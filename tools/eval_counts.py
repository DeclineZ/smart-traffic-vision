"""
End-to-end count accuracy: compare recorded payloads (--record-payloads JSONL)
with human ground truth for the same clip.

Ground truth format (JSON):
{
  "clip": "north-2026-10-12-0730",
  "occupancy": [                       # sampled lane states, wall time ISO-8601
    {"t": "2026-10-12T00:30:05Z", "laneId": "N1", "count": 7, "queued": 5},
    ...
  ],
  "gateEvents": [                      # one entry per vehicle crossing
    {"t": "2026-10-12T00:30:07.4Z", "gateId": "GATE_N_STOPLINE"},
    ...
  ]
}

Gate annotations must be complete for the evaluated intervals, including empty
intervals. Optional "gates": ["GATE_N_STOPLINE"] limits evaluation to labelled
gates; otherwise every reported gate is assumed labelled. False crossings on
empty gates are included. Unknown intervals and uncovered truth events are
reported separately from accuracy on valid intervals.

For each labelled occupancy sample the payload with the closest observedAt
within --max-dt seconds is used. Reported per lane and overall:
  MAE, signed bias, p95 absolute error (count and queued), empty-lane false
  positives (truth 0, predicted >0), false zeros (truth >0, predicted 0),
  and the share of samples where vision had no valid reading.
Gate events are compared per publish interval (vision reports interval
counts, not per-vehicle crossing times): precision, recall, F1 per gate.
Intervals where the gate was not validly observed are skipped.

    .venv\\Scripts\\python.exe -m tools.eval_counts --payloads runs/replay/payloads.jsonl \\
        --truth data/groundtruth/north-0730.json --out runs/eval/north-0730.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_payloads(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("payload", rec)
            out.append((ts(p.get("observedAt") or p["timestamp"]), p))
    out.sort(key=lambda x: x[0])
    return out


def stats(errors):
    if not errors:
        return None
    e = np.asarray(errors, dtype=float)
    return {"n": int(e.size), "mae": round(float(np.abs(e).mean()), 3), "bias": round(float(e.mean()), 3),
            "p95": round(float(np.percentile(np.abs(e), 95)), 3)}


def evaluate(payloads, truth, max_dt=1.0):
    observations = defaultdict(list)
    for wrapper_t, p in payloads:
        for pl in p["lanes"]:
            valid = pl.get("valid", True) and pl.get("count") is not None
            at = pl.get("observedAt") if valid else p.get("publishedAt")
            observations[pl["laneId"]].append((ts(at) if at else wrapper_t, pl))
    for rows in observations.values():
        rows.sort(key=lambda row: row[0])
    lane_times = {lid: [t for t, _ in rows] for lid, rows in observations.items()}
    per_lane = defaultdict(lambda: {"count": [], "queued": [], "fp_empty": 0, "false_zero": 0, "empty_truth": 0,
                                    "nonempty_truth": 0, "unknown": 0, "unmatched": 0})
    for s in truth.get("occupancy", []):
        t = ts(s["t"])
        lane = per_lane[s["laneId"]]
        times = lane_times.get(s["laneId"], [])
        i = bisect.bisect_left(times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
        j = min(cands, key=lambda k: abs(times[k] - t)) if cands else None
        if j is None or abs(times[j] - t) > max_dt:
            lane["unmatched"] += 1
            continue
        pl = observations[s["laneId"]][j][1]
        if pl is None or not pl.get("valid", True) or pl.get("count") is None:
            lane["unknown"] += 1
            continue
        lane["count"].append(pl["count"] - s["count"])
        if "queued" in s and pl.get("queuedCount") is not None:
            lane["queued"].append(pl["queuedCount"] - s["queued"])
        if s["count"] == 0:
            lane["empty_truth"] += 1
            lane["fp_empty"] += int(pl["count"] > 0)
        else:
            lane["nonempty_truth"] += 1
            lane["false_zero"] += int(pl["count"] == 0)

    lanes_out = {}
    all_c, all_q = [], []
    for lid, d in sorted(per_lane.items()):
        all_c += d["count"]
        all_q += d["queued"]
        total = len(d["count"]) + d["unknown"] + d["unmatched"]
        lanes_out[lid] = {
            "count": stats(d["count"]),
            "queued": stats(d["queued"]),
            "emptyLaneFalsePositiveRate": round(d["fp_empty"] / d["empty_truth"], 3) if d["empty_truth"] else None,
            "falseZeroRate": round(d["false_zero"] / d["nonempty_truth"], 3) if d["nonempty_truth"] else None,
            "unknownShare": round((d["unknown"] + d["unmatched"]) / total, 3) if total else None,
            "unmatchedSamples": d["unmatched"],
        }

    gates_out = {}
    truth_by_gate = defaultdict(list)
    for e in truth.get("gateEvents", []):
        truth_by_gate[e["gateId"]].append(ts(e["t"]))
    reported_gates = {g["gateId"] for _, p in payloads
                      for g in ((p.get("traffic_flow") or {}).get("interval") or {}).get("gates", [])}
    gate_ids = set(truth.get("gates", reported_gates | set(truth_by_gate)))
    for gid in sorted(gate_ids):
        tt = truth_by_gate[gid]
        tt.sort()
        tp = fp = fn = 0
        intervals = {}
        for _, p in payloads:
            iv = (p.get("traffic_flow") or {}).get("interval")
            if not iv:
                continue
            g = next((x for x in iv["gates"] if x["gateId"] == gid), None)
            start, end = ts(iv["windowStart"]), ts(iv["windowEnd"])
            # Failed delivery can record the same growing interval repeatedly.
            if start not in intervals or end >= intervals[start][0]:
                intervals[start] = (end, g)
        valid_windows = []
        invalid_intervals = 0
        for start, (end, g) in sorted(intervals.items()):
            if g is None or not g["valid"] or g["count"] is None:
                invalid_intervals += 1
                continue
            valid_windows.append((start, end))
            n_truth = sum(1 for x in tt if start <= x < end)
            n_pred = g["count"]
            tp += min(n_truth, n_pred)
            fp += max(0, n_pred - n_truth)
            fn += max(0, n_truth - n_pred)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        gates_out[gid] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": round(prec, 3) if prec is not None else None,
                          "recall": round(rec, 3) if rec is not None else None,
                          "f1": round(f1, 3) if f1 is not None else None,
                          "validIntervals": len(valid_windows), "invalidIntervals": invalid_intervals,
                          "unobservedTruthEvents": sum(not any(a <= x < b for a, b in valid_windows) for x in tt)}

    return {"clip": truth.get("clip"), "overall": {"count": stats(all_c), "queued": stats(all_q)},
            "lanes": lanes_out, "gates": gates_out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payloads", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--max-dt", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    with open(args.truth, encoding="utf-8") as f:
        truth = json.load(f)
    res = evaluate(load_payloads(args.payloads), truth, args.max_dt)
    text = json.dumps(res, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
