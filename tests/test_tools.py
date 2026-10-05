"""Operator/evaluation tools: count accuracy evaluator, healthcheck, diagnostic export."""

import json
import os
import shutil
import sys
import time
import unittest
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.helpers import tmpdir
from tools.eval_counts import evaluate
from tools.healthcheck import check


def payload(t, lanes, gates=None, w0=None, w1=None):
    p = {"observedAt": t, "timestamp": t, "lanes": lanes}
    if gates is not None:
        p["traffic_flow"] = {"interval": {"windowStart": w0, "windowEnd": w1, "gates": gates}}
    return p


class TestEvalCounts(unittest.TestCase):
    def test_lane_errors_false_zero_unknown_and_gates(self):
        from tools.eval_counts import ts
        P = [
            (ts("2026-10-12T00:00:01Z"), payload("2026-10-12T00:00:01Z", [
                {"laneId": "N1", "valid": True, "count": 5, "queuedCount": 4},
                {"laneId": "N2", "valid": True, "count": 0, "queuedCount": 0}],
                gates=[{"gateId": "G", "count": 2, "valid": True}], w0="2026-10-12T00:00:00Z", w1="2026-10-12T00:00:01Z")),
            (ts("2026-10-12T00:00:02Z"), payload("2026-10-12T00:00:02Z", [
                {"laneId": "N1", "valid": False, "count": None, "queuedCount": None},
                {"laneId": "N2", "valid": True, "count": 1, "queuedCount": 0}],
                gates=[{"gateId": "G", "count": 0, "valid": True}], w0="2026-10-12T00:00:01Z", w1="2026-10-12T00:00:02Z")),
        ]
        truth = {
            "occupancy": [
                {"t": "2026-10-12T00:00:01.2Z", "laneId": "N1", "count": 6, "queued": 4},
                {"t": "2026-10-12T00:00:01.1Z", "laneId": "N2", "count": 2, "queued": 0},
                {"t": "2026-10-12T00:00:02Z", "laneId": "N1", "count": 3, "queued": 3},
                {"t": "2026-10-12T00:00:02Z", "laneId": "N2", "count": 0, "queued": 0},
            ],
            "gateEvents": [{"t": "2026-10-12T00:00:00.5Z", "gateId": "G"}, {"t": "2026-10-12T00:00:01.5Z", "gateId": "G"}],
        }
        r = evaluate(P, truth)
        self.assertEqual(r["lanes"]["N1"]["count"], {"n": 1, "mae": 1.0, "bias": -1.0, "p95": 1.0})
        self.assertEqual(r["lanes"]["N1"]["unknownShare"], 0.5)
        self.assertEqual(r["lanes"]["N2"]["falseZeroRate"], 1.0)
        self.assertEqual(r["lanes"]["N2"]["emptyLaneFalsePositiveRate"], 1.0)
        self.assertEqual(r["gates"]["G"], {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                                            "validIntervals": 2, "invalidIntervals": 0, "unobservedTruthEvents": 0})

    def test_false_crossings_on_empty_labelled_gate_have_zero_f1(self):
        p = payload("2026-10-12T00:00:01Z", [],
                    gates=[{"gateId": "G", "count": 2, "valid": True}],
                    w0="2026-10-12T00:00:00Z", w1="2026-10-12T00:00:01Z")
        result = evaluate([(0, p)], {"gates": ["G"], "gateEvents": []})["gates"]["G"]
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["f1"], 0)

    def test_lane_alignment_uses_its_own_observation_time(self):
        p = payload("2026-10-12T00:00:01Z", [
            {"laneId": "E1", "observedAt": "2026-10-12T00:00:03Z", "valid": True, "count": 2}])
        result = evaluate([(0, p)], {"occupancy": [
            {"laneId": "E1", "t": "2026-10-12T00:00:03Z", "count": 2}]}, max_dt=.1)
        self.assertEqual(result["lanes"]["E1"]["count"]["mae"], 0)

    def test_unmatched_observations_are_part_of_unknown_share(self):
        result = evaluate([], {"occupancy": [{"laneId": "N1", "t": "2026-10-12T00:00:03Z", "count": 1}]})
        self.assertEqual(result["lanes"]["N1"]["unknownShare"], 1)

    def test_failed_publish_intervals_are_not_counted_twice(self):
        rows = []
        for end in ("01", "02"):
            rows.append((0, payload("2026-10-12T00:00:01Z", [],
                         gates=[{"gateId": "G", "count": 1, "valid": True}],
                         w0="2026-10-12T00:00:00Z", w1=f"2026-10-12T00:00:{end}Z")))
        result = evaluate(rows, {"gateEvents": [{"gateId": "G", "t": "2026-10-12T00:00:00.5Z"}]})
        self.assertEqual(result["gates"]["G"]["tp"], 1)
        self.assertEqual(result["gates"]["G"]["validIntervals"], 1)


class TestHealthcheck(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()
        self.path = os.path.join(self.dir, "health.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, cams):
        with open(self.path, "w") as f:
            json.dump({"validCameras": sum(c["status"] == "ok" for c in cams), "totalCameras": len(cams), "cameras": cams}, f)

    def test_codes(self):
        self.write([{"name": "n", "status": "ok"}, {"name": "s", "status": "ok"}])
        self.assertEqual(check(self.path, 10)[0], 0)
        self.write([{"name": "n", "status": "ok"}, {"name": "s", "status": "offline"}])
        self.assertEqual(check(self.path, 10)[0], 1)
        self.write([{"name": "n", "status": "stale"}])
        self.assertEqual(check(self.path, 10)[0], 2)
        self.assertEqual(check(self.path, 10, now=time.time() + 60)[0], 2)
        self.assertEqual(check(os.path.join(self.dir, "missing.json"), 10)[0], 2)


class TestDiagnosticExport(unittest.TestCase):
    def test_bundle_contains_window_and_manifest(self):
        from tools.diagnostic_export import main
        d = tmpdir()
        try:
            pl = os.path.join(d, "p.jsonl")
            with open(pl, "w") as f:
                for t in ("2026-10-12T00:00:00Z", "2026-10-12T00:05:00Z", "2026-10-12T01:00:00Z"):
                    f.write(json.dumps({"delivered": True, "payload": {"observedAt": t, "lanes": []}}) + "\n")
            out = os.path.join(d, "bundle.zip")
            main(["--payloads", pl, "--from", "2026-10-12T00:00:00Z", "--to", "2026-10-12T00:10:00Z",
                  "--out", out, "--models"])
            with zipfile.ZipFile(out) as z:
                manifest = json.loads(z.read("manifest.json"))
                self.assertEqual(manifest["payloadCount"], 2)
                self.assertIn("config/config_north.json", z.namelist())
                self.assertIn("2026-10-04-batch2b", json.dumps(manifest["configs"]))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
