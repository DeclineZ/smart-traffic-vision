"""
Compare a TensorRT engine against its source .pt checkpoint on real camera frames.

Checks, per batch size (1..N cameras, as happens when cameras drop out):
  * detection count parity per frame and in total
  * one-to-one box matching (IoU >= 0.5): matched fraction, mean IoU, class agreement
  * per-lane occupancy parity using the camera's lane polygons (bottom-centre anchor)
  * warmed-up latency p50/p95/p99 for each backend

Usage:
    .venv\\Scripts\\python.exe -m tools.trt_parity --engine models/yolo26s_thai_traffic.engine \\
        --model models/yolo26s_thai_traffic.pt --frames 40 --out docs/trt-parity/report.json

Exit code 1 if any acceptance threshold fails (thresholds are CLI options).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2 as cv
import numpy as np
import shapely

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.utils import iou_batch  # noqa: E402
from trt_pipeline.lane_validation import validate_config_file  # noqa: E402

CAMERAS = ["north", "south", "east", "west", "northeast"]


def load_frames(n_per_cam: int, stride_s: float):
    frames, lanes = {c: [] for c in CAMERAS}, {}
    for cam in CAMERAS:
        cfg_path = f"config/config_{cam}.json"
        report = validate_config_file(cfg_path)
        lanes[cam] = report.valid_polygons
        cap = cv.VideoCapture(report.raw_config["video"]["path"])
        fps = cap.get(cv.CAP_PROP_FPS) or 25.0
        for i in range(n_per_cam):
            cap.set(cv.CAP_PROP_POS_FRAMES, int((30 + i * stride_s) * fps))
            ok, f = cap.read()
            if ok:
                frames[cam].append(f)
        cap.release()
    return frames, lanes


def dets(result):
    b = result.boxes
    if b is None or len(b) == 0:
        return np.empty((0, 6))
    return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]])


def lane_counts(d, polys):
    if len(d) == 0:
        return {k: 0 for k in polys}
    xs, ys = (d[:, 0] + d[:, 2]) / 2, d[:, 3]
    taken = np.zeros(len(d), bool)
    out = {}
    for k, p in polys.items():
        m = shapely.intersects_xy(p, xs, ys) & ~taken
        taken |= m
        out[k] = int(m.sum())
    return out


def compare(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0, [], 0
    from scipy.optimize import linear_sum_assignment
    ious = iou_batch(a[:, :4], b[:, :4])
    r, c = linear_sum_assignment(-ious)
    keep = ious[r, c] >= 0.5
    same_cls = int(np.sum(a[r[keep], 5] == b[c[keep], 5]))
    return int(keep.sum()), ious[r, c][keep].tolist(), same_cls


def run_backend(model, batches, kw, warmup=3):
    for _ in range(warmup):
        model(batches[0], **kw)
    out, lat = [], []
    for batch in batches:
        t0 = time.perf_counter()
        res = model(batch, **kw)
        lat.append((time.perf_counter() - t0) * 1000)
        out.append([dets(r) for r in res])
    return out, lat


def pct(v, q):
    return round(float(np.percentile(v, q)), 2) if v else None


def detection_agreement(n_pt, n_trt, matched, same_cls):
    if n_pt == n_trt == 0:
        return 1.0, 1.0, 1.0
    return matched / max(1, n_pt), matched / max(1, n_trt), same_cls / max(1, matched)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="models/yolo26s_thai_traffic.pt")
    ap.add_argument("--frames", type=int, default=40, help="frames per camera")
    ap.add_argument("--stride", type=float, default=7.0, help="seconds between sampled frames")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--min-match", type=float, default=0.95, help="min fraction of .pt boxes matched by the engine")
    ap.add_argument("--min-class-agreement", type=float, default=0.99, help="min class agreement of matched boxes")
    ap.add_argument("--max-lane-mae", type=float, default=0.10, help="max mean abs per-lane occupancy difference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.frames < 1 or not np.isfinite(args.stride) or args.stride <= 0:
        ap.error("--frames and --stride must be positive")
    if not 0 <= args.min_match <= 1 or not 0 <= args.min_class_agreement <= 1:
        ap.error("agreement thresholds must be between 0 and 1")

    from ultralytics import YOLO

    frames, lanes = load_frames(args.frames, args.stride)
    pt, trt = YOLO(args.model), YOLO(args.engine, task="detect")
    kw = {"verbose": False, "imgsz": args.imgsz, "conf": args.conf, "device": 0}

    report = {"engine": args.engine, "model": args.model, "conf": args.conf, "imgsz": args.imgsz, "batches": {}}
    ok = True
    for bs in range(1, len(CAMERAS) + 1):
        cams = CAMERAS[:bs]
        n = min(len(frames[c]) for c in cams)
        batches = [[frames[c][i] for c in cams] for i in range(n)]
        pt_out, pt_lat = run_backend(pt, batches, kw)
        trt_out, trt_lat = run_backend(trt, batches, kw)

        n_pt = n_trt = matched = same_cls = 0
        ious, lane_err = [], []
        for bi in range(n):
            for ci, cam in enumerate(cams):
                a, b = pt_out[bi][ci], trt_out[bi][ci]
                n_pt += len(a)
                n_trt += len(b)
                m, iou, sc = compare(a, b)
                matched += m
                same_cls += sc
                ious += iou
                la, lb = lane_counts(a, lanes[cam]), lane_counts(b, lanes[cam])
                lane_err += [abs(la[k] - lb[k]) for k in la]
        match_frac, engine_precision, class_agreement = detection_agreement(n_pt, n_trt, matched, same_cls)
        lane_mae = float(np.mean(lane_err)) if lane_err else 0.0
        passed = (match_frac >= args.min_match and engine_precision >= args.min_match
                  and class_agreement >= args.min_class_agreement and lane_mae <= args.max_lane_mae)
        ok &= passed
        report["batches"][bs] = {
            "frames": n * bs,
            "ptDetections": n_pt,
            "engineDetections": n_trt,
            "matchedFraction": round(match_frac, 4),
            "engineMatchedFraction": round(engine_precision, 4),
            "meanIoU": round(float(np.mean(ious)), 4) if ious else None,
            "classAgreement": round(class_agreement, 4),
            "laneOccupancyMAE": round(lane_mae, 4),
            "laneOccupancyMaxAbs": int(max(lane_err)) if lane_err else 0,
            "ptLatencyMs": {"p50": pct(pt_lat, 50), "p95": pct(pt_lat, 95), "p99": pct(pt_lat, 99)},
            "engineLatencyMs": {"p50": pct(trt_lat, 50), "p95": pct(trt_lat, 95), "p99": pct(trt_lat, 99)},
            "passed": passed,
        }
        print(f"batch={bs}: " + json.dumps(report["batches"][bs]))

    report["passed"] = ok
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
