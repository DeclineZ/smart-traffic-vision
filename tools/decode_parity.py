"""
Check that GPU decoding (NVDEC + on-GPU resize) gives the same detections as OpenCV
full-resolution decoding, on the same frames of every calibrated camera.

Both decoders read each recording sequentially from the start, so frame i is the same
picture in both; every --stride-th frame is compared. Boxes from the downscaled frames are
mapped back to camera pixels before matching, exactly as run_multi_camera.py does.

With --val-dir (the hand-labelled split, e.g. data/manual_v1_dataset), each labelled frame is
also decoded from its recording through the GPU path and scored against the hand labels next
to the original full-resolution image. That is the acceptance test: recall on the labels must
not drop by more than --max-recall-drop. Box-level agreement with OpenCV decoding is reported
for information; hardware and software decoders differ by ~1 grey level, which flips some
low-confidence boxes either way.

Usage:
    .venv\\Scripts\\python.exe -m tools.decode_parity --model models/yolo26s_thai_traffic_manual_v1_640.engine \\
        --frames 30 --val-dir data/manual_v1_dataset --out output/decode-parity.json

Exit code 1 if the label check fails (or, without --val-dir, if agreement is below thresholds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.utils import iou_batch  # noqa: E402
from tools.trt_parity import CAMERAS, compare, dets, detection_agreement, lane_counts  # noqa: E402
from trt_pipeline.lane_validation import validate_config_file  # noqa: E402
from trt_pipeline.nvdec import NvdecCapture  # noqa: E402
from trt_pipeline.tools import engine_imgsz  # noqa: E402


def sample_pairs(path: str, out_width: int, n: int, stride: int):
    """[(full-res OpenCV frame, NVDEC frame, scale)] for frames stride, 2*stride, ..."""
    cv_cap = cv.VideoCapture(path)
    gpu = NvdecCapture(path, out_width=out_width)
    pairs, i = [], 0
    while len(pairs) < n:
        ok_a, a = cv_cap.read()
        ok_b, b = gpu.read()
        if not (ok_a and ok_b):
            break
        i += 1
        if i % stride == 0:
            pairs.append((a, b))
    cv_cap.release()
    gpu.release()
    return pairs, gpu.scale


def label_score(d: np.ndarray, label_path: str, w: int, h: int):
    """(labelled vehicles, found at IoU >= 0.5, found with the right class); greedy one-to-one."""
    gt = []
    if os.path.exists(label_path):
        for line in open(label_path, encoding="utf-8").read().split("\n"):
            parts = line.split()
            if len(parts) == 5:
                c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, c])
    gt = np.array(gt, dtype=float).reshape(-1, 5)
    if len(gt) == 0 or len(d) == 0:
        return len(gt), 0, 0
    iou = iou_batch(gt[:, :4], d[:, :4])
    found = right = 0
    used_g, used_d = set(), set()
    for i, j in sorted(zip(*np.nonzero(iou >= 0.5)), key=lambda ij: -iou[ij]):
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        found += 1
        right += int(gt[i, 4]) == int(d[j, 5])
    return len(gt), found, right


def label_check(model, kw, val_dir: str, out_width: int, conf: float):
    """Score labelled frames decoded through NVDEC against the original images, on the hand labels."""
    images = os.path.join(val_dir, "images", "val")
    labels = os.path.join(val_dir, "labels", "val")
    kw = dict(kw, conf=conf)
    tot = {"gt": 0, "frames": 0, "image": [0, 0], "nvdec": [0, 0]}
    for name in sorted(os.listdir(images)):
        stem, _ = os.path.splitext(name)
        cam, idx = stem.rsplit("_f", 1)
        video = os.path.join("videos", f"{cam}.avi")
        if not os.path.exists(video):
            continue
        cap = cv.VideoCapture(video)  # labelled frames were taken at OpenCV frame indices; get that frame's time
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        cap.read()
        t = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
        cap.release()
        gpu = NvdecCapture(video, out_width=out_width, start_s=t)
        ok, small = gpu.read()
        gpu.release()
        if not ok:
            continue
        img = cv.imread(os.path.join(images, name))
        label = os.path.join(labels, f"{stem}.txt")
        a = dets(model(img, **kw)[0])
        b = dets(model(small, **kw)[0])
        if len(b):
            b[:, :4] *= gpu.scale
        n, fa, ra = label_score(a, label, img.shape[1], img.shape[0])
        _, fb, rb = label_score(b, label, img.shape[1], img.shape[0])
        tot["gt"] += n
        tot["frames"] += 1
        tot["image"][0] += fa
        tot["image"][1] += ra
        tot["nvdec"][0] += fb
        tot["nvdec"][1] += rb
    g = max(1, tot["gt"])
    return {
        "frames": tot["frames"], "labelledVehicles": tot["gt"], "conf": conf,
        "imageRecall": round(tot["image"][0] / g, 4), "nvdecRecall": round(tot["nvdec"][0] / g, 4),
        "imageClassRecall": round(tot["image"][1] / g, 4), "nvdecClassRecall": round(tot["nvdec"][1] / g, 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/yolo26s_thai_traffic_manual_v1_640.engine")
    ap.add_argument("--imgsz", type=int, default=None, help="default: the engine's build size, else 640")
    ap.add_argument("--frames", type=int, default=30, help="frames compared per camera")
    ap.add_argument("--stride", type=int, default=100, help="compare every Nth decoded frame")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--min-match", type=float, default=0.95, help="min fraction of boxes matched both ways")
    ap.add_argument("--min-class-agreement", type=float, default=0.98)
    ap.add_argument("--max-lane-mae", type=float, default=0.10, help="max mean abs per-lane occupancy difference")
    ap.add_argument("--val-dir", default=None, help="hand-labelled split (images/val, labels/val) for the label check")
    ap.add_argument("--max-recall-drop", type=float, default=0.01, help="max recall loss on the hand labels")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    imgsz = args.imgsz or engine_imgsz(args.model) or 640

    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    kw = {"verbose": False, "imgsz": imgsz, "conf": args.conf, "device": 0}
    report = {"model": args.model, "imgsz": imgsz, "conf": args.conf, "cameras": {}}
    totals = {"cv": 0, "gpu": 0, "matched": 0, "same_cls": 0, "lane_err": [], "ious": []}

    for cam in CAMERAS:
        cfg = validate_config_file(f"config/config_{cam}.json")
        polys = cfg.valid_polygons
        pairs, scale = sample_pairs(cfg.raw_config["video"]["path"], imgsz, args.frames, args.stride)
        n_cv = n_gpu = matched = same_cls = 0
        lane_err, ious, pixel_diff = [], [], []
        for full, small in pairs:
            a = dets(model(full, **kw)[0])
            b = dets(model(small, **kw)[0])
            if len(b):
                b[:, :4] *= scale
            n_cv += len(a)
            n_gpu += len(b)
            m, iou, sc = compare(a, b)
            matched += m
            same_cls += sc
            ious += iou
            la, lb = lane_counts(a, polys), lane_counts(b, polys)
            lane_err += [abs(la[k] - lb[k]) for k in la]
            ref = cv.resize(full, (small.shape[1], small.shape[0]), interpolation=cv.INTER_AREA)
            pixel_diff.append(float(np.mean(cv.absdiff(ref, small))))
        cv_frac, gpu_frac, cls_frac = detection_agreement(n_cv, n_gpu, matched, same_cls)
        report["cameras"][cam] = {
            "frames": len(pairs), "opencvDetections": n_cv, "nvdecDetections": n_gpu,
            "matchedFraction": round(cv_frac, 4), "nvdecMatchedFraction": round(gpu_frac, 4),
            "classAgreement": round(cls_frac, 4), "meanIoU": round(float(np.mean(ious)), 4) if ious else None,
            "laneOccupancyMAE": round(float(np.mean(lane_err)), 4) if lane_err else 0.0,
            "meanPixelDiff": round(float(np.mean(pixel_diff)), 2) if pixel_diff else None,
        }
        print(f"{cam}: " + json.dumps(report["cameras"][cam]))
        totals["cv"] += n_cv
        totals["gpu"] += n_gpu
        totals["matched"] += matched
        totals["same_cls"] += same_cls
        totals["lane_err"] += lane_err
        totals["ious"] += ious

    cv_frac, gpu_frac, cls_frac = detection_agreement(totals["cv"], totals["gpu"], totals["matched"], totals["same_cls"])
    lane_mae = float(np.mean(totals["lane_err"])) if totals["lane_err"] else 0.0
    agreement_ok = (cv_frac >= args.min_match and gpu_frac >= args.min_match
                    and cls_frac >= args.min_class_agreement and lane_mae <= args.max_lane_mae)
    passed = agreement_ok
    if args.val_dir:
        labels = label_check(model, kw, args.val_dir, imgsz, conf=0.25)
        labels["passed"] = (labels["nvdecRecall"] >= labels["imageRecall"] - args.max_recall_drop
                            and labels["nvdecClassRecall"] >= labels["imageClassRecall"] - args.max_recall_drop)
        report["handLabels"] = labels
        print("hand labels: " + json.dumps(labels))
        passed = labels["passed"]
    report["overall"] = {
        "opencvDetections": totals["cv"], "nvdecDetections": totals["gpu"],
        "matchedFraction": round(cv_frac, 4), "nvdecMatchedFraction": round(gpu_frac, 4),
        "classAgreement": round(cls_frac, 4), "meanIoU": round(float(np.mean(totals["ious"])), 4) if totals["ious"] else None,
        "laneOccupancyMAE": round(lane_mae, 4), "withinAgreementThresholds": agreement_ok,
    }
    report["passed"] = passed
    print("overall: " + json.dumps(report["overall"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
