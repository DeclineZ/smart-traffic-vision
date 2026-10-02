#!/usr/bin/env python3
"""
Pre-label pack frames so annotators correct boxes instead of drawing every one.

Three passes are merged per frame:
  base640   - deployed baseline (models/yolo26s_thai_traffic.pt) at its native 640
  base1280  - same model at 1280, which recovers some small/distant vehicles
  coco1280  - COCO teacher (models/yolo26x.pt) at 1280; good at parked and background
              cars the baseline never learned, but has no three_wheeler class and calls
              most pickups "truck"

Boxes from different passes that overlap (IoU >= 0.55) are clustered. The baseline's class
wins when it saw the object. A box is marked `check` (drawn dashed with a "?" in the UI) when
the passes disagree in a way that matters - mostly the pickup-vs-truck confusion - or when
confidence is low. Nothing here is ground truth; every frame still needs a human pass.

Usage:
  python tools/manual_label/prelabel.py --pack data/manual_v1
  python tools/manual_label/prelabel.py --pack data/manual_v1 --baseline runs/train/manual_v1_ft/weights/best.pt --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import (
    BUS, CAR, CLASS_NAMES, MOTORCYCLE, REPO_ROOT, THREE_WHEELER, TRUCK,
    Pack, iou_xyxy, now_iso, write_json_atomic,
)

DEFAULT_BASELINE = REPO_ROOT / "models" / "yolo26s_thai_traffic.pt"
DEFAULT_TEACHER = REPO_ROOT / "models" / "yolo26x.pt"

# COCO ids -> our ids. Bicycle and everything else is dropped.
COCO_TO_OURS = {2: CAR, 3: MOTORCYCLE, 5: BUS, 7: TRUCK}
COCO_VEHICLE_IDS = list(COCO_TO_OURS)

# COCO classes that are *consistent* with a baseline class (COCO "truck" covers pickups,
# so baseline car + COCO truck is the normal case, not a disagreement).
COMPATIBLE_COCO = {
    CAR: {CAR, TRUCK},
    MOTORCYCLE: {MOTORCYCLE},
    BUS: {BUS, TRUCK},
    TRUCK: {TRUCK},
    THREE_WHEELER: {CAR, MOTORCYCLE, TRUCK},
}

CLUSTER_IOU = 0.55
BASE_MIN_CONF = 0.25
COCO_ONLY_MIN_CONF = 0.30
SURE_TRUCK_CONF = 0.70


def predict(model, images: List[Any], imgsz: int, conf: float, src: str, coco: bool = False) -> List[List[Dict]]:
    """Run one pass over a batch of BGR images. Returns per-image proposal lists in our class ids."""
    kwargs = dict(imgsz=imgsz, conf=conf, verbose=False)
    if coco:
        kwargs["classes"] = COCO_VEHICLE_IDS
    out = []
    for r in model.predict(images, **kwargs):
        props = []
        if r.boxes is not None and len(r.boxes):
            for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                cls = int(c)
                if coco:
                    if cls not in COCO_TO_OURS:
                        continue
                    cls = COCO_TO_OURS[cls]
                props.append({"cls": cls, "xyxy": [float(v) for v in xyxy], "conf": float(s), "src": src})
        out.append(props)
    return out


def merge_proposals(props: Sequence[Dict]) -> List[Dict]:
    """Cluster proposals from several passes into one box per vehicle, flagging doubtful ones."""
    ordered = sorted(props, key=lambda p: (not p["src"].startswith("base"), -p["conf"]))
    clusters: List[List[Dict]] = []
    for p in ordered:
        best, best_iou = None, CLUSTER_IOU
        for cl in clusters:
            iou = iou_xyxy(p["xyxy"], cl[0]["xyxy"])
            if iou >= best_iou:
                best, best_iou = cl, iou
        if best is None:
            clusters.append([p])
        else:
            best.append(p)

    boxes = []
    for cl in clusters:
        base = [p for p in cl if p["src"].startswith("base")]
        coco = [p for p in cl if p["src"].startswith("coco")]
        notes = []
        if base:
            rep = max(base, key=lambda p: p["conf"])
            if rep["conf"] < BASE_MIN_CONF:
                continue
            cls = rep["cls"]
            base_classes = {p["cls"] for p in base}
            if len(base_classes) > 1:
                notes.append("baseline passes disagree: " + " vs ".join(CLASS_NAMES[c] for c in sorted(base_classes)))
            if coco and not ({p["cls"] for p in coco} & COMPATIBLE_COCO[cls]):
                coco_cls = max(coco, key=lambda p: p["conf"])["cls"]
                notes.append(f"COCO model says {CLASS_NAMES[coco_cls]}")
            if cls == TRUCK and rep["conf"] < SURE_TRUCK_CONF:
                notes.append("truck or pickup/van/songthaew?")
        else:
            rep = max(coco, key=lambda p: p["conf"])
            if rep["conf"] < COCO_ONLY_MIN_CONF:
                continue
            cls = rep["cls"]
            if cls == TRUCK:
                # Baseline missed it entirely and COCO calls it a truck: far more often a pickup.
                cls = CAR
                notes.append("COCO-only 'truck' set to car - check pickup vs truck")
            elif rep["conf"] < 0.5:
                notes.append("only the COCO model saw this")
        if rep["conf"] < 0.35 and not notes:
            notes.append("low confidence")

        x1, y1, x2, y2 = rep["xyxy"]
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        boxes.append({
            "cls": cls,
            "xyxy": [round(v, 1) for v in rep["xyxy"]],
            "conf": round(rep["conf"], 3),
            "check": bool(notes),
            "note": "; ".join(notes),
            "sources": sorted({p["src"] for p in cl}),
        })
    boxes.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    return boxes


def prelabel_pack(pack: Pack, baseline: Path, teacher: Path | None, overwrite: bool, batch: int = 8) -> None:
    import cv2
    from ultralytics import YOLO

    todo = [f for f in pack.frames if overwrite or not pack.prelabel_path(f["id"]).exists()]
    if not todo:
        print("[prelabel] every frame already has prelabels (use --overwrite to redo)")
        return
    print(f"[prelabel] {len(todo)} frames | baseline={baseline.name} teacher={teacher.name if teacher else 'none'}")
    base_model = YOLO(str(baseline))
    teacher_model = YOLO(str(teacher)) if teacher else None
    models_meta = {"baseline": str(baseline.relative_to(REPO_ROOT) if baseline.is_relative_to(REPO_ROOT) else baseline),
                   "teacher": str(teacher.name) if teacher else None}

    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        images = [cv2.imread(str(pack.image_path(f["id"]))) for f in chunk]
        passes = [
            predict(base_model, images, 640, 0.15, "base640"),
            predict(base_model, images, 1280, 0.15, "base1280"),
        ]
        if teacher_model is not None:
            passes.append(predict(teacher_model, images, 1280, 0.25, "coco1280", coco=True))
        for i, f in enumerate(chunk):
            h, w = images[i].shape[:2]
            boxes = merge_proposals([p for ps in passes for p in ps[i]])
            write_json_atomic(pack.prelabel_path(f["id"]), {
                "frame_id": f["id"], "img_w": w, "img_h": h, "created_at": now_iso(),
                "models": models_meta, "boxes": boxes,
            })
        print(f"  {min(start + batch, len(todo))}/{len(todo)}", end="\r", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--no-teacher", action="store_true", help="skip the COCO yolo26x pass")
    ap.add_argument("--overwrite", action="store_true", help="redo frames that already have prelabels")
    args = ap.parse_args()

    teacher = None if args.no_teacher else args.teacher.resolve()
    prelabel_pack(Pack(args.pack), args.baseline.resolve(), teacher, args.overwrite)


if __name__ == "__main__":
    main()
