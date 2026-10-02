#!/usr/bin/env python3
"""
Compare detectors on a manually labeled validation set and write a markdown report.

For every model:
  - Ultralytics val() at conf=0.001: mAP50, mAP50-95, per-class AP50
  - operational numbers at --conf (default 0.25, what the live pipeline uses):
    per-class precision / recall / F1, class-agnostic recall, recall by object size,
    day vs night, per camera, false positives per frame, and the car<->truck confusion
    that pickups cause.

Usage:
  python tools/manual_label/evaluate.py --data data/manual_v1_dataset/data.yaml \
      --models models/yolo26s_thai_traffic.pt runs/train/manual_v1_ft_baseline/weights/best.pt
  python tools/manual_label/evaluate.py --data data/eval_snapshot_v1/dataset.yaml --models ...   # old 42-frame diagnostic set
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import CAR, CLASS_NAMES, REPO_ROOT, TRUCK, iou_xyxy, parse_frame_id, read_yolo_labels

IOU_THR = 0.5


def resolve(p: Path) -> Path:
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def val_images(data_yaml: Path) -> List[Path]:
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    val = root / cfg["val"]
    return sorted(p for p in val.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def greedy_match(preds: List[Dict], gts: List[Dict]) -> List[Tuple[int, int]]:
    """Predictions in confidence order each take the best still-free GT with IoU >= 0.5."""
    used, pairs = set(), []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["conf"]):
        best, best_iou = -1, IOU_THR
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            iou = iou_xyxy(preds[pi]["xyxy"], g["xyxy"])
            if iou >= best_iou:
                best, best_iou = gi, iou
        if best >= 0:
            used.add(best)
            pairs.append((pi, best))
    return pairs


def size_bucket(box: List[float], img_w: int, img_h: int) -> str:
    s = 640 / max(img_w, img_h)
    area = (box[2] - box[0]) * s * (box[3] - box[1]) * s
    return "small" if area < 32 * 32 else "medium" if area <= 96 * 96 else "large"


def slice_of(img: Path) -> Tuple[str, str]:
    parsed = parse_frame_id(img.name)
    cam = parsed[0] if parsed else "other"
    return cam.replace("_night", ""), "night" if cam.endswith("_night") else "day"


def operational(model, images: List[Path], conf: float, imgsz: int) -> Dict:
    import cv2
    nc = len(CLASS_NAMES)
    per_class = {c: Counter() for c in range(nc)}
    confusion = [[0] * (nc + 1) for _ in range(nc + 1)]  # rows GT (+bg), cols pred (+missed)
    size_tot, size_hit = Counter(), Counter()
    slices = defaultdict(Counter)
    agn = Counter()
    for start in range(0, len(images), 8):
        chunk = images[start:start + 8]
        frames = [cv2.imread(str(p)) for p in chunk]
        results = model.predict(frames, imgsz=imgsz, conf=conf, verbose=False)
        for img_path, frame, r in zip(chunk, frames, results):
            h, w = frame.shape[:2]
            gts = read_yolo_labels(label_for(img_path), w, h)
            preds = [{"cls": int(c), "xyxy": b, "conf": float(s)}
                     for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            cam, tod = slice_of(img_path)
            # class-aware matching per class
            for c in range(nc):
                pc = [p for p in preds if p["cls"] == c]
                gc = [g for g in gts if g["cls"] == c]
                tp = len(greedy_match(pc, gc))
                per_class[c].update(tp=tp, fp=len(pc) - tp, fn=len(gc) - tp)
                for key in (f"cam:{cam}", f"tod:{tod}"):
                    slices[key].update(tp=tp, fp=len(pc) - tp, fn=len(gc) - tp)
            # class-agnostic matching for confusion / size recall
            pairs = greedy_match(preds, gts)
            matched_p = {p for p, _ in pairs}
            matched_g = {g for _, g in pairs}
            for p, g in pairs:
                confusion[gts[g]["cls"]][preds[p]["cls"]] += 1
            for gi, g in enumerate(gts):
                if gi not in matched_g:
                    confusion[g["cls"]][nc] += 1
                b = size_bucket(g["xyxy"], w, h)
                size_tot[b] += 1
                size_hit[b] += gi in matched_g
            for pi, p in enumerate(preds):
                if pi not in matched_p:
                    confusion[nc][p["cls"]] += 1
            agn.update(tp=len(pairs), fp=len(preds) - len(pairs), fn=len(gts) - len(pairs), frames=1)
    return {"per_class": per_class, "confusion": confusion, "size_tot": size_tot, "size_hit": size_hit,
            "slices": slices, "agnostic": agn}


def prf(c: Counter) -> Tuple[float, float, float]:
    p = c["tp"] / max(c["tp"] + c["fp"], 1)
    r = c["tp"] / max(c["tp"] + c["fn"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def evaluate(models: List[Path], data_yaml: Path, conf: float, imgsz: int) -> Dict:
    from ultralytics import YOLO
    images = val_images(data_yaml)
    if not images:
        raise SystemExit(f"no val images found for {data_yaml}")
    out = {"data": str(data_yaml), "n_images": len(images), "conf": conf, "imgsz": imgsz, "models": {}}
    for mp in models:
        print(f"[eval] {mp}")
        model = YOLO(str(mp))
        if list(model.names.values()) != CLASS_NAMES:
            raise SystemExit(f"{mp} predicts {len(model.names)} classes ({list(model.names.values())[:6]}...), "
                             f"not our 5 ({CLASS_NAMES}) - only Thai 5-class checkpoints can be compared")
        v = model.val(data=str(data_yaml), imgsz=imgsz, conf=0.001, batch=8, plots=False, verbose=False, split="val")
        ap50 = {CLASS_NAMES[int(c)]: float(v.box.all_ap[i, 0]) for i, c in enumerate(v.box.ap_class_index)}
        op = operational(model, images, conf, imgsz)
        out["models"][str(mp)] = {"map50": float(v.box.map50), "map": float(v.box.map), "ap50": ap50, "op": op}
    return out


def write_report(res: Dict, path: Path) -> None:
    names = list(res["models"])
    short = [f"{Path(n).parent.parent.name}/{Path(n).stem}" if Path(n).parent.name == "weights" else Path(n).stem
             for n in names]
    hdr = "| Metric | " + " | ".join(short) + " |\n|---|" + "---:|" * len(short)
    L = [f"# Manual-label evaluation\n",
         f"- Data: `{res['data']}` ({res['n_images']} val images)",
         f"- Operational threshold: conf={res['conf']}, IoU>={IOU_THR}, imgsz={res['imgsz']}",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n",
         "## Headline\n", hdr]
    M = [res["models"][n] for n in names]

    def row(label, vals):
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    row("mAP50", [f"{m['map50']:.4f}" for m in M])
    row("mAP50-95", [f"{m['map']:.4f}" for m in M])
    tot = []
    for m in M:
        c = Counter()
        for pc in m["op"]["per_class"].values():
            c.update(pc)
        tot.append(c)
    row("Precision (class-aware)", [pct(prf(c)[0]) for c in tot])
    row("Recall (class-aware)", [pct(prf(c)[1]) for c in tot])
    row("F1 (class-aware)", [pct(prf(c)[2]) for c in tot])
    row("Recall (any class, vehicle found)", [pct(prf(m["op"]["agnostic"])[1]) for m in M])
    row("False positives / frame", [f"{m['op']['agnostic']['fp'] / max(m['op']['agnostic']['frames'], 1):.2f}" for m in M])
    for b in ("small", "medium", "large"):
        row(f"Recall {b} (n={M[0]['op']['size_tot'][b]})",
            [pct(m["op"]["size_hit"][b] / max(m["op"]["size_tot"][b], 1)) for m in M])

    L += ["\n## Per class\n", hdr]
    for c, name in enumerate(CLASS_NAMES):
        n_gt = M[0]["op"]["per_class"][c]["tp"] + M[0]["op"]["per_class"][c]["fn"]
        row(f"{name} AP50", [f"{m['ap50'].get(name, 0):.3f}" for m in M])
        row(f"{name} P / R (n={n_gt})", [f"{pct(prf(m['op']['per_class'][c])[0])} / {pct(prf(m['op']['per_class'][c])[1])}" for m in M])

    L += ["\n## Pickup problem: car <-> truck confusion (matched boxes)\n", hdr]
    row("GT car predicted truck", [str(m["op"]["confusion"][CAR][TRUCK]) for m in M])
    row("GT truck predicted car", [str(m["op"]["confusion"][TRUCK][CAR]) for m in M])

    L += ["\n## Slices (class-aware recall / precision)\n", hdr]
    keys = sorted({k for m in M for k in m["op"]["slices"]})
    for k in keys:
        row(k.replace("cam:", "camera ").replace("tod:", ""),
            [f"{pct(prf(m['op']['slices'][k])[1])} / {pct(prf(m['op']['slices'][k])[0])}" for m in M])

    for n, s, m in zip(names, short, M):
        L += [f"\n## Confusion matrix: {s}\n", f"`{n}`\n",
              "| GT \\ Pred | " + " | ".join(CLASS_NAMES) + " | missed |", "|---|" + "---:|" * (len(CLASS_NAMES) + 1)]
        for r, rname in enumerate(CLASS_NAMES + ["background (FP)"]):
            L.append(f"| {rname} | " + " | ".join(str(v) for v in m["op"]["confusion"][r]) + " |")
    L.append("\nNote: cam45_northeast was never in the baseline's training data; the new models may train on it. "
             "Compare the per-camera rows before reading too much into the overall delta.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", nargs="+", type=Path, default=[Path("models/yolo26s_thai_traffic.pt")])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, help="report path (default runs/manual_eval/<time>/report.md)")
    args = ap.parse_args()

    out = resolve(args.out) if args.out else REPO_ROOT / "runs" / "manual_eval" / datetime.now().strftime("%Y%m%d_%H%M%S") / "report.md"
    res = evaluate([resolve(m) for m in args.models], resolve(args.data), args.conf, args.imgsz)
    write_report(res, out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))  # Counters serialise as plain dicts
    print(f"\nreport: {out}")
    print(out.read_text(encoding="utf-8").split("## Per class")[0])


if __name__ == "__main__":
    main()
