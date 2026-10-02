#!/usr/bin/env python3
"""
Train YOLO26s on the manually labeled dataset, then compare against the deployed baseline.

Two starting points (run one or both):
  baseline  fine-tune models/yolo26s_thai_traffic.pt on the clean labels. Keeps what the
            baseline already knows and corrects it; the most likely winner with 300 frames.
  coco      train from the COCO-pretrained models/yolo26s.pt using only the clean labels.
            Tells you how far clean data alone gets you (no noisy old labels at all).

Weights land in runs/train/<name>/weights/{best,last}.pt. Nothing in models/ is overwritten;
promote a candidate by hand once the report says it wins.

Usage:
  python tools/manual_label/train.py --data data/manual_v1_dataset/data.yaml                 # both inits
  python tools/manual_label/train.py --data data/manual_v1_dataset/data.yaml --init baseline
  python tools/manual_label/train.py --data data/manual_v1_dataset/data.yaml --imgsz 960      # small-object experiment
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import REPO_ROOT

INITS = {
    "baseline": dict(weights="models/yolo26s_thai_traffic.pt", epochs=80, optimizer="SGD", lr0=0.002, warmup_epochs=2.0),
    "coco": dict(weights="models/yolo26s.pt", epochs=150, optimizer="auto", lr0=0.01, warmup_epochs=3.0),
}


def train_one(init: str, data: Path, name: str, args) -> Path:
    from ultralytics import YOLO
    cfg = INITS[init]
    model = YOLO(str(REPO_ROOT / cfg["weights"]))
    model.train(
        data=str(data),
        epochs=args.epochs or cfg["epochs"],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=2,
        project=str(REPO_ROOT / "runs" / "train"),
        name=name,
        exist_ok=True,
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        cos_lr=True,
        warmup_epochs=cfg["warmup_epochs"],
        patience=0,
        close_mosaic=10,
        mosaic=1.0,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        copy_paste=0.0,
        mixup=0.0,
        seed=args.seed,
        plots=True,
        verbose=True,
    )
    return REPO_ROOT / "runs" / "train" / name / "weights"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--init", choices=["baseline", "coco", "both"], default="both")
    ap.add_argument("--name", default=None, help="run name prefix (default: <dataset dir name>)")
    ap.add_argument("--epochs", type=int, default=None, help="override (default 80 for baseline, 150 for coco)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower to 8 if you run out of GPU memory (e.g. at --imgsz 960)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="skip the comparison report after training")
    args = ap.parse_args()

    data = (args.data if args.data.is_absolute() else REPO_ROOT / args.data).resolve()
    prefix = args.name or data.parent.name.replace("_dataset", "")
    inits = ["baseline", "coco"] if args.init == "both" else [args.init]
    candidates = []
    for init in inits:
        name = f"{prefix}_{init}" + (f"_{args.imgsz}" if args.imgsz != 640 else "")
        print(f"\n===== training {name} from {INITS[init]['weights']} =====")
        weights = train_one(init, data, name, args)
        candidates += [weights / "best.pt", weights / "last.pt"]

    if args.no_eval:
        return
    from tools.manual_label.evaluate import evaluate, write_report
    report = REPO_ROOT / "runs" / "manual_eval" / f"{prefix}_report.md"
    models = [REPO_ROOT / "models" / "yolo26s_thai_traffic.pt"] + [c for c in candidates if c.exists()]
    write_report(evaluate(models, data, conf=0.25, imgsz=args.imgsz), report)
    print(f"\ncomparison report: {report}")
    print(report.read_text(encoding="utf-8").split("## Per class")[0])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
