"""
Fine-tune Ultralytics YOLO11 on Unified Thai Traffic Dataset.

Features:
- Frozen backbone features to prevent catastrophic forgetting of base COCO classes.
- Mosaic disablement in final epochs for bounding box coordinate stabilization.
- Copy-Paste augmentation (copy_paste: 0.3) to resolve extreme minority class imbalance (saleng, truck).
- Broad color and lighting augmentation (HSV desaturation/value shifts) to bridge daylight and infrared night surveillance streams.
- Direct export of best model weights to .pt and optional FP16 TensorRT .engine export for trt_pipeline.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO


def train(args):
    dataset_yaml = Path(args.data)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found at {dataset_yaml}. Run tools/compile_multiclass_dataset.py first.")

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model weights not found at {base_model_path}")

    print(f"Loading base YOLO weights: {base_model_path}...")
    model = YOLO(str(base_model_path))

    print("\nStarting YOLO Fine-Tuning:")
    print(f"  Target Run:    {args.name}")
    print(f"  Dataset:       {dataset_yaml.resolve()}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Frozen Layers: {args.freeze} (preserving general features)")
    print(f"  Close Mosaic:  {args.close_mosaic} (final {args.close_mosaic} epochs without mosaic)")
    print(f"  Copy-Paste:    {args.copy_paste} (minority class boost)")
    print(f"  Batch Size:    {args.batch}")
    print(f"  Image Size:    {args.imgsz}")
    print(f"  Device:        {args.device}")
    print(f"  Initial LR:    {args.lr0} (cosine schedule={args.cos_lr})")
    print(f"  Weight Decay:  {args.weight_decay}")
    print(f"  Patience:      {args.patience}")

    model.train(
        data=str(dataset_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        fliplr=0.5,
        scale=0.5,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True
    )

    run_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    trainer_best = getattr(getattr(model, "trainer", None), "best", None)

    best_weights = Path(trainer_best) if trainer_best and Path(trainer_best).exists() else None
    if not best_weights or not best_weights.exists():
        candidates = list(Path("runs").rglob(f"*{args.name}*/weights/best.pt"))
        if candidates:
            best_weights = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    export_path = Path(args.export)
    if best_weights and best_weights.exists():
        shutil.copy(best_weights, export_path)
        print(f"\n[OK] Training completed successfully.")
        print(f"Best model weights exported to: {export_path.resolve()}")
    else:
        print(f"Training finished. Check run artifacts in: {run_dir}")

    target_weights = export_path if export_path.exists() else best_weights
    if target_weights and target_weights.exists():
        print("\nEvaluating fine-tuned model on validation split...")
        val_model = YOLO(str(target_weights))
        metrics = val_model.val(data=str(dataset_yaml.resolve()), imgsz=args.imgsz, device=args.device, verbose=False)
        print(f"\n=======================================================")
        print(f"Validation Summary across all classes:")
        print(f"  Overall mAP50:    {metrics.box.map50:.4f}")
        print(f"  Overall mAP50-95: {metrics.box.map:.4f}")
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            for idx, name in enumerate(val_model.names.values()):
                if idx < len(metrics.box.maps):
                    print(f"  - {name:<16}: mAP50-95 = {metrics.box.maps[idx]:.4f}")
        print(f"=======================================================\n")

        if args.export_engine:
            print("Exporting model to FP16 TensorRT engine...")
            try:
                engine_path = val_model.export(format="engine", half=True, device=args.device)
                print(f"[OK] TensorRT Engine exported: {engine_path}")
            except Exception as e:
                print(f"Warning: TensorRT export failed ({e}). You can export later via trtexec or model.export().")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Ultralytics YOLO26s on Unified 5-Class Thai Traffic Dataset.")
    parser.add_argument("--data", default="data/multiclass_dataset/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--base-model", default="yolo26s.pt", help="Baseline YOLO weights (yolo26s.pt).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution.")
    parser.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze.")
    parser.add_argument("--close-mosaic", type=int, default=5, help="Epochs before end to disable mosaic augmentation.")
    parser.add_argument("--copy-paste", type=float, default=0.35, help="Copy-paste augmentation probability for rare vehicles.")
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction.")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay regularization.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 to disable).")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Use cosine learning rate scheduler.")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--project", default="runs/train", help="Run save directory.")
    parser.add_argument("--name", default="yolo26s_thai_traffic", help="Run name.")
    parser.add_argument("--export", default="models/yolo26s_thai_traffic.pt", help="Exported weights path.")
    parser.add_argument("--export-engine", action="store_true", default=False, help="Export to TensorRT engine after training.")

    args = parser.parse_args()

    if args.data is None:
        if Path("data/multiclass_dataset/data.yaml").exists():
            args.data = "data/multiclass_dataset/data.yaml"
        else:
            args.data = "data/tuktuk/dataset/data.yaml"

    train(args)


if __name__ == "__main__":
    main()
