"""
YOLOv8 Weather & Rain Classifier Training Script.
Trains a lightweight image classification model (YOLOv8-cls) to recognize
rainy vs clear/non-rainy environmental conditions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def get_optimal_device() -> str:
    """Detect the best hardware acceleration available (MPS, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return "0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_classifier(
    data_dir: Path,
    model_name: str = "yolov8n-cls.pt",
    epochs: int = 20,
    batch_size: int = 32,
    img_size: int = 224,
    device: str | None = None,
    project_dir: str = "runs/weather_cls",
    experiment_name: str = "yolov8n_rain_detector",
) -> Path:
    """Train YOLOv8-cls model and return the best checkpoint path."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir.resolve()}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(
            f"Dataset must contain 'train' and 'val' subdirectories: {data_dir.resolve()}"
        )

    device = device or get_optimal_device()
    print("=" * 60)
    print("       STARTING WEATHER CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"[*] Base Model:       {model_name}")
    print(f"[*] Dataset Path:     {data_dir.resolve()}")
    print(f"[*] Compute Device:   {device}")
    print(f"[*] Image Resolution: {img_size}x{img_size}")
    print(f"[*] Epochs:           {epochs}")
    print(f"[*] Batch Size:       {batch_size}")
    print("=" * 60)

    # Initialize model
    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=str(data_dir.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=experiment_name,
        save=True,
        plots=True,
        workers=2,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project_dir) / experiment_name
    best_weights = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("       TRAINING COMPLETE!")
    print("=" * 60)
    print(f"[✔] Best Model Weights: {best_weights.resolve()}")
    print(f"[✔] Training Logs & Plots: {save_dir.resolve()}")

    # Validate model
    print("\n[*] Evaluating Best Checkpoint on Validation Set...")
    val_results = model.val()
    top1 = val_results.top1 if hasattr(val_results, "top1") else "N/A"
    top5 = val_results.top5 if hasattr(val_results, "top5") else "N/A"
    print(f"[✔] Validation Top-1 Accuracy: {top1:.4f}" if isinstance(top1, float) else f"[✔] Validation Top-1 Accuracy: {top1}")

    # Copy to models directory for convenience
    models_dir = data_dir.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_weights = models_dir / "weather_yolov8n_cls.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"[✔] Copied deployment model to: {target_weights.resolve()}")

    print("=" * 60)
    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-cls weather & rain classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/weather_dataset",
        help="Path to organized dataset directory containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Pretrained YOLO classification model (e.g. yolov8n-cls.pt, yolov8s-cls.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size (default: 224).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('mps', '0', 'cpu', or leave empty for auto).",
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir

    train_classifier(
        data_dir=data_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
