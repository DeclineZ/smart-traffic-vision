"""
External CCTV & Weather Dataset Ingestion Pipeline.
Automates scanning, sampling, and balancing external datasets such as
RainCityscapes, AAU RainSnow, and external CCTV surveillance frames.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import cv2


def sample_and_copy(
    image_paths: list[Path],
    dest_train: Path,
    dest_val: Path,
    prefix: str,
    max_samples: int = 150,
    train_ratio: float = 0.8,
) -> tuple[int, int]:
    """Samples up to max_samples images and copies them to train/val directories."""
    dest_train.mkdir(parents=True, exist_ok=True)
    dest_val.mkdir(parents=True, exist_ok=True)

    selected = random.sample(image_paths, min(len(image_paths), max_samples))
    n_train = int(len(selected) * train_ratio)
    train_count, val_count = 0, 0

    for i, src in enumerate(selected):
        is_train = i < n_train
        target_dir = dest_train if is_train else dest_val
        out_name = f"{prefix}_{src.stem}_{i:04d}{src.suffix.lower()}"
        dest_path = target_dir / out_name

        try:
            shutil.copy2(src, dest_path)
            if is_train:
                train_count += 1
            else:
                val_count += 1
        except Exception as e:
            print(f"[!] Warning: failed to copy {src.name}: {e}")

    return train_count, val_count


def ingest_rain_cityscapes(
    dataset_dir: Path,
    target_dataset_dir: Path,
    max_samples_per_class: int = 150,
) -> None:
    """
    Ingests RainCityscapes dataset.
    Scans for rainy images and corresponding clear/dry cityscapes images.
    """
    print(f"[*] Scanning RainCityscapes directory: {dataset_dir.resolve()}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"RainCityscapes directory not found at: {dataset_dir}")

    # Search for all image files
    all_imgs = (
        list(dataset_dir.rglob("*.png"))
        + list(dataset_dir.rglob("*.jpg"))
        + list(dataset_dir.rglob("*.jpeg"))
    )
    print(f"[*] Total images found in RainCityscapes: {len(all_imgs)}")

    # In RainCityscapes, filenames often contain 'rain' or are organized in 'rain' vs 'clear'
    rain_imgs = [p for p in all_imgs if "rain" in p.name.lower() or "rain" in str(p.parent).lower()]
    clear_imgs = [p for p in all_imgs if p not in rain_imgs]

    print(f"    - Detected Rainy images: {len(rain_imgs)}")
    print(f"    - Detected Clear images: {len(clear_imgs)}")

    train_clear = target_dataset_dir / "train" / "clear"
    val_clear = target_dataset_dir / "val" / "clear"
    train_rain = target_dataset_dir / "train" / "rainy"
    val_rain = target_dataset_dir / "val" / "rainy"

    random.seed(42)

    # Ingest Rainy
    if rain_imgs:
        tr_r, val_r = sample_and_copy(
            rain_imgs,
            train_rain,
            val_rain,
            prefix="cityscapes_rain",
            max_samples=max_samples_per_class,
        )
        print(f"[✔] Ingested Rainy: {tr_r} train, {val_r} val")

    # Ingest Clear to preserve 50:50 balance
    if clear_imgs:
        tr_c, val_c = sample_and_copy(
            clear_imgs,
            train_clear,
            val_clear,
            prefix="cityscapes_clear",
            max_samples=max_samples_per_class,
        )
        print(f"[✔] Ingested Clear: {tr_c} train, {val_c} val")


def main():
    parser = argparse.ArgumentParser(description="Ingest external weather/CCTV datasets.")
    parser.add_argument("--source", type=str, required=True, help="Path to external dataset folder.")
    parser.add_argument("--target-dataset", type=str, default="data/weather_dataset", help="Target dataset path.")
    parser.add_argument("--max-samples", type=int, default=150, help="Max samples per class.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    src_p = Path(args.source) if Path(args.source).is_absolute() else project_root / args.source
    target_p = Path(args.target_dataset) if Path(args.target_dataset).is_absolute() else project_root / args.target_dataset

    ingest_rain_cityscapes(src_p, target_p, max_samples_per_class=args.max_samples)


if __name__ == "__main__":
    main()
