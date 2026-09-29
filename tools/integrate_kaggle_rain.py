"""
Integrate bshaurya/rain-dataset from Kaggle into our weather dataset.
Adds 300 rain streak images and 100 norain images to boost model robustness.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def integrate_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    random.seed(seed)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir.resolve()}")

    files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    rain_files = [f for f in files if "rain" in f.name.lower() and "norain" not in f.name.lower()]
    norain_files = [f for f in files if "norain" in f.name.lower()]

    print("=" * 60)
    print("      INTEGRATING KAGGLE RAIN DATASET")
    print("=" * 60)
    print(f"[*] Source: {source_dir.resolve()}")
    print(f"[*] Found Rain images:   {len(rain_files)}")
    print(f"[*] Found Norain images: {len(norain_files)}")
    print("=" * 60)

    random.shuffle(rain_files)
    random.shuffle(norain_files)

    def add_to_split(image_list: List[Path], class_name: str) -> Tuple[int, int]:
        n_train = int(len(image_list) * train_ratio)
        train_list = image_list[:n_train]
        val_list = image_list[n_train:]

        train_dest = target_dir / "train" / class_name
        val_dest = target_dir / "val" / class_name
        train_dest.mkdir(parents=True, exist_ok=True)
        val_dest.mkdir(parents=True, exist_ok=True)

        for img in train_list:
            shutil.copy2(img, train_dest / f"kgl_{class_name}_{img.name}")
        for img in val_list:
            shutil.copy2(img, val_dest / f"kgl_{class_name}_{img.name}")

        return len(train_list), len(val_list)

    rain_train, rain_val = add_to_split(rain_files, "rainy")
    clear_train, clear_val = add_to_split(norain_files, "clear")

    print(f"[✔] Added to 'rainy': {rain_train} train, {rain_val} val images")
    print(f"[✔] Added to 'clear': {clear_train} train, {clear_val} val images")

    # Count total images now
    all_train_rain = len(list((target_dir / "train" / "rainy").glob("*")))
    all_train_clear = len(list((target_dir / "train" / "clear").glob("*")))
    all_val_rain = len(list((target_dir / "val" / "rainy").glob("*")))
    all_val_clear = len(list((target_dir / "val" / "clear").glob("*")))

    print("\n" + "=" * 60)
    print("           UPDATED DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Split':<10} | {'Rainy':<10} | {'Clear':<10} | {'Total':<10}")
    print("-" * 60)
    print(f"{'train':<10} | {all_train_rain:<10} | {all_train_clear:<10} | {all_train_rain + all_train_clear:<10}")
    print(f"{'val':<10} | {all_val_rain:<10} | {all_val_clear:<10} | {all_val_rain + all_val_clear:<10}")
    print("-" * 60)
    print(f"{'TOTAL':<10} | {all_train_rain + all_val_rain:<10} | {all_train_clear + all_val_clear:<10} | "
          f"{all_train_rain + all_val_rain + all_train_clear + all_val_clear:<10}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Integrate Kaggle rain dataset into weather dataset.")
    parser.add_argument(
        "--source",
        type=str,
        default="/Users/febbuary/.cache/kagglehub/datasets/bshaurya/rain-dataset/versions/1",
        help="Path to downloaded Kaggle rain dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/weather_dataset",
        help="Path to project weather dataset.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    source_path = Path(args.source)
    target_path = Path(args.target) if Path(args.target).is_absolute() else project_root / args.target

    integrate_dataset(source_path, target_path)


if __name__ == "__main__":
    main()
