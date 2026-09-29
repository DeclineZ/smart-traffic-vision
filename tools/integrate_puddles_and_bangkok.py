"""
Integrate Stagnant Water / Puddles dataset and Bangkok Rain Video frames
into the 'rainy/wet' class of our weather dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import cv2


def extract_bangkok_video_frames(
    video_path: Path,
    output_dir: Path,
    num_frames: int = 60,
    train_ratio: float = 0.8,
) -> tuple[int, int]:
    print(f"[*] Extracting frames from Bangkok Rain Video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    train_dir = output_dir / "train" / "rainy"
    val_dir = output_dir / "val" / "rainy"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(num_frames * train_ratio)
    train_count, val_count = 0, 0

    for i in range(num_frames):
        pos = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        is_train = (i < n_train)
        target_folder = train_dir if is_train else val_dir
        dest_name = f"bkk_rain_f{pos:04d}.jpg"
        cv2.imwrite(str(target_folder / dest_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if is_train:
            train_count += 1
        else:
            val_count += 1

    cap.release()
    print(f"[✔] Bangkok Video: Extracted {train_count} train, {val_count} val frames")
    return train_count, val_count


def integrate_puddles(
    puddles_dir: Path,
    output_dir: Path,
    sample_count: int = 400,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[int, int]:
    random.seed(seed)
    print(f"[*] Sampling puddle images from: {puddles_dir.resolve()}")
    images = [f for f in puddles_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"    Total available puddle images: {len(images)}")

    random.shuffle(images)
    chosen = images[:min(sample_count, len(images))]

    train_dir = output_dir / "train" / "rainy"
    val_dir = output_dir / "val" / "rainy"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(len(chosen) * train_ratio)
    train_list = chosen[:n_train]
    val_list = chosen[n_train:]

    for img in train_list:
        shutil.copy2(img, train_dir / f"puddle_{img.name}")
    for img in val_list:
        shutil.copy2(img, val_dir / f"puddle_{img.name}")

    print(f"[✔] Puddles: Copied {len(train_list)} train, {len(val_list)} val images")
    return len(train_list), len(val_list)


def main():
    project_root = Path(__file__).resolve().parent.parent
    bkk_video = project_root / "videos" / "bangkok_rain_sample.mp4"
    puddles_path = Path("/Users/febbuary/.cache/kagglehub/datasets/meraxes10/stagnantwaterdata/versions/2/Stagnant water and Wet surface Dataset")
    dataset_dir = project_root / "data" / "weather_dataset"

    print("=" * 60)
    print("      INTEGRATING PUDDLES & REAL BANGKOK RAIN DATA")
    print("=" * 60)

    bkk_train, bkk_val = extract_bangkok_video_frames(bkk_video, dataset_dir, num_frames=60)
    puddle_train, puddle_val = integrate_puddles(puddles_path, dataset_dir, sample_count=400)

    # Count final dataset
    all_train_rain = len(list((dataset_dir / "train" / "rainy").glob("*")))
    all_train_clear = len(list((dataset_dir / "train" / "clear").glob("*")))
    all_val_rain = len(list((dataset_dir / "val" / "rainy").glob("*")))
    all_val_clear = len(list((dataset_dir / "val" / "clear").glob("*")))

    print("\n" + "=" * 60)
    print("             FINAL UPDATED DATASET STATS")
    print("=" * 60)
    print(f"{'Split':<10} | {'Rainy/Puddles':<15} | {'Clear/Dry':<12} | {'Total':<10}")
    print("-" * 60)
    print(f"{'train':<10} | {all_train_rain:<15} | {all_train_clear:<12} | {all_train_rain + all_train_clear:<10}")
    print(f"{'val':<10} | {all_val_rain:<15} | {all_val_clear:<12} | {all_val_rain + all_val_clear:<10}")
    print("-" * 60)
    print(f"{'TOTAL':<10} | {all_train_rain + all_val_rain:<15} | {all_train_clear + all_val_clear:<12} | "
          f"{all_train_rain + all_val_rain + all_train_clear + all_val_clear:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
