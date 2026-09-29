"""
Extract and integrate flash flood video frames into rainy/wet class.
Teaches the model to recognize urban waterlogged streets under daylight conditions.
"""

from __future__ import annotations

from pathlib import Path
import cv2


def extract_flood_frames(
    video_path: Path,
    output_dir: Path,
    num_frames: int = 80,
    train_ratio: float = 0.8,
) -> tuple[int, int]:
    print(f"[*] Extracting flood frames from: {video_path.resolve()}")
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
        dest_name = f"flood_bkk_f{pos:04d}.jpg"
        cv2.imwrite(str(target_folder / dest_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if is_train:
            train_count += 1
        else:
            val_count += 1

    cap.release()
    print(f"[✔] Extracted: {train_count} train frames, {val_count} val frames")
    return train_count, val_count


def main():
    project_root = Path(__file__).resolve().parent.parent
    video_p = project_root / "videos" / "flash-flood-traffic-in-bangkok.mp4"
    dataset_p = project_root / "data" / "weather_dataset"

    extract_flood_frames(video_p, dataset_p, num_frames=80)

    # Count final dataset
    all_train_rain = len(list((dataset_p / "train" / "rainy").glob("*")))
    all_train_clear = len(list((dataset_p / "train" / "clear").glob("*")))
    all_val_rain = len(list((dataset_p / "val" / "rainy").glob("*")))
    all_val_clear = len(list((dataset_p / "val" / "clear").glob("*")))

    print("\n" + "=" * 60)
    print("             UPDATED DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Split':<10} | {'Rainy/Flood':<15} | {'Clear/Dry':<12} | {'Total':<10}")
    print("-" * 60)
    print(f"{'train':<10} | {all_train_rain:<15} | {all_train_clear:<12} | {all_train_rain + all_train_clear:<10}")
    print(f"{'val':<10} | {all_val_rain:<15} | {all_val_clear:<12} | {all_val_rain + all_val_clear:<10}")
    print("-" * 60)
    print(f"{'TOTAL':<10} | {all_train_rain + all_val_rain:<15} | {all_train_clear + all_val_clear:<12} | "
          f"{all_train_rain + all_val_rain + all_train_clear + all_val_clear:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
