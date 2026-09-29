"""
Extract and integrate dry traffic jam frames and new wet/rain traffic frames into dataset.
Solves domain imbalance where dense Bangkok traffic scenes only existed in the rainy class.
"""

from __future__ import annotations

from pathlib import Path
import cv2


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    class_name: str,
    prefix: str,
    num_frames: int = 80,
    train_ratio: float = 0.8,
) -> tuple[int, int]:
    print(f"[*] Extracting {num_frames} frames from: {video_path.name} -> class: {class_name}")
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    train_dir = output_dir / "train" / class_name
    val_dir = output_dir / "val" / class_name
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
        dest_name = f"{prefix}_f{pos:04d}.jpg"
        cv2.imwrite(str(target_folder / dest_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if is_train:
            train_count += 1
        else:
            val_count += 1

    cap.release()
    print(f"[✔] Saved {train_count} train and {val_count} val frames for {video_path.name}")
    return train_count, val_count


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_p = project_root / "data" / "weather_dataset"

    # 1. Dry Bangkok traffic jam (counter-example for Bangkok taxis/buses in dry conditions)
    dry_jam_video = project_root / "videos" / "dry" / "gettyimages-151939150-640_adpp.mp4"
    if dry_jam_video.exists():
        extract_video_frames(
            dry_jam_video,
            dataset_p,
            class_name="clear",
            prefix="dry_jam_bkk",
            num_frames=100,  # 80 train, 20 val
        )

    # 2. Wet Bangkok street scene with bus, songthaew, umbrellas, wet asphalt
    wet_street_video = project_root / "videos" / "wet-flood" / "gettyimages-1203092299-640_adpp.mp4"
    if wet_street_video.exists():
        extract_video_frames(
            wet_street_video,
            dataset_p,
            class_name="rainy",
            prefix="wet_street_bkk",
            num_frames=50,   # 40 train, 10 val
        )

    # 3. Wet night asphalt with water reflections
    wet_night_video = project_root / "videos" / "wet-flood" / "gettyimages-1454390722-640_adpp.mp4"
    if wet_night_video.exists():
        extract_video_frames(
            wet_night_video,
            dataset_p,
            class_name="rainy",
            prefix="wet_night",
            num_frames=50,   # 40 train, 10 val
        )

    # Summary table
    all_train_rain = len(list((dataset_p / "train" / "rainy").glob("*.jpg")))
    all_train_clear = len(list((dataset_p / "train" / "clear").glob("*.jpg")))
    all_val_rain = len(list((dataset_p / "val" / "rainy").glob("*.jpg")))
    all_val_clear = len(list((dataset_p / "val" / "clear").glob("*.jpg")))

    print("\n" + "=" * 60)
    print("             UPDATED DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Split':<10} | {'Rainy/Wet':<15} | {'Clear/Dry':<12} | {'Total':<10}")
    print("-" * 60)
    print(f"{'train':<10} | {all_train_rain:<15} | {all_train_clear:<12} | {all_train_rain + all_train_clear:<10}")
    print(f"{'val':<10} | {all_val_rain:<15} | {all_val_clear:<12} | {all_val_rain + all_val_clear:<10}")
    print("-" * 60)
    print(f"{'TOTAL':<10} | {all_train_rain + all_val_rain:<15} | {all_train_clear + all_val_clear:<12} | "
          f"{all_train_rain + all_val_rain + all_train_clear + all_val_clear:<10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
