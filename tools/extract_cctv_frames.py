"""
Extract frames from real CCTV traffic videos to augment the 'clear' weather class.
Samples diverse frames across multiple cameras to bridge the domain gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def extract_cctv_clear_frames(
    videos_dir: Path,
    output_dir: Path,
    samples_per_cam: int = 100,
    train_ratio: float = 0.8,
) -> None:
    train_clear_dir = output_dir / "train" / "clear"
    val_clear_dir = output_dir / "val" / "clear"
    train_clear_dir.mkdir(parents=True, exist_ok=True)
    val_clear_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(videos_dir.glob("*.avi")) + sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in: {videos_dir.resolve()}")

    print("=" * 60)
    print("      EXTRACTING CCTV FRAMES FOR CLEAR WEATHER CLASS")
    print("=" * 60)
    print(f"[*] Found {len(video_files)} video sources in: {videos_dir.resolve()}")
    print(f"[*] Samples per camera: {samples_per_cam}")
    print(f"[*] Train/Val Split:    {train_ratio:.0%} / {1 - train_ratio:.0%}")
    print("=" * 60)

    total_train = 0
    total_val = 0

    for v_idx, v_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(v_path.resolve()))
        if not cap.isOpened():
            print(f"[!] Warning: Unable to open {v_path.name}, skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Pick frames evenly across the first 30,000 frames (approx 3-5 minutes)
        max_search_frame = min(total_frames, 30000)
        step = max(1, max_search_frame // samples_per_cam)
        
        cam_tag = v_path.stem
        cam_train = 0
        cam_val = 0
        n_train_target = int(samples_per_cam * train_ratio)

        for i in range(samples_per_cam):
            target_pos = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            is_train = (i < n_train_target)
            dest_folder = train_clear_dir if is_train else val_clear_dir
            out_filename = f"cctv_{cam_tag}_f{target_pos:06d}.jpg"
            dest_path = dest_folder / out_filename

            cv2.imwrite(str(dest_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if is_train:
                cam_train += 1
            else:
                cam_val += 1

        cap.release()
        total_train += cam_train
        total_val += cam_val
        print(f"[{v_idx}/{len(video_files)}] {v_path.name:<20} -> {cam_train} train, {cam_val} val frames")

    print("\n" + "=" * 60)
    print("            CCTV EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"[✔] Total New CCTV Train Frames: {total_train}")
    print(f"[✔] Total New CCTV Val Frames:   {total_val}")
    print(f"[✔] Saved into: {output_dir.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from CCTV videos for clear class.")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="videos",
        help="Directory containing traffic camera videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/weather_dataset",
        help="Root dataset directory containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--samples-per-cam",
        type=int,
        default=100,
        help="Number of frames to extract per video camera (default: 100).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of frames allocated to training (default: 0.8).",
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent

    videos_dir = Path(args.videos_dir) if Path(args.videos_dir).is_absolute() else project_root / args.videos_dir
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir

    extract_cctv_clear_frames(
        videos_dir=videos_dir,
        output_dir=output_dir,
        samples_per_cam=args.samples_per_cam,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
