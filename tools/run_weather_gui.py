"""
Interactive Smart Traffic Weather Detection Simulation GUI.
Uses the production-ready WeatherClassifier engine with Auto-Road Focus
and real-time Lane Detection Mode switching.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2

# Add project root to sys.path to allow imports from algorithm
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algorithm.weather_classifier import WeatherClassifier


def run_interactive_simulation(
    video_path: Path,
    model_path: Path,
    check_interval: int = 5,
    window_size: int = 10,
    top_crop_ratio: float = 0.30,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print("=" * 60)
    print("   STARTING PRODUCTION SMART TRAFFIC WEATHER SIMULATION")
    print("=" * 60)
    print(f"[*] Video Source:      {video_path.name}")
    print(f"[*] Engine Weights:    {model_path.name}")
    print(f"[*] Top-Crop Road ROI: Discard top {top_crop_ratio*100:.0f}% sky/buildings")
    print(f"[*] Check Interval:    Every {check_interval} frames")
    print("=" * 60)

    # Initialize production engine
    classifier = WeatherClassifier(
        model_path=model_path,
        check_interval=check_interval,
        window_size=window_size,
        auto_road_roi=True,
        top_crop_ratio=top_crop_ratio,
    )

    cap = cv2.VideoCapture(str(video_path.resolve()))
    window_title = f"Smart Traffic Vision - Road Condition Monitor [{video_path.name}]"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    is_paused = False
    show_roi = True
    frame_idx = 0

    print("\n[Controls] [Space]=Pause/Resume  [R]=Toggle Road ROI Line  [Q/Esc]=Quit\n")

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video seamlessly
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            # Run production classification engine
            result = classifier.update(frame, frame_idx=frame_idx)

            # Render production HUD overlay
            display_frame = classifier.draw_hud(
                frame=frame,
                result=result,
                show_roi=show_roi,
                camera_id=video_path.name,
            )

            cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1 if not is_paused else 50) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == 32:  # Spacebar
            is_paused = not is_paused
        elif key in (ord('r'), ord('R')):  # Toggle Road ROI display
            show_roi = not show_roi

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Simulation terminated gracefully.")


def main():
    parser = argparse.ArgumentParser(description="Run interactive Smart Traffic Weather Simulation GUI.")
    parser.add_argument(
        "--video",
        type=str,
        default="videos/flash-flood-traffic-in-bangkok.mp4",
        help="Path to video file to monitor.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/weather_yolov8n_cls.pt",
        help="Path to trained YOLOv8 classification model weights.",
    )
    parser.add_argument(
        "--top-crop",
        type=float,
        default=0.30,
        help="Fraction of upper frame to discard to isolate road surface (default 0.30 = top 30%%).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Frame evaluation interval (default: 5).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    video_p = Path(args.video) if Path(args.video).is_absolute() else project_root / args.video
    model_p = Path(args.model) if Path(args.model).is_absolute() else project_root / args.model

    run_interactive_simulation(
        video_path=video_p,
        model_path=model_p,
        check_interval=args.interval,
        top_crop_ratio=args.top_crop,
    )


if __name__ == "__main__":
    main()
