"""
Weather & Road Condition Video Test Tool.
Evaluates arbitrary traffic videos using the production WeatherClassifier engine.
Supports single-video analysis and batch verification across all clips in videos/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import cv2

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from algorithm.weather_classifier import WeatherClassifier


def evaluate_single_video(
    video_path: Path,
    classifier: WeatherClassifier,
    output_path: Path | None = None,
    snapshot_path: Path | None = None,
    max_frames: int = 250,
) -> dict:
    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(output_path.resolve()), fourcc, min(fps, 30.0), (width, height))

    frame_count = 0
    clear_evals, rainy_evals = 0, 0
    t_start = time.time()
    last_res = None
    snapshot_saved = False

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1
        res = classifier.update(frame, frame_idx=frame_count)
        last_res = res

        if frame_count % classifier.check_interval == 0 or frame_count == 1:
            if res.raw_class == "rainy":
                rainy_evals += 1
            else:
                clear_evals += 1

        annotated = classifier.draw_hud(frame, res, show_roi=True, camera_id=video_path.name)

        if out_writer is not None:
            out_writer.write(annotated)

        if snapshot_path is not None and not snapshot_saved and frame_count >= 30:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_path.resolve()), annotated)
            snapshot_saved = True

    cap.release()
    if out_writer is not None:
        out_writer.release()

    total_checks = clear_evals + rainy_evals
    return {
        "video": video_path.name,
        "frames": frame_count,
        "total_video_frames": total_frames,
        "time_s": time.time() - t_start,
        "final_state": last_res.state if last_res else "UNKNOWN",
        "lane_mode": last_res.lane_mode if last_res else "UNKNOWN",
        "clear_pct": (clear_evals / total_checks * 100) if total_checks else 0.0,
        "rainy_pct": (rainy_evals / total_checks * 100) if total_checks else 0.0,
        "confidence": last_res.confidence if last_res else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test WeatherClassifier on traffic video clips.")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to specific video file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all video clips found in videos/ directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/weather_yolov8n_cls.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Max frames to process per video (default: 200).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output video.",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Path to save annotated snapshot frame.",
    )
    args = parser.parse_args()

    videos_dir = project_root / "videos"
    model_p = Path(args.model) if Path(args.model).is_absolute() else project_root / args.model

    classifier = WeatherClassifier(
        model_path=model_p,
        auto_road_roi=True,
        top_crop_ratio=0.30,
        check_interval=5,
    )

    if args.all or (args.video is None):
        video_files = sorted(list(set(videos_dir.rglob("*.avi")) | set(videos_dir.rglob("*.mp4"))))
        print("=" * 70)
        print("      BATCH EVALUATION ON ALL TRAFFIC CLIPS (PRODUCTION ENGINE)")
        print("=" * 70)
        results = []
        for v in video_files:
            out_p = project_root / "output" / f"eval_{v.stem}.mp4"
            snap_p = project_root / "output" / f"eval_{v.stem}_snap.jpg"
            # Reset history for each video
            classifier.history.clear()
            classifier.current_state = "CLEAR"
            r = evaluate_single_video(v, classifier, output_path=out_p, snapshot_path=snap_p, max_frames=args.max_frames)
            results.append(r)
            status_symbol = "🌧️" if r["final_state"] == "RAINY" else "☀️"
            print(f"[{status_symbol}] {r['video']:<45} -> {r['final_state']:<7} ({r['confidence']*100:.1f}%) | Lane: {r['lane_mode']}")

        print("\n" + "=" * 70)
        print("                    EVALUATION SUMMARY TABLE")
        print("=" * 70)
        print(f"{'Video Source':<42} | {'State':<7} | {'Conf':<6} | {'Lane Mode':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['video']:<42} | {r['final_state']:<7} | {r['confidence']*100:5.1f}% | {r['lane_mode']:<15}")
        print("=" * 70)
    else:
        vpath = Path(args.video) if Path(args.video).is_absolute() else project_root / args.video
        out_p = Path(args.output) if args.output else project_root / "output" / f"eval_{vpath.stem}.mp4"
        snap_p = Path(args.snapshot) if args.snapshot else project_root / "output" / f"eval_{vpath.stem}_snap.jpg"
        print(f"[*] Running single video evaluation on: {vpath.name}")
        r = evaluate_single_video(vpath, classifier, output_path=out_p, snapshot_path=snap_p, max_frames=args.max_frames)
        print("\n" + "=" * 60)
        print("                 EVALUATION RESULT")
        print("=" * 60)
        print(f"Video Source:       {r['video']}")
        print(f"Frames Analyzed:    {r['frames']}")
        print(f"Final Weather State:{r['final_state']}")
        print(f"Lane Mode Switch:   {r['lane_mode']}")
        print(f"Model Confidence:   {r['confidence']*100:.1f}%")
        print(f"[✔] Output Video:   {out_p.resolve()}")
        print(f"[✔] Snapshot Frame: {snap_p.resolve()}")
        print("=" * 60)


if __name__ == "__main__":
    main()
