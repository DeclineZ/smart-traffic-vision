"""
Smart Traffic Vision - Unified Command Line Interface.
Central entry point for multi-camera streaming, hardware benchmarking, and lane polygon calibration.
"""

from __future__ import annotations

import argparse
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Smart Traffic Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Multi-Camera Streaming Runner
    run_parser = subparsers.add_parser(
        "run",
        help="Run multi-camera real-time traffic tracking & MQTT streaming pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--num-cams",
        type=int,
        default=4,
        help="Number of camera feeds to run (1 to 8+)",
    )
    run_parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Named camera feeds to run: north south east west all",
    )
    run_parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Custom JSON configuration file paths per camera feed",
    )
    run_parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Custom video file paths or RTSP stream URLs",
    )
    run_parser.add_argument(
        "--display",
        action="store_true",
        help="Display live multi-camera HUD window",
    )
    run_parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model weights or engine path",
    )
    run_parser.add_argument(
        "--device",
        default=None,
        help="Inference device: 'cuda', 'cuda:0', 'cpu' (default: auto)",
    )
    run_parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold",
    )
    run_parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target processing frame rate per stream",
    )
    run_parser.add_argument(
        "--pub-interval",
        type=float,
        default=2.0,
        help="MQTT broadcast interval in seconds",
    )
    run_parser.add_argument(
        "--mqtt-broker",
        default="mqtt://localhost:1883",
        help="MQTT broker URL",
    )
    run_parser.add_argument(
        "--mqtt-topic",
        default="traffic/counts",
        help="MQTT destination topic",
    )
    run_parser.add_argument(
        "--intersection-id",
        default="INT-001",
        help="Intersection identifier string",
    )

    # 2. Hardware Benchmarking & Sizing Suite
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run comprehensive multi-camera hardware profiling & sizing benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bench_parser.add_argument(
        "--streams",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Camera feed counts to test sequentially",
    )
    bench_parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov8s.pt"],
        help="YOLO model checkpoint files to benchmark",
    )
    bench_parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds per individual benchmark run",
    )
    bench_parser.add_argument(
        "--target-cams",
        type=int,
        default=8,
        help="Target number of camera feeds for sizing evaluation",
    )
    bench_parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target frame rate per camera feed",
    )
    bench_parser.add_argument(
        "--mode",
        choices=["threaded", "batched", "both"],
        default="threaded",
        help="Vision pipeline architecture mode",
    )
    bench_parser.add_argument(
        "--frame-skips",
        nargs="+",
        type=int,
        default=[0],
        help="Frame skipping intervals (0 = every frame, 1 = every 2nd, 2 = every 3rd)",
    )
    bench_parser.add_argument(
        "--display",
        action="store_true",
        help="Open live multi-camera preview window to measure rendering overhead",
    )
    bench_parser.add_argument(
        "--unpaced",
        action="store_true",
        help="Disable stream pacing for uncapped maximum throughput stress testing",
    )
    bench_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    bench_parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Custom list of video file paths or RTSP URLs to stream",
    )
    bench_parser.add_argument(
        "--out-dir",
        default="benchmark/hardware-results",
        help="Directory to save JSON, CSV, Markdown, and chart reports",
    )
    bench_parser.add_argument(
        "--keep-latest",
        type=int,
        default=3,
        help="Maximum historical benchmark result sets to retain",
    )
    bench_parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate 4-panel analysis charts (PNG)",
    )

    # 3. Interactive Lane Polygon Calibration
    calib_parser = subparsers.add_parser(
        "calibrate",
        aliases=["segment"],
        help="Launch interactive lane polygon segmentor & perspective calibration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    calib_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file or RTSP stream URL to extract calibration frame",
    )
    calib_parser.add_argument(
        "--sec",
        type=float,
        default=0.0,
        help="Timestamp in seconds to grab video frame",
    )
    calib_parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of consecutive lane polygons to calibrate",
    )

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "run":
        import run_multi_camera
        # Delegate directly
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_multi_camera.main()

    elif args.command == "benchmark":
        import benchmark_hardware
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        benchmark_hardware.main()

    elif args.command in ("calibrate", "segment"):
        from tools.segmentor import RoadSegmenter
        for lane_idx in range(args.n):
            print(f"\n--- Calibrating Lane {lane_idx + 1} of {args.n} ---")
            segmentor = RoadSegmenter(video_path=args.video, time_in_seconds=args.sec)
            segmentor.segment()


if __name__ == "__main__":
    main()