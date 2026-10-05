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
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        import run_multi_camera
        sys.exit(run_multi_camera.main(sys.argv[2:]))

    parser = argparse.ArgumentParser(
        description="Smart Traffic Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Multi-Camera Streaming Runner. Its options live in run_multi_camera.build_pipeline_args();
    # "main.py run ..." is forwarded there unchanged (see the top of main()).
    subparsers.add_parser(
        "run",
        help="Run the multi-camera measurement pipeline (python main.py run --help for options)",
        add_help=False,
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

    # 3. Interactive Lane Polygon & Counting Gate Calibration
    calib_parser = subparsers.add_parser(
        "calibrate",
        aliases=["segment"],
        help="Launch interactive lane polygon and virtual counting gate calibration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    calib_parser.add_argument(
        "--direction",
        "--approach",
        type=str,
        dest="direction",
        choices=["north", "south", "east", "west", "northeast", "custom"],
        default=None,
        help="Target approach / camera direction (automatically resolves video and config files)",
    )
    calib_parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file or RTSP stream URL",
    )
    calib_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to target camera configuration JSON file to update",
    )
    calib_parser.add_argument(
        "--sec",
        type=float,
        default=0.0,
        help="Timestamp in seconds to grab initial video frame",
    )

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "benchmark":
        import benchmark_hardware
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        benchmark_hardware.main()

    elif args.command in ("calibrate", "segment"):
        from tools.segmentor import run_calibration
        run_calibration(
            direction=args.direction,
            video=args.video,
            config=args.config,
            sec=args.sec,
        )


if __name__ == "__main__":
    main()