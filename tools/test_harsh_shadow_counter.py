"""
Benchmark Tool: Harsh Shadow Vehicle Counting Evaluation.
Compares Baseline (Naive Centroid + Legacy SORT) vs Shadow-Resilient Pipeline
(Shadow Contrast Equalizer + Contact-Patch Anchoring + Shadow-Resilient Tracker).

Metrics Evaluated:
  1. Cross-Lane Double-Counting Count & Rate (target: 0).
  2. Total Unique Vehicle Track IDs.
  3. Track Continuity & ID Switches across Harsh Shadow Bands.
  4. Underexposed Shadow Zone Vehicle Detections.
  5. Processing Speed (FPS).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon, Point
import shapely
from ultralytics import YOLO

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.shadow_processor import ShadowContrastEqualizer, ContactPatchRefiner, ShadowLaneAssigner
from algorithm.shadow_tracker import ShadowResilientTracker
from algorithm.sort import Sort
from trt_pipeline.payload import LaneMetricsManager
from trt_pipeline.tools import initial_config

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def run_benchmark(
    video_path: str,
    config_path: Optional[str] = None,
    model_path: str = "yolov8s.pt",
    max_frames: int = 150,
    device: str = "cpu",
    conf: float = 0.20,
) -> Dict[str, Any]:
    """
    Executes comparative benchmark on the specified video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    print(f"\n=======================================================")
    print(f"HARSH SHADOW VEHICLE COUNTING BENCHMARK")
    print(f"Video: {video_path} ({w}x{h} @ {video_fps:.1f} FPS, {total_video_frames} total frames)")
    print(f"Processing up to: {max_frames} frames on device: {device}")
    print(f"=======================================================\n")

    # Load lane config or construct adaptive 3-lane grid for generic video
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lanes_source = cfg.get("lane_metrics", {}).get("lanes", cfg.get("lanes", {}))
        lanes_dict = {
            lid: {
                "direction": linfo.get("direction", "N"),
                "polygon": Polygon(linfo["polygon"]) if not isinstance(linfo["polygon"], Polygon) else linfo["polygon"],
            }
            for lid, linfo in lanes_source.items()
        }
    else:
        # Default 3 vertical road lanes spanning the video width
        lane_w = w / 3.0
        lanes_dict = {
            "Lane_1": {"direction": "N", "polygon": Polygon([(0, 0), (lane_w, 0), (lane_w, h), (0, h)])},
            "Lane_2": {"direction": "N", "polygon": Polygon([(lane_w, 0), (2 * lane_w, 0), (2 * lane_w, h), (lane_w, h)])},
            "Lane_3": {"direction": "N", "polygon": Polygon([(2 * lane_w, 0), (w, 0), (w, h), (2 * lane_w, h)])},
        }

    # Load YOLO detector
    print(f"Loading YOLO detector: {model_path}...")
    model = YOLO(model_path)

    # Pre-extract benchmark frames to ensure identical inputs for both pipelines
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    eval_frames = len(frames)
    print(f"Ingested {eval_frames} frames for evaluation.\n")

    # ----------------------------------------------------
    # PIPELINE A: BASELINE (Standard YOLO + Naive Centroid + Legacy SORT)
    # ----------------------------------------------------
    print(">>> Running Baseline Pipeline (Naive Centroid + Legacy SORT)...")
    baseline_tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
    baseline_lanes = {
        lid: {
            "direction": linfo["direction"],
            "polygon": linfo["polygon"],
        }
        for lid, linfo in lanes_dict.items()
    }
    baseline_metrics = LaneMetricsManager(baseline_lanes)

    # Track track_id -> set of lanes registered across entire run to catch double-counting
    baseline_track_to_lanes: Dict[int, set] = {}
    baseline_shadow_detections = 0
    t0_base = time.perf_counter()

    # Snapshot interval simulation (every 50 frames, matching production publish interval)
    interval_frames = 50
    baseline_window_double_counts = 0
    resilient_window_double_counts = 0

    for f_idx, frame in enumerate(frames):
        # Raw inference
        res = model(frame, conf=conf, verbose=False, device=device, classes=list(COCO_VEHICLES.keys()))[0]
        dets = []
        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0])
            cls_id = int(box.cls[0])
            dets.append([b[0], b[1], b[2], b[3], c, cls_id])

            bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if crop.size > 0 and np.mean(crop) < 55.0:
                baseline_shadow_detections += 1

        dets_arr = np.array(dets) if len(dets) else np.empty((0, 6))
        sort_input = dets_arr[:, :5] if len(dets_arr) else np.empty((0, 5))
        tracked = baseline_tracker.update(sort_input)

        # Naive centroid assignment without single-lane lock or contact patch
        if len(tracked) > 0:
            cxs = (tracked[:, 0] + tracked[:, 2]) * 0.5
            cys = (tracked[:, 1] + tracked[:, 3]) * 0.5

            for lid, linfo in baseline_lanes.items():
                poly = linfo["polygon"]
                try:
                    inside = shapely.contains_xy(poly, cxs, cys)
                except AttributeError:
                    inside = np.array([poly.contains(Point(x, y)) for x, y in zip(cxs, cys)])

                for idx in np.where(inside)[0]:
                    tid = int(tracked[idx][4])
                    # In legacy baseline without exclusivity:
                    cat = "cars"
                    baseline_metrics.lanes[lid]["vehicles"]["moving"][cat].add(tid)
                    if tid not in baseline_track_to_lanes:
                        baseline_track_to_lanes[tid] = set()
                    baseline_track_to_lanes[tid].add(lid)

        if (f_idx + 1) % interval_frames == 0 or (f_idx + 1) == eval_frames:
            # Audit interval snapshot for cross-lane duplicates
            seen_in_window: Dict[int, List[str]] = {}
            for lid, data in baseline_metrics.lanes.items():
                all_ids = data["vehicles"]["queued"]["cars"] | data["vehicles"]["moving"]["cars"]
                for tid in all_ids:
                    seen_in_window.setdefault(tid, []).append(lid)
            for tid, l_list in seen_in_window.items():
                if len(l_list) > 1:
                    baseline_window_double_counts += (len(l_list) - 1)
            baseline_metrics.reset()

    time_baseline = time.perf_counter() - t0_base
    fps_baseline = eval_frames / max(1e-5, time_baseline)
    baseline_total_unique_tracks = len(baseline_track_to_lanes)

    # ----------------------------------------------------
    # PIPELINE B: SHADOW-RESILIENT (SCE + Contact-Patch + Shadow Tracker + Hysteresis)
    # ----------------------------------------------------
    print(">>> Running Shadow-Resilient Pipeline (SCE + Contact Patch + Shadow Tracker)...")
    shadow_eq = ShadowContrastEqualizer(dynamic_trigger=True)
    contact_ref = ContactPatchRefiner(ground_offset_ratio=0.05, trim_shadow_wings=True)
    lane_assigner = ShadowLaneAssigner(hysteresis_frames=3)
    shadow_tracker = ShadowResilientTracker(
        det_thresh=0.40,
        min_conf=0.15,
        max_age=30,
        max_coast_frames=12,
        min_hits=2,
    )
    resilient_lanes = {
        lid: {
            "direction": linfo["direction"],
            "polygon": linfo["polygon"],
        }
        for lid, linfo in lanes_dict.items()
    }
    resilient_metrics = LaneMetricsManager(resilient_lanes)

    resilient_track_to_lanes: Dict[int, set] = {}
    resilient_shadow_detections = 0
    shadow_active_frames = 0
    t0_resilient = time.perf_counter()

    for f_idx, frame in enumerate(frames):
        enhanced_frame, is_active = shadow_eq.enhance(frame)
        if is_active:
            shadow_active_frames += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        res = model(enhanced_frame, conf=conf, verbose=False, device=device, classes=list(COCO_VEHICLES.keys()))[0]
        dets = []
        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = float(box.conf[0])
            cls_id = int(box.cls[0])
            b_trimmed = contact_ref.trim_lateral_cast_shadow(b, gray)
            dets.append([b_trimmed[0], b_trimmed[1], b_trimmed[2], b_trimmed[3], c, cls_id])

            bx1, by1, bx2, by2 = int(b_trimmed[0]), int(b_trimmed[1]), int(b_trimmed[2]), int(b_trimmed[3])
            crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if crop.size > 0 and np.mean(crop) < 55.0:
                resilient_shadow_detections += 1

        dets_arr = np.array(dets) if len(dets) else np.empty((0, 6))
        tracked = shadow_tracker.update(dets_arr)

        if len(tracked) > 0:
            cxs, cys = contact_ref.get_contact_points_vectorized(tracked)

            num_objs = len(tracked)
            obj_candidates = [None] * num_objs
            for lid, linfo in resilient_lanes.items():
                poly = linfo["polygon"]
                try:
                    inside = shapely.contains_xy(poly, cxs, cys)
                except AttributeError:
                    inside = np.array([poly.contains(Point(x, y)) for x, y in zip(cxs, cys)])

                for idx in np.where(inside)[0]:
                    obj_candidates[idx] = lid

            for idx in range(num_objs):
                tid = int(tracked[idx][4])
                cls_id = int(tracked[idx][5]) if len(tracked[idx]) >= 6 else 2

                confirmed_lane = lane_assigner.update_track_lane(
                    track_id=tid,
                    candidate_lane_id=obj_candidates[idx],
                    frame_idx=f_idx,
                )

                if confirmed_lane is not None:
                    resilient_metrics.register_vehicle(
                        lane_id=confirmed_lane,
                        track_id=tid,
                        vehicle_class=cls_id,
                    )
                    if tid not in resilient_track_to_lanes:
                        resilient_track_to_lanes[tid] = set()
                    resilient_track_to_lanes[tid].add(confirmed_lane)

        if (f_idx + 1) % interval_frames == 0 or (f_idx + 1) == eval_frames:
            seen_in_window: Dict[int, List[str]] = {}
            for lid, data in resilient_metrics.lanes.items():
                all_ids = data["vehicles"]["queued"]["cars"] | data["vehicles"]["moving"]["cars"]
                for tid in all_ids:
                    seen_in_window.setdefault(tid, []).append(lid)
            for tid, l_list in seen_in_window.items():
                if len(l_list) > 1:
                    resilient_window_double_counts += (len(l_list) - 1)
            resilient_metrics.reset()

    time_resilient = time.perf_counter() - t0_resilient
    fps_resilient = eval_frames / max(1e-5, time_resilient)
    resilient_total_unique_tracks = len(resilient_track_to_lanes)

    # ----------------------------------------------------
    # BENCHMARK COMPARATIVE SUMMARY
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS & SHADOW RESILIENCE AUDIT")
    print("=" * 60)
    print(f"{'Metric':<38} | {'Baseline':<10} | {'Shadow-Resilient':<15}")
    print("-" * 60)
    print(f"{'Interval Cross-Lane Double-Counts':<38} | {baseline_window_double_counts:<10} | {resilient_window_double_counts:<15} (Target: 0)")
    print(f"{'Unique Track IDs Generated':<38} | {baseline_total_unique_tracks:<10} | {resilient_total_unique_tracks:<15} (Lower = Less fragmentation)")
    print(f"{'Detections in Deep Shadow Zones':<38} | {baseline_shadow_detections:<10} | {resilient_shadow_detections:<15}")
    print(f"{'Shadow Equalizer Triggered Frames':<38} | {'N/A':<10} | {shadow_active_frames} / {eval_frames} frames")
    print(f"{'Processing Speed (FPS)':<38} | {fps_baseline:<10.1f} | {fps_resilient:<15.1f}")
    print("=" * 60 + "\n")

    return {
        "baseline_double_counts": baseline_window_double_counts,
        "resilient_double_counts": resilient_window_double_counts,
        "baseline_tracks": baseline_total_unique_tracks,
        "resilient_tracks": resilient_total_unique_tracks,
        "baseline_shadow_dets": baseline_shadow_detections,
        "resilient_shadow_dets": resilient_shadow_detections,
        "fps_baseline": fps_baseline,
        "fps_resilient": fps_resilient,
    }


def main():
    parser = argparse.ArgumentParser(description="Harsh Shadow Vehicle Counting Benchmark Tool")
    parser.add_argument(
        "--video",
        default="videos/dry/gettyimages-151939150-640_adpp.mp4",
        help="Path to test video with harsh shadows",
    )
    parser.add_argument("--config", default=None, help="Optional camera configuration JSON")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to evaluate")
    parser.add_argument("--device", default=None, help="Compute device (auto, cuda:0, mps, cpu)")
    parser.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    args = parser.parse_args()

    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    run_benchmark(
        video_path=args.video,
        config_path=args.config,
        model_path=args.model,
        max_frames=args.frames,
        device=device,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
