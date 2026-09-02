"""
Multi-Camera Traffic Tracking & MQTT Streaming Pipeline for Laptop / Edge Development.
Supports 1 to 8+ parallel camera feeds with YOLO detection, SORT tracking, dynamic grid HUD,
and real-time MQTT streaming to the traffic controller engine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
from shapely.geometry import Point, Polygon
from shapely import contains
from ultralytics import YOLO

# Add root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm.sort import Sort
from trt_pipeline.payload import LaneMetricsManager, PayloadBuilder
from trt_pipeline.publisher import MQTTPublisher
from trt_pipeline.tools import initial_config, get_logger

logger = get_logger("MultiCameraRunner")

DEFAULT_CONFIGS = {
    "north": "config/config_north.json",
    "south": "config/config_south.json",
    "east": "config/config_east.json",
    "west": "config/config_west.json",
}

COCO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def stitch_camera_grid(frames: List[np.ndarray], tile_size: Tuple[int, int] = (480, 270)) -> np.ndarray:
    """
    Dynamically tiles N camera preview frames into an adaptive multi-view grid.
    Supports 1, 2, 4, 6, 8, or arbitrary N streams with uniform aspect ratio.
    """
    if not frames:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)

    resized = [cv.resize(f, tile_size) for f in frames]
    n = len(resized)

    if n == 1:
        return resized[0]
    elif n == 2:
        return np.hstack(resized)
    elif n in (3, 4):
        while len(resized) < 4:
            resized.append(np.zeros_like(resized[0]))
        top = np.hstack(resized[:2])
        bot = np.hstack(resized[2:4])
        return np.vstack([top, bot])
    elif n in (5, 6):
        while len(resized) < 6:
            resized.append(np.zeros_like(resized[0]))
        row1 = np.hstack(resized[:3])
        row2 = np.hstack(resized[3:6])
        return np.vstack([row1, row2])
    else:
        # 7, 8 or larger grids (2 or more rows of 4 columns)
        cols = 4
        while len(resized) % cols != 0:
            resized.append(np.zeros_like(resized[0]))
        rows = [np.hstack(resized[i : i + cols]) for i in range(0, len(resized), cols)]
        return np.vstack(rows)


class CameraWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        config_path: str,
        model: YOLO,
        video_override: Optional[str] = None,
        camera_id_override: Optional[str] = None,
        device: str = "cuda:0",
        display: bool = False,
        max_fps: float = 25.0,
        conf: float = 0.20,
        model_lock: Optional[threading.Lock] = None,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.config_path = config_path
        self.model = model
        self.device = device
        self.display = display
        self.max_fps = max_fps
        self.conf = conf
        self.model_lock = model_lock or threading.Lock()
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None

        # Load config
        self.config = initial_config(config_path)
        self.camera_info = dict(self.config["camera_info"])
        if camera_id_override:
            self.camera_info["camera_id"] = camera_id_override

        self.video_path = video_override or self.config["video"]["path"]
        self.metrics_config = self.config["lane_metrics"]
        self.publish_interval = int(self.metrics_config.get("publish_interval_frames", 50))
        self.queue_speed_thresh = float(self.metrics_config.get("queue_speed_threshold", 2.0))

        # Lane metrics
        self.lane_config = {
            lane_id: {
                "direction": l_info.get("direction", lane_id[0]),
                "polygon": Polygon(l_info["polygon"]),
            }
            for lane_id, l_info in self.metrics_config.get("lanes", {}).items()
        }
        self.metrics = LaneMetricsManager(self.lane_config)

        # Tracker & motion memory
        self.tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
        self.track_history: Dict[int, deque] = {}

        self.latest_lanes: List[Dict[str, Any]] = []
        self._lanes_lock = threading.Lock()

    def get_latest_lanes(self) -> List[Dict[str, Any]]:
        with self._lanes_lock:
            return list(self.latest_lanes)

    def _is_queued(self, track_id: int, pt: Tuple[float, float], frame_idx: int) -> bool:
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=15)
            self.track_history[track_id].append((frame_idx, pt))
            return False

        hist = self.track_history[track_id]
        hist.append((frame_idx, pt))

        if len(hist) < 3:
            return False

        first_frame, first_pt = hist[0]
        dt = max(1, frame_idx - first_frame)
        dist = float(np.linalg.norm(np.array(pt) - np.array(first_pt)))
        return (dist / dt) < self.queue_speed_thresh

    def _prune_track_history(self, active_track_ids: set[int]):
        """Prunes dead track IDs from history to prevent unbounded memory growth."""
        if len(self.track_history) > 300:
            stale_ids = [tid for tid in self.track_history if tid not in active_track_ids]
            for tid in stale_ids:
                del self.track_history[tid]

    def run(self):
        is_url = "://" in self.video_path
        if not is_url and not os.path.exists(self.video_path):
            logger.error(f"[{self.name}] Video file not found: {self.video_path}")
            return

        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"[{self.name}] Failed to open video stream: {self.video_path}")
            return

        self.running = True
        frame_idx = 0
        last_pub_time = time.perf_counter()

        logger.info(f"[{self.name}] Started processing: {self.video_path} (CAM: {self.camera_info['camera_id']})")
        frame_delay = 1.0 / self.max_fps if self.max_fps > 0 else 0.0

        try:
            while self.running:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    if not self.running:
                        break
                    # Loop local video for continuous streaming simulation
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue

                frame_idx += 1

                # Thread-safe YOLO model inference
                with self.model_lock:
                    results = self.model(
                        frame,
                        verbose=False,
                        device=self.device,
                        classes=list(COCO_CLASSES.keys()),
                        conf=self.conf,
                    )
                    if "cuda" in self.device and torch.cuda.is_available():
                        torch.cuda.synchronize()

                dets = []
                if len(results) and len(results[0].boxes):
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        dets.append([x1, y1, x2, y2, conf, cls_id])

                dets_arr = np.array(dets) if len(dets) else np.empty((0, 6))

                # Update SORT tracker
                track_input = dets_arr[:, :5] if len(dets_arr) else np.empty((0, 5))
                tracked_objs = self.tracker.update(track_input)

                # Map track identities and classes once per frame
                active_ids = set()
                track_meta: Dict[int, Tuple[int, str, float]] = {}

                for obj in tracked_objs:
                    x1, y1, x2, y2, track_id = obj[:5]
                    tid = int(track_id)
                    active_ids.add(tid)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    cls_id = 2  # Default car
                    cname = "car"
                    conf_val = self.conf
                    if len(dets_arr):
                        centers = (dets_arr[:, :2] + dets_arr[:, 2:4]) / 2.0
                        dists = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
                        closest_idx = int(np.argmin(dists))
                        cls_id = int(dets_arr[closest_idx, 5])
                        cname = COCO_CLASSES.get(cls_id, "car")
                        conf_val = float(dets_arr[closest_idx, 4])

                    track_meta[tid] = (cls_id, cname, conf_val)

                    is_q = self._is_queued(tid, (cx, cy), frame_idx)
                    centroid = Point(cx, cy)

                    for lane_id, lcfg in self.lane_config.items():
                        if contains(lcfg["polygon"], centroid):
                            self.metrics.register_vehicle(
                                lane_id=lane_id,
                                track_id=tid,
                                vehicle_class=cls_id,
                                is_queued=is_q,
                            )

                self._prune_track_history(active_ids)

                # Periodic lane metrics snapshot
                if frame_idx % self.publish_interval == 0:
                    lanes_snapshot = self.metrics.snapshot()
                    self.metrics.reset()

                    with self._lanes_lock:
                        self.latest_lanes = lanes_snapshot

                    now = time.perf_counter()
                    fps = self.publish_interval / max(1e-5, now - last_pub_time)
                    last_pub_time = now

                    total_q = sum(l["vehicles"]["queued"]["cars"] + l["vehicles"]["queued"]["motorbike"] for l in lanes_snapshot)
                    total_m = sum(l["vehicles"]["moving"]["cars"] + l["vehicles"]["moving"]["motorbike"] for l in lanes_snapshot)
                    logger.info(f"[{self.name}] Frame {frame_idx:05d} | Queued: {total_q} | Moving: {total_m} | FPS: {fps:.1f}")

                # Visual HUD preview
                if self.display:
                    vis_w, vis_h = 640, 360
                    vis_frame = cv.resize(frame, (vis_w, vis_h))
                    scale_x, scale_y = vis_w / frame.shape[1], vis_h / frame.shape[0]

                    # Draw lane polygons
                    for lane_id, lcfg in self.lane_config.items():
                        poly_scaled = np.array(
                            [[int(x * scale_x), int(y * scale_y)] for x, y in lcfg["polygon"].exterior.coords],
                            dtype=np.int32,
                        )
                        cv.polylines(vis_frame, [poly_scaled], True, (0, 255, 0), 2)
                        cv.putText(vis_frame, lane_id, tuple(poly_scaled[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Draw tracked bounding boxes
                    for obj in tracked_objs:
                        ox1, oy1, ox2, oy2, otrack_id = obj[:5]
                        tid = int(otrack_id)
                        bx1, by1 = int(ox1 * scale_x), int(oy1 * scale_y)
                        bx2, by2 = int(ox2 * scale_x), int(oy2 * scale_y)

                        cls_id, cname, conf_val = track_meta.get(tid, (2, "car", self.conf))
                        box_color = (0, 255, 255) if cname in ("motorcycle", "bicycle") else (0, 220, 100)
                        cv.rectangle(vis_frame, (bx1, by1), (bx2, by2), box_color, 2)
                        label = f"#{tid} {cname} {conf_val:.2f}"
                        cv.putText(
                            vis_frame,
                            label,
                            (bx1, max(14, by1 - 4)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv.LINE_AA,
                        )

                    cv.putText(
                        vis_frame,
                        f"{self.name.upper()} (CAM: {self.camera_info['camera_id']})",
                        (15, 25),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
                    self.latest_frame = vis_frame

                # Frame pacing
                elapsed = time.perf_counter() - t0
                if frame_delay > elapsed:
                    time.sleep(frame_delay - elapsed)

        finally:
            cap.release()
            logger.info(f"[{self.name}] Camera feed released and stopped.")

    def stop(self):
        self.running = False


def build_camera_workers(
    args: argparse.Namespace,
    model: YOLO,
    device: str,
    model_lock: threading.Lock,
) -> List[CameraWorker]:
    """Constructs dynamic CameraWorker instances supporting 1 to 8+ feeds."""
    workers: List[CameraWorker] = []

    # Case 1: Explicit config list provided
    if args.configs:
        for idx, cfg_path in enumerate(args.configs):
            cam_name = f"cam_{idx + 1:02d}"
            video_override = args.videos[idx] if args.videos and idx < len(args.videos) else None
            w = CameraWorker(
                name=cam_name,
                config_path=cfg_path,
                model=model,
                video_override=video_override,
                camera_id_override=f"CAM_{idx + 1:02d}",
                device=device,
                display=args.display,
                max_fps=args.fps,
                conf=args.conf,
                model_lock=model_lock,
            )
            workers.append(w)
        return workers

    # Case 2: Named cameras specified (e.g. north south)
    if args.cameras:
        target_names = list(DEFAULT_CONFIGS.keys()) if "all" in args.cameras else args.cameras
        for idx, cam_name in enumerate(target_names):
            cfg_path = DEFAULT_CONFIGS.get(cam_name, DEFAULT_CONFIGS["north"])
            video_override = args.videos[idx] if args.videos and idx < len(args.videos) else None
            w = CameraWorker(
                name=cam_name,
                config_path=cfg_path,
                model=model,
                video_override=video_override,
                camera_id_override=f"CAM_{idx + 1:02d}",
                device=device,
                display=args.display,
                max_fps=args.fps,
                conf=args.conf,
                model_lock=model_lock,
            )
            workers.append(w)
        return workers

    # Case 3: Dynamic stream count (4 to 8+ feeds)
    num_cams = max(1, args.num_cams)
    default_keys = list(DEFAULT_CONFIGS.keys())

    for idx in range(num_cams):
        key = default_keys[idx % len(default_keys)]
        cfg_path = DEFAULT_CONFIGS[key]
        cam_name = f"{key}_{idx // len(default_keys) + 1}" if num_cams > 4 else key
        video_override = args.videos[idx] if args.videos and idx < len(args.videos) else None

        w = CameraWorker(
            name=cam_name,
            config_path=cfg_path,
            model=model,
            video_override=video_override,
            camera_id_override=f"CAM_{idx + 1:02d}",
            device=device,
            display=args.display,
            max_fps=args.fps,
            conf=args.conf,
            model_lock=model_lock,
        )
        workers.append(w)

    return workers


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Traffic Tracking & MQTT Streaming Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-cams",
        type=int,
        default=4,
        help="Number of dynamic camera feeds to run for this intersection (e.g. 4, 6, 8)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Named camera feeds to run: north south east west all (overrides --num-cams)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Custom JSON configuration file paths per camera feed",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Video source file paths or RTSP URLs for camera feeds",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display adaptive multi-camera live grid window",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model checkpoint or TensorRT engine path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: 'cuda', 'cuda:0', 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Target processing FPS per camera feed",
    )
    parser.add_argument(
        "--pub-interval",
        type=float,
        default=2.0,
        help="MQTT broadcast interval in seconds",
    )
    parser.add_argument(
        "--mqtt-broker",
        default="mqtt://localhost:1883",
        help="MQTT broker URL",
    )
    parser.add_argument(
        "--mqtt-topic",
        default="traffic/counts",
        help="MQTT destination topic",
    )
    parser.add_argument(
        "--intersection-id",
        default="INT-001",
        help="Intersection identifier string",
    )
    args = parser.parse_args()

    # Determine compute device
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in device else "CPU"
    logger.info(f"Using compute device: {device} ({gpu_name})")

    logger.info(f"Loading YOLO model: {args.model} onto {device} (conf threshold: {args.conf})...")
    model = YOLO(args.model)
    model_lock = threading.Lock()

    workers = build_camera_workers(args=args, model=model, device=device, model_lock=model_lock)

    for w in workers:
        w.start()

    # Centralized MQTT Publisher for intersection aggregation
    publisher = MQTTPublisher(
        broker_url=args.mqtt_broker,
        topic=args.mqtt_topic,
        qos=1,
    )
    publisher.start()
    payload_builder = PayloadBuilder(intersection_id=args.intersection_id, camera_id="MULTI-CAM")

    logger.info(f"Started {len(workers)} camera feed(s) streaming to MQTT ({args.mqtt_topic}).")
    print("==================================================")
    print(f" Multi-Camera Tracker active: {len(workers)} camera feeds")
    print(f" Publishing to MQTT topic: {args.mqtt_topic}")
    print(" Press Ctrl+C to stop")
    print("==================================================")

    last_broadcast_time = time.perf_counter()
    broadcast_frame = 0

    try:
        while True:
            now = time.perf_counter()

            # Broadcast combined intersection snapshot periodically
            if now - last_broadcast_time >= args.pub_interval:
                combined_lanes = []
                for w in workers:
                    combined_lanes.extend(w.get_latest_lanes())

                if combined_lanes:
                    broadcast_frame += 1
                    payload = payload_builder.build(
                        frame_idx=broadcast_frame,
                        lanes_snapshot=combined_lanes,
                        meta={"active_cameras": [w.camera_info["camera_id"] for w in workers]},
                    )
                    publisher.publish(payload)
                last_broadcast_time = now

            if args.display:
                frames = [w.latest_frame for w in workers if w.latest_frame is not None]
                if len(frames) == len(workers) and len(frames) > 0:
                    grid = stitch_camera_grid(frames)
                    cv.imshow("Smart Traffic Vision - Adaptive Multi-Camera Monitor", grid)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Stopping camera workers...")
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=1.5)
        publisher.stop()
        if args.display:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
