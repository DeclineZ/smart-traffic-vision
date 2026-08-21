"""
Multi-Camera Traffic Tracking & MQTT Streaming Pipeline for Laptop / Edge Development.
Runs multiple video feeds in parallel with YOLO detection, SORT tracking, and real-time MQTT streaming.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from collections import deque
from typing import Any

import cv2 as cv
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import contains
from ultralytics import YOLO

# Add root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm.sort import Sort
from trt_pipeline.payload import LaneMetricsManager, PayloadBuilder, classify_vehicle
from trt_pipeline.publisher import MQTTPublisher
from trt_pipeline.tools import initial_config, parse_zones, side_of_line, get_logger

logger = get_logger("MultiCameraRunner")

DEFAULT_CONFIGS = {
    "north": "config/config_north.json",
    "south": "config/config_south.json",
    "east": "config/config_east.json",
    "west": "config/config_west.json",
}

# COCO Class mapping for YOLO
COCO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class CameraWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        config_path: str,
        model: YOLO,
        device: str = "cuda:0",
        display: bool = False,
        max_fps: float = 25.0,
        conf: float = 0.20,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.config_path = config_path
        self.model = model
        self.device = device
        self.display = display
        self.max_fps = max_fps
        self.conf = conf
        self.running = False
        self.latest_frame = None

        # Load config
        self.config = initial_config(config_path)
        self.camera_info = self.config["camera_info"]
        self.metrics_config = self.config["lane_metrics"]
        self.video_path = self.config["video"]["path"]
        self.publish_interval = int(self.metrics_config.get("publish_interval_frames", 50))
        self.queue_speed_thresh = float(self.metrics_config.get("queue_speed_threshold", 2.0))

        # Lane metrics & Payload builder
        self.lane_config = {
            lane_id: {
                "direction": l_info.get("direction", lane_id[0]),
                "polygon": Polygon(l_info["polygon"]),
            }
            for lane_id, l_info in self.metrics_config.get("lanes", {}).items()
        }
        self.metrics = LaneMetricsManager(self.lane_config)
        self.payload_builder = PayloadBuilder(
            intersection_id=self.camera_info["intersection_id"],
            camera_id=self.camera_info["camera_id"],
        )

        # Tracker & motion memory
        self.tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
        self.track_history: dict[int, deque] = {}

        # MQTT Publisher
        mqtt_cfg = self.config.get("mqtt", {})
        self.publisher = None
        if mqtt_cfg.get("enabled", True):
            self.publisher = MQTTPublisher(
                broker_url=mqtt_cfg.get("broker_url", "mqtt://localhost:1883"),
                topic=mqtt_cfg.get("topic", "traffic/counts"),
                username=mqtt_cfg.get("username"),
                password=mqtt_cfg.get("password"),
                qos=mqtt_cfg.get("qos", 1),
            )

        self.latest_lanes = []
        self._lanes_lock = threading.Lock()

    def get_latest_lanes(self) -> list[dict[str, Any]]:
        with self._lanes_lock:
            return list(self.latest_lanes)

    def _is_queued(self, track_id: int, pt: tuple[float, float], frame_idx: int) -> bool:
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

    def run(self):
        if not os.path.exists(self.video_path):
            logger.error(f"[{self.name}] Video file not found: {self.video_path}")
            return

        cap = cv.VideoCapture(self.video_path)
        self.running = True
        frame_idx = 0
        last_pub_time = time.perf_counter()

        logger.info(f"[{self.name}] Started processing: {self.video_path}")

        frame_delay = 1.0 / self.max_fps if self.max_fps > 0 else 0

        while self.running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                # Loop video for continuous streaming simulation
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1

            # Run YOLO inference on GPU / configured device with configurable confidence
            results = self.model(frame, verbose=False, device=self.device, classes=list(COCO_CLASSES.keys()), conf=self.conf)
            dets = []
            if len(results) and len(results[0].boxes):
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    dets.append([x1, y1, x2, y2, conf, cls_id])

            dets_arr = np.array(dets) if len(dets) else np.empty((0, 6))

            # Update tracker
            track_input = dets_arr[:, :5] if len(dets_arr) else np.empty((0, 5))
            tracked_objs = self.tracker.update(track_input)

            # Process tracked objects in lane polygons
            for obj in tracked_objs:
                x1, y1, x2, y2, track_id = obj[:5]
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                centroid = Point(cx, cy)

                # Match class from closest detection
                cls_id = 2  # default car
                if len(dets_arr):
                    centers = (dets_arr[:, :2] + dets_arr[:, 2:4]) / 2
                    dists = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
                    cls_id = int(dets_arr[np.argmin(dists), 5])

                is_q = self._is_queued(track_id, (cx, cy), frame_idx)

                for lane_id, lcfg in self.lane_config.items():
                    if contains(lcfg["polygon"], centroid):
                        self.metrics.register_vehicle(
                            lane_id=lane_id,
                            track_id=track_id,
                            vehicle_class=cls_id,
                            is_queued=is_q,
                        )

            # Take snapshot periodically
            if frame_idx % self.publish_interval == 0:
                lanes_snapshot = self.metrics.snapshot()
                self.metrics.reset()

                with self._lanes_lock:
                    self.latest_lanes = lanes_snapshot

                now = time.perf_counter()
                fps = self.publish_interval / max(1e-5, now - last_pub_time)
                last_pub_time = now

                # Log summary
                total_q = sum(l["vehicles"]["queued"]["cars"] + l["vehicles"]["queued"]["motorbike"] for l in lanes_snapshot)
                total_m = sum(l["vehicles"]["moving"]["cars"] + l["vehicles"]["moving"]["motorbike"] for l in lanes_snapshot)
                logger.info(f"[{self.name}] Frame {frame_idx:05d} | Queued: {total_q} | Moving: {total_m} | FPS: {fps:.1f}")

            # Store preview frame if display is requested
            if self.display:
                vis_frame = cv.resize(frame, (640, 360))
                scale_x, scale_y = 640 / frame.shape[1], 360 / frame.shape[0]

                # Draw lane polygons
                for lane_id, lcfg in self.lane_config.items():
                    poly_scaled = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in lcfg["polygon"].exterior.coords], dtype=np.int32)
                    cv.polylines(vis_frame, [poly_scaled], True, (0, 255, 0), 2)
                    cv.putText(vis_frame, lane_id, tuple(poly_scaled[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw tracked bounding boxes, IDs, and confidence scores
                for obj in tracked_objs:
                    ox1, oy1, ox2, oy2, otrack_id = obj[:5]
                    bx1, by1 = int(ox1 * scale_x), int(oy1 * scale_y)
                    bx2, by2 = int(ox2 * scale_x), int(oy2 * scale_y)

                    cname = "car"
                    conf_val = 0.85
                    if len(dets_arr):
                        centers = (dets_arr[:, :2] + dets_arr[:, 2:4]) / 2
                        dists = np.linalg.norm(centers - np.array([(ox1 + ox2) / 2, (oy1 + oy2) / 2]), axis=1)
                        closest_idx = np.argmin(dists)
                        cname = COCO_CLASSES.get(int(dets_arr[closest_idx, 5]), "car")
                        conf_val = float(dets_arr[closest_idx, 4])

                    box_color = (0, 255, 255) if cname in ("motorcycle", "bicycle") else (0, 220, 100)
                    cv.rectangle(vis_frame, (bx1, by1), (bx2, by2), box_color, 2)
                    label = f"#{int(otrack_id)} {cname} {conf_val:.2f}"
                    cv.putText(vis_frame, label, (bx1, max(14, by1 - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

                cv.putText(vis_frame, f"{self.name.upper()} (CAM: {self.camera_info['camera_id']})", (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.latest_frame = vis_frame

            # Throttle frame rate if needed
            elapsed = time.perf_counter() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)

        cap.release()

    def stop(self):
        self.running = False


def main():
    import torch

    parser = argparse.ArgumentParser(description="Multi-Camera Traffic Tracking & MQTT Streaming")
    parser.add_argument("--camera", choices=["north", "south", "east", "west", "all"], default="all", help="Camera feed to run (default: all)")
    parser.add_argument("--all", action="store_true", help="Run all 4 camera feeds simultaneously (default)")
    parser.add_argument("--display", action="store_true", help="Display visual live grid window")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path or name (default: yolov8n.pt)")
    parser.add_argument("--device", default=None, help="Inference device: 'cuda', 'cuda:0', 'cpu' (default: auto)")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence detection threshold (default: 0.20)")
    parser.add_argument("--fps", type=float, default=25.0, help="Target processing FPS (default: 25.0)")
    parser.add_argument("--pub-interval", type=float, default=2.0, help="MQTT broadcast interval in seconds (default: 2.0)")
    args = parser.parse_args()

    # Determine compute device
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in device else "CPU"
    logger.info(f"Using compute device: {device} ({gpu_name})")

    logger.info(f"Loading YOLO model: {args.model} onto {device} (conf threshold: {args.conf})...")
    model = YOLO(args.model)

    run_all = args.all or args.camera == "all"
    targets = list(DEFAULT_CONFIGS.keys()) if run_all else [args.camera]
    workers = []

    for cam_name in targets:
        cfg_path = DEFAULT_CONFIGS[cam_name]
        worker = CameraWorker(
            name=cam_name,
            config_path=cfg_path,
            model=model,
            device=device,
            display=args.display,
            max_fps=args.fps,
            conf=args.conf,
        )
        workers.append(worker)
        worker.start()

    # Global MQTT Publisher for integrated intersection streaming
    publisher = MQTTPublisher(
        broker_url="mqtt://localhost:1883",
        topic="traffic/counts",
        qos=1,
    )
    publisher.start()
    payload_builder = PayloadBuilder(intersection_id="INT-001", camera_id="MULTI-CAM")

    logger.info(f"Started {len(workers)} camera feeds streaming to MQTT.")
    print("==================================================")
    print(" Multi-Camera Tracker active!")
    print(" Publishing traffic counts to MQTT topic: traffic/counts")
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
                        meta={"active_cameras": [w.name for w in workers]},
                    )
                    publisher.publish(payload)
                last_broadcast_time = now
            if args.display:
                if run_all:
                    frames = [w.latest_frame for w in workers if w.latest_frame is not None]
                    if len(frames) == len(workers):
                        top_row = np.hstack([frames[0], frames[1]])
                        bot_row = np.hstack([frames[2], frames[3]])
                        grid = np.vstack([top_row, bot_row])
                        cv.imshow("4-Camera Adaptive Traffic Monitor", grid)
                else:
                    if workers and workers[0].latest_frame is not None:
                        cv.imshow(f"{args.camera.upper()} Traffic Monitor", workers[0].latest_frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stopping camera workers...")
    finally:
        for w in workers:
            w.stop()
        publisher.stop()
        if args.display:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
