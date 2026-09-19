"""
High-Throughput Batched Multi-Camera Traffic Tracking & MQTT Streaming Pipeline.
Production architecture engineered for high-FPS edge & workstation deployment:
- Centralized Dynamic Batching (N, 3, H, W) into TensorRT FP16 / PyTorch
- Intermittent Inference (Frame Skipping 1-in-2 / 1-in-3) with Kalman Tracker Continuity
- Decoupled Asynchronous Display Worker (AsyncDisplayWorker) + optional NVENC
- Jitter-Absorbing Double-Buffered Ring Buffers (queue_size = 2)
- Vectorized Spatial Lane Analytics using shapely.contains_xy
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import signal
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import json
import logging

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure standard logger
logger = logging.getLogger("ProductionMultiCameraRunner")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def initial_config(config_path: str) -> dict:
    """Loads JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


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


class BatchedCameraPipeline:
    """
    Centralized multi-camera orchestration engine:
    - Coordinates N asynchronous double-buffered camera ingestion workers
    - Assembles dynamic batch tensors for single-pass TensorRT / PyTorch inference
    - Manages intermittent frame skipping with Kalman filter continuity
    - Executes vectorized spatial lane containment
    - Dispatches non-blocking snapshots to AsyncDisplayWorker and MQTT publisher
    """

    def __init__(
        self,
        camera_names: List[str],
        config_paths: List[str],
        video_sources: List[str],
        model_path: str,
        device: str = "cuda:0",
        conf: float = 0.20,
        target_fps: float = 25.0,
        skip_frames: int = 1,
        buffer_size: int = 2,
        display: bool = False,
        nvenc_out: Optional[str] = None,
        pub_interval: float = 2.0,
        mqtt_broker: str = "mqtt://localhost:1883",
        mqtt_topic: str = "traffic/counts",
        intersection_id: str = "INT-001",
        voting_window: int = 15,
        pickup_bias: float = 1.15,
        enable_voting: bool = True,
    ):
        global cv, np, shapely, Point, Polygon, YOLO, Sort
        global AsyncDisplayWorker, NVENCVideoWriter, is_nvenc_available
        global LaneMetricsManager, PayloadBuilder, MQTTPublisher, StreamBufferWorker, TrackClassVotingFilter

        import cv2 as cv
        import numpy as np
        import shapely
        from shapely.geometry import Point, Polygon
        from ultralytics import YOLO

        from algorithm.sort import Sort
        from trt_pipeline.display import AsyncDisplayWorker, NVENCVideoWriter, is_nvenc_available
        from trt_pipeline.payload import LaneMetricsManager, PayloadBuilder
        from trt_pipeline.publisher import MQTTPublisher
        from trt_pipeline.stream import StreamBufferWorker
        from trt_pipeline.voter import TrackClassVotingFilter
        self.num_streams = len(camera_names)
        self.camera_names = camera_names
        self.config_paths = config_paths
        self.video_sources = video_sources
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.target_fps = target_fps
        self.skip_frames = max(0, skip_frames)
        self.buffer_size = buffer_size
        self.display = display
        self.nvenc_out = nvenc_out
        self.pub_interval = pub_interval
        self.intersection_id = intersection_id

        # 1. Load Camera Configs, Lane Polygons & Metrics Managers
        self.camera_configs = []
        self.lane_configs: List[Dict[str, Dict[str, Any]]] = []
        self.metrics_managers: List[LaneMetricsManager] = []
        self.queue_speed_thresholds: List[float] = []

        for idx in range(self.num_streams):
            cfg = initial_config(config_paths[idx])
            self.camera_configs.append(cfg)
            metrics_cfg = cfg.get("lane_metrics", {})
            self.queue_speed_thresholds.append(float(metrics_cfg.get("queue_speed_threshold", 2.0)))

            # Parse lane polygons
            lane_dict = {}
            for lane_id, l_info in metrics_cfg.get("lanes", {}).items():
                poly_coords = l_info.get("polygon", [])
                lane_dict[lane_id] = {
                    "direction": l_info.get("direction", lane_id[0] if lane_id else "N"),
                    "polygon": Polygon(poly_coords) if not isinstance(poly_coords, Polygon) else poly_coords,
                }
            self.lane_configs.append(lane_dict)
            self.metrics_managers.append(LaneMetricsManager(lane_dict))

        # 2. Trackers & Motion Memory per Camera
        min_hits = 1 if self.skip_frames > 0 else 2
        self.trackers = [
            Sort(max_age=30, min_hits=min_hits, iou_threshold=0.3)
            for _ in range(self.num_streams)
        ]
        self.last_tracked: List[np.ndarray] = [np.empty((0, 6)) for _ in range(self.num_streams)]
        self.track_histories: List[Dict[int, deque]] = [{} for _ in range(self.num_streams)]

        # 3. Stream Ingestion Workers (Double-Buffered Ring Buffer)
        self.stream_workers = [
            StreamBufferWorker(
                name=self.camera_names[i],
                source=self.video_sources[i],
                target_fps=self.target_fps,
                buffer_size=self.buffer_size,
                is_paced=True,
                loop_video=True,
            )
            for i in range(self.num_streams)
        ]

        # 4. Load Inference Model (Auto-detects TensorRT FP16 .engine vs PyTorch .pt)
        logger.info(f"Loading vision model: '{self.model_path}' onto device '{self.device}'...")
        self.model = YOLO(self.model_path)

        # Auto-configure class mappings from model metadata
        raw_names = getattr(self.model, "names", None)
        if raw_names and isinstance(raw_names, dict):
            model_names = {int(k): str(v) for k, v in raw_names.items()}
        elif raw_names and isinstance(raw_names, list):
            model_names = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            model_names = COCO_CLASSES

        # Check if model has domain-specific traffic classes (e.g. Thai Traffic 5-class model)
        traffic_keywords = {"car", "motorcycle", "bus", "truck", "three_wheeler", "tuktuk", "bicycle"}
        is_traffic_model = len(model_names) <= 15 and any(v.lower() in traffic_keywords for v in model_names.values())

        if is_traffic_model:
            self.class_names = model_names
            self.target_classes = None  # Infer across all domain classes
            logger.info(f"Loaded domain-specific traffic model with {len(self.class_names)} classes: {self.class_names}")
        else:
            self.class_names = COCO_CLASSES
            self.target_classes = list(COCO_CLASSES.keys())
            logger.info(f"Loaded general model with {len(model_names)} classes. Filtering to COCO traffic classes: {self.target_classes}")

        self.default_car_cls = next((k for k, v in self.class_names.items() if v.lower() == "car"), 0)
        truck_cls_id = next((k for k, v in self.class_names.items() if v.lower() == "truck"), 3)
        self.class_voter = TrackClassVotingFilter(
            num_streams=self.num_streams,
            window_size=voting_window,
            car_truck_bias=pickup_bias,
            car_cls_id=self.default_car_cls,
            truck_cls_id=truck_cls_id,
            enabled=enable_voting,
        )

        # 5. Decoupled Display & Video Encoding Worker
        nvenc_writer = None
        if self.nvenc_out:
            if is_nvenc_available():
                # Default 4-cam grid resolution (960x540)
                cols = min(4, self.num_streams)
                rows = (self.num_streams + cols - 1) // cols
                grid_w, grid_h = cols * 480, rows * 270
                nvenc_writer = NVENCVideoWriter(self.nvenc_out, width=grid_w, height=grid_h, fps=self.target_fps)
            else:
                logger.warning("Hardware NVENC is not available on this host. Falling back to OpenCV display.")

        self.display_worker = (
            AsyncDisplayWorker(
                display=self.display,
                nvenc_writer=nvenc_writer,
                window_name=f"Smart Traffic Vision - Multi-Camera Production Grid ({self.num_streams} Cams)",
                class_names=self.class_names,
            )
            if (self.display or nvenc_writer)
            else None
        )

        # 6. MQTT Integration
        self.publisher = MQTTPublisher(broker_url=mqtt_broker, topic=mqtt_topic, qos=1)
        self.payload_builder = PayloadBuilder(intersection_id=self.intersection_id, camera_id="MULTI-CAM")

        self.running = False
        self.total_processed_batches = 0
        self.total_inferred_batches = 0
        self.total_skipped_batches = 0

    def _is_queued(self, cam_idx: int, track_id: int, pt: Tuple[float, float], frame_idx: int) -> bool:
        """Determines if a tracked vehicle is queued (stopped/slow) based on centroid velocity."""
        hist_map = self.track_histories[cam_idx]
        if track_id not in hist_map:
            hist_map[track_id] = deque(maxlen=15)
            hist_map[track_id].append((frame_idx, pt))
            return False

        hist = hist_map[track_id]
        hist.append((frame_idx, pt))

        if len(hist) < 3:
            return False

        first_frame, first_pt = hist[0]
        dt = max(1, frame_idx - first_frame)
        dist = float(np.linalg.norm(np.array(pt) - np.array(first_pt)))
        speed = dist / dt
        return speed < self.queue_speed_thresholds[cam_idx]

    def _evaluate_vectorized_lanes(self, cam_idx: int, tracked_objs: np.ndarray, frame_idx: int) -> None:
        """
        High-performance vectorized spatial lane assignment:
        Evaluates all tracked bounding box centroids against lane polygons using
        C-accelerated shapely.contains_xy to eliminate Python loop overhead.
        """
        if tracked_objs is None or len(tracked_objs) == 0:
            return

        cxs = (tracked_objs[:, 0] + tracked_objs[:, 2]) * 0.5
        cys = (tracked_objs[:, 1] + tracked_objs[:, 3]) * 0.5
        lane_cfg = self.lane_configs[cam_idx]
        metrics = self.metrics_managers[cam_idx]

        for lane_id, linfo in lane_cfg.items():
            poly = linfo["polygon"]
            if poly is None or poly.is_empty:
                continue

            try:
                # Vectorized evaluation across all centroids simultaneously
                inside_mask = shapely.contains_xy(poly, cxs, cys)
            except AttributeError:
                # Fallback for older Shapely versions (< 2.0)
                inside_mask = np.array([poly.contains(Point(cx, cy)) for cx, cy in zip(cxs, cys)], dtype=bool)

            inside_indices = np.where(inside_mask)[0]
            for idx in inside_indices:
                obj = tracked_objs[idx]
                track_id = int(obj[4])
                cx, cy = float(cxs[idx]), float(cys[idx])
                cls_id = int(obj[5]) if len(obj) >= 6 else self.default_car_cls
                cls_name = self.class_names.get(cls_id, "car")

                is_q = self._is_queued(cam_idx, track_id, (cx, cy), frame_idx)
                metrics.register_vehicle(
                    lane_id=lane_id,
                    track_id=track_id,
                    vehicle_class=cls_name,
                    is_queued=is_q,
                )

    def start(self) -> None:
        """Starts stream workers, background display, and MQTT publisher."""
        for worker in self.stream_workers:
            worker.start()

        if self.display_worker:
            self.display_worker.start()

        self.publisher.start()
        self.running = True

    def stop(self) -> None:
        """Gracefully stops all workers and releases resources."""
        self.running = False
        for worker in self.stream_workers:
            worker.stop()

        if self.display_worker:
            self.display_worker.stop()

        self.publisher.stop()
        logger.info("BatchedCameraPipeline stopped cleanly.")

    def run(self) -> None:
        """Main centralized batching and processing loop."""
        self.start()
        logger.info(f"Pipeline running: {self.num_streams} camera streams | Frame skipping: 1-in-{self.skip_frames + 1}")

        batch_idx = 0
        last_pub_time = time.perf_counter()
        last_fps_time = time.perf_counter()
        fps_batch_counter = 0
        current_fps = 0.0

        try:
            while self.running:
                t0 = time.perf_counter()

                # 1. Ingest batch: pull 1 frame from each camera ring buffer
                batch_frames = []
                for worker in self.stream_workers:
                    item = worker.get_frame(timeout=0.1)
                    if item:
                        batch_frames.append(item[1])

                if len(batch_frames) < self.num_streams:
                    # Waiting for all streams to deliver synced frames
                    continue

                batch_idx += 1
                fps_batch_counter += 1
                should_run_yolo = (batch_idx % (self.skip_frames + 1)) == 0

                # 2. Centralized Model Forward Pass
                results = []
                if should_run_yolo:
                    infer_kwargs = {
                        "verbose": False,
                        "device": self.device,
                        "conf": self.conf,
                    }
                    if self.target_classes is not None:
                        infer_kwargs["classes"] = self.target_classes

                    results = self.model(batch_frames, **infer_kwargs)
                    self.total_inferred_batches += 1
                else:
                    self.total_skipped_batches += 1

                # 3. Fan-out to Trackers with Kalman Continuity & Vectorized Spatial Analytics
                tracked_list = []
                for idx in range(self.num_streams):
                    if should_run_yolo:
                        dets = []
                        if idx < len(results) and len(results[idx].boxes):
                            for box in results[idx].boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                cls_id = int(box.cls[0])
                                dets.append([x1, y1, x2, y2, conf, cls_id])

                        dets_arr = np.array(dets) if len(dets) else np.empty((0, 6))
                        # SORT tracker update with detection boxes [x1, y1, x2, y2, conf]
                        track_input = dets_arr[:, :5] if len(dets_arr) else np.empty((0, 5))
                        tracked_out = self.trackers[idx].update(track_input)

                        # Re-associate class IDs from closest detections with temporal voting smoothing
                        if len(tracked_out) > 0 and len(dets_arr) > 0:
                            matched_tracked = []
                            det_centers = (dets_arr[:, :2] + dets_arr[:, 2:4]) * 0.5
                            for tobj in tracked_out:
                                t_cx = (tobj[0] + tobj[2]) * 0.5
                                t_cy = (tobj[1] + tobj[3]) * 0.5
                                dists = np.linalg.norm(det_centers - np.array([t_cx, t_cy]), axis=1)
                                best_idx = int(np.argmin(dists))
                                raw_c_id = int(dets_arr[best_idx, 5])
                                det_conf = float(dets_arr[best_idx, 4])
                                tid = int(tobj[4])

                                smoothed_c_id = self.class_voter.update(
                                    cam_idx=idx,
                                    track_id=tid,
                                    raw_cls_id=raw_c_id,
                                    conf=det_conf,
                                )
                                matched_tracked.append(np.append(tobj[:5], smoothed_c_id))
                            tracked_objs = np.array(matched_tracked)
                        elif len(tracked_out) > 0:
                            matched_tracked = []
                            for tobj in tracked_out:
                                tid = int(tobj[4])
                                smoothed_c_id = self.class_voter.get_class(
                                    cam_idx=idx,
                                    track_id=tid,
                                    fallback=self.default_car_cls,
                                )
                                matched_tracked.append(np.append(tobj[:5], smoothed_c_id))
                            tracked_objs = np.array(matched_tracked)
                        else:
                            tracked_objs = np.empty((0, 6))

                        self.last_tracked[idx] = tracked_objs
                    else:
                        # Skip frame: use tracker's Kalman predicted state
                        tracked_objs = self.last_tracked[idx]

                    tracked_list.append(tracked_objs)

                    # Prune stale track voting history
                    if len(tracked_objs) > 0:
                        active_tids = set(int(o[4]) for o in tracked_objs)
                        self.class_voter.prune(cam_idx=idx, active_track_ids=active_tids)

                    # Vectorized Lane Analytics
                    self._evaluate_vectorized_lanes(cam_idx=idx, tracked_objs=tracked_objs, frame_idx=batch_idx)

                # 4. Decoupled Asynchronous Display Submission (0 ms GPU blocking)
                if self.display_worker:
                    stats_str = f"BATCH {batch_idx:06d} | FPS: {current_fps:.1f} | INFER: {'YES' if should_run_yolo else 'SKIP (Kalman)'} | DROPS: {sum(w.frames_dropped for w in self.stream_workers)}"
                    self.display_worker.submit(
                        frames=batch_frames,
                        tracked_list=tracked_list,
                        cam_names=self.camera_names,
                        lane_configs=self.lane_configs,
                        header_stats=stats_str,
                    )
                    # Poll GUI window from main thread (100% thread-safe on macOS/Linux/Windows)
                    if not self.display_worker.poll_window():
                        logger.info("User requested exit from preview window (pressed 'q').")
                        break

                # 5. Periodic MQTT Broadcast
                now = time.perf_counter()
                if now - last_pub_time >= self.pub_interval:
                    combined_lanes = []
                    for m_mgr in self.metrics_managers:
                        combined_lanes.extend(m_mgr.snapshot())
                        m_mgr.reset()

                    if combined_lanes:
                        payload = self.payload_builder.build(
                            frame_idx=batch_idx,
                            lanes_snapshot=combined_lanes,
                            meta={
                                "active_cameras": self.camera_names,
                                "fps": round(current_fps, 1),
                                "mode": "batched_production",
                                "skip_frames": self.skip_frames,
                            },
                        )
                        self.publisher.publish(payload)

                    last_pub_time = now

                # FPS Calculation
                if now - last_fps_time >= 1.0:
                    current_fps = fps_batch_counter / (now - last_fps_time)
                    fps_batch_counter = 0
                    last_fps_time = now

                self.total_processed_batches = batch_idx

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught. Shutting down pipeline...")
        finally:
            self.stop()


def build_pipeline_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart Traffic Vision - High-Throughput Production Multi-Camera Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-cams", type=int, default=4, help="Number of camera streams (1 to 8+)")
    parser.add_argument("--cameras", nargs="+", default=None, help="Named cameras to run (north south east west all)")
    parser.add_argument("--configs", nargs="+", default=None, help="Custom JSON configuration paths")
    parser.add_argument("--videos", nargs="+", default=None, help="Custom video paths or RTSP stream URLs")
    default_model = "models/yolo26s_thai_traffic.pt" if os.path.exists("models/yolo26s_thai_traffic.pt") else "yolov8s.pt"
    parser.add_argument("--model", default=default_model, help="YOLO model checkpoint or TensorRT .engine path")
    parser.add_argument("--device", default=None, help="Inference compute device: 'cuda:0', 'cpu' (default: auto)")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO detection confidence threshold")
    parser.add_argument("--fps", type=float, default=25.0, help="Target ingestion frame rate per camera")
    parser.add_argument("--skip-frames", type=int, default=1, help="Frame skipping ratio (0=none, 1=1-in-2, 2=1-in-3)")
    parser.add_argument("--buffer-size", type=int, default=2, help="Ring buffer size for jitter absorption (default: 2)")
    parser.add_argument("--display", action="store_true", help="Display live asynchronous multi-camera HUD window")
    parser.add_argument("--nvenc", default=None, help="Save live stream to hardware-encoded H.264 video (e.g. out.mp4)")
    parser.add_argument("--pub-interval", type=float, default=2.0, help="MQTT broadcast interval in seconds")
    parser.add_argument("--mqtt-broker", default="mqtt://localhost:1883", help="MQTT broker URL")
    parser.add_argument("--mqtt-topic", default="traffic/counts", help="MQTT destination topic")
    parser.add_argument("--intersection-id", default="INT-001", help="Intersection identifier string")
    parser.add_argument("--voting-window", type=int, default=15, help="Temporal voting window size in frames for class smoothing (default: 15)")
    parser.add_argument("--pickup-bias", type=float, default=1.15, help="Prior weight multiplier favoring car over truck for pickup trucks (default: 1.15)")
    parser.add_argument("--no-voting", action="store_true", help="Disable temporal class smoothing filter")
    return parser


def main():
    parser = build_pipeline_args()
    args = parser.parse_args()

    # Determine Device
    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Initialized compute device: {device}")

    # Resolve Cameras and Configs
    default_keys = list(DEFAULT_CONFIGS.keys())
    if args.cameras:
        if "all" in args.cameras:
            chosen_cams = default_keys
        else:
            chosen_cams = args.cameras
    else:
        num = max(1, args.num_cams)
        chosen_cams = [default_keys[i % len(default_keys)] if num <= 4 else f"{default_keys[i % 4]}_{i // 4 + 1}" for i in range(num)]

    configs = []
    videos = []
    for idx, cname in enumerate(chosen_cams):
        base_key = cname.split("_")[0] if "_" in cname else cname
        cfg_file = args.configs[idx] if (args.configs and idx < len(args.configs)) else DEFAULT_CONFIGS.get(base_key, DEFAULT_CONFIGS["north"])
        configs.append(cfg_file)

        if args.videos and idx < len(args.videos):
            v_src = args.videos[idx]
        else:
            cfg_data = initial_config(cfg_file)
            v_src = cfg_data["video"]["path"]
        videos.append(v_src)

    pipeline = BatchedCameraPipeline(
        camera_names=chosen_cams,
        config_paths=configs,
        video_sources=videos,
        model_path=args.model,
        device=device,
        conf=args.conf,
        target_fps=args.fps,
        skip_frames=args.skip_frames,
        buffer_size=args.buffer_size,
        display=args.display,
        nvenc_out=args.nvenc,
        pub_interval=args.pub_interval,
        mqtt_broker=args.mqtt_broker,
        mqtt_topic=args.mqtt_topic,
        intersection_id=args.intersection_id,
        voting_window=args.voting_window,
        pickup_bias=args.pickup_bias,
        enable_voting=(not args.no_voting),
    )

    # Handle OS termination signals
    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received. Stopping pipeline...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
