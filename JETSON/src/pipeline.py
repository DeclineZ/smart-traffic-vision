import os
import sys
import time
import logging
import numpy as np
import torch
import cv2 as cv
from shapely.geometry import Point
import psutil
import subprocess
from collections import defaultdict

# Add parent directory to path to find trt_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm.sort import Sort
from algorithm.ocsort import OcSort
from trt_pipeline.trt_model import TRTModel
from trt_pipeline.video_stream import VideoStream, AsyncImageSaver, letterbox
from trt_pipeline.tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, save_lane_data
)


class Pipeline:
    def __init__(self, config_path: str, engine_path: str, save_crop: bool = False, root_dir: str = None):
        self.logger = get_logger("JetsonPipeline")
        self.device = torch.device(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

        self.config = initial_config(config_path, root_dir=root_dir)
        self.save_crop = save_crop

        self.skip = int(self.config.get("skip", 0))
        self.frame_stride = max(1, self.skip)

        self.model = TRTModel(
            engine_path=engine_path,
            input_shape=(1, 3, 640, 640),
            device=self.device,
        )

        self.tracker = self.initial_tracker(self.config)
        self.dict_class = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.target_classes = set(self.dict_class.keys())

        self.tracking_zone = parse_zones(self.config["tracking"])
        self.lane_data = initial_lane_data(self.tracking_zone, self.dict_class)

        self.image_saver = AsyncImageSaver()
        self.save_dir = {}
        for lane_name, _ in self.lane_data.items():
            for cls in list(self.dict_class.values()):
                save_dir = os.path.join(self.config["output"], lane_name, cls)
                os.makedirs(save_dir, exist_ok=True)
                self.save_dir[(lane_name, cls)] = save_dir
        
        # Resource monitoring
        self.process = psutil.Process()
        self.timing_stats = defaultdict(list)
        self.resource_log_interval = 100

    def initial_tracker(self, config):
        """Initialize tracker with configuration parameters.
        
        Supports tracker types: 'sort' (default), 'ocsort', 'bytetrack'
        """
        # Extract tracker config if it exists
        tracker_config = config.get("tracker", {}) if isinstance(config, dict) else {}
        
        # Get tracker type (default to 'sort')
        tracker_type = tracker_config.get("type", "sort").lower()
        
        # Extract parameters
        max_age = tracker_config.get("max_age", 30)
        min_hits = tracker_config.get("min_hits", 3)
        iou_threshold = tracker_config.get("iou_threshold", 0.3)
        
        self.logger.info(f"Initializing tracker: {tracker_type}")
        
        if tracker_type == "ocsort":
            det_thresh = tracker_config.get("det_thresh", 0.6)
            delta_t = tracker_config.get("delta_t", 3)
            inertia = tracker_config.get("inertia", 0.2)
            use_byte = tracker_config.get("use_byte", False)
            min_conf = tracker_config.get("min_conf", 0.1)
            self._ocsort_min_conf = float(min_conf)
            return OcSort(
                det_thresh=float(det_thresh),
                max_age=int(max_age),
                min_hits=int(min_hits),
                iou_threshold=float(iou_threshold),
                delta_t=int(delta_t),
                inertia=float(inertia),
                use_byte=bool(use_byte)
            )

        self._ocsort_min_conf = None
        return Sort(
            max_age=int(max_age),
            min_hits=int(min_hits),
            iou_threshold=float(iou_threshold)
        )

    def get_gpu_memory(self):
        """Get GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                # PyTorch CUDA memory
                allocated = torch.cuda.memory_allocated(self.device) / 1024**2
                reserved = torch.cuda.memory_reserved(self.device) / 1024**2
                return allocated, reserved
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory: {e}")
        return 0, 0

    def get_system_resources(self):
        """Get system CPU and RAM usage."""
        try:
            cpu_percent = self.process.cpu_percent()
            ram_mb = self.process.memory_info().rss / 1024**2
            ram_percent = self.process.memory_percent()
            return cpu_percent, ram_mb, ram_percent
        except Exception as e:
            self.logger.warning(f"Could not get system resources: {e}")
            return 0, 0, 0

    def log_resources(self, frame_idx, stage_timings=None):
        """Log current resource usage."""
        gpu_alloc, gpu_reserved = self.get_gpu_memory()
        cpu_percent, ram_mb, ram_percent = self.get_system_resources()
        
        log_msg = (
            f"\n{'='*60}\n"
            f"Frame {frame_idx} Resource Usage:\n"
            f"  GPU Memory: {gpu_alloc:.1f} MB allocated, {gpu_reserved:.1f} MB reserved\n"
            f"  CPU Usage: {cpu_percent:.1f}%\n"
            f"  RAM Usage: {ram_mb:.1f} MB ({ram_percent:.1f}%)\n"
        )
        
        if stage_timings:
            log_msg += "  Stage Timings (ms):\n"
            for stage, duration in stage_timings.items():
                log_msg += f"    {stage}: {duration*1000:.2f} ms\n"
        
        log_msg += f"{'='*60}"
        self.logger.info(log_msg)

    def _closest_class_id(self, centroid, dets):
        if dets.size == 0:
            return None
        centers = np.stack([
            (dets[:, 0] + dets[:, 2]) / 2,
            (dets[:, 1] + dets[:, 3]) / 2
        ], axis=1)
        dists = np.linalg.norm(centers - np.array(centroid), axis=1)
        idx = int(np.argmin(dists))
        return int(dets[idx, 5])

    def run(self):
        cv.setNumThreads(0)
        cv.ocl.setUseOpenCL(False)
        max_frames = int(25 * 3600)

        stream = VideoStream(
            video_path=self.config["video"],
            skip=self.frame_stride,
            queue_size=2,
        )

        total_start = time.perf_counter()
        processed_frames = 0
        
        self.logger.info("Starting pipeline with resource monitoring...")

        while True:
            frame_start = time.perf_counter()
            stage_timings = {}
            
            # Video read
            t0 = time.perf_counter()
            item = stream.read()
            if item is None:
                break
            stage_timings['video_read'] = time.perf_counter() - t0

            frame_idx, frame_bgr = item
            processed_frames = frame_idx
            if frame_idx % 10 == 0:
                print(f"\rframe: {frame_idx}", end="", flush=True)

            if frame_idx > max_frames:
                break

            # Preprocessing
            t0 = time.perf_counter()
            img_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=(640, 640), auto=False)
            img_chw = img_lb.transpose(2, 0, 1)
            img_chw = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
            stage_timings['preprocessing'] = time.perf_counter() - t0

            # Inference
            t0 = time.perf_counter()
            _, outputs = self.model.infer(input_tensor)
            stage_timings['inference'] = time.perf_counter() - t0
            
            # Post-inference processing
            t0 = time.perf_counter()
            num = int(outputs["num_dets"][0])
            boxes = outputs["det_boxes"][0][:num].cpu().numpy()
            scores = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            if dets.size:
                dets = dets[np.isin(dets[:, 5].astype(int), list(self.target_classes))]
            stage_timings['post_inference'] = time.perf_counter() - t0

            # Tracking
            t0 = time.perf_counter()
            boxes_only = dets[:, :4] if dets.size else np.empty((0, 4))
            if isinstance(self.tracker, OcSort):
                tracker_input = dets[:, :5] if dets.size else np.empty((0, 5))
                min_conf = 0.1 if self._ocsort_min_conf is None else self._ocsort_min_conf
                tracker_objects = self.tracker.update(tracker_input, min_conf=min_conf)
            else:
                tracker_objects = self.tracker.update(boxes_only)
            stage_timings['tracking'] = time.perf_counter() - t0

            # Lane crossing detection and cropping
            t0 = time.perf_counter()
            h, w = frame_bgr.shape[:2]

            for x1, y1, x2, y2, track_id in tracker_objects:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_id = self._closest_class_id((cx, cy), dets)
                if class_id is None or class_id not in self.target_classes:
                    continue

                cxo, cyo = to_original_coords(cx, cy, dw, dh, ratio)
                x1o, y1o = to_original_coords(x1, y1, dw, dh, ratio)
                x2o, y2o = to_original_coords(x2, y2, dw, dh, ratio)

                x1c = int(max(0, min(w, x1o)))
                y1c = int(max(0, min(h, y1o)))
                x2c = int(max(0, min(w, x2o)))
                y2c = int(max(0, min(h, y2o)))

                if x2c <= x1c or y2c <= y1c:
                    continue

                for lane_name, lane in self.lane_data.items():
                    if int(track_id) in lane["cross_ids"]:
                        continue

                    if (
                        lane["polygon"] is not None
                        and lane["polygon"].contains(Point(cxo, cyo))
                        and side_of_line((cxo, cyo), lane["line"][0], lane["line"][1]) < 0
                    ):
                        lane["cross_ids"].add(int(track_id))
                        lane["count_cls"][class_id] += 1
                        lane["cross_obj"].append(
                            {
                                "id": int(track_id),
                                "frame": frame_idx,
                                "class_id": int(class_id),
                                "bbox": (x1c, y1c, x2c, y2c),
                            }
                        )

                        if self.save_crop:
                            crop = frame_bgr[y1c:y2c, x1c:x2c]
                            save_path = os.path.join(
                                self.save_dir[(lane_name, self.dict_class[int(class_id)])],
                                f"frame_{frame_idx}_id_{int(track_id)}.jpg"
                            )
                            self.image_saver.save(save_path, crop)
            
            stage_timings['lane_crossing'] = time.perf_counter() - t0
            
            # Total frame time
            frame_time = time.perf_counter() - frame_start
            stage_timings['total_frame'] = frame_time
            
            # Store timing stats
            for stage, duration in stage_timings.items():
                self.timing_stats[stage].append(duration)
            
            # Periodic resource logging
            if frame_idx % self.resource_log_interval == 0:
                avg_timings = {
                    stage: np.mean(durations[-self.resource_log_interval:])
                    for stage, durations in self.timing_stats.items()
                }
                self.log_resources(frame_idx, avg_timings)

        total_time = time.perf_counter() - total_start
        fps = (processed_frames / total_time) if total_time > 0 else 0.0
        self.logger.info(f"Total time: {total_time:.2f}s | FPS: {fps:.2f}")
        
        # Compute and log detailed timing statistics
        self.logger.info("\n" + "="*60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("="*60)
        
        for stage in ['video_read', 'preprocessing', 'inference', 'post_inference', 
                      'tracking', 'lane_crossing', 'total_frame']:
            if stage in self.timing_stats:
                timings = self.timing_stats[stage]
                avg_ms = np.mean(timings) * 1000
                std_ms = np.std(timings) * 1000
                min_ms = np.min(timings) * 1000
                max_ms = np.max(timings) * 1000
                median_ms = np.median(timings) * 1000
                
                self.logger.info(
                    f"{stage:20s}: avg={avg_ms:6.2f}ms  std={std_ms:5.2f}ms  "
                    f"min={min_ms:6.2f}ms  max={max_ms:6.2f}ms  median={median_ms:6.2f}ms"
                )
        
        # Final resource check
        gpu_alloc, gpu_reserved = self.get_gpu_memory()
        cpu_percent, ram_mb, ram_percent = self.get_system_resources()
        self.logger.info("="*60)
        self.logger.info(f"Final GPU Memory: {gpu_alloc:.1f} MB allocated, {gpu_reserved:.1f} MB reserved")
        self.logger.info(f"Final CPU Usage: {cpu_percent:.1f}%")
        self.logger.info(f"Final RAM Usage: {ram_mb:.1f} MB ({ram_percent:.1f}%)")
        self.logger.info("="*60)

        inference_path = os.path.join(self.config["output"], "inference.txt")
        with open(inference_path, "w") as f:
            f.write(f"Total frames: {processed_frames}\n")
            f.write(f"Total time (s): {total_time:.2f}\n")
            f.write(f"Average FPS: {fps:.2f}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("Stage Timings (ms):\n")
            f.write("="*60 + "\n")
            for stage in ['video_read', 'preprocessing', 'inference', 'post_inference',
                          'tracking', 'lane_crossing', 'total_frame']:
                if stage in self.timing_stats:
                    timings = self.timing_stats[stage]
                    avg_ms = np.mean(timings) * 1000
                    std_ms = np.std(timings) * 1000
                    min_ms = np.min(timings) * 1000
                    max_ms = np.max(timings) * 1000
                    median_ms = np.median(timings) * 1000
                    f.write(
                        f"{stage:20s}: avg={avg_ms:6.2f}ms  std={std_ms:5.2f}ms  "
                        f"min={min_ms:6.2f}ms  max={max_ms:6.2f}ms  median={median_ms:6.2f}ms\n"
                    )
            f.write("\n" + "="*60 + "\n")
            f.write("Resource Usage:\n")
            f.write("="*60 + "\n")
            f.write(f"Final GPU Memory: {gpu_alloc:.1f} MB allocated, {gpu_reserved:.1f} MB reserved\n")
            f.write(f"Final CPU Usage: {cpu_percent:.1f}%\n")
            f.write(f"Final RAM Usage: {ram_mb:.1f} MB ({ram_percent:.1f}%)\n")

        cleanup()
        self.image_saver.stop()
        self.logger.info("Resources cleaned up.")

        save_lane_data(self.lane_data, os.path.join(self.config["output"], "lane_data.json"))
        self.logger.info("Outputs saved successfully.")