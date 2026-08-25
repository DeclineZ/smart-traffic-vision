from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import psutil
import torch
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm.sort import Sort

# Optional NVML for zero-overhead C-level GPU metrics
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# Default 1080p surveillance video feeds
DEFAULT_VIDEOS = [
    "videos/cam44_north.avi",
    "videos/cam43_south.avi",
    "videos/cam03_east.avi",
    "videos/cam46_west.avi",
]

COCO_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ==============================================================================
# 1. HARDWARE PROFILER (Direct NVML + psutil)
# ==============================================================================
class HardwareProfiler:
    """
    High-frequency background hardware sampler measuring:
    - GPU Core Compute Utilization (%) & Memory Controller Bus Utilization (%)
    - Dedicated VRAM Used (MB / GB) & PyTorch Allocated / Reserved VRAM
    - GPU Temperature (°C) & Power Draw (Watts)
    - Process CPU % & System Total CPU % & Per-Core CPU Distribution (Min, Max, Std)
    - Process RAM (RSS, Peak RSS) & Total System RAM (GB, %)
    - Process Thread Count
    """

    def __init__(self, sample_interval: float = 0.05, gpu_index: int = 0):
        self.interval = sample_interval
        self.gpu_index = gpu_index
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process = psutil.Process(os.getpid())

        # NVML Handle
        self.nvml_handle = None
        if HAS_NVML:
            try:
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self.nvml_handle = None

        # Sample history
        self.timestamps: List[float] = []
        self.process_cpu_samples: List[float] = []
        self.system_cpu_samples: List[float] = []
        self.per_core_cpu_samples: List[List[float]] = []
        self.process_ram_mb_samples: List[float] = []
        self.system_ram_gb_samples: List[float] = []
        self.system_ram_pct_samples: List[float] = []

        self.gpu_util_samples: List[float] = []
        self.gpu_mem_bus_samples: List[float] = []
        self.vram_used_mb_samples: List[float] = []
        self.torch_vram_alloc_mb_samples: List[float] = []
        self.torch_vram_res_mb_samples: List[float] = []
        self.gpu_temp_samples: List[float] = []
        self.gpu_power_w_samples: List[float] = []

        self.baseline_metrics: Dict[str, float] = {}

    def capture_baseline(self, duration: float = 1.0) -> Dict[str, float]:
        """Samples idle hardware baseline before workload runs."""
        print("   [~] Calibrating baseline idle hardware load...")
        b_cpu, b_proc_ram, b_gpu, b_vram, b_pwr = [], [], [], [], []
        t_end = time.perf_counter() + duration

        # Prime psutil cpu measurement
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        while time.perf_counter() < t_end:
            b_cpu.append(psutil.cpu_percent(interval=None))
            b_proc_ram.append(self.process.memory_info().rss / (1024 * 1024))
            if self.nvml_handle:
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    pwr = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                    b_gpu.append(rates.gpu)
                    b_vram.append(mem.used / (1024 * 1024))
                    b_pwr.append(pwr)
                except Exception:
                    pass
            elif torch.cuda.is_available():
                b_vram.append(torch.cuda.memory_allocated(self.gpu_index) / (1024 * 1024))
                b_gpu.append(0.0)
                b_pwr.append(0.0)
            time.sleep(0.05)

        self.baseline_metrics = {
            "baseline_cpu_pct": float(np.mean(b_cpu)) if b_cpu else 0.0,
            "baseline_ram_mb": float(np.mean(b_proc_ram)) if b_proc_ram else 0.0,
            "baseline_gpu_pct": float(np.mean(b_gpu)) if b_gpu else 0.0,
            "baseline_vram_mb": float(np.mean(b_vram)) if b_vram else 0.0,
            "baseline_power_w": float(np.mean(b_pwr)) if b_pwr else 0.0,
        }
        return self.baseline_metrics

    def start(self):
        self._clear_samples()
        self.running = True
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def _clear_samples(self):
        self.timestamps.clear()
        self.process_cpu_samples.clear()
        self.system_cpu_samples.clear()
        self.per_core_cpu_samples.clear()
        self.process_ram_mb_samples.clear()
        self.system_ram_gb_samples.clear()
        self.system_ram_pct_samples.clear()
        self.gpu_util_samples.clear()
        self.gpu_mem_bus_samples.clear()
        self.vram_used_mb_samples.clear()
        self.torch_vram_alloc_mb_samples.clear()
        self.torch_vram_res_mb_samples.clear()
        self.gpu_temp_samples.clear()
        self.gpu_power_w_samples.clear()

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Calculate summary statistics
        num_cores = psutil.cpu_count(logical=True) or 1
        proc_cpu = np.array(self.process_cpu_samples) if self.process_cpu_samples else np.zeros(1)
        proc_cpu_norm = proc_cpu / num_cores

        sys_cpu = np.array(self.system_cpu_samples) if self.system_cpu_samples else np.zeros(1)
        proc_ram = np.array(self.process_ram_mb_samples) if self.process_ram_mb_samples else np.zeros(1)
        sys_ram_gb = np.array(self.system_ram_gb_samples) if self.system_ram_gb_samples else np.zeros(1)
        gpu_util = np.array(self.gpu_util_samples) if self.gpu_util_samples else np.zeros(1)
        gpu_mem_bus = np.array(self.gpu_mem_bus_samples) if self.gpu_mem_bus_samples else np.zeros(1)
        vram_used = np.array(self.vram_used_mb_samples) if self.vram_used_mb_samples else np.zeros(1)
        torch_vram = np.array(self.torch_vram_alloc_mb_samples) if self.torch_vram_alloc_mb_samples else np.zeros(1)
        gpu_temp = np.array(self.gpu_temp_samples) if self.gpu_temp_samples else np.zeros(1)
        gpu_pwr = np.array(self.gpu_power_w_samples) if self.gpu_power_w_samples else np.zeros(1)

        # Per core spread
        per_core_arr = np.array(self.per_core_cpu_samples) if self.per_core_cpu_samples else np.zeros((1, num_cores))
        core_means = np.mean(per_core_arr, axis=0) if len(per_core_arr) else np.zeros(num_cores)

        return {
            "duration_sec": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0,
            "sample_count": len(self.timestamps),
            # CPU Metrics
            "avg_process_cpu_pct": float(np.mean(proc_cpu)),
            "avg_process_cpu_norm_pct": float(np.mean(proc_cpu_norm)),
            "peak_process_cpu_pct": float(np.max(proc_cpu)),
            "avg_system_cpu_pct": float(np.mean(sys_cpu)),
            "peak_system_cpu_pct": float(np.max(sys_cpu)),
            "p95_system_cpu_pct": float(np.percentile(sys_cpu, 95)),
            "core_max_util_pct": float(np.max(core_means)),
            "core_min_util_pct": float(np.min(core_means)),
            "core_load_std_pct": float(np.std(core_means)),
            # RAM Metrics
            "avg_process_ram_mb": float(np.mean(proc_ram)),
            "peak_process_ram_mb": float(np.max(proc_ram)),
            "delta_process_ram_mb": float(np.max(proc_ram) - self.baseline_metrics.get("baseline_ram_mb", 0.0)),
            "avg_system_ram_gb": float(np.mean(sys_ram_gb)),
            "peak_system_ram_gb": float(np.max(sys_ram_gb)),
            # GPU Metrics
            "avg_gpu_util_pct": float(np.mean(gpu_util)),
            "peak_gpu_util_pct": float(np.max(gpu_util)),
            "p95_gpu_util_pct": float(np.percentile(gpu_util, 95)),
            "avg_gpu_mem_bus_pct": float(np.mean(gpu_mem_bus)),
            "peak_gpu_mem_bus_pct": float(np.max(gpu_mem_bus)),
            "avg_vram_mb": float(np.mean(vram_used)),
            "peak_vram_mb": float(np.max(vram_used)),
            "delta_vram_mb": float(np.max(vram_used) - self.baseline_metrics.get("baseline_vram_mb", 0.0)),
            "peak_torch_vram_mb": float(np.max(torch_vram)),
            "avg_gpu_temp_c": float(np.mean(gpu_temp)),
            "peak_gpu_temp_c": float(np.max(gpu_temp)),
            "avg_gpu_power_w": float(np.mean(gpu_pwr)),
            "peak_gpu_power_w": float(np.max(gpu_pwr)),
            # Raw time series for plotting
            "time_series": {
                "timestamps": [t - self.timestamps[0] for t in self.timestamps] if self.timestamps else [],
                "process_cpu": [float(x) for x in self.process_cpu_samples],
                "system_cpu": [float(x) for x in self.system_cpu_samples],
                "gpu_util": [float(x) for x in self.gpu_util_samples],
                "gpu_mem_bus": [float(x) for x in self.gpu_mem_bus_samples],
                "vram_mb": [float(x) for x in self.vram_used_mb_samples],
                "process_ram_mb": [float(x) for x in self.process_ram_mb_samples],
                "gpu_temp_c": [float(x) for x in self.gpu_temp_samples],
                "gpu_power_w": [float(x) for x in self.gpu_power_w_samples],
            },
        }

    def _sample_loop(self):
        while self.running:
            t_now = time.perf_counter()
            self.timestamps.append(t_now)

            # CPU & RAM
            try:
                self.process_cpu_samples.append(self.process.cpu_percent(interval=None))
                self.system_cpu_samples.append(psutil.cpu_percent(interval=None))
                self.per_core_cpu_samples.append(psutil.cpu_percent(interval=None, percpu=True))
                mem_info = self.process.memory_info()
                self.process_ram_mb_samples.append(mem_info.rss / (1024 * 1024))

                sys_mem = psutil.virtual_memory()
                self.system_ram_gb_samples.append(sys_mem.used / (1024**3))
                self.system_ram_pct_samples.append(sys_mem.percent)
            except Exception:
                pass

            # GPU Metrics via NVML
            if self.nvml_handle:
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    pwr = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0

                    self.gpu_util_samples.append(float(rates.gpu))
                    self.gpu_mem_bus_samples.append(float(rates.memory))
                    self.vram_used_mb_samples.append(float(mem.used / (1024 * 1024)))
                    self.gpu_temp_samples.append(float(temp))
                    self.gpu_power_w_samples.append(float(pwr))
                except Exception:
                    pass
            elif torch.cuda.is_available():
                self.gpu_util_samples.append(0.0)
                self.gpu_mem_bus_samples.append(0.0)
                self.vram_used_mb_samples.append(torch.cuda.memory_allocated(self.gpu_index) / (1024 * 1024))
                self.gpu_temp_samples.append(0.0)
                self.gpu_power_w_samples.append(0.0)

            if torch.cuda.is_available():
                self.torch_vram_alloc_mb_samples.append(torch.cuda.memory_allocated(self.gpu_index) / (1024 * 1024))
                self.torch_vram_res_mb_samples.append(torch.cuda.memory_reserved(self.gpu_index) / (1024 * 1024))

            time.sleep(self.interval)


# ==============================================================================
# 2. RTSP STREAM SIMULATOR & LIVE INGESTION
# ==============================================================================
class RTSPStreamSimulator:
    """
    Decoupled background stream producer that simulates real IP / RTSP network cameras:
    - Paces frame ingestion at camera native FPS (e.g. 15, 25, 30 FPS).
    - Maintains a bounded ring buffer (queue size = 1 or 2).
    - If the inference worker experiences backpressure / lag, it drops intermediate
      frames just like a real RTSP client (`cv.CAP_PROP_BUFFERSIZE = 1`).
    - Tracks dropped frames count, drop rate %, and frame arrival latency.
    """

    def __init__(self, source: str, target_fps: float = 25.0, buffer_size: int = 1, is_paced: bool = True):
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.is_paced = is_paced
        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.frames_ingested = 0
        self.frames_dropped = 0
        self.cap: Optional[cv.VideoCapture] = None

    def start(self):
        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video stream source: {self.source}")

        self.running = True
        self.frames_ingested = 0
        self.frames_dropped = 0
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _capture_loop(self):
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        while self.running and self.cap and self.cap.isOpened():
            t_loop_start = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                # Video file loop
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

            t_captured = time.perf_counter()
            self.frames_ingested += 1

            # Push to bounded queue; drop oldest frame if full (simulating live RTSP stream)
            item = (t_captured, frame)
            if self.queue.full():
                try:
                    _ = self.queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.frames_dropped += 1

            # Pacing
            if self.is_paced and frame_interval > 0:
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

    def get_frame(self, timeout: float = 0.5) -> Optional[Tuple[float, np.ndarray]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def drop_rate_pct(self) -> float:
        total = self.frames_ingested
        return (self.frames_dropped / total * 100.0) if total > 0 else 0.0


# ==============================================================================
# 3. STAGE LATENCY TRACKER
# ==============================================================================
class StageLatencyTracker:
    """Collects microsecond-accurate latency measurements for each pipeline stage."""

    def __init__(self):
        self.t_decode: List[float] = []
        self.t_preprocess: List[float] = []
        self.t_inference: List[float] = []
        self.t_tracking: List[float] = []
        self.t_analytics: List[float] = []
        self.t_visualize: List[float] = []
        self.t_e2e: List[float] = []

    def record(
        self,
        t_dec_ms: float,
        t_pre_ms: float,
        t_inf_ms: float,
        t_trk_ms: float,
        t_ana_ms: float,
        t_vis_ms: float,
        t_e2e_ms: float,
    ):
        self.t_decode.append(t_dec_ms)
        self.t_preprocess.append(t_pre_ms)
        self.t_inference.append(t_inf_ms)
        self.t_tracking.append(t_trk_ms)
        self.t_analytics.append(t_ana_ms)
        self.t_visualize.append(t_vis_ms)
        self.t_e2e.append(t_e2e_ms)

    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
            np_a = np.array(arr)
            return {
                "mean": float(np.mean(np_a)),
                "p50": float(np.percentile(np_a, 50)),
                "p90": float(np.percentile(np_a, 90)),
                "p95": float(np.percentile(np_a, 95)),
                "p99": float(np.percentile(np_a, 99)),
                "max": float(np.max(np_a)),
            }

        return {
            "decode_ms": _stats(self.t_decode),
            "preprocess_ms": _stats(self.t_preprocess),
            "inference_ms": _stats(self.t_inference),
            "tracking_ms": _stats(self.t_tracking),
            "analytics_ms": _stats(self.t_analytics),
            "visualize_ms": _stats(self.t_visualize),
            "e2e_ms": _stats(self.t_e2e),
        }


# ==============================================================================
# 4. WORKER: THREADED PIPELINE (Independent Camera Workers)
# ==============================================================================
class ThreadedCameraWorker(threading.Thread):
    def __init__(
        self,
        stream_id: int,
        stream_source: str,
        model: YOLO,
        device: str = "cuda:0",
        conf: float = 0.20,
        imgsz: int = 640,
        skip_frames: int = 0,
        target_fps: float = 25.0,
        is_paced: bool = True,
        display: bool = False,
        model_lock: Optional[threading.Lock] = None,
    ):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.source = stream_source
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.skip_frames = skip_frames
        self.display = display
        self.model_lock = model_lock or threading.Lock()

        self.simulator = RTSPStreamSimulator(source=stream_source, target_fps=target_fps, is_paced=is_paced)
        self.tracker = Sort(max_age=25, min_hits=2, iou_threshold=0.3)
        self.test_poly = Polygon([[100, 100], [1800, 100], [1800, 1000], [100, 1000]])

        self.latency_tracker = StageLatencyTracker()
        self.running = False
        self.processed_frames = 0
        self.inferred_frames = 0
        self.skipped_frames_count = 0
        self.last_dets = np.empty((0, 5))
        self.latest_frame: Optional[np.ndarray] = None

    def run(self):
        self.simulator.start()
        self.running = True
        frame_counter = 0

        while self.running:
            t0 = time.perf_counter()

            # 1. Decode & Ingestion
            item = self.simulator.get_frame(timeout=0.2)
            if item is None:
                continue
            t_captured, frame = item
            t_dec = (time.perf_counter() - t0) * 1000.0

            frame_counter += 1
            should_run_yolo = (frame_counter % (self.skip_frames + 1)) == 0

            # 2. Preprocess & 3. Inference
            t_pre = 0.0
            t_inf = 0.0

            dets_arr = np.empty((0, 5))

            if should_run_yolo:
                t_pre0 = time.perf_counter()
                with self.model_lock:
                    t_pre = (time.perf_counter() - t_pre0) * 1000.0
                    t_infer0 = time.perf_counter()

                    results = self.model(
                        frame,
                        verbose=False,
                        device=self.device,
                        classes=list(COCO_CLASSES.keys()),
                        conf=self.conf,
                        imgsz=self.imgsz,
                    )
                    if "cuda" in self.device and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    t_inf = (time.perf_counter() - t_infer0) * 1000.0

                dets = []
                if len(results) and len(results[0].boxes):
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        dets.append([x1, y1, x2, y2, conf])
                dets_arr = np.array(dets) if len(dets) else np.empty((0, 5))
                self.last_dets = dets_arr
                self.inferred_frames += 1
            else:
                self.skipped_frames_count += 1
                dets_arr = np.empty((0, 5))

            # 4. Tracking (SORT)
            t_trk0 = time.perf_counter()
            tracked_objs = self.tracker.update(dets_arr)
            t_trk = (time.perf_counter() - t_trk0) * 1000.0

            # 5. Spatial Analytics (Polygon containment)
            t_ana0 = time.perf_counter()
            for obj in tracked_objs:
                cx, cy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
                _ = self.test_poly.contains(Point(cx, cy))
            t_ana = (time.perf_counter() - t_ana0) * 1000.0

            # 6. UI Drawing & Visual HUD (if display is enabled)
            t_vis0 = time.perf_counter()
            if self.display:
                vis = cv.resize(frame, (640, 360))
                scale_x, scale_y = 640 / frame.shape[1], 360 / frame.shape[0]

                # Draw polygon
                poly_pts = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in self.test_poly.exterior.coords], dtype=np.int32)
                cv.polylines(vis, [poly_pts], True, (0, 255, 0), 2)

                # Draw bounding boxes
                for obj in tracked_objs:
                    ox1, oy1, ox2, oy2, otrack_id = obj[:5]
                    bx1, by1 = int(ox1 * scale_x), int(oy1 * scale_y)
                    bx2, by2 = int(ox2 * scale_x), int(oy2 * scale_y)
                    cv.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv.putText(vis, f"#{int(otrack_id)}", (bx1, max(15, by1 - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv.putText(vis, f"CAM {self.stream_id:02d}", (15, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.latest_frame = vis
            t_vis = (time.perf_counter() - t_vis0) * 1000.0

            # End-to-End Latency
            t_e2e = (time.perf_counter() - t_captured) * 1000.0

            self.latency_tracker.record(t_dec, t_pre, t_inf, t_trk, t_ana, t_vis, t_e2e)
            self.processed_frames += 1

    def stop(self):
        self.running = False
        self.simulator.stop()


# ==============================================================================
# 5. WORKER: BATCHED PIPELINE (High-Throughput Centralized Batching)
# ==============================================================================
class BatchedCameraPipeline:
    """
    Centralized batching engine: pulls frames from N cameras and feeds
    batch tensor [B, 3, H, W] into YOLO in a single forward pass.
    """

    def __init__(
        self,
        stream_sources: List[str],
        model: YOLO,
        device: str = "cuda:0",
        conf: float = 0.20,
        imgsz: int = 640,
        skip_frames: int = 0,
        target_fps: float = 25.0,
        is_paced: bool = True,
        display: bool = False,
    ):
        self.stream_sources = stream_sources
        self.num_streams = len(stream_sources)
        self.model = model
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.skip_frames = skip_frames
        self.display = display

        self.simulators = [
            RTSPStreamSimulator(source=src, target_fps=target_fps, is_paced=is_paced) for src in stream_sources
        ]
        self.trackers = [Sort(max_age=25, min_hits=2, iou_threshold=0.3) for _ in range(self.num_streams)]
        self.test_poly = Polygon([[100, 100], [1800, 100], [1800, 1000], [100, 1000]])

        self.latency_tracker = StageLatencyTracker()
        self.running = False
        self.processed_frames = 0
        self.inferred_batches = 0
        self.total_dropped_frames = 0

    def run_for(self, duration_sec: float):
        for sim in self.simulators:
            sim.start()

        self.running = True
        t_start = time.perf_counter()
        batch_counter = 0

        while self.running and (time.perf_counter() - t_start < duration_sec):
            t0 = time.perf_counter()

            # 1. Decode & collect batch
            frames = []
            capture_times = []
            for sim in self.simulators:
                item = sim.get_frame(timeout=0.2)
                if item:
                    capture_times.append(item[0])
                    frames.append(item[1])

            if len(frames) < self.num_streams:
                continue

            t_dec = (time.perf_counter() - t0) * 1000.0
            batch_counter += 1
            should_run_yolo = (batch_counter % (self.skip_frames + 1)) == 0

            # 2. Batch Inference
            t_pre = 0.0
            t_inf = 0.0
            results = []

            if should_run_yolo:
                t_pre0 = time.perf_counter()
                t_pre = (time.perf_counter() - t_pre0) * 1000.0
                t_inf0 = time.perf_counter()

                results = self.model(
                    frames,
                    verbose=False,
                    device=self.device,
                    classes=list(COCO_CLASSES.keys()),
                    conf=self.conf,
                    imgsz=self.imgsz,
                )
                if "cuda" in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_inf = (time.perf_counter() - t_inf0) * 1000.0
                self.inferred_batches += 1

            # 3. Fan-out to Trackers & Analytics
            t_trk0 = time.perf_counter()
            tracked_list = []
            for idx in range(self.num_streams):
                dets_arr = np.empty((0, 5))
                if should_run_yolo and idx < len(results) and len(results[idx].boxes):
                    dets = []
                    for box in results[idx].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        dets.append([x1, y1, x2, y2, conf])
                    dets_arr = np.array(dets) if len(dets) else np.empty((0, 5))

                tracked = self.trackers[idx].update(dets_arr)
                tracked_list.append(tracked)

                # Geometry
                for obj in tracked:
                    cx, cy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
                    _ = self.test_poly.contains(Point(cx, cy))

            t_trk_ana = (time.perf_counter() - t_trk0) * 1000.0

            # 4. Visual Rendering & Multi-Camera Display Grid
            t_vis0 = time.perf_counter()
            if self.display:
                vis_frames = []
                for idx in range(self.num_streams):
                    f = frames[idx]
                    vis = cv.resize(f, (480, 270))
                    scale_x, scale_y = 480 / f.shape[1], 270 / f.shape[0]

                    # Draw poly
                    poly_pts = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in self.test_poly.exterior.coords], dtype=np.int32)
                    cv.polylines(vis, [poly_pts], True, (0, 255, 0), 2)

                    # Draw boxes
                    for obj in tracked_list[idx]:
                        ox1, oy1, ox2, oy2, otrack_id = obj[:5]
                        bx1, by1 = int(ox1 * scale_x), int(oy1 * scale_y)
                        bx2, by2 = int(ox2 * scale_x), int(oy2 * scale_y)
                        cv.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                        cv.putText(vis, f"#{int(otrack_id)}", (bx1, max(12, by1 - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    cv.putText(vis, f"CAM {idx+1:02d}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    vis_frames.append(vis)

                # Stitch Grid
                if len(vis_frames) == 1:
                    grid = vis_frames[0]
                elif len(vis_frames) == 2:
                    grid = np.hstack(vis_frames)
                elif len(vis_frames) == 4:
                    top = np.hstack(vis_frames[:2])
                    bot = np.hstack(vis_frames[2:])
                    grid = np.vstack([top, bot])
                elif len(vis_frames) == 8:
                    row1 = np.hstack(vis_frames[:4])
                    row2 = np.hstack(vis_frames[4:])
                    grid = np.vstack([row1, row2])
                else:
                    grid = vis_frames[0]

                cv.imshow("Smart Traffic Vision - Multi-Camera Hardware Benchmark HUD", grid)
                cv.waitKey(1)
            t_vis = (time.perf_counter() - t_vis0) * 1000.0

            t_e2e = (time.perf_counter() - np.mean(capture_times)) * 1000.0

            self.latency_tracker.record(
                t_dec / self.num_streams,
                t_pre,
                t_inf / self.num_streams,
                t_trk_ana / self.num_streams * 0.7,
                t_trk_ana / self.num_streams * 0.3,
                t_vis / self.num_streams,
                t_e2e,
            )
            self.processed_frames += self.num_streams

        self.running = False
        if self.display:
            cv.destroyAllWindows()
        for sim in self.simulators:
            self.total_dropped_frames += sim.frames_dropped
            sim.stop()


# ==============================================================================
# 6. BENCHMARK RUNNER & EXPERIMENT ORCHESTRATOR
# ==============================================================================
def run_single_test(
    model_name: str,
    n_streams: int,
    video_sources: List[str],
    pipeline_mode: str = "threaded",
    duration_sec: float = 10.0,
    skip_frames: int = 0,
    target_fps: float = 25.0,
    is_paced: bool = True,
    display: bool = False,
    imgsz: int = 640,
    device: str = "cuda:0",
    profiler: Optional[HardwareProfiler] = None,
) -> Dict[str, Any]:
    print(f"\n   ----------------------------------------------------------------------")
    print(
        f"   ▶ Running: {model_name} | {n_streams} Camera Stream(s) | Mode: {pipeline_mode.upper()} | Skip: {skip_frames} | Display HUD: {display} | Paced: {is_paced}"
    )
    print(f"   ----------------------------------------------------------------------")

    # Load Model
    model = YOLO(model_name)

    # Prepare stream sources
    sources = [video_sources[i % len(video_sources)] for i in range(n_streams)]

    if profiler is None:
        profiler = HardwareProfiler(sample_interval=0.05)

    # Warmup Model
    print("   [~] Warming up GPU model caches...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, verbose=False, device=device, imgsz=imgsz)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Start Profiling
    profiler.start()
    t_start = time.perf_counter()

    latencies_summary = {}
    total_processed = 0
    total_dropped = 0
    total_ingested = 0

    if pipeline_mode == "threaded":
        model_lock = threading.Lock()
        workers: List[ThreadedCameraWorker] = []
        for idx in range(n_streams):
            w = ThreadedCameraWorker(
                stream_id=idx + 1,
                stream_source=sources[idx],
                model=model,
                device=device,
                imgsz=imgsz,
                skip_frames=skip_frames,
                target_fps=target_fps,
                is_paced=is_paced,
                display=display,
                model_lock=model_lock,
            )
            workers.append(w)
            w.start()

        # Run duration with optional display pump
        if display:
            t_end_loop = time.perf_counter() + duration_sec
            while time.perf_counter() < t_end_loop:
                frames = [w.latest_frame for w in workers if w.latest_frame is not None]
                if len(frames) == len(workers) and len(frames) > 0:
                    if len(frames) == 1:
                        grid = frames[0]
                    elif len(frames) == 2:
                        grid = np.hstack(frames)
                    elif len(frames) == 4:
                        grid = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
                    elif len(frames) == 8:
                        grid = np.vstack([np.hstack(frames[:4]), np.hstack(frames[4:])])
                    else:
                        grid = frames[0]
                    cv.imshow("Smart Traffic Vision - Multi-Camera Hardware Benchmark HUD", grid)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(0.02)
            cv.destroyAllWindows()
        else:
            time.sleep(duration_sec)

        # Stop workers
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2.0)

        t_elapsed = time.perf_counter() - t_start
        hw_metrics = profiler.stop()

        total_processed = sum(w.processed_frames for w in workers)
        total_dropped = sum(w.simulator.frames_dropped for w in workers)
        total_ingested = sum(w.simulator.frames_ingested for w in workers)

        # Combine latencies
        combined_tracker = StageLatencyTracker()
        for w in workers:
            st = w.latency_tracker
            combined_tracker.t_decode.extend(st.t_decode)
            combined_tracker.t_preprocess.extend(st.t_preprocess)
            combined_tracker.t_inference.extend(st.t_inference)
            combined_tracker.t_tracking.extend(st.t_tracking)
            combined_tracker.t_analytics.extend(st.t_analytics)
            combined_tracker.t_visualize.extend(st.t_visualize)
            combined_tracker.t_e2e.extend(st.t_e2e)
        latencies_summary = combined_tracker.compute_summary()

    elif pipeline_mode == "batched":
        pipeline = BatchedCameraPipeline(
            stream_sources=sources,
            model=model,
            device=device,
            imgsz=imgsz,
            skip_frames=skip_frames,
            target_fps=target_fps,
            is_paced=is_paced,
            display=display,
        )
        pipeline.run_for(duration_sec)
        t_elapsed = time.perf_counter() - t_start
        hw_metrics = profiler.stop()

        total_processed = pipeline.processed_frames
        total_dropped = pipeline.total_dropped_frames
        total_ingested = sum(sim.frames_ingested for sim in pipeline.simulators)
        latencies_summary = pipeline.latency_tracker.compute_summary()

    # Calculate throughput
    total_fps = total_processed / max(1e-5, t_elapsed)
    fps_per_camera = total_fps / max(1, n_streams)
    drop_rate_pct = (total_dropped / max(1, total_ingested)) * 100.0

    result = {
        "model": model_name,
        "streams": n_streams,
        "mode": pipeline_mode,
        "skip_frames": skip_frames,
        "display": display,
        "paced": is_paced,
        "target_fps": target_fps,
        "total_fps": round(total_fps, 2),
        "fps_per_camera": round(fps_per_camera, 2),
        "frames_processed": total_processed,
        "frames_dropped": total_dropped,
        "drop_rate_pct": round(drop_rate_pct, 2),
        "elapsed_sec": round(t_elapsed, 2),
        # Hardware Metrics
        "avg_gpu_util_pct": round(hw_metrics["avg_gpu_util_pct"], 1),
        "peak_gpu_util_pct": round(hw_metrics["peak_gpu_util_pct"], 1),
        "avg_gpu_mem_bus_pct": round(hw_metrics["avg_gpu_mem_bus_pct"], 1),
        "avg_vram_mb": round(hw_metrics["avg_vram_mb"], 1),
        "peak_vram_mb": round(hw_metrics["peak_vram_mb"], 1),
        "delta_vram_mb": round(hw_metrics["delta_vram_mb"], 1),
        "avg_gpu_temp_c": round(hw_metrics["avg_gpu_temp_c"], 1),
        "peak_gpu_temp_c": round(hw_metrics["peak_gpu_temp_c"], 1),
        "avg_gpu_power_w": round(hw_metrics["avg_gpu_power_w"], 1),
        "peak_gpu_power_w": round(hw_metrics["peak_gpu_power_w"], 1),
        "avg_process_cpu_pct": round(hw_metrics["avg_process_cpu_pct"], 1),
        "avg_system_cpu_pct": round(hw_metrics["avg_system_cpu_pct"], 1),
        "peak_system_cpu_pct": round(hw_metrics["peak_system_cpu_pct"], 1),
        "avg_process_ram_mb": round(hw_metrics["avg_process_ram_mb"], 1),
        "peak_process_ram_mb": round(hw_metrics["peak_process_ram_mb"], 1),
        "delta_process_ram_mb": round(hw_metrics["delta_process_ram_mb"], 1),
        "peak_system_ram_gb": round(hw_metrics["peak_system_ram_gb"], 2),
        # Latencies
        "latencies": latencies_summary,
        "time_series": hw_metrics.get("time_series", {}),
    }

    # Print summary
    print(
        f"   ✔ Completed: {fps_per_camera:.1f} FPS/cam (Total: {total_fps:.1f} FPS) | "
        f"GPU: {hw_metrics['avg_gpu_util_pct']:.1f}% | VRAM: {hw_metrics['peak_vram_mb']:.0f} MB | "
        f"CPU: {hw_metrics['avg_system_cpu_pct']:.1f}% | Latency (E2E P50): {latencies_summary.get('e2e_ms', {}).get('p50', 0.0):.1f} ms | "
        f"Drops: {drop_rate_pct:.1f}%"
    )

    return result


# ==============================================================================
# 7. FEASIBILITY & 8-CAMERA HARDWARE SIZING ANALYZER
# ==============================================================================
class FeasibilityAnalyzer:
    """
    Evaluates real-time feasibility for 8 cameras, determines system bottlenecks,
    calculates maximum supportable streams, and generates hardware procurement tiers.
    """

    def __init__(self, benchmark_results: List[Dict[str, Any]], target_cameras: int = 8, target_fps: float = 15.0):
        self.results = benchmark_results
        self.target_cameras = target_cameras
        self.target_fps = target_fps
        self.target_total_fps = target_cameras * target_fps

    def evaluate_8cam_feasibility(self) -> Dict[str, Any]:
        eight_cam_runs = [r for r in self.results if r["streams"] == self.target_cameras]
        if not eight_cam_runs:
            eight_cam_runs = sorted(self.results, key=lambda x: x["streams"], reverse=True)

        best_run = eight_cam_runs[0]
        actual_fps_per_cam = best_run["fps_per_camera"]
        actual_total_fps = best_run["total_fps"]
        drop_rate = best_run.get("drop_rate_pct", 0.0)
        gpu_util = best_run["avg_gpu_util_pct"]
        vram_mb = best_run["peak_vram_mb"]
        cpu_util = best_run["avg_system_cpu_pct"]
        e2e_latency_ms = best_run.get("latencies", {}).get("e2e_ms", {}).get("p50", 0.0)

        # Headroom calculations
        total_vram_mb = 8192.0
        if HAS_NVML and torch.cuda.is_available():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                total_vram_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
            except Exception:
                pass

        gpu_headroom_pct = max(0.0, 100.0 - gpu_util)
        cpu_headroom_pct = max(0.0, 100.0 - cpu_util)
        vram_headroom_mb = max(0.0, total_vram_mb - vram_mb)
        frame_budget_ms = 1000.0 / self.target_fps

        meets_fps = actual_fps_per_cam >= (self.target_fps * 0.90)
        drops_acceptable = drop_rate < 5.0
        gpu_safe = gpu_util < 90.0
        cpu_safe = cpu_util < 90.0

        if meets_fps and drops_acceptable and gpu_safe and cpu_safe:
            if gpu_util < 65.0 and cpu_util < 65.0 and drop_rate < 1.0:
                verdict = "EXCELLENT (Production Ready with High Headroom)"
                grade = "A+"
            else:
                verdict = "GOOD (Production Ready)"
                grade = "A"
        elif meets_fps and (gpu_util >= 90.0 or cpu_util >= 90.0 or drop_rate >= 5.0):
            verdict = "BORDERLINE (Meets target FPS but near hardware limits; frame drops or thermal throttling likely under peak traffic)"
            grade = "B-"
        else:
            verdict = "INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)"
            grade = "C / FAIL"

        bottlenecks = []
        if gpu_util >= 85.0:
            bottlenecks.append(f"GPU Compute Core Saturation ({gpu_util:.1f}% load)")
        if vram_headroom_mb < 1500.0:
            bottlenecks.append(f"VRAM Capacity Limit ({vram_mb:.0f} MB / {total_vram_mb:.0f} MB used)")
        if cpu_util >= 80.0:
            bottlenecks.append(f"Host CPU Multithreading / Video Decoding Backpressure ({cpu_util:.1f}% load)")
        if best_run.get("avg_gpu_mem_bus_pct", 0.0) >= 70.0:
            bottlenecks.append(f"GPU PCIe / Memory Controller Bus Bandwidth Limit ({best_run['avg_gpu_mem_bus_pct']:.1f}%)")
        if not bottlenecks:
            bottlenecks.append("None (Hardware operates with healthy margin across all subsystems)")

        cost_per_stream_fps = actual_total_fps / max(1, best_run["streams"])
        cost_per_stream_gpu = max(5.0, gpu_util / max(1, best_run["streams"]))
        cost_per_stream_cpu = max(4.0, cpu_util / max(1, best_run["streams"]))
        cost_per_stream_vram = max(150.0, (vram_mb - 800.0) / max(1, best_run["streams"]))

        max_streams_gpu = int(85.0 / cost_per_stream_gpu) if cost_per_stream_gpu > 0 else 8
        max_streams_cpu = int(85.0 / cost_per_stream_cpu) if cost_per_stream_cpu > 0 else 8
        max_streams_vram = int((total_vram_mb - 1500.0) / cost_per_stream_vram) if cost_per_stream_vram > 0 else 8
        max_safe_streams = max(1, min(max_streams_gpu, max_streams_cpu, max_streams_vram))

        return {
            "verdict": verdict,
            "grade": grade,
            "target_cameras": self.target_cameras,
            "target_fps": self.target_fps,
            "target_total_fps": self.target_total_fps,
            "actual_fps_per_cam": actual_fps_per_cam,
            "actual_total_fps": actual_total_fps,
            "drop_rate_pct": drop_rate,
            "gpu_util_pct": gpu_util,
            "gpu_headroom_pct": gpu_headroom_pct,
            "vram_used_mb": vram_mb,
            "total_vram_mb": total_vram_mb,
            "vram_headroom_mb": vram_headroom_mb,
            "cpu_util_pct": cpu_util,
            "cpu_headroom_pct": cpu_headroom_pct,
            "e2e_latency_ms": e2e_latency_ms,
            "frame_budget_ms": frame_budget_ms,
            "bottlenecks": bottlenecks,
            "max_safe_streams_at_target_fps": max_safe_streams,
        }


# ==============================================================================
# 8. VISUALIZATION & MULTI-FORMAT REPORT GENERATOR
# ==============================================================================
class ReportGenerator:
    @staticmethod
    def print_terminal_report(
        benchmark_results: List[Dict[str, Any]],
        feasibility: Dict[str, Any],
        system_info: Dict[str, Any],
    ):
        print("\n" + "=" * 95)
        print("🚦 SMART TRAFFIC VISION - MULTI-CAMERA BENCHMARK & SIZING REPORT")
        print("=" * 95)
        print(f" • Host Hardware  : {system_info.get('gpu_name', 'N/A')}")
        print(
            f" • CPU Architecture: {system_info.get('cpu_physical', 0)} Physical Cores / {system_info.get('cpu_logical', 0)} Logical Threads"
        )
        print(
            f" • Memory Capacity : {system_info.get('system_ram_gb', 0.0):.1f} GB System RAM | {system_info.get('vram_total_gb', 0.0):.1f} GB Dedicated VRAM"
        )
        print(f" • PyTorch / CUDA  : PyTorch {torch.__version__} (CUDA: {torch.version.cuda})")
        print("=" * 95)

        # Results Table
        print("\n### 1. Multi-Camera Scalability & Resource Demand Matrix:")
        print("-" * 95)
        header = (
            f"{'Model':<10} | {'Streams':<7} | {'Mode':<8} | {'Skip':<4} | {'Display':<7} | "
            f"{'FPS/Cam':<9} | {'Total FPS':<9} | {'GPU %':<6} | {'VRAM(MB)':<8} | {'CPU %':<6} | {'Drops'}"
        )
        print(header)
        print("-" * 95)

        for r in benchmark_results:
            disp_str = "YES" if r.get("display") else "NO"
            row = (
                f"{r['model']:<10} | {r['streams']:<7} | {r['mode']:<8} | {r['skip_frames']:<4} | {disp_str:<7} | "
                f"{r['fps_per_camera']:<9.1f} | {r['total_fps']:<9.1f} | {r['avg_gpu_util_pct']:<6.1f} | "
                f"{r['peak_vram_mb']:<8.0f} | {r['avg_system_cpu_pct']:<6.1f} | {r.get('drop_rate_pct', 0.0):.1f}%"
            )
            print(row)
        print("-" * 95)

        # Stage Latency Breakdown Table
        print("\n### 2. Stage-by-Stage Latency Breakdown (Averaged Across Streams):")
        print("-" * 95)
        print(f"{'Model / Streams':<23} | {'Decode':<8} | {'Inference':<10} | {'SORT':<8} | {'Render':<8} | {'E2E P50 (ms)'}")
        print("-" * 95)
        for r in benchmark_results:
            l = r.get("latencies", {})
            dec = l.get("decode_ms", {}).get("mean", 0.0)
            inf = l.get("inference_ms", {}).get("mean", 0.0)
            trk = l.get("tracking_ms", {}).get("mean", 0.0)
            vis = l.get("visualize_ms", {}).get("mean", 0.0)
            e2e = l.get("e2e_ms", {}).get("p50", 0.0)
            name_str = f"{r['model']} ({r['streams']} cams)"
            print(f"{name_str:<23} | {dec:<8.2f} | {inf:<10.2f} | {trk:<8.2f} | {vis:<8.2f} | {e2e:<.2f}")
        print("-" * 95)

        # 8-Camera Sizing Verdict
        print("\n" + "=" * 95)
        print(
            f"🎯 8-CAMERA HARDWARE SIZING VERDICT (Target: {feasibility['target_cameras']} Cams @ {feasibility['target_fps']} FPS = {feasibility['target_total_fps']:.0f} Total FPS)"
        )
        print("=" * 95)
        print(f" [★] VERDICT GRADE       : [{feasibility['grade']}] - {feasibility['verdict']}")
        print(
            f" [★] Achieved Throughput : {feasibility['actual_fps_per_cam']:.1f} FPS/cam (Total: {feasibility['actual_total_fps']:.1f} FPS)"
        )
        print(
            f" [★] GPU Load & Headroom : {feasibility['gpu_util_pct']:.1f}% Load ({feasibility['gpu_headroom_pct']:.1f}% Compute Headroom remaining)"
        )
        print(
            f" [★] VRAM Load & Headroom: {feasibility['vram_used_mb']:.0f} MB / {feasibility['total_vram_mb']:.0f} MB ({feasibility['vram_headroom_mb']/1024.0:.2f} GB Free)"
        )
        print(
            f" [★] CPU Load & Headroom : {feasibility['cpu_util_pct']:.1f}% Load ({feasibility['cpu_headroom_pct']:.1f}% CPU Headroom remaining)"
        )
        print(
            f" [★] Frame Drop Rate     : {feasibility['drop_rate_pct']:.2f}% (Threshold: < 5.0% for real-time safety)"
        )
        print(
            f" [★] Max Safe Streams    : {feasibility['max_safe_streams_at_target_fps']} Cameras simultaneously @ {feasibility['target_fps']} FPS"
        )
        print(f" [★] Primary Bottlenecks : {', '.join(feasibility['bottlenecks'])}")
        print("=" * 95)

        # Procurement Specifications
        print("""
### 3. Recommended Deployment & Procurement Specs:
-----------------------------------------------------------------------------------------------
[TIER 1: OPTIMAL EDGE BOX / WORKSTATION (Target: 8 Cameras @ 15-25 FPS)]
  • GPU     : NVIDIA GeForce RTX 4060 8GB / RTX 5060 Laptop/Desktop (TensorRT FP16 enabled)
  • CPU     : Intel Core i5-13400 / 14400 (10 Cores, 16 Threads) or AMD Ryzen 5 7600
  • RAM     : 16 GB - 32 GB DDR4/DDR5 (Dual Channel)
  • Storage : 500 GB - 1 TB NVMe M.2 SSD (for OS, PostgreSQL incident DB, and video logs)
  • Network : Dual Gigabit / 2.5GbE LAN (handles 8x 1080p RTSP feeds @ ~32 Mbps total)

[TIER 2: INDUSTRIAL EDGE APPLIANCE (Outdoor Traffic Cabinet / Fanless)]
  • Platform: NVIDIA Jetson Orin Nano 8GB (40 TOPS) or Orin NX 16GB (100 TOPS)
  • Pipeline: TensorRT Engine + DeepStream / Batched Inference + Frame Skip=1
  • Power   : 15W - 25W low-power industrial DC power input

[TIER 3: ENTERPRISE INTERSECTION SERVER (Target: 16+ Cameras or Multi-Model YOLOv8m)]
  • GPU     : NVIDIA RTX 4070 Ti Super 16GB / RTX 4080 / RTX A4000
  • CPU     : Intel Core i7-14700 / AMD Ryzen 7 7700X (32 GB DDR5 RAM)
-----------------------------------------------------------------------------------------------
""")

    @staticmethod
    def export_json(data: Dict[str, Any], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"   [+] Saved structured benchmark JSON to: {filepath}")

    @staticmethod
    def export_csv(results: List[Dict[str, Any]], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not results:
            return

        keys = [
            "model",
            "streams",
            "mode",
            "skip_frames",
            "display",
            "paced",
            "target_fps",
            "total_fps",
            "fps_per_camera",
            "drop_rate_pct",
            "avg_gpu_util_pct",
            "peak_gpu_util_pct",
            "avg_gpu_mem_bus_pct",
            "avg_vram_mb",
            "peak_vram_mb",
            "avg_system_cpu_pct",
            "peak_system_cpu_pct",
            "avg_process_ram_mb",
            "peak_process_ram_mb",
            "avg_gpu_temp_c",
            "avg_gpu_power_w",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"   [+] Saved benchmark summary CSV to: {filepath}")

    @staticmethod
    def export_markdown_report(
        results: List[Dict[str, Any]],
        feasibility: Dict[str, Any],
        system_info: Dict[str, Any],
        filepath: str,
    ):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        md = []
        md.append("# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report\n")
        md.append(f"**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("## 1. System Hardware Baseline\n")
        md.append(f"- **GPU Model:** {system_info.get('gpu_name', 'N/A')}")
        md.append(
            f"- **CPU:** {system_info.get('cpu_physical', 0)} Physical Cores / {system_info.get('cpu_logical', 0)} Logical Threads"
        )
        md.append(f"- **System RAM:** {system_info.get('system_ram_gb', 0.0):.1f} GB")
        md.append(f"- **Dedicated VRAM:** {system_info.get('vram_total_gb', 0.0):.1f} GB")
        md.append(f"- **PyTorch / CUDA:** {torch.__version__} (CUDA: {torch.version.cuda})\n")

        md.append("## 2. 8-Camera Sizing Feasibility Verdict\n")
        md.append(f"> **Verdict:** **[{feasibility['grade']}] - {feasibility['verdict']}**\n")
        md.append(
            f"- **Target Throughput:** {feasibility['target_cameras']} Cameras @ {feasibility['target_fps']} FPS = **{feasibility['target_total_fps']:.0f} Total FPS**"
        )
        md.append(
            f"- **Achieved Throughput:** **{feasibility['actual_fps_per_cam']:.1f} FPS/cam** (Total: **{feasibility['actual_total_fps']:.1f} FPS**)"
        )
        md.append(
            f"- **GPU Utilization & Headroom:** **{feasibility['gpu_util_pct']:.1f}%** load (Headroom: **{feasibility['gpu_headroom_pct']:.1f}%**)"
        )
        md.append(
            f"- **VRAM Footprint & Free Space:** **{feasibility['vram_used_mb']:.0f} MB** / {feasibility['total_vram_mb']:.0f} MB (Headroom: **{feasibility['vram_headroom_mb']/1024.0:.2f} GB**)"
        )
        md.append(
            f"- **CPU Utilization & Headroom:** **{feasibility['cpu_util_pct']:.1f}%** load (Headroom: **{feasibility['cpu_headroom_pct']:.1f}%**)"
        )
        md.append(f"- **Frame Drop Rate:** **{feasibility['drop_rate_pct']:.2f}%**")
        md.append(
            f"- **Max Safe Stream Capacity:** **{feasibility['max_safe_streams_at_target_fps']} Camera Streams** @ {feasibility['target_fps']} FPS"
        )
        md.append(f"- **Identified Bottlenecks:** {', '.join(feasibility['bottlenecks'])}\n")

        md.append("## 3. Detailed Results Matrix\n")
        md.append(
            "| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |"
        )
        md.append(
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        )
        for r in results:
            disp_str = "YES" if r.get("display") else "NO"
            md.append(
                f"| `{r['model']}` | {r['streams']} | {r['mode']} | {r['skip_frames']} | {disp_str} | "
                f"**{r['fps_per_camera']:.1f}** | **{r['total_fps']:.1f}** | {r['avg_gpu_util_pct']:.1f}% | "
                f"{r['peak_vram_mb']:.0f} | {r['avg_system_cpu_pct']:.1f}% | {r['peak_process_ram_mb']:.0f} | {r.get('drop_rate_pct', 0.0):.1f}% |"
            )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"   [+] Saved Executive Markdown Report to: {filepath}")

    @staticmethod
    def generate_plots(results: List[Dict[str, Any]], filepath: str):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"   [!] Skipping plot generation (matplotlib error: {e})")
            return

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        fig.suptitle("Smart Traffic Vision - Multi-Camera Hardware Performance Analysis", fontsize=14, fontweight="bold")

        # Sort results by streams
        sorted_res = sorted(results, key=lambda x: (x["model"], x["streams"]))
        models = list(set(r["model"] for r in sorted_res))

        # Panel 1: Throughput Scaling (Streams vs FPS/camera)
        ax1 = axs[0, 0]
        for m in models:
            m_res = [r for r in sorted_res if r["model"] == m]
            streams = [r["streams"] for r in m_res]
            fps_cam = [r["fps_per_camera"] for r in m_res]
            ax1.plot(streams, fps_cam, marker="o", linewidth=2.5, label=f"{m} (FPS/cam)")
        ax1.axhline(15.0, color="green", linestyle="--", alpha=0.7, label="15 FPS Target")
        ax1.axhline(25.0, color="orange", linestyle="--", alpha=0.7, label="25 FPS Target")
        ax1.set_title("Per-Camera Throughput vs Stream Count", fontweight="bold")
        ax1.set_xlabel("Number of Parallel Cameras")
        ax1.set_ylabel("FPS per Camera")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # Panel 2: Resource Utilization vs Stream Count
        ax2 = axs[0, 1]
        for m in models:
            m_res = [r for r in sorted_res if r["model"] == m]
            streams = [r["streams"] for r in m_res]
            gpu_u = [r["avg_gpu_util_pct"] for r in m_res]
            cpu_u = [r["avg_system_cpu_pct"] for r in m_res]
            ax2.plot(streams, gpu_u, marker="s", linewidth=2, label=f"{m} GPU %")
            ax2.plot(streams, cpu_u, marker="^", linewidth=2, linestyle="--", label=f"{m} CPU %")
        ax2.set_title("GPU & CPU Load vs Stream Count", fontweight="bold")
        ax2.set_xlabel("Number of Parallel Cameras")
        ax2.set_ylabel("Utilization (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        # Panel 3: Memory Footprint (VRAM & RAM MB)
        ax3 = axs[1, 0]
        for m in models:
            m_res = [r for r in sorted_res if r["model"] == m]
            streams = [r["streams"] for r in m_res]
            vram = [r["peak_vram_mb"] for r in m_res]
            ram = [r["peak_process_ram_mb"] for r in m_res]
            ax3.plot(streams, vram, marker="D", linewidth=2, label=f"{m} VRAM (MB)")
            ax3.plot(streams, ram, marker="v", linewidth=2, linestyle=":", label=f"{m} Host RAM (MB)")
        ax3.set_title("Memory Demand vs Stream Count", fontweight="bold")
        ax3.set_xlabel("Number of Parallel Cameras")
        ax3.set_ylabel("Memory (MB)")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend()

        # Panel 4: Latency Stage Breakdown (for largest run)
        ax4 = axs[1, 1]
        largest_run = sorted_res[-1]
        l = largest_run.get("latencies", {})
        stages = ["Decode", "Inference", "SORT", "Render"]
        times = [
            l.get("decode_ms", {}).get("mean", 0.0),
            l.get("inference_ms", {}).get("mean", 0.0),
            l.get("tracking_ms", {}).get("mean", 0.0),
            l.get("visualize_ms", {}).get("mean", 0.0),
        ]
        colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
        bars = ax4.bar(stages, times, color=colors, edgecolor="black", alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.1,
                f"{yval:.2f} ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax4.set_title(
            f"Stage Latency Breakdown ({largest_run['model']} - {largest_run['streams']} Cams)", fontweight="bold"
        )
        ax4.set_ylabel("Mean Execution Time (ms)")
        ax4.grid(True, linestyle=":", alpha=0.6, axis="y")

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   [+] Saved Performance Charts PNG to: {filepath}")

    @staticmethod
    def cleanup_old_results(out_dir: str, keep_latest: int = 3):
        """
        Prunes older benchmark result files, retaining at most `keep_latest` historical runs.
        """
        if not os.path.exists(out_dir) or keep_latest <= 0:
            return

        import glob

        patterns = [
            "BENCHMARK_REPORT_*.md",
            "benchmark_summary_*.csv",
            "benchmark_results_*.json",
            "benchmark_analysis_*.png",
        ]

        deleted_count = 0
        for pattern in patterns:
            files = glob.glob(os.path.join(out_dir, pattern))
            # Sort by modification time descending (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            for old_file in files[keep_latest:]:
                try:
                    os.remove(old_file)
                    deleted_count += 1
                except Exception:
                    pass

        if deleted_count > 0:
            print(f"   [~] Cleaned up {deleted_count} older benchmark artifact file(s) (retaining latest {keep_latest} runs).")


# ==============================================================================
# 9. MAIN CLI INTERFACE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Camera Hardware Benchmark & Sizing Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="List of parallel camera stream counts to test (e.g. 1 2 4 8)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov8s.pt"],
        help="YOLO model checkpoint files to benchmark",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds per individual benchmark run",
    )
    parser.add_argument(
        "--target-cams",
        type=int,
        default=8,
        help="Target number of camera feeds for feasibility sizing",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target frame rate per camera feed (e.g. 15.0 or 25.0 FPS)",
    )
    parser.add_argument(
        "--mode",
        choices=["threaded", "batched", "both"],
        default="threaded",
        help="Vision pipeline architecture mode",
    )
    parser.add_argument(
        "--frame-skips",
        nargs="+",
        type=int,
        default=[0],
        help="Frame skipping intervals to evaluate (0 = every frame, 1 = every 2nd frame, 2 = every 3rd)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open live OpenCV multi-camera visual HUD window with bounding boxes & polygons",
    )
    parser.add_argument(
        "--unpaced",
        action="store_true",
        help="Disable RTSP stream pacing to run in uncapped max-throughput stress mode",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (e.g. 640, 480, 1280)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEOS,
        help="Custom list of video file paths or RTSP URLs to stream",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/hardware-results",
        help="Output directory to store JSON, CSV, Markdown, and chart reports",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=3,
        help="Maximum number of historical benchmark result runs to retain in results folder",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save graphical performance charts (PNG)",
    )

    args = parser.parse_args()

    # Hardware Info Detection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"
    cpu_logical = psutil.cpu_count(logical=True) or 1
    cpu_physical = psutil.cpu_count(logical=False) or 1
    sys_ram_gb = psutil.virtual_memory().total / (1024**3)

    vram_total_gb = 0.0
    if HAS_NVML and torch.cuda.is_available():
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            vram_total_gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024**3)
        except Exception:
            pass

    system_info = {
        "gpu_name": gpu_name,
        "device": device,
        "cpu_physical": cpu_physical,
        "cpu_logical": cpu_logical,
        "system_ram_gb": sys_ram_gb,
        "vram_total_gb": vram_total_gb,
    }

    print("\n" + "=" * 80)
    print("🚦 INITIALIZING SMART TRAFFIC VISION BENCHMARK SUITE")
    print(f" Target Compute Device : {device} ({gpu_name})")
    print(f" CPU Subsystem          : {cpu_physical} Physical Cores / {cpu_logical} Logical Threads")
    print(f" Host & GPU Memory      : {sys_ram_gb:.1f} GB System RAM | {vram_total_gb:.1f} GB Dedicated VRAM")
    print(f" Test Models            : {args.models}")
    print(f" Stream Targets         : {args.streams} cameras")
    print(f" Frame Skip Intervals   : {args.frame_skips}")
    print(f" Live Display Window    : {'ENABLED (Visual HUD Active)' if args.display else 'DISABLED (Headless Server Mode)'}")
    print(f" RTSP Stream Pacing     : {'DISABLED (Uncapped Stress)' if args.unpaced else f'ENABLED ({args.target_fps} FPS)'}")
    print("=" * 80)

    # Initialize Profiler & calibrate baseline
    profiler = HardwareProfiler(sample_interval=0.05)
    baseline = profiler.capture_baseline(duration=1.5)
    print(
        f"   ✔ Baseline Idle State: CPU: {baseline['baseline_cpu_pct']:.1f}% | "
        f"GPU: {baseline['baseline_gpu_pct']:.1f}% | VRAM: {baseline['baseline_vram_mb']:.0f} MB | "
        f"Power: {baseline['baseline_power_w']:.1f} W"
    )

    modes_to_test = ["threaded", "batched"] if args.mode == "both" else [args.mode]
    all_results: List[Dict[str, Any]] = []

    # Run Benchmark Experiments
    for model_name in args.models:
        for mode in modes_to_test:
            for skip in args.frame_skips:
                for n_streams in args.streams:
                    res = run_single_test(
                        model_name=model_name,
                        n_streams=n_streams,
                        video_sources=args.videos,
                        pipeline_mode=mode,
                        duration_sec=args.duration,
                        skip_frames=skip,
                        target_fps=args.target_fps,
                        is_paced=not args.unpaced,
                        display=args.display,
                        imgsz=args.imgsz,
                        device=device,
                        profiler=profiler,
                    )
                    all_results.append(res)
                    time.sleep(1.0)  # Brief thermal cooldown between tests

    # Analyze 8-Camera Feasibility
    analyzer = FeasibilityAnalyzer(
        benchmark_results=all_results,
        target_cameras=args.target_cams,
        target_fps=args.target_fps,
    )
    feasibility = analyzer.evaluate_8cam_feasibility()

    # Terminal Report
    ReportGenerator.print_terminal_report(
        benchmark_results=all_results,
        feasibility=feasibility,
        system_info=system_info,
    )

    # Export Artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"benchmark_results_{timestamp}.json")
    csv_path = os.path.join(args.out_dir, f"benchmark_summary_{timestamp}.csv")
    md_path = os.path.join(args.out_dir, f"BENCHMARK_REPORT_{timestamp}.md")
    png_path = os.path.join(args.out_dir, f"benchmark_analysis_{timestamp}.png")

    full_export = {
        "system_info": system_info,
        "baseline": baseline,
        "feasibility_8cam": feasibility,
        "benchmark_runs": all_results,
    }

    ReportGenerator.export_json(full_export, json_path)
    ReportGenerator.export_csv(all_results, csv_path)
    ReportGenerator.export_markdown_report(all_results, feasibility, system_info, md_path)

    if args.save_plots:
        ReportGenerator.generate_plots(all_results, png_path)

    # Cleanup older benchmark result sets beyond keep_latest
    ReportGenerator.cleanup_old_results(args.out_dir, keep_latest=args.keep_latest)

    print("\n[+] Benchmark Suite Execution Completed Successfully!")


if __name__ == "__main__":
    main()
