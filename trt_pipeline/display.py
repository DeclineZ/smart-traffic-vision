"""
Asynchronous Multi-Camera Display & Video Streaming Engine for Smart Traffic Vision.
Decouples visual HUD rendering, grid stitching, and optional hardware NVENC encoding
into a dedicated background thread to prevent UI bottlenecks in the GPU inference pipeline.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
from shapely.geometry import Polygon

from .tools import get_logger

logger = get_logger("AsyncDisplayWorker")

COCO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def is_nvenc_available() -> bool:
    """Probes whether NVIDIA NVENC hardware encoder is operational on the host."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=0.1:size=640x360:rate=30",
        "-c:v", "h264_nvenc",
        "-f", "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
        return res.returncode == 0
    except Exception:
        return False


class NVENCVideoWriter:
    """
    Hardware-accelerated H.264 video encoder utilizing NVIDIA NVENC silicon.
    Spawns an FFmpeg pipe to encode frames with near-zero CPU load.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: float = 25.0, bitrate: str = "4M"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.process: Optional[subprocess.Popen] = None
        self._init_ffmpeg()

    def _init_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-tune", "ll",  # Low latency
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
            self.output_path,
        ]
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            logger.info(f"NVENC Video Writer initialized -> {self.output_path} ({self.width}x{self.height} @ {self.fps} FPS)")
        except Exception as e:
            logger.warning(f"Failed to start NVENC Video Writer: {e}")
            self.process = None

    def write(self, frame: np.ndarray) -> None:
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                self.close()

    def close(self) -> None:
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()
            self.process = None


def stitch_camera_grid(frames: List[np.ndarray], tile_size: Tuple[int, int] = (480, 270)) -> np.ndarray:
    """
    Dynamically tiles N camera preview frames into an adaptive multi-view grid.
    Supports 1, 2, 4, 6, 8, or arbitrary N streams with uniform tile resolution.
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


class AsyncDisplayWorker:
    """
    Decoupled visual rendering & HUD display worker.
    Runs asynchronously in a dedicated background thread to prevent GUI rendering,
    OpenCV resizing, grid stitching, and NVENC encoding from throttling
    the high-throughput GPU inference loop.
    """

    def __init__(
        self,
        display: bool = True,
        nvenc_writer: Optional[NVENCVideoWriter] = None,
        window_name: str = "Smart Traffic Vision - Multi-Camera Production Grid",
        tile_size: Tuple[int, int] = (480, 270),
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.display = display
        self.nvenc_writer = nvenc_writer
        self.window_name = window_name
        self.tile_size = tile_size
        self.class_names = class_names

        # Queue size = 1 ensures we always display the freshest frame, dropping preview if UI is slow
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_grid: Optional[np.ndarray] = None
        self._grid_lock = threading.Lock()

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._render_loop, name="AsyncDisplayWorker", daemon=True)
        self.thread.start()
        logger.info("AsyncDisplayWorker started.")

    def get_latest_grid(self) -> Optional[np.ndarray]:
        """Thread-safe retrieval of the most recently stitched preview grid."""
        with self._grid_lock:
            return self.latest_grid

    def poll_window(self) -> bool:
        """
        Renders the pre-stitched multi-camera grid to the OpenCV GUI window.
        Must be invoked from the main thread to ensure full cross-platform compatibility
        (especially macOS Cocoa AppKit requirements).
        Returns False if the user pressed 'q', True otherwise.
        """
        if not self.display:
            return True

        grid = self.get_latest_grid()
        if grid is not None:
            cv.imshow(self.window_name, grid)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                return False
        return True

    def submit(
        self,
        frames: List[np.ndarray],
        tracked_list: List[np.ndarray],
        cam_names: List[str],
        lane_configs: Optional[List[Dict[str, Any]]] = None,
        header_stats: Optional[str] = None,
    ) -> None:
        """
        Non-blocking snapshot submit. If rendering is busy, drop preview frame
        so inference is NEVER stalled.
        """
        if not self.running:
            return

        payload = (frames, tracked_list, cam_names, lane_configs, header_stats)
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.nvenc_writer:
            self.nvenc_writer.close()
        if self.display:
            try:
                cv.destroyAllWindows()
            except Exception:
                pass
        logger.info("AsyncDisplayWorker stopped.")

    def _render_loop(self) -> None:
        while self.running:
            try:
                payload = self.queue.get(timeout=0.04)
            except queue.Empty:
                time.sleep(0.01)
                continue

            frames, tracked_list, cam_names, lane_configs, header_stats = payload
            num_streams = len(frames)
            vis_frames = []

            for idx in range(num_streams):
                f = frames[idx]
                if f is None or f.size == 0:
                    continue

                vis = cv.resize(f, self.tile_size)
                scale_x = self.tile_size[0] / f.shape[1]
                scale_y = self.tile_size[1] / f.shape[0]

                # 1. Draw Lane Polygons if configured
                if lane_configs and idx < len(lane_configs):
                    for lane_id, lcfg in lane_configs[idx].items():
                        poly = lcfg.get("polygon")
                        if poly is not None and isinstance(poly, Polygon):
                            pts = np.array(
                                [[int(x * scale_x), int(y * scale_y)] for x, y in poly.exterior.coords],
                                dtype=np.int32,
                            )
                            cv.polylines(vis, [pts], True, (0, 255, 120), 1)
                            first_pt = (pts[0][0], max(15, pts[0][1]))
                            cv.putText(vis, lane_id, first_pt, cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)

                # 2. Draw Tracked Bounding Boxes & IDs
                if idx < len(tracked_list) and len(tracked_list[idx]) > 0:
                    for obj in tracked_list[idx]:
                        ox1, oy1, ox2, oy2, track_id = obj[:5]
                        bx1, by1 = int(ox1 * scale_x), int(oy1 * scale_y)
                        bx2, by2 = int(ox2 * scale_x), int(oy2 * scale_y)

                        cls_name = "car"
                        if len(obj) >= 6:
                            cls_id = int(obj[5])
                            if self.class_names and cls_id in self.class_names:
                                cls_name = self.class_names[cls_id]
                            else:
                                cls_name = COCO_CLASSES.get(cls_id, "car")

                        # Dynamic vehicle category colors
                        if cls_name in ("motorcycle", "bicycle"):
                            color = (0, 255, 255)  # Yellow for 2-wheelers
                        elif cls_name in ("three_wheeler", "tuktuk"):
                            color = (255, 165, 0)  # Orange for 3-wheelers / Tuk-tuks
                        elif cls_name in ("bus", "truck"):
                            color = (0, 165, 255)  # Amber for heavy vehicles
                        else:
                            color = (0, 220, 100)  # Vibrant Green for cars

                        cv.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
                        label = f"#{int(track_id)} {cls_name}"
                        cv.putText(vis, label, (bx1, max(12, by1 - 3)), cv.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv.LINE_AA)

                # 3. Overlay Camera Identifier
                c_name = cam_names[idx] if idx < len(cam_names) else f"CAM_{idx+1:02d}"
                cv.putText(vis, c_name.upper(), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv.LINE_AA)
                vis_frames.append(vis)

            if not vis_frames:
                continue

            # Stitch into Multi-View Grid
            grid = stitch_camera_grid(vis_frames, tile_size=self.tile_size)

            # Optional Header Banner
            if header_stats:
                banner_h = 28
                banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
                cv.putText(banner, header_stats, (12, 19), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv.LINE_AA)
                grid = np.vstack([banner, grid])

            with self._grid_lock:
                self.latest_grid = grid

            # Hardware NVENC write
            if self.nvenc_writer:
                self.nvenc_writer.write(grid)
