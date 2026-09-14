"""
High-Throughput Stream Ingestion & Ring Buffer Worker for Smart Traffic Vision.
Decouples camera I/O from GPU inference using a bounded ring buffer (queue_size=2)
to absorb OS thread jitter and prevent frame lag.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from .tools import get_logger

logger = get_logger("StreamBufferWorker")


class StreamBufferWorker:
    """
    Decoupled background stream ingestion worker.
    Supports RTSP IP camera streams, local video files, and video looping.
    Implements a jitter-absorbing ring buffer (queue_size=2) that discards stale
    frames upon buffer overflow to maintain real-time low-latency synchronization.
    """

    def __init__(
        self,
        name: str,
        source: str | int,
        target_fps: float = 25.0,
        buffer_size: int = 2,
        is_paced: bool = True,
        loop_video: bool = True,
    ):
        self.name = name
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = max(1, buffer_size)
        self.is_paced = is_paced
        self.loop_video = loop_video

        self.queue: queue.Queue[Tuple[float, np.ndarray]] = queue.Queue(maxsize=self.buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv.VideoCapture] = None

        self.frames_ingested = 0
        self.frames_dropped = 0
        self.source_fps = 25.0
        self.frame_shape: Optional[Tuple[int, int]] = None

    def start(self) -> None:
        """Starts the capture background thread."""
        self.cap = cv.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"[{self.name}] Failed to open video source: {self.source}")
            raise RuntimeError(f"[{self.name}] Unable to open video source: {self.source}")

        # Set low internal buffer on OpenCV backend if supported (Linux / RTSP)
        try:
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fps = self.cap.get(cv.CAP_PROP_FPS)
        if fps and fps > 1.0:
            self.source_fps = fps

        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self.frame_shape = (h, w)

        self.running = True
        self.thread = threading.Thread(target=self._ingest_loop, name=f"Ingest-{self.name}", daemon=True)
        self.thread.start()
        logger.info(f"[{self.name}] Ingestion started: source='{self.source}', target_fps={self.target_fps:.1f}, buffer_size={self.buffer_size}")

    def stop(self) -> None:
        """Stops the ingestion thread and releases video capture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info(f"[{self.name}] Ingestion stopped. (Total: {self.frames_ingested}, Dropped: {self.frames_dropped})")

    def _ingest_loop(self) -> None:
        effective_fps = self.target_fps if self.target_fps > 0 else self.source_fps
        frame_interval = 1.0 / effective_fps if effective_fps > 0 else 0.04
        next_frame_time = time.perf_counter()

        while self.running and self.cap:
            now = time.perf_counter()
            if self.is_paced and now < next_frame_time:
                time.sleep(max(0.0005, next_frame_time - now))

            ret, frame = self.cap.read()
            if not ret:
                if self.loop_video and isinstance(self.source, str) and os.path.exists(self.source):
                    self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning(f"[{self.name}] Stream reached end-of-stream or read failed.")
                    time.sleep(0.05)
                    continue

            capture_ts = time.perf_counter()
            self.frames_ingested += 1

            # Ring buffer overflow management (double buffering with latest-frame priority)
            if self.queue.full():
                try:
                    _ = self.queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self.queue.put_nowait((capture_ts, frame))
            except queue.Full:
                self.frames_dropped += 1

            if self.is_paced:
                next_frame_time += frame_interval
                if time.perf_counter() - next_frame_time > 0.5:
                    # Reset sync if scheduling drift occurs
                    next_frame_time = time.perf_counter()

    def get_frame(self, timeout: float = 0.2) -> Optional[Tuple[float, np.ndarray]]:
        """
        Retrieves the latest available frame from the ring buffer.
        Returns (capture_timestamp, frame_ndarray) or None if timed out.
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        total = self.frames_ingested
        dropped = self.frames_dropped
        drop_rate = (dropped / max(1, total)) * 100.0
        return {
            "name": self.name,
            "ingested": total,
            "dropped": dropped,
            "drop_rate_pct": drop_rate,
        }
