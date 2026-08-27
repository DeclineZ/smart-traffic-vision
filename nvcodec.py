"""
Hardware-Accelerated Video Codecs (NVENC / NVDEC) for Smart Traffic Vision
Leverages dedicated NVIDIA Silicon Video Engines:
- NVENC: Hardware H.264 / HEVC video encoder for visual HUD, live re-streaming, or recording.
- NVDEC (CUVID): Hardware video decoder for multi-stream 1080p surveillance cameras.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2 as cv
import imageio_ffmpeg
import numpy as np


def get_ffmpeg_exe() -> str:
    """Returns the bundled high-performance FFmpeg binary with NVENC/CUVID support."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def is_nvenc_available() -> bool:
    """Probes whether NVIDIA NVENC hardware encoder is operational on the host."""
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=0.1:size=640x360:rate=30",
        "-c:v", "h264_nvenc",
        "-f", "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


def is_nvdec_available(sample_video: Optional[str] = None) -> bool:
    """Probes whether NVIDIA CUVID hardware decoder is operational."""
    ffmpeg_exe = get_ffmpeg_exe()
    if sample_video and os.path.exists(sample_video):
        cmd = [
            ffmpeg_exe,
            "-hwaccel", "cuvid",
            "-c:v", "h264_cuvid",
            "-i", sample_video,
            "-t", "0.2",
            "-f", "null",
            "-",
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            return res.returncode == 0
        except Exception:
            return False
    return is_nvenc_available()


class NVENCVideoWriter:
    """
    High-throughput hardware video writer backed by NVIDIA NVENC (h264_nvenc).
    Runs asynchronously with a bounded queue to prevent blocking the vision loop.
    """

    def __init__(
        self,
        output_path: Optional[str],
        width: int,
        height: int,
        fps: float = 25.0,
        bitrate: str = "4M",
        preset: str = "p1",  # p1 (fastest) to p7 (slowest/highest quality)
        tune: str = "zerolatency",
        queue_size: int = 64,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.preset = preset
        self.tune = tune
        self.queue_size = queue_size

        self.ffmpeg_exe = get_ffmpeg_exe()
        self.process: Optional[subprocess.Popen] = None
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        self.frames_written = 0
        self.frames_dropped = 0
        self.total_encode_time_ms = 0.0

        self._start_process()

    def _start_process(self):
        is_null_sink = not self.output_path or self.output_path == "-"

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", self.preset,
            "-zerolatency", "1",
            "-tune", "ull",
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
        ]

        if is_null_sink:
            cmd.extend(["-f", "null", "-"])
        else:
            cmd.append(self.output_path)

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10 * 1024 * 1024,
        )

        self.running = True
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()

    def _writer_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame_data = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame_data is None:
                self.queue.task_done()
                break

            t0 = time.perf_counter()
            try:
                if self.process and self.process.stdin:
                    self.process.stdin.write(frame_data)
                self.frames_written += 1
                self.total_encode_time_ms += (time.perf_counter() - t0) * 1000.0
            except (BrokenPipeError, OSError):
                break
            finally:
                self.queue.task_done()

    def write(self, frame: np.ndarray):
        """Pushes a frame to the NVENC encoder."""
        if not self.running or self.process is None:
            return

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv.resize(frame, (self.width, self.height))

        raw_bytes = frame.tobytes()
        try:
            self.queue.put_nowait(raw_bytes)
        except queue.Full:
            self.frames_dropped += 1

    def release(self):
        """Flushes buffered frames and terminates FFmpeg NVENC process."""
        if not self.running:
            return

        self.running = False
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

        if self.process and self.process.stdin:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()
        self.process = None

    @property
    def avg_encode_ms(self) -> float:
        return (self.total_encode_time_ms / self.frames_written) if self.frames_written > 0 else 0.0


class NVDECStreamSimulator:
    """
    Simulates high-speed live surveillance camera ingestion accelerated by NVIDIA CUVID/NVDEC.
    Decodes 1080p H.264 directly on GPU silicon, dropping CPU decode backpressure to near-zero.
    """

    def __init__(
        self,
        source: str,
        target_fps: float = 25.0,
        buffer_size: int = 1,
        is_paced: bool = True,
    ):
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.is_paced = is_paced

        self.queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.frames_ingested = 0
        self.frames_dropped = 0

        # Probe video dimensions
        self.width, self.height = self._probe_resolution(source)
        self.frame_size_bytes = self.width * self.height * 3
        self.ffmpeg_exe = get_ffmpeg_exe()

    def _probe_resolution(self, path: str) -> Tuple[int, int]:
        cap = cv.VideoCapture(path)
        if cap.isOpened():
            w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
        return 1920, 1080

    def start(self):
        self.running = True
        self.frames_ingested = 0
        self.frames_dropped = 0
        self.thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _decode_loop(self):
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        while self.running:
            # Build FFmpeg hardware decode command with looping
            cmd = [
                self.ffmpeg_exe,
                "-hwaccel", "cuda",
                "-stream_loop", "-1",
                "-i", self.source,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_size_bytes * 3,
            )

            while self.running and process.poll() is None:
                t_loop_start = time.perf_counter()

                raw_frame = process.stdout.read(self.frame_size_bytes)
                if len(raw_frame) < self.frame_size_bytes:
                    break

                t_captured = time.perf_counter()
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
                self.frames_ingested += 1

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

            if process.poll() is None:
                process.kill()

    def get_frame(self, timeout: float = 0.5) -> Optional[Tuple[float, np.ndarray]]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def drop_rate_pct(self) -> float:
        total = self.frames_ingested
        return (self.frames_dropped / total * 100.0) if total > 0 else 0.0
