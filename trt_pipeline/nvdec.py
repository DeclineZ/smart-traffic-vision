"""
GPU video decoding for camera ingestion.

``NvdecCapture`` behaves like ``cv.VideoCapture`` (``isOpened``/``read``/``get``/``set``/``release``)
so the stream workers can use it unchanged. An ffmpeg child process decodes on the GPU's
NVDEC engine and resizes there (``scale_cuda``) to the detector's input width, then pipes
small NV12 frames to Python. Decoding eight 1080p H.264 streams in software costs about 11 ms
of CPU per frame; this path costs about 1 ms, which is what lets one PC serve 8 cameras.

The resize reproduces what Ultralytics does to a full-resolution frame (cv2 INTER_LINEAR),
because that is what the model's accuracy was validated on: at an odd integer ratio
(1920 -> 640) INTER_LINEAR samples single pixels, so ``nearest`` matches it; at other ratios
(1920 -> 960) it is a 2-tap average, which ``bilinear`` matches. An anti-aliased (area)
resize was measured to lose ~5 points of small-vehicle recall.

Frames come out smaller than the camera's native resolution. ``scale`` (native pixels per
delivered pixel) travels with every frame so detections can be mapped back to the
full-resolution coordinates the lane calibration uses.
"""

from __future__ import annotations

import collections
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Optional, Tuple

import cv2 as cv
import numpy as np

from .tools import get_logger

logger = get_logger("NvdecCapture")

_STREAM_RE = re.compile(r"Stream #\S+.*?Video: (?P<codec>\w+).*?, (?P<w>\d{2,5})x(?P<h>\d{2,5})")
_FPS_RE = re.compile(r"(?P<fps>\d+(?:\.\d+)?) fps")
# Full-range ("JPEG", yuvj*/pc) to video-range YUV, so cv2's NV12 conversion (video range) is correct.
_Y_FULL_TO_VIDEO = np.round(np.arange(256) * 219.0 / 255.0 + 16.0).astype(np.uint8)
_C_FULL_TO_VIDEO = np.round((np.arange(256) - 128.0) * 224.0 / 255.0 + 128.0).astype(np.uint8)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def ffmpeg_exe() -> Optional[str]:
    """ffmpeg to use: $VISION_FFMPEG, else the imageio-ffmpeg build (has CUVID on Windows/Linux), else PATH."""
    env = os.environ.get("VISION_FFMPEG")
    if env:
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


def _is_network(source: Any) -> bool:
    return isinstance(source, str) and "://" in source


def _input_args(source: str, timeout_s: float) -> list:
    if source.lower().startswith("rtsp://"):
        return ["-rtsp_transport", "tcp", "-timeout", str(int(timeout_s * 1e6))]
    if _is_network(source):
        return ["-rw_timeout", str(int(timeout_s * 1e6))]
    return []


def probe(source: str, exe: str, timeout_s: float = 10.0) -> Tuple[str, int, int, float, bool]:
    """(codec, width, height, fps, full_range) of the first video stream, from ffmpeg's input banner."""
    cmd = [exe, "-hide_banner", "-nostdin", *_input_args(source, timeout_s), "-i", source]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout_s + 2.0, creationflags=_NO_WINDOW).stderr
    except subprocess.TimeoutExpired as exc:
        raise OSError("ffmpeg probe timed out") from exc
    return parse_banner(out.decode("utf-8", "replace"))


def parse_banner(text: str) -> Tuple[str, int, int, float, bool]:
    """(codec, width, height, fps, full_range) from the "Stream #0:0: Video: ..." line."""
    m = _STREAM_RE.search(text)
    if not m:
        raise OSError("ffmpeg found no video stream")
    end = text.find("\n", m.start())
    line = text[m.start():end if end >= 0 else len(text)]
    fps_m = _FPS_RE.search(line)
    full_range = "yuvj" in line or "(pc" in line
    return (m.group("codec"), int(m.group("w")), int(m.group("h")),
            float(fps_m.group("fps")) if fps_m else 0.0, full_range)


def output_size(src_w: int, src_h: int, out_width: int) -> Tuple[int, int]:
    """Downscale only, keep the aspect ratio, even dimensions (NV12 needs them)."""
    w = min(int(out_width), src_w) // 2 * 2
    return w, max(2, int(round(src_h * w / src_w / 2.0)) * 2)


def resize_interp(scale: float) -> str:
    """scale_cuda filter that reproduces cv2.INTER_LINEAR (what Ultralytics applies to a full frame):
    single-pixel sampling at odd integer ratios, a 2-tap average otherwise."""
    ratio = round(scale)
    return "nearest" if abs(scale - ratio) < 1e-6 and ratio % 2 == 1 else "bilinear"


class NvdecCapture:
    """cv.VideoCapture-compatible reader backed by ffmpeg + NVDEC with on-GPU resize."""

    def __init__(self, source: str, out_width: int = 640, realtime: bool = False,
                 open_timeout_s: float = 10.0, read_timeout_s: float = 5.0, exe: Optional[str] = None,
                 start_s: float = 0.0):
        self.source = str(source)
        self.start_s = float(start_s)  # files only: frame-accurate seek (decodes from the previous keyframe)
        self.realtime = realtime
        self.read_timeout_s = read_timeout_s
        self.exe = exe or ffmpeg_exe()
        if not self.exe:
            raise OSError("ffmpeg not found (pip install imageio-ffmpeg or set VISION_FFMPEG)")
        self.codec, self.src_w, self.src_h, self.fps, self.full_range = probe(self.source, self.exe, open_timeout_s)
        self.out_w, self.out_h = output_size(self.src_w, self.src_h, out_width)
        self.scale = self.src_w / float(self.out_w)
        self.interp = resize_interp(self.scale)
        self._nbytes = self.out_w * self.out_h * 3 // 2
        self._proc: Optional[subprocess.Popen] = None
        self._frames: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2)
        self._halt = threading.Event()
        self._stderr_tail: collections.deque = collections.deque(maxlen=8)
        self._pending: Optional[np.ndarray] = None
        self._ended = False
        self._start()
        # Fail here, not on every reconnect, when the GPU decoder cannot start (no NVIDIA GPU/driver).
        # The first frame can wait for a keyframe, so it gets the open timeout.
        ok, frame = self._next(open_timeout_s)
        if not ok:
            proc = self._proc
            if proc is not None:
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            self._stop()
            raise OSError(f"NVDEC decode failed: {self.last_error or 'no frames'}")
        self._pending = frame
        logger.info(f"GPU decoding {self.codec} {self.src_w}x{self.src_h} -> {self.out_w}x{self.out_h} "
                    f"({self.interp}{', full range' if self.full_range else ''})")

    # ------------------------------------------------------------------ process
    def _start(self) -> None:
        # read() notices a stall after read_timeout_s and stops ffmpeg. ffmpeg's own socket timeout is
        # only a backstop: on an RTSP stall ffmpeg 7.1 needs two of them to exit and flushes stale
        # frames after the first, so it must not fire before read() does.
        cmd = [self.exe, "-hide_banner", "-loglevel", "error", "-nostdin",
               *_input_args(self.source, self.read_timeout_s + 1.0)]
        if self.realtime and not _is_network(self.source):
            cmd += ["-re"]
        if self.start_s > 0 and not _is_network(self.source):
            cmd += ["-ss", f"{self.start_s:.6f}"]
        # -threads 1: more decoder threads make NVDEC allocate more than its 32 surfaces and fail
        cmd += ["-threads", "1", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", self.source,
                "-vf", f"scale_cuda={self.out_w}:{self.out_h}:interp_algo={self.interp},hwdownload,format=nv12",
                # one output frame per decoded frame: no timestamp-driven drops or duplicates
                "-fps_mode", "passthrough", "-an", "-sn", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                      bufsize=0, creationflags=_NO_WINDOW)
        self._frames = queue.Queue(maxsize=2)
        self._halt = threading.Event()
        self._ended = False
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._pump, args=(self._proc, self._frames, self._halt), daemon=True).start()

    def _pump(self, proc: subprocess.Popen, frames: queue.Queue, halt: threading.Event) -> None:
        """Reads and colour-converts the next frames while the caller runs inference. The queue is
        bounded and never drops, so callers still see every frame in order (replay stays exact)."""
        buf = bytearray(self._nbytes)
        view = memoryview(buf)
        while not halt.is_set():
            got = 0
            while got < self._nbytes:  # a pipe read may return part of a frame
                try:
                    n = proc.stdout.readinto(view[got:])
                except (OSError, ValueError):
                    n = 0
                if not n:
                    break
                got += n
            frame = self._convert(buf) if got == self._nbytes else None
            while not halt.is_set():
                try:
                    frames.put(frame, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if frame is None:
                return

    def _convert(self, buf: bytearray) -> np.ndarray:
        yuv = np.frombuffer(buf, np.uint8).reshape(self.out_h * 3 // 2, self.out_w)
        if self.full_range:
            yuv = np.vstack([cv.LUT(yuv[:self.out_h], _Y_FULL_TO_VIDEO), cv.LUT(yuv[self.out_h:], _C_FULL_TO_VIDEO)])
        return cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV12)  # new array; buf is reused

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        for line in iter(proc.stderr.readline, b""):
            self._stderr_tail.append(line.decode("utf-8", "replace").strip())

    def _stop(self) -> None:
        self._halt.set()
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass

    @property
    def last_error(self) -> Optional[str]:
        return self._stderr_tail[-1] if self._stderr_tail else None

    # ------------------------------------------------------------------ cv.VideoCapture interface
    def isOpened(self) -> bool:
        return self._proc is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return True, frame
        if self._proc is None or self._ended:
            return False, None
        return self._next(self.read_timeout_s)

    def _next(self, timeout_s: float) -> Tuple[bool, Optional[np.ndarray]]:
        """Next decoded frame. A stream that ends, or delivers nothing for ``timeout_s``, stays ended
        until restarted, so the caller's later reads fail at once and it can reconnect."""
        try:
            frame = self._frames.get(timeout=timeout_s)
        except queue.Empty:
            frame = None
            self._stop()  # stalled; stop it before it can deliver stale frames
        if frame is None:
            self._ended = True
            return False, None
        return True, frame

    def get(self, prop: int) -> float:
        if prop == cv.CAP_PROP_FPS:
            return self.fps
        if prop == cv.CAP_PROP_FRAME_WIDTH:
            return float(self.src_w)
        if prop == cv.CAP_PROP_FRAME_HEIGHT:
            return float(self.src_h)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        """Only rewinding to the start is supported (used to loop recordings)."""
        if prop == cv.CAP_PROP_POS_FRAMES and value == 0:
            self._pending = None
            self.start_s = 0.0
            self._stop()
            self._start()
            return True
        return False

    def release(self) -> None:
        self._stop()


def make_capture(source: Any, decoder: str = "auto", out_width: int = 640, realtime: bool = False,
                 open_timeout_s: float = 10.0, read_timeout_s: float = 5.0):
    """NVDEC capture when ``decoder`` is 'nvdec' (or 'auto' and it works), else an OpenCV capture."""
    if decoder in ("auto", "nvdec") and not isinstance(source, int):
        try:
            return NvdecCapture(str(source), out_width=out_width, realtime=realtime,
                                open_timeout_s=open_timeout_s, read_timeout_s=read_timeout_s)
        except OSError as exc:
            if decoder == "nvdec":
                raise
            logger.warning(f"NVDEC decoding unavailable ({exc}); using OpenCV")
    from .stream import open_capture

    return open_capture(source, open_timeout_s, read_timeout_s)
