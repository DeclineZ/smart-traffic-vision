"""
Camera ingestion for Smart Traffic Vision.

Two source types share one polling interface (``poll()``, ``status()``, ``start()``, ``stop()``):

* ``StreamBufferWorker`` -- live sources (RTSP/HTTP/device). A background thread
  opens the capture with bounded open/read timeouts, reads continuously so the
  decoder/network backlog cannot grow, keeps only the newest frame, and reconnects
  with exponential backoff. Each reconnect starts a new *epoch* so the runner can
  reset tracker/motion/gate state across the discontinuity.
* ``FileReplaySource`` -- local recordings, read synchronously in lockstep. Frame
  timestamps come from the frame index and an explicit replay FPS (media time),
  not from how fast this machine happens to process them. Looping to the start of
  the file starts a new epoch.

Every frame is delivered as a ``FramePacket`` carrying its local capture wall time
(for observation age/freshness) and a per-epoch monotonic observation clock in
seconds (for speeds and durations).
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import cv2 as cv
import numpy as np

from .tools import get_logger

logger = get_logger("StreamBufferWorker")

_CREDENTIALS_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<cred>[^/@\s]+)@")


def redact_source(source: Any) -> str:
    """Hides user:password in stream URLs before they reach logs or payloads."""
    return _CREDENTIALS_RE.sub(lambda m: f"{m.group('scheme')}***@", str(source))


def is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    s = str(source)
    return "://" in s or not os.path.exists(s)


@dataclass
class FramePacket:
    frame: np.ndarray
    seq: int
    epoch: int
    captured_wall: float   # time.time() when the frame was captured/decoded locally
    obs_t: float           # seconds on this camera's observation clock (monotonic within an epoch)
    captured_mono: Optional[float] = None  # local freshness clock, independent of wall-clock corrections

    def age_s(self, now_wall: Optional[float] = None) -> float:
        if self.captured_mono is not None:
            return max(0.0, time.monotonic() - self.captured_mono)
        return max(0.0, (now_wall if now_wall is not None else time.time()) - self.captured_wall)


def open_capture(source: Any, open_timeout_s: float, read_timeout_s: float) -> cv.VideoCapture:
    """Opens a capture with timeouts applied at open time (OpenCV only honours them there)."""
    params = [
        cv.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000),
        cv.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_s * 1000),
    ]
    if isinstance(source, str) and "://" in source:
        try:
            return cv.VideoCapture(source, cv.CAP_FFMPEG, params)
        except (cv.error, TypeError):
            pass
    try:
        return cv.VideoCapture(source, cv.CAP_ANY, params)
    except (cv.error, TypeError):
        return cv.VideoCapture(source)


class StreamBufferWorker:
    """Background live-camera reader with latest-frame delivery and reconnection."""

    def __init__(
        self,
        name: str,
        source: str | int,
        target_fps: float = 25.0,
        buffer_size: int = 1,
        is_paced: bool = False,
        loop_video: bool = False,
        open_timeout_s: float = 10.0,
        read_timeout_s: float = 5.0,
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 30.0,
        max_consecutive_failures: int = 3,
        capture_factory: Optional[Callable[[], Any]] = None,
    ):
        self.name = name
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = 1  # latest-only; older frames are never processed
        self.is_paced = is_paced
        self.loop_video = loop_video
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self._capture_factory = capture_factory or (
            lambda: open_capture(self.source, self.open_timeout_s, self.read_timeout_s)
        )

        self.cap: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self._stop_evt = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._consumed_seq = 0

        self.state = "starting"
        self.epoch = 0
        self.seq = 0
        self.frames_ingested = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.open_failures = 0
        self.last_error: Optional[str] = None
        self.last_frame_wall: Optional[float] = None
        self.last_frame_mono: Optional[float] = None
        self.frame_shape: Optional[tuple] = None
        self.source_fps = 25.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Starts the reader thread. Opening happens in the thread, so start never blocks."""
        self.running = True
        self._stop_evt.clear()
        self.thread = threading.Thread(target=self._run, name=f"Ingest-{self.name}", daemon=True)
        self.thread.start()
        logger.info(f"[{self.name}] ingestion started: source='{redact_source(self.source)}'")

    def stop(self) -> None:
        self.running = False
        self._stop_evt.set()
        with self._cond:
            self._cond.notify_all()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=max(1.0, self.open_timeout_s, self.read_timeout_s) + 0.5)
        if not self.thread or not self.thread.is_alive():
            self._release()
        else:
            logger.warning(f"[{self.name}] capture backend has not returned; reader will release it on exit")
        self.state = "stopped"
        logger.info(f"[{self.name}] ingestion stopped (ingested={self.frames_ingested}, "
                    f"dropped={self.frames_dropped}, reconnects={self.reconnects})")

    def _release(self) -> None:
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                logger.debug(f"[{self.name}] capture release failed: {redact_source(exc)}")

    # ------------------------------------------------------------------ reader thread
    def _run(self) -> None:
        backoff = self.reconnect_initial_s
        failures = 0
        try:
            while self.running:
                if self.cap is None:
                    try:
                        self.cap = self._capture_factory()
                        if not self.running:
                            break
                        if self.cap is None or not self.cap.isOpened():
                            raise OSError("capture could not be opened")
                        fps = self.cap.get(cv.CAP_PROP_FPS)
                    except Exception as exc:  # native capture backends use different exception types
                        self.last_error = redact_source(exc)
                        self.open_failures += 1
                        self.state = "reconnecting" if self.epoch > 0 else "offline"
                        self._release()
                        logger.warning(f"[{self.name}] open failed; retrying in {backoff:.1f}s")
                        if self._stop_evt.wait(backoff):
                            break
                        backoff = min(self.reconnect_max_s, backoff * 2.0)
                        continue
                    self.epoch += 1
                    if self.epoch > 1:
                        self.reconnects += 1
                    failures = 0
                    if fps and fps > 1.0:
                        self.source_fps = float(fps)
                    self.state = "connecting"
                    logger.info(f"[{self.name}] connected (epoch {self.epoch})")

                try:
                    ok, frame = self.cap.read()
                except Exception as exc:
                    self.last_error = redact_source(exc)
                    ok, frame = False, None
                    failures = self.max_consecutive_failures - 1
                if not self.running:
                    break
                if not ok or frame is None:
                    failures += 1
                    if failures >= self.max_consecutive_failures:
                        self.last_error = self.last_error or "read failed"
                        logger.warning(f"[{self.name}] capture read failed; retrying in {backoff:.1f}s")
                        self.state = "reconnecting"
                        self._release()
                        if self._stop_evt.wait(backoff):
                            break
                        backoff = min(self.reconnect_max_s, backoff * 2.0)
                    continue
                failures = 0
                backoff = self.reconnect_initial_s
                self.last_error = None
                packet = FramePacket(frame=frame, seq=self.seq + 1, epoch=self.epoch,
                                     captured_wall=time.time(), obs_t=time.perf_counter(), captured_mono=time.monotonic())
                with self._cond:
                    if self._latest is not None and self._latest.seq > self._consumed_seq:
                        self.frames_dropped += 1
                    self.seq = packet.seq
                    self._latest = packet
                    self._cond.notify_all()
                self.frames_ingested += 1
                self.last_frame_wall = packet.captured_wall
                self.last_frame_mono = packet.captured_mono
                self.frame_shape = frame.shape[:2]
                self.state = "ok"
        finally:
            self._release()
            self.running = False

    # ------------------------------------------------------------------ consumer API
    def poll(self) -> Optional[FramePacket]:
        """Returns the newest frame not yet handed out, without waiting."""
        with self._cond:
            if self._latest is None or self._latest.seq <= self._consumed_seq:
                return None
            self._consumed_seq = self._latest.seq
            return self._latest

    def get_frame(self, timeout: float = 0.2) -> Optional[tuple]:
        """Compatibility helper: waits for a newer frame and returns (captured_wall, frame)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.running and (self._latest is None or self._latest.seq <= self._consumed_seq):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._latest is None or self._latest.seq <= self._consumed_seq:
                return None
            self._consumed_seq = self._latest.seq
            return (self._latest.captured_wall, self._latest.frame)

    def status(self) -> Dict[str, Any]:
        age = None if self.last_frame_mono is None else max(0.0, time.monotonic() - self.last_frame_mono)
        return {
            "name": self.name,
            "source": redact_source(self.source),
            "state": self.state,
            "epoch": self.epoch,
            "lastFrameAgeSec": age,
            "framesIngested": self.frames_ingested,
            "framesDropped": self.frames_dropped,
            "reconnects": self.reconnects,
            "openFailures": self.open_failures,
            "lastError": self.last_error,
        }

    def get_stats(self) -> dict:
        total = self.frames_ingested
        return {
            "name": self.name,
            "ingested": total,
            "dropped": self.frames_dropped,
            "drop_rate_pct": (self.frames_dropped / max(1, total)) * 100.0,
        }


class FileReplaySource:
    """Synchronous lockstep reader for recorded files with explicit media time."""

    def __init__(
        self,
        name: str,
        path: str,
        replay_fps: Optional[float] = None,
        loop_video: bool = True,
        capture_factory: Optional[Callable[[], Any]] = None,
        reopen_interval_s: float = 5.0,
    ):
        self.name = name
        self.source = path
        self.loop_video = loop_video
        self._capture_factory = capture_factory or (lambda: cv.VideoCapture(path))
        self.reopen_interval_s = reopen_interval_s
        self.cap: Optional[Any] = None
        self.replay_fps = replay_fps
        self.state = "starting"
        self.epoch = 0
        self.seq = 0
        self.frame_index = 0
        self.frames_ingested = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.open_failures = 0
        self.last_error: Optional[str] = None
        self.last_frame_wall: Optional[float] = None
        self.last_frame_mono: Optional[float] = None
        self.frame_shape: Optional[tuple] = None
        self._next_open_mono = 0.0

    def start(self) -> None:
        self._open()

    def _open(self) -> bool:
        self._next_open_mono = time.monotonic() + self.reopen_interval_s
        try:
            self.cap = self._capture_factory()
            if self.cap is None or not self.cap.isOpened():
                raise OSError("cannot open file")
            if self.replay_fps is None:
                fps = self.cap.get(cv.CAP_PROP_FPS)
                self.replay_fps = float(fps) if fps and fps > 1.0 else 25.0
                if self.replay_fps > 60.0:
                    logger.warning(f"[{self.name}] file reports {self.replay_fps:.0f} FPS; verify and pass --replay-fps")
        except Exception as exc:
            self.open_failures += 1
            self._failed(redact_source(exc))
            logger.error(f"[{self.name}] cannot open recording '{redact_source(self.source)}'")
            return False
        self._new_epoch()
        return True

    def _failed(self, reason: str) -> None:
        self.stop()
        self.state = "offline"
        self.last_error = reason
        self._next_open_mono = time.monotonic() + self.reopen_interval_s

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.frame_index = 0

    def stop(self) -> None:
        if self.cap is not None:
            cap, self.cap = self.cap, None
            try:
                cap.release()
            except Exception as exc:
                logger.debug(f"[{self.name}] capture release failed: {redact_source(exc)}")
        self.state = "stopped"

    def poll(self) -> Optional[FramePacket]:
        if self.cap is None:
            if time.monotonic() < self._next_open_mono or not self._open():
                return None
        try:
            ok, frame = self.cap.read()
            if (not ok or frame is None) and self.loop_video:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._new_epoch()
                ok, frame = self.cap.read()
        except Exception as exc:
            self._failed(redact_source(exc))
            return None
        if not ok or frame is None:
            self._failed("end of file" if not self.loop_video else "read failed")
            return None

        self.seq += 1
        self.frames_ingested += 1
        obs_t = self.frame_index / float(self.replay_fps or 25.0)
        self.frame_index += 1
        self.last_frame_wall = time.time()
        self.last_frame_mono = time.monotonic()
        self.frame_shape = frame.shape[:2]
        self.state = "ok"
        return FramePacket(frame=frame, seq=self.seq, epoch=self.epoch,
                           captured_wall=self.last_frame_wall, obs_t=obs_t, captured_mono=self.last_frame_mono)

    def status(self) -> Dict[str, Any]:
        age = None if self.last_frame_mono is None else max(0.0, time.monotonic() - self.last_frame_mono)
        return {
            "name": self.name,
            "source": redact_source(self.source),
            "state": self.state,
            "epoch": self.epoch,
            "lastFrameAgeSec": age,
            "framesIngested": self.frames_ingested,
            "framesDropped": self.frames_dropped,
            "reconnects": self.reconnects,
            "openFailures": self.open_failures,
            "lastError": self.last_error,
            "replayFps": self.replay_fps,
        }
