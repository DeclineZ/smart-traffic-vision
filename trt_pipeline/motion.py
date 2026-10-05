"""
Time-based queued/moving classification for tracked vehicles.

Speed is measured on the camera's observation clock (seconds), so the result does
not change with processing FPS or frame skipping. Distance is normalised for
perspective in one of two ways:

* ``homography`` (3x3, image pixels -> road-plane metres) when the camera has a
  road-plane calibration. Speeds are then metres/second.
* otherwise by the vehicle's own box height, so a distant car that moves one
  body-length per second is treated the same as a near one. Speeds are then
  box-heights/second.

A vehicle enters the queued state only after its speed stays below
``enter_speed`` for ``enter_duration_s``, and leaves it only after exceeding
``exit_speed`` for ``exit_duration_s`` (hysteresis). Until a track has
``min_history_s`` of history its state is ``unknown``.

Histories are bounded per track and pruned by observation time, so long runs
with many vehicles do not grow memory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

QUEUED = "queued"
MOVING = "moving"
UNKNOWN = "unknown"


@dataclass
class QueueSettings:
    enter_speed: float = 0.15
    exit_speed: float = 0.35
    enter_duration_s: float = 1.5
    exit_duration_s: float = 0.5
    window_s: float = 1.0
    min_history_s: float = 0.5
    track_ttl_s: float = 3.0
    max_samples: int = 64
    homography: Optional[np.ndarray] = None

    @property
    def units(self) -> str:
        return "m/s" if self.homography is not None else "box_heights/s"

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "QueueSettings":
        cfg = cfg or {}
        s = cls()
        for key in ("enter_speed", "exit_speed", "enter_duration_s", "exit_duration_s",
                    "window_s", "min_history_s", "track_ttl_s"):
            if key in cfg:
                setattr(s, key, float(cfg[key]))
        if cfg.get("homography") is not None:
            h = np.asarray(cfg["homography"], dtype=float)
            if h.shape != (3, 3) or not np.all(np.isfinite(h)):
                raise ValueError("queue.homography must be a finite 3x3 matrix")
            s.homography = h
        if s.exit_speed < s.enter_speed:
            raise ValueError("queue.exit_speed must be >= queue.enter_speed (hysteresis)")
        if min(s.window_s, s.min_history_s, s.track_ttl_s) <= 0:
            raise ValueError("queue window/history/ttl durations must be positive")
        return s


@dataclass
class _TrackMotion:
    samples: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)  # (t, x, y, box_h)
    state: str = UNKNOWN
    below_since: Optional[float] = None
    above_since: Optional[float] = None
    last_t: float = 0.0
    last_speed: Optional[float] = None


class MotionStateClassifier:
    """Per-camera queued/moving state machine keyed by track ID."""

    def __init__(self, settings: Optional[QueueSettings] = None):
        self.settings = settings or QueueSettings()
        self._tracks: Dict[int, _TrackMotion] = {}

    def __len__(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def _project(self, x: float, y: float) -> Tuple[float, float]:
        h = self.settings.homography
        v = h @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-9:
            return float("nan"), float("nan")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def _speed(self, tm: _TrackMotion, now_t: float) -> Optional[float]:
        s = self.settings
        window = [p for p in tm.samples if p[0] >= now_t - s.window_s]
        if len(window) < 2:
            # fall back to the oldest retained sample if the window is sparse
            window = list(tm.samples)
        if len(window) < 2:
            return None
        dt = window[-1][0] - window[0][0]
        if dt < s.min_history_s:
            return None
        # average the first/last few positions to reduce detector jitter
        k = max(1, min(3, len(window) // 3))
        first = np.mean([(p[1], p[2]) for p in window[:k]], axis=0)
        last = np.mean([(p[1], p[2]) for p in window[-k:]], axis=0)
        if s.homography is not None:
            a = self._project(*first)
            b = self._project(*last)
            dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        else:
            box_h = float(np.median([p[3] for p in window]))
            dist = float(np.hypot(*(last - first))) / max(box_h, 1.0)
        if not np.isfinite(dist):
            return None
        return dist / dt

    def update(self, track_id: int, anchor_xy: Sequence[float], box_h: float, t: float) -> str:
        """Adds one observation and returns the track's current state."""
        s = self.settings
        tm = self._tracks.get(track_id)
        if tm is None or (tm.samples and t - tm.samples[-1][0] > s.window_s):
            tm = _TrackMotion()
            self._tracks[track_id] = tm
        if tm.samples and t <= tm.samples[-1][0]:
            return tm.state  # duplicate/held observation: no new motion information
        tm.samples.append((float(t), float(anchor_xy[0]), float(anchor_xy[1]), float(box_h)))
        while len(tm.samples) > s.max_samples or (tm.samples and tm.samples[0][0] < t - 2.0 * s.window_s):
            tm.samples.popleft()
        tm.last_t = t

        speed = self._speed(tm, t)
        tm.last_speed = speed
        if speed is None:
            return tm.state

        if speed < s.enter_speed:
            tm.below_since = tm.below_since if tm.below_since is not None else t
        else:
            tm.below_since = None
        if speed > s.exit_speed:
            tm.above_since = tm.above_since if tm.above_since is not None else t
        else:
            tm.above_since = None

        if tm.state == QUEUED:
            if tm.above_since is not None and t - tm.above_since >= s.exit_duration_s:
                tm.state = MOVING
        else:
            if tm.below_since is not None and t - tm.below_since >= s.enter_duration_s:
                tm.state = QUEUED
            elif tm.state == UNKNOWN and speed >= s.enter_speed:
                tm.state = MOVING
            elif tm.state == UNKNOWN:
                pass  # slow but not yet slow for long enough: still unknown
        return tm.state

    def state_of(self, track_id: int) -> str:
        tm = self._tracks.get(track_id)
        return tm.state if tm else UNKNOWN

    def speed_of(self, track_id: int) -> Optional[float]:
        tm = self._tracks.get(track_id)
        return tm.last_speed if tm else None

    def prune(self, now_t: float) -> int:
        """Drops tracks not observed within ``track_ttl_s``; returns how many were removed."""
        ttl = self.settings.track_ttl_s
        dead = [tid for tid, tm in self._tracks.items() if now_t - tm.last_t > ttl]
        for tid in dead:
            del self._tracks[tid]
        return len(dead)
