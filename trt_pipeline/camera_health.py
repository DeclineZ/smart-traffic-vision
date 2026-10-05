"""
Per-camera image health checks: black/no-signal, frozen feed, obstructed or
blurred view, and camera shift relative to the calibration reference image.

Checks run on a small grayscale thumbnail at a bounded rate, so cost stays
negligible next to inference.

Frozen detection requires *bit-identical* thumbnails for ``frozen_after_s``. A
live sensor always has some noise, so a genuinely empty, stationary road still
changes slightly between frames; an encoder/NVR repeating the last frame does
not. Thresholds are deliberately conservative and should be tuned per site.

Camera shift is estimated from ORB features matched against the calibration
reference frame, with a RANSAC similarity fit: moving vehicles become outliers
and the static scene (buildings, poles, markings) determines the motion.
Whole-image phase correlation was tried first and reported shifts of hundreds of
pixels on fixed cameras because traffic dominates the image. A shift is only
reported after ``shift_confirmations`` consecutive checks agree, and a check
with too few inliers (night vs a day reference, heavy occlusion) is
inconclusive rather than a shift.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import cv2 as cv
import numpy as np

THUMB_W = 320


def _thumb(frame: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    scale = THUMB_W / float(w)
    return cv.resize(gray, (THUMB_W, max(1, int(round(h * scale)))), interpolation=cv.INTER_AREA)


@dataclass
class HealthSettings:
    check_interval_s: float = 1.0
    black_mean: float = 12.0
    black_std: float = 4.0
    frozen_after_s: float = 10.0
    low_detail_laplacian: float = 15.0
    obstruction_std: float = 8.0
    recovery_duration_s: float = 2.0
    shift_check_interval_s: float = 30.0
    max_shift_px: float = 12.0          # in full-resolution pixels
    min_inliers: int = 40
    shift_confirmations: int = 3

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "HealthSettings":
        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise ValueError("camera_health must be an object")
        names = {f.name for f in fields(cls)}
        if set(config) - names:
            raise ValueError(f"Unknown camera_health settings: {sorted(set(config) - names)}")
        values = {}
        for name, value in config.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"camera_health.{name} must be a finite nonnegative number")
            if name in ("min_inliers", "shift_confirmations") and (value < 1 or int(value) != value):
                raise ValueError(f"camera_health.{name} must be a positive integer")
            values[name] = int(value) if name in ("min_inliers", "shift_confirmations") else float(value)
        return cls(**values)


SHIFT_W = 640


class ImageHealthMonitor:
    """Tracks the latest health verdict for one camera."""

    def __init__(self, reference: Optional[np.ndarray] = None, settings: Optional[HealthSettings] = None):
        self.s = settings or HealthSettings()
        self._orb = cv.ORB_create(nfeatures=1500)
        self._matcher = cv.BFMatcher(cv.NORM_HAMMING)
        self._ref_kp, self._ref_desc = (None, None)
        if reference is not None:
            self._ref_kp, self._ref_desc = self._orb.detectAndCompute(self._shift_image(reference), None)
        self._shift_votes = 0
        self._last_check = -1e9
        self._last_shift_check = -1e9
        self._prev: Optional[np.ndarray] = None
        self._same_since: Optional[float] = None
        self._clear_since: Optional[float] = None
        self.flags: Dict[str, Any] = {"black": False, "frozen": False, "lowDetail": False,
                                      "obstructed": False,
                                      "shifted": False, "shiftPx": None, "shiftInliers": None}

    @staticmethod
    def _shift_image(frame: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        return cv.resize(gray, (SHIFT_W, max(1, int(round(h * SHIFT_W / float(w))))), interpolation=cv.INTER_AREA)

    def measure_shift(self, frame: np.ndarray) -> Optional[tuple]:
        """(shift in full-resolution px, inlier count) or None when inconclusive."""
        if self._ref_desc is None or len(self._ref_kp) < self.s.min_inliers:
            return None
        kp, desc = self._orb.detectAndCompute(self._shift_image(frame), None)
        if desc is None or len(kp) < self.s.min_inliers:
            return None
        pairs = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < self.s.min_inliers:
            return None
        src = np.float32([self._ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        model, inliers = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=3.0)
        n_in = int(inliers.sum()) if inliers is not None else 0
        if model is None or n_in < self.s.min_inliers:
            return None
        # displacement of the image centre under the fitted similarity
        cx, cy = SHIFT_W / 2.0, SHIFT_W * frame.shape[0] / (2.0 * frame.shape[1])
        moved = model @ np.array([cx, cy, 1.0])
        scale = frame.shape[1] / float(SHIFT_W)
        return float(np.hypot(moved[0] - cx, moved[1] - cy) * scale), n_in

    def reset(self) -> None:
        self._prev = None
        self._same_since = None
        self._clear_since = None
        self._last_check = self._last_shift_check = -1e9
        self._shift_votes = 0
        self.flags.update(black=False, frozen=False, lowDetail=False)

    def check(self, frame: np.ndarray, now: float) -> Dict[str, Any]:
        """``now`` is a monotonic time in seconds."""
        if now - self._last_check < self.s.check_interval_s:
            return self.flags
        self._last_check = now
        th = _thumb(frame)

        mean, std = float(th.mean()), float(th.std())
        self.flags["black"] = mean < self.s.black_mean and std < self.s.black_std

        if self._prev is not None and self._prev.shape == th.shape and np.array_equal(self._prev, th):
            if self._same_since is None:
                self._same_since = now
        else:
            self._same_since = None
        self._prev = th
        self.flags["frozen"] = self._same_since is not None and now - self._same_since >= self.s.frozen_after_s

        self.flags["lowDetail"] = float(cv.Laplacian(th, cv.CV_64F).var()) < self.s.low_detail_laplacian
        if self.flags["lowDetail"] and std < self.s.obstruction_std:
            self.flags["obstructed"] = True
            self._clear_since = None
        elif not self.flags["lowDetail"] and not self.flags["black"] and not self.flags["frozen"]:
            if self._clear_since is None:
                self._clear_since = now
            if now - self._clear_since >= self.s.recovery_duration_s:
                self.flags["obstructed"] = False
        else:
            self._clear_since = None

        if self._ref_desc is not None and now - self._last_shift_check >= self.s.shift_check_interval_s:
            self._last_shift_check = now
            res = None if self.flags["black"] else self.measure_shift(frame)
            if res is not None:
                shift, n_in = res
                self.flags["shiftPx"] = round(shift, 1)
                self.flags["shiftInliers"] = n_in
                if shift > self.s.max_shift_px:
                    self._shift_votes += 1
                    if self._shift_votes >= self.s.shift_confirmations:
                        self.flags["shifted"] = True
                else:
                    self._shift_votes = 0
                    self.flags["shifted"] = False
            else:
                self._shift_votes = 0
        return self.flags

    def verdict(self) -> Optional[str]:
        """Reason the camera's measurements are invalid, or None."""
        for key, reason in (("black", "no_signal"), ("frozen", "frozen_feed"), ("shifted", "camera_shifted"),
                            ("obstructed", "low_detail")):
            if self.flags.get(key):
                return reason
        return None

    def degraded(self) -> Optional[str]:
        return "low_detail" if self.flags.get("lowDetail") else None
