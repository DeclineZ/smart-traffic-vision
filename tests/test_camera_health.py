"""Black, frozen, shifted and low-detail camera views."""

import os
import sys
import unittest

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.camera_health import HealthSettings, ImageHealthMonitor


def scene(seed=0, shift=(0, 0), noise=0.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((360, 640, 3), np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, 600), rng.integers(0, 330)
        cv.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 20), tuple(int(c) for c in rng.integers(40, 255, 3)), -1)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img = cv.warpAffine(img, m, (640, 360), borderMode=cv.BORDER_REFLECT)
    if noise:
        img = np.clip(img + np.random.default_rng(seed + 1).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


FAST = HealthSettings(check_interval_s=0.0, frozen_after_s=5.0, shift_check_interval_s=0.0, max_shift_px=12.0, min_inliers=20)


class TestImageHealth(unittest.TestCase):
    def test_normal_scene_is_healthy(self):
        mon = ImageHealthMonitor(reference=scene(), settings=FAST)
        for t in range(10):
            mon.check(scene(noise=3.0 + t * 0.01), float(t))
        self.assertIsNone(mon.verdict())

    def test_black_frame(self):
        mon = ImageHealthMonitor(settings=FAST)
        mon.check(np.zeros((360, 640, 3), np.uint8), 0.0)
        self.assertEqual(mon.verdict(), "no_signal")

    def test_repeated_identical_frames_are_frozen_but_noisy_stationary_scene_is_not(self):
        frozen = ImageHealthMonitor(settings=FAST)
        img = scene()
        for t in range(8):
            frozen.check(img, float(t))
        self.assertEqual(frozen.verdict(), "frozen_feed")

        # An empty, unchanging road from a live sensor still has pixel noise.
        rng = np.random.default_rng(5)
        live = ImageHealthMonitor(settings=FAST)
        base = scene()
        for t in range(8):
            live.check(np.clip(base + rng.normal(0, 1.5, base.shape), 0, 255).astype(np.uint8), float(t))
        self.assertNotEqual(live.verdict(), "frozen_feed")

    def test_camera_shift_needs_consecutive_confirmations(self):
        mon = ImageHealthMonitor(reference=scene(), settings=FAST)
        moved = scene(shift=(40, 25))
        mon.check(moved, 0.0)
        mon.check(moved, 1.0)
        self.assertIsNone(mon.verdict(), "two checks are not enough")
        self.assertAlmostEqual(mon.flags["shiftPx"], 47.2, delta=3.0)
        mon.check(moved, 2.0)
        self.assertEqual(mon.verdict(), "camera_shifted")
        mon.check(scene(noise=1.0), 3.0)  # view restored
        self.assertIsNone(mon.verdict())

    def test_unshifted_view_with_traffic_changes_is_not_shifted(self):
        mon = ImageHealthMonitor(reference=scene(), settings=FAST)
        for t in range(5):
            img = scene(noise=2.0)
            cv.rectangle(img, (100 + 60 * t, 150), (220 + 60 * t, 230), (200, 200, 200), -1)  # a passing vehicle
            mon.check(img, float(t))
        self.assertIsNone(mon.verdict())
        self.assertLess(mon.flags["shiftPx"], 12)

    def test_low_detail_is_degraded_not_invalid(self):
        mon = ImageHealthMonitor(settings=FAST)
        gradient = np.tile(np.linspace(40, 220, 640, dtype=np.uint8), (360, 1))
        mon.check(np.repeat(gradient[:, :, None], 3, axis=2), 0.0)
        self.assertIsNone(mon.verdict())
        self.assertEqual(mon.degraded(), "low_detail")


if __name__ == "__main__":
    unittest.main()
