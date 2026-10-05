"""Time-based queued/moving classification: FPS invariance, perspective, hysteresis, bounded memory."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.motion import MOVING, QUEUED, UNKNOWN, MotionStateClassifier, QueueSettings


def run(clf, tid, fps, seconds, speed_px_s, box_h=20.0, x0=0.0, t0=0.0, jitter=0.0, rng=None):
    state = None
    n = int(round(seconds * fps))
    for i in range(n + 1):
        t = t0 + i / fps
        x = x0 + speed_px_s * (t - t0)
        if jitter:
            x += rng.normal(0, jitter)
        state = clf.update(tid, (x, 100.0), box_h, t)
    return state


class TestMotionStateClassifier(unittest.TestCase):
    def test_new_track_is_unknown(self):
        clf = MotionStateClassifier()
        self.assertEqual(clf.update(1, (0, 0), 20, 0.0), UNKNOWN)

    def test_same_physical_speed_same_state_at_any_fps(self):
        # 30 px/s with 20 px boxes = 1.5 box-heights/s: moving at every processing rate.
        for fps in (5, 10, 12.5, 25, 30):
            clf = MotionStateClassifier()
            self.assertEqual(run(clf, 1, fps, 4.0, 30.0), MOVING, f"fps={fps}")
        # 1 px/s = 0.05 box-heights/s: queued at every processing rate.
        for fps in (5, 10, 12.5, 25, 30):
            clf = MotionStateClassifier()
            self.assertEqual(run(clf, 1, fps, 4.0, 1.0), QUEUED, f"fps={fps}")

    def test_perspective_normalisation(self):
        # A distant vehicle (6 px tall) creeping 3 px/s is moving 0.5 body-lengths/s: not queued,
        # while a near vehicle (60 px tall) at 3 px/s is effectively stopped.
        far, near = MotionStateClassifier(), MotionStateClassifier()
        self.assertEqual(run(far, 1, 10, 4.0, 3.0, box_h=6.0), MOVING)
        self.assertEqual(run(near, 1, 10, 4.0, 3.0, box_h=60.0), QUEUED)

    def test_requires_sustained_low_speed(self):
        clf = MotionStateClassifier(QueueSettings(enter_duration_s=1.5))
        run(clf, 1, 10, 2.0, 40.0)                              # moving
        state = run(clf, 1, 10, 1.0, 0.0, x0=80.0, t0=2.1)      # stopped for only ~1 s
        self.assertEqual(state, MOVING)
        state = run(clf, 1, 10, 2.0, 0.0, x0=80.0, t0=3.2)      # stopped long enough
        self.assertEqual(state, QUEUED)

    def test_hysteresis_creeping_queue_stays_queued(self):
        clf = MotionStateClassifier()
        run(clf, 1, 10, 4.0, 0.0)
        self.assertEqual(clf.state_of(1), QUEUED)
        # creeping at 0.25 box-heights/s: between enter (0.15) and exit (0.35) thresholds
        self.assertEqual(run(clf, 1, 10, 3.0, 5.0, t0=4.1), QUEUED)
        # clearly moving for longer than exit_duration
        self.assertEqual(run(clf, 1, 10, 2.0, 40.0, x0=15.0, t0=7.2), MOVING)

    def test_detection_jitter_does_not_release_queue(self):
        rng = np.random.default_rng(0)
        clf = MotionStateClassifier()
        self.assertEqual(run(clf, 1, 10, 6.0, 0.0, jitter=1.0, rng=rng), QUEUED)

    def test_duplicate_timestamps_ignored(self):
        clf = MotionStateClassifier()
        clf.update(1, (0, 0), 20, 0.0)
        clf.update(1, (50, 0), 20, 0.0)  # held box / repeated frame
        self.assertEqual(len(clf._tracks[1].samples), 1)

    def test_homography_mode_uses_metres(self):
        # 10 px = 1 m
        h = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]
        s = QueueSettings.from_config({"homography": h, "enter_speed": 0.5, "exit_speed": 1.5})
        self.assertEqual(s.units, "m/s")
        clf = MotionStateClassifier(s)
        self.assertEqual(run(clf, 1, 10, 4.0, 100.0, box_h=5.0), MOVING)   # 10 m/s
        clf2 = MotionStateClassifier(s)
        self.assertEqual(run(clf2, 1, 10, 4.0, 2.0, box_h=500.0), QUEUED)  # 0.2 m/s

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            QueueSettings.from_config({"enter_speed": 0.5, "exit_speed": 0.2})
        with self.assertRaises(ValueError):
            QueueSettings.from_config({"homography": [[1, 0], [0, 1]]})

    def test_memory_bounded_under_sustained_new_ids(self):
        clf = MotionStateClassifier(QueueSettings(track_ttl_s=3.0))
        t = 0.0
        for tid in range(20000):        # a new vehicle every 0.1 s for ~33 minutes
            t = tid * 0.1
            clf.update(tid, (tid % 500, 100), 20, t)
            clf.update(tid, (tid % 500 + 1, 100), 20, t + 0.05)
            if tid % 10 == 0:
                clf.prune(t)
        clf.prune(t)
        self.assertLessEqual(len(clf), 40)
        longest = max(len(tm.samples) for tm in clf._tracks.values())
        self.assertLessEqual(longest, clf.settings.max_samples)


if __name__ == "__main__":
    unittest.main()
