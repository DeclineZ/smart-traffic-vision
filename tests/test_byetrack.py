"""
Unit tests for ByteTrack multi-object tracking implementation:
- Two-stage association (D_high and D_low)
- Occlusion recovery
- Track lifecycle (new, tracked, lost, removed)
- Consistent ID persistence across frames
- Bounding box output integrity (no stretching/deforming)
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.byetrack import ByteTrack, STrack


class TestByteTrack(unittest.TestCase):
    def setUp(self):
        STrack.reset_counter()
        self.tracker = ByteTrack(
            track_thresh=0.40,
            low_thresh=0.10,
            match_thresh=0.70,
            max_age=5,
            min_hits=1,
        )

    def test_single_object_tracking_and_id_persistence(self):
        # Frame 1: Detection of vehicle at (100, 100, 200, 200) with high score 0.85
        dets_f1 = np.array([[100, 100, 200, 200, 0.85, 0]], dtype=np.float32)
        tracks_f1 = self.tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 1)
        tid = int(tracks_f1[0, 4])

        # Frame 2: Vehicle moves slightly to (105, 105, 205, 205)
        dets_f2 = np.array([[105, 105, 205, 205, 0.88, 0]], dtype=np.float32)
        tracks_f2 = self.tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(int(tracks_f2[0, 4]), tid, "Track ID must persist across moving frames")

    def test_low_confidence_secondary_association_during_occlusion(self):
        # Frame 1: Vehicle detected clearly
        dets_f1 = np.array([[300, 300, 400, 400, 0.90, 0]], dtype=np.float32)
        tracks_f1 = self.tracker.update(dets_f1)
        tid = int(tracks_f1[0, 4])

        # Frame 2: Vehicle partially occluded (score drops to 0.25, below track_thresh=0.40)
        # Legacy SORT would DROP this detection and lose the track!
        # ByteTrack must match it via second association (low_thresh=0.10 <= 0.25 < 0.40)
        dets_f2 = np.array([[305, 305, 405, 405, 0.25, 0]], dtype=np.float32)
        tracks_f2 = self.tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 1, "ByteTrack must maintain track using low-score detection")
        self.assertEqual(int(tracks_f2[0, 4]), tid, "Track ID must be preserved during low-score occlusion")

    def test_bounding_box_output_integrity(self):
        # Ensure bounding boxes do not stretch or explode
        dets = np.array([[50, 60, 150, 160, 0.92, 0]], dtype=np.float32)
        tracks = self.tracker.update(dets)
        x1, y1, x2, y2 = tracks[0, :4]
        self.assertTrue(x2 > x1, "Width must be positive")
        self.assertTrue(y2 > y1, "Height must be positive")
        w = x2 - x1
        h = y2 - y1
        self.assertAlmostEqual(w, 100.0, delta=10.0)
        self.assertAlmostEqual(h, 100.0, delta=10.0)

    def test_untracked_removal_after_max_age(self):
        dets_f1 = np.array([[10, 10, 50, 50, 0.85, 0]], dtype=np.float32)
        tracks_f1 = self.tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 1)

        # Vehicle disappears for 10 frames (exceeds max_age=5)
        for _ in range(10):
            tracks = self.tracker.update(np.empty((0, 6), dtype=np.float32))

        self.assertEqual(len(tracks), 0)
        self.assertEqual(len(self.tracker.lost_stracks), 0, "Lost tracks must be purged after max_age")

    def test_min_hits_confirmation(self):
        tracker = ByteTrack(min_hits=2, max_age=5)
        d1 = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
        t1 = tracker.update(d1)
        # Should not be confirmed on frame 1
        self.assertEqual(len(t1), 0)

        # Confirmed on frame 2
        d2 = np.array([[102, 102, 202, 202, 0.9, 0]], dtype=np.float32)
        t2 = tracker.update(d2)
        self.assertEqual(len(t2), 1)
        tid = int(t2[0, 4])

        # Preserved on frame 3
        d3 = np.array([[104, 104, 204, 204, 0.9, 0]], dtype=np.float32)
        t3 = tracker.update(d3)
        self.assertEqual(len(t3), 1)
        self.assertEqual(int(t3[0, 4]), tid)


if __name__ == "__main__":
    unittest.main()
