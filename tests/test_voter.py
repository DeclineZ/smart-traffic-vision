"""
Unit tests for TrackClassVotingFilter:
- Temporal vote accumulation
- Pickup truck (car vs truck) ambiguity resolution
- Hysteresis switching threshold
- Multi-camera namespace isolation and memory pruning
"""

import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.voter import TrackClassVotingFilter


class TestTrackClassVotingFilter(unittest.TestCase):
    def setUp(self):
        self.voter = TrackClassVotingFilter(
            num_streams=2,
            window_size=10,
            car_truck_bias=1.15,
            hysteresis_margin=1.20,
            decay_factor=0.95,
            car_cls_id=0,
            truck_cls_id=3,
            enabled=True,
        )

    def test_single_observation(self):
        cls_out = self.voter.update(cam_idx=0, track_id=1, raw_cls_id=0, conf=0.85)
        self.assertEqual(cls_out, 0)

    def test_pickup_truck_ambiguity_favors_car(self):
        """
        Alternating detections between 0 (car) and 3 (truck) with similar confidence
        should reliably resolve and lock to 0 (car) due to the pickup prior.
        """
        # Alternate: car, truck, car, truck, car
        feed = [(0, 0.70), (3, 0.65), (0, 0.72), (3, 0.60), (0, 0.75)]
        final_cls = 0
        for raw_cls, conf in feed:
            final_cls = self.voter.update(cam_idx=0, track_id=10, raw_cls_id=raw_cls, conf=conf)

        self.assertEqual(final_cls, 0, "Alternating pickup truck detections should stabilize to car (0)")

    def test_single_frame_flicker_rejected(self):
        """
        An established car track should not flip to truck on a single rogue detection.
        """
        # Establish car with 5 solid detections
        for _ in range(5):
            self.voter.update(cam_idx=0, track_id=20, raw_cls_id=0, conf=0.85)

        # Single rogue truck detection
        smoothed = self.voter.update(cam_idx=0, track_id=20, raw_cls_id=3, conf=0.60)
        self.assertEqual(smoothed, 0, "Single-frame truck outlier must not flip an established car track")

    def test_legitimate_transition_allowed(self):
        """
        If a vehicle genuinely and consistently receives new class detections
        (e.g., long sequence of confident truck detections), it should transition.
        """
        # Start with 2 car detections
        self.voter.update(cam_idx=0, track_id=30, raw_cls_id=0, conf=0.50)
        self.voter.update(cam_idx=0, track_id=30, raw_cls_id=0, conf=0.50)

        # Sustained stream of 10 confident truck detections (exceeding window_size)
        final_cls = 0
        for _ in range(10):
            final_cls = self.voter.update(cam_idx=0, track_id=30, raw_cls_id=3, conf=0.95)

        self.assertEqual(final_cls, 3, "Sustained high-confidence truck detections should legitimately transition to truck")

    def test_stream_isolation(self):
        """Track IDs across different camera streams must not interfere."""
        # Stream 0: track #5 is car
        cls_cam0 = self.voter.update(cam_idx=0, track_id=5, raw_cls_id=0, conf=0.90)
        # Stream 1: track #5 is motorcycle (1)
        cls_cam1 = self.voter.update(cam_idx=1, track_id=5, raw_cls_id=1, conf=0.90)

        self.assertEqual(cls_cam0, 0)
        self.assertEqual(cls_cam1, 1)

    def test_get_class_fallback(self):
        self.assertEqual(self.voter.get_class(cam_idx=0, track_id=999, fallback=0), 0)
        self.voter.update(cam_idx=0, track_id=42, raw_cls_id=1, conf=0.9)
        self.assertEqual(self.voter.get_class(cam_idx=0, track_id=42, fallback=0), 1)

    def test_memory_pruning(self):
        # Create 100 tracks
        for tid in range(100):
            self.voter.update(cam_idx=0, track_id=tid, raw_cls_id=0, conf=0.8)

        self.assertEqual(len(self.voter.stream_tracks[0]), 100)
        # Prune with only active set [90..99]
        active = set(range(90, 100))
        self.voter.prune(cam_idx=0, active_track_ids=active)
        self.assertEqual(len(self.voter.stream_tracks[0]), len(active))

    def test_disabled_bypass(self):
        disabled_voter = TrackClassVotingFilter(enabled=False)
        self.assertEqual(disabled_voter.update(0, 1, raw_cls_id=3, conf=0.5), 3)
        self.assertEqual(disabled_voter.update(0, 1, raw_cls_id=0, conf=0.5), 0)


if __name__ == "__main__":
    unittest.main()
