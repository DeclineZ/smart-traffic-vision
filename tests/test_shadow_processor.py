"""
Unit tests for Shadow Processor, Contact Patch Refinement, and Shadow-Resilient Counting.
"""

import unittest
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon

from algorithm.shadow_processor import (
    ShadowContrastEqualizer,
    ContactPatchRefiner,
    ShadowLaneAssigner,
)
from algorithm.shadow_tracker import ShadowResilientTracker
from trt_pipeline.payload import LaneMetricsManager


class TestShadowProcessor(unittest.TestCase):

    def test_shadow_contrast_equalizer(self):
        eq = ShadowContrastEqualizer(dynamic_trigger=False)
        # Create synthetic image with a dark shadow region and a bright sunlit region
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        img[:50, :50] = 30  # deep shadow block

        enhanced, is_active = eq.enhance(img)
        self.assertTrue(is_active)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        # Contrast inside shadow should be increased (mean intensity lifted)
        orig_shadow_mean = np.mean(img[:50, :50])
        enh_shadow_mean = np.mean(enhanced[:50, :50])
        self.assertGreater(enh_shadow_mean, orig_shadow_mean)

    def test_contact_patch_refiner(self):
        refiner = ContactPatchRefiner(ground_offset_ratio=0.05)
        bbox = [100.0, 100.0, 200.0, 300.0]  # w = 100, h = 200
        cx, cy = refiner.get_contact_point(bbox)

        # Expected x = (100 + 200)/2 = 150
        self.assertEqual(cx, 150.0)
        # Expected y = 300 - 0.05 * 200 = 290.0
        self.assertEqual(cy, 290.0)

        # Vectorized test
        bboxes = np.array([
            [100.0, 100.0, 200.0, 300.0],
            [50.0, 50.0, 150.0, 250.0],
        ])
        xs, ys = refiner.get_contact_points_vectorized(bboxes)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0], 150.0)
        self.assertEqual(ys[0], 290.0)
        self.assertEqual(xs[1], 100.0)
        self.assertEqual(ys[1], 240.0)

    def test_lateral_shadow_trimming(self):
        refiner = ContactPatchRefiner(ground_offset_ratio=0.05, trim_shadow_wings=True)
        # Create a test frame with vehicle tires (high gradient) and lateral shadow (flat dark)
        frame_gray = np.full((200, 200), 180, dtype=np.uint8)
        # Vehicle chassis: [50, 50, 150, 150]
        # In bottom strip (y=130 to 150):
        # x=50 to 70: flat dark shadow wing (no edges)
        frame_gray[130:150, 50:70] = 30
        # x=70 to 150: tire pattern with strong alternating edges
        for x in range(70, 150, 4):
            frame_gray[130:150, x:x+2] = 20
            frame_gray[130:150, x+2:x+4] = 100

        box_with_shadow = [50.0, 50.0, 150.0, 150.0]
        trimmed = refiner.trim_lateral_cast_shadow(box_with_shadow, frame_gray)
        # The left shadow wing should be trimmed inward
        self.assertGreater(trimmed[0], box_with_shadow[0])

    def test_shadow_lane_assigner_hysteresis(self):
        assigner = ShadowLaneAssigner(hysteresis_frames=3)
        track_id = 101

        # Frame 1: Vehicle appears in Lane 1
        l1 = assigner.update_track_lane(track_id, "L1", frame_idx=1)
        self.assertIsNone(l1)  # Needs 3 consecutive frames

        # Frame 2: Still in Lane 1
        l2 = assigner.update_track_lane(track_id, "L1", frame_idx=2)
        self.assertIsNone(l2)

        # Frame 3: Confirmed in Lane 1
        l3 = assigner.update_track_lane(track_id, "L1", frame_idx=3)
        self.assertEqual(l3, "L1")

        # Frame 4: Shadow flickers vehicle into Lane 2 for 1 frame
        l4 = assigner.update_track_lane(track_id, "L2", frame_idx=4)
        # Should stay stably locked to L1!
        self.assertEqual(l4, "L1")

        # Frame 5: Flickers back to L1
        l5 = assigner.update_track_lane(track_id, "L1", frame_idx=5)
        self.assertEqual(l5, "L1")

        # Frame 6, 7, 8: Genuine lane change to Lane 2 for 3 consecutive frames
        assigner.update_track_lane(track_id, "L2", frame_idx=6)
        assigner.update_track_lane(track_id, "L2", frame_idx=7)
        l8 = assigner.update_track_lane(track_id, "L2", frame_idx=8)
        self.assertEqual(l8, "L2")

    def test_lane_exclusivity_prevents_double_counting(self):
        lane_cfg = {
            "L1": {"direction": "N", "polygon": Polygon([(0, 0), (100, 0), (100, 500), (0, 500)])},
            "L2": {"direction": "N", "polygon": Polygon([(100, 0), (200, 0), (200, 500), (100, 500)])},
        }
        manager = LaneMetricsManager(lane_cfg)

        track_id = 55
        # Vehicle initially registered in L1
        manager.register_vehicle(lane_id="L1", track_id=track_id, vehicle_class=2, is_queued=False)

        snap1 = {item["laneId"]: item["vehicles"]["moving"]["cars"] for item in manager.snapshot()}
        self.assertEqual(snap1["L1"], 1)
        self.assertEqual(snap1["L2"], 0)

        # Shadow flickers into L2 in the same interval
        manager.register_vehicle(lane_id="L2", track_id=track_id, vehicle_class=2, is_queued=False)

        # Snapshot should strictly enforce exclusivity: 1 vehicle in L2, 0 in L1
        snap2 = {item["laneId"]: item["vehicles"]["moving"]["cars"] for item in manager.snapshot()}
        self.assertEqual(snap2["L1"], 0)
        self.assertEqual(snap2["L2"], 1)
        # Total vehicle count across both lanes MUST be exactly 1 (0 duplicate counts!)
        total_cars = sum(snap2.values())
        self.assertEqual(total_cars, 1)

    def test_shadow_tracker_byte_recovery_and_coasting(self):
        tracker = ShadowResilientTracker(
            det_thresh=0.40,
            min_conf=0.15,
            max_age=15,
            max_coast_frames=5,
            min_hits=1,
        )

        # Frame 1: High confidence detection in sunlight
        det1 = np.array([[100.0, 100.0, 150.0, 180.0, 0.85, 2]])
        res1 = tracker.update(det1)
        self.assertEqual(len(res1), 1)
        initial_id = res1[0][4]

        # Frame 2: Vehicle enters harsh shadow, confidence dips to 0.25 (below det_thresh=0.40)
        det2 = np.array([[102.0, 105.0, 152.0, 185.0, 0.25, 2]])
        res2 = tracker.update(det2)
        # ByteTrack Tier 2 should recover this low-conf detection with the same track ID!
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0][4], initial_id)

        # Frame 3: Vehicle passes under overpass shadow band (completely undetected for 2 frames)
        res3 = tracker.update(np.empty((0, 6)))
        # Coasting mode keeps the track alive
        self.assertEqual(len(res3), 1)
        self.assertEqual(res3[0][4], initial_id)

        # Frame 4: Second coasting frame
        res4 = tracker.update(np.empty((0, 6)))
        self.assertEqual(len(res4), 1)
        self.assertEqual(res4[0][4], initial_id)

        # Frame 5: Vehicle emerges from underpass into light (high conf detection)
        det5 = np.array([[110.0, 120.0, 160.0, 200.0, 0.88, 2]])
        res5 = tracker.update(det5)
        self.assertEqual(len(res5), 1)
        # Same track ID maintained! No ID switch!
        self.assertEqual(res5[0][4], initial_id)


if __name__ == "__main__":
    unittest.main()
