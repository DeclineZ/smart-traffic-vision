"""
Unit tests for Batch 1: Correct current lane occupancy and prevent duplicate vehicle counting.
Tests runner-level and evaluation-level behaviors with synthetic inputs.
No GPU, cameras, or live MQTT brokers required.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
from shapely.geometry import Polygon

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_multi_camera import BatchedCameraPipeline
from trt_pipeline.payload import LaneMetricsManager, PayloadBuilder
from trt_pipeline.gates import GateFlowManager


class TestLaneOccupancyRunner(unittest.TestCase):
    def setUp(self):
        self.pipeline = BatchedCameraPipeline.__new__(BatchedCameraPipeline)
        self.pipeline.num_streams = 1
        self.pipeline.camera_names = ["cam_north"]
        self.pipeline.skip_frames = 1
        self.pipeline.is_file_mode = False
        self.pipeline.default_car_cls = 0
        self.pipeline.class_names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "three_wheeler"}
        self.pipeline.queue_speed_thresholds = [2.0]
        self.pipeline.track_histories = [{}]

        # Standard polygon: [0, 0] to [100, 100]
        self.lanes = {
            "N1": {
                "direction": "N",
                "polygon": Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            }
        }
        self.pipeline.lane_configs = [self.lanes]
        self.pipeline.metrics_managers = [LaneMetricsManager(self.lanes)]
        self.pipeline.last_tracked = [np.empty((0, 6))]
        self.pipeline.gate_manager = GateFlowManager(camera_names=["cam_north"])
        self.pipeline.publisher = MagicMock()
        self.pipeline.payload_builder = PayloadBuilder("INT-001", "MULTI-CAM")
        self.pipeline.pub_interval = 2.0

    def test_nonempty_then_empty_evaluation_clears_occupancy(self):
        # Frame 0: 1 car in N1
        tracked_objs = np.array([[10, 10, 30, 30, 101, 0]], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs, frame_idx=0)
        snap0 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap0[0]["count"], 1)

        # Frame 1: Empty frame (0 tracks)
        empty_tracked = np.empty((0, 6), dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, empty_tracked, frame_idx=1)
        snap1 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap1[0]["count"], 0)
        self.assertEqual(snap1[0]["queuedCount"], 0)
        self.assertEqual(snap1[0]["movingCount"], 0)
        self.assertEqual(snap1[0]["vehicles"]["queued"]["cars"], 0)
        self.assertEqual(snap1[0]["vehicles"]["moving"]["cars"], 0)

        # Frame 2: Non-empty again, then None frame
        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs, frame_idx=2)
        self.assertEqual(self.pipeline.metrics_managers[0].snapshot()[0]["count"], 1)
        self.pipeline._evaluate_vectorized_lanes(0, None, frame_idx=3)
        snap3 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap3[0]["count"], 0)

    def test_track_moves_outside_polygons_disappears(self):
        # Frame 0: track inside polygon (centroid at 20, 20)
        tracked_objs_inside = np.array([[10, 10, 30, 30, 101, 0]], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs_inside, frame_idx=0)
        self.assertEqual(self.pipeline.metrics_managers[0].snapshot()[0]["count"], 1)

        # Frame 1: track moves outside polygon (centroid at 510, 510)
        tracked_objs_outside = np.array([[500, 500, 520, 520, 101, 0]], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs_outside, frame_idx=1)
        snap1 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap1[0]["count"], 0)

    def test_successive_evaluations_different_ids(self):
        # Frame 0: IDs 101 and 102
        frame0_tracks = np.array([
            [10, 10, 30, 30, 101, 0],
            [40, 40, 60, 60, 102, 0],
        ], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, frame0_tracks, frame_idx=0)
        snap0 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap0[0]["count"], 2)

        # Frame 1: IDs 201 and 202 (completely new IDs)
        frame1_tracks = np.array([
            [20, 20, 40, 40, 201, 0],
            [50, 50, 70, 70, 202, 0],
        ], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, frame1_tracks, frame_idx=1)
        snap1 = self.pipeline.metrics_managers[0].snapshot()
        self.assertEqual(snap1[0]["count"], 2)

        vehicles = self.pipeline.metrics_managers[0].lanes["N1"]["vehicles"]
        active_ids = vehicles["queued"]["cars"] | vehicles["moving"]["cars"]
        self.assertEqual(active_ids, {201, 202})
        self.assertNotIn(101, active_ids)
        self.assertNotIn(102, active_ids)

    def test_overlapping_polygons_first_lane_precedence_and_single_is_queued(self):
        overlap_lanes = {
            "LaneA": {
                "direction": "N",
                "polygon": Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            },
            "LaneB": {
                "direction": "N",
                "polygon": Polygon([(50, 0), (150, 0), (150, 100), (50, 100)]),
            },
        }
        self.pipeline.lane_configs = [overlap_lanes]
        self.pipeline.metrics_managers = [LaneMetricsManager(overlap_lanes)]

        # Track centroid at (70, 50), which is inside LaneA (0..100) AND LaneB (50..150)
        tracked_objs = np.array([[60, 40, 80, 60, 55, 0]], dtype=float)

        is_queued_calls = []
        original_is_queued = self.pipeline._is_queued

        def spy_is_queued(cam_idx, track_id, pt, frame_idx):
            is_queued_calls.append((cam_idx, track_id, pt, frame_idx))
            return original_is_queued(cam_idx, track_id, pt, frame_idx)

        self.pipeline._is_queued = spy_is_queued

        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs, frame_idx=0)
        snap = self.pipeline.metrics_managers[0].snapshot()

        lane_a = next(l for l in snap if l["laneId"] == "LaneA")
        lane_b = next(l for l in snap if l["laneId"] == "LaneB")

        # Must be assigned to LaneA (first in config order), not LaneB
        self.assertEqual(lane_a["count"], 1)
        self.assertEqual(lane_b["count"], 0)
        self.assertEqual(sum(l["count"] for l in snap), 1)

        # _is_queued must be called exactly once
        self.assertEqual(len(is_queued_calls), 1)
        self.assertEqual(is_queued_calls[0][1], 55)

    def test_repeated_held_boxes_do_not_accumulate(self):
        # Frame 0: inference frame
        tracked_objs = np.array([[10, 10, 30, 30, 10, 0]], dtype=float)
        self.pipeline._evaluate_vectorized_lanes(0, tracked_objs, frame_idx=0)
        self.assertEqual(self.pipeline.metrics_managers[0].snapshot()[0]["count"], 1)

        # Frames 1-3: skipped frames holding the same tracked_objs
        for f in range(1, 4):
            self.pipeline._evaluate_vectorized_lanes(0, tracked_objs, frame_idx=f)
            snap = self.pipeline.metrics_managers[0].snapshot()
            self.assertEqual(snap[0]["count"], 1)

    def test_publication_does_not_consume_occupancy(self):
        """
        Executes BatchedCameraPipeline.run() through its actual production publication branch.
        Stubs external capture, model/tracker outputs, and publisher while keeping real
        lane evaluation, metrics aggregation, and payload construction.
        Stops deterministically immediately on publication, before subsequent frames can run,
        verifying that publication does not reset or consume evaluated lane occupancy.
        """
        self.pipeline.skip_frames = 0
        self.pipeline.is_file_mode = False
        self.pipeline.target_classes = None
        self.pipeline.device = "cpu"
        self.pipeline.conf = 0.20
        self.pipeline.imgsz = 640
        self.pipeline.total_inferred_batches = 0
        self.pipeline.total_skipped_batches = 0
        self.pipeline.total_processed_batches = 0
        self.pipeline.tracker_type = "byetrack"

        # Pub interval 0.0 ensures publication branch executes on the first batch
        self.pipeline.pub_interval = 0.0

        # Stub stream worker supplying dummy frames
        worker = MagicMock()
        worker.get_frame.return_value = (0, np.zeros((100, 100, 3), dtype=np.uint8))
        self.pipeline.stream_workers = [worker]
        self.pipeline.file_caps = []
        self.pipeline.display_worker = None

        # Stub model forward pass to return empty boxes (avoiding torch/GPU dependency)
        self.pipeline.model = MagicMock(return_value=[MagicMock(boxes=[])])

        # Stub tracker to return 1 vehicle inside lane N1 (centroid at 20, 20), ID 101, class 0 (car)
        tracker = MagicMock()
        tracker.update.return_value = np.array([[10.0, 10.0, 30.0, 30.0, 101, 0]], dtype=float)
        self.pipeline.trackers = [tracker]
        from trt_pipeline.voter import TrackClassVotingFilter
        self.pipeline.class_voter = TrackClassVotingFilter(num_streams=1)

        # Publisher spy that captures the payload and stops the pipeline immediately
        published_payloads = []

        def spy_publish(payload):
            published_payloads.append(payload)
            # Stop immediately on publication, before another lane evaluation can run
            self.pipeline.running = False

        self.pipeline.publisher = MagicMock()
        self.pipeline.publisher.publish.side_effect = spy_publish

        # Execute production run() loop
        self.pipeline.run()

        # Verify publication branch executed exactly once
        self.assertEqual(len(published_payloads), 1)
        pub_lanes = published_payloads[0]["lanes"]
        n1_pub = next(l for l in pub_lanes if l["laneId"] == "N1")
        self.assertEqual(n1_pub["count"], 1)
        self.assertEqual(n1_pub["movingCount"], 1)
        self.assertEqual(n1_pub["vehicles"]["moving"]["cars"], 1)

        # Verify that the metrics manager retained occupancy and was NOT reset by publication
        retained_snapshot = self.pipeline.metrics_managers[0].snapshot()
        n1_retained = next(l for l in retained_snapshot if l["laneId"] == "N1")
        self.assertEqual(n1_retained["count"], 1)
        self.assertEqual(n1_retained["movingCount"], 1)
        self.assertEqual(n1_retained["vehicles"]["moving"]["cars"], 1)
        self.assertEqual(pub_lanes, retained_snapshot)


if __name__ == "__main__":
    unittest.main()
