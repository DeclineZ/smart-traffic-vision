"""
Unit tests for trt_pipeline.payload:
- Vehicle classification
- LaneMetricsManager state aggregation
- PayloadBuilder controller compatibility
"""

import os
import sys
import unittest
from datetime import datetime

# Add project root to sys.path for test discovery
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.payload import classify_vehicle, LaneMetricsManager, PayloadBuilder
from shapely.geometry import Polygon


class TestVehicleClassification(unittest.TestCase):
    def test_class_ids(self):
        # 1: bicycle, 3: motorcycle -> motorbike
        self.assertEqual(classify_vehicle(1), "motorbike")
        self.assertEqual(classify_vehicle(3), "motorbike")
        # 2: car, 5: bus, 7: truck -> cars
        self.assertEqual(classify_vehicle(2), "cars")
        self.assertEqual(classify_vehicle(5), "cars")
        self.assertEqual(classify_vehicle(7), "cars")

    def test_class_names(self):
        self.assertEqual(classify_vehicle("motorcycle"), "motorbike")
        self.assertEqual(classify_vehicle("motorbike"), "motorbike")
        self.assertEqual(classify_vehicle("bicycle"), "motorbike")
        self.assertEqual(classify_vehicle("bike"), "motorbike")

        self.assertEqual(classify_vehicle("car"), "cars")
        self.assertEqual(classify_vehicle("truck"), "cars")
        self.assertEqual(classify_vehicle("bus"), "cars")
        self.assertEqual(classify_vehicle("van"), "cars")
        self.assertEqual(classify_vehicle("three_wheeler"), "cars")
        self.assertEqual(classify_vehicle("tuktuk"), "cars")

    def test_custom_class_mapping(self):
        thai_classes = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "three_wheeler"}
        self.assertEqual(classify_vehicle(0, thai_classes), "cars")
        self.assertEqual(classify_vehicle(1, thai_classes), "motorbike")
        self.assertEqual(classify_vehicle(2, thai_classes), "cars")
        self.assertEqual(classify_vehicle(3, thai_classes), "cars")
        self.assertEqual(classify_vehicle(4, thai_classes), "cars")


class TestLaneMetricsManager(unittest.TestCase):
    def setUp(self):
        self.lane_config = {
            "N1": {
                "direction": "N",
                "polygon": Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            },
            "S1": {
                "direction": "S",
                "polygon": Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            },
        }
        self.manager = LaneMetricsManager(self.lane_config)

    def test_initial_snapshot(self):
        snapshot = self.manager.snapshot()
        self.assertIsInstance(snapshot, list)
        self.assertEqual(len(snapshot), 2)

        n1 = next(l for l in snapshot if l["laneId"] == "N1")
        self.assertEqual(n1["direction"], "N")
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 0)
        self.assertEqual(n1["vehicles"]["queued"]["motorbike"], 0)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 0)
        self.assertEqual(n1["vehicles"]["moving"]["motorbike"], 0)

    def test_register_vehicles(self):
        # Register 2 queued cars, 1 queued motorbike, 1 moving car in N1
        self.manager.register_vehicle("N1", track_id=101, vehicle_class=2, is_queued=True)
        self.manager.register_vehicle("N1", track_id=102, vehicle_class="car", is_queued=True)
        self.manager.register_vehicle("N1", track_id=103, vehicle_class=3, is_queued=True)
        self.manager.register_vehicle("N1", track_id=104, vehicle_class=2, is_queued=False)

        snapshot = self.manager.snapshot()
        n1 = next(l for l in snapshot if l["laneId"] == "N1")
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 2)
        self.assertEqual(n1["vehicles"]["queued"]["motorbike"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["motorbike"], 0)

    def test_state_transition(self):
        # Vehicle initially registered as queued then starts moving
        self.manager.register_vehicle("N1", track_id=201, vehicle_class="car", is_queued=True)
        self.manager.register_vehicle("N1", track_id=201, vehicle_class="car", is_queued=False)

        snapshot = self.manager.snapshot()
        n1 = next(l for l in snapshot if l["laneId"] == "N1")
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 0)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 1)

    def test_reset(self):
        self.manager.register_vehicle("N1", track_id=301, vehicle_class="car", is_queued=True)
        self.manager.reset()
        snapshot = self.manager.snapshot()
        n1 = next(l for l in snapshot if l["laneId"] == "N1")
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 0)

    def test_category_change_both_directions_counts_once(self):
        # Track 1 starts as car
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["motorbike"], 0)

        # Changes to motorcycle
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="motorcycle", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 0)
        self.assertEqual(n1["vehicles"]["moving"]["motorbike"], 1)

        # Changes back to car
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["motorbike"], 0)

    def test_category_and_state_change_together(self):
        # Track 1 starts as moving car
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["movingCount"], 1)
        self.assertEqual(n1["queuedCount"], 0)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 1)

        # Changes to queued motorbike
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="motorcycle", is_queued=True)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["movingCount"], 0)
        self.assertEqual(n1["queuedCount"], 1)
        self.assertEqual(n1["vehicles"]["queued"]["motorbike"], 1)
        self.assertEqual(n1["vehicles"]["moving"]["cars"], 0)

    def test_repeated_registration_idempotent(self):
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["queuedCount"], 1)
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 1)

    def test_lane_transfer_latest_only(self):
        # Register in N1
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        s1 = next(l for l in snap if l["laneId"] == "S1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(s1["count"], 0)

        # Transfer to S1
        self.manager.register_vehicle("S1", track_id=1, vehicle_class="car", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        s1 = next(l for l in snap if l["laneId"] == "S1")
        self.assertEqual(n1["count"], 0)
        self.assertEqual(s1["count"], 1)
        self.assertEqual(sum(l["count"] for l in snap), 1)

    def test_unknown_lane_leaves_valid_state_unchanged(self):
        self.manager.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        # Attempt to register in nonexistent lane
        self.manager.register_vehicle("NONEXISTENT", track_id=1, vehicle_class="motorcycle", is_queued=False)
        snap = self.manager.snapshot()
        n1 = next(l for l in snap if l["laneId"] == "N1")
        self.assertEqual(n1["count"], 1)
        self.assertEqual(n1["queuedCount"], 1)
        self.assertEqual(n1["vehicles"]["queued"]["cars"], 1)
        self.assertNotIn("NONEXISTENT", [l["laneId"] for l in snap])

        # Attempt to register a brand new track in nonexistent lane
        self.manager.register_vehicle("NONEXISTENT", track_id=999, vehicle_class="car", is_queued=False)
        snap = self.manager.snapshot()
        self.assertEqual(sum(l["count"] for l in snap), 1)

    def test_separate_cameras_independent_track_ids(self):
        cam0_mgr = LaneMetricsManager({"N1": {"direction": "N", "polygon": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}})
        cam1_mgr = LaneMetricsManager({"S1": {"direction": "S", "polygon": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}})

        cam0_mgr.register_vehicle("N1", track_id=42, vehicle_class="car", is_queued=True)
        cam1_mgr.register_vehicle("S1", track_id=42, vehicle_class="motorcycle", is_queued=False)

        snap0 = cam0_mgr.snapshot()
        snap1 = cam1_mgr.snapshot()

        self.assertEqual(snap0[0]["count"], 1)
        self.assertEqual(snap0[0]["vehicles"]["queued"]["cars"], 1)
        self.assertEqual(snap1[0]["count"], 1)
        self.assertEqual(snap1[0]["vehicles"]["moving"]["motorbike"], 1)

    def test_repeated_snapshots_do_not_mutate(self):
        self.manager.register_vehicle("N1", track_id=10, vehicle_class="car", is_queued=True)
        self.manager.register_vehicle("S1", track_id=20, vehicle_class="motorcycle", is_queued=False)
        snap1 = self.manager.snapshot()
        snap2 = self.manager.snapshot()
        self.assertEqual(snap1, snap2)

    def test_all_configured_lanes_present_and_satisfy_invariants(self):
        config = {
            "N1": {"direction": "N", "polygon": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
            "N2": {"direction": "N", "polygon": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])},
            "S1": {"direction": "S", "polygon": Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])},
        }
        mgr = LaneMetricsManager(config)
        mgr.register_vehicle("N1", track_id=1, vehicle_class="car", is_queued=True)
        mgr.register_vehicle("N1", track_id=2, vehicle_class="motorcycle", is_queued=False)

        snapshot = mgr.snapshot()
        self.assertEqual(len(snapshot), 3)
        lane_ids = {l["laneId"] for l in snapshot}
        self.assertEqual(lane_ids, {"N1", "N2", "S1"})

        for lane in snapshot:
            q_cars = lane["vehicles"]["queued"]["cars"]
            q_bikes = lane["vehicles"]["queued"]["motorbike"]
            m_cars = lane["vehicles"]["moving"]["cars"]
            m_bikes = lane["vehicles"]["moving"]["motorbike"]

            self.assertEqual(lane["queuedCount"], q_cars + q_bikes)
            self.assertEqual(lane["movingCount"], m_cars + m_bikes)
            self.assertEqual(lane["count"], lane["queuedCount"] + lane["movingCount"])

        # Also test with PayloadBuilder
        builder = PayloadBuilder("INT-001", "CAM-01")
        payload = builder.build(1, snapshot)
        self.assertEqual(len(payload["lanes"]), 3)


class TestPayloadBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PayloadBuilder(intersection_id="INT-001", camera_id="CAM-01")

    def test_payload_schema_controller_compatibility(self):
        lanes_data = [
            {
                "laneId": "N1",
                "direction": "N",
                "vehicles": {
                    "queued": {"cars": 3, "motorbike": 1},
                    "moving": {"cars": 1, "motorbike": 0},
                },
            },
            {
                "laneId": "S1",
                "direction": "S",
                "vehicles": {
                    "queued": {"cars": 0, "motorbike": 0},
                    "moving": {"cars": 2, "motorbike": 1},
                },
            },
        ]

        payload = self.builder.build(frame_idx=100, lanes_snapshot=lanes_data, meta={"fps": 25.0})

        # Test controller validation requirements:
        # 1. intersectionId
        self.assertIn("intersectionId", payload)
        self.assertEqual(payload["intersectionId"], "INT-001")

        # 2. cameraId
        self.assertIn("cameraId", payload)
        self.assertEqual(payload["cameraId"], "CAM-01")

        # 3. timestamp (valid ISO date parseable by Date.parse in JS)
        self.assertIn("timestamp", payload)
        # Should parse with datetime.fromisoformat
        parsed_ts = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
        self.assertIsNotNone(parsed_ts)

        # 4. meta with frameId
        self.assertIn("meta", payload)
        self.assertEqual(payload["meta"]["frameId"], "frame_100")
        self.assertEqual(payload["meta"]["fps"], 25.0)

        # 5. lanes (must be array/list)
        self.assertIn("lanes", payload)
        self.assertIsInstance(payload["lanes"], list)
        self.assertEqual(len(payload["lanes"]), 2)

        # 6. lane vehicle structure
        lane_n1 = payload["lanes"][0]
        self.assertEqual(lane_n1["laneId"], "N1")
        self.assertEqual(lane_n1["direction"], "N")
        self.assertIn("queued", lane_n1["vehicles"])
        self.assertIn("moving", lane_n1["vehicles"])
        self.assertEqual(lane_n1["vehicles"]["queued"]["cars"], 3)
        self.assertEqual(lane_n1["vehicles"]["queued"]["motorbike"], 1)


if __name__ == "__main__":
    unittest.main()
