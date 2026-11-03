"""
Unit tests for Dual-Camera Smart Traffic Counting & Virtual Cart Handover System
(TRAFFIC_SYSTEM_LOGIC_FULL.md specification):
- Per-Lane Virtual Memory Carts (FIFO) & 100m Check-in
- Adaptive Per-Lane TTL & Freeze Mechanism (EC-01, EC-02)
- 4-Tier Priority FIFO Handover & Adjacent Lane Borrowing at 55m (EC-03, EC-04)
- Dual-Verification Spillback Override Mode (EC-05, EC-06)
- Motorcycle Cam 1 Only Exclusion Policy (EC-07)
- Average speed, flow rate, and Max-Pressure telemetry per lane & direction
"""

import os
import sys
import time
import unittest
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.corridor import CorridorQueueEstimator, is_motorcycle
from trt_pipeline.gates import GateFlowManager, VirtualGate
from trt_pipeline.motion import MotionStateClassifier, QueueSettings, QUEUED, MOVING


class TestVirtualCartAndSpillbackEngine(unittest.TestCase):
    def setUp(self):
        # Setup 3 lanes on North approach: N1, N2, N3
        # Upstream lanes NE1, NE2, NE3 map to N1, N2, N3
        self.lane_mappings = {"NE1": "N1", "NE2": "N2", "NE3": "N3"}
        self.estimator = CorridorQueueEstimator(
            lane_mappings=self.lane_mappings,
            max_lane_capacity=40,
            default_cap_0_100m=15,
            default_ttl_s=8.5,
            freeze_speed_kmh=8.0,
            drift_reset_duration_s=2.0,
        )
        self.estimator.register_lane("N1", "N")
        self.estimator.register_lane("N2", "N")
        self.estimator.register_lane("N3", "N")

    def test_per_lane_cars_inflow_and_outflow(self):
        """Cars entering NE1 enqueue tokens in N1 cart; arriving/exiting N1 hands over tokens."""
        # 3 cars cross 100m ingress gate in NE1
        for _ in range(3):
            self.estimator.record_gate_crossing("ingress", "N", "NE1", vehicle_class="car")
        # 2 cars cross 100m ingress gate in NE2
        for _ in range(2):
            self.estimator.record_gate_crossing("ingress", "N", "NE2", vehicle_class="car")

        self.assertEqual(len(self.estimator.lanes["N1"].cart), 3)
        self.assertEqual(len(self.estimator.lanes["N2"].cart), 2)
        self.assertEqual(self.estimator.lanes["N1"].demand_count, 3)
        self.assertEqual(self.estimator.lanes["N2"].demand_count, 2)
        self.assertEqual(self.estimator.lanes["N3"].demand_count, 0)

        # 1 car exits N1 at stopline (without prior Cam 1 track -> consumes 1 cart token)
        self.estimator.record_gate_crossing("stopline", "N", "N1", vehicle_class="car")
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 2)
        self.assertEqual(self.estimator.lanes["N1"].demand_count, 2)
        self.assertEqual(self.estimator.lanes["N1"].interval_outflow, 1)

    def test_ec01_ttl_freeze_when_stopped_at_cam1(self):
        """
        EC-01: Vehicle in blind spot (89m) while Cam 1 sees stopped queue (speed < 8 km/h).
        TTL is frozen (is_frozen = True) so tokens never expire during red light.
        """
        self.estimator.enqueue_token("N1", "car", ttl=8.5)
        self.estimator.enqueue_token("N1", "car", ttl=8.5)

        # Cam 1 sees 3 queued cars in N1 (has_stopped = True)
        visuals = {
            "N1": {"cars_queued": 3, "cars_total": 3, "motorcycles": 0, "has_stopped": True, "avg_speed_kmh": 0.0},
            "N2": {"cars_queued": 0, "cars_total": 1, "motorcycles": 0, "has_stopped": False, "avg_speed_kmh": 35.0},
        }
        # Advance 20 seconds (well past 8.5s TTL)
        self.estimator.update_visual_occupancy(visuals, now_mono=10.0, dt=10.0)
        self.estimator.update_visual_occupancy(visuals, now_mono=20.0, dt=10.0)

        n1 = self.estimator.lanes["N1"]
        self.assertTrue(n1.is_frozen)
        self.assertEqual(len(n1.cart), 2)
        self.assertAlmostEqual(n1.cart[0].ttl, 8.5)  # TTL frozen!
        self.assertAlmostEqual(n1.cart[0].age, 20.0)  # Age still increments
        # Hybrid demand = vis_cam1 (3) + vis_cam2 (0) + cart (2) = 5
        self.assertEqual(n1.demand_count, 5)

    def test_ec02_midblock_soi_turn_in_auto_expires(self):
        """
        EC-02: Car passes 100m (drops token), turns into side soi at 70m, and Cam 1 is flowing freely.
        Token TTL ticks down to <= 0 and is automatically purged.
        """
        self.estimator.enqueue_token("N1", "car", ttl=8.5)
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 1)

        visuals = {
            "N1": {"cars_queued": 0, "cars_total": 1, "motorcycles": 0, "has_stopped": False, "avg_speed_kmh": 40.0},
        }
        # Advance 5.0s -> token still alive (ttl = 3.5s)
        self.estimator.update_visual_occupancy(visuals, now_mono=5.0, dt=5.0)
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 1)
        self.assertEqual(self.estimator.lanes["N1"].demand_count, 2)  # 1 vis_cam1 + 1 token

        # Advance another 4.0s (total 9.0s > 8.5s) -> token auto-expires!
        self.estimator.update_visual_occupancy(visuals, now_mono=9.0, dt=4.0)
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 0)
        self.assertEqual(self.estimator.lanes["N1"].expired_tokens_count, 1)
        self.assertEqual(self.estimator.lanes["N1"].demand_count, 1)  # Only 1 vis_cam1 left

    def test_ec03_and_ec04_adjacent_borrowing_even_when_frozen(self):
        """
        EC-03 & EC-04: Car enters L2 (N2), N2 is frozen due to stopped car at front,
        and a trailing car in N2 switches lanes in the blind spot to L1 (N1) and arrives at 55m.
        N1 cart is empty, so N1 borrows the token from frozen N2.
        """
        # N2 has 2 tokens in cart and is frozen
        self.estimator.enqueue_token("N2", "car", ttl=8.5)
        self.estimator.enqueue_token("N2", "truck", ttl=8.5)
        self.estimator.lanes["N2"].stopped_cam1 = True
        self.estimator.lanes["N2"].is_frozen = True

        # Car arrives at 55m on N1 (where N1 cart is empty)
        borrowed = self.estimator.handover_token("N1", "car")
        self.assertIsNotNone(borrowed)
        self.assertEqual(len(self.estimator.lanes["N2"].cart), 1)
        self.assertEqual(self.estimator.lanes["N1"].handover_borrowed_count, 1)
        self.assertEqual(self.estimator.lanes["N1"].last_borrowed_from, "N2")

    def test_ec05_and_ec06_dual_verification_spillback_override(self):
        """
        EC-05: Both Cam 1 and Cam 2 detect stopped cars -> Spillback Override: MaxCap(15) + N_vis_Cam2.
        EC-06: Only Cam 2 detects stopped car while Cam 1 is flowing -> No Spillback Override!
        """
        # EC-06: Stopped at Cam 2 (NE1), flowing at Cam 1 (N1) -> False override prevented
        visuals_ec06 = {
            "N1": {"role": "queue", "cars_queued": 0, "cars_total": 2, "motorcycles": 0, "has_stopped": False, "avg_speed_kmh": 32.0},
            "NE1": {"role": "upstream", "cars_queued": 3, "cars_total": 4, "motorcycles": 0, "has_stopped": True, "avg_speed_kmh": 3.0},
        }
        self.estimator.update_visual_occupancy(visuals_ec06, now_mono=1.0, dt=0.1)
        n1 = self.estimator.lanes["N1"]
        self.assertFalse(n1.spillback)
        self.assertEqual(n1.demand_count, 6)  # 2 (Cam 1) + 4 (Cam 2) + 0 (Cart)

        # EC-05: Both Cam 1 (N1) and Cam 2 (NE1) stopped -> Spillback Override triggered!
        visuals_ec05 = {
            "N1": {"role": "queue", "cars_queued": 6, "cars_total": 6, "motorcycles": 0, "has_stopped": True, "avg_speed_kmh": 0.0},
            "NE1": {"role": "upstream", "cars_queued": 4, "cars_total": 5, "motorcycles": 0, "has_stopped": True, "avg_speed_kmh": 0.0},
        }
        self.estimator.update_visual_occupancy(visuals_ec05, now_mono=2.0, dt=1.0)
        self.assertTrue(n1.spillback)
        # Override formula: MaxCap(15) + N_vis_Cam2(5) = 20
        self.assertEqual(n1.demand_count, 20)
        tel = self.estimator.get_telemetry(interval_sec=1.0)
        self.assertTrue(tel["alarm_flags"]["emergency_spillback_override"])
        self.assertIn("N1", tel["alarm_flags"]["target_override_lanes"])

    def test_ec07_motorcycle_cam1_only_policy(self):
        """
        EC-07: Motorcycles crossing 100m at Cam 2 never enqueue tokens into Virtual Carts,
        and motorcycles at Cam 1 are counted independently as motorcycle_count_cam1 without
        consuming car tokens or inflating car demand_count.
        """
        # 5 motorcycles cross 100m ingress gate in NE1
        for _ in range(5):
            self.estimator.record_gate_crossing("ingress", "N", "NE1", vehicle_class="motorcycle")
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 0)

        # 1 car crosses 100m ingress gate in NE1
        self.estimator.record_gate_crossing("ingress", "N", "NE1", vehicle_class="car")
        self.assertEqual(len(self.estimator.lanes["N1"].cart), 1)

        # Cam 1 sees 2 cars and 4 motorcycles in N1
        visuals = {
            "N1": {"role": "queue", "cars_queued": 2, "cars_total": 2, "motorcycles": 4, "has_stopped": True, "avg_speed_kmh": 0.0},
        }
        self.estimator.update_visual_occupancy(visuals, now_mono=3.0, dt=0.1)
        n1 = self.estimator.lanes["N1"]
        # Token was NOT consumed by motorcycles; demand = 2 (vis_cam1) + 0 (vis_cam2) + 1 (cart) = 3
        self.assertEqual(len(n1.cart), 1)
        self.assertEqual(n1.demand_count, 3)
        self.assertEqual(n1.vis_motorcycles_cam1, 4)

        tel = self.estimator.get_telemetry(interval_sec=1.0)
        self.assertEqual(tel["motorcycle_count_cam1"], 4)
        self.assertEqual(tel["by_direction"]["N"]["motorcycles"]["waiting_count"], 4)

    def test_spec_section_7_reference_step_api(self):
        """Verifies the direct step(dt, cam1_cars, cam2_cars) method from Section 7 of spec."""
        cam2_cars = [
            {"lane": 0, "x_meters": 110.0, "class": "car", "speed_kmh": 40.0, "stopped_time": 0.0, "crossed_100m_trigger": True},
            {"lane": 0, "x_meters": 105.0, "class": "motorcycle", "speed_kmh": 45.0, "stopped_time": 0.0, "crossed_100m_trigger": True},
        ]
        cam1_cars = [
            {"lane": 0, "x_meters": 20.0, "class": "car", "speed_kmh": 30.0, "stopped_time": 0.0},
            {"lane": 0, "x_meters": 15.0, "class": "motorcycle", "speed_kmh": 25.0, "stopped_time": 0.0},
        ]
        res = self.estimator.step(0.1, cam1_cars, cam2_cars)
        self.assertEqual(res["n_vis_cam1"][0], 1)
        self.assertEqual(res["n_vis_cam2"][0], 1)
        self.assertEqual(res["bikes_cam1"], 1)
        self.assertEqual(res["cart_tokens"][0], 1)
        self.assertEqual(res["final_demand"][0], 3)


class TestGateLaneClassification(unittest.TestCase):
    def setUp(self):
        self.manager = GateFlowManager(camera_names=["north"], camera_ids=["CAM-01"])
        # Stopline gate spanning x=100..400 at y=200
        self.gate = VirtualGate("G_STOP", cam_idx=0, p1=(100.0, 200.0), p2=(400.0, 200.0),
                                gate_type="stopline", target_dir="N", direction_vec=(0.0, 1.0))
        self.manager.add_gate(self.gate)

        # Configure lane N1 (x: 100..250, y: 100..195) and N2 (x: 250..400, y: 100..195)
        self.poly_n1 = Polygon([(100, 100), (250, 100), (250, 195), (100, 195)])
        self.poly_n2 = Polygon([(250, 100), (400, 100), (400, 195), (250, 195)])
        self.manager.set_camera_lanes(0, {
            "N1": {"direction": "N", "polygon": self.poly_n1},
            "N2": {"direction": "N", "polygon": self.poly_n2},
        })

    def test_edge_case_2_gate_outside_polygon_lookup(self):
        """
        EDGE CASE 2: Gate is at y=200, while lane polygon ends at y=195.
        Crossing from (150, 190) to (150, 210) resolves to lane N1 via history/lateral projection.
        """
        crossed = self.manager.update_tracks(
            cam_idx=0,
            tracked_objs=np.array([[140, 180, 160, 190, 1, 0]]),
            now=1.0,
            track_lanes={1: "N1"},
        )
        self.assertEqual(len(crossed), 0)

        crossed = self.manager.update_tracks(
            cam_idx=0,
            tracked_objs=np.array([[140, 200, 160, 210, 1, 0]]),
            now=1.1,
            track_lanes={},
        )
        self.assertEqual(crossed, ["G_STOP"])
        self.assertEqual(self.gate.interval_lane_counts["N1"], 1)
        self.assertEqual(self.manager.corridor_estimator.lanes["N1"].total_outflow, 1)

    def test_edge_case_4_id_switch_cooldown_on_gate(self):
        """
        EDGE CASE 4: A vehicle crosses the gate (Track 5). In the very next frame (0.1s later),
        a new ID (Track 6) appears at the exact same location across the gate.
        Spatial-temporal cooldown prevents double-counting.
        """
        self.gate.check_crossing(5, (180.0, 190.0), (180.0, 210.0), class_id=0, now=2.0, lane_id="N1")
        self.assertEqual(self.gate.count, 1)

        res = self.gate.check_crossing(6, (182.0, 190.0), (182.0, 211.0), class_id=0, now=2.1, lane_id="N1")
        self.assertFalse(res)
        self.assertEqual(self.gate.count, 1)


class TestMotionStateHandover(unittest.TestCase):
    def test_edge_case_4_queued_state_handover_on_id_switch(self):
        """
        EDGE CASE 4: Car is queued at red light. Track ID 10 is pruned (occlusion),
        and Track 11 appears within 1.0s at the same position. Track 11 inherits QUEUED state.
        """
        classifier = MotionStateClassifier()
        for step in range(40):
            t = step * 0.1
            classifier.update(track_id=10, anchor_xy=(200.0, 300.0), box_h=40.0, t=t)
        self.assertEqual(classifier.state_of(10), QUEUED)

        st = classifier.update(track_id=11, anchor_xy=(202.0, 301.0), box_h=40.0, t=4.2)
        self.assertEqual(st, QUEUED)


class TestEndToEndPayloadIntegration(unittest.TestCase):
    def test_base_and_enriched_payload_layout(self):
        """
        Verifies end-to-end payload generation for both controller mode (base layout + enriched metrics)
        and shadow mode (canonical schema 2.0 + traffic_flow).
        """
        from tests.helpers import SQUARE, FakeModel, FakePublisher, make_pipeline, tmpdir
        import json
        import shutil

        d = tmpdir()
        try:
            expected_lanes = {
                "N1": "N", "N2": "N", "N3": "N",
                "E1": "E",
                "S1": "S", "S2": "S", "S3": "S",
                "W1": "W", "W2": "W",
            }
            profile_path = os.path.join(d, "controller_main.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump({
                    "intersection_id": "INT-001",
                    "camera_id": "CAM-01",
                    "expected_lanes": expected_lanes,
                    "max_observation_age_ms": 750,
                    "controller_freshness_ms": 2000,
                }, f)

            cameras = {
                "north": {"lanes": {f"N{i}": {"direction": "N", "polygon": SQUARE} for i in (1, 2, 3)}},
                "east": {"lanes": {"E1": {"direction": "E", "polygon": SQUARE}}},
                "south": {"lanes": {f"S{i}": {"direction": "S", "polygon": SQUARE} for i in (1, 2, 3)}},
                "west": {"lanes": {f"W{i}": {"direction": "W", "polygon": SQUARE} for i in (1, 2)}},
                "northeast": {
                    "lanes": {f"NE{i}": {"direction": "N", "target_lane": f"N{i}", "role": "upstream", "polygon": SQUARE} for i in (1, 2, 3)},
                    "gates": [{"gate_id": "GATE_N_IN", "p1": [0, 50], "p2": [100, 50], "type": "ingress", "direction": [0, 1], "target_dir": "N", "lane_id": "NE1"}],
                },
            }

            pub = FakePublisher()
            pipeline, sources = make_pipeline(
                d, cameras, publisher=pub, pub_interval=1.0,
                output_mode="controller", controller_config=profile_path,
            )
            for c in pipeline.cams:
                c.tracker.min_hits = 1

            # Drop 2 Virtual Cart tokens into N1 (via NE1 ingress)
            pipeline.gate_manager.corridor_estimator.enqueue_token("NE1", "car")
            pipeline.gate_manager.corridor_estimator.enqueue_token("NE1", "car")

            # Feed frames across all 5 cameras
            for i in range(5):
                sources[i].push(obs_t=0.1)
            pipeline.step()
            for i in range(5):
                sources[i].push(obs_t=0.2)
            pipeline.step()

            canonical = pipeline.publish_once()
            self.assertEqual(len(pub.payloads), 1)
            wire = pub.payloads[-1]

            self.assertEqual(wire["intersectionId"], "INT-001")
            self.assertEqual(wire["cameraId"], "CAM-01")
            self.assertEqual(len(wire["lanes"]), 9)

            n1_wire = next(l for l in wire["lanes"] if l["laneId"] == "N1")
            # N1 has 2 tokens in cart -> count is 2, and lane item contains only laneId, direction, count
            self.assertEqual(n1_wire, {"laneId": "N1", "direction": "N", "count": 2})
            for lane_item in wire["lanes"]:
                self.assertEqual(set(lane_item.keys()), {"laneId", "direction", "count"})
            self.assertEqual(
                wire["outgoingfull"],
                {"N": False, "E": False, "S": False, "W": False},
            )
            self.assertEqual(
                canonical["outgoingfull"],
                {"N": False, "E": False, "S": False, "W": False},
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
