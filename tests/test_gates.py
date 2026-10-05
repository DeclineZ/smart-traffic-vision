"""
Unit tests for VirtualGate and GateFlowManager:
- segment crossing, forward direction, per-track de-duplication
- explicit direction / unique IDs
- bounded continuity across short occlusions, no bridging of long gaps
- jitter suppression
- interval counts, observed coverage, "not instrumented" vs zero
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.gates import GateFlowManager, VirtualGate, segments_intersect


def gate(gid="G", cam=0, y=200.0, gtype="stopline", d="N", **kw):
    return VirtualGate(gid, cam_idx=cam, p1=(100.0, y), p2=(300.0, y), gate_type=gtype,
                       direction_vec=(0.0, 1.0), target_dir=d, **kw)


def obj(tid, x, y_bottom, h=20):
    return [x - 10, y_bottom - h, x + 10, y_bottom, tid, 0]


class TestVirtualGates(unittest.TestCase):
    def test_segment_intersection(self):
        p1, p2 = (100.0, 200.0), (300.0, 200.0)
        self.assertTrue(segments_intersect((200.0, 150.0), (200.0, 250.0), p1, p2))
        self.assertFalse(segments_intersect((50.0, 150.0), (50.0, 250.0), p1, p2))

    def test_forward_crossing_increments_count(self):
        g = gate()
        self.assertTrue(g.check_crossing(1, (200.0, 180.0), (200.0, 220.0)))
        self.assertEqual(g.count, 1)
        self.assertEqual(g.interval_count, 1)

    def test_reverse_crossing_rejected(self):
        g = gate()
        self.assertFalse(g.check_crossing(2, (200.0, 220.0), (200.0, 180.0)))
        self.assertEqual(g.count, 0)

    def test_deduplication_per_track_id(self):
        g = gate()
        self.assertTrue(g.check_crossing(5, (200.0, 180.0), (200.0, 220.0)))
        self.assertFalse(g.check_crossing(5, (200.0, 190.0), (200.0, 230.0)))
        self.assertEqual(g.count, 1)

    def test_subpixel_jitter_on_line_is_not_a_crossing(self):
        g = gate()
        self.assertFalse(g.check_crossing(1, (200.0, 199.5), (200.0, 200.5)))
        self.assertEqual(g.count, 0)

    def test_direction_and_type_are_required_and_validated(self):
        with self.assertRaises(ValueError):
            VirtualGate("G", 0, (0, 0), (10, 0))  # no target_dir
        with self.assertRaises(ValueError):
            gate(d="NE")
        with self.assertRaises(ValueError):
            gate(gtype="unknown")

    def test_crossed_ids_pruned_by_time_not_unbounded(self):
        g = gate(crossed_ttl_s=10.0)
        for tid in range(1000):
            g.check_crossing(tid, (200.0, 180.0), (200.0, 220.0), now=float(tid) * 0.01)
        g.prune_stale_tracks(set(), now=100.0)
        self.assertEqual(len(g.crossed_track_ids), 0)


class TestGateFlowManager(unittest.TestCase):
    def test_duplicate_gate_ids_rejected(self):
        m = GateFlowManager(["a", "b"])
        m.add_gate(gate("G1", cam=0))
        with self.assertRaises(ValueError):
            m.add_gate(gate("G1", cam=1))

    def test_crossing_across_short_occlusion_is_counted(self):
        m = GateFlowManager(["a"], max_gap_s=1.0)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.array([obj(7, 200, 190)]), now=0.0)
        m.update_tracks(0, np.array([obj(8, 400, 100)]), now=0.2)   # track 7 occluded, other track visible
        m.update_tracks(0, np.array([obj(7, 200, 215)]), now=0.5)   # reappears past the line
        self.assertEqual(m.gates["G1"].count, 1)

    def test_long_gap_is_not_bridged(self):
        m = GateFlowManager(["a"], max_gap_s=1.0)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.array([obj(7, 200, 190)]), now=0.0)
        m.update_tracks(0, np.array([obj(8, 400, 100)]), now=1.5)
        m.update_tracks(0, np.array([obj(7, 200, 215)]), now=3.0)
        self.assertEqual(m.gates["G1"].count, 0)

    def test_long_gap_without_intermediate_frames_is_not_bridged(self):
        # The camera itself produced no frames for 3 s (outage): no crossing across the gap.
        m = GateFlowManager(["a"], max_gap_s=1.0)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.array([obj(7, 200, 190)]), now=0.0)
        m.update_tracks(0, np.array([obj(7, 200, 215)]), now=3.0)
        self.assertEqual(m.gates["G1"].count, 0)

    def test_empty_frame_does_not_clear_recent_tracks(self):
        m = GateFlowManager(["a"], max_gap_s=1.0)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.array([obj(7, 200, 190)]), now=0.0)
        m.update_tracks(0, np.empty((0, 6)), now=0.1)
        m.update_tracks(0, np.array([obj(7, 200, 215)]), now=0.2)
        self.assertEqual(m.gates["G1"].count, 1)

    def test_reset_camera_drops_history(self):
        m = GateFlowManager(["a"], max_gap_s=1.0)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.array([obj(7, 200, 190)]), now=0.0)
        m.reset_camera(0)
        m.update_tracks(0, np.array([obj(7, 200, 215)]), now=0.1)
        self.assertEqual(m.gates["G1"].count, 0)

    def test_interval_report_counts_and_resets(self):
        m = GateFlowManager(["ne", "n"], camera_ids=["CAM-05", "CAM-01"])
        m.add_gate(gate("GATE_N_IN", cam=0, gtype="ingress", d="N"))
        m.add_gate(gate("GATE_N_STOP", cam=1, y=400.0, gtype="stopline", d="N"))
        for tid in range(1, 6):
            m.gates["GATE_N_IN"].check_crossing(tid, (250, 180), (250, 220))
        for tid in range(1, 3):
            m.gates["GATE_N_STOP"].check_crossing(tid, (250, 380), (250, 420))
        # both cameras observed continuously for the whole window
        m.interval_start_mono = time.monotonic() - 0.5
        for k in range(6):
            m._cam_observed_s[0] = m._cam_observed_s[1] = 0.5
        rep = m.interval_report()
        n = rep["by_direction"]["N"]
        self.assertEqual(n["arrivals"]["count"], 5)
        self.assertEqual(n["departures"]["count"], 2)
        self.assertTrue(n["departures"]["valid"])
        self.assertIsNone(rep["by_direction"]["S"]["arrivals"])   # not instrumented: unknown
        self.assertIsNone(rep["by_direction"]["S"]["departures"])
        cams = {g["gateId"]: g["cameraId"] for g in rep["gates"]}
        self.assertEqual(cams, {"GATE_N_IN": "CAM-05", "GATE_N_STOP": "CAM-01"})

        acc = m.get_corridor_accounting()
        self.assertEqual(acc["total_inflow"], 5)
        self.assertEqual(acc["total_stopline_cleared"], 2)

        tel = m.get_mqtt_telemetry()
        self.assertEqual(set(tel["by_direction"]), {"N"})  # uninstrumented directions are omitted, not zero
        m.reset_interval()
        rep2 = m.interval_report()
        self.assertTrue(all(g["count"] is None and not g["valid"] for g in rep2["gates"]))
        self.assertTrue(all(g.interval_count == 0 for g in m.gates.values()))
        self.assertEqual(m.gates["GATE_N_IN"].count, 5)  # session cumulative kept

    def test_unobserved_gate_is_invalid_not_zero(self):
        m = GateFlowManager(["a"], min_coverage=0.8)
        m.add_gate(gate("G1"))
        m.interval_start_mono = time.monotonic() - 2.0  # 2 s window, camera never observed
        rep = m.interval_report()
        self.assertFalse(rep["gates"][0]["valid"])
        self.assertIsNone(rep["by_direction"]["N"]["departures"]["count"])

    def test_coverage_counts_only_continuous_observation(self):
        m = GateFlowManager(["a"], max_gap_s=0.2)
        m.add_gate(gate("G1"))
        m.update_tracks(0, np.empty((0, 6)), now=0.0)
        time.sleep(0.5)
        m.update_tracks(0, np.empty((0, 6)), now=0.5)
        self.assertLessEqual(m._cam_observed_s[0], 0.2 + 1e-6)


if __name__ == "__main__":
    unittest.main()
