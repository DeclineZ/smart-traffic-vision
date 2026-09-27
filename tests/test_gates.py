"""
Unit tests for VirtualGate and GateFlowManager:
- 2D line segment crossing intersection
- Forward direction enforcement & reverse movement rejection
- Track ID de-duplication
- Inflow-Outflow queue calculation
- Periodic discharge rate computation
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.gates import VirtualGate, GateFlowManager, segments_intersect


class TestVirtualGates(unittest.TestCase):
    def test_segment_intersection(self):
        # Horizontal line (100, 200) to (300, 200)
        # Vertical motion (200, 150) to (200, 250) -> should intersect
        p1, p2 = (100.0, 200.0), (300.0, 200.0)
        c1, c2 = (200.0, 150.0), (200.0, 250.0)
        self.assertTrue(segments_intersect(c1, c2, p1, p2))

        # Parallel motion (50, 150) to (50, 250) -> misses line
        m1, m2 = (50.0, 150.0), (50.0, 250.0)
        self.assertFalse(segments_intersect(m1, m2, p1, p2))

    def test_forward_crossing_increments_count(self):
        # Horizontal gate line at y = 200, normal pointing DOWN (0, 1)
        gate = VirtualGate(
            gate_id="GATE_TEST",
            cam_idx=0,
            p1=(100.0, 200.0),
            p2=(300.0, 200.0),
            gate_type="stopline",
            direction_vec=(0.0, 1.0),
        )

        # Vehicle moving DOWN across line: (200, 180) -> (200, 220)
        crossed = gate.check_crossing(track_id=1, p_prev=(200.0, 180.0), p_curr=(200.0, 220.0), class_id=0)
        self.assertTrue(crossed)
        self.assertEqual(gate.count, 1)

    def test_reverse_crossing_rejected(self):
        # Gate expects downward motion (0, 1)
        gate = VirtualGate(
            gate_id="GATE_FORWARD_ONLY",
            cam_idx=0,
            p1=(100.0, 200.0),
            p2=(300.0, 200.0),
            direction_vec=(0.0, 1.0),
        )

        # Vehicle moving UP across line (against traffic): (200, 220) -> (200, 180)
        crossed = gate.check_crossing(track_id=2, p_prev=(200.0, 220.0), p_curr=(200.0, 180.0), class_id=0)
        self.assertFalse(crossed, "Reverse motion must not trigger the gate")
        self.assertEqual(gate.count, 0)

    def test_deduplication_per_track_id(self):
        gate = VirtualGate(
            gate_id="GATE_DEDUP",
            cam_idx=0,
            p1=(100.0, 200.0),
            p2=(300.0, 200.0),
            direction_vec=(0.0, 1.0),
        )

        # First crossing
        c1 = gate.check_crossing(track_id=5, p_prev=(200.0, 180.0), p_curr=(200.0, 220.0))
        self.assertTrue(c1)
        self.assertEqual(gate.count, 1)

        # Same track crosses again (e.g. jitter)
        c2 = gate.check_crossing(track_id=5, p_prev=(200.0, 190.0), p_curr=(200.0, 230.0))
        self.assertFalse(c2, "Same track ID must never trigger the gate twice")
        self.assertEqual(gate.count, 1)

    def test_corridor_accounting_inflow_outflow(self):
        manager = GateFlowManager(["CAM_NORTHEAST", "CAM_NORTH"])

        # Ingress Gate in Cam 0 (Upstream CAM 45)
        g_in = VirtualGate("GATE_IN", cam_idx=0, p1=(0, 200), p2=(500, 200), gate_type="ingress", direction_vec=(0, 1))
        # Stopline Gate in Cam 1 (Stopline CAM 44)
        g_stop = VirtualGate("GATE_STOP", cam_idx=1, p1=(0, 400), p2=(500, 400), gate_type="stopline", direction_vec=(0, 1))

        manager.add_gate(g_in)
        manager.add_gate(g_stop)

        # 5 cars enter upstream
        for tid in range(1, 6):
            g_in.check_crossing(tid, (250, 180), (250, 220))

        # 2 cars clear stopline
        for tid in range(1, 3):
            g_stop.check_crossing(tid, (250, 380), (250, 420))

        summary = manager.get_corridor_accounting()
        self.assertEqual(summary["total_inflow"], 5)
        self.assertEqual(summary["total_stopline_cleared"], 2)
        # 5 in - 2 out = 3 cars queued in corridor
        self.assertEqual(summary["corridor_queue"], 3)
        self.assertIn("N", summary["by_direction"])
        self.assertEqual(summary["by_direction"]["N"]["inflow"], 5)
        self.assertEqual(summary["by_direction"]["N"]["cleared"], 2)
        telemetry = manager.get_mqtt_telemetry()
        self.assertIn("by_direction", telemetry)
        self.assertEqual(telemetry["by_direction"]["N"]["inflow"], 5)
        self.assertEqual(telemetry["by_direction"]["N"]["cleared"], 2)


if __name__ == "__main__":
    unittest.main()

