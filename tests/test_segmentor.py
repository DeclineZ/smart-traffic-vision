"""
Unit tests for the Interactive Segmentor & Calibrator Tool:
- Camera preset resolution
- Normal vector computation & flipping
- Point-to-line Euclidean distance
- Geometry loading and saving with automatic .bak backup
- Multi-exit gate configuration persistence
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.segmentor import (
    CAMERA_PRESETS,
    compute_gate_normal,
    point_to_line_dist,
    load_config_geometry,
    save_config_geometry,
)


class TestSegmentor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_presets_completeness(self):
        for direction in ["north", "south", "east", "west", "northeast"]:
            self.assertIn(direction, CAMERA_PRESETS)
            preset = CAMERA_PRESETS[direction]
            self.assertIn("video", preset)
            self.assertIn("config", preset)
            self.assertIn("lane_prefix", preset)
            self.assertIn("suggested_gates", preset)

    def test_gate_normal_computation(self):
        # Horizontal line from (100, 200) to (300, 200)
        # dx = 200, dy = 0
        # Clockwise perpendicular (-dy, dx) -> (0, 1) [pointing down]
        n_std = compute_gate_normal((100, 200), (300, 200), flip=False)
        self.assertAlmostEqual(n_std[0], 0.0, places=2)
        self.assertAlmostEqual(n_std[1], 1.0, places=2)

        # Flipped perpendicular -> (0, -1) [pointing up]
        n_flip = compute_gate_normal((100, 200), (300, 200), flip=True)
        self.assertAlmostEqual(n_flip[0], 0.0, places=2)
        self.assertAlmostEqual(n_flip[1], -1.0, places=2)

    def test_point_to_line_distance(self):
        p1 = (100.0, 200.0)
        p2 = (300.0, 200.0)

        # Point directly 15px above midpoint
        p_above = (200.0, 185.0)
        self.assertAlmostEqual(point_to_line_dist(p_above, p1, p2), 15.0, places=3)

        # Point collinear beyond segment endpoint
        p_far = (350.0, 200.0)
        self.assertAlmostEqual(point_to_line_dist(p_far, p1, p2), 50.0, places=3)

    def test_config_geometry_save_and_load(self):
        config_path = os.path.join(self.test_dir, "config_test_south.json")

        # Initial dummy config
        initial_data = {
            "camera_info": {"camera_id": "CAM-02", "location": "South"},
            "model": {"engine_path": "models/yolo.engine"},
            "lane_metrics": {
                "enabled": True,
                "lanes": {
                    "S1": {"direction": "S", "polygon": [[100, 200], [200, 200], [200, 300], [100, 300]]}
                }
            },
            "gates": []
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)

        # Update geometry with multi-exit setup
        new_lanes = {
            "S1": {"direction": "S", "polygon": [[100, 200], [200, 200], [200, 300], [100, 300]]},
            "S2": {"direction": "S", "polygon": [[200, 200], [300, 200], [300, 300], [200, 300]]},
        }
        new_gates = [
            {
                "gate_id": "GATE_S_STOPLINE",
                "p1": [850, 400],
                "p2": [1600, 500],
                "type": "stopline",
                "direction": [0.0, 1.0],
                "label": "STOP_S",
                "target_dir": "S"
            },
            {
                "gate_id": "GATE_S_EXIT",
                "p1": [100, 450],
                "p2": [750, 380],
                "type": "egress",
                "direction": [0.0, -1.0],
                "label": "EXIT_S",
                "target_dir": "S"
            },
            {
                "gate_id": "GATE_E_EXIT",
                "p1": [1500, 350],
                "p2": [1850, 600],
                "type": "egress",
                "direction": [1.0, 0.0],
                "label": "EXIT_E",
                "target_dir": "E"
            }
        ]

        saved = save_config_geometry(config_path, new_lanes, new_gates)
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(f"{config_path}.bak"))

        # Load back
        loaded_lanes, loaded_gates, loaded_cfg = load_config_geometry(config_path)
        self.assertEqual(len(loaded_lanes), 2)
        self.assertIn("S2", loaded_lanes)
        self.assertEqual(len(loaded_gates), 3)
        self.assertEqual(loaded_gates[2]["gate_id"], "GATE_E_EXIT")
        # Ensure original fields are preserved
        self.assertEqual(loaded_cfg["camera_info"]["camera_id"], "CAM-02")


if __name__ == "__main__":
    unittest.main()
