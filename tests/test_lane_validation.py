"""
Unit and integration tests for Batch 2A:
Lane calibration validation, overlap detection, safe saving, and preflight startup rejection.
No GPU, cameras, or MQTT brokers required.
"""

import json
import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from shapely.geometry import Polygon

# Ensure project root is in sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.lane_validation import (
    validate_single_polygon,
    validate_camera_lanes,
    validate_config_file,
)
from tools.segmentor import (
    load_config_geometry,
    save_config_geometry,
)
from tools.validate_calibration import main as cli_main
from run_multi_camera import BatchedCameraPipeline


class TestLaneGeometryValidation(unittest.TestCase):
    def test_valid_polygon_and_repeated_closing_point(self):
        # 4 distinct points without closing
        pts_open = [[0, 0], [10, 0], [10, 10], [0, 10]]
        poly, errs = validate_single_polygon(pts_open, lane_id="N1")
        self.assertIsNotNone(poly)
        self.assertEqual(errs, [])
        self.assertEqual(poly.area, 100.0)

        # 4 distinct points with repeated closing point
        pts_closed = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        poly2, errs2 = validate_single_polygon(pts_closed, lane_id="N1")
        self.assertIsNotNone(poly2)
        self.assertEqual(errs2, [])
        self.assertEqual(poly2.area, 100.0)

    def test_too_few_distinct_vertices_zero_area_self_intersection(self):
        # Only 2 distinct points
        pts_2 = [[0, 0], [10, 0], [0, 0]]
        poly, errs = validate_single_polygon(pts_2, lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("at least 3 distinct vertices" in e for e in errs))

        # Collinear points (zero area)
        pts_collinear = [[0, 0], [5, 0], [10, 0]]
        poly, errs = validate_single_polygon(pts_collinear, lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("positive" in e.lower() for e in errs))

        # Self-intersecting bowtie
        pts_bowtie = [[0, 0], [10, 10], [0, 10], [10, 0]]
        poly, errs = validate_single_polygon(pts_bowtie, lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("Self-intersection" in e or "Invalid polygon" in e for e in errs))

    def test_malformed_nonnumeric_nan_infinity(self):
        # Not a pair
        poly, errs = validate_single_polygon([[0, 0], [10]], lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("pair of (x, y)" in e for e in errs))

        # Non-numeric string
        poly, errs = validate_single_polygon([[0, 0], ["10", 0], [10, 10]], lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("numeric" in e for e in errs))

        # Boolean coordinate (bool is subclass of int)
        poly, errs = validate_single_polygon([[0, 0], [True, 0], [10, 10]], lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("numeric" in e for e in errs))

        # NaN coordinate
        poly, errs = validate_single_polygon([[0, 0], [float("nan"), 0], [10, 10]], lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("finite" in e for e in errs))

        # Infinity coordinate
        poly, errs = validate_single_polygon([[0, 0], [float("inf"), 0], [10, 10]], lane_id="N1")
        self.assertIsNone(poly)
        self.assertTrue(any("finite" in e for e in errs))

    def test_overlap_warning_vs_shared_edge_and_vertex(self):
        # Shared edge (inter.area == 0) -> no warning
        lanes_shared_edge = {
            "L1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "L2": {"polygon": [[10, 0], [20, 0], [20, 10], [10, 10]]},
        }
        report = validate_camera_lanes(lanes_shared_edge, context="edge_test")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.warnings), 0)

        # Shared vertex (inter.area == 0) -> no warning
        lanes_shared_vertex = {
            "L1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "L2": {"polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]},
        }
        report = validate_camera_lanes(lanes_shared_vertex, context="vertex_test")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.warnings), 0)

        # Positive-area overlap (50.0 sq px) -> warning
        lanes_overlapping = {
            "L1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "L2": {"polygon": [[5, 0], [15, 0], [15, 10], [5, 10]]},
        }
        report = validate_camera_lanes(lanes_overlapping, context="overlap_test")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(report.warnings[0].overlap_area, 50.0)
        self.assertIn("L1", report.warnings[0].message)
        self.assertIn("L2", report.warnings[0].message)

    def test_small_positive_overlap_preserves_precision(self):
        # Small overlap that would be truncated to 0.0 with 2 decimals
        # L1: [0, 0] to [10, 10]
        # L2: overlaps by width 0.001, height 10 -> area = 0.01
        lanes_small = {
            "L1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "L2": {"polygon": [[9.999, 0], [20, 0], [20, 10], [9.999, 10]]},
        }
        report = validate_camera_lanes(lanes_small, context="small_overlap")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.warnings), 1)
        self.assertGreater(report.warnings[0].overlap_area, 0.0)
        # Message must not display as '0.0 sq px'
        self.assertNotIn("overlap by 0.0 sq px", report.warnings[0].message)
        self.assertNotIn("overlap by 0 sq px", report.warnings[0].message)

    def test_overlap_calculation_failure_reported_as_warning_not_swallowed(self):
        lanes = {
            "L1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            "L2": {"polygon": [[5, 0], [15, 0], [15, 10], [5, 10]]},
        }
        # Simulate an exception inside Polygon.intersection
        with patch.object(Polygon, "intersection", side_effect=RuntimeError("GEOS topology fault")):
            report = validate_camera_lanes(lanes, context="test_cam")
            self.assertTrue(report.is_valid)
            self.assertEqual(len(report.warnings), 1)
            warn = report.warnings[0]
            self.assertEqual(warn.context, "test_cam")
            self.assertEqual(warn.lane_a, "L1")
            self.assertEqual(warn.lane_b, "L2")
            self.assertIn("could not complete", warn.message)
            self.assertIn("GEOS topology fault", warn.message)

    def test_empty_lane_collection(self):
        report = validate_camera_lanes({}, context="empty_cam")
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(len(report.warnings), 0)

    def test_no_comparisons_between_different_cameras(self):
        # Two cameras with identical overlapping pixel coordinates
        cam1_lanes = {"N1": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}}
        cam2_lanes = {"S1": {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}}

        report1 = validate_camera_lanes(cam1_lanes, context="cam1")
        report2 = validate_camera_lanes(cam2_lanes, context="cam2")

        self.assertTrue(report1.is_valid)
        self.assertTrue(report2.is_valid)
        self.assertEqual(len(report1.warnings), 0)
        self.assertEqual(len(report2.warnings), 0)


class TestContainerAndStructureValidation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_root_and_metrics_container_types(self):
        # 1. Root not an object
        f_list_root = os.path.join(self.test_dir, "list_root.json")
        with open(f_list_root, "w") as f:
            json.dump([1, 2, 3], f)
        rep = validate_config_file(f_list_root)
        self.assertFalse(rep.is_valid)
        self.assertTrue(any("JSON root must be an object" in e.reason for e in rep.errors))

        # 2. lane_metrics is explicitly null (Failure 1 regression)
        f_null_metrics = os.path.join(self.test_dir, "null_metrics.json")
        with open(f_null_metrics, "w") as f:
            json.dump({"lane_metrics": None}, f)
        rep_null = validate_config_file(f_null_metrics)
        self.assertFalse(rep_null.is_valid)
        self.assertTrue(any("'lane_metrics' must be an object" in e.reason for e in rep_null.errors))

        # 3. lane_metrics is a list
        f_list_metrics = os.path.join(self.test_dir, "list_metrics.json")
        with open(f_list_metrics, "w") as f:
            json.dump({"lane_metrics": ["bad"]}, f)
        rep_list = validate_config_file(f_list_metrics)
        self.assertFalse(rep_list.is_valid)
        self.assertTrue(any("'lane_metrics' must be an object" in e.reason for e in rep_list.errors))

        # 4. lanes is explicitly null
        f_null_lanes = os.path.join(self.test_dir, "null_lanes.json")
        with open(f_null_lanes, "w") as f:
            json.dump({"lane_metrics": {"lanes": None}}, f)
        rep_null_lanes = validate_config_file(f_null_lanes)
        self.assertFalse(rep_null_lanes.is_valid)
        self.assertTrue(any("'lanes' collection must be an object" in e.reason for e in rep_null_lanes.errors))

        # 5. lanes is a list
        f_list_lanes = os.path.join(self.test_dir, "list_lanes.json")
        with open(f_list_lanes, "w") as f:
            json.dump({"lane_metrics": {"lanes": []}}, f)
        rep_list_lanes = validate_config_file(f_list_lanes)
        self.assertFalse(rep_list_lanes.is_valid)
        self.assertTrue(any("'lanes' collection must be an object" in e.reason for e in rep_list_lanes.errors))

    def test_lane_entry_structure_validation(self):
        # Lane entry is null (Failure 2 regression)
        rep_bad_null = validate_camera_lanes({"BAD": None}, context="cam_null")
        self.assertFalse(rep_bad_null.is_valid)
        self.assertTrue(any("Lane entry must be an object" in e.reason and e.lane_id == "BAD" for e in rep_bad_null.errors))

        # Lane entry is a list
        rep_bad_list = validate_camera_lanes({"BAD": [[0, 0], [10, 0], [10, 10]]}, context="cam_list")
        self.assertFalse(rep_bad_list.is_valid)
        self.assertTrue(any("Lane entry must be an object" in e.reason for e in rep_bad_list.errors))

        # Lane entry missing 'polygon' key
        rep_no_poly = validate_camera_lanes({"N1": {"direction": "N"}}, context="cam_no_poly")
        self.assertFalse(rep_no_poly.is_valid)
        self.assertTrue(any("Missing 'polygon' key" in e.reason and e.lane_id == "N1" for e in rep_no_poly.errors))

        # Missing optional lane_metrics or empty dict is valid
        rep_empty = validate_camera_lanes({}, context="cam_empty")
        self.assertTrue(rep_empty.is_valid)


class TestSafeConfigurationSaving(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_invalid_save_preserves_destination_and_backup(self):
        cfg_path = os.path.join(self.test_dir, "test_config.json")
        bak_path = f"{cfg_path}.bak"

        # Create original file
        original_data = {
            "custom_metadata": "keep_this",
            "lane_metrics": {
                "lanes": {
                    "N1": {"direction": "N", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
                }
            }
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, indent=2)

        # Create sentinel backup bytes
        sentinel_bak = b"ORIGINAL_BACKUP_BYTES_SENTINEL"
        with open(bak_path, "wb") as f:
            f.write(sentinel_bak)

        # Attempt to save invalid geometry (self-intersecting bowtie)
        invalid_lanes = {
            "N1": {"direction": "N", "polygon": [[0, 0], [10, 10], [0, 10], [10, 0]]}
        }
        success = save_config_geometry(cfg_path, invalid_lanes, gates=[])

        self.assertFalse(success)

        # Destination must remain untouched
        with open(cfg_path, "r", encoding="utf-8") as f:
            retained_data = json.load(f)
        self.assertEqual(retained_data["custom_metadata"], "keep_this")
        self.assertEqual(
            retained_data["lane_metrics"]["lanes"]["N1"]["polygon"],
            [[0, 0], [10, 0], [10, 10], [0, 10]],
        )

        # Backup bytes must remain identical
        with open(bak_path, "rb") as f:
            self.assertEqual(f.read(), sentinel_bak)

    def test_malformed_save_inputs_preserve_destination_and_backup(self):
        # Failure 3 regression: save_config_geometry(path, [], []) and other malformed inputs
        cfg_path = os.path.join(self.test_dir, "strict_save.json")
        bak_path = f"{cfg_path}.bak"

        original_content = json.dumps({"intact": True, "lane_metrics": {"lanes": {}}}).encode("utf-8")
        backup_content = b"ORIGINAL_BACKUP_SENTINEL"

        with open(cfg_path, "wb") as f:
            f.write(original_content)
        with open(bak_path, "wb") as f:
            f.write(backup_content)

        malformed_lanes = [
            [],                      # list instead of dict (Failure 3)
            None,                    # None instead of dict
            "invalid_string",        # string
            {"BAD": None},           # lane entry null
            {"BAD": []},             # lane entry list
            {"BAD": "not_a_dict"},   # lane entry string
            {"BAD": {"direction": "N"}}, # missing polygon key
            {"BAD": {"polygon": None}},  # null polygon
            {"BAD": {"polygon": "non_list"}}, # non-list polygon
        ]

        for mal_input in malformed_lanes:
            res = save_config_geometry(cfg_path, mal_input, gates=[])  # type: ignore
            self.assertFalse(res, f"Expected save_config_geometry to fail for input: {mal_input!r}")

            # Verify byte-for-byte unchanged
            with open(cfg_path, "rb") as f:
                self.assertEqual(f.read(), original_content)
            with open(bak_path, "rb") as f:
                self.assertEqual(f.read(), backup_content)

        # Also verify malformed gates (non-list)
        res_gate = save_config_geometry(cfg_path, {}, gates="not_a_list")  # type: ignore
        self.assertFalse(res_gate)
        with open(cfg_path, "rb") as f:
            self.assertEqual(f.read(), original_content)
        with open(bak_path, "rb") as f:
            self.assertEqual(f.read(), backup_content)

    def test_valid_save_preserves_unrelated_fields_and_original_backup(self):
        cfg_path = os.path.join(self.test_dir, "valid_config.json")
        bak_path = f"{cfg_path}.bak"

        original_data = {
            "camera_id": "CAM-NORTH",
            "unrelated_setting": 999,
            "lane_metrics": {
                "enabled": True,
                "lanes": {
                    "N1": {"direction": "N", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
                }
            }
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, indent=2)

        # Save new valid lanes and gates
        new_lanes = {
            "N1": {"direction": "N", "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]]},
            "N2": {"direction": "N", "polygon": [[20, 0], [40, 0], [40, 20], [20, 20]]},
        }
        new_gates = [{"gate_id": "GATE_1", "p1": [0, 0], "p2": [10, 0], "target_dir": "N"}]

        success = save_config_geometry(cfg_path, new_lanes, new_gates)
        self.assertTrue(success)

        # Backup must contain original data
        self.assertTrue(os.path.exists(bak_path))
        with open(bak_path, "r", encoding="utf-8") as f:
            bak_data = json.load(f)
        self.assertEqual(bak_data["unrelated_setting"], 999)
        self.assertEqual(len(bak_data["lane_metrics"]["lanes"]), 1)

        # Destination must contain preserved unrelated fields and new lanes/gates
        with open(cfg_path, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        self.assertEqual(updated_data["camera_id"], "CAM-NORTH")
        self.assertEqual(updated_data["unrelated_setting"], 999)
        self.assertEqual(len(updated_data["lane_metrics"]["lanes"]), 2)
        self.assertEqual(len(updated_data["gates"]), 1)

    def test_save_records_revision_and_archives_history(self):
        import numpy as np
        from tools.calibration_history import list_revisions, restore
        cfg_path = os.path.join(self.test_dir, "versioned.json")
        lanes_a = {"N1": {"direction": "N", "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]]}}
        lanes_b = {"N1": {"direction": "N", "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]]}}
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.assertTrue(save_config_geometry(cfg_path, lanes_a, [], frame=frame, operator="tester"))
        with open(cfg_path, encoding="utf-8") as f:
            first = json.load(f)["calibration"]
        self.assertEqual(first["resolution"], [64, 48])
        self.assertEqual(first["saved_by"], "tester")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, first["reference_image"])))
        import time as _t
        _t.sleep(0.01)
        self.assertTrue(save_config_geometry(cfg_path, lanes_b, [], operator="tester"))
        revs = list_revisions(cfg_path)
        self.assertEqual(len(revs), 2)
        self.assertTrue(revs[-1]["current"])
        self.assertEqual(restore(cfg_path, first["revision"]), 0)
        with open(cfg_path, encoding="utf-8") as f:
            restored = json.load(f)
        self.assertEqual(restored["lane_metrics"]["lanes"]["N1"]["polygon"][1], [20, 0])

    def test_gates_without_direction_or_with_duplicate_ids_are_not_saved(self):
        cfg_path = os.path.join(self.test_dir, "gates.json")
        lanes = {"N1": {"direction": "N", "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]]}}
        self.assertFalse(save_config_geometry(cfg_path, lanes, [{"gate_id": "G", "p1": [0, 0], "p2": [9, 0]}]))
        dup = [{"gate_id": "G", "p1": [0, 0], "p2": [9, 0], "target_dir": "N"}] * 2
        self.assertFalse(save_config_geometry(cfg_path, lanes, dup))
        self.assertFalse(os.path.exists(cfg_path))

    def test_simulated_serialization_and_write_failure_preserves_original(self):
        valid_lanes = {"N1": {"direction": "N", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}}

        # 1. Independent fixture for serialization failure
        cfg_path_ser = os.path.join(self.test_dir, "serialization_failure.json")
        bak_path_ser = f"{cfg_path_ser}.bak"
        original_bytes_ser = b'{"intact": true}'
        previous_backup_bytes = b"PREVIOUS_BACKUP_BYTES"

        with open(cfg_path_ser, "wb") as f:
            f.write(original_bytes_ser)
        with open(bak_path_ser, "wb") as f:
            f.write(previous_backup_bytes)

        with patch("json.dump", side_effect=OSError("Disk full during write")) as mock_dump:
            success_ser = save_config_geometry(cfg_path_ser, valid_lanes, gates=[])
            self.assertFalse(success_ser)
            mock_dump.assert_called()

        with open(cfg_path_ser, "rb") as f:
            self.assertEqual(f.read(), original_bytes_ser)
        with open(bak_path_ser, "rb") as f:
            self.assertEqual(f.read(), previous_backup_bytes)

        temp_files_ser = [f for f in os.listdir(self.test_dir) if f.startswith(".tmp_")]
        self.assertEqual(len(temp_files_ser), 0)

        # 2. Independent fixture for os.replace failure
        cfg_path_rep = os.path.join(self.test_dir, "replacement_failure.json")
        bak_path_rep = f"{cfg_path_rep}.bak"
        original_bytes_rep = b'{"intact": true}'

        with open(cfg_path_rep, "wb") as f:
            f.write(original_bytes_rep)

        with patch("os.replace", side_effect=OSError("Rename permission denied")) as mock_replace:
            success_rep = save_config_geometry(cfg_path_rep, valid_lanes, gates=[])
            self.assertFalse(success_rep)
            mock_replace.assert_called()

        # Destination file remains intact and unchanged
        with open(cfg_path_rep, "rb") as f:
            self.assertEqual(f.read(), original_bytes_rep)

        # Replacement step creates backup of original destination before os.replace
        self.assertTrue(os.path.exists(bak_path_rep))
        with open(bak_path_rep, "rb") as f:
            self.assertEqual(f.read(), original_bytes_rep)

        temp_files_rep = [f for f in os.listdir(self.test_dir) if f.startswith(".tmp_")]
        self.assertEqual(len(temp_files_rep), 0)

    def test_editor_loader_can_load_invalid_polygon(self):
        cfg_path = os.path.join(self.test_dir, "broken_for_repair.json")
        data = {
            "lane_metrics": {
                "lanes": {
                    "BROKEN": {"polygon": [[0, 0], [10, 10], [0, 10], [10, 0]]}
                }
            },
            "gates": []
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # load_config_geometry must load it without raising exceptions
        lanes, gates, full_cfg = load_config_geometry(cfg_path)
        self.assertIn("BROKEN", lanes)
        self.assertEqual(lanes["BROKEN"]["polygon"], [[0, 0], [10, 10], [0, 10], [10, 0]])


class TestValidationCLIAndPreflightStartup(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli_checks_multiple_files_and_returns_exit_status(self):
        valid_path = os.path.join(self.test_dir, "valid.json")
        invalid_path = os.path.join(self.test_dir, "invalid.json")

        with open(valid_path, "w", encoding="utf-8") as f:
            json.dump({"lane_metrics": {"lanes": {"N1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}}}}, f)

        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump({"lane_metrics": {"lanes": {"E1": {"polygon": [[0, 0], [10, 10], [0, 10], [10, 0]]}}}}, f)

        # Single valid file -> exit 0
        ret_valid = cli_main(["--configs", valid_path])
        self.assertEqual(ret_valid, 0)

        # Both files (one invalid) -> exit 1
        ret_mixed = cli_main(["--configs", valid_path, invalid_path])
        self.assertEqual(ret_mixed, 1)

        # Nonexistent file -> exit 1
        ret_missing = cli_main(["--configs", os.path.join(self.test_dir, "missing.json")])
        self.assertEqual(ret_missing, 1)

    def test_regression_cli_null_lane_metrics_and_mixed_inputs(self):
        # Failure 1 regression: {"lane_metrics": null} must not crash CLI
        good1 = os.path.join(self.test_dir, "good1.json")
        bad_null = os.path.join(self.test_dir, "bad_null.json")
        bad_json = os.path.join(self.test_dir, "bad_json.json")
        good2 = os.path.join(self.test_dir, "good2.json")

        with open(good1, "w") as f:
            json.dump({"lane_metrics": {"lanes": {"N1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}}}}, f)
        with open(bad_null, "w") as f:
            json.dump({"lane_metrics": None}, f)
        with open(bad_json, "w") as f:
            f.write("{ invalid json")
        with open(good2, "w") as f:
            json.dump({"lane_metrics": {"lanes": {"W1": {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}}}}, f)

        # Must check every file, report context for all without unhandled exception traceback, and return 1
        ret = cli_main(["--configs", good1, bad_null, bad_json, good2])
        self.assertEqual(ret, 1)

    def test_startup_preflight_prevents_pipeline_resource_allocation_with_spies(self):
        invalid_cfg_path = os.path.join(self.test_dir, "invalid_preflight.json")
        with open(invalid_cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lane_metrics": {
                        "lanes": {
                            "BAD_LANE": {"polygon": [[0, 0], [10, 10], [0, 10], [10, 0]]}
                        }
                    },
                    "gates": []
                },
                f,
            )

        with patch("cv2.VideoCapture") as mock_cap, \
             patch("trt_pipeline.stream.StreamBufferWorker") as mock_worker, \
             patch("ultralytics.YOLO") as mock_model, \
             patch("trt_pipeline.publisher.MQTTPublisher") as mock_pub:

            with self.assertRaises(ValueError) as ctx:
                BatchedCameraPipeline(
                    camera_names=["cam_test"],
                    config_paths=[invalid_cfg_path],
                    video_sources=["nonexistent_video.mp4"],
                    model_path="nonexistent_model.pt",
                )

            self.assertIn("BAD_LANE", str(ctx.exception))
            self.assertIn("Self-intersection", str(ctx.exception))

            # Explicit spies prove capture opening, model construction, and publisher setup were not reached
            mock_cap.assert_not_called()
            mock_worker.assert_not_called()
            mock_model.assert_not_called()
            mock_pub.assert_not_called()

    def test_regression_startup_null_lane_entry(self):
        # Failure 2 regression: {"lane_metrics": {"lanes": {"BAD": null}}}
        bad_entry_cfg = os.path.join(self.test_dir, "null_lane_entry.json")
        with open(bad_entry_cfg, "w") as f:
            json.dump({"lane_metrics": {"lanes": {"BAD": None}}, "gates": []}, f)

        with patch("ultralytics.YOLO") as mock_model, \
             patch("cv2.VideoCapture") as mock_cap:
            with self.assertRaises(ValueError) as ctx:
                BatchedCameraPipeline(
                    camera_names=["cam_null_lane"],
                    config_paths=[bad_entry_cfg],
                    video_sources=["dummy.mp4"],
                    model_path="dummy.pt",
                )
            self.assertIn("BAD", str(ctx.exception))
            self.assertIn("NoneType", str(ctx.exception))
            mock_model.assert_not_called()
            mock_cap.assert_not_called()

    def test_regression_startup_malformed_json_checks_all_configs(self):
        # Failure 4 regression: malformed JSON in camera 0 does not abort before checking camera 1
        cfg_malformed = os.path.join(self.test_dir, "malformed.json")
        cfg_invalid_geom = os.path.join(self.test_dir, "invalid_geom.json")

        with open(cfg_malformed, "w") as f:
            f.write("{ this is not valid json")

        with open(cfg_invalid_geom, "w") as f:
            json.dump({"lane_metrics": {"lanes": {"E1": {"polygon": [[0, 0], [10, 10], [0, 10], [10, 0]]}}}}, f)

        with patch("ultralytics.YOLO") as mock_model, \
             patch("cv2.VideoCapture") as mock_cap:
            with self.assertRaises(ValueError) as ctx:
                BatchedCameraPipeline(
                    camera_names=["cam_malformed", "cam_invalid_geom"],
                    config_paths=[cfg_malformed, cfg_invalid_geom],
                    video_sources=["dummy0.mp4", "dummy1.mp4"],
                    model_path="dummy.pt",
                )

            err_msg = str(ctx.exception)
            # Verify BOTH errors are present in the single raised startup report
            self.assertIn("cam_malformed", err_msg)
            self.assertIn("Invalid JSON", err_msg)
            self.assertIn("cam_invalid_geom", err_msg)
            self.assertIn("E1", err_msg)
            mock_model.assert_not_called()
            mock_cap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
