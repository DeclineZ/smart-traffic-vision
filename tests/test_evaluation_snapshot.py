"""
Unit Tests for Review Pack Validation, Snapshot Creation, and Baseline Evaluation (Batch 3).

Covers:
1. Annotation vs YOLO label disagreement detection (count, class, coordinate mismatch).
2. Duplicate canonical source ID detection.
3. Distinction between verified empty frames and unannotated frames.
4. Strict invalid box rejection (negative dims, coords out of bounds, invalid classes, subtype conflict).
5. One-to-one bipartite matching logic (TP, FP, FN, class confusion, localization error).
6. Evaluation snapshot creation and source pack immutability.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.evaluate_baseline import match_one_to_one
from tools.prepare_review_pack import compute_file_sha256
from tools.validate_review_pack import (
    create_evaluation_snapshot,
    validate_review_pack,
)


class TestEvaluationValidationAndMatching(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pack_dir = Path(self.temp_dir.name) / "test_pack"
        self.pack_dir.mkdir(parents=True, exist_ok=True)
        (self.pack_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.pack_dir / "annotations" / "labels").mkdir(parents=True, exist_ok=True)

        # Create 1 valid clean image (100x100 RGB)
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(self.pack_dir / "images" / "test_frame_01.jpg"), img)
        cv2.imwrite(str(self.pack_dir / "images" / "test_frame_02.jpg"), img)

        # Create valid baseline records
        self.base_records = [
            {
                "frame_id": "test_frame_01",
                "canonical_source_id": "videoA_f100",
                "camera": "cam03_east",
                "lighting_type": "real_day",
                "diagnostic_group": "current_val_diagnostic",
                "training_exposure": "unproven_checkpoint_exposure",
                "review_status": "verified",
                "annotation_state": "annotated",
                "is_unannotated": False,
                "is_ambiguous": False,
                "reviewer_notes": "",
                "boxes": [
                    {
                        "instance_id": "test_frame_01_inst_000",
                        "class_id": 0,
                        "class_name": "car",
                        "subtype": "pickup",
                        "is_ambiguous": False,
                        "ambiguity_reason": "",
                        "bbox_norm": [0.500000, 0.500000, 0.200000, 0.300000]
                    }
                ]
            },
            {
                "frame_id": "test_frame_02",
                "canonical_source_id": "videoB_f200",
                "camera": "cam43_south",
                "lighting_type": "real_night",
                "diagnostic_group": "unproven_checkpoint_candidate",
                "training_exposure": "unproven_checkpoint_exposure",
                "review_status": "verified",
                "annotation_state": "annotated",
                "is_unannotated": False,
                "is_ambiguous": False,
                "reviewer_notes": "",
                "boxes": [
                    {
                        "instance_id": "test_frame_02_inst_000",
                        "class_id": 1,
                        "class_name": "motorcycle",
                        "subtype": "motorcycle_commuter",
                        "is_ambiguous": False,
                        "ambiguity_reason": "",
                        "bbox_norm": [0.400000, 0.400000, 0.100000, 0.150000]
                    }
                ]
            }
        ]

        self._write_pack(self.base_records)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_pack(self, records):
        annos_file = self.pack_dir / "annotations" / "annotations.json"
        with open(annos_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        labels_dir = self.pack_dir / "annotations" / "labels"
        for r in records:
            fid = r["frame_id"]
            lines = []
            for b in r.get("boxes", []):
                cid = b["class_id"]
                xc, yc, bw, bh = b["bbox_norm"]
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (labels_dir / f"{fid}.txt").write_text("\n".join(lines), encoding="utf-8")

        manifest_file = self.pack_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as mf:
            json.dump({"samples": records}, mf, indent=2)

    # --------------------------------------------------------------------------
    # 1. Annotation vs YOLO Label Disagreement Tests
    # --------------------------------------------------------------------------

    def test_yolo_label_count_mismatch_detected(self):
        """When YOLO label file has different box count than annotations.json, report blocker."""
        label_file = self.pack_dir / "annotations" / "labels" / "test_frame_01.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n0 0.6 0.6 0.1 0.1", encoding="utf-8")

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("label_count_mismatch" in b["category"] for b in res.blockers))

    def test_yolo_label_class_mismatch_detected(self):
        """When YOLO label file has different class than annotations.json, report blocker."""
        label_file = self.pack_dir / "annotations" / "labels" / "test_frame_01.txt"
        label_file.write_text("3 0.500000 0.500000 0.200000 0.300000", encoding="utf-8")  # Class 3 truck vs 0 car

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("label_class_mismatch" in b["category"] for b in res.blockers))

    def test_yolo_label_coordinate_mismatch_detected(self):
        """When YOLO label coordinates deviate beyond float precision, report blocker."""
        label_file = self.pack_dir / "annotations" / "labels" / "test_frame_01.txt"
        label_file.write_text("0 0.550000 0.500000 0.200000 0.300000", encoding="utf-8")  # xc moved by 0.05

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("label_coord_mismatch" in b["category"] for b in res.blockers))

    # --------------------------------------------------------------------------
    # 2. Duplicate Canonical Source ID Tests
    # --------------------------------------------------------------------------

    def test_duplicate_canonical_source_id_rejected(self):
        """When two frames share the exact same canonical_source_id, report blocker."""
        dup_records = list(self.base_records)
        dup_records[1]["canonical_source_id"] = dup_records[0]["canonical_source_id"]
        self._write_pack(dup_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("duplicate_source" in b["category"] for b in res.blockers))

    # --------------------------------------------------------------------------
    # 3. Verified Empty vs Unannotated Frames Distinction Tests
    # --------------------------------------------------------------------------

    def test_verified_empty_frame_is_valid(self):
        """A verified frame with 0 boxes and state='verified_empty_background' is valid."""
        empty_records = list(self.base_records)
        empty_records[1]["boxes"] = []
        empty_records[1]["review_status"] = "verified"
        empty_records[1]["annotation_state"] = "verified_empty_background"
        empty_records[1]["is_unannotated"] = False
        self._write_pack(empty_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertTrue(res.is_valid)
        self.assertIn("test_frame_02", res.verified_empty_frames)

    def test_unannotated_frame_rejected_from_benchmark(self):
        """Unannotated frames must never be treated as verified empty backgrounds."""
        unann_records = list(self.base_records)
        unann_records[1]["boxes"] = []
        unann_records[1]["review_status"] = "unreviewed"
        unann_records[1]["annotation_state"] = "unannotated"
        unann_records[1]["is_unannotated"] = True
        self._write_pack(unann_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("unannotated_frame" in b["category"] for b in res.blockers))
        self.assertIn("test_frame_02", res.unannotated_frames)

    def test_draft_status_rejected_from_benchmark(self):
        """Frames with review_status='draft' must not pass validation."""
        draft_records = list(self.base_records)
        draft_records[0]["review_status"] = "draft"
        self._write_pack(draft_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertIn("test_frame_01", res.draft_frames)

    # --------------------------------------------------------------------------
    # 4. Strict Invalid Box Rejection Tests
    # --------------------------------------------------------------------------

    def test_negative_dimensions_rejected(self):
        """Boxes with negative or zero width/height must be rejected."""
        bad_records = list(self.base_records)
        bad_records[0]["boxes"][0]["bbox_norm"] = [0.5, 0.5, -0.1, 0.2]
        self._write_pack(bad_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("invalid_box_coords" in b["category"] for b in res.blockers))

    def test_out_of_bounds_coords_rejected(self):
        """Boxes with coordinates well outside [0, 1] must be rejected."""
        bad_records = list(self.base_records)
        bad_records[0]["boxes"][0]["bbox_norm"] = [1.2, 0.5, 0.2, 0.2]
        self._write_pack(bad_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("invalid_box_coords" in b["category"] for b in res.blockers))

    def test_unknown_class_id_rejected(self):
        """Boxes with class IDs outside 0..4 must be rejected."""
        bad_records = list(self.base_records)
        bad_records[0]["boxes"][0]["class_id"] = 7
        self._write_pack(bad_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("invalid_class" in b["category"] for b in res.blockers))

    def test_subtype_conflict_rejected(self):
        """Subtypes belonging to one class must not be assigned to another class."""
        bad_records = list(self.base_records)
        # medium_truck_6w belongs to class 3 (truck), cannot be assigned to class 0 (car)
        bad_records[0]["boxes"][0]["subtype"] = "medium_truck_6w"
        self._write_pack(bad_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("subtype_conflict" in b["category"] for b in res.blockers))

    def test_duplicate_instance_id_within_frame_rejected(self):
        """Multiple boxes in the same frame cannot have identical instance_ids."""
        bad_records = list(self.base_records)
        bad_records[0]["boxes"].append(dict(bad_records[0]["boxes"][0]))  # duplicate
        self._write_pack(bad_records)

        res = validate_review_pack(self.pack_dir, expected_frames_count=2)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("duplicate_instance_id" in b["category"] for b in res.blockers))

    # --------------------------------------------------------------------------
    # 5. One-to-One Bipartite Matching Logic Tests
    # --------------------------------------------------------------------------

    def test_one_to_one_exact_match(self):
        """Identical GT and prediction produces True Positive."""
        gt = [{"class_id": 0, "bbox_norm": [0.5, 0.5, 0.2, 0.2]}]
        pred = [{"class_id": 0, "bbox_norm": [0.5, 0.5, 0.2, 0.2], "conf": 0.95}]

        res = match_one_to_one(gt, pred, iou_threshold=0.50)
        self.assertEqual(len(res["matched_pairs"]), 1)
        self.assertTrue(res["matched_pairs"][0][3])  # is_class_match True
        self.assertEqual(len(res["unmatched_gt"]), 0)
        self.assertEqual(len(res["unmatched_pred"]), 0)

    def test_one_to_one_class_disagreement(self):
        """High IoU overlap with differing class produces class disagreement."""
        gt = [{"class_id": 3, "bbox_norm": [0.5, 0.5, 0.3, 0.3]}]  # GT: truck
        pred = [{"class_id": 0, "bbox_norm": [0.5, 0.5, 0.3, 0.3], "conf": 0.88}]  # Pred: car

        res = match_one_to_one(gt, pred, iou_threshold=0.50)
        self.assertEqual(len(res["matched_pairs"]), 1)
        self.assertFalse(res["matched_pairs"][0][3])  # is_class_match False
        self.assertEqual(len(res["class_disagreements"]), 1)
        self.assertEqual(res["class_disagreements"][0][3], 3)  # GT class 3
        self.assertEqual(res["class_disagreements"][0][4], 0)  # Pred class 0

    def test_one_to_one_localization_error(self):
        """Moderate IoU overlap (0.10 <= IoU < 0.50) produces localization error."""
        gt = [{"class_id": 0, "bbox_norm": [0.50, 0.50, 0.20, 0.20]}]
        # Shifted box with IoU ~ 0.28
        pred = [{"class_id": 0, "bbox_norm": [0.60, 0.50, 0.20, 0.20], "conf": 0.85}]

        res = match_one_to_one(gt, pred, iou_threshold=0.50, loc_iou_threshold=0.10)
        self.assertEqual(len(res["matched_pairs"]), 0)
        self.assertEqual(len(res["localization_errors"]), 1)

    def test_one_to_one_unmatched_missing_and_extra(self):
        """Unmatched boxes correctly categorized as missing (FN) and extra (FP)."""
        gt = [{"class_id": 0, "bbox_norm": [0.2, 0.2, 0.1, 0.1]}]
        pred = [{"class_id": 0, "bbox_norm": [0.8, 0.8, 0.1, 0.1], "conf": 0.90}]

        res = match_one_to_one(gt, pred, iou_threshold=0.50)
        self.assertEqual(len(res["matched_pairs"]), 0)
        self.assertEqual(res["unmatched_gt"], [0])
        self.assertEqual(res["unmatched_pred"], [0])

    # --------------------------------------------------------------------------
    # 6. Snapshot Creation & Immutability Tests
    # --------------------------------------------------------------------------

    def test_snapshot_creation_preserves_original_pack(self):
        """Creating an evaluation snapshot must leave the original review pack 100% untouched."""
        annos_file = self.pack_dir / "annotations" / "annotations.json"
        initial_sha = compute_file_sha256(annos_file)

        snap_dst = Path(self.temp_dir.name) / "eval_snapshot"
        manifest = create_evaluation_snapshot(self.pack_dir, snap_dst)

        # Verify snapshot files
        self.assertTrue((snap_dst / "images" / "test_frame_01.jpg").exists())
        self.assertTrue((snap_dst / "labels" / "test_frame_01.txt").exists())
        self.assertTrue((snap_dst / "dataset.yaml").exists())
        self.assertTrue((snap_dst / "manifest.json").exists())
        self.assertTrue((snap_dst / "README.md").exists())

        # Verify original pack immutability
        final_sha = compute_file_sha256(annos_file)
        self.assertEqual(initial_sha, final_sha, "Original annotations.json was modified!")

    def test_refuse_overwrite_nonempty_snapshot(self):
        """Creating an evaluation snapshot into an existing non-empty directory must raise FileExistsError."""
        snap_dst = Path(self.temp_dir.name) / "nonempty_snapshot"
        snap_dst.mkdir(parents=True, exist_ok=True)
        (snap_dst / "existing_file.txt").write_text("pre-existing content", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            create_evaluation_snapshot(self.pack_dir, snap_dst)

    def test_snapshot_integrity_verification_catches_tampering(self):
        """Tampering with an image or label in the snapshot must cause verify_snapshot_integrity to fail clearly."""
        from tools.validate_review_pack import verify_snapshot_integrity

        snap_dst = Path(self.temp_dir.name) / "eval_snapshot_integrity"
        create_evaluation_snapshot(self.pack_dir, snap_dst)

        # Baseline snapshot passes integrity
        res = verify_snapshot_integrity(snap_dst)
        self.assertEqual(res["status"], "verified")

        # Tamper with a label file
        label_file = snap_dst / "labels" / "test_frame_01.txt"
        label_file.write_text("0 0.999 0.999 0.1 0.1", encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            verify_snapshot_integrity(snap_dst)
        self.assertIn("Label hash mismatch", str(ctx.exception))

    def test_checkpoint_hash_mismatch_fails_integrity(self):
        """Providing an incorrect baseline checkpoint hash must raise RuntimeError."""
        from tools.validate_review_pack import verify_snapshot_integrity

        snap_dst = Path(self.temp_dir.name) / "eval_snapshot_ckpt"
        create_evaluation_snapshot(self.pack_dir, snap_dst)

        # Create dummy checkpoint file
        dummy_ckpt = Path(self.temp_dir.name) / "dummy_model.pt"
        dummy_ckpt.write_bytes(b"dummy model weights")

        with self.assertRaises(RuntimeError) as ctx:
            verify_snapshot_integrity(
                snap_dst,
                expected_checkpoint_path=dummy_ckpt,
                expected_checkpoint_sha256="0000000000000000000000000000000000000000000000000000000000000000"
            )
        self.assertIn("Baseline checkpoint hash mismatch", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 7. Size-Slice Matching Regression Tests
    # --------------------------------------------------------------------------

    def test_small_object_matched_by_slightly_larger_prediction(self):
        """
        Regression: A 31x31 letterboxed GT box (area 961 < 1024, small) matches a centered
        33x33 prediction (area 1089 > 1024, medium) at IoU ~ 0.882 > 0.50.
        Full-frame matching must count it as a detected small GT object, NOT a missed FN.
        """
        from tools.evaluate_baseline import compute_ground_truth_size_slices

        # Frame dims: 1920x1080 -> scale = 640 / 1920 = 1/3; resized = 640x360
        # 31x31 px -> norm: w = 31 / 640, h = 31 / 360
        gt_w = 31.0 / 640.0
        gt_h = 31.0 / 360.0

        # 33x33 px -> norm: w = 33 / 640, h = 33 / 360
        pred_w = 33.0 / 640.0
        pred_h = 33.0 / 360.0

        frame_data = [{
            "frame_id": "test_small_frame",
            "dimensions": [1920, 1080],
            "gt_boxes": [{
                "class_id": 0,
                "bbox_norm": [0.5, 0.5, gt_w, gt_h]
            }],
            "pred_boxes": [{
                "class_id": 0,
                "bbox_norm": [0.5, 0.5, pred_w, pred_h],
                "conf": 0.90
            }]
        }]

        stats = compute_ground_truth_size_slices(frame_data, match_iou_thresh=0.50)

        self.assertEqual(stats["small"]["gt_count"], 1)
        self.assertEqual(stats["small"]["detected_count"], 1)
        self.assertEqual(stats["small"]["missed_count"], 0)
        self.assertEqual(stats["small"]["recall"], 1.0)

    # --------------------------------------------------------------------------
    # 8. Machine Proposals Granular Decomposition Regression Test
    # --------------------------------------------------------------------------

    def test_proposal_error_decomposition_exact_counts(self):
        """
        Regression: Verify exact decomposition of machine proposals on the 26 real frames:
        290 correct matches, 304 unmatched GT, 4 unmatched predictions, 28 class disagreements,
        and 2 localization errors. Total GT = 624, total proposals = 324.
        """
        from tools.evaluate_baseline import assess_machine_proposals

        real_pack = Path("data/review_pack_v1")
        manifest_file = real_pack / "manifest.json"
        if not manifest_file.exists():
            self.skipTest("Real pack manifest not found at data/review_pack_v1/manifest.json")

        with open(manifest_file, "r", encoding="utf-8") as f:
            samples = json.load(f).get("samples", [])

        res = assess_machine_proposals(real_pack, samples)
        self.assertEqual(res["evaluated_frames_count"], 26)
        self.assertEqual(res["total_human_gt"], 624)
        self.assertEqual(res["total_proposals"], 324)

        gran = res["granular_error_breakdown"]
        self.assertEqual(gran["correct_matches"], 290)
        self.assertEqual(gran["unmatched_gt"], 304)
        self.assertEqual(gran["unmatched_pred"], 4)
        self.assertEqual(gran["class_disagreements"], 28)
        self.assertEqual(gran["localization_errors"], 2)


if __name__ == "__main__":
    unittest.main()

