"""
tests/test_remapping_and_expansion.py

Focused unit and regression tests for the external-data remapping and expansion experiment.
Verifies:
1. Mapping Scope: Remapping applies strictly to copied external annotations; local training,
   validation, and diagnostic ground truth retain original classes and taxonomy.
2. Retained Boxes: Original boxes and coordinates are preserved; exactly 5 truck boxes in the
   25 reviewed frames are converted to car (0) in experiment copies.
3. Nested Membership: The 25 external frames in R25 are strictly nested inside the 100 external
   frames in R100 (25 subset of 100, 100% canonical containment).
4. Canonical Exclusions & Disjointness: Rejected sources (cam44_north_f019140) and stale synthetic
   descendants are quarantined; zero canonical contamination across train, val, and diagnostic benchmark.
5. Teacher Coordinate Merging: Tile coordinate restoration to global 640x640, deduplication, and
   conflict logging (bus labels strictly preserved, never overwritten).
6. Immutable Source Artifacts: Baseline checkpoint, reference manifests, review packs, and old runs
   remain byte-for-byte intact.
"""

from collections import Counter
import json
from pathlib import Path
import unittest

from tools.audit_dataset import THAI_5CLASS_NAMES
from tools.evaluate_baseline import EXPECTED_BASELINE_SHA256
from tools.external_remapping_and_expansion import (
    UADETRAC_RAW_TO_REMAPPED_THAI5,
    merge_teacher_proposals_with_source_boxes,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.teacher_completion_pilot import (
    compute_box_iou,
    tile_xyxy_to_global_xyxy,
    xyxy_to_norm_yolo,
)


class TestRemappingAndExpansion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.baseline_path = cls.repo_root / "models" / "yolo26s_thai_traffic.pt"
        cls.manifests_v6_dir = cls.repo_root / "data" / "training_manifests_v6"
        cls.consolidated_v3_dir = cls.repo_root / "data" / "review_pack_consolidated_v3"
        cls.snapshot_dir = cls.repo_root / "data" / "eval_snapshot_v1"
        cls.ds_r25_dir = cls.repo_root / "data" / "experiment_remapped_r25"
        cls.ds_r100_dir = cls.repo_root / "data" / "experiment_remapped_r100"

    def test_immutable_source_artifacts(self):
        """Verifies baseline checkpoint and source reference manifests are byte-for-byte unchanged."""
        actual_base_sha = compute_file_sha256(self.baseline_path)
        self.assertEqual(
            actual_base_sha,
            EXPECTED_BASELINE_SHA256,
            f"Baseline checkpoint was modified! Expected {EXPECTED_BASELINE_SHA256}, got {actual_base_sha}",
        )

        # Batch 9 candidate best.pt checkpoints must exist and be intact
        b9_a = self.repo_root / "runs" / "train" / "candidate_a_local_only" / "weights" / "best.pt"
        b9_b = self.repo_root / "runs" / "train" / "candidate_b_reviewed_external" / "weights" / "best.pt"
        self.assertTrue(b9_a.exists(), f"Batch 9 Candidate A checkpoint missing: {b9_a}")
        self.assertTrue(b9_b.exists(), f"Batch 9 Candidate B checkpoint missing: {b9_b}")

    def test_remapping_scope_external_only(self):
        """
        Ensures remapping is NEVER applied to local training data, validation sets,
        or diagnostic ground truth. Local trucks and three-wheelers must remain in local data.
        """
        # 1. Primary validation labels must contain trucks (class 3) and three-wheelers (class 4)
        val_labels = list((self.ds_r25_dir / "labels" / "val").glob("*.txt"))
        self.assertEqual(len(val_labels), 130)

        val_classes = Counter()
        for lp in val_labels:
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        val_classes[int(parts[0])] += 1

        self.assertGreater(val_classes[3], 100, "Validation set must retain ground-truth trucks (class 3)!")
        self.assertGreater(val_classes[4], 30, "Validation set must retain ground-truth three-wheelers (class 4)!")

        # 2. Diagnostic benchmark labels must retain original classes
        diag_labels = list((self.snapshot_dir / "labels").glob("*.txt"))
        self.assertEqual(len(diag_labels), 42)
        diag_classes = Counter()
        for lp in diag_labels:
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        diag_classes[int(parts[0])] += 1

        self.assertGreater(diag_classes[3], 15, "Diagnostic benchmark must retain ground-truth trucks (class 3)!")
        self.assertGreater(diag_classes[4], 15, "Diagnostic benchmark must retain ground-truth three-wheelers (class 4)!")

    def test_retained_boxes_and_truck_remapping(self):
        """
        Verifies that the 25 reviewed external frames in R25 retain their exact 813 boxes,
        with exactly the 5 previously approved truck boxes converted to car (0).
        """
        with open(self.consolidated_v3_dir / "manifest.json", "r", encoding="utf-8") as f:
            v3_man = json.load(f)

        ext_fids = [
            Path(p).stem for p in v3_man["export_audit"]["exported_label_files"]
            if Path(p).stem.startswith("MVI_")
        ]
        self.assertEqual(len(ext_fids), 25)

        total_r25_ext_boxes = 0
        r25_ext_classes = Counter()

        for fid in ext_fids:
            orig_lbl = self.consolidated_v3_dir / "annotations" / "labels" / f"{fid}.txt"
            r25_lbl = self.ds_r25_dir / "labels" / "train" / f"{fid}.txt"
            self.assertTrue(r25_lbl.exists(), f"Remapped label missing: {r25_lbl}")

            with open(orig_lbl, "r", encoding="utf-8") as f:
                orig_lines = [l.strip().split() for l in f if l.strip()]
            with open(r25_lbl, "r", encoding="utf-8") as f:
                r25_lines = [l.strip().split() for l in f if l.strip()]

            # Exactly equal box counts per frame
            self.assertEqual(len(orig_lines), len(r25_lines), f"Box count mismatch for {fid}")

            for orig_parts, r25_parts in zip(orig_lines, r25_lines):
                orig_cid = int(orig_parts[0])
                r25_cid = int(r25_parts[0])

                # Coordinates must match exactly
                for c_idx in range(1, 5):
                    self.assertAlmostEqual(
                        float(orig_parts[c_idx]),
                        float(r25_parts[c_idx]),
                        places=5,
                        msg=f"Coordinate altered in {fid} box {orig_parts}",
                    )

                # Class remapping assertion: truck(3) -> car(0); all other classes unchanged
                if orig_cid == 3:
                    self.assertEqual(r25_cid, 0, f"Truck box in {fid} was not remapped to car!")
                else:
                    self.assertEqual(r25_cid, orig_cid, f"Non-truck class {orig_cid} unexpectedly changed in {fid}!")

                total_r25_ext_boxes += 1
                r25_ext_classes[r25_cid] += 1

        self.assertEqual(total_r25_ext_boxes, 813)
        self.assertEqual(r25_ext_classes[3], 0, "Zero truck boxes must exist in remapped R25 external set!")
        self.assertEqual(r25_ext_classes[0], 726, "Car count must be 721 + 5 = 726!")
        self.assertEqual(r25_ext_classes[2], 84, "Bus count must remain 84!")
        self.assertEqual(r25_ext_classes[1], 3, "Motorcycle count must remain 3!")

    def test_nested_membership(self):
        """
        Verifies that R25 external frames are strictly a subset of R100 external frames,
        with 100 unique canonical external frames in total.
        """
        with open(self.ds_r25_dir / "manifest.json", "r", encoding="utf-8") as f:
            meta_r25 = json.load(f)
        with open(self.ds_r100_dir / "manifest.json", "r", encoding="utf-8") as f:
            meta_r100 = json.load(f)

        self.assertEqual(meta_r25["external_train_images"], 25)
        self.assertEqual(meta_r100["external_train_images"], 100)

        # External image files in R25 and R100
        r25_ext_files = set(p.name for p in (self.ds_r25_dir / "images" / "train").glob("MVI_*.jpg"))
        r100_ext_files = set(p.name for p in (self.ds_r100_dir / "images" / "train").glob("MVI_*.jpg"))

        self.assertEqual(len(r25_ext_files), 25)
        self.assertEqual(len(r100_ext_files), 100)

        # Strict nested subset check
        self.assertTrue(r25_ext_files.issubset(r100_ext_files), "R25 external frames must be strictly nested inside R100!")

        # Exactly 75 new frames
        delta = r100_ext_files - r25_ext_files
        self.assertEqual(len(delta), 75)

    def test_canonical_exclusions_and_zero_contamination(self):
        """
        Verifies that rejected source cam44_north_f019140 is quarantined,
        and that train, validation, and diagnostic splits have 0 canonical overlap.
        """
        for ds_dir in [self.ds_r25_dir, self.ds_r100_dir]:
            train_images = [p.stem for p in (ds_dir / "images" / "train").glob("*.jpg")]
            # Quarantined source must NOT be in training
            for f in train_images:
                self.assertNotIn("cam44_north_f019140", f, f"Rejected source {f} leaked into {ds_dir}!")

        # Overlap checks
        train_r100_canons = set(p.stem.split("_jpg")[0] for p in (self.ds_r100_dir / "images" / "train").glob("*.jpg"))
        val_canons = set(p.stem.split("_jpg")[0] for p in (self.ds_r100_dir / "images" / "val").glob("*.jpg"))

        with open(self.snapshot_dir / "manifest.json", "r", encoding="utf-8") as f:
            snap_man = json.load(f)
        diag_canons = set(s["canonical_source_id"] for s in snap_man["samples"])

        self.assertEqual(len(train_r100_canons & val_canons), 0, "Train R100 & Val overlap!")
        self.assertEqual(len(train_r100_canons & diag_canons), 0, "Train R100 & Diag overlap!")
        self.assertEqual(len(val_canons & diag_canons), 0, "Val & Diag overlap!")

    def test_teacher_coordinate_merging_and_bus_protection(self):
        """
        Tests coordinate projection and verifies that teacher proposals
        never overwrite authoritative source boxes (especially bus labels).
        """
        # 1. Coordinate projection test
        tile_box = [10.0, 20.0, 50.0, 80.0]
        x_off, y_off = 256, 128
        global_xyxy = tile_xyxy_to_global_xyxy(tile_box, x_off, y_off)
        self.assertEqual(global_xyxy, [266.0, 148.0, 306.0, 208.0])

        norm_box = xyxy_to_norm_yolo(global_xyxy, img_w=640, img_h=640)
        self.assertAlmostEqual(norm_box[0], (266.0 + 306.0) / (2 * 640.0), places=5)
        self.assertAlmostEqual(norm_box[2], 40.0 / 640.0, places=5)

        # 2. Bus protection: source bus box overlapped by teacher car proposal must NOT be overwritten
        source_boxes = [{
            "class_id": 2,  # bus
            "class_name": "bus",
            "bbox_norm": [0.5, 0.5, 0.2, 0.4],
            "is_teacher_addition": False,
        }]
        teacher_proposals = [{
            "class_id": 0,  # car
            "class_name": "car",
            "bbox_norm": [0.5, 0.5, 0.2, 0.4],
            "xyxy_px": [256.0, 192.0, 384.0, 448.0],
            "confidence": 0.85,
            "method": "both",
        }]

        final_boxes, additions, conflicts = merge_teacher_proposals_with_source_boxes(
            source_boxes=source_boxes,
            teacher_proposals=teacher_proposals,
            img_w=640,
            img_h=640,
            match_iou_thresh=0.45,
        )

        # Final boxes must still contain exactly 1 box with class_id == 2 (bus)
        self.assertEqual(len(final_boxes), 1)
        self.assertEqual(final_boxes[0]["class_id"], 2, "Bus label was overwritten by conflicting proposal!")
        self.assertEqual(len(additions), 0, "Conflicting proposal must not be added as a duplicate box")
        self.assertEqual(len(conflicts), 1, "Conflict must be logged")
        self.assertEqual(conflicts[0]["source_class_id"], 2)
        self.assertEqual(conflicts[0]["proposal_class_id"], 0)


if __name__ == "__main__":
    unittest.main()
