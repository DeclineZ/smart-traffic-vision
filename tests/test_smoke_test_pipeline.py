"""
Unit Tests for Batch 8 Smoke Test Pipeline, Candidate Evaluation, and Dataset Separation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    THAI_5CLASS_NAMES,
    validate_box,
)
from tools.evaluate_baseline import (
    EXPECTED_BASELINE_SHA256,
    run_baseline_evaluation,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.run_smoke_test import (
    check_preflight_invariants,
    inspect_layer_structure_and_freezing,
)


class TestSmokeTestPipeline(unittest.TestCase):

    def setUp(self):
        self.baseline_path = Path("models/yolo26s_thai_traffic.pt")
        self.snapshot_dir = Path("data/eval_snapshot_v1")
        self.consolidated_dir = Path("data/review_pack_consolidated_v3")
        self.manifests_dir = Path("data/training_manifests_v6")

    def test_baseline_checkpoint_exists_and_matches_expected_hash(self):
        """Baseline checkpoint must exist at models/yolo26s_thai_traffic.pt with expected SHA256."""
        self.assertTrue(self.baseline_path.exists(), f"Baseline model missing at {self.baseline_path}")
        actual_sha = compute_file_sha256(self.baseline_path)
        self.assertEqual(actual_sha, EXPECTED_BASELINE_SHA256)

    def test_preflight_invariants(self):
        """Pre-flight check must succeed and return tracked reference file hashes."""
        initial_hashes = check_preflight_invariants(
            baseline_path=self.baseline_path,
            snapshot_dir=self.snapshot_dir,
            consolidated_dir=self.consolidated_dir,
            manifests_dir=self.manifests_dir,
        )
        self.assertGreaterEqual(len(initial_hashes), 5)
        self.assertIn(str(self.baseline_path.resolve()), initial_hashes)
        self.assertEqual(initial_hashes[str(self.baseline_path.resolve())], EXPECTED_BASELINE_SHA256)

    def test_dataset_membership_and_exclusions(self):
        """Training dataset must include exactly 43 eligible frames and exclude the 1 rejected frame."""
        annot_file = self.consolidated_dir / "annotations" / "annotations.json"
        with open(annot_file, "r", encoding="utf-8") as f:
            records = json.load(f)

        eligible = [r for r in records if r.get("training_eligible")]
        rejected = [r for r in records if r.get("is_rejected")]

        self.assertEqual(len(eligible), 43)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["frame_id"], "cam44_north_f019140")

        # Sum of approved boxes across 43 eligible frames must equal 1,488
        total_boxes = sum(r.get("approved_boxes_count", len(r.get("boxes", []))) for r in eligible)
        self.assertEqual(total_boxes, 1488)

    def test_canonical_source_disjointness(self):
        """Train, Val, and Eval sources must have zero mutual intersection."""
        with open(self.consolidated_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
            v3_records = json.load(f)
        train_sources = set(r["canonical_source_id"] for r in v3_records if r.get("training_eligible"))

        with open(self.manifests_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
            prim_val = json.load(f)
        with open(self.manifests_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            inv = json.load(f)
        inv_map = {r["inventory_id"]: r for r in inv["records"]}
        val_sources = set(inv_map[r]["canonical_source_id"] for r in prim_val["records"])

        with open(self.snapshot_dir / "manifest.json", "r", encoding="utf-8") as f:
            snap = json.load(f)
        eval_sources = set(s["canonical_source_id"] for s in snap["samples"])

        self.assertEqual(len(train_sources), 43)
        self.assertEqual(len(val_sources), 130)
        self.assertEqual(len(eval_sources), 42)

        self.assertEqual(len(train_sources & val_sources), 0)
        self.assertEqual(len(train_sources & eval_sources), 0)
        self.assertEqual(len(val_sources & eval_sources), 0)

    def test_layer_freezing_structure(self):
        """freeze=10 must freeze layers 0-9 (Backbone) and leave 10-23 trainable (Neck & Head)."""
        arch = inspect_layer_structure_and_freezing(self.baseline_path, freeze_layer_count=10)
        self.assertEqual(arch["total_parameters"], 9951734)
        self.assertEqual(arch["frozen_parameters"], 4451008)
        self.assertEqual(arch["trainable_parameters"], 5500726)
        self.assertEqual(len(arch["frozen_layers"]), 10)
        self.assertEqual(len(arch["trainable_layers"]), 14)


if __name__ == "__main__":
    unittest.main()
