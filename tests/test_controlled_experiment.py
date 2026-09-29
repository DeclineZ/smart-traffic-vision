"""
Tests for Batch 9 Controlled Experiment Runner and Invariants.
"""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_baseline import EXPECTED_BASELINE_SHA256
from tools.prepare_review_pack import compute_file_sha256
from tools.run_controlled_experiment import (
    check_preflight_invariants,
    materialize_controlled_datasets,
)


class TestControlledExperiment(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.baseline_path = self.repo_root / "models" / "yolo26s_thai_traffic.pt"
        self.snapshot_dir = self.repo_root / "data" / "eval_snapshot_v1"
        self.manifests_dir = self.repo_root / "data" / "training_manifests_v6"
        self.consolidated_dir = self.repo_root / "data" / "review_pack_consolidated_v3"

    def test_baseline_checkpoint_sha256(self):
        """Verifies that the deployed baseline has the exact expected SHA256."""
        self.assertTrue(self.baseline_path.exists(), f"Baseline model not found at {self.baseline_path}")
        actual_sha = compute_file_sha256(self.baseline_path)
        self.assertEqual(
            actual_sha,
            EXPECTED_BASELINE_SHA256,
            f"Baseline checkpoint SHA256 altered! Expected {EXPECTED_BASELINE_SHA256}, got {actual_sha}",
        )

    def test_preflight_invariants(self):
        """Verifies pre-flight invariant check returns correct file hashes."""
        hashes = check_preflight_invariants(
            baseline_path=self.baseline_path,
            snapshot_dir=self.snapshot_dir,
            manifests_dir=self.manifests_dir,
            consolidated_dir=self.consolidated_dir,
        )
        self.assertIn(str(self.baseline_path.resolve()), hashes)
        self.assertEqual(hashes[str(self.baseline_path.resolve())], EXPECTED_BASELINE_SHA256)

    def test_refuse_nonempty_materialization_destination(self):
        """Ensures materialization refuses to overwrite nonempty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dir_a = tmp_path / "ds_a"
            dir_b = tmp_path / "ds_b"
            dir_a.mkdir()
            (dir_a / "dummy.txt").write_text("dummy", encoding="utf-8")

            with self.assertRaises(RuntimeError) as ctx:
                materialize_controlled_datasets(
                    dataset_a_dir=dir_a,
                    dataset_b_dir=dir_b,
                    manifests_dir=self.manifests_dir,
                    consolidated_dir=self.consolidated_dir,
                    snapshot_dir=self.snapshot_dir,
                )
            self.assertIn("already exists and is not empty", str(ctx.exception))

    def test_dataset_lineage_and_exact_external_delta(self):
        """
        Verifies that Manifest A and Manifest B differ by strictly the 25
        reviewed external frames (813 boxes), and that canonical splits are 100% disjoint.
        """
        with open(self.manifests_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
            man_a = json.load(f)
        with open(self.manifests_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
            prim_val = json.load(f)
        with open(self.manifests_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            inv = json.load(f)
        with open(self.consolidated_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
            consolidated_annots = json.load(f)
        with open(self.snapshot_dir / "manifest.json", "r", encoding="utf-8") as f:
            snap_man = json.load(f)

        inv_map = {r["inventory_id"]: r for r in inv["records"]}

        # 1. Dataset A counts
        local_ids = man_a["train_records"]
        self.assertEqual(len(local_ids), 1092)

        local_sources = set(inv_map[rid]["canonical_source_id"] for rid in local_ids)
        self.assertEqual(len(local_sources), 527)

        # 2. External reviewed delta
        ext_reviewed = [
            r for r in consolidated_annots
            if r.get("data_origin") == "external_ua_detrac" and r.get("training_eligible")
        ]
        self.assertEqual(len(ext_reviewed), 25)
        ext_sources = set(r["canonical_source_id"] for r in ext_reviewed)
        self.assertEqual(len(ext_sources), 25)

        # Total boxes in 25 reviewed frames
        ext_box_count = sum(r.get("approved_boxes_count", 0) for r in ext_reviewed)
        self.assertEqual(ext_box_count, 813)

        # 3. Primary validation counts & lineage exposure
        val_ids = prim_val["records"]
        self.assertEqual(len(val_ids), 130)
        val_sources = set(inv_map[rid]["canonical_source_id"] for rid in val_ids)
        self.assertEqual(len(val_sources), 130)

        # Confirm 107 of 130 originated from old training split
        val_splits = Counter(inv_map[rid].get("original_split") for rid in val_ids)
        self.assertEqual(val_splits["train"], 107)
        self.assertEqual(val_splits["val"], 23)

        # 4. Diagnostic snapshot
        diag_sources = set(s["canonical_source_id"] for s in snap_man["samples"])
        self.assertEqual(len(diag_sources), 42)

        # 5. Canonical split disjointness
        b_sources = local_sources.union(ext_sources)
        self.assertEqual(len(local_sources & val_sources), 0, "Train A & Val overlap!")
        self.assertEqual(len(local_sources & diag_sources), 0, "Train A & Eval overlap!")
        self.assertEqual(len(b_sources & val_sources), 0, "Train B & Val overlap!")
        self.assertEqual(len(b_sources & diag_sources), 0, "Train B & Eval overlap!")
        self.assertEqual(len(val_sources & diag_sources), 0, "Val & Eval overlap!")


if __name__ == "__main__":
    unittest.main()
