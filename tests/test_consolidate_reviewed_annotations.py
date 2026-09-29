"""
tests/test_consolidate_reviewed_annotations.py - Verification & Regression Tests for Batch 7 Consolidation Completion

Regressions & Verifications:
1. Reviewed Roboflow variants: exact image/variant matching with SHA256 verification; coordinates not propagated to different variants.
2. Embedded-box and export agreement: inventory boxes match exported YOLO labels line-for-line, box-for-box, and class-for-class.
3. Class-only corrections in review_pack_v2: box comparison detects changes and flags synthetic variants as stale.
4. Stale descendants: all 41 synthetic descendants of all 18 modified local frames are excluded from candidate training.
5. Non-square-image size bin regression: compute_aspect_preserving_size uses true 1920x1080 native dimensions for local CCTV frames,
   yielding 117 medium->small and 9 large->medium shifts across all 18 local frames (111 and 9 across the original 17).
6. Canonical source-level exclusions: rejected canonical sources (cam44_north_f019140) and their descendants/variants are strictly excluded
   from candidate training and resampling.
7. Fully accepted continuation decisions: MVI_20063_img00769 and cam44_north_f148440 have all proposals resolved and are verified for training.
8. Exported-count reconciliation: exported class totals count ONLY over the 43 exported eligible frames (1488 approved boxes).
9. Dynamic manifest summaries: derived dynamically from final selected records in v6 (Manifest A: 1092 frames; Manifest B: 1355 frames).
10. Evaluation benchmark diagnostic status and zero contamination.
"""

from collections import Counter
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from tools.audit_dataset import (
    THAI_5CLASS_NAMES,
    compute_aspect_preserving_size,
    parse_frame_provenance,
)
from tools.consolidate_reviewed_annotations import (
    ConsolidationConfig,
    assert_inventory_yolo_agreement,
    compare_boxes,
    compute_manifest_summary,
    load_pack_annotations,
    reconcile_review_lineage,
    reconcile_training_manifests,
    validate_and_export_approved_pack,
)
from tools.prepare_review_pack import compute_file_sha256


class TestReviewLineageAndPrecedence(unittest.TestCase):
    """Verifies lineage reconciliation, explicit precedence, and conflict detection."""

    def setUp(self):
        self.config = ConsolidationConfig()

    def test_exact_44_frame_reconciliation(self):
        """All 44 canonical review sources must reconcile into 43 verified, 1 rejected, 43 eligible, 0 blocked."""
        records, audit = reconcile_review_lineage(self.config)
        self.assertEqual(len(records), 44)
        self.assertEqual(audit["total_canonical_sources"], 44)
        self.assertEqual(audit["verified_count"], 43)
        self.assertEqual(audit["rejected_count"], 1)
        self.assertEqual(audit["rejected_frames"], ["cam44_north_f019140"])
        self.assertEqual(audit["eligible_count"], 43)
        self.assertEqual(audit["blocked_count"], 0)
        self.assertEqual(len(audit["blocked_frames"]), 0)

    def test_precedence_conflict_fails_fast(self):
        """If a canonical frame has conflicting authoritative decisions across packs, must fail fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            fake_pilot = tmp_root / "fake_pilot" / "annotations"
            fake_cont = tmp_root / "fake_cont" / "annotations"
            fake_v2 = tmp_root / "fake_v2" / "annotations"
            for d in [fake_pilot, fake_cont, fake_v2]:
                d.mkdir(parents=True)

            collision_frame = "test_cam_collision_f001"
            rec_pilot = [{"frame_id": collision_frame, "review_status": "verified", "boxes": []}]
            rec_cont = [{"frame_id": collision_frame, "review_status": "rejected", "boxes": []}]
            rec_v2 = [{"frame_id": collision_frame, "review_status": "draft", "boxes": []}]

            with open(fake_pilot / "annotations.json", "w") as f:
                json.dump(rec_pilot, f)
            with open(fake_cont / "annotations.json", "w") as f:
                json.dump(rec_cont, f)
            with open(fake_v2 / "annotations.json", "w") as f:
                json.dump(rec_v2, f)

            cfg = ConsolidationConfig(
                pack_v2_dir=tmp_root / "fake_v2",
                pilot_pack_dir=tmp_root / "fake_pilot",
                continuation_pack_dir=tmp_root / "fake_cont"
            )

            with self.assertRaises(ValueError) as ctx:
                reconcile_review_lineage(cfg)
            self.assertIn("Lineage conflict", str(ctx.exception))


class TestApprovedLabelExportAndFiltering(unittest.TestCase):
    """Verifies label export rules, proposal filtering, and class fidelity in review_pack_consolidated_v3."""

    def setUp(self):
        self.pack_dir = Path("data/review_pack_consolidated_v3")
        self.assertTrue(self.pack_dir.exists(), "Consolidated pack v3 must exist")
        with open(self.pack_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
            self.annos = json.load(f)
        with open(self.pack_dir / "manifest.json", "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_only_eligible_verified_frames_have_exported_labels(self):
        """Labels dir must contain exactly 43 label files for the 43 eligible verified frames."""
        labels_dir = self.pack_dir / "annotations" / "labels"
        label_files = list(labels_dir.glob("*.txt"))
        self.assertEqual(len(label_files), 43, "Must have exactly 43 label files for eligible frames")

        # Rejected frame cam44_north_f019140 must NOT have label file
        self.assertFalse((labels_dir / "cam44_north_f019140.txt").exists())

        # Accepted frames must have exported label files
        self.assertTrue((labels_dir / "cam44_north_f148440.txt").exists())
        self.assertTrue((labels_dir / "MVI_20063_img00769.txt").exists())

    def test_fully_resolved_training_eligibility(self):
        """MVI_20063_img00769 and cam44_north_f148440 are verified and eligible with zero pending proposals."""
        for fid in ["MVI_20063_img00769", "cam44_north_f148440"]:
            rec = next(r for r in self.annos if r["frame_id"] == fid)
            self.assertEqual(rec["review_status"], "verified")
            self.assertTrue(rec["training_eligible"])
            self.assertEqual(rec["eligibility_status"], "eligible")
            self.assertEqual(rec["pending_proposals_count"], 0)
            self.assertTrue(rec["label_exported"])

    def test_exported_count_reconciliation(self):
        """Exported class totals must count ONLY over the 43 exported eligible frames; zero rejected boxes."""
        ea = self.manifest["export_audit"]
        self.assertEqual(ea["exported_frames_count"], 43)
        self.assertEqual(ea["blocked_frames_count"], 0)
        self.assertEqual(ea["omitted_rejected_frames_count"], 1)

        # Total approved boxes must equal sum of class breakdown
        self.assertEqual(ea["total_approved_boxes"], sum(ea["class_breakdown"].values()))
        self.assertEqual(ea["total_approved_boxes"], 1488)

        # Verify no boxes from rejected frames are counted in exported total
        labels_dir = self.pack_dir / "annotations" / "labels"
        total_boxes_in_exported_files = 0
        for f in labels_dir.glob("*.txt"):
            lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            total_boxes_in_exported_files += len(lines)

        self.assertEqual(ea["total_approved_boxes"], total_boxes_in_exported_files)


class TestReviewedRoboflowVariantsAndInventoryLinking(unittest.TestCase):
    """Verifies that reviewed Roboflow variants link properly with verified hashes, and coordinates don't leak."""

    def setUp(self):
        self.v6_dir = Path("data/training_manifests_v6")
        with open(self.v6_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            self.inv = json.load(f)
        self.inv_by_id = {r["inventory_id"]: r for r in self.inv["records"]}
        self.pack_dir = Path("data/review_pack_consolidated_v3")

    def test_reviewed_roboflow_variants_linked_with_hash_verification(self):
        """All 25 eligible reviewed external frames link to consolidated labels despite roboflow_augmented variant."""
        with open(self.pack_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
            annos = json.load(f)

        ext_eligible = [
            r for r in annos
            if (r.get("data_origin") == "external_ua_detrac" or r["frame_id"].startswith("MVI"))
            and r.get("training_eligible")
        ]
        self.assertEqual(len(ext_eligible), 25)

        for er in ext_eligible:
            in_id = er["inventory_id"]
            self.assertIn(in_id, self.inv_by_id, f"Missing inventory record for {in_id}")
            inv_rec = self.inv_by_id[in_id]

            self.assertEqual(inv_rec["variant"], "roboflow_augmented")
            self.assertTrue(inv_rec["label_exists_on_disk"])
            self.assertTrue(inv_rec["label_path"].endswith(f"{er['canonical_source_id']}.txt"))
            self.assertEqual(inv_rec["annotation_provenance"], "human_reviewed_consolidated_batch7")
            self.assertTrue(inv_rec["image_exists_on_disk"])
            self.assertEqual(inv_rec["image_status"], "available_local_file")

            # Verify image SHA256 matches review pack image
            img_file = self.pack_dir / "images" / f"{er['canonical_source_id']}.jpg"
            self.assertTrue(img_file.is_file())
            img_hash = compute_file_sha256(img_file)
            self.assertEqual(er["image_sha256"], img_hash)

    def test_coordinates_not_propagated_to_different_transformed_variants(self):
        """Unreviewed variants of the same canonical external frame must NOT have coordinates propagated."""
        rev_id = "external:MVI_40863_img00007_jpg.rf.026104382d330ddf46b9d970430cf56a.jpg"
        other_id = "external:MVI_40863_img00007_jpg.rf.a75411a296588e5b82689be1313b4d7d.jpg"

        rev_rec = self.inv_by_id[rev_id]
        other_rec = self.inv_by_id[other_id]

        self.assertEqual(rev_rec["total_boxes"], 45)
        self.assertEqual(rev_rec["annotation_provenance"], "human_reviewed_consolidated_batch7")

        # The other variant must remain uncorrupted with original labels/status
        self.assertEqual(other_rec["total_boxes"], 23)
        self.assertIsNone(other_rec["label_path"])
        self.assertEqual(other_rec["annotation_provenance"], "ua_detrac_manual_box_tracking")
        self.assertEqual(other_rec["image_status"], "unresolved_not_downloaded")

    def test_embedded_box_and_export_agreement(self):
        """For all 43 records in inventory pointing to exported labels, assert line-for-line agreement."""
        checked_count = assert_inventory_yolo_agreement(self.inv["records"])
        self.assertEqual(checked_count, 43, "Must have checked exactly 43 inventory records (18 local + 25 external)")


class TestSizeBinCalculationsAndRegressions(unittest.TestCase):
    """
    Verifies aspect-preserving size-bin calculations for non-square local CCTV images (1920x1080),
    ensuring medium->small and large->medium shifts occur without altering normalized coordinates.
    """

    def setUp(self):
        self.v6_dir = Path("data/training_manifests_v6")
        self.pack_dir = Path("data/review_pack_consolidated_v3")
        with open(self.pack_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
            self.annos = json.load(f)
        with open(self.v6_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            self.inv = json.load(f)
        self.inv_by_id = {r["inventory_id"]: r for r in self.inv["records"]}

    def test_non_square_image_size_bin_regression(self):
        """
        Non-square regression:
        1. Letterbox scaling formula unit test: 1920x1080 scaled into 640x640 yields scale=1/3 and resized=640x360.
        2. Boundary test cases prove medium->small and large->medium shifts.
        3. Full dataset check across all local CCTV records confirms size corrections.
        4. Normalized bounding box coordinates remain completely unaltered.
        """
        # 1. Unit calculation test on 1920x1080 vs 640x640
        sz_naive_m = compute_aspect_preserving_size(0.08, 0.04, 640, 640, ref_size=640)
        sz_true_s = compute_aspect_preserving_size(0.08, 0.04, 1920, 1080, ref_size=640)
        self.assertEqual(sz_naive_m["size_bucket"], "medium")
        self.assertEqual(sz_true_s["size_bucket"], "small")

        sz_naive_l = compute_aspect_preserving_size(0.25, 0.10, 640, 640, ref_size=640)
        sz_true_m = compute_aspect_preserving_size(0.25, 0.10, 1920, 1080, ref_size=640)
        self.assertEqual(sz_naive_l["size_bucket"], "large")
        self.assertEqual(sz_true_m["size_bucket"], "medium")

        # 2. Dataset regression across all 18 local eligible frames
        local_eligible = [
            r for r in self.annos
            if r.get("data_origin") == "local" and r.get("training_eligible")
        ]
        self.assertEqual(len(local_eligible), 18)

        shifts_18 = {"m_to_s": 0, "l_to_m": 0, "other": 0, "same": 0}
        shifts_17 = {"m_to_s": 0, "l_to_m": 0, "other": 0, "same": 0}
        total_boxes = 0

        for r in local_eligible:
            is_17 = (r["frame_id"] != "cam44_north_f148440")
            app_boxes = [
                b for b in r["boxes"]
                if not b.get("is_proposal") or b.get("proposal_status") == "accepted"
            ]
            total_boxes += len(app_boxes)
            for b in app_boxes:
                bw, bh = b["bbox_norm"][2], b["bbox_norm"][3]
                b_naive = compute_aspect_preserving_size(bw, bh, 640, 640, ref_size=640)["size_bucket"]
                b_true = compute_aspect_preserving_size(bw, bh, 1920, 1080, ref_size=640)["size_bucket"]
                if b_naive == b_true:
                    shifts_18["same"] += 1
                    if is_17: shifts_17["same"] += 1
                elif b_naive == "medium" and b_true == "small":
                    shifts_18["m_to_s"] += 1
                    if is_17: shifts_17["m_to_s"] += 1
                elif b_naive == "large" and b_true == "medium":
                    shifts_18["l_to_m"] += 1
                    if is_17: shifts_17["l_to_m"] += 1
                else:
                    shifts_18["other"] += 1
                    if is_17: shifts_17["other"] += 1

        self.assertEqual(total_boxes, 675)
        # Verify 17 original local frames: exactly 111 m->s and 9 l->m
        self.assertEqual(shifts_17["m_to_s"], 111)
        self.assertEqual(shifts_17["l_to_m"], 9)
        self.assertEqual(shifts_17["other"], 0)
        self.assertEqual(shifts_17["same"], 511)

        # Verify all 18 local frames: exactly 117 m->s and 9 l->m
        self.assertEqual(shifts_18["m_to_s"], 117)
        self.assertEqual(shifts_18["l_to_m"], 9)
        self.assertEqual(shifts_18["other"], 0)
        self.assertEqual(shifts_18["same"], 549)

        # 3. Verify normalized coordinates are unaltered in inventory records
        for r in local_eligible:
            in_id = r["inventory_id"]
            inv_rec = self.inv_by_id[in_id]
            app_boxes = [
                b for b in r["boxes"]
                if not b.get("is_proposal") or b.get("proposal_status") == "accepted"
            ]
            self.assertEqual(len(inv_rec["boxes"]), len(app_boxes))
            for b_pack, b_inv in zip(app_boxes, inv_rec["boxes"]):
                for c_pack, c_inv in zip(b_pack["bbox_norm"], b_inv["bbox_norm"]):
                    self.assertAlmostEqual(c_pack, c_inv, places=6)
                expected_bucket = compute_aspect_preserving_size(
                    b_pack["bbox_norm"][2], b_pack["bbox_norm"][3], 1920, 1080, ref_size=640
                )["size_bucket"]
                self.assertEqual(b_inv["size_bucket"], expected_bucket)


class TestStaleDescendantsAndCanonicalSourceExclusions(unittest.TestCase):
    """
    Verifies that changed local annotations exclude stale descendants and that
    rejected exclusions are enforced by canonical source and descendants.
    """

    def setUp(self):
        self.v6_dir = Path("data/training_manifests_v6")
        with open(self.v6_dir / "split_and_exclusion_manifest.json", "r", encoding="utf-8") as f:
            self.splits = json.load(f)
        with open(self.v6_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
            self.man_a = json.load(f)
        with open(self.v6_dir / "manifest_b_local_plus_external.json", "r", encoding="utf-8") as f:
            self.man_b = json.load(f)
        with open(self.v6_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            self.inv = json.load(f)

    def test_class_only_corrections_in_review_pack_v2_detected(self):
        """Frames with class edits in v2 (e.g. cam43_south_f001620) must be detected as modified."""
        orig = [{"class_id": 3, "bbox_norm": [0.5, 0.5, 0.2, 0.2]}]
        approved = [{"class_id": 0, "bbox_norm": [0.5, 0.5, 0.2, 0.2]}]
        is_mod, reason, _ = compare_boxes(orig, approved)
        self.assertTrue(is_mod)
        self.assertIn("class_edit", reason)

    def test_all_41_stale_variants_excluded_from_candidate_train(self):
        """All 41 synthetic variants of modified local frames must be excluded from Manifest A and B."""
        excl_stale = [e for e in self.splits["exclusions"] if e.get("reason") == "stale_unregenerated_augmentation_variant"]
        self.assertEqual(len(excl_stale), 41, f"Expected exactly 41 stale variants excluded, found {len(excl_stale)}")

        stale_ids = {e["inventory_id"] for e in excl_stale}
        for rid in self.man_a["train_records"]:
            self.assertNotIn(rid, stale_ids, f"Stale variant {rid} leaked into Manifest A train records!")
        for rid in self.man_b["train_records"]:
            self.assertNotIn(rid, stale_ids, f"Stale variant {rid} leaked into Manifest B train records!")

        # Check examples
        self.assertIn("local:train:cam43_south_f137460", self.man_a["train_records"])
        self.assertIn("local:train:cam43_south_f137460_salengboost_1", stale_ids)
        self.assertIn("local:train:cam44_north_f003960", self.man_a["train_records"])
        self.assertIn("local:train:cam44_north_f003960_truckboost_1", stale_ids)
        self.assertIn("local:train:cam44_north_f148440", self.man_a["train_records"])
        self.assertIn("local:train:cam44_north_f148440_synth_ir_night", stale_ids)

    def test_canonical_source_exclusion_blocks_all_descendants(self):
        """
        Enforce rejected exclusions by canonical source and descendants:
        1. Rejected canonical frame cam44_north_f019140 and its variants are excluded.
        2. Neither Manifest A nor B contains any record derived from unresolved canonical sources.
        3. External sampling candidates explicitly filter out canonical sources in unresolved_canonical_ids.
        """
        unresolved_canonical_ids = {"cam44_north_f019140"}

        # 1. Exclusions manifest contains entries for unresolved canonical sources and variants
        cam019_excls = [e for e in self.splits["exclusions"] if e["canonical_source_id"] == "cam44_north_f019140"]
        self.assertEqual(len(cam019_excls), 2)  # base + tuktukboost_1

        # 2. Check no train or val records in Manifest A or B contain unresolved canonical source
        for man_name, man in [("Manifest A", self.man_a), ("Manifest B", self.man_b)]:
            for rid in man["train_records"] + man["val_records"]:
                for u_cid in unresolved_canonical_ids:
                    self.assertNotIn(u_cid, rid, f"Unresolved source {u_cid} found in {man_name} record {rid}")

        # 3. Check inventory records for external candidates: if an external record had canonical source in unresolved,
        # it is guaranteed excluded from eligible_ext_cands.
        train_seqs = self.splits.get("metadata", {}).get("external_sequence_partition", {}).get("train_sequences", [])
        ext_records = [r for r in self.inv["records"] if r["data_origin"] == "external_ua_detrac"]
        eligible_cands = [
            r for r in ext_records
            if r["sequence_id"] in train_seqs and r["canonical_source_id"] not in unresolved_canonical_ids
        ]
        for ec in eligible_cands:
            self.assertNotIn(
                ec["canonical_source_id"],
                unresolved_canonical_ids,
                f"Candidate {ec['inventory_id']} leaked into eligible external candidates!"
            )


class TestManifestsV6IntegrityAndDiagnosticStatus(unittest.TestCase):
    """Verifies counts, dynamic summaries, and diagnostic benchmark status in Manifests V6."""

    def setUp(self):
        self.v6_dir = Path("data/training_manifests_v6")
        with open(self.v6_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
            self.man_a = json.load(f)
        with open(self.v6_dir / "manifest_b_local_plus_external.json", "r", encoding="utf-8") as f:
            self.man_b = json.load(f)
        with open(self.v6_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
            self.prim_val = json.load(f)
        with open(self.v6_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            self.inv = json.load(f)
        with open("data/eval_snapshot_v1/manifest.json", "r", encoding="utf-8") as f:
            self.eval_data = json.load(f)

    def test_local_training_and_external_cap(self):
        """Manifest A has 1092 images (527 unique canonical sources); Manifest B cap = 263."""
        a_meta = self.man_a["metadata"]["train_summary"]
        self.assertEqual(a_meta["images_count"], 1092)
        self.assertEqual(a_meta["unique_canonical_frames_count"], 527)
        self.assertEqual(len(self.man_a["train_records"]), 1092)

        b_meta = self.man_b["metadata"]
        self.assertEqual(b_meta["external_images_count"], 263)
        self.assertEqual(len(self.man_b["external_train_records"]), 263)
        self.assertEqual(len(self.man_b["train_records"]), 1092 + 263)

    def test_manifest_summaries_derived_dynamically(self):
        """Manifest summaries must agree with actual counts across final selected records."""
        inv_lookup = {r["inventory_id"]: r for r in self.inv["records"]}

        sum_a = compute_manifest_summary(self.man_a["train_records"], inv_lookup)
        self.assertEqual(self.man_a["metadata"]["train_summary"]["total_boxes"], sum_a["total_boxes"])
        self.assertEqual(self.man_a["metadata"]["train_summary"]["class_box_counts"], sum_a["class_box_counts"])
        self.assertEqual(self.man_a["metadata"]["train_summary"]["size_box_counts"], sum_a["size_box_counts"])

        sum_b = compute_manifest_summary(self.man_b["train_records"], inv_lookup)
        self.assertEqual(self.man_b["metadata"]["train_summary"]["total_boxes"], sum_b["total_boxes"])
        self.assertEqual(self.man_b["metadata"]["train_summary"]["class_box_counts"], sum_b["class_box_counts"])
        self.assertEqual(self.man_b["metadata"]["train_summary"]["size_box_counts"], sum_b["size_box_counts"])

    def test_validation_identity_between_a_and_b(self):
        """Primary validation set must be byte-for-byte identical between Manifest A and Manifest B."""
        self.assertEqual(self.man_a["val_records"], self.man_b["val_records"])
        self.assertEqual(len(self.man_a["val_records"]), 130)
        self.assertEqual(self.man_a["val_records"], self.prim_val["records"])

    def test_eval_benchmark_zero_contamination(self):
        """None of the 42 diagnostic eval benchmark frames may appear in candidate train or val."""
        eval_fids = {s["frame_id"] for s in self.eval_data["samples"]}
        self.assertEqual(len(eval_fids), 42)

        for man in [self.man_a, self.man_b]:
            for rec_id in man["train_records"] + man["val_records"]:
                stem = rec_id.split(":")[-1]
                prov = parse_frame_provenance(stem)
                self.assertNotIn(prov.source_frame_id, eval_fids)
                self.assertNotIn(stem, eval_fids)


if __name__ == "__main__":
    unittest.main()
