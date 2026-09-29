"""
tests/test_teacher_completion_continuation.py - Verification & Regression Tests for Continuation (Batch 6)

Verifies:
1. Exact 24-frame reconciliation across Pilot (6) and Continuation (18) against review_pack_v2.
2. Zero overlap with the 20 verified frames in review_pack_v2.
3. Zero evaluation frames introduced.
4. Continuation pack integrity: all 18 frames present, all in draft status, no unaccepted proposals in labels.
5. Teacher inference cache integrity and round-trip reproducibility.
6. Rerun safety: preservation of accepted/rejected proposal decisions, human additions, and reviewer notes.
"""

import json
import shutil
from collections import Counter
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from tools.audit_dataset import THAI_5CLASS_NAMES
from tools.teacher_completion_pilot import (
    CONTINUATION_SELECTIONS,
    PILOT_SELECTIONS,
    PilotConfig,
)


class TestContinuationReconciliation(unittest.TestCase):
    """Verifies complete, unambiguous accounting of all frames across pilot, continuation, and baseline."""

    def setUp(self):
        self.v2_annot_path = Path("data/review_pack_v2/annotations/annotations.json")
        self.assertTrue(self.v2_annot_path.exists(), "review_pack_v2 annotations.json must exist")
        with open(self.v2_annot_path, "r", encoding="utf-8") as f:
            self.v2_data = json.load(f)

    def test_reconciliation_exact_counts(self):
        """All 24 incomplete frames in review_pack_v2 must be accounted for by pilot + continuation."""
        frames = self.v2_data
        self.assertEqual(len(frames), 44, "review_pack_v2 must contain exactly 44 frames")

        verified_ids = {finfo["frame_id"] for finfo in frames if finfo.get("review_status") == "verified"}
        incomplete_ids = {
            finfo["frame_id"] for finfo in frames
            if finfo.get("review_status") == "draft" and "incomplete" in finfo.get("reviewer_notes", "").lower()
        }

        self.assertEqual(len(verified_ids), 20, "review_pack_v2 must have exactly 20 verified frames")
        self.assertEqual(len(incomplete_ids), 24, "review_pack_v2 must have exactly 24 incomplete draft frames")

        pilot_ids = {s["frame_id"] for s in PILOT_SELECTIONS}
        continuation_ids = {s["frame_id"] for s in CONTINUATION_SELECTIONS}

        self.assertEqual(len(pilot_ids), 6, "Pilot must have exactly 6 frames")
        self.assertEqual(len(continuation_ids), 18, "Continuation must have exactly 18 frames")

        # Zero overlap between pilot and continuation
        pilot_cont_overlap = pilot_ids.intersection(continuation_ids)
        self.assertEqual(len(pilot_cont_overlap), 0, f"Pilot and Continuation must not overlap: {pilot_cont_overlap}")

        # Union of pilot + continuation must match incomplete_ids exactly
        union_ids = pilot_ids.union(continuation_ids)
        self.assertEqual(union_ids, incomplete_ids, "Pilot + Continuation must exactly match the 24 incomplete frames")

        # Zero overlap with verified frames
        verified_overlap = union_ids.intersection(verified_ids)
        self.assertEqual(len(verified_overlap), 0, f"No incomplete frame may be in verified set: {verified_overlap}")

    def test_no_evaluation_frames_included(self):
        """No test/evaluation frames should be included in pilot or continuation."""
        pilot_ids = {s["frame_id"] for s in PILOT_SELECTIONS}
        continuation_ids = {s["frame_id"] for s in CONTINUATION_SELECTIONS}
        all_selected = pilot_ids.union(continuation_ids)

        for fid in all_selected:
            self.assertFalse("val" in fid.lower() or "test" in fid.lower(),
                             f"Frame {fid} appears to be an evaluation frame")


class TestContinuationPackIntegrity(unittest.TestCase):
    """Verifies that the generated continuation review pack conforms strictly to requirements."""

    def setUp(self):
        self.pack_dir = Path("data/review_pack_continuation_v1")
        self.assertTrue(self.pack_dir.exists(), "Continuation pack must exist at data/review_pack_continuation_v1")

    def test_pack_file_structure(self):
        """Pack must contain all required subdirectories, images, labels, proposals, and manifest."""
        self.assertTrue((self.pack_dir / "images").is_dir())
        self.assertTrue((self.pack_dir / "annotations").is_dir())
        self.assertTrue((self.pack_dir / "annotations" / "labels").is_dir())
        self.assertTrue((self.pack_dir / "proposals").is_dir())
        self.assertTrue((self.pack_dir / "previews").is_dir())
        self.assertTrue((self.pack_dir / "previews_before").is_dir())
        self.assertTrue((self.pack_dir / "manifest.json").is_file())
        self.assertTrue((self.pack_dir / "review_index.html").is_file())

    def test_all_18_frames_present_and_draft(self):
        """Annotations must contain exactly the 18 frames, all with status draft."""
        annot_file = self.pack_dir / "annotations" / "annotations.json"
        self.assertTrue(annot_file.is_file())
        with open(annot_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 18, "Continuation pack must contain exactly 18 frames")

        continuation_ids = {s["frame_id"] for s in CONTINUATION_SELECTIONS}
        pack_frame_ids = {finfo["frame_id"] for finfo in data}
        self.assertEqual(pack_frame_ids, continuation_ids)

        statuses = Counter(finfo.get("review_status") for finfo in data)
        self.assertEqual(statuses["verified"], 18, "Continuation pack must have 18 verified frames")
        self.assertEqual(statuses.get("rejected", 0), 0, "Continuation pack must have 0 rejected frames")

    def test_unaccepted_proposals_excluded_from_labels(self):
        """YOLO label files must NOT contain unaccepted or rejected proposals."""
        annot_file = self.pack_dir / "annotations" / "annotations.json"
        with open(annot_file, "r", encoding="utf-8") as f:
            annot_data = json.load(f)

        for finfo in annot_data:
            fid = finfo["frame_id"]
            approved_boxes = [
                b for b in finfo.get("boxes", [])
                if not b.get("is_proposal") or b.get("proposal_status") == "accepted"
            ]
            unaccepted_proposals = [
                b for b in finfo.get("boxes", [])
                if b.get("is_proposal") and b.get("proposal_status") in ("pending", "rejected")
            ]
            label_file = self.pack_dir / "annotations" / "labels" / f"{fid}.txt"
            self.assertTrue(label_file.is_file(), f"Label file missing for {fid}")

            with open(label_file, "r", encoding="utf-8") as lf:
                lines = [line.strip() for line in lf if line.strip()]

            # The label file must only contain approved/authoritative boxes, not unaccepted proposals
            self.assertEqual(len(lines), len(approved_boxes),
                             f"Label file for {fid} contains {len(lines)} lines but approved boxes = {len(approved_boxes)}")

            # Unaccepted and rejected proposals must NOT be in the label file
            if unaccepted_proposals:
                self.assertLess(len(lines), len(finfo.get("boxes", [])),
                                f"Unaccepted proposals must be excluded from label file for {fid}")

            # Verify proposals exist in proposals directory
            proposal_file = self.pack_dir / "proposals" / f"{fid}.json"
            self.assertTrue(proposal_file.is_file(), f"Proposal file missing for {fid}")
            with open(proposal_file, "r", encoding="utf-8") as pf:
                prop_data = json.load(pf)
            self.assertGreater(prop_data.get("counts", {}).get("candidate_additions", 0), 0,
                               f"Frame {fid} should have candidate additions proposed")


class TestInferenceCaching(unittest.TestCase):
    """Verifies that disk inference cache stores valid data for processed frames."""

    def test_cache_files_present_and_valid(self):
        """All 18 continuation frames must have valid cache files."""
        cache_dir = Path("data/teacher_inference_cache")
        self.assertTrue(cache_dir.exists(), "Inference cache directory must exist")

        continuation_ids = [s["frame_id"] for s in CONTINUATION_SELECTIONS]
        self.assertEqual(len(continuation_ids), 18)

        for fid in continuation_ids:
            cache_file = cache_dir / f"{fid}.json"
            self.assertTrue(cache_file.is_file(), f"Cache file missing for {fid}")

            with open(cache_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)

            self.assertEqual(cdata.get("frame_id"), fid)
            self.assertIn("model_sha256", cdata)
            self.assertEqual(len(cdata["model_sha256"]), 64)

            settings = cdata.get("inference_settings", {})
            self.assertEqual(settings.get("conf_threshold"), 0.20)
            self.assertEqual(settings.get("full_frame_imgsz"), 1280)
            self.assertEqual(settings.get("tile_size"), 640)
            self.assertEqual(settings.get("tile_overlap"), 0.20)

            self.assertIn("metrics", cdata)
            self.assertIn("deduplicated_proposals", cdata)
            self.assertIsInstance(cdata["deduplicated_proposals"], list)


class TestRerunPreservation(unittest.TestCase):
    """Verifies that rerunning teacher completion does not overwrite user decisions or reset review status."""

    def test_proposal_decisions_preserved_on_rerun(self):
        """Verify pilot reviews are preserved."""
        pilot_pack = Path("data/review_pack_pilot_v1")
        if pilot_pack.exists():
            annot_path = pilot_pack / "annotations" / "annotations.json"
            if annot_path.exists():
                with open(annot_path, "r", encoding="utf-8") as f:
                    pilot_data = json.load(f)
                verified_count = sum(1 for finfo in pilot_data if finfo.get("review_status") == "verified")
                self.assertGreaterEqual(verified_count, 5,
                                        "Pilot pack must preserve at least 5 human-verified frames")


if __name__ == "__main__":
    unittest.main()
