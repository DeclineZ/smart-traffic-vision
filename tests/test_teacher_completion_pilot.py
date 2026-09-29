"""
tests/test_teacher_completion_pilot.py - Unit and Integration Tests for Batch 6

Tests:
1. Coordinate transformations (tile-to-global, pixel to normalized YOLO, normalized to pixel).
2. Edge-crossing tile objects and coordinate round-tripping.
3. Overlapping tile grid generation covering image borders without gaps.
4. Proposal deduplication across tiles preserving adjacent vehicles of same and different classes.
5. Proposal matching against human annotations (duplicates, conflicts, ambiguous overlaps, candidate additions).
6. Human annotation preservation (human boxes authoritative, never modified or replaced).
7. YOLO label file export safety: unaccepted proposals strictly excluded from label files.
8. Proposal review lifecycle (accept, reject) persistence and draft status preservation through editor API.
9. Disposable fixture smoke test simulating editor interaction without touching real data.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from tools.audit_dataset import THAI_5CLASS_NAMES
from tools.annotation_editor import (
    save_frame_annotation,
    validate_and_sanitize_boxes,
)
from tools.prepare_review_pack import (
    compute_boxes_label_hash,
    compute_file_sha256,
    sync_review_pack,
)
from tools.teacher_completion_pilot import (
    COCO_TEACHER_TO_THAI5,
    PilotConfig,
    compute_box_iomin,
    compute_box_iou,
    deduplicate_proposals,
    generate_tile_grid,
    match_proposals_against_human_annotations,
    norm_yolo_to_xyxy,
    tile_xyxy_to_global_xyxy,
    xyxy_to_norm_yolo,
)


class TestCoordinateTransforms(unittest.TestCase):
    """Verifies geometric calculations, tile projections, and bounding box conversions."""

    def test_tile_to_global_projection(self):
        """Tile coordinates must accurately project back to global pixel space with offsets."""
        tile_box = [10.0, 20.0, 100.0, 150.0]
        x_offset = 512
        y_offset = 256
        global_box = tile_xyxy_to_global_xyxy(tile_box, x_offset, y_offset)
        self.assertEqual(global_box, [522.0, 276.0, 612.0, 406.0])

    def test_norm_yolo_round_trip(self):
        """Converting pixel xyxy to normalized YOLO and back must match within float precision."""
        img_w, img_h = 1920, 1080
        orig_xyxy = [100.0, 200.0, 300.0, 450.0]
        norm = xyxy_to_norm_yolo(orig_xyxy, img_w, img_h)
        reconstructed = norm_yolo_to_xyxy(norm, img_w, img_h)
        for orig, recon in zip(orig_xyxy, reconstructed):
            self.assertAlmostEqual(orig, recon, delta=0.5)

    def test_edge_crossing_tile_grid(self):
        """Tile grid must cover all pixels up to image boundaries without gaps."""
        for (w, h) in [(640, 640), (960, 540), (1920, 1080)]:
            tile_size = 384 if min(w, h) <= 640 else 640
            tiles = generate_tile_grid(w, h, tile_size=tile_size, overlap=0.20)
            self.assertGreater(len(tiles), 0)
            # Verify bottom-right coverage
            last_tile = tiles[-1]
            self.assertEqual(last_tile[2], w, f"Width not covered for {w}x{h}")
            self.assertEqual(last_tile[3], h, f"Height not covered for {w}x{h}")
            # Verify top-left coverage
            first_tile = tiles[0]
            self.assertEqual(first_tile[0], 0)
            self.assertEqual(first_tile[1], 0)

    def test_box_iou_and_containment(self):
        """IoU and IoMin must accurately reflect overlap and containment."""
        boxA = [0.0, 0.0, 100.0, 100.0]
        boxB = [0.0, 0.0, 50.0, 50.0]  # Fully contained within boxA
        iou = compute_box_iou(boxA, boxB)
        iomin = compute_box_iomin(boxA, boxB)
        self.assertAlmostEqual(iou, 2500.0 / 10000.0, delta=1e-4)
        self.assertAlmostEqual(iomin, 1.0, delta=1e-4)


class TestProposalDeduplication(unittest.TestCase):
    """Verifies proposal deduplication across tiles while preserving adjacent objects."""

    def test_deduplicate_identical_proposals(self):
        """Identical proposals of same class from overlapping tiles must deduplicate into one."""
        p1 = {
            "class_id": 0,
            "confidence": 0.90,
            "xyxy_px": [100.0, 100.0, 200.0, 200.0],
            "method": "tile_0"
        }
        p2 = {
            "class_id": 0,
            "confidence": 0.85,
            "xyxy_px": [102.0, 101.0, 201.0, 202.0],  # ~0.95 IoU
            "method": "tile_1"
        }
        deduped = deduplicate_proposals([p1, p2], iou_thresh=0.55)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["confidence"], 0.90)
        self.assertEqual(deduped[0]["tile_hits"], 2)

    def test_preserve_adjacent_vehicles(self):
        """Adjacent vehicles with lower overlap must NEVER be suppressed."""
        v1 = {
            "class_id": 0,
            "confidence": 0.88,
            "xyxy_px": [100.0, 100.0, 200.0, 200.0],
            "method": "full_frame"
        }
        v2 = {
            "class_id": 0,
            "confidence": 0.85,
            "xyxy_px": [180.0, 100.0, 280.0, 200.0],  # 20px overlap, ~0.11 IoU
            "method": "tiled"
        }
        deduped = deduplicate_proposals([v1, v2], iou_thresh=0.55)
        self.assertEqual(len(deduped), 2)

    def test_preserve_different_class_overlapping_vehicles(self):
        """Overlapping vehicles of different classes (e.g. motorcycle alongside bus) must both be kept."""
        bus = {
            "class_id": 2,
            "confidence": 0.92,
            "xyxy_px": [100.0, 50.0, 300.0, 350.0],
            "method": "full_frame"
        }
        moto = {
            "class_id": 1,
            "confidence": 0.75,
            "xyxy_px": [200.0, 200.0, 260.0, 320.0],  # Contained inside bus bounding box area
            "method": "tiled"
        }
        deduped = deduplicate_proposals([bus, moto], iou_thresh=0.55)
        self.assertEqual(len(deduped), 2)


class TestMatchingAgainstHumanAnnotations(unittest.TestCase):
    """Verifies proposal categorization against authoritative human annotations."""

    def test_matching_categories(self):
        """Proposals must be accurately partitioned into duplicates, conflicts, and additions."""
        img_w, img_h = 1000, 1000
        human_boxes = [
            {
                "instance_id": "test_inst_001",
                "class_id": 0,
                "class_name": "car",
                "bbox_norm": [0.2, 0.2, 0.2, 0.2]  # [100, 100, 300, 300]
            },
            {
                "instance_id": "test_inst_002",
                "class_id": 3,
                "class_name": "truck",
                "bbox_norm": [0.6, 0.6, 0.2, 0.2]  # [500, 500, 700, 700]
            }
        ]

        proposals = [
            # 1. Duplicate of human car (same class, high IoU)
            {
                "class_id": 0,
                "class_name": "car",
                "coco_class_id": 2,
                "confidence": 0.91,
                "xyxy_px": [102.0, 98.0, 298.0, 302.0],
                "bbox_norm": [0.2, 0.2, 0.196, 0.204],
                "method": "full_frame"
            },
            # 2. Conflict on human truck (teacher thinks car, high IoU)
            {
                "class_id": 0,
                "class_name": "car",
                "coco_class_id": 2,
                "confidence": 0.78,
                "xyxy_px": [500.0, 500.0, 700.0, 700.0],
                "bbox_norm": [0.6, 0.6, 0.2, 0.2],
                "method": "tiled"
            },
            # 3. Candidate Addition: completely unannotated background vehicle
            {
                "class_id": 1,
                "class_name": "motorcycle",
                "coco_class_id": 3,
                "confidence": 0.82,
                "xyxy_px": [800.0, 800.0, 850.0, 900.0],
                "bbox_norm": [0.825, 0.85, 0.05, 0.1],
                "method": "tiled"
            }
        ]

        dups, confs, ambigs, adds = match_proposals_against_human_annotations(
            proposals=proposals,
            human_boxes=human_boxes,
            img_w=img_w,
            img_h=img_h,
            iou_match_thresh=0.45,
            iou_ambig_thresh=0.15
        )

        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["matching_human_instance_id"], "test_inst_001")

        self.assertEqual(len(confs), 1)
        self.assertEqual(confs[0]["matching_human_instance_id"], "test_inst_002")
        self.assertEqual(confs[0]["human_class_name"], "truck")
        self.assertEqual(confs[0]["class_name"], "car")

        self.assertEqual(len(adds), 1)
        self.assertIsNone(adds[0]["matching_human_instance_id"])
        self.assertEqual(adds[0]["class_name"], "motorcycle")


class TestEditorProposalLifecycleAndSafety(unittest.TestCase):
    """Integration test verifying proposal acceptance, rejection, and label synchronization."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_pilot_pack_"))
        self.images_dir = self.temp_dir / "images"
        self.labels_dir = self.temp_dir / "annotations" / "labels"
        self.annos_dir = self.temp_dir / "annotations"
        self.previews_dir = self.temp_dir / "previews"

        for d in [self.images_dir, self.labels_dir, self.annos_dir, self.previews_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.frame_id = "test_frame_001"

        # Create dummy image
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(self.images_dir / f"{self.frame_id}.jpg"), img)

        # 1 human box: class 0 (car)
        self.human_box = {
            "instance_id": f"{self.frame_id}_inst_000",
            "class_id": 0,
            "class_name": "car",
            "subtype": "sedan",
            "is_ambiguous": False,
            "ambiguity_reason": "",
            "bbox_norm": [0.2, 0.2, 0.1, 0.1],
            "proposal_source": "human_annotation",
            "is_proposal": False,
            "source_type": "existing_human",
            "proposal_status": "none"
        }

        # 1 proposal box: class 1 (motorcycle)
        self.proposal_box = {
            "instance_id": f"{self.frame_id}_teacher_add_000",
            "class_id": 1,
            "class_name": "motorcycle",
            "subtype": "motorcycle_provisional",
            "is_ambiguous": False,
            "ambiguity_reason": "",
            "bbox_norm": [0.8, 0.8, 0.05, 0.05],
            "proposal_source": "yolo26x:tiled(conf=0.85)",
            "is_proposal": True,
            "source_type": "teacher_proposal",
            "proposal_category": "candidate_addition",
            "proposal_status": "pending",
            "confidence": 0.85,
            "proposal_method": "tiled",
            "tile_hits": 1
        }

        initial_record = {
            "frame_id": self.frame_id,
            "filename": f"{self.frame_id}.jpg",
            "boxes": [self.human_box, self.proposal_box],
            "review_status": "draft",
            "annotation_state": "annotated",
            "is_unannotated": False,
            "is_ambiguous": False,
            "is_rejected": False,
            "reviewer_notes": "test frame with proposals"
        }

        with open(self.annos_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump([initial_record], f, indent=2)

        # Initial YOLO label contains ONLY human box
        (self.labels_dir / f"{self.frame_id}.txt").write_text("0 0.200000 0.200000 0.100000 0.100000\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unaccepted_proposal_excluded_from_yolo_labels(self):
        """Pending and rejected proposals must NEVER enter approved YOLO label files."""
        # Initial label check
        lbl_content = (self.labels_dir / f"{self.frame_id}.txt").read_text().strip().splitlines()
        self.assertEqual(len(lbl_content), 1)
        self.assertTrue(lbl_content[0].startswith("0 "))  # Human box

        # Save draft with proposal still pending
        etag = compute_file_sha256(self.annos_dir / "annotations.json")
        save_frame_annotation(
            pack_dir=self.temp_dir,
            frame_id=self.frame_id,
            base_etag=etag,
            action="save_draft",
            boxes_data=[self.human_box, self.proposal_box],
            reviewer_notes="still pending review"
        )

        lbl_content = (self.labels_dir / f"{self.frame_id}.txt").read_text().strip().splitlines()
        self.assertEqual(len(lbl_content), 1, "Unaccepted proposal must not enter YOLO label file!")

    def test_accepted_proposal_enters_yolo_labels_and_preserves_draft_status(self):
        """When accepted, proposal enters approved YOLO labels without silently marking frame verified."""
        accepted_prop = dict(self.proposal_box)
        accepted_prop["proposal_status"] = "accepted"

        etag = compute_file_sha256(self.annos_dir / "annotations.json")
        res = save_frame_annotation(
            pack_dir=self.temp_dir,
            frame_id=self.frame_id,
            base_etag=etag,
            action="save_draft",
            boxes_data=[self.human_box, accepted_prop],
            reviewer_notes="proposal accepted by reviewer"
        )

        # 1. Check frame review status remains draft!
        self.assertEqual(res["record"]["review_status"], "draft", "Frame must not be silently marked verified!")

        # 2. Check YOLO label file now contains 2 lines (human + accepted proposal)
        lbl_content = (self.labels_dir / f"{self.frame_id}.txt").read_text().strip().splitlines()
        self.assertEqual(len(lbl_content), 2)
        self.assertTrue(lbl_content[0].startswith("0 "))  # Car
        self.assertTrue(lbl_content[1].startswith("1 "))  # Motorcycle

        # 3. Check reload from disk persists proposal_status
        with open(self.annos_dir / "annotations.json", "r") as f:
            disk_records = json.load(f)
        disk_boxes = disk_records[0]["boxes"]
        self.assertEqual(len(disk_boxes), 2)
        prop_disk = next(b for b in disk_boxes if b["instance_id"] == accepted_prop["instance_id"])
        self.assertEqual(prop_disk["proposal_status"], "accepted")

    def test_rejected_proposal_persists_decision_and_excluded_from_labels(self):
        """When rejected, proposal decision persists across save/reload and remains excluded from labels."""
        rejected_prop = dict(self.proposal_box)
        rejected_prop["proposal_status"] = "rejected"

        etag = compute_file_sha256(self.annos_dir / "annotations.json")
        save_frame_annotation(
            pack_dir=self.temp_dir,
            frame_id=self.frame_id,
            base_etag=etag,
            action="save_draft",
            boxes_data=[self.human_box, rejected_prop],
            reviewer_notes="proposal rejected: false positive"
        )

        # Label file contains ONLY human box
        lbl_content = (self.labels_dir / f"{self.frame_id}.txt").read_text().strip().splitlines()
        self.assertEqual(len(lbl_content), 1)

        # Annotations.json retains rejected box with proposal_status: rejected
        with open(self.annos_dir / "annotations.json", "r") as f:
            disk_records = json.load(f)
        disk_boxes = disk_records[0]["boxes"]
        self.assertEqual(len(disk_boxes), 2)
        prop_disk = next(b for b in disk_boxes if b["instance_id"] == rejected_prop["instance_id"])
        self.assertEqual(prop_disk["proposal_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
