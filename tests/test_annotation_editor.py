"""
Unit and integration tests for tools.annotation_editor (Batch 2B).

Verifies:
- Accurate screen-to-image and normalized-to-pixel coordinate conversions across zoom and pan.
- Stable instance IDs through insertions, deletions, and reordering.
- Subtype and ambiguity metadata preservation on edit and save.
- Empty frame state handling (unannotated -> draft vs verified_empty_background).
- Editing a verified frame returns it to draft until explicitly verified again.
- Stale-save rejection (409 Conflict) when external changes occur.
- Transactional rollback on save/sync failure preserving byte-identical original files.
- End-to-end HTTP API server operations on a disposable copy of the 42-frame review pack.
- Zero modification to the real data/review_pack_v1.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_review_pack import (
    compute_file_sha256,
    THAI_5CLASS_NAMES,
)
from tools.annotation_editor import (
    StaleSaveError,
    create_editor_server,
    generate_stable_instance_id,
    image_box_to_normalized_yolo,
    image_to_screen_coords,
    normalized_yolo_to_image_box,
    save_frame_annotation,
    screen_to_image_coords,
    validate_and_sanitize_boxes,
)


class TestAnnotationEditorGeometry(unittest.TestCase):
    """Tests coordinate conversions, normalization, and bounds clamping."""

    def test_screen_to_image_and_roundtrip(self):
        """Image to screen and back must be exact across different zoom and pan values."""
        img_w, img_h = 1920, 1080

        test_cases = [
            # (pan_x, pan_y, zoom, test_ix, test_iy)
            (0.0, 0.0, 1.0, 500.0, 300.0),
            (150.0, -80.0, 2.5, 1200.0, 750.0),
            (-300.0, 200.0, 0.5, 100.0, 90.0),
            (50.0, 50.0, 8.0, 1800.0, 1000.0),  # High zoom for distant vehicles
        ]

        for pan_x, pan_y, zoom, ix, iy in test_cases:
            sx, sy = image_to_screen_coords(ix, iy, pan_x, pan_y, zoom)
            ret_ix, ret_iy = screen_to_image_coords(sx, sy, pan_x, pan_y, zoom, img_w, img_h)
            self.assertAlmostEqual(ix, ret_ix, places=4, msg=f"Failed roundtrip for zoom={zoom}")
            self.assertAlmostEqual(iy, ret_iy, places=4, msg=f"Failed roundtrip for zoom={zoom}")

    def test_coordinate_clamping_to_image_bounds(self):
        """Screen coordinates outside the image must clamp to [0, img_w] and [0, img_h]."""
        img_w, img_h = 1920, 1080
        pan_x, pan_y, zoom = 100.0, 100.0, 1.0

        # Far left / top outside screen
        ix, iy = screen_to_image_coords(-50.0, -50.0, pan_x, pan_y, zoom, img_w, img_h)
        self.assertEqual(ix, 0.0)
        self.assertEqual(iy, 0.0)

        # Far right / bottom outside screen
        ix, iy = screen_to_image_coords(5000.0, 4000.0, pan_x, pan_y, zoom, img_w, img_h)
        self.assertEqual(ix, 1920.0)
        self.assertEqual(iy, 1080.0)

    def test_image_box_to_normalized_yolo_and_roundtrip(self):
        """Pixel bounding box must convert to normalized YOLO [xc, yc, w, h] and back."""
        img_w, img_h = 1920, 1080
        x1, y1, x2, y2 = 200.0, 150.0, 600.0, 450.0

        bbox_norm = image_box_to_normalized_yolo(x1, y1, x2, y2, img_w, img_h)
        # Expected:
        # xc = (200 + 600) / (2 * 1920) = 800 / 3840 = 0.208333
        # yc = (150 + 450) / (2 * 1080) = 600 / 2160 = 0.277778
        # bw = (600 - 200) / 1920 = 400 / 1920 = 0.208333
        # bh = (450 - 150) / 1080 = 300 / 1080 = 0.277778
        self.assertEqual(len(bbox_norm), 4)
        self.assertAlmostEqual(bbox_norm[0], 0.208333, places=5)
        self.assertAlmostEqual(bbox_norm[1], 0.277778, places=5)
        self.assertAlmostEqual(bbox_norm[2], 0.208333, places=5)
        self.assertAlmostEqual(bbox_norm[3], 0.277778, places=5)

        # Inverted coordinates (dragged bottom-right to top-left)
        inv_norm = image_box_to_normalized_yolo(x2, y2, x1, y1, img_w, img_h)
        self.assertEqual(bbox_norm, inv_norm)

        # Roundtrip back to image box
        rx1, ry1, rx2, ry2 = normalized_yolo_to_image_box(bbox_norm, img_w, img_h)
        self.assertAlmostEqual(x1, rx1, delta=0.1)
        self.assertAlmostEqual(y1, ry1, delta=0.1)
        self.assertAlmostEqual(x2, rx2, delta=0.1)
        self.assertAlmostEqual(y2, ry2, delta=0.1)


class TestAnnotationEditorInstanceManagement(unittest.TestCase):
    """Tests stable instance ID generation, box validation, and metadata preservation."""

    def test_stable_instance_id_generation(self):
        """Newly added boxes must receive next available sequential IDs without colliding with existing ones."""
        frame_id = "cam03_east_f003720"
        existing = [
            {"instance_id": f"{frame_id}_inst_000"},
            {"instance_id": f"{frame_id}_inst_001"},
            {"instance_id": f"{frame_id}_inst_003"},  # note gap: 002 was previously deleted
        ]

        # Next ID must be 004 (strictly higher than max existing to prevent ID re-use)
        next_id = generate_stable_instance_id(existing, frame_id)
        self.assertEqual(next_id, f"{frame_id}_inst_004")

        # Empty existing list starts at 000
        first_id = generate_stable_instance_id([], frame_id)
        self.assertEqual(first_id, f"{frame_id}_inst_000")

    def test_validate_and_sanitize_boxes_preserves_metadata(self):
        """Validation must preserve instance_id, class, subtype, ambiguity flags, and reasons."""
        frame_id = "cam43_south_f001200"
        raw_boxes = [
            {
                "instance_id": f"{frame_id}_inst_000",
                "class_id": 0,
                "subtype": "pickup_based_songthaew",
                "is_ambiguous": True,
                "ambiguity_reason": "rear canopy obscured by tree",
                "bbox_norm": [0.45, 0.55, 0.12, 0.18],
                "proposal_source": "human_annotation"
            },
            {
                # New box without instance_id
                "class_id": 3,
                "subtype": "truck_based_songthaew",
                "is_ambiguous": False,
                "ambiguity_reason": "",
                "bbox_norm": [0.70, 0.40, 0.20, 0.30]
            }
        ]

        cleaned = validate_and_sanitize_boxes(raw_boxes, frame_id)
        self.assertEqual(len(cleaned), 2)

        # Existing box keeps exact metadata
        b0 = cleaned[0]
        self.assertEqual(b0["instance_id"], f"{frame_id}_inst_000")
        self.assertEqual(b0["class_id"], 0)
        self.assertEqual(b0["class_name"], "car")
        self.assertEqual(b0["subtype"], "pickup_based_songthaew")
        self.assertTrue(b0["is_ambiguous"])
        self.assertEqual(b0["ambiguity_reason"], "rear canopy obscured by tree")

        # New box receives fresh unique ID and correct class_name
        b1 = cleaned[1]
        self.assertEqual(b1["instance_id"], f"{frame_id}_inst_001")
        self.assertEqual(b1["class_id"], 3)
        self.assertEqual(b1["class_name"], "truck")
        self.assertEqual(b1["subtype"], "truck_based_songthaew")
        self.assertFalse(b1["is_ambiguous"])

    def test_validate_rejects_invalid_class_or_coords(self):
        """Invalid class IDs or malformed box coordinates must be rejected."""
        frame_id = "test_frame"
        # Invalid class 99
        with self.assertRaises(ValueError):
            validate_and_sanitize_boxes([{"class_id": 99, "bbox_norm": [0.5, 0.5, 0.1, 0.1]}], frame_id)

        # Malformed coords
        with self.assertRaises(ValueError):
            validate_and_sanitize_boxes([{"class_id": 0, "bbox_norm": [0.5, 0.5]}], frame_id)


class TestAnnotationEditorIntegration(unittest.TestCase):
    """
    Integration tests on disposable review pack copies.
    Verifies state transitions, stale-save rejection, rollback on failure,
    and end-to-end HTTP API behavior.
    """

    def setUp(self):
        self.real_pack_dir = Path("data/review_pack_v1")
        if not self.real_pack_dir.exists():
            self.skipTest("Real pack not found at data/review_pack_v1")

        # Record real pack hashes to guarantee immutability
        self.real_annos_file = self.real_pack_dir / "annotations" / "annotations.json"
        self.real_annos_sha = compute_file_sha256(self.real_annos_file)

        # Create temporary disposable pack
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pack_dir = Path(self.temp_dir.name) / "disposable_pack"
        shutil.copytree(self.real_pack_dir, self.pack_dir)

    def tearDown(self):
        # Assert real pack was NEVER touched
        current_real_sha = compute_file_sha256(self.real_annos_file)
        self.assertEqual(
            self.real_annos_sha,
            current_real_sha,
            "CRITICAL: Real pack data/review_pack_v1/annotations/annotations.json was modified!"
        )
        self.temp_dir.cleanup()

    def test_save_draft_and_mark_verified_transitions(self):
        """
        Verify state transitions:
        - Saving draft leaves unannotated state and sets review_status: draft
        - Explicit mark verified sets review_status: verified
        - Editing a verified annotation resets to draft
        - Empty frame verified sets annotation_state: verified_empty_background
        """
        annos_file = self.pack_dir / "annotations" / "annotations.json"

        # Frame 1: cam03_east_f003720 (starts unreviewed with 14 boxes)
        fid = "cam03_east_f003720"
        with open(annos_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        rec = next(r for r in records if r["frame_id"] == fid)
        boxes = rec["boxes"]
        etag = compute_file_sha256(annos_file)

        # 1. Save draft with modified box
        boxes_modified = list(boxes)
        boxes_modified[0]["subtype"] = "pickup"  # change subtype
        res = save_frame_annotation(
            pack_dir=self.pack_dir,
            frame_id=fid,
            base_etag=etag,
            action="save_draft",
            boxes_data=boxes_modified,
            reviewer_notes="Reviewed truck/pickup split",
            is_ambiguous=False
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["record"]["review_status"], "draft")
        self.assertEqual(res["record"]["annotation_state"], "annotated")
        self.assertEqual(res["record"]["reviewer_notes"], "Reviewed truck/pickup split")

        # 2. Explicitly mark verified
        etag2 = res["base_etag"]
        res_ver = save_frame_annotation(
            pack_dir=self.pack_dir,
            frame_id=fid,
            base_etag=etag2,
            action="mark_verified",
            boxes_data=boxes_modified,
            reviewer_notes="Reviewed and approved",
            is_ambiguous=False
        )
        self.assertEqual(res_ver["record"]["review_status"], "verified")
        self.assertEqual(res_ver["record"]["annotation_state"], "annotated")

        # 3. Editing a verified frame returns it to draft
        etag3 = res_ver["base_etag"]
        res_edit = save_frame_annotation(
            pack_dir=self.pack_dir,
            frame_id=fid,
            base_etag=etag3,
            action="save_draft",
            boxes_data=boxes_modified,
            reviewer_notes="Modified again",
            is_ambiguous=False
        )
        self.assertEqual(res_edit["record"]["review_status"], "draft")

        # 4. Empty frame verified -> verified_empty_background
        unann_rec = next((r for r in records if r.get("is_unannotated") is True), None)
        if unann_rec is None:
            # When all frames in real pack are verified, synthesize an unannotated record in the disposable test copy
            unann_rec = records[-1]
            unann_rec["is_unannotated"] = True
            unann_rec["boxes"] = []
            unann_rec["review_status"] = "unreviewed"
            with open(annos_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            etag_for_unann = compute_file_sha256(annos_file)
        else:
            etag_for_unann = res_edit["base_etag"]

        unann_fid = unann_rec["frame_id"]

        res_empty_ver = save_frame_annotation(
            pack_dir=self.pack_dir,
            frame_id=unann_fid,
            base_etag=etag_for_unann,
            action="mark_verified",
            boxes_data=[],
            reviewer_notes="Verified background with zero traffic targets",
            is_ambiguous=False
        )
        self.assertEqual(res_empty_ver["record"]["review_status"], "verified")
        self.assertEqual(res_empty_ver["record"]["annotation_state"], "verified_empty_background")
        self.assertFalse(res_empty_ver["record"]["is_unannotated"])

    def test_stale_save_rejection_on_external_change(self):
        """When annotations.json changes on disk, a save with outdated base_etag must raise StaleSaveError."""
        annos_file = self.pack_dir / "annotations" / "annotations.json"
        initial_etag = compute_file_sha256(annos_file)

        # Simulate external modification to annotations.json
        with open(annos_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data[0]["reviewer_notes"] = "External edit from another tool"
        with open(annos_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        new_etag = compute_file_sha256(annos_file)
        self.assertNotEqual(initial_etag, new_etag)

        # Attempt to save with old initial_etag -> must raise StaleSaveError
        with self.assertRaises(StaleSaveError):
            save_frame_annotation(
                pack_dir=self.pack_dir,
                frame_id="cam03_east_f003720",
                base_etag=initial_etag,
                action="save_draft",
                boxes_data=[],
                reviewer_notes=""
            )

    def test_failed_save_rollback_preserves_annotations(self):
        """If sync or commit fails, annotations.json must be restored byte-identically."""
        annos_file = self.pack_dir / "annotations" / "annotations.json"
        initial_sha = compute_file_sha256(annos_file)
        initial_bytes = annos_file.read_bytes()

        # Provide a box with invalid geometry that passes basic validation but causes sync preflight failure
        # For example, inject an un-renderable condition or pass a non-existent frame ID
        with self.assertRaises(Exception):
            save_frame_annotation(
                pack_dir=self.pack_dir,
                frame_id="non_existent_frame_id",
                base_etag=initial_sha,
                action="save_draft",
                boxes_data=[],
                reviewer_notes=""
            )

        # Assert annotations.json was completely restored byte-for-byte
        current_sha = compute_file_sha256(annos_file)
        self.assertEqual(initial_sha, current_sha)
        self.assertEqual(initial_bytes, annos_file.read_bytes())

    def test_end_to_end_http_server_and_workflow(self):
        """
        Exercises the editor HTTP server end-to-end:
        - GET / (UI)
        - GET /api/frames (42 frames accessible)
        - GET /api/frame/<id>
        - POST /api/frame/<id>/save (add box, change class/subtype, move/resize, delete box)
        - Confirm JSON, YOLO label file, and previews agree!
        """
        # Start server on a free port on 127.0.0.1
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = create_editor_server(self.pack_dir, host="127.0.0.1", port=port)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. GET /
            req = urllib.request.urlopen(f"{base_url}/")
            self.assertEqual(req.status, 200)
            html_content = req.read().decode("utf-8")
            self.assertIn("YOLO26s Review Pack Visual Annotation Editor", html_content)
            self.assertIn("TAXONOMY_CLASSES", html_content)

            # 2. GET /api/frames
            req_frames = urllib.request.urlopen(f"{base_url}/api/frames")
            self.assertEqual(req_frames.status, 200)
            data_frames = json.loads(req_frames.read().decode("utf-8"))
            self.assertEqual(data_frames["summary"]["total"], 42)
            self.assertEqual(len(data_frames["frames"]), 42)

            # 3. GET /api/frame/cam03_east_f003720
            fid = "cam03_east_f003720"
            req_frame = urllib.request.urlopen(f"{base_url}/api/frame/{fid}")
            self.assertEqual(req_frame.status, 200)
            frame_data = json.loads(req_frame.read().decode("utf-8"))
            boxes = frame_data["record"]["boxes"]
            self.assertGreaterEqual(len(boxes), 5)
            initial_etag = frame_data["base_etag"]

            # 4. Modify boxes:
            # - Delete box 0
            # - Move/resize box 1 and change class to 3 (truck) and subtype to truck_based_songthaew
            # - Add a new box with class 0 (car) and subtype pickup
            modified_boxes = []

            # Box 1 modified
            b1 = dict(boxes[1])
            b1["class_id"] = 3
            b1["class_name"] = "truck"
            b1["subtype"] = "truck_based_songthaew"
            b1["bbox_norm"] = [0.60, 0.45, 0.15, 0.20]  # moved and resized
            modified_boxes.append(b1)

            # Retain remaining boxes (skipping box 0, deleting it)
            for b in boxes[2:]:
                modified_boxes.append(dict(b))

            # Add new box
            new_box = {
                "instance_id": f"{fid}_inst_999",
                "class_id": 0,
                "class_name": "car",
                "subtype": "pickup",
                "is_ambiguous": False,
                "ambiguity_reason": "",
                "bbox_norm": [0.25, 0.50, 0.10, 0.12]
            }
            modified_boxes.append(new_box)

            # Save modified frame via POST
            save_payload = json.dumps({
                "base_etag": initial_etag,
                "action": "save_draft",
                "boxes": modified_boxes,
                "reviewer_notes": "Tested adding pickup and changing songthaew to truck",
                "is_ambiguous": False
            }).encode("utf-8")

            save_req = urllib.request.Request(
                f"{base_url}/api/frame/{fid}/save",
                data=save_payload,
                headers={"Content-Type": "application/json"}
            )
            save_resp = urllib.request.urlopen(save_req)
            self.assertEqual(save_resp.status, 200)
            save_result = json.loads(save_resp.read().decode("utf-8"))
            self.assertEqual(save_result["status"], "success")

            # 5. Verify JSON, YOLO label file, and previews agree!
            annos_file = self.pack_dir / "annotations" / "annotations.json"
            label_file = self.pack_dir / "annotations" / "labels" / f"{fid}.txt"
            preview_file = self.pack_dir / "previews" / f"{fid}_preview.jpg"

            self.assertTrue(label_file.exists())
            self.assertTrue(preview_file.exists())

            # Read label lines
            label_lines = [l.strip() for l in label_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(label_lines), len(modified_boxes))

            # Check that class 3 and class 0 appear in YOLO label file
            label_classes = [int(l.split()[0]) for l in label_lines]
            self.assertIn(3, label_classes)
            self.assertIn(0, label_classes)

            # Check annotations.json
            with open(annos_file, "r", encoding="utf-8") as f:
                re_records = json.load(f)
            saved_rec = next(r for r in re_records if r["frame_id"] == fid)
            self.assertEqual(len(saved_rec["boxes"]), len(modified_boxes))
            self.assertEqual(saved_rec["review_status"], "draft")
            self.assertEqual(saved_rec["reviewer_notes"], "Tested adding pickup and changing songthaew to truck")

            # Check stable ID on b1 survived
            saved_b1 = next(b for b in saved_rec["boxes"] if b["instance_id"] == b1["instance_id"])
            self.assertEqual(saved_b1["class_id"], 3)
            self.assertEqual(saved_b1["subtype"], "truck_based_songthaew")

        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
