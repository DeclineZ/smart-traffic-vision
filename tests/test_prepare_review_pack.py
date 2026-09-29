"""
Unit and regression tests for tools.prepare_review_pack (Batch 2).

Verifies:
- Every manifest candidate is either represented once in the pack or listed with a concrete failure reason.
- Clean images, source identities, label proposals, and subtype metadata remain linked.
- Duplicate sources are grouped by canonical source ID.
- Missing images and videos log concrete failure reasons without crashing.
- Failed or out-of-range extraction is safely handled and logged.
- Original proposals are preserved in proposals/ and unannotated holdouts remain explicitly unannotated.
- Re-running preparation preserves human review edits and does not overwrite modified annotations.
- Prediction-overlay images are never used as annotation inputs.
- Baseline model checkpoint hash is calculated and stored for traceability.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.prepare_review_pack import (
    ConflictingEditError,
    assign_diagnostic_group,
    compute_box_iou,
    compute_file_sha256,
    deduplicate_manifest_samples,
    extract_clean_frame_from_video,
    infer_provisional_subtype,
    match_box_instances_spatial,
    prepare_review_pack,
    render_box_preview_image,
    sync_review_pack,
    verify_image_file,
)


class TestPrepareReviewPack(unittest.TestCase):

    def test_baseline_checkpoint_hash_recorded(self):
        """Baseline model weight hash must be computed and recorded for audit traceability."""
        model_path = Path("models/yolo26s_thai_traffic.pt")
        if not model_path.exists():
            self.skipTest("Baseline checkpoint not found at models/yolo26s_thai_traffic.pt")

        h = compute_file_sha256(model_path)
        self.assertIsNotNone(h)
        self.assertEqual(len(h), 64)
        # Expected baseline hash verified in Batch 1
        self.assertEqual(h, "cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c")

    def test_diagnostic_group_assignment(self):
        """Samples must be assigned to unambiguous diagnostic categories; never independent test benchmarks."""
        # 1. Nearby training diagnostic (<= 3.0s from training frame)
        s_nearby = {"training_exposure": "nearby_training_exposure"}
        self.assertEqual(assign_diagnostic_group(s_nearby), "nearby_training_diagnostic")

        # 2. Validation split diagnostic
        s_val = {"training_exposure": "current_val_split", "artifact_references": ["val_split:cam03_east_f003720"]}
        self.assertEqual(assign_diagnostic_group(s_val), "current_val_diagnostic")

        # 3. Training split reference
        s_train = {"training_exposure": "current_train_split"}
        self.assertEqual(assign_diagnostic_group(s_train), "training_split_reference")

        # 4. Unproven checkpoint exposure (e.g. northeast holdout)
        s_unproven = {"training_exposure": "unproven_checkpoint_exposure", "artifact_references": ["video_frame:cam45_northeast.avi#12611"]}
        self.assertEqual(assign_diagnostic_group(s_unproven), "unproven_checkpoint_candidate")

    def test_provisional_subtypes_follow_agreed_taxonomy(self):
        """Subtypes must map pickups and pickup-songthaews to light vehicle (0), and truck-songthaews to truck (3)."""
        # Pickup -> class 0: car, subtype pickup
        sub_pk = infer_provisional_subtype(0, "cam43_south_pickup_f001000", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_pk, "pickup")

        # Pickup-based songthaew -> class 0: car, subtype pickup_based_songthaew
        sub_st = infer_provisional_subtype(0, "cam44_north_songthaew_f001500", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_st, "pickup_based_songthaew")

        # Commercial truck-based songthaew -> class 3: truck, subtype truck_based_songthaew
        sub_tst = infer_provisional_subtype(3, "cam46_west_songthaew_f002000", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_tst, "truck_based_songthaew")

        # Commuter van -> class 0: car, subtype passenger_van
        sub_van = infer_provisional_subtype(0, "cam03_east_van_f003000", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_van, "passenger_van")

        # Saleng & Tuk-Tuk -> class 4: three_wheeler
        sub_sl = infer_provisional_subtype(4, "cam43_south_saleng_f001000", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_sl, "saleng")
        sub_tt = infer_provisional_subtype(4, "cam43_south_tuktuk_f001000", (0.5, 0.5, 0.2, 0.2))
        self.assertEqual(sub_tt, "tuktuk")

    def test_missing_image_and_video_logs_concrete_failure(self):
        """When clean image is missing and video cannot be found, log concrete failure without crashing."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            manifest_path = temp_dir / "test_manifest.json"
            manifest_data = {
                "samples": [
                    {
                        "frame_id": "cam99_missing_f001000",
                        "source_frame_id": "cam99_missing_f001000",
                        "video_name": "cam99_missing",
                        "camera": "cam99_missing",
                        "frame_idx": 1000,
                        "clean_input_status": "requires_video_extraction",
                        "training_exposure": "unproven_checkpoint_exposure"
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            out_pack_dir = temp_dir / "pack_out"
            res = prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack_dir,
                videos_dir=temp_dir / "empty_videos",
                dataset_dir=temp_dir / "empty_dataset"
            )

            meta = res["metadata"]
            self.assertEqual(meta["total_candidates"], 1)
            self.assertEqual(meta["successfully_prepared_count"], 0)
            self.assertEqual(meta["extraction_failures_count"], 1)

            failure = meta["extraction_failures"][0]
            self.assertEqual(failure["frame_id"], "cam99_missing_f001000")
            self.assertIn("Source video file not found", failure["reason"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_out_of_range_video_extraction_failure(self):
        """Out of range frame index extraction must report a concrete bounds failure reason."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Check OpenCV availability
            try:
                import cv2
                import numpy as np
            except ImportError:
                self.skipTest("cv2 not available for video test")

            vid_path = temp_dir / "test_short.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (320, 240))
            for _ in range(10):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

            out_frame_path = temp_dir / "out.jpg"

            # Frame 5 is within bounds -> success
            ok, err, d = extract_clean_frame_from_video(vid_path, 5, out_frame_path)
            self.assertTrue(ok)
            self.assertIsNone(err)
            self.assertEqual(d, (320, 240))

            # Frame 9999 is out of bounds -> concrete failure
            ok, err, d = extract_clean_frame_from_video(vid_path, 9999, out_frame_path)
            self.assertFalse(ok)
            self.assertIn("out of bounds", err)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_unannotated_northeast_holdout_proposals_preserved_as_unannotated(self):
        """Unannotated holdouts must remain explicitly unannotated, not treated as verified background."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)
            (val_img_dir / "cam45_northeast_f012611.jpg").write_text("fake_raw")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {
                        "frame_id": "cam45_northeast_f012611",
                        "source_frame_id": "cam45_northeast_f012611",
                        "video_name": "cam45_northeast",
                        "camera": "cam45_northeast",
                        "frame_idx": 12611,
                        "clean_input_status": "existing_raw_image",
                        "annotations_status": "missing_unannotated",
                        "training_exposure": "unproven_checkpoint_exposure"
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            out_pack = temp_dir / "pack_unanno"
            res = prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack,
                videos_dir=temp_dir,
                dataset_dir=temp_dir / "dataset"
            )

            sample_record = res["samples"][0]
            self.assertTrue(sample_record["is_unannotated"])
            self.assertEqual(len(sample_record["boxes"]), 0)

            # Proposal text file must have unannotated marker
            prop_file = out_pack / "proposals" / "cam45_northeast_f012611.txt"
            self.assertTrue(prop_file.exists())
            self.assertIn("UNANNOTATED CANDIDATE", prop_file.read_text(encoding="utf-8"))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rerun_preserves_human_review_edits(self):
        """Re-running the preparation tool must not overwrite existing human annotations."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Setup synthetic dataset with 1 val image and 1 label
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            (val_img_dir / f"{fid}.jpg").write_text("fake_raw")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {
                        "frame_id": fid,
                        "source_frame_id": fid,
                        "video_name": "cam03_east",
                        "camera": "cam03_east",
                        "frame_idx": 3720,
                        "clean_input_status": "existing_raw_image",
                        "training_exposure": "current_val_split"
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            out_pack = temp_dir / "pack_rerun"

            # 1. First run: generates initial unreviewed pack
            prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack,
                videos_dir=temp_dir,
                dataset_dir=temp_dir / "dataset"
            )

            # 2. Simulate human annotator review:
            annos_json_path = out_pack / "annotations" / "annotations.json"
            annos_data = json.loads(annos_json_path.read_text(encoding="utf-8"))
            annos_data[0]["review_status"] = "verified"
            annos_data[0]["reviewer_notes"] = "Human verified pickup carrying ladder"
            annos_data[0]["is_ambiguous"] = False
            annos_data[0]["boxes"][0]["subtype"] = "pickup"
            annos_json_path.write_text(json.dumps(annos_data, indent=2), encoding="utf-8")

            # 3. Second run: must PRESERVE human annotations
            res2 = prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack,
                videos_dir=temp_dir,
                dataset_dir=temp_dir / "dataset"
            )

            rec_preserved = res2["samples"][0]
            self.assertEqual(rec_preserved["review_status"], "verified")
            self.assertEqual(rec_preserved["reviewer_notes"], "Human verified pickup carrying ladder")
            self.assertEqual(rec_preserved["boxes"][0]["subtype"], "pickup")

            # 4. Third run: verifies human work is never reset by preparation
            res3 = prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack,
                videos_dir=temp_dir,
                dataset_dir=temp_dir / "dataset"
            )
            rec_still_preserved = res3["samples"][0]
            self.assertEqual(rec_still_preserved["review_status"], "verified")
            self.assertEqual(rec_still_preserved["reviewer_notes"], "Human verified pickup carrying ladder")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_never_uses_prediction_overlay_as_clean_input(self):
        """Prediction overlay images must never be copied into clean images directory."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            try:
                import cv2
                import numpy as np
            except ImportError:
                self.skipTest("cv2 not available")

            # Synthetic video
            vid_dir = temp_dir / "videos"
            vid_dir.mkdir(parents=True)
            vid_path = vid_dir / "cam43_south.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (100, 100))
            # Frame 0 is all black (0)
            out.write(np.zeros((100, 100, 3), dtype=np.uint8))
            # Frame 1 is all gray (128) -> this is what clean video has
            clean_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
            out.write(clean_frame)
            out.release()

            # Prediction artifact has burned-in red box (value 255)
            eval_pred_dir = temp_dir / "eval_predictions"
            eval_pred_dir.mkdir(parents=True)
            pred_img = np.full((100, 100, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(eval_pred_dir / "pred_cam43_south_f000001.jpg"), pred_img)

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {
                        "frame_id": "pred_cam43_south_f000001",
                        "source_frame_id": "cam43_south_f000001",
                        "video_name": "cam43_south",
                        "camera": "cam43_south",
                        "frame_idx": 1,
                        "clean_input_status": "requires_video_extraction",
                        "training_exposure": "nearby_training_exposure"
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            out_pack = temp_dir / "pack_clean"
            prepare_review_pack(
                manifest_path=manifest_path,
                output_dir=out_pack,
                videos_dir=vid_dir,
                dataset_dir=temp_dir
            )

            # Extracted image must be from video (gray 128), NOT prediction overlay (255)
            extracted_path = out_pack / "images" / "pred_cam43_south_f000001.jpg"
            self.assertTrue(extracted_path.exists())
            read_back = cv2.imread(str(extracted_path))
            self.assertEqual(read_back[50, 50, 0], 128)
            self.assertNotEqual(read_back[50, 50, 0], 255)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _create_synthetic_image(self, path: Path, width: int = 100, height: int = 100):
        """Helper to create a valid decodable image file."""
        import cv2
        import numpy as np
        im = np.full((height, width, 3), 120, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), im)

    def test_subtype_only_edits_preserved_while_unreviewed(self):
        """Subtype-only edits made while review_status is 'unreviewed' must be preserved on rerun."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. First run: generates initial pack
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 2. Human changes only the subtype: review_status remains 'unreviewed', notes empty
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            self.assertEqual(annos[0]["review_status"], "unreviewed")
            annos[0]["boxes"][0]["subtype"] = "pickup"
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # 3. Rerun preparation
            res = prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 4. Must preserve subtype 'pickup' even though unreviewed!
            rec = res["samples"][0]
            self.assertEqual(rec["review_status"], "unreviewed")
            self.assertEqual(rec["boxes"][0]["subtype"], "pickup")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_deletion_of_all_boxes_preserved(self):
        """Intentionally empty box lists (0 boxes) must be preserved on rerun, not restored to proposals."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. First run
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 2. Annotator deletes all boxes
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            annos[0]["boxes"] = []
            annos[0]["review_status"] = "verified"
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # 3. Rerun preparation
            res = prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 4. Box list must remain empty!
            rec = res["samples"][0]
            self.assertEqual(len(rec["boxes"]), 0)
            self.assertEqual(rec["annotation_state"], "verified_empty_background")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_edited_yolo_labels_updating_previews(self):
        """Editing YOLO label files must update structured metadata and re-render decodable previews."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. Prepare initial pack
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Check initial preview
            preview_file = out_pack / "previews" / f"{fid}_preview.jpg"
            self.assertTrue(preview_file.exists())
            old_preview_time = preview_file.stat().st_mtime_ns

            # 2. Annotator edits YOLO label file: adds motorcycle (class 1)
            yolo_label_file = out_pack / "annotations" / "labels" / f"{fid}.txt"
            yolo_label_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n", encoding="utf-8")

            # 3. Synchronize
            res = sync_review_pack(out_pack, strategy="from_yolo")
            self.assertEqual(res["synced_records"], 1)

            # 4. Check annotations.json and preview image
            annos = json.loads((out_pack / "annotations" / "annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(len(annos[0]["boxes"]), 2)
            self.assertEqual(annos[0]["boxes"][0]["instance_id"], f"{fid}_inst_000")
            self.assertEqual(annos[0]["boxes"][1]["class_id"], 1)

            # Preview was re-rendered and decodes
            is_v, err, dims = verify_image_file(preview_file)
            self.assertTrue(is_v)
            self.assertEqual(dims, (100, 100))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_malformed_existing_annotation_json_stops_safely(self):
        """If annotations.json cannot be parsed, stop execution without overwriting it."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Corrupt annotations.json
            annos_path = out_pack / "annotations" / "annotations.json"
            corrupt_content = '{\n  "corrupt": true, [syntax error]\n'
            annos_path.write_text(corrupt_content, encoding="utf-8")

            # Rerun must raise ValueError and NOT overwrite the file
            with self.assertRaises(ValueError) as ctx:
                prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            self.assertIn("Failed to parse existing annotations JSON", str(ctx.exception))
            # Verify file content is completely unchanged
            self.assertEqual(annos_path.read_text(encoding="utf-8"), corrupt_content)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_changed_source_proposals_requires_new_version(self):
        """If source proposals change, report mismatch and require a new version without resetting proposals."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            prop_file = out_pack / "proposals" / f"{fid}.txt"
            original_prop_text = prop_file.read_text(encoding="utf-8")

            # Modify source proposals in dataset labels
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.8 0.8 0.3 0.3\n")

            # Rerun must raise RuntimeError requiring a new pack version
            with self.assertRaises(RuntimeError) as ctx:
                prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            self.assertIn("Source proposals changed", str(ctx.exception))
            self.assertIn("immutable within a pack version", str(ctx.exception))
            # Verify original proposal file remained immutable
            self.assertEqual(prop_file.read_text(encoding="utf-8"), original_prop_text)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_duplicate_canonical_source_ids_deduplicated(self):
        """Multiple manifest samples sharing canonical_source_id must be deduplicated into one record."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            # 2 samples with identical canonical_source_id
            manifest_data = {
                "samples": [
                    {
                        "frame_id": fid,
                        "source_frame_id": fid,
                        "video_name": "cam03_east",
                        "camera": "cam03_east",
                        "frame_idx": 3720,
                        "clean_input_status": "existing_raw_image",
                        "selection_reasons": ["reason_alpha"],
                        "artifact_references": ["ref_1"],
                        "training_exposure": "nearby_training_exposure",
                        "nearby_training_delta_frames": 45
                    },
                    {
                        "frame_id": fid,
                        "source_frame_id": fid,
                        "video_name": "cam03_east",
                        "camera": "cam03_east",
                        "frame_idx": 3720,
                        "clean_input_status": "existing_raw_image",
                        "selection_reasons": ["reason_beta"],
                        "artifact_references": ["ref_2"],
                        "training_exposure": "current_train_split",
                        "nearby_training_delta_frames": 0
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            res = prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Must have exactly 1 deduplicated record
            self.assertEqual(len(res["samples"]), 1)
            rec = res["samples"][0]
            # Must combine selection reasons and artifact references
            self.assertIn("reason_alpha", rec["selection_reasons"])
            self.assertIn("reason_beta", rec["selection_reasons"])
            self.assertIn("ref_1", rec["artifact_references"])
            self.assertIn("ref_2", rec["artifact_references"])
            # Exposure precedence: current_train_split > nearby_training_exposure
            self.assertEqual(rec["training_exposure"], "current_train_split")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_historical_frames_with_no_labels_are_explicitly_unannotated(self):
        """Historical frames without label files must be explicitly unannotated."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            fid = "cam43_south_hist_f000500"
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam43_south",
                    "camera": "cam43_south",
                    "frame_idx": 500,
                    "clean_input_status": "existing_raw_image",
                    "annotations_status": "missing_unannotated",
                    "training_exposure": "unproven_checkpoint_exposure"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            res = prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")
            rec = res["samples"][0]
            self.assertTrue(rec["is_unannotated"])
            self.assertEqual(rec["annotation_state"], "unannotated")
            self.assertEqual(len(rec["boxes"]), 0)

            # Proposals file must contain unannotated marker
            prop = (out_pack / "proposals" / f"{fid}.txt").read_text(encoding="utf-8")
            self.assertIn("UNANNOTATED CANDIDATE", prop)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_human_annotated_northeast_leaves_unannotated_state(self):
        """Northeast frames with human-added annotations must leave the unannotated state."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            fid = "cam45_northeast_f012611"
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam45_northeast",
                    "camera": "cam45_northeast",
                    "frame_idx": 12611,
                    "clean_input_status": "existing_raw_image",
                    "annotations_status": "missing_unannotated",
                    "training_exposure": "unproven_checkpoint_exposure"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. Initial run: starts unannotated
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 2. Annotator adds a box in YOLO label file
            (out_pack / "annotations" / "labels" / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            # 3. Synchronize
            res = sync_review_pack(out_pack, strategy="from_yolo")
            self.assertEqual(res["synced_records"], 1)

            # 4. Check that frame has left unannotated state
            annos = json.loads((out_pack / "annotations" / "annotations.json").read_text(encoding="utf-8"))
            rec = annos[0]
            self.assertFalse(rec["is_unannotated"])
            self.assertEqual(rec["annotation_state"], "annotated")
            self.assertEqual(len(rec["boxes"]), 1)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_distinguish_unannotated_from_verified_empty_background(self):
        """Distinguish unannotated frames from human-verified empty background frames."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)

            f_unanno = "cam45_northeast_f001000"
            f_empty = "cam03_east_f002000"
            self._create_synthetic_image(val_img_dir / f"{f_unanno}.jpg")
            self._create_synthetic_image(val_img_dir / f"{f_empty}.jpg")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {
                        "frame_id": f_unanno,
                        "source_frame_id": f_unanno,
                        "video_name": "cam45_northeast",
                        "camera": "cam45_northeast",
                        "frame_idx": 1000,
                        "clean_input_status": "existing_raw_image",
                        "annotations_status": "missing_unannotated",
                        "training_exposure": "unproven_checkpoint_exposure"
                    },
                    {
                        "frame_id": f_empty,
                        "source_frame_id": f_empty,
                        "video_name": "cam03_east",
                        "camera": "cam03_east",
                        "frame_idx": 2000,
                        "clean_input_status": "existing_raw_image",
                        "annotations_status": "missing_unannotated",
                        "training_exposure": "current_val_split"
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Annotator reviews f_empty and confirms 0 vehicles
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            for r in annos:
                if r["frame_id"] == f_empty:
                    r["review_status"] = "verified"
                    r["reviewer_notes"] = "Verified empty background - 0 vehicles"
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Rerun
            res = prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")
            records_by_id = {r["frame_id"]: r for r in res["samples"]}

            # f_unanno must remain unannotated
            self.assertTrue(records_by_id[f_unanno]["is_unannotated"])
            self.assertEqual(records_by_id[f_unanno]["annotation_state"], "unannotated")

            # f_empty must be verified_empty_background and NOT unannotated
            self.assertFalse(records_by_id[f_empty]["is_unannotated"])
            self.assertEqual(records_by_id[f_empty]["annotation_state"], "verified_empty_background")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_retains_records_omitted_from_later_manifest(self):
        """Existing annotation records omitted from a later manifest must be retained."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)
            f1, f2 = "cam01_f001000", "cam02_f002000"
            self._create_synthetic_image(val_img_dir / f"{f1}.jpg")
            self._create_synthetic_image(val_img_dir / f"{f2}.jpg")

            # Manifest 1 contains both f1 and f2
            m1_path = temp_dir / "manifest1.json"
            m1_data = {
                "samples": [
                    {"frame_id": f1, "source_frame_id": f1, "video_name": "cam01", "camera": "cam01", "clean_input_status": "existing_raw_image"},
                    {"frame_id": f2, "source_frame_id": f2, "video_name": "cam02", "camera": "cam02", "clean_input_status": "existing_raw_image"}
                ]
            }
            m1_path.write_text(json.dumps(m1_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(m1_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Manifest 2 only contains f1
            m2_path = temp_dir / "manifest2.json"
            m2_data = {
                "samples": [
                    {"frame_id": f1, "source_frame_id": f1, "video_name": "cam01", "camera": "cam01", "clean_input_status": "existing_raw_image"}
                ]
            }
            m2_path.write_text(json.dumps(m2_data), encoding="utf-8")

            # Run with Manifest 2
            res = prepare_review_pack(m2_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")
            fids = [r["frame_id"] for r in res["samples"]]

            # Both f1 and f2 must be retained
            self.assertIn(f1, fids)
            self.assertIn(f2, fids)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_conflicting_edits_detected(self):
        """If both YOLO label and annotations.json are modified incompatibly, detect conflict."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Annotator A edits YOLO label
            (out_pack / "annotations" / "labels" / f"{fid}.txt").write_text("0 0.1 0.1 0.1 0.1\n")

            # Annotator B edits JSON boxes to a different box
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            annos[0]["boxes"][0]["bbox_norm"] = [0.9, 0.9, 0.2, 0.2]
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Auto sync must detect conflict rather than guessing
            with self.assertRaises(ConflictingEditError) as ctx:
                sync_review_pack(out_pack, strategy="auto")

            self.assertIn("Conflicting edits detected", str(ctx.exception))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_demonstrate_edit_synchronize_preview_workflow(self):
        """Demonstrate the complete documented edit -> synchronize -> preview workflow."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg", width=320, height=240)
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image",
                    "training_exposure": "current_val_split"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # Step 1: Prepare pack
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Step 2: Human edits YOLO label (adds a truck)
            yolo_file = out_pack / "annotations" / "labels" / f"{fid}.txt"
            yolo_file.write_text("0 0.5 0.5 0.2 0.2\n3 0.8 0.8 0.15 0.15\n")

            # Step 3: Run synchronization
            sync_res = sync_review_pack(out_pack, strategy="from_yolo")
            self.assertEqual(sync_res["status"], "success")
            self.assertEqual(sync_res["synced_records"], 1)

            # Step 4: Verify structured annotations
            annos = json.loads((out_pack / "annotations" / "annotations.json").read_text(encoding="utf-8"))
            rec = annos[0]
            self.assertEqual(len(rec["boxes"]), 2)
            self.assertEqual(rec["boxes"][0]["instance_id"], f"{fid}_inst_000")
            self.assertEqual(rec["boxes"][1]["class_id"], 3)
            self.assertEqual(rec["boxes"][1]["class_name"], "truck")

            # Step 5: Verify preview updated and decodable
            preview_file = out_pack / "previews" / f"{fid}_preview.jpg"
            is_v, err, dims = verify_image_file(preview_file)
            self.assertTrue(is_v)
            self.assertEqual(dims, (320, 240))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_independent_edits_rerun_preparation_preserves_both_files(self):
        """Independently edit JSON and YOLO geometry, rerun preparation, and verify both files remain unchanged."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "frame_idx": 3720,
                    "clean_input_status": "existing_raw_image"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. Prepare initial pack
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 2. Annotator A independently edits YOLO label file
            yolo_file = out_pack / "annotations" / "labels" / f"{fid}.txt"
            yolo_content_edited = "0 0.900000 0.900000 0.100000 0.100000\n"
            yolo_file.write_text(yolo_content_edited, encoding="utf-8")

            # 3. Annotator B independently edits JSON coordinates
            annos_file = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_file.read_text(encoding="utf-8"))
            annos[0]["boxes"][0]["bbox_norm"] = [0.123456, 0.654321, 0.2, 0.2]
            json_content_edited = json.dumps(annos, indent=2)
            annos_file.write_text(json_content_edited, encoding="utf-8")

            # 4. Rerun preparation
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # 5. Verify BOTH files remain completely unchanged
            self.assertEqual(yolo_file.read_text(encoding="utf-8"), yolo_content_edited)
            self.assertEqual(annos_file.read_text(encoding="utf-8"), json_content_edited)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_preflight_failure_preserves_all_files_byte_identical(self):
        """First frame needs synchronization; a later frame conflicts. Verify every existing file remains byte-identical."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            f1, f2 = "cam01_f001000", "cam02_f002000"
            self._create_synthetic_image(val_img_dir / f"{f1}.jpg")
            self._create_synthetic_image(val_img_dir / f"{f2}.jpg")
            (val_lbl_dir / f"{f1}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (val_lbl_dir / f"{f2}.txt").write_text("0 0.4 0.4 0.3 0.3\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {"frame_id": f1, "source_frame_id": f1, "video_name": "cam01", "camera": "cam01", "clean_input_status": "existing_raw_image"},
                    {"frame_id": f2, "source_frame_id": f2, "video_name": "cam02", "camera": "cam02", "clean_input_status": "existing_raw_image"}
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # f1: needs synchronization (YOLO label was updated to add a box)
            f1_lbl = out_pack / "annotations" / "labels" / f"{f1}.txt"
            f1_lbl.write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")

            # f2: has a conflict (both YOLO and JSON modified independently)
            f2_lbl = out_pack / "annotations" / "labels" / f"{f2}.txt"
            f2_lbl.write_text("0 0.9 0.9 0.1 0.1\n")

            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            for r in annos:
                if r["frame_id"] == f2:
                    r["boxes"][0]["bbox_norm"] = [0.1, 0.1, 0.2, 0.2]
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Record pre-sync exact file contents
            annos_before = annos_path.read_text(encoding="utf-8")
            f1_before = f1_lbl.read_text(encoding="utf-8")
            f2_before = f2_lbl.read_text(encoding="utf-8")

            # Run auto sync: must fail on f2 during preflight
            with self.assertRaises(ConflictingEditError):
                sync_review_pack(out_pack, strategy="auto")

            # Verify every existing file remains 100% byte-identical
            self.assertEqual(annos_path.read_text(encoding="utf-8"), annos_before)
            self.assertEqual(f1_lbl.read_text(encoding="utf-8"), f1_before)
            self.assertEqual(f2_lbl.read_text(encoding="utf-8"), f2_before)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_metadata_only_changes_refresh_preview_in_auto_sync(self):
        """Change only subtype/ambiguity and verify automatic sync refreshes the preview."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "clean_input_status": "existing_raw_image"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")
            preview_file = out_pack / "previews" / f"{fid}_preview.jpg"
            preview_hash_initial = hashlib.sha256(preview_file.read_bytes()).hexdigest()

            # Change only subtype and ambiguity flag; geometry is unchanged
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            annos[0]["boxes"][0]["subtype"] = "pickup"
            annos[0]["boxes"][0]["is_ambiguous"] = True
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Run auto sync
            res = sync_review_pack(out_pack, strategy="auto")
            self.assertEqual(res["status"], "success")
            self.assertGreaterEqual(res["updated_previews"], 1)

            # Verify preview was refreshed with new content
            preview_hash_updated = hashlib.sha256(preview_file.read_bytes()).hexdigest()
            self.assertNotEqual(preview_hash_initial, preview_hash_updated)
            is_v, _, dims = verify_image_file(preview_file)
            self.assertTrue(is_v)
            self.assertEqual(dims, (100, 100))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mark_empty_frame_verified_becomes_verified_empty_background(self):
        """Mark an empty frame verified without changing geometry; verify its state becomes verified_empty_background."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_img_dir.mkdir(parents=True)

            fid = "cam45_northeast_f001000"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam45_northeast",
                    "camera": "cam45_northeast",
                    "clean_input_status": "existing_raw_image",
                    "annotations_status": "missing_unannotated"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Initially unannotated
            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            self.assertEqual(annos[0]["annotation_state"], "unannotated")
            self.assertTrue(annos[0]["is_unannotated"])

            # Reviewer verifies empty background without adding any boxes
            annos[0]["review_status"] = "verified"
            annos[0]["reviewer_notes"] = "Human verified: zero vehicles in frame"
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Run auto sync
            res = sync_review_pack(out_pack, strategy="auto")
            self.assertEqual(res["status"], "success")
            self.assertGreaterEqual(res["updated_previews"], 1)

            # Check state transition
            annos_updated = json.loads(annos_path.read_text(encoding="utf-8"))
            self.assertEqual(annos_updated[0]["annotation_state"], "verified_empty_background")
            self.assertFalse(annos_updated[0]["is_unannotated"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_preview_write_failure_preserves_human_files(self):
        """Simulate preview/write failure and verify no human files are partially replaced."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            fid = "cam03_east_f003720"
            self._create_synthetic_image(val_img_dir / f"{fid}.jpg")
            (val_lbl_dir / f"{fid}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [{
                    "frame_id": fid,
                    "source_frame_id": fid,
                    "video_name": "cam03_east",
                    "camera": "cam03_east",
                    "clean_input_status": "existing_raw_image"
                }]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            # Annotator edits label file
            lbl_file = out_pack / "annotations" / "labels" / f"{fid}.txt"
            lbl_file.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
            lbl_before = lbl_file.read_text(encoding="utf-8")

            annos_file = out_pack / "annotations" / "annotations.json"
            annos_before = annos_file.read_text(encoding="utf-8")

            # Mock render_box_preview_image to fail during staging
            import tools.prepare_review_pack as prep_mod
            orig_render = prep_mod.render_box_preview_image
            try:
                prep_mod.render_box_preview_image = lambda *args, **kwargs: False

                with self.assertRaises(RuntimeError) as ctx:
                    sync_review_pack(out_pack, strategy="from_yolo")

                self.assertIn("Failed to render preview", str(ctx.exception))

                # Verify human files on disk were not replaced or corrupted
                self.assertEqual(annos_file.read_text(encoding="utf-8"), annos_before)
                self.assertEqual(lbl_file.read_text(encoding="utf-8"), lbl_before)

            finally:
                prep_mod.render_box_preview_image = orig_render

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_destination_commit_failure_rolls_back_to_byte_identical_originals(self):
        """
        Create a two-frame pack requiring updates. Inject failure after the first successful
        destination replacement. Assert every existing label and annotations.json remains
        byte-identical and no new destination survives. Also verify a successful commit
        leaves labels and JSON synchronized.
        """
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            f1 = "cam01_f001000"
            f2 = "cam02_f002000"
            self._create_synthetic_image(val_img_dir / f"{f1}.jpg")
            self._create_synthetic_image(val_img_dir / f"{f2}.jpg")
            (val_lbl_dir / f"{f1}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (val_lbl_dir / f"{f2}.txt").write_text("0 0.4 0.4 0.3 0.3\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {"frame_id": f1, "source_frame_id": f1, "video_name": "cam01", "camera": "cam01", "clean_input_status": "existing_raw_image"},
                    {"frame_id": f2, "source_frame_id": f2, "video_name": "cam02", "camera": "cam02", "clean_input_status": "existing_raw_image"}
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            # 1. Prepare initial review pack
            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            annos_path = out_pack / "annotations" / "annotations.json"
            f1_lbl = out_pack / "annotations" / "labels" / f"{f1}.txt"
            f2_lbl = out_pack / "annotations" / "labels" / f"{f2}.txt"
            f1_preview = out_pack / "previews" / f"{f1}_preview.jpg"
            f2_preview = out_pack / "previews" / f"{f2}_preview.jpg"

            # 2. Both frames require updates: edit annotations.json so strategy="from_json" updates both labels
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            annos[0]["boxes"][0]["bbox_norm"] = [0.111111, 0.222222, 0.333333, 0.444444]
            annos[1]["boxes"][0]["bbox_norm"] = [0.555555, 0.666666, 0.222222, 0.333333]
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            # Snapshot exact bytes before sync attempt
            annos_bytes_before = annos_path.read_bytes()
            f1_lbl_bytes_before = f1_lbl.read_bytes()
            f2_lbl_bytes_before = f2_lbl.read_bytes()
            f1_prev_bytes_before = f1_preview.read_bytes()
            f2_prev_bytes_before = f2_preview.read_bytes()

            # 3. Inject failure after the first successful destination replacement
            import tools.prepare_review_pack as prep_mod
            orig_replace = prep_mod.atomic_replace_file
            replace_count = 0

            def failing_replace(src, dst):
                nonlocal replace_count
                replace_count += 1
                if replace_count == 2:
                    raise OSError("Injected destination replacement failure on 2nd file")
                return orig_replace(src, dst)

            prep_mod.atomic_replace_file = failing_replace
            try:
                with self.assertRaises(OSError) as ctx:
                    sync_review_pack(out_pack, strategy="from_json")
                self.assertIn("Injected destination replacement failure on 2nd file", str(ctx.exception))
            finally:
                prep_mod.atomic_replace_file = orig_replace

            # 4. Assert every existing label and annotations.json remains byte-identical
            self.assertEqual(annos_path.read_bytes(), annos_bytes_before)
            self.assertEqual(f1_lbl.read_bytes(), f1_lbl_bytes_before)
            self.assertEqual(f2_lbl.read_bytes(), f2_lbl_bytes_before)
            self.assertEqual(f1_preview.read_bytes(), f1_prev_bytes_before)
            self.assertEqual(f2_preview.read_bytes(), f2_prev_bytes_before)

            # Assert no new destination survives (clean rollback, no lingering backups or temps)
            self.assertFalse((out_pack / ".recovery_backups").exists())
            temp_files = list(out_pack.glob("**/.tmp_*"))
            self.assertEqual(len(temp_files), 0, f"Found lingering temporary files: {temp_files}")

            # 5. Verify a subsequent successful commit leaves labels and JSON synchronized
            res = sync_review_pack(out_pack, strategy="from_json")
            self.assertEqual(res["status"], "success")
            self.assertGreaterEqual(res["synced_records"], 2)

            f1_lbl_after = f1_lbl.read_text(encoding="utf-8")
            f2_lbl_after = f2_lbl.read_text(encoding="utf-8")
            self.assertIn("0.111111 0.222222", f1_lbl_after)
            self.assertIn("0.555555 0.666666", f2_lbl_after)

            annos_after = json.loads(annos_path.read_text(encoding="utf-8"))
            self.assertEqual(annos_after[0]["synced_label_hash"], prep_mod.compute_label_file_hash(f1_lbl))
            self.assertEqual(annos_after[1]["synced_label_hash"], prep_mod.compute_label_file_hash(f2_lbl))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_rollback_failure_retains_recovery_backups_and_reports_path(self):
        """If rollback itself encounters errors, recovery backups must be retained and reported."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            val_img_dir = temp_dir / "dataset" / "images" / "val"
            val_lbl_dir = temp_dir / "dataset" / "labels" / "val"
            val_img_dir.mkdir(parents=True)
            val_lbl_dir.mkdir(parents=True)

            f1 = "cam01_f001000"
            f2 = "cam02_f002000"
            self._create_synthetic_image(val_img_dir / f"{f1}.jpg")
            self._create_synthetic_image(val_img_dir / f"{f2}.jpg")
            (val_lbl_dir / f"{f1}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (val_lbl_dir / f"{f2}.txt").write_text("0 0.4 0.4 0.3 0.3\n")

            manifest_path = temp_dir / "manifest.json"
            manifest_data = {
                "samples": [
                    {"frame_id": f1, "source_frame_id": f1, "video_name": "cam01", "camera": "cam01", "clean_input_status": "existing_raw_image"},
                    {"frame_id": f2, "source_frame_id": f2, "video_name": "cam02", "camera": "cam02", "clean_input_status": "existing_raw_image"}
                ]
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
            out_pack = temp_dir / "pack"

            prepare_review_pack(manifest_path, out_pack, videos_dir=temp_dir, dataset_dir=temp_dir / "dataset")

            annos_path = out_pack / "annotations" / "annotations.json"
            annos = json.loads(annos_path.read_text(encoding="utf-8"))
            annos[0]["boxes"][0]["bbox_norm"] = [0.111111, 0.222222, 0.333333, 0.444444]
            annos[1]["boxes"][0]["bbox_norm"] = [0.555555, 0.666666, 0.222222, 0.333333]
            annos_path.write_text(json.dumps(annos, indent=2), encoding="utf-8")

            import tools.prepare_review_pack as prep_mod
            orig_replace = prep_mod.atomic_replace_file
            replace_count = 0

            def failing_replace_and_rollback(src, dst):
                nonlocal replace_count
                replace_count += 1
                if replace_count >= 2:
                    raise OSError("Injected disk failure during commit/rollback")
                return orig_replace(src, dst)

            prep_mod.atomic_replace_file = failing_replace_and_rollback
            try:
                with self.assertRaises(RuntimeError) as ctx:
                    sync_review_pack(out_pack, strategy="from_json")
                err_text = str(ctx.exception)
                self.assertIn("Commit failed", err_text)
                self.assertIn("rollback failed", err_text)
                self.assertIn("Recovery backups are retained at", err_text)
                # Verify recovery backups directory was actually retained on disk
                backups_dir = out_pack / ".recovery_backups"
                self.assertTrue(backups_dir.exists())
                backup_files = list(backups_dir.glob("**/*.*"))
                self.assertGreater(len(backup_files), 0)
            finally:
                prep_mod.atomic_replace_file = orig_replace

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

