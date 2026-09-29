"""
Unit and regression tests for tools.audit_dataset:
- Independent reproduction of external UA-DETRAC totals (16,584 records, 9,716 source frames)
- Validation of external IDs against dataset header, rejecting fractional/unknown IDs, non-finite coords
- Box edge auditing with documented rounding tolerance
- Unknown provenance preservation without merging distinct files
- Aspect-preserving resizing and object-size calculation for non-square images and unknown dimensions
- Absent video handling: reporting frame distances and unknown timing without guessing FPS
- Cross-checking prediction samples against training frames and nearby exposure
- Dynamic report values changing with fixture data (no hardcoded totals)
- Deterministic reproducibility between identical runs
"""

import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    DEFAULT_UADETRAC_CLASS_NAMES,
    THAI_5CLASS_NAMES,
    audit_compiled_dataset,
    audit_external_ndjson,
    build_eval_sampling_manifest,
    compute_aspect_preserving_size,
    compute_box_iou,
    get_manual_discrepancy_review,
    parse_frame_provenance,
    render_markdown_report,
    validate_box,
)


class TestAuditDataset(unittest.TestCase):

    def test_external_totals_reproduction(self):
        """Independently reproduce exactly 16,584 image records and 9,716 source-frame identifiers on actual NDJSON."""
        ndjson_path = Path("data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson")
        if not ndjson_path.exists():
            self.skipTest(f"Missing external test file: {ndjson_path}")

        res = audit_external_ndjson(ndjson_path)
        self.assertTrue(res["exists"])
        self.assertEqual(res["image_records"], 16584)
        self.assertEqual(res["source_frame_identifiers"], 9716)
        self.assertEqual(res["unique_sequences"], 100)
        self.assertEqual(res["malformed_count"], 0)

        # Verify 100% sequence leakage finding across all split pairs
        for pair_name, o_info in res["split_overlaps"].items():
            self.assertEqual(o_info["shared_sequences"], 100)

    def test_invalid_external_ids_and_malformed_boxes(self):
        """Validate box parser rejects fractional/unknown IDs, non-finite coords, and out-of-bounds edges."""
        allowed_classes = {0: "bus", 1: "car", 2: "truck", 3: "van"}

        # 1. Valid box
        ok, reason, clean = validate_box([1, 0.5, 0.5, 0.2, 0.2], allowed_classes)
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(clean, (1, 0.5, 0.5, 0.2, 0.2))

        # 2. Fractional class ID (e.g. 1.5) must be rejected
        ok, reason, _ = validate_box([1.5, 0.5, 0.5, 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Fractional class ID", reason)

        # 3. Unknown class ID (e.g. 99 or negative -1)
        ok, reason, _ = validate_box([99, 0.5, 0.5, 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Unknown class ID", reason)

        # 4. Non-numeric class token
        ok, reason, _ = validate_box(["car", 0.5, 0.5, 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Non-numeric", reason)

        # 5. Non-finite coordinate (NaN / Inf)
        ok, reason, _ = validate_box([1, float("nan"), 0.5, 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Non-finite", reason)

        ok, reason, _ = validate_box([1, 0.5, float("inf"), 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Non-finite", reason)

        # 6. Non-positive dimensions
        ok, reason, _ = validate_box([1, 0.5, 0.5, 0.0, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Non-positive", reason)

        # 7. Box edge out of bounds beyond tolerance
        # x2 = 0.95 + 0.2 / 2 = 1.05 (> 1.0 + 1e-3)
        ok, reason, _ = validate_box([1, 0.95, 0.5, 0.2, 0.2], allowed_classes)
        self.assertFalse(ok)
        self.assertIn("Box edges out of bounds", reason)

        # 8. Box edge within rounding tolerance (1.0005 <= 1.0 + 1e-3)
        # x2 = 0.50025 + 1.0 / 2 = 1.00025 (within 1e-3 tolerance)
        ok, reason, clean = validate_box([1, 0.50025, 0.5, 1.0, 0.2], allowed_classes, edge_tolerance=1e-3)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_source_class_vs_target_class_distinction(self):
        """Verify external UA-DETRAC classes and Thai 5-class standard maintain semantic distinction."""
        self.assertEqual(DEFAULT_UADETRAC_CLASS_NAMES[0], "truck")
        self.assertEqual(DEFAULT_UADETRAC_CLASS_NAMES[1], "car")
        self.assertEqual(DEFAULT_UADETRAC_CLASS_NAMES[2], "van")
        self.assertEqual(DEFAULT_UADETRAC_CLASS_NAMES[3], "bus")

        self.assertEqual(THAI_5CLASS_NAMES[0], "car")
        self.assertEqual(THAI_5CLASS_NAMES[1], "motorcycle")
        self.assertEqual(THAI_5CLASS_NAMES[2], "bus")
        self.assertEqual(THAI_5CLASS_NAMES[3], "truck")
        self.assertEqual(THAI_5CLASS_NAMES[4], "three_wheeler")

        # Class ID 0 represents distinct concepts in each standard (truck vs car)
        self.assertNotEqual(DEFAULT_UADETRAC_CLASS_NAMES[0], THAI_5CLASS_NAMES[0])

    def test_unknown_provenance_preserves_unique_identities(self):
        """Unrecognized stems must retain distinct source IDs rather than collapsing into one 'unknown' key."""
        stems = ["arbitrary_download_001", "arbitrary_download_002", "unstructured_frame_abc"]
        provs = [parse_frame_provenance(s) for s in stems]

        # Each must be marked as not known
        for p in provs:
            self.assertFalse(p.is_provenance_known)
            self.assertEqual(p.video, "unknown")
            self.assertEqual(p.camera, "unknown")
            self.assertEqual(p.lighting, "unknown")

        # Distinct source IDs: no accidental deduplication
        source_ids = {p.source_frame_id for p in provs}
        self.assertEqual(len(source_ids), len(stems))
        self.assertIn("unknown_provenance:arbitrary_download_001", source_ids)
        self.assertIn("unknown_provenance:arbitrary_download_002", source_ids)

    def test_non_square_images_aspect_preserving_resizing(self):
        """Aspect-preserving resizing must properly scale 16:9 images without vertical distortion."""
        # 1. 1920x1080 native CCTV image resized to ref_size 640
        # scale = 640 / 1920 = 1/3
        # resized_w = 640, resized_h = 360
        sz_16_9 = compute_aspect_preserving_size(
            norm_w=0.10,
            norm_h=0.20,
            img_w=1920,
            img_h=1080,
            ref_size=640
        )
        self.assertTrue(sz_16_9["dimensions_known"])
        self.assertAlmostEqual(sz_16_9["resized_w"], 640.0)
        self.assertAlmostEqual(sz_16_9["resized_h"], 360.0)
        self.assertAlmostEqual(sz_16_9["pixel_w"], 64.0)   # 0.10 * 640
        self.assertAlmostEqual(sz_16_9["pixel_h"], 72.0)   # 0.20 * 360
        self.assertAlmostEqual(sz_16_9["pixel_area"], 4608.0) # 64 * 72
        self.assertEqual(sz_16_9["size_bucket"], "medium") # 1024 <= 4608 <= 9216

        # 2. Square 640x640 image
        sz_sq = compute_aspect_preserving_size(
            norm_w=0.10,
            norm_h=0.20,
            img_w=640,
            img_h=640,
            ref_size=640
        )
        self.assertTrue(sz_sq["dimensions_known"])
        self.assertAlmostEqual(sz_sq["pixel_w"], 64.0)
        self.assertAlmostEqual(sz_sq["pixel_h"], 128.0)
        self.assertAlmostEqual(sz_sq["pixel_area"], 8192.0)

        # 3. Unknown dimensions must NOT assume 1920x1080
        sz_unk = compute_aspect_preserving_size(
            norm_w=0.10,
            norm_h=0.20,
            img_w=None,
            img_h=None,
            ref_size=640
        )
        self.assertFalse(sz_unk["dimensions_known"])
        self.assertIsNone(sz_unk["pixel_area"])
        self.assertEqual(sz_unk["size_bucket"], "unknown_dimensions")

    def test_absent_videos(self):
        """When video recordings are absent, report frame distances and unknown timing without guessing FPS."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create synthetic train and val dataset
            for s in ["train", "val"]:
                (temp_dir / "images" / s).mkdir(parents=True)
                (temp_dir / "labels" / s).mkdir(parents=True)

            (temp_dir / "images" / "train" / "cam44_north_f001000.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam44_north_f001000.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            (temp_dir / "images" / "val" / "cam44_north_f001500.jpg").write_text("fake")
            (temp_dir / "labels" / "val" / "cam44_north_f001500.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            empty_videos_dir = temp_dir / "empty_videos"
            empty_videos_dir.mkdir()

            res = audit_compiled_dataset(temp_dir, videos_dir=empty_videos_dir, ref_size=640)
            leakage = res["cross_split_leakage"]
            min_sep = leakage["minimum_separation"]

            # Video metadata was absent: delta frames is 500, but timing is flagged unknown
            self.assertIsNotNone(min_sep)
            self.assertEqual(min_sep["delta_frames"], 500)
            self.assertIsNone(min_sep["delta_sec"])
            self.assertEqual(min_sep["timing_status"], "unknown_timing_fps_unavailable")

            # Evaluation manifest handles absent holdout video gracefully
            manifest = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=empty_videos_dir,
                out_paths=[temp_dir / "eval_manifest.json"]
            )
            # cam45_northeast holdout samples should be omitted because video file is absent
            holdout_cams = [s["camera"] for s in manifest["samples"] if s["camera"] == "cam45_northeast"]
            self.assertEqual(len(holdout_cams), 0)
            self.assertTrue(any("cam45_northeast" in g for g in manifest["metadata"]["coverage_gaps"]))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_prediction_sample_matching_training_frame(self):
        """Prediction sample near a training frame must be flagged with nearby exposure when FPS is evidenced."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            (temp_dir / "images" / "train").mkdir(parents=True)
            (temp_dir / "labels" / "train").mkdir(parents=True)
            (temp_dir / "images" / "val").mkdir(parents=True)
            (temp_dir / "labels" / "val").mkdir(parents=True)
            (temp_dir / "eval_predictions").mkdir(parents=True)

            # Train has cam43_south frame 1980
            (temp_dir / "images" / "train" / "cam43_south_f001980.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam43_south_f001980.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # Prediction artifact has cam43_south frame 2000 (delta = 20 frames = 0.13s at evidenced 150fps)
            (temp_dir / "eval_predictions" / "pred_cam43_south_f002000.jpg").write_text("fake_rendered_image")

            manifest = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=temp_dir / "nonexistent_videos",
                out_paths=[temp_dir / "eval.json"],
                evidenced_fps_map={"cam43_south": 150.0}
            )

            pred_sample = next((s for s in manifest["samples"] if "f002000" in s["frame_id"]), None)
            self.assertIsNotNone(pred_sample)

            # Must NOT be marked as an unseen holdout; must detect nearby training exposure
            self.assertNotEqual(pred_sample["training_exposure"], "proven_camera_holdout")
            self.assertEqual(pred_sample["training_exposure"], "nearby_training_exposure")
            self.assertEqual(pred_sample["nearby_training_delta_frames"], 20)
            self.assertAlmostEqual(pred_sample["nearby_training_delta_sec"], 0.13, places=2)
            # Must not be eligible for quantitative evaluation because it is an unreviewed overlay render
            self.assertFalse(pred_sample["eligible_for_quantitative_eval"])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_northeast_merged_record_with_training_precedence(self):
        """Regression: when one northeast frame exists in train, val, and predictions, produce 1 merged record with train exposure."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            for s in ["train", "val"]:
                (temp_dir / "images" / s).mkdir(parents=True)
                (temp_dir / "labels" / s).mkdir(parents=True)
            (temp_dir / "eval_predictions").mkdir(parents=True)

            ne_frame_id = "cam45_northeast_f001200"

            # 1. Northeast frame exists in training split
            (temp_dir / "images" / "train" / f"{ne_frame_id}.jpg").write_text("fake_train_img")
            (temp_dir / "labels" / "train" / f"{ne_frame_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # 2. Northeast frame also exists in validation split
            (temp_dir / "images" / "val" / f"{ne_frame_id}.jpg").write_text("fake_val_img")
            (temp_dir / "labels" / "val" / f"{ne_frame_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")

            # 3. Northeast frame also exists in evaluation prediction artifacts
            (temp_dir / "eval_predictions" / f"pred_{ne_frame_id}.jpg").write_text("fake_pred_img")

            manifest = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=temp_dir / "empty_videos",
                out_paths=[temp_dir / "eval_manifest.json"],
                evidenced_fps_map={"cam45_northeast": 150.0}
            )

            # Must merge all candidates into ONE record keyed by canonical source_frame_id
            matching_samples = [s for s in manifest["samples"] if s["source_frame_id"] == ne_frame_id]
            self.assertEqual(len(matching_samples), 1)

            merged_record = matching_samples[0]
            # Training exposure must take precedence over holdout claims or val exposure
            self.assertEqual(merged_record["training_exposure"], "current_train_split")
            self.assertEqual(merged_record["nearby_training_delta_frames"], 0)

            # Retains selection reasons from both validation split and prediction artifact
            reasons = merged_record["selection_reasons"]
            self.assertTrue(any("class_representation" in r for r in reasons))
            self.assertIn("historical_qualitative_prediction_artifact", reasons)

            # Retains all artifact references across candidate sources
            refs = merged_record["artifact_references"]
            self.assertIn(f"val_split:{ne_frame_id}", refs)
            self.assertIn(f"eval_prediction:pred_{ne_frame_id}.jpg", refs)

            # All candidate counts derived from unique registry
            self.assertEqual(manifest["metadata"]["registry_total_candidates"], 1)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_missing_validation_images_and_unreviewed_eligibility(self):
        """A missing validation image cannot be clean_source_available; unreviewed annotations must be ineligible."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            (temp_dir / "images" / "train").mkdir(parents=True)
            (temp_dir / "labels" / "train").mkdir(parents=True)
            (temp_dir / "images" / "val").mkdir(parents=True)
            (temp_dir / "labels" / "val").mkdir(parents=True)

            # Val label exists, but image file does NOT exist
            (temp_dir / "labels" / "val" / "cam44_north_f001000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            # Note: temp_dir / "images" / "val" / "cam44_north_f001000.jpg" is intentionally missing!

            # Case A: Source video exists
            videos_dir_a = temp_dir / "videos_a"
            videos_dir_a.mkdir(parents=True)
            (videos_dir_a / "cam44_north.avi").write_bytes(b"RIFF....AVI ")

            manifest_a = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=videos_dir_a,
                out_paths=[temp_dir / "eval_a.json"]
            )
            sample_a = next(s for s in manifest_a["samples"] if "f001000" in s["frame_id"])
            # Must distinguish requires_video_extraction from clean existing image
            self.assertEqual(sample_a["clean_input_status"], "requires_video_extraction")
            self.assertNotEqual(sample_a["clean_input_status"], "clean_source_available")
            self.assertNotEqual(sample_a["clean_input_status"], "existing_raw_image")
            # Must be ineligible for quantitative benchmark
            self.assertFalse(sample_a["eligible_for_quantitative_eval"])
            self.assertIn("Unreviewed annotations", sample_a["eligibility_reason"])

            # Case B: Neither image nor video exists -> unavailable_input
            videos_dir_b = temp_dir / "empty_videos_b"
            videos_dir_b.mkdir(parents=True)

            manifest_b = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=videos_dir_b,
                out_paths=[temp_dir / "eval_b.json"]
            )
            sample_b = next(s for s in manifest_b["samples"] if "f001000" in s["frame_id"])
            self.assertEqual(sample_b["clean_input_status"], "unavailable_input")
            self.assertNotEqual(sample_b["clean_input_status"], "clean_source_available")
            self.assertFalse(sample_b["eligible_for_quantitative_eval"])

            # Manifest summary must reflect ineligibility
            self.assertEqual(manifest_b["metadata"]["eligibility_summary"]["eligible_for_quantitative_eval_count"], 0)
            self.assertEqual(manifest_b["metadata"]["eligibility_summary"]["ineligible_unreviewed_count"], len(manifest_b["samples"]))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_audit_external_ndjson_malformed_inputs(self):
        """Directly exercise malformed inputs through audit_external_ndjson (arrays, nulls, invalid dimensions)."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            bad_ndjson_path = temp_dir / "malformed_test.ndjson"
            lines = [
                # Line 1: Valid header
                json.dumps({"type": "dataset", "class_names": {"0": "bus", "1": "car"}}),
                # Line 2: Top-level array record (malformed)
                json.dumps([{"type": "image", "file": "array_rec.jpg"}]),
                # Line 3: Null annotations field (malformed)
                json.dumps({"type": "image", "file": "null_anno.jpg", "width": 640, "height": 640, "annotations": None}),
                # Line 4: Non-dict annotations field (malformed)
                json.dumps({"type": "image", "file": "bad_anno_type.jpg", "width": 640, "height": 640, "annotations": [1, 2, 3]}),
                # Line 5: Null boxes collection (malformed)
                json.dumps({"type": "image", "file": "null_boxes.jpg", "width": 640, "height": 640, "annotations": {"boxes": None}}),
                # Line 6: Non-list boxes collection (malformed)
                json.dumps({"type": "image", "file": "bad_boxes_type.jpg", "width": 640, "height": 640, "annotations": {"boxes": "not_a_list"}}),
                # Line 7: Null box item in boxes (malformed)
                json.dumps({"type": "image", "file": "null_box_item.jpg", "width": 640, "height": 640, "annotations": {"boxes": [None]}}),
                # Line 8: Non-numeric dimension types (malformed)
                json.dumps({"type": "image", "file": "non_numeric_dim.jpg", "width": "wide", "height": 640, "annotations": {"boxes": []}}),
                # Line 9: Negative dimension values (malformed)
                json.dumps({"type": "image", "file": "negative_dim.jpg", "width": -640, "height": 640, "annotations": {"boxes": []}}),
                # Line 10: Incomplete dimensions (malformed)
                json.dumps({"type": "image", "file": "missing_h.jpg", "width": 640, "annotations": {"boxes": []}}),
                # Line 11: Valid background image without annotations key (valid, 0 boxes)
                json.dumps({"type": "image", "file": "valid_bg.jpg", "width": 640, "height": 640, "split": "train"}),
                # Line 12: Valid annotated image (valid, 1 box)
                json.dumps({"type": "image", "file": "valid_annotated.jpg", "width": 640, "height": 640, "split": "train", "annotations": {"boxes": [[1, 0.5, 0.5, 0.2, 0.2]]}}),
            ]
            bad_ndjson_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            res = audit_external_ndjson(bad_ndjson_path)
            self.assertTrue(res["exists"])
            # Lines 2 through 10 are malformed (9 records)
            self.assertEqual(res["malformed_count"], 9)

            malformed_lines = [m["line"] for m in res["malformed_sample"]]
            for expected_line in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                self.assertIn(expected_line, malformed_lines)

            # Valid records (lines 11 and 12) were processed safely
            self.assertEqual(res["source_class_box_counts"]["car"], 1)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_unknown_timing_vs_measured_temporal_leakage(self):
        """Temporal leakage requires evidenced FPS; without FPS, timing is unknown and only frame distance is reported."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            (temp_dir / "images" / "train").mkdir(parents=True)
            (temp_dir / "labels" / "train").mkdir(parents=True)
            (temp_dir / "eval_predictions").mkdir(parents=True)

            # Train has frame 1000
            (temp_dir / "images" / "train" / "cam43_south_f001000.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam43_south_f001000.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # Prediction has frame 1020 (delta = 20 frames)
            (temp_dir / "eval_predictions" / "pred_cam43_south_f001020.jpg").write_text("fake")

            # Case A: FPS is UNAVAILABLE (no videos)
            manifest_no_fps = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=temp_dir / "empty_videos",
                out_paths=[temp_dir / "eval_no_fps.json"]
            )
            sample_no_fps = next(s for s in manifest_no_fps["samples"] if "f001020" in s["frame_id"])
            # Must NOT substitute implicit 90/300 frames; timing is unknown and exposure is unproven
            self.assertEqual(sample_no_fps["timing_status"], "unknown_timing_fps_unavailable")
            self.assertIsNone(sample_no_fps["timestamp_sec"])
            self.assertIsNone(sample_no_fps["nearby_training_delta_sec"])
            self.assertEqual(sample_no_fps["nearby_training_delta_frames"], 20)
            self.assertEqual(sample_no_fps["training_exposure"], "unproven_checkpoint_exposure")

            # Case B: FPS is EVIDENCED (150.0 FPS)
            manifest_with_fps = build_eval_sampling_manifest(
                dataset_dir=temp_dir,
                data_dir=temp_dir,
                videos_dir=temp_dir / "empty_videos",
                out_paths=[temp_dir / "eval_with_fps.json"],
                evidenced_fps_map={"cam43_south": 150.0}
            )
            sample_with_fps = next(s for s in manifest_with_fps["samples"] if "f001020" in s["frame_id"])
            # At 150 FPS, 20 frames = 0.13s <= 3.0s -> measured nearby exposure
            self.assertEqual(sample_with_fps["timing_status"], "evidenced_from_video_fps")
            self.assertIsNotNone(sample_with_fps["timestamp_sec"])
            self.assertEqual(sample_with_fps["nearby_training_delta_frames"], 20)
            self.assertAlmostEqual(sample_with_fps["nearby_training_delta_sec"], 0.13, places=2)
            self.assertEqual(sample_with_fps["training_exposure"], "nearby_training_exposure")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_report_values_change_with_fixture_data(self):
        """Verifies report markdown values dynamically change when fixture data changes (no hardcoded totals)."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            (temp_dir / "images" / "train").mkdir(parents=True)
            (temp_dir / "labels" / "train").mkdir(parents=True)
            (temp_dir / "images" / "val").mkdir(parents=True)
            (temp_dir / "labels" / "val").mkdir(parents=True)

            # Fixture A: 1 train image, 1 box
            (temp_dir / "images" / "train" / "cam44_north_f001000.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam44_north_f001000.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            report_a_path = temp_dir / "report_a.md"
            ext_mock = {"image_records": 100, "source_frame_identifiers": 50, "unique_sequences": 10, "split_overlaps": {}}
            stag_mock = {"total_verified_crops": 10, "total_matched_verified_crops": 10, "unmatched_verified_crops": [], "stashed_crops": []}
            manifest_mock = {"metadata": {"actual_coverage": {}, "exposure_breakdown": {}}}

            res_a = audit_compiled_dataset(temp_dir, videos_dir=temp_dir)
            render_markdown_report(ext_mock, res_a, stag_mock, [], manifest_mock, report_a_path)
            content_a = report_a_path.read_text(encoding="utf-8")

            self.assertIn("**1 train**", content_a)
            self.assertIn("**1 train** + **0 val** boxes", content_a)

            # Fixture B: Add 2 images and 3 more boxes
            (temp_dir / "images" / "train" / "cam44_north_f002000.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam44_north_f002000.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n")

            (temp_dir / "images" / "val" / "cam46_west_f003000.jpg").write_text("fake")
            (temp_dir / "labels" / "val" / "cam46_west_f003000.txt").write_text("3 0.5 0.5 0.2 0.2\n")

            report_b_path = temp_dir / "report_b.md"
            res_b = audit_compiled_dataset(temp_dir, videos_dir=temp_dir)
            render_markdown_report(ext_mock, res_b, stag_mock, [], manifest_mock, report_b_path)
            content_b = report_b_path.read_text(encoding="utf-8")

            # Must dynamically reflect Fixture B numbers
            self.assertIn("**2 train** + **1 val** images", content_b)
            self.assertIn("**3 train** + **1 val** boxes", content_b)
            self.assertNotEqual(content_a, content_b)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_deterministic_reproducibility(self):
        """Comparing complete outputs from two identical runs must yield byte-for-byte identity."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            (temp_dir / "images" / "train").mkdir(parents=True)
            (temp_dir / "labels" / "train").mkdir(parents=True)
            (temp_dir / "images" / "val").mkdir(parents=True)
            (temp_dir / "labels" / "val").mkdir(parents=True)

            (temp_dir / "images" / "train" / "cam44_north_f001000.jpg").write_text("fake")
            (temp_dir / "labels" / "train" / "cam44_north_f001000.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            rep1_path = temp_dir / "rep1.md"
            rep2_path = temp_dir / "rep2.md"
            man1_path = temp_dir / "man1.json"
            man2_path = temp_dir / "man2.json"

            ext_mock = {"image_records": 100, "source_frame_identifiers": 50, "unique_sequences": 10, "split_overlaps": {}}
            stag_mock = {"total_verified_crops": 5, "total_matched_verified_crops": 5, "unmatched_verified_crops": [], "stashed_crops": []}
            discrepancies = get_manual_discrepancy_review()

            # Run 1
            res1 = audit_compiled_dataset(temp_dir, videos_dir=temp_dir)
            man1 = build_eval_sampling_manifest(temp_dir, temp_dir, temp_dir, [man1_path])
            render_markdown_report(ext_mock, res1, stag_mock, discrepancies, man1, rep1_path)

            # Run 2
            res2 = audit_compiled_dataset(temp_dir, videos_dir=temp_dir)
            man2 = build_eval_sampling_manifest(temp_dir, temp_dir, temp_dir, [man2_path])
            render_markdown_report(ext_mock, res2, stag_mock, discrepancies, man2, rep2_path)

            # Must be completely identical
            self.assertEqual(rep1_path.read_text(encoding="utf-8"), rep2_path.read_text(encoding="utf-8"))
            self.assertEqual(man1_path.read_text(encoding="utf-8"), man2_path.read_text(encoding="utf-8"))

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
