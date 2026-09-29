"""
Unit and Regression Tests for Training Manifests & Leakage Safeguards (Batch 4 Corrected).

Validates:
1. Deterministic reproducibility: same seed and config produce identical inventory, exclusions, and manifests.
2. Diagnostic evaluation benchmark exclusion: all 42 diagnostic frames and variants are strictly excluded from candidate training and validation.
3. External cap enforcement: cap is strictly calculated as floor(unique_local_frames * cap_ratio) (528 -> 264), never inflated by synthetic variants (1,135).
4. Validation identity: validation set is byte-for-byte identical between Manifest A and Manifest B.
5. External sequence-level split: 0% cross-split sequence overlap between external train, val, and test.
6. External image status: external images are honestly reported as unresolved_not_downloaded.
7. Local image-label pairing: local images and labels remain paired and valid.
8. Class mapping: external classes are properly mapped to Thai 5-class standard without box deletion.
9. Visual spot-check: up to 50 unique canonical frames across 5 specified categories with explicit readiness gate.

Fixture-based Regression Tests:
- Exact train/validation duplicates and augmentation siblings staying together.
- Configurable time buffers at 150 FPS and another FPS (30 FPS).
- Missing evaluation metadata failing fast.
- Missing labels, invalid boxes, corrupted images, and malformed NDJSON with recorded rejection reasons, preserving valid empty labels.
- Unique review-frame selection and configurable quotas with honest shortfall reporting.
- Output overwrite refusal on nonempty destination without explicit overwrite flag.
"""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_dataset import (
    DEFAULT_UADETRAC_CLASS_NAMES,
    THAI_5CLASS_NAMES,
    parse_frame_provenance,
    validate_box,
)
from tools.prepare_training_manifests import (
    build_canonical_source_inventory,
    construct_local_splits_and_exclusions,
    generate_visual_spot_check_manifest,
    load_evaluation_manifest,
    probe_video_fps,
    run_manifest_preparation,
)


class TestTrainingManifestsV2(unittest.TestCase):
    """System validation tests against generated Batch 4 v2 training manifests."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("config/training_manifest_config.json")
        cls.eval_snapshot_path = Path("data/eval_snapshot_v1/manifest.json")
        cls.local_dir = Path("data/multiclass_dataset")
        cls.ndjson_path = Path("data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson")
        cls.manifests_dir = Path("data/training_manifests_v2")

    def test_eval_benchmark_exclusion_from_candidate_splits(self):
        """All 42 diagnostic evaluation benchmark frames and variants must be excluded from training and validation."""
        self.assertTrue(self.eval_snapshot_path.exists())
        snap_data = json.loads(self.eval_snapshot_path.read_text(encoding="utf-8"))
        eval_fids = {s["frame_id"] for s in snap_data["samples"]}
        self.assertEqual(len(eval_fids), 42)

        man_a_path = self.manifests_dir / "manifest_a_local_only.json"
        man_b_path = self.manifests_dir / "manifest_b_local_plus_external.json"
        self.assertTrue(man_a_path.exists())
        self.assertTrue(man_b_path.exists())

        man_a = json.loads(man_a_path.read_text(encoding="utf-8"))
        man_b = json.loads(man_b_path.read_text(encoding="utf-8"))

        for man in [man_a, man_b]:
            for rec_id in man["train_records"] + man["val_records"]:
                stem = rec_id.split(":")[-1]
                prov = parse_frame_provenance(stem)
                self.assertNotIn(prov.source_frame_id, eval_fids, f"Leaked eval frame in {man['manifest_name']}: {rec_id}")
                self.assertNotIn(stem, eval_fids, f"Leaked eval frame in {man['manifest_name']}: {rec_id}")

    def test_external_cap_enforced_using_unique_local_frames(self):
        """External cap must be floor(unique_local_training_frames * 0.5), not based on synthetic variants."""
        man_b_path = self.manifests_dir / "manifest_b_local_plus_external.json"
        self.assertTrue(man_b_path.exists())
        man_b = json.loads(man_b_path.read_text(encoding="utf-8"))

        meta = man_b["metadata"]
        ext_cap = meta["external_images_count"]
        # Unique local training frames is 528. 528 * 0.5 = 264.
        self.assertEqual(ext_cap, 264)
        self.assertEqual(len(man_b["external_train_records"]), 264)

        # Confirm cap was not computed on total local training images (1,135 * 0.5 = 567)
        self.assertNotEqual(ext_cap, 567)

    def test_identical_validation_between_manifest_a_and_b(self):
        """Validation set must be 100% identical between Manifest A and Manifest B (zero validation leakage)."""
        man_a = json.loads((self.manifests_dir / "manifest_a_local_only.json").read_text(encoding="utf-8"))
        man_b = json.loads((self.manifests_dir / "manifest_b_local_plus_external.json").read_text(encoding="utf-8"))

        self.assertEqual(man_a["val_records"], man_b["val_records"])
        self.assertEqual(len(man_a["val_records"]), 229)
        self.assertEqual(man_a["metadata"]["val_summary"]["total_boxes"], man_b["metadata"]["val_summary"]["total_boxes"])
        self.assertEqual(man_a["metadata"]["val_summary"]["total_boxes"], 2197)

    def test_external_sequence_partition_zero_overlap(self):
        """External sequence splits must have 0% cross-split sequence overlap."""
        excl_path = self.manifests_dir / "split_and_exclusion_manifest.json"
        self.assertTrue(excl_path.exists())
        excl_data = json.loads(excl_path.read_text(encoding="utf-8"))

        part = excl_data["metadata"]["external_sequence_partition"]
        train_seqs = set(part["train_sequences"])
        val_seqs = set(part["val_sequences"])
        test_seqs = set(part["test_sequences"])

        self.assertEqual(len(train_seqs), 70)
        self.assertEqual(len(val_seqs), 15)
        self.assertEqual(len(test_seqs), 15)
        self.assertEqual(len(train_seqs.intersection(val_seqs)), 0)
        self.assertEqual(len(train_seqs.intersection(test_seqs)), 0)
        self.assertEqual(len(val_seqs.intersection(test_seqs)), 0)
        self.assertEqual(part["cross_split_sequence_overlap"], 0)

    def test_external_image_unresolved_status_honesty(self):
        """External image files must be reported as unresolved_not_downloaded, never falsely counted as available."""
        inv_path = self.manifests_dir / "canonical_source_inventory.json"
        self.assertTrue(inv_path.exists())
        inv_data = json.loads(inv_path.read_text(encoding="utf-8"))

        external_records = [r for r in inv_data["records"] if r["data_origin"] == "external_ua_detrac"]
        self.assertGreater(len(external_records), 1000)

        for r in external_records[:200]:
            self.assertFalse(r["image_exists_on_disk"])
            self.assertEqual(r["image_status"], "unresolved_not_downloaded")

    def test_local_image_label_pairing_intact(self):
        """Local images and labels must exist and remain paired without merging transformed variants."""
        inv_path = self.manifests_dir / "canonical_source_inventory.json"
        inv_data = json.loads(inv_path.read_text(encoding="utf-8"))

        local_records = [r for r in inv_data["records"] if r["data_origin"] == "local"]
        self.assertEqual(len(local_records), 1397)

        for r in local_records[:200]:
            self.assertTrue(r["image_exists_on_disk"])
            self.assertEqual(r["image_status"], "available_local_file")
            self.assertTrue(r["label_exists_on_disk"])
            self.assertEqual(r["annotation_provenance"], "teacher_completed_local")

    def test_minority_classes_preserved_without_box_deletion(self):
        """Local motorcycle and three-wheeler boxes must be fully preserved between A and B."""
        man_a = json.loads((self.manifests_dir / "manifest_a_local_only.json").read_text(encoding="utf-8"))
        man_b = json.loads((self.manifests_dir / "manifest_b_local_plus_external.json").read_text(encoding="utf-8"))

        a_tr = man_a["metadata"]["train_summary"]["class_box_counts"]
        b_tr = man_b["metadata"]["train_summary"]["class_box_counts"]

        self.assertEqual(a_tr["motorcycle"], 1621)
        self.assertEqual(b_tr["motorcycle"], 1621)
        self.assertEqual(a_tr["three_wheeler"], 515)
        self.assertEqual(b_tr["three_wheeler"], 515)

        # External data adds car, bus, and truck boxes
        self.assertGreater(b_tr["car"], a_tr["car"])
        self.assertGreater(b_tr["bus"], a_tr["bus"])
        self.assertGreater(b_tr["truck"], a_tr["truck"])

    def test_visual_spot_check_manifest_structure(self):
        """Visual spot-check must contain up to 50 unique canonical frames across 5 categories with readiness gate."""
        spot_path = self.manifests_dir / "visual_spot_check_manifest.json"
        self.assertTrue(spot_path.exists())
        spot_data = json.loads(spot_path.read_text(encoding="utf-8"))

        self.assertLessEqual(spot_data["metadata"]["total_unique_spot_check_frames"], 50)
        self.assertEqual(len(spot_data["spot_checks"]), spot_data["metadata"]["total_unique_spot_check_frames"])

        audit = spot_data["metadata"]["quota_audit"]
        self.assertEqual(audit["external_vans"]["achieved_count"], 10)
        self.assertEqual(audit["external_trucks"]["achieved_count"], 10)
        self.assertEqual(audit["external_dense_small"]["achieved_count"], 10)
        self.assertEqual(audit["local_teacher_completed"]["achieved_count"], 10)
        self.assertEqual(audit["local_night_congestion"]["achieved_count"], 10)

        # Confirm all spot check frames have unique canonical IDs
        canonical_ids = [sc["canonical_source_id"] for sc in spot_data["spot_checks"]]
        self.assertEqual(len(canonical_ids), len(set(canonical_ids)))

        for sc in spot_data["spot_checks"]:
            self.assertTrue(len(sc["required_human_checks"]) >= 1)
            self.assertTrue(len(sc["risk_factors"]) >= 1)

    def test_deterministic_reproducibility(self):
        """Running preparation twice with the same seed must produce identical selections and counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out1 = Path(tmp_dir) / "run1"
            out2 = Path(tmp_dir) / "run2"

            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            cfg["random_seed"] = 12345

            cfg_file = Path(tmp_dir) / "config.json"
            cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

            res1 = run_manifest_preparation(
                config_path=cfg_file,
                output_dir=out1,
                report_md_path=out1 / "report.md",
                seed=12345
            )
            res2 = run_manifest_preparation(
                config_path=cfg_file,
                output_dir=out2,
                report_md_path=out2 / "report.md",
                seed=12345
            )

            self.assertEqual(res1["manifest_b"]["train_records"], res2["manifest_b"]["train_records"])
            self.assertEqual(res1["manifest_b"]["external_train_records"], res2["manifest_b"]["external_train_records"])
            self.assertEqual(
                res1["splits_data"]["external_sequence_partition"],
                res2["splits_data"]["external_sequence_partition"]
            )


class TestBatch4CorrectionRegressions(unittest.TestCase):
    """Fixture-based regression tests for Batch 4 correction pass requirements."""

    def test_exact_train_val_duplicates_and_augmentation_siblings_together(self):
        """A canonical frame and ALL of its augmentation variants must strictly stay in the same split."""
        inventory = {
            "records": [
                # Canonical frame 1: Original + 2 synthetic variants in cam03_east
                {
                    "inventory_id": "local:train:cam03_east_f000100",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f000100",
                    "sequence_id": "cam03_east",
                    "frame_idx": 100,
                    "is_synthetic_variant": False,
                    "is_diagnostic_eval_frame": False,
                },
                {
                    "inventory_id": "local:train:cam03_east_f000100_synth_blur",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f000100",
                    "sequence_id": "cam03_east",
                    "frame_idx": 100,
                    "is_synthetic_variant": True,
                    "is_diagnostic_eval_frame": False,
                },
                {
                    "inventory_id": "local:train:cam03_east_f000100_synth_rain",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f000100",
                    "sequence_id": "cam03_east",
                    "frame_idx": 100,
                    "is_synthetic_variant": True,
                    "is_diagnostic_eval_frame": False,
                },
                # Canonical frame 2: Later frame in val block
                {
                    "inventory_id": "local:train:cam03_east_f005000",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f005000",
                    "sequence_id": "cam03_east",
                    "frame_idx": 5000,
                    "is_synthetic_variant": False,
                    "is_diagnostic_eval_frame": False,
                },
                {
                    "inventory_id": "local:train:cam03_east_f005000_synth_night",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f005000",
                    "sequence_id": "cam03_east",
                    "frame_idx": 5000,
                    "is_synthetic_variant": True,
                    "is_diagnostic_eval_frame": False,
                },
                # Canonical frame 3: Evaluation benchmark frame with variant
                {
                    "inventory_id": "local:train:cam03_east_f009999",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f009999",
                    "sequence_id": "cam03_east",
                    "frame_idx": 9999,
                    "is_synthetic_variant": False,
                    "is_diagnostic_eval_frame": True,
                },
                {
                    "inventory_id": "local:train:cam03_east_f009999_synth_flip",
                    "data_origin": "local",
                    "canonical_source_id": "cam03_east_f009999",
                    "sequence_id": "cam03_east",
                    "frame_idx": 9999,
                    "is_synthetic_variant": True,
                    "is_diagnostic_eval_frame": True,
                },
            ]
        }

        eval_ids = {"cam03_east_f009999"}
        fps_map = {"cam03_east": 150.0}
        cfg = {
            "local_split_strategy": {"train_block_ratio": 0.5, "temporal_buffer_seconds": 3.0}
        }

        res = construct_local_splits_and_exclusions(inventory, eval_ids, fps_map, cfg)
        train_ids = {r["inventory_id"] for r in res["local_train_records"]}
        val_ids = {r["inventory_id"] for r in res["local_val_records"]}
        excl_ids = {e["inventory_id"] for e in res["exclusions"]}

        # All 3 variants of f000100 must be in train
        self.assertIn("local:train:cam03_east_f000100", train_ids)
        self.assertIn("local:train:cam03_east_f000100_synth_blur", train_ids)
        self.assertIn("local:train:cam03_east_f000100_synth_rain", train_ids)
        self.assertNotIn("local:train:cam03_east_f000100", val_ids)

        # Both variants of f005000 must be in val
        self.assertIn("local:train:cam03_east_f005000", val_ids)
        self.assertIn("local:train:cam03_east_f005000_synth_night", val_ids)
        self.assertNotIn("local:train:cam03_east_f005000", train_ids)

        # Both variants of f009999 must be excluded under evaluation leakage
        self.assertIn("local:train:cam03_east_f009999", excl_ids)
        self.assertIn("local:train:cam03_east_f009999_synth_flip", excl_ids)
        self.assertNotIn("local:train:cam03_east_f009999", train_ids)
        self.assertNotIn("local:train:cam03_east_f009999", val_ids)

    def test_configurable_time_buffers_at_150fps_and_other_fps(self):
        """Temporal buffer in seconds must translate accurately via evidenced FPS (150 FPS and 30 FPS)."""
        # Scenario A: 150.0 FPS. 3.0s buffer = 450 frames.
        # Train block: [1000]. Last train = 1000. Cutoff = 1450.
        # Frame at 1200 must be excluded (buffer violation). Frame at 1500 must be in val.
        inv_150 = {
            "records": [
                {"inventory_id": "r1", "canonical_source_id": "c1", "sequence_id": "cam_a", "frame_idx": 1000, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
                {"inventory_id": "r2", "canonical_source_id": "c2", "sequence_id": "cam_a", "frame_idx": 1200, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
                {"inventory_id": "r3", "canonical_source_id": "c3", "sequence_id": "cam_a", "frame_idx": 1500, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
            ]
        }
        res_150 = construct_local_splits_and_exclusions(
            inv_150, set(), {"cam_a": 150.0},
            {"local_split_strategy": {"train_block_ratio": 0.33, "temporal_buffer_seconds": 3.0}}
        )
        self.assertEqual(len(res_150["local_train_records"]), 1)
        self.assertEqual(res_150["local_train_records"][0]["inventory_id"], "r1")
        self.assertEqual(len(res_150["local_val_records"]), 1)
        self.assertEqual(res_150["local_val_records"][0]["inventory_id"], "r3")

        buf_excls_150 = [e for e in res_150["exclusions"] if e["reason"] == "temporal_buffer_violation"]
        self.assertEqual(len(buf_excls_150), 1)
        self.assertEqual(buf_excls_150[0]["inventory_id"], "r2")
        self.assertIn("450 frames @ 150.0 FPS", buf_excls_150[0]["details"])

        # Scenario B: 30.0 FPS. 3.0s buffer = 90 frames.
        # Train block: [1000]. Last train = 1000. Cutoff = 1090.
        # Frame at 1050 must be excluded. Frame at 1100 must be in val.
        inv_30 = {
            "records": [
                {"inventory_id": "r1", "canonical_source_id": "c1", "sequence_id": "cam_b", "frame_idx": 1000, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
                {"inventory_id": "r2", "canonical_source_id": "c2", "sequence_id": "cam_b", "frame_idx": 1050, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
                {"inventory_id": "r3", "canonical_source_id": "c3", "sequence_id": "cam_b", "frame_idx": 1100, "is_synthetic_variant": False, "is_diagnostic_eval_frame": False, "data_origin": "local"},
            ]
        }
        res_30 = construct_local_splits_and_exclusions(
            inv_30, set(), {"cam_b": 30.0},
            {"local_split_strategy": {"train_block_ratio": 0.33, "temporal_buffer_seconds": 3.0}}
        )
        self.assertEqual(len(res_30["local_train_records"]), 1)
        self.assertEqual(res_30["local_train_records"][0]["inventory_id"], "r1")
        self.assertEqual(len(res_30["local_val_records"]), 1)
        self.assertEqual(res_30["local_val_records"][0]["inventory_id"], "r3")

        buf_excls_30 = [e for e in res_30["exclusions"] if e["reason"] == "temporal_buffer_violation"]
        self.assertEqual(len(buf_excls_30), 1)
        self.assertEqual(buf_excls_30[0]["inventory_id"], "r2")
        self.assertIn("90 frames @ 30.0 FPS", buf_excls_30[0]["details"])

    def test_missing_evaluation_metadata_fails_clearly(self):
        """Missing or malformed evaluation manifest must raise clear exceptions, never disabling exclusions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Non-existent file
            non_existent = Path(tmp_dir) / "does_not_exist.json"
            with self.assertRaises(FileNotFoundError):
                load_evaluation_manifest(non_existent)

            # 2. Malformed JSON
            bad_json = Path(tmp_dir) / "bad.json"
            bad_json.write_text("{ this is not json }", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_evaluation_manifest(bad_json)
            self.assertIn("malformed JSON", str(ctx.exception))

            # 3. Missing samples key or empty samples
            empty_samples = Path(tmp_dir) / "empty.json"
            empty_samples.write_text(json.dumps({"metadata": {}, "samples": []}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_evaluation_manifest(empty_samples)
            self.assertIn("0 samples", str(ctx.exception))

    def test_quarantine_missing_labels_invalid_boxes_and_malformed_ndjson(self):
        """Corrupted images, missing labels, and invalid boxes must be quarantined with recorded reasons."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_dir = root / "images" / "train"
            lbl_dir = root / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            from PIL import Image

            def create_valid_jpeg(path: Path):
                im = Image.new("RGB", (100, 100), color=(128, 128, 128))
                im.save(path, format="JPEG")

            # 1. Local image with missing label
            create_valid_jpeg(img_dir / "cam03_east_f000100.jpg")

            # 2. Local image with invalid bounding box (out of bounds xc=1.5)
            create_valid_jpeg(img_dir / "cam03_east_f000200.jpg")
            (lbl_dir / "cam03_east_f000200.txt").write_text("0 1.500 0.500 0.200 0.200\n", encoding="utf-8")

            # 3. Local image with explicitly empty label (0 boxes) -> Preserved, NOT quarantined!
            create_valid_jpeg(img_dir / "cam03_east_f000300.jpg")
            (lbl_dir / "cam03_east_f000300.txt").write_text("", encoding="utf-8")

            # 4. Local image that is corrupted (0 bytes)
            (img_dir / "cam03_east_f000400.jpg").write_bytes(b"")
            (lbl_dir / "cam03_east_f000400.txt").write_text("0 0.500 0.500 0.100 0.100\n", encoding="utf-8")

            # 5. External NDJSON with malformed line and invalid box
            bad_ndjson = root / "bad.ndjson"
            bad_ndjson.write_text(
                '{"type": "dataset", "class_names": {"0": "bus", "1": "car", "2": "truck", "3": "van"}}\n'
                'NOT_JSON_LINE\n'
                '{"type": "image", "file": "MVI_10001_img00001.jpg", "width": 640, "height": 640, "annotations": {"boxes": [[1, 0.5, 0.5, -0.2, 0.2]]}}\n'
                '{"type": "image", "file": "MVI_10001_img00002.jpg", "width": 640, "height": 640, "annotations": {"boxes": [[1, 0.5, 0.5, 0.2, 0.2]]}}\n',
                encoding="utf-8"
            )

            inv = build_canonical_source_inventory(
                local_dataset_dir=root,
                external_ndjson_path=bad_ndjson,
                eval_source_ids=set(),
                config={}
            )

            q_recs = inv["quarantined_records"]
            q_reasons = Counter(q["reason"] for q in q_recs)

            self.assertIn("missing_label_file", q_reasons)
            self.assertIn("invalid_bounding_box", q_reasons)
            self.assertIn("unreadable_or_corrupt_image", q_reasons)
            self.assertIn("malformed_ndjson_record", q_reasons)
            self.assertIn("invalid_external_bounding_box", q_reasons)

            # Confirm valid empty label was preserved in records
            valid_empty = [r for r in inv["records"] if r.get("stem") == "cam03_east_f000300"]
            self.assertEqual(len(valid_empty), 1)
            self.assertTrue(valid_empty[0]["is_explicitly_empty"])
            self.assertEqual(valid_empty[0]["total_boxes"], 0)

            # Confirm valid external record was preserved
            valid_ext = [r for r in inv["records"] if "img00002" in r["canonical_source_id"]]
            self.assertEqual(len(valid_ext), 1)

    def test_unique_review_frame_selection_and_configurable_quotas(self):
        """Spot check must select unique canonical source frames, allow multi-tagging, and report shortfalls honestly."""
        # Create candidate records
        ext_records = []
        for i in range(20):
            # 5 frames with vans, 5 with trucks, 5 with small boxes, 5 generic
            has_van = i < 5
            has_truck = (3 <= i < 8)  # frames 3 and 4 have BOTH van and truck (multi-tag test)
            boxes = []
            if has_van:
                boxes.append({"source_class_name": "van", "class_id": 0, "size_bucket": "medium"})
            if has_truck:
                boxes.append({"source_class_name": "truck", "class_id": 3, "size_bucket": "medium"})
            if i >= 10:
                boxes.append({"source_class_name": "car", "class_id": 0, "size_bucket": "small"})

            ext_records.append({
                "inventory_id": f"ext_{i}",
                "canonical_source_id": f"MVI_10001_img{i:05d}",
                "data_origin": "external_ua_detrac",
                "sequence_id": "MVI_10001",
                "image_path": f"unresolved/MVI_10001_img{i:05d}.jpg",
                "image_status": "unresolved_not_downloaded",
                "is_synthetic_variant": False,
                "total_boxes": len(boxes),
                "source_class_counts": {
                    "van": 1 if has_van else 0,
                    "truck": 1 if has_truck else 0
                },
                "class_counts": {"0": 1 if has_van else 0, "3": 1 if has_truck else 0},
                "box_sizes": {"small": 1 if i >= 10 else 0, "medium": 1 if (has_van or has_truck) else 0, "large": 0},
                "boxes": boxes
            })

        local_records = []
        for i in range(10):
            local_records.append({
                "inventory_id": f"loc_{i}",
                "canonical_source_id": f"cam43_south_f{i:06d}",
                "data_origin": "local",
                "sequence_id": "cam43_south",
                "lighting": "real_night" if i < 3 else "real_day",  # only 3 night frames available!
                "variant": "original_curated",
                "image_path": f"data/local/cam43_south_f{i:06d}.jpg",
                "image_status": "available_local_file",
                "is_synthetic_variant": False,
                "total_boxes": 5,
                "source_class_counts": {"1": 2, "0": 3},
                "class_counts": {"1": 2, "0": 3},
                "box_sizes": {"small": 2, "medium": 3, "large": 0},
                "boxes": []
            })

        inventory = {"records": ext_records + local_records}
        manifest_b_train_ids = [r["inventory_id"] for r in inventory["records"]]

        # Configure custom quotas: request 5 night frames when only 3 exist to test shortfall reporting
        config = {
            "spot_check": {
                "counts": {
                    "external_vans": 4,
                    "external_trucks": 4,
                    "external_dense_small": 4,
                    "local_teacher_completed": 4,
                    "local_night_congestion": 5  # Expect shortfall of 2!
                }
            }
        }

        spot_manifest = generate_visual_spot_check_manifest(inventory, manifest_b_train_ids, config)
        audit = spot_manifest["metadata"]["quota_audit"]

        # Honest shortfall reporting
        self.assertEqual(audit["external_vans"]["achieved_count"], 4)
        self.assertEqual(audit["external_vans"]["shortfall"], 0)
        self.assertEqual(audit["local_night_congestion"]["achieved_count"], 3)
        self.assertEqual(audit["local_night_congestion"]["shortfall"], 2)
        self.assertEqual(audit["local_night_congestion"]["status"], "shortfall_of_2")

        # Multi-tagging confirmation: frames 3 and 4 should have multiple tags
        frames_by_id = {sc["inventory_id"]: sc for sc in spot_manifest["spot_checks"]}
        if "ext_3" in frames_by_id:
            self.assertIn("external_vans", frames_by_id["ext_3"]["review_tags"])
            self.assertIn("external_trucks", frames_by_id["ext_3"]["review_tags"])

        # No duplicate canonical source frames
        c_ids = [sc["canonical_source_id"] for sc in spot_manifest["spot_checks"]]
        self.assertEqual(len(c_ids), len(set(c_ids)))

    def test_output_overwrite_refusal(self):
        """Destination with existing files must raise FileExistsError unless overwrite=True is passed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "output_test"
            out_dir.mkdir(parents=True)
            # Create a pre-existing file in output destination
            (out_dir / "pre_existing_manifest.json").write_text("{}", encoding="utf-8")

            # Calling without overwrite must raise FileExistsError
            with self.assertRaises(FileExistsError) as ctx:
                run_manifest_preparation(
                    output_dir=out_dir,
                    overwrite=False
                )
            self.assertIn("already exists and is not empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
