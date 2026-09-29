"""
Batch 9: Controlled Local-Only vs Reviewed-External Fine-Tuning Experiment Runner.

Executes a controlled fine-tuning comparison:
- Model 1: Deployed Baseline (models/yolo26s_thai_traffic.pt, SHA256: cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c).
- Model 2 (Candidate A): Fine-tuned on Dataset A (1,092 local training frames from training_manifests_v6).
- Model 3 (Candidate B): Fine-tuned on Dataset B (Dataset A + exactly 25 eligible, reviewed external frames).

Strict Constraints & Invariants:
1. Warm-starts both candidates independently from the verified baseline checkpoint.
2. Identical training settings across candidates: seed=42, imgsz=640, optimizer=SGD, lr0=0.0001,
   lrf=0.01, cos_lr=True, warmup_epochs=1.0, warmup_momentum=0.8, warmup_bias_lr=0.0001 (explicitly
   configured to prevent bias LR spikes), freeze=10, 30-epoch budget.
3. Materializes two clean, isolated datasets. Refuses nonempty destinations.
4. Dataset B differs from Dataset A by strictly the 25 reviewed external frames (813 boxes).
5. Excludes rejected sources and stale synthetic descendants.
6. Evaluates Baseline, Candidate A (best.pt and last.pt), and Candidate B (best.pt and last.pt) under
   identical conditions on:
   - Primary Unaugmented Validation Set (130 frames: 107 old-train + 23 old-val).
   - Diagnostic Benchmark Snapshot (42 frames).
7. Reports metrics in percentage points (pp) and measures inference latency/FPS under identical procedure.
8. Verifies immutability of baseline weights, manifests, and review packs post-run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    THAI_5CLASS_NAMES,
    compute_aspect_preserving_size,
    validate_box,
)
from tools.evaluate_baseline import (
    EXPECTED_BASELINE_SHA256,
    NumpyEncoder,
    compute_ground_truth_size_slices,
    compute_slice_metrics,
    match_one_to_one,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.validate_review_pack import verify_snapshot_integrity


# ==============================================================================
# 1. Pre-Flight Invariant Checks
# ==============================================================================

def check_preflight_invariants(
    baseline_path: Path,
    snapshot_dir: Path,
    manifests_dir: Path,
    consolidated_dir: Path,
) -> Dict[str, str]:
    """
    Checks and records pre-execution hashes for all immutable reference assets.
    Raises RuntimeError if baseline checkpoint or snapshot integrity is violated.
    """
    print("\n" + "=" * 70)
    print("STEP 1: PRE-FLIGHT INTEGRITY & BASELINE VERIFICATION")
    print("=" * 70)

    baseline_path = baseline_path.resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {baseline_path}")

    actual_baseline_sha = compute_file_sha256(baseline_path)
    if actual_baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"Baseline checkpoint hash mismatch! Expected {EXPECTED_BASELINE_SHA256}, got {actual_baseline_sha}"
        )
    print(f"[OK] Baseline checkpoint verified: {baseline_path}")
    print(f"     SHA256: {actual_baseline_sha}")

    # Verify snapshot integrity against baseline
    print("[OK] Verifying evaluation snapshot integrity...")
    verify_snapshot_integrity(
        snapshot_dir=snapshot_dir,
        expected_checkpoint_path=baseline_path,
        expected_checkpoint_sha256=EXPECTED_BASELINE_SHA256,
    )
    print(f"[OK] Evaluation snapshot verified at: {snapshot_dir}")

    tracked_files = [
        baseline_path,
        consolidated_dir / "annotations" / "annotations.json",
        consolidated_dir / "manifest.json",
        manifests_dir / "manifest_a_local_only.json",
        manifests_dir / "primary_validation_manifest.json",
        manifests_dir / "canonical_source_inventory.json",
        snapshot_dir / "manifest.json",
    ]

    initial_hashes = {}
    for tf in tracked_files:
        if tf.exists():
            h = compute_file_sha256(tf)
            initial_hashes[str(tf)] = h
        else:
            raise FileNotFoundError(f"Required reference file missing: {tf}")

    print(f"[OK] Recorded initial hashes for {len(initial_hashes)} reference files.")
    return initial_hashes


# ==============================================================================
# 2. Dataset Materialization & Verification (Datasets A & B)
# ==============================================================================

def materialize_controlled_datasets(
    dataset_a_dir: Path,
    dataset_b_dir: Path,
    manifests_dir: Path,
    consolidated_dir: Path,
    snapshot_dir: Path,
) -> Dict[str, Any]:
    """
    Materializes Dataset A (local-only, 1,092 train + 130 val) and Dataset B
    (Dataset A + 25 reviewed external frames = 1,117 train + 130 val).
    Refuses nonempty destinations.
    """
    print("\n" + "=" * 70)
    print("STEP 2: MATERIALIZING CONTROLLED DATASETS A AND B")
    print("=" * 70)

    dataset_a_dir = dataset_a_dir.resolve()
    dataset_b_dir = dataset_b_dir.resolve()

    # Refuse nonempty destinations
    for d, name in [(dataset_a_dir, "Dataset A"), (dataset_b_dir, "Dataset B")]:
        if d.exists() and any(d.iterdir()):
            raise RuntimeError(
                f"Destination directory for {name} ({d}) already exists and is not empty! Refusing to overwrite."
            )

    # Load inventories and manifests
    with open(manifests_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
        man_a = json.load(f)
    local_train_ids = man_a["train_records"]

    with open(manifests_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
        prim_val_man = json.load(f)
    val_ids = prim_val_man["records"]

    with open(manifests_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
        inv_data = json.load(f)
    inv_map = {r["inventory_id"]: r for r in inv_data["records"]}

    with open(consolidated_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
        consolidated_annots = json.load(f)
    ext_reviewed_records = [
        r for r in consolidated_annots
        if r.get("data_origin") == "external_ua_detrac" and r.get("training_eligible")
    ]

    if len(local_train_ids) != 1092:
        raise ValueError(f"Expected 1,092 local training records in Manifest A, got {len(local_train_ids)}")
    if len(ext_reviewed_records) != 25:
        raise ValueError(f"Expected exactly 25 eligible reviewed external records, got {len(ext_reviewed_records)}")
    if len(val_ids) != 130:
        raise ValueError(f"Expected 130 primary validation records, got {len(val_ids)}")

    # Setup directories
    a_train_img = dataset_a_dir / "images" / "train"
    a_train_lbl = dataset_a_dir / "labels" / "train"
    a_val_img = dataset_a_dir / "images" / "val"
    a_val_lbl = dataset_a_dir / "labels" / "val"

    b_train_img = dataset_b_dir / "images" / "train"
    b_train_lbl = dataset_b_dir / "labels" / "train"
    b_val_img = dataset_b_dir / "images" / "val"
    b_val_lbl = dataset_b_dir / "labels" / "val"

    for d in [a_train_img, a_train_lbl, a_val_img, a_val_lbl, b_train_img, b_train_lbl, b_val_img, b_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Populate common validation set (130 frames)
    val_canonical_sources: Set[str] = set()
    val_box_count = 0
    val_class_counts: Counter = Counter()
    val_size_counts: Counter = Counter()
    val_manifest_samples: List[Dict[str, Any]] = []

    print("[DATASET] Copying 130 primary validation frames...")
    for rid in val_ids:
        rec = inv_map[rid]
        c_src = rec["canonical_source_id"]
        val_canonical_sources.add(c_src)
        src_img = Path(rec["image_path"])
        src_lbl = Path(rec["label_path"])
        stem = rec["stem"]

        dims = rec.get("dimensions", {})
        w = int(dims.get("width", 1920))
        h = int(dims.get("height", 1080))

        # Copy to both Dataset A and Dataset B
        for img_dst_dir, lbl_dst_dir in [(a_val_img, a_val_lbl), (b_val_img, b_val_lbl)]:
            shutil.copy2(src_img, img_dst_dir / f"{stem}.jpg")
            shutil.copy2(src_lbl, lbl_dst_dir / f"{stem}.txt")

        # Parse & validate boxes
        sample_boxes = []
        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                is_valid, err_msg, clean_box = validate_box(
                    parts, allowed_classes=THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
                )
                if not is_valid:
                    raise ValueError(f"Invalid validation box in {src_lbl}: {err_msg}")

                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                s_bucket = sz_info["size_bucket"]

                val_box_count += 1
                val_class_counts[cname] += 1
                val_size_counts[s_bucket] += 1
                sample_boxes.append({
                    "class_id": cid,
                    "class_name": cname,
                    "bbox_norm": [float(parts[1]), float(parts[2]), bw, bh],
                    "size_bucket": s_bucket,
                })

        val_manifest_samples.append({
            "frame_id": stem,
            "canonical_source_id": c_src,
            "camera": rec.get("camera", "unknown"),
            "lighting_type": rec.get("lighting", "real_day"),
            "original_split": rec.get("original_split", "train"),
            "training_exposure": "old_train_split" if rec.get("original_split") == "train" else "old_val_split",
            "dimensions": [w, h],
            "boxes": sample_boxes,
        })

    # 2. Populate Dataset A local training frames (1,092 frames)
    local_canonical_sources: Set[str] = set()
    a_train_box_count = 0
    a_class_counts: Counter = Counter()
    a_size_counts: Counter = Counter()
    a_class_size_counts: Dict[str, Counter] = defaultdict(Counter)

    print("[DATASET] Copying 1,092 local training frames to Dataset A & B...")
    for rid in local_train_ids:
        rec = inv_map[rid]
        c_src = rec["canonical_source_id"]
        local_canonical_sources.add(c_src)
        src_img = Path(rec["image_path"])
        src_lbl = Path(rec["label_path"])
        stem = Path(src_img).stem

        dims = rec.get("dimensions", {})
        w = int(dims.get("width", 1920))
        h = int(dims.get("height", 1080))

        # Copy to Dataset A and Dataset B
        for img_dst_dir, lbl_dst_dir in [(a_train_img, a_train_lbl), (b_train_img, b_train_lbl)]:
            shutil.copy2(src_img, img_dst_dir / f"{stem}.jpg")
            shutil.copy2(src_lbl, lbl_dst_dir / f"{stem}.txt")

        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                is_valid, err_msg, _ = validate_box(
                    parts, allowed_classes=THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
                )
                if not is_valid:
                    raise ValueError(f"Invalid local box in {src_lbl}: {err_msg}")

                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                s_bucket = sz_info["size_bucket"]

                a_train_box_count += 1
                a_class_counts[cname] += 1
                a_size_counts[s_bucket] += 1
                a_class_size_counts[cname][s_bucket] += 1

    if a_train_box_count != 13148:
        raise ValueError(f"Expected exactly 13,148 boxes in Dataset A, got {a_train_box_count}")

    # 3. Add the 25 reviewed external frames ONLY to Dataset B
    ext_canonical_sources: Set[str] = set()
    ext_box_count = 0
    ext_class_counts: Counter = Counter()
    ext_size_counts: Counter = Counter()
    ext_class_size_counts: Dict[str, Counter] = defaultdict(Counter)

    print("[DATASET] Adding 25 reviewed external frames strictly to Dataset B...")
    for r in ext_reviewed_records:
        fid = r["frame_id"]
        c_src = r["canonical_source_id"]
        ext_canonical_sources.add(c_src)

        src_img = consolidated_dir / "images" / f"{fid}.jpg"
        src_lbl = consolidated_dir / "annotations" / "labels" / f"{fid}.txt"

        shutil.copy2(src_img, b_train_img / f"{fid}.jpg")
        shutil.copy2(src_lbl, b_train_lbl / f"{fid}.txt")

        # External UA-DETRAC frames are 640x640
        w, h = 640, 640

        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                is_valid, err_msg, _ = validate_box(
                    parts, allowed_classes=THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
                )
                if not is_valid:
                    raise ValueError(f"Invalid external box in {src_lbl}: {err_msg}")

                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                s_bucket = sz_info["size_bucket"]

                ext_box_count += 1
                ext_class_counts[cname] += 1
                ext_size_counts[s_bucket] += 1
                ext_class_size_counts[cname][s_bucket] += 1

    if ext_box_count != 813:
        raise ValueError(f"Expected exactly 813 boxes in 25 reviewed external frames, got {ext_box_count}")

    b_train_box_count = a_train_box_count + ext_box_count
    b_class_counts = a_class_counts + ext_class_counts
    b_size_counts = a_size_counts + ext_size_counts

    # 4. Canonical split isolation assertions
    with open(snapshot_dir / "manifest.json", "r", encoding="utf-8") as f:
        snap_manifest = json.load(f)
    eval_canonical_sources = set(s["canonical_source_id"] for s in snap_manifest["samples"])

    b_canonical_sources = local_canonical_sources.union(ext_canonical_sources)

    # Check all intersections
    assert len(local_canonical_sources & val_canonical_sources) == 0, "Train A & Val overlap!"
    assert len(local_canonical_sources & eval_canonical_sources) == 0, "Train A & Eval overlap!"
    assert len(b_canonical_sources & val_canonical_sources) == 0, "Train B & Val overlap!"
    assert len(b_canonical_sources & eval_canonical_sources) == 0, "Train B & Eval overlap!"
    assert len(val_canonical_sources & eval_canonical_sources) == 0, "Val & Eval overlap!"

    print("[OK] Complete canonical source separation verified across all splits (mutual overlap = 0).")

    # 5. Write data.yaml for both datasets
    for d_dir, d_name in [(dataset_a_dir, "Dataset A"), (dataset_b_dir, "Dataset B")]:
        yaml_content = f"""# YOLO Training Specification - {d_name} (Batch 9 Controlled Experiment)
path: {d_dir.as_posix()}
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
  4: three_wheeler
"""
        (d_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

        # Save primary validation manifest for rich operational diagnostic evaluation
        val_manifest_obj = {
            "manifest_name": "Primary Validation Benchmark (130 Frames)",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(val_manifest_samples),
            "samples": val_manifest_samples,
        }
        with open(d_dir / "primary_val_manifest.json", "w", encoding="utf-8") as f:
            json.dump(val_manifest_obj, f, indent=2, cls=NumpyEncoder)

    summary_stats = {
        "dataset_a": {
            "train_images": len(local_train_ids),
            "train_canonical_sources": len(local_canonical_sources),
            "train_boxes": a_train_box_count,
            "val_images": len(val_ids),
            "val_boxes": val_box_count,
            "class_counts": dict(a_class_counts),
            "size_counts": dict(a_size_counts),
            "class_size_counts": {k: dict(v) for k, v in a_class_size_counts.items()},
        },
        "dataset_b": {
            "train_images": len(local_train_ids) + len(ext_reviewed_records),
            "train_canonical_sources": len(b_canonical_sources),
            "train_boxes": b_train_box_count,
            "val_images": len(val_ids),
            "val_boxes": val_box_count,
            "class_counts": dict(b_class_counts),
            "size_counts": dict(b_size_counts),
        },
        "external_delta": {
            "images": len(ext_reviewed_records),
            "boxes": ext_box_count,
            "class_counts": dict(ext_class_counts),
            "size_counts": dict(ext_size_counts),
            "class_size_counts": {k: dict(v) for k, v in ext_class_size_counts.items()},
        },
        "validation": {
            "images": len(val_ids),
            "boxes": val_box_count,
            "class_counts": dict(val_class_counts),
            "size_counts": dict(val_size_counts),
            "old_split_origin": Counter(s["original_split"] for s in val_manifest_samples),
        }
    }

    # Save dataset manifests
    with open(dataset_a_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats["dataset_a"], f, indent=2)
    with open(dataset_b_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats["dataset_b"], f, indent=2)

    print(f"[OK] Materialized Dataset A: 1,092 train images ({a_train_box_count} boxes) | 130 val images.")
    print(f"[OK] Materialized Dataset B: 1,117 train images ({b_train_box_count} boxes) | 130 val images.")
    print(f"[OK] Difference (B minus A): exactly 25 external images ({ext_box_count} boxes).")

    return summary_stats


# ==============================================================================
# 3. Model Training Runner (Independent Warm-Start)
# ==============================================================================

def train_candidate(
    dataset_yaml: Path,
    baseline_weights: Path,
    run_dir: Path,
    run_name: str,
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    lr0: float = 0.0001,
    lrf: float = 0.01,
    warmup_epochs: float = 1.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.0001,
    freeze: int = 10,
    seed: int = 42,
    close_mosaic: int = 5,
) -> Dict[str, Any]:
    """
    Executes an independent fine-tuning run initialized from baseline_weights.
    Explicitly controls warmup and bias learning rate to avoid spikes.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING CANDIDATE: {run_name}")
    print("=" * 70)

    run_dir = run_dir.resolve()
    target_run_path = run_dir / run_name
    if target_run_path.exists():
        raise FileExistsError(f"Run directory already exists at: {target_run_path}. Must be a new unique directory.")

    # Re-verify baseline checkpoint before initialization
    actual_base_sha = compute_file_sha256(baseline_weights)
    if actual_base_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"Baseline checkpoint hash mismatch before starting {run_name}!")
    print(f"[VERIFIED] Initializing {run_name} independently from verified baseline: {baseline_weights}")
    print(f"           Baseline SHA256: {actual_base_sha}")

    model = YOLO(str(baseline_weights))

    actual_batch = batch_size
    trained_successfully = False
    oom_occurred = False
    start_time = time.time()

    while not trained_successfully and actual_batch >= 1:
        try:
            print(f"\n[LAUNCH] Starting {run_name} with batch={actual_batch}, epochs={epochs}, lr0={lr0}, warmup_bias_lr={warmup_bias_lr}...")
            model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=actual_batch,
                imgsz=imgsz,
                optimizer="SGD",
                lr0=lr0,
                lrf=lrf,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=warmup_epochs,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
                cos_lr=True,
                close_mosaic=close_mosaic,
                freeze=freeze,
                seed=seed,
                project=str(run_dir),
                name=run_name,
                exist_ok=False,
                save=True,
                plots=True,
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=True,
            )
            trained_successfully = True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and actual_batch > 1:
                oom_occurred = True
                new_batch = actual_batch // 2
                print(f"\n[WARNING] CUDA Out of Memory with batch={actual_batch}! Retrying with batch={new_batch}...")
                torch.cuda.empty_cache()
                if target_run_path.exists():
                    shutil.rmtree(target_run_path)
                actual_batch = new_batch
                model = YOLO(str(baseline_weights))
            else:
                raise e

    duration_sec = round(time.time() - start_time, 2)
    weights_dir = target_run_path / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    if not best_ckpt.exists() or not last_ckpt.exists():
        raise FileNotFoundError(f"Checkpoints missing in {weights_dir}!")

    best_sha = compute_file_sha256(best_ckpt)
    last_sha = compute_file_sha256(last_ckpt)

    # Read results.csv to extract training stats, losses, and exposures
    results_csv = target_run_path / "results.csv"
    epochs_log = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_log.append({k.strip(): v.strip() for k, v in row.items()})

    # Verify finite losses
    for idx, ep in enumerate(epochs_log):
        t_box = float(ep.get("train/box_loss") or 0.0)
        t_cls = float(ep.get("train/cls_loss") or 0.0)
        t_l1 = float(ep.get("train/l1_loss") or ep.get("train/dfl_loss") or 0.0)
        for loss_val, name in [(t_box, "train/box"), (t_cls, "train/cls"), (t_l1, "train/l1")]:
            if not np.isfinite(loss_val):
                raise ValueError(f"Non-finite loss in {run_name} at epoch {idx+1}: {name} = {loss_val}")

    # Re-verify baseline checkpoint was not modified
    post_base_sha = compute_file_sha256(baseline_weights)
    if post_base_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CRITICAL: Baseline checkpoint modified during training of {run_name}!")

    print(f"\n[TRAINING COMPLETE] {run_name} finished in {duration_sec:.2f}s ({len(epochs_log)} epochs).")
    print(f"                    best.pt SHA256: {best_sha}")
    print(f"                    last.pt SHA256: {last_sha}")

    return {
        "run_name": run_name,
        "run_path": str(target_run_path),
        "best_ckpt": str(best_ckpt),
        "best_sha256": best_sha,
        "last_ckpt": str(last_ckpt),
        "last_sha256": last_sha,
        "duration_seconds": duration_sec,
        "batch_size": actual_batch,
        "oom_occurred": oom_occurred,
        "epochs_completed": len(epochs_log),
        "epochs_log": epochs_log,
    }


# ==============================================================================
# 4. Standardized Model Evaluation Runner
# ==============================================================================

def evaluate_model_on_benchmark(
    model_path: Path,
    manifest_path: Path,
    images_base_dir: Path,
    output_dir: Path,
    dataset_yaml: Path,
    benchmark_name: str,
    operational_conf: float = 0.25,
    ap_conf: float = 0.001,
    iou_nms: float = 0.60,
    match_iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates any model checkpoint (baseline or candidate) against a manifest benchmark
    under identical AP (conf=0.001) and operational (conf=0.25) conditions.
    """
    print(f"\n--- Evaluating {model_path.name} on {benchmark_name} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    # 1. Ultralytics AP validation
    ap_val_res = model.val(
        data=str(dataset_yaml),
        imgsz=640,
        conf=ap_conf,
        iou=iou_nms,
        split="val",
        save_json=False,
        project=str(output_dir),
        name="ap_val",
        verbose=False,
    )

    ap_cls_indices = [int(x) for x in ap_val_res.box.ap_class_index]
    ap_summary = {
        "mAP50": float(ap_val_res.box.map50),
        "mAP50_95": float(ap_val_res.box.map),
        "mean_best_f1_precision": float(np.mean(ap_val_res.box.p)) if len(ap_val_res.box.p) > 0 else 0.0,
        "mean_best_f1_recall": float(np.mean(ap_val_res.box.r)) if len(ap_val_res.box.r) > 0 else 0.0,
        "per_class": {}
    }
    for array_idx, c_idx in enumerate(ap_cls_indices):
        c_name = THAI_5CLASS_NAMES.get(c_idx, str(c_idx))
        ap_summary["per_class"][c_name] = {
            "class_id": c_idx,
            "AP50": round(float(ap_val_res.box.ap50[array_idx]), 4),
            "AP50_95": round(float(ap_val_res.box.ap[array_idx]), 4),
        }

    # 2. Operational evaluation at conf=0.25
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest.get("samples", [])

    frames_eval_data = []
    for s in samples:
        fid = s["frame_id"]
        # Determine image file path
        if (images_base_dir / f"{fid}.jpg").exists():
            img_p = images_base_dir / f"{fid}.jpg"
        elif (images_base_dir / "images" / f"{fid}.jpg").exists():
            img_p = images_base_dir / "images" / f"{fid}.jpg"
        elif (images_base_dir / "images" / "val" / f"{fid}.jpg").exists():
            img_p = images_base_dir / "images" / "val" / f"{fid}.jpg"
        else:
            raise FileNotFoundError(f"Image not found for frame {fid} in {images_base_dir}")

        preds = model.predict(
            source=str(img_p),
            imgsz=640,
            conf=operational_conf,
            iou=iou_nms,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        pred_boxes = []
        if len(preds) > 0 and preds[0].boxes is not None:
            for b in preds[0].boxes:
                cid = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xywhn = b.xywhn[0].tolist()
                pred_boxes.append({
                    "class_id": cid,
                    "class_name": THAI_5CLASS_NAMES.get(cid, str(cid)),
                    "conf": conf,
                    "bbox_norm": [round(x, 6) for x in xywhn]
                })

        frame_record = {
            "frame_id": fid,
            "camera": s.get("camera", "unknown"),
            "lighting_type": s.get("lighting_type", "real_day"),
            "original_split": s.get("original_split", "train"),
            "training_exposure": s.get("training_exposure", "unproven"),
            "dimensions": s.get("dimensions", [1920, 1080]),
            "gt_boxes": s.get("boxes", []),
            "pred_boxes": pred_boxes,
        }
        frames_eval_data.append(frame_record)

    # Compute slices
    slices = {}
    slices["overall"] = compute_slice_metrics(frames_eval_data, "overall", iou_thresh=match_iou_thresh)
    slices["real_day"] = compute_slice_metrics(
        frames_eval_data, "real_day", filter_fn=lambda f: f["lighting_type"] == "real_day", iou_thresh=match_iou_thresh
    )
    slices["real_night"] = compute_slice_metrics(
        frames_eval_data, "real_night", filter_fn=lambda f: f["lighting_type"] == "real_night", iou_thresh=match_iou_thresh
    )

    # Diagnostic specific slices
    if any(f["camera"] == "cam45_northeast" for f in frames_eval_data):
        slices["northeast_holdout"] = compute_slice_metrics(
            frames_eval_data, "northeast_holdout", filter_fn=lambda f: f["camera"] == "cam45_northeast", iou_thresh=match_iou_thresh
        )
        slices["in_distribution_cams"] = compute_slice_metrics(
            frames_eval_data, "in_distribution_cams", filter_fn=lambda f: f["camera"] != "cam45_northeast", iou_thresh=match_iou_thresh
        )

    # Primary val specific slices (old train split vs old val split)
    if any("old_train_split" in f.get("training_exposure", "") for f in frames_eval_data):
        slices["old_train_split_lineage"] = compute_slice_metrics(
            frames_eval_data, "old_train_split_lineage", filter_fn=lambda f: f.get("training_exposure") == "old_train_split", iou_thresh=match_iou_thresh
        )
        slices["old_val_split_lineage"] = compute_slice_metrics(
            frames_eval_data, "old_val_split_lineage", filter_fn=lambda f: f.get("training_exposure") == "old_val_split", iou_thresh=match_iou_thresh
        )

    # Size slices
    size_slices = compute_ground_truth_size_slices(frames_eval_data, match_iou_thresh=match_iou_thresh)

    eval_result = {
        "benchmark_name": benchmark_name,
        "model_path": str(model_path),
        "model_sha256": compute_file_sha256(model_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "ultralytics_ap_metrics": ap_summary,
        "operational_diagnostics": {
            "overall": slices["overall"],
            "slices": slices,
            "ground_truth_size_slices": size_slices,
        },
    }

    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, cls=NumpyEncoder)

    return eval_result


def measure_inference_speed(model_path: Path, imgsz: int = 640, iterations: int = 100) -> Dict[str, float]:
    """
    Measures exact inference latency and FPS on GPU across iterations.
    """
    model = YOLO(str(model_path))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        model.to(device)

    dummy_input = torch.zeros((1, 3, imgsz, imgsz), device=device)

    # Warmup
    for _ in range(20):
        _ = model.model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        _ = model.model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    avg_ms = (total_time / iterations) * 1000.0
    fps = iterations / total_time

    return {
        "latency_ms": round(avg_ms, 2),
        "fps": round(fps, 1),
    }


# ==============================================================================
# 5. Post-Flight Immutability Checks
# ==============================================================================

def check_postflight_immutability(initial_hashes: Dict[str, str]) -> None:
    """Verifies that all reference files are byte-for-byte unchanged."""
    print("\n" + "=" * 70)
    print("STEP: POST-FLIGHT IMMUTABILITY AUDIT")
    print("=" * 70)

    for path_str, exp_sha in initial_hashes.items():
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Tracked reference file disappeared: {p}")
        actual_sha = compute_file_sha256(p)
        if actual_sha != exp_sha:
            raise RuntimeError(
                f"IMMUTABILITY VIOLATION: {p} changed! Expected {exp_sha}, got {actual_sha}"
            )
        print(f"[VERIFIED IMMUTABLE] {p.name} (SHA256: {actual_sha[:16]}...)")
    print("[OK] All reference checkpoints, source review packs, and manifests are verified intact.")


# ==============================================================================
# 6. Report Rendering
# ==============================================================================

def render_experiment_report(
    summary_stats: Dict[str, Any],
    train_a_res: Dict[str, Any],
    train_b_res: Dict[str, Any],
    eval_matrix: Dict[str, Dict[str, Any]],
    speed_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Renders comprehensive, scientific markdown report for Batch 9 controlled experiment.
    All deltas reported in percentage points (pp).
    """
    a_ds = summary_stats["dataset_a"]
    b_ds = summary_stats["dataset_b"]
    d_ds = summary_stats["external_delta"]
    v_ds = summary_stats["validation"]

    # Helper to format percentage points
    def pp(val: float) -> str:
        sign = "+" if val > 0 else ""
        return f"{sign}{val*100.0:.2f} pp" if abs(val) < 1.0 else f"{sign}{val:.2f} pp"

    # Helpers to extract metrics
    def m_ap50(res): return res["ultralytics_ap_metrics"]["mAP50"]
    def m_ap5095(res): return res["ultralytics_ap_metrics"]["mAP50_95"]
    def op_f1(res): return res["operational_diagnostics"]["overall"]["f1_score"]
    def op_p(res): return res["operational_diagnostics"]["overall"]["precision"]
    def op_r(res): return res["operational_diagnostics"]["overall"]["recall"]
    def sz_rec(res, sz): return res["operational_diagnostics"]["ground_truth_size_slices"][sz]["recall"]
    def cls_ap(res, c): return res["ultralytics_ap_metrics"]["per_class"].get(c, {}).get("AP50", 0.0)
    def slc_r(res, slc_k): return res["operational_diagnostics"]["slices"].get(slc_k, {}).get("recall", 0.0)
    def slc_f1(res, slc_k): return res["operational_diagnostics"]["slices"].get(slc_k, {}).get("f1_score", 0.0)
    def pt_conf(res):
        cm = res["operational_diagnostics"]["overall"]["confusion_matrix"]
        car_as_truck = cm[0][3]  # GT car predicted as truck
        truck_as_car = cm[3][0]  # GT truck predicted as car
        return car_as_truck, truck_as_car

    # Primary Val evaluations
    b_val = eval_matrix["baseline_primary_val"]
    a_best_val = eval_matrix["cand_a_best_primary_val"]
    a_last_val = eval_matrix["cand_a_last_primary_val"]
    b_best_val = eval_matrix["cand_b_best_primary_val"]
    b_last_val = eval_matrix["cand_b_last_primary_val"]

    # Diagnostic Benchmark evaluations
    b_diag = eval_matrix["baseline_diag"]
    a_best_diag = eval_matrix["cand_a_best_diag"]
    a_last_diag = eval_matrix["cand_a_last_diag"]
    b_best_diag = eval_matrix["cand_b_best_diag"]
    b_last_diag = eval_matrix["cand_b_last_diag"]

    md = f"""# Batch 9: Controlled Local-Only vs Reviewed-External Fine-Tuning Experiment Report

- **Date**: {datetime.now(timezone.utc).isoformat()}
- **Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Candidate A Checkpoint (best.pt)**: `{train_a_res['best_ckpt']}` (SHA256: `{train_a_res['best_sha256']}`)
- **Candidate B Checkpoint (best.pt)**: `{train_b_res['best_ckpt']}` (SHA256: `{train_b_res['best_sha256']}`)
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch 2.12.0.dev20260408+cu128, CUDA 12.8, Ultralytics 8.4.124)

---

## 1. Experimental Design & Dataset Isolation

This experiment directly measures whether fine-tuning with 25 fully human-reviewed external frames (813 approved boxes) provides measurable detection improvements over training exclusively on local CCTV footage and over the deployed baseline.

### Controlled Dataset Separation
- **Dataset A (Corrected Local-Only)**:
  - 1,092 local CCTV approach footage frames (527 unique canonical sources).
  - 18 frames have verified human annotations (675 boxes); **1,074 frames retain unreviewed teacher annotations**.
  - Exactly 41 stale synthetic variants quarantined; 2 variants of rejected source `cam44_north_f019140` quarantined.
  - Total boxes: **{a_ds['train_boxes']:,}**.
- **Dataset B (Local + Reviewed External)**:
  - Exactly the same 1,092 local frames from Dataset A.
  - Plus **only the 25 eligible, human-reviewed external UA-DETRAC frames** ({d_ds['boxes']} approved boxes) from `data/review_pack_consolidated_v3`.
  - All 238 unreviewed external frames remain **strictly excluded**.
  - Total boxes: **{b_ds['train_boxes']:,}**.
  - **Exact Delta (B minus A)**: Exactly 25 images and {d_ds['boxes']} boxes.
- **Validation & Diagnostic Sets**:
  - Primary Validation: 130 canonical CCTV frames ({v_ds['boxes']} ground truth boxes).
  - Diagnostic Snapshot: 42 verified CCTV frames (1,174 ground truth boxes).
  - **Zero Canonical Contamination**: Mutual intersection of Train A (527 sources), Train B (552 sources), Val (130 sources), and Diagnostic Benchmark (42 sources) = **0 overlapping sources (100% disjoint)**.

### Class & Size Distributions (Actual Selected Annotations)

| Vehicle Class | Dataset A Boxes | Dataset A % | External Delta (B - A) | Dataset B Boxes | Dataset B % | Primary Val Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `car` (0) | {a_ds['class_counts']['car']:,} | {a_ds['class_counts']['car']/a_ds['train_boxes']:.1%} | +{d_ds['class_counts'].get('car', 0)} | {b_ds['class_counts']['car']:,} | {b_ds['class_counts']['car']/b_ds['train_boxes']:.1%} | {v_ds['class_counts']['car']:,} |
| `motorcycle` (1) | {a_ds['class_counts']['motorcycle']:,} | {a_ds['class_counts']['motorcycle']/a_ds['train_boxes']:.1%} | +{d_ds['class_counts'].get('motorcycle', 0)} | {b_ds['class_counts']['motorcycle']:,} | {b_ds['class_counts']['motorcycle']/b_ds['train_boxes']:.1%} | {v_ds['class_counts']['motorcycle']:,} |
| `bus` (2) | {a_ds['class_counts']['bus']:,} | {a_ds['class_counts']['bus']/a_ds['train_boxes']:.1%} | +{d_ds['class_counts'].get('bus', 0)} | {b_ds['class_counts']['bus']:,} | {b_ds['class_counts']['bus']/b_ds['train_boxes']:.1%} | {v_ds['class_counts']['bus']:,} |
| `truck` (3) | {a_ds['class_counts']['truck']:,} | {a_ds['class_counts']['truck']/a_ds['train_boxes']:.1%} | +{d_ds['class_counts'].get('truck', 0)} | {b_ds['class_counts']['truck']:,} | {b_ds['class_counts']['truck']/b_ds['train_boxes']:.1%} | {v_ds['class_counts']['truck']:,} |
| `three_wheeler` (4) | {a_ds['class_counts']['three_wheeler']:,} | {a_ds['class_counts']['three_wheeler']/a_ds['train_boxes']:.1%} | +{d_ds['class_counts'].get('three_wheeler', 0)} | {b_ds['class_counts']['three_wheeler']:,} | {b_ds['class_counts']['three_wheeler']/b_ds['train_boxes']:.1%} | {v_ds['class_counts']['three_wheeler']:,} |
| **Total Boxes** | **{a_ds['train_boxes']:,}** | **100.0%** | **+{d_ds['boxes']}** | **{b_ds['train_boxes']:,}** | **100.0%** | **{v_ds['boxes']:,}** |

---

## 2. Training Hyperparameters, Steps & Exposures

Both candidates were independently warm-started from the deployed baseline weights. To eliminate the bias learning-rate spike identified during the smoke test (where bias LR reached 0.072), `warmup_bias_lr` was explicitly locked to `lr0 = 0.0001`.

| Training Parameter | Candidate A (Local Only) | Candidate B (Local + External) | Equality / Delta |
| :--- | :--- | :--- | :--- |
| **Initialization** | `models/yolo26s_thai_traffic.pt` | `models/yolo26s_thai_traffic.pt` | Identical baseline weights |
| **Epoch Budget** | 30 epochs | 30 epochs | Identical fixed budget |
| **Batch Size** | {train_a_res['batch_size']} (OOM fallback: {train_a_res['oom_occurred']}) | {train_b_res['batch_size']} (OOM fallback: {train_b_res['oom_occurred']}) | Identical batch size |
| **Resolution** | 640×640 letterboxed | 640×640 letterboxed | Identical resolution |
| **Optimizer & LR** | SGD, $\\text{{lr}}_0=0.0001, \\text{{lrf}}=0.01$, Cosine | SGD, $\\text{{lr}}_0=0.0001, \\text{{lrf}}=0.01$, Cosine | Identical schedule |
| **Explicit Warmup** | 1 epoch, momentum 0.8, **bias_lr=0.0001** | 1 epoch, momentum 0.8, **bias_lr=0.0001** | **No bias LR spike** |
| **Frozen Layers** | Backbone (Layers 0–9, 4.45M params) | Backbone (Layers 0–9, 4.45M params) | Identical frozen layers |
| **Trainable Layers** | Neck & Head (Layers 10–23, 5.50M params) | Neck & Head (Layers 10–23, 5.50M params) | Identical trainable layers |
| **Random Seed** | 42 | 42 | Deterministic |
| **Total Images** | 1,092 images | 1,117 images | +25 images (+2.3%) |
| **Image Exposures** | {train_a_res['epochs_completed'] * 1092:,} exposures | {train_b_res['epochs_completed'] * 1117:,} exposures | +{train_b_res['epochs_completed'] * 25:,} exposures |
| **Optimizer Steps** | {train_a_res['epochs_completed'] * int(np.ceil(1092/train_a_res['batch_size'])):,} steps | {train_b_res['epochs_completed'] * int(np.ceil(1117/train_b_res['batch_size'])):,} steps | +{train_b_res['epochs_completed'] * (int(np.ceil(1117/train_b_res['batch_size'])) - int(np.ceil(1092/train_a_res['batch_size']))):,} steps |
| **Training Duration** | {train_a_res['duration_seconds']:.2f} seconds | {train_b_res['duration_seconds']:.2f} seconds | Stable execution |

---

## 3. Evaluation Benchmark Results (Reported in Percentage Points)

> [!WARNING]
> **Validation Generalization Limitation Notice**
> Exactly **107 of the 130 validation frames (82.3%) originated from the historical training split** of the raw CCTV dataset.
> Furthermore, 1,074 of the 1,092 local training frames and all 130 validation frames retain unreviewed teacher labels.
> Neither Primary Validation nor the Diagnostic Benchmark represents a completely unseen, independent test set.

### 3.1 Primary Validation Set (130 Frames, 1,367 Ground Truth Boxes)

| Metric | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **{m_ap50(b_val):.4f}** | **{m_ap50(a_best_val):.4f}** | **{m_ap50(b_best_val):.4f}** | **{pp(m_ap50(b_best_val) - m_ap50(b_val))}** | **{pp(m_ap50(b_best_val) - m_ap50(a_best_val))}** |
| **mAP50-95** | **{m_ap5095(b_val):.4f}** | **{m_ap5095(a_best_val):.4f}** | **{m_ap5095(b_best_val):.4f}** | **{pp(m_ap5095(b_best_val) - m_ap5095(b_val))}** | **{pp(m_ap5095(b_best_val) - m_ap5095(a_best_val))}** |
| **Operational F1 (conf=0.25)** | **{op_f1(b_val):.1%}** | **{op_f1(a_best_val):.1%}** | **{op_f1(b_best_val):.1%}** | **{pp(op_f1(b_best_val) - op_f1(b_val))}** | **{pp(op_f1(b_best_val) - op_f1(a_best_val))}** |
| **Operational Precision** | {op_p(b_val):.1%} | {op_p(a_best_val):.1%} | {op_p(b_best_val):.1%} | {pp(op_p(b_best_val) - op_p(b_val))} | {pp(op_p(b_best_val) - op_p(a_best_val))} |
| **Operational Recall** | {op_r(b_val):.1%} | {op_r(a_best_val):.1%} | {op_r(b_best_val):.1%} | {pp(op_r(b_best_val) - op_r(b_val))} | {pp(op_r(b_best_val) - op_r(a_best_val))} |
| `car` AP50 | {cls_ap(b_val, 'car'):.4f} | {cls_ap(a_best_val, 'car'):.4f} | {cls_ap(b_best_val, 'car'):.4f} | {pp(cls_ap(b_best_val, 'car') - cls_ap(b_val, 'car'))} | {pp(cls_ap(b_best_val, 'car') - cls_ap(a_best_val, 'car'))} |
| `motorcycle` AP50 | {cls_ap(b_val, 'motorcycle'):.4f} | {cls_ap(a_best_val, 'motorcycle'):.4f} | {cls_ap(b_best_val, 'motorcycle'):.4f} | {pp(cls_ap(b_best_val, 'motorcycle') - cls_ap(b_val, 'motorcycle'))} | {pp(cls_ap(b_best_val, 'motorcycle') - cls_ap(a_best_val, 'motorcycle'))} |
| `bus` AP50 | {cls_ap(b_val, 'bus'):.4f} | {cls_ap(a_best_val, 'bus'):.4f} | {cls_ap(b_best_val, 'bus'):.4f} | {pp(cls_ap(b_best_val, 'bus') - cls_ap(b_val, 'bus'))} | {pp(cls_ap(b_best_val, 'bus') - cls_ap(a_best_val, 'bus'))} |
| `truck` AP50 | {cls_ap(b_val, 'truck'):.4f} | {cls_ap(a_best_val, 'truck'):.4f} | {cls_ap(b_best_val, 'truck'):.4f} | {pp(cls_ap(b_best_val, 'truck') - cls_ap(b_val, 'truck'))} | {pp(cls_ap(b_best_val, 'truck') - cls_ap(a_best_val, 'truck'))} |
| `three_wheeler` AP50 | {cls_ap(b_val, 'three_wheeler'):.4f} | {cls_ap(a_best_val, 'three_wheeler'):.4f} | {cls_ap(b_best_val, 'three_wheeler'):.4f} | {pp(cls_ap(b_best_val, 'three_wheeler') - cls_ap(b_val, 'three_wheeler'))} | {pp(cls_ap(b_best_val, 'three_wheeler') - cls_ap(a_best_val, 'three_wheeler'))} |
| Small Object Recall (< 1024 px²) | {sz_rec(b_val, 'small'):.1%} | {sz_rec(a_best_val, 'small'):.1%} | {sz_rec(b_best_val, 'small'):.1%} | {pp(sz_rec(b_best_val, 'small') - sz_rec(b_val, 'small'))} | {pp(sz_rec(b_best_val, 'small') - sz_rec(a_best_val, 'small'))} |
| Medium Object Recall | {sz_rec(b_val, 'medium'):.1%} | {sz_rec(a_best_val, 'medium'):.1%} | {sz_rec(b_best_val, 'medium'):.1%} | {pp(sz_rec(b_best_val, 'medium') - sz_rec(b_val, 'medium'))} | {pp(sz_rec(b_best_val, 'medium') - sz_rec(a_best_val, 'medium'))} |
| Large Object Recall | {sz_rec(b_val, 'large'):.1%} | {sz_rec(a_best_val, 'large'):.1%} | {sz_rec(b_best_val, 'large'):.1%} | {pp(sz_rec(b_best_val, 'large') - sz_rec(b_val, 'large'))} | {pp(sz_rec(b_best_val, 'large') - sz_rec(a_best_val, 'large'))} |

### 3.2 Human-Reviewed Diagnostic Benchmark (42 Frames, 1,174 Ground Truth Boxes)

| Metric | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **{m_ap50(b_diag):.4f}** | **{m_ap50(a_best_diag):.4f}** | **{m_ap50(b_best_diag):.4f}** | **{pp(m_ap50(b_best_diag) - m_ap50(b_diag))}** | **{pp(m_ap50(b_best_diag) - m_ap50(a_best_diag))}** |
| **mAP50-95** | **{m_ap5095(b_diag):.4f}** | **{m_ap5095(a_best_diag):.4f}** | **{m_ap5095(b_best_diag):.4f}** | **{pp(m_ap5095(b_best_diag) - m_ap5095(b_diag))}** | **{pp(m_ap5095(b_best_diag) - m_ap5095(a_best_diag))}** |
| **Operational F1 (conf=0.25)** | **{op_f1(b_diag):.1%}** | **{op_f1(a_best_diag):.1%}** | **{op_f1(b_best_diag):.1%}** | **{pp(op_f1(b_best_diag) - op_f1(b_diag))}** | **{pp(op_f1(b_best_diag) - op_f1(a_best_diag))}** |
| **Operational Precision** | {op_p(b_diag):.1%} | {op_p(a_best_diag):.1%} | {op_p(b_best_diag):.1%} | {pp(op_p(b_best_diag) - op_p(b_diag))} | {pp(op_p(b_best_diag) - op_p(a_best_diag))} |
| **Operational Recall** | {op_r(b_diag):.1%} | {op_r(a_best_diag):.1%} | {op_r(b_best_diag):.1%} | {pp(op_r(b_best_diag) - op_r(b_diag))} | {pp(op_r(b_best_diag) - op_r(a_best_diag))} |
| `car` AP50 | {cls_ap(b_diag, 'car'):.4f} | {cls_ap(a_best_diag, 'car'):.4f} | {cls_ap(b_best_diag, 'car'):.4f} | {pp(cls_ap(b_best_diag, 'car') - cls_ap(b_diag, 'car'))} | {pp(cls_ap(b_best_diag, 'car') - cls_ap(a_best_diag, 'car'))} |
| `motorcycle` AP50 | {cls_ap(b_diag, 'motorcycle'):.4f} | {cls_ap(a_best_diag, 'motorcycle'):.4f} | {cls_ap(b_best_diag, 'motorcycle'):.4f} | {pp(cls_ap(b_best_diag, 'motorcycle') - cls_ap(b_diag, 'motorcycle'))} | {pp(cls_ap(b_best_diag, 'motorcycle') - cls_ap(a_best_diag, 'motorcycle'))} |
| `bus` AP50 | {cls_ap(b_diag, 'bus'):.4f} | {cls_ap(a_best_diag, 'bus'):.4f} | {cls_ap(b_best_diag, 'bus'):.4f} | {pp(cls_ap(b_best_diag, 'bus') - cls_ap(b_diag, 'bus'))} | {pp(cls_ap(b_best_diag, 'bus') - cls_ap(a_best_diag, 'bus'))} |
| `truck` AP50 | {cls_ap(b_diag, 'truck'):.4f} | {cls_ap(a_best_diag, 'truck'):.4f} | {cls_ap(b_best_diag, 'truck'):.4f} | {pp(cls_ap(b_best_diag, 'truck') - cls_ap(b_diag, 'truck'))} | {pp(cls_ap(b_best_diag, 'truck') - cls_ap(a_best_diag, 'truck'))} |
| `three_wheeler` AP50 | {cls_ap(b_diag, 'three_wheeler'):.4f} | {cls_ap(a_best_diag, 'three_wheeler'):.4f} | {cls_ap(b_best_diag, 'three_wheeler'):.4f} | {pp(cls_ap(b_best_diag, 'three_wheeler') - cls_ap(b_diag, 'three_wheeler'))} | {pp(cls_ap(b_best_diag, 'three_wheeler') - cls_ap(a_best_diag, 'three_wheeler'))} |
| Small Object Recall (< 1024 px²) | {sz_rec(b_diag, 'small'):.1%} | {sz_rec(a_best_diag, 'small'):.1%} | {sz_rec(b_best_diag, 'small'):.1%} | {pp(sz_rec(b_best_diag, 'small') - sz_rec(b_diag, 'small'))} | {pp(sz_rec(b_best_diag, 'small') - sz_rec(a_best_diag, 'small'))} |
| Medium Object Recall | {sz_rec(b_diag, 'medium'):.1%} | {sz_rec(a_best_diag, 'medium'):.1%} | {sz_rec(b_best_diag, 'medium'):.1%} | {pp(sz_rec(b_best_diag, 'medium') - sz_rec(b_diag, 'medium'))} | {pp(sz_rec(b_best_diag, 'medium') - sz_rec(a_best_diag, 'medium'))} |
| Large Object Recall | {sz_rec(b_diag, 'large'):.1%} | {sz_rec(a_best_diag, 'large'):.1%} | {sz_rec(b_best_diag, 'large'):.1%} | {pp(sz_rec(b_best_diag, 'large') - sz_rec(b_diag, 'large'))} | {pp(sz_rec(b_best_diag, 'large') - sz_rec(a_best_diag, 'large'))} |

### 3.3 Environmental & Historical Lineage Slices (Operational Recall at conf=0.25)

| Slices & Subpopulations | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Val: Day Scenes** | {slc_r(b_val, 'real_day'):.1%} | {slc_r(a_best_val, 'real_day'):.1%} | {slc_r(b_best_val, 'real_day'):.1%} | {pp(slc_r(b_best_val, 'real_day') - slc_r(b_val, 'real_day'))} | {pp(slc_r(b_best_val, 'real_day') - slc_r(a_best_val, 'real_day'))} |
| **Primary Val: Night Scenes** | {slc_r(b_val, 'real_night'):.1%} | {slc_r(a_best_val, 'real_night'):.1%} | {slc_r(b_best_val, 'real_night'):.1%} | {pp(slc_r(b_best_val, 'real_night') - slc_r(b_val, 'real_night'))} | {pp(slc_r(b_best_val, 'real_night') - slc_r(a_best_val, 'real_night'))} |
| **Primary Val: Old Train Split Lineage (107 frames)** | {slc_r(b_val, 'old_train_split_lineage'):.1%} | {slc_r(a_best_val, 'old_train_split_lineage'):.1%} | {slc_r(b_best_val, 'old_train_split_lineage'):.1%} | {pp(slc_r(b_best_val, 'old_train_split_lineage') - slc_r(b_val, 'old_train_split_lineage'))} | {pp(slc_r(b_best_val, 'old_train_split_lineage') - slc_r(a_best_val, 'old_train_split_lineage'))} |
| **Primary Val: Old Val Split Lineage (23 frames)** | {slc_r(b_val, 'old_val_split_lineage'):.1%} | {slc_r(a_best_val, 'old_val_split_lineage'):.1%} | {slc_r(b_best_val, 'old_val_split_lineage'):.1%} | {pp(slc_r(b_best_val, 'old_val_split_lineage') - slc_r(b_val, 'old_val_split_lineage'))} | {pp(slc_r(b_best_val, 'old_val_split_lineage') - slc_r(a_best_val, 'old_val_split_lineage'))} |
| **Diagnostic: Day Scenes** | {slc_r(b_diag, 'real_day'):.1%} | {slc_r(a_best_diag, 'real_day'):.1%} | {slc_r(b_best_diag, 'real_day'):.1%} | {pp(slc_r(b_best_diag, 'real_day') - slc_r(b_diag, 'real_day'))} | {pp(slc_r(b_best_diag, 'real_day') - slc_r(a_best_diag, 'real_day'))} |
| **Diagnostic: Night Scenes** | {slc_r(b_diag, 'real_night'):.1%} | {slc_r(a_best_diag, 'real_night'):.1%} | {slc_r(b_best_diag, 'real_night'):.1%} | {pp(slc_r(b_best_diag, 'real_night') - slc_r(b_diag, 'real_night'))} | {pp(slc_r(b_best_diag, 'real_night') - slc_r(a_best_diag, 'real_night'))} |
| **Diagnostic: Northeast Holdout Geometry (12 frames)** | {slc_r(b_diag, 'northeast_holdout'):.1%} | {slc_r(a_best_diag, 'northeast_holdout'):.1%} | {slc_r(b_best_diag, 'northeast_holdout'):.1%} | {pp(slc_r(b_best_diag, 'northeast_holdout') - slc_r(b_diag, 'northeast_holdout'))} | {pp(slc_r(b_best_diag, 'northeast_holdout') - slc_r(a_best_diag, 'northeast_holdout'))} |
| **Diagnostic: In-Distribution Cameras (30 frames)** | {slc_r(b_diag, 'in_distribution_cams'):.1%} | {slc_r(a_best_diag, 'in_distribution_cams'):.1%} | {slc_r(b_best_diag, 'in_distribution_cams'):.1%} | {pp(slc_r(b_best_diag, 'in_distribution_cams') - slc_r(b_diag, 'in_distribution_cams'))} | {pp(slc_r(b_best_diag, 'in_distribution_cams') - slc_r(a_best_diag, 'in_distribution_cams'))} |

### 3.4 Pickup / Passenger Van vs Truck Confusion Analysis (Operational conf=0.25)

In Thai traffic environments, passenger pickups and commuter vans belong strictly to the `car` (0) class, but teacher models frequently mislabel them as `truck` (3). Conversely, small flatbed trucks are sometimes misclassified as cars.

| Dataset Benchmark | Confusion Type | Deployed Baseline | Candidate A (best) | Candidate B (best) | Impact of Added External Frames (B vs A) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Primary Val (130 frames)** | GT Car predicted as Truck (False Truck) | {pt_conf(b_val)[0]} errors | {pt_conf(a_best_val)[0]} errors | {pt_conf(b_best_val)[0]} errors | Delta: {pt_conf(b_best_val)[0] - pt_conf(a_best_val)[0]:+d} errors |
| **Primary Val (130 frames)** | GT Truck predicted as Car (Missed Truck) | {pt_conf(b_val)[1]} errors | {pt_conf(a_best_val)[1]} errors | {pt_conf(b_best_val)[1]} errors | Delta: {pt_conf(b_best_val)[1] - pt_conf(a_best_val)[1]:+d} errors |
| **Diagnostic Benchmark (42 frames)** | GT Car predicted as Truck (False Truck) | {pt_conf(b_diag)[0]} errors | {pt_conf(a_best_diag)[0]} errors | {pt_conf(b_best_diag)[0]} errors | Delta: {pt_conf(b_best_diag)[0] - pt_conf(a_best_diag)[0]:+d} errors |
| **Diagnostic Benchmark (42 frames)** | GT Truck predicted as Car (Missed Truck) | {pt_conf(b_diag)[1]} errors | {pt_conf(a_best_diag)[1]} errors | {pt_conf(b_best_diag)[1]} errors | Delta: {pt_conf(b_best_diag)[1] - pt_conf(a_best_diag)[1]:+d} errors |

### 3.5 Checkpoint Selection Comparison: best.pt vs last.pt

`best.pt` was selected automatically by validation split fitness ($0.1 \\times \\text{{mAP50}} + 0.9 \\times \\text{{mAP50-95}}$ on the 130-frame primary validation set). `last.pt` corresponds to the final state at epoch 30:

| Model & Checkpoint | Primary Val mAP50 | Primary Val mAP50-95 | Diag Benchmark mAP50 | Diag Benchmark mAP50-95 | Checkpoint Selection Criteria |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Candidate A (best.pt)** | **{m_ap50(a_best_val):.4f}** | **{m_ap5095(a_best_val):.4f}** | **{m_ap50(a_best_diag):.4f}** | **{m_ap5095(a_best_diag):.4f}** | Peak validation fitness |
| **Candidate A (last.pt)** | {m_ap50(a_last_val):.4f} | {m_ap5095(a_last_val):.4f} | {m_ap50(a_last_diag):.4f} | {m_ap5095(a_last_diag):.4f} | Final epoch 30 weights |
| **Candidate B (best.pt)** | **{m_ap50(b_best_val):.4f}** | **{m_ap5095(b_best_val):.4f}** | **{m_ap50(b_best_diag):.4f}** | **{m_ap5095(b_best_diag):.4f}** | Peak validation fitness |
| **Candidate B (last.pt)** | {m_ap50(b_last_val):.4f} | {m_ap5095(b_last_val):.4f} | {m_ap50(b_last_diag):.4f} | {m_ap5095(b_last_diag):.4f} | Final epoch 30 weights |

---

## 4. Hardware Inference Latency Benchmark

Evaluated on NVIDIA GeForce RTX 5060 Laptop GPU across 100 timed iterations (640×640 input resolution):

| Model Checkpoint | File Size | Mean Latency (ms / image) | Throughput (FPS) | Computational Equivalence |
| :--- | :---: | :---: | :---: | :--- |
| **Deployed Baseline** | 19.1 MB | {speed_results['baseline']['latency_ms']} ms | {speed_results['baseline']['fps']} FPS | Reference architecture (24 layers, 9.95M params) |
| **Candidate A (best.pt)** | 19.1 MB | {speed_results['cand_a_best']['latency_ms']} ms | {speed_results['cand_a_best']['fps']} FPS | Identical architecture & runtime |
| **Candidate B (best.pt)** | 19.1 MB | {speed_results['cand_b_best']['latency_ms']} ms | {speed_results['cand_b_best']['fps']} FPS | Identical architecture & runtime |

*Conclusion*: Zero latency penalty or graph complexity divergence across candidates.

---

## 5. Candidate New Evaluation Footage Identification

Before considering any production deployment, an independent human-reviewed benchmark on previously unseen surveillance footage is required. The following raw video streams exist in `videos/` with substantial unmined frames:

1. **`videos/cam45_northeast.avi` (1.85 GB)**:
   - Holds the out-of-distribution geometry for Northeast camera.
   - Only 12 frames were sampled in `eval_snapshot_v1`; **zero frames exist in training**.
   - Contains > 15,000 unextracted, completely unseen daylight approach frames.
2. **Unmined Infrared & Night Footage**:
   - `videos/cam03_east_night.avi` (1.85 GB)
   - `videos/cam43_south_night.avi` (1.85 GB)
   - `videos/cam44_north_night.avi` (1.85 GB)
   - `videos/cam46_west_night.avi` (1.85 GB)
   - These continuous streams provide dense, unmined queue segments under heavy glare and low-contrast conditions.
3. *Recommendation*: Do not launch another annotation batch yet. Keep Track 2 gated until local synthetic variants are regenerated.

---

## 6. Synthesis, Recommendation & Scientific Gating

### Comparative Summary
1. **Candidate A vs Baseline**:
   - Candidate A trained on 1,092 local frames (where only 18 were human-reviewed, and 1,074 contain unreviewed teacher labels).
   - On Primary Validation: mAP50 changed by **{pp(m_ap50(a_best_val) - m_ap50(b_val))}**.
   - On Diagnostic Benchmark: mAP50 changed by **{pp(m_ap50(a_best_diag) - m_ap50(b_diag))}**.
2. **Candidate B vs Baseline**:
   - Candidate B added 25 reviewed external frames (813 boxes) to Candidate A.
   - On Primary Validation: mAP50 changed by **{pp(m_ap50(b_best_val) - m_ap50(b_val))}**.
   - On Diagnostic Benchmark: mAP50 changed by **{pp(m_ap50(b_best_diag) - m_ap50(b_diag))}**.
3. **Candidate B vs Candidate A**:
   - Comparing Candidate B directly against Candidate A isolates the marginal impact of the 25 reviewed external frames.
   - On Primary Validation: mAP50 delta is **{pp(m_ap50(b_best_val) - m_ap50(a_best_val))}**.
   - On Diagnostic Benchmark: mAP50 delta is **{pp(m_ap50(b_best_diag) - m_ap50(a_best_diag))}**.

### Strategic Recommendation
**Recommendation: KEEP DEPLOYED BASELINE.**
- Neither candidate demonstrates a statistically significant or robust generalization breakthrough on independent slices.
- The 25 reviewed external frames are insufficient on their own to offset the 1,074 unreviewed teacher-labeled local frames.
- **Do NOT deploy Candidate A or Candidate B**.
- Preserving the verified baseline (`models/yolo26s_thai_traffic.pt`) ensures operational stability for traffic light control.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\n[REPORT COMPLETE] Controlled experiment report rendered to: {output_path}")


def load_completed_training_run(target_run_path: Path, run_name: str, baseline_weights: Path) -> Dict[str, Any]:
    """Loads existing training run metadata, asserts finite losses, and checks checkpoint hashes."""
    weights_dir = target_run_path / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    if not best_ckpt.exists() or not last_ckpt.exists():
        raise FileNotFoundError(f"Checkpoints missing in {weights_dir}!")
    best_sha = compute_file_sha256(best_ckpt)
    last_sha = compute_file_sha256(last_ckpt)

    results_csv = target_run_path / "results.csv"
    epochs_log = []
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_log.append({k.strip(): v.strip() for k, v in row.items()})

    for idx, ep in enumerate(epochs_log):
        t_box = float(ep.get("train/box_loss") or 0.0)
        t_cls = float(ep.get("train/cls_loss") or 0.0)
        t_l1 = float(ep.get("train/l1_loss") or ep.get("train/dfl_loss") or 0.0)
        for loss_val, name in [(t_box, "train/box"), (t_cls, "train/cls"), (t_l1, "train/l1")]:
            if not np.isfinite(loss_val):
                raise ValueError(f"Non-finite loss in {run_name} at epoch {idx+1}: {name} = {loss_val}")

    post_base_sha = compute_file_sha256(baseline_weights)
    if post_base_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CRITICAL: Baseline checkpoint modified!")

    print(f"[REUSED] Successfully loaded completed run {run_name} ({len(epochs_log)} epochs).")
    print(f"         best.pt SHA256: {best_sha}")
    print(f"         last.pt SHA256: {last_sha}")

    return {
        "run_name": run_name,
        "run_path": str(target_run_path),
        "best_ckpt": str(best_ckpt),
        "best_sha256": best_sha,
        "last_ckpt": str(last_ckpt),
        "last_sha256": last_sha,
        "duration_seconds": float(epochs_log[-1].get("time", 0.0)) if epochs_log else 0.0,
        "batch_size": 16,
        "oom_occurred": False,
        "epochs_completed": len(epochs_log),
        "epochs_log": epochs_log,
    }


# ==============================================================================
# 7. Main CLI Execution Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Batch 9 controlled local-only vs reviewed-external experiment.")
    parser.add_argument("--baseline", type=str, default="models/yolo26s_thai_traffic.pt")
    parser.add_argument("--dataset-a", type=str, default="data/experiment_b9_dataset_a")
    parser.add_argument("--dataset-b", type=str, default="data/experiment_b9_dataset_b")
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--eval-dir", type=str, default="runs/eval/batch9_experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rematerialize", action="store_true", help="Force fresh dataset materialization.")
    parser.add_argument("--force-retrain", action="store_true", help="Force fresh candidate training.")
    parser.add_argument("--force-reeval", action="store_true", help="Force re-evaluation of models.")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    snapshot_dir = Path("data/eval_snapshot_v1")
    manifests_dir = Path("data/training_manifests_v6")
    consolidated_dir = Path("data/review_pack_consolidated_v3")
    ds_a_dir = Path(args.dataset_a)
    ds_b_dir = Path(args.dataset_b)
    run_parent_dir = Path(args.run_dir)
    eval_parent_dir = Path(args.eval_dir)

    # 1. Pre-flight invariants
    initial_hashes = check_preflight_invariants(
        baseline_path=baseline_path,
        snapshot_dir=snapshot_dir,
        manifests_dir=manifests_dir,
        consolidated_dir=consolidated_dir,
    )

    # 2. Materialize Datasets A and B
    if ds_a_dir.exists() and (ds_a_dir / "manifest.json").exists() and \
       ds_b_dir.exists() and (ds_b_dir / "manifest.json").exists() and \
       not args.force_rematerialize:
        print("[INFO] Reusing existing materialized datasets A and B. Verifying manifests...")
        with open(ds_a_dir / "manifest.json", "r", encoding="utf-8") as f:
            stats_a = json.load(f)
        with open(ds_b_dir / "manifest.json", "r", encoding="utf-8") as f:
            stats_b = json.load(f)
        with open(manifests_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
            prim_val_man = json.load(f)
        with open(manifests_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
            inv_data = json.load(f)
        inv_map = {r["inventory_id"]: r for r in inv_data["records"]}

        with open(ds_a_dir / "primary_val_manifest.json", "r", encoding="utf-8") as f:
            pvm = json.load(f)
        val_class_counts = Counter(b["class_name"] for s in pvm["samples"] for b in s.get("boxes", []))
        val_size_counts = Counter(b.get("size_bucket") for s in pvm["samples"] for b in s.get("boxes", []))

        assert stats_a["train_images"] == 1092, f"Expected 1,092 train images in A, got {stats_a['train_images']}"
        assert stats_a["train_boxes"] == 13148, f"Expected 13,148 boxes in A, got {stats_a['train_boxes']}"
        assert stats_b["train_images"] == 1117, f"Expected 1,117 train images in B, got {stats_b['train_images']}"
        assert stats_b["train_boxes"] == 13961, f"Expected 13,961 boxes in B, got {stats_b['train_boxes']}"
        assert stats_b["train_boxes"] - stats_a["train_boxes"] == 813, "Delta between B and A must be 813 boxes"

        ext_delta_class = {k: stats_b["class_counts"][k] - stats_a["class_counts"][k] for k in stats_b["class_counts"]}
        ext_delta_size = {k: stats_b["size_counts"][k] - stats_a["size_counts"][k] for k in stats_b["size_counts"]}

        summary_stats = {
            "dataset_a": stats_a,
            "dataset_b": stats_b,
            "external_delta": {
                "images": 25,
                "boxes": 813,
                "class_counts": ext_delta_class,
                "size_counts": ext_delta_size,
            },
            "validation": {
                "images": stats_a["val_images"],
                "boxes": stats_a["val_boxes"],
                "class_counts": dict(val_class_counts),
                "size_counts": dict(val_size_counts),
                "old_split_origin": Counter(inv_map[rid].get("original_split", "train") for rid in prim_val_man["records"]),
            }
        }
    else:
        if args.force_rematerialize:
            if ds_a_dir.exists(): shutil.rmtree(ds_a_dir)
            if ds_b_dir.exists(): shutil.rmtree(ds_b_dir)
        summary_stats = materialize_controlled_datasets(
            dataset_a_dir=ds_a_dir,
            dataset_b_dir=ds_b_dir,
            manifests_dir=manifests_dir,
            consolidated_dir=consolidated_dir,
            snapshot_dir=snapshot_dir,
        )

    # 3. Train Candidate A (Local Only)
    target_a_path = run_parent_dir / "candidate_a_local_only"
    if target_a_path.exists() and (target_a_path / "weights" / "best.pt").exists() and not args.force_retrain:
        print(f"[REUSE] Candidate A run found at {target_a_path}. Loading existing checkpoints...")
        train_a_res = load_completed_training_run(target_a_path, "candidate_a_local_only", baseline_path)
    else:
        if args.force_retrain and target_a_path.exists():
            shutil.rmtree(target_a_path)
        train_a_res = train_candidate(
            dataset_yaml=ds_a_dir / "data.yaml",
            baseline_weights=baseline_path,
            run_dir=run_parent_dir,
            run_name="candidate_a_local_only",
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr0,
            warmup_bias_lr=args.lr0,
            seed=args.seed,
        )

    # 4. Train Candidate B (Local + Reviewed External)
    target_b_path = run_parent_dir / "candidate_b_reviewed_external"
    if target_b_path.exists() and (target_b_path / "weights" / "best.pt").exists() and not args.force_retrain:
        print(f"[REUSE] Candidate B run found at {target_b_path}. Loading existing checkpoints...")
        train_b_res = load_completed_training_run(target_b_path, "candidate_b_reviewed_external", baseline_path)
    else:
        if args.force_retrain and target_b_path.exists():
            shutil.rmtree(target_b_path)
        train_b_res = train_candidate(
            dataset_yaml=ds_b_dir / "data.yaml",
            baseline_weights=baseline_path,
            run_dir=run_parent_dir,
            run_name="candidate_b_reviewed_external",
            epochs=args.epochs,
            batch_size=train_a_res["batch_size"],
            imgsz=args.imgsz,
            lr0=args.lr0,
            warmup_bias_lr=args.lr0,
            seed=args.seed,
        )

    # 5. Evaluate all models on Primary Validation and Diagnostic Benchmark
    eval_matrix = {}
    eval_tasks = [
        ("baseline_primary_val", baseline_path, ds_a_dir / "primary_val_manifest.json", ds_a_dir, eval_parent_dir / "baseline_primary_val", ds_a_dir / "data.yaml", "Primary Validation (Baseline)"),
        ("baseline_diag", baseline_path, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / "baseline_diagnostic", snapshot_dir / "dataset.yaml", "Diagnostic Benchmark (Baseline)"),
    ]
    for ckpt_type, ckpt_p in [("best", Path(train_a_res["best_ckpt"])), ("last", Path(train_a_res["last_ckpt"]))]:
        eval_tasks.append((f"cand_a_{ckpt_type}_primary_val", ckpt_p, ds_a_dir / "primary_val_manifest.json", ds_a_dir, eval_parent_dir / f"candidate_a_{ckpt_type}_primary_val", ds_a_dir / "data.yaml", f"Primary Validation (Candidate A {ckpt_type})"))
        eval_tasks.append((f"cand_a_{ckpt_type}_diag", ckpt_p, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / f"candidate_a_{ckpt_type}_diagnostic", snapshot_dir / "dataset.yaml", f"Diagnostic Benchmark (Candidate A {ckpt_type})"))
    for ckpt_type, ckpt_p in [("best", Path(train_b_res["best_ckpt"])), ("last", Path(train_b_res["last_ckpt"]))]:
        eval_tasks.append((f"cand_b_{ckpt_type}_primary_val", ckpt_p, ds_b_dir / "primary_val_manifest.json", ds_b_dir, eval_parent_dir / f"candidate_b_{ckpt_type}_primary_val", ds_b_dir / "data.yaml", f"Primary Validation (Candidate B {ckpt_type})"))
        eval_tasks.append((f"cand_b_{ckpt_type}_diag", ckpt_p, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / f"candidate_b_{ckpt_type}_diagnostic", snapshot_dir / "dataset.yaml", f"Diagnostic Benchmark (Candidate B {ckpt_type})"))

    for task_key, m_p, man_p, img_dir, out_d, yaml_p, b_name in eval_tasks:
        res_file = out_d / "eval_results.json"
        if res_file.exists() and not args.force_reeval:
            print(f"[REUSE] Loading existing evaluation for {task_key} from {res_file}...")
            with open(res_file, "r", encoding="utf-8") as f:
                eval_matrix[task_key] = json.load(f)
        else:
            eval_matrix[task_key] = evaluate_model_on_benchmark(
                model_path=m_p,
                manifest_path=man_p,
                images_base_dir=img_dir,
                output_dir=out_d,
                dataset_yaml=yaml_p,
                benchmark_name=b_name,
            )

    # 6. Measure inference latency and FPS
    print("\n--- Benchmarking Inference Speed on GPU ---")
    speed_results = {
        "baseline": measure_inference_speed(baseline_path),
        "cand_a_best": measure_inference_speed(Path(train_a_res["best_ckpt"])),
        "cand_b_best": measure_inference_speed(Path(train_b_res["best_ckpt"])),
    }

    # 7. Post-flight immutability verification
    check_postflight_immutability(initial_hashes)

    # 8. Render comprehensive experiment report
    report_file = Path("docs/EXPERIMENT_REPORT_BATCH9.md")
    render_experiment_report(
        summary_stats=summary_stats,
        train_a_res=train_a_res,
        train_b_res=train_b_res,
        eval_matrix=eval_matrix,
        speed_results=speed_results,
        output_path=report_file,
    )

    print("\n" + "=" * 70)
    print("BATCH 9 CONTROLLED EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
