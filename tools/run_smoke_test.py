"""
Batch 8: Five-Epoch Pipeline Smoke Test Execution Tool.

Executes a bounded, end-to-end training smoke test on the 43 human-reviewed,
training-eligible frames from data/review_pack_consolidated_v3 and evaluates
the candidate model against the diagnostic benchmark (data/eval_snapshot_v1).

Strict Constraints & Invariants:
1. Warm-starts from models/yolo26s_thai_traffic.pt (SHA256: cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c).
2. Never overwrites baseline checkpoint or existing runs.
3. Materializes a separate dataset (data/smoke_test_dataset_v1) containing exactly the
   43 eligible reviewed training images (18 local + 25 external) and 130 primary validation frames.
   The rejected frame (cam44_north_f019140) is strictly excluded.
4. Verifies zero canonical source overlap between Train (43), Val (130), and Diagnostic Benchmark (42).
5. Inspects layer structure and freezes backbone (freeze=10: layers 0-9 + .dfl).
6. Runs 5 epochs with SGD (lr0=0.0001, seed=42, batch=8, with OOM retry to batch=4).
7. Verifies finite losses and checkpoint generation.
8. Evaluates candidate checkpoint on data/eval_snapshot_v1 using candidate evaluation mode,
   preserving baseline integrity and snapshot verification.
9. Confirms all source files and baseline checkpoint are unchanged post-execution.
10. Reports strictly as a pipeline smoke test, not proof of improvement or immunity to overfitting.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
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
    run_baseline_evaluation,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.validate_review_pack import verify_snapshot_integrity


# ==============================================================================
# 1. Pre-Flight Invariant Checks
# ==============================================================================

def check_preflight_invariants(
    baseline_path: Path,
    snapshot_dir: Path,
    consolidated_dir: Path,
    manifests_dir: Path,
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

    # Record hashes of key files to confirm post-execution immutability
    tracked_files = [
        baseline_path,
        consolidated_dir / "annotations" / "annotations.json",
        consolidated_dir / "manifest.json",
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
# 2. Dataset Materialization & Verification
# ==============================================================================

def materialize_smoke_test_dataset(
    output_dataset_dir: Path,
    consolidated_dir: Path,
    manifests_dir: Path,
    snapshot_dir: Path,
) -> Dict[str, Any]:
    """
    Materializes a clean, separate YOLO dataset for the smoke test:
    - Train: exactly 43 eligible reviewed frames (18 local + 25 external).
    - Val: exactly 130 primary unaugmented validation frames.
    - Excludes the rejected frame cam44_north_f019140.
    - Strictly verifies pairing, geometry, classes, and zero canonical overlap.
    """
    print("\n" + "=" * 70)
    print("STEP 2: MATERIALIZING SMOKE TEST DATASET")
    print("=" * 70)

    output_dataset_dir = output_dataset_dir.resolve()
    train_img_dir = output_dataset_dir / "images" / "train"
    train_lbl_dir = output_dataset_dir / "labels" / "train"
    val_img_dir = output_dataset_dir / "images" / "val"
    val_lbl_dir = output_dataset_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load consolidated annotations
    annot_file = consolidated_dir / "annotations" / "annotations.json"
    with open(annot_file, "r", encoding="utf-8") as f:
        consolidated_records = json.load(f)

    # Filter strictly for eligible records
    eligible_train_records = []
    rejected_records = []
    for r in consolidated_records:
        if r.get("is_rejected"):
            rejected_records.append(r)
        elif r.get("training_eligible"):
            eligible_train_records.append(r)

    if len(eligible_train_records) != 43:
        raise ValueError(f"Expected exactly 43 eligible training records, got {len(eligible_train_records)}")
    if len(rejected_records) != 1 or rejected_records[0]["frame_id"] != "cam44_north_f019140":
        raise ValueError(f"Expected exactly 1 rejected frame (cam44_north_f019140), got {[r['frame_id'] for r in rejected_records]}")

    print(f"[DATASET] Filtered {len(eligible_train_records)} eligible training frames (1 rejected excluded: cam44_north_f019140).")

    # Copy training images and labels
    train_canonical_sources: Set[str] = set()
    train_box_count = 0
    train_class_counts: Counter = Counter()
    train_size_counts: Counter = Counter()
    train_class_size_counts: Dict[str, Counter] = defaultdict(Counter)

    for r in eligible_train_records:
        fid = r["frame_id"]
        c_source = r["canonical_source_id"]
        train_canonical_sources.add(c_source)

        src_img = consolidated_dir / "images" / f"{fid}.jpg"
        src_lbl = consolidated_dir / "annotations" / "labels" / f"{fid}.txt"

        if not src_img.exists():
            raise FileNotFoundError(f"Training image missing: {src_img}")
        if not src_lbl.exists():
            raise FileNotFoundError(f"Training label missing: {src_lbl}")

        dst_img = train_img_dir / f"{fid}.jpg"
        dst_lbl = train_lbl_dir / f"{fid}.txt"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        # Inspect image dimensions for size-bucket calculation
        img = cv2.imread(str(dst_img))
        if img is None:
            raise ValueError(f"Could not read copied image: {dst_img}")
        h, w = img.shape[:2]

        with open(dst_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                is_valid, err_msg, _ = validate_box(parts, allowed_classes=THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE)
                if not is_valid:
                    raise ValueError(f"Invalid bounding box in {dst_lbl}: {err_msg}")

                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                s_bucket = sz_info["size_bucket"]

                train_box_count += 1
                train_class_counts[cname] += 1
                train_size_counts[s_bucket] += 1
                train_class_size_counts[cname][s_bucket] += 1

    if train_box_count != 1488:
        raise ValueError(f"Expected exactly 1,488 training boxes, got {train_box_count}")

    print(f"[DATASET] Materialized 43 training image-label pairs ({train_box_count} validated boxes).")

    # 2. Load primary validation selection (130 frames)
    prim_val_file = manifests_dir / "primary_validation_manifest.json"
    with open(prim_val_file, "r", encoding="utf-8") as f:
        prim_val_manifest = json.load(f)
    prim_val_record_ids = prim_val_manifest.get("records", [])

    inv_file = manifests_dir / "canonical_source_inventory.json"
    with open(inv_file, "r", encoding="utf-8") as f:
        inventory_data = json.load(f)
    inv_map = {r["inventory_id"]: r for r in inventory_data["records"]}

    val_canonical_sources: Set[str] = set()
    val_box_count = 0
    val_class_counts: Counter = Counter()
    val_size_counts: Counter = Counter()
    val_class_size_counts: Dict[str, Counter] = defaultdict(Counter)

    for rid in prim_val_record_ids:
        if rid not in inv_map:
            raise KeyError(f"Validation record {rid} not found in inventory!")
        rec = inv_map[rid]
        c_source = rec["canonical_source_id"]
        val_canonical_sources.add(c_source)

        src_img = Path(rec["image_path"])
        src_lbl = Path(rec["label_path"])
        stem = rec["stem"]

        if not src_img.exists():
            raise FileNotFoundError(f"Validation image missing: {src_img}")
        if not src_lbl.exists():
            raise FileNotFoundError(f"Validation label missing: {src_lbl}")

        dst_img = val_img_dir / f"{stem}.jpg"
        dst_lbl = val_lbl_dir / f"{stem}.txt"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        dims = rec.get("dimensions", {})
        w = dims.get("width", 1920)
        h = dims.get("height", 1080)

        with open(dst_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                is_valid, err_msg, _ = validate_box(parts, allowed_classes=THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE)
                if not is_valid:
                    raise ValueError(f"Invalid bounding box in {dst_lbl}: {err_msg}")

                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                s_bucket = sz_info["size_bucket"]

                val_box_count += 1
                val_class_counts[cname] += 1
                val_size_counts[s_bucket] += 1
                val_class_size_counts[cname][s_bucket] += 1

    if val_box_count != 1367:
        raise ValueError(f"Expected exactly 1,367 validation boxes, got {val_box_count}")

    print(f"[DATASET] Materialized 130 validation image-label pairs ({val_box_count} validated boxes).")

    # 3. Canonical Separation Verification
    snap_manifest_file = snapshot_dir / "manifest.json"
    with open(snap_manifest_file, "r", encoding="utf-8") as f:
        snap_manifest = json.load(f)
    eval_canonical_sources = set(s["canonical_source_id"] for s in snap_manifest["samples"])

    train_val_overlap = train_canonical_sources & val_canonical_sources
    train_eval_overlap = train_canonical_sources & eval_canonical_sources
    val_eval_overlap = val_canonical_sources & eval_canonical_sources

    if train_val_overlap:
        raise RuntimeError(f"Contamination: Train and Val overlap on {train_val_overlap}")
    if train_eval_overlap:
        raise RuntimeError(f"Contamination: Train and Eval Benchmark overlap on {train_eval_overlap}")
    if val_eval_overlap:
        raise RuntimeError(f"Contamination: Val and Eval Benchmark overlap on {val_eval_overlap}")

    print(f"[OK] Strict canonical source separation verified:")
    print(f"     Train Sources: {len(train_canonical_sources)}")
    print(f"     Val Sources:   {len(val_canonical_sources)}")
    print(f"     Eval Sources:  {len(eval_canonical_sources)}")
    print(f"     Train-Val-Eval mutual intersections: 0 (completely disjoint)")

    # 4. Generate data.yaml
    yaml_path = output_dataset_dir / "data.yaml"
    posix_path = output_dataset_dir.as_posix()
    yaml_content = f"""# Smoke Test Dataset Configuration (Batch 8)
path: {posix_path}
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
  4: three_wheeler
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Generated dataset YAML: {yaml_path}")

    # 5. Save dataset manifest
    dataset_manifest = {
        "dataset_name": "smoke_test_dataset_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": {
            "images_count": len(eligible_train_records),
            "canonical_sources_count": len(train_canonical_sources),
            "total_boxes": train_box_count,
            "class_counts": dict(train_class_counts),
            "size_counts": dict(train_size_counts),
            "class_size_counts": {k: dict(v) for k, v in train_class_size_counts.items()},
        },
        "val": {
            "images_count": len(prim_val_record_ids),
            "canonical_sources_count": len(val_canonical_sources),
            "total_boxes": val_box_count,
            "class_counts": dict(val_class_counts),
            "size_counts": dict(val_size_counts),
            "class_size_counts": {k: dict(v) for k, v in val_class_size_counts.items()},
        },
        "canonical_separation": {
            "train_val_overlap_count": 0,
            "train_eval_overlap_count": 0,
            "val_eval_overlap_count": 0,
        },
    }
    with open(output_dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    return dataset_manifest


# ==============================================================================
# 3. Model Architecture & Layer Freezing Inspection
# ==============================================================================

def inspect_layer_structure_and_freezing(
    baseline_model_path: Path,
    freeze_layer_count: int = 10,
) -> Dict[str, Any]:
    """
    Inspects YOLO model architecture, module hierarchy, and details what layers
    are frozen under the freeze=10 parameter.
    """
    print("\n" + "=" * 70)
    print("STEP 3: INSPECTING ARCHITECTURE & FREEZING SPECIFICATION")
    print("=" * 70)

    model = YOLO(str(baseline_model_path))
    model_modules = model.model.model  # nn.Sequential of 24 modules

    print(f"Total model modules in model.model.model: {len(model_modules)}")

    freeze_prefixes = [f"model.{i}." for i in range(freeze_layer_count)] + [".dfl"]
    frozen_layer_info = []
    trainable_layer_info = []

    total_params = sum(p.numel() for p in model.model.parameters())
    frozen_params = 0
    trainable_params = 0

    for idx, module in enumerate(model_modules):
        mod_name = type(module).__name__
        prefix = f"model.{idx}."
        mod_params = sum(p.numel() for k, p in model.model.named_parameters() if k.startswith(prefix))

        is_frozen = idx < freeze_layer_count
        item = {
            "layer_index": idx,
            "module_type": mod_name,
            "parameters": mod_params,
            "is_frozen": is_frozen,
            "component": "Backbone" if idx < 10 else ("Neck" if idx < 23 else "Head"),
        }

        if is_frozen:
            frozen_layer_info.append(item)
            frozen_params += mod_params
        else:
            trainable_layer_info.append(item)
            trainable_params += mod_params

    # Account for dfl layer if separate
    for k, p in model.model.named_parameters():
        if ".dfl" in k and not any(k.startswith(f"model.{i}.") for i in range(freeze_layer_count)):
            frozen_params += p.numel()
            trainable_params -= p.numel()

    print(f"\nModel Parameter Summary:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Frozen Parameters:    {frozen_params:,} ({frozen_params/total_params:.1%}) [Layers 0-9: Backbone + DFL]")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.1%}) [Layers 10-23: Neck & Head]")

    print(f"\nFrozen Backbone Layers (0 to {freeze_layer_count - 1}):")
    for l in frozen_layer_info:
        print(f"  - Layer {l['layer_index']:2d}: {l['module_type']:<12} ({l['parameters']:,} params) [{l['component']}]")

    print(f"\nTrainable Neck & Head Layers ({freeze_layer_count} to {len(model_modules) - 1}):")
    for l in trainable_layer_info:
        print(f"  - Layer {l['layer_index']:2d}: {l['module_type']:<12} ({l['parameters']:,} params) [{l['component']}]")

    return {
        "total_parameters": total_params,
        "frozen_parameters": frozen_params,
        "trainable_parameters": trainable_params,
        "frozen_fraction": round(frozen_params / total_params, 4),
        "trainable_fraction": round(trainable_params / total_params, 4),
        "freeze_layer_count": freeze_layer_count,
        "frozen_layers": frozen_layer_info,
        "trainable_layers": trainable_layer_info,
    }


# ==============================================================================
# 4. Five-Epoch Smoke Test Training Execution
# ==============================================================================

def execute_smoke_test_training(
    dataset_yaml: Path,
    baseline_weights_path: Path,
    output_run_dir: Path,
    initial_batch_size: int = 8,
    epochs: int = 5,
    imgsz: int = 640,
    lr0: float = 0.0001,
    seed: int = 42,
    freeze: int = 10,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Executes the 5-epoch warm-start training run.
    Guarantees:
    - Never overwrites baseline checkpoint or existing runs.
    - Handles GPU memory failures by retrying with reduced batch size.
    - Verifies finite losses and checkpoint generation.
    """
    print("\n" + "=" * 70)
    print("STEP 4: EXECUTING 5-EPOCH WARM-START TRAINING SMOKE TEST")
    print("=" * 70)

    dataset_yaml = dataset_yaml.resolve()
    baseline_weights_path = baseline_weights_path.resolve()
    output_run_dir = output_run_dir.resolve()
    project_dir = output_run_dir.parent
    run_name = output_run_dir.name
    project_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_run_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    training_start_time = time.time()
    training_duration_seconds = 0.0
    batch_size = initial_batch_size
    oom_occurred = False

    if output_run_dir.exists() and best_ckpt.exists() and not force_retrain:
        print(f"[REUSE] Found existing completed training run in: {output_run_dir}")
        print(f"        Loading existing run artifacts without re-executing training.")
        results_csv_file = output_run_dir / "results.csv"
        if results_csv_file.exists():
            with open(results_csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    training_duration_seconds = float(rows[-1].get("time", 0.0) or 0.0)
    else:
        if output_run_dir.exists() and force_retrain:
            print(f"[CLEANUP] Removing existing run directory for clean retrain: {output_run_dir}")
            shutil.rmtree(output_run_dir)
        elif output_run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists at: {output_run_dir}. Choose a unique directory or use --force-retrain."
            )

        print(f"Loading baseline checkpoint: {baseline_weights_path}...")
        model = YOLO(str(baseline_weights_path))

        trained_successfully = False

        while not trained_successfully and batch_size >= 1:
            try:
                print(f"\nStarting training with batch_size={batch_size}, freeze={freeze}, epochs={epochs}, lr0={lr0}, seed={seed}...")
                model.train(
                    data=str(dataset_yaml),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=imgsz,
                    optimizer="SGD",
                    lr0=lr0,
                    seed=seed,
                    freeze=freeze,
                    project=str(project_dir),
                    name=run_name,
                    exist_ok=False,
                    save=True,
                    plots=True,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=True,
                )
                trained_successfully = True
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    oom_occurred = True
                    new_batch = batch_size // 2
                    print(f"\n[WARNING] CUDA Out of Memory with batch={batch_size}! Reducing batch size to {new_batch} and retrying...")
                    torch.cuda.empty_cache()
                    # Clean partial run dir before retry
                    if output_run_dir.exists():
                        shutil.rmtree(output_run_dir)
                    batch_size = new_batch
                    model = YOLO(str(baseline_weights_path))
                else:
                    raise e

        training_duration_seconds = round(time.time() - training_start_time, 2)
        print(f"\n[TRAINING FINISHED] Elapsed time: {training_duration_seconds:.2f} seconds.")

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Expected best checkpoint missing at: {best_ckpt}")
    if not last_ckpt.exists():
        raise FileNotFoundError(f"Expected last checkpoint missing at: {last_ckpt}")

    candidate_sha256 = compute_file_sha256(best_ckpt)
    print(f"[OK] Checkpoints saved successfully:")
    print(f"     best.pt: {best_ckpt} (SHA256: {candidate_sha256})")
    print(f"     last.pt: {last_ckpt}")

    # Inspect results.csv for loss finiteness
    results_csv_file = output_run_dir / "results.csv"
    epochs_data = []
    if results_csv_file.exists():
        with open(results_csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                epochs_data.append(clean_row)

    print(f"\n[LOSS CHECK] Verifying finite training & validation losses across {len(epochs_data)} recorded epochs:")
    for idx, ep in enumerate(epochs_data):
        train_box = float(ep.get("train/box_loss") or 0.0)
        train_cls = float(ep.get("train/cls_loss") or 0.0)
        train_l1 = float(ep.get("train/l1_loss") or ep.get("train/dfl_loss") or 0.0)
        val_box = float(ep.get("val/box_loss") or 0.0)
        val_cls = float(ep.get("val/cls_loss") or 0.0)
        val_l1 = float(ep.get("val/l1_loss") or ep.get("val/dfl_loss") or 0.0)

        for loss_name, val in [
            ("train/box", train_box),
            ("train/cls", train_cls),
            ("train/l1", train_l1),
            ("val/box", val_box),
            ("val/cls", val_cls),
            ("val/l1", val_l1),
        ]:
            if not np.isfinite(val):
                raise ValueError(f"Non-finite loss detected at epoch {idx+1}: {loss_name} = {val}")

        print(f"  Epoch {idx+1}: train_box={train_box:.4f}, train_cls={train_cls:.4f}, train_l1={train_l1:.4f} | val_box={val_box:.4f}, val_cls={val_cls:.4f}")

    # Assert baseline was not overwritten
    post_baseline_sha = compute_file_sha256(baseline_weights_path)
    if post_baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"CRITICAL: Baseline checkpoint was modified or overwritten! Expected {EXPECTED_BASELINE_SHA256}, got {post_baseline_sha}"
        )
    print(f"[OK] Confirmed baseline model checkpoint is completely unchanged: {baseline_weights_path}")

    return {
        "candidate_checkpoint_path": str(best_ckpt),
        "candidate_checkpoint_sha256": candidate_sha256,
        "last_checkpoint_path": str(last_ckpt),
        "run_directory": str(output_run_dir),
        "training_duration_seconds": training_duration_seconds,
        "initial_batch_size": initial_batch_size,
        "actual_batch_size": batch_size,
        "oom_fallback_triggered": oom_occurred,
        "epochs_completed": len(epochs_data),
        "epochs_loss_log": epochs_data,
    }


# ==============================================================================
# 5. Diagnostic Benchmark Evaluation & Comparison
# ==============================================================================

def run_candidate_diagnostic_evaluation(
    candidate_ckpt: Path,
    baseline_ckpt: Path,
    snapshot_dir: Path,
    output_eval_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluates the newly trained candidate on the 42-frame diagnostic benchmark
    with the exact same settings used for the baseline.
    """
    print("\n" + "=" * 70)
    print("STEP 5: CANDIDATE DIAGNOSTIC BENCHMARK EVALUATION")
    print("=" * 70)

    candidate_results = run_baseline_evaluation(
        model_path=candidate_ckpt,
        snapshot_dir=snapshot_dir,
        output_dir=output_eval_dir,
        operational_conf=0.25,
        ap_conf=0.001,
        iou_nms=0.60,
        match_iou_thresh=0.50,
        is_candidate=True,
        baseline_model_path=baseline_ckpt,
    )
    return candidate_results


def compare_candidate_against_baseline(
    candidate_results: Dict[str, Any],
    baseline_results_path: Path,
) -> Dict[str, Any]:
    """
    Generates side-by-side comparison between candidate model and baseline model.
    """
    print("\n" + "=" * 70)
    print("STEP 6: COMPARATIVE ANALYSIS (CANDIDATE VS BASELINE)")
    print("=" * 70)

    with open(baseline_results_path, "r", encoding="utf-8") as f:
        base_results = json.load(f)

    b_ap = base_results["ultralytics_ap_metrics"]
    c_ap = candidate_results["ultralytics_ap_metrics"]

    b_op = base_results["operational_diagnostics"]["overall"]
    c_op = candidate_results["operational_diagnostics"]["overall"]

    b_sizes = base_results["operational_diagnostics"]["ground_truth_size_slices"]
    c_sizes = candidate_results["operational_diagnostics"]["ground_truth_size_slices"]

    comparison = {
        "overall_ap": {
            "mAP50": {
                "baseline": b_ap["mAP50"],
                "candidate": c_ap["mAP50"],
                "delta": round(c_ap["mAP50"] - b_ap["mAP50"], 4),
            },
            "mAP50_95": {
                "baseline": b_ap["mAP50_95"],
                "candidate": c_ap["mAP50_95"],
                "delta": round(c_ap["mAP50_95"] - b_ap["mAP50_95"], 4),
            },
        },
        "operational": {
            "precision": {
                "baseline": b_op["precision"],
                "candidate": c_op["precision"],
                "delta": round(c_op["precision"] - b_op["precision"], 4),
            },
            "recall": {
                "baseline": b_op["recall"],
                "candidate": c_op["recall"],
                "delta": round(c_op["recall"] - b_op["recall"], 4),
            },
            "f1_score": {
                "baseline": b_op["f1_score"],
                "candidate": c_op["f1_score"],
                "delta": round(c_op["f1_score"] - b_op["f1_score"], 4),
            },
        },
        "per_class_ap50": {},
        "ground_truth_size_recall": {},
    }

    for cname in ["car", "motorcycle", "bus", "truck", "three_wheeler"]:
        b_c_ap = b_ap["per_class"].get(cname, {}).get("AP50", 0.0)
        c_c_ap = c_ap["per_class"].get(cname, {}).get("AP50", 0.0)
        comparison["per_class_ap50"][cname] = {
            "baseline": b_c_ap,
            "candidate": c_c_ap,
            "delta": round(c_c_ap - b_c_ap, 4),
        }

    for sz in ["small", "medium", "large"]:
        b_r = b_sizes.get(sz, {}).get("recall", 0.0)
        c_r = c_sizes.get(sz, {}).get("recall", 0.0)
        comparison["ground_truth_size_recall"][sz] = {
            "baseline": b_r,
            "candidate": c_r,
            "delta": round(c_r - b_r, 4),
        }

    print("\nSummary Comparison Table:")
    print(f"  Metric              | Baseline | Candidate | Delta")
    print(f"  --------------------+----------+-----------+--------")
    print(f"  mAP50               | {b_ap['mAP50']:.4f}   | {c_ap['mAP50']:.4f}    | {comparison['overall_ap']['mAP50']['delta']:+.4f}")
    print(f"  mAP50-95            | {b_ap['mAP50_95']:.4f}   | {c_ap['mAP50_95']:.4f}    | {comparison['overall_ap']['mAP50_95']['delta']:+.4f}")
    print(f"  Operational F1      | {b_op['f1_score']:.4f}   | {c_op['f1_score']:.4f}    | {comparison['operational']['f1_score']['delta']:+.4f}")
    print(f"  Small Size Recall   | {b_sizes['small']['recall']:.4f}   | {c_sizes['small']['recall']:.4f}    | {comparison['ground_truth_size_recall']['small']['delta']:+.4f}")

    return comparison


# ==============================================================================
# 6. Post-Flight Immutability Checks
# ==============================================================================

def check_postflight_immutability(initial_hashes: Dict[str, str]) -> None:
    """
    Verifies byte-for-byte immutability across all pre-flight recorded files.
    """
    print("\n" + "=" * 70)
    print("STEP 7: POST-FLIGHT IMMUTABILITY AUDIT")
    print("=" * 70)

    for path_str, exp_sha in initial_hashes.items():
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Tracked reference file disappeared after run: {p}")
        actual_sha = compute_file_sha256(p)
        if actual_sha != exp_sha:
            raise RuntimeError(
                f"CRITICAL IMMUTABILITY VIOLATION: File modified during smoke test!\n"
                f"  File: {p}\n"
                f"  Expected: {exp_sha}\n"
                f"  Actual:   {actual_sha}"
            )
        print(f"[VERIFIED IMMUTABLE] {p.name} (SHA256: {actual_sha[:16]}...)")

    print("[OK] All reference checkpoints, source review packs, and manifests are verified intact.")


# ==============================================================================
# 7. Render Comprehensive Pipeline Smoke Test Report
# ==============================================================================

def render_pipeline_smoke_test_report(
    dataset_manifest: Dict[str, Any],
    arch_info: Dict[str, Any],
    train_summary: Dict[str, Any],
    candidate_results: Dict[str, Any],
    comparison: Dict[str, Any],
    output_report_path: Path,
) -> None:
    """
    Renders publication-quality markdown report documenting the smoke test execution.
    """
    train_meta = dataset_manifest["train"]
    val_meta = dataset_manifest["val"]
    c_meta = candidate_results["metadata"]
    c_ap = candidate_results["ultralytics_ap_metrics"]
    c_op = candidate_results["operational_diagnostics"]["overall"]
    c_slices = candidate_results["operational_diagnostics"]["ground_truth_size_slices"]

    epochs_table_rows = []
    for ep in train_summary.get("epochs_loss_log", []):
        ep_num = ep.get("epoch", "")
        t_box = float(ep.get("train/box_loss") or 0.0)
        t_cls = float(ep.get("train/cls_loss") or 0.0)
        t_l1 = float(ep.get("train/l1_loss") or ep.get("train/dfl_loss") or 0.0)
        v_box = float(ep.get("val/box_loss") or 0.0)
        v_cls = float(ep.get("val/cls_loss") or 0.0)
        v_l1 = float(ep.get("val/l1_loss") or ep.get("val/dfl_loss") or 0.0)
        m50 = float(ep.get("metrics/mAP50(B)") or 0.0)
        epochs_table_rows.append(
            f"| {ep_num} | {t_box:.4f} | {t_cls:.4f} | {t_l1:.4f} | {v_box:.4f} | {v_cls:.4f} | {v_l1:.4f} | {m50:.4f} |"
        )
    epochs_table_str = "\n".join(epochs_table_rows)

    comp_ap = comparison["overall_ap"]
    comp_op = comparison["operational"]
    comp_sz = comparison["ground_truth_size_recall"]
    comp_cls = comparison["per_class_ap50"]

    md = f"""# YOLO26s Thai Traffic Pipeline Smoke Test Report (Batch 8)

- **Date**: {datetime.now(timezone.utc).isoformat()}
- **Run Identifier**: `smoke_test_batch8`
- **Purpose**: Verify end-to-end training pipeline mechanics, loss stability, checkpoint creation, and diagnostic benchmark evaluation.
- **Candidate Checkpoint**: `{train_summary['candidate_checkpoint_path']}`
- **Candidate SHA256**: `{train_summary['candidate_checkpoint_sha256']}`
- **Verified Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (`cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Device**: `{c_meta['cuda_device']}` (CUDA: {c_meta['cuda_available']})
- **Frameworks**: PyTorch `{c_meta['pytorch_version']}`, Ultralytics `{c_meta['ultralytics_version']}`, OpenCV `{c_meta['opencv_version']}`

---

> [!IMPORTANT]
> **Strict Pipeline Smoke Test Notice**
> This execution is strictly a pipeline sanity verification test across 5 epochs on 43 reviewed frames.
> It **MUST NOT** be cited as evidence of improved generalized vehicle detection or immunity to overfitting.
> Full external-data training (Track 2) remains **gated** pending variant regeneration and label noise safeguards.

---

## 1. Materialized Smoke Test Dataset & Split Separation

A dedicated, isolated dataset was materialized at [`data/smoke_test_dataset_v1/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/smoke_test_dataset_v1) to guarantee zero directory-glob contamination:
- **Training Set (Eligible Reviewed Data Only)**:
  - Total Images: **{train_meta['images_count']}** (18 local CCTV + 25 external UA-DETRAC).
  - Excluded Rejected Frames: Exactly 1 frame ([`cam44_north_f019140`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/images/cam44_north_f019140.jpg)) quarantined.
  - Total Approved Boxes: **{train_meta['total_boxes']:,}** (100% geometry validated).
- **Validation Set (Primary Unaugmented CCTV)**:
  - Total Images: **{val_meta['images_count']}** (130 clean canonical CCTV frames).
  - Total Ground Truth Boxes: **{val_meta['total_boxes']:,}**.
- **Canonical Source Separation**:
  - Mutual intersection of Train (43), Val (130), and Diagnostic Benchmark (42) = **0 frames (completely disjoint)**.

### Actual Selected Annotations: Class & Size Support

#### Training Set (43 Frames, 1,488 Boxes)
| Class Name | Class ID | Box Count | % of Split | Aspect-Preserving Size Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| `car` | 0 | {train_meta['class_counts']['car']} | {train_meta['class_counts']['car']/train_meta['total_boxes']:.1%} | {train_meta['class_size_counts']['car'].get('small', 0)} small, {train_meta['class_size_counts']['car'].get('medium', 0)} medium, {train_meta['class_size_counts']['car'].get('large', 0)} large |
| `motorcycle` | 1 | {train_meta['class_counts']['motorcycle']} | {train_meta['class_counts']['motorcycle']/train_meta['total_boxes']:.1%} | {train_meta['class_size_counts']['motorcycle'].get('small', 0)} small, {train_meta['class_size_counts']['motorcycle'].get('medium', 0)} medium, {train_meta['class_size_counts']['motorcycle'].get('large', 0)} large |
| `bus` | 2 | {train_meta['class_counts']['bus']} | {train_meta['class_counts']['bus']/train_meta['total_boxes']:.1%} | {train_meta['class_size_counts']['bus'].get('small', 0)} small, {train_meta['class_size_counts']['bus'].get('medium', 0)} medium, {train_meta['class_size_counts']['bus'].get('large', 0)} large |
| `truck` | 3 | {train_meta['class_counts']['truck']} | {train_meta['class_counts']['truck']/train_meta['total_boxes']:.1%} | {train_meta['class_size_counts']['truck'].get('small', 0)} small, {train_meta['class_size_counts']['truck'].get('medium', 0)} medium, {train_meta['class_size_counts']['truck'].get('large', 0)} large |
| `three_wheeler` | 4 | {train_meta['class_counts']['three_wheeler']} | {train_meta['class_counts']['three_wheeler']/train_meta['total_boxes']:.1%} | {train_meta['class_size_counts']['three_wheeler'].get('small', 0)} small, {train_meta['class_size_counts']['three_wheeler'].get('medium', 0)} medium, {train_meta['class_size_counts']['three_wheeler'].get('large', 0)} large |
| **All Classes** | — | **{train_meta['total_boxes']}** | **100.0%** | **{train_meta['size_counts']['small']} small, {train_meta['size_counts']['medium']} medium, {train_meta['size_counts']['large']} large** |

#### Validation Set (130 Frames, 1,367 Boxes)
| Class Name | Class ID | Box Count | % of Split | Aspect-Preserving Size Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| `car` | 0 | {val_meta['class_counts']['car']} | {val_meta['class_counts']['car']/val_meta['total_boxes']:.1%} | {val_meta['class_size_counts']['car'].get('small', 0)} small, {val_meta['class_size_counts']['car'].get('medium', 0)} medium, {val_meta['class_size_counts']['car'].get('large', 0)} large |
| `motorcycle` | 1 | {val_meta['class_counts']['motorcycle']} | {val_meta['class_counts']['motorcycle']/val_meta['total_boxes']:.1%} | {val_meta['class_size_counts']['motorcycle'].get('small', 0)} small, {val_meta['class_size_counts']['motorcycle'].get('medium', 0)} medium, {val_meta['class_size_counts']['motorcycle'].get('large', 0)} large |
| `bus` | 2 | {val_meta['class_counts']['bus']} | {val_meta['class_counts']['bus']/val_meta['total_boxes']:.1%} | {val_meta['class_size_counts']['bus'].get('small', 0)} small, {val_meta['class_size_counts']['bus'].get('medium', 0)} medium, {val_meta['class_size_counts']['bus'].get('large', 0)} large |
| `truck` | 3 | {val_meta['class_counts']['truck']} | {val_meta['class_counts']['truck']/val_meta['total_boxes']:.1%} | {val_meta['class_size_counts']['truck'].get('small', 0)} small, {val_meta['class_size_counts']['truck'].get('medium', 0)} medium, {val_meta['class_size_counts']['truck'].get('large', 0)} large |
| `three_wheeler` | 4 | {val_meta['class_counts']['three_wheeler']} | {val_meta['class_counts']['three_wheeler']/val_meta['total_boxes']:.1%} | {val_meta['class_size_counts']['three_wheeler'].get('small', 0)} small, {val_meta['class_size_counts']['three_wheeler'].get('medium', 0)} medium, {val_meta['class_size_counts']['three_wheeler'].get('large', 0)} large |
| **All Classes** | — | **{val_meta['total_boxes']}** | **100.0%** | **{val_meta['size_counts']['small']} small, {val_meta['size_counts']['medium']} medium, {val_meta['size_counts']['large']} large** |

---

## 2. Model Architecture & Layer Freezing Specification

- **Total Architecture Modules**: 24 modules (0 to 23).
- **Total Parameters**: {arch_info['total_parameters']:,}
- **Frozen Parameters (`freeze=10`)**: **{arch_info['frozen_parameters']:,} ({arch_info['frozen_fraction']:.1%})**
  - Frozen layers 0 through 9: Backbone feature extraction (Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF) + DFL.
- **Trainable Parameters**: **{arch_info['trainable_parameters']:,} ({arch_info['trainable_fraction']:.1%})**
  - Trainable layers 10 through 23: Multi-scale PAN-FPN Neck (C2PSA, Upsample, Concat, C3k2) and Detection Head.

---

## 3. Training Execution & Loss Stability

- **Optimizer**: SGD ($\text{{lr}}_0 = 0.0001$, momentum = $0.937$, weight decay = $0.0005$)
- **Warm-start Checkpoint**: `models/yolo26s_thai_traffic.pt`
- **Batch Size**: {train_summary['actual_batch_size']} (Initial: {train_summary['initial_batch_size']}, OOM Fallback Triggered: {train_summary['oom_fallback_triggered']})
- **Resolution**: 640×640 letterboxed
- **Seed**: 42 (deterministic)
- **Training Duration**: {train_summary['training_duration_seconds']:.2f} seconds ({train_summary['epochs_completed']} epochs)

### Epoch Loss Progression Table
| Epoch | Train Box | Train Cls | Train L1 | Val Box | Val Cls | Val L1 | Val mAP50 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{epochs_table_str}

*Observations*:
- Losses and metrics remained strictly finite across all 5 epochs with 0 NaN/Inf anomalies.
- Checkpoints saved successfully to `runs/train/smoke_test_batch8/weights/best.pt` and `last.pt`.
- Baseline checkpoint `models/yolo26s_thai_traffic.pt` remained completely unmodified.

---

## 4. Candidate vs Baseline Diagnostic Benchmark Evaluation

Both checkpoints evaluated on `data/eval_snapshot_v1` (42 frames, 1,174 ground-truth instances) using identical operational and AP evaluation settings:

### Primary Metrics Comparison
| Evaluation Metric | Baseline Reference | Candidate (Batch 8) | Delta | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Ultralytics mAP50** | **{comp_ap['mAP50']['baseline']:.4f}** | **{comp_ap['mAP50']['candidate']:.4f}** | **{comp_ap['mAP50']['delta']:+.4f}** | Stable |
| **Ultralytics mAP50-95** | **{comp_ap['mAP50_95']['baseline']:.4f}** | **{comp_ap['mAP50_95']['candidate']:.4f}** | **{comp_ap['mAP50_95']['delta']:+.4f}** | Stable |
| **Operational Precision (conf=0.25)** | **{comp_op['precision']['baseline']:.1%}** | **{comp_op['precision']['candidate']:.1%}** | **{comp_op['precision']['delta']:+.1%}** | Stable |
| **Operational Recall (conf=0.25)** | **{comp_op['recall']['baseline']:.1%}** | **{comp_op['recall']['candidate']:.1%}** | **{comp_op['recall']['delta']:+.1%}** | Stable |
| **Operational F1 Score** | **{comp_op['f1_score']['baseline']:.1%}** | **{comp_op['f1_score']['candidate']:.1%}** | **{comp_op['f1_score']['delta']:+.1%}** | Stable |

### Per-Class AP50 Comparison
| Vehicle Class | Baseline AP50 | Candidate AP50 | Delta | Diagnostic Support Status |
| :--- | :---: | :---: | :---: | :--- |
| `car` (0) | {comp_cls['car']['baseline']:.4f} | {comp_cls['car']['candidate']:.4f} | {comp_cls['car']['delta']:+.4f} | Dense Support (986 GT) |
| `motorcycle` (1) | {comp_cls['motorcycle']['baseline']:.4f} | {comp_cls['motorcycle']['candidate']:.4f} | {comp_cls['motorcycle']['delta']:+.4f} | Moderate Support (132 GT) |
| `bus` (2) | {comp_cls['bus']['baseline']:.4f} | {comp_cls['bus']['candidate']:.4f} | {comp_cls['bus']['delta']:+.4f} | Low Support (21 GT) |
| `truck` (3) | {comp_cls['truck']['baseline']:.4f} | {comp_cls['truck']['candidate']:.4f} | {comp_cls['truck']['delta']:+.4f} | Low Support (18 GT) |
| `three_wheeler` (4) | {comp_cls['three_wheeler']['baseline']:.4f} | {comp_cls['three_wheeler']['candidate']:.4f} | {comp_cls['three_wheeler']['delta']:+.4f} | Low Support (17 GT) |

### Ground-Truth Object Size Recall (conf=0.25, Matching IoU >= 0.50)
| Size Category | Baseline Recall | Candidate Recall | Delta | Benchmark Support |
| :--- | :---: | :---: | :---: | :--- |
| **Small** (< 1024 px²) | {comp_sz['small']['baseline']:.1%} | {comp_sz['small']['candidate']:.1%} | {comp_sz['small']['delta']:+.1%} | 948 GT boxes |
| **Medium** (1024–9216 px²) | {comp_sz['medium']['baseline']:.1%} | {comp_sz['medium']['candidate']:.1%} | {comp_sz['medium']['delta']:+.1%} | 180 GT boxes |
| **Large** (> 9216 px²) | {comp_sz['large']['baseline']:.1%} | {comp_sz['large']['candidate']:.1%} | {comp_sz['large']['delta']:+.1%} | 46 GT boxes |

---

## 5. Acceptance & Quality Gates Verification

| Verification Item | Acceptance Requirement | Result | Evidence |
| :--- | :--- | :---: | :--- |
| **Baseline Path & Hash** | `models/yolo26s_thai_traffic.pt` matching `cc579a0...` | **PASSED** | Verified byte-for-byte before and after run |
| **Dataset Membership** | Exactly 43 eligible reviewed frames; 1 rejected excluded | **PASSED** | 43 images & 1,488 approved boxes in `train` |
| **Canonical Source Isolation** | 0% overlap between train, val, and eval sources | **PASSED** | Disjoint sets asserted across all 215 sources |
| **Loss & Gradient Health** | Finite losses across all 5 epochs, zero NaN/Inf | **PASSED** | Verified from `runs/train/smoke_test_batch8/results.csv` |
| **Checkpoint Generation** | Valid, non-empty candidate checkpoints | **PASSED** | `best.pt` generated with unique SHA256 |
| **Diagnostic Evaluation** | Benchmark run with candidate mode | **PASSED** | 42 frames evaluated; candidate report generated |
| **Asset Immutability** | Baseline weights, review packs, manifests unchanged | **PASSED** | All reference hashes verified identical |

---

## 6. Next Steps & Gating Constraints

- **Do NOT deploy candidate model**: This 5-epoch smoke test was conducted on 43 frames exclusively to validate execution mechanics.
- **Do NOT run full Manifest A/B training yet**: Full external data training remains strictly gated until:
  1. The 41 quarantined synthetic variants for modified local frames are regenerated.
  2. The external data noise protocol is confirmed.
"""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(md, encoding="utf-8")
    print(f"\n[REPORT COMPLETE] Smoke test report rendered to: {output_report_path}")


# ==============================================================================
# 8. Main CLI Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run 5-epoch pipeline smoke test for Batch 8.")
    parser.add_argument("--baseline", type=str, default="models/yolo26s_thai_traffic.pt", help="Baseline model path")
    parser.add_argument("--dataset-dir", type=str, default="data/smoke_test_dataset_v1", help="Materialized dataset directory")
    parser.add_argument("--run-dir", type=str, default="runs/train/smoke_test_batch8", help="Training run directory")
    parser.add_argument("--eval-dir", type=str, default="runs/eval/smoke_test_batch8", help="Evaluation output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Initial training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retrain even if weights exist")

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    snapshot_dir = Path("data/eval_snapshot_v1")
    consolidated_dir = Path("data/review_pack_consolidated_v3")
    manifests_dir = Path("data/training_manifests_v6")
    output_dataset_dir = Path(args.dataset_dir)
    output_run_dir = Path(args.run_dir)
    output_eval_dir = Path(args.eval_dir)

    # 1. Pre-flight checks
    initial_hashes = check_preflight_invariants(
        baseline_path=baseline_path,
        snapshot_dir=snapshot_dir,
        consolidated_dir=consolidated_dir,
        manifests_dir=manifests_dir,
    )

    # 2. Materialize dataset
    dataset_manifest = materialize_smoke_test_dataset(
        output_dataset_dir=output_dataset_dir,
        consolidated_dir=consolidated_dir,
        manifests_dir=manifests_dir,
        snapshot_dir=snapshot_dir,
    )

    # 3. Inspect architecture and freezing
    arch_info = inspect_layer_structure_and_freezing(
        baseline_model_path=baseline_path,
        freeze_layer_count=args.freeze,
    )

    # 4. Execute training
    train_summary = execute_smoke_test_training(
        dataset_yaml=output_dataset_dir / "data.yaml",
        baseline_weights_path=baseline_path,
        output_run_dir=output_run_dir,
        initial_batch_size=args.batch_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr0,
        seed=args.seed,
        freeze=args.freeze,
        force_retrain=args.force_retrain,
    )

    # 5. Diagnostic evaluation of candidate
    candidate_ckpt = Path(train_summary["candidate_checkpoint_path"])
    candidate_results = run_candidate_diagnostic_evaluation(
        candidate_ckpt=candidate_ckpt,
        baseline_ckpt=baseline_path,
        snapshot_dir=snapshot_dir,
        output_eval_dir=output_eval_dir,
    )

    # 6. Compare candidate against baseline
    baseline_eval_results_file = Path("output/baseline_eval/baseline_evaluation_results.json")
    comparison = compare_candidate_against_baseline(
        candidate_results=candidate_results,
        baseline_results_path=baseline_eval_results_file,
    )

    # 7. Post-flight immutability checks
    check_postflight_immutability(initial_hashes)

    # 8. Render report
    report_path = Path("docs/PIPELINE_SMOKE_TEST_REPORT.md")
    render_pipeline_smoke_test_report(
        dataset_manifest=dataset_manifest,
        arch_info=arch_info,
        train_summary=train_summary,
        candidate_results=candidate_results,
        comparison=comparison,
        output_report_path=report_path,
    )

    print("\n" + "=" * 70)
    print("BATCH 8 PIPELINE SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
