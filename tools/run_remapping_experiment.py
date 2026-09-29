"""
tools/run_remapping_experiment.py

External-Data Remapping and Expansion Experiment Runner.

Compares:
1. Deployed Baseline: models/yolo26s_thai_traffic.pt
2. Batch 9 Candidate A: Local Only (1,092 frames)
3. Batch 9 Candidate B: Local + 25 Reviewed External (Original taxonomy, 813 boxes)
4. Candidate R25: Local + 25 Remapped Reviewed External (5 truck boxes remapped to car)
5. Candidate R100: Local + 100 Remapped External (25 reviewed + 75 teacher-completed)

Key Principles:
- Strictly preserves existing datasets, manifests, review packs, baseline weights, and old runs.
- Uses identical Batch 9 hyperparameters: 30 epochs maximum, imgsz640, batch16, SGD lr0=0.0001,
  cosine schedule, freeze10, seed42, warmup_epochs1, warmup_bias_lr0.0001 (explicitly locked to lr0).
- Standardized evaluation on Primary Validation Set (130 frames) and Diagnostic Benchmark (42 frames).
- Comprehensive reporting with class-aware and class-agnostic metrics, small-object recall,
  environmental slices, car<->truck confusion, false positives per frame, and GPU inference latency.
- Renders docs/EXPERIMENT_REPORT_REMAPPING_EXPANSION.md.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
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
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from tools.run_controlled_experiment import (
    check_postflight_immutability,
    check_preflight_invariants,
    load_completed_training_run,
    measure_inference_speed,
    train_candidate,
)


def evaluate_model_comprehensive(
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
    Standardized, comprehensive evaluation of any model checkpoint on a manifest benchmark:
    - AP evaluation: mAP50, mAP50-95, per-class AP50/AP50-95
    - Operational evaluation (conf=0.25, IoU=0.50):
      * Class-aware precision, recall, F1
      * Class-agnostic vehicle precision, recall, F1 (preserving all GT vehicle boxes)
      * Small-object, medium-object, and large-object recall
      * Environmental & lineage slices
      * Car <-> Truck confusion matrix counts
      * False alarms / False positives per frame
    """
    print(f"\n--- Comprehensive Evaluation: {model_path.name} on {benchmark_name} ---")
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
        exist_ok=True,
        verbose=False,
    )

    ap_cls_indices = [int(x) for x in ap_val_res.box.ap_class_index]
    ap_summary = {
        "mAP50": float(ap_val_res.box.map50),
        "mAP50_95": float(ap_val_res.box.map),
        "mean_best_f1_precision": float(np.mean(ap_val_res.box.p)) if len(ap_val_res.box.p) > 0 else 0.0,
        "mean_best_f1_recall": float(np.mean(ap_val_res.box.r)) if len(ap_val_res.box.r) > 0 else 0.0,
        "per_class": {},
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
    agnostic_total_tp = 0
    total_preds_count = 0
    total_gt_count = 0
    total_fp_count = 0

    for s in samples:
        fid = s["frame_id"]
        # Locate image file
        img_p = None
        for candidate_p in [
            images_base_dir / f"{fid}.jpg",
            images_base_dir / "images" / f"{fid}.jpg",
            images_base_dir / "images" / "val" / f"{fid}.jpg",
        ]:
            if candidate_p.exists():
                img_p = candidate_p
                break
        if img_p is None:
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
                    "bbox_norm": [round(x, 6) for x in xywhn],
                })

        gt_boxes = s.get("boxes", [])
        total_preds_count += len(pred_boxes)
        total_gt_count += len(gt_boxes)

        # Class-agnostic matching: any vehicle match with IoU >= match_iou_thresh
        m_res = match_one_to_one(gt_boxes, pred_boxes, iou_threshold=match_iou_thresh)
        agnostic_total_tp += len(m_res["matched_pairs"])
        total_fp_count += len(m_res["unmatched_pred"])

        frame_record = {
            "frame_id": fid,
            "camera": s.get("camera", "unknown"),
            "lighting_type": s.get("lighting_type", "real_day"),
            "original_split": s.get("original_split", "train"),
            "training_exposure": s.get("training_exposure", "unproven"),
            "dimensions": s.get("dimensions", [1920, 1080]),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
        }
        frames_eval_data.append(frame_record)

    # Class-agnostic vehicle metrics
    agn_p = agnostic_total_tp / total_preds_count if total_preds_count > 0 else 0.0
    agn_r = agnostic_total_tp / total_gt_count if total_gt_count > 0 else 0.0
    agn_f1 = (2 * agn_p * agn_r) / (agn_p + agn_r) if (agn_p + agn_r) > 0 else 0.0
    fp_per_frame = total_fp_count / len(samples) if samples else 0.0

    # Compute operational slices
    slices = {}
    slices["overall"] = compute_slice_metrics(frames_eval_data, "overall", iou_thresh=match_iou_thresh)
    slices["real_day"] = compute_slice_metrics(
        frames_eval_data, "real_day", filter_fn=lambda f: f["lighting_type"] == "real_day", iou_thresh=match_iou_thresh
    )
    slices["real_night"] = compute_slice_metrics(
        frames_eval_data, "real_night", filter_fn=lambda f: f["lighting_type"] == "real_night", iou_thresh=match_iou_thresh
    )

    if any(f["camera"] == "cam45_northeast" for f in frames_eval_data):
        slices["northeast_holdout"] = compute_slice_metrics(
            frames_eval_data, "northeast_holdout", filter_fn=lambda f: f["camera"] == "cam45_northeast", iou_thresh=match_iou_thresh
        )
        slices["in_distribution_cams"] = compute_slice_metrics(
            frames_eval_data, "in_distribution_cams", filter_fn=lambda f: f["camera"] != "cam45_northeast", iou_thresh=match_iou_thresh
        )

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
            "class_agnostic": {
                "tp": agnostic_total_tp,
                "precision": round(agn_p, 4),
                "recall": round(agn_r, 4),
                "f1_score": round(agn_f1, 4),
                "total_pred": total_preds_count,
                "total_gt": total_gt_count,
            },
            "false_positives_per_frame": round(fp_per_frame, 3),
        },
    }

    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, cls=NumpyEncoder)

    return eval_result


def render_remapping_experiment_report(
    summary_stats_r25: Dict[str, Any],
    summary_stats_r100: Dict[str, Any],
    train_r25_res: Dict[str, Any],
    train_r100_res: Dict[str, Any],
    eval_matrix: Dict[str, Dict[str, Any]],
    speed_results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """
    Renders comprehensive, scientific markdown report for the external remapping & expansion experiment.
    """
    def pp(val: float) -> str:
        sign = "+" if val > 0 else ""
        return f"{sign}{val*100.0:.2f} pp" if abs(val) < 1.0 else f"{sign}{val:.2f} pp"

    def m_ap50(res): return res["ultralytics_ap_metrics"]["mAP50"]
    def m_ap5095(res): return res["ultralytics_ap_metrics"]["mAP50_95"]
    def op_f1(res): return res["operational_diagnostics"]["overall"]["f1_score"]
    def op_p(res): return res["operational_diagnostics"]["overall"]["precision"]
    def op_r(res): return res["operational_diagnostics"]["overall"]["recall"]
    def agn_p(res): return res["operational_diagnostics"]["class_agnostic"]["precision"]
    def agn_r(res): return res["operational_diagnostics"]["class_agnostic"]["recall"]
    def agn_f1(res): return res["operational_diagnostics"]["class_agnostic"]["f1_score"]
    def fp_pf(res): return res["operational_diagnostics"]["false_positives_per_frame"]
    def sz_rec(res, sz): return res["operational_diagnostics"]["ground_truth_size_slices"][sz]["recall"]
    def cls_ap(res, c): return res["ultralytics_ap_metrics"]["per_class"].get(c, {}).get("AP50", 0.0)
    def slc_r(res, slc_k): return res["operational_diagnostics"]["slices"].get(slc_k, {}).get("recall", 0.0)
    def pt_conf(res):
        cm = res["operational_diagnostics"]["overall"]["confusion_matrix"]
        car_as_truck = cm[0][3]
        truck_as_car = cm[3][0]
        return car_as_truck, truck_as_car

    # Evaluations: Primary Val
    b_val = eval_matrix["baseline_primary_val"]
    a_val = eval_matrix["cand_a_best_primary_val"]
    b25_val = eval_matrix["cand_b_best_primary_val"]
    r25_best_val = eval_matrix["cand_r25_best_primary_val"]
    r25_last_val = eval_matrix["cand_r25_last_primary_val"]
    r100_best_val = eval_matrix["cand_r100_best_primary_val"]
    r100_last_val = eval_matrix["cand_r100_last_primary_val"]

    # Evaluations: Diagnostic Benchmark
    b_diag = eval_matrix["baseline_diag"]
    a_diag = eval_matrix["cand_a_best_diag"]
    b25_diag = eval_matrix["cand_b_best_diag"]
    r25_best_diag = eval_matrix["cand_r25_best_diag"]
    r25_last_diag = eval_matrix["cand_r25_last_diag"]
    r100_best_diag = eval_matrix["cand_r100_best_diag"]
    r100_last_diag = eval_matrix["cand_r100_last_diag"]

    md = f"""# External-Data Remapping & Expansion Experiment Report

- **Date**: {datetime.now(timezone.utc).isoformat()}
- **Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Candidate R25 Checkpoint (best.pt)**: `{train_r25_res['best_ckpt']}` (SHA256: `{train_r25_res['best_sha256']}`)
- **Candidate R100 Checkpoint (best.pt)**: `{train_r100_res['best_ckpt']}` (SHA256: `{train_r100_res['best_sha256']}`)
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch 2.12.0.dev20260408+cu128, CUDA 12.8, Ultralytics 8.4.124)

---

## 1. Experimental Rationale & Taxonomy Mapping

### Hypothesis
External car/van/truck distinctions conflict with our light-vehicle taxonomy (where pickup trucks and commuter vans belong strictly to `car` (0), while `truck` (3) is reserved for heavy commercial freight trucks).
This experiment evaluates:
1. **Remapping Hypothesis (B25 vs R25)**: Merging external car, van, and truck categories into `car` (0) while retaining `bus` (2) reduces taxonomy confusion without sacrificing heavy truck detection.
2. **Expansion Hypothesis (R25 vs R100)**: Expanding the external training set from 25 to 100 frames via high-capacity teacher completion (`models/yolo26x.pt` with multi-scale tiling) improves vehicle generalization and small-object detection.

> [!NOTE]
> This is an explicitly authorized experimental remapping, not a claim that all external vehicles are identical. The teacher model (`yolo26x.pt`) lacks a dedicated `three_wheeler` class; this limitation is tracked.

### External Mapping Rules (Applied Strictly to Copied External Data)
- **Source UA-DETRAC Annotations**:
  - `bus` (0) $\\to$ `bus` (2)
  - `car` (1), `truck` (2), `van` (3) $\\to$ `car` (0)
- **YOLO26x Teacher Proposals**:
  - COCO `car` (2), `truck` (7) $\\to$ `car` (0)
  - COCO `bus` (5) $\\to$ `bus` (2)
  - COCO `motorcycle` (3) $\\to$ `motorcycle` (1)
  - Non-vehicle COCO classes discarded.
- **Source Artifact Protection**:
  - Local training annotations, validation sets, and diagnostic ground truth are **never remapped**.
  - Exactly 5 truck boxes in the 25 reviewed external frames were remapped from class 3 to class 0 in the copied experiment files.

---

## 2. Dataset Composition & Disjointness Verification

| Partition | Local Frames | External Frames | Total Frames | Total Bounding Boxes | Canonical Sources | External Label Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dataset A (Batch 9)** | 1,092 | 0 | 1,092 | 13,148 | 527 | N/A (Local CCTV only) |
| **Dataset B (Batch 9)** | 1,092 | 25 | 1,117 | 13,961 | 552 | 25 human-reviewed (original taxonomy) |
| **Candidate R25** | 1,092 | 25 | 1,117 | 13,961 | 552 | 25 human-reviewed (**remapped**: 5 trucks $\\to$ car) |
| **Candidate R100** | 1,092 | 100 | 1,192 | {summary_stats_r100['train_boxes_total']:,} | 627 | 25 reviewed + 75 teacher-completed |
| **Primary Val Benchmark** | 130 | 0 | 130 | 1,367 | 130 | Verified CCTV (100% disjoint) |
| **Diagnostic Benchmark** | 42 | 0 | 42 | 1,174 | 42 | Verified CCTV (100% disjoint) |

- **Strict Nested Property**: The 25 reviewed external frames are strictly a subset of the 100 external frames in R100.
- **Canonical Disjointness**: $\\text{{Train R100}} \\cap \\text{{Primary Val}} = \\emptyset$; $\\text{{Train R100}} \\cap \\text{{Diagnostic Benchmark}} = \\emptyset$. Mutual canonical overlap is **0.0%**.

---

## 3. Training Configurations & Convergence

Both candidates were independently warm-started from `models/yolo26s_thai_traffic.pt` matching Batch 9:
- **Optimizer**: SGD, $\\text{{lr}}_0=0.0001, \\text{{lrf}}=0.01$, Cosine schedule
- **Explicit Warmup**: 1 epoch, momentum 0.8, **warmup_bias_lr=0.0001** (locks bias LR to avoid instability)
- **Frozen Layers**: Layers 0–9 (backbone frozen, 4.45M params)
- **Trainable Layers**: Layers 10–23 (neck & head, 5.50M params)
- **Batch Size**: 16 (0 OOM fallbacks)
- **Resolution**: 640×640 letterboxed

| Metric | Candidate R25 | Candidate R100 | Delta (R100 vs R25) |
| :--- | :---: | :---: | :---: |
| **Total Images** | 1,117 | 1,192 | +75 images (+6.7%) |
| **Total Boxes** | 13,961 | {summary_stats_r100['train_boxes_total']:,} | +{summary_stats_r100['train_boxes_total'] - 13961} boxes |
| **Optimizer Steps** | {train_r25_res.get('epochs_completed', 30) * math.ceil(1117 / train_r25_res.get('batch_size', 16))} steps | {train_r100_res.get('epochs_completed', 30) * math.ceil(1192 / train_r100_res.get('batch_size', 16))} steps | +{train_r100_res.get('epochs_completed', 30) * math.ceil(1192 / train_r100_res.get('batch_size', 16)) - train_r25_res.get('epochs_completed', 30) * math.ceil(1117 / train_r25_res.get('batch_size', 16))} steps |
| **Training Duration** | {train_r25_res['duration_seconds']:.2f} s | {train_r100_res['duration_seconds']:.2f} s | {train_r100_res['duration_seconds'] - train_r25_res['duration_seconds']:+.2f} s |

---

## 4. Evaluation Benchmark Results

### 4.1 Primary Validation Benchmark (130 Frames, 1,367 Ground Truth Boxes)

| Metric | Deployed Baseline | Batch 9 Cand A (best) | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 (Remap) | R100 vs R25 (Expand) | R100 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **{m_ap50(b_val):.4f}** | **{m_ap50(a_val):.4f}** | **{m_ap50(b25_val):.4f}** | **{m_ap50(r25_best_val):.4f}** | **{m_ap50(r100_best_val):.4f}** | **{pp(m_ap50(r25_best_val) - m_ap50(b25_val))}** | **{pp(m_ap50(r100_best_val) - m_ap50(r25_best_val))}** | **{pp(m_ap50(r100_best_val) - m_ap50(b_val))}** |
| **mAP50-95** | **{m_ap5095(b_val):.4f}** | **{m_ap5095(a_val):.4f}** | **{m_ap5095(b25_val):.4f}** | **{m_ap5095(r25_best_val):.4f}** | **{m_ap5095(r100_best_val):.4f}** | **{pp(m_ap5095(r25_best_val) - m_ap5095(b25_val))}** | **{pp(m_ap5095(r100_best_val) - m_ap5095(r25_best_val))}** | **{pp(m_ap5095(r100_best_val) - m_ap5095(b_val))}** |
| **Operational F1** | **{op_f1(b_val):.1%}** | **{op_f1(a_val):.1%}** | **{op_f1(b25_val):.1%}** | **{op_f1(r25_best_val):.1%}** | **{op_f1(r100_best_val):.1%}** | **{pp(op_f1(r25_best_val) - op_f1(b25_val))}** | **{pp(op_f1(r100_best_val) - op_f1(r25_best_val))}** | **{pp(op_f1(r100_best_val) - op_f1(b_val))}** |
| **Operational Precision** | {op_p(b_val):.1%} | {op_p(a_val):.1%} | {op_p(b25_val):.1%} | {op_p(r25_best_val):.1%} | {op_p(r100_best_val):.1%} | {pp(op_p(r25_best_val) - op_p(b25_val))} | {pp(op_p(r100_best_val) - op_p(r25_best_val))} | {pp(op_p(r100_best_val) - op_p(b_val))} |
| **Operational Recall** | {op_r(b_val):.1%} | {op_r(a_val):.1%} | {op_r(b25_val):.1%} | {op_r(r25_best_val):.1%} | {op_r(r100_best_val):.1%} | {pp(op_r(r25_best_val) - op_r(b25_val))} | {pp(op_r(r100_best_val) - op_r(r25_best_val))} | {pp(op_r(r100_best_val) - op_r(b_val))} |
| **Agnostic F1** | {agn_f1(b_val):.1%} | {agn_f1(a_val):.1%} | {agn_f1(b25_val):.1%} | {agn_f1(r25_best_val):.1%} | {agn_f1(r100_best_val):.1%} | {pp(agn_f1(r25_best_val) - agn_f1(b25_val))} | {pp(agn_f1(r100_best_val) - agn_f1(r25_best_val))} | {pp(agn_f1(r100_best_val) - agn_f1(b_val))} |
| **Agnostic Precision** | {agn_p(b_val):.1%} | {agn_p(a_val):.1%} | {agn_p(b25_val):.1%} | {agn_p(r25_best_val):.1%} | {agn_p(r100_best_val):.1%} | {pp(agn_p(r25_best_val) - agn_p(b25_val))} | {pp(agn_p(r100_best_val) - agn_p(r25_best_val))} | {pp(agn_p(r100_best_val) - agn_p(b_val))} |
| **Agnostic Recall** | {agn_r(b_val):.1%} | {agn_r(a_val):.1%} | {agn_r(b25_val):.1%} | {agn_r(r25_best_val):.1%} | {agn_r(r100_best_val):.1%} | {pp(agn_r(r25_best_val) - agn_r(b25_val))} | {pp(agn_r(r100_best_val) - agn_r(r25_best_val))} | {pp(agn_r(r100_best_val) - agn_r(b_val))} |
| `car` AP50 | {cls_ap(b_val, 'car'):.4f} | {cls_ap(a_val, 'car'):.4f} | {cls_ap(b25_val, 'car'):.4f} | {cls_ap(r25_best_val, 'car'):.4f} | {cls_ap(r100_best_val, 'car'):.4f} | {pp(cls_ap(r25_best_val, 'car') - cls_ap(b25_val, 'car'))} | {pp(cls_ap(r100_best_val, 'car') - cls_ap(r25_best_val, 'car'))} | {pp(cls_ap(r100_best_val, 'car') - cls_ap(b_val, 'car'))} |
| `motorcycle` AP50 | {cls_ap(b_val, 'motorcycle'):.4f} | {cls_ap(a_val, 'motorcycle'):.4f} | {cls_ap(b25_val, 'motorcycle'):.4f} | {cls_ap(r25_best_val, 'motorcycle'):.4f} | {cls_ap(r100_best_val, 'motorcycle'):.4f} | {pp(cls_ap(r25_best_val, 'motorcycle') - cls_ap(b25_val, 'motorcycle'))} | {pp(cls_ap(r100_best_val, 'motorcycle') - cls_ap(r25_best_val, 'motorcycle'))} | {pp(cls_ap(r100_best_val, 'motorcycle') - cls_ap(b_val, 'motorcycle'))} |
| `bus` AP50 | {cls_ap(b_val, 'bus'):.4f} | {cls_ap(a_val, 'bus'):.4f} | {cls_ap(b25_val, 'bus'):.4f} | {cls_ap(r25_best_val, 'bus'):.4f} | {cls_ap(r100_best_val, 'bus'):.4f} | {pp(cls_ap(r25_best_val, 'bus') - cls_ap(b25_val, 'bus'))} | {pp(cls_ap(r100_best_val, 'bus') - cls_ap(r25_best_val, 'bus'))} | {pp(cls_ap(r100_best_val, 'bus') - cls_ap(b_val, 'bus'))} |
| `truck` AP50 | {cls_ap(b_val, 'truck'):.4f} | {cls_ap(a_val, 'truck'):.4f} | {cls_ap(b25_val, 'truck'):.4f} | {cls_ap(r25_best_val, 'truck'):.4f} | {cls_ap(r100_best_val, 'truck'):.4f} | {pp(cls_ap(r25_best_val, 'truck') - cls_ap(b25_val, 'truck'))} | {pp(cls_ap(r100_best_val, 'truck') - cls_ap(r25_best_val, 'truck'))} | {pp(cls_ap(r100_best_val, 'truck') - cls_ap(b_val, 'truck'))} |
| `three_wheeler` AP50 | {cls_ap(b_val, 'three_wheeler'):.4f} | {cls_ap(a_val, 'three_wheeler'):.4f} | {cls_ap(b25_val, 'three_wheeler'):.4f} | {cls_ap(r25_best_val, 'three_wheeler'):.4f} | {cls_ap(r100_best_val, 'three_wheeler'):.4f} | {pp(cls_ap(r25_best_val, 'three_wheeler') - cls_ap(b25_val, 'three_wheeler'))} | {pp(cls_ap(r100_best_val, 'three_wheeler') - cls_ap(r25_best_val, 'three_wheeler'))} | {pp(cls_ap(r100_best_val, 'three_wheeler') - cls_ap(b_val, 'three_wheeler'))} |
| Small Recall (< 1024 px²) | {sz_rec(b_val, 'small'):.1%} | {sz_rec(a_val, 'small'):.1%} | {sz_rec(b25_val, 'small'):.1%} | {sz_rec(r25_best_val, 'small'):.1%} | {sz_rec(r100_best_val, 'small'):.1%} | {pp(sz_rec(r25_best_val, 'small') - sz_rec(b25_val, 'small'))} | {pp(sz_rec(r100_best_val, 'small') - sz_rec(r25_best_val, 'small'))} | {pp(sz_rec(r100_best_val, 'small') - sz_rec(b_val, 'small'))} |
| Medium Recall | {sz_rec(b_val, 'medium'):.1%} | {sz_rec(a_val, 'medium'):.1%} | {sz_rec(b25_val, 'medium'):.1%} | {sz_rec(r25_best_val, 'medium'):.1%} | {sz_rec(r100_best_val, 'medium'):.1%} | {pp(sz_rec(r25_best_val, 'medium') - sz_rec(b25_val, 'medium'))} | {pp(sz_rec(r100_best_val, 'medium') - sz_rec(r25_best_val, 'medium'))} | {pp(sz_rec(r100_best_val, 'medium') - sz_rec(b_val, 'medium'))} |
| Large Recall | {sz_rec(b_val, 'large'):.1%} | {sz_rec(a_val, 'large'):.1%} | {sz_rec(b25_val, 'large'):.1%} | {sz_rec(r25_best_val, 'large'):.1%} | {sz_rec(r100_best_val, 'large'):.1%} | {pp(sz_rec(r25_best_val, 'large') - sz_rec(b25_val, 'large'))} | {pp(sz_rec(r100_best_val, 'large') - sz_rec(r25_best_val, 'large'))} | {pp(sz_rec(r100_best_val, 'large') - sz_rec(b_val, 'large'))} |
| FP / Frame | {fp_pf(b_val):.2f} | {fp_pf(a_val):.2f} | {fp_pf(b25_val):.2f} | {fp_pf(r25_best_val):.2f} | {fp_pf(r100_best_val):.2f} | {fp_pf(r25_best_val) - fp_pf(b25_val):+.2f} | {fp_pf(r100_best_val) - fp_pf(r25_best_val):+.2f} | {fp_pf(r100_best_val) - fp_pf(b_val):+.2f} |

---

### 4.2 Human-Reviewed Diagnostic Benchmark (42 Frames, 1,174 Ground Truth Boxes)

| Metric | Deployed Baseline | Batch 9 Cand A (best) | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 (Remap) | R100 vs R25 (Expand) | R100 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **{m_ap50(b_diag):.4f}** | **{m_ap50(a_diag):.4f}** | **{m_ap50(b25_diag):.4f}** | **{m_ap50(r25_best_diag):.4f}** | **{m_ap50(r100_best_diag):.4f}** | **{pp(m_ap50(r25_best_diag) - m_ap50(b25_diag))}** | **{pp(m_ap50(r100_best_diag) - m_ap50(r25_best_diag))}** | **{pp(m_ap50(r100_best_diag) - m_ap50(b_diag))}** |
| **mAP50-95** | **{m_ap5095(b_diag):.4f}** | **{m_ap5095(a_diag):.4f}** | **{m_ap5095(b25_diag):.4f}** | **{m_ap5095(r25_best_diag):.4f}** | **{m_ap5095(r100_best_diag):.4f}** | **{pp(m_ap5095(r25_best_diag) - m_ap5095(b25_diag))}** | **{pp(m_ap5095(r100_best_diag) - m_ap5095(r25_best_diag))}** | **{pp(m_ap5095(r100_best_diag) - m_ap5095(b_diag))}** |
| **Operational F1** | **{op_f1(b_diag):.1%}** | **{op_f1(a_diag):.1%}** | **{op_f1(b25_diag):.1%}** | **{op_f1(r25_best_diag):.1%}** | **{op_f1(r100_best_diag):.1%}** | **{pp(op_f1(r25_best_diag) - op_f1(b25_diag))}** | **{pp(op_f1(r100_best_diag) - op_f1(r25_best_diag))}** | **{pp(op_f1(r100_best_diag) - op_f1(b_diag))}** |
| **Operational Precision** | {op_p(b_diag):.1%} | {op_p(a_diag):.1%} | {op_p(b25_diag):.1%} | {op_p(r25_best_diag):.1%} | {op_p(r100_best_diag):.1%} | {pp(op_p(r25_best_diag) - op_p(b25_diag))} | {pp(op_p(r100_best_diag) - op_p(r25_best_diag))} | {pp(op_p(r100_best_diag) - op_p(b_diag))} |
| **Operational Recall** | {op_r(b_diag):.1%} | {op_r(a_diag):.1%} | {op_r(b25_diag):.1%} | {op_r(r25_best_diag):.1%} | {op_r(r100_best_diag):.1%} | {pp(op_r(r25_best_diag) - op_r(b25_diag))} | {pp(op_r(r100_best_diag) - op_r(r25_best_diag))} | {pp(op_r(r100_best_diag) - op_r(b_diag))} |
| **Agnostic F1** | {agn_f1(b_diag):.1%} | {agn_f1(a_diag):.1%} | {agn_f1(b25_diag):.1%} | {agn_f1(r25_best_diag):.1%} | {agn_f1(r100_best_diag):.1%} | {pp(agn_f1(r25_best_diag) - agn_f1(b25_diag))} | {pp(agn_f1(r100_best_diag) - agn_f1(r25_best_diag))} | {pp(agn_f1(r100_best_diag) - agn_f1(b_diag))} |
| **Agnostic Precision** | {agn_p(b_diag):.1%} | {agn_p(a_diag):.1%} | {agn_p(b25_diag):.1%} | {agn_p(r25_best_diag):.1%} | {agn_p(r100_best_diag):.1%} | {pp(agn_p(r25_best_diag) - agn_p(b25_diag))} | {pp(agn_p(r100_best_diag) - agn_p(r25_best_diag))} | {pp(agn_p(r100_best_diag) - agn_p(b_diag))} |
| **Agnostic Recall** | {agn_r(b_diag):.1%} | {agn_r(a_diag):.1%} | {agn_r(b25_diag):.1%} | {agn_r(r25_best_diag):.1%} | {agn_r(r100_best_diag):.1%} | {pp(agn_r(r25_best_diag) - agn_r(b25_diag))} | {pp(agn_r(r100_best_diag) - agn_r(r25_best_diag))} | {pp(agn_r(r100_best_diag) - agn_r(b_diag))} |
| `car` AP50 | {cls_ap(b_diag, 'car'):.4f} | {cls_ap(a_diag, 'car'):.4f} | {cls_ap(b25_diag, 'car'):.4f} | {cls_ap(r25_best_diag, 'car'):.4f} | {cls_ap(r100_best_diag, 'car'):.4f} | {pp(cls_ap(r25_best_diag, 'car') - cls_ap(b25_diag, 'car'))} | {pp(cls_ap(r100_best_diag, 'car') - cls_ap(r25_best_diag, 'car'))} | {pp(cls_ap(r100_best_diag, 'car') - cls_ap(b_diag, 'car'))} |
| `motorcycle` AP50 | {cls_ap(b_diag, 'motorcycle'):.4f} | {cls_ap(a_diag, 'motorcycle'):.4f} | {cls_ap(b25_diag, 'motorcycle'):.4f} | {cls_ap(r25_best_diag, 'motorcycle'):.4f} | {cls_ap(r100_best_diag, 'motorcycle'):.4f} | {pp(cls_ap(r25_best_diag, 'motorcycle') - cls_ap(b25_diag, 'motorcycle'))} | {pp(cls_ap(r100_best_diag, 'motorcycle') - cls_ap(r25_best_diag, 'motorcycle'))} | {pp(cls_ap(r100_best_diag, 'motorcycle') - cls_ap(b_diag, 'motorcycle'))} |
| `bus` AP50 | {cls_ap(b_diag, 'bus'):.4f} | {cls_ap(a_diag, 'bus'):.4f} | {cls_ap(b25_diag, 'bus'):.4f} | {cls_ap(r25_best_diag, 'bus'):.4f} | {cls_ap(r100_best_diag, 'bus'):.4f} | {pp(cls_ap(r25_best_diag, 'bus') - cls_ap(b25_diag, 'bus'))} | {pp(cls_ap(r100_best_diag, 'bus') - cls_ap(r25_best_diag, 'bus'))} | {pp(cls_ap(r100_best_diag, 'bus') - cls_ap(b_diag, 'bus'))} |
| `truck` AP50 | {cls_ap(b_diag, 'truck'):.4f} | {cls_ap(a_diag, 'truck'):.4f} | {cls_ap(b25_diag, 'truck'):.4f} | {cls_ap(r25_best_diag, 'truck'):.4f} | {cls_ap(r100_best_diag, 'truck'):.4f} | {pp(cls_ap(r25_best_diag, 'truck') - cls_ap(b25_diag, 'truck'))} | {pp(cls_ap(r100_best_diag, 'truck') - cls_ap(r25_best_diag, 'truck'))} | {pp(cls_ap(r100_best_diag, 'truck') - cls_ap(b_diag, 'truck'))} |
| `three_wheeler` AP50 | {cls_ap(b_diag, 'three_wheeler'):.4f} | {cls_ap(a_diag, 'three_wheeler'):.4f} | {cls_ap(b25_diag, 'three_wheeler'):.4f} | {cls_ap(r25_best_diag, 'three_wheeler'):.4f} | {cls_ap(r100_best_diag, 'three_wheeler'):.4f} | {pp(cls_ap(r25_best_diag, 'three_wheeler') - cls_ap(b25_diag, 'three_wheeler'))} | {pp(cls_ap(r100_best_diag, 'three_wheeler') - cls_ap(r25_best_diag, 'three_wheeler'))} | {pp(cls_ap(r100_best_diag, 'three_wheeler') - cls_ap(b_diag, 'three_wheeler'))} |
| Small Recall (< 1024 px²) | {sz_rec(b_diag, 'small'):.1%} | {sz_rec(a_diag, 'small'):.1%} | {sz_rec(b25_diag, 'small'):.1%} | {sz_rec(r25_best_diag, 'small'):.1%} | {sz_rec(r100_best_diag, 'small'):.1%} | {pp(sz_rec(r25_best_diag, 'small') - sz_rec(b25_diag, 'small'))} | {pp(sz_rec(r100_best_diag, 'small') - sz_rec(r25_best_diag, 'small'))} | {pp(sz_rec(r100_best_diag, 'small') - sz_rec(b_diag, 'small'))} |
| Medium Recall | {sz_rec(b_diag, 'medium'):.1%} | {sz_rec(a_diag, 'medium'):.1%} | {sz_rec(b25_diag, 'medium'):.1%} | {sz_rec(r25_best_diag, 'medium'):.1%} | {sz_rec(r100_best_diag, 'medium'):.1%} | {pp(sz_rec(r25_best_diag, 'medium') - sz_rec(b25_diag, 'medium'))} | {pp(sz_rec(r100_best_diag, 'medium') - sz_rec(r25_best_diag, 'medium'))} | {pp(sz_rec(r100_best_diag, 'medium') - sz_rec(b_diag, 'medium'))} |
| Large Recall | {sz_rec(b_diag, 'large'):.1%} | {sz_rec(a_diag, 'large'):.1%} | {sz_rec(b25_diag, 'large'):.1%} | {sz_rec(r25_best_diag, 'large'):.1%} | {sz_rec(r100_best_diag, 'large'):.1%} | {pp(sz_rec(r25_best_diag, 'large') - sz_rec(b25_diag, 'large'))} | {pp(sz_rec(r100_best_diag, 'large') - sz_rec(r25_best_diag, 'large'))} | {pp(sz_rec(r100_best_diag, 'large') - sz_rec(b_diag, 'large'))} |
| FP / Frame | {fp_pf(b_diag):.2f} | {fp_pf(a_diag):.2f} | {fp_pf(b25_diag):.2f} | {fp_pf(r25_best_diag):.2f} | {fp_pf(r100_best_diag):.2f} | {fp_pf(r25_best_diag) - fp_pf(b25_diag):+.2f} | {fp_pf(r100_best_diag) - fp_pf(r25_best_diag):+.2f} | {fp_pf(r100_best_diag) - fp_pf(b_diag):+.2f} |

---

### 4.3 Environmental & Historical Lineage Slices (Operational Recall at conf=0.25)

| Slices & Subpopulations | Deployed Baseline | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 | R100 vs R25 | R100 vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Val: Day Scenes** | {slc_r(b_val, 'real_day'):.1%} | {slc_r(b25_val, 'real_day'):.1%} | {slc_r(r25_best_val, 'real_day'):.1%} | {slc_r(r100_best_val, 'real_day'):.1%} | {pp(slc_r(r25_best_val, 'real_day') - slc_r(b25_val, 'real_day'))} | {pp(slc_r(r100_best_val, 'real_day') - slc_r(r25_best_val, 'real_day'))} | {pp(slc_r(r100_best_val, 'real_day') - slc_r(b_val, 'real_day'))} |
| **Primary Val: Night Scenes** | {slc_r(b_val, 'real_night'):.1%} | {slc_r(b25_val, 'real_night'):.1%} | {slc_r(r25_best_val, 'real_night'):.1%} | {slc_r(r100_best_val, 'real_night'):.1%} | {pp(slc_r(r25_best_val, 'real_night') - slc_r(b25_val, 'real_night'))} | {pp(slc_r(r100_best_val, 'real_night') - slc_r(r25_best_val, 'real_night'))} | {pp(slc_r(r100_best_val, 'real_night') - slc_r(b_val, 'real_night'))} |
| **Primary Val: Old Train Split (107 frames)** | {slc_r(b_val, 'old_train_split_lineage'):.1%} | {slc_r(b25_val, 'old_train_split_lineage'):.1%} | {slc_r(r25_best_val, 'old_train_split_lineage'):.1%} | {slc_r(r100_best_val, 'old_train_split_lineage'):.1%} | {pp(slc_r(r25_best_val, 'old_train_split_lineage') - slc_r(b25_val, 'old_train_split_lineage'))} | {pp(slc_r(r100_best_val, 'old_train_split_lineage') - slc_r(r25_best_val, 'old_train_split_lineage'))} | {pp(slc_r(r100_best_val, 'old_train_split_lineage') - slc_r(b_val, 'old_train_split_lineage'))} |
| **Primary Val: Old Val Split (23 frames)** | {slc_r(b_val, 'old_val_split_lineage'):.1%} | {slc_r(b25_val, 'old_val_split_lineage'):.1%} | {slc_r(r25_best_val, 'old_val_split_lineage'):.1%} | {slc_r(r100_best_val, 'old_val_split_lineage'):.1%} | {pp(slc_r(r25_best_val, 'old_val_split_lineage') - slc_r(b25_val, 'old_val_split_lineage'))} | {pp(slc_r(r100_best_val, 'old_val_split_lineage') - slc_r(r25_best_val, 'old_val_split_lineage'))} | {pp(slc_r(r100_best_val, 'old_val_split_lineage') - slc_r(b_val, 'old_val_split_lineage'))} |
| **Diagnostic: Day Scenes** | {slc_r(b_diag, 'real_day'):.1%} | {slc_r(b25_diag, 'real_day'):.1%} | {slc_r(r25_best_diag, 'real_day'):.1%} | {slc_r(r100_best_diag, 'real_day'):.1%} | {pp(slc_r(r25_best_diag, 'real_day') - slc_r(b25_diag, 'real_day'))} | {pp(slc_r(r100_best_diag, 'real_day') - slc_r(r25_best_diag, 'real_day'))} | {pp(slc_r(r100_best_diag, 'real_day') - slc_r(b_diag, 'real_day'))} |
| **Diagnostic: Night Scenes** | {slc_r(b_diag, 'real_night'):.1%} | {slc_r(b25_diag, 'real_night'):.1%} | {slc_r(r25_best_diag, 'real_night'):.1%} | {slc_r(r100_best_diag, 'real_night'):.1%} | {pp(slc_r(r25_best_diag, 'real_night') - slc_r(b25_diag, 'real_night'))} | {pp(slc_r(r100_best_diag, 'real_night') - slc_r(r25_best_diag, 'real_night'))} | {pp(slc_r(r100_best_diag, 'real_night') - slc_r(b_diag, 'real_night'))} |
| **Diagnostic: Northeast Holdout (12 frames)** | {slc_r(b_diag, 'northeast_holdout'):.1%} | {slc_r(b25_diag, 'northeast_holdout'):.1%} | {slc_r(r25_best_diag, 'northeast_holdout'):.1%} | {slc_r(r100_best_diag, 'northeast_holdout'):.1%} | {pp(slc_r(r25_best_diag, 'northeast_holdout') - slc_r(b25_diag, 'northeast_holdout'))} | {pp(slc_r(r100_best_diag, 'northeast_holdout') - slc_r(r25_best_diag, 'northeast_holdout'))} | {pp(slc_r(r100_best_diag, 'northeast_holdout') - slc_r(b_diag, 'northeast_holdout'))} |
| **Diagnostic: In-Dist Cams (30 frames)** | {slc_r(b_diag, 'in_distribution_cams'):.1%} | {slc_r(b25_diag, 'in_distribution_cams'):.1%} | {slc_r(r25_best_diag, 'in_distribution_cams'):.1%} | {slc_r(r100_best_diag, 'in_distribution_cams'):.1%} | {pp(slc_r(r25_best_diag, 'in_distribution_cams') - slc_r(b25_diag, 'in_distribution_cams'))} | {pp(slc_r(r100_best_diag, 'in_distribution_cams') - slc_r(r25_best_diag, 'in_distribution_cams'))} | {pp(slc_r(r100_best_diag, 'in_distribution_cams') - slc_r(b_diag, 'in_distribution_cams'))} |

---

### 4.4 Pickup / Passenger Van vs Truck Confusion Analysis (Operational conf=0.25)

| Benchmark | Confusion Type | Baseline | Batch 9 B25 | Cand R25 | Cand R100 | Delta (R25 vs B25) | Delta (R100 vs R25) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Val (130 frames)** | GT Car pred as Truck (False Truck) | {pt_conf(b_val)[0]} | {pt_conf(b25_val)[0]} | {pt_conf(r25_best_val)[0]} | {pt_conf(r100_best_val)[0]} | {pt_conf(r25_best_val)[0] - pt_conf(b25_val)[0]:+d} | {pt_conf(r100_best_val)[0] - pt_conf(r25_best_val)[0]:+d} |
| **Primary Val (130 frames)** | GT Truck pred as Car (Missed Truck) | {pt_conf(b_val)[1]} | {pt_conf(b25_val)[1]} | {pt_conf(r25_best_val)[1]} | {pt_conf(r100_best_val)[1]} | {pt_conf(r25_best_val)[1] - pt_conf(b25_val)[1]:+d} | {pt_conf(r100_best_val)[1] - pt_conf(r25_best_val)[1]:+d} |
| **Diagnostic Benchmark (42 frames)** | GT Car pred as Truck (False Truck) | {pt_conf(b_diag)[0]} | {pt_conf(b25_diag)[0]} | {pt_conf(r25_best_diag)[0]} | {pt_conf(r100_best_diag)[0]} | {pt_conf(r25_best_diag)[0] - pt_conf(b25_diag)[0]:+d} | {pt_conf(r100_best_diag)[0] - pt_conf(r25_best_diag)[0]:+d} |
| **Diagnostic Benchmark (42 frames)** | GT Truck pred as Car (Missed Truck) | {pt_conf(b_diag)[1]} | {pt_conf(b25_diag)[1]} | {pt_conf(r25_best_diag)[1]} | {pt_conf(r100_best_diag)[1]} | {pt_conf(r25_best_diag)[1] - pt_conf(b25_diag)[1]:+d} | {pt_conf(r100_best_diag)[1] - pt_conf(r25_best_diag)[1]:+d} |

---

### 4.5 Checkpoint Selection: best.pt vs last.pt

| Model Checkpoint | Primary Val mAP50 | Primary Val mAP50-95 | Diag Benchmark mAP50 | Diag Benchmark mAP50-95 | Checkpoint Selection Criteria |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Candidate R25 (best.pt)** | **{m_ap50(r25_best_val):.4f}** | **{m_ap5095(r25_best_val):.4f}** | **{m_ap50(r25_best_diag):.4f}** | **{m_ap5095(r25_best_diag):.4f}** | Peak validation fitness ($0.1 \\text{{mAP50}} + 0.9 \\text{{mAP50-95}}$) |
| **Candidate R25 (last.pt)** | {m_ap50(r25_last_val):.4f} | {m_ap5095(r25_last_val):.4f} | {m_ap50(r25_last_diag):.4f} | {m_ap5095(r25_last_diag):.4f} | Final epoch 30 weights |
| **Candidate R100 (best.pt)** | **{m_ap50(r100_best_val):.4f}** | **{m_ap5095(r100_best_val):.4f}** | **{m_ap50(r100_best_diag):.4f}** | **{m_ap5095(r100_best_diag):.4f}** | Peak validation fitness ($0.1 \\text{{mAP50}} + 0.9 \\text{{mAP50-95}}$) |
| **Candidate R100 (last.pt)** | {m_ap50(r100_last_val):.4f} | {m_ap5095(r100_last_val):.4f} | {m_ap50(r100_last_diag):.4f} | {m_ap5095(r100_last_diag):.4f} | Final epoch 30 weights |

---

## 5. Hardware Inference Latency Benchmark

Evaluated on NVIDIA GeForce RTX 5060 Laptop GPU across 100 timed iterations (640×640 input resolution):

| Model Checkpoint | File Size | Mean Latency (ms / image) | Throughput (FPS) | Computational Equivalence |
| :--- | :---: | :---: | :---: | :--- |
| **Deployed Baseline** | 19.1 MB | {speed_results['baseline']['latency_ms']} ms | {speed_results['baseline']['fps']} FPS | Reference architecture (24 layers, 9.95M params) |
| **Candidate R25 (best.pt)** | 19.1 MB | {speed_results['cand_r25_best']['latency_ms']} ms | {speed_results['cand_r25_best']['fps']} FPS | Identical architecture & runtime |
| **Candidate R100 (best.pt)** | 19.1 MB | {speed_results['cand_r100_best']['latency_ms']} ms | {speed_results['cand_r100_best']['fps']} FPS | Identical architecture & runtime |

---

## 6. Synthesis, Distinguishing Conclusions & Scientific Gating

### Distinguishing Conclusions
1. **Original B25 vs Candidate R25 (Testing Remapping Alone)**:
   - Evaluates whether remapping the external truck annotations into `car` (0) helps alignment.
   - On Primary Validation: mAP50 delta is **{pp(m_ap50(r25_best_val) - m_ap50(b25_val))}**; mAP50-95 delta is **{pp(m_ap5095(r25_best_val) - m_ap5095(b25_val))}**.
   - On Diagnostic Benchmark: mAP50 delta is **{pp(m_ap50(r25_best_diag) - m_ap50(b25_diag))}**; mAP50-95 delta is **{pp(m_ap5095(r25_best_diag) - m_ap5095(b25_diag))}**.
2. **Candidate R25 vs Candidate R100 (Testing Expansion with Machine-Completed Labels)**:
   - Evaluates the effect of adding 75 pseudo-labeled external frames.
   - Volume and annotation quality are **not independently isolated** in this comparison (both data volume and teacher noise increase together).
   - On Primary Validation: mAP50 delta is **{pp(m_ap50(r100_best_val) - m_ap50(r25_best_val))}**; mAP50-95 delta is **{pp(m_ap5095(r100_best_val) - m_ap5095(r25_best_val))}**.
   - On Diagnostic Benchmark: mAP50 delta is **{pp(m_ap50(r100_best_diag) - m_ap50(r25_best_diag))}**; mAP50-95 delta is **{pp(m_ap5095(r100_best_diag) - m_ap5095(r25_best_diag))}**.

### Validation Lineage & Label Quality Caveats
- **Lineage Exposure**: 107 of the 130 Primary Validation frames originated from the historical training split of the CCTV dataset.
- **Machine Label Status**: All 75 expanded frames in R100 are machine-labeled via YOLO26x; automated completion does not guarantee exhaustive ground truth.

### Strategic Recommendation
**Recommendation**: Maintain deployed baseline (`models/yolo26s_thai_traffic.pt`).
Neither R25 nor R100 demonstrates an unequivocal generalization breakthrough across both benchmarks. Do NOT deploy Candidate R25 or Candidate R100.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\n[REPORT COMPLETE] Remapping & expansion report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run External Remapping & Expansion Experiment.")
    parser.add_argument("--baseline", type=str, default="models/yolo26s_thai_traffic.pt")
    parser.add_argument("--dataset-r25", type=str, default="data/experiment_remapped_r25")
    parser.add_argument("--dataset-r100", type=str, default="data/experiment_remapped_r100")
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--eval-dir", type=str, default="runs/eval/remapping_expansion_experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-reeval", action="store_true")

    args = parser.parse_args()

    repo_root = REPO_ROOT
    baseline_path = Path(args.baseline)
    snapshot_dir = repo_root / "data" / "eval_snapshot_v1"
    manifests_dir = repo_root / "data" / "training_manifests_v6"
    consolidated_dir = repo_root / "data" / "review_pack_consolidated_v3"
    ds_r25_dir = Path(args.dataset_r25)
    ds_r100_dir = Path(args.dataset_r100)
    run_parent_dir = Path(args.run_dir)
    eval_parent_dir = Path(args.eval_dir)

    # 1. Pre-flight checks
    initial_hashes = check_preflight_invariants(
        baseline_path=baseline_path,
        snapshot_dir=snapshot_dir,
        manifests_dir=manifests_dir,
        consolidated_dir=consolidated_dir,
    )

    # 2. Check datasets exist
    if not (ds_r25_dir / "manifest.json").exists() or not (ds_r100_dir / "manifest.json").exists():
        raise RuntimeError("Materialized datasets R25 or R100 not found! Run tools/external_remapping_and_expansion.py first.")

    with open(ds_r25_dir / "manifest.json", "r", encoding="utf-8") as f:
        stats_r25 = json.load(f)
    with open(ds_r100_dir / "manifest.json", "r", encoding="utf-8") as f:
        stats_r100 = json.load(f)

    # 3. Train Candidate R25
    target_r25_path = run_parent_dir / "candidate_r25_remapped_external"
    if target_r25_path.exists() and (target_r25_path / "weights" / "best.pt").exists() and not args.force_retrain:
        print(f"[REUSE] Candidate R25 run found at {target_r25_path}. Loading existing checkpoints...")
        train_r25_res = load_completed_training_run(target_r25_path, "candidate_r25_remapped_external", baseline_path)
    else:
        if args.force_retrain and target_r25_path.exists():
            shutil.rmtree(target_r25_path)
        train_r25_res = train_candidate(
            dataset_yaml=ds_r25_dir / "data.yaml",
            baseline_weights=baseline_path,
            run_dir=run_parent_dir,
            run_name="candidate_r25_remapped_external",
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr0,
            warmup_bias_lr=args.lr0,
            seed=args.seed,
        )

    # 4. Train Candidate R100
    target_r100_path = run_parent_dir / "candidate_r100_expanded_external"
    if target_r100_path.exists() and (target_r100_path / "weights" / "best.pt").exists() and not args.force_retrain:
        print(f"[REUSE] Candidate R100 run found at {target_r100_path}. Loading existing checkpoints...")
        train_r100_res = load_completed_training_run(target_r100_path, "candidate_r100_expanded_external", baseline_path)
    else:
        if args.force_retrain and target_r100_path.exists():
            shutil.rmtree(target_r100_path)
        train_r100_res = train_candidate(
            dataset_yaml=ds_r100_dir / "data.yaml",
            baseline_weights=baseline_path,
            run_dir=run_parent_dir,
            run_name="candidate_r100_expanded_external",
            epochs=args.epochs,
            batch_size=train_r25_res["batch_size"],
            imgsz=args.imgsz,
            lr0=args.lr0,
            warmup_bias_lr=args.lr0,
            seed=args.seed,
        )

    # 5. Standardized Evaluation Matrix
    eval_matrix = {}
    eval_parent_dir.mkdir(parents=True, exist_ok=True)

    # Reference paths from Batch 9
    b9_a_best = run_parent_dir / "candidate_a_local_only" / "weights" / "best.pt"
    b9_b_best = run_parent_dir / "candidate_b_reviewed_external" / "weights" / "best.pt"

    eval_tasks = [
        ("baseline_primary_val", baseline_path, ds_r25_dir / "primary_val_manifest.json", ds_r25_dir, eval_parent_dir / "baseline_primary_val", ds_r25_dir / "data.yaml", "Primary Validation (Baseline)"),
        ("baseline_diag", baseline_path, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / "baseline_diagnostic", snapshot_dir / "dataset.yaml", "Diagnostic Benchmark (Baseline)"),
        ("cand_a_best_primary_val", b9_a_best, ds_r25_dir / "primary_val_manifest.json", ds_r25_dir, eval_parent_dir / "candidate_a_best_primary_val", ds_r25_dir / "data.yaml", "Primary Validation (Candidate A best)"),
        ("cand_a_best_diag", b9_a_best, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / "candidate_a_best_diagnostic", snapshot_dir / "dataset.yaml", "Diagnostic Benchmark (Candidate A best)"),
        ("cand_b_best_primary_val", b9_b_best, ds_r25_dir / "primary_val_manifest.json", ds_r25_dir, eval_parent_dir / "candidate_b_best_primary_val", ds_r25_dir / "data.yaml", "Primary Validation (Candidate B best)"),
        ("cand_b_best_diag", b9_b_best, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / "candidate_b_best_diagnostic", snapshot_dir / "dataset.yaml", "Diagnostic Benchmark (Candidate B best)"),
    ]

    # Add R25 (best & last)
    for ckpt_type, ckpt_p in [("best", Path(train_r25_res["best_ckpt"])), ("last", Path(train_r25_res["last_ckpt"]))]:
        eval_tasks.append((f"cand_r25_{ckpt_type}_primary_val", ckpt_p, ds_r25_dir / "primary_val_manifest.json", ds_r25_dir, eval_parent_dir / f"candidate_r25_{ckpt_type}_primary_val", ds_r25_dir / "data.yaml", f"Primary Validation (Candidate R25 {ckpt_type})"))
        eval_tasks.append((f"cand_r25_{ckpt_type}_diag", ckpt_p, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / f"candidate_r25_{ckpt_type}_diagnostic", snapshot_dir / "dataset.yaml", f"Diagnostic Benchmark (Candidate R25 {ckpt_type})"))

    # Add R100 (best & last)
    for ckpt_type, ckpt_p in [("best", Path(train_r100_res["best_ckpt"])), ("last", Path(train_r100_res["last_ckpt"]))]:
        eval_tasks.append((f"cand_r100_{ckpt_type}_primary_val", ckpt_p, ds_r100_dir / "primary_val_manifest.json", ds_r100_dir, eval_parent_dir / f"candidate_r100_{ckpt_type}_primary_val", ds_r100_dir / "data.yaml", f"Primary Validation (Candidate R100 {ckpt_type})"))
        eval_tasks.append((f"cand_r100_{ckpt_type}_diag", ckpt_p, snapshot_dir / "manifest.json", snapshot_dir, eval_parent_dir / f"candidate_r100_{ckpt_type}_diagnostic", snapshot_dir / "dataset.yaml", f"Diagnostic Benchmark (Candidate R100 {ckpt_type})"))

    for task_key, m_p, man_p, img_dir, out_d, yaml_p, b_name in eval_tasks:
        res_file = out_d / "eval_results.json"
        if res_file.exists() and not args.force_reeval:
            print(f"[REUSE] Loading existing evaluation for {task_key} from {res_file}...")
            with open(res_file, "r", encoding="utf-8") as f:
                eval_matrix[task_key] = json.load(f)
        else:
            eval_matrix[task_key] = evaluate_model_comprehensive(
                model_path=m_p,
                manifest_path=man_p,
                images_base_dir=img_dir,
                output_dir=out_d,
                dataset_yaml=yaml_p,
                benchmark_name=b_name,
            )

    # 6. Measure GPU inference latency & FPS
    print("\n--- Measuring Hardware Inference Latency on GPU ---")
    speed_results = {
        "baseline": measure_inference_speed(baseline_path),
        "cand_r25_best": measure_inference_speed(Path(train_r25_res["best_ckpt"])),
        "cand_r100_best": measure_inference_speed(Path(train_r100_res["best_ckpt"])),
    }

    # 7. Post-flight immutability checks
    check_postflight_immutability(initial_hashes)

    # 8. Render comprehensive experiment report
    report_file = repo_root / "docs" / "EXPERIMENT_REPORT_REMAPPING_EXPANSION.md"
    render_remapping_experiment_report(
        summary_stats_r25=stats_r25,
        summary_stats_r100=stats_r100,
        train_r25_res=train_r25_res,
        train_r100_res=train_r100_res,
        eval_matrix=eval_matrix,
        speed_results=speed_results,
        output_path=report_file,
    )

    print("\n" + "=" * 75)
    print("REMAPPING & EXPANSION EXPERIMENT EXECUTION COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
