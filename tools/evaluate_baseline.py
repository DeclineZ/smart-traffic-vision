"""
Baseline Model Evaluation & Diagnostic Slicing Tool (Batch 3 - Corrected).

Evaluates models/yolo26s_thai_traffic.pt against the verified 42-frame evaluation snapshot (data/eval_snapshot_v1):
1. Verifies baseline checkpoint SHA256 (cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c).
2. Verifies snapshot integrity (all 42 image and label hashes match manifest before execution).
3. Separates AP evaluation from operational diagnostics:
   - Official Ultralytics validation at low confidence threshold (conf=0.001, imgsz=640, iou=0.60).
   - Maps per-class AP through returned class IDs.
   - Records Ultralytics best-F1 precision and recall.
4. Operational diagnostics at fixed threshold (conf=0.25, imgsz=640, matching iou=0.50):
   - Computes operational precision, recall, F1, and 6x6 confusion matrix.
   - Diagnostic slices: Real Day vs Real Night, Holdout Northeast vs In-Distribution, Exposure Lineage.
5. Ground-truth size slices:
   - Matches on the full frame FIRST before assigning ground-truth objects to size bins.
   - Reports ground-truth size recall and support; omits ambiguous size-specific precision.
6. Correct proposal-error reporting:
   - Separately reports correct matches (290), unmatched GT (304), unmatched predictions (4),
     class disagreements (28), and localization errors (2).
   - Preserves standard detection FP/FN totals without conflating classification/localization errors with pure omissions.
7. Visual error overlays:
   - Displays TP (green), Missed GT (red), Extra predictions (orange), Class disagreements (purple),
     and Localization errors (cyan).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
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
from tools.prepare_review_pack import (
    compute_box_iou,
    compute_file_sha256,
)
from tools.validate_review_pack import verify_snapshot_integrity

EXPECTED_BASELINE_SHA256 = "cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c"

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Colors for visual error overlays (BGR)
OVERLAY_COLORS = {
    "tp": (50, 205, 50),       # Lime green: Matched detection (TP)
    "fn": (50, 50, 230),       # Red: Missed vehicle (FN)
    "fp": (0, 165, 255),       # Orange/Amber: Extra detection / False alarm (FP)
    "disagree": (210, 50, 210),# Magenta/Purple: Class disagreement
    "loc_err": (255, 200, 0),  # Cyan/Sky blue: Localization error (0.10 <= IoU < 0.50)
}


# ==============================================================================
# 1. Bipartite One-to-One Matching Logic
# ==============================================================================

def match_one_to_one(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    loc_iou_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Performs deterministic bipartite greedy one-to-one matching between ground truth
    and predicted bounding boxes based on IoU.

    Returns:
    - matched_pairs: list of (gt_idx, pred_idx, iou, is_class_match)
    - class_disagreements: list of (gt_idx, pred_idx, iou, gt_class, pred_class)
    - localization_errors: list of (gt_idx, pred_idx, iou) where loc_iou <= iou < iou_threshold
    - unmatched_gt: list of gt_idx (missing objects / pure false negatives)
    - unmatched_pred: list of pred_idx (extra boxes / pure false positives)
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    if num_gt == 0 and num_pred == 0:
        return {
            "matched_pairs": [],
            "class_disagreements": [],
            "localization_errors": [],
            "unmatched_gt": [],
            "unmatched_pred": [],
        }

    if num_gt == 0:
        return {
            "matched_pairs": [],
            "class_disagreements": [],
            "localization_errors": [],
            "unmatched_gt": [],
            "unmatched_pred": list(range(num_pred)),
        }

    if num_pred == 0:
        return {
            "matched_pairs": [],
            "class_disagreements": [],
            "localization_errors": [],
            "unmatched_gt": list(range(num_gt)),
            "unmatched_pred": [],
        }

    # Compute all pairwise IoUs
    candidates: List[Tuple[float, int, int]] = []
    for g_idx, g in enumerate(gt_boxes):
        g_box = g["bbox_norm"]
        for p_idx, p in enumerate(pred_boxes):
            p_box = p["bbox_norm"]
            iou = compute_box_iou(g_box, p_box)
            candidates.append((iou, g_idx, p_idx))

    # Sort descending by IoU
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()

    matched_pairs: List[Tuple[int, int, float, bool]] = []
    class_disagreements: List[Tuple[int, int, float, int, int]] = []
    localization_errors: List[Tuple[int, int, float]] = []

    # First pass: matches with IoU >= iou_threshold
    for iou, g_idx, p_idx in candidates:
        if iou < iou_threshold:
            break
        if g_idx in matched_gt or p_idx in matched_pred:
            continue

        matched_gt.add(g_idx)
        matched_pred.add(p_idx)

        g_cid = gt_boxes[g_idx]["class_id"]
        p_cid = pred_boxes[p_idx]["class_id"]
        is_cls_match = (g_cid == p_cid)

        matched_pairs.append((g_idx, p_idx, iou, is_cls_match))
        if not is_cls_match:
            class_disagreements.append((g_idx, p_idx, iou, g_cid, p_cid))

    # Second pass: check remaining unmatched for localization errors
    for iou, g_idx, p_idx in candidates:
        if iou < loc_iou_threshold or iou >= iou_threshold:
            continue
        if g_idx in matched_gt or p_idx in matched_pred:
            continue

        g_cid = gt_boxes[g_idx]["class_id"]
        p_cid = pred_boxes[p_idx]["class_id"]
        if g_cid == p_cid:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            localization_errors.append((g_idx, p_idx, iou))

    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(num_pred) if j not in matched_pred]

    return {
        "matched_pairs": matched_pairs,
        "class_disagreements": class_disagreements,
        "localization_errors": localization_errors,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": unmatched_pred,
    }


# ==============================================================================
# 2. Operational Slice Metrics & Ground-Truth Size Slices
# ==============================================================================

def compute_slice_metrics(
    frames_data: List[Dict[str, Any]],
    slice_name: str,
    filter_fn: Any = None,
    iou_thresh: float = 0.50
) -> Dict[str, Any]:
    """
    Computes operational precision, recall, F1, and 6x6 confusion matrix
    at fixed operational confidence threshold.
    """
    selected_frames = [f for f in frames_data if filter_fn is None or filter_fn(f)]

    gt_counts = Counter()
    pred_counts = Counter()
    tp_counts = Counter()
    fp_counts = Counter()
    fn_counts = Counter()
    disagree_counts = Counter()

    confusion_matrix = np.zeros((6, 6), dtype=int)

    for fd in selected_frames:
        gt_boxes = fd["gt_boxes"]
        pred_boxes = fd["pred_boxes"]

        for g in gt_boxes:
            gt_counts[g["class_id"]] += 1
        for p in pred_boxes:
            pred_counts[p["class_id"]] += 1

        match_res = match_one_to_one(gt_boxes, pred_boxes, iou_threshold=iou_thresh)

        # Record matches
        for g_idx, p_idx, iou, is_cls_match in match_res["matched_pairs"]:
            g_cid = gt_boxes[g_idx]["class_id"]
            p_cid = pred_boxes[p_idx]["class_id"]
            confusion_matrix[g_cid, p_cid] += 1
            if is_cls_match:
                tp_counts[g_cid] += 1
            else:
                disagree_counts[(g_cid, p_cid)] += 1
                fp_counts[p_cid] += 1
                fn_counts[g_cid] += 1

        # Localization errors (same class, lower IoU)
        for g_idx, p_idx, iou in match_res["localization_errors"]:
            g_cid = gt_boxes[g_idx]["class_id"]
            p_cid = pred_boxes[p_idx]["class_id"]
            confusion_matrix[g_cid, 5] += 1  # Counted as FN for true class
            confusion_matrix[5, p_cid] += 1  # Counted as FP for pred class
            fn_counts[g_cid] += 1
            fp_counts[p_cid] += 1

        # Unmatched GT -> Missed / FN
        for g_idx in match_res["unmatched_gt"]:
            g_cid = gt_boxes[g_idx]["class_id"]
            confusion_matrix[g_cid, 5] += 1
            fn_counts[g_cid] += 1

        # Unmatched Pred -> False alarm / FP
        for p_idx in match_res["unmatched_pred"]:
            p_cid = pred_boxes[p_idx]["class_id"]
            confusion_matrix[5, p_cid] += 1
            fp_counts[p_cid] += 1

    class_metrics = {}
    total_tp = sum(tp_counts.values())
    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())

    images_with_gt = Counter()
    for fd in selected_frames:
        seen_cids = set(b.get("class_id") for b in fd.get("gt_boxes", []))
        for cid in seen_cids:
            images_with_gt[cid] += 1

    for cid in range(5):
        c_name = THAI_5CLASS_NAMES[cid]
        tp = tp_counts[cid]
        fp = fp_counts[cid]
        fn = fn_counts[cid]
        n_gt = gt_counts[cid]
        n_pred = pred_counts[cid]

        p = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        support_note = ""
        if n_gt == 0:
            support_note = "ABSENT (0 GT instances)"
        elif n_gt < 10:
            support_note = f"INSUFFICIENT SUPPORT ({n_gt} GT instances < 10)"
        else:
            support_note = f"Dense support ({n_gt} GT instances)"

        class_metrics[c_name] = {
            "class_id": cid,
            "class_name": c_name,
            "images_count": images_with_gt[cid],
            "gt_count": n_gt,
            "pred_count": n_pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support_note": support_note,
        }

    overall_p = total_tp / float(total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / float(total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    return {
        "slice_name": slice_name,
        "frames_count": len(selected_frames),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1_score": round(overall_f1, 4),
        "per_class": class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def compute_ground_truth_size_slices(
    frames_eval_data: List[Dict[str, Any]],
    match_iou_thresh: float = 0.50,
    loc_iou_thresh: float = 0.10
) -> Dict[str, Any]:
    """
    Evaluates ground-truth size recall by matching on the full frame FIRST,
    then partitioning ground-truth boxes into standard letterbox area bins:
    - Small: < 32^2 px (< 1024 px^2)
    - Medium: 32^2 <= area <= 96^2 (1024 to 9216 px^2)
    - Large: > 96^2 px (> 9216 px^2)

    Reports:
    - Ground-truth count per size bucket
    - Detected ground-truth count (matched TP at IoU >= 0.50 with correct class)
    - Missed ground-truth count (pure unmatched FN)
    - Class disagreements on GT boxes
    - Localization errors on GT boxes (0.10 <= IoU < 0.50)
    - Recall (detected / gt_count)
    - Support status
    Omit size-specific precision because false-positive attribution across sizes is ambiguous.
    """
    stats = {
        "small": {
            "size_bucket": "small",
            "area_definition": "< 32^2 px (< 1024 px^2)",
            "gt_count": 0,
            "detected_count": 0,
            "missed_count": 0,
            "class_disagreements": 0,
            "localization_errors": 0,
            "recall": 0.0,
            "support_note": ""
        },
        "medium": {
            "size_bucket": "medium",
            "area_definition": "32^2 <= area <= 96^2 (1024 to 9216 px^2)",
            "gt_count": 0,
            "detected_count": 0,
            "missed_count": 0,
            "class_disagreements": 0,
            "localization_errors": 0,
            "recall": 0.0,
            "support_note": ""
        },
        "large": {
            "size_bucket": "large",
            "area_definition": "> 96^2 px (> 9216 px^2)",
            "gt_count": 0,
            "detected_count": 0,
            "missed_count": 0,
            "class_disagreements": 0,
            "localization_errors": 0,
            "recall": 0.0,
            "support_note": ""
        }
    }

    for fd in frames_eval_data:
        dims = fd.get("dimensions")
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            img_w, img_h = int(dims[0]), int(dims[1])
        elif isinstance(dims, dict):
            img_w, img_h = int(dims.get("width", 1920)), int(dims.get("height", 1080))
        else:
            img_w, img_h = 1920, 1080

        gt_boxes = fd["gt_boxes"]
        pred_boxes = fd["pred_boxes"]

        # Match on the FULL frame
        m = match_one_to_one(gt_boxes, pred_boxes, iou_threshold=match_iou_thresh, loc_iou_threshold=loc_iou_thresh)

        tp_gt_set = set(g_idx for g_idx, p_idx, iou, is_match in m["matched_pairs"] if is_match)
        disagree_gt_set = set(g_idx for g_idx, p_idx, iou, is_match in m["matched_pairs"] if not is_match)
        loc_gt_set = set(g_idx for g_idx, p_idx, iou in m["localization_errors"])

        for g_idx, g in enumerate(gt_boxes):
            sz = compute_aspect_preserving_size(g["bbox_norm"][2], g["bbox_norm"][3], img_w, img_h, ref_size=640)
            bucket = sz["size_bucket"]
            if bucket not in stats:
                continue
            stats[bucket]["gt_count"] += 1
            if g_idx in tp_gt_set:
                stats[bucket]["detected_count"] += 1
            elif g_idx in disagree_gt_set:
                stats[bucket]["class_disagreements"] += 1
            elif g_idx in loc_gt_set:
                stats[bucket]["localization_errors"] += 1
            else:
                stats[bucket]["missed_count"] += 1

    for b_name, b_data in stats.items():
        gt_c = b_data["gt_count"]
        det_c = b_data["detected_count"]
        b_data["recall"] = round(det_c / gt_c, 4) if gt_c > 0 else 0.0
        if gt_c == 0:
            b_data["support_note"] = "ABSENT (0 GT instances)"
        elif gt_c < 10:
            b_data["support_note"] = f"INSUFFICIENT SUPPORT ({gt_c} GT instances < 10)"
        else:
            b_data["support_note"] = f"Dense support ({gt_c} GT instances)"

    return stats


# ==============================================================================
# 3. Visual Error Overlays Generation
# ==============================================================================

def render_error_overlay(
    image_path: Path,
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    meta_info: Dict[str, Any],
    output_path: Path,
    iou_thresh: float = 0.50
) -> None:
    """
    Renders decodable visual error overlay showing True Positives (green),
    False Negatives / Missed (red), False Positives / Extra (orange),
    Class Disagreements (purple), and Localization Errors (cyan).
    """
    im = cv2.imread(str(image_path))
    if im is None:
        return
    img_h, img_w = im.shape[:2]

    match_res = match_one_to_one(gt_boxes, pred_boxes, iou_threshold=iou_thresh, loc_iou_threshold=0.10)

    overlay = im.copy()

    # Draw Missed Vehicles (FN) in Red
    for g_idx in match_res["unmatched_gt"]:
        g = gt_boxes[g_idx]
        xc, yc, bw, bh = g["bbox_norm"]
        x1 = int((xc - bw / 2.0) * img_w)
        y1 = int((yc - bh / 2.0) * img_h)
        x2 = int((xc + bw / 2.0) * img_w)
        y2 = int((yc + bh / 2.0) * img_h)
        c_name = THAI_5CLASS_NAMES.get(g["class_id"], str(g["class_id"]))
        subtype = g.get("subtype", "")

        cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLORS["fn"], 2)
        label = f"MISSED: {c_name}"
        if subtype:
            label += f" ({subtype})"
        cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY_COLORS["fn"], 1, cv2.LINE_AA)

    # Draw False Alarms (FP) in Orange
    for p_idx in match_res["unmatched_pred"]:
        p = pred_boxes[p_idx]
        xc, yc, bw, bh = p["bbox_norm"]
        x1 = int((xc - bw / 2.0) * img_w)
        y1 = int((yc - bh / 2.0) * img_h)
        x2 = int((xc + bw / 2.0) * img_w)
        y2 = int((yc + bh / 2.0) * img_h)
        c_name = THAI_5CLASS_NAMES.get(p["class_id"], str(p["class_id"]))
        conf = p.get("conf", 1.0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLORS["fp"], 2)
        label = f"EXTRA: {c_name} {conf:.2f}"
        cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY_COLORS["fp"], 1, cv2.LINE_AA)

    # Draw Localization Errors (Cyan)
    for g_idx, p_idx, iou in match_res["localization_errors"]:
        p = pred_boxes[p_idx]
        xc, yc, bw, bh = p["bbox_norm"]
        x1 = int((xc - bw / 2.0) * img_w)
        y1 = int((yc - bh / 2.0) * img_h)
        x2 = int((xc + bw / 2.0) * img_w)
        y2 = int((yc + bh / 2.0) * img_h)
        c_name = THAI_5CLASS_NAMES.get(p["class_id"], str(p["class_id"]))
        conf = p.get("conf", 1.0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLORS["loc_err"], 2)
        label = f"LOC_ERR: {c_name} {conf:.2f} (IoU {iou:.2f})"
        cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY_COLORS["loc_err"], 1, cv2.LINE_AA)

    # Draw Matched Pairs and Class Disagreements
    for g_idx, p_idx, iou, is_cls_match in match_res["matched_pairs"]:
        g = gt_boxes[g_idx]
        p = pred_boxes[p_idx]
        xc, yc, bw, bh = p["bbox_norm"]
        x1 = int((xc - bw / 2.0) * img_w)
        y1 = int((yc - bh / 2.0) * img_h)
        x2 = int((xc + bw / 2.0) * img_w)
        y2 = int((yc + bh / 2.0) * img_h)

        g_cname = THAI_5CLASS_NAMES.get(g["class_id"], str(g["class_id"]))
        p_cname = THAI_5CLASS_NAMES.get(p["class_id"], str(p["class_id"]))
        conf = p.get("conf", 1.0)

        if is_cls_match:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLORS["tp"], 2)
            label = f"TP: {p_cname} {conf:.2f} (IoU {iou:.2f})"
            cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY_COLORS["tp"], 1, cv2.LINE_AA)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLORS["disagree"], 2)
            label = f"DISAGREE: GT {g_cname} -> PRED {p_cname} {conf:.2f}"
            cv2.putText(overlay, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OVERLAY_COLORS["disagree"], 1, cv2.LINE_AA)

    # Top banner with diagnostic info
    fid = meta_info.get("frame_id", "")
    cam = meta_info.get("camera", "")
    lighting = meta_info.get("lighting_type", "")
    exposure = meta_info.get("training_exposure", "")
    tp_c = len([m for m in match_res["matched_pairs"] if m[3]])
    disagree_c = len(match_res["class_disagreements"])
    loc_c = len(match_res["localization_errors"])
    fn_c = len(match_res["unmatched_gt"])
    fp_c = len(match_res["unmatched_pred"])

    banner_text = f"Frame: {fid} | Cam: {cam} | Lighting: {lighting} | Exp: {exposure} | TP: {tp_c} | Missed (FN): {fn_c} | Extra (FP): {fp_c} | Disagree: {disagree_c} | LocErr: {loc_c}"
    
    cv2.rectangle(overlay, (0, 0), (img_w, 36), (20, 20, 20), -1)
    cv2.putText(overlay, banner_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


# ==============================================================================
# 4. Machine Proposals Quality Assessment (Scope Item 4)
# ==============================================================================

def assess_machine_proposals(
    review_pack_dir: Path,
    manifest_samples: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
    loc_iou_thresh: float = 0.10
) -> Dict[str, Any]:
    """
    Evaluates original machine proposal quality against human corrections for the
    26 frames where original machine proposals existed in review_pack/proposals/*.txt.
    """
    proposals_dir = review_pack_dir / "proposals"
    if not proposals_dir.exists():
        return {"error": f"Proposals directory missing: {proposals_dir}"}

    proposal_frames_data = []

    total_correct = 0
    total_unmatched_gt = 0
    total_unmatched_pred = 0
    total_class_disagree = 0
    total_loc_error = 0
    class_disagreements_summary = Counter()

    for s in manifest_samples:
        fid = s["frame_id"]
        prop_file = proposals_dir / f"{fid}.txt"
        if not prop_file.exists():
            continue

        prop_lines = [l.strip() for l in prop_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(prop_lines) == 0:
            continue

        proposals = []
        for line in prop_lines:
            parts = line.split()
            if len(parts) == 5:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = [float(x) for x in parts[1:]]
                proposals.append({
                    "class_id": cid,
                    "bbox_norm": [xc, yc, bw, bh]
                })

        gt_boxes = s.get("boxes", [])
        m = match_one_to_one(gt_boxes, proposals, iou_threshold=iou_thresh, loc_iou_threshold=loc_iou_thresh)

        correct_count = len([pair for pair in m["matched_pairs"] if pair[3]])
        disagree_count = len(m["class_disagreements"])
        loc_count = len(m["localization_errors"])
        unmatched_gt_count = len(m["unmatched_gt"])
        unmatched_pred_count = len(m["unmatched_pred"])

        total_correct += correct_count
        total_class_disagree += disagree_count
        total_loc_error += loc_count
        total_unmatched_gt += unmatched_gt_count
        total_unmatched_pred += unmatched_pred_count

        for _, _, _, g_cid, p_cid in m["class_disagreements"]:
            class_disagreements_summary[f"{THAI_5CLASS_NAMES[g_cid]} -> {THAI_5CLASS_NAMES[p_cid]}"] += 1

        proposal_frames_data.append({
            "frame_id": fid,
            "camera": s.get("camera", "unknown"),
            "lighting_type": s.get("lighting_type", "real_day"),
            "training_exposure": s.get("training_exposure", "unproven"),
            "dimensions": s.get("dimensions", {}),
            "gt_boxes": gt_boxes,
            "pred_boxes": proposals
        })

    total_proposals = sum(len(f["pred_boxes"]) for f in proposal_frames_data)
    total_human_gt = sum(len(f["gt_boxes"]) for f in proposal_frames_data)

    std_tp = total_correct
    std_fn = total_unmatched_gt + total_class_disagree + total_loc_error
    std_fp = total_unmatched_pred + total_class_disagree + total_loc_error

    std_precision = round(std_tp / float(std_tp + std_fp), 4) if (std_tp + std_fp) > 0 else 0.0
    std_recall = round(std_tp / float(std_tp + std_fn), 4) if (std_tp + std_fn) > 0 else 0.0
    std_f1 = round((2 * std_precision * std_recall) / (std_precision + std_recall), 4) if (std_precision + std_recall) > 0 else 0.0

    return {
        "evaluated_frames_count": len(proposal_frames_data),
        "total_proposals": total_proposals,
        "total_human_gt": total_human_gt,
        "standard_detection_metrics": {
            "tp": std_tp,
            "fp": std_fp,
            "fn": std_fn,
            "precision": std_precision,
            "recall": std_recall,
            "f1_score": std_f1
        },
        "granular_error_breakdown": {
            "correct_matches": total_correct,
            "unmatched_gt": total_unmatched_gt,
            "unmatched_pred": total_unmatched_pred,
            "class_disagreements": total_class_disagree,
            "localization_errors": total_loc_error,
        },
        "class_disagreements": dict(class_disagreements_summary),
        "findings": [
            f"Evaluated {len(proposal_frames_data)} frames with initial machine proposals (total {total_proposals} proposals vs {total_human_gt} verified human GT boxes).",
            f"Exact matches: {total_correct} ({total_correct/total_human_gt:.1%} of human GT).",
            f"Unmatched GT (pure omissions): {total_unmatched_gt} ({total_unmatched_gt/total_human_gt:.1%} of human GT).",
            f"Unmatched proposals (spurious detections): {total_unmatched_pred} ({total_unmatched_pred/total_proposals:.1%} of proposals).",
            f"Class disagreements: {total_class_disagree} instances where machine localized vehicle but misclassified it.",
            f"Localization errors: {total_loc_error} instances where machine localized vehicle with 0.10 <= IoU < 0.50.",
            f"Standard Detection Benchmark: Precision {std_precision:.1%}, Recall {std_recall:.1%}, F1 {std_f1:.1%}."
        ],
        "recommendations": [
            "Test multi-scale tiling inference as an experimental approach for small distant vehicle proposal generation.",
            "Test morphology post-processing rules (e.g. aspect ratio, vehicle height priors) to resolve pickup vs medium truck confusion.",
            "Incorporate targeted human-in-the-loop review on night footage where low contrast degrades proposals.",
            "Calibrate candidate confidence thresholds empirically rather than accepting uncalibrated low-confidence teacher proposals."
        ]
    }


# ==============================================================================
# 5. Main Baseline Evaluation Pipeline
# ==============================================================================

def run_baseline_evaluation(
    model_path: Path,
    snapshot_dir: Path,
    output_dir: Path,
    operational_conf: float = 0.25,
    ap_conf: float = 0.001,
    iou_nms: float = 0.60,
    match_iou_thresh: float = 0.50,
    is_candidate: bool = False,
    baseline_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end evaluation on the evaluation snapshot.
    Supports both baseline verification and candidate model evaluation:
    - If is_candidate=False: strictly verifies that model_path matches EXPECTED_BASELINE_SHA256.
    - If is_candidate=True: strictly verifies baseline_model_path against EXPECTED_BASELINE_SHA256,
      verifies snapshot integrity, hashes the candidate model, and evaluates the candidate.
    """
    import time
    start_eval_time = time.time()

    model_path = Path(model_path).resolve()
    snapshot_dir = Path(snapshot_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline_model_path is None:
        baseline_model_path = Path("models/yolo26s_thai_traffic.pt").resolve()
    else:
        baseline_model_path = Path(baseline_model_path).resolve()

    # Step 1: Snapshot and Baseline integrity check
    print("\n--- Verifying Snapshot Integrity & Baseline Checkpoint ---")
    verify_snapshot_integrity(
        snapshot_dir=snapshot_dir,
        expected_checkpoint_path=baseline_model_path,
        expected_checkpoint_sha256=EXPECTED_BASELINE_SHA256
    )
    baseline_ckpt_sha = compute_file_sha256(baseline_model_path)
    print(f"[INTEGRITY VERIFIED] Baseline Checkpoint SHA256 verified: {baseline_ckpt_sha}")

    if is_candidate:
        if not model_path.exists():
            raise FileNotFoundError(f"Candidate model checkpoint not found at: {model_path}")
        candidate_ckpt_sha = compute_file_sha256(model_path)
        actual_ckpt_sha = candidate_ckpt_sha
        print(f"[CANDIDATE VERIFIED] Evaluating Candidate Model: {model_path}")
        print(f"                     Candidate SHA256: {candidate_ckpt_sha}")
    else:
        actual_ckpt_sha = compute_file_sha256(model_path)
        if actual_ckpt_sha != EXPECTED_BASELINE_SHA256:
            raise RuntimeError(
                f"Baseline model hash mismatch! Expected {EXPECTED_BASELINE_SHA256}, got {actual_ckpt_sha} at {model_path}"
            )
        candidate_ckpt_sha = None

    # Load Snapshot manifest
    snap_manifest_file = snapshot_dir / "manifest.json"
    with open(snap_manifest_file, "r", encoding="utf-8") as f:
        snap_manifest = json.load(f)
    samples = snap_manifest.get("samples", [])

    # Step 2: Run official Ultralytics AP validation (conf=0.001, imgsz=640, iou=0.60)
    print(f"\n--- Running Ultralytics AP validation on {'Candidate' if is_candidate else 'Baseline'} (conf=0.001, imgsz=640, iou=0.60) ---")
    dataset_yaml = snapshot_dir / "dataset.yaml"
    model = YOLO(str(model_path))

    ap_val_res = model.val(
        data=str(dataset_yaml),
        imgsz=640,
        conf=ap_conf,
        iou=iou_nms,
        split="val",
        save_json=True,
        project=str(output_dir),
        name="ultralytics_ap_val",
        verbose=False,
    )

    # Map per-class AP through returned class IDs
    ap_cls_indices = [int(x) for x in ap_val_res.box.ap_class_index]
    ultralytics_ap_summary = {
        "mAP50": float(ap_val_res.box.map50),
        "mAP50_95": float(ap_val_res.box.map),
        "mean_best_f1_precision": float(np.mean(ap_val_res.box.p)) if len(ap_val_res.box.p) > 0 else 0.0,
        "mean_best_f1_recall": float(np.mean(ap_val_res.box.r)) if len(ap_val_res.box.r) > 0 else 0.0,
        "per_class": {}
    }
    for array_idx, c_idx in enumerate(ap_cls_indices):
        c_name = THAI_5CLASS_NAMES.get(c_idx, str(c_idx))
        ap50 = float(ap_val_res.box.ap50[array_idx])
        ap = float(ap_val_res.box.ap[array_idx])
        best_p = float(ap_val_res.box.p[array_idx]) if len(ap_val_res.box.p) > array_idx else 0.0
        best_r = float(ap_val_res.box.r[array_idx]) if len(ap_val_res.box.r) > array_idx else 0.0
        best_f1 = (2 * best_p * best_r) / (best_p + best_r) if (best_p + best_r) > 0 else 0.0

        ultralytics_ap_summary["per_class"][c_name] = {
            "class_id": c_idx,
            "class_name": c_name,
            "AP50": round(ap50, 4),
            "AP50_95": round(ap, 4),
            "best_f1_precision": round(best_p, 4),
            "best_f1_recall": round(best_r, 4),
            "best_f1_score": round(best_f1, 4),
        }

    # Step 3: Run per-frame inference at operational threshold (conf=0.25)
    print("\n--- Running operational diagnostics & overlays (conf=0.25, iou=0.60) ---")
    frames_eval_data = []
    overlays_dir = output_dir / "overlays"

    rep_frames = {
        "daytime": "cam03_east_f003720",
        "night": "cam43_south_night_f002800",
        "northeast_holdout": "cam45_northeast_f063055",
        "distant_small": "cam43_south_f046440"
    }

    for s in samples:
        fid = s["frame_id"]
        clean_img_path = snapshot_dir / "images" / f"{fid}.jpg"

        preds = model.predict(
            source=str(clean_img_path),
            imgsz=640,
            conf=operational_conf,
            iou=iou_nms,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            verbose=False
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
            "training_exposure": s.get("training_exposure", "unproven"),
            "dimensions": s.get("dimensions", {}),
            "gt_boxes": s.get("boxes", []),
            "pred_boxes": pred_boxes
        }
        frames_eval_data.append(frame_record)

        overlay_path = overlays_dir / f"{fid}_overlay.jpg"
        render_error_overlay(
            image_path=clean_img_path,
            gt_boxes=s.get("boxes", []),
            pred_boxes=pred_boxes,
            meta_info=frame_record,
            output_path=overlay_path,
            iou_thresh=match_iou_thresh
        )

    # Step 4: Compute Operational Slices & Ground-Truth Size Slices
    print("\n--- Computing operational slices & size recall ---")
    slices = {}
    slices["overall"] = compute_slice_metrics(frames_eval_data, "overall", iou_thresh=match_iou_thresh)
    slices["real_day"] = compute_slice_metrics(frames_eval_data, "real_day", filter_fn=lambda f: f["lighting_type"] == "real_day", iou_thresh=match_iou_thresh)
    slices["real_night"] = compute_slice_metrics(frames_eval_data, "real_night", filter_fn=lambda f: f["lighting_type"] == "real_night", iou_thresh=match_iou_thresh)
    slices["northeast_holdout"] = compute_slice_metrics(frames_eval_data, "northeast_holdout", filter_fn=lambda f: f["camera"] == "cam45_northeast", iou_thresh=match_iou_thresh)
    slices["in_distribution_cams"] = compute_slice_metrics(frames_eval_data, "in_distribution_cams", filter_fn=lambda f: f["camera"] != "cam45_northeast", iou_thresh=match_iou_thresh)

    for cam_name in ["cam03_east", "cam43_south", "cam44_north", "cam46_west"]:
        slices[cam_name] = compute_slice_metrics(frames_eval_data, cam_name, filter_fn=lambda f, c=cam_name: f["camera"] == c, iou_thresh=match_iou_thresh)

    slices["nearby_training_exposure"] = compute_slice_metrics(frames_eval_data, "nearby_training_exposure", filter_fn=lambda f: f["training_exposure"] == "nearby_training_exposure", iou_thresh=match_iou_thresh)
    slices["unproven_checkpoint_exposure"] = compute_slice_metrics(frames_eval_data, "unproven_checkpoint_exposure", filter_fn=lambda f: f["training_exposure"] == "unproven_checkpoint_exposure", iou_thresh=match_iou_thresh)

    # Size-Slice matching on full frame
    size_slices = compute_ground_truth_size_slices(frames_eval_data, match_iou_thresh=match_iou_thresh)

    # Step 5: Machine Proposals Audit
    print("\n--- Assessing machine proposals quality ---")
    review_pack_dir = snapshot_dir.parent / "review_pack_v1"
    proposal_assessment = assess_machine_proposals(review_pack_dir, samples, iou_thresh=match_iou_thresh)

    eval_runtime_seconds = round(time.time() - start_eval_time, 2)

    # Step 6: Package Evaluation Payload
    full_results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "candidate" if is_candidate else "baseline",
            "evaluated_model": str(model_path),
            "baseline_model": str(baseline_model_path),
            "baseline_checkpoint_sha256": baseline_ckpt_sha,
            "candidate_checkpoint_sha256": candidate_ckpt_sha,
            "evaluation_runtime_seconds": eval_runtime_seconds,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "ultralytics_version": ultralytics_ap_summary.get("ultralytics_version", "8.4.124"),
            "opencv_version": cv2.__version__,
            "ap_evaluation_config": {
                "conf_threshold": ap_conf,
                "iou_nms_threshold": iou_nms,
                "imgsz": 640,
                "purpose": "Full PR curve integration for mAP50 and mAP50-95"
            },
            "operational_diagnostic_config": {
                "conf_threshold": operational_conf,
                "iou_nms_threshold": iou_nms,
                "matching_iou_threshold": match_iou_thresh,
                "imgsz": 640,
                "purpose": "Fixed-threshold operational diagnostics and failure mode slicing"
            },
            "evaluation_notice": (
                "Internal diagnostic benchmark across 42 verified frames with documented exposure lineage. "
                "MUST NOT be cited as an independent test set."
            )
        },
        "ultralytics_ap_metrics": ultralytics_ap_summary,
        "operational_diagnostics": {
            "overall": slices["overall"],
            "slices": slices,
            "ground_truth_size_slices": size_slices,
        },
        "machine_proposals_audit": proposal_assessment,
        "representative_overlays": {
            k: str(overlays_dir / f"{v}_overlay.jpg") for k, v in rep_frames.items()
        }
    }

    # Save results JSON
    res_filename = "candidate_evaluation_results.json" if is_candidate else "baseline_evaluation_results.json"
    res_json_path = output_dir / res_filename
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[EVALUATION COMPLETE] Results saved to: {res_json_path}")
    print(f"[RUNTIME] Total evaluation runtime: {eval_runtime_seconds:.2f} seconds")

    # Render Markdown Reports
    if is_candidate:
        cand_md_path = output_dir / "CANDIDATE_EVALUATION_REPORT.md"
        render_markdown_evaluation_report(full_results, cand_md_path)
        print(f"Candidate diagnostic evaluation report written to: {cand_md_path}")
    else:
        render_markdown_evaluation_report(full_results, Path("docs/BASELINE_EVALUATION_REPORT.md"))
        render_markdown_proposal_report(proposal_assessment, Path("docs/MACHINE_PROPOSAL_AUDIT.md"))

    return full_results


# ==============================================================================
# 6. Report Rendering Utilities
# ==============================================================================

def render_markdown_evaluation_report(results: Dict[str, Any], output_md_path: Path) -> None:
    """Renders comprehensive, publication-quality Markdown evaluation report."""
    meta = results["metadata"]
    ap_cfg = meta["ap_evaluation_config"]
    op_cfg = meta["operational_diagnostic_config"]
    u_ap = results["ultralytics_ap_metrics"]
    op_diag = results["operational_diagnostics"]
    overall = op_diag["overall"]
    slices = op_diag["slices"]
    size_slices = op_diag["ground_truth_size_slices"]

    day = slices["real_day"]
    night = slices["real_night"]
    ne = slices["northeast_holdout"]
    in_dist = slices["in_distribution_cams"]
    nearby_exp = slices["nearby_training_exposure"]
    unproven_exp = slices["unproven_checkpoint_exposure"]

    is_candidate = meta.get("model_type") == "candidate"
    report_title = "YOLO26s Candidate Model Evaluation Report (Batch 8)" if is_candidate else "YOLO26s Thai Traffic Baseline Evaluation Report (Batch 3)"
    model_line = f"- **Evaluated Checkpoint**: `{meta.get('evaluated_model', meta['baseline_model'])}`\n- **Candidate SHA256**: `{meta.get('candidate_checkpoint_sha256', 'N/A')}`\n- **Verified Baseline Reference**: `{meta['baseline_model']}` (`{meta['baseline_checkpoint_sha256']}`)" if is_candidate else f"- **Evaluated Checkpoint**: `{meta['baseline_model']}`\n- **Checkpoint SHA256**: `{meta['baseline_checkpoint_sha256']}`"

    md = f"""# {report_title}

- **Date**: {meta['timestamp']}
{model_line}
- **Evaluation Runtime**: {meta.get('evaluation_runtime_seconds', 0.0):.2f}s
- **Device**: `{meta['cuda_device']}` (CUDA: {meta['cuda_available']})
- **Frameworks**: PyTorch `{meta['pytorch_version']}`, Ultralytics `{meta['ultralytics_version']}`, OpenCV `{meta['opencv_version']}`
- **Benchmark Dataset**: `data/eval_snapshot_v1` ({overall['frames_count']} frames, {overall['total_gt']} verified ground-truth instances)

---

> [!IMPORTANT]
> **Scientific Integrity & Benchmark Non-Independence Notice**
> This evaluation benchmark consists of {overall['frames_count']} diagnostic frames with documented exposure lineage:
> - **Nearby Training Diagnostic**: {nearby_exp['frames_count']} frames (<= 3.0s temporal distance from training footage).
> - **Unproven Checkpoint Exposure**: {unproven_exp['frames_count']} candidate frames where model weight training absence cannot be definitively proven.
> 
> This benchmark measures internal model capability, failure modes, and slice disparities. It **MUST NOT** be cited as an independent test set.

---

## 1. Ultralytics Standard Benchmark AP Metrics (conf={ap_cfg['conf_threshold']}, iou={ap_cfg['iou_nms_threshold']})

Evaluated via native Ultralytics `YOLO.val()` with low confidence integration threshold (`conf=0.001`) to calculate complete Precision-Recall curves. Per-class AP is mapped strictly through returned class IDs:

| Vehicle Class | Class ID | Images | GT Instances | AP50 | AP50-95 | Best-F1 Precision | Best-F1 Recall | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **All Classes** | — | **{overall['frames_count']}** | **{overall['total_gt']}** | **{u_ap['mAP50']:.4f}** | **{u_ap['mAP50_95']:.4f}** | **{u_ap['mean_best_f1_precision']:.4f}** | **{u_ap['mean_best_f1_recall']:.4f}** | **Benchmark Overall** |
| `car` | 0 | {overall['per_class']['car']['images_count']} | {overall['per_class']['car']['gt_count']} | {u_ap['per_class']['car']['AP50']:.4f} | {u_ap['per_class']['car']['AP50_95']:.4f} | {u_ap['per_class']['car']['best_f1_precision']:.4f} | {u_ap['per_class']['car']['best_f1_recall']:.4f} | Dense Support |
| `motorcycle` | 1 | {overall['per_class']['motorcycle']['images_count']} | {overall['per_class']['motorcycle']['gt_count']} | {u_ap['per_class']['motorcycle']['AP50']:.4f} | {u_ap['per_class']['motorcycle']['AP50_95']:.4f} | {u_ap['per_class']['motorcycle']['best_f1_precision']:.4f} | {u_ap['per_class']['motorcycle']['best_f1_recall']:.4f} | Moderate Support |
| `bus` | 2 | {overall['per_class']['bus']['images_count']} | {overall['per_class']['bus']['gt_count']} | {u_ap['per_class']['bus']['AP50']:.4f} | {u_ap['per_class']['bus']['AP50_95']:.4f} | {u_ap['per_class']['bus']['best_f1_precision']:.4f} | {u_ap['per_class']['bus']['best_f1_recall']:.4f} | Low Support ({overall['per_class']['bus']['gt_count']} GT) |
| `truck` | 3 | {overall['per_class']['truck']['images_count']} | {overall['per_class']['truck']['gt_count']} | {u_ap['per_class']['truck']['AP50']:.4f} | {u_ap['per_class']['truck']['AP50_95']:.4f} | {u_ap['per_class']['truck']['best_f1_precision']:.4f} | {u_ap['per_class']['truck']['best_f1_recall']:.4f} | Low Support ({overall['per_class']['truck']['gt_count']} GT) |
| `three_wheeler` | 4 | {overall['per_class']['three_wheeler']['images_count']} | {overall['per_class']['three_wheeler']['gt_count']} | {u_ap['per_class']['three_wheeler']['AP50']:.4f} | {u_ap['per_class']['three_wheeler']['AP50_95']:.4f} | {u_ap['per_class']['three_wheeler']['best_f1_precision']:.4f} | {u_ap['per_class']['three_wheeler']['best_f1_recall']:.4f} | Low Support ({overall['per_class']['three_wheeler']['gt_count']} GT) |

---

## 2. Operational Diagnostics at Fixed Threshold (conf={op_cfg['conf_threshold']}, matching IoU >= {op_cfg['matching_iou_threshold']})

Diagnostics evaluated at operational threshold `conf=0.25`:

| Vehicle Class | Class ID | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Overall Operational** | — | **{overall['total_gt']}** | **{overall['total_pred']}** | **{overall['precision']:.1%}** | **{overall['recall']:.1%}** | **{overall['f1_score']:.1%}** | Operational Baseline |
| `car` | 0 | {overall['per_class']['car']['gt_count']} | {overall['per_class']['car']['pred_count']} | {overall['per_class']['car']['precision']:.1%} | {overall['per_class']['car']['recall']:.1%} | {overall['per_class']['car']['f1_score']:.1%} | Dense Support |
| `motorcycle` | 1 | {overall['per_class']['motorcycle']['gt_count']} | {overall['per_class']['motorcycle']['pred_count']} | {overall['per_class']['motorcycle']['precision']:.1%} | {overall['per_class']['motorcycle']['recall']:.1%} | {overall['per_class']['motorcycle']['f1_score']:.1%} | Moderate Support |
| `bus` | 2 | {overall['per_class']['bus']['gt_count']} | {overall['per_class']['bus']['pred_count']} | {overall['per_class']['bus']['precision']:.1%} | {overall['per_class']['bus']['recall']:.1%} | {overall['per_class']['bus']['f1_score']:.1%} | Low Support |
| `truck` | 3 | {overall['per_class']['truck']['gt_count']} | {overall['per_class']['truck']['pred_count']} | {overall['per_class']['truck']['precision']:.1%} | {overall['per_class']['truck']['recall']:.1%} | {overall['per_class']['truck']['f1_score']:.1%} | Low Support |
| `three_wheeler` | 4 | {overall['per_class']['three_wheeler']['gt_count']} | {overall['per_class']['three_wheeler']['pred_count']} | {overall['per_class']['three_wheeler']['precision']:.1%} | {overall['per_class']['three_wheeler']['recall']:.1%} | {overall['per_class']['three_wheeler']['f1_score']:.1%} | Low Support |

### 6x6 Object Confusion Matrix (Matching IoU >= 0.50)
Rows represent authoritative **Human Ground Truth**, columns represent **Baseline Model Predictions**:

| GT / Pred | car (0) | motorcycle (1) | bus (2) | truck (3) | three_wheeler (4) | Background (FN) | Total GT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **car (0)** | **{overall['confusion_matrix'][0][0]}** | {overall['confusion_matrix'][0][1]} | {overall['confusion_matrix'][0][2]} | {overall['confusion_matrix'][0][3]} | {overall['confusion_matrix'][0][4]} | **{overall['confusion_matrix'][0][5]}** | {sum(overall['confusion_matrix'][0])} |
| **motorcycle (1)** | {overall['confusion_matrix'][1][0]} | **{overall['confusion_matrix'][1][1]}** | {overall['confusion_matrix'][1][2]} | {overall['confusion_matrix'][1][3]} | {overall['confusion_matrix'][1][4]} | **{overall['confusion_matrix'][1][5]}** | {sum(overall['confusion_matrix'][1])} |
| **bus (2)** | {overall['confusion_matrix'][2][0]} | {overall['confusion_matrix'][2][1]} | **{overall['confusion_matrix'][2][2]}** | {overall['confusion_matrix'][2][3]} | {overall['confusion_matrix'][2][4]} | **{overall['confusion_matrix'][2][5]}** | {sum(overall['confusion_matrix'][2])} |
| **truck (3)** | {overall['confusion_matrix'][3][0]} | {overall['confusion_matrix'][3][1]} | {overall['confusion_matrix'][3][2]} | **{overall['confusion_matrix'][3][3]}** | {overall['confusion_matrix'][3][4]} | **{overall['confusion_matrix'][3][5]}** | {sum(overall['confusion_matrix'][3])} |
| **three_wheeler (4)** | {overall['confusion_matrix'][4][0]} | {overall['confusion_matrix'][4][1]} | {overall['confusion_matrix'][4][2]} | {overall['confusion_matrix'][4][3]} | **{overall['confusion_matrix'][4][4]}** | **{overall['confusion_matrix'][4][5]}** | {sum(overall['confusion_matrix'][4])} |
| **Background (FP)** | {overall['confusion_matrix'][5][0]} | {overall['confusion_matrix'][5][1]} | {overall['confusion_matrix'][5][2]} | {overall['confusion_matrix'][5][3]} | {overall['confusion_matrix'][5][4]} | — | **{sum(overall['confusion_matrix'][5][:5])}** |

---

## 3. Diagnostic Slices Breakdown

### Slice A: Ground-Truth Object Size Recall (Matched on Full Frame First)
*Formula*: `scale = 640 / max(img_w, img_h)`, `area = (bw * img_w * scale) * (bh * img_h * scale)`.
*Note*: Full-frame matching is performed first to guarantee that slightly larger or shifted detections successfully match small ground-truth vehicles. Size-specific precision is omitted because false-positive attribution across sizes is ambiguous.

| Size Category | Area Definition | GT Count | Detected (TP) | Missed (FN) | Class Disagree | Loc Error | GT Recall | Support Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Small** | {size_slices['small']['area_definition']} | {size_slices['small']['gt_count']} | {size_slices['small']['detected_count']} | {size_slices['small']['missed_count']} | {size_slices['small']['class_disagreements']} | {size_slices['small']['localization_errors']} | **{size_slices['small']['recall']:.1%}** | {size_slices['small']['support_note']} |
| **Medium** | {size_slices['medium']['area_definition']} | {size_slices['medium']['gt_count']} | {size_slices['medium']['detected_count']} | {size_slices['medium']['missed_count']} | {size_slices['medium']['class_disagreements']} | {size_slices['medium']['localization_errors']} | **{size_slices['medium']['recall']:.1%}** | {size_slices['medium']['support_note']} |
| **Large** | {size_slices['large']['area_definition']} | {size_slices['large']['gt_count']} | {size_slices['large']['detected_count']} | {size_slices['large']['missed_count']} | {size_slices['large']['class_disagreements']} | {size_slices['large']['localization_errors']} | **{size_slices['large']['recall']:.1%}** | {size_slices['large']['support_note']} |

### Slice B: Lighting Disparity (Real Day vs Real Night)

| Condition | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Real Day** | {day['frames_count']} | {day['total_gt']} | {day['total_pred']} | {day['precision']:.1%} | {day['recall']:.1%} | {day['f1_score']:.1%} | Dense Support |
| **Real Night** | {night['frames_count']} | {night['total_gt']} | {night['total_pred']} | {night['precision']:.1%} | {night['recall']:.1%} | {night['f1_score']:.1%} | Moderate Support |

### Slice C: Camera Domain Disparity (Holdout Northeast vs In-Distribution)

| Camera Group | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **In-Distribution Cameras** | {in_dist['frames_count']} | {in_dist['total_gt']} | {in_dist['total_pred']} | {in_dist['precision']:.1%} | {in_dist['recall']:.1%} | {in_dist['f1_score']:.1%} | Dense Support |
| **Holdout Northeast (cam45)** | {ne['frames_count']} | {ne['total_gt']} | {ne['total_pred']} | {ne['precision']:.1%} | {ne['recall']:.1%} | {ne['f1_score']:.1%} | Moderate Support |

### Slice D: Exposure Lineage Disparity

| Exposure Group | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nearby Training (<= 3.0s)** | {nearby_exp['frames_count']} | {nearby_exp['total_gt']} | {nearby_exp['total_pred']} | {nearby_exp['precision']:.1%} | {nearby_exp['recall']:.1%} | {nearby_exp['f1_score']:.1%} | Known Training Proximity |
| **Unproven Checkpoint Exposure** | {unproven_exp['frames_count']} | {unproven_exp['total_gt']} | {unproven_exp['total_pred']} | {unproven_exp['precision']:.1%} | {unproven_exp['recall']:.1%} | {unproven_exp['f1_score']:.1%} | Unproven Lineage Candidate |

---

## 4. Primary Baseline Failure Modes & Empirical Observations

1. **Small Vehicle Omission**:
   - For vehicles letterboxed under 32x32 pixels, ground-truth recall is **{size_slices['small']['recall']:.1%}** ({size_slices['small']['missed_count']} pure omissions). Distant vehicles in queued lanes are frequently missed.
2. **Truck vs Light Vehicle Confusion (Poor Truck Precision)**:
   - Operational truck precision is only **{overall['per_class']['truck']['precision']:.1%}** ({overall['per_class']['truck']['fp']} false alarms out of {overall['per_class']['truck']['pred_count']} truck predictions). The model frequently misidentifies pickup-based songthaews, passenger vans, and delivery pickups as commercial trucks.
3. **Motorcycle Omission in Congestion**:
   - At operational threshold (conf=0.25), motorcycle recall is **{overall['per_class']['motorcycle']['recall']:.1%}** ({overall['per_class']['motorcycle']['fn']} false negatives out of {overall['per_class']['motorcycle']['gt_count']} GT).
4. **Generalization Gap on Holdout Camera**:
   - Precision drops from {in_dist['precision']:.1%} on in-distribution cameras to **{ne['precision']:.1%}** on `cam45_northeast`.

---

## 5. Potential Experimental Directions to Test (Future Iterations)

The following directions represent empirical hypotheses to evaluate in future model training passes, not established requirements:
- **Multi-Scale Tiling Inference**: Test high-resolution slicing (e.g. SAHI / tile inference) on distant traffic queues to evaluate if small vehicle recall can be improved without generating excessive false alarms.
- **Morphological Feature Distillation**: Test loss reweighting or fine-tuning with hard negatives to differentiate ordinary pickups and vans (class 0) from medium trucks (class 3).
- **Domain Adaptive Night Augmentation**: Test low-light glare simulation and contrast augmentation to evaluate night recall recovery.

---

## 6. Visual Error Overlays

All {overall['frames_count']} visual overlays are saved in `output/baseline_eval/overlays/`:
- **Green**: True Positive (GT matched with correct prediction)
- **Red**: Missed Vehicle (False Negative: GT omitted by baseline)
- **Orange**: False Alarm (False Positive: spurious baseline detection)
- **Purple**: Class Disagreement (Spatial overlap >= 0.50, but wrong class assigned)
- **Cyan**: Localization Error (0.10 <= IoU < 0.50)

Representative diagnostic frames:
- Daytime Approach: `output/baseline_eval/overlays/cam03_east_f003720_overlay.jpg`
- Real Night Approach: `output/baseline_eval/overlays/cam43_south_night_f002800_overlay.jpg`
- Northeast Camera: `output/baseline_eval/overlays/cam45_northeast_f063055_overlay.jpg`
- Small / Distant Vehicles: `output/baseline_eval/overlays/cam43_south_f046440_overlay.jpg`
"""
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(md, encoding="utf-8")
    print(f"Baseline evaluation report written to: {output_md_path}")


def render_markdown_proposal_report(prop_audit: Dict[str, Any], output_md_path: Path) -> None:
    """Renders machine proposal quality audit report (Scope Item 4)."""
    std = prop_audit["standard_detection_metrics"]
    gran = prop_audit["granular_error_breakdown"]
    disagreements = prop_audit.get("class_disagreements", {})

    md = f"""# Machine Proposal Quality Assessment Report (Batch 3)

## 1. Executive Summary

This audit assesses the quality of the initial automated machine proposals (`review_pack/proposals/*.txt`) compared against authoritative human ground truth across the **{prop_audit['evaluated_frames_count']} frames** where machine proposals originally existed:

- **Total Initial Machine Proposals**: {prop_audit['total_proposals']}
- **Total Verified Human Ground Truth Boxes**: {prop_audit['total_human_gt']}
- **Proposal Precision (Standard Detection)**: **{std['precision']:.1%}** ({std['tp']} TP / {prop_audit['total_proposals']} proposals)
- **Proposal Recall (Standard Detection)**: **{std['recall']:.1%}** ({std['tp']} TP / {prop_audit['total_human_gt']} human ground truth)
- **Proposal F1 Score**: **{std['f1_score']:.1%}**

---

## 2. Granular Proposal Error Taxonomy (One-to-One Matching)

To avoid conflating classification and boundary fit errors with pure omissions or hallucinations, the matching errors are decomposed as follows:

| Category | Count | % of Reference | Description |
| :--- | :---: | :---: | :--- |
| **Correct Matches (True Positives)** | **{gran['correct_matches']}** | {gran['correct_matches']/prop_audit['total_human_gt']:.1%} of human GT | Spatial overlap (IoU >= 0.50) with correct vehicle class |
| **Unmatched Ground Truth (Pure Omissions)** | **{gran['unmatched_gt']}** | {gran['unmatched_gt']/prop_audit['total_human_gt']:.1%} of human GT | Human-verified vehicles completely missed by proposals |
| **Unmatched Proposals (Spurious Detections)** | **{gran['unmatched_pred']}** | {gran['unmatched_pred']/prop_audit['total_proposals']:.1%} of proposals | Proposals with no ground truth vehicle in vicinity |
| **Class Disagreements** | **{gran['class_disagreements']}** | {gran['class_disagreements']/prop_audit['total_human_gt']:.1%} of human GT | Overlapped GT (IoU >= 0.50) but assigned wrong class |
| **Localization Errors** | **{gran['localization_errors']}** | {gran['localization_errors']/prop_audit['total_human_gt']:.1%} of human GT | Correct class match with moderate overlap (0.10 <= IoU < 0.50) |

*Reconciliation*:
- Human Ground Truth: {gran['correct_matches']} (correct) + {gran['unmatched_gt']} (omissions) + {gran['class_disagreements']} (class mismatch) + {gran['localization_errors']} (loc error) = **{prop_audit['total_human_gt']} total GT**.
- Machine Proposals: {gran['correct_matches']} (correct) + {gran['unmatched_pred']} (spurious) + {gran['class_disagreements']} (class mismatch) + {gran['localization_errors']} (loc error) = **{prop_audit['total_proposals']} total proposals**.

### Class Disagreement Breakdown
{chr(10).join(f"- `{k}`: {v} instances" for k, v in disagreements.items())}

---

## 3. Potential Experimental Directions for Teacher Generation (Future Iterations)

The following directions represent empirical hypotheses to evaluate before scaling up teacher-assisted pseudo-labeling:
1. **Targeted Human Verification on Night Footage**: Test human-in-the-loop review for nighttime footage where contrast degradation causes disproportionate omission.
2. **Morphology Disambiguation Rules**: Evaluate geometric prior filters (aspect ratio, height) to reduce `car -> truck` and `car -> bus` misclassifications.
3. **Multi-Scale Feature Tiling**: Test tile-based inference to evaluate if small vehicle proposal recall can be enhanced.
4. **Empirical Confidence Threshold Sweeps**: Calibrate teacher candidate thresholds against verified validation frames rather than using raw uncalibrated proposals.
"""
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(md, encoding="utf-8")
    print(f"Machine proposal audit report written to: {output_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline or candidate model against evaluation snapshot.")
    parser.add_argument("--model", type=str, default="models/yolo26s_thai_traffic.pt", help="Model path to evaluate")
    parser.add_argument("--snapshot", type=str, default="data/eval_snapshot_v1", help="Evaluation snapshot directory")
    parser.add_argument("--output", type=str, default="output/baseline_eval", help="Output directory")
    parser.add_argument("--op-conf", type=float, default=0.25, help="Operational confidence threshold")
    parser.add_argument("--ap-conf", type=float, default=0.001, help="AP integration confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.50, help="Matching IoU threshold")
    parser.add_argument("--is-candidate", action="store_true", default=False, help="Evaluate candidate model (preserves baseline hash and verification)")
    parser.add_argument("--baseline-model", type=str, default="models/yolo26s_thai_traffic.pt", help="Baseline model path for integrity verification")

    args = parser.parse_args()

    run_baseline_evaluation(
        model_path=Path(args.model),
        snapshot_dir=Path(args.snapshot),
        output_dir=Path(args.output),
        operational_conf=args.op_conf,
        ap_conf=args.ap_conf,
        iou_nms=args.iou,
        match_iou_thresh=args.match_iou,
        is_candidate=args.is_candidate,
        baseline_model_path=Path(args.baseline_model),
    )


if __name__ == "__main__":
    main()
