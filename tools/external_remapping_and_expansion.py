"""
tools/external_remapping_and_expansion.py

Prepares remapped, nested external datasets and teacher-assisted completions
for the external-data remapping and expansion experiment.

Core Responsibilities:
1. Remap copied UA-DETRAC annotations:
   source bus(0) -> bus(2)
   source car(1), truck(2), van(3) -> car(0)
2. Build nested external subsets:
   - Small subset (R25): 25 reviewed external training frames from review_pack_consolidated_v3,
     with the 5 truck boxes explicitly remapped to car(0).
   - Expanded subset (R100): 100 unique canonical external training frames TOTAL,
     containing those exact same 25 plus 75 deterministically selected candidates.
3. Complete annotations on the additional 75 frames using local models/yolo26x.pt:
   - Teacher mapping: COCO car(2) & truck(7) -> car(0); bus(5) -> bus(2); motorcycle(3) -> motorcycle(1).
   - Full-frame + multi-scale tiled inference with coordinate restoration.
   - Deduplicate proposals and merge non-overlapping candidate additions with source boxes.
   - Do not overwrite existing source boxes (especially bus). Log conflicts.
   - Mark completed frames as machine-labeled, NOT human-verified.
4. Materialize isolated training datasets:
   - Dataset R25: 1,092 local CCTV frames + 25 remapped reviewed external frames.
   - Dataset R100: 1,092 local CCTV frames + 100 remapped external frames.
   - Preserves local exclusions (cam44_north_f019140 rejected, 41 stale synthetic variants).
   - Leaves local training, primary validation, and diagnostic ground truth strictly untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    THAI_5CLASS_NAMES,
    compute_aspect_preserving_size,
    validate_box,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.teacher_completion_pilot import (
    compute_box_iou,
    compute_box_iomin,
    deduplicate_proposals,
    generate_tile_grid,
    norm_yolo_to_xyxy,
    tile_xyxy_to_global_xyxy,
    xyxy_to_norm_yolo,
)

# Experimental remapping for external UA-DETRAC annotations:
# Raw NDJSON header: 0: bus, 1: car, 2: truck, 3: van
# Mapping: bus(0) -> bus(2); car(1), truck(2), van(3) -> car(0)
UADETRAC_RAW_TO_REMAPPED_THAI5 = {
    0: 2,  # bus -> bus
    1: 0,  # car -> car
    2: 0,  # truck -> car (experimental remapping)
    3: 0,  # van -> car (light vehicle)
}

# Teacher COCO 80-class mapping under experimental remapping:
# COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TEACHER_TO_REMAPPED_THAI5 = {
    2: 0,  # car -> car
    7: 0,  # truck -> car (experimental remapping)
    5: 2,  # bus -> bus
    3: 1,  # motorcycle -> motorcycle
}
COCO_ALLOWED_CLASSES = list(COCO_TEACHER_TO_REMAPPED_THAI5.keys())


def select_deterministic_75_frames(
    candidate_records: List[Dict[str, Any]],
    excluded_canonical_ids: Set[str],
) -> List[Dict[str, Any]]:
    """
    Selects exactly 75 additional external frames deterministically from candidate records:
    - Excludes frames whose canonical source ID is already in excluded_canonical_ids (the 25 reviewed frames).
    - Groups candidates by sequence to ensure broad scene coverage.
    - Within each sequence, prioritizes frames by:
      1. Number of small vehicle boxes (< 1024 px^2) descending
      2. Total boxes descending
      3. Canonical source ID ascending (tie-breaker)
    - Distributes selection across all available sequences (1 frame each),
      then takes remaining frames from sequences with highest small-vehicle count.
    """
    eligible = [
        r for r in candidate_records
        if r["canonical_source_id"] not in excluded_canonical_ids
    ]

    by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in eligible:
        by_seq[r["sequence_id"]].append(r)

    # Sort each sequence's candidates deterministically
    for seq_id in by_seq:
        by_seq[seq_id].sort(
            key=lambda r: (
                -r.get("box_sizes", {}).get("small", 0),
                -r.get("total_boxes", 0),
                r["canonical_source_id"]
            )
        )

    selected: List[Dict[str, Any]] = []
    sorted_seqs = sorted(by_seq.keys())

    # Pass 1: Select 1 frame per sequence for maximum scene diversity
    for seq_id in sorted_seqs:
        if by_seq[seq_id]:
            selected.append(by_seq[seq_id][0])

    needed = 75 - len(selected)
    if needed > 0:
        # Pass 2: Select remaining from unused candidates across all sequences,
        # ordered by small-vehicle count descending
        remaining: List[Dict[str, Any]] = []
        for seq_id in sorted_seqs:
            for r in by_seq[seq_id][1:]:
                remaining.append(r)

        remaining.sort(
            key=lambda r: (
                -r.get("box_sizes", {}).get("small", 0),
                -r.get("total_boxes", 0),
                r["canonical_source_id"]
            )
        )
        selected.extend(remaining[:needed])

    if len(selected) != 75:
        raise ValueError(f"Expected to select exactly 75 frames, but selected {len(selected)}")

    # Verify canonical uniqueness
    sel_canons = set(r["canonical_source_id"] for r in selected)
    if len(sel_canons) != 75:
        raise ValueError(f"Selected 75 frames contain duplicate canonical sources ({len(sel_canons)} unique)")

    # Verify disjointness from excluded
    overlap = sel_canons & excluded_canonical_ids
    if overlap:
        raise ValueError(f"Selected frames overlap with excluded canonical sources: {overlap}")

    return selected


def download_and_verify_image(
    url: str,
    dest_path: Path,
    expected_w: int = 640,
    expected_h: int = 640,
    timeout: int = 15,
) -> Tuple[bool, Optional[str]]:
    """
    Downloads image from URL (or reuses existing valid image) and verifies exact dimensions.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        try:
            with Image.open(dest_path) as im:
                im.load()
                if im.size == (expected_w, expected_h):
                    return True, None
        except Exception:
            pass  # Re-download if corrupted

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        return False, f"HTTP download error from {url}: {e}"

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        if im.size != (expected_w, expected_h):
            return False, f"Dimension mismatch: expected {expected_w}x{expected_h}, got {im.size[0]}x{im.size[1]}"
    except Exception as e:
        return False, f"Image decode error: {e}"

    with open(dest_path, "wb") as f:
        f.write(data)

    return True, None


def run_teacher_on_frame(
    model: YOLO,
    img_bgr: np.ndarray,
    device: str,
    conf_thresh: float = 0.20,
    full_frame_imgsz: int = 1280,
    tile_size: int = 384,
    tile_overlap: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Runs YOLO26x teacher inference using both full-frame and tiled passes.
    Maps detected COCO vehicle classes to experimental Thai 5-class taxonomy:
    COCO car(2) & truck(7) -> car(0), bus(5) -> bus(2), motorcycle(3) -> motorcycle(1).
    Restores tile coordinates and returns deduplicated proposals.
    """
    img_h, img_w = img_bgr.shape[:2]
    raw_proposals: List[Dict[str, Any]] = []

    # 1. Full-frame inference
    res_full = model.predict(
        img_bgr,
        conf=conf_thresh,
        imgsz=full_frame_imgsz,
        classes=COCO_ALLOWED_CLASSES,
        device=device,
        verbose=False,
    )

    if len(res_full) > 0 and res_full[0].boxes is not None:
        for b in res_full[0].boxes:
            coco_cid = int(b.cls[0].item())
            if coco_cid not in COCO_TEACHER_TO_REMAPPED_THAI5:
                continue
            thai_cid = COCO_TEACHER_TO_REMAPPED_THAI5[coco_cid]
            conf = float(b.conf[0].item())
            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().tolist()
            norm_box = xyxy_to_norm_yolo([bx1, by1, bx2, by2], img_w, img_h)

            raw_proposals.append({
                "class_id": thai_cid,
                "class_name": THAI_5CLASS_NAMES[thai_cid],
                "coco_class_id": coco_cid,
                "confidence": round(conf, 4),
                "bbox_norm": norm_box,
                "xyxy_px": [bx1, by1, bx2, by2],
                "method": "full_frame",
            })

    # 2. Tiled inference
    tiles = generate_tile_grid(img_w, img_h, tile_size=tile_size, overlap=tile_overlap)
    for tx1, ty1, tx2, ty2 in tiles:
        tile_img = img_bgr[ty1:ty2, tx1:tx2]
        res_tile = model.predict(
            tile_img,
            conf=conf_thresh,
            imgsz=tile_size,
            classes=COCO_ALLOWED_CLASSES,
            device=device,
            verbose=False,
        )
        if len(res_tile) > 0 and res_tile[0].boxes is not None:
            for b in res_tile[0].boxes:
                coco_cid = int(b.cls[0].item())
                if coco_cid not in COCO_TEACHER_TO_REMAPPED_THAI5:
                    continue
                thai_cid = COCO_TEACHER_TO_REMAPPED_THAI5[coco_cid]
                conf = float(b.conf[0].item())
                t_bx1, t_by1, t_bx2, t_by2 = b.xyxy[0].cpu().numpy().tolist()
                g_xyxy = tile_xyxy_to_global_xyxy([t_bx1, t_by1, t_bx2, t_by2], tx1, ty1)
                norm_box = xyxy_to_norm_yolo(g_xyxy, img_w, img_h)

                raw_proposals.append({
                    "class_id": thai_cid,
                    "class_name": THAI_5CLASS_NAMES[thai_cid],
                    "coco_class_id": coco_cid,
                    "confidence": round(conf, 4),
                    "bbox_norm": norm_box,
                    "xyxy_px": g_xyxy,
                    "method": "tiled",
                })

    # Deduplicate proposals across full-frame and tiles
    deduped = deduplicate_proposals(raw_proposals, iou_thresh=0.55)
    return deduped


def merge_teacher_proposals_with_source_boxes(
    source_boxes: List[Dict[str, Any]],
    teacher_proposals: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    match_iou_thresh: float = 0.45,
    ambig_iou_thresh: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merges supplemental teacher proposals with existing source boxes:
    - Retains ALL existing source boxes.
    - If proposal matches existing source box (IoU >= 0.45):
      * If classes match: confirmed duplicate, discard proposal.
      * If classes disagree: preserve source box! Especially protect bus(2) labels. Log conflict.
    - If 0.15 <= IoU < 0.45: ambiguous spatial overlap, discard proposal to avoid halo artifacts.
    - If IoU < 0.15: candidate addition! Append proposal as new machine-completed box.
    Returns: (final_combined_boxes, accepted_additions, logged_conflicts)
    """
    # Convert source boxes to pixel coordinates
    src_pixel_boxes = []
    for sb in source_boxes:
        xyxy = norm_yolo_to_xyxy(sb["bbox_norm"], img_w, img_h)
        src_pixel_boxes.append({
            "source_box": sb,
            "xyxy_px": xyxy,
            "class_id": sb["class_id"],
        })

    accepted_additions: List[Dict[str, Any]] = []
    logged_conflicts: List[Dict[str, Any]] = []

    for prop in teacher_proposals:
        p_xyxy = prop["xyxy_px"]
        best_iou = 0.0
        best_src = None

        for sp in src_pixel_boxes:
            iou = compute_box_iou(p_xyxy, sp["xyxy_px"])
            if iou > best_iou:
                best_iou = iou
                best_src = sp

        if best_iou >= match_iou_thresh and best_src is not None:
            # Overlaps an existing source box
            if prop["class_id"] != best_src["class_id"]:
                # Class conflict: do NOT overwrite source box! Log conflict
                logged_conflicts.append({
                    "proposal_class_id": prop["class_id"],
                    "proposal_class_name": prop["class_name"],
                    "source_class_id": best_src["class_id"],
                    "source_class_name": THAI_5CLASS_NAMES[best_src["class_id"]],
                    "iou": round(best_iou, 4),
                    "confidence": prop["confidence"],
                    "bbox_norm": prop["bbox_norm"],
                })
            # In either case (duplicate or conflict), discard proposal to preserve source box
        elif best_iou >= ambig_iou_thresh:
            # Ambiguous overlap, discard
            pass
        else:
            # Candidate addition!
            addition_box = {
                "class_id": prop["class_id"],
                "class_name": prop["class_name"],
                "bbox_norm": prop["bbox_norm"],
                "confidence": prop["confidence"],
                "provenance": f"yolo26x_teacher_{prop['method']}(conf={prop['confidence']})",
                "is_teacher_addition": True,
            }
            accepted_additions.append(addition_box)

    final_boxes = list(source_boxes) + accepted_additions
    return final_boxes, accepted_additions, logged_conflicts


def render_review_overlay(
    img_path: Path,
    boxes: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Renders an inspection overlay showing bounding boxes and labels:
    - Blue/Green for source boxes
    - Amber/Orange for teacher additions
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return
    img_h, img_w = img_bgr.shape[:2]

    for b in boxes:
        cid = b["class_id"]
        cname = b.get("class_name", THAI_5CLASS_NAMES.get(cid, str(cid)))
        is_add = b.get("is_teacher_addition", False)
        xc, yc, bw, bh = b["bbox_norm"]
        x1 = int(round((xc - bw / 2.0) * img_w))
        y1 = int(round((yc - bh / 2.0) * img_h))
        x2 = int(round((xc + bw / 2.0) * img_w))
        y2 = int(round((yc + bh / 2.0) * img_h))

        # Color: source boxes in green (or cyan for bus), teacher additions in amber
        if is_add:
            color = (0, 165, 255)  # Amber
            label = f"+{cname} ({b.get('confidence', 0.0):.2f})"
        else:
            color = (0, 220, 0) if cid == 0 else ((255, 100, 0) if cid == 2 else (0, 200, 255))
            label = cname

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img_bgr)


def prepare_external_remapped_and_expanded_datasets(
    repo_root: Path,
    dataset_r25_dir: Path,
    dataset_r100_dir: Path,
    manifests_v6_dir: Path,
    consolidated_v3_dir: Path,
    snapshot_dir: Path,
    ndjson_path: Path,
    teacher_model_path: Path,
    overlays_dir: Path,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete preparation pipeline:
    1. Prepares 25 remapped reviewed external frames (remapping 5 truck boxes to car).
    2. Selects 75 additional external frames deterministically across eligible sequences.
    3. Downloads and verifies images.
    4. Completes annotations on the 75 frames using YOLO26x teacher with remapping.
    5. Materializes Dataset R25 (1,092 local + 25 remapped external) and
       Dataset R100 (1,092 local + 100 remapped external).
    """
    print("\n" + "=" * 75)
    print("PREPARATION: EXTERNAL REMAPPING AND EXPANSION PIPELINE")
    print("=" * 75)

    dataset_r25_dir = dataset_r25_dir.resolve()
    dataset_r100_dir = dataset_r100_dir.resolve()

    if dataset_r25_dir.exists() and any(dataset_r25_dir.iterdir()) and not force_rebuild:
        raise RuntimeError(f"Directory {dataset_r25_dir} already exists and is not empty! Pass force_rebuild to overwrite.")
    if dataset_r100_dir.exists() and any(dataset_r100_dir.iterdir()) and not force_rebuild:
        raise RuntimeError(f"Directory {dataset_r100_dir} already exists and is not empty! Pass force_rebuild to overwrite.")

    if force_rebuild:
        if dataset_r25_dir.exists():
            shutil.rmtree(dataset_r25_dir)
        if dataset_r100_dir.exists():
            shutil.rmtree(dataset_r100_dir)

    for d in [dataset_r25_dir, dataset_r100_dir]:
        for sub in ["images/train", "labels/train", "images/val", "labels/val"]:
            (d / sub).mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inventories and reference manifests
    with open(manifests_v6_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
        man_a = json.load(f)
    local_train_ids = man_a["train_records"]

    with open(manifests_v6_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
        prim_val_man = json.load(f)
    val_ids = prim_val_man["records"]

    with open(manifests_v6_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
        inv_data = json.load(f)
    inv_map = {r["inventory_id"]: r for r in inv_data["records"]}

    with open(manifests_v6_dir / "manifest_b_local_plus_external.json", "r", encoding="utf-8") as f:
        man_b = json.load(f)
    man_b_ext_ids = [rid for rid in man_b["train_records"] if rid.startswith("external:")]

    with open(consolidated_v3_dir / "annotations" / "annotations.json", "r", encoding="utf-8") as f:
        consolidated_annots = json.load(f)

    with open(snapshot_dir / "manifest.json", "r", encoding="utf-8") as f:
        snap_manifest = json.load(f)
    diagnostic_sources = set(s["canonical_source_id"] for s in snap_manifest["samples"])

    # 2. Populate Common Validation Set (130 frames, strictly identical to Batch 9)
    print("\n[STEP 1/5] Populating Primary Validation Set (130 frames, local taxonomy)...")
    val_manifest_samples: List[Dict[str, Any]] = []
    val_box_count = 0
    val_class_counts: Counter = Counter()

    for rid in val_ids:
        rec = inv_map[rid]
        src_img = Path(rec["image_path"])
        src_lbl = Path(rec["label_path"])
        stem = rec["stem"]
        w = int(rec.get("dimensions", {}).get("width", 1920))
        h = int(rec.get("dimensions", {}).get("height", 1080))

        # Copy validation images and labels to both R25 and R100
        for ds_dir in [dataset_r25_dir, dataset_r100_dir]:
            shutil.copy2(src_img, ds_dir / "images" / "val" / f"{stem}.jpg")
            shutil.copy2(src_lbl, ds_dir / "labels" / "val" / f"{stem}.txt")

        # Parse boxes for validation manifest
        sample_boxes = []
        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                bw, bh = float(parts[3]), float(parts[4])
                sz_info = compute_aspect_preserving_size(bw, bh, img_w=w, img_h=h, ref_size=640)
                val_box_count += 1
                val_class_counts[cname] += 1
                sample_boxes.append({
                    "class_id": cid,
                    "class_name": cname,
                    "bbox_norm": [float(parts[1]), float(parts[2]), bw, bh],
                    "size_bucket": sz_info["size_bucket"],
                })

        val_manifest_samples.append({
            "frame_id": stem,
            "canonical_source_id": rec["canonical_source_id"],
            "camera": rec.get("camera", "unknown"),
            "lighting_type": rec.get("lighting", "real_day"),
            "original_split": rec.get("original_split", "train"),
            "training_exposure": "old_train_split" if rec.get("original_split") == "train" else "old_val_split",
            "dimensions": [w, h],
            "boxes": sample_boxes,
        })

    print(f"  [OK] Validation set verified: 130 frames, {val_box_count} ground truth boxes.")

    # 3. Populate Local Training Set (1,092 frames, strictly identical to Batch 9)
    print("\n[STEP 2/5] Populating Local Training Set (1,092 frames, local taxonomy)...")
    local_box_count = 0
    local_class_counts: Counter = Counter()

    for rid in local_train_ids:
        rec = inv_map[rid]
        src_img = Path(rec["image_path"])
        src_lbl = Path(rec["label_path"])
        stem = Path(src_img).stem

        for ds_dir in [dataset_r25_dir, dataset_r100_dir]:
            shutil.copy2(src_img, ds_dir / "images" / "train" / f"{stem}.jpg")
            shutil.copy2(src_lbl, ds_dir / "labels" / "train" / f"{stem}.txt")

        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = THAI_5CLASS_NAMES[cid]
                local_box_count += 1
                local_class_counts[cname] += 1

    if local_box_count != 13148:
        raise ValueError(f"Expected 13,148 local training boxes, got {local_box_count}")
    print(f"  [OK] Local training set verified: 1,092 frames, {local_box_count} boxes.")

    # 4. Remap and Copy the 25 Reviewed External Frames
    print("\n[STEP 3/5] Remapping the 25 Reviewed External Frames (source truck -> car)...")
    ext_reviewed_records = [
        r for r in consolidated_annots
        if r.get("data_origin") == "external_ua_detrac" and r.get("training_eligible")
    ]
    if len(ext_reviewed_records) != 25:
        raise ValueError(f"Expected 25 eligible reviewed external records, got {len(ext_reviewed_records)}")

    reviewed_25_canons = set(r["canonical_source_id"] for r in ext_reviewed_records)
    r25_box_count = 0
    r25_class_counts: Counter = Counter()
    truck_to_car_remapping_log: List[Dict[str, Any]] = []

    for r in ext_reviewed_records:
        fid = r["frame_id"]
        src_img = consolidated_v3_dir / "images" / f"{fid}.jpg"
        src_lbl = consolidated_v3_dir / "annotations" / "labels" / f"{fid}.txt"

        if not src_img.exists():
            raise FileNotFoundError(f"Missing reviewed external image: {src_img}")
        if not src_lbl.exists():
            raise FileNotFoundError(f"Missing reviewed external label: {src_lbl}")

        # Read original reviewed boxes and apply experimental remapping
        remapped_lines = []
        with open(src_lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                orig_cid = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Remap truck(3) -> car(0); preserve bus(2), car(0), motorcycle(1)
                if orig_cid == 3:
                    new_cid = 0  # truck -> car
                    truck_to_car_remapping_log.append({
                        "frame_id": fid,
                        "original_class_id": 3,
                        "original_class_name": "truck",
                        "remapped_class_id": 0,
                        "remapped_class_name": "car",
                        "bbox_norm": [xc, yc, bw, bh],
                    })
                else:
                    new_cid = orig_cid

                cname = THAI_5CLASS_NAMES[new_cid]
                r25_box_count += 1
                r25_class_counts[cname] += 1
                remapped_lines.append(f"{new_cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Write to both R25 and R100 training sets
        for ds_dir in [dataset_r25_dir, dataset_r100_dir]:
            shutil.copy2(src_img, ds_dir / "images" / "train" / f"{fid}.jpg")
            lbl_dst = ds_dir / "labels" / "train" / f"{fid}.txt"
            lbl_dst.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")

    if r25_box_count != 813:
        raise ValueError(f"Expected exactly 813 boxes in 25 reviewed frames, got {r25_box_count}")
    if len(truck_to_car_remapping_log) != 5:
        raise ValueError(f"Expected exactly 5 truck boxes remapped to car, got {len(truck_to_car_remapping_log)}")

    print(f"  [OK] 25 reviewed external frames remapped: {r25_box_count} total boxes.")
    print(f"       Class distribution: {dict(r25_class_counts)}")
    print(f"       Explicit remapping overrides: {len(truck_to_car_remapping_log)} truck boxes -> car (0)")

    # 5. Deterministically Select, Download, and Machine-Complete the Additional 75 External Frames
    print("\n[STEP 4/5] Selecting and Machine-Completing 75 Additional External Frames...")

    cand_records = [inv_map[rid] for rid in man_b_ext_ids]
    selected_75_records = select_deterministic_75_frames(
        candidate_records=cand_records,
        excluded_canonical_ids=reviewed_25_canons,
    )

    print(f"  [OK] Selected 75 canonical external frames across {len(set(r['sequence_id'] for r in selected_75_records))} sequences.")

    # Index NDJSON entries for the 75 frames
    sel_75_files = {r["image_path"]: r for r in selected_75_records}
    ndjson_map: Dict[str, Dict[str, Any]] = {}
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            fl = obj.get("file")
            if fl in sel_75_files:
                ndjson_map[fl] = obj

    if len(ndjson_map) != 75:
        raise RuntimeError(f"Expected 75 matching NDJSON entries, got {len(ndjson_map)}")

    # Load teacher model YOLO26x
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  [TEACHER] Loading {teacher_model_path} on {device}...")
    teacher_model = YOLO(str(teacher_model_path))

    # External images cache directory
    ext_cache_dir = repo_root / "data" / "usdetrac" / "images_cache"
    ext_cache_dir.mkdir(parents=True, exist_ok=True)

    completed_75_summary = []
    total_75_source_boxes = 0
    total_75_teacher_additions = 0
    total_75_logged_conflicts = 0
    r75_class_counts: Counter = Counter()

    for idx, r in enumerate(selected_75_records, 1):
        c_src = r["canonical_source_id"]
        raw_file = r["image_path"]
        nd_entry = ndjson_map[raw_file]
        url = nd_entry["url"]
        exp_w = int(nd_entry.get("width", 640))
        exp_h = int(nd_entry.get("height", 640))

        # Destination paths in R100
        img_dst = dataset_r100_dir / "images" / "train" / f"{c_src}.jpg"
        lbl_dst = dataset_r100_dir / "labels" / "train" / f"{c_src}.txt"
        cached_img = ext_cache_dir / f"{c_src}.jpg"

        # Download / retrieve cached image
        if cached_img.exists():
            shutil.copy2(cached_img, img_dst)
            success, err_msg = True, None
        else:
            success, err_msg = download_and_verify_image(
                url=url, dest_path=img_dst, expected_w=exp_w, expected_h=exp_h
            )
            if success:
                shutil.copy2(img_dst, cached_img)

        if not success:
            raise RuntimeError(f"Failed downloading {c_src} ({raw_file}): {err_msg}")

        # Parse and remap raw source boxes from NDJSON:
        # 0: bus -> bus(2); 1: car, 2: truck, 3: van -> car(0)
        raw_boxes = nd_entry.get("annotations", {}).get("boxes", [])
        remapped_source_boxes: List[Dict[str, Any]] = []

        for b in raw_boxes:
            src_cid, xc, yc, bw, bh = b
            if src_cid not in UADETRAC_RAW_TO_REMAPPED_THAI5:
                continue
            remapped_cid = UADETRAC_RAW_TO_REMAPPED_THAI5[src_cid]
            remapped_source_boxes.append({
                "class_id": remapped_cid,
                "class_name": THAI_5CLASS_NAMES[remapped_cid],
                "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                "source_class_id": src_cid,
                "source_class_name": {0: "bus", 1: "car", 2: "truck", 3: "van"}.get(src_cid, str(src_cid)),
                "provenance": f"raw_ndjson_remapped({src_cid}->{remapped_cid})",
                "is_teacher_addition": False,
            })

        # Run teacher inference (full-frame + tiled)
        img_bgr = cv2.imread(str(img_dst))
        teacher_proposals = run_teacher_on_frame(
            model=teacher_model,
            img_bgr=img_bgr,
            device=device,
            conf_thresh=0.20,
            full_frame_imgsz=1280,
            tile_size=384,
            tile_overlap=0.20,
        )

        # Merge proposals with source boxes
        final_boxes, additions, conflicts = merge_teacher_proposals_with_source_boxes(
            source_boxes=remapped_source_boxes,
            teacher_proposals=teacher_proposals,
            img_w=exp_w,
            img_h=exp_h,
        )

        # Write labels to R100
        lbl_lines = []
        for b in final_boxes:
            cid = b["class_id"]
            xc, yc, bw, bh = b["bbox_norm"]
            cname = THAI_5CLASS_NAMES[cid]
            r75_class_counts[cname] += 1
            lbl_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        lbl_dst.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")

        total_75_source_boxes += len(remapped_source_boxes)
        total_75_teacher_additions += len(additions)
        total_75_logged_conflicts += len(conflicts)

        # Render review overlay
        overlay_dst = overlays_dir / f"{c_src}_completed_overlay.jpg"
        render_review_overlay(img_dst, final_boxes, overlay_dst)

        completed_75_summary.append({
            "canonical_source_id": c_src,
            "sequence_id": r["sequence_id"],
            "raw_file": raw_file,
            "source_boxes_count": len(remapped_source_boxes),
            "teacher_additions_count": len(additions),
            "total_boxes_count": len(final_boxes),
            "conflicts_count": len(conflicts),
            "conflicts": conflicts,
            "label_file": str(lbl_dst.relative_to(repo_root)),
            "overlay_file": str(overlay_dst.relative_to(repo_root)),
        })

        if idx % 15 == 0 or idx == 75:
            print(f"  [PROGRESS] {idx}/75 frames completed: {total_75_source_boxes} source boxes, +{total_75_teacher_additions} teacher additions.")

    print(f"  [OK] 75 frames machine-completed:")
    print(f"       Source boxes: {total_75_source_boxes}")
    print(f"       Teacher additions: +{total_75_teacher_additions}")
    print(f"       Total boxes in 75: {total_75_source_boxes + total_75_teacher_additions}")
    print(f"       Logged conflicts: {total_75_logged_conflicts}")
    print(f"       75-frame class distribution: {dict(r75_class_counts)}")

    # 6. Verification and Invariant Checks
    print("\n[STEP 5/5] Verifying Dataset Invariants & Canonical Disjointness...")

    # Canonical source sets
    r25_canons = reviewed_25_canons
    r100_canons = reviewed_25_canons.union(set(r["canonical_source_id"] for r in selected_75_records))
    local_canons = set(inv_map[rid]["canonical_source_id"] for rid in local_train_ids)
    val_canons = set(inv_map[rid]["canonical_source_id"] for rid in val_ids)

    # Invariant assertions
    assert len(r25_canons) == 25, "R25 external canonical count must be 25"
    assert len(r100_canons) == 100, "R100 external canonical count must be 100"
    assert r25_canons.issubset(r100_canons), "R25 external frames must be strictly nested inside R100"
    assert len(r100_canons & local_canons) == 0, "External frames must not overlap with local training frames"
    assert len(r100_canons & val_canons) == 0, "External frames must not overlap with validation frames"
    assert len(r100_canons & diagnostic_sources) == 0, "External frames must not overlap with diagnostic benchmark"
    assert len(local_canons & val_canons) == 0, "Local training frames must not overlap with validation frames"
    assert len(local_canons & diagnostic_sources) == 0, "Local training frames must not overlap with diagnostic benchmark"

    # Write data.yaml for both datasets
    for ds_dir, ds_name in [(dataset_r25_dir, "Dataset R25 (Remapped Reviewed External)"),
                            (dataset_r100_dir, "Dataset R100 (Remapped + Expanded External)")]:
        yaml_content = f"""# YOLO Training Specification - {ds_name}
path: {ds_dir.as_posix()}
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
  4: three_wheeler
"""
        (ds_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

        val_manifest_obj = {
            "manifest_name": "Primary Validation Benchmark (130 Frames)",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_samples": len(val_manifest_samples),
            "samples": val_manifest_samples,
        }
        with open(ds_dir / "primary_val_manifest.json", "w", encoding="utf-8") as f:
            json.dump(val_manifest_obj, f, indent=2)

    total_r25_boxes = local_box_count + r25_box_count
    total_r100_boxes = local_box_count + r25_box_count + total_75_source_boxes + total_75_teacher_additions

    # Save summary metadata
    manifest_meta = {
        "dataset_r25": {
            "train_images_total": len(local_train_ids) + 25,
            "local_train_images": len(local_train_ids),
            "external_train_images": 25,
            "train_boxes_total": total_r25_boxes,
            "local_boxes": local_box_count,
            "external_boxes": r25_box_count,
            "external_class_counts": dict(r25_class_counts),
            "truck_boxes_remapped_to_car": len(truck_to_car_remapping_log),
            "val_images": len(val_ids),
            "val_boxes": val_box_count,
        },
        "dataset_r100": {
            "train_images_total": len(local_train_ids) + 100,
            "local_train_images": len(local_train_ids),
            "external_train_images": 100,
            "train_boxes_total": total_r100_boxes,
            "local_boxes": local_box_count,
            "external_boxes_total": r25_box_count + total_75_source_boxes + total_75_teacher_additions,
            "reviewed_external_boxes": r25_box_count,
            "machine_completed_75_source_boxes": total_75_source_boxes,
            "machine_completed_75_teacher_additions": total_75_teacher_additions,
            "machine_completed_75_conflicts": total_75_logged_conflicts,
            "val_images": len(val_ids),
            "val_boxes": val_box_count,
        },
        "truck_to_car_remapping_log": truck_to_car_remapping_log,
        "machine_completed_75_records": completed_75_summary,
    }

    with open(dataset_r25_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_meta["dataset_r25"], f, indent=2)
    with open(dataset_r100_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_meta["dataset_r100"], f, indent=2)
    with open(dataset_r100_dir / "external_expansion_audit.json", "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2)

    print("\n" + "=" * 75)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"  Dataset R25:  {len(local_train_ids) + 25} train images ({total_r25_boxes} boxes) | 130 val images")
    print(f"  Dataset R100: {len(local_train_ids) + 100} train images ({total_r100_boxes} boxes) | 130 val images")
    print(f"  External Expansion Delta: +75 frames (+{total_75_source_boxes + total_75_teacher_additions} boxes)")
    print("=" * 75)

    return manifest_meta


def main():
    parser = argparse.ArgumentParser(description="External remapping and expansion dataset builder.")
    parser.add_argument("--r25-dir", type=str, default="data/experiment_remapped_r25")
    parser.add_argument("--r100-dir", type=str, default="data/experiment_remapped_r100")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    prepare_external_remapped_and_expanded_datasets(
        repo_root=REPO_ROOT,
        dataset_r25_dir=Path(args.r25_dir),
        dataset_r100_dir=Path(args.r100_dir),
        manifests_v6_dir=REPO_ROOT / "data" / "training_manifests_v6",
        consolidated_v3_dir=REPO_ROOT / "data" / "review_pack_consolidated_v3",
        snapshot_dir=REPO_ROOT / "data" / "eval_snapshot_v1",
        ndjson_path=REPO_ROOT / "data" / "usdetrac" / "ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson",
        teacher_model_path=REPO_ROOT / "models" / "yolo26x.pt",
        overlays_dir=REPO_ROOT / "data" / "machine_completion_overlays_r75",
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
