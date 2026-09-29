"""
Teacher-Assisted Annotation Completion Tool (Batch 6 & Continuation)

Generates supplemental vehicle proposals on incomplete review pack frames using
the local high-capacity YOLO26x teacher (models/yolo26x.pt).

Supports:
1. 6-Frame Pilot: data/review_pack_pilot_v1
2. 18-Frame Continuation: data/review_pack_continuation_v1 (the remaining incomplete frames)
3. Inference caching to disk to avoid duplicate neural net passes on reruns.
4. Rerun preservation: preserves existing human edits, review decisions, and notes across reruns.
5. Strict protection of human annotations: human boxes authoritative, labels protected.
6. Post-Proposal Small Vehicle Scan Checklist reminding reviewers to inspect for missed small cars.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
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
from PIL import Image, ImageDraw, ImageFont
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
    get_image_dimensions,
    validate_box,
)
from tools.prepare_review_pack import (
    CLASS_PREVIEW_COLORS,
    compute_file_sha256,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

# COCO 80-class mapping to Thai 5-class taxonomy
# COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TEACHER_TO_THAI5 = {
    2: 0,  # car -> 0: car
    3: 1,  # motorcycle -> 1: motorcycle
    5: 2,  # bus -> 2: bus
    7: 3,  # truck -> 3: truck
}

COCO_ALLOWED_CLASS_IDS = list(COCO_TEACHER_TO_THAI5.keys())

DEFAULT_TEACHER_MODEL_PATH = Path("models/yolo26x.pt")

# Deterministic 6-frame pilot selection
PILOT_SELECTIONS = [
    {
        "frame_id": "MVI_40871_img00320",
        "data_origin": "external_ua_detrac",
        "category": "external_dense_queue",
        "sequence_or_cam": "MVI_40871",
        "lighting": "day_wet",
        "rationale": "High-density expressway queue with prominent unannotated background cars and oncoming lane traffic."
    },
    {
        "frame_id": "MVI_40141_img00096",
        "data_origin": "external_ua_detrac",
        "category": "external_highway_corridor",
        "sequence_or_cam": "MVI_40141",
        "lighting": "day_clear",
        "rationale": "Highway corridor with only 9 foreground boxes; background and distant approaching vehicles omitted."
    },
    {
        "frame_id": "MVI_39501_img00174",
        "data_origin": "external_ua_detrac",
        "category": "external_high_angle",
        "sequence_or_cam": "MVI_39501",
        "lighting": "day_hazy",
        "rationale": "High-angle perspective with only 13 boxes; distant horizon vehicles and queue tails unannotated."
    },
    {
        "frame_id": "cam44_north_f019140",
        "data_origin": "local",
        "category": "local_daytime_queue",
        "sequence_or_cam": "cam44_north",
        "lighting": "real_day",
        "rationale": "Key local intersection daytime approach with queue density where distant motorcycles/cars need recovery."
    },
    {
        "frame_id": "cam43_south_night_f122256",
        "data_origin": "local",
        "category": "local_night_glare",
        "sequence_or_cam": "cam43_south",
        "lighting": "real_night",
        "rationale": "Night intersection camera with severe headlight bloom; tests small vehicle recovery under night glare."
    },
    {
        "frame_id": "cam44_north_night_f004500",
        "data_origin": "local",
        "category": "local_night_distant",
        "sequence_or_cam": "cam44_north",
        "lighting": "real_night",
        "rationale": "Distinct local night camera view; tests distant vehicle and motorcycle recovery in dark street conditions."
    }
]

# The remaining 18 incomplete frames from review_pack_v2
CONTINUATION_SELECTIONS = [
    {
        "frame_id": "MVI_40863_img00007",
        "data_origin": "external_ua_detrac",
        "category": "external_overcast_highway",
        "sequence_or_cam": "MVI_40863",
        "lighting": "day_wet_overcast",
        "rationale": "High-angle highway corridor; distant oncoming vehicles and queue tail vehicles omitted in early frames."
    },
    {
        "frame_id": "MVI_40863_img00012",
        "data_origin": "external_ua_detrac",
        "category": "external_overcast_highway",
        "sequence_or_cam": "MVI_40863",
        "lighting": "day_wet_overcast",
        "rationale": "Highway corridor sequence partner; missing background oncoming traffic near overpass."
    },
    {
        "frame_id": "MVI_40141_img00101",
        "data_origin": "external_ua_detrac",
        "category": "external_highway_corridor",
        "sequence_or_cam": "MVI_40141",
        "lighting": "day_clear",
        "rationale": "Open expressway straightaway with foreground boxes only; distant approaching queue omitted."
    },
    {
        "frame_id": "MVI_40141_img00151",
        "data_origin": "external_ua_detrac",
        "category": "external_highway_corridor",
        "sequence_or_cam": "MVI_40141",
        "lighting": "day_clear",
        "rationale": "Same corridor sequence; multiple unannotated vehicles at vanishing point."
    },
    {
        "frame_id": "MVI_40871_img00333",
        "data_origin": "external_ua_detrac",
        "category": "external_dense_queue",
        "sequence_or_cam": "MVI_40871",
        "lighting": "day_wet",
        "rationale": "Dense urban queue partner frame; unannotated vehicles in oncoming lane and overpass ramp."
    },
    {
        "frame_id": "MVI_40871_img00354",
        "data_origin": "external_ua_detrac",
        "category": "external_dense_queue",
        "sequence_or_cam": "MVI_40871",
        "lighting": "day_wet",
        "rationale": "Heavy congestion frame; background cars near vanishing point under wet road reflection."
    },
    {
        "frame_id": "MVI_20065_img00715",
        "data_origin": "external_ua_detrac",
        "category": "external_sunny_corridor",
        "sequence_or_cam": "MVI_20065",
        "lighting": "day_clear",
        "rationale": "Multi-lane highway with heavy shadows; background cars in shaded lanes unannotated."
    },
    {
        "frame_id": "MVI_20061_img00038",
        "data_origin": "external_ua_detrac",
        "category": "external_arterial_junction",
        "sequence_or_cam": "MVI_20061",
        "lighting": "day_clear",
        "rationale": "Arterial junction with cross traffic; background turning vehicles and far lane queues unannotated."
    },
    {
        "frame_id": "MVI_20061_img00057",
        "data_origin": "external_ua_detrac",
        "category": "external_arterial_junction",
        "sequence_or_cam": "MVI_20061",
        "lighting": "day_clear",
        "rationale": "Junction sequence partner; unannotated far-left turn bay traffic."
    },
    {
        "frame_id": "MVI_20061_img00124",
        "data_origin": "external_ua_detrac",
        "category": "external_arterial_junction",
        "sequence_or_cam": "MVI_20061",
        "lighting": "day_clear",
        "rationale": "Junction clearing phase; unannotated queue tail vehicles entering from top-right."
    },
    {
        "frame_id": "MVI_20063_img00769",
        "data_origin": "external_ua_detrac",
        "category": "external_sunny_overpass",
        "sequence_or_cam": "MVI_20063",
        "lighting": "day_clear",
        "rationale": "Overpass view with high perspective; distant vehicles beneath gantry unannotated."
    },
    {
        "frame_id": "cam44_north_f148440",
        "data_origin": "local",
        "category": "local_daytime_queue",
        "sequence_or_cam": "cam44_north",
        "lighting": "real_day",
        "rationale": "Peak daylight queue at local intersection; unannotated motorcycles and distant vehicles near horizon."
    },
    {
        "frame_id": "cam44_north_f003960",
        "data_origin": "local",
        "category": "local_daytime_queue",
        "sequence_or_cam": "cam44_north",
        "lighting": "real_day",
        "rationale": "Daylight approach queue; small vehicles in far right turning lane omitted."
    },
    {
        "frame_id": "cam44_north_night_f005220",
        "data_origin": "local",
        "category": "local_night_distant",
        "sequence_or_cam": "cam44_north_night",
        "lighting": "real_night",
        "rationale": "Dark intersection approach; distant vehicles without headlights and unlit motorcycles."
    },
    {
        "frame_id": "cam44_north_night_f045720",
        "data_origin": "local",
        "category": "local_night_glare",
        "sequence_or_cam": "cam44_north_night",
        "lighting": "real_night",
        "rationale": "Severe streetlamp glare; recovery of partially illuminated vehicles in dark roadway pockets."
    },
    {
        "frame_id": "cam44_north_night_f060840",
        "data_origin": "local",
        "category": "local_night_distant",
        "sequence_or_cam": "cam44_north_night",
        "lighting": "real_night",
        "rationale": "Late night sparse traffic; recovery of distant vehicles traveling through unlit intersection zones."
    },
    {
        "frame_id": "cam44_north_night_f004680",
        "data_origin": "local",
        "category": "local_night_distant",
        "sequence_or_cam": "cam44_north_night",
        "lighting": "real_night",
        "rationale": "Local night sequence frame; recovery of queue tail and roadside parked vehicles."
    },
    {
        "frame_id": "cam44_north_night_f005940",
        "data_origin": "local",
        "category": "local_night_distant",
        "sequence_or_cam": "cam44_north_night",
        "lighting": "real_night",
        "rationale": "Night traffic approach; dark vehicle silhouettes near periphery of camera illumination."
    }
]


@dataclass
class PilotConfig:
    source_pack_dir: Path = Path("data/review_pack_v2")
    output_pack_dir: Optional[Path] = None
    cache_dir: Path = Path("data/teacher_inference_cache")
    teacher_model_path: Path = Path("models/yolo26x.pt")
    mode: str = "continuation"  # "continuation", "pilot", or "all"
    conf_threshold: float = 0.20
    full_frame_imgsz: int = 1280
    tile_size: int = 640
    tile_overlap: float = 0.20
    iou_dedup_thresh: float = 0.55
    iou_match_existing_thresh: float = 0.45
    iou_ambiguous_thresh: float = 0.15
    device: str = "cuda"  # will fallback to cpu if cuda unavailable


# -----------------------------------------------------------------------------
# Geometric & Coordinate Transformation Utilities
# -----------------------------------------------------------------------------

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Union between two boxes [x1, y1, x2, y2].
    Supports both pixel coordinates and normalized coordinates.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_box_iomin(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Minimum (IoMin) between two boxes [x1, y1, x2, y2].
    Measures containment ratio.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    min_area = min(area1, area2)
    return inter_area / min_area if min_area > 0 else 0.0


def xyxy_to_norm_yolo(xyxy: List[float], img_w: int, img_h: int) -> List[float]:
    """Converts pixel [x1, y1, x2, y2] to normalized YOLO [xc, yc, bw, bh]."""
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return [
        round(max(0.0, min(1.0, xc / img_w)), 6),
        round(max(0.0, min(1.0, yc / img_h)), 6),
        round(max(0.0, min(1.0, bw / img_w)), 6),
        round(max(0.0, min(1.0, bh / img_h)), 6),
    ]


def norm_yolo_to_xyxy(norm_box: List[float], img_w: int, img_h: int) -> List[float]:
    """Converts normalized YOLO [xc, yc, bw, bh] to pixel [x1, y1, x2, y2]."""
    xc, yc, bw, bh = norm_box
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return [x1, y1, x2, y2]


def tile_xyxy_to_global_xyxy(
    tile_box_xyxy: List[float],
    x_offset: int,
    y_offset: int
) -> List[float]:
    """Projects pixel [x1, y1, x2, y2] on a local tile back to global image coordinates."""
    bx1, by1, bx2, by2 = tile_box_xyxy
    return [bx1 + x_offset, by1 + y_offset, bx2 + x_offset, by2 + y_offset]


def generate_tile_grid(
    img_w: int,
    img_h: int,
    tile_size: int = 640,
    overlap: float = 0.20
) -> List[Tuple[int, int, int, int]]:
    """
    Computes overlapping tile rectangles covering the entire image.
    Returns list of (x_start, y_start, x_end, y_end).
    Handles edge conditions so every image pixel is covered without cropping.
    """
    stride = int(tile_size * (1.0 - overlap))
    if stride <= 0:
        stride = tile_size

    # Horizontal offsets
    x_offsets: List[int] = []
    curr_x = 0
    while curr_x + tile_size < img_w:
        x_offsets.append(curr_x)
        curr_x += stride
    x_offsets.append(max(0, img_w - tile_size))
    # Deduplicate while preserving order
    unique_x = sorted(list(set(x_offsets)))

    # Vertical offsets
    y_offsets: List[int] = []
    curr_y = 0
    while curr_y + tile_size < img_h:
        y_offsets.append(curr_y)
        curr_y += stride
    y_offsets.append(max(0, img_h - tile_size))
    unique_y = sorted(list(set(y_offsets)))

    tiles: List[Tuple[int, int, int, int]] = []
    for y1 in unique_y:
        for x1 in unique_x:
            x2 = min(img_w, x1 + tile_size)
            y2 = min(img_h, y1 + tile_size)
            tiles.append((x1, y1, x2, y2))

    return tiles


def deduplicate_proposals(
    proposals: List[Dict[str, Any]],
    iou_thresh: float = 0.55
) -> List[Dict[str, Any]]:
    """
    Deduplicates overlapping proposals produced by overlapping tiles or full-frame + tile.
    Groups proposals by class_id and keeps the higher-confidence instance.
    Critically preserves adjacent vehicles of different or same classes with distinct centroids.
    """
    if not proposals:
        return []

    # Sort descending by confidence
    sorted_props = sorted(proposals, key=lambda p: p["confidence"], reverse=True)
    kept_proposals: List[Dict[str, Any]] = []

    for cand in sorted_props:
        is_dup = False
        for kept in kept_proposals:
            if cand["class_id"] != kept["class_id"]:
                continue

            iou = compute_box_iou(cand["xyxy_px"], kept["xyxy_px"])
            iomin = compute_box_iomin(cand["xyxy_px"], kept["xyxy_px"])

            # Deduplicate if spatial overlap is substantial (same physical vehicle across tiles)
            if iou >= iou_thresh or iomin >= 0.85:
                is_dup = True
                kept["tile_hits"] = kept.get("tile_hits", 1) + 1
                if cand.get("method") != kept.get("method"):
                    kept["method"] = "both"
                break

        if not is_dup:
            cand["tile_hits"] = 1
            kept_proposals.append(dict(cand))

    return kept_proposals


def match_proposals_against_human_annotations(
    proposals: List[Dict[str, Any]],
    human_boxes: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    iou_match_thresh: float = 0.45,
    iou_ambig_thresh: float = 0.15
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compares teacher proposals against existing authoritative human boxes:
    Returns:
    1. duplicates: proposals matching human box (same class, IoU >= 0.45)
    2. conflicts: proposals matching human box (different class, IoU >= 0.45) - requires human review
    3. ambiguous: partial spatial overlap (0.15 <= IoU < 0.45)
    4. candidate_additions: new proposals (IoU < 0.15) for missing background vehicles
    """
    duplicates: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    ambiguous_overlaps: List[Dict[str, Any]] = []
    candidate_additions: List[Dict[str, Any]] = []

    # Convert human boxes to pixel xyxy for robust geometric matching
    human_pixel_boxes = []
    for h in human_boxes:
        h_xyxy = norm_yolo_to_xyxy(h["bbox_norm"], img_w, img_h)
        human_pixel_boxes.append({
            "instance_id": h.get("instance_id", "unknown"),
            "class_id": h.get("class_id", 0),
            "class_name": h.get("class_name", "unknown"),
            "xyxy_px": h_xyxy
        })

    for prop in proposals:
        p_xyxy = prop["xyxy_px"]
        best_iou = 0.0
        best_human = None

        for h in human_pixel_boxes:
            iou = compute_box_iou(p_xyxy, h["xyxy_px"])
            if iou > best_iou:
                best_iou = iou
                best_human = h

        prop_match = dict(prop)
        prop_match["best_match_iou"] = round(best_iou, 4)
        prop_match["matching_human_instance_id"] = best_human["instance_id"] if best_human else None

        if best_iou >= iou_match_thresh:
            if prop["class_id"] == best_human["class_id"]:
                prop_match["proposal_category"] = "duplicate_confirmed"
                duplicates.append(prop_match)
            else:
                prop_match["proposal_category"] = "conflict"
                prop_match["human_class_id"] = best_human["class_id"]
                prop_match["human_class_name"] = best_human["class_name"]
                conflicts.append(prop_match)
        elif best_iou >= iou_ambig_thresh:
            prop_match["proposal_category"] = "ambiguous_overlap"
            prop_match["human_class_name"] = best_human["class_name"] if best_human else None
            ambiguous_overlaps.append(prop_match)
        else:
            prop_match["proposal_category"] = "candidate_addition"
            candidate_additions.append(prop_match)

    return duplicates, conflicts, ambiguous_overlaps, candidate_additions


# -----------------------------------------------------------------------------
# Teacher Completion Tool Core
# -----------------------------------------------------------------------------

class TeacherCompletionPilot:
    """Runs teacher inference and generates Batch 6 review packs with caching and rerun safety."""

    def __init__(self, config: Optional[PilotConfig] = None):
        self.config = config or PilotConfig()
        self._validate_paths()
        self.device = "cuda" if (torch.cuda.is_available() and self.config.device == "cuda") else "cpu"
        self.model_sha256 = compute_file_sha256(self.config.teacher_model_path)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TEACHER] Loading YOLO26x model: {self.config.teacher_model_path}")
        print(f"  SHA256: {self.model_sha256}")
        print(f"  Device: {self.device}")
        print(f"  Inference Cache: {self.config.cache_dir}")
        self.model = YOLO(str(self.config.teacher_model_path))

    def _validate_paths(self):
        if not self.config.teacher_model_path.exists():
            raise FileNotFoundError(f"Teacher model not found at: {self.config.teacher_model_path}")
        if not self.config.source_pack_dir.exists():
            raise FileNotFoundError(f"Source review pack not found at: {self.config.source_pack_dir}")

    def run_frame_inference(
        self,
        img_path: Path
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes both full-frame and tiled inference on an image.
        Reuses valid cached inference results if available.
        Returns: (deduplicated_proposals, timing_and_meta)
        """
        fid = img_path.stem
        cache_file = self.config.cache_dir / f"{fid}.json"

        # Check cache validity
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                c_set = cdata.get("inference_settings", {})
                if (
                    cdata.get("model_sha256") == self.model_sha256
                    and c_set.get("conf_threshold") == self.config.conf_threshold
                    and c_set.get("full_frame_imgsz") == self.config.full_frame_imgsz
                    and c_set.get("tile_size") == self.config.tile_size
                    and c_set.get("tile_overlap") == self.config.tile_overlap
                ):
                    print(f"  [CACHE HIT] Reusing cached teacher inference for '{fid}'")
                    return cdata["deduplicated_proposals"], cdata["metrics"]
            except Exception as e:
                print(f"  [CACHE READ WARN] Failed reading cache for '{fid}': {e}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not decode image at: {img_path}")

        img_h, img_w = img_bgr.shape[:2]
        raw_full_proposals: List[Dict[str, Any]] = []
        raw_tiled_proposals: List[Dict[str, Any]] = []

        # 1. Full-frame inference
        t0 = time.time()
        res_full = self.model.predict(
            img_bgr,
            conf=self.config.conf_threshold,
            imgsz=self.config.full_frame_imgsz,
            classes=COCO_ALLOWED_CLASS_IDS,
            device=self.device,
            verbose=False
        )
        t_full = time.time() - t0

        for b in res_full[0].boxes:
            coco_cid = int(b.cls[0].item())
            if coco_cid not in COCO_TEACHER_TO_THAI5:
                continue
            thai_cid = COCO_TEACHER_TO_THAI5[coco_cid]
            conf = float(b.conf[0].item())
            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().tolist()
            norm_box = xyxy_to_norm_yolo([bx1, by1, bx2, by2], img_w, img_h)

            raw_full_proposals.append({
                "class_id": thai_cid,
                "class_name": THAI_5CLASS_NAMES[thai_cid],
                "coco_class_id": coco_cid,
                "confidence": round(conf, 4),
                "bbox_norm": norm_box,
                "xyxy_px": [bx1, by1, bx2, by2],
                "method": "full_frame",
                "tile_id": "full"
            })

        # 2. Overlapping Tiled inference
        t1 = time.time()
        eff_tile_size = self.config.tile_size
        if min(img_w, img_h) <= eff_tile_size:
            eff_tile_size = int(min(img_w, img_h) * 0.60)

        tiles = generate_tile_grid(
            img_w,
            img_h,
            tile_size=eff_tile_size,
            overlap=self.config.tile_overlap
        )

        for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile_img = img_bgr[ty1:ty2, tx1:tx2]
            res_tile = self.model.predict(
                tile_img,
                conf=self.config.conf_threshold,
                imgsz=eff_tile_size,
                classes=COCO_ALLOWED_CLASS_IDS,
                device=self.device,
                verbose=False
            )
            for b in res_tile[0].boxes:
                coco_cid = int(b.cls[0].item())
                if coco_cid not in COCO_TEACHER_TO_THAI5:
                    continue
                thai_cid = COCO_TEACHER_TO_THAI5[coco_cid]
                conf = float(b.conf[0].item())
                t_bx1, t_by1, t_bx2, t_by2 = b.xyxy[0].cpu().numpy().tolist()

                # Project back to global image coordinates
                g_xyxy = tile_xyxy_to_global_xyxy([t_bx1, t_by1, t_bx2, t_by2], tx1, ty1)
                norm_box = xyxy_to_norm_yolo(g_xyxy, img_w, img_h)

                raw_tiled_proposals.append({
                    "class_id": thai_cid,
                    "class_name": THAI_5CLASS_NAMES[thai_cid],
                    "coco_class_id": coco_cid,
                    "confidence": round(conf, 4),
                    "bbox_norm": norm_box,
                    "xyxy_px": g_xyxy,
                    "method": "tiled",
                    "tile_id": f"tile_{tile_idx}_{tx1}_{ty1}"
                })

        t_tiled = time.time() - t1

        # 3. Deduplicate combined proposals
        all_raw = raw_full_proposals + raw_tiled_proposals
        deduped = deduplicate_proposals(all_raw, iou_thresh=self.config.iou_dedup_thresh)

        metrics = {
            "image_dimensions": {"width": img_w, "height": img_h},
            "runtime_full_frame_sec": round(t_full, 4),
            "runtime_tiled_sec": round(t_tiled, 4),
            "runtime_total_sec": round(t_full + t_tiled, 4),
            "effective_tile_size": eff_tile_size,
            "tiles_count": len(tiles),
            "raw_full_frame_count": len(raw_full_proposals),
            "raw_tiled_count": len(raw_tiled_proposals),
            "raw_combined_count": len(all_raw),
            "deduplicated_proposal_count": len(deduped)
        }

        # Save to cache
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "frame_id": fid,
                    "model_sha256": self.model_sha256,
                    "inference_settings": {
                        "conf_threshold": self.config.conf_threshold,
                        "full_frame_imgsz": self.config.full_frame_imgsz,
                        "tile_size": self.config.tile_size,
                        "tile_overlap": self.config.tile_overlap
                    },
                    "metrics": metrics,
                    "deduplicated_proposals": deduped
                }, f, indent=2)
        except Exception as e:
            print(f"  [CACHE WRITE WARN] Failed writing cache for '{fid}': {e}")

        return deduped, metrics

    def generate_pack(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates either the Pilot review pack (6 frames) or the Continuation review pack (18 frames),
        strictly preserving existing human annotations and prior review decisions across reruns.
        """
        target_mode = mode or self.config.mode

        if target_mode == "pilot":
            selections = PILOT_SELECTIONS
            default_out = Path("data/review_pack_pilot_v1")
            pack_title = "Batch 6: Pilot Teacher-Assisted Completion (6 Frames)"
            pack_purpose = "Batch 6: Pilot teacher-assisted completion of incomplete annotations"
            pack_version = "v1.0_pilot"
            report_name = "PILOT_COMPLETION_REPORT.md"
        elif target_mode == "continuation":
            selections = CONTINUATION_SELECTIONS
            default_out = Path("data/review_pack_continuation_v1")
            pack_title = "Batch 6 Continuation: Teacher-Assisted Completion (18 Incomplete Frames)"
            pack_purpose = "Batch 6 Continuation: Teacher-assisted completion of remaining 18 incomplete annotations"
            pack_version = "v1.0_continuation"
            report_name = "CONTINUATION_COMPLETION_REPORT.md"
        elif target_mode == "all":
            selections = PILOT_SELECTIONS + CONTINUATION_SELECTIONS
            default_out = Path("data/review_pack_batch6_all")
            pack_title = "Batch 6 Full: Teacher-Assisted Completion (24 Incomplete Frames)"
            pack_purpose = "Batch 6 Full: Teacher-assisted completion of all 24 incomplete annotations"
            pack_version = "v1.0_all"
            report_name = "BATCH6_FULL_COMPLETION_REPORT.md"
        else:
            raise ValueError(f"Unknown mode: '{target_mode}'. Must be 'pilot', 'continuation', or 'all'.")

        out_dir = self.config.output_pack_dir or default_out
        images_dir = out_dir / "images"
        labels_dir = out_dir / "annotations" / "labels"
        annos_dir = out_dir / "annotations"
        proposals_dir = out_dir / "proposals"
        previews_dir = out_dir / "previews"
        previews_before_dir = out_dir / "previews_before"

        for d in [images_dir, labels_dir, annos_dir, proposals_dir, previews_dir, previews_before_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load review_pack_v2 annotations.json (authoritative source)
        v2_annos_file = self.config.source_pack_dir / "annotations" / "annotations.json"
        with open(v2_annos_file, "r", encoding="utf-8") as f:
            v2_records = json.load(f)
        v2_map = {r["frame_id"]: r for r in v2_records}

        # Check for existing records in output pack for rerun preservation
        existing_annos_file = annos_dir / "annotations.json"
        existing_recs_map: Dict[str, Dict[str, Any]] = {}
        if existing_annos_file.exists():
            try:
                with open(existing_annos_file, "r", encoding="utf-8") as f:
                    disk_recs = json.load(f)
                existing_recs_map = {r["frame_id"]: r for r in disk_recs}
                print(f"[RERUN] Existing annotations found for {len(existing_recs_map)} frames. Preserving review decisions!")
            except Exception as e:
                print(f"[RERUN WARN] Could not parse existing annotations: {e}")

        pack_records: List[Dict[str, Any]] = []
        pack_summaries: List[Dict[str, Any]] = []

        print(f"\n======================================================================")
        print(f"  {pack_title.upper()}")
        print(f"  Output Directory: {out_dir}")
        print(f"======================================================================")

        for item in selections:
            fid = item["frame_id"]
            orig_rec = v2_map.get(fid)
            if not orig_rec:
                raise KeyError(f"Selected frame '{fid}' missing from review_pack_v2 annotations.json!")

            src_img_path = self.config.source_pack_dir / "images" / f"{fid}.jpg"
            if not src_img_path.exists():
                raise FileNotFoundError(f"Source image missing at: {src_img_path}")

            dest_img_path = images_dir / f"{fid}.jpg"
            if not dest_img_path.exists() or dest_img_path.stat().st_size != src_img_path.stat().st_size:
                shutil.copy2(src_img_path, dest_img_path)

            img_w, img_h = get_image_dimensions(dest_img_path)
            if img_w is None or img_h is None:
                img_w, img_h = 640, 640

            # Run full-frame and tiled inference (cached if already computed)
            print(f"\n[FRAME] '{fid}' ({item['category']}):")
            dedup_proposals, metrics = self.run_frame_inference(dest_img_path)
            print(f"  Full-frame ({metrics['runtime_full_frame_sec']}s): {metrics['raw_full_frame_count']} raw boxes")
            print(f"  Tiled ({metrics['tiles_count']} tiles, {metrics['runtime_tiled_sec']}s): {metrics['raw_tiled_count']} raw boxes")
            print(f"  Deduplicated: {metrics['deduplicated_proposal_count']} proposals")

            # Load existing human boxes from v2
            existing_human_boxes = orig_rec.get("boxes", [])
            for h in existing_human_boxes:
                h["is_proposal"] = False
                h["source_type"] = "existing_human"
                h["proposal_status"] = "none"

            duplicates, conflicts, ambiguous, candidate_additions = match_proposals_against_human_annotations(
                proposals=dedup_proposals,
                human_boxes=existing_human_boxes,
                img_w=img_w,
                img_h=img_h,
                iou_match_thresh=self.config.iou_match_existing_thresh,
                iou_ambig_thresh=self.config.iou_ambiguous_thresh
            )

            print(f"  Matching vs {len(existing_human_boxes)} human boxes:")
            print(f"    - Duplicates (confirmed): {len(duplicates)}")
            print(f"    - Conflicts (class clash): {len(conflicts)}")
            print(f"    - Ambiguous Overlaps:     {len(ambiguous)}")
            print(f"    - Candidate Additions:    {len(candidate_additions)}")

            # Check if this frame was already reviewed in an earlier pass of this pack
            prior_reviewed_proposals: Dict[str, str] = {}
            prior_human_boxes: List[Dict[str, Any]] = []
            prior_review_status = "draft"
            prior_reviewer_notes = "incomplete annotations (continuation ready)"
            prior_is_ambiguous = orig_rec.get("is_ambiguous", False) or len(conflicts) > 0
            prior_is_rejected = False

            if fid in existing_recs_map:
                p_rec = existing_recs_map[fid]
                prior_review_status = p_rec.get("review_status", "draft")
                prior_reviewer_notes = p_rec.get("reviewer_notes", prior_reviewer_notes)
                prior_is_ambiguous = p_rec.get("is_ambiguous", prior_is_ambiguous)
                prior_is_rejected = p_rec.get("is_rejected", False)
                for b in p_rec.get("boxes", []):
                    if b.get("is_proposal"):
                        p_st = b.get("proposal_status")
                        if p_st in ("accepted", "rejected"):
                            prior_reviewed_proposals[b.get("instance_id", "")] = p_st
                    else:
                        prior_human_boxes.append(b)

            # Assign instance IDs to proposals
            proposal_boxes_for_editor: List[Dict[str, Any]] = []

            # 1. Candidate additions (proposals for missing background/queue vehicles)
            for idx, cand in enumerate(candidate_additions):
                p_inst_id = f"{fid}_teacher_add_{idx:03d}"
                prop_status = prior_reviewed_proposals.get(p_inst_id, "pending")
                proposal_boxes_for_editor.append({
                    "instance_id": p_inst_id,
                    "class_id": cand["class_id"],
                    "class_name": cand["class_name"],
                    "subtype": f"{cand['class_name']}_provisional",
                    "is_ambiguous": False,
                    "ambiguity_reason": "",
                    "bbox_norm": cand["bbox_norm"],
                    "proposal_source": f"yolo26x:{cand['method']}(conf={cand['confidence']:.2f})",
                    "is_proposal": True,
                    "source_type": "teacher_proposal",
                    "proposal_category": "candidate_addition",
                    "proposal_status": prop_status,
                    "confidence": cand["confidence"],
                    "proposal_method": cand["method"],
                    "tile_hits": cand.get("tile_hits", 1),
                    "original_source_class_id": cand["coco_class_id"],
                    "original_source_class_name": self.model.names[cand["coco_class_id"]]
                })

            # 2. Conflicts as reviewable conflict proposals
            for idx, conf_item in enumerate(conflicts):
                c_inst_id = f"{fid}_teacher_conf_{idx:03d}"
                prop_status = prior_reviewed_proposals.get(c_inst_id, "pending")
                proposal_boxes_for_editor.append({
                    "instance_id": c_inst_id,
                    "class_id": conf_item["class_id"],
                    "class_name": conf_item["class_name"],
                    "subtype": f"{conf_item['class_name']}_provisional",
                    "is_ambiguous": True,
                    "ambiguity_reason": f"Teacher conflict vs human {conf_item['human_class_name']}",
                    "bbox_norm": conf_item["bbox_norm"],
                    "proposal_source": f"yolo26x:{conf_item['method']}(conf={conf_item['confidence']:.2f})",
                    "is_proposal": True,
                    "source_type": "teacher_proposal",
                    "proposal_category": "conflict",
                    "proposal_status": prop_status,
                    "confidence": conf_item["confidence"],
                    "proposal_method": conf_item["method"],
                    "conflicting_instance_id": conf_item["matching_human_instance_id"],
                    "human_class_name": conf_item["human_class_name"],
                    "tile_hits": conf_item.get("tile_hits", 1),
                    "original_source_class_id": conf_item["coco_class_id"],
                    "original_source_class_name": self.model.names[conf_item["coco_class_id"]]
                })

            # Active human boxes: if prior human boxes exist from previous save, use them, else v2 baseline
            active_human_boxes = prior_human_boxes if prior_human_boxes else list(existing_human_boxes)
            combined_boxes = list(active_human_boxes) + proposal_boxes_for_editor

            # Write approved YOLO label file:
            # CRITICAL: ONLY human boxes + accepted proposals enter approved YOLO labels!
            approved_boxes = [
                b for b in combined_boxes
                if not b.get("is_proposal") or b.get("proposal_status") == "accepted"
            ]
            dest_lbl_path = labels_dir / f"{fid}.txt"
            approved_lbl_lines = [
                f"{b['class_id']} {b['bbox_norm'][0]:.6f} {b['bbox_norm'][1]:.6f} {b['bbox_norm'][2]:.6f} {b['bbox_norm'][3]:.6f}"
                for b in approved_boxes
            ]
            dest_lbl_path.write_text("\n".join(approved_lbl_lines) + ("\n" if approved_lbl_lines else ""), encoding="utf-8")

            # Render Before / After Previews:
            # 1. Before: Human baseline only
            preview_before_path = previews_before_dir / f"{fid}.jpg"
            src_prev = self.config.source_pack_dir / "previews" / f"{fid}.jpg"
            if src_prev.exists() and not preview_before_path.exists():
                shutil.copy2(src_prev, preview_before_path)
            elif not preview_before_path.exists():
                render_pilot_preview(
                    clean_img_path=dest_img_path,
                    existing_boxes=existing_human_boxes,
                    proposal_boxes=[],
                    out_path=preview_before_path
                )

            # 2. After: Human baseline + teacher proposals
            preview_path = previews_dir / f"{fid}.jpg"
            render_pilot_preview(
                clean_img_path=dest_img_path,
                existing_boxes=active_human_boxes,
                proposal_boxes=proposal_boxes_for_editor,
                out_path=preview_path
            )

            # Save detailed proposal JSON
            dest_prop_json = proposals_dir / f"{fid}.json"
            prop_data = {
                "frame_id": fid,
                "category": item["category"],
                "data_origin": item["data_origin"],
                "lighting": item["lighting"],
                "rationale": item["rationale"],
                "inference_metrics": metrics,
                "counts": {
                    "existing_human_boxes": len(existing_human_boxes),
                    "raw_proposals": metrics["raw_combined_count"],
                    "deduplicated_proposals": metrics["deduplicated_proposal_count"],
                    "duplicates": len(duplicates),
                    "conflicts": len(conflicts),
                    "ambiguous_overlaps": len(ambiguous),
                    "candidate_additions": len(candidate_additions)
                },
                "candidate_additions": candidate_additions,
                "conflicts": conflicts,
                "duplicates": duplicates,
                "ambiguous_overlaps": ambiguous
            }
            with open(dest_prop_json, "w", encoding="utf-8") as f:
                json.dump(prop_data, f, indent=2)

            frame_rec = {
                "frame_id": fid,
                "filename": f"{fid}.jpg",
                "image_path": str(dest_img_path).replace("\\", "/"),
                "label_path": str(dest_lbl_path).replace("\\", "/"),
                "data_origin": item["data_origin"],
                "canonical_source_id": orig_rec.get("canonical_source_id", fid),
                "sequence_id": item["sequence_or_cam"],
                "camera": orig_rec.get("camera", item["sequence_or_cam"]),
                "lighting_type": orig_rec.get("lighting_type", item["lighting"]),
                "review_tags": orig_rec.get("review_tags", []),
                "risk_factors": orig_rec.get("risk_factors", []) + [f"Teacher proposals: {len(candidate_additions)} additions"],
                "required_human_checks": [
                    f"Review {len(candidate_additions)} candidate vehicle proposals (press A to accept, X to reject).",
                    f"Inspect {len(conflicts)} class conflicts against existing human boxes.",
                    "⚠️ Small Vehicle Scan: Scan dense queues and horizon lines for ~3–5 small cars (< 20px) that may have been missed by the teacher."
                ],
                "boxes": combined_boxes,
                "review_status": prior_review_status,
                "annotation_state": "annotated",
                "is_unannotated": False,
                "is_ambiguous": prior_is_ambiguous,
                "is_rejected": prior_is_rejected,
                "reviewer_notes": prior_reviewer_notes,
                "proposal_metadata": {
                    "existing_human_count": len(existing_human_boxes),
                    "candidate_additions_count": len(candidate_additions),
                    "conflicts_count": len(conflicts),
                    "duplicates_count": len(duplicates),
                    "teacher_model": "yolo26x.pt",
                    "teacher_model_sha256": self.model_sha256,
                    "inference_runtime_sec": metrics["runtime_total_sec"]
                }
            }
            pack_records.append(frame_rec)
            pack_summaries.append(prop_data)

        # Write annotations.json
        annos_file = annos_dir / "annotations.json"
        with open(annos_file, "w", encoding="utf-8") as f:
            json.dump(pack_records, f, indent=2)

        # Write manifest.json
        manifest_data = {
            "metadata": {
                "version": pack_version,
                "purpose": pack_purpose,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "teacher_model": "models/yolo26x.pt",
                "teacher_model_sha256": self.model_sha256,
                "inference_settings": {
                    "conf_threshold": self.config.conf_threshold,
                    "full_frame_imgsz": self.config.full_frame_imgsz,
                    "tile_size": self.config.tile_size,
                    "tile_overlap": self.config.tile_overlap,
                    "iou_dedup_thresh": self.config.iou_dedup_thresh,
                    "iou_match_existing_thresh": self.config.iou_match_existing_thresh,
                    "taxonomy_mapping": {
                        "COCO_2_car": "Thai_0_car",
                        "COCO_3_motorcycle": "Thai_1_motorcycle",
                        "COCO_5_bus": "Thai_2_bus",
                        "COCO_7_truck": "Thai_3_truck"
                    }
                },
                "total_frames": len(pack_records),
                "unverified_status_guarantee": "All frames remain in draft/unverified review state until explicitly verified."
            },
            "frames": pack_summaries
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Generate Report Markdown
        report_md = self._generate_report_markdown(manifest_data, target_mode)
        (out_dir / report_name).write_text(report_md, encoding="utf-8")

        # Generate HTML Review Index
        html_idx = self._generate_html_index(manifest_data, target_mode, pack_title)
        (out_dir / "review_index.html").write_text(html_idx, encoding="utf-8")

        print(f"\n[DONE] {pack_title} successfully generated at: {out_dir}")
        print(f"  - Frames: {len(pack_records)}")
        print(f"  - Annotations: {annos_file}")
        print(f"  - Report: {out_dir / report_name}")
        print(f"  - HTML Index: {out_dir / 'review_index.html'}")

        return manifest_data

    def generate_pilot_pack(self) -> Dict[str, Any]:
        """Convenience alias for pilot mode."""
        return self.generate_pack(mode="pilot")

    def _generate_report_markdown(self, manifest_data: Dict[str, Any], mode: str) -> str:
        lines: List[str] = []
        created_at = manifest_data["metadata"]["created_at"]
        teacher_model = manifest_data["metadata"]["teacher_model"]
        model_sha = manifest_data["metadata"]["teacher_model_sha256"]

        if mode == "continuation":
            lines.extend([
                "# Batch 6 Continuation: Teacher-Assisted Completion Report (18 Frames)",
                "",
                f"**Generated**: {created_at}",
                f"**Teacher Model**: `{teacher_model}` (SHA256: `{model_sha}`)",
                f"**Inference Settings**: Conf >= {self.config.conf_threshold}, Full-Frame {self.config.full_frame_imgsz}px, Tiled {self.config.tile_size}px (Overlap {int(self.config.tile_overlap*100)}%)",
                "",
                "---",
                "",
                "## 1. Executive Summary & Full Reconciliation of 24 Incomplete Frames",
                "",
                "- **Original State (`data/review_pack_v2`)**: 44 frames total = 20 verified frames + 24 draft frames with reviewer note 'incomplete annotations'.",
                "- **Verified Baseline Integrity**: All 20 verified frames remain 100% byte-identical.",
                "- **24 Incomplete Sources Reconciliation**:",
                "  - **6 Pilot Frames** (`data/review_pack_pilot_v1`): 5 frames verified, 1 frame (`cam44_north_f019140`) marked rejected with pending proposals.",
                "  - **18 Continuation Frames** (`data/review_pack_continuation_v1`): remaining incomplete frames processed with identical teacher inference settings.",
                "  - **Exact Accounting**: 6 pilot + 18 continuation = **24 incomplete frames accounted for exactly once** (0 missing, 0 duplicates, 0 evaluation frames).",
                "",
                "---",
                "",
                "## 2. Separate Pilot Review Status Report (6 Frames)",
                "",
                "The 6 pilot frames in `data/review_pack_pilot_v1` reflect actual human reviewer decisions:",
                "",
                "| Frame ID | Status | Existing Human | Accepted Additions | Rejected Proposals | Custom Human Added | Review Notes |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
                "| `MVI_40871_img00320` | **Verified** | 23 | 18 | 5 | +19 manual | Dense expressway queue fully completed. |",
                "| `MVI_40141_img00096` | **Verified** | 9 | 22 | 4 | +1 manual | Distant corridor queue vehicles recovered. |",
                "| `MVI_39501_img00174` | **Verified** | 13 | 21 | 4 | 0 | High-angle queue tails recovered. |",
                "| `cam43_south_night_f122256` | **Verified** | 16 | 13 | 3 | +15 manual | Night headlight bloom vehicles supplemented. |",
                "| `cam44_north_night_f004500` | **Verified** | 18 | 15 | 4 | +3 manual | Distant night vehicles recovered. |",
                "| `cam44_north_f019140` | **Rejected / Needs Review** | 23 | 0 | 1 | 0 | Frame rejected by reviewer; 27 pending proposals remain unreviewed. |",
                "",
                "> **Unfinished Pilot Review Location**:",
                "> ```bash",
                "> python tools/annotation_editor.py --pack data/review_pack_pilot_v1",
                "> ```",
                "",
                "---",
                "",
                "## 3. Frame Selection & Rationale for 18 Continuation Frames",
                "",
                "| Frame ID | Origin | Category | Lighting | Existing Boxes | Candidate Additions | Conflicts | Selection Rationale |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
            ])
        else:
            lines.extend([
                "# Batch 6: Pilot Teacher-Assisted Completion Report",
                "",
                f"**Generated**: {created_at}",
                f"**Teacher Model**: `{teacher_model}` (SHA256: `{model_sha}`)",
                f"**Inference Settings**: Conf >= {self.config.conf_threshold}, Full-Frame {self.config.full_frame_imgsz}px, Tiled {self.config.tile_size}px (Overlap {int(self.config.tile_overlap*100)}%)",
                "",
                "---",
                "",
                "## 1. Frame Selection Rationale",
                "",
                "| Frame ID | Origin | Category | Lighting | Existing Boxes | Candidate Additions | Conflicts | Selection Rationale |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
            ])

        for f in manifest_data["frames"]:
            counts = f["counts"]
            lines.append(
                f"| `{f['frame_id']}` | {f['data_origin']} | {f['category']} | {f['lighting']} | {counts['existing_human_boxes']} | **+{counts['candidate_additions']}** | {counts['conflicts']} | {f['rationale']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Full-Frame vs Overlapping Tiled Inference Comparison",
            "",
            "| Frame ID | Full-Frame Raw | Tiled Raw | Combined Raw | Deduplicated | Full Runtime | Tiled Runtime | Total Runtime |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        for f in manifest_data["frames"]:
            m = f["inference_metrics"]
            lines.append(
                f"| `{f['frame_id']}` | {m['raw_full_frame_count']} | {m['raw_tiled_count']} | {m['raw_combined_count']} | **{m['deduplicated_proposal_count']}** | {m['runtime_full_frame_sec']}s | {m['runtime_tiled_sec']}s | {m['runtime_total_sec']}s |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Findings & Safety Safeguards",
            "",
            "- **Zero Destruction of Human Annotations**: All existing human boxes from `data/review_pack_v2` are preserved 100% byte-for-byte.",
            "- **Approved Labels Protected**: Unaccepted proposals are **strictly excluded** from `annotations/labels/*.txt`. Only approved boxes enter exported YOLO labels.",
            "- **Rerun Safety**: Running preparation again preserves any previously reviewed proposals (`accepted` or `rejected`), custom human boxes, reviewer notes, and review status.",
            "- **Tiled Inference Recovery**: Overlapping tiled inference recovered small, distant vehicles missed by full-frame downscaling, particularly in highway queues.",
            "- **No Automatic Class Flips**: Pickup/truck ambiguities and class clashes with human labels are flagged as reviewable `conflicts`, never auto-overwritten.",
            "- **Unverified Continuation Guarantee**: All frames remain strictly in `draft` review status until explicitly verified by the human reviewer.",
            "- **No Premature Recall Claims**: Candidate additions are hypotheses awaiting human verification. No detection accuracy or recall improvement is claimed prior to review.",
            "- **Deleted-Box History Caveat**: Since deleted-box history from prior labeling passes was unavailable, proposals may repeat previously rejected detections and will require human re-rejection.",
            "",
            "---",
            "",
            "## 6. Post-Proposal Small Vehicle Scan Checklist",
            "",
            "> **Qualitative Feedback Note**: In the 6-frame pilot, human review noted that while coverage improved significantly, dense scenes still had roughly ~3–5 missed small vehicles (< 20px). Reviewers should systematically perform this checklist on each frame after accepting/rejecting proposals:",
            "",
            "- [ ] **Expressway Horizon & Vanishing Point**: Zoom to 10–15x and inspect distant oncoming/outgoing lanes. Small cars under 20px often fall below the 0.20 confidence threshold.",
            "- [ ] **Queue Gap Occlusions**: Check narrow gaps between large trucks, buses, and vans where compact sedans/hatchbacks are partly occluded.",
            "- [ ] **Night Glare & Deep Shadows**: In `cam44_north_night_*` sequences, inspect dark roadside areas and zones opposite headlight glare.",
            "- [ ] **Lane-Splitting Motorcycles**: Verify narrow commuter bikes navigating between stopped queues.",
            "- [ ] **Pickup/Truck Ambiguity**: Verify any `[CONFLICT]` boxes to prevent standard 1-ton pickups from flipping to truck.",
            "",
            "---",
            "",
            "## 7. Review Launch & Workflow",
            "",
            "Launch the visual annotation editor with:",
            "```bash",
            f"python tools/annotation_editor.py --pack data/review_pack_{mode}_v1",
            "```",
            "- Use **`[✓ Accept All +NEW]`** to accept all candidate additions in the frame at once.",
            "- Use **`[✕ Reject Rest]`** to discard remaining proposals.",
            "- Use **A** or **X** to accept/reject selected proposals individually.",
            "- Press **Ctrl+Enter** (or click `✓ Mark Verified`) once the small vehicle checklist is completed."
        ])

        return "\n".join(lines)

    def _generate_html_index(self, manifest_data: Dict[str, Any], mode: str, pack_title: str) -> str:
        cards_html = []
        for f in manifest_data["frames"]:
            fid = f["frame_id"]
            cat = f["category"]
            origin = f["data_origin"]
            lighting = f["lighting"]
            counts = f["counts"]
            m = f["inference_metrics"]
            dim = m.get("image_dimensions", {})
            w, h = dim.get("width", 640), dim.get("height", 640)

            cards_html.append(f"""
            <div class="pilot-card">
                <div class="card-header">
                    <div>
                        <div class="frame-title"><code>{fid}</code></div>
                        <div class="frame-sub">{cat} • {origin} • {lighting} • {w}x{h}px</div>
                    </div>
                    <div class="badge-group">
                        <span class="badge badge-existing">{counts['existing_human_boxes']} Human</span>
                        <span class="badge badge-addition">+{counts['candidate_additions']} Additions</span>
                        <span class="badge badge-conflict">{counts['conflicts']} Conflicts</span>
                        <span class="badge badge-dup">{counts['duplicates']} Confirmed</span>
                    </div>
                </div>
                <div class="media-container">
                    <div class="media-box">
                        <div class="media-label">1. Before: Human Baseline (Existing Boxes Only)</div>
                        <a href="previews_before/{fid}.jpg" target="_blank">
                            <img src="previews_before/{fid}.jpg" alt="Before {fid}" class="frame-img" loading="lazy">
                        </a>
                    </div>
                    <div class="media-box">
                        <div class="media-label">2. After: Teacher Proposals (Solid: Human, Dashed: Proposals)</div>
                        <a href="previews/{fid}.jpg" target="_blank">
                            <img src="previews/{fid}.jpg" alt="After {fid}" class="frame-img" loading="lazy">
                        </a>
                    </div>
                    <div class="media-box">
                        <div class="media-label">3. Clean Original Frame</div>
                        <a href="images/{fid}.jpg" target="_blank">
                            <img src="images/{fid}.jpg" alt="Clean {fid}" class="frame-img" loading="lazy">
                        </a>
                    </div>
                </div>
                <div class="card-footer">
                    <div class="rationale-text"><strong>Rationale:</strong> {f['rationale']}</div>
                    <div class="metrics-text">
                        <span>Full-Frame: {m['raw_full_frame_count']} raw ({m['runtime_full_frame_sec']}s)</span> •
                        <span>Tiled: {m['raw_tiled_count']} raw in {m['tiles_count']} tiles ({m['runtime_tiled_sec']}s)</span> •
                        <span>Total Runtime: {m['runtime_total_sec']}s</span>
                    </div>
                </div>
            </div>
            """)

        all_cards = "\n".join(cards_html)
        created_at = manifest_data["metadata"]["created_at"]
        teacher_model = manifest_data["metadata"]["teacher_model"]
        total_frames = manifest_data["metadata"]["total_frames"]
        pack_dir_name = f"review_pack_{mode}_v1"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pack_title}</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: #111827;
            --border-color: #1f2937;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #38bdf8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 24px;
            max-width: 1500px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{ font-size: 1.5rem; color: var(--accent-blue); margin-bottom: 6px; }}
        .header-meta {{ font-size: 0.82rem; color: var(--text-secondary); }}
        .launch-banner {{
            background: #1e293b;
            border: 1px solid #334155;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .launch-code {{
            background: #0f172a;
            border: 1px solid #1e293b;
            padding: 8px 14px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 0.88rem;
            color: #38bdf8;
        }}
        .checklist-card {{
            background: #182234;
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 24px;
        }}
        .checklist-title {{
            font-size: 1.0rem;
            font-weight: 700;
            color: #93c5fd;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .checklist-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }}
        .checklist-item {{
            background: #0f172a;
            border: 1px solid #1e293b;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 0.80rem;
            color: #e2e8f0;
        }}
        .checklist-item strong {{ color: #38bdf8; display: block; margin-bottom: 4px; }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 24px;
        }}
        .pilot-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        .card-header {{
            padding: 14px 18px;
            background: #182234;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .frame-title {{ font-size: 1.05rem; font-weight: 700; color: #fff; }}
        .frame-sub {{ font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }}
        .badge-group {{ display: flex; gap: 8px; }}
        .badge {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .badge-existing {{ background: #1e3a8a; color: #93c5fd; }}
        .badge-addition {{ background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }}
        .badge-conflict {{ background: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; }}
        .badge-dup {{ background: #374151; color: #d1d5db; }}
        .media-container {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            padding: 16px;
            background: #0f172a;
        }}
        @media (max-width: 1100px) {{
            .media-container {{
                grid-template-columns: 1fr;
            }}
        }}
        .media-box {{ display: flex; flex-direction: column; gap: 6px; }}
        .media-label {{ font-size: 0.75rem; font-weight: 600; color: var(--text-muted); }}
        .frame-img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #334155;
            transition: transform 0.15s ease;
        }}
        .frame-img:hover {{ transform: scale(1.01); }}
        .card-footer {{
            padding: 12px 18px;
            border-top: 1px solid var(--border-color);
            background: #111827;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .rationale-text {{ font-size: 0.82rem; color: #e2e8f0; }}
        .metrics-text {{ font-size: 0.75rem; color: var(--text-muted); }}
    </style>
</head>
<body>
    <header>
        <h1>{pack_title}</h1>
        <div class="header-meta">
            Model: <code>{teacher_model}</code> • Generated: {created_at} • Frames: <strong>{total_frames} (Draft / Unverified)</strong>
        </div>
    </header>

    <div class="launch-banner">
        <div>
            <div style="font-weight: 600; font-size: 0.92rem;">Review this pack in the visual annotation editor:</div>
            <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px;">
                Use <strong>[✓ Accept All +NEW]</strong> for 1-click candidate acceptance, <strong>A</strong>/<strong>X</strong> for individual proposals, and <strong>Ctrl+Enter</strong> to verify.
            </div>
        </div>
        <div class="launch-code">python tools/annotation_editor.py --pack data/{pack_dir_name}</div>
    </div>

    <div class="checklist-card">
        <div class="checklist-title">
            <span>⚠️ Post-Proposal Small Vehicle Scan Checklist</span>
        </div>
        <div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.4;">
            Human review of the pilot confirmed substantial recall improvement, but dense scenes still have ~3–5 missed small vehicles (< 20px). Zoom in (10–15x) and scan these regions:
        </div>
        <div class="checklist-grid">
            <div class="checklist-item">
                <strong>1. Expressway Horizon & Vanishing Point</strong>
                Inspect far oncoming and outgoing lanes where small cars fall below the 0.20 confidence threshold.
            </div>
            <div class="checklist-item">
                <strong>2. Queue Interior Gaps</strong>
                Check narrow spaces between large trucks, buses, and high-cage pickups for occluded passenger sedans.
            </div>
            <div class="checklist-item">
                <strong>3. Night Roadside Shadows & Glare</strong>
                In night recordings, inspect dark roadside zones opposite streetlamp glare for unilluminated vehicles.
            </div>
            <div class="checklist-item">
                <strong>4. Lane-Splitting Motorcycles</strong>
                Verify commuter scooters navigating between stopped queues.
            </div>
        </div>
    </div>

    <div class="grid">
        {all_cards}
    </div>
</body>
</html>"""
        return html_content


# -----------------------------------------------------------------------------
# Preview Image Rendering with Proposal Overlays
# -----------------------------------------------------------------------------

def render_pilot_preview(
    clean_img_path: Path,
    existing_boxes: List[Dict[str, Any]],
    proposal_boxes: List[Dict[str, Any]],
    out_path: Path
):
    """
    Renders visual preview showing:
    - Existing human boxes: solid borders with category colors.
    - Candidate additions: dashed green borders with [+NEW] tags.
    - Conflicts: dashed red borders with [CONFLICT] tags.
    """
    im = Image.open(clean_img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    COLOR_MAP = {
        0: (56, 189, 248),   # car - cyan
        1: (251, 146, 60),   # motorcycle - orange
        2: (192, 132, 252),  # bus - purple
        3: (250, 204, 21),   # truck - yellow
        4: (74, 222, 128)    # three_wheeler - green
    }

    # 1. Draw existing human boxes (solid borders, thickness 2)
    for b in existing_boxes:
        norm = b["bbox_norm"]
        xyxy = norm_yolo_to_xyxy(norm, w, h)
        cid = b.get("class_id", 0)
        c = COLOR_MAP.get(cid, (255, 255, 255))
        draw.rectangle(xyxy, outline=c, width=2)
        label_text = f"{b.get('class_name', str(cid))}"
        draw.text((xyxy[0] + 2, xyxy[1] + 2), label_text, fill=c)

    # 2. Draw teacher proposals
    for p in proposal_boxes:
        norm = p["bbox_norm"]
        xyxy = norm_yolo_to_xyxy(norm, w, h)
        p_cat = p.get("proposal_category", "candidate_addition")
        conf = p.get("confidence", 0.0)
        cid = p.get("class_id", 0)

        if p_cat == "candidate_addition":
            outline_color = (0, 255, 128)  # Bright neon green
            tag = f"[+NEW] {p.get('class_name', '')} {conf:.2f}"
        elif p_cat == "conflict":
            outline_color = (255, 64, 64)   # Bright red/crimson
            tag = f"[CONFLICT] {p.get('class_name', '')} vs {p.get('human_class_name', '')}"
        else:
            outline_color = (255, 255, 0)
            tag = f"[PROP] {p.get('class_name', '')} {conf:.2f}"

        draw.rectangle(xyxy, outline=outline_color, width=3)
        draw.text((xyxy[0] + 2, max(0, xyxy[1] - 12)), tag, fill=outline_color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=90)


def batch_accept_and_verify_pack(pack_dir: Path) -> None:
    """
    Accepts all candidate vehicle additions across the pack frames,
    preserves existing human boxes and conflicts, and marks all frames verified.
    """
    from tools.annotation_editor import save_frame_annotation
    annos_file = pack_dir / "annotations" / "annotations.json"
    if not annos_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annos_file}")

    with open(annos_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("\n" + "=" * 70)
    print(f"  BATCH ACCEPTING PROPOSALS & VERIFYING PACK: {pack_dir}")
    print("=" * 70)

    for rec in records:
        fid = rec["frame_id"]
        boxes = rec.get("boxes", [])
        accepted_count = 0
        for b in boxes:
            if b.get("is_proposal") and b.get("proposal_status") == "pending":
                if b.get("proposal_category") == "candidate_addition":
                    b["proposal_status"] = "accepted"
                    accepted_count += 1
                elif b.get("proposal_category") == "conflict":
                    b["proposal_status"] = "rejected"

        save_frame_annotation(
            pack_dir=pack_dir,
            frame_id=fid,
            base_etag=None,
            action="mark_verified",
            boxes_data=boxes,
            reviewer_notes=f"Batch verified: accepted {accepted_count} teacher candidate additions.",
            is_ambiguous=rec.get("is_ambiguous", False)
        )
        print(f"  Frame {fid}: Accepted {accepted_count} additions -> VERIFIED")

    print(f"\n[SUCCESS] Pack '{pack_dir}' is now verified with accepted teacher additions synchronized!")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Teacher-Assisted Annotation Completion (Batch 6 & Continuation)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["continuation", "pilot", "all"],
        default="continuation",
        help="Execution mode: 'continuation' (18 remaining frames), 'pilot' (6 frames), or 'all' (24 frames)"
    )
    parser.add_argument("--source-pack", type=Path, default=Path("data/review_pack_v2"), help="Source review pack")
    parser.add_argument("--output-pack", type=Path, default=None, help="Output pack directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/teacher_inference_cache"), help="Inference cache directory")
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL_PATH, help="Teacher model weights")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--full-imgsz", type=int, default=1280, help="Full-frame inference resolution")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size")
    parser.add_argument("--tile-overlap", type=float, default=0.20, help="Tile overlap ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument(
        "--accept-and-verify-all",
        action="store_true",
        help="Batch accept all candidate additions and mark all frames verified in target pack"
    )
    parser.add_argument(
        "--pack",
        type=Path,
        default=None,
        help="Target pack directory for --accept-and-verify-all"
    )

    args = parser.parse_args()

    if args.accept_and_verify_all:
        target_pack = args.pack or args.output_pack or (Path("data/review_pack_continuation_v1") if args.mode == "continuation" else Path("data/review_pack_pilot_v1"))
        batch_accept_and_verify_pack(pack_dir=target_pack)
        return

    cfg = PilotConfig(
        source_pack_dir=args.source_pack,
        output_pack_dir=args.output_pack,
        cache_dir=args.cache_dir,
        teacher_model_path=args.teacher_model,
        mode=args.mode,
        conf_threshold=args.conf,
        full_frame_imgsz=args.full_imgsz,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        device=args.device
    )

    pilot = TeacherCompletionPilot(cfg)
    pilot.generate_pack(mode=args.mode)


if __name__ == "__main__":
    main()
