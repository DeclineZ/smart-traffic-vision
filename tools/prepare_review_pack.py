"""
Preparation Tool for YOLO26s Evaluation & Annotation Review Pack (Batch 2).

Consumes docs/eval_sampling_manifest.json and creates a versioned review pack:
- Copies clean existing images or extracts exact requested frames from source videos.
- Never uses prediction-overlay images as annotation inputs.
- Verifies extraction succeeds and logs source video, frame index, dimensions, and failures.
- Preserves original labels as clearly marked annotation proposals.
- Associates each image with its source ID, subtype metadata, exposure evidence, and review status.
- Keeps unannotated northeast frames explicitly unannotated (not verified background).
- Generates box-overlay previews for visual triage alongside clean raw images.
- Provides documented editable format for correcting boxes, classes, and subtypes.
- Preserves existing human annotations when rerun (never resets human work; new proposals require a new pack version).
- Records preparation configuration and baseline checkpoint hash for traceability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directory to sys.path to import tools.audit_dataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    THAI_5CLASS_NAMES,
    parse_frame_provenance,
    validate_box,
)

# Colors for box-overlay previews (BGR format for OpenCV)
CLASS_PREVIEW_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (235, 160, 50),   # car: bright blue/cyan
    1: (0, 215, 255),    # motorcycle: yellow/amber
    2: (210, 50, 210),   # bus: magenta/purple
    3: (50, 100, 240),   # truck: orange/coral
    4: (80, 200, 50),    # three_wheeler: green
}

DIAGNOSTIC_GROUP_LABELS: Dict[str, str] = {
    "current_val_diagnostic": "Validation Split Diagnostic (Unproven Lineage)",
    "nearby_training_diagnostic": "Nearby Training Diagnostic (<= 3.0s from Train)",
    "unproven_checkpoint_candidate": "Unproven Checkpoint Candidate (Absence Lineage Unproven)",
    "training_split_reference": "Training Split Reference (Exact Train Match)",
}


class ConflictingEditError(Exception):
    """Raised when conflicting human or machine edits cannot be resolved automatically."""
    pass


def compute_file_sha256(file_path: Path) -> Optional[str]:
    """Computes SHA256 hash of a file if it exists."""
    if not file_path.exists() or not file_path.is_file():
        return None
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_label_file_hash(label_path: Path) -> Optional[str]:
    """Computes SHA256 of normalized YOLO label lines."""
    if not label_path.exists() or not label_path.is_file():
        return None
    content = label_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_boxes_label_hash(boxes: List[Dict[str, Any]]) -> str:
    """Computes SHA256 of normalized YOLO label lines derived from a structured box list."""
    lines = []
    for b in boxes:
        if b.get("is_proposal") and b.get("proposal_status") != "accepted":
            continue
        c = b["class_id"]
        xc, yc, bw, bh = b["bbox_norm"]
        lines.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()



def compute_preview_meta_hash(
    boxes: List[Dict[str, Any]],
    annotation_state: Optional[str] = None,
    is_unannotated: bool = False
) -> str:
    """Computes SHA256 of visual metadata that affects preview rendering."""
    box_repr = [
        (
            b.get("class_id"),
            [round(x, 6) for x in b.get("bbox_norm", [])],
            b.get("subtype", ""),
            bool(b.get("is_ambiguous", False))
        )
        for b in boxes
    ]
    payload = {
        "annotation_state": annotation_state,
        "is_unannotated": bool(is_unannotated),
        "boxes": box_repr,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def derive_annotation_state(
    boxes: List[Dict[str, Any]],
    review_status: str,
    has_source_labels: bool,
    has_human_annotation: bool = False,
) -> Tuple[str, bool]:
    """
    Derives annotation_state and is_unannotated based on evidence:
    - len(boxes) > 0: ("annotated", False)
    - len(boxes) == 0:
      - review_status == "verified": ("verified_empty_background", False)
      - not has_source_labels and (not has_human_annotation or review_status == "unreviewed"): ("unannotated", True)
      - otherwise: ("unreviewed_machine_empty", False)
    """
    if len(boxes) > 0:
        return "annotated", False
    if review_status == "verified":
        return "verified_empty_background", False
    if not has_source_labels and (not has_human_annotation or review_status == "unreviewed"):
        return "unannotated", True
    return "unreviewed_machine_empty", False


def compute_box_iou(
    box_a: Tuple[float, float, float, float] | List[float],
    box_b: Tuple[float, float, float, float] | List[float]
) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes
    in normalized format [center_x, center_y, width, height].
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x1 = ax - aw / 2.0
    a_y1 = ay - ah / 2.0
    a_x2 = ax + aw / 2.0
    a_y2 = ay + ah / 2.0

    b_x1 = bx - bw / 2.0
    b_y1 = by - bh / 2.0
    b_x2 = bx + bw / 2.0
    b_y2 = by + bh / 2.0

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw * ah)
    area_b = max(0.0, bw * bh)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def verify_image_file(file_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Verifies that an image file exists, is non-empty, and decodes successfully with valid dimensions.
    Returns: (is_valid, error_reason_if_any, (width, height))
    """
    if not file_path.exists():
        return False, f"Image file does not exist: {file_path}", None
    try:
        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            return False, f"Image file is empty (0 bytes): {file_path}", None
    except OSError as e:
        return False, f"Failed to stat image file {file_path}: {e}", None

    try:
        import cv2
        im = cv2.imread(str(file_path))
        if im is None:
            return False, f"Failed to decode image with cv2: {file_path}", None
        h, w = im.shape[:2]
        if w <= 0 or h <= 0:
            return False, f"Decoded image has non-positive dimensions: {w}x{h}", None
        return True, None, (w, h)
    except Exception as e:
        return False, f"Exception while decoding image {file_path}: {e}", None


def atomic_replace_file(src: Path, dst: Path) -> None:
    """
    Atomically replaces dst with src using a temporary file in dst's parent directory.
    Ensures same-directory and same-filesystem atomic rename.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dst = dst.parent / f".tmp_{dst.name}.{os.getpid()}_{datetime.now().strftime('%f')}"
    try:
        shutil.copy2(src, temp_dst)
        os.replace(temp_dst, dst)
    except Exception:
        if temp_dst.exists():
            try:
                temp_dst.unlink()
            except OSError:
                pass
        raise


def parse_yolo_label_file(label_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a standard YOLO format label file (class_id xc yc w h).
    Validates classes and coordinates with repo standard tolerance.
    """
    if not label_path.exists():
        return []
    boxes = []
    content = label_path.read_text(encoding="utf-8")
    for line_idx, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        is_v, err, clean_box = validate_box(parts, allowed_classes=THAI_5CLASS_NAMES)
        if not is_v:
            raise ValueError(f"Invalid YOLO label in {label_path} (line {line_idx + 1}): {err}")
        cid, xc, yc, bw, bh = clean_box
        boxes.append({
            "class_id": cid,
            "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)]
        })
    return boxes


def match_box_instances_spatial(
    new_boxes: List[Dict[str, Any]],
    existing_instances: List[Dict[str, Any]],
    frame_id: str,
    iou_threshold: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Spatially pairs incoming YOLO boxes with existing structured instance records
    using greedy bipartite IoU matching.
    
    Preserves:
    - Stable instance_id
    - Subtype metadata
    - Ambiguity status and notes
    
    Detects conflicts if multiple incoming boxes ambiguously match the same instance
    with near-identical high IoU.
    """
    if not existing_instances:
        # All incoming boxes are new instances
        result = []
        for idx, nb in enumerate(new_boxes):
            cid = nb["class_id"]
            bbox = nb["bbox_norm"]
            inst_id = f"{frame_id}_inst_{idx:03d}"
            c_name = THAI_5CLASS_NAMES.get(cid, "unknown")
            sub = nb.get("subtype") or infer_provisional_subtype(cid, frame_id, tuple(bbox))
            result.append({
                "instance_id": inst_id,
                "class_id": cid,
                "class_name": c_name,
                "subtype": sub,
                "is_ambiguous": nb.get("is_ambiguous", False),
                "ambiguity_reason": nb.get("ambiguity_reason", ""),
                "bbox_norm": [round(float(c), 6) for c in bbox],
                "proposal_source": nb.get("proposal_source", "human_annotation")
            })
        return result

    # Compute pairwise IoU
    matches: List[Tuple[float, int, int]] = []
    for n_idx, nb in enumerate(new_boxes):
        for e_idx, eb in enumerate(existing_instances):
            iou = compute_box_iou(nb["bbox_norm"], eb["bbox_norm"])
            if iou >= iou_threshold:
                matches.append((iou, n_idx, e_idx))

    # Sort descending by IoU
    matches.sort(key=lambda x: x[0], reverse=True)

    matched_new: Set[int] = set()
    matched_existing: Set[int] = set()
    pairings: Dict[int, int] = {}  # n_idx -> e_idx

    # Check for severe ambiguity / conflict
    instance_matches: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for iou, n_idx, e_idx in matches:
        instance_matches[e_idx].append((iou, n_idx))

    for e_idx, candidates in instance_matches.items():
        if len(candidates) >= 2:
            top_iou, top_n = candidates[0]
            second_iou, second_n = candidates[1]
            if top_iou >= 0.70 and second_iou >= 0.70 and abs(top_iou - second_iou) < 0.02:
                raise ConflictingEditError(
                    f"Conflicting box match detected in frame '{frame_id}': two boxes (indices {top_n} and {second_n}) "
                    f"compete for existing instance '{existing_instances[e_idx].get('instance_id')}' with near-identical IoU "
                    f"({top_iou:.3f} vs {second_iou:.3f})."
                )

    for iou, n_idx, e_idx in matches:
        if n_idx not in matched_new and e_idx not in matched_existing:
            matched_new.add(n_idx)
            matched_existing.add(e_idx)
            pairings[n_idx] = e_idx

    # Determine highest existing instance index to generate unique IDs for additions
    used_ids: Set[str] = {eb.get("instance_id", "") for eb in existing_instances if eb.get("instance_id")}
    counter = len(existing_instances)

    def generate_next_id() -> str:
        nonlocal counter
        while True:
            candidate = f"{frame_id}_inst_{counter:03d}"
            counter += 1
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    result = []
    for n_idx, nb in enumerate(new_boxes):
        cid = nb["class_id"]
        bbox = [round(float(c), 6) for c in nb["bbox_norm"]]
        c_name = THAI_5CLASS_NAMES.get(cid, "unknown")

        if n_idx in pairings:
            e_idx = pairings[n_idx]
            eb = existing_instances[e_idx]
            inst_id = eb.get("instance_id") or generate_next_id()
            old_cid = eb.get("class_id")

            # Preserve subtype & ambiguity if class matches or subtype was human-assigned
            if old_cid == cid:
                sub = eb.get("subtype") or infer_provisional_subtype(cid, frame_id, tuple(bbox))
                is_amb = eb.get("is_ambiguous", False)
                amb_reason = eb.get("ambiguity_reason", "")
            else:
                # Class changed! Infer new provisional subtype for the new class
                sub = infer_provisional_subtype(cid, frame_id, tuple(bbox))
                is_amb = eb.get("is_ambiguous", False)
                amb_reason = eb.get("ambiguity_reason", "")

            result.append({
                "instance_id": inst_id,
                "class_id": cid,
                "class_name": c_name,
                "subtype": sub,
                "is_ambiguous": is_amb,
                "ambiguity_reason": amb_reason,
                "bbox_norm": bbox,
                "proposal_source": eb.get("proposal_source", "human_annotation")
            })
        else:
            # Newly inserted box
            inst_id = generate_next_id()
            sub = nb.get("subtype") or infer_provisional_subtype(cid, frame_id, tuple(bbox))
            result.append({
                "instance_id": inst_id,
                "class_id": cid,
                "class_name": c_name,
                "subtype": sub,
                "is_ambiguous": nb.get("is_ambiguous", False),
                "ambiguity_reason": nb.get("ambiguity_reason", ""),
                "bbox_norm": bbox,
                "proposal_source": nb.get("proposal_source", "human_annotation")
            })

    return result


def deduplicate_manifest_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates manifest candidates by canonical source frame ID.
    Merges duplicate records into a single canonical entry, combining selection reasons,
    artifact references, and resolving training exposure precedence:
    current_train_split > nearby_training_exposure > current_val_split > unproven_checkpoint_exposure.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        cid = s.get("source_frame_id", s.get("canonical_source_id", s["frame_id"]))
        grouped[cid].append(s)

    exposure_precedence = {
        "current_train_split": 4,
        "nearby_training_exposure": 3,
        "current_val_split": 2,
        "unproven_checkpoint_exposure": 1
    }

    canonical_samples: List[Dict[str, Any]] = []
    for cid, dups in grouped.items():
        if len(dups) == 1:
            canonical_samples.append(dups[0])
            continue

        # Merge duplicates
        primary = dups[0]
        merged = dict(primary)

        # Merge selection reasons
        combined_reasons: List[str] = []
        for d in dups:
            for r in d.get("selection_reasons", []):
                if r not in combined_reasons:
                    combined_reasons.append(r)
        merged["selection_reasons"] = combined_reasons

        # Merge artifact references
        combined_refs: List[str] = []
        for d in dups:
            for ref in d.get("artifact_references", []):
                if ref not in combined_refs:
                    combined_refs.append(ref)
        merged["artifact_references"] = combined_refs

        # Merge verified classes present
        combined_classes: List[str] = []
        for d in dups:
            for c in d.get("verified_classes_present", []):
                if c not in combined_classes:
                    combined_classes.append(c)
        merged["verified_classes_present"] = combined_classes

        # Exposure precedence
        best_exposure = primary.get("training_exposure", "unproven_checkpoint_exposure")
        best_score = exposure_precedence.get(best_exposure, 0)
        min_delta_f = primary.get("nearby_training_delta_frames")
        min_delta_t = primary.get("nearby_training_delta_sec")

        for d in dups[1:]:
            d_exp = d.get("training_exposure", "unproven_checkpoint_exposure")
            d_score = exposure_precedence.get(d_exp, 0)
            if d_score > best_score:
                best_score = d_score
                best_exposure = d_exp

            df = d.get("nearby_training_delta_frames")
            dt = d.get("nearby_training_delta_sec")
            if df is not None and (min_delta_f is None or df < min_delta_f):
                min_delta_f = df
            if dt is not None and (min_delta_t is None or dt < min_delta_t):
                min_delta_t = dt

        merged["training_exposure"] = best_exposure
        merged["nearby_training_delta_frames"] = min_delta_f
        merged["nearby_training_delta_sec"] = min_delta_t
        merged["canonical_source_id"] = cid

        canonical_samples.append(merged)

    return canonical_samples


def assign_diagnostic_group(sample: Dict[str, Any]) -> str:
    """
    Classifies a manifest sample into an explicit diagnostic category.
    Never infers an 'independent test benchmark' without lineage proof.
    """
    exposure = sample.get("training_exposure", "unproven_checkpoint_exposure")
    reasons = sample.get("selection_reasons", [])
    artifact_refs = sample.get("artifact_references", [])

    if exposure == "current_train_split":
        return "training_split_reference"
    if exposure == "nearby_training_exposure":
        return "nearby_training_diagnostic"
    if any(r.startswith("val_split:") for r in artifact_refs) or exposure == "current_val_split":
        return "current_val_diagnostic"
    return "unproven_checkpoint_candidate"


def infer_provisional_subtype(
    class_id: int,
    frame_id: str,
    box_coords: Tuple[float, float, float, float],
    staged_mining_index: Optional[Dict[str, Any]] = None
) -> str:
    """
    Assigns a provisional subtype string based on candidate source or class category.
    All assignments are provisional proposals subject to human verification.
    """
    c_name = THAI_5CLASS_NAMES.get(class_id, "unknown")
    if class_id == 0:
        # Car light vehicle category
        if "songthaew" in frame_id.lower():
            return "pickup_based_songthaew"
        if "pickup" in frame_id.lower():
            return "pickup"
        if "van" in frame_id.lower():
            return "passenger_van"
        return "light_vehicle_provisional"

    if class_id == 1:
        return "motorcycle_provisional"

    if class_id == 2:
        return "bus_provisional"

    if class_id == 3:
        if "songthaew" in frame_id.lower():
            return "truck_based_songthaew"
        if "trailer" in frame_id.lower():
            return "articulated_trailer_18w"
        return "commercial_truck_provisional"

    if class_id == 4:
        if "saleng" in frame_id.lower():
            return "saleng"
        if "tuktuk" in frame_id.lower():
            return "tuktuk"
        return "three_wheeler_provisional"

    return f"{c_name}_provisional"


def extract_clean_frame_from_video(
    video_path: Path,
    frame_idx: int,
    output_path: Path
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Extracts an exact frame index from a video container using OpenCV.
    Verifies frame decode and write success before returning.
    Returns: (success, failure_reason_if_any, (width, height) if successful).
    """
    if not video_path.exists():
        return False, f"Source video file not found: {video_path}", None

    try:
        import cv2
    except ImportError:
        return False, "OpenCV (cv2) is required for video frame extraction", None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"Could not open video file: {video_path}", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or (total_frames > 0 and frame_idx >= total_frames):
        cap.release()
        return False, f"Frame index {frame_idx} is out of bounds (total frames in container: {total_frames})", None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, f"Failed to decode frame {frame_idx} from {video_path.name}", None

    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return False, f"Extracted frame {frame_idx} has invalid dimensions: {w}x{h}", None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ok = cv2.imwrite(str(output_path), frame)
    if not write_ok:
        return False, f"cv2.imwrite returned False when writing frame {frame_idx} to {output_path}", None

    # Verify written image on disk
    is_v, err, dims = verify_image_file(output_path)
    if not is_v:
        if output_path.exists():
            output_path.unlink()
        return False, f"Extracted frame verification failed: {err}", None

    return True, None, dims


def render_box_preview_image(
    clean_image_path: Path,
    boxes: List[Dict[str, Any]],
    output_preview_path: Path,
    is_unannotated: bool = False,
    annotation_state: Optional[str] = None
) -> bool:
    """
    Renders bounding box proposals and subtype badges on a copy of the clean image
    for human review visual inspection. Verifies write and decode success.
    """
    is_v, err, _ = verify_image_file(clean_image_path)
    if not is_v:
        return False

    try:
        import cv2
    except ImportError:
        return False

    im = cv2.imread(str(clean_image_path))
    if im is None:
        return False

    h, w = im.shape[:2]

    effective_state = annotation_state
    if effective_state is None:
        if is_unannotated:
            effective_state = "unannotated"
        elif len(boxes) > 0:
            effective_state = "annotated"
        else:
            effective_state = "empty"

    if effective_state == "unannotated":
        badge_text = "[UNANNOTATED CANDIDATE - REQUIRES FULL HUMAN ANNOTATION]"
        cv2.rectangle(im, (20, 20), (740, 60), (30, 30, 30), -1)
        cv2.putText(im, badge_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
    elif effective_state == "verified_empty_background":
        badge_text = "[VERIFIED EMPTY BACKGROUND - 0 VEHICLES]"
        cv2.rectangle(im, (20, 20), (580, 60), (20, 60, 20), -1)
        cv2.putText(im, badge_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (74, 222, 128), 2, cv2.LINE_AA)
    else:
        for b in boxes:
            if b.get("is_proposal") and b.get("proposal_status") == "rejected":
                continue

            c_id = b.get("class_id", 0)
            xc, yc, bw, bh = b.get("bbox_norm", [0, 0, 0, 0])
            color = CLASS_PREVIEW_COLORS.get(c_id, (0, 255, 0))

            is_prop = bool(b.get("is_proposal"))
            prop_status = b.get("proposal_status", "none")
            prop_cat = b.get("proposal_category", "")

            if is_prop and prop_status == "pending":
                if prop_cat == "conflict":
                    color = (0, 0, 255)  # BGR Red
                else:
                    color = (0, 255, 128)  # BGR Neon Green
            elif is_prop and prop_status == "accepted":
                color = (0, 200, 0)  # BGR Green

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

            # Label text: class + subtype
            c_name = b.get("class_name", str(c_id))
            sub = b.get("subtype", "")
            prefix = ""
            if is_prop:
                if prop_status == "accepted":
                    prefix = "[ACCEPTED] "
                elif prop_cat == "conflict":
                    prefix = "[CONFLICT] "
                else:
                    prefix = "[+NEW] "
            lbl = f"{prefix}{c_name}:{sub}" if sub else f"{prefix}{c_name}"
            if b.get("is_ambiguous"):
                lbl += " [AMBIGUOUS]"

            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            by1 = max(0, y1 - th - 6)
            by2 = y1
            bx2 = min(w - 1, x1 + tw + 6)
            cv2.rectangle(im, (x1, by1), (bx2, by2), color, -1)
            cv2.putText(im, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    output_preview_path.parent.mkdir(parents=True, exist_ok=True)
    write_ok = cv2.imwrite(str(output_preview_path), im)
    if not write_ok:
        return False

    is_pv, _, _ = verify_image_file(output_preview_path)
    return is_pv


def render_html_review_index(
    pack_dir: Path,
    manifest_records: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Path:
    """
    Renders an interactive, responsive HTML review index adhering to repo UX standards.
    Provides direct visual triage, preview links, and metadata display.
    """
    html_path = pack_dir / "review_index.html"

    cards_html: List[str] = []
    for r in manifest_records:
        fid = r["frame_id"]
        sid = r["canonical_source_id"]
        cam = r["camera"]
        lighting = r["lighting_type"]
        group = r["diagnostic_group"]
        group_label = DIAGNOSTIC_GROUP_LABELS.get(group, group)
        clean_status = r["clean_input_status"]
        dim_str = f"{r['dimensions'][0]}x{r['dimensions'][1]}" if r.get("dimensions") else "Unknown"
        is_unanno = r.get("is_unannotated", False)
        rev_status = r.get("review_status", "unreviewed")

        boxes = r.get("boxes", [])
        box_summary_parts = []
        for b in boxes:
            c = b.get("class_name", "")
            sub = b.get("subtype", "")
            box_summary_parts.append(f"<span class='badge class-{b.get('class_id', 0)}'>{c}: {sub}</span>")
        box_summary = " ".join(box_summary_parts) if box_summary_parts else (
            "<span class='badge unannotated-badge'>Unannotated Candidate</span>" if is_unanno else "<span class='badge'>0 Boxes</span>"
        )

        preview_rel = f"previews/{fid}_preview.jpg"
        clean_rel = f"images/{fid}.jpg"
        label_rel = f"annotations/labels/{fid}.txt"

        status_class = "status-unreviewed" if rev_status == "unreviewed" else "status-verified"

        cards_html.append(f"""
        <tr class="sample-row" data-group="{group}" data-camera="{cam}" data-lighting="{lighting}" data-status="{rev_status}">
            <td class="thumb-cell">
                <a href="{preview_rel}" target="_blank" title="Click to view full preview">
                    <img src="{preview_rel}" alt="Preview {fid}" class="sample-thumb" loading="lazy" />
                </a>
            </td>
            <td class="info-cell">
                <div class="sample-title"><code>{fid}</code></div>
                <div class="sample-meta">
                    <strong>Source ID:</strong> <code>{sid}</code> |
                    <strong>Camera:</strong> {cam} |
                    <strong>Lighting:</strong> <span class="lighting-tag">{lighting}</span> |
                    <strong>Dimensions:</strong> {dim_str}
                </div>
                <div class="group-tag group-{group}">{group_label}</div>
                <div class="boxes-container">{box_summary}</div>
            </td>
            <td class="action-cell">
                <div class="status-badge {status_class}">{rev_status.upper()}</div>
                <div class="links-stack">
                    <a href="{clean_rel}" target="_blank" class="action-link">Clean Image</a>
                    <a href="{preview_rel}" target="_blank" class="action-link">Box Preview</a>
                    <a href="{label_rel}" target="_blank" class="action-link">Edit YOLO Label</a>
                </div>
            </td>
        </tr>
        """)

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26s Thai Traffic Vision - Evaluation Review Pack v1</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #182234;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-yellow: #fbbf24;
            --accent-orange: #fb923c;
            --accent-purple: #c084fc;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .header-panel {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 1.6rem;
            color: var(--accent-blue);
        }}
        .meta-summary {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 12px;
        }}
        .meta-item {{
            background-color: rgba(255, 255, 255, 0.04);
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .filter-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        .filter-btn {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.15s ease-in-out;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background-color: var(--accent-blue);
            color: #0f172a;
            font-weight: 600;
        }}
        table.samples-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background-color: var(--bg-secondary);
            text-align: left;
            padding: 12px 16px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
        }}
        .sample-thumb {{
            width: 240px;
            height: 135px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s;
            display: block;
        }}
        .sample-thumb:hover {{
            transform: scale(1.03);
            border-color: var(--accent-blue);
        }}
        .sample-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        .sample-meta {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }}
        .group-tag {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .group-current_val_diagnostic {{ background-color: rgba(56, 189, 248, 0.15); color: var(--accent-blue); }}
        .group-nearby_training_diagnostic {{ background-color: rgba(251, 146, 60, 0.15); color: var(--accent-orange); }}
        .group-unproven_checkpoint_candidate {{ background-color: rgba(192, 132, 252, 0.15); color: var(--accent-purple); }}
        .badge {{
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 6px;
            border-radius: 4px;
            margin: 2px;
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
        }}
        .badge.class-0 {{ border-color: #38bdf8; color: #38bdf8; }}
        .badge.class-1 {{ border-color: #fbbf24; color: #fbbf24; }}
        .badge.class-2 {{ border-color: #c084fc; color: #c084fc; }}
        .badge.class-3 {{ border-color: #fb923c; color: #fb923c; }}
        .badge.class-4 {{ border-color: #4ade80; color: #4ade80; }}
        .unannotated-badge {{ background-color: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); border-color: var(--accent-yellow); }}
        .status-badge {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .status-unreviewed {{ background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }}
        .status-verified {{ background-color: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #4ade80; }}
        .links-stack {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .action-link {{
            color: var(--accent-blue);
            text-decoration: none;
            font-size: 0.82rem;
        }}
        .action-link:hover {{
            text-decoration: underline;
        }}
        .checklist-box {{
            background-color: rgba(56, 189, 248, 0.05);
            border-left: 4px solid var(--accent-blue);
            padding: 12px 16px;
            margin-top: 12px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="header-panel">
        <h1>YOLO26s Thai Traffic Vision - Evaluation Review Pack (Batch 2)</h1>
        <div style="font-size: 0.9rem; color: var(--text-secondary);">
            Baseline Weight: <code>models/yolo26s_thai_traffic.pt</code> (SHA256: <code>{metadata.get('baseline_checkpoint_sha256', 'N/A')[:16]}...</code>) |
            Created: {metadata.get('generated_at', '')}
        </div>
        <div class="meta-summary">
            <div class="meta-item"><strong>Total Samples:</strong> {len(manifest_records)}</div>
            <div class="meta-item"><strong>Validation Diagnostic:</strong> {metadata.get('diagnostic_group_counts', {}).get('current_val_diagnostic', 0)}</div>
            <div class="meta-item"><strong>Nearby Training Diagnostic:</strong> {metadata.get('diagnostic_group_counts', {}).get('nearby_training_diagnostic', 0)}</div>
            <div class="meta-item"><strong>Unproven Checkpoint Candidates:</strong> {metadata.get('diagnostic_group_counts', {}).get('unproven_checkpoint_candidate', 0)}</div>
            <div class="meta-item"><strong>Extraction Failures:</strong> {len(metadata.get('extraction_failures', []))}</div>
        </div>
        <div class="checklist-box">
            <strong>Mandatory Human Review Checklist:</strong><br>
            1. <strong>Light vs Heavy boundary</strong>: Ordinary pickups (Hilux, D-Max) & pickup songthaews = <code>0: car</code>. Medium/heavy trucks & truck songthaews = <code>3: truck</code>.<br>
            2. <strong>Occlusion & Horizon</strong>: Check distant vehicles (< 32&sup2; px) and occluded motorcycles. Do not drop valid instances.<br>
            3. <strong>Uncertainty</strong>: Mark ambiguous vehicles with <code>is_ambiguous: true</code> instead of guessing.<br>
            4. <strong>Unannotated frames</strong>: Northeast frames start unannotated (not empty background). Add visible bounding boxes from scratch.<br>
            5. <strong>Policy</strong>: All records start unreviewed. Neither machine proposals nor previews constitute human approval.
        </div>
    </div>

    <div class="filter-bar">
        <button class="filter-btn active" onclick="filterSamples('all')">Show All ({len(manifest_records)})</button>
        <button class="filter-btn" onclick="filterSamples('current_val_diagnostic')">Validation Diagnostic ({metadata.get('diagnostic_group_counts', {}).get('current_val_diagnostic', 0)})</button>
        <button class="filter-btn" onclick="filterSamples('nearby_training_diagnostic')">Nearby Training Diagnostic ({metadata.get('diagnostic_group_counts', {}).get('nearby_training_diagnostic', 0)})</button>
        <button class="filter-btn" onclick="filterSamples('unproven_checkpoint_candidate')">Northeast / Unproven Holdout ({metadata.get('diagnostic_group_counts', {}).get('unproven_checkpoint_candidate', 0)})</button>
    </div>

    <table class="samples-table">
        <thead>
            <tr>
                <th style="width: 250px;">Visual Preview</th>
                <th>Candidate Metadata & Proposal Summary</th>
                <th style="width: 160px;">Review Actions</th>
            </tr>
        </thead>
        <tbody>
            {"".join(cards_html)}
        </tbody>
    </table>

    <script>
        function filterSamples(group) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            const rows = document.querySelectorAll('.sample-row');
            rows.forEach(r => {{
                if (group === 'all' || r.getAttribute('data-group') === group) {{
                    r.style.display = '';
                }} else {{
                    r.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    return html_path


def render_markdown_readme(
    pack_dir: Path,
    manifest_records: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Path:
    """Renders data/review_pack_v1/README.md with instructions and summary tables."""
    readme_path = pack_dir / "README.md"
    lines: List[str] = []
    w = lines.append

    w("# YOLO26s Thai Traffic Vision - Evaluation Review Pack (Batch 2)")
    w("")
    w("> **Pack Directory**: `" + str(pack_dir) + "`  ")
    w(f"> **Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (SHA256: `{metadata.get('baseline_checkpoint_sha256', 'N/A')}`)  ")
    w(f"> **Generated At**: {metadata.get('generated_at', '')}  ")
    w(f"> **Total Review Candidates**: **{len(manifest_records)}**  ")
    w(f"> **Extraction Failures**: **{len(metadata.get('extraction_failures', []))}**")
    w("")
    w("---")
    w("")
    w("## 1. Quick Navigation")
    w("")
    w("- **Interactive Dashboard**: Open [`review_index.html`](review_index.html) in any browser for visual card filtering and inspection.")
    w("- **Clean Raw Images**: `images/` (Original, clean CCTV frames; zero overlays).")
    w("- **Box Previews**: `previews/` (Visual bounding box overlay renders for rapid triage).")
    w("- **Original Machine Proposals**: `proposals/` (Immutable baseline proposals).")
    w("- **Editable YOLO Labels**: `annotations/labels/` (Standard `class xc yc w h` files for CVAT/LabelStudio/VSCode).")
    w("- **Editable Structured Annotations**: `annotations/annotations.json` (Includes subtype metadata, ambiguity flags, review status).")
    w("- **Traceability Manifest**: `manifest.json` (Full execution parameters and baseline hashes).")
    w("")
    w("---")
    w("")
    w("## 2. Agreed Vehicle Taxonomy (5-Class Standard)")
    w("")
    w("| Class ID | Class Name | Included Vehicles & Subtypes | Key Rule |")
    w("| :---: | :--- | :--- | :--- |")
    w("| **`0`** | **`car`** | Sedans, hatchbacks, taxis, SUVs, PPVs, passenger commuter vans, ordinary pickups (Hilux, D-Max, etc.), and pickup-based songthaews. | Light 4-wheel vehicle visual morphology. |")
    w("| **`1`** | **`motorcycle`** | Commuter scooters, underbones, big bikes, delivery motorbikes (Grab, Lineman, Shopee). | Two-wheel motorized vehicles. |")
    w("| **`2`** | **`bus`** | BMTA city transit buses, EV Thai Smile Bus, tour coaches, double-deckers. | Large transit passenger vehicles. |")
    w("| **`3`** | **`truck`** | Medium 6-wheelers, heavy 10-wheelers, articulated 18-wheelers (รถพ่วง), and truck-based songthaews. | Medium/heavy commercial freight chassis. |")
    w("| **`4`** | **`three_wheeler`** | Tuk-Tuks (auto-rickshaw) and Salengs (motorized sidecar cargo tricycle). | Motorized 3-wheel configuration. |")
    w("")
    w("> [!IMPORTANT]")
    w("> **Detector vs. Controller Separation**: Detector classes model visual vehicle morphology only. Traffic signal controllers assign PCE or delay weights downstream independently; do NOT assign traffic weights in the detector or annotations.")
    w("")
    w("---")
    w("")
    w("## 3. Human Review Checklist")
    w("")
    w("1. **Every Visible Target Vehicle**: Check all approaches, oncoming traffic, and turn lanes.")
    w("2. **Distant Vehicles & Occlusions**: Inspect distant vehicles (< 32² px) near the intersection horizon.")
    w("3. **Light vs Heavy Consistency**: Pickups and pickup songthaews $\\rightarrow$ `0: car`; commercial 6/10/18-wheelers & truck songthaews $\\rightarrow$ `3: truck`.")
    w("4. **Mark Ambiguity (Never Guess)**: If night glare, extreme distance, or occlusion prevents distinction, set `is_ambiguous: true`.")
    w("5. **Unannotated Frames**: `cam45_northeast` frames are unannotated candidates, NOT empty backgrounds. Annotate all visible targets from scratch.")
    w("6. **Unreviewed Baseline**: All records start unreviewed. Neither machine proposals nor generated previews constitute human approval.")
    w("")
    w("---")
    w("")
    w("## 4. Candidate Summary Table")
    w("")
    w("| Frame ID | Canonical Source ID | Camera | Diagnostic Category | Clean Input Status | Proposed Classes / Subtypes | Review Status |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- | :---: |")

    for r in manifest_records:
        fid = r["frame_id"]
        sid = r["canonical_source_id"]
        cam = r["camera"]
        diag = r["diagnostic_group"]
        clean_in = r["clean_input_status"]
        boxes = r.get("boxes", [])
        if r.get("is_unannotated"):
            c_desc = "*Unannotated*"
        elif boxes:
            c_desc = ", ".join(f"{b.get('class_name')}:{b.get('subtype', '')}" for b in boxes)
        else:
            c_desc = "*0 proposals*"
        w(f"| [`{fid}`](images/{fid}.jpg) | `{sid}` | {cam} | `{diag}` | `{clean_in}` | {c_desc} | **{r.get('review_status', 'unreviewed')}** |")

    w("")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return readme_path


def prepare_review_pack(
    manifest_path: Path,
    output_dir: Path,
    videos_dir: Path = Path("videos"),
    dataset_dir: Path = Path("data/multiclass_dataset")
) -> Dict[str, Any]:
    """
    Main preparation workflow:
    1. Loads evaluation sampling manifest and deduplicates canonical source IDs.
    2. Hashes baseline checkpoint.
    3. Safely parses existing annotations.json; stops immediately if malformed to preserve human work.
    4. Enforces immutability of proposals and clean images within the pack version.
    5. Preserves all human edits on rerun (regardless of review_status or notes; preserves empty box lists).
    6. Retains existing annotation records omitted from the incoming manifest.
    7. Determines annotation availability from evidence (no camera name heuristics).
    8. Verifies image decoding and write success for clean images and previews.
    9. Generates review index (HTML and Markdown) and manifest.
    
    Note: Preparation preserves existing annotations and directs users to explicit
    synchronization when edits are pending. It never resets human work.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    raw_samples = manifest_data.get("samples", [])
    if not raw_samples:
        raise ValueError(f"No samples found in manifest: {manifest_path}")

    # Deduplicate canonical source IDs rather than merely counting groups
    samples = deduplicate_manifest_samples(raw_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    previews_dir = output_dir / "previews"
    proposals_dir = output_dir / "proposals"
    annos_dir = output_dir / "annotations"
    annos_labels_dir = annos_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    annos_labels_dir.mkdir(parents=True, exist_ok=True)

    # Compute baseline checkpoint hash
    baseline_pt_path = Path("models/yolo26s_thai_traffic.pt")
    baseline_hash = compute_file_sha256(baseline_pt_path)

    # Parse existing annotations safely: stop without overwriting if malformed
    existing_annotations: Dict[str, Dict[str, Any]] = {}
    existing_annos_file = annos_dir / "annotations.json"
    if existing_annos_file.exists():
        try:
            with open(existing_annos_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError(f"Existing {existing_annos_file} must contain a JSON list, got {type(loaded).__name__}")
            for item in loaded:
                fid = item.get("frame_id")
                if fid:
                    existing_annotations[fid] = item
        except Exception as e:
            raise ValueError(
                f"Failed to parse existing annotations JSON ({existing_annos_file}): {e}. "
                f"Stopping execution immediately without overwriting existing files to preserve human work."
            ) from e

    records: List[Dict[str, Any]] = []
    extraction_failures: List[Dict[str, Any]] = []
    diagnostic_group_counts: Dict[str, int] = Counter()
    clean_input_counts: Dict[str, int] = Counter()

    processed_frame_ids: Set[str] = set()

    for s in samples:
        frame_id = s["frame_id"]
        source_frame_id = s.get("source_frame_id", s.get("canonical_source_id", frame_id))
        video_name = s.get("video_name", "unknown")
        camera = s.get("camera", "unknown")
        frame_idx = s.get("frame_idx")
        lighting = s.get("lighting_type", "unknown")
        anno_status = s.get("annotations_status", "")
        clean_input_status = s.get("clean_input_status", "")
        exposure = s.get("training_exposure", "unproven_checkpoint_exposure")
        delta_f = s.get("nearby_training_delta_frames")
        delta_t = s.get("nearby_training_delta_sec")
        reasons = s.get("selection_reasons", [])
        artifact_refs = s.get("artifact_references", [])

        diag_group = assign_diagnostic_group(s)
        diagnostic_group_counts[diag_group] += 1
        processed_frame_ids.add(frame_id)

        clean_img_dest = images_dir / f"{frame_id}.jpg"
        clean_extracted = False
        extraction_method = "failed"
        dims: Optional[Tuple[int, int]] = None
        fail_reason: Optional[str] = None

        # 1. Clean image acquisition: immutable within pack version; NEVER use prediction overlays
        if clean_img_dest.exists():
            is_v, err, d = verify_image_file(clean_img_dest)
            if is_v:
                clean_extracted = True
                extraction_method = "existing_immutable_clean_image"
                dims = d
            else:
                fail_reason = f"Existing clean image {clean_img_dest} failed verification: {err}"
        elif clean_input_status == "existing_raw_image":
            source_raw_path = dataset_dir / "images" / "val" / f"{frame_id}.jpg"
            if source_raw_path.exists():
                shutil.copy2(source_raw_path, clean_img_dest)
                is_v, err, d = verify_image_file(clean_img_dest)
                if is_v:
                    clean_extracted = True
                    extraction_method = "copied_from_val_images"
                    dims = d
                else:
                    if clean_img_dest.exists():
                        clean_img_dest.unlink()
                    fail_reason = f"Copied clean image {clean_img_dest} failed verification: {err}"
            else:
                vid_path = videos_dir / f"{video_name}.avi"
                if vid_path.exists() and frame_idx is not None:
                    ok, err, d = extract_clean_frame_from_video(vid_path, frame_idx, clean_img_dest)
                    if ok:
                        clean_extracted = True
                        extraction_method = "extracted_from_source_video"
                        dims = d
                    else:
                        fail_reason = err
                else:
                    fail_reason = f"Image {source_raw_path} not found and video extraction unavailable"
        else:
            vid_path = videos_dir / f"{video_name}.avi"
            if frame_idx is not None:
                ok, err, d = extract_clean_frame_from_video(vid_path, frame_idx, clean_img_dest)
                if ok:
                    clean_extracted = True
                    extraction_method = "extracted_from_source_video"
                    dims = d
                else:
                    fail_reason = err
            else:
                fail_reason = f"Missing frame index for extraction from {video_name}"

        if not clean_extracted:
            extraction_failures.append({
                "frame_id": frame_id,
                "source_frame_id": source_frame_id,
                "video": video_name,
                "frame_idx": frame_idx,
                "reason": fail_reason
            })
            clean_input_counts["failed"] += 1
        else:
            clean_input_counts[extraction_method] += 1

        # 2. Proposals extraction & Subtype assignment: EVIDENCE-BASED (no camera name checks)
        val_lbl_file = dataset_dir / "labels" / "val" / f"{frame_id}.txt"
        has_source_labels = False
        proposal_boxes: List[Dict[str, Any]] = []

        if anno_status not in ("missing_unannotated", "no_labels") and val_lbl_file.exists():
            with open(val_lbl_file, "r", encoding="utf-8") as lf:
                for b_idx, line in enumerate(lf):
                    parts = line.strip().split()
                    is_v, _, clean_box = validate_box(parts, allowed_classes=THAI_5CLASS_NAMES)
                    if is_v:
                        cid, xc, yc, bw, bh = clean_box
                        sub = infer_provisional_subtype(cid, frame_id, (xc, yc, bw, bh))
                        proposal_boxes.append({
                            "instance_id": f"{frame_id}_inst_{b_idx:03d}",
                            "class_id": cid,
                            "class_name": THAI_5CLASS_NAMES[cid],
                            "subtype": sub,
                            "is_ambiguous": False,
                            "ambiguity_reason": "",
                            "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                            "proposal_source": f"val_label:{val_lbl_file.name}",
                        })
            has_source_labels = True

        # Generate expected proposal text content
        if not has_source_labels:
            expected_proposal_content = "# UNANNOTATED CANDIDATE - NO MACHINE PROPOSALS\n"
        elif proposal_boxes:
            expected_proposal_content = "".join(
                f"{b['class_id']} {b['bbox_norm'][0]:.6f} {b['bbox_norm'][1]:.6f} {b['bbox_norm'][2]:.6f} {b['bbox_norm'][3]:.6f}\n"
                for b in proposal_boxes
            )
        else:
            expected_proposal_content = "# MACHINE PROPOSALS: 0 BOXES\n"

        # Check proposal file immutability within pack version: new proposals require a new pack version
        proposal_file = proposals_dir / f"{frame_id}.txt"
        if proposal_file.exists():
            existing_content = proposal_file.read_text(encoding="utf-8")
            if existing_content.strip() != expected_proposal_content.strip():
                raise RuntimeError(
                    f"Source proposals changed for frame '{frame_id}' within pack version! "
                    f"Existing proposal file '{proposal_file}' differs from newly generated proposals. "
                    f"Proposal files and clean input images are immutable within a pack version. "
                    f"To introduce new source proposals, create a new pack version (e.g. --output-dir data/review_pack_v2)."
                )
        else:
            proposal_file.write_text(expected_proposal_content, encoding="utf-8")

        # 3. Check for existing human edits to preserve: NEVER reset human work
        editable_label_file = annos_labels_dir / f"{frame_id}.txt"

        if frame_id in existing_annotations:
            prev = existing_annotations[frame_id]
            editable_boxes = prev.get("boxes", [])
            review_status = prev.get("review_status", "unreviewed")
            reviewer_notes = prev.get("reviewer_notes", "")
            is_ambiguous_sample = prev.get("is_ambiguous", False)
            synced_label_hash = prev.get("synced_label_hash")

            annotation_state, is_unannotated = derive_annotation_state(
                editable_boxes, review_status, has_source_labels, has_human_annotation=True
            )

            # Preparation must NOT overwrite or sync editable_label_file or boxes!
            # If label file does not exist, initialize it from editable_boxes
            if not editable_label_file.exists():
                with open(editable_label_file, "w", encoding="utf-8") as ef:
                    for b in editable_boxes:
                        c = b["class_id"]
                        xc, yc, bw, bh = b["bbox_norm"]
                        ef.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                synced_label_hash = compute_label_file_hash(editable_label_file)

            preview_meta_hash = prev.get("preview_meta_hash")
            if not preview_meta_hash:
                preview_meta_hash = compute_preview_meta_hash(editable_boxes, annotation_state, is_unannotated)

            record = dict(prev)
            record["annotation_state"] = annotation_state
            record["is_unannotated"] = is_unannotated
            record["preview_meta_hash"] = preview_meta_hash
            record["synced_label_hash"] = synced_label_hash
        else:
            # Brand new sample: write initial editable YOLO format label
            editable_boxes = proposal_boxes
            if not editable_label_file.exists():
                with open(editable_label_file, "w", encoding="utf-8") as ef:
                    for b in editable_boxes:
                        c = b["class_id"]
                        xc, yc, bw, bh = b["bbox_norm"]
                        ef.write(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            synced_label_hash = compute_label_file_hash(editable_label_file) if editable_label_file.exists() else None

            annotation_state, is_unannotated = derive_annotation_state(
                editable_boxes, "unreviewed", has_source_labels, has_human_annotation=False
            )
            preview_meta_hash = compute_preview_meta_hash(editable_boxes, annotation_state, is_unannotated)

            record = {
                "frame_id": frame_id,
                "canonical_source_id": source_frame_id,
                "camera": camera,
                "video_name": video_name,
                "frame_idx": frame_idx,
                "lighting_type": lighting,
                "diagnostic_group": diag_group,
                "diagnostic_group_label": DIAGNOSTIC_GROUP_LABELS.get(diag_group, diag_group),
                "training_exposure": exposure,
                "nearby_training_delta_frames": delta_f,
                "nearby_training_delta_sec": delta_t,
                "clean_input_status": extraction_method if clean_extracted else "failed",
                "clean_image_file": f"images/{frame_id}.jpg" if clean_extracted else None,
                "preview_image_file": f"previews/{frame_id}_preview.jpg" if clean_extracted else None,
                "proposal_label_file": f"proposals/{frame_id}.txt",
                "editable_label_file": f"annotations/labels/{frame_id}.txt",
                "synced_label_hash": synced_label_hash,
                "preview_meta_hash": preview_meta_hash,
                "dimensions": list(dims) if dims else None,
                "annotation_state": annotation_state,
                "is_unannotated": is_unannotated,
                "review_status": "unreviewed",
                "is_ambiguous": False,
                "reviewer_notes": "",
                "selection_reasons": reasons,
                "artifact_references": artifact_refs,
                "boxes": editable_boxes
            }

        # 4. Visual preview: render if missing or verify if existing
        preview_dest = previews_dir / f"{frame_id}_preview.jpg"
        if clean_extracted:
            if not preview_dest.exists():
                preview_ok = render_box_preview_image(
                    clean_image_path=clean_img_dest,
                    boxes=record.get("boxes", []),
                    output_preview_path=preview_dest,
                    is_unannotated=record.get("is_unannotated", False),
                    annotation_state=record.get("annotation_state")
                )
                if not preview_ok or not verify_image_file(preview_dest)[0]:
                    extraction_failures.append({
                        "frame_id": frame_id,
                        "source_frame_id": source_frame_id,
                        "video": video_name,
                        "frame_idx": frame_idx,
                        "reason": f"Failed to generate valid preview image at {preview_dest}"
                    })
                    clean_extracted = False
            else:
                is_pv, err, _ = verify_image_file(preview_dest)
                if not is_pv:
                    extraction_failures.append({
                        "frame_id": frame_id,
                        "source_frame_id": source_frame_id,
                        "video": video_name,
                        "frame_idx": frame_idx,
                        "reason": f"Existing preview image failed verification: {err}"
                    })
                    clean_extracted = False

        records.append(record)

    # 6. Retain existing annotation records omitted from the incoming manifest
    for prev_fid, prev_rec in existing_annotations.items():
        if prev_fid not in processed_frame_ids:
            records.append(prev_rec)
            processed_frame_ids.add(prev_fid)

    # Direct users to explicit synchronization when edits are pending
    pending_sync_count = 0
    for r in records:
        fid = r["frame_id"]
        lbl_file = annos_labels_dir / f"{fid}.txt"
        if lbl_file.exists():
            lbl_hash = compute_label_file_hash(lbl_file)
            last_hash = r.get("synced_label_hash")
            curr_geom_hash = compute_boxes_label_hash(r.get("boxes", []))
            if last_hash and (lbl_hash != last_hash or curr_geom_hash != last_hash):
                pending_sync_count += 1
            elif not last_hash and lbl_hash != curr_geom_hash:
                pending_sync_count += 1

    if pending_sync_count > 0:
        print(
            f"[NOTICE] {pending_sync_count} candidate(s) have pending unsynchronized edits between "
            f"annotations.json and YOLO label files. Preparation has preserved all existing annotations. "
            f"Run explicit synchronization to align them:\n"
            f"    python tools/prepare_review_pack.py --sync --strategy auto\n"
            f"or specify --strategy from_yolo / from_json if there are competing changes."
        )

    # Save editable annotations JSON
    with open(annos_dir / "annotations.json", "w", encoding="utf-8") as af:
        json.dump(records, af, indent=2)

    now_iso = datetime.now(timezone.utc).isoformat()
    pack_manifest = {
        "metadata": {
            "pack_version": "1.0",
            "description": "YOLO26s Thai Traffic Vision Evaluation Review Pack",
            "generated_at": now_iso,
            "baseline_model_path": str(baseline_pt_path),
            "baseline_checkpoint_sha256": baseline_hash,
            "manifest_source": str(manifest_path),
            "total_candidates": len(records),
            "successfully_prepared_count": len(records) - len(extraction_failures),
            "extraction_failures_count": len(extraction_failures),
            "clean_input_counts": dict(clean_input_counts),
            "diagnostic_group_counts": dict(diagnostic_group_counts),
            "extraction_failures": extraction_failures,
            "canonical_source_groups_count": len(records),
            "human_annotation_policy": {
                "detector_vs_controller_weights": (
                    "Detector classes model visual vehicle morphology only. Traffic signal controllers assign PCE or delay "
                    "weights downstream independently; do NOT assign traffic weights in the detector or annotations."
                ),
                "light_vs_heavy_rule": (
                    "Ordinary pickups (Hilux, D-Max) and pickup-based songthaews belong to class 0 (car). "
                    "Medium/heavy commercial trucks (6/10/18-wheelers) and truck-based songthaews belong to class 3 (truck)."
                ),
                "unannotated_holdouts": (
                    "Historical and holdout frames without source labels start explicitly unannotated, NOT verified background images. "
                    "Annotators must add visible target vehicles from scratch or verify empty backgrounds."
                ),
                "ambiguity_rule": (
                    "Mark ambiguous vehicles with is_ambiguous: true instead of guessing."
                )
            }
        },
        "samples": records
    }

    manifest_out_path = output_dir / "manifest.json"
    with open(manifest_out_path, "w", encoding="utf-8") as mf:
        json.dump(pack_manifest, mf, indent=2)

    # Render review indices
    render_html_review_index(output_dir, records, pack_manifest["metadata"])
    render_markdown_readme(output_dir, records, pack_manifest["metadata"])

    return pack_manifest


def sync_review_pack(
    pack_dir: Path,
    strategy: str = "auto",
    frame_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Synchronizes YOLO labels, structured metadata, and preview images explicitly.
    Preserves subtype/ambiguity metadata using stable instance identities via spatial IoU matching.
    Preflights all selected records for conflicts and invalid annotations before writing anything.
    Stages and validates all outputs before replacing existing human files.
    
    Strategies:
    - 'auto': checks modification hashes; detects two-way conflict if both changed independently.
    - 'from_yolo': updates annotations.json and previews from editable YOLO labels.
    - 'from_json': updates YOLO label files and previews from annotations.json.
    """
    annos_file = pack_dir / "annotations" / "annotations.json"
    if not annos_file.exists():
        raise FileNotFoundError(f"Annotations JSON file not found at {annos_file}")

    try:
        with open(annos_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Annotations JSON must be a list of records: {annos_file}")
    except Exception as e:
        raise ValueError(f"Failed to parse annotations JSON {annos_file}: {e}") from e

    images_dir = pack_dir / "images"
    previews_dir = pack_dir / "previews"
    labels_dir = pack_dir / "annotations" / "labels"

    manifest_file = pack_dir / "manifest.json"
    manifest_samples: Dict[str, Dict[str, Any]] = {}
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as mf:
                p_man = json.load(mf)
            manifest_samples = {s["frame_id"]: s for s in p_man.get("samples", [])}
        except Exception:
            manifest_samples = {}

    # =========================================================================
    # PHASE 1: PREFLIGHT CHECK (Read-Only)
    # Validate all annotations, parse YOLO labels, detect conflicts across ALL records.
    # If ANY conflict or invalid annotation is detected, abort before writing anything.
    # =========================================================================
    plan = []

    for rec in records:
        fid = rec["frame_id"]
        if frame_ids and fid not in frame_ids:
            plan.append({"rec": rec, "skip": True})
            continue

        label_file = labels_dir / f"{fid}.txt"
        clean_img_file = images_dir / f"{fid}.jpg"
        # Support both review pack conventions: {fid}.jpg (v2) or {fid}_preview.jpg (v1)
        if (previews_dir / f"{fid}.jpg").exists() or rec.get("preview_image_file", "").endswith(f"/{fid}.jpg"):
            preview_file = previews_dir / f"{fid}.jpg"
        else:
            preview_file = previews_dir / f"{fid}_preview.jpg"

        # Preflight validation: parse YOLO label file if present
        yolo_boxes: Optional[List[Dict[str, Any]]] = None
        curr_label_hash: Optional[str] = None
        if label_file.exists():
            curr_label_hash = compute_label_file_hash(label_file)
            yolo_boxes = parse_yolo_label_file(label_file)  # Validates syntax & box bounds

        curr_json_boxes = rec.get("boxes", [])
        for b in curr_json_boxes:
            cid = b.get("class_id")
            bbox = b.get("bbox_norm")
            if cid not in THAI_5CLASS_NAMES or not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid box in annotations.json for frame '{fid}': {b}")

        curr_json_geom_hash = compute_boxes_label_hash(curr_json_boxes)
        last_synced_hash = rec.get("synced_label_hash")

        # Determine effective geometry strategy & detect conflicts
        effective_strategy = strategy
        if strategy == "auto":
            if curr_label_hash is None:
                effective_strategy = "from_json"
            elif last_synced_hash is not None:
                yolo_geom_changed = (curr_label_hash != last_synced_hash)
                json_geom_changed = (curr_json_geom_hash != last_synced_hash)
                if yolo_geom_changed and json_geom_changed and curr_label_hash != curr_json_geom_hash:
                    raise ConflictingEditError(
                        f"Conflicting edits detected for frame '{fid}': both YOLO label file and annotations.json "
                        f"were modified independently since last sync. "
                        f"Specify --strategy=from_yolo or --strategy=from_json to resolve."
                    )
                elif yolo_geom_changed:
                    effective_strategy = "from_yolo"
                elif json_geom_changed:
                    effective_strategy = "from_json"
                else:
                    effective_strategy = "in_sync"
            else:
                # No previous sync hash: check identity
                if curr_label_hash == curr_json_geom_hash:
                    effective_strategy = "in_sync"
                else:
                    raise ConflictingEditError(
                        f"Conflicting edits detected for frame '{fid}': both YOLO label file and annotations.json "
                        f"exist with differing contents and no prior sync record. "
                        f"Specify --strategy=from_yolo or --strategy=from_json to resolve."
                    )

        # Compute new box list based on effective strategy
        staged_label_text: Optional[str] = None
        if effective_strategy == "from_yolo":
            if not label_file.exists():
                raise FileNotFoundError(f"YOLO label file missing for frame {fid}: {label_file}")
            new_boxes = match_box_instances_spatial(yolo_boxes, curr_json_boxes, fid)
            target_label_hash = curr_label_hash
            geom_synced = (curr_label_hash != curr_json_geom_hash)
        elif effective_strategy == "from_json":
            new_boxes = curr_json_boxes
            export_boxes = [b for b in new_boxes if not b.get("is_proposal") or b.get("proposal_status") == "accepted"]
            lines = [f"{b['class_id']} {b['bbox_norm'][0]:.6f} {b['bbox_norm'][1]:.6f} {b['bbox_norm'][2]:.6f} {b['bbox_norm'][3]:.6f}\n" for b in export_boxes]
            staged_label_text = "".join(lines)
            target_label_hash = compute_boxes_label_hash(export_boxes)
            geom_synced = (curr_label_hash != target_label_hash)
        else:
            new_boxes = curr_json_boxes
            target_label_hash = curr_label_hash or last_synced_hash or curr_json_geom_hash
            geom_synced = False

        # Derive updated annotation state
        r_status = rec.get("review_status", "unreviewed")
        if r_status in ("rejected", "uncertain"):
            new_annotation_state = r_status
            new_is_unannotated = False
        elif len(new_boxes) > 0:
            new_annotation_state = "annotated"
            new_is_unannotated = False
        elif r_status == "verified":
            new_annotation_state = "verified_empty_background"
            new_is_unannotated = False
        elif rec.get("is_unannotated"):
            new_annotation_state = "unannotated"
            new_is_unannotated = True
        else:
            new_annotation_state = "unreviewed_machine_empty"
            new_is_unannotated = False

        # Separate metadata change detection (subtypes, ambiguity, review_status, annotation_state)
        old_meta_hash = rec.get("preview_meta_hash")
        m_rec = manifest_samples.get(fid)
        if old_meta_hash is None and m_rec:
            old_meta_hash = m_rec.get("preview_meta_hash") or compute_preview_meta_hash(
                m_rec.get("boxes", []), m_rec.get("annotation_state"), m_rec.get("is_unannotated", False)
            )

        new_meta_hash = compute_preview_meta_hash(new_boxes, new_annotation_state, new_is_unannotated)

        meta_changed = False
        if old_meta_hash is not None and old_meta_hash != new_meta_hash:
            meta_changed = True
        if rec.get("annotation_state") != new_annotation_state:
            meta_changed = True
        if rec.get("is_unannotated") != new_is_unannotated:
            meta_changed = True

        if m_rec:
            m_boxes = m_rec.get("boxes", [])
            m_subtypes = [b.get("subtype") for b in m_boxes]
            n_subtypes = [b.get("subtype") for b in new_boxes]
            m_ambiguities = [b.get("is_ambiguous") for b in m_boxes]
            n_ambiguities = [b.get("is_ambiguous") for b in new_boxes]
            if m_subtypes != n_subtypes or m_ambiguities != n_ambiguities:
                meta_changed = True
            if m_rec.get("review_status") != r_status:
                meta_changed = True
            if m_rec.get("annotation_state") != new_annotation_state:
                meta_changed = True
            if m_rec.get("is_ambiguous") != rec.get("is_ambiguous"):
                meta_changed = True

        preview_missing = not preview_file.exists()
        needs_preview_refresh = geom_synced or meta_changed or preview_missing

        plan.append({
            "rec": rec,
            "fid": fid,
            "label_file": label_file,
            "clean_img_file": clean_img_file,
            "preview_file": preview_file,
            "new_boxes": new_boxes,
            "target_label_hash": target_label_hash,
            "staged_label_text": staged_label_text,
            "new_annotation_state": new_annotation_state,
            "new_is_unannotated": new_is_unannotated,
            "new_meta_hash": new_meta_hash,
            "geom_synced": geom_synced,
            "meta_changed": meta_changed,
            "needs_preview_refresh": needs_preview_refresh,
            "skip": False
        })

    # =========================================================================
    # PHASE 2: STAGING & VALIDATION (Isolated Temporary Directory)
    # Stage all planned writes into a temporary directory; validate completely.
    # If preview generation or disk writes fail, human files remain untouched.
    # =========================================================================
    with tempfile.TemporaryDirectory() as stage_dir_str:
        stage_dir = Path(stage_dir_str)
        staged_labels_dir = stage_dir / "labels"
        staged_previews_dir = stage_dir / "previews"
        staged_labels_dir.mkdir(parents=True, exist_ok=True)
        staged_previews_dir.mkdir(parents=True, exist_ok=True)

        labels_to_commit: List[Tuple[Path, Path]] = []
        previews_to_commit: List[Tuple[Path, Path]] = []
        updated_records: List[Dict[str, Any]] = []

        synced_records_count = 0
        updated_previews_count = 0

        for item in plan:
            if item.get("skip"):
                updated_records.append(item["rec"])
                continue

            rec = item["rec"]
            fid = item["fid"]

            up_rec = dict(rec)
            up_rec["boxes"] = item["new_boxes"]
            up_rec["annotation_state"] = item["new_annotation_state"]
            up_rec["is_unannotated"] = item["new_is_unannotated"]
            up_rec["synced_label_hash"] = item["target_label_hash"]
            up_rec["preview_meta_hash"] = item["new_meta_hash"]
            updated_records.append(up_rec)

            if item["geom_synced"] or item["meta_changed"]:
                synced_records_count += 1

            # Stage label file write
            if item["staged_label_text"] is not None:
                staged_label = staged_labels_dir / f"{fid}.txt"
                staged_label.write_text(item["staged_label_text"], encoding="utf-8")
                labels_to_commit.append((staged_label, item["label_file"]))

            # Stage preview render & validate
            if item["needs_preview_refresh"] and item["clean_img_file"].exists():
                staged_preview = staged_previews_dir / item["preview_file"].name
                render_ok = render_box_preview_image(
                    clean_image_path=item["clean_img_file"],
                    boxes=item["new_boxes"],
                    output_preview_path=staged_preview,
                    is_unannotated=item["new_is_unannotated"],
                    annotation_state=item["new_annotation_state"]
                )
                if not render_ok:
                    raise RuntimeError(f"Failed to render preview for frame '{fid}'")
                is_v, err, _ = verify_image_file(staged_preview)
                if not is_v:
                    raise RuntimeError(f"Staged preview image verification failed for frame '{fid}': {err}")
                previews_to_commit.append((staged_preview, item["preview_file"]))
                updated_previews_count += 1

        # Stage and validate annotations.json
        staged_annos_file = stage_dir / "annotations.json"
        staged_annos_file.write_text(json.dumps(updated_records, indent=2), encoding="utf-8")
        with open(staged_annos_file, "r", encoding="utf-8") as f:
            test_load = json.load(f)
            if len(test_load) != len(records):
                raise RuntimeError("Staged annotations.json record count mismatch")

        # Stage updated manifest if it exists
        manifest_file = pack_dir / "manifest.json"
        staged_manifest_file: Optional[Path] = None
        pack_manifest: Optional[Dict[str, Any]] = None
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                pack_manifest = json.load(f)
            pack_manifest["samples"] = updated_records
            pack_manifest["metadata"]["last_synced_at"] = datetime.now(timezone.utc).isoformat()
            staged_manifest_file = stage_dir / "manifest.json"
            staged_manifest_file.write_text(json.dumps(pack_manifest, indent=2), encoding="utf-8")

        # =========================================================================
        # PHASE 3: ATOMIC COMMIT WITH PRE-COMMIT BACKUP & ROLLBACK
        # 1. Preserve recoverable originals before replacing destination files.
        # 2. Use same-directory temporary files and atomic replacement for individual files.
        # 3. If any replacement fails, restore every already-replaced human file
        #    and remove newly created destinations.
        # 4. Retain recovery backups and report their location if rollback itself fails.
        # 5. Do not report success or suppress commit/recovery errors.
        # =========================================================================
        commit_queue: List[Tuple[Path, Path]] = []
        commit_queue.extend(labels_to_commit)
        commit_queue.extend(previews_to_commit)
        commit_queue.append((staged_annos_file, annos_file))
        if staged_manifest_file and manifest_file.exists():
            commit_queue.append((staged_manifest_file, manifest_file))

        recovery_dir = pack_dir / ".recovery_backups" / f"backup_{os.getpid()}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        recovery_dir.mkdir(parents=True, exist_ok=True)

        applied_records: List[Dict[str, Any]] = []

        try:
            for idx, (src, dst) in enumerate(commit_queue):
                existed = dst.exists()
                backup_path: Optional[Path] = None
                if existed:
                    backup_path = recovery_dir / f"{idx:04d}_{dst.name}"
                    shutil.copy2(dst, backup_path)

                atomic_replace_file(src, dst)
                applied_records.append({
                    "dst": dst,
                    "existed": existed,
                    "backup_path": backup_path
                })
        except Exception as commit_exc:
            rollback_errors: List[str] = []
            for entry in reversed(applied_records):
                target = entry["dst"]
                if entry["existed"]:
                    backup = entry["backup_path"]
                    try:
                        if backup and backup.exists():
                            atomic_replace_file(backup, target)
                    except Exception as rb_err:
                        rollback_errors.append(f"Failed to restore {target} from {backup}: {rb_err}")
                else:
                    try:
                        if target.exists():
                            target.unlink()
                    except Exception as rb_err:
                        rollback_errors.append(f"Failed to remove new file {target}: {rb_err}")

            if rollback_errors:
                err_msg = (
                    f"Commit failed: {commit_exc}. "
                    f"Additionally, rollback failed for {len(rollback_errors)} file(s): {'; '.join(rollback_errors)}. "
                    f"Recovery backups are retained at: {recovery_dir.resolve()}"
                )
                raise RuntimeError(err_msg) from commit_exc

            # Clean up recovery directory on successful rollback
            shutil.rmtree(recovery_dir, ignore_errors=True)
            try:
                (pack_dir / ".recovery_backups").rmdir()
            except OSError:
                pass
            raise

        # Clean up recovery directory after all replacements succeed
        shutil.rmtree(recovery_dir, ignore_errors=True)
        try:
            (pack_dir / ".recovery_backups").rmdir()
        except OSError:
            pass

    # Update review index and readme if manifest was updated
    if pack_manifest:
        try:
            render_html_review_index(pack_dir, updated_records, pack_manifest["metadata"])
            render_markdown_readme(pack_dir, updated_records, pack_manifest["metadata"])
        except Exception:
            pass

    return {
        "status": "success",
        "synced_records": synced_records_count,
        "updated_previews": updated_previews_count,
        "total_records": len(records)
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Evaluation & Annotation Review Pack for YOLO26s (Batch 2)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/eval_sampling_manifest.json"),
        help="Path to evaluation sampling manifest"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/review_pack_v1"),
        help="Destination directory for review pack"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("videos"),
        help="Directory containing surveillance video recordings"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/multiclass_dataset"),
        help="Path to compiled multiclass dataset"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Synchronize existing review pack annotations, YOLO labels, and previews"
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "from_yolo", "from_json"],
        default="auto",
        help="Synchronization strategy when running --sync"
    )

    args = parser.parse_args()

    if args.sync:
        print(f"Synchronizing review pack in {args.output_dir} using strategy '{args.strategy}'...")
        res = sync_review_pack(pack_dir=args.output_dir, strategy=args.strategy)
        print(f"Sync complete: {res['synced_records']} records synced, {res['updated_previews']} previews updated.")
        return

    # Fallback to data/eval_sampling_manifest.json if docs path does not exist
    manifest_p = args.manifest
    if not manifest_p.exists() and Path("data/eval_sampling_manifest.json").exists():
        manifest_p = Path("data/eval_sampling_manifest.json")

    print(f"Loading manifest from: {manifest_p}")
    print(f"Preparing review pack in: {args.output_dir}...")
    manifest = prepare_review_pack(
        manifest_path=manifest_p,
        output_dir=args.output_dir,
        videos_dir=args.videos_dir,
        dataset_dir=args.dataset_dir
    )

    meta = manifest["metadata"]
    print("\n" + "=" * 70)
    print("Review Pack Preparation Completed Successfully!")
    print(f"  -> Total Candidates: {meta['total_candidates']}")
    print(f"  -> Successfully Prepared: {meta['successfully_prepared_count']}")
    print(f"  -> Extraction Failures: {meta['extraction_failures_count']}")
    print(f"  -> Diagnostic Groups: {meta['diagnostic_group_counts']}")
    print(f"  -> Baseline Checkpoint SHA256: {meta['baseline_checkpoint_sha256'][:16]}...")
    print(f"  -> Review Pack Dashboard: {args.output_dir / 'review_index.html'}")
    print(f"  -> Review Pack Guide: {args.output_dir / 'README.md'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

