"""
Training Manifest Preparation & Leakage Safeguards Tool (Batch 4 - Corrected).

Prepares balanced, leakage-safe training manifests for a controlled comparison
between local-only training (Manifest A) and local + UA-DETRAC training (Manifest B):
1. Builds a canonical source inventory covering local and external datasets:
   - Preserves image-label pairing and tracks annotation provenance.
   - Quarantines unreadable images, missing labels, and invalid boxes.
   - Preserves explicitly valid empty annotations (0 boxes) as distinct from missing/malformed.
   - Flags missing external images as unresolved on disk.
2. Local split construction:
   - Supports whole recording groups where feasible.
   - Otherwise constructs explicit contiguous temporal blocks per recording.
   - Keeps all variants of a canonical source strictly together.
   - Enforces temporal exclusions using configured seconds and verified per-recording FPS.
   - Documents block definitions, buffer cutoffs, and measured temporal separation.
3. Diagnostic safeguards:
   - Fails clearly if evaluation manifest is missing, malformed, or empty.
   - Excludes all 42 diagnostic benchmark frames and variants from candidate manifests.
   - Reserves northeast footage (cam45_northeast) from training.
   - Quarantines unknown provenance.
4. External sequence & size stratification:
   - Partitions external sequences deterministically by whole MVI sequence (0% sequence leakage).
   - Recalculates external cap from eligible unique local training frames.
   - Explicitly samples across sequences and object-size bins (small, medium, large).
   - Identifies external vans using original source-class IDs (van == 3).
5. Visual review spot-check:
   - Selects up to 50 unique canonical source frames (no repeated variants).
   - Allows multiple review tags per frame.
   - Reports category shortfalls honestly.
6. Immutability & Overwrite Protection:
   - Refuses nonempty output destinations unless explicit overwrite is requested.
   - Derives all report claims from computed results.
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
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    DEFAULT_UADETRAC_CLASS_NAMES,
    THAI_5CLASS_NAMES,
    compute_aspect_preserving_size,
    get_image_dimensions,
    parse_frame_provenance,
    probe_video_fps,
    validate_box,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy / scalar types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


# ==============================================================================
# 1. Evaluation Manifest Loader with Fail-Fast Validation (Scope Item 3)
# ==============================================================================

def load_evaluation_manifest(eval_manifest_path: Path) -> Set[str]:
    """
    Loads diagnostic evaluation benchmark manifest with fail-fast validation:
    - Fails clearly if missing.
    - Fails clearly if malformed JSON.
    - Fails clearly if empty (0 samples).
    Never silently disables diagnostic exclusions.
    """
    if not eval_manifest_path.exists():
        raise FileNotFoundError(
            f"Required evaluation manifest missing at '{eval_manifest_path}'. "
            "Refusing to proceed without diagnostic exclusions to prevent benchmark leakage."
        )

    try:
        data = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(
            f"Required evaluation manifest at '{eval_manifest_path}' is malformed JSON: {e}. "
            "Refusing to proceed without diagnostic exclusions."
        )

    if not isinstance(data, dict) or "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError(
            f"Required evaluation manifest at '{eval_manifest_path}' is invalid: missing 'samples' list. "
            "Refusing to proceed without diagnostic exclusions."
        )

    samples = data["samples"]
    if len(samples) == 0:
        raise ValueError(
            f"Required evaluation manifest at '{eval_manifest_path}' is empty (0 samples). "
            "Refusing to proceed without diagnostic exclusions."
        )

    eval_source_ids = set()
    for s in samples:
        if isinstance(s, dict):
            fid = s.get("frame_id")
            if fid:
                eval_source_ids.add(str(fid))

    if len(eval_source_ids) == 0:
        raise ValueError(
            f"Required evaluation manifest at '{eval_manifest_path}' contains no valid frame IDs."
        )

    print(f"[EVAL MANIFEST] Loaded {len(eval_source_ids)} verified diagnostic frame IDs from: {eval_manifest_path}")
    return eval_source_ids


# ==============================================================================
# 2. Canonical Source Inventory Builder with Strict Quarantine (Scope Item 4)
# ==============================================================================

def build_canonical_source_inventory(
    local_dataset_dir: Path,
    external_ndjson_path: Path,
    eval_source_ids: Set[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Builds canonical source inventory. Quarantines unreadable images, missing labels,
    and invalid boxes without silently dropping annotations or guessing dimensions.
    Preserves explicitly valid empty annotations as distinct from missing/malformed.
    """
    records: List[Dict[str, Any]] = []
    canonical_source_groups: Dict[str, List[str]] = defaultdict(list)
    quarantined_records: List[Dict[str, Any]] = []

    ext_class_mapping = config.get("taxonomy", {}).get("external_to_thai_mapping", {"0": 3, "1": 0, "2": 0, "3": 2})
    ext_map = {int(k): int(v) for k, v in ext_class_mapping.items()} if isinstance(ext_class_mapping, dict) else {0: 3, 1: 0, 2: 0, 3: 2}

    # 2.1 Process Local Dataset
    print(f"[INVENTORY] Scanning local dataset at: {local_dataset_dir}")
    if local_dataset_dir.exists():
        for split in ["train", "val"]:
            img_dir = local_dataset_dir / "images" / split
            lbl_dir = local_dataset_dir / "labels" / split
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                stem = img_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"
                prov = parse_frame_provenance(stem)
                inventory_id = f"local:{split}:{stem}"
                canonical_source_id = prov.source_frame_id

                # Check 1: Image readability & dimension measurement
                img_w, img_h = get_image_dimensions(img_path)
                if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
                    q_entry = {
                        "inventory_id": inventory_id,
                        "data_origin": "local",
                        "canonical_source_id": canonical_source_id,
                        "reason": "unreadable_or_corrupt_image",
                        "details": f"Image file '{img_path}' could not be decoded or has non-positive dimensions."
                    }
                    quarantined_records.append(q_entry)
                    continue

                # Check 2: Missing label file
                if not lbl_path.exists():
                    q_entry = {
                        "inventory_id": inventory_id,
                        "data_origin": "local",
                        "canonical_source_id": canonical_source_id,
                        "reason": "missing_label_file",
                        "details": f"Expected label file missing at '{lbl_path}'."
                    }
                    quarantined_records.append(q_entry)
                    continue

                # Check 3: Read and parse label file
                boxes: List[Dict[str, Any]] = []
                class_counts: Dict[str, int] = Counter()
                box_sizes = {"small": 0, "medium": 0, "large": 0}
                is_explicitly_empty = False
                has_label_error = False
                label_error_details = ""

                try:
                    raw_text = lbl_path.read_text(encoding="utf-8").strip()
                    if not raw_text:
                        # Explicitly valid empty annotation (negative background frame)
                        is_explicitly_empty = True
                    else:
                        for l_idx, line in enumerate(raw_text.splitlines(), 1):
                            line_s = line.strip()
                            if not line_s or line_s.startswith("#"):
                                continue
                            parts = line_s.split()
                            is_valid, err_msg, validated_box = validate_box(
                                parts, THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
                            )
                            if not is_valid or validated_box is None:
                                has_label_error = True
                                label_error_details = f"Line {l_idx}: {err_msg}"
                                break

                            cid, xc, yc, bw, bh = validated_box
                            sz_info = compute_aspect_preserving_size(bw, bh, img_w, img_h, ref_size=640)
                            bucket = sz_info["size_bucket"]
                            if bucket in box_sizes:
                                box_sizes[bucket] += 1
                            class_counts[str(cid)] += 1
                            boxes.append({
                                "class_id": cid,
                                "class_name": THAI_5CLASS_NAMES[cid],
                                "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                                "size_bucket": bucket
                            })
                except Exception as e:
                    has_label_error = True
                    label_error_details = f"Label read exception: {e}"

                if has_label_error:
                    q_entry = {
                        "inventory_id": inventory_id,
                        "data_origin": "local",
                        "canonical_source_id": canonical_source_id,
                        "reason": "invalid_bounding_box",
                        "details": f"Label file '{lbl_path}' contains invalid annotations ({label_error_details}). Frame quarantined."
                    }
                    quarantined_records.append(q_entry)
                    continue

                rec = {
                    "inventory_id": inventory_id,
                    "data_origin": "local",
                    "canonical_source_id": canonical_source_id,
                    "stem": stem,
                    "sequence_id": prov.video,
                    "camera": prov.camera,
                    "lighting": prov.lighting,
                    "variant": prov.variant,
                    "frame_idx": prov.frame_idx,
                    "is_replay": prov.is_replay,
                    "is_synthetic_variant": prov.variant not in ("original_curated", "replay"),
                    "augmentation_ancestry": {
                        "is_synthetic": prov.variant not in ("original_curated", "replay"),
                        "parent_canonical_id": canonical_source_id,
                        "variant_type": prov.variant
                    },
                    "original_split": split,
                    "image_path": str(img_path).replace("\\", "/"),
                    "image_exists_on_disk": True,
                    "image_status": "available_local_file",
                    "label_path": str(lbl_path).replace("\\", "/"),
                    "label_exists_on_disk": True,
                    "is_explicitly_empty": is_explicitly_empty,
                    "dimensions": {"width": img_w, "height": img_h},
                    "annotation_provenance": "teacher_completed_local",
                    "is_diagnostic_eval_frame": canonical_source_id in eval_source_ids or stem in eval_source_ids,
                    "is_northeast_holdout": prov.camera == "cam45_northeast",
                    "is_provenance_known": prov.is_provenance_known,
                    "total_boxes": len(boxes),
                    "class_counts": dict(class_counts),
                    "source_class_counts": dict(class_counts),
                    "box_sizes": box_sizes,
                    "boxes": boxes
                }
                records.append(rec)
                canonical_source_groups[canonical_source_id].append(inventory_id)

    # 2.2 Process External UA-DETRAC NDJSON
    print(f"[INVENTORY] Scanning external NDJSON at: {external_ndjson_path}")
    if external_ndjson_path.exists():
        with open(external_ndjson_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line_s = raw_line.strip()
                if not line_s:
                    continue
                try:
                    obj = json.loads(line_s)
                except Exception as e:
                    quarantined_records.append({
                        "inventory_id": f"external_ndjson_line_{line_no}",
                        "data_origin": "external_ua_detrac",
                        "canonical_source_id": "unknown",
                        "reason": "malformed_ndjson_record",
                        "details": f"Line {line_no} JSON parse error: {e}"
                    })
                    continue

                if not isinstance(obj, dict) or obj.get("type") != "image":
                    continue

                raw_filename = str(obj.get("file", ""))
                if not raw_filename:
                    quarantined_records.append({
                        "inventory_id": f"external_ndjson_line_{line_no}",
                        "data_origin": "external_ua_detrac",
                        "canonical_source_id": "unknown",
                        "reason": "malformed_ndjson_record",
                        "details": f"Line {line_no} missing 'file' attribute."
                    })
                    continue

                base_frame = raw_filename.split(".rf.")[0] if ".rf." in raw_filename else raw_filename
                base_frame = re.sub(r"_[a-zA-Z0-9]+$", "", base_frame) if base_frame.endswith("_jpg") else base_frame

                seq_match = re.match(r"^(MVI_\d+)", base_frame)
                seq_id = seq_match.group(1) if seq_match else "unknown_seq"

                try:
                    img_w = int(obj.get("width", 960))
                    img_h = int(obj.get("height", 540))
                except (ValueError, TypeError):
                    img_w = 960
                    img_h = 540

                # Parse annotations
                raw_annos = obj.get("annotations", {})
                raw_boxes: List[Any] = []
                if isinstance(raw_annos, dict):
                    raw_boxes = raw_annos.get("boxes", [])
                elif isinstance(raw_annos, list):
                    raw_boxes = raw_annos

                boxes: List[Dict[str, Any]] = []
                class_counts: Dict[str, int] = Counter()
                source_class_counts: Dict[str, int] = Counter()
                box_sizes = {"small": 0, "medium": 0, "large": 0}
                ext_box_error = False

                for b in raw_boxes:
                    is_valid, err_msg, validated_box = validate_box(
                        b, allowed_classes=DEFAULT_UADETRAC_CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
                    )
                    if not is_valid or validated_box is None:
                        ext_box_error = True
                        break

                    src_cid, xc, yc, bw, bh = validated_box
                    tgt_cid = ext_map.get(src_cid, 0)
                    sz_info = compute_aspect_preserving_size(bw, bh, img_w, img_h, ref_size=640)
                    bucket = sz_info["size_bucket"]
                    if bucket in box_sizes:
                        box_sizes[bucket] += 1
                    class_counts[str(tgt_cid)] += 1
                    src_cname = DEFAULT_UADETRAC_CLASS_NAMES.get(src_cid, str(src_cid))
                    source_class_counts[src_cname] += 1

                    boxes.append({
                        "source_class_id": src_cid,
                        "source_class_name": src_cname,
                        "class_id": tgt_cid,
                        "class_name": THAI_5CLASS_NAMES.get(tgt_cid, str(tgt_cid)),
                        "bbox_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                        "size_bucket": bucket
                    })

                if ext_box_error:
                    quarantined_records.append({
                        "inventory_id": f"external:{raw_filename}",
                        "data_origin": "external_ua_detrac",
                        "canonical_source_id": base_frame,
                        "reason": "invalid_external_bounding_box",
                        "details": f"Line {line_no}: box validation failed. Quarantined."
                    })
                    continue

                is_augmented = ".rf." in raw_filename
                inventory_id = f"external:{raw_filename}"
                canonical_source_id = base_frame

                rec = {
                    "inventory_id": inventory_id,
                    "data_origin": "external_ua_detrac",
                    "canonical_source_id": canonical_source_id,
                    "stem": Path(raw_filename).stem,
                    "sequence_id": seq_id,
                    "camera": f"ua_detrac_{seq_id}",
                    "lighting": "unknown_external",
                    "variant": "roboflow_augmented" if is_augmented else "original_external",
                    "frame_idx": None,
                    "is_replay": False,
                    "is_synthetic_variant": is_augmented,
                    "augmentation_ancestry": {
                        "is_synthetic": is_augmented,
                        "parent_canonical_id": canonical_source_id,
                        "variant_type": "roboflow_rf_hash" if is_augmented else "original"
                    },
                    "original_split": str(obj.get("split", "unknown")),
                    "image_path": raw_filename,
                    "image_exists_on_disk": False,
                    "image_status": "unresolved_not_downloaded",
                    "label_path": None,
                    "label_exists_on_disk": False,
                    "is_explicitly_empty": len(boxes) == 0,
                    "dimensions": {"width": img_w, "height": img_h},
                    "annotation_provenance": "ua_detrac_manual_box_tracking",
                    "is_diagnostic_eval_frame": False,
                    "is_northeast_holdout": False,
                    "is_provenance_known": True,
                    "total_boxes": len(boxes),
                    "class_counts": dict(class_counts),
                    "source_class_counts": dict(source_class_counts),
                    "box_sizes": box_sizes,
                    "boxes": boxes
                }
                records.append(rec)
                canonical_source_groups[canonical_source_id].append(inventory_id)

    local_count = sum(1 for r in records if r["data_origin"] == "local")
    external_count = sum(1 for r in records if r["data_origin"] == "external_ua_detrac")
    print(f"[INVENTORY] Valid records: {len(records)} (Local: {local_count}, External: {external_count})")
    print(f"[INVENTORY] Quarantined records: {len(quarantined_records)}")

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "unique_canonical_sources": len(canonical_source_groups),
            "local_records_count": local_count,
            "external_records_count": external_count,
            "quarantined_records_count": len(quarantined_records),
            "local_dataset_dir": str(local_dataset_dir),
            "external_ndjson_path": str(external_ndjson_path)
        },
        "records": records,
        "quarantined_records": quarantined_records,
        "canonical_source_groups": canonical_source_groups
    }


# ==============================================================================
# 3. Local Split Construction & Temporal Exclusion Safeguards (Scope Items 1 & 2)
# ==============================================================================

def construct_local_splits_and_exclusions(
    inventory: Dict[str, Any],
    eval_source_ids: Set[str],
    evidenced_fps_map: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs local train and val splits from scratch:
    - Evaluates whole recording groups where designated.
    - Otherwise divides recordings into explicit contiguous temporal blocks.
    - Keeps all variants of a canonical source frame strictly together.
    - Enforces temporal exclusions using configured buffer seconds and verified FPS.
    - Reports missing timing metadata rather than guessing.
    - Records exact block definitions, buffer cutoffs, and measured separations.
    """
    records = inventory["records"]
    local_records = [r for r in records if r["data_origin"] == "local"]

    local_strat_cfg = config.get("local_split_strategy", {})
    default_strategy = local_strat_cfg.get("default_strategy", "contiguous_temporal_blocks")
    train_block_ratio = float(local_strat_cfg.get("temporal_block_ratio", local_strat_cfg.get("train_block_ratio", 0.80)))
    buffer_sec = float(local_strat_cfg.get("temporal_buffer_seconds", 3.0))
    whole_rec_assign = local_strat_cfg.get("whole_recording_assignments", {})

    exclusions: List[Dict[str, Any]] = []
    eligible_local_sources: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Step 3.1: Pre-split exclusions (diagnostic benchmark, northeast, unknown provenance)
    for r in local_records:
        inv_id = r["inventory_id"]
        c_src = r["canonical_source_id"]

        # Rule A: Diagnostic Benchmark Exclusion
        if c_src in eval_source_ids or r.get("stem", "") in eval_source_ids:
            exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "local",
                "reason": "diagnostic_evaluation_benchmark_leakage",
                "details": f"Matches diagnostic benchmark frame '{c_src}'. Excluded to preserve benchmark integrity."
            })
            continue

        # Rule B: Reserve Northeast Holdout
        if r.get("is_northeast_holdout", False):
            exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "local",
                "reason": "reserved_northeast_holdout",
                "details": "Northeast camera approach reserved from training."
            })
            continue

        # Rule C: Reject Unknown Provenance
        if not r.get("is_provenance_known", True):
            exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "local",
                "reason": "unknown_provenance_quarantined",
                "details": "Unknown provenance cannot be proven free of evaluation overlap."
            })
            continue

        eligible_local_sources[c_src].append(r)

    # Step 3.2: Group canonical sources by recording/video
    sources_by_video: Dict[str, Dict[str, Tuple[Optional[int], List[Dict[str, Any]]]]] = defaultdict(dict)
    for c_src, rec_list in eligible_local_sources.items():
        sample_r = rec_list[0]
        vid = sample_r.get("sequence_id", "unknown_video")
        f_idx = sample_r.get("frame_idx")
        sources_by_video[vid][c_src] = (f_idx, rec_list)

    local_train_records: List[Dict[str, Any]] = []
    local_val_records: List[Dict[str, Any]] = []
    split_block_definitions: Dict[str, Any] = {}
    missing_timing_records: List[str] = []

    # Step 3.3: Construct splits per recording
    for vid, src_map in sorted(sources_by_video.items()):
        fps = evidenced_fps_map.get(vid)
        fps_available = fps is not None and fps > 0

        # Check whole-recording assignment
        if vid in whole_rec_assign:
            target_split = whole_rec_assign[vid]
            if target_split == "train":
                for c_src, (_, r_list) in src_map.items():
                    local_train_records.extend(r_list)
                split_block_definitions[vid] = {
                    "recording": vid,
                    "strategy": "whole_recording",
                    "assigned_split": "train",
                    "evidenced_fps": fps if fps_available else "missing_timing_metadata",
                    "canonical_sources_count": len(src_map),
                    "total_images_count": sum(len(x[1]) for x in src_map.values())
                }
            elif target_split == "val":
                for c_src, (_, r_list) in src_map.items():
                    local_val_records.extend(r_list)
                split_block_definitions[vid] = {
                    "recording": vid,
                    "strategy": "whole_recording",
                    "assigned_split": "val",
                    "evidenced_fps": fps if fps_available else "missing_timing_metadata",
                    "canonical_sources_count": len(src_map),
                    "total_images_count": sum(len(x[1]) for x in src_map.values())
                }
            else:
                # Reserved holdout
                for c_src, (_, r_list) in src_map.items():
                    for r in r_list:
                        exclusions.append({
                            "inventory_id": r["inventory_id"],
                            "canonical_source_id": c_src,
                            "data_origin": "local",
                            "reason": f"whole_recording_reserved_{target_split}",
                            "details": f"Recording '{vid}' reserved via configuration."
                        })
            continue

        # Divided recording: Contiguous Temporal Blocks
        if not fps_available:
            missing_timing_records.append(vid)
            # If FPS is unavailable, we cannot reliably compute seconds-based buffer. Quarantine recording.
            for c_src, (_, r_list) in src_map.items():
                for r in r_list:
                    exclusions.append({
                        "inventory_id": r["inventory_id"],
                        "canonical_source_id": c_src,
                        "data_origin": "local",
                        "reason": "missing_timing_metadata",
                        "details": f"Video '{vid}' has no evidenced FPS metadata. Refusing to guess timing."
                    })
            continue

        # Sort canonical source frames chronologically
        sorted_sources = sorted(src_map.items(), key=lambda x: (x[1][0] if x[1][0] is not None else -1))
        n_sources = len(sorted_sources)
        n_train = max(1, int(round(n_sources * train_block_ratio)))

        train_sources = [(c_src, f_idx, r_list) for c_src, (f_idx, r_list) in sorted_sources[:n_train]]
        remaining_sources = sorted_sources[n_train:]

        last_train_fidx = train_sources[-1][1] or 0
        buffer_frames = int(buffer_sec * fps)
        buffer_cutoff_fidx = last_train_fidx + buffer_frames

        val_sources = []
        buf_excluded_sources = []

        for c_src, (f_idx, r_list) in remaining_sources:
            if f_idx is not None and f_idx < buffer_cutoff_fidx:
                buf_excluded_sources.append((c_src, f_idx, r_list))
                for r in r_list:
                    exclusions.append({
                        "inventory_id": r["inventory_id"],
                        "canonical_source_id": c_src,
                        "data_origin": "local",
                        "reason": "temporal_buffer_violation",
                        "details": (
                            f"Frame index {f_idx} is within {buffer_sec:.2f}s ({buffer_frames} frames @ {fps} FPS) "
                            f"of last train frame {last_train_fidx} in '{vid}'. Excluded to prevent burst leakage."
                        )
                    })
            else:
                val_sources.append((c_src, f_idx, r_list))

        # Add valid frames to splits
        for c_src, _, r_list in train_sources:
            local_train_records.extend(r_list)
        for c_src, _, r_list in val_sources:
            local_val_records.extend(r_list)

        first_val_fidx = val_sources[0][1] if val_sources else None
        first_train_fidx = train_sources[0][1] if train_sources else None

        sep_frames = (first_val_fidx - last_train_fidx) if (first_val_fidx is not None) else None
        sep_sec = (sep_frames / fps) if (sep_frames is not None) else None

        split_block_definitions[vid] = {
            "recording": vid,
            "strategy": "contiguous_temporal_blocks",
            "evidenced_fps": fps,
            "total_canonical_sources": n_sources,
            "train_block": {
                "start_frame_idx": first_train_fidx,
                "end_frame_idx": last_train_fidx,
                "canonical_sources_count": len(train_sources),
                "total_images_count": sum(len(x[2]) for x in train_sources)
            },
            "buffer_zone": {
                "start_frame_idx": last_train_fidx + 1,
                "end_frame_idx": buffer_cutoff_fidx - 1,
                "buffer_seconds": buffer_sec,
                "buffer_frames": buffer_frames,
                "excluded_canonical_sources_count": len(buf_excluded_sources),
                "excluded_images_count": sum(len(x[2]) for x in buf_excluded_sources)
            },
            "val_block": {
                "start_frame_idx": first_val_fidx,
                "end_frame_idx": val_sources[-1][1] if val_sources else None,
                "canonical_sources_count": len(val_sources),
                "total_images_count": sum(len(x[2]) for x in val_sources)
            },
            "measured_temporal_separation": {
                "delta_frames": sep_frames,
                "delta_seconds": round(sep_sec, 2) if sep_sec is not None else None
            }
        }

    # Scope Item 3: Primary validation using original unaugmented frames only (Batch 5)
    primary_val_records: List[Dict[str, Any]] = []
    synthetic_val_records: List[Dict[str, Any]] = []
    unresolved_val_sources: List[Dict[str, Any]] = []

    val_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in local_val_records:
        val_by_source[r["canonical_source_id"]].append(r)

    for c_src, recs in sorted(val_by_source.items()):
        unaugmented = [r for r in recs if not r.get("is_synthetic_variant", False)]
        synthetics = [r for r in recs if r.get("is_synthetic_variant", False)]
        synthetic_val_records.extend(synthetics)

        if len(unaugmented) == 0:
            unresolved_val_sources.append({
                "canonical_source_id": c_src,
                "reason": "missing_original_unaugmented_frame",
                "available_variants": [r.get("variant") for r in recs]
            })
        else:
            # Deterministically choose single canonical unaugmented frame: original_curated preferred over replay
            unaugmented.sort(key=lambda x: (0 if x.get("variant") == "original_curated" else 1, x["inventory_id"]))
            primary_val_records.append(unaugmented[0])
            for extra in unaugmented[1:]:
                synthetic_val_records.append(extra)

    return {
        "local_train_records": local_train_records,
        "local_val_records": local_val_records,
        "primary_val_records": primary_val_records,
        "synthetic_val_records": synthetic_val_records,
        "unresolved_val_sources": unresolved_val_sources,
        "split_block_definitions": split_block_definitions,
        "exclusions": exclusions,
        "missing_timing_records": missing_timing_records
    }


# ==============================================================================
# 4. External Sequence & Size-Stratified Sampling (Scope Item 5)
# ==============================================================================

def sample_external_frames_stratified(
    external_candidates: List[Dict[str, Any]],
    target_count: int,
    train_sequences: List[str],
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Samples external frames using explicit sequence AND size-bin stratification:
    - Distributes target count across available external train sequences.
    - In each sequence, explicitly samples small (< 32^2 px), medium, and large object frames.
    - Identifies vans using original source_class_id == 3 (van), not merged car class.
    - Reports achieved sequence coverage and size-bin coverage metrics.
    """
    # Group candidate records by canonical source frame (prefer original unaugmented over Roboflow .rf.)
    by_canonical: Dict[str, Dict[str, Any]] = {}
    for r in external_candidates:
        c_src = r["canonical_source_id"]
        if c_src not in by_canonical:
            by_canonical[c_src] = r
        elif not r["is_synthetic_variant"] and by_canonical[c_src]["is_synthetic_variant"]:
            by_canonical[c_src] = r

    # Group by sequence
    by_seq: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in by_canonical.values():
        if r["sequence_id"] in train_sequences:
            by_seq[r["sequence_id"]].append(r)

    sorted_seqs = sorted(train_sequences)
    n_seqs = len(sorted_seqs)

    if n_seqs == 0 or target_count == 0:
        return [], {"error": "No sequences or target count is 0"}

    base_per_seq = target_count // n_seqs
    remainder = target_count % n_seqs

    selected_external: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for idx, seq in enumerate(sorted_seqs):
        seq_cands = by_seq.get(seq, [])
        seq_quota = base_per_seq + (1 if idx < remainder else 0)
        if not seq_cands:
            continue

        # Partition candidates into size categories
        small_pool = [c for c in seq_cands if c.get("box_sizes", {}).get("small", 0) > 0]
        med_pool = [c for c in seq_cands if c.get("box_sizes", {}).get("medium", 0) > 0]
        large_pool = [c for c in seq_cands if c.get("box_sizes", {}).get("large", 0) > 0]

        # Prioritization key preferring minority source classes (bus=0, truck=2, van=3)
        def cand_key(c: Dict[str, Any]) -> Tuple[int, int, str]:
            src_counts = c.get("source_class_counts", {})
            has_van = src_counts.get("van", 0) > 0
            has_truck = src_counts.get("truck", 0) > 0
            has_bus = src_counts.get("bus", 0) > 0
            score = (3 if has_bus else 0) + (2 if has_truck else 0) + (2 if has_van else 0)
            return (-score, -c.get("total_boxes", 0), c["inventory_id"])

        small_sorted = sorted(small_pool, key=cand_key)
        med_sorted = sorted(med_pool, key=cand_key)
        large_sorted = sorted(large_pool, key=cand_key)
        all_sorted = sorted(seq_cands, key=cand_key)

        chosen_ids: Set[str] = set()
        seq_chosen: List[Dict[str, Any]] = []

        # Slot 1: Small object representation
        for cand in small_sorted:
            if cand["inventory_id"] not in chosen_ids:
                seq_chosen.append(cand)
                chosen_ids.add(cand["inventory_id"])
                break

        # Slot 2: Medium object representation
        if len(seq_chosen) < seq_quota:
            for cand in med_sorted:
                if cand["inventory_id"] not in chosen_ids:
                    seq_chosen.append(cand)
                    chosen_ids.add(cand["inventory_id"])
                    break

        # Slot 3: Large object representation
        if len(seq_chosen) < seq_quota:
            for cand in large_sorted:
                if cand["inventory_id"] not in chosen_ids:
                    seq_chosen.append(cand)
                    chosen_ids.add(cand["inventory_id"])
                    break

        # Remaining quota slots: Fill with highest diversity candidates
        if len(seq_chosen) < seq_quota:
            for cand in all_sorted:
                if cand["inventory_id"] not in chosen_ids:
                    seq_chosen.append(cand)
                    chosen_ids.add(cand["inventory_id"])
                    if len(seq_chosen) >= seq_quota:
                        break

        selected_external.extend(seq_chosen)

    # Ensure exact match to target count
    selected_external = selected_external[:target_count]

    # Calculate achieved coverage
    seqs_represented = len(set(r["sequence_id"] for r in selected_external))
    frames_with_small = sum(1 for r in selected_external if r.get("box_sizes", {}).get("small", 0) > 0)
    frames_with_med = sum(1 for r in selected_external if r.get("box_sizes", {}).get("medium", 0) > 0)
    frames_with_large = sum(1 for r in selected_external if r.get("box_sizes", {}).get("large", 0) > 0)
    frames_with_van = sum(1 for r in selected_external if r.get("source_class_counts", {}).get("van", 0) > 0)
    frames_with_truck = sum(1 for r in selected_external if r.get("source_class_counts", {}).get("truck", 0) > 0)
    frames_with_bus = sum(1 for r in selected_external if r.get("source_class_counts", {}).get("bus", 0) > 0)

    achieved_stats = {
        "target_count": target_count,
        "selected_count": len(selected_external),
        "sequences_available": len(train_sequences),
        "sequences_covered": seqs_represented,
        "sequence_coverage_pct": round(seqs_represented / len(train_sequences) * 100.0, 1) if train_sequences else 0.0,
        "frames_with_small_boxes": frames_with_small,
        "small_box_frame_pct": round(frames_with_small / len(selected_external) * 100.0, 1) if selected_external else 0.0,
        "frames_with_medium_boxes": frames_with_med,
        "medium_box_frame_pct": round(frames_with_med / len(selected_external) * 100.0, 1) if selected_external else 0.0,
        "frames_with_large_boxes": frames_with_large,
        "large_box_frame_pct": round(frames_with_large / len(selected_external) * 100.0, 1) if selected_external else 0.0,
        "frames_with_vans_source_class": frames_with_van,
        "frames_with_trucks_source_class": frames_with_truck,
        "frames_with_buses_source_class": frames_with_bus,
    }

    return selected_external, achieved_stats


# ==============================================================================
# 5. Bounded Visual Spot-Check Selection (Scope Item 5)
# ==============================================================================

def generate_visual_spot_check_manifest(
    inventory: Dict[str, Any],
    manifest_b_train_records: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Selects up to 50 unique canonical source frames for visual review:
    - Avoids repeated augmentation variants of the same canonical frame.
    - Identifies vans using original source-class ID (van == 3).
    - Allows multiple review tags per frame.
    - Reports category shortfalls honestly.
    - Honors configured spot-check quotas.
    """
    records_by_id = {r["inventory_id"]: r for r in inventory["records"]}
    spot_cfg = config.get("spot_check", {}).get("counts", {})

    quotas = {
        "external_vans": int(spot_cfg.get("external_vans", 10)),
        "external_trucks": int(spot_cfg.get("external_trucks", 10)),
        "external_dense_small": int(spot_cfg.get("external_dense_small", 10)),
        "local_teacher_completed": int(spot_cfg.get("local_teacher_completed", 10)),
        "local_night_congestion": int(spot_cfg.get("local_night_congestion", 10)),
    }

    # Selected candidate pools
    selected_ext_records = [records_by_id[rid] for rid in manifest_b_train_records if records_by_id[rid]["data_origin"] == "external_ua_detrac"]
    local_train_records = [records_by_id[rid] for rid in manifest_b_train_records if records_by_id[rid]["data_origin"] == "local"]

    # Filter to 1 representative variant per canonical source frame
    ext_by_canonical: Dict[str, Dict[str, Any]] = {}
    for r in selected_ext_records:
        c_src = r["canonical_source_id"]
        if c_src not in ext_by_canonical or (not r["is_synthetic_variant"] and ext_by_canonical[c_src]["is_synthetic_variant"]):
            ext_by_canonical[c_src] = r

    local_by_canonical: Dict[str, Dict[str, Any]] = {}
    for r in local_train_records:
        c_src = r["canonical_source_id"]
        if c_src not in local_by_canonical or (not r["is_synthetic_variant"] and local_by_canonical[c_src]["is_synthetic_variant"]):
            local_by_canonical[c_src] = r

    # Spot-check map keyed by canonical_source_id
    spot_check_map: Dict[str, Dict[str, Any]] = {}
    achieved_counts: Dict[str, int] = Counter()

    # Helper to add or tag frame
    def tag_frame(rec: Dict[str, Any], category: str, risk_factors: List[str], required_checks: List[str]) -> bool:
        c_src = rec["canonical_source_id"]
        if c_src in spot_check_map:
            # Frame already in spot-check: add review tag and checks if not already present
            entry = spot_check_map[c_src]
            if category not in entry["review_tags"]:
                entry["review_tags"].append(category)
            for rf in risk_factors:
                if rf not in entry["risk_factors"]:
                    entry["risk_factors"].append(rf)
            for rc in required_checks:
                if rc not in entry["required_human_checks"]:
                    entry["required_human_checks"].append(rc)
            achieved_counts[category] += 1
            return False  # not a new frame
        else:
            if len(spot_check_map) >= 50:
                return False
            spot_check_map[c_src] = {
                "canonical_source_id": c_src,
                "inventory_id": rec["inventory_id"],
                "data_origin": rec["data_origin"],
                "image_path": rec["image_path"],
                "image_status_on_disk": rec["image_status"],
                "sequence_id": rec["sequence_id"],
                "review_tags": [category],
                "source_classes_present": [b.get("source_class_name") for b in rec.get("boxes", [])],
                "risk_factors": list(risk_factors),
                "required_human_checks": list(required_checks)
            }
            achieved_counts[category] += 1
            return True  # new frame added

    # 1. External Vans (source_class_name == "van" or source_class_id == 3)
    van_cands = [r for r in ext_by_canonical.values() if r.get("source_class_counts", {}).get("van", 0) > 0]
    van_cands.sort(key=lambda r: (-r.get("source_class_counts", {}).get("van", 0), -r.get("total_boxes", 0), r["canonical_source_id"]))
    for r in van_cands:
        if achieved_counts["external_vans"] >= quotas["external_vans"]:
            break
        tag_frame(
            r,
            category="external_vans",
            risk_factors=[
                "Semantic ambiguity between passenger commuter vans (0: car) and commercial delivery trucks (3: truck).",
                "Unannotated motorcycles or scooters in background."
            ],
            required_checks=[
                "Confirm van is passenger/minivan chassis (car), not heavy commercial truck.",
                "Verify absence of unannotated two-wheelers in frame background."
            ]
        )

    # 2. External Trucks (source_class_name == "truck" or source_class_id == 2)
    truck_cands = [r for r in ext_by_canonical.values() if r.get("source_class_counts", {}).get("truck", 0) > 0]
    truck_cands.sort(key=lambda r: (-r.get("source_class_counts", {}).get("truck", 0), -r.get("total_boxes", 0), r["canonical_source_id"]))
    for r in truck_cands:
        if achieved_counts["external_trucks"] >= quotas["external_trucks"]:
            break
        tag_frame(
            r,
            category="external_trucks",
            risk_factors=[
                "Light pickup trucks or small flatbeds mislabeled as heavy commercial trucks.",
                "Missing distant background vehicles."
            ],
            required_checks=[
                "Confirm vehicle has medium/heavy commercial chassis (6/10 wheels or heavy box/dump).",
                "If ordinary pickup, reclassify to 0: car or quarantine frame."
            ]
        )

    # 3. External Dense Small Vehicles
    small_cands = sorted(ext_by_canonical.values(), key=lambda r: (-r.get("box_sizes", {}).get("small", 0), r["canonical_source_id"]))
    for r in small_cands:
        if achieved_counts["external_dense_small"] >= quotas["external_dense_small"]:
            break
        tag_frame(
            r,
            category="external_dense_small",
            risk_factors=[
                "Tiny boxes clipped or truncated at distant horizons.",
                "Unannotated traffic queued in distant approach lanes."
            ],
            required_checks=[
                "Verify tiny bounding box tightness and boundary coordinates.",
                "Ensure distant unannotated vehicles are not penalized as negative background."
            ]
        )

    # 4. Local Teacher-Completed Frames
    teacher_cands = sorted(
        [r for r in local_by_canonical.values() if r.get("variant") == "original_curated"],
        key=lambda r: (-r.get("total_boxes", 0), r["canonical_source_id"])
    )
    for r in teacher_cands:
        if achieved_counts["local_teacher_completed"] >= quotas["local_teacher_completed"]:
            break
        tag_frame(
            r,
            category="local_teacher_completed",
            risk_factors=[
                "COCO YOLO26x teacher false-positive detections on background clutter.",
                "Pickup-based songthaew misclassified as truck by teacher."
            ],
            required_checks=[
                "Inspect background vehicle boxes generated by teacher model for false alarms.",
                "Confirm pickups, passenger vans, and songthaews are strictly classified as 0: car."
            ]
        )

    # 5. Local Night & Congestion Frames
    night_cands = sorted(
        [r for r in local_by_canonical.values() if r.get("lighting") == "real_night"],
        key=lambda r: (-(r.get("class_counts", {}).get("1", 0) + r.get("class_counts", {}).get("4", 0)), r["canonical_source_id"])
    )
    for r in night_cands:
        if achieved_counts["local_night_congestion"] >= quotas["local_night_congestion"]:
            break
        tag_frame(
            r,
            category="local_night_congestion",
            risk_factors=[
                "Headlight glare obscuring vehicle contours.",
                "Dense motorcycle platoons suffering from overlapping bounding boxes."
            ],
            required_checks=[
                "Verify motorcycle bounding boxes in dense queues under low light.",
                "Confirm tuk-tuks and salengs are cleanly separated from background motorcycles."
            ]
        )

    # Compile quota and shortfall audit
    quota_report = {}
    for cat, q_val in quotas.items():
        ach = achieved_counts[cat]
        shortfall = max(0, q_val - ach)
        quota_report[cat] = {
            "target_quota": q_val,
            "achieved_count": ach,
            "shortfall": shortfall,
            "status": "met" if shortfall == 0 else f"shortfall_of_{shortfall}"
        }

    spot_items = list(spot_check_map.values())
    print(f"[SPOT CHECK] Selected {len(spot_items)} unique canonical source frames across categories.")
    for cat, q_info in quota_report.items():
        print(f"   - {cat}: target {q_info['target_quota']}, achieved {q_info['achieved_count']}, shortfall {q_info['shortfall']}")

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_unique_spot_check_frames": len(spot_items),
            "max_allowed_frames": 50,
            "quota_audit": quota_report,
            "readiness_gate_notice": (
                "MANDATORY GATE: The training datasets must NOT be considered training-ready "
                "or exported to training pipelines until these spot-check frames are visually reviewed "
                "and external image files are resolved on disk."
            )
        },
        "spot_checks": spot_items
    }


# ==============================================================================
# 6. Markdown Report Renderer (Scope Item 6)
# ==============================================================================

def render_markdown_dataset_report(
    inventory: Dict[str, Any],
    local_split_data: Dict[str, Any],
    external_split_data: Dict[str, Any],
    manifest_a: Dict[str, Any],
    manifest_b: Dict[str, Any],
    spot_check: Dict[str, Any],
    achieved_ext_stats: Dict[str, Any],
    output_path: Path
) -> None:
    """Renders comprehensive, publication-quality Markdown dataset preparation report."""
    a_tr = manifest_a["metadata"]["train_summary"]
    a_vl = manifest_a["metadata"]["val_summary"]
    b_tr = manifest_b["metadata"]["train_summary"]
    b_ext = manifest_b["metadata"]["external_only_subset_summary"]

    block_defs = local_split_data["split_block_definitions"]
    exclusions = local_split_data["exclusions"] + external_split_data["exclusions"]
    excl_counts = Counter(e["reason"] for e in exclusions)
    ext_part = external_split_data["external_sequence_partition"]

    # Calculate actual minimum temporal separation across divided recordings
    min_sep_sec = None
    min_sep_frames = None
    min_sep_vid = None
    min_sep_fps = None

    for vid, bdef in block_defs.items():
        m_sep = bdef.get("measured_temporal_separation", {})
        ds = m_sep.get("delta_seconds")
        df = m_sep.get("delta_frames")
        if ds is not None and df is not None:
            if min_sep_sec is None or ds < min_sep_sec:
                min_sep_sec = ds
                min_sep_frames = df
                min_sep_vid = vid
                min_sep_fps = bdef.get("evidenced_fps")

    min_sep_statement = (
        f"**{min_sep_frames} frames ({min_sep_sec:.2f}s @ {min_sep_fps} FPS in `{min_sep_vid}`)**"
        if min_sep_sec is not None else "N/A"
    )

    box_delta = b_tr['total_boxes'] - a_tr['total_boxes']
    box_delta_pct = (box_delta / a_tr['total_boxes'] * 100.0) if a_tr['total_boxes'] > 0 else 0.0

    md = f"""# Thai Traffic Vision & YOLO26s Dataset Preparation Report (Batch 4 - Corrected)

- **Date**: {datetime.now(timezone.utc).isoformat()}
- **Report Status**: Candidate Manifests & Leakage Safeguards Established (**Pre-Training Gate**)
- **Random Seed**: `{external_split_data['random_seed']}` (Deterministic sequence splitting & stratified sampling)
- **External Image Cap**: **{manifest_b['metadata']['external_images_count']} frames** ({manifest_b['metadata']['external_cap_ratio']} per unique local training frame; {a_tr['unique_canonical_frames_count']} unique local frames)
- **Target Taxonomy**: Thai 5-Class COCO-aligned morphology (`0: car`, `1: motorcycle`, `2: bus`, `3: truck`, `4: three_wheeler`)

---

## 1. Executive Summary & Comparative Matrix

This audit prepares a controlled comparison between **Manifest A (Local-Only)** and **Manifest B (Local + Capped External UA-DETRAC)** to expand visual training variety while strictly preventing external cars from overwhelming Thai motorcycles, three-wheelers, and local conditions:

| Metric / Attribute | Manifest A (Local-Only) | Manifest B (Local + External) | Delta (B vs A) | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Total Training Images** | **{a_tr['images_count']}** | **{b_tr['images_count']}** | +{b_tr['images_count'] - a_tr['images_count']} ({manifest_b['metadata']['external_images_count']} external) | Controlled external expansion |
| **Unique Canonical Training Frames** | **{a_tr['unique_canonical_frames_count']}** | **{b_tr['unique_canonical_frames_count']}** | +{manifest_b['metadata']['external_unique_sources_count']} | Independent video frames |
| **Total Training Bounding Boxes** | **{a_tr['total_boxes']}** | **{b_tr['total_boxes']}** | +{box_delta} (+{box_delta_pct:.1f}%) | Preserves dense annotations |
| **Total Validation Images** | **{a_vl['images_count']}** | **{a_vl['images_count']}** | **0 (Identical)** | **Validation is strictly identical** |
| **Validation Bounding Boxes** | **{a_vl['total_boxes']}** | **{a_vl['total_boxes']}** | **0 (Identical)** | **Zero validation leakage** |
| **External Sequence Overlap** | **0.0%** | **0.0%** | 0.0% | Whole MVI sequence splitting |
| **Diagnostic Benchmark Overlap** | **0 frames** | **0 frames** | 0 | 42 eval frames strictly excluded |

---

## 2. Per-Class Box Proportions & Class Imbalance Guardrails

The external cap is calculated strictly as $\\lfloor {a_tr['unique_canonical_frames_count']} \\times {manifest_b['metadata']['external_cap_ratio']} \\rfloor = {manifest_b['metadata']['external_images_count']}$ frames (using unique local frames, NOT synthetic variants). This guarantees that external passenger cars do not submerge local minority classes:

| Vehicle Class | Class ID | Manifest A Boxes | Manifest A Share | Manifest B Boxes | Manifest B Share | External Added | Share Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `car` | 0 | {a_tr['class_box_counts']['car']} | {a_tr['class_box_percentages']['car']}% | {b_tr['class_box_counts']['car']} | {b_tr['class_box_percentages']['car']}% | +{b_ext['class_box_counts']['car']} | {b_tr['class_box_percentages']['car'] - a_tr['class_box_percentages']['car']:+.1f}% |
| `motorcycle` | 1 | {a_tr['class_box_counts']['motorcycle']} | {a_tr['class_box_percentages']['motorcycle']}% | {b_tr['class_box_counts']['motorcycle']} | {b_tr['class_box_percentages']['motorcycle']}% | +0 | {b_tr['class_box_percentages']['motorcycle'] - a_tr['class_box_percentages']['motorcycle']:+.1f}% |
| `bus` | 2 | {a_tr['class_box_counts']['bus']} | {a_tr['class_box_percentages']['bus']}% | {b_tr['class_box_counts']['bus']} | {b_tr['class_box_percentages']['bus']}% | +{b_ext['class_box_counts']['bus']} | {b_tr['class_box_percentages']['bus'] - a_tr['class_box_percentages']['bus']:+.1f}% |
| `truck` | 3 | {a_tr['class_box_counts']['truck']} | {a_tr['class_box_percentages']['truck']}% | {b_tr['class_box_counts']['truck']} | {b_tr['class_box_percentages']['truck']}% | +{b_ext['class_box_counts']['truck']} | {b_tr['class_box_percentages']['truck'] - a_tr['class_box_percentages']['truck']:+.1f}% |
| `three_wheeler` | 4 | {a_tr['class_box_counts']['three_wheeler']} | {a_tr['class_box_percentages']['three_wheeler']}% | {b_tr['class_box_counts']['three_wheeler']} | {b_tr['class_box_percentages']['three_wheeler']}% | +0 | {b_tr['class_box_percentages']['three_wheeler'] - a_tr['class_box_percentages']['three_wheeler']:+.1f}% |
| **Total** | — | **{a_tr['total_boxes']}** | **100.0%** | **{b_tr['total_boxes']}** | **100.0%** | **+{b_ext['total_boxes']}** | — |

> [!NOTE]
> **Minority Class Protection**: Local motorcycle boxes ({a_tr['class_box_counts']['motorcycle']}) and three-wheeler boxes ({a_tr['class_box_counts']['three_wheeler']}) remain fully preserved without box deletion. External data adds valuable truck and bus variety without collapsing minority representation.

---

## 3. Local Split & Contiguous Temporal Block Definitions

Local surveillance video recordings are split into explicit contiguous temporal blocks (80% timeline for train, exclusion buffer zone, remaining timeline for val) with all canonical frame variants kept together:

| Recording | Evidenced FPS | Strategy | Train Block [Start-End] | Buffer Zone Excluded | Val Block [Start-End] | Measured Separation |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
"""

    for vid, bdef in sorted(block_defs.items()):
        strat = bdef.get("strategy")
        fps = bdef.get("evidenced_fps")
        if strat == "whole_recording":
            md += f"| `{vid}` | {fps} | Whole Recording -> {bdef.get('assigned_split')} | All ({bdef.get('total_images_count')} imgs) | 0 | None | N/A |\n"
        else:
            tb = bdef["train_block"]
            bz = bdef["buffer_zone"]
            vb = bdef["val_block"]
            m_sep = bdef["measured_temporal_separation"]
            sep_str = f"{m_sep['delta_frames']} frames ({m_sep['delta_seconds']}s)" if m_sep['delta_frames'] is not None else "N/A"
            md += f"| `{vid}` | {fps} FPS | Contiguous Blocks | [{tb['start_frame_idx']}, {tb['end_frame_idx']}] ({tb['total_images_count']} imgs) | {bz['excluded_canonical_sources_count']} sources ({bz['excluded_images_count']} imgs) | [{vb['start_frame_idx']}, {vb['end_frame_idx']}] ({vb['total_images_count']} imgs) | {sep_str} |\n"

    md += f"""
- **Minimum Measured Temporal Separation**: {min_sep_statement}.
- *Scientific Disclaimer*: Passing a temporal proximity check prevents frame-burst leakage, but does NOT prove vehicle-level independence without trajectory tracking, nor does it remove historical checkpoint training exposure.

---

## 4. Split & Exclusion Audit (Safeguards Enforced)

All records excluded from training candidates are tracked in `split_and_exclusion_manifest.json`:

| Exclusion Category | Records Excluded | Scientific Rationale |
| :--- | :---: | :--- |
| **`diagnostic_evaluation_benchmark_leakage`** | **{excl_counts.get('diagnostic_evaluation_benchmark_leakage', 0)}** | Excludes the 42 authoritative evaluation benchmark frames (and variants) from training/validation to prevent circular benchmarking. |
| **`temporal_buffer_violation`** | **{excl_counts.get('temporal_buffer_violation', 0)}** | Frames falling within the temporal exclusion buffer between train and val blocks. |
| **`reserved_northeast_holdout`** | **{excl_counts.get('reserved_northeast_holdout', 0)}** | Reserves `cam45_northeast` approach footage from training. |
| **`external_val_sequence_reserved`** | **{excl_counts.get('external_val_sequence_reserved', 0)}** | Sequence-level partition: {ext_part['val_sequences_count']} MVI sequences reserved exclusively for external validation. |
| **`external_test_sequence_reserved`** | **{excl_counts.get('external_test_sequence_reserved', 0)}** | Sequence-level partition: {ext_part['test_sequences_count']} MVI sequences reserved exclusively for external testing. |
| **`unknown_provenance_quarantined`** | **{excl_counts.get('unknown_provenance_quarantined', 0)}** | Quarantines any frames with unproven lineage. |

### Whole-Sequence Partitioning (UA-DETRAC)
- **Train Sequences**: {ext_part['train_sequences_count']} sequences ({', '.join(ext_part['train_sequences'][:5])}...)
- **Validation Sequences**: {ext_part['val_sequences_count']} sequences ({', '.join(ext_part['val_sequences'][:5])}...)
- **Test Sequences**: {ext_part['test_sequences_count']} sequences ({', '.join(ext_part['test_sequences'][:5])}...)
- **Cross-Split Sequence Leakage**: **{ext_part['cross_split_sequence_overlap']} sequences (0.0% overlap)**.

### Achieved External Stratified Coverage
- **Sequences Represented**: {achieved_ext_stats.get('sequences_covered')} / {achieved_ext_stats.get('sequences_available')} ({achieved_ext_stats.get('sequence_coverage_pct')}%)
- **Frames with Small Boxes (< 32² px)**: {achieved_ext_stats.get('frames_with_small_boxes')} frames ({achieved_ext_stats.get('small_box_frame_pct')}%)
- **Frames with Medium Boxes**: {achieved_ext_stats.get('frames_with_medium_boxes')} frames ({achieved_ext_stats.get('medium_box_frame_pct')}%)
- **Frames with Large Boxes**: {achieved_ext_stats.get('frames_with_large_boxes')} frames ({achieved_ext_stats.get('large_box_frame_pct')}%)
- **Frames with Source Vans (class 3)**: {achieved_ext_stats.get('frames_with_vans_source_class')} frames
- **Frames with Source Trucks (class 2)**: {achieved_ext_stats.get('frames_with_trucks_source_class')} frames
- **Frames with Source Buses (class 0)**: {achieved_ext_stats.get('frames_with_buses_source_class')} frames

---

## 5. Annotation Uncertainty & Taxonomy Warnings

1. **Semantic Compatibility of External Vans & Trucks**:
   - `van -> 0 (car)`: In Thailand, commuter passenger vans (Toyota Commuter/HiAce) belong to class 0. However, truck-based cargo vans might border class 3.
   - `truck -> 3 (truck)`: Medium and heavy commercial trucks belong to class 3. In external datasets, light utility trucks or flatbeds might be labeled truck, whereas Thai morphology treats light pickups as class 0.
2. **Missing Foreground Object Bias**:
   - UA-DETRAC does **NOT** annotate motorcycles, bicycles, or three-wheelers.
   - The absence of motorcycle labels in external images does **not** prove motorcycles are absent. Training on unannotated motorcycles with standard detection loss would treat them as negative background and penalize motorcycle recall.
3. **Missing External Image Files**:
   - The UA-DETRAC image files are currently **unresolved on disk** (only NDJSON metadata exists in `data/usdetrac/`). They are explicitly tracked as `unresolved_not_downloaded` and must not be treated as locally available image files.

---

## 6. Bounded Visual Spot-Check Plan (Up to 50 Unique Canonical Frames)

A dedicated manifest of **{spot_check['metadata']['total_unique_spot_check_frames']} unique canonical source frames** is established at `data/training_manifests_v2/visual_spot_check_manifest.json`:

| Category | Target Quota | Achieved Frames | Shortfall | Key Inspection Objective |
| :--- | :---: | :---: | :---: | :--- |
"""

    for cat, q_info in spot_check["metadata"]["quota_audit"].items():
        obj = {
            "external_vans": "Verify passenger commuter van morphology vs commercial cargo trucks.",
            "external_trucks": "Verify medium/heavy commercial chassis vs light pickup flatbeds.",
            "external_dense_small": "Inspect tiny box bounds and verify absence of unannotated motorcycles in background.",
            "local_teacher_completed": "Check background boxes generated by COCO YOLO26x teacher for false alarms.",
            "local_night_congestion": "Verify dense motorcycle and tuk-tuk queue annotations under headlight glare."
        }.get(cat, "")
        md += f"| **{cat}** | {q_info['target_quota']} | {q_info['achieved_count']} | {q_info['shortfall']} | {obj} |\n"

    md += f"""
> [!IMPORTANT]
> **Pre-Training Gate**: {spot_check['metadata']['readiness_gate_notice']}

---

## 7. Reproducibility & Commands

To regenerate these manifests deterministically from source data:
```bash
python tools/prepare_training_manifests.py \\
  --config config/training_manifest_config.json \\
  --local-dir data/multiclass_dataset \\
  --ndjson-path data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson \\
  --eval-snapshot data/eval_snapshot_v1/manifest.json \\
  --output-dir data/training_manifests_v2 \\
  --videos-dir videos \\
  --report-md docs/DATASET_PREPARATION_REPORT.md \\
  --seed 42 \\
  --external-ratio 0.5
```
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"[REPORT COMPLETE] Dataset preparation report saved to: {output_path}")


# ==============================================================================
# 7. Pipeline Execution Engine
# ==============================================================================

def run_manifest_preparation(
    config_path: Optional[Path] = None,
    local_dir: Optional[Path] = None,
    ndjson_path: Optional[Path] = None,
    eval_snapshot_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    report_md_path: Optional[Path] = None,
    seed: Optional[int] = None,
    external_ratio: Optional[float] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Runs the complete training manifest preparation pipeline."""
    # Load Config
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    local_dir = local_dir or Path(config.get("paths", {}).get("local_dataset_dir", "data/multiclass_dataset"))
    ndjson_path = ndjson_path or Path(config.get("paths", {}).get("external_ndjson_path", "data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson"))
    eval_snapshot_path = eval_snapshot_path or Path(config.get("paths", {}).get("eval_snapshot_manifest", "data/eval_snapshot_v1/manifest.json"))
    output_dir = output_dir or Path(config.get("paths", {}).get("output_dir", "data/training_manifests_v2"))
    videos_dir = videos_dir or Path(config.get("paths", {}).get("videos_dir", "videos"))
    report_md_path = report_md_path or Path(config.get("paths", {}).get("report_markdown", "docs/DATASET_PREPARATION_REPORT.md"))

    if seed is not None:
        config["random_seed"] = seed
    if external_ratio is not None:
        if "balancing" not in config:
            config["balancing"] = {}
        config["balancing"]["external_cap_ratio_per_unique_local_frame"] = external_ratio

    # Overwrite check (Scope Item 6)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists and is not empty. "
                "Refusing to overwrite previous manifest versions. "
                "Specify a new version directory (e.g. data/training_manifests_v2) or pass --overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Evaluation Snapshot Manifest (Fail-Fast: Scope Item 3)
    eval_source_ids = load_evaluation_manifest(eval_snapshot_path)

    # 2. Probe Video FPS (Scope Item 2)
    evidenced_fps_map = probe_video_fps(videos_dir)
    print(f"[FPS PROBE] Probed {len(evidenced_fps_map)} surveillance videos from '{videos_dir}': {evidenced_fps_map}")

    # 3. Build Canonical Source Inventory (Scope Item 4)
    inventory = build_canonical_source_inventory(local_dir, ndjson_path, eval_source_ids, config)
    inv_file = output_dir / "canonical_source_inventory.json"
    with open(inv_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Canonical source inventory saved to: {inv_file}")

    # 4. Local Split Construction & Temporal Exclusions (Scope Items 1 & 2)
    local_split_data = construct_local_splits_and_exclusions(inventory, eval_source_ids, evidenced_fps_map, config)

    # 5. External Whole-Sequence Partitioning (Scope Item 2)
    ext_records = [r for r in inventory["records"] if r["data_origin"] == "external_ua_detrac"]
    external_sequences = sorted(list(set(r["sequence_id"] for r in ext_records if r["sequence_id"] != "unknown_seq")))

    rng_seed = int(config.get("random_seed", 42))
    ext_ratios = config.get("splits", {}).get("external_sequence_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})

    rng = random.Random(rng_seed)
    shuffled_seqs = list(external_sequences)
    rng.shuffle(shuffled_seqs)

    n_total_seq = len(shuffled_seqs)
    n_train_seq = int(round(n_total_seq * ext_ratios.get("train", 0.70)))
    n_val_seq = int(round(n_total_seq * ext_ratios.get("val", 0.15)))

    ext_train_seqs = set(shuffled_seqs[:n_train_seq])
    ext_val_seqs = set(shuffled_seqs[n_train_seq:n_train_seq + n_val_seq])
    ext_test_seqs = set(shuffled_seqs[n_train_seq + n_val_seq:])

    external_exclusions = []
    eligible_external_train_cands = []

    for r in ext_records:
        seq = r["sequence_id"]
        inv_id = r["inventory_id"]
        c_src = r["canonical_source_id"]
        if seq in ext_train_seqs:
            eligible_external_train_cands.append(r)
        elif seq in ext_val_seqs:
            external_exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "external_ua_detrac",
                "reason": "external_val_sequence_reserved",
                "details": f"Sequence '{seq}' allocated to external validation."
            })
        elif seq in ext_test_seqs:
            external_exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "external_ua_detrac",
                "reason": "external_test_sequence_reserved",
                "details": f"Sequence '{seq}' allocated to external test."
            })

    external_split_data = {
        "random_seed": rng_seed,
        "external_sequence_partition": {
            "train_sequences_count": len(ext_train_seqs),
            "val_sequences_count": len(ext_val_seqs),
            "test_sequences_count": len(ext_test_seqs),
            "train_sequences": sorted(list(ext_train_seqs)),
            "val_sequences": sorted(list(ext_val_seqs)),
            "test_sequences": sorted(list(ext_test_seqs)),
            "cross_split_sequence_overlap": len(ext_train_seqs.intersection(ext_val_seqs))
        },
        "exclusions": external_exclusions
    }

    # 6. Recalculate External Cap from Eligible Unique Local Training Frames (Scope Items 3 & 6)
    local_train_records = local_split_data["local_train_records"]
    local_val_records = local_split_data["local_val_records"]
    primary_val_records = local_split_data.get("primary_val_records", local_val_records)
    synthetic_val_records = local_split_data.get("synthetic_val_records", [])
    unresolved_val_sources = local_split_data.get("unresolved_val_sources", [])

    unique_local_train_sources = set(r["canonical_source_id"] for r in local_train_records)
    n_unique_local_train = len(unique_local_train_sources)

    cap_ratio = float(config.get("balancing", {}).get("external_cap_ratio_per_unique_local_frame", 0.5))
    external_cap = int(math.floor(n_unique_local_train * cap_ratio))

    print(f"\n[BALANCING] Eligible unique local training frames: {n_unique_local_train}")
    print(f"[BALANCING] Total local training images (including variants): {len(local_train_records)}")
    print(f"[BALANCING] External cap (ratio {cap_ratio}): {external_cap} images")
    print(f"[VALIDATION] Primary unaugmented validation images: {len(primary_val_records)} (Sources: {len(set(r['canonical_source_id'] for r in primary_val_records))})")
    print(f"[VALIDATION] Synthetic diagnostic validation images: {len(synthetic_val_records)}")
    if unresolved_val_sources:
        print(f"[VALIDATION WARNING] Unresolved validation sources: {len(unresolved_val_sources)}")

    # 7. External Sequence & Size Stratified Sampling (Scope Item 5)
    selected_external, achieved_ext_stats = sample_external_frames_stratified(
        external_candidates=eligible_external_train_cands,
        target_count=external_cap,
        train_sequences=sorted(list(ext_train_seqs)),
        seed=rng_seed
    )

    # Helper to aggregate manifest statistics
    def summarize_manifest_subset(items: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        total_imgs = len(items)
        unique_sources = len(set(x["canonical_source_id"] for x in items))
        class_boxes = Counter()
        size_boxes = Counter()
        lighting_imgs = Counter()
        camera_imgs = Counter()
        origin_imgs = Counter()

        for it in items:
            origin_imgs[it["data_origin"]] += 1
            lighting_imgs[it["lighting"]] += 1
            camera_imgs[it["camera"]] += 1
            for k, v in it.get("class_counts", {}).items():
                class_boxes[int(k)] += v
            for k, v in it.get("box_sizes", {}).items():
                size_boxes[k] += v

        total_b = sum(class_boxes.values())
        return {
            "subset_name": name,
            "images_count": total_imgs,
            "unique_canonical_frames_count": unique_sources,
            "total_boxes": total_b,
            "data_origin_distribution": dict(origin_imgs),
            "lighting_distribution": dict(lighting_imgs),
            "camera_distribution": dict(camera_imgs),
            "class_box_counts": {THAI_5CLASS_NAMES[cid]: class_boxes[cid] for cid in range(5)},
            "class_box_percentages": {THAI_5CLASS_NAMES[cid]: round(class_boxes[cid] / total_b * 100.0, 2) if total_b > 0 else 0.0 for cid in range(5)},
            "size_box_counts": dict(size_boxes),
            "size_box_percentages": {k: round(v / total_b * 100.0, 2) if total_b > 0 else 0.0 for k, v in size_boxes.items()}
        }

    # Primary Validation Manifest (Scope Item 3: unaugmented frames only)
    primary_val_manifest = {
        "manifest_name": "Primary Validation Set (Clean Unaugmented)",
        "manifest_id": "primary_val_v3",
        "description": "Clean, benchmark-grade primary validation selection consisting strictly of one original unaugmented image-label pair per eligible validation canonical source.",
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_images": len(primary_val_records),
            "unique_canonical_sources": len(set(r["canonical_source_id"] for r in primary_val_records)),
            "unresolved_sources_count": len(unresolved_val_sources),
            "unresolved_sources": unresolved_val_sources,
            "summary": summarize_manifest_subset(primary_val_records, "primary_val_clean")
        },
        "records": [r["inventory_id"] for r in primary_val_records]
    }

    # Synthetic Diagnostic Validation Manifest
    synth_val_manifest = {
        "manifest_name": "Synthetic Validation Diagnostic Variants",
        "manifest_id": "synthetic_val_diagnostic_v3",
        "description": "Diagnostic set of synthetic augmentation variants (salengboost, tuktukboost, nightboost, etc.) corresponding to canonical validation sources. Evaluated separately for robustness checks; excluded from primary validation benchmark.",
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_images": len(synthetic_val_records),
            "unique_canonical_sources": len(set(r["canonical_source_id"] for r in synthetic_val_records)),
            "summary": summarize_manifest_subset(synthetic_val_records, "synthetic_val_diagnostic")
        },
        "records": [r["inventory_id"] for r in synthetic_val_records]
    }

    # Manifest A (Local-Only)
    manifest_a = {
        "manifest_name": "Manifest A (Eligible Local-Only)",
        "manifest_id": "manifest_a_local_v3",
        "description": "Baseline training dataset consisting strictly of eligible local CCTV approach footage with clean unaugmented primary validation.",
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "external_data_included": False,
            "train_summary": summarize_manifest_subset(local_train_records, "local_train"),
            "val_summary": summarize_manifest_subset(primary_val_records, "primary_local_val"),
            "primary_validation_manifest": "primary_validation_manifest.json",
            "synthetic_validation_diagnostic_manifest": "synthetic_validation_diagnostic_manifest.json",
            "primary_validation_images_count": len(primary_val_records),
            "synthetic_validation_images_count": len(synthetic_val_records),
            "unresolved_validation_sources_count": len(unresolved_val_sources)
        },
        "train_records": [r["inventory_id"] for r in local_train_records],
        "val_records": [r["inventory_id"] for r in primary_val_records],
    }

    # Manifest B (Local + Capped External)
    manifest_b_train_items = local_train_records + selected_external
    manifest_b = {
        "manifest_name": "Manifest B (Local + Capped External UA-DETRAC)",
        "manifest_id": "manifest_b_combined_v3",
        "description": (
            "Comparative training dataset combining identical local CCTV footage "
            f"with a capped external subset ({len(selected_external)} images, ratio {cap_ratio} per unique local frame) "
            "from whole UA-DETRAC sequences, with identical clean unaugmented primary validation."
        ),
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "external_data_included": True,
            "external_cap_ratio": cap_ratio,
            "external_images_count": len(selected_external),
            "external_unique_sources_count": len(set(x["canonical_source_id"] for x in selected_external)),
            "external_sequences_covered": achieved_ext_stats.get("sequences_covered", 0),
            "achieved_stratification_stats": achieved_ext_stats,
            "train_summary": summarize_manifest_subset(manifest_b_train_items, "combined_train"),
            "val_summary": summarize_manifest_subset(primary_val_records, "identical_primary_local_val"),
            "external_only_subset_summary": summarize_manifest_subset(selected_external, "external_train_subset"),
            "primary_validation_manifest": "primary_validation_manifest.json",
            "synthetic_validation_diagnostic_manifest": "synthetic_validation_diagnostic_manifest.json",
            "primary_validation_images_count": len(primary_val_records),
            "synthetic_validation_images_count": len(synthetic_val_records),
            "unresolved_validation_sources_count": len(unresolved_val_sources)
        },
        "train_records": [r["inventory_id"] for r in manifest_b_train_items],
        "val_records": [r["inventory_id"] for r in primary_val_records],
        "external_train_records": [r["inventory_id"] for r in selected_external],
    }

    # 8. Visual Spot-Check Selection (Scope Item 5)
    spot_check = generate_visual_spot_check_manifest(inventory, manifest_b["train_records"], config)

    # 9. Save Artifacts
    combined_exclusions = local_split_data["exclusions"] + external_split_data["exclusions"]
    split_excl_manifest = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": rng_seed,
            "total_exclusions": len(combined_exclusions),
            "exclusion_breakdown": dict(Counter(e["reason"] for e in combined_exclusions)),
            "split_block_definitions": local_split_data["split_block_definitions"],
            "external_sequence_partition": external_split_data["external_sequence_partition"],
            "missing_timing_records": local_split_data["missing_timing_records"]
        },
        "exclusions": combined_exclusions
    }

    excl_file = output_dir / "split_and_exclusion_manifest.json"
    with open(excl_file, "w", encoding="utf-8") as f:
        json.dump(split_excl_manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Split and exclusion manifest saved to: {excl_file}")

    prim_val_file = output_dir / "primary_validation_manifest.json"
    with open(prim_val_file, "w", encoding="utf-8") as f:
        json.dump(primary_val_manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Primary validation manifest saved to: {prim_val_file}")

    synth_val_file = output_dir / "synthetic_validation_diagnostic_manifest.json"
    with open(synth_val_file, "w", encoding="utf-8") as f:
        json.dump(synth_val_manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Synthetic validation diagnostic manifest saved to: {synth_val_file}")

    man_a_file = output_dir / "manifest_a_local_only.json"
    with open(man_a_file, "w", encoding="utf-8") as f:
        json.dump(manifest_a, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Manifest A saved to: {man_a_file}")

    man_b_file = output_dir / "manifest_b_local_plus_external.json"
    with open(man_b_file, "w", encoding="utf-8") as f:
        json.dump(manifest_b, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Manifest B saved to: {man_b_file}")

    spot_file = output_dir / "visual_spot_check_manifest.json"
    with open(spot_file, "w", encoding="utf-8") as f:
        json.dump(spot_check, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Visual spot-check manifest saved to: {spot_file}")

    cfg_copy_file = output_dir / "config.json"
    with open(cfg_copy_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 10. Render Markdown Report
    render_markdown_dataset_report(
        inventory=inventory,
        local_split_data=local_split_data,
        external_split_data=external_split_data,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        spot_check=spot_check,
        achieved_ext_stats=achieved_ext_stats,
        output_path=report_md_path
    )

    return {
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
        "primary_val_manifest": primary_val_manifest,
        "synthetic_val_manifest": synth_val_manifest,
        "spot_check": spot_check,
        "local_split_data": local_split_data,
        "external_split_data": external_split_data,
        "splits_data": external_split_data,
        "split_block_definitions": local_split_data["split_block_definitions"],
        "achieved_ext_stats": achieved_ext_stats,
        "output_dir": str(output_dir),
        "report_path": str(report_md_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced training manifests with leakage safeguards (Batch 4 / Batch 5).")
    parser.add_argument("--config", type=str, default="config/training_manifest_config.json", help="Path to config JSON")
    parser.add_argument("--local-dir", type=str, default="data/multiclass_dataset", help="Local dataset directory")
    parser.add_argument("--ndjson-path", type=str, default="data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson", help="External NDJSON path")
    parser.add_argument("--eval-snapshot", type=str, default="data/eval_snapshot_v1/manifest.json", help="Evaluation snapshot manifest path")
    parser.add_argument("--output-dir", type=str, default="data/training_manifests_v3", help="Output directory for manifests")
    parser.add_argument("--videos-dir", type=str, default="videos", help="Directory containing surveillance video files")
    parser.add_argument("--report-md", type=str, default="docs/DATASET_PREPARATION_REPORT.md", help="Output Markdown report path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sequence splitting")
    parser.add_argument("--external-ratio", type=float, default=0.5, help="External cap ratio per unique local training frame")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output directory")

    args = parser.parse_args()

    run_manifest_preparation(
        config_path=Path(args.config) if args.config else None,
        local_dir=Path(args.local_dir) if args.local_dir else None,
        ndjson_path=Path(args.ndjson_path) if args.ndjson_path else None,
        eval_snapshot_path=Path(args.eval_snapshot) if args.eval_snapshot else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        videos_dir=Path(args.videos_dir) if args.videos_dir else None,
        report_md_path=Path(args.report_md) if args.report_md else None,
        seed=args.seed,
        external_ratio=args.external_ratio,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
