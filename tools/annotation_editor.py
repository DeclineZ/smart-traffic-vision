#!/usr/bin/env python3
"""
tools/annotation_editor.py - Visual Annotation Editor for YOLO26s Review Packs (Batch 2B)

Provides a lightweight, zero-cloud local annotation editor for reviewing, drawing,
resizing, moving, deleting, and classifying bounding boxes with pixel-level precision.

Integrates directly with the authoritative structured store (annotations/annotations.json)
and reuses the existing validated synchronization and rollback mechanisms (sync_review_pack).

Workflow:
  1. Launch: python tools/annotation_editor.py --pack data/review_pack_v1
  2. Browse 42 frames with prev/next navigation, filters, and progress counts.
  3. Zoom and pan for distant/tiny vehicles (< 32^2 px letterboxed).
  4. Draw, move, resize (8 handles), and delete boxes with undo support.
  5. Edit detailed subtype taxonomy, ambiguity flags, and reviewer notes.
  6. Save as draft or explicitly mark fully reviewed frames verified.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import http.server
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import socketserver
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import webbrowser

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_review_pack import (
    THAI_5CLASS_NAMES,
    atomic_replace_file,
    compute_file_sha256,
    infer_provisional_subtype,
    sync_review_pack,
)

UI_HTML_PATH = Path(__file__).resolve().parent / "editor_ui.html"

# Agreed 5-class taxonomy presets for editor controls
TAXONOMY_SUBTYPE_PRESETS = {
    0: [
        "pickup",
        "pickup_based_songthaew",
        "passenger_van",
        "sedan",
        "hatchback",
        "suv_ppv",
        "taxi",
        "high_cage_pickup",
        "light_vehicle_provisional",
        "other_car"
    ],
    1: [
        "motorcycle_commuter",
        "sport_bike",
        "delivery_bike_with_box",
        "motorcycle_provisional",
        "other_motorcycle"
    ],
    2: [
        "bmta_city_bus",
        "thai_smile_bus",
        "tour_coach",
        "double_decker",
        "bus_provisional",
        "other_bus"
    ],
    3: [
        "medium_truck_6w",
        "heavy_truck_10w",
        "articulated_trailer_18w",
        "truck_based_songthaew",
        "construction_truck",
        "truck_provisional",
        "other_truck"
    ],
    4: [
        "tuktuk",
        "saleng",
        "three_wheeler_provisional",
        "other_three_wheeler"
    ]
}


class StaleSaveError(Exception):
    """Raised when an incoming save request has an outdated ETag (concurrent external edit)."""
    pass


# =============================================================================
# Mathematical Coordinate Conversions & Geometry Helpers
# =============================================================================

def screen_to_image_coords(
    screen_x: float,
    screen_y: float,
    pan_x: float,
    pan_y: float,
    zoom: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float]:
    """
    Converts screen/canvas viewport coordinates to underlying image pixel coordinates,
    accounting for current zoom scale and pan offset.
    Clamps coordinates strictly to [0, img_w] and [0, img_h].
    """
    if zoom <= 0:
        zoom = 1.0
    ix = (screen_x - pan_x) / zoom
    iy = (screen_y - pan_y) / zoom
    clamped_x = max(0.0, min(float(img_w), ix))
    clamped_y = max(0.0, min(float(img_h), iy))
    return clamped_x, clamped_y


def image_to_screen_coords(
    img_x: float,
    img_y: float,
    pan_x: float,
    pan_y: float,
    zoom: float
) -> Tuple[float, float]:
    """
    Converts image pixel coordinates to screen/canvas viewport coordinates.
    """
    sx = pan_x + img_x * zoom
    sy = pan_y + img_y * zoom
    return sx, sy


def image_box_to_normalized_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int
) -> List[float]:
    """
    Converts two corner image pixel coordinates (x1, y1, x2, y2) to
    standard normalized YOLO format [xc, yc, w, h] clamped to [0.0, 1.0].
    Rounded to 6 decimal places.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    min_x = max(0.0, min(float(img_w), min(x1, x2)))
    max_x = max(0.0, min(float(img_w), max(x1, x2)))
    min_y = max(0.0, min(float(img_h), min(y1, y2)))
    max_y = max(0.0, min(float(img_h), max(y1, y2)))

    xc = (min_x + max_x) / (2.0 * img_w)
    yc = (min_y + max_y) / (2.0 * img_h)
    bw = (max_x - min_x) / float(img_w)
    bh = (max_y - min_y) / float(img_h)

    # Clamp normalized values
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)]


def normalized_yolo_to_image_box(
    bbox_norm: List[float],
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts normalized YOLO format [xc, yc, w, h] to corner image pixels (x1, y1, x2, y2).
    """
    if len(bbox_norm) != 4:
        raise ValueError(f"bbox_norm must contain 4 elements, got: {bbox_norm}")
    xc, yc, bw, bh = bbox_norm
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return x1, y1, x2, y2


def generate_stable_instance_id(existing_boxes: List[Dict[str, Any]], frame_id: str) -> str:
    """
    Generates a new unique, sequential instance ID (frame_id_inst_XXX) that does not
    collide with any existing instance IDs, preserving stable IDs for existing objects.
    """
    pattern = re.compile(rf"^{re.escape(frame_id)}_inst_(\d+)$")
    max_idx = -1
    used_ids = set()

    for b in existing_boxes:
        iid = b.get("instance_id", "")
        if iid:
            used_ids.add(iid)
            m = pattern.match(iid)
            if m:
                val = int(m.group(1))
                if val > max_idx:
                    max_idx = val

    next_idx = max_idx + 1
    while True:
        candidate = f"{frame_id}_inst_{next_idx:03d}"
        if candidate not in used_ids:
            return candidate
        next_idx += 1


def validate_and_sanitize_boxes(
    boxes: List[Dict[str, Any]],
    frame_id: str
) -> List[Dict[str, Any]]:
    """
    Strictly validates and sanitizes incoming bounding box annotations:
    - Enforces valid class_id (0..4) and corresponding canonical class_name
    - Ensures valid normalized bounding box coordinates in [0.0, 1.0]
    - Generates stable unique instance_id for any newly created box
    - Preserves subtype metadata and ambiguity flags
    """
    sanitized: List[Dict[str, Any]] = []

    for b in boxes:
        if not isinstance(b, dict):
            continue

        cid = b.get("class_id")
        try:
            cid = int(cid)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid class_id in box for frame '{frame_id}': {cid}")

        if cid not in THAI_5CLASS_NAMES:
            raise ValueError(f"Unknown class_id {cid} for frame '{frame_id}'. Must be in 0..4.")

        bbox = b.get("bbox_norm")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Invalid bbox_norm in box for frame '{frame_id}': {bbox}")

        try:
            clean_bbox = [
                round(max(0.0, min(1.0, float(c))), 6) for c in bbox
            ]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Non-numeric bbox coordinates in frame '{frame_id}': {e}") from e

        # Ensure minimal box dimensions
        if clean_bbox[2] <= 0.0 or clean_bbox[3] <= 0.0:
            continue

        inst_id = b.get("instance_id")
        if not inst_id or not isinstance(inst_id, str) or not inst_id.strip():
            inst_id = generate_stable_instance_id(sanitized, frame_id)

        subtype = b.get("subtype")
        if not subtype or not isinstance(subtype, str) or not subtype.strip():
            subtype = infer_provisional_subtype(cid, frame_id, tuple(clean_bbox))

        is_ambiguous = bool(b.get("is_ambiguous", False))
        ambiguity_reason = str(b.get("ambiguity_reason", "") or "").strip()
        proposal_source = str(b.get("proposal_source", "") or "manual_annotation")

        box_item = {
            "instance_id": inst_id.strip(),
            "class_id": cid,
            "class_name": THAI_5CLASS_NAMES[cid],
            "subtype": subtype.strip(),
            "is_ambiguous": is_ambiguous,
            "ambiguity_reason": ambiguity_reason,
            "bbox_norm": clean_bbox,
            "proposal_source": proposal_source
        }
        if "original_source_class_id" in b and b["original_source_class_id"] is not None:
            box_item["original_source_class_id"] = b["original_source_class_id"]
        if "original_source_class_name" in b and b["original_source_class_name"] is not None:
            box_item["original_source_class_name"] = str(b["original_source_class_name"])

        # Preserve proposal review lifecycle metadata
        if "is_proposal" in b:
            box_item["is_proposal"] = bool(b["is_proposal"])
        if "proposal_status" in b and b["proposal_status"] is not None:
            box_item["proposal_status"] = str(b["proposal_status"])
        if "proposal_category" in b and b["proposal_category"] is not None:
            box_item["proposal_category"] = str(b["proposal_category"])
        if "confidence" in b and b["confidence"] is not None:
            try:
                box_item["confidence"] = round(float(b["confidence"]), 4)
            except (ValueError, TypeError):
                pass
        if "proposal_method" in b and b["proposal_method"] is not None:
            box_item["proposal_method"] = str(b["proposal_method"])
        if "tile_hits" in b and b["tile_hits"] is not None:
            try:
                box_item["tile_hits"] = int(b["tile_hits"])
            except (ValueError, TypeError):
                pass
        if "conflicting_instance_id" in b and b["conflicting_instance_id"] is not None:
            box_item["conflicting_instance_id"] = str(b["conflicting_instance_id"])
        if "human_class_name" in b and b["human_class_name"] is not None:
            box_item["human_class_name"] = str(b["human_class_name"])
        if "source_type" in b and b["source_type"] is not None:
            box_item["source_type"] = str(b["source_type"])

        sanitized.append(box_item)

    return sanitized


# =============================================================================
# Core Storage & Transactional Synchronization
# =============================================================================

def save_frame_annotation(
    pack_dir: Path,
    frame_id: str,
    base_etag: Optional[str],
    action: str,
    boxes_data: List[Dict[str, Any]],
    reviewer_notes: str = "",
    is_ambiguous: bool = False
) -> Dict[str, Any]:
    """
    Saves an updated frame annotation into annotations.json and synchronizes
    YOLO label files, preview images, and manifest through sync_review_pack.

    Transactional guarantees:
    - Verifies base_etag against current annotations.json sha256 to reject stale saves (409 Conflict).
    - Preserves backup of annotations.json before writing.
    - If sync_review_pack fails, rolls back annotations.json and raises an error.
    - Never automatically marks a frame verified on save; only explicit action == 'mark_verified' does.
    - Editing a verified annotation returns review_status to 'draft'.
    """
    annos_file = pack_dir / "annotations" / "annotations.json"
    if not annos_file.exists():
        raise FileNotFoundError(f"Annotations JSON file not found at {annos_file}")

    # 1. Stale-Save ETag Check
    curr_etag = compute_file_sha256(annos_file)
    if base_etag and base_etag != curr_etag:
        raise StaleSaveError(
            f"External modification detected: annotations.json was changed since this frame was loaded. "
            f"Expected ETag {base_etag[:10]}, current ETag {curr_etag[:10]}."
        )

    # 2. Parse current annotations.json
    try:
        with open(annos_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"Corrupt annotations.json at {annos_file}")
    except Exception as e:
        raise ValueError(f"Failed to read annotations.json: {e}") from e

    rec_idx = next((i for i, r in enumerate(records) if r.get("frame_id") == frame_id), None)
    if rec_idx is None:
        raise KeyError(f"Frame '{frame_id}' not found in annotations.json")

    orig_rec = records[rec_idx]

    # 3. Validate & sanitize box annotations
    clean_boxes = validate_and_sanitize_boxes(boxes_data, frame_id)

    # 4. State transitions
    if action == "mark_verified":
        new_review_status = "verified"
        if len(clean_boxes) == 0:
            new_annotation_state = "verified_empty_background"
            new_is_unannotated = False
        else:
            new_annotation_state = "annotated"
            new_is_unannotated = False
    elif action == "save_draft":
        new_review_status = "draft"
        if len(clean_boxes) == 0:
            # Empty frame saved as draft (leaves unannotated state, but unreviewed empty)
            new_annotation_state = "unreviewed_machine_empty"
            new_is_unannotated = False
        else:
            new_annotation_state = "annotated"
            new_is_unannotated = False
    elif action == "mark_rejected":
        new_review_status = "rejected"
        new_annotation_state = "rejected"
        new_is_unannotated = False
    elif action == "mark_uncertain":
        new_review_status = "uncertain"
        new_annotation_state = "uncertain"
        new_is_unannotated = False
    else:
        raise ValueError(f"Unknown save action '{action}'. Must be 'save_draft', 'mark_verified', 'mark_rejected', or 'mark_uncertain'.")

    up_rec = dict(orig_rec)
    up_rec["boxes"] = clean_boxes
    up_rec["review_status"] = new_review_status
    up_rec["annotation_state"] = new_annotation_state
    up_rec["is_unannotated"] = new_is_unannotated
    up_rec["reviewer_notes"] = reviewer_notes.strip()
    up_rec["is_ambiguous"] = bool(is_ambiguous) or (action == "mark_uncertain")
    up_rec["is_rejected"] = (action == "mark_rejected")

    updated_records = list(records)
    updated_records[rec_idx] = up_rec

    # 5. Pre-save backup for rollback on sync failure
    backup_file = annos_file.parent / f".tmp_backup_{frame_id}_{os.getpid()}_{int(time.time() * 1000)}.json"
    shutil.copy2(annos_file, backup_file)

    try:
        # Atomic write of updated annotations.json via same-directory temporary file
        temp_annos = annos_file.parent / f".tmp_write_{frame_id}_{os.getpid()}.json"
        with open(temp_annos, "w", encoding="utf-8") as f:
            json.dump(updated_records, f, indent=2)
        atomic_replace_file(temp_annos, annos_file)

        # 6. Reuse validated synchronization & rollback mechanisms
        # Passing frame_ids=[frame_id] updates labels, preview images, manifest, and review index
        sync_res = sync_review_pack(pack_dir=pack_dir, strategy="from_json", frame_ids=[frame_id])

    except Exception as exc:
        # Rollback annotations.json from backup!
        try:
            if backup_file.exists():
                atomic_replace_file(backup_file, annos_file)
        except Exception as rb_err:
            raise RuntimeError(f"Sync failed ({exc}) and annotations rollback also failed: {rb_err}") from exc
        raise exc
    finally:
        if backup_file.exists():
            try:
                backup_file.unlink()
            except OSError:
                pass

    # 7. Compute new ETag and retrieve final synchronized record
    new_etag = compute_file_sha256(annos_file)
    with open(annos_file, "r", encoding="utf-8") as f:
        final_records = json.load(f)
    final_rec = final_records[rec_idx]

    return {
        "status": "success",
        "frame_id": frame_id,
        "base_etag": new_etag,
        "record": final_rec,
        "sync_result": sync_res
    }


# =============================================================================
# Local HTTP Server & API Handlers (Restricted to Localhost & Pack Directory)
# =============================================================================

class ReviewEditorRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Local HTTP request handler for the annotation editor.
    Binds strictly to localhost (127.0.0.1) and enforces path sandboxing
    within the target review pack directory.
    """
    server_pack_dir: Path

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal output clean; uncomment for debugging
        # sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")
        pass

    def send_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status_code: int, message: str) -> None:
        self.send_json(status_code, {"status": "error", "message": message})

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # 1. Root: Serve Editor UI
        if path in ("/", "/index.html"):
            if not UI_HTML_PATH.exists():
                self.send_error_json(500, f"UI template missing: {UI_HTML_PATH}")
                return
            html_bytes = UI_HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        # 2. API: Status & Metadata
        if path == "/api/status":
            annos_file = self.server_pack_dir / "annotations" / "annotations.json"
            etag = compute_file_sha256(annos_file) if annos_file.exists() else None
            self.send_json(200, {
                "pack_dir": str(self.server_pack_dir),
                "base_etag": etag,
                "server_time": datetime.now(timezone.utc).isoformat()
            })
            return

        # 3. API: List Frames & Progress Summary
        if path == "/api/frames":
            annos_file = self.server_pack_dir / "annotations" / "annotations.json"
            if not annos_file.exists():
                self.send_error_json(404, "Pack annotations.json not found")
                return

            with open(annos_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            summary = {
                "total": len(records),
                "verified": 0,
                "draft": 0,
                "unreviewed": 0,
                "unannotated": 0,
                "rejected": 0,
                "uncertain": 0,
                "total_boxes": 0
            }

            frame_list = []
            for r in records:
                fid = r["frame_id"]
                r_status = r.get("review_status", "unreviewed")
                is_unann = bool(r.get("is_unannotated", False))
                box_count = len(r.get("boxes", []))

                if is_unann:
                    summary["unannotated"] += 1
                elif r_status == "verified":
                    summary["verified"] += 1
                elif r_status == "draft":
                    summary["draft"] += 1
                elif r_status == "rejected":
                    summary["rejected"] += 1
                elif r_status == "uncertain":
                    summary["uncertain"] += 1
                else:
                    summary["unreviewed"] += 1

                summary["total_boxes"] += box_count

                frame_list.append({
                    "frame_id": fid,
                    "canonical_source_id": r.get("canonical_source_id", fid),
                    "data_origin": r.get("data_origin", ""),
                    "camera": r.get("camera", ""),
                    "lighting_type": r.get("lighting_type", ""),
                    "diagnostic_group": r.get("diagnostic_group", ""),
                    "diagnostic_group_label": r.get("diagnostic_group_label", ""),
                    "review_tags": r.get("review_tags", []),
                    "risk_factors": r.get("risk_factors", []),
                    "required_human_checks": r.get("required_human_checks", []),
                    "review_status": r_status,
                    "annotation_state": r.get("annotation_state", ""),
                    "is_unannotated": is_unann,
                    "is_ambiguous": bool(r.get("is_ambiguous", False)),
                    "is_rejected": bool(r.get("is_rejected", False)),
                    "box_count": box_count
                })

            self.send_json(200, {
                "frames": frame_list,
                "summary": summary
            })
            return

        # 4. API: Get Single Frame Details
        frame_match = re.match(r"^/api/frame/([a-zA-Z0-9_\-]+)$", path)
        if frame_match:
            frame_id = frame_match.group(1)
            annos_file = self.server_pack_dir / "annotations" / "annotations.json"
            if not annos_file.exists():
                self.send_error_json(404, "Annotations file not found")
                return

            with open(annos_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            rec = next((r for r in records if r.get("frame_id") == frame_id), None)
            if not rec:
                self.send_error_json(404, f"Frame '{frame_id}' not found")
                return

            etag = compute_file_sha256(annos_file)
            self.send_json(200, {
                "frame_id": frame_id,
                "record": rec,
                "base_etag": etag
            })
            return

        # 5. Static Images & Previews (Strict Path Sandboxing)
        if path.startswith("/images/") or path.startswith("/previews/"):
            parts = path.strip("/").split("/")
            folder = parts[0]
            filename = parts[1] if len(parts) > 1 else ""

            # Prevent directory traversal
            clean_filename = Path(filename).name
            if not clean_filename or clean_filename != filename:
                self.send_error_json(403, "Invalid image path")
                return

            target_file = (self.server_pack_dir / folder / clean_filename).resolve()
            expected_parent = (self.server_pack_dir / folder).resolve()

            if expected_parent not in target_file.parents or not target_file.exists():
                self.send_error_json(404, "Image file not found")
                return

            content_type, _ = mimetypes.guess_type(str(target_file))
            content_type = content_type or "image/jpeg"

            img_bytes = target_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(img_bytes)))
            # No cache on previews so live changes update immediately
            if folder == "previews":
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            else:
                self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(img_bytes)
            return

        self.send_error_json(404, f"Endpoint not found: {path}")

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Frame Save Endpoint: POST /api/frame/<frame_id>/save
        save_match = re.match(r"^/api/frame/([a-zA-Z0-9_\-]+)/save$", path)
        if save_match:
            frame_id = save_match.group(1)
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_error_json(400, "Empty request body")
                return

            raw_body = self.wfile.read(content_length) if hasattr(self.wfile, "_read") else self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self.send_error_json(400, f"Malformed JSON: {e}")
                return

            base_etag = payload.get("base_etag")
            action = payload.get("action", "save_draft")
            boxes = payload.get("boxes", [])
            reviewer_notes = payload.get("reviewer_notes", "")
            is_ambiguous = payload.get("is_ambiguous", False)

            try:
                res = save_frame_annotation(
                    pack_dir=self.server_pack_dir,
                    frame_id=frame_id,
                    base_etag=base_etag,
                    action=action,
                    boxes_data=boxes,
                    reviewer_notes=reviewer_notes,
                    is_ambiguous=is_ambiguous
                )
                self.send_json(200, res)
            except StaleSaveError as e:
                self.send_json(409, {
                    "status": "conflict",
                    "error": "stale_save",
                    "message": str(e)
                })
            except (ValueError, KeyError) as e:
                self.send_error_json(400, str(e))
            except Exception as e:
                self.send_error_json(500, f"Save failed: {e}")
            return

        self.send_error_json(404, f"Endpoint not found: {path}")


def create_editor_server(pack_dir: Path, host: str = "127.0.0.1", port: int = 8080) -> socketserver.TCPServer:
    """
    Creates and configures the TCPServer instance for the annotation editor.
    """
    class ConfiguredHandler(ReviewEditorRequestHandler):
        server_pack_dir = pack_dir.resolve()

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer((host, port), ConfiguredHandler)
    return server


def run_editor(
    pack_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True
) -> None:
    """
    Runs the visual annotation editor HTTP server.
    """
    pack_dir = Path(pack_dir).resolve()
    if not pack_dir.exists():
        raise FileNotFoundError(f"Review pack directory not found: {pack_dir}")

    annos_file = pack_dir / "annotations" / "annotations.json"
    if not annos_file.exists():
        raise FileNotFoundError(f"Review pack missing annotations.json: {annos_file}")

    server = create_editor_server(pack_dir, host=host, port=port)
    url = f"http://{host}:{port}/"

    print("\n" + "=" * 70)
    print("  YOLO26s Review Pack Visual Annotation Editor (Batch 2B)")
    print("=" * 70)
    print(f"  Pack Directory:  {pack_dir}")
    print(f"  Local URL:       {url}")
    print(f"  Taxonomy:        0: car | 1: motorcycle | 2: bus | 3: truck | 4: three_wheeler")
    print(f"  Security:        Bound to {host} (local only)")
    print("=" * 70)
    print("  Press Ctrl+C to stop the editor server.\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation editor server...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Visual Annotation Editor for YOLO26s Review Pack (Batch 2B)")
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("data/review_pack_v1"),
        help="Path to evaluation review pack directory (default: data/review_pack_v1)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1 for local only)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to listen on (default: 8080)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser"
    )

    args = parser.parse_args()
    run_editor(
        pack_dir=args.pack,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()
