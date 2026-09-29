"""
tools/prepare_spot_check_review_pack.py - Preparation Tool for Visual Spot-Check Review Pack (Batch 5).

Consumes data/training_manifests_v2/visual_spot_check_manifest.json to build an editable review pack (data/review_pack_v2):
- Downloads the 25 external UA-DETRAC images from their recorded URLs with disk caching.
- Validates decoding and exact dimensions against NDJSON metadata; reports any mismatch/failure.
- Copies the 19 matching local images and labels from data/multiclass_dataset.
- Preserves exact image-label pairing and augmentation ancestry without modifying source labels.
- Initializes labels as unverified proposals with stable instance IDs, keeping an immutable copy in proposals/.
- Preserves saved human edits, reviewer notes, and review statuses on rerun (never resets human work).
- Generates visual box-overlay previews, manifest.json, and interactive review_index.html.
- Fully compatible with tools/annotation_editor.py.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    THAI_5CLASS_NAMES,
    parse_frame_provenance,
    validate_box,
)
from tools.prepare_review_pack import (
    CLASS_PREVIEW_COLORS,
    atomic_replace_file,
    compute_boxes_label_hash,
    compute_file_sha256,
    compute_label_file_hash,
    compute_preview_meta_hash,
    infer_provisional_subtype,
    render_box_preview_image,
    sync_review_pack,
    verify_image_file,
)
from tools.annotation_editor import generate_stable_instance_id

# UA-DETRAC true class name mapping (correcting for alphabetical header in raw NDJSON)
# Raw UA-DETRAC YOLO indices: 0: truck (others), 1: car, 2: van, 3: bus
DEFAULT_UADETRAC_CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}

# Mapping from UA-DETRAC raw classes to Thai 5-Class taxonomy
# 0: truck/others -> 3: truck
# 1: car          -> 0: car
# 2: van          -> 0: car (passenger van)
# 3: bus          -> 2: bus
UADETRAC_TO_THAI5_MAP = {
    0: 3,  # truck/others -> truck
    1: 0,  # car          -> car
    2: 0,  # van          -> car (passenger van)
    3: 2,  # bus          -> bus
}

# Provisional subtypes for external proposals
EXTERNAL_CLASS_SUBTYPES = {
    0: "light_vehicle_provisional",
    2: "bus_provisional",
    3: "commercial_truck_provisional",
}


def download_and_validate_external_image(
    url: str,
    expected_filename: str,
    expected_w: int,
    expected_h: int,
    dest_path: Path,
    cache_dir: Optional[Path] = None,
    timeout: int = 15
) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """
    Downloads an external image from its URL with disk caching.
    Validates decoding and exact pixel dimensions against expected NDJSON metadata.
    Returns (success, error_message, (actual_w, actual_h)).
    """
    # 1. Check cache first
    cached_file: Optional[Path] = None
    if cache_dir and (cache_dir / expected_filename).exists():
        cached_file = cache_dir / expected_filename
    elif dest_path.exists():
        cached_file = dest_path

    if cached_file and cached_file.exists():
        try:
            with Image.open(cached_file) as im:
                w, h = im.size
                if w == expected_w and h == expected_h:
                    if cached_file != dest_path:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cached_file, dest_path)
                    return True, None, (w, h)
                else:
                    return False, f"Cached image dimension mismatch: expected {expected_w}x{expected_h}, got {w}x{h}", (w, h)
        except Exception as e:
            # Corrupted cache file: remove and re-download
            try:
                cached_file.unlink()
            except OSError:
                pass

    # 2. Download from recorded URL
    if not url:
        return False, "Missing URL for external image", None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        return False, f"HTTP download failed from {url}: {e}", None

    # 3. Validate image decoding and dimensions
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        # Verify complete decoding
        im.load()
    except Exception as e:
        return False, f"Image decoding failed for downloaded bytes: {e}", None

    if w != expected_w or h != expected_h:
        return False, f"Dimension mismatch: expected {expected_w}x{expected_h}, got {w}x{h}", (w, h)

    # 4. Save to destination and cache
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(data)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_target = cache_dir / expected_filename
        if not cached_target.exists():
            with open(cached_target, "wb") as f:
                f.write(data)

    return True, None, (w, h)


def render_spot_check_html_index(
    output_dir: Path,
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """Renders visual review index dashboard for the 44-frame spot check pack."""
    index_file = output_dir / "review_index.html"

    # Category counts
    tag_counts = Counter()
    status_counts = Counter()
    for r in records:
        status_counts[r.get("review_status", "unreviewed")] += 1
        for t in r.get("review_tags", []):
            tag_counts[t] += 1

    rows_html = []
    for idx, r in enumerate(records, 1):
        fid = r["frame_id"]
        c_src = r.get("canonical_source_id", fid)
        origin = r.get("data_origin", "unknown")
        status = r.get("review_status", "unreviewed")
        tags = r.get("review_tags", [])
        risk_factors = r.get("risk_factors", [])
        checks = r.get("required_human_checks", [])
        boxes = r.get("boxes", [])

        # Format tag badges
        tag_badges = "".join([f'<span class="badge tag-badge">{t}</span>' for t in tags])

        # Format status badge
        status_class = f"status-{status}"
        status_badge = f'<span class="status-badge {status_class}">{status.upper()}</span>'

        # Class counts in frame
        c_counts = Counter(b.get("class_name", "unknown") for b in boxes)
        box_summary_str = ", ".join(f"{k}: {v}" for k, v in sorted(c_counts.items())) if boxes else "No boxes (empty)"

        # Thumbnail path relative to HTML
        preview_rel = f"previews/{fid}.jpg"
        clean_rel = f"images/{fid}.jpg"

        checks_html = "".join([f"<li>{c}</li>" for c in checks])
        risks_html = "".join([f"<li>{rf}</li>" for rf in risk_factors])

        row = f"""
        <tr class="review-row" data-status="{status}" data-tags="{' '.join(tags)}" data-origin="{origin}">
            <td class="col-idx">{idx}</td>
            <td class="col-thumb">
                <a href="{preview_rel}" target="_blank" title="View Full Preview">
                    <img src="{preview_rel}" class="thumb-img" alt="{fid}" loading="lazy">
                </a>
            </td>
            <td class="col-info">
                <div class="frame-title"><code>{fid}</code></div>
                <div class="frame-sub">Canonical: <code>{c_src}</code> | Origin: <strong>{origin}</strong></div>
                <div class="tags-container">{tag_badges}</div>
                <div class="box-summary"><strong>Boxes ({len(boxes)}):</strong> {box_summary_str}</div>
            </td>
            <td class="col-checks">
                <div class="checklist-section">
                    <div class="check-title">Inspection Objectives:</div>
                    <ul class="check-list">{checks_html}</ul>
                    <div class="check-title" style="margin-top: 4px; color: var(--color-orange);">Known Risk Factors:</div>
                    <ul class="check-list">{risks_html}</ul>
                </div>
            </td>
            <td class="col-status">
                {status_badge}
                <div class="links-stack">
                    <a href="http://127.0.0.1:8080/?frame={fid}" target="_blank" class="action-btn">Open in Editor ↗</a>
                    <a href="{clean_rel}" target="_blank" class="sub-link">Clean Image</a>
                    <a href="proposals/{fid}.txt" target="_blank" class="sub-link">Initial Proposal</a>
                </div>
            </td>
        </tr>
        """
        rows_html.append(row)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26s Spot-Check Visual Review Pack (Batch 5)</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-surface: #111827;
            --bg-card: #182234;
            --border-color: #2e3b52;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-orange: #fb923c;
            --accent-purple: #c084fc;
            --accent-yellow: #facc15;
            --accent-red: #f87171;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }}
        .title {{ font-size: 1.5rem; font-weight: 700; color: var(--accent-blue); margin-bottom: 6px; }}
        .subtitle {{ font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 14px; }}
        .meta-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            padding-top: 12px;
        }}
        .meta-pill {{ background: var(--bg-card); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border-color); }}
        .filter-panel {{
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
        }}
        .filter-label {{ font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); }}
        .filter-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .filter-btn:hover {{ border-color: var(--accent-blue); }}
        .filter-btn.active {{ background: rgba(56, 189, 248, 0.18); border-color: var(--accent-blue); color: var(--accent-blue); font-weight: 600; }}
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-surface);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th {{
            background: #151e30;
            color: var(--text-secondary);
            font-weight: 600;
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            vertical-align: top;
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .col-idx {{ width: 40px; color: var(--text-muted); text-align: center; }}
        .col-thumb {{ width: 140px; text-align: center; }}
        .thumb-img {{
            width: 120px;
            height: 68px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            transition: transform 0.15s ease;
        }}
        .thumb-img:hover {{ transform: scale(1.06); border-color: var(--accent-blue); }}
        .frame-title {{ font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 3px; }}
        .frame-sub {{ font-size: 0.76rem; color: var(--text-muted); margin-bottom: 6px; }}
        .tags-container {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }}
        .badge {{
            font-size: 0.72rem;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .tag-badge {{ background: rgba(192, 132, 252, 0.15); border: 1px solid rgba(192, 132, 252, 0.4); color: var(--accent-purple); }}
        .box-summary {{ font-size: 0.78rem; color: var(--text-secondary); }}
        .checklist-section {{ font-size: 0.78rem; line-height: 1.4; }}
        .check-title {{ font-weight: 700; color: var(--accent-blue); margin-bottom: 2px; }}
        .check-list {{ padding-left: 18px; color: var(--text-secondary); }}
        .status-badge {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            margin-bottom: 8px;
        }}
        .status-unreviewed {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .status-verified {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.4); }}
        .status-draft {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .status-rejected {{ background: rgba(239, 68, 68, 0.25); color: #fca5a5; border: 1px solid #ef4444; }}
        .status-uncertain {{ background: rgba(192, 132, 252, 0.2); color: #d8b4fe; border: 1px solid #c084fc; }}
        .links-stack {{ display: flex; flex-direction: column; gap: 4px; }}
        .action-btn {{
            display: inline-block;
            background: var(--accent-blue);
            color: #0b0f19;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.75rem;
            text-align: center;
        }}
        .action-btn:hover {{ background: #7dd3fc; }}
        .sub-link {{ font-size: 0.72rem; color: var(--text-muted); text-decoration: none; }}
        .sub-link:hover {{ color: var(--accent-blue); text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">YOLO26s Thai Traffic Vision - Spot-Check Visual Review Pack (Batch 5)</h1>
        <div class="subtitle">
            Pre-Training Verification Gate: 44 unique canonical source frames selected to audit class boundaries, teacher false positives, and unannotated background vehicles.
        </div>
        <div class="meta-strip">
            <div class="meta-pill"><strong>Total Review Frames:</strong> {len(records)}</div>
            <div class="meta-pill"><strong>External UA-DETRAC:</strong> {sum(1 for r in records if r.get('data_origin') == 'external_ua_detrac')}</div>
            <div class="meta-pill"><strong>Local Thai Footage:</strong> {sum(1 for r in records if r.get('data_origin') == 'local')}</div>
            <div class="meta-pill"><strong>Unreviewed:</strong> <span style="color: var(--accent-red);">{status_counts['unreviewed']}</span></div>
            <div class="meta-pill"><strong>Draft:</strong> <span style="color: var(--accent-yellow);">{status_counts['draft']}</span></div>
            <div class="meta-pill"><strong>Verified:</strong> <span style="color: var(--accent-green);">{status_counts['verified']}</span></div>
            <div class="meta-pill"><strong>Rejected:</strong> <span style="color: var(--accent-orange);">{status_counts['rejected']}</span></div>
        </div>
    </div>

    <div class="filter-panel">
        <span class="filter-label">Filter Category:</span>
        <button class="filter-btn active" onclick="filterTable('all')">All ({len(records)})</button>
        <button class="filter-btn" onclick="filterTable('external_vans')">External Vans ({tag_counts['external_vans']})</button>
        <button class="filter-btn" onclick="filterTable('external_trucks')">External Trucks ({tag_counts['external_trucks']})</button>
        <button class="filter-btn" onclick="filterTable('external_dense_small')">External Dense Small ({tag_counts['external_dense_small']})</button>
        <button class="filter-btn" onclick="filterTable('local_teacher_completed')">Local Teacher ({tag_counts['local_teacher_completed']})</button>
        <button class="filter-btn" onclick="filterTable('local_night_congestion')">Local Night ({tag_counts['local_night_congestion']})</button>
        <span class="filter-label" style="margin-left: 12px;">Origin:</span>
        <button class="filter-btn" onclick="filterOrigin('external_ua_detrac')">External Only (25)</button>
        <button class="filter-btn" onclick="filterOrigin('local')">Local Only (19)</button>
    </div>

    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th class="col-idx">#</th>
                    <th class="col-thumb">Preview</th>
                    <th class="col-info">Frame Information & Tags</th>
                    <th class="col-checks">Human Review Checklist</th>
                    <th class="col-status">Review Status & Actions</th>
                </tr>
            </thead>
            <tbody id="review-tbody">
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>

    <script>
        function filterTable(tag) {{
            const rows = document.querySelectorAll('.review-row');
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            rows.forEach(r => {{
                if (tag === 'all') {{
                    r.style.display = '';
                }} else {{
                    const tags = (r.getAttribute('data-tags') || '').split(' ');
                    r.style.display = tags.includes(tag) ? '' : 'none';
                }}
            }});
        }}

        function filterOrigin(origin) {{
            const rows = document.querySelectorAll('.review-row');
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            rows.forEach(r => {{
                r.style.display = (r.getAttribute('data-origin') === origin) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    index_file.write_text(html, encoding="utf-8")


def prepare_spot_check_review_pack(
    spot_manifest_path: Path,
    ndjson_path: Path,
    local_dataset_dir: Path,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Builds the 44-frame visual spot check review pack:
    - 25 external frames downloaded & validated against NDJSON
    - 19 local frames copied from multiclass_dataset
    - Proposals initialized with stable IDs
    - Preserves saved human annotations on rerun
    """
    if not spot_manifest_path.exists():
        raise FileNotFoundError(f"Spot check manifest not found: {spot_manifest_path}")
    if not ndjson_path.exists():
        raise FileNotFoundError(f"UA-DETRAC NDJSON file not found: {ndjson_path}")
    if not local_dataset_dir.exists():
        raise FileNotFoundError(f"Local multiclass dataset directory not found: {local_dataset_dir}")

    spot_manifest = json.loads(spot_manifest_path.read_text(encoding="utf-8"))
    spot_frames = spot_manifest.get("spot_checks", [])
    if not spot_frames:
        raise ValueError(f"Spot check manifest has 0 frames: {spot_manifest_path}")

    # Set up directory layout
    images_dir = output_dir / "images"
    previews_dir = output_dir / "previews"
    proposals_dir = output_dir / "proposals"
    annos_dir = output_dir / "annotations"
    labels_dir = annos_dir / "labels"

    for d in [images_dir, previews_dir, proposals_dir, labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load existing human annotations if rerun to preserve progress
    existing_records_by_fid: Dict[str, Dict[str, Any]] = {}
    annos_file = annos_dir / "annotations.json"
    if annos_file.exists():
        try:
            with open(annos_file, "r", encoding="utf-8") as f:
                ex_list = json.load(f)
                for ex_rec in ex_list:
                    existing_records_by_fid[ex_rec["frame_id"]] = ex_rec
            print(f"[RELOAD] Found {len(existing_records_by_fid)} existing records in {annos_file}. Human edits will be preserved.")
        except Exception as e:
            print(f"[WARN] Failed to read existing annotations.json: {e}")

    # Separate external and local frames
    ext_frames = [f for f in spot_frames if f.get("data_origin") == "external_ua_detrac"]
    loc_frames = [f for f in spot_frames if f.get("data_origin") == "local"]

    print(f"[PREPARE] Spot check frames: {len(spot_frames)} total (External: {len(ext_frames)}, Local: {len(loc_frames)})")

    # Index external NDJSON for requested files
    ext_files_set = {f["image_path"] for f in ext_frames}
    ndjson_records: Dict[str, Dict[str, Any]] = {}
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                obj = json.loads(line_s)
                fl = obj.get("file")
                if fl in ext_files_set:
                    ndjson_records[fl] = obj
            except Exception:
                continue

    download_failures: List[Dict[str, Any]] = []
    dimension_mismatches: List[Dict[str, Any]] = []
    processed_records: List[Dict[str, Any]] = []

    # 1. Process 25 External Frames
    print("\n--- Processing 25 External UA-DETRAC Frames ---")
    for ext_info in ext_frames:
        c_src = ext_info["canonical_source_id"]
        inv_id = ext_info["inventory_id"]
        raw_file = ext_info["image_path"]
        seq_id = ext_info.get("sequence_id", "unknown_seq")
        review_tags = ext_info.get("review_tags", [])
        risk_factors = ext_info.get("risk_factors", [])
        checks = ext_info.get("required_human_checks", [])

        # Frame ID in review pack: clean canonical ID
        frame_id = c_src
        dest_img_path = images_dir / f"{frame_id}.jpg"
        dest_prop_path = proposals_dir / f"{frame_id}.txt"
        dest_lbl_path = labels_dir / f"{frame_id}.txt"
        dest_preview_path = previews_dir / f"{frame_id}.jpg"

        nd_entry = ndjson_records.get(raw_file)
        if not nd_entry:
            err = f"External file '{raw_file}' not found in NDJSON {ndjson_path}"
            download_failures.append({"frame_id": frame_id, "file": raw_file, "error": err})
            print(f"  [ERROR] {err}")
            continue

        url = nd_entry.get("url", "")
        exp_w = int(nd_entry.get("width", 640))
        exp_h = int(nd_entry.get("height", 640))

        # Download and validate image
        success, err_msg, actual_dims = download_and_validate_external_image(
            url=url,
            expected_filename=raw_file,
            expected_w=exp_w,
            expected_h=exp_h,
            dest_path=dest_img_path,
            cache_dir=cache_dir
        )

        if not success or actual_dims is None:
            if "Dimension mismatch" in str(err_msg):
                dimension_mismatches.append({"frame_id": frame_id, "file": raw_file, "error": err_msg})
            else:
                download_failures.append({"frame_id": frame_id, "file": raw_file, "error": err_msg})
            print(f"  [FAIL] {frame_id}: {err_msg}")
            continue

        img_w, img_h = actual_dims

        # Check existing human work
        existing_rec = existing_records_by_fid.get(frame_id)
        has_human_edits = (
            existing_rec is not None and
            (existing_rec.get("review_status") in ("verified", "draft", "rejected", "uncertain") or
             bool(existing_rec.get("reviewer_notes")) or
             bool(existing_rec.get("is_ambiguous")) or
             bool(existing_rec.get("is_rejected")))
        )

        # Parse initial proposal boxes from NDJSON
        raw_annos = nd_entry.get("annotations", {})
        raw_boxes = raw_annos.get("boxes", []) if isinstance(raw_annos, dict) else (raw_annos if isinstance(raw_annos, list) else [])

        initial_boxes: List[Dict[str, Any]] = []
        proposal_lines: List[str] = []

        for b_idx, b in enumerate(raw_boxes):
            is_valid, b_err, val_box = validate_box(
                b, allowed_classes=DEFAULT_UADETRAC_CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE
            )
            if not is_valid or val_box is None:
                continue

            src_cid, xc, yc, bw, bh = val_box
            src_name = DEFAULT_UADETRAC_CLASS_NAMES.get(src_cid, "unknown")
            tgt_cid = UADETRAC_TO_THAI5_MAP.get(src_cid, 0)
            tgt_name = THAI_5CLASS_NAMES[tgt_cid]

            # Subtype
            subtype = "passenger_van" if src_name == "van" else EXTERNAL_CLASS_SUBTYPES.get(tgt_cid, f"{tgt_name}_provisional")
            inst_id = f"{frame_id}_inst_{b_idx:03d}"

            norm_box = [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)]
            initial_boxes.append({
                "instance_id": inst_id,
                "class_id": tgt_cid,
                "class_name": tgt_name,
                "subtype": subtype,
                "is_ambiguous": False,
                "ambiguity_reason": "",
                "bbox_norm": norm_box,
                "proposal_source": f"external_ndjson:{src_name}({src_cid})",
                "original_source_class_id": src_cid,
                "original_source_class_name": src_name
            })
            proposal_lines.append(f"{tgt_cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Write immutable proposal file
        dest_prop_path.write_text("\n".join(proposal_lines) + ("\n" if proposal_lines else ""), encoding="utf-8")

        if has_human_edits and existing_rec:
            # Preserve human edited boxes, notes, and status
            final_boxes = existing_rec.get("boxes", initial_boxes)
            review_status = existing_rec.get("review_status", "unreviewed")
            reviewer_notes = existing_rec.get("reviewer_notes", "")
            is_ambiguous = existing_rec.get("is_ambiguous", False)
            is_rejected = existing_rec.get("is_rejected", False)
            annotation_state = existing_rec.get("annotation_state", "annotated")
            # If label file exists on disk, keep it
            if not dest_lbl_path.exists():
                lbl_lines = [f"{b['class_id']} {b['bbox_norm'][0]:.6f} {b['bbox_norm'][1]:.6f} {b['bbox_norm'][2]:.6f} {b['bbox_norm'][3]:.6f}" for b in final_boxes]
                dest_lbl_path.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")
        else:
            final_boxes = initial_boxes
            review_status = "unreviewed"
            reviewer_notes = ""
            is_ambiguous = False
            is_rejected = False
            annotation_state = "annotated" if final_boxes else "unreviewed_machine_empty"
            # Write editable label file initialized from proposals
            dest_lbl_path.write_text("\n".join(proposal_lines) + ("\n" if proposal_lines else ""), encoding="utf-8")

        # Render preview image
        render_box_preview_image(
            clean_image_path=dest_img_path,
            boxes=final_boxes,
            output_preview_path=dest_preview_path,
            is_unannotated=False,
            annotation_state=annotation_state
        )

        rec = {
            "frame_id": frame_id,
            "canonical_source_id": c_src,
            "inventory_id": inv_id,
            "data_origin": "external_ua_detrac",
            "source_raw_file": raw_file,
            "sequence_id": seq_id,
            "camera": seq_id,
            "lighting_type": "daylight",
            "review_tags": review_tags,
            "risk_factors": risk_factors,
            "required_human_checks": checks,
            "dimensions": {"width": img_w, "height": img_h},
            "clean_image_file": f"images/{frame_id}.jpg",
            "preview_image_file": f"previews/{frame_id}.jpg",
            "proposal_label_file": f"proposals/{frame_id}.txt",
            "editable_label_file": f"annotations/labels/{frame_id}.txt",
            "synced_label_hash": compute_label_file_hash(dest_lbl_path),
            "preview_meta_hash": compute_preview_meta_hash(final_boxes, annotation_state, False),
            "review_status": review_status,
            "annotation_state": annotation_state,
            "is_unannotated": False,
            "is_ambiguous": is_ambiguous,
            "is_rejected": is_rejected,
            "reviewer_notes": reviewer_notes,
            "boxes": final_boxes
        }
        processed_records.append(rec)
        print(f"  [OK] {frame_id} ({img_w}x{img_h}) -> {len(final_boxes)} boxes, tags: {review_tags}")

    # 2. Process 19 Local Frames
    print("\n--- Processing 19 Local Frames ---")
    for loc_info in loc_frames:
        c_src = loc_info["canonical_source_id"]
        inv_id = loc_info["inventory_id"]
        img_rel = loc_info["image_path"]
        seq_id = loc_info.get("sequence_id", "unknown_seq")
        review_tags = loc_info.get("review_tags", [])
        risk_factors = loc_info.get("risk_factors", [])
        checks = loc_info.get("required_human_checks", [])

        frame_id = c_src
        dest_img_path = images_dir / f"{frame_id}.jpg"
        dest_prop_path = proposals_dir / f"{frame_id}.txt"
        dest_lbl_path = labels_dir / f"{frame_id}.txt"
        dest_preview_path = previews_dir / f"{frame_id}.jpg"

        src_img_path = Path(img_rel)
        if not src_img_path.is_absolute():
            src_img_path = Path(REPO_ROOT) / img_rel if not src_img_path.exists() else src_img_path

        src_lbl_path = Path(str(src_img_path).replace("images", "labels").replace(".jpg", ".txt"))

        if not src_img_path.exists():
            download_failures.append({"frame_id": frame_id, "file": str(src_img_path), "error": "Local image missing on disk"})
            print(f"  [FAIL] {frame_id}: Local image missing at {src_img_path}")
            continue

        # Validate local image
        try:
            with Image.open(src_img_path) as im:
                img_w, img_h = im.size
                im.load()
        except Exception as e:
            download_failures.append({"frame_id": frame_id, "file": str(src_img_path), "error": f"Local image corrupt: {e}"})
            print(f"  [FAIL] {frame_id}: Local image corrupt: {e}")
            continue

        # Copy clean local image
        shutil.copy2(src_img_path, dest_img_path)

        # Parse local provenance
        stem = src_img_path.stem
        prov = parse_frame_provenance(stem)

        # Check existing human work
        existing_rec = existing_records_by_fid.get(frame_id)
        has_human_edits = (
            existing_rec is not None and
            (existing_rec.get("review_status") in ("verified", "draft", "rejected", "uncertain") or
             bool(existing_rec.get("reviewer_notes")) or
             bool(existing_rec.get("is_ambiguous")) or
             bool(existing_rec.get("is_rejected")))
        )

        # Parse initial proposal boxes from local label
        initial_boxes = []
        proposal_lines = []

        if src_lbl_path.exists():
            content = src_lbl_path.read_text(encoding="utf-8").strip()
            if content:
                for b_idx, line in enumerate(content.splitlines()):
                    line_s = line.strip()
                    if not line_s or line_s.startswith("#"):
                        continue
                    parts = line_s.split()
                    is_valid, b_err, val_box = validate_box(parts, THAI_5CLASS_NAMES, edge_tolerance=BOX_EDGE_ROUNDING_TOLERANCE)
                    if not is_valid or val_box is None:
                        continue
                    cid, xc, yc, bw, bh = val_box
                    cname = THAI_5CLASS_NAMES[cid]
                    subtype = infer_provisional_subtype(cid, frame_id, (xc, yc, bw, bh))
                    inst_id = f"{frame_id}_inst_{b_idx:03d}"
                    norm_box = [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)]

                    initial_boxes.append({
                        "instance_id": inst_id,
                        "class_id": cid,
                        "class_name": cname,
                        "subtype": subtype,
                        "is_ambiguous": False,
                        "ambiguity_reason": "",
                        "bbox_norm": norm_box,
                        "proposal_source": f"local_dataset:{src_lbl_path.name}"
                    })
                    proposal_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Write immutable proposal file
        dest_prop_path.write_text("\n".join(proposal_lines) + ("\n" if proposal_lines else ""), encoding="utf-8")

        if has_human_edits and existing_rec:
            final_boxes = existing_rec.get("boxes", initial_boxes)
            review_status = existing_rec.get("review_status", "unreviewed")
            reviewer_notes = existing_rec.get("reviewer_notes", "")
            is_ambiguous = existing_rec.get("is_ambiguous", False)
            is_rejected = existing_rec.get("is_rejected", False)
            annotation_state = existing_rec.get("annotation_state", "annotated")
            if not dest_lbl_path.exists():
                lbl_lines = [f"{b['class_id']} {b['bbox_norm'][0]:.6f} {b['bbox_norm'][1]:.6f} {b['bbox_norm'][2]:.6f} {b['bbox_norm'][3]:.6f}" for b in final_boxes]
                dest_lbl_path.write_text("\n".join(lbl_lines) + ("\n" if lbl_lines else ""), encoding="utf-8")
        else:
            final_boxes = initial_boxes
            review_status = "unreviewed"
            reviewer_notes = ""
            is_ambiguous = False
            is_rejected = False
            annotation_state = "annotated" if final_boxes else "unreviewed_machine_empty"
            dest_lbl_path.write_text("\n".join(proposal_lines) + ("\n" if proposal_lines else ""), encoding="utf-8")

        # Render preview image
        render_box_preview_image(
            clean_image_path=dest_img_path,
            boxes=final_boxes,
            output_preview_path=dest_preview_path,
            is_unannotated=False,
            annotation_state=annotation_state
        )

        rec = {
            "frame_id": frame_id,
            "canonical_source_id": c_src,
            "inventory_id": inv_id,
            "data_origin": "local",
            "source_raw_file": str(src_img_path).replace("\\", "/"),
            "sequence_id": prov.video,
            "camera": prov.camera,
            "lighting_type": prov.lighting,
            "variant": prov.variant,
            "review_tags": review_tags,
            "risk_factors": risk_factors,
            "required_human_checks": checks,
            "dimensions": {"width": img_w, "height": img_h},
            "clean_image_file": f"images/{frame_id}.jpg",
            "preview_image_file": f"previews/{frame_id}.jpg",
            "proposal_label_file": f"proposals/{frame_id}.txt",
            "editable_label_file": f"annotations/labels/{frame_id}.txt",
            "synced_label_hash": compute_label_file_hash(dest_lbl_path),
            "preview_meta_hash": compute_preview_meta_hash(final_boxes, annotation_state, False),
            "review_status": review_status,
            "annotation_state": annotation_state,
            "is_unannotated": False,
            "is_ambiguous": is_ambiguous,
            "is_rejected": is_rejected,
            "reviewer_notes": reviewer_notes,
            "boxes": final_boxes
        }
        processed_records.append(rec)
        print(f"  [OK] {frame_id} ({img_w}x{img_h}) -> {len(final_boxes)} boxes, tags: {review_tags}")

    # 3. Save Structured Store (annotations.json)
    annos_file = annos_dir / "annotations.json"
    with open(annos_file, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2)
    print(f"\n[SAVED] Structured store saved to: {annos_file}")

    # 4. Save Manifest
    manifest_meta = {
        "pack_name": "YOLO26s Thai Traffic Vision - Spot-Check Visual Review Pack (Batch 5)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(spot_frames),
        "successfully_prepared_count": len(processed_records),
        "external_prepared_count": sum(1 for r in processed_records if r["data_origin"] == "external_ua_detrac"),
        "local_prepared_count": sum(1 for r in processed_records if r["data_origin"] == "local"),
        "download_failures_count": len(download_failures),
        "dimension_mismatches_count": len(dimension_mismatches),
        "download_failures": download_failures,
        "dimension_mismatches": dimension_mismatches,
        "spot_check_quotas": spot_manifest.get("metadata", {}).get("quota_audit", {})
    }

    manifest_data = {
        "metadata": manifest_meta,
        "samples": processed_records
    }
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[SAVED] Manifest saved to: {manifest_file}")

    # 5. Render HTML Dashboard
    render_spot_check_html_index(output_dir, processed_records, manifest_meta)
    print(f"[SAVED] Review dashboard saved to: {output_dir / 'review_index.html'}")

    # 6. Render README.md
    readme_path = output_dir / "README.md"
    readme_content = f"""# YOLO26s Spot-Check Visual Review Pack (Batch 5)

- **Total Frames**: {len(processed_records)} unique canonical source frames
  - **External UA-DETRAC**: {sum(1 for r in processed_records if r['data_origin'] == 'external_ua_detrac')}
  - **Local Thai CCTV**: {sum(1 for r in processed_records if r['data_origin'] == 'local')}
- **Creation Date**: {datetime.now(timezone.utc).isoformat()}
- **Dashboard**: `data/review_pack_v2/review_index.html`

## How to Review and Correct Frames
1. Start the visual annotation editor:
   ```bash
   python tools/annotation_editor.py --pack data/review_pack_v2 --port 8080
   ```
2. Open your browser at `http://127.0.0.1:8080/`
3. Inspect each frame according to the priority checklist:
   - **External Vans**: Confirm passenger vans (0: car) vs commercial cargo trucks (3: truck).
   - **External Trucks**: Confirm medium/heavy chassis (3: truck) vs pickup flatbeds (0: car).
   - **External Small Vehicles**: Check for unannotated motorcycles or tiny truncated boxes in distant approach lanes.
   - **Local Teacher Completed**: Check background clutter boxes for false positives; verify songthaews.
   - **Local Night**: Verify dense motorcycle/three-wheeler queues under glare.
4. Correct boxes (add, resize, reclassify, move, delete) and click **✓ Mark Verified** or **✕ Reject Frame**.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[SAVED] Guide saved to: {readme_path}")

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "total_prepared": len(processed_records),
        "external_prepared": sum(1 for r in processed_records if r["data_origin"] == "external_ua_detrac"),
        "local_prepared": sum(1 for r in processed_records if r["data_origin"] == "local"),
        "download_failures": download_failures,
        "dimension_mismatches": dimension_mismatches
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Spot-Check Visual Review Pack (Batch 5)")
    parser.add_argument("--spot-manifest", type=Path, default=Path("data/training_manifests_v2/visual_spot_check_manifest.json"), help="Spot check manifest path")
    parser.add_argument("--ndjson-path", type=Path, default=Path("data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson"), help="UA-DETRAC NDJSON path")
    parser.add_argument("--local-dataset-dir", type=Path, default=Path("data/multiclass_dataset"), help="Local dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/review_pack_v2"), help="Output review pack directory")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for downloaded images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing review pack while preserving human annotations")

    args = parser.parse_args()
    res = prepare_spot_check_review_pack(
        spot_manifest_path=args.spot_manifest,
        ndjson_path=args.ndjson_path,
        local_dataset_dir=args.local_dataset_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite
    )
    print("\n" + "=" * 70)
    print(f"Preparation Summary: {res['total_prepared']} frames ready in {res['output_dir']}")
    print(f"  - External UA-DETRAC: {res['external_prepared']}")
    print(f"  - Local Thai CCTV:    {res['local_prepared']}")
    print(f"  - Download Failures:  {len(res['download_failures'])}")
    print(f"  - Dimension Mismatch: {len(res['dimension_mismatches'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
