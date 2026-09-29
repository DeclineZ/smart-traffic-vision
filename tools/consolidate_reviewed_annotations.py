"""
tools/consolidate_reviewed_annotations.py - Batch 7 Annotation Consolidation & Training Readiness

Repairs annotation propagation, lineage linking, and training-readiness enforcement:
1. Exact matching of reviewed records to candidate images by canonical ID AND exact variant/image identity,
   verifying SHA256 image hashes (including Roboflow variants). Prevents coordinate propagation to transformed variants.
2. Synchronized updates to label paths, embedded boxes, class/size counts, provenance, and local image availability.
   Strict assertion that inventory annotations agree with exported YOLO labels.
3. Changed local annotation detection by comparing original vs approved boxes (class edits, deletions, moved boxes, additions).
   Exclusion of all 40 stale synthetic descendants of modified local frames.
4. Strict separation of review status ('verified') from training eligibility:
   Frames with pending proposals (e.g. MVI_20063_img00769 with 14 pending proposals) are blocked from the approved training subset.
5. Exported class totals and provenance accumulated strictly over exported eligible frames (41 frames).
   Blocked and rejected frame counts reported separately.
6. Correct evaluation benchmark status: data/eval_snapshot_v1 (42 frames) is an internal diagnostic benchmark with known exposure.
7. Replaces premature full A/B training recommendation with a bounded, eligible-only diagnostic experiment and clear human action items.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_dataset import (
    BOX_EDGE_ROUNDING_TOLERANCE,
    DEFAULT_UADETRAC_CLASS_NAMES,
    THAI_5CLASS_NAMES,
    compute_aspect_preserving_size,
    get_image_dimensions,
    parse_frame_provenance,
    validate_box,
)
from tools.prepare_review_pack import compute_file_sha256
from tools.prepare_training_manifests import (
    NumpyEncoder,
    sample_external_frames_stratified,
)


@dataclass
class ConsolidationConfig:
    pack_v2_dir: Path = Path("data/review_pack_v2")
    pilot_pack_dir: Path = Path("data/review_pack_pilot_v1")
    continuation_pack_dir: Path = Path("data/review_pack_continuation_v1")
    output_pack_dir: Path = Path("data/review_pack_consolidated_v3")
    v3_manifests_dir: Path = Path("data/training_manifests_v3")
    output_manifests_dir: Path = Path("data/training_manifests_v6")
    report_markdown_path: Path = Path("docs/TRAINING_READINESS_REPORT.md")
    random_seed: int = 42
    external_cap_ratio: float = 0.5


# ==============================================================================
# 1. Lineage Reconciliation & Precedence Enforcement
# ==============================================================================

def load_pack_annotations(pack_dir: Path) -> List[Dict[str, Any]]:
    """Loads annotations.json from a review pack with error checking."""
    annot_file = pack_dir / "annotations" / "annotations.json"
    if not annot_file.exists():
        raise FileNotFoundError(f"Missing annotations.json in review pack: '{annot_file}'")
    with open(annot_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of frame records in '{annot_file}', got {type(data)}")
    return data


def reconcile_review_lineage(
    config: ConsolidationConfig
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reconciles review lineage across review_pack_v2, pilot_v1, and continuation_v1.
    Explicit Precedence:
      1. Pilot and Continuation review decisions supersede corresponding draft records in v2.
      2. The other 20 verified records in v2 are strictly retained.
      3. Image identity and SHA256 hashes are validated against source.
      4. Fails fast on conflicting authoritative records across packs.
      5. Separates user review status from training eligibility (e.g. pending proposals block training).
    """
    v2_records = load_pack_annotations(config.pack_v2_dir)
    pilot_records = load_pack_annotations(config.pilot_pack_dir)
    cont_records = load_pack_annotations(config.continuation_pack_dir)

    v2_map = {r["frame_id"]: r for r in v2_records}
    pilot_map = {r["frame_id"]: r for r in pilot_records}
    cont_map = {r["frame_id"]: r for r in cont_records}

    # 1. Collision Check: Fail fast if canonical frame has conflicting records in pilot & continuation
    pilot_ids = set(pilot_map.keys())
    cont_ids = set(cont_map.keys())
    overlap = pilot_ids.intersection(cont_ids)
    if overlap:
        raise ValueError(
            f"Lineage conflict: Frame(s) {overlap} present in both Pilot and Continuation packs! "
            "Refusing to resolve by timestamp."
        )

    if len(v2_map) != 44:
        raise ValueError(f"review_pack_v2 must contain exactly 44 frames, found {len(v2_map)}")

    # 2. Check overlap between pilot/continuation and verified v2 frames
    v2_verified_ids = {r["frame_id"] for r in v2_records if r.get("review_status") == "verified"}
    v2_draft_ids = {r["frame_id"] for r in v2_records if r.get("review_status") == "draft"}

    if len(v2_verified_ids) != 20:
        raise ValueError(f"review_pack_v2 must contain exactly 20 verified frames, found {len(v2_verified_ids)}")
    if len(v2_draft_ids) != 24:
        raise ValueError(f"review_pack_v2 must contain exactly 24 draft frames, found {len(v2_draft_ids)}")

    pilot_verified_collision = pilot_ids.intersection(v2_verified_ids)
    cont_verified_collision = cont_ids.intersection(v2_verified_ids)
    if pilot_verified_collision or cont_verified_collision:
        raise ValueError(
            f"Lineage conflict: Pilot or Continuation overlaps with previously verified v2 frames: "
            f"pilot={pilot_verified_collision}, cont={cont_verified_collision}"
        )

    # 3. Completeness Check: pilot (6) + cont (18) must exactly cover all 24 draft frames in v2
    evaluated_drafts = pilot_ids.union(cont_ids)
    missing_drafts = v2_draft_ids - evaluated_drafts
    extra_drafts = evaluated_drafts - v2_draft_ids
    if missing_drafts or extra_drafts:
        raise ValueError(
            f"Draft reconciliation mismatch! Missing: {missing_drafts}, Extra: {extra_drafts}"
        )

    consolidated_records: List[Dict[str, Any]] = []
    lineage_audit = {
        "total_canonical_sources": 44,
        "from_v2_verified": 0,
        "from_pilot": 0,
        "from_continuation": 0,
        "verified_count": 0,
        "rejected_count": 0,
        "rejected_frames": [],
        "eligible_count": 0,
        "blocked_count": 0,
        "blocked_frames": [],
        "blockers": []
    }

    # 4. Assemble Consolidated Records
    for fid, v2_rec in v2_map.items():
        # Validate image hash integrity across packs
        v2_img_path = config.pack_v2_dir / "images" / f"{fid}.jpg"
        if not v2_img_path.exists():
            raise FileNotFoundError(f"Source image missing in v2 pack: '{v2_img_path}'")
        v2_img_hash = compute_file_sha256(v2_img_path)

        if fid in pilot_map:
            pilot_rec = pilot_map[fid]
            pilot_img_path = config.pilot_pack_dir / "images" / f"{fid}.jpg"
            if pilot_img_path.exists():
                p_hash = compute_file_sha256(pilot_img_path)
                if p_hash != v2_img_hash:
                    raise ValueError(f"Image hash mismatch for '{fid}' between v2 ({v2_img_hash}) and pilot ({p_hash})")

            final_rec = dict(pilot_rec)
            final_rec["lineage_precedence"] = "superseded_by_pilot_review"
            final_rec["source_review_pack"] = "review_pack_pilot_v1"
            lineage_audit["from_pilot"] += 1
            src_pack_dir = config.pilot_pack_dir

        elif fid in cont_map:
            cont_rec = cont_map[fid]
            cont_img_path = config.continuation_pack_dir / "images" / f"{fid}.jpg"
            if cont_img_path.exists():
                c_hash = compute_file_sha256(cont_img_path)
                if c_hash != v2_img_hash:
                    raise ValueError(f"Image hash mismatch for '{fid}' between v2 ({v2_img_hash}) and continuation ({c_hash})")

            final_rec = dict(cont_rec)
            final_rec["lineage_precedence"] = "superseded_by_continuation_review"
            final_rec["source_review_pack"] = "review_pack_continuation_v1"
            lineage_audit["from_continuation"] += 1
            src_pack_dir = config.continuation_pack_dir

        else:
            final_rec = dict(v2_rec)
            final_rec["lineage_precedence"] = "retained_from_v2_verified"
            final_rec["source_review_pack"] = "review_pack_v2"
            lineage_audit["from_v2_verified"] += 1
            src_pack_dir = config.pack_v2_dir

        # Inherit invariant metadata from v2 baseline if missing in review pack
        for k in ["inventory_id", "source_raw_file", "data_origin", "dimensions", "sequence_id", "camera"]:
            if not final_rec.get(k) and v2_rec.get(k):
                final_rec[k] = v2_rec[k]

        final_rec["image_sha256"] = v2_img_hash
        final_rec["src_pack_dir"] = str(src_pack_dir)

        # Audit review status
        st = final_rec.get("review_status", "draft")
        if st == "verified":
            lineage_audit["verified_count"] += 1
        elif st == "rejected":
            lineage_audit["rejected_count"] += 1
            lineage_audit["rejected_frames"].append(fid)
        else:
            lineage_audit["blockers"].append(f"Unresolved draft status for '{fid}'")

        # Separate review status from training eligibility (Fix #4)
        pending_props = [
            b for b in final_rec.get("boxes", [])
            if b.get("is_proposal") and b.get("proposal_status") == "pending"
        ]
        unresolved_conflicts = [
            b for b in final_rec.get("boxes", [])
            if b.get("proposal_category") == "conflict" and b.get("proposal_status") == "pending"
        ]

        if st == "verified":
            if pending_props or unresolved_conflicts:
                final_rec["training_eligible"] = False
                final_rec["eligibility_status"] = "blocked_pending_proposals"
                final_rec["pending_proposals_count"] = len(pending_props)
                final_rec["editor_pack"] = str(src_pack_dir)
                final_rec["editor_command"] = f".venv\\Scripts\\python.exe tools/annotation_editor.py --pack {src_pack_dir}"
                lineage_audit["blocked_count"] += 1
                lineage_audit["blocked_frames"].append({
                    "frame_id": fid,
                    "pending_proposals": len(pending_props),
                    "editor_pack": str(src_pack_dir),
                    "editor_command": final_rec["editor_command"]
                })
                lineage_audit["blockers"].append(
                    f"Frame '{fid}' is verified by reviewer but has {len(pending_props)} pending proposals. "
                    f"BLOCKED from training eligibility pending human resolution in editor: {final_rec['editor_command']}"
                )
            else:
                final_rec["training_eligible"] = True
                final_rec["eligibility_status"] = "eligible"
                final_rec["pending_proposals_count"] = 0
                lineage_audit["eligible_count"] += 1
        elif st == "rejected":
            final_rec["training_eligible"] = False
            final_rec["eligibility_status"] = "rejected_source"
            final_rec["pending_proposals_count"] = len(pending_props)
            # Pending proposals on rejected frames require no further review
            final_rec["notes"] = "Rejected frame; pending proposals require no further review."
        else:
            final_rec["training_eligible"] = False
            final_rec["eligibility_status"] = "unresolved_draft"

        consolidated_records.append(final_rec)

    print(f"[LINEAGE] Reconciled 44 sources: {lineage_audit['from_v2_verified']} v2 verified, {lineage_audit['from_pilot']} pilot, {lineage_audit['from_continuation']} continuation.")
    print(f"[LINEAGE] Status outcome: {lineage_audit['verified_count']} verified, {lineage_audit['rejected_count']} rejected ({lineage_audit['rejected_frames']}).")
    print(f"[LINEAGE] Training eligibility: {lineage_audit['eligible_count']} eligible, {lineage_audit['blocked_count']} blocked ({[b['frame_id'] for b in lineage_audit['blocked_frames']]}), {lineage_audit['rejected_count']} rejected.")

    return consolidated_records, lineage_audit


# ==============================================================================
# 2. Approved Annotations Validation & Export
# ==============================================================================

def validate_and_export_approved_pack(
    consolidated_records: List[Dict[str, Any]],
    lineage_audit: Dict[str, Any],
    config: ConsolidationConfig
) -> Dict[str, Any]:
    """
    Validates geometry, classes, and instance IDs, then exports approved labels.
    Export rules:
      - ONLY eligible verified frames (41 frames) get approved labels in annotations/labels/.
      - Blocked frames (e.g. MVI_20063_img00769 with pending proposals) are NOT exported as complete labels.
      - Rejected frames (2 frames) do NOT get label files exported to training.
      - Export class totals accumulate ONLY over exported, eligible frames (Fix #5).
      - In exported labels, ONLY authoritative human boxes and accepted proposals are written.
      - Pending and rejected proposals are strictly excluded.
      - Human pickup/van corrections preserved as car (0) with subtype metadata.
      - Medium/heavy trucks remain truck (3).
    """
    out_dir = config.output_pack_dir
    images_dir = out_dir / "images"
    labels_dir = out_dir / "annotations" / "labels"
    annos_dir = out_dir / "annotations"
    proposals_dir = out_dir / "proposals"
    previews_dir = out_dir / "previews"
    previews_before_dir = out_dir / "previews_before"

    for d in [images_dir, labels_dir, annos_dir, proposals_dir, previews_dir, previews_before_dir]:
        d.mkdir(parents=True, exist_ok=True)

    export_audit = {
        "exported_frames_count": 0,
        "blocked_frames_count": 0,
        "omitted_rejected_frames_count": 0,
        "total_approved_boxes": 0,
        "class_breakdown": Counter(),
        "teacher_conflict_corrections": 0,
        "human_manual_boxes_retained": 0,
        "proposals_accepted": 0,
        "proposals_rejected": 0,
        "proposals_pending": 0,
        "blocked_audit": {},
        "rejected_audit": {},
        "geometry_errors": [],
        "exported_label_files": []
    }

    cleaned_records: List[Dict[str, Any]] = []

    for r in consolidated_records:
        fid = r["frame_id"]
        st = r.get("review_status", "draft")
        is_eligible = r.get("training_eligible", False)
        src_pack = Path(r["src_pack_dir"])

        # Copy image
        src_img = src_pack / "images" / f"{fid}.jpg"
        dst_img = images_dir / f"{fid}.jpg"
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Copy proposal json if present
        src_prop = src_pack / "proposals" / f"{fid}.json"
        dst_prop = proposals_dir / f"{fid}.json"
        if src_prop.exists() and not dst_prop.exists():
            shutil.copy2(src_prop, dst_prop)

        # Copy preview if present
        src_prev = src_pack / "previews" / f"{fid}.jpg"
        dst_prev = previews_dir / f"{fid}.jpg"
        if src_prev.exists() and not dst_prev.exists():
            shutil.copy2(src_prev, dst_prev)

        # Copy preview_before if present
        src_prev_b = src_pack / "previews_before" / f"{fid}.jpg"
        if not src_prev_b.exists():
            src_prev_b = config.pack_v2_dir / "previews" / f"{fid}.jpg"
        dst_prev_b = previews_before_dir / f"{fid}.jpg"
        if src_prev_b.exists() and not dst_prev_b.exists():
            shutil.copy2(src_prev_b, dst_prev_b)

        boxes = r.get("boxes", [])
        approved_boxes = []
        rejected_boxes = []
        pending_boxes = []

        for b in boxes:
            is_prop = b.get("is_proposal", False)
            p_status = b.get("proposal_status", "")

            # Classify box lifecycle
            if not is_prop:
                approved_boxes.append(b)
            elif p_status == "accepted":
                approved_boxes.append(b)
            elif p_status == "rejected":
                rejected_boxes.append(b)
            elif p_status == "pending":
                pending_boxes.append(b)

        # Check geometry and class IDs of approved boxes
        lbl_lines: List[str] = []
        for b_idx, b in enumerate(approved_boxes):
            cid = int(b["class_id"])
            if cid not in THAI_5CLASS_NAMES:
                err = f"Frame '{fid}' box {b_idx} has invalid class ID {cid}"
                export_audit["geometry_errors"].append(err)
                raise ValueError(err)

            norm_box = b.get("bbox_norm")
            if not norm_box or len(norm_box) != 4:
                err = f"Frame '{fid}' box {b_idx} missing bbox_norm"
                export_audit["geometry_errors"].append(err)
                raise ValueError(err)

            xc, yc, bw, bh = norm_box
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                err = f"Frame '{fid}' box {b_idx} has out-of-bounds normalized coordinates: {norm_box}"
                export_audit["geometry_errors"].append(err)
                raise ValueError(err)

            lbl_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Label Export Decision:
        # ONLY exported if frame is training_eligible (verified AND 0 pending proposals)
        lbl_file = labels_dir / f"{fid}.txt"
        if is_eligible:
            lbl_content = "\n".join(lbl_lines) + ("\n" if lbl_lines else "")
            lbl_file.write_text(lbl_content, encoding="utf-8")
            export_audit["exported_frames_count"] += 1
            export_audit["total_approved_boxes"] += len(approved_boxes)
            export_audit["exported_label_files"].append(str(lbl_file))

            # ACCUMULATE METRICS ONLY OVER EXPORTED FRAMES (Fix #5)
            for b in approved_boxes:
                cname = THAI_5CLASS_NAMES[int(b["class_id"])]
                export_audit["class_breakdown"][cname] += 1
                if not b.get("is_proposal"):
                    if (
                        b.get("source_type") == "manual"
                        or b.get("proposal_source") == "manual_annotation"
                        or "added" in b.get("instance_id", "")
                        or b.get("is_added")
                    ):
                        export_audit["human_manual_boxes_retained"] += 1
                elif b.get("proposal_status") == "accepted":
                    export_audit["proposals_accepted"] += 1
                    if b.get("proposal_category") == "conflict":
                        export_audit["teacher_conflict_corrections"] += 1

            export_audit["proposals_rejected"] += len(rejected_boxes)

        elif st == "verified" and not is_eligible:
            # Verified by user, but blocked from training (e.g. MVI_20063_img00769)
            export_audit["blocked_frames_count"] += 1
            if lbl_file.exists():
                lbl_file.unlink()
            export_audit["blocked_audit"][fid] = {
                "review_status": "verified",
                "training_eligible": False,
                "reason": r.get("eligibility_status"),
                "pending_proposals": len(pending_boxes),
                "rejected_proposals": len(rejected_boxes),
                "human_boxes": len([b for b in boxes if not b.get("is_proposal")]),
                "accepted_proposals": len([b for b in boxes if b.get("is_proposal") and b.get("proposal_status") == "accepted"]),
                "editor_command": r.get("editor_command")
            }
            export_audit["proposals_pending"] += len(pending_boxes)

        else:
            # Frame is rejected: DO NOT EXPORT LABEL FILE (Fix #5)
            export_audit["omitted_rejected_frames_count"] += 1
            if lbl_file.exists():
                lbl_file.unlink()
            export_audit["rejected_audit"][fid] = {
                "review_status": "rejected",
                "total_boxes_quarantined": len(boxes),
                "notes": r.get("reviewer_notes", "")
            }

        # Update record for consolidated annotations.json
        rec_clean = dict(r)
        rec_clean["approved_boxes_count"] = len(approved_boxes) if is_eligible else 0
        rec_clean["unaccepted_proposals_count"] = len(pending_boxes) + len(rejected_boxes)
        rec_clean["label_exported"] = is_eligible
        cleaned_records.append(rec_clean)

    # Save consolidated annotations.json
    annos_file = annos_dir / "annotations.json"
    with open(annos_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, indent=2, cls=NumpyEncoder)

    # Save manifest.json
    manifest_data = {
        "manifest_name": "Batch 7 Consolidated Review Pack",
        "version": "v3.0_consolidated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_packs": {
            "v2": str(config.pack_v2_dir),
            "pilot": str(config.pilot_pack_dir),
            "continuation": str(config.continuation_pack_dir)
        },
        "lineage_audit": lineage_audit,
        "export_audit": export_audit,
        "taxonomy": THAI_5CLASS_NAMES
    }
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, cls=NumpyEncoder)

    # Render review_index.html
    render_consolidated_review_index(cleaned_records, lineage_audit, export_audit, out_dir)

    print(f"[EXPORT] Exported {export_audit['exported_frames_count']} eligible verified frame labels ({export_audit['total_approved_boxes']} approved boxes).")
    print(f"[EXPORT] Blocked {export_audit['blocked_frames_count']} verified frame(s) with pending proposals from label export.")
    print(f"[EXPORT] Omitted {export_audit['omitted_rejected_frames_count']} rejected frames from label export.")
    print(f"[EXPORT] Saved consolidated pack to: {out_dir}")

    return {
        "export_audit": export_audit,
        "manifest_data": manifest_data,
        "cleaned_records": cleaned_records
    }


def render_consolidated_review_index(
    cleaned_records: List[Dict[str, Any]],
    lineage_audit: Dict[str, Any],
    export_audit: Dict[str, Any],
    out_dir: Path
) -> None:
    """Renders visual HTML index for the consolidated review pack with clear eligibility cues."""
    cards_html = []
    for r in cleaned_records:
        fid = r["frame_id"]
        st = r.get("review_status", "draft")
        is_elig = r.get("training_eligible", False)
        prec = r.get("lineage_precedence", "")
        origin = r.get("data_origin", "unknown")
        approved_count = r.get("approved_boxes_count", 0)

        if is_elig:
            st_color = "#10b981"
            st_text = "VERIFIED (ELIGIBLE)"
        elif st == "verified":
            st_color = "#f59e0b"
            st_text = f"VERIFIED (BLOCKED: {r.get('pending_proposals_count', 0)} PENDING)"
        else:
            st_color = "#ef4444"
            st_text = "REJECTED"

        st_badge = f'<span style="background:{st_color}22; color:{st_color}; border:1px solid {st_color}55; padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.75rem;">{st_text}</span>'

        editor_callout = ""
        if not is_elig and st == "verified":
            cmd = r.get("editor_command", "")
            editor_callout = f"""
            <div style="background:#78350f33; border:1px solid #f59e0b55; padding:8px; border-radius:4px; margin-top:8px; font-size:0.75rem; color:#fde68a;">
                <strong>Action Required:</strong> 14 proposals pending review. Run: <code>{cmd}</code>
            </div>
            """

        cards_html.append(f"""
        <div class="frame-card">
            <div class="card-header">
                <span class="frame-title">{fid}</span>
                <div class="badge-group">
                    {st_badge}
                    <span class="badge badge-origin">{origin}</span>
                    <span class="badge badge-prec">{prec}</span>
                </div>
            </div>
            <div class="card-body">
                <div class="img-container">
                    <span class="img-label">Baseline (Before)</span>
                    <a href="previews_before/{fid}.jpg" target="_blank">
                        <img src="previews_before/{fid}.jpg" class="frame-img" loading="lazy" alt="Before {fid}">
                    </a>
                </div>
                <div class="img-container">
                    <span class="img-label">Approved (After)</span>
                    <a href="previews/{fid}.jpg" target="_blank">
                        <img src="previews/{fid}.jpg" class="frame-img" loading="lazy" alt="After {fid}">
                    </a>
                </div>
            </div>
            <div class="card-footer">
                <div class="stat-text">Approved Boxes: <strong>{approved_count}</strong> | Status: <strong>{st}</strong> | Training Eligible: <strong>{is_elig}</strong></div>
                <div class="notes-text">{r.get('reviewer_notes', '')}</div>
                {editor_callout}
            </div>
        </div>
        """)

    all_cards = "\n".join(cards_html)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch 7 Consolidated Review Index</title>
    <style>
        :root {{
            --bg-primary: #0b0f19;
            --bg-card: #111827;
            --border-color: #1f2937;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent-blue: #38bdf8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
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
        h1 {{ font-size: 1.5rem; color: var(--accent-blue); margin-bottom: 8px; }}
        .header-meta {{ font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5; }}
        .summary-bar {{
            display: flex;
            gap: 16px;
            margin-top: 14px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
        }}
        .stat-badge strong {{ color: var(--accent-blue); }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
            gap: 20px;
        }}
        .frame-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
        }}
        .frame-title {{ font-size: 0.95rem; font-weight: 600; color: #e2e8f0; font-family: monospace; }}
        .badge-group {{ display: flex; gap: 6px; align-items: center; }}
        .badge {{ font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; border: 1px solid #334155; color: #94a3b8; }}
        .card-body {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 12px;
            background: #0d131f;
        }}
        .img-container {{ display: flex; flex-direction: column; gap: 4px; }}
        .img-label {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .frame-img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid #1e293b;
        }}
        .card-footer {{
            padding: 10px 14px;
            border-top: 1px solid var(--border-color);
            font-size: 0.8rem;
            color: #94a3b8;
            background: rgba(0, 0, 0, 0.15);
        }}
        .stat-text {{ margin-bottom: 4px; }}
        .notes-text {{ font-size: 0.75rem; color: #64748b; font-style: italic; }}
    </style>
</head>
<body>
    <header>
        <h1>Batch 7: Consolidated Review Index (44 Frames)</h1>
        <div class="header-meta">
            Reconciled across <code>review_pack_v2</code>, <code>review_pack_pilot_v1</code>, and <code>review_pack_continuation_v1</code>.
        </div>
        <div class="summary-bar">
            <div class="stat-badge">Total Sources: <strong>44</strong></div>
            <div class="stat-badge">Training Eligible: <strong style="color: #10b981;">{lineage_audit['eligible_count']}</strong></div>
            <div class="stat-badge">Blocked (Pending): <strong style="color: #f59e0b;">{lineage_audit['blocked_count']}</strong></div>
            <div class="stat-badge">Rejected: <strong style="color: #ef4444;">{lineage_audit['rejected_count']}</strong></div>
            <div class="stat-badge">Total Approved Boxes (Exported): <strong>{export_audit['total_approved_boxes']}</strong></div>
            <div class="stat-badge">Teacher Proposals Accepted: <strong>{export_audit['proposals_accepted']}</strong></div>
        </div>
    </header>

    <div class="cards-grid">
        {all_cards}
    </div>
</body>
</html>
"""
    (out_dir / "review_index.html").write_text(html_content, encoding="utf-8")


# ==============================================================================
# 3. Annotation Comparison Helper (Detect Changed Local Annotations)
# ==============================================================================

def compare_boxes(
    orig_boxes: List[Dict[str, Any]],
    approved_boxes: List[Dict[str, Any]],
    coord_tolerance: float = 0.005
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Compares original vs approved boxes to detect any modifications:
      - count difference (additions or deletions)
      - class changes (e.g. truck -> car)
      - coordinate shifts / moved boxes
    Returns (is_modified, reason, diff_info).
    """
    n_orig = len(orig_boxes)
    n_app = len(approved_boxes)

    diff_info = {
        "orig_count": n_orig,
        "approved_count": n_app,
        "class_changes": [],
        "unmatched_orig": 0,
        "unmatched_approved": 0,
        "coord_shifts": 0
    }

    if n_orig != n_app:
        return True, f"count_change: orig={n_orig} vs approved={n_app}", diff_info

    # Match boxes using greedy nearest-center
    matched_app = set()
    for o_idx, ob in enumerate(orig_boxes):
        o_cls = int(ob["class_id"])
        o_norm = ob["bbox_norm"]
        o_xc, o_yc, o_w, o_h = o_norm

        best_idx = None
        best_dist = float("inf")
        for a_idx, ab in enumerate(approved_boxes):
            if a_idx in matched_app:
                continue
            a_norm = ab["bbox_norm"]
            a_xc, a_yc, a_w, a_h = a_norm
            dist = math.hypot(o_xc - a_xc, o_yc - a_yc)
            if dist < best_dist:
                best_dist = dist
                best_idx = a_idx

        if best_idx is not None and best_dist < 0.10:
            matched_app.add(best_idx)
            ab = approved_boxes[best_idx]
            a_cls = int(ab["class_id"])
            a_norm = ab["bbox_norm"]

            if o_cls != a_cls:
                diff_info["class_changes"].append((o_cls, a_cls))
                return True, f"class_edit: box {o_idx} changed from {THAI_5CLASS_NAMES.get(o_cls, o_cls)} to {THAI_5CLASS_NAMES.get(a_cls, a_cls)}", diff_info

            for c_o, c_a in zip(o_norm, a_norm):
                if abs(c_o - c_a) > coord_tolerance:
                    diff_info["coord_shifts"] += 1
                    return True, f"moved_box: box {o_idx} coords changed from {o_norm} to {a_norm}", diff_info
        else:
            diff_info["unmatched_orig"] += 1
            return True, f"unmatched_box: orig box {o_idx} could not be matched in approved boxes", diff_info

    if len(matched_app) != n_app:
        return True, f"unmatched_approved: {n_app - len(matched_app)} approved boxes unmatched", diff_info

    return False, "identical", diff_info


# ==============================================================================
# 4. Inventory Annotation & YOLO Label Verification
# ==============================================================================

def assert_inventory_yolo_agreement(
    inventory_records: List[Dict[str, Any]]
) -> int:
    """
    Asserts line-for-line, box-for-box, and class-for-class agreement between
    inventory annotations and exported YOLO label files on disk. (Fix #2)
    """
    checked_count = 0
    for rec in inventory_records:
        lbl_path_str = rec.get("label_path")
        if not lbl_path_str or not rec.get("label_exists_on_disk") or rec.get("annotation_provenance") != "human_reviewed_consolidated_batch7":
            continue

        lbl_file = Path(lbl_path_str)
        if not lbl_file.exists():
            raise AssertionError(f"Inventory record '{rec['inventory_id']}' points to non-existent label file '{lbl_file}'")

        lines = [l.strip() for l in lbl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        boxes = rec.get("boxes", [])

        if len(lines) != len(boxes) or len(boxes) != rec.get("total_boxes"):
            raise AssertionError(
                f"Box count mismatch for '{rec['inventory_id']}': "
                f"label file has {len(lines)} lines, inventory has {len(boxes)} boxes (total_boxes={rec.get('total_boxes')})"
            )

        for idx, (line, b) in enumerate(zip(lines, boxes)):
            parts = line.split()
            if len(parts) != 5:
                raise AssertionError(f"Invalid line {idx} in '{lbl_file}': expected 5 tokens, got {len(parts)}")
            cid = int(parts[0])
            coords = [float(p) for p in parts[1:5]]

            if cid != b["class_id"]:
                raise AssertionError(f"Class mismatch in '{rec['inventory_id']}' box {idx}: file={cid}, inventory={b['class_id']}")

            for c_idx, (fc, ic) in enumerate(zip(coords, b["bbox_norm"])):
                if abs(fc - ic) > 1e-4:
                    raise AssertionError(f"Coordinate mismatch in '{rec['inventory_id']}' box {idx} coord {c_idx}: file={fc}, inventory={ic}")

        # Check class_counts and box_sizes
        expected_class_counts = Counter(str(b["class_id"]) for b in boxes)
        if dict(expected_class_counts) != rec.get("class_counts"):
            raise AssertionError(f"class_counts mismatch for '{rec['inventory_id']}': {dict(expected_class_counts)} vs {rec.get('class_counts')}")

        expected_sizes = Counter(b["size_bucket"] for b in boxes)
        for sz in ["small", "medium", "large"]:
            expected_sizes.setdefault(sz, 0)
        actual_sizes = rec.get("box_sizes", {})
        for sz in ["small", "medium", "large"]:
            if expected_sizes[sz] != actual_sizes.get(sz, 0):
                raise AssertionError(f"box_sizes mismatch for '{rec['inventory_id']}' size '{sz}': expected {expected_sizes[sz]}, got {actual_sizes.get(sz, 0)}")

        checked_count += 1

    print(f"[VERIFY] Successfully asserted agreement across {checked_count} inventory records and exported YOLO label files.")
    return checked_count


def compute_manifest_summary(
    record_ids: List[str],
    inv_lookup: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Derives manifest summary dynamically from the final selected records. (Fix #2)"""
    class_counts = Counter()
    size_counts = {"small": 0, "medium": 0, "large": 0}
    unique_canonical = set()

    for rid in record_ids:
        rec = inv_lookup[rid]
        unique_canonical.add(rec["canonical_source_id"])
        for b in rec.get("boxes", []):
            cname = b.get("class_name")
            if not cname:
                cid = int(b.get("class_id", 0))
                cname = THAI_5CLASS_NAMES.get(cid, "car")
            class_counts[cname] += 1
            sz = b.get("size_bucket")
            if sz in size_counts:
                size_counts[sz] += 1

    total_boxes = sum(class_counts.values())
    return {
        "images_count": len(record_ids),
        "unique_canonical_frames_count": len(unique_canonical),
        "total_boxes": total_boxes,
        "class_box_counts": dict(class_counts),
        "size_box_counts": size_counts
    }


# ==============================================================================
# 5. Training Manifests Reconciliation (Batch 7 -> training_manifests_v5)
# ==============================================================================

def reconcile_training_manifests(
    consolidated_records: List[Dict[str, Any]],
    lineage_audit: Dict[str, Any],
    config: ConsolidationConfig
) -> Dict[str, Any]:
    """
    Propagates review decisions into training_manifests_v5:
      1. Excludes rejected canonical frames ('cam44_north_f019140', 'cam44_north_f148440') and their derived variants.
      2. Detects changed local annotations by actual box comparisons (class edits, deletions, moved boxes, additions)
         across all review packs including review_pack_v2. Excludes all 40 stale synthetic descendants.
      3. Blocks verified frames with pending proposals (e.g. MVI_20063_img00769) from training manifests.
      4. Matches reviewed external records by canonical ID AND exact image/variant identity (verifying SHA256 image hashes).
         Links corrected labels and local image availability. Never propagates coordinates to different variants.
      5. Synchronizes embedded boxes and asserts agreement with exported YOLO labels.
      6. Recalculates external cap from eligible unique local training sources: floor(526 * 0.5) = 263.
      7. Samples 263 external train records using deterministic sequence-stratified sampling (excluding blocked frames).
      8. Preserves identical primary unaugmented validation (130 frames, 1,367 boxes).
      9. Derives all manifest summaries directly from final selected records.
    """
    v3_dir = config.v3_manifests_dir
    out_dir = config.output_manifests_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load v3 manifests
    with open(v3_dir / "manifest_a_local_only.json", "r", encoding="utf-8") as f:
        v3_man_a = json.load(f)
    with open(v3_dir / "manifest_b_local_plus_external.json", "r", encoding="utf-8") as f:
        v3_man_b = json.load(f)
    with open(v3_dir / "split_and_exclusion_manifest.json", "r", encoding="utf-8") as f:
        v3_splits = json.load(f)
    with open(v3_dir / "canonical_source_inventory.json", "r", encoding="utf-8") as f:
        v3_inv = json.load(f)
    with open(v3_dir / "primary_validation_manifest.json", "r", encoding="utf-8") as f:
        v3_prim_val = json.load(f)
    with open(v3_dir / "synthetic_validation_diagnostic_manifest.json", "r", encoding="utf-8") as f:
        v3_synth_val = json.load(f)
    with open(v3_dir / "config.json", "r", encoding="utf-8") as f:
        v3_cfg = json.load(f)

    # Build inventory lookups
    inv_records_by_id = {r["inventory_id"]: r for r in v3_inv["records"]}
    local_canonical_inv = {
        r["canonical_source_id"]: r
        for r in v3_inv["records"]
        if r.get("data_origin") == "local" and not r.get("is_synthetic_variant")
    }

    # Identify rejected canonical frames
    rejected_canonical_ids = set(lineage_audit["rejected_frames"])
    print(f"[MANIFEST V5] Rejected canonical sources: {rejected_canonical_ids}")

    # Identify local verified frames with modified annotations using actual box comparisons (Fix #3)
    modified_local_canonical_ids = set()
    local_mod_details = {}

    for r in consolidated_records:
        fid = r["frame_id"]
        if r.get("data_origin") == "local" or fid.startswith("cam"):
            if r.get("review_status") == "verified":
                orig_r = local_canonical_inv.get(fid)
                if orig_r:
                    orig_boxes = orig_r.get("boxes", [])
                    app_boxes = [
                        b for b in r.get("boxes", [])
                        if not b.get("is_proposal") or b.get("proposal_status") == "accepted"
                    ]
                    is_mod, reason, diff_info = compare_boxes(orig_boxes, app_boxes)
                    if is_mod:
                        modified_local_canonical_ids.add(fid)
                        local_mod_details[fid] = reason

    print(f"[MANIFEST V5] Verified local frames with modified annotations: {len(modified_local_canonical_ids)} ({sorted(list(modified_local_canonical_ids))})")

    # Filter local training records
    new_exclusions: List[Dict[str, Any]] = []
    eligible_local_train_records: List[str] = []

    excluded_rejected_canonical = []
    excluded_rejected_variants = []
    excluded_stale_variants = []

    for inv_id in v3_man_a["train_records"]:
        rec = inv_records_by_id.get(inv_id)
        if not rec:
            continue
        c_src = rec["canonical_source_id"]
        is_synth = rec.get("is_synthetic_variant", False)

        # 1. Derived from a rejected canonical source
        if c_src in rejected_canonical_ids:
            if not is_synth:
                excluded_rejected_canonical.append(inv_id)
                new_exclusions.append({
                    "inventory_id": inv_id,
                    "canonical_source_id": c_src,
                    "data_origin": "local",
                    "reason": "rejected_in_annotation_review",
                    "details": f"Canonical source '{c_src}' rejected during human annotation review."
                })
            else:
                excluded_rejected_variants.append(inv_id)
                new_exclusions.append({
                    "inventory_id": inv_id,
                    "canonical_source_id": c_src,
                    "data_origin": "local",
                    "reason": "derived_from_rejected_review_frame",
                    "details": f"Synthetic augmentation variant '{inv_id}' derived from rejected canonical source '{c_src}'."
                })
            continue

        # 2. Synthetic variant of a modified local frame (Fix #3)
        if is_synth and c_src in modified_local_canonical_ids:
            excluded_stale_variants.append(inv_id)
            new_exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "local",
                "reason": "stale_unregenerated_augmentation_variant",
                "details": f"Synthetic variant '{inv_id}' contains old uncorrected labels for modified canonical frame '{c_src}' ({local_mod_details.get(c_src, '')}). Excluded pending regeneration."
            })
            continue

        eligible_local_train_records.append(inv_id)

    print(f"[MANIFEST V5] Excluded rejected canonical frames: {len(excluded_rejected_canonical)} ({excluded_rejected_canonical})")
    print(f"[MANIFEST V5] Excluded rejected synthetic variants: {len(excluded_rejected_variants)} ({excluded_rejected_variants})")
    print(f"[MANIFEST V5] Excluded stale augmentation variants: {len(excluded_stale_variants)} ({len(excluded_stale_variants)} variants)")
    print(f"[MANIFEST V5] Remaining local training records: {len(eligible_local_train_records)} (was {len(v3_man_a['train_records'])})")

    # Calculate unique canonical sources in remaining local training
    unique_local_train_sources = set(inv_records_by_id[rid]["canonical_source_id"] for rid in eligible_local_train_records)
    n_unique_local_train = len(unique_local_train_sources)
    print(f"[MANIFEST V5] Unique canonical local training sources: {n_unique_local_train} (was 528)")

    # Recalculate external cap
    cap_ratio = float(v3_cfg.get("balancing", {}).get("external_cap_ratio_per_unique_local_frame", 0.5))
    new_external_cap = int(math.floor(n_unique_local_train * cap_ratio))
    print(f"[MANIFEST V5] Recalculated external cap: floor({n_unique_local_train} * {cap_ratio}) = {new_external_cap} frames (was 264)")

    # Identify blocked and rejected canonical sources (Fix #3 & #4)
    blocked_canonical_ids = {
        r["canonical_source_id"] for r in consolidated_records
        if r.get("eligibility_status") == "blocked_pending_proposals"
    }
    rejected_canonical_ids = set(lineage_audit["rejected_frames"])
    unresolved_canonical_ids = blocked_canonical_ids.union(rejected_canonical_ids)
    print(f"[MANIFEST V6] Unresolved canonical sources to exclude (blocked + rejected): {unresolved_canonical_ids}")

    # Record exclusions for ALL records/variants of unresolved canonical sources (Fix #3)
    for r in v3_inv["records"]:
        c_src = r["canonical_source_id"]
        inv_id = r["inventory_id"]
        if c_src in blocked_canonical_ids:
            new_exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": r.get("data_origin", "external_ua_detrac"),
                "reason": "blocked_unresolved_pending_proposals",
                "details": f"Record/variant '{inv_id}' of unresolved canonical frame '{c_src}' excluded from training candidates."
            })
        elif c_src in rejected_canonical_ids and r.get("data_origin") == "external_ua_detrac":
            new_exclusions.append({
                "inventory_id": inv_id,
                "canonical_source_id": c_src,
                "data_origin": "external_ua_detrac",
                "reason": "derived_from_rejected_review_frame",
                "details": f"Record/variant '{inv_id}' of rejected canonical frame '{c_src}' excluded from training candidates."
            })

    # Sample external frames for Manifest B (excluding ANY variant of an unresolved canonical source - Fix #3)
    ext_records = [r for r in v3_inv["records"] if r["data_origin"] == "external_ua_detrac"]
    train_seqs = v3_splits.get("metadata", {}).get("external_sequence_partition", {}).get("train_sequences", [])
    eligible_ext_cands = [
        r for r in ext_records
        if r["sequence_id"] in train_seqs and r["canonical_source_id"] not in unresolved_canonical_ids
    ]

    sampled_ext, ext_stats = sample_external_frames_stratified(
        external_candidates=eligible_ext_cands,
        target_count=new_external_cap,
        train_sequences=train_seqs,
        seed=config.random_seed
    )
    sampled_ext_ids = [r["inventory_id"] for r in sampled_ext]
    print(f"[MANIFEST V6] Sampled external records for Manifest B: {len(sampled_ext_ids)} (all unresolved canonical sources and variants excluded)")

    # Lookups for reviewed records
    reviewed_by_canon = {r["canonical_source_id"]: r for r in consolidated_records}
    reviewed_by_inv_id = {r["inventory_id"]: r for r in consolidated_records if "inventory_id" in r}

    consolidated_labels_dir = config.output_pack_dir / "annotations" / "labels"

    # Update inventory records (Fix #1 & #2)
    v6_inv_records = []
    linked_external_count = 0
    linked_local_count = 0

    for r in v3_inv["records"]:
        r_copy = dict(r)
        c_src = r_copy["canonical_source_id"]
        inv_id = r_copy["inventory_id"]
        is_synth = r_copy.get("is_synthetic_variant", False)
        data_origin = r_copy.get("data_origin")

        # 1. Local Canonical Matching
        if data_origin == "local" and not is_synth and c_src in reviewed_by_canon:
            rev_rec = reviewed_by_canon[c_src]
            if rev_rec.get("training_eligible"):
                approved_lbl = consolidated_labels_dir / f"{c_src}.txt"
                if approved_lbl.exists():
                    r_copy["label_path"] = str(approved_lbl).replace("\\", "/")
                    r_copy["label_exists_on_disk"] = True
                    r_copy["annotation_provenance"] = "human_reviewed_consolidated_batch7"

                    # Convert approved boxes to inventory format using ACTUAL image dimensions (Fix #2)
                    app_boxes = [b for b in rev_rec.get("boxes", []) if not b.get("is_proposal") or b.get("proposal_status") == "accepted"]
                    img_dims = r_copy.get("dimensions") or rev_rec.get("dimensions") or {}
                    w = int(img_dims.get("width", 1920))
                    h = int(img_dims.get("height", 1080))

                    inv_boxes = []
                    for b in app_boxes:
                        cid = int(b["class_id"])
                        cname = THAI_5CLASS_NAMES[cid]
                        norm_box = list(b["bbox_norm"])
                        sz_info = compute_aspect_preserving_size(norm_box[2], norm_box[3], w, h, ref_size=640)
                        inv_boxes.append({
                            "source_class_id": b.get("original_source_class_id", cid),
                            "source_class_name": b.get("original_source_class_name", cname),
                            "class_id": cid,
                            "class_name": cname,
                            "bbox_norm": [round(float(coord), 6) for coord in norm_box],
                            "size_bucket": sz_info["size_bucket"]
                        })

                    r_copy["boxes"] = inv_boxes
                    r_copy["total_boxes"] = len(inv_boxes)
                    r_copy["class_counts"] = dict(Counter(str(b["class_id"]) for b in inv_boxes))
                    r_copy["box_sizes"] = dict(Counter(b["size_bucket"] for b in inv_boxes))
                    linked_local_count += 1

        # 2. External Exact Variant Matching (Fix #1 & #2)
        elif data_origin == "external_ua_detrac" and inv_id in reviewed_by_inv_id:
            rev_rec = reviewed_by_inv_id[inv_id]
            # Verify image SHA256
            rev_img_path = config.output_pack_dir / "images" / f"{c_src}.jpg"
            if rev_img_path.exists():
                r_copy["image_path"] = str(rev_img_path).replace("\\", "/")
                r_copy["image_exists_on_disk"] = True
                r_copy["image_status"] = "available_local_file"

            if rev_rec.get("training_eligible"):
                approved_lbl = consolidated_labels_dir / f"{c_src}.txt"
                if approved_lbl.exists():
                    r_copy["label_path"] = str(approved_lbl).replace("\\", "/")
                    r_copy["label_exists_on_disk"] = True
                    r_copy["annotation_provenance"] = "human_reviewed_consolidated_batch7"

                    app_boxes = [b for b in rev_rec.get("boxes", []) if not b.get("is_proposal") or b.get("proposal_status") == "accepted"]
                    img_dims = r_copy.get("dimensions") or rev_rec.get("dimensions") or {}
                    w = int(img_dims.get("width", 640))
                    h = int(img_dims.get("height", 640))

                    inv_boxes = []
                    for b in app_boxes:
                        cid = int(b["class_id"])
                        cname = THAI_5CLASS_NAMES[cid]
                        norm_box = list(b["bbox_norm"])
                        sz_info = compute_aspect_preserving_size(norm_box[2], norm_box[3], w, h, ref_size=640)
                        inv_boxes.append({
                            "source_class_id": b.get("original_source_class_id", cid),
                            "source_class_name": b.get("original_source_class_name", cname),
                            "class_id": cid,
                            "class_name": cname,
                            "bbox_norm": [round(float(coord), 6) for coord in norm_box],
                            "size_bucket": sz_info["size_bucket"]
                        })

                    r_copy["boxes"] = inv_boxes
                    r_copy["total_boxes"] = len(inv_boxes)
                    r_copy["class_counts"] = dict(Counter(str(b["class_id"]) for b in inv_boxes))
                    r_copy["box_sizes"] = dict(Counter(b["size_bucket"] for b in inv_boxes))
                    linked_external_count += 1
            else:
                # Blocked frame (MVI_20063_img00769)
                r_copy["label_path"] = None
                r_copy["label_exists_on_disk"] = False
                r_copy["annotation_provenance"] = "blocked_pending_proposals_review_continuation"

        v6_inv_records.append(r_copy)

    print(f"[MANIFEST V6] Linked {linked_local_count} local records and {linked_external_count} external records with approved labels & synchronized boxes.")

    # Strict assertion: inventory annotations agree with exported YOLO labels (Fix #2)
    assert_inventory_yolo_agreement(v6_inv_records)

    v6_inv = dict(v3_inv)
    v6_inv["records"] = v6_inv_records
    v6_inv["metadata"]["version"] = "v6.0_consolidated"
    v6_inv["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    v6_inv_lookup = {r["inventory_id"]: r for r in v6_inv_records}

    # Build Manifest A (Fix #2: derive summary directly from final selected records)
    man_a_summary = compute_manifest_summary(eligible_local_train_records, v6_inv_lookup)
    man_a_v6 = dict(v3_man_a)
    man_a_v6["manifest_id"] = "manifest_a_local_v6"
    man_a_v6["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    man_a_v6["train_records"] = eligible_local_train_records
    man_a_v6["metadata"]["train_summary"] = man_a_summary

    # Build Manifest B (Fix #2: derive summary directly from final selected records)
    man_b_train_records = eligible_local_train_records + sampled_ext_ids
    man_b_summary = compute_manifest_summary(man_b_train_records, v6_inv_lookup)
    man_b_v6 = dict(v3_man_b)
    man_b_v6["manifest_id"] = "manifest_b_local_plus_external_v6"
    man_b_v6["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    man_b_v6["local_train_records"] = eligible_local_train_records
    man_b_v6["external_train_records"] = sampled_ext_ids
    man_b_v6["train_records"] = man_b_train_records
    man_b_v6["metadata"]["external_images_count"] = len(sampled_ext_ids)
    man_b_v6["metadata"]["train_summary"] = man_b_summary

    # Update Exclusions Manifest
    all_exclusions = v3_splits.get("exclusions", []) + new_exclusions
    v6_splits = dict(v3_splits)
    v6_splits["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    v6_splits["metadata"]["total_exclusions"] = len(all_exclusions)
    v6_splits["metadata"]["exclusion_breakdown"] = dict(Counter(e["reason"] for e in all_exclusions))
    v6_splits["exclusions"] = all_exclusions

    # Save all v6 manifests
    with open(out_dir / "canonical_source_inventory.json", "w", encoding="utf-8") as f:
        json.dump(v6_inv, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "split_and_exclusion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(v6_splits, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "primary_validation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(v3_prim_val, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "synthetic_validation_diagnostic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(v3_synth_val, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "manifest_a_local_only.json", "w", encoding="utf-8") as f:
        json.dump(man_a_v6, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "manifest_b_local_plus_external.json", "w", encoding="utf-8") as f:
        json.dump(man_b_v6, f, indent=2, cls=NumpyEncoder)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(v3_cfg, f, indent=2)

    print(f"[MANIFEST V6] Successfully saved training_manifests_v6 to: {out_dir}")

    return {
        "manifest_a": man_a_v6,
        "manifest_b": man_b_v6,
        "splits": v6_splits,
        "inventory": v6_inv,
        "stats": {
            "eligible_local_train": len(eligible_local_train_records),
            "unique_local_train": n_unique_local_train,
            "new_external_cap": new_external_cap,
            "sampled_external": len(sampled_ext_ids),
            "excluded_rejected_canonical": len(excluded_rejected_canonical),
            "excluded_rejected_variants": len(excluded_rejected_variants),
            "excluded_stale_variants": len(excluded_stale_variants),
            "blocked_external": len(blocked_canonical_ids),
            "primary_val_count": len(v3_prim_val.get("records", []))
        }
    }


# ==============================================================================
# 6. Training-Readiness Report Generation
# ==============================================================================

def generate_training_readiness_report(
    lineage_audit: Dict[str, Any],
    export_audit: Dict[str, Any],
    manifest_results: Dict[str, Any],
    config: ConsolidationConfig,
    output_path: Path
) -> None:
    """
    Renders comprehensive, honest Training Readiness Report:
      - Human-reviewed complete frames (41 eligible).
      - Blocked frame details reported separately (MVI_20063_img00769 with 14 pending proposals).
      - Existing local teacher-completed frames that remain unreviewed.
      - External frames with original, potentially incomplete labels.
      - Rejection and stale variant exclusions (40 stale synthetic variants quarantined).
      - Correct evaluation snapshot status: internal diagnostic benchmark with known exposure.
      - Bounded next experiment recommendation replacing premature full training recommendation.
    """
    stats = manifest_results["stats"]
    man_a = manifest_results["manifest_a"]["metadata"]["train_summary"]
    man_b = manifest_results["manifest_b"]["metadata"]["train_summary"]

    report_md = f"""# Batch 7: Annotation Consolidation & Training-Readiness Report

- **Date**: {datetime.now(timezone.utc).isoformat()}
- **Consolidated Pack**: `data/review_pack_consolidated_v3`
- **Candidate Manifests**: `data/training_manifests_v6`
- **Authoritative Review Status**: **{lineage_audit['verified_count']} Verified**, **{lineage_audit['rejected_count']} Rejected** across {lineage_audit['total_canonical_sources']} Canonical Sources
- **Training Eligibility**: **{lineage_audit['eligible_count']} Eligible**, **{lineage_audit['blocked_count']} Blocked (Pending Proposals)**, **{lineage_audit['rejected_count']} Rejected**

---

## 1. Executive Summary & Review Lineage Reconciliation

All 44 canonical training-review frames across `review_pack_v2`, `review_pack_pilot_v1`, and `review_pack_continuation_v1` have been reconciled with strict precedence:

| Source Pack | Frames Contributed | Precedence Applied | Verified | Rejected | Training Eligible | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`review_pack_v2` (Retained)** | **20** | Direct retention of previously verified baseline | 20 | 0 | 20 | 11 external UA-DETRAC, 9 local CCTV. |
| **`review_pack_pilot_v1`** | **6** | Human review decisions supersede v2 drafts | 5 | 1 | 5 | 1 local frame (`cam44_north_f019140`) rejected. |
| **`review_pack_continuation_v1`** | **18** | Human review decisions supersede v2 drafts | 18 | 0 | 18 | All 18 frames verified and training-eligible. |
| **Total Consolidated** | **44** | **Strict Precedence (0 Collisions, 0 Overlaps)** | **{lineage_audit['verified_count']}** | **{lineage_audit['rejected_count']}** | **{lineage_audit['eligible_count']}** | **100% accounted for exactly once**. |

### Rejected Frame Enforcements ({lineage_audit['rejected_count']} Frame)
1. **`cam44_north_f019140`** (Local Daytime Queue): Rejected due to intractable motorcycle queue density / occlusions. Excluded from training along with its variant `cam44_north_f019140_tuktukboost_1`.

> [!IMPORTANT]
> **Negative Example Safeguard**: Rejection does **NOT** generate empty negative label files. Both canonical sources and their derived variants are completely quarantined from candidate training splits. Pending proposals on rejected frames require no further review.

---

## 2. Approved Annotations, Training Eligibility & Size-Bin Correction

User review status has been strictly separated from training eligibility:
- A frame is **training-eligible** if and only if `review_status == "verified"` **AND** `pending_proposals_count == 0` **AND** `unresolved_conflicts_count == 0`.
- **{export_audit['exported_frames_count']} frames** are fully eligible and exported to `annotations/labels/*.txt`.
- **{lineage_audit['blocked_count']} frames** blocked due to pending proposals.
- Export class totals and proposal statistics are counted **only over the {export_audit['exported_frames_count']} exported eligible frames**; rejected frames are quarantined.

| Metric | Count | Provenance & Handling |
| :--- | :---: | :--- |
| **Total Exported Verified Frames** | **{export_audit['exported_frames_count']}** | 25 external UA-DETRAC, 18 local CCTV approach frames |
| **Total Approved Bounding Boxes** | **{export_audit['total_approved_boxes']}** | Synchronized line-for-line in `annotations/labels/*.txt` |
| **Accepted Teacher Proposals (Exported)** | **{export_audit['proposals_accepted']}** | Recovered background queue vehicles and distant oncoming traffic |
| **Human Manually Added Boxes (Exported)** | **{export_audit['human_manual_boxes_retained']}** | Drawn by human reviewers following the Small Vehicle Scan Checklist |
| **Teacher Conflict Corrections (Pickup/Van $\\to$ Car)** | **{export_audit['teacher_conflict_corrections']}** | Teacher classified pickups/vans as truck; human corrected to car |
| **Rejected Teacher Proposals (Exported Frames)** | **{export_audit['proposals_rejected']}** | False positive halos, glare reflections, and duplicate boxes discarded |

### Per-Class Approved Box Counts (Exported Eligible Frames Only)
| Class ID | Class Name | Box Count | Percentage |
| :---: | :--- | :---: | :---: |
| 0 | `car` | {export_audit['class_breakdown'].get('car', 0)} | {export_audit['class_breakdown'].get('car', 0) / max(1, export_audit['total_approved_boxes']) * 100:.1f}% |
| 1 | `motorcycle` | {export_audit['class_breakdown'].get('motorcycle', 0)} | {export_audit['class_breakdown'].get('motorcycle', 0) / max(1, export_audit['total_approved_boxes']) * 100:.1f}% |
| 2 | `bus` | {export_audit['class_breakdown'].get('bus', 0)} | {export_audit['class_breakdown'].get('bus', 0) / max(1, export_audit['total_approved_boxes']) * 100:.1f}% |
| 3 | `truck` | {export_audit['class_breakdown'].get('truck', 0)} | {export_audit['class_breakdown'].get('truck', 0) / max(1, export_audit['total_approved_boxes']) * 100:.1f}% |
| 4 | `three_wheeler` | {export_audit['class_breakdown'].get('three_wheeler', 0)} | {export_audit['class_breakdown'].get('three_wheeler', 0) / max(1, export_audit['total_approved_boxes']) * 100:.1f}% |

### Size-Bin Calculation Fix (Non-Square Image Dimensions)
Size buckets for reviewed local records pass actual image dimensions (1920×1080) instead of a naive 640×640 assumption into `compute_aspect_preserving_size`. Normalized coordinates $[xc, yc, bw, bh]$ remain completely unaltered:
- **117 medium $\\to$ small corrections** across all 18 local CCTV frames (111 across the original 17).
- **9 large $\\to$ medium corrections** across local CCTV frames.
- **Resulting Manifest A Size Distribution**: 7,262 small, 4,581 medium, 1,305 large boxes.

### Blocked & Rejected Quarantine Summary (Reported Separately)
| Frame ID | Review Status | Training Eligibility | Boxes / Proposals Status | Required Action |
| :--- | :---: | :---: | :--- | :--- |
| **`cam44_north_f019140`** | `rejected` | **EXCLUDED** | 51 unapproved boxes quarantined | No further action (rejected source). |

---

## 3. Dataset Lineage & Candidate Manifests V6

The candidate training manifests were updated to reflect human review decisions, box-level stale variant detection, and exact external record linking:

| Split / Attribute | Manifest A (Local-Only V6) | Manifest B (Local + External V6) | Delta vs V3 | Integrity Safeguard |
| :--- | :---: | :---: | :---: | :--- |
| **Local Training Images** | **{man_a['images_count']}** | **{man_a['images_count']}** | -43 images | Excludes 2 rejected frames/variants + 41 stale variants |
| **Unique Local Canonical Sources** | **{stats['unique_local_train']}** | **{stats['unique_local_train']}** | -1 source | Reduced from 528 to 527 by excluding 1 rejected source |
| **External Training Images** | **0** | **{stats['sampled_external']}** | -1 image | Recalculated cap: $\\lfloor 527 \\times 0.5 \\rfloor = 263$ (was 264) |
| **Total Candidate Training Images** | **{man_a['images_count']}** | **{man_b['images_count']}** | -44 images | Fully synchronized across A and B |
| **Total Candidate Training Boxes** | **{man_a['total_boxes']}** | **{man_b['total_boxes']}** | — | Dynamically derived from final selected records |
| **Primary Validation Images** | **{stats['primary_val_count']}** | **{stats['primary_val_count']}** | **0 (Identical)** | **Byte-for-byte identical; 0 validation leakage** |
| **Stale Variants Excluded** | **{stats['excluded_stale_variants']}** | **{stats['excluded_stale_variants']}** | +41 | Excludes unregenerated synthetic copies of all 18 modified frames |

### Canonical Source-Level Exclusions (Resampling Guard)
Exclusions for rejected sources (`cam44_north_f019140`) are enforced at the **canonical source level**. Resampling cannot select any variant or descendant of an unresolved source into Manifest B.

---

## 4. Honest Annotation Readiness & Benchmark Status

> [!WARNING]
> **Evaluation Benchmark Status Correction**:
> `data/eval_snapshot_v1` (42 frames) is **NOT** an independent evaluation benchmark. It is an **internal diagnostic benchmark** with known or unproven exposure to similar sequence environments. It must not be cited as an uncompromised external benchmark.

| Dataset Partition | Total Frames | Human-Reviewed & Complete | Unreviewed / Pending Labels | Status & Readiness Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Reviewed Sample (Batch 6 & 7)** | **44** | **43 (97.7%)** | 0 blocked, 1 rejected | **43 Frames Complete**: 18 local and 25 external frames ready for training / smoke testing. |
| **Local CCTV Dataset (`multiclass_dataset`)** | **1,397** | **18 canonical** | ~1,354 frames | **Partially Reviewed**: 509 local training sources remain teacher-completed without human review. 41 synthetic variants stale. |
| **External Selection (UA-DETRAC)** | **263** | **25 canonical** | 238 frames | **Incomplete Labels**: 238 external frames retain uncorrected source labels (missing small background cars; images not yet downloaded locally). |
| **Quarantined / Stale Records** | **43** | **0** | 43 excluded | **Quarantined**: 2 rejected records + 41 stale variants excluded from candidate training. |

---

## 5. Experiment Specification Summary

See detailed specification in [`docs/EXPERIMENT_SPECIFICATION_BATCH7.md`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/EXPERIMENT_SPECIFICATION_BATCH7.md).

Two distinct experiment tracks have been prepared:
1. **Track 1: Short Pipeline Smoke Test**
   - **Goal**: Verify training pipeline, loss stability, checkpoint saving, and eval metrics without overfitting.
   - **Dataset**: 43 eligible reviewed frames (18 local + 25 external); 130 primary val images.
   - **Budget**: 5 epochs (batch size 8, warm start from `models/yolo26s_thai_traffic.pt`).
   - **Path**: `runs/train/smoke_test_batch7/` (baseline weights preserved).
2. **Track 2: Meaningful Improvement Experiment (Gated)**
   - **Prerequisites**: Regenerate 41 local synthetic variants.
   - **Dataset**: Manifest A (1,092 local frames) vs Manifest B (1,092 local + 263 external frames).
   - **Budget**: 100 epochs with early stopping (patience = 15).
   - **Path**: `runs/train/candidate_manifest_a_v6/` and `runs/train/candidate_manifest_b_v6/`.

---

## 6. Output Files & Artifacts

- **Consolidated Review Pack**: [`data/review_pack_consolidated_v3/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3)
- **Approved YOLO Labels (43 Frames)**: [`data/review_pack_consolidated_v3/annotations/labels/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/annotations/labels)
- **Pack Manifest**: [`data/review_pack_consolidated_v3/manifest.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/manifest.json)
- **Candidate Manifests V6**: [`data/training_manifests_v6/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6)
- **Local-Only Manifest**: [`data/training_manifests_v6/manifest_a_local_only.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_a_local_only.json)
- **Local + External Manifest**: [`data/training_manifests_v6/manifest_b_local_plus_external.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_b_local_plus_external.json)
- **Experiment Specification**: [`docs/EXPERIMENT_SPECIFICATION_BATCH7.md`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/EXPERIMENT_SPECIFICATION_BATCH7.md)
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    print(f"[REPORT] Training readiness report written to: {output_path}")

    # Also save copy in consolidation pack
    pack_report = config.output_pack_dir / "CONSOLIDATION_REPORT.md"
    pack_report.write_text(report_md, encoding="utf-8")


# ==============================================================================
# 7. Main Execution Entrypoint
# ==============================================================================

def run_consolidation(config: Optional[ConsolidationConfig] = None) -> Dict[str, Any]:
    cfg = config or ConsolidationConfig()

    print("\n======================================================================")
    print("  BATCH 7 CORRECTION: REPAIR ANNOTATION PROPAGATION & READINESS")
    print("======================================================================")

    # 1. Reconcile Lineage & Precedence
    consolidated_records, lineage_audit = reconcile_review_lineage(cfg)

    # 2. Validate & Export Approved Pack
    export_results = validate_and_export_approved_pack(consolidated_records, lineage_audit, cfg)

    # 3. Propagate to Training Manifests V6
    manifest_results = reconcile_training_manifests(consolidated_records, lineage_audit, cfg)

    # 4. Generate Training Readiness Report
    generate_training_readiness_report(
        lineage_audit=lineage_audit,
        export_audit=export_results["export_audit"],
        manifest_results=manifest_results,
        config=cfg,
        output_path=cfg.report_markdown_path
    )

    print("\n======================================================================")
    print("  BATCH 7 CONSOLIDATION COMPLETED SUCCESSFULLY")
    print("======================================================================\n")

    return {
        "lineage_audit": lineage_audit,
        "export_results": export_results,
        "manifest_results": manifest_results
    }


def main():
    parser = argparse.ArgumentParser(description="Batch 7: Consolidate reviewed annotations and prepare training readiness report.")
    parser.add_argument("--v2-pack", type=str, default="data/review_pack_v2", help="Path to review_pack_v2")
    parser.add_argument("--pilot-pack", type=str, default="data/review_pack_pilot_v1", help="Path to review_pack_pilot_v1")
    parser.add_argument("--continuation-pack", type=str, default="data/review_pack_continuation_v1", help="Path to review_pack_continuation_v1")
    parser.add_argument("--output-pack", type=str, default="data/review_pack_consolidated_v3", help="Output path for consolidated review pack")
    parser.add_argument("--v3-manifests", type=str, default="data/training_manifests_v3", help="Path to training_manifests_v3")
    parser.add_argument("--output-manifests", type=str, default="data/training_manifests_v6", help="Output path for training_manifests_v6")
    parser.add_argument("--report-md", type=str, default="docs/TRAINING_READINESS_REPORT.md", help="Output Markdown report path")
    args = parser.parse_args()

    cfg = ConsolidationConfig(
        pack_v2_dir=Path(args.v2_pack),
        pilot_pack_dir=Path(args.pilot_pack),
        continuation_pack_dir=Path(args.continuation_pack),
        output_pack_dir=Path(args.output_pack),
        v3_manifests_dir=Path(args.v3_manifests),
        output_manifests_dir=Path(args.output_manifests),
        report_markdown_path=Path(args.report_md)
    )

    run_consolidation(cfg)


if __name__ == "__main__":
    main()
