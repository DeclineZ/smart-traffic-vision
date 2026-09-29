"""
Smoke test exercising the Batch 2B visual annotation editor workflow
on a disposable copy of the 42-frame review pack.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_review_pack import compute_file_sha256
from tools.annotation_editor import (
    create_editor_server,
    image_box_to_normalized_yolo,
    image_to_screen_coords,
    normalized_yolo_to_image_box,
    screen_to_image_coords,
)


def run_smoke_test():
    print("=" * 70)
    print("  RUNNING BATCH 2B VISUAL ANNOTATION EDITOR SMOKE TEST")
    print("=" * 70)

    real_pack_dir = REPO_ROOT / "data" / "review_pack_v1"
    real_annos_file = real_pack_dir / "annotations" / "annotations.json"
    real_sha_before = compute_file_sha256(real_annos_file)
    print(f"[CHECK] Real pack annotations.json SHA256: {real_sha_before}")

    disposable_dir = REPO_ROOT / "data" / "smoke_disposable_pack"
    if disposable_dir.exists():
        shutil.rmtree(disposable_dir)
    shutil.copytree(real_pack_dir, disposable_dir)
    print(f"[STEP 1] Created disposable copy at {disposable_dir.name}")

    # Find free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = create_editor_server(disposable_dir, host="127.0.0.1", port=port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[STEP 2] Launched local editor server on {base_url}")

    try:
        # 1. Check UI HTML loads
        req_ui = urllib.request.urlopen(f"{base_url}/")
        assert req_ui.status == 200
        html = req_ui.read().decode("utf-8")
        assert "YOLO26s Review Pack Visual Annotation Editor" in html
        print("[STEP 3] Verified UI root endpoint (GET /) serves visual editor HTML")

        # 2. Check all 42 frames accessible
        req_frames = urllib.request.urlopen(f"{base_url}/api/frames")
        assert req_frames.status == 200
        frames_data = json.loads(req_frames.read().decode("utf-8"))
        assert frames_data["summary"]["total"] == 42
        assert len(frames_data["frames"]) == 42
        print(f"[STEP 4] Verified all 42 frames accessible via API (Total: {frames_data['summary']['total']})")

        # 3. Load frame cam03_east_f003720
        target_fid = "cam03_east_f003720"
        req_frame = urllib.request.urlopen(f"{base_url}/api/frame/{target_fid}")
        assert req_frame.status == 200
        frame_resp = json.loads(req_frame.read().decode("utf-8"))
        record = frame_resp["record"]
        initial_etag = frame_resp["base_etag"]
        boxes = list(record["boxes"])
        initial_box_count = len(boxes)
        img_w, img_h = record["dimensions"]
        print(f"[STEP 5] Loaded target frame '{target_fid}': {initial_box_count} boxes, ETag {initial_etag[:10]}")

        # 4. Simulate visual interaction:
        # A) Zoom & Pan: zoom=3.0, pan_x=120, pan_y=-50
        zoom = 3.0
        pan_x, pan_y = 120.0, -50.0
        b0 = boxes[0]
        # Convert box 0 to screen coordinates while zoomed
        bx1, by1, bx2, by2 = normalized_yolo_to_image_box(b0["bbox_norm"], img_w, img_h)
        s_x1, s_y1 = image_to_screen_coords(bx1, by1, pan_x, pan_y, zoom)
        s_x2, s_y2 = image_to_screen_coords(bx2, by2, pan_x, pan_y, zoom)
        # Move on screen by (+30px, -20px) and expand width by +15px
        s_x1_new = s_x1 + 30
        s_y1_new = s_y1 - 20
        s_x2_new = s_x2 + 45
        s_y2_new = s_y2 - 20
        # Convert back from screen to image coordinates while zoomed
        new_ix1, new_iy1 = screen_to_image_coords(s_x1_new, s_y1_new, pan_x, pan_y, zoom, img_w, img_h)
        new_ix2, new_iy2 = screen_to_image_coords(s_x2_new, s_y2_new, pan_x, pan_y, zoom, img_w, img_h)
        new_norm_box = image_box_to_normalized_yolo(new_ix1, new_iy1, new_ix2, new_iy2, img_w, img_h)

        # Apply edits to box 0:
        b0_edited = dict(b0)
        b0_edited["bbox_norm"] = new_norm_box
        b0_edited["class_id"] = 3
        b0_edited["class_name"] = "truck"
        b0_edited["subtype"] = "truck_based_songthaew"
        b0_edited["is_ambiguous"] = True
        b0_edited["ambiguity_reason"] = "commercial chassis songthaew conversion"

        # B) Delete box 1
        deleted_box_id = boxes[1]["instance_id"]

        # C) Retain remaining boxes (boxes[2:])
        surviving_boxes = [dict(b) for b in boxes[2:]]

        # D) Add new box (pickup)
        new_box = {
            "instance_id": f"{target_fid}_inst_999",
            "class_id": 0,
            "class_name": "car",
            "subtype": "pickup",
            "is_ambiguous": False,
            "ambiguity_reason": "",
            "bbox_norm": [0.300000, 0.400000, 0.080000, 0.120000],
            "proposal_source": "manual_annotation"
        }

        edited_box_list = [b0_edited] + surviving_boxes + [new_box]
        expected_count = len(edited_box_list)

        # 5. Save Draft
        save_payload = json.dumps({
            "base_etag": initial_etag,
            "action": "save_draft",
            "boxes": edited_box_list,
            "reviewer_notes": "Smoke test: converted b0 to truck_based_songthaew, deleted b1, added pickup",
            "is_ambiguous": False
        }).encode("utf-8")

        req_save = urllib.request.Request(
            f"{base_url}/api/frame/{target_fid}/save",
            data=save_payload,
            headers={"Content-Type": "application/json"}
        )
        resp_save = urllib.request.urlopen(req_save)
        assert resp_save.status == 200
        save_res = json.loads(resp_save.read().decode("utf-8"))
        draft_etag = save_res["base_etag"]
        print(f"[STEP 6] Saved draft successfully. Status: {save_res['record']['review_status']}, New ETag: {draft_etag[:10]}")

        # 6. Reload and confirm JSON, YOLO labels, and previews agree
        req_reload = urllib.request.urlopen(f"{base_url}/api/frame/{target_fid}")
        reloaded = json.loads(req_reload.read().decode("utf-8"))
        re_rec = reloaded["record"]

        # Check JSON
        assert len(re_rec["boxes"]) == expected_count, f"Expected {expected_count} boxes, got {len(re_rec['boxes'])}"
        assert re_rec["review_status"] == "draft"
        # Check b0 metadata
        re_b0 = next(b for b in re_rec["boxes"] if b["instance_id"] == b0["instance_id"])
        assert re_b0["class_id"] == 3
        assert re_b0["class_name"] == "truck"
        assert re_b0["subtype"] == "truck_based_songthaew"
        assert re_b0["is_ambiguous"] is True
        # Check deleted box is gone
        assert not any(b["instance_id"] == deleted_box_id for b in re_rec["boxes"])
        # Check new box exists
        re_new = next(b for b in re_rec["boxes"] if b["instance_id"] == new_box["instance_id"])
        assert re_new["subtype"] == "pickup"

        # Check YOLO labels on disk
        label_file = disposable_dir / "annotations" / "labels" / f"{target_fid}.txt"
        assert label_file.exists(), f"Label file missing: {label_file}"
        label_lines = [l.strip() for l in label_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(label_lines) == expected_count, f"Label line count mismatch: {len(label_lines)} vs {expected_count}"

        # Check Preview image on disk
        preview_file = disposable_dir / "previews" / f"{target_fid}_preview.jpg"
        assert preview_file.exists(), f"Preview file missing: {preview_file}"
        assert preview_file.stat().st_size > 10000, "Preview file suspiciously small"
        print(f"[STEP 7] Confirmed annotations.json ({expected_count} boxes), labels ({len(label_lines)} lines), and preview image agree!")

        # 7. Explicitly mark verified
        ver_payload = json.dumps({
            "base_etag": draft_etag,
            "action": "mark_verified",
            "boxes": re_rec["boxes"],
            "reviewer_notes": "Reviewed and verified",
            "is_ambiguous": False
        }).encode("utf-8")
        req_ver = urllib.request.Request(
            f"{base_url}/api/frame/{target_fid}/save",
            data=ver_payload,
            headers={"Content-Type": "application/json"}
        )
        resp_ver = urllib.request.urlopen(req_ver)
        ver_res = json.loads(resp_ver.read().decode("utf-8"))
        assert ver_res["record"]["review_status"] == "verified"
        assert ver_res["record"]["annotation_state"] == "annotated"
        ver_etag = ver_res["base_etag"]
        print(f"[STEP 8] Verified explicit 'mark_verified': review_status={ver_res['record']['review_status']}")

        # 8. Re-edit verified frame with save_draft -> resets to draft!
        re_edit_payload = json.dumps({
            "base_etag": ver_etag,
            "action": "save_draft",
            "boxes": ver_res["record"]["boxes"],
            "reviewer_notes": "Modified note on verified frame",
            "is_ambiguous": False
        }).encode("utf-8")
        req_re_edit = urllib.request.Request(
            f"{base_url}/api/frame/{target_fid}/save",
            data=re_edit_payload,
            headers={"Content-Type": "application/json"}
        )
        resp_re_edit = urllib.request.urlopen(req_re_edit)
        re_edit_res = json.loads(resp_re_edit.read().decode("utf-8"))
        assert re_edit_res["record"]["review_status"] == "draft", "Editing verified frame must reset to draft!"
        print(f"[STEP 9] Verified editing a verified frame resets review_status to draft")

        # 9. Verify empty frame handling (empty frame verified empty background)
        unann_rec = next((f for f in frames_data["frames"] if f.get("is_unannotated") is True), None)
        if not unann_rec:
            # All frames in review_pack_v1 are now human reviewed. Test empty background on a disposable frame.
            unann_fid = frames_data["frames"][1]["frame_id"]
        else:
            unann_fid = unann_rec["frame_id"]

        req_unann = urllib.request.urlopen(f"{base_url}/api/frame/{unann_fid}")
        unann_resp = json.loads(req_unann.read().decode("utf-8"))
        unann_etag = unann_resp["base_etag"]

        unann_ver_payload = json.dumps({
            "base_etag": unann_etag,
            "action": "mark_verified",
            "boxes": [],
            "reviewer_notes": "Human verified empty background",
            "is_ambiguous": False
        }).encode("utf-8")
        req_unann_ver = urllib.request.Request(
            f"{base_url}/api/frame/{unann_fid}/save",
            data=unann_ver_payload,
            headers={"Content-Type": "application/json"}
        )
        resp_unann_ver = urllib.request.urlopen(req_unann_ver)
        unann_ver_res = json.loads(resp_unann_ver.read().decode("utf-8"))
        assert unann_ver_res["record"]["review_status"] == "verified"
        assert unann_ver_res["record"]["annotation_state"] == "verified_empty_background"
        assert unann_ver_res["record"]["is_unannotated"] is False
        print(f"[STEP 10] Verified empty frame explicitly marked verified -> verified_empty_background")

        print("\n" + "=" * 70)
        print("  SMOKE TEST PASSED ALL CHECKS!")
        print("=" * 70)

    finally:
        server.shutdown()
        server.server_close()
        if disposable_dir.exists():
            shutil.rmtree(disposable_dir)
        print(f"[CLEANUP] Removed disposable pack directory {disposable_dir.name}")

    # Final assertion: real pack was NEVER touched
    real_sha_after = compute_file_sha256(real_annos_file)
    assert real_sha_before == real_sha_after, "CRITICAL: Real pack was modified!"
    print(f"[CONFIRM] Real pack annotations.json SHA256 untouched: {real_sha_after}\n")


if __name__ == "__main__":
    run_smoke_test()
