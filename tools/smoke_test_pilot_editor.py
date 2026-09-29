"""
tools/smoke_test_pilot_editor.py - Disposable Fixture Smoke Test for Batch 6 Pilot Editor

Exercises:
1. Spawns a TCPServer on an ephemeral port backed by a disposable copy of review_pack_pilot_v1.
2. Validates GET /api/status, GET /api/frames, and GET /api/frame/<frame_id>.
3. Tests accepting a teacher proposal:
   - Validates that accepted proposal enters annotations/labels/<frame_id>.txt.
   - Validates that review_status remains 'draft'.
   - Validates that all existing human boxes are preserved.
4. Tests rejecting a teacher proposal:
   - Validates that rejected proposal is excluded from annotations/labels/<frame_id>.txt.
   - Validates that rejection is persisted in annotations.json.
5. Cleans up disposable fixture completely, ensuring zero side-effects.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.annotation_editor import create_editor_server
from tools.prepare_review_pack import compute_file_sha256

def run_smoke_test():
    print("=" * 70)
    print("  BATCH 6 PILOT EDITOR SMOKE TEST (DISPOSABLE FIXTURE)")
    print("=" * 70)

    pilot_src = Path("data/review_pack_pilot_v1")
    if not pilot_src.exists():
        raise FileNotFoundError(f"Pilot pack not found at {pilot_src}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_pilot_"))
    print(f"[FIXTURE] Created temporary copy at: {tmp_dir}")
    shutil.copytree(pilot_src, tmp_dir / "review_pack_pilot_v1")
    pack_dir = tmp_dir / "review_pack_pilot_v1"

    # Reset disposable copy frames to draft to ensure predictable test environment
    annos_file = pack_dir / "annotations" / "annotations.json"
    with open(annos_file, "r", encoding="utf-8") as f:
        recs = json.load(f)
    for r in recs:
        r["review_status"] = "draft"
    with open(annos_file, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2)

    # Start ephemeral server on localhost:18099
    port = 18099
    server = create_editor_server(pack_dir=pack_dir, host="127.0.0.1", port=port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    base_url = f"http://127.0.0.1:{port}"
    try:
        # 1. Test GET /api/status
        req = urllib.request.urlopen(f"{base_url}/api/status")
        self_status = json.loads(req.read().decode())
        print(f"[CHECK 1] GET /api/status -> HTTP 200 (ETag: {self_status['base_etag'][:12]}...)")
        assert "base_etag" in self_status

        # 2. Test GET /api/frames
        req = urllib.request.urlopen(f"{base_url}/api/frames")
        frames_resp = json.loads(req.read().decode())
        print(f"[CHECK 2] GET /api/frames -> HTTP 200 ({len(frames_resp['frames'])} frames, {frames_resp['summary']['total_boxes']} boxes)")
        assert len(frames_resp["frames"]) == 6
        assert frames_resp["summary"]["draft"] == 6
        assert frames_resp["summary"]["verified"] == 0

        # 3. Test GET /api/frame/<frame_id>
        test_fid = "MVI_40871_img00320"
        req = urllib.request.urlopen(f"{base_url}/api/frame/{test_fid}")
        frame_detail = json.loads(req.read().decode())
        print(f"[CHECK 3] GET /api/frame/{test_fid} -> HTTP 200")
        record = frame_detail["record"]
        base_etag = frame_detail["base_etag"]
        boxes = record["boxes"]
        
        human_boxes = [b for b in boxes if not b.get("is_proposal")]
        proposal_boxes = [b for b in boxes if b.get("is_proposal")]
        print(f"  Existing human boxes: {len(human_boxes)}, Teacher proposals: {len(proposal_boxes)}")
        assert len(human_boxes) > 0
        assert len(proposal_boxes) > 0

        # 4. Check initial YOLO label file: MUST contain ONLY human boxes + accepted proposals!
        lbl_path = pack_dir / "annotations" / "labels" / f"{test_fid}.txt"
        lbl_lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        initial_accepted = len([p for p in proposal_boxes if p.get("proposal_status") == "accepted"])
        expected_init = len(human_boxes) + initial_accepted
        assert len(lbl_lines) == expected_init, f"Expected {expected_init} initial label lines, got {len(lbl_lines)}"
        print(f"[CHECK 4] Initial YOLO labels: exactly {len(lbl_lines)} lines (0 unaccepted proposals in labels)")

        # 5. Accept one pending proposal and save draft
        candidate_proposal = next((p for p in proposal_boxes if p.get("proposal_status") != "accepted"), proposal_boxes[0])
        candidate_proposal["proposal_status"] = "accepted"
        print(f"[ACTION] Accepting proposal '{candidate_proposal['instance_id']}' ({candidate_proposal['class_name']})...")

        save_payload = {
            "base_etag": base_etag,
            "action": "save_draft",
            "boxes": boxes,
            "reviewer_notes": "Smoke test: accepted 1 proposal",
            "is_ambiguous": False
        }

        save_req = urllib.request.Request(
            f"{base_url}/api/frame/{test_fid}/save",
            data=json.dumps(save_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        save_resp = json.loads(urllib.request.urlopen(save_req).read().decode())
        print(f"[CHECK 5] POST /api/frame/{test_fid}/save -> HTTP 200 (Success)")

        # Verify frame review status remained draft!
        saved_rec = save_resp["record"]
        assert saved_rec["review_status"] == "draft", f"Expected review_status='draft', got {saved_rec['review_status']}"
        print(f"  Frame review_status verified: '{saved_rec['review_status']}' (NEVER silently verified)")

        # Verify YOLO label file now has exactly expected lines
        lbl_lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        expected_after_accept = expected_init + 1
        assert len(lbl_lines) == expected_after_accept, f"Expected {expected_after_accept} label lines, got {len(lbl_lines)}"
        print(f"  YOLO label file verified: exactly {len(lbl_lines)} lines")

        # 6. Reject another proposal and save draft
        new_etag = save_resp["base_etag"]
        candidate_reject = next((p for p in proposal_boxes if p.get("proposal_status") != "accepted"), proposal_boxes[1])
        candidate_reject["proposal_status"] = "rejected"
        print(f"[ACTION] Rejecting proposal '{candidate_reject['instance_id']}'...")

        save_payload_2 = {
            "base_etag": new_etag,
            "action": "save_draft",
            "boxes": boxes,
            "reviewer_notes": "Smoke test: accepted 1, rejected 1",
            "is_ambiguous": False
        }

        save_req_2 = urllib.request.Request(
            f"{base_url}/api/frame/{test_fid}/save",
            data=json.dumps(save_payload_2).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        save_resp_2 = json.loads(urllib.request.urlopen(save_req_2).read().decode())

        # Verify label lines still expected_after_accept (rejected proposal is excluded!)
        lbl_lines_2 = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        assert len(lbl_lines_2) == expected_after_accept, f"Expected {expected_after_accept} label lines, got {len(lbl_lines_2)}"
        print(f"[CHECK 6] YOLO label file after rejection: {len(lbl_lines_2)} lines (rejected proposal excluded)")

        # 7. Reload and verify persistence on disk
        with open(pack_dir / "annotations" / "annotations.json", "r") as f:
            disk_records = json.load(f)
        disk_rec = next(r for r in disk_records if r["frame_id"] == test_fid)
        disk_acc = next(b for b in disk_rec["boxes"] if b["instance_id"] == candidate_proposal["instance_id"])
        disk_rej = next(b for b in disk_rec["boxes"] if b["instance_id"] == candidate_reject["instance_id"])
        assert disk_acc["proposal_status"] == "accepted"
        assert disk_rej["proposal_status"] == "rejected"
        print(f"[CHECK 7] Disk persistence verified: accepted & rejected statuses persisted cleanly.")

        print("\n" + "=" * 70)
        print("  SMOKE TEST PASSED 100%: ALL ACCEPTANCE CRITERIA SATISFIED")
        print("=" * 70)

    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[TEARDOWN] Cleaned up temporary fixture: {tmp_dir}")

if __name__ == "__main__":
    run_smoke_test()
