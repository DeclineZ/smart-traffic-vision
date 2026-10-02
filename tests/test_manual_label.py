"""
tests/test_manual_label.py

Unit tests for the manual labeling workflow (tools/manual_label). CPU only, no models or videos:
- pre-label merging flags the pickup/truck disagreement and keeps the baseline class
- frame selection keeps train/val apart, respects the min gap, and gives each frame one owner
- the labeling server refuses saves on someone else's frame and keeps reviews in the reviewer's folder
- merge builds YOLO labels only from done frames and measures calibration agreement
"""

import io
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from argparse import Namespace
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path

from tools.manual_label import label_server, merge, sync
from tools.manual_label.common import Pack, read_json, validate_boxes, write_json_atomic, xyxy_to_yolo
from tools.manual_label.prelabel import merge_proposals
from tools.manual_label.select_frames import assign, candidate_positions, select

# SOI + a SOF0 header saying 1920x1080 + EOI: all the server's size parser reads
TINY_JPEG = bytes.fromhex("ffd8" "ffc0000b080438078001011100" "ffd9")


def make_pack(root: Path, frames):
    (root / "images").mkdir(parents=True)
    for f in frames:
        (root / "images" / f"{f['id']}.jpg").write_bytes(TINY_JPEG)
    write_json_atomic(root / "frames.json", {"pack_id": root.name, "annotators": ["alice", "bob"], "frames": frames})
    return Pack(root)


def frame(fid, split, assignee, order):
    return {"id": fid, "camera": fid.split("_f")[0], "video": "x.avi", "frame_idx": int(fid.split("_f")[1]),
            "night": False, "split": split, "assignee": assignee, "reason": "test", "order": order, "scan": {}}


class TestPrelabelMerge(unittest.TestCase):
    def test_pickup_called_truck_is_flagged_and_keeps_baseline_class(self):
        boxes = merge_proposals([
            {"cls": 3, "xyxy": [100, 100, 200, 180], "conf": 0.55, "src": "base640"},
            {"cls": 0, "xyxy": [102, 101, 199, 181], "conf": 0.80, "src": "coco1280"},
        ])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["cls"], 3)
        self.assertTrue(boxes[0]["check"])
        self.assertIn("COCO model says car", boxes[0]["note"])

    def test_coco_truck_on_baseline_car_is_not_a_disagreement(self):
        boxes = merge_proposals([
            {"cls": 0, "xyxy": [10, 10, 60, 40], "conf": 0.9, "src": "base640"},
            {"cls": 3, "xyxy": [11, 10, 61, 41], "conf": 0.7, "src": "coco1280"},
        ])
        self.assertEqual([(b["cls"], b["check"]) for b in boxes], [(0, False)])

    def test_coco_only_truck_becomes_flagged_car(self):
        boxes = merge_proposals([{"cls": 3, "xyxy": [300, 50, 330, 70], "conf": 0.6, "src": "coco1280"}])
        self.assertEqual(boxes[0]["cls"], 0)
        self.assertTrue(boxes[0]["check"])

    def test_low_confidence_coco_only_is_dropped(self):
        self.assertEqual(merge_proposals([{"cls": 0, "xyxy": [0, 0, 20, 20], "conf": 0.2, "src": "coco1280"}]), [])


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(start_skip=100_000, per_video_candidates=200, guard=5_000, exclusion_gap=5_000,
                              val=20, train=40, calibration=2, general_frac=0.2, train_floor=5, train_cap=20,
                              min_gap=3_000, seed=0)

    def candidates(self):
        stats = {"car": 5, "motorcycle": 2, "bus": 0, "truck": 1, "three_wheeler": 0, "total": 8, "small": 3,
                 "sure_truck": 0, "truck_suspect": 1, "pickup_proxy": 1, "coco_extra": 0}
        out = []
        for cam in ("cam01_a", "cam02_b", "cam03_c_night"):
            for pos in candidate_positions(1_000_000, self.args, used=[400_000]):
                out.append({**pos, "camera": cam, "video": f"{cam}.avi", "stats": dict(stats, motorcycle=pos["frame_idx"] % 7)})
        return out

    def test_positions_respect_guards(self):
        pos = candidate_positions(1_000_000, self.args, used=[400_000])
        self.assertTrue(all(p["frame_idx"] >= 100_000 for p in pos))
        self.assertFalse(any(abs(p["frame_idx"] - 400_000) < 5_000 for p in pos))
        val = [p["frame_idx"] for p in pos if p["block"] == "val"]
        train = [p["frame_idx"] for p in pos if p["block"] == "train"]
        self.assertTrue(val and train)
        self.assertGreaterEqual(min(abs(v - t) for v in val for t in train), 5_000 - 1)

    def test_selection_counts_gaps_and_owners(self):
        frames = assign(select(self.candidates(), self.args), ["alice", "bob"], seed=0)
        splits = [f["split"] for f in frames]
        self.assertEqual(splits.count("val"), 20)
        self.assertEqual(splits.count("train"), 40)
        self.assertEqual(splits.count("calib"), 2)
        self.assertEqual(len({f["id"] for f in frames}), len(frames), "a frame was picked twice")
        for f in frames:
            self.assertEqual(f["assignee"] == "*", f["split"] == "calib")
        for split in ("val", "train"):
            owners = [f["assignee"] for f in frames if f["split"] == split]
            self.assertLessEqual(abs(owners.count("alice") - owners.count("bob")), 1)
            by_cam = {}
            for f in frames:
                if f["split"] == split:
                    by_cam.setdefault(f["camera"], []).append(f["frame_idx"])
            for idxs in by_cam.values():
                idxs.sort()
                self.assertTrue(all(b - a >= 3_000 for a, b in zip(idxs, idxs[1:])))


class TestServerAndMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pack = make_pack(self.tmp / "pack", [
            frame("cam01_a_f0000100", "calib", "*", 0),
            frame("cam01_a_f0200000", "val", "alice", 1),
            frame("cam01_a_f0300000", "train", "bob", 2),
            frame("cam01_a_f0400000", "train", "alice", 3),
        ])
        label_server.Handler.pack = self.pack
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), label_server.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def save(self, user, fid, boxes, status="done"):
        return self.post("/api/save", {"id": fid, "user": user, "status": status, "boxes": boxes})

    def test_cannot_save_someone_elses_frame(self):
        code, body = self.save("bob", "cam01_a_f0200000", [])
        self.assertEqual(code, 403)
        self.assertFalse(self.pack.label_path("bob", "cam01_a_f0200000").exists())

    def test_save_writes_to_own_folder_and_keeps_check_flag(self):
        code, _ = self.save("alice", "cam01_a_f0200000",
                            [{"cls": 1, "xyxy": [10, 10, 40, 50], "check": True, "note": "?"}, {"cls": 0, "xyxy": [5, 5, 6, 6]}],
                            status="in_progress")
        self.assertEqual(code, 200)
        rec = read_json(self.pack.label_path("alice", "cam01_a_f0200000"))
        self.assertEqual(rec["img_w"], 1920)
        self.assertEqual(len(rec["boxes"]), 1, "degenerate box should be dropped")
        self.assertTrue(rec["boxes"][0]["check"])

    def test_review_lands_in_reviewer_folder(self):
        self.save("bob", "cam01_a_f0300000", [{"cls": 0, "xyxy": [0, 0, 50, 50]}])
        code, _ = self.post("/api/review", {"id": "cam01_a_f0300000", "owner": "bob", "user": "alice",
                                            "verdict": "issue", "comment": "missed a car"})
        self.assertEqual(code, 200)
        self.assertTrue(self.pack.review_path("alice", "cam01_a_f0300000").exists())
        code, _ = self.post("/api/review", {"id": "cam01_a_f0300000", "owner": "bob", "user": "bob", "verdict": "ok"})
        self.assertEqual(code, 403)
        self.assertEqual(len(merge.open_review_issues(self.pack)), 1)

    def test_build_uses_only_done_frames_and_writes_yolo(self):
        self.save("alice", "cam01_a_f0200000", [{"cls": 3, "xyxy": [0, 0, 960, 540]}])
        self.save("bob", "cam01_a_f0300000", [])  # verified empty frame
        self.save("alice", "cam01_a_f0400000", [{"cls": 0, "xyxy": [0, 0, 10, 10]}], status="in_progress")
        out = self.tmp / "ds"
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                merge.cmd_build([self.pack], out, allow_incomplete=False, link=False)
            merge.cmd_build([self.pack], out, allow_incomplete=True, link=False)
        self.assertEqual((out / "labels/val/cam01_a_f0200000.txt").read_text().split(),
                         ["3", "0.250000", "0.250000", "0.500000", "0.500000"])
        self.assertEqual((out / "labels/train/cam01_a_f0300000.txt").read_text(), "")
        self.assertFalse((out / "labels/train/cam01_a_f0400000.txt").exists())
        self.assertFalse(any("f0000100" in p.name for p in out.rglob("*")), "calibration frames must stay out")

    def test_agreement_counts_missed_boxes(self):
        fid = "cam01_a_f0000100"
        self.save("alice", fid, [{"cls": 0, "xyxy": [0, 0, 50, 50]}, {"cls": 1, "xyxy": [100, 100, 120, 140]}])
        self.save("bob", fid, [{"cls": 3, "xyxy": [1, 1, 50, 50]}])
        buf = io.StringIO()
        with redirect_stdout(buf):
            merge.cmd_agreement([self.pack])
        text = buf.getvalue()
        self.assertIn("matched=1 same_class=0 only_alice=1 only_bob=0", text)
        self.assertIn("car->truck=1", text)


    def test_sync_refuses_to_overwrite_newer_work(self):
        self.save("bob", "cam01_a_f0300000", [{"cls": 0, "xyxy": [0, 0, 50, 50]}])
        zip_path = self.tmp / "bob.zip"
        with redirect_stdout(io.StringIO()):
            sync.send(self.pack, "bob", zip_path)
            self.save("bob", "cam01_a_f0300000", [])  # newer local edit
            with self.assertRaises(SystemExit):
                sync.receive(zip_path, self.tmp, force=False)
            sync.receive(zip_path, self.tmp, force=True)
        self.assertEqual(len(read_json(self.pack.label_path("bob", "cam01_a_f0300000"))["boxes"]), 1)


class TestGeometry(unittest.TestCase):
    def test_validate_clamps_and_sorts(self):
        boxes = validate_boxes([{"cls": 2, "xyxy": [2000, -5, 1800, 100]}], 1920, 1080)
        self.assertEqual(boxes[0]["xyxy"], [1800.0, 0.0, 1920.0, 100.0])
        with self.assertRaises(ValueError):
            validate_boxes([{"cls": 7, "xyxy": [0, 0, 10, 10]}], 100, 100)

    def test_yolo_roundtrip(self):
        self.assertEqual(xyxy_to_yolo([0, 0, 100, 50], 200, 100), (0.25, 0.25, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
