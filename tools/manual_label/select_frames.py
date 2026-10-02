#!/usr/bin/env python3
"""
Pick frames from the local CCTV videos for manual labeling, split them into train/val,
assign each frame to exactly one annotator, extract the images and pre-label them.

How frames are chosen
---------------------
1. Leakage guards. The first --start-skip frames of every video are ignored, and so is
   anything within --exclusion-gap frames of a frame already used by the old datasets
   (data/multiclass_dataset, data/eval_snapshot_v1) or by packs passed via --exclude-packs.
2. Time-block split. The rest of each video is cut into 10 equal blocks; blocks 2 and 7
   are validation-only, the others are train-only, and train candidates must also sit
   --guard frames away from a val block. Train and val never share a scene moment.
3. Candidate scan. Every video is sampled at ~--per-video-candidates evenly spaced frames;
   each candidate is scored with the baseline (640) and the COCO teacher (640). Results are
   cached in <out>/_cache/ so re-running with other selection settings is instant.
4. Selection.
   - val:   ~11 frames per video, 60% uniformly random (keeps val representative) and
            40% rare-class enriched (so truck / bus / three_wheeler have some support).
   - train: 85% targeted at motorcycles, three-wheelers, trucks, buses, likely pickups
            (COCO "truck" on a baseline "car") and suspected pickup-labelled-as-truck
            errors, with a per-video floor/cap; 15% random "general" frames.
   - calib: a handful of frames labeled by *everyone*, used only to measure how
            consistently the annotators label. Not used for train or val.
   Frames of the same video and split are at least --min-gap frames apart.
5. Assignment alternates annotators per split after sorting by video, so each person gets
   the same mix of cameras, day/night, train/val. Frames never overlap between people
   (except calib).

Usage:
  python tools/manual_label/select_frames.py --out data/manual_v1
  python tools/manual_label/select_frames.py --out data/manual_v1 --dry-run        # just show the selection
  python tools/manual_label/select_frames.py --out data/manual_v2 --exclude-packs data/manual_v1 --train 300 --val 0
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.manual_label.common import (
    BUS, CAR, CLASS_NAMES, EXISTING_DATASETS, MOTORCYCLE, REPO_ROOT, THREE_WHEELER, TRUCK,
    Pack, iou_xyxy, make_frame_id, now_iso, parse_frame_id, read_json, write_json_atomic,
)
from tools.manual_label.prelabel import DEFAULT_BASELINE, DEFAULT_TEACHER, predict, prelabel_pack

NUM_BLOCKS = 10
VAL_BLOCKS = (2, 7)
SCAN_VERSION = 1


# --------------------------------------------------------------------------- leakage guards


def existing_frame_indices(exclude_packs: Sequence[Path]) -> Dict[str, List[int]]:
    used: Dict[str, set] = defaultdict(set)
    for ds in EXISTING_DATASETS:
        root = REPO_ROOT / ds
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in (".jpg", ".png", ".txt"):
                parsed = parse_frame_id(p.name)
                if parsed:
                    used[parsed[0]].add(parsed[1])
    for pack_dir in exclude_packs:
        meta = read_json(Path(pack_dir) / "frames.json")
        if meta is None:
            raise SystemExit(f"--exclude-packs: {pack_dir} has no frames.json")
        for f in meta["frames"]:
            used[f["camera"]].add(f["frame_idx"])
    return {cam: sorted(v) for cam, v in used.items()}


def near_any(idx: int, sorted_vals: Sequence[int], gap: int) -> bool:
    import bisect
    i = bisect.bisect_left(sorted_vals, idx)
    for j in (i - 1, i):
        if 0 <= j < len(sorted_vals) and abs(sorted_vals[j] - idx) < gap:
            return True
    return False


def candidate_positions(n_frames: int, args, used: Sequence[int]) -> List[Dict]:
    start, end = args.start_skip, n_frames - 2000
    if end - start < NUM_BLOCKS * 1000:
        return []
    block_len = (end - start) / NUM_BLOCKS
    val_ranges = [(start + b * block_len, start + (b + 1) * block_len) for b in VAL_BLOCKS]
    stride = max(1, int((end - start) / args.per_video_candidates))
    out = []
    for idx in range(start + stride // 2, end, stride):
        if near_any(idx, used, args.exclusion_gap):
            continue
        in_val = any(lo <= idx < hi for lo, hi in val_ranges)
        if not in_val and any(lo - args.guard <= idx < hi + args.guard for lo, hi in val_ranges):
            continue
        out.append({"frame_idx": idx, "block": "val" if in_val else "train"})
    return out


# --------------------------------------------------------------------------- scan


def frame_stats(base: List[Dict], coco: List[Dict], img_w: int, img_h: int) -> Dict:
    scale = 640 / max(img_w, img_h)
    base = [b for b in base if b["conf"] >= 0.30]
    coco = [b for b in coco if b["conf"] >= 0.35]
    counts = Counter(b["cls"] for b in base)
    coco_cars = [b for b in coco if b["cls"] == CAR]
    coco_trucks = [b for b in coco if b["cls"] == TRUCK]

    def overlaps(box, others):
        return any(iou_xyxy(box["xyxy"], o["xyxy"]) >= 0.5 for o in others)

    truck_suspect = sure_truck = 0
    for b in base:
        if b["cls"] != TRUCK:
            continue
        if b["conf"] < 0.7 or overlaps(b, coco_cars):
            truck_suspect += 1
        else:
            sure_truck += 1
    small = sum(1 for b in base
                if (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]) * scale * scale < 32 * 32)
    return {
        "car": counts[CAR], "motorcycle": counts[MOTORCYCLE], "bus": counts[BUS],
        "truck": counts[TRUCK], "three_wheeler": counts[THREE_WHEELER],
        "total": len(base), "small": small,
        "sure_truck": sure_truck, "truck_suspect": truck_suspect,
        "pickup_proxy": sum(1 for b in base if b["cls"] == CAR and overlaps(b, coco_trucks)),
        "coco_extra": sum(1 for c in coco if not overlaps(c, base)),
    }


def read_frame(cap, idx: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None or float(frame.std()) < 6.0:  # unreadable / blank / frozen grey
        return None
    return frame


def scan_videos(videos: List[Path], args, used: Dict[str, List[int]], cache_path: Path) -> List[Dict]:
    import cv2
    from ultralytics import YOLO

    cache = read_json(cache_path, default={})
    params = {"v": SCAN_VERSION, "start_skip": args.start_skip, "per_video": args.per_video_candidates,
              "guard": args.guard, "exclusion_gap": args.exclusion_gap,
              "exclude_packs": sorted(str(Path(p).resolve()) for p in args.exclude_packs)}
    if cache.get("params") != params:
        cache = {"params": params, "videos": {}}

    base_model = teacher = None
    for video in videos:
        cam = video.stem
        if cam in cache["videos"]:
            print(f"[scan] {cam}: cached ({len(cache['videos'][cam])} candidates)")
            continue
        if base_model is None:
            base_model = YOLO(str(args.baseline))
            teacher = YOLO(str(args.teacher))
        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        positions = candidate_positions(n_frames, args, used.get(cam, []))
        print(f"[scan] {cam}: {n_frames} frames -> {len(positions)} candidates")
        results = []
        for start in range(0, len(positions), 8):
            chunk, images = [], []
            for pos in positions[start:start + 8]:
                frame = read_frame(cap, pos["frame_idx"])
                if frame is not None:
                    chunk.append(pos)
                    images.append(frame)
            if not images:
                continue
            base = predict(base_model, images, 640, 0.25, "base640")
            coco = predict(teacher, images, 640, 0.25, "coco640", coco=True)
            for pos, img, b, c in zip(chunk, images, base, coco):
                h, w = img.shape[:2]
                results.append({**pos, "camera": cam, "video": video.name, "stats": frame_stats(b, c, w, h)})
            print(f"  {min(start + 8, len(positions))}/{len(positions)}", end="\r", flush=True)
        print()
        cap.release()
        cache["videos"][cam] = results
        write_json_atomic(cache_path, cache)  # per-video checkpoint so an interrupted scan resumes
    return [c for cam in sorted(cache["videos"]) for c in cache["videos"][cam]]


# --------------------------------------------------------------------------- selection


def target_score(s: Dict) -> float:
    return (3.0 * min(s["three_wheeler"], 3)
            + 0.6 * min(s["motorcycle"], 12)
            + 1.5 * min(s["sure_truck"], 3)
            + 2.0 * min(s["truck_suspect"], 3)
            + 0.8 * min(s["pickup_proxy"], 8)
            + 1.5 * min(s["bus"], 2)
            + 0.1 * min(s["total"], 40))


def rare_score(s: Dict) -> float:
    return 3.0 * min(s["three_wheeler"], 2) + 2.0 * min(s["truck"], 2) + 2.0 * min(s["bus"], 2) + 0.3 * min(s["motorcycle"], 6)


class Picker:
    """Tracks picks and enforces the minimum frame gap within one (video, split)."""

    def __init__(self, min_gap: int):
        self.min_gap = min_gap
        self.taken: Dict[tuple, List[int]] = defaultdict(list)
        self.picked: List[Dict] = []
        self.ids = set()

    def ok(self, c: Dict, split: str) -> bool:
        key = (c["camera"], split)
        return id(c) not in self.ids and all(abs(c["frame_idx"] - t) >= self.min_gap for t in self.taken[key])

    def take(self, c: Dict, split: str, reason: str) -> None:
        self.taken[(c["camera"], split)].append(c["frame_idx"])
        self.ids.add(id(c))
        self.picked.append({**c, "split": split, "reason": reason})

    def count(self, split: str, camera: Optional[str] = None) -> int:
        return sum(1 for p in self.picked if p["split"] == split and (camera is None or p["camera"] == camera))

    def fill(self, pool: List[Dict], split: str, reason: str, n: int, cap: Optional[int] = None) -> int:
        got = 0
        for c in pool:
            if got >= n:
                break
            if cap is not None and self.count(split, c["camera"]) >= cap:
                continue
            if self.ok(c, split):
                self.take(c, split, reason)
                got += 1
        return got


def split_quota(total: int, cams: List[str], available: Dict[str, int]) -> Dict[str, int]:
    quota = {c: 0 for c in cams}
    for i in range(total):
        # round-robin over cameras that still have candidates, most-available first on ties
        order = sorted(cams, key=lambda c: (quota[c], -available[c]))
        for c in order:
            if quota[c] < available[c]:
                quota[c] += 1
                break
    return quota


def select(cands: List[Dict], args) -> List[Dict]:
    rng = random.Random(args.seed)
    picker = Picker(args.min_gap)
    cams = sorted({c["camera"] for c in cands})
    by_cam_block = defaultdict(list)
    for c in cands:
        by_cam_block[(c["camera"], c["block"])].append(c)

    # ---- val: per-camera quota, 60% random + 40% rare-enriched
    if args.val:
        quota = split_quota(args.val, cams, {c: len(by_cam_block[(c, "val")]) for c in cams})
        for cam in cams:
            pool = by_cam_block[(cam, "val")]
            n_rand = round(quota[cam] * 0.6)
            shuffled = pool[:]
            rng.shuffle(shuffled)
            picker.fill(shuffled, "val", "random", n_rand)
            ranked = sorted(pool, key=lambda c: -rare_score(c["stats"]))
            picker.fill(ranked, "val", "rare_enriched", quota[cam] - picker.count("val", cam))
            picker.fill(shuffled, "val", "random", quota[cam] - picker.count("val", cam))

    # ---- train: targeted (per-camera floor, then global with cap), then random general
    if args.train:
        n_general = round(args.train * args.general_frac)
        n_targeted = args.train - n_general
        floor = min(args.train_floor, n_targeted // max(1, len(cams)))
        ranked_by_cam = {cam: sorted(by_cam_block[(cam, "train")], key=lambda c: -target_score(c["stats"])) for cam in cams}
        for cam in cams:
            picker.fill(ranked_by_cam[cam], "train", "targeted", floor)
        global_ranked = sorted((c for cam in cams for c in by_cam_block[(cam, "train")]),
                               key=lambda c: -target_score(c["stats"]))
        picker.fill(global_ranked, "train", "targeted", n_targeted - picker.count("train"), cap=args.train_cap)
        rest = [c for cam in cams for c in by_cam_block[(cam, "train")]]
        rng.shuffle(rest)
        picker.fill(rest, "train", "general", args.train - picker.count("train"))

    # ---- calibration: one interesting frame from several different videos, labeled by everyone
    if args.calibration:
        cam_order = cams[:]
        rng.shuffle(cam_order)
        for cam in cam_order:
            if picker.count("calib") >= args.calibration:
                break
            ranked = sorted(by_cam_block[(cam, "train")], key=lambda c: -target_score(c["stats"]))
            ranked = [c for c in ranked if picker.ok(c, "train")][len(ranked) // 10:]  # skip the very top: those are train picks
            picker.fill(ranked, "calib", "calibration", 1)
    return picker.picked


def assign(picked: List[Dict], annotators: List[str], seed: int) -> List[Dict]:
    rng = random.Random(seed + 1)
    frames = []
    for split in ("calib", "val", "train"):
        group = sorted((p for p in picked if p["split"] == split), key=lambda p: (p["camera"], p["frame_idx"]))
        for i, p in enumerate(group):
            p["assignee"] = "*" if split == "calib" else annotators[i % len(annotators)]
        # queue order: calib first, then val, then train; shuffled within split so a half-done
        # queue still covers every camera
        rng.shuffle(group)
        frames.extend(group)
    out = []
    for order, p in enumerate(frames):
        out.append({
            "id": make_frame_id(p["camera"], p["frame_idx"]),
            "camera": p["camera"],
            "video": p["video"],
            "frame_idx": p["frame_idx"],
            "night": p["camera"].endswith("_night"),
            "split": p["split"],
            "assignee": p["assignee"],
            "reason": p["reason"],
            "order": order,
            "scan": p["stats"],
        })
    return out


def print_summary(frames: List[Dict], annotators: List[str]) -> None:
    print("\n=== selection ===")
    by = Counter((f["split"], f["camera"]) for f in frames)
    cams = sorted({f["camera"] for f in frames})
    print(f"{'camera':22s} {'calib':>5s} {'val':>5s} {'train':>5s}")
    for cam in cams:
        print(f"{cam:22s} {by[('calib', cam)]:5d} {by[('val', cam)]:5d} {by[('train', cam)]:5d}")
    for split in ("calib", "val", "train"):
        fs = [f for f in frames if f["split"] == split]
        if not fs:
            continue
        tot = Counter()
        for f in fs:
            tot.update({k: f["scan"][k] for k in CLASS_NAMES + ["truck_suspect", "pickup_proxy", "coco_extra"]})
        reasons = Counter(f["reason"] for f in fs)
        print(f"\n{split}: {len(fs)} frames  reasons={dict(reasons)}")
        print("  baseline detections (lower bound, true counts are higher): "
              + ", ".join(f"{k}={tot[k]}" for k in CLASS_NAMES))
        print(f"  suspected pickup->truck errors={tot['truck_suspect']}  likely pickups={tot['pickup_proxy']}"
              f"  extra COCO-only vehicles={tot['coco_extra']}")
    print("\nper annotator:")
    for a in annotators:
        mine = [f for f in frames if f["assignee"] in ("*", a)]
        c = Counter(f["split"] for f in mine)
        night = sum(f["night"] for f in mine)
        print(f"  {a:10s} calib={c['calib']} val={c['val']} train={c['train']} total={len(mine)} (night {night})")


# --------------------------------------------------------------------------- extraction


def extract_images(frames: List[Dict], out: Path) -> None:
    import cv2
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    by_video = defaultdict(list)
    for f in frames:
        by_video[f["video"]].append(f)
    done = 0
    for video, fs in sorted(by_video.items()):
        cap = cv2.VideoCapture(str(REPO_ROOT / "videos" / video))
        for f in sorted(fs, key=lambda f: f["frame_idx"]):
            dst = img_dir / f"{f['id']}.jpg"
            if not dst.exists():
                frame = read_frame(cap, f["frame_idx"])
                if frame is None:
                    raise RuntimeError(f"could not re-read {video} frame {f['frame_idx']}")
                cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            done += 1
            print(f"[extract] {done}/{len(frames)}", end="\r", flush=True)
        cap.release()
    print()


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="pack directory to create, e.g. data/manual_v1")
    ap.add_argument("--annotators", nargs="+", default=["me", "friend"],
                    help="annotator ids (lowercase, no spaces); frames alternate between them")
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--calibration", type=int, default=6, help="frames labeled by everyone to check agreement (0 to disable)")
    ap.add_argument("--general-frac", type=float, default=0.15, help="share of train frames picked at random")
    ap.add_argument("--train-floor", type=int, default=20, help="minimum targeted train frames per video")
    ap.add_argument("--train-cap", type=int, default=45, help="maximum targeted train frames per video")
    ap.add_argument("--videos", nargs="*", help="video stems to use (default: all videos/cam*.avi)")
    ap.add_argument("--start-skip", type=int, default=260_000, help="ignore the first N frames of every video")
    ap.add_argument("--exclusion-gap", type=int, default=9_000, help="min distance (frames) from old dataset frames")
    ap.add_argument("--guard", type=int, default=9_000, help="min distance (frames) between train candidates and val blocks")
    ap.add_argument("--min-gap", type=int, default=4_500, help="min distance between picks of the same video+split")
    ap.add_argument("--per-video-candidates", type=int, default=360)
    ap.add_argument("--exclude-packs", nargs="*", default=[], type=Path, help="earlier packs whose frames must be avoided")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="scan + print the selection, write nothing else")
    ap.add_argument("--no-prelabel", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if annotators already saved work in this pack")
    args = ap.parse_args()

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    for a in args.annotators:
        if not a.replace("_", "").isalnum() or a != a.lower() or a == "*":
            raise SystemExit(f"annotator id '{a}' must be lowercase letters/digits/underscore")
    work = out / "work"
    if work.exists() and any(work.rglob("*.json")) and not args.force and not args.dry_run:
        raise SystemExit(f"{work} already contains saved labels - rebuilding would reshuffle assignments. "
                         "Use a new --out, or --force if you really mean it.")

    videos = sorted((REPO_ROOT / "videos").glob("cam*.avi"))
    if args.videos:
        videos = [v for v in videos if v.stem in set(args.videos)]
    if not videos:
        raise SystemExit("no videos found")

    used = existing_frame_indices(args.exclude_packs)
    print(f"[guard] excluding frames near {sum(map(len, used.values()))} previously used frames "
          f"across {len(used)} cameras")
    cands = scan_videos(videos, args, used, out / "_cache" / "candidates.json")
    picked = select(cands, args)
    frames = assign(picked, args.annotators, args.seed)
    print_summary(frames, args.annotators)
    short = [(s, n) for s, n in (("val", args.val), ("train", args.train))
             if sum(f["split"] == s for f in frames) < n]
    for s, n in short:
        print(f"[warn] only {sum(f['split'] == s for f in frames)}/{n} {s} frames found - lower --min-gap or raise --per-video-candidates")
    if args.dry_run:
        return

    extract_images(frames, out)
    meta = {
        "pack_id": out.name,
        "created_at": now_iso(),
        "annotators": args.annotators,
        "classes": CLASS_NAMES,
        "selection": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                      if k not in ("out", "dry_run", "force", "exclude_packs")}
                     | {"exclude_packs": [str(p) for p in args.exclude_packs],
                        "num_blocks": NUM_BLOCKS, "val_blocks": list(VAL_BLOCKS)},
        "frames": frames,
    }
    write_json_atomic(out / "frames.json", meta)
    print(f"[pack] wrote {out / 'frames.json'} ({len(frames)} frames)")

    if not args.no_prelabel:
        prelabel_pack(Pack(out), args.baseline.resolve(), args.teacher.resolve(), overwrite=False)
    print(f"\nDone. Start labeling with:\n  python tools/manual_label/label_server.py --pack {args.out}")


if __name__ == "__main__":
    main()
