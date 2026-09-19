"""
Compile Unified 5-Class Thai Traffic Dataset with YOLO26x Co-Training Teacher.

Classes:
  0: car (sedan, hatchback, taxi, SUV, passenger commuter van - aligned with COCO)
  1: motorcycle (scooter, underbone, commuter bike, big bike)
  2: bus (BMTA transit bus, EV Thai Smile, tour coach, double-decker)
  3: truck (pickup, songthaew, delivery box pickup, 6/10/18-wheeler)
  4: three_wheeler (tuk-tuk, motorized saleng sidecar)

Features:
- Multi-folder aggregation across curated hits (saleng, pickup, van, truck_trailer, bus, tuktuk).
- Co-training teacher YOLO26x with IoMin suppression to annotate background vehicles.
- Cross-category anti-corruption checks preventing false teacher labels on Thai custom vehicles.
- Minority class targeted oversampling (saleng 4x, bus 3x, heavy truck 2x).
- Temporal event clustering (< 3.0s) preventing train/val split leakage.
- Synthetic monochrome IR generation and nighttime photometric jitter.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
    4: "three_wheeler"
}

# Source category folder to custom class mapping (5-Class Standard)
CATEGORY_CONFIG = {
    "pickup": {
        "class_id": 0,  # Passenger 4-wheeler (1.0 PCE, aligned with COCO car category)
        "verified_dir": "data/pickup/verified_hits",
        "manifest_path": "data/pickup/mined_candidates/manifest.json",
        "raw_frames_dir": "data/pickup/mined_candidates/raw_frames",
    },
    "songthaew": {
        "class_id": 0,  # Built on pickup chassis, passenger-class local transport (1.0 PCE)
        "verified_dir": "data/songthaew/verified_hits",
        "manifest_path": "data/songthaew/mined_candidates/manifest.json",
        "raw_frames_dir": "data/songthaew/mined_candidates/raw_frames",
    },
    "van": {
        "class_id": 0,  # Aligned with COCO car category (minivans, passenger commuter vans)
        "verified_dir": "data/van/verified_hits",
        "manifest_path": "data/van/mined_candidates/manifest.json",
        "raw_frames_dir": "data/van/mined_candidates/raw_frames",
    },
    "truck_trailer": {
        "class_id": 3,  # Heavy commercial transport only (6/10/18-wheelers)
        "verified_dir": "data/truck_trailer/verified_hits",
        "manifest_path": "data/truck_trailer/mined_candidates/manifest.json",
        "raw_frames_dir": "data/truck_trailer/mined_candidates/raw_frames",
    },
    "bus": {
        "class_id": 2,  # BMTA, tour coaches, transit buses
        "verified_dir": "data/bus/verified_hits",
        "manifest_path": "data/bus/mined_candidates/manifest.json",
        "raw_frames_dir": "data/bus/mined_candidates/raw_frames",
    },
    "saleng": {
        "class_id": 4,  # Motorized sidecar tricycles
        "verified_dir": "data/saleng/verified_hits",
        "manifest_path": "data/saleng/mined_candidates/manifest.json",
        "raw_frames_dir": "data/saleng/mined_candidates/raw_frames",
        "is_saleng": True,
    },
    "tuktuk": {
        "class_id": 4,  # Auto-rickshaws / Tuk-tuks
        "verified_dir": "data/tuktuk/verified_hits",
        "manifest_path": "data/tuktuk/mined_candidates/manifest.json",
        "raw_frames_dir": "data/tuktuk/mined_candidates/raw_frames",
    },
}

COCO_TEACHER_TO_CUSTOM = {
    2: 0,  # COCO car -> custom car (sedan/taxi)
    3: 1,  # COCO motorcycle -> custom motorcycle
    5: 2,  # COCO bus -> custom bus
    7: 3,  # COCO truck -> custom truck (size disambiguated)
}


def deduplicate_boxes(boxes: List[Dict], iou_thresh: float = 0.55, iomin_thresh: float = 0.85) -> List[Dict]:
    """
    Deduplicates boxes across all classes using IoU and IoMin suppression.
    Safe thresholds (IoU >= 0.55, IoMin >= 0.85) prevent adjacent queued vehicles
    and small vehicles near larger ones from being falsely erased.
    """
    keep: List[Dict] = []
    for cand in boxes:
        c_box = cand["xyxy"]
        suppress = False
        for acc in keep:
            a_box = acc["xyxy"]
            if compute_iou(c_box, a_box) >= iou_thresh or compute_iomin(c_box, a_box) >= iomin_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(cand)
    return keep


def compute_iou(box1: List[float], box2: List[float]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_iomin(box1: List[float], box2: List[float]) -> float:
    """Intersection over Minimum (IoMin). Detects when a small box is nested inside a large box."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    min_area = min(area1, area2)

    return inter_area / min_area if min_area > 0 else 0.0


def xyxy_to_yolo(xyxy: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return max(0.0, min(1.0, xc / img_w)), max(0.0, min(1.0, yc / img_h)), max(0.0, min(1.0, bw / img_w)), max(0.0, min(1.0, bh / img_h))


def apply_night_jitter(img: np.ndarray, seed: int = None) -> np.ndarray:
    """Subtle low-light photometric perturbations."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    gamma = random.uniform(0.82, 1.25)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    jittered = cv2.LUT(img, table)

    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-8.0, 8.0)
    jittered = np.clip(alpha * jittered.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    noise = np.random.normal(0, 2.0, jittered.shape).astype(np.float32)
    jittered = np.clip(jittered.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return jittered


def get_roadway_crop_box(video_name: str, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Return full uncropped frame to maintain natural CCTV perspective and prevent
    clipping tall vehicles (such as double-decker tour buses) or distant vehicles.
    """
    return (0, 0, img_w, img_h)


def run_compilation(args):
    dataset_dir = Path(args.dataset_dir)
    img_train_dir = dataset_dir / "images" / "train"
    img_val_dir = dataset_dir / "images" / "val"
    lbl_train_dir = dataset_dir / "labels" / "train"
    lbl_val_dir = dataset_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for old_f in d.glob("*.*"):
            old_f.unlink()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading high-capacity teacher detector: {args.teacher_model} on {device}...")
    teacher_yolo = YOLO(args.teacher_model)

    # 1. Aggregate curated frames across all category directories
    curated_frames: Dict[str, Dict[str, Any]] = {}
    category_counts = {cid: 0 for cid in CLASS_NAMES.keys()}

    # 1. Build a universal global crop index across all category manifests
    print("\n--- Indexing All Available Staged Manifests ---")
    global_crop_index: Dict[str, Tuple[str, Dict, Dict, Path]] = {}
    high_sim_pickups_by_frame: Dict[str, List[List[float]]] = {}
    high_sim_vans_by_frame: Dict[str, List[List[float]]] = {}
    for cat_name, cat_cfg in CATEGORY_CONFIG.items():
        mpath = Path(cat_cfg["manifest_path"])
        rdir = Path(cat_cfg["raw_frames_dir"])
        if mpath.exists():
            try:
                with open(mpath, "r") as f:
                    mdata = json.load(f)
                indexed_count = 0
                for fk, finfo in mdata.items():
                    for tgt in finfo.get("targets", []):
                        cfname = tgt["crop_filename"]
                        sim = tgt.get("sim", 0.0)
                        if cat_name in ["pickup", "songthaew"] and sim >= 0.65:
                            high_sim_pickups_by_frame.setdefault(fk, []).append(tgt["xyxy"])
                        elif cat_name == "van" and sim >= 0.65:
                            high_sim_vans_by_frame.setdefault(fk, []).append(tgt["xyxy"])
                        if cfname not in global_crop_index:
                            global_crop_index[cfname] = (fk, finfo, tgt, rdir)
                            indexed_count += 1
                print(f"  Indexed {indexed_count} candidate crop records from {cat_name} manifest.")
            except Exception as e:
                print(f"  Warning loading {cat_name} manifest: {e}")

    print(f"Total global crop records indexed: {len(global_crop_index)}")
    print(f"Indexed high-sim candidate frames: {len(high_sim_pickups_by_frame)} pickup frames, {len(high_sim_vans_by_frame)} van frames.")

    # 2. Aggregate curated frames across all category directories (supporting cross-category moves)
    print("\n--- Aggregating Verified Hits Across Categories ---")
    for cat_name, cat_cfg in CATEGORY_CONFIG.items():
        vdir = Path(cat_cfg["verified_dir"])
        if not vdir.exists():
            continue

        verified_files = set(os.listdir(vdir))
        if not verified_files:
            continue

        matched_for_cat = 0
        cls_id = cat_cfg["class_id"]
        is_saleng = cat_cfg.get("is_saleng", False)

        for cfname in verified_files:
            # Handle accidental Windows ' - Copy' or ' (1)' suffix
            clean_cfname = cfname
            if " - Copy" in clean_cfname or " (" in clean_cfname:
                clean_cfname = re.sub(r' - Copy(?:\s*\(\d+\))?', '', clean_cfname)
                clean_cfname = re.sub(r'\s*\(\d+\)', '', clean_cfname)

            lookup_name = cfname if cfname in global_crop_index else clean_cfname

            if lookup_name in global_crop_index:
                fk, finfo, tgt, rdir = global_crop_index[lookup_name]
                raw_path = rdir / finfo["raw_frame_file"]
                if not raw_path.exists():
                    continue

                target_item = {
                    "class_id": cls_id,
                    "xyxy": tgt["xyxy"],
                    "crop_filename": cfname,
                    "category": cat_name,
                    "is_saleng": is_saleng
                }

                if fk not in curated_frames:
                    curated_frames[fk] = {
                        "video_name": finfo["video_name"],
                        "frame_idx": finfo["frame_idx"],
                        "timestamp_sec": finfo["timestamp_sec"],
                        "img_width": finfo["img_width"],
                        "img_height": finfo["img_height"],
                        "raw_frame_path": raw_path,
                        "targets": [target_item],
                        "has_saleng": is_saleng,
                        "categories": {cat_name}
                    }
                else:
                    # Deduplicate overlapping targets (e.g. if copied across category folders)
                    already_present = False
                    for existing_tgt in curated_frames[fk]["targets"]:
                        if compute_iou(existing_tgt["xyxy"], tgt["xyxy"]) > 0.70:
                            already_present = True
                            existing_tgt["class_id"] = cls_id
                            existing_tgt["category"] = cat_name
                            break
                    if not already_present:
                        curated_frames[fk]["targets"].append(target_item)
                    if is_saleng:
                        curated_frames[fk]["has_saleng"] = True
                    curated_frames[fk].setdefault("categories", set()).add(cat_name)

                    if cat_name in ["pickup", "songthaew"]:
                        high_sim_pickups_by_frame.setdefault(fk, []).append(tgt["xyxy"])
                    elif cat_name == "van":
                        high_sim_vans_by_frame.setdefault(fk, []).append(tgt["xyxy"])

                matched_for_cat += 1
                category_counts[cls_id] += 1

        print(f"  [{cat_name:<14}] Successfully loaded {matched_for_cat} verified instances from {len(verified_files)} files in {vdir}.")

    print(f"\nTotal curated frames collected: {len(curated_frames)}")
    if not curated_frames:
        print("No verified hits found in any category. Run mining and copy verified crops before compiling.")
        return

    # 2. Smart Teacher Co-Annotation with IoMin Suppression
    print(f"\n--- Running Teacher Co-Annotation (conf >= {args.teacher_conf:.2f}, IoMin suppression >= 0.65) ---")
    teacher_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for fk, finfo in curated_frames.items():
        src_path = finfo["raw_frame_path"]
        if not src_path.exists():
            continue
        frame_img = cv2.imread(str(src_path))
        if frame_img is None:
            continue

        res = teacher_yolo.predict(frame_img, conf=args.teacher_conf, verbose=False, device=device)
        boxes = res[0].boxes
        cand_background_boxes = []

        for b in boxes:
            c_id = int(b.cls[0].item())
            if c_id not in COCO_TEACHER_TO_CUSTOM:
                continue

            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].item())

            # Strict suppression against verified targets (verified targets always win)
            suppress = False
            for target in finfo["targets"]:
                iou = compute_iou(xyxy, target["xyxy"])
                iomin = compute_iomin(xyxy, target["xyxy"])
                if iou >= 0.35 or iomin >= 0.60:
                    suppress = True
                    break

            if not suppress:
                custom_cls = COCO_TEACHER_TO_CUSTOM[c_id]

                # CORRECTION 1: Protect commuter vans from false Teacher Truck/Bus labels
                # If teacher says truck (custom_cls == 3) or bus (custom_cls == 2),
                # but the box matches a mined van candidate:
                # Force custom_cls = 0 (car) so commuter vans are never trained as trucks or buses.
                if custom_cls in [2, 3]:
                    for v_box in high_sim_vans_by_frame.get(fk, []):
                        if compute_iou(xyxy, v_box) >= 0.30 or compute_iomin(xyxy, v_box) >= 0.50:
                            custom_cls = 0
                            break

                # CORRECTION 2: All pickups (open-bed & box) belong to Class 0 (car)
                # If teacher predicted truck (custom_cls == 3) or bus (custom_cls == 2) on a pickup candidate,
                # force custom_cls = 0 (car) so pickups are unified as passenger vehicles (1.0 PCE).
                if custom_cls in [2, 3]:
                    for p_box in high_sim_pickups_by_frame.get(fk, []):
                        if compute_iou(xyxy, p_box) >= 0.30 or compute_iomin(xyxy, p_box) >= 0.50:
                            custom_cls = 0
                            break

                # Calibrated class-specific confidence thresholds for surveillance CCTV
                # Distant perspective cars & bikes naturally score 0.25-0.55; labeling them prevents background penalty.
                if custom_cls == 0 and conf < 0.25:
                    continue
                if custom_cls == 1 and conf < 0.25:
                    continue
                if custom_cls == 2 and conf < 0.35:
                    continue
                if custom_cls == 3 and conf < 0.35:
                    continue

                cand_background_boxes.append({
                    "class_id": custom_cls,
                    "xyxy": xyxy,
                    "conf": conf
                })

        # Cross-class NMS: highest confidence wins, eliminate duplicate conflicting labels
        cand_background_boxes.sort(key=lambda x: x["conf"], reverse=True)
        background_boxes = deduplicate_boxes(cand_background_boxes, iou_thresh=0.55, iomin_thresh=0.85)
        for bb in background_boxes:
            teacher_counts[bb["class_id"]] += 1
            category_counts[bb["class_id"]] += 1

        finfo["background_vehicles"] = background_boxes

    print(f"Teacher Co-Annotation Summary:")
    print(f"  - Cars (sedans/taxis): {teacher_counts[0]}")
    print(f"  - Motorcycles:        {teacher_counts[1]}")
    print(f"  - Buses:              {teacher_counts[2]}")
    print(f"  - Trucks:             {teacher_counts[3]}")

    # 3. Temporal Event Clustering (< 3.0s) to prevent train/val data leakage
    print("\n--- Partitioning Splits via Temporal Event Clustering ---")
    frames_by_video: Dict[str, List[Tuple[str, Dict]]] = {}
    for fk, finfo in curated_frames.items():
        frames_by_video.setdefault(finfo["video_name"], []).append((fk, finfo))

    events_day = []
    events_night = []

    for vname, vframes in frames_by_video.items():
        vframes.sort(key=lambda x: x[1]["timestamp_sec"])
        current_event = []
        is_night = "_night" in vname

        for fk, finfo in vframes:
            if not current_event:
                current_event.append((fk, finfo))
            else:
                last_t = current_event[-1][1]["timestamp_sec"]
                if (finfo["timestamp_sec"] - last_t) <= 3.0:
                    current_event.append((fk, finfo))
                else:
                    (events_night if is_night else events_day).append((vname, current_event))
                    current_event = [(fk, finfo)]

        if current_event:
            (events_night if is_night else events_day).append((vname, current_event))

    print(f"Clustered into {len(events_day)} daytime events and {len(events_night)} nighttime events.")

    def split_events(events, val_ratio=0.20):
        random.seed(42)
        from collections import defaultdict
        category_events = defaultdict(list)
        for item in events:
            vname, ev = item
            ev_cats = set()
            for fk, finfo in ev:
                ev_cats.update(finfo.get("categories", set()))
                if finfo.get("has_saleng"):
                    ev_cats.add("saleng")

            assigned = False
            for cat in ["bus", "saleng", "truck_trailer", "tuktuk", "songthaew", "pickup", "van"]:
                if cat in ev_cats:
                    category_events[cat].append(item)
                    assigned = True
                    break
            if not assigned:
                category_events["other"].append(item)

        train_evs, val_evs = [], []
        for cat, ev_list in category_events.items():
            shuffled = list(ev_list)
            random.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
            val_evs.extend(shuffled[:n_val])
            train_evs.extend(shuffled[n_val:])
        return train_evs, val_evs

    train_day_evs, val_day_evs = split_events(events_day, args.val_ratio)
    train_night_evs, val_night_evs = split_events(events_night, args.val_ratio)

    train_keys = set()
    val_keys = set()
    for _, ev in train_day_evs + train_night_evs:
        for fk, _ in ev:
            train_keys.add(fk)
    for _, ev in val_day_evs + val_night_evs:
        for fk, _ in ev:
            val_keys.add(fk)

    print(f"Partitioned: {len(train_keys)} train frames, {len(val_keys)} validation frames.")

    # 4. Replay Buffer Collection (Clean Negative/Positive Standard Feeds)
    replay_target = int(len(curated_frames) * args.replay_ratio)
    replay_frames = []
    valid_videos = [Path(v) for v in args.videos if Path(v).exists()]

    if replay_target > 0 and valid_videos:
        print(f"\n[Replay Buffer] Sampling {replay_target} clean frames across feeds (ratio={args.replay_ratio:.0%})...")
        samples_per_vid = max(1, int(np.ceil(replay_target / len(valid_videos))))

        for vpath in valid_videos:
            cap = cv2.VideoCapture(str(vpath))
            vname = vpath.stem
            v_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Exclude intervals where verified vehicles occurred
            excl = []
            for fk, finfo in curated_frames.items():
                if finfo["video_name"] == vname:
                    f_idx = finfo["frame_idx"]
                    excl.append((f_idx - int(fps * 5), f_idx + int(fps * 5)))

            cands = np.linspace(100, max(101, v_total - 100), samples_per_vid * 3, dtype=int)
            collected = 0

            for cand_f in cands:
                if any(s <= cand_f <= e for s, e in excl):
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cand_f))
                ret, frame = cap.read()
                if not ret:
                    continue

                res = teacher_yolo.predict(frame, conf=0.25, verbose=False, device=device)
                cand_annos = []
                for b in res[0].boxes:
                    c_id = int(b.cls[0].item())
                    if c_id in COCO_TEACHER_TO_CUSTOM:
                        conf = float(b.conf[0].item())
                        custom_cls = COCO_TEACHER_TO_CUSTOM[c_id]
                        if custom_cls == 0 and conf < 0.25:
                            continue
                        if custom_cls == 1 and conf < 0.25:
                            continue
                        if custom_cls == 2 and conf < 0.35:
                            continue
                        if custom_cls == 3 and conf < 0.45:
                            continue
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        cand_annos.append({
                            "class_id": custom_cls,
                            "xyxy": xyxy,
                            "conf": conf
                        })

                # Cross-class NMS on replay background boxes
                cand_annos.sort(key=lambda x: x["conf"], reverse=True)
                bg_annos = deduplicate_boxes(cand_annos, iou_thresh=0.55, iomin_thresh=0.85)

                if bg_annos:
                    replay_frames.append({
                        "video_name": vname,
                        "frame_idx": cand_f,
                        "frame_img": frame,
                        "annotations": bg_annos
                    })
                    collected += 1
                if collected >= samples_per_vid:
                    break
            cap.release()

        print(f"Collected {len(replay_frames)} clean replay buffer frames.")

    # 5. Write Samples with ROI Bounding Crop & Oversampling
    print("\n--- Exporting Final Dataset Samples ---")

    def write_sample(
        img: np.ndarray,
        out_img_dir: Path,
        out_lbl_dir: Path,
        stem: str,
        targets: List[Dict],
        bg: List[Dict],
        crop_box: Tuple[int, int, int, int]
    ):
        cx1, cy1, cx2, cy2 = crop_box
        cropped_img = img[cy1:cy2, cx1:cx2]
        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw <= 0 or ch <= 0 or cropped_img.size == 0:
            return

        # Target priority deduplication: targets first, then background
        # Safe thresholds prevent adjacent queued vehicles from erasing each other
        all_boxes = deduplicate_boxes(targets + bg, iou_thresh=0.55, iomin_thresh=0.85)

        lbl_lines = []
        for item in all_boxes:
            bx1, by1, bx2, by2 = item["xyxy"]
            # Shift coordinates relative to ROI crop box
            nx1 = max(0.0, bx1 - cx1)
            ny1 = max(0.0, by1 - cy1)
            nx2 = min(float(cw), bx2 - cx1)
            ny2 = min(float(ch), by2 - cy1)

            if (nx2 - nx1) >= 10 and (ny2 - ny1) >= 10:
                xc, yc, bw, bh = xyxy_to_yolo([nx1, ny1, nx2, ny2], cw, ch)
                lbl_lines.append(f"{item['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if lbl_lines:
            cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), cropped_img)
            with open(out_lbl_dir / f"{stem}.txt", "w") as lf:
                lf.write("\n".join(lbl_lines) + "\n")

    saleng_oversample_count = 0
    tuktuk_oversample_count = 0
    bus_oversample_count = 0
    truck_oversample_count = 0
    night_oversample_count = 0
    synth_ir_count = 0
    day_frames_for_synth = []

    for fk, finfo in curated_frames.items():
        src_path = finfo["raw_frame_path"]
        if not src_path.exists():
            continue
        full_img = cv2.imread(str(src_path))
        if full_img is None:
            continue

        is_train = (fk in train_keys)
        is_night = "_night" in finfo["video_name"]
        categories = finfo.get("categories", set())
        has_saleng = finfo.get("has_saleng", False)
        has_tuktuk = "tuktuk" in categories
        has_bus = "bus" in categories
        has_truck = "truck_trailer" in categories

        t_img_dir = img_train_dir if is_train else img_val_dir
        t_lbl_dir = lbl_train_dir if is_train else lbl_val_dir

        crop_box = get_roadway_crop_box(finfo["video_name"], finfo["img_width"], finfo["img_height"]) if args.roadway_crop else (0, 0, finfo["img_width"], finfo["img_height"])

        # Base sample
        write_sample(full_img, t_img_dir, t_lbl_dir, fk, finfo["targets"], finfo["background_vehicles"], crop_box)

        # 4x Oversampling for Saleng to solve extreme class imbalance
        if is_train and has_saleng:
            for s_idx in range(1, max(1, args.saleng_boost)):
                s_stem = f"{fk}_salengboost_{s_idx}"
                s_seed = abs(hash(f"{fk}_saleng_{s_idx}")) % (2**31)
                jittered = apply_night_jitter(full_img, seed=s_seed)
                write_sample(jittered, img_train_dir, lbl_train_dir, s_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
                saleng_oversample_count += 1

        # 2x Oversampling for Tuk-Tuk to strongly emphasize 3-wheeler detection
        if is_train and has_tuktuk:
            for tt_idx in range(1, max(1, args.tuktuk_boost)):
                tt_stem = f"{fk}_tuktukboost_{tt_idx}"
                tt_seed = abs(hash(f"{fk}_tuktuk_{tt_idx}")) % (2**31)
                jittered = apply_night_jitter(full_img, seed=tt_seed)
                write_sample(jittered, img_train_dir, lbl_train_dir, tt_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
                tuktuk_oversample_count += 1

        # 3x Oversampling for Bus minority class
        if is_train and has_bus:
            for b_idx in range(1, max(1, args.bus_boost)):
                b_stem = f"{fk}_busboost_{b_idx}"
                b_seed = abs(hash(f"{fk}_bus_{b_idx}")) % (2**31)
                jittered = apply_night_jitter(full_img, seed=b_seed)
                write_sample(jittered, img_train_dir, lbl_train_dir, b_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
                bus_oversample_count += 1

        # 3x Oversampling for Heavy Commercial Truck/Trailer minority class
        if is_train and has_truck:
            for tr_idx in range(1, max(1, args.truck_boost)):
                tr_stem = f"{fk}_truckboost_{tr_idx}"
                tr_seed = abs(hash(f"{fk}_truck_{tr_idx}")) % (2**31)
                jittered = apply_night_jitter(full_img, seed=tr_seed)
                write_sample(jittered, img_train_dir, lbl_train_dir, tr_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
                truck_oversample_count += 1

        # Nighttime Photometric Jitter
        if is_train and is_night:
            for n_idx in range(1, max(1, args.night_boost)):
                n_stem = f"{fk}_nightboost_{n_idx}"
                n_seed = abs(hash(f"{fk}_night_{n_idx}")) % (2**31)
                jittered = apply_night_jitter(full_img, seed=n_seed)
                write_sample(jittered, img_train_dir, lbl_train_dir, n_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
                night_oversample_count += 1
        elif is_train and not is_night:
            day_frames_for_synth.append((fk, finfo, full_img, crop_box))

    # Synthetic IR Frames
    random.seed(42)
    selected_synth = random.sample(day_frames_for_synth, min(args.synth_ir_count, len(day_frames_for_synth)))
    for fk, finfo, d_img, crop_box in selected_synth:
        gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
        gamma_table = np.array([((i / 255.0) ** 1.6) * 255 for i in range(256)]).astype("uint8")
        dark_gray = cv2.LUT(gray, gamma_table)
        synth_bgr = cv2.cvtColor(dark_gray, cv2.COLOR_GRAY2BGR)
        synth_stem = f"{fk}_synth_ir_night"
        write_sample(synth_bgr, img_train_dir, lbl_train_dir, synth_stem, finfo["targets"], finfo["background_vehicles"], crop_box)
        synth_ir_count += 1

    # Write Replay Buffer
    random.seed(42)
    random.shuffle(replay_frames)
    split_rep = int(len(replay_frames) * (1.0 - args.val_ratio))

    for idx, rep in enumerate(replay_frames):
        stem = f"replay_{rep['video_name']}_f{rep['frame_idx']:06d}"
        out_i = img_train_dir if idx < split_rep else img_val_dir
        out_l = lbl_train_dir if idx < split_rep else lbl_val_dir
        h, w = rep["frame_img"].shape[:2]
        crop_box = get_roadway_crop_box(rep["video_name"], w, h) if args.roadway_crop else (0, 0, w, h)
        write_sample(rep["frame_img"], out_i, out_l, stem, [], rep["annotations"], crop_box)

    print(f"\nApplied Data Augmentations:")
    print(f"  - Saleng minority oversample ({args.saleng_boost}x): +{saleng_oversample_count} frames")
    print(f"  - Tuk-tuk minority oversample ({args.tuktuk_boost}x): +{tuktuk_oversample_count} frames")
    print(f"  - Bus minority oversample ({args.bus_boost}x):    +{bus_oversample_count} frames")
    print(f"  - Heavy Truck oversample ({args.truck_boost}x):    +{truck_oversample_count} frames")
    print(f"  - Nighttime photometric jitter:       +{night_oversample_count} frames")
    print(f"  - Synthetic monochrome IR:            +{synth_ir_count} frames")
    print(f"  - Clean replay buffer frames:          +{len(replay_frames)} frames")

    # 6. Generate data.yaml (5-Class Standard)
    data_yaml_path = dataset_dir / "data.yaml"

    yaml_lines = [
        "# Autogenerated 5-Class Thai Traffic Dataset",
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:"
    ]
    for cid in sorted(CLASS_NAMES.keys()):
        yaml_lines.append(f"  {cid}: {CLASS_NAMES[cid]}")

    with open(data_yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    train_img_count = len(list(img_train_dir.glob("*.jpg")))
    val_img_count = len(list(img_val_dir.glob("*.jpg")))

    print(f"\n=======================================================")
    print(f"Unified 5-Class Dataset Compilation Complete!")
    print(f"  Dataset directory: {dataset_dir.resolve()}")
    print(f"  Train images:      {train_img_count}")
    print(f"  Val images:        {val_img_count}")
    print(f"  Config file:       {data_yaml_path}")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Compile Unified 5-Class Thai Traffic Dataset with YOLO26x Co-Training Teacher.")
    parser.add_argument("--dataset-dir", default="data/multiclass_dataset", help="Output YOLO dataset folder.")
    parser.add_argument("--teacher-model", default="yolo26x.pt", help="Teacher detector for co-annotation (default: yolo26x.pt).")
    parser.add_argument("--device", default="cuda:0", help="Inference device.")
    parser.add_argument("--teacher-conf", type=float, default=0.25, help="Teacher co-annotation confidence threshold.")
    parser.add_argument("--iou-suppress", type=float, default=0.55, help="IoU threshold for suppressing overlapping teacher boxes.")
    parser.add_argument("--iomin-suppress", type=float, default=0.85, help="IoMin threshold for suppressing nested teacher boxes (e.g. cab in trailer).")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation event split ratio.")
    parser.add_argument("--replay-ratio", type=float, default=0.08, help="Replay buffer size relative to curated frames.")
    parser.add_argument("--saleng-boost", type=int, default=4, help="Oversampling multiplier for saleng frames to fix class imbalance.")
    parser.add_argument("--tuktuk-boost", type=int, default=2, help="Oversampling multiplier for tuk-tuk frames to strongly emphasize 3-wheelers.")
    parser.add_argument("--bus-boost", type=int, default=3, help="Oversampling multiplier for bus frames to fix class imbalance.")
    parser.add_argument("--truck-boost", type=int, default=3, help="Oversampling multiplier for heavy truck/trailer frames.")
    parser.add_argument("--night-boost", type=int, default=3, help="Oversampling multiplier for real nighttime frames.")
    parser.add_argument("--synth-ir-count", type=int, default=40, help="Number of synthetic infrared monochrome frames.")
    parser.add_argument("--roadway-crop", action="store_true", default=True, help="Crop active roadway to avoid unannotated negative penalties.")
    parser.add_argument("--no-roadway-crop", dest="roadway_crop", action="store_false", help="Disable roadway cropping.")
    parser.add_argument("--videos", nargs="+", default=[
        "videos/cam44_north.avi",
        "videos/cam46_west.avi",
        "videos/cam43_south.avi",
        "videos/cam03_east.avi",
        "videos/cam44_north_night.avi",
        "videos/cam46_west_night.avi",
        "videos/cam43_south_night.avi",
        "videos/cam03_east_night.avi"
    ], help="Surveillance video paths.")

    args = parser.parse_args()
    run_compilation(args)


if __name__ == "__main__":
    main()

