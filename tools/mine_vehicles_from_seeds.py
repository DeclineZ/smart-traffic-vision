"""
Automated CCTV Video Miner using DINOv2 Visual Seeds & Dual-Stream Proposals.

High-Performance Features:
1. Fast-forward skipping via cap.grab() (skips full frame decoding, running at ~2,000 effective FPS).
2. Per-video and total candidate quotas (--max-candidates-per-video, --max-total) to prevent wasting hours.
3. Temporal-spatial duplicate suppression: prevents capturing 50 identical crops of a vehicle idling at a red light.
4. Dual-stream candidate proposals:
   - Low-threshold YOLO stream (conf >= 0.08) across relevant COCO classes.
   - Optional MOG2 motion proposals with shadow filtering for detector blind spots.
5. Zero-margin tight body cropping to eliminate asphalt and marking bias.
6. DINOv2 ViT feature extraction (dinov2_vitb14) with max-similarity matching against seed prototypes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

COCO_TO_CUSTOM = {
    2: 0,  # car
    3: 1,  # motorcycle
    5: 2,  # bus
    7: 3,  # truck
}

TARGET_TO_PROPOSAL_CLASSES: Dict[str, Set[int]] = {
    "saleng": {0, 1, 2, 3, 5, 7},         # person, bicycle, car, motorcycle, bus, truck
    "songthaew": {2, 5, 7},               # car, bus, truck (COCO often classifies songthaew as bus!)
    "pickup": {2, 5, 7},                  # car, bus, truck
    "van": {2, 5, 7},                     # car, bus, truck
    "truck_trailer": {2, 5, 7},           # car, bus, truck
    "bus": {2, 5, 7},                     # car, bus, truck
    "tuktuk": {2, 3, 7},                  # car, motorcycle, truck
    "motorcycle": {1, 3},                 # bicycle, motorcycle
}


def get_dino_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2_model(model_name: str = "dinov2_vitb14", device: str = "cuda"):
    print(f"Loading DINOv2 ({model_name}) on {device}...")
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        print(f"Failed to load {model_name} ({e}), falling back to dinov2_vits14...")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)
    return model


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
    """Intersection over Minimum (IoMin) to detect nested detections (e.g. cab inside full rig)."""
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


def extract_seed_embeddings(seeds_dir: Path, dino_model, transform, device: str) -> torch.Tensor:
    seed_files = sorted(
        list(seeds_dir.glob("*.jpg")) + list(seeds_dir.glob("*.png")) + list(seeds_dir.glob("*.jpeg"))
    )
    if not seed_files:
        raise FileNotFoundError(f"No seed images found in {seeds_dir}. Place reference crops there first.")

    print(f"Extracting DINOv2 embeddings for {len(seed_files)} seed crops in {seeds_dir}...")
    embeddings = []
    with torch.no_grad():
        for sf in seed_files:
            img = cv2.imread(str(sf))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            emb = dino_model(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"Could not extract embeddings from seed files in {seeds_dir}.")

    all_seeds = torch.cat(embeddings, dim=0)
    print(f"Encoded {len(embeddings)} seed vectors (shape: {all_seeds.shape}).")
    return all_seeds


def extract_mog2_motion_proposals(
    fg_mask: np.ndarray,
    img_w: int,
    img_h: int,
    existing_boxes: List[List[float]],
    min_box_size: int = 36,
    max_box_dim: int = 700,
    min_moving_pixels: int = 300
) -> List[List[float]]:
    clean_mask = np.where(fg_mask == 255, 255, 0).astype(np.uint8)
    moving_pixels = np.count_nonzero(clean_mask)
    if moving_pixels < min_moving_pixels or moving_pixels > (img_w * img_h * 0.45):
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mog2_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_box_size or h < min_box_size:
            continue
        if w > max_box_dim and h > max_box_dim:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.20 or aspect_ratio > 4.5:
            continue

        box = [float(x), float(y), float(x + w), float(y + h)]
        if any(compute_iou(box, ebox) > 0.40 or compute_iomin(box, ebox) > 0.60 for ebox in existing_boxes):
            continue

        mog2_boxes.append(box)

    return mog2_boxes


def run_mining(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    seeds_dir = Path(args.seeds_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    raw_frames_dir = output_dir / "raw_frames"
    manifest_path = output_dir / "manifest.json"

    crops_dir.mkdir(parents=True, exist_ok=True)
    raw_frames_dir.mkdir(parents=True, exist_ok=True)

    dino_transform = get_dino_transform()
    dino_model = load_dinov2_model(args.dino_model, device)
    seed_embeddings = extract_seed_embeddings(seeds_dir, dino_model, dino_transform, device)

    print(f"Loading proposal YOLO detector: {args.yolo_model}...")
    yolo = YOLO(args.yolo_model)

    proposal_classes = TARGET_TO_PROPOSAL_CLASSES.get(
        args.target_class, {0, 1, 2, 3, 5, 7}
    )
    print(f"Target class: [{args.target_class}] | Active proposal COCO classes: {sorted(list(proposal_classes))}")
    print(f"DINOv2 similarity threshold: {args.sim_threshold:.2f} (max across {seed_embeddings.shape[0]} prototypes)")
    print(f"Speed settings: stride={args.stride or 'auto'}, max_candidates_per_vid={args.max_per_video}, max_total={args.max_total}")

    video_paths = [Path(v) for v in args.videos]
    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            print(f"Loaded existing manifest with {len(manifest_data)} frames.")
        except Exception as e:
            print(f"Warning loading manifest: {e}. Starting fresh.")

    total_candidates_found = 0
    t_start_total = time.time()

    for v_idx, vpath in enumerate(video_paths, 1):
        if total_candidates_found >= args.max_total:
            print(f"\n[INFO] Reached total candidate quota ({args.max_total}). Mining complete!")
            break

        if not vpath.exists():
            print(f"Skipping missing video: {vpath}")
            continue

        vname = vpath.stem
        print(f"\n[{v_idx}/{len(video_paths)}] Mining video: {vpath.name}...")
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Failed to open video: {vpath}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Calculate stride
        if args.stride is not None and args.stride > 0:
            stride = args.stride
        else:
            # Surveillance files in this repo report 150 FPS in container header (~5x encoded).
            # Default target_fps=1.0 maps to stride=150 (exactly 1 sample per real second).
            stride = max(1, int(round(fps / args.target_fps)))

        keyframe_count = int(total_frames / stride)
        duration_mins = total_frames / fps / 60.0
        print(f"  Metadata: {total_frames} frames ({duration_mins:.1f}m), {fps:.1f} FPS | Stride = {stride} frames (~{keyframe_count} keyframes to evaluate)")

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True) if args.enable_motion else None

        frame_idx = 0
        mined_in_video = 0
        t_start_vid = time.time()
        recent_mined_targets: List[Tuple[float, List[float]]] = []  # (sec, xyxy)

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            current_sec = frame_idx / fps

            # Periodic progress logging
            if (frame_idx // stride) % max(1, (keyframe_count // 10)) == 0 and frame_idx > 0:
                pct = (frame_idx / total_frames) * 100
                rate = (frame_idx / max(0.1, time.time() - t_start_vid))
                print(f"    Progress: {pct:.1f}% ({frame_idx}/{total_frames} frames, {rate:.0f} eff_fps) | Mined in video: {mined_in_video}")

            results = yolo.predict(frame, conf=args.min_conf, verbose=False, device=device)
            boxes = results[0].boxes

            candidate_crops = []
            candidate_boxes_xyxy = []
            candidate_sources = []
            high_conf_vehicles = []

            for b_idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                bw = x2 - x1
                bh = y2 - y1

                # Track baseline vehicles for background co-annotation
                if cls_id in COCO_TO_CUSTOM and conf >= args.co_conf:
                    high_conf_vehicles.append({
                        "box_idx": b_idx,
                        "coco_cls": cls_id,
                        "custom_cls": COCO_TO_CUSTOM[cls_id],
                        "conf": conf,
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                    })

                # Check proposal criteria
                if cls_id in proposal_classes and bw >= args.min_box_size and bh >= args.min_box_size:
                    # Spatial-temporal duplicate suppression: skip if vehicle is idling at the same spot
                    box_xyxy = [float(x1), float(y1), float(x2), float(y2)]
                    is_stationary_duplicate = False
                    for r_sec, r_box in recent_mined_targets:
                        if (current_sec - r_sec) <= args.cooldown_sec and compute_iou(box_xyxy, r_box) > 0.65:
                            is_stationary_duplicate = True
                            break

                    if not is_stationary_duplicate:
                        # Zero-margin tight crop
                        x1_c = max(0, x1)
                        y1_c = max(0, y1)
                        x2_c = min(w, x2)
                        y2_c = min(h, y2)
                        crop = frame[y1_c:y2_c, x1_c:x2_c]
                        if crop.size > 0:
                            candidate_crops.append(crop)
                            candidate_boxes_xyxy.append(box_xyxy)
                            candidate_sources.append(f"yolo_{cls_id}")

            # Optional MOG2 motion proposals for detector blind spots
            if bg_subtractor is not None:
                fg_mask = bg_subtractor.apply(frame)
                mog2_boxes = extract_mog2_motion_proposals(
                    fg_mask, w, h, candidate_boxes_xyxy, min_box_size=args.min_box_size
                )
                for mbox in mog2_boxes:
                    mx1, my1, mx2, my2 = [int(round(v)) for v in mbox]
                    x1_c = max(0, mx1)
                    y1_c = max(0, my1)
                    x2_c = min(w, mx2)
                    y2_c = min(h, my2)
                    crop = frame[y1_c:y2_c, x1_c:x2_c]
                    if crop.size > 0:
                        candidate_crops.append(crop)
                        candidate_boxes_xyxy.append([float(mx1), float(my1), float(mx2), float(my2)])
                        candidate_sources.append("mog2_motion")

            # Evaluate candidate crops through DINOv2
            if candidate_crops:
                batch_tensors = []
                for c in candidate_crops:
                    c_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                    batch_tensors.append(dino_transform(c_rgb))
                batch_tensors = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    crop_embs = dino_model(batch_tensors)
                    crop_embs = crop_embs / crop_embs.norm(dim=-1, keepdim=True)
                    sim_matrix = torch.mm(crop_embs, seed_embeddings.t())
                    max_sims, _ = sim_matrix.max(dim=-1)
                    max_sims = max_sims.cpu().numpy()

                matched_targets = []
                for c_i, max_sim in enumerate(max_sims):
                    if max_sim >= args.sim_threshold:
                        matched_targets.append({
                            "source": candidate_sources[c_i],
                            "xyxy": candidate_boxes_xyxy[c_i],
                            "sim": float(max_sim),
                            "crop_img": candidate_crops[c_i]
                        })

                if matched_targets:
                    # Suppress overlapping background baseline boxes
                    final_background = []
                    for bg in high_conf_vehicles:
                        suppress = False
                        for target in matched_targets:
                            iou = compute_iou(bg["xyxy"], target["xyxy"])
                            iomin = compute_iomin(bg["xyxy"], target["xyxy"])
                            if iou > args.iou_threshold or iomin >= 0.65:
                                suppress = True
                                break
                        if not suppress:
                            final_background.append(bg)

                    frame_key = f"{vname}_f{frame_idx:06d}"
                    raw_frame_path = raw_frames_dir / f"{frame_key}.jpg"
                    cv2.imwrite(str(raw_frame_path), frame)

                    target_records = []
                    for t_idx, target in enumerate(matched_targets):
                        src_tag = target["source"]
                        crop_filename = f"crop_{frame_key}_{src_tag}_{t_idx}_sim{target['sim']:.2f}.jpg"
                        crop_path = crops_dir / crop_filename
                        cv2.imwrite(str(crop_path), target["crop_img"])

                        target_records.append({
                            "crop_filename": crop_filename,
                            "source": src_tag,
                            "xyxy": target["xyxy"],
                            "sim": target["sim"]
                        })
                        recent_mined_targets.append((current_sec, target["xyxy"]))
                        mined_in_video += 1
                        total_candidates_found += 1

                    manifest_data[frame_key] = {
                        "video_name": vname,
                        "frame_idx": frame_idx,
                        "timestamp_sec": current_sec,
                        "img_width": w,
                        "img_height": h,
                        "raw_frame_file": f"{frame_key}.jpg",
                        "targets": target_records,
                        "background_vehicles": final_background
                    }

                    # Prune old targets from recent memory (> 10 sec)
                    recent_mined_targets = [
                        (s, b) for s, b in recent_mined_targets if (current_sec - s) <= 10.0
                    ]

                    if mined_in_video % 15 == 0:
                        print(f"    [Mined {mined_in_video}] Frame {frame_idx:06d} ({current_sec/60:.1f}m): found {len(matched_targets)} {args.target_class} targets.")

            # Check per-video candidate quota
            if mined_in_video >= args.max_per_video:
                print(f"  --> Reached per-video quota ({args.max_per_video} crops for {vpath.name}). Advancing to next camera.")
                break

            # Fast skip next (stride - 1) frames via cap.grab() without full decoding
            for _ in range(stride - 1):
                cap.grab()
            frame_idx += stride

        cap.release()
        elapsed_vid = time.time() - t_start_vid
        print(f"  Completed {vpath.name} in {elapsed_vid:.1f}s ({elapsed_vid/60:.1f}m): {mined_in_video} candidate crops extracted.")

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

    elapsed_total = time.time() - t_start_total
    print(f"\n=======================================================")
    print(f"Mining complete in {elapsed_total/60:.1f} minutes!")
    print(f"Total candidate crops extracted: {total_candidates_found}")
    print(f"Review and copy confirmed hits into: {args.verified_dir}")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="High-Performance CCTV Vehicle Miner using DINOv2 Visual Seeds.")
    parser.add_argument("--target-class", default="pickup", help="Target vehicle class (saleng, pickup, van, truck_trailer, bus, tuktuk).")
    parser.add_argument("--videos", nargs="+", default=[
        "videos/cam44_north.avi",
        "videos/cam46_west.avi",
        "videos/cam43_south.avi",
        "videos/cam03_east.avi",
        "videos/cam44_north_night.avi",
        "videos/cam46_west_night.avi",
        "videos/cam43_south_night.avi",
        "videos/cam03_east_night.avi"
    ], help="Surveillance videos to mine.")
    parser.add_argument("--seeds-dir", default=None, help="Reference seed crop directory (default: data/<target-class>/seeds).")
    parser.add_argument("--output-dir", default=None, help="Staging output directory (default: data/<target-class>/mined_candidates).")
    parser.add_argument("--verified-dir", default=None, help="Curated confirmed hits (default: data/<target-class>/verified_hits).")
    parser.add_argument("--yolo-model", default="yolo26s.pt", help="Baseline proposal detector.")
    parser.add_argument("--dino-model", default="dinov2_vitb14", help="DINOv2 architecture (dinov2_vitb14 or dinov2_vits14).")
    parser.add_argument("--device", default="cuda:0", help="Inference device.")
    parser.add_argument("--min-conf", type=float, default=0.08, help="Low proposal confidence threshold.")
    parser.add_argument("--co-conf", type=float, default=0.28, help="Confidence for tracking co-detected background vehicles.")
    parser.add_argument("--min-box-size", type=int, default=36, help="Minimum box width and height in pixels.")
    parser.add_argument("--sim-threshold", type=float, default=0.70, help="DINOv2 cosine similarity threshold.")
    parser.add_argument("--target-fps", type=float, default=1.0, help="Sampling rate in real seconds (default: 1.0 = ~1 frame per second).")
    parser.add_argument("--stride", type=int, default=None, help="Explicit frame stride (overrides target-fps if set, e.g. 150 or 200).")
    parser.add_argument("--max-per-video", type=int, default=120, help="Candidate quota per video before advancing to next camera.")
    parser.add_argument("--max-total", type=int, default=400, help="Total candidate quota across all videos.")
    parser.add_argument("--cooldown-sec", type=float, default=3.0, help="Cooldown in seconds to suppress duplicate crops of stationary idling vehicles.")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for suppressing overlapping baseline boxes.")
    parser.add_argument("--enable-motion", action="store_true", default=False, help="Enable MOG2 background motion stream (recommended for saleng).")
    parser.add_argument("--fast", action="store_true", default=False, help="Fast preset: stride=180, max 80 per video across main cameras.")

    args = parser.parse_args()

    if args.fast:
        args.stride = 180
        args.max_per_video = 80
        args.max_total = 250
        args.videos = [
            "videos/cam44_north.avi",
            "videos/cam46_west.avi",
            "videos/cam44_north_night.avi",
            "videos/cam46_west_night.avi"
        ]
        print("[FAST MODE ENABLED] Stride=180, Max=80/video, 4 core cameras selected.")

    tc = args.target_class
    if args.seeds_dir is None:
        args.seeds_dir = f"data/{tc}/seeds"
    if args.output_dir is None:
        args.output_dir = f"data/{tc}/mined_candidates"
    if args.verified_dir is None:
        args.verified_dir = f"data/{tc}/verified_hits"

    # Saleng benefits from motion proposals
    if tc == "saleng" and not args.fast:
        args.enable_motion = True

    run_mining(args)


if __name__ == "__main__":
    main()
