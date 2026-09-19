"""
High-Performance Multi-Threaded Presentation Video Renderer.
Generates full 4-camera stitched comparison videos (Base YOLO26s vs Fine-Tuned Thai Traffic YOLO26s).
Features:
- Parallel multi-threaded video stream decoding (4 cameras concurrently)
- Batch GPU inference (RTX 5060 AMP)
- Asynchronous disk video writing queue
- Clean HUD banner with live vehicle statistics
"""

import argparse
import os
import sys
import time
from pathlib import Path
from queue import Queue
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Standard color palette (BGR)
CLASS_COLORS = {
    "car": (50, 205, 50),            # Lime / Vibrant Green (PCE 1.0)
    "motorcycle": (240, 200, 30),    # Cyan / Sky Blue (VRU)
    "bus": (230, 110, 20),           # Royal Blue
    "truck": (30, 130, 255),         # Deep Amber / Orange (Heavy commercial)
    "three_wheeler": (210, 40, 210), # Magenta / Vivid Purple (Tuk-tuk / Saleng)
}
DEFAULT_COLOR = (180, 180, 180)

CAM_LABELS = [
    "CAM-01 NORTH",
    "CAM-02 WEST",
    "CAM-03 SOUTH",
    "CAM-04 EAST"
]

VIDEO_SOURCES = [
    "videos/cam44_north.avi",
    "videos/cam46_west.avi",
    "videos/cam43_south.avi",
    "videos/cam03_east.avi"
]


class AsyncFrameReader:
    """Multi-threaded reader that concurrently decodes and pre-scales 4 video feeds."""
    def __init__(self, sources: List[str], target_size: Tuple[int, int] = (640, 360), queue_size: int = 32):
        self.sources = sources
        self.target_size = target_size
        self.queue: Queue = Queue(maxsize=queue_size)
        self.running = False
        self.caps = [cv2.VideoCapture(s) for s in sources]
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def _worker(self):
        while self.running:
            tiles = []
            valid = True
            for c in self.caps:
                ret, frame = c.read()
                if not ret:
                    c.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = c.read()
                if ret and frame is not None:
                    small = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                    tiles.append(small)
                else:
                    valid = False
                    break
            if valid and len(tiles) == len(self.sources):
                self.queue.put(tiles)
            else:
                time.sleep(0.01)

    def get(self) -> List[np.ndarray]:
        return self.queue.get()

    def stop(self):
        self.running = False
        for c in self.caps:
            c.release()


class AsyncVideoWriter:
    """Asynchronous video writer offloading disk I/O to a background thread."""
    def __init__(self, path: str, fps: int, frame_size: Tuple[int, int], queue_size: int = 64):
        self.path = path
        self.fps = fps
        self.frame_size = frame_size
        self.queue: Queue = Queue(maxsize=queue_size)
        self.running = False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
                self.writer.write(frame)
                self.queue.task_done()
            except Exception:
                continue

    def write(self, frame: np.ndarray):
        self.queue.put(frame)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=10.0)
        self.writer.release()


def draw_bounding_boxes(
    img: np.ndarray,
    boxes,
    class_names: Dict[int, str],
    conf_thresh: float = 0.15,
) -> Dict[str, int]:
    """Draw clean bounding boxes with colored label badges on a 640x360 tile."""
    counts: Dict[str, int] = {}
    h, w = img.shape[:2]

    for b in boxes:
        conf = float(b.conf[0].item())
        if conf < conf_thresh:
            continue
        cid = int(b.cls[0].item())
        cname = class_names.get(cid, str(cid))

        # Filter non-vehicle COCO classes for fair comparison
        if cname not in ("car", "motorcycle", "bus", "truck", "three_wheeler", "bicycle"):
            continue

        counts[cname] = counts.get(cname, 0) + 1
        color = CLASS_COLORS.get(cname, DEFAULT_COLOR)

        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Draw box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw label badge
        label = f"{cname} {int(conf * 100)}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tag_y1 = max(0, y1 - th - 5)
        tag_y2 = y1
        cv2.rectangle(img, (x1, tag_y1), (x1 + tw + 4, tag_y2), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, tag_y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return counts


def render_model_video(
    model_path: str,
    model_title: str,
    out_path: str,
    duration_sec: int = 600,
    fps: int = 25,
    conf: float = 0.15,
    device: str = "0",
) -> None:
    print(f"\n=======================================================")
    print(f"Rendering Presentation Video: {model_title}")
    print(f"  Model weights: {model_path}")
    print(f"  Output path:   {out_path}")
    print(f"  Duration:      {duration_sec // 60}m {duration_sec % 60}s ({duration_sec}s @ {fps} FPS)")
    print(f"  Total frames:  {duration_sec * fps}")
    print(f"=======================================================")

    model = YOLO(model_path)
    total_frames = duration_sec * fps

    # Tile size 640x360 -> Grid 1280x720 + 60px banner = 1280x780
    tile_w, tile_h = 640, 360
    banner_h = 60
    out_w, out_h = 1280, banner_h + (tile_h * 2)

    reader = AsyncFrameReader(VIDEO_SOURCES, target_size=(tile_w, tile_h), queue_size=32).start()
    writer = AsyncVideoWriter(out_path, fps=fps, frame_size=(out_w, out_h), queue_size=64).start()

    start_time = time.time()
    last_log = start_time

    for f_idx in range(total_frames):
        raw_tiles = reader.get()

        # Batch GPU inference on 4 pre-scaled tiles (RTX 5060)
        results = model.predict(
            raw_tiles,
            imgsz=640,
            conf=conf,
            verbose=False,
            device=device,
        )

        processed_tiles = []
        total_counts: Dict[str, int] = {}

        for cam_idx, res in enumerate(results):
            tile = raw_tiles[cam_idx]
            counts = draw_bounding_boxes(tile, res.boxes, model.names, conf_thresh=conf)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

            # Camera label overlay badge
            cam_badge = CAM_LABELS[cam_idx]
            cv2.rectangle(tile, (8, 8), (145, 32), (18, 18, 22), -1)
            cv2.rectangle(tile, (8, 8), (145, 32), (70, 70, 75), 1)
            cv2.putText(
                tile,
                cam_badge,
                (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            processed_tiles.append(tile)

        # Stitch 2x2 grid
        top_row = np.hstack([processed_tiles[0], processed_tiles[1]])
        bot_row = np.hstack([processed_tiles[2], processed_tiles[3]])
        grid = np.vstack([top_row, bot_row])

        # Render Header Banner
        banner = np.zeros((banner_h, out_w, 3), dtype=np.uint8)
        banner[:] = (18, 18, 22)
        cv2.line(banner, (0, banner_h - 1), (out_w, banner_h - 1), (45, 45, 50), 2)

        # Title
        cv2.putText(
            banner,
            model_title,
            (18, 38),
            cv2.FONT_HERSHEY_DUPLEX,
            0.68,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Elapsed timestamp
        cur_sec = f_idx // fps
        tot_sec = duration_sec
        time_str = f"Time: {cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}"
        cv2.putText(
            banner,
            time_str,
            (570, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        # Live vehicle stats badges
        c_car = total_counts.get("car", 0)
        c_moto = total_counts.get("motorcycle", 0)
        c_bus = total_counts.get("bus", 0)
        c_truck = total_counts.get("truck", 0)
        c_3w = total_counts.get("three_wheeler", 0)

        stats_text = f"Cars: {c_car} | Motos: {c_moto} | Trucks: {c_truck} | Buses: {c_bus} | 3-Whl: {c_3w}"
        cv2.putText(
            banner,
            stats_text,
            (780, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 220, 120),
            1,
            cv2.LINE_AA,
        )

        full_canvas = np.vstack([banner, grid])
        writer.write(full_canvas)

        now = time.time()
        if now - last_log >= 10.0 or f_idx == total_frames - 1:
            pct = (f_idx + 1) / total_frames * 100
            elapsed = now - start_time
            render_fps = (f_idx + 1) / max(0.001, elapsed)
            eta_sec = (total_frames - (f_idx + 1)) / max(0.001, render_fps)
            print(
                f"  [{f_idx + 1:05d}/{total_frames:05d}] {pct:5.1f}% | "
                f"Speed: {render_fps:5.1f} fps | "
                f"Elapsed: {elapsed / 60:4.1f}m | ETA: {eta_sec / 60:4.1f}m"
            )
            last_log = now

    reader.stop()
    writer.stop()

    total_sec = time.time() - start_time
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[DONE] Successfully generated: {out_path}")
    print(f"  File size: {file_size_mb:.1f} MB in {total_sec / 60:.1f} minutes.\n")


def main():
    parser = argparse.ArgumentParser(description="Render comparison videos for Base vs Fine-Tuned YOLO26s.")
    parser.add_argument("--duration-sec", type=int, default=600, help="Video duration in seconds (default: 600 = 10 minutes).")
    parser.add_argument("--fps", type=int, default=25, help="Output video frame rate (default: 25).")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold.")
    parser.add_argument("--out-dir", default="temp_presentation", help="Output directory for presentation videos.")
    parser.add_argument("--device", default="0", help="CUDA inference device.")
    parser.add_argument("--only-model", choices=["both", "base", "custom"], default="both", help="Which video to render.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_model in ("both", "base"):
        base_video_path = str(out_dir / "base_yolo26s_4cams.mp4")
        render_model_video(
            model_path="yolo26s.pt",
            model_title="BASE YOLO26s (COCO Standard)",
            out_path=base_video_path,
            duration_sec=args.duration_sec,
            fps=args.fps,
            conf=args.conf,
            device=args.device,
        )

    if args.only_model in ("both", "custom"):
        custom_video_path = str(out_dir / "thai_traffic_yolo26s_4cams.mp4")
        render_model_video(
            model_path="models/yolo26s_thai_traffic.pt",
            model_title="FINE-TUNED YOLO26s (Thai Traffic Adaptive Vision)",
            out_path=custom_video_path,
            duration_sec=args.duration_sec,
            fps=args.fps,
            conf=args.conf,
            device=args.device,
        )


if __name__ == "__main__":
    main()
