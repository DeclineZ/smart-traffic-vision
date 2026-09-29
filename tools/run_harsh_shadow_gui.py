"""
Interactive Visual Inspection GUI for Harsh Shadow Vehicle Counting.
Provides real-time interactive preview of:
  - Side-by-side or toggled view of Raw vs Shadow Contrast Equalized video.
  - Bounding boxes and bottom contact-patch anchor points.
  - Lane polygons with single-lane debounced occupancy.
  - Keyboard controls:
      's': Toggle Shadow Contrast Equalizer ON/OFF
      'c': Toggle Contact-Patch Anchoring ON/OFF
      'space': Pause/Play
      'q' or ESC: Exit
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon, Point
import shapely
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.shadow_processor import ShadowContrastEqualizer, ContactPatchRefiner, ShadowLaneAssigner, remap_shadow_detection, suppress_duplicate_shadow_boxes, COCO_VEHICLES
from algorithm.shadow_tracker import ShadowResilientTracker
from trt_pipeline.payload import LaneMetricsManager
from trt_pipeline.tools import initial_config


def run_gui(
    video_path: str,
    config_path: Optional[str] = None,
    model_path: str = "yolov8s.pt",
    device: str = "cpu",
    conf: float = 0.20,
):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 25.0

    print(f"\n=======================================================")
    print(f"HARSH SHADOW VISUAL INSPECTION GUI")
    print(f"Video: {video_path} ({w}x{h} @ {fps:.1f} FPS)")
    print(f"Controls: 's' = Toggle SCE | 'c' = Toggle Contact-Patch | 'space' = Pause | 'q' = Quit")
    print(f"=======================================================\n")

    # Load lane config or construct adaptive 3-lane grid
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lanes_source = cfg.get("lane_metrics", {}).get("lanes", cfg.get("lanes", {}))
        lanes_dict = {
            lid: {
                "direction": linfo.get("direction", "N"),
                "polygon": Polygon(linfo["polygon"]) if not isinstance(linfo["polygon"], Polygon) else linfo["polygon"],
            }
            for lid, linfo in lanes_source.items()
        }
    else:
        lane_w = w / 3.0
        lanes_dict = {
            "Lane_1": {"direction": "N", "polygon": Polygon([(0, 0), (lane_w, 0), (lane_w, h), (0, h)])},
            "Lane_2": {"direction": "N", "polygon": Polygon([(lane_w, 0), (2 * lane_w, 0), (2 * lane_w, h), (lane_w, h)])},
            "Lane_3": {"direction": "N", "polygon": Polygon([(2 * lane_w, 0), (w, 0), (w, h), (2 * lane_w, h)])},
        }

    print(f"Loading YOLO detector: {model_path} onto {device}...")
    model = YOLO(model_path)

    shadow_eq = ShadowContrastEqualizer(dynamic_trigger=False)
    contact_ref = ContactPatchRefiner(ground_offset_ratio=0.05, trim_shadow_wings=True)
    lane_assigner = ShadowLaneAssigner(hysteresis_frames=3)
    tracker = ShadowResilientTracker(det_thresh=0.35, min_conf=0.10, max_age=35, max_coast_frames=18, min_hits=2)
    metrics_mgr = LaneMetricsManager(lanes_dict)

    enable_sce = True
    enable_cp = True
    paused = False

    window_name = "Harsh Shadow Vehicle Counting HUD"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)

    frame_idx = 0
    fps_calc = 0.0
    t_prev = time.perf_counter()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_idx += 1

            # 1. Shadow Contrast Equalization
            if enable_sce:
                proc_frame, _ = shadow_eq.enhance(frame)
            else:
                proc_frame = frame

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # 2. Inference: run without hard class dropping so pillar-shadow confused vehicles are recovered
            res = model(proc_frame, conf=conf, verbose=False, device=device)[0]
            dets = []
            for box in res.boxes:
                b = box.xyxy[0].cpu().numpy()
                c = float(box.conf[0])
                raw_cls = int(box.cls[0])
                # Remap shadow confusion (e.g. dark car next to pillar misclassified as suitcase/chair)
                cls_id = remap_shadow_detection(raw_cls, b)
                if cls_id not in COCO_VEHICLES:
                    continue

                if enable_cp:
                    b = contact_ref.trim_lateral_cast_shadow(b, gray)
                dets.append([b[0], b[1], b[2], b[3], c, cls_id])

            dets_filtered = suppress_duplicate_shadow_boxes(dets, iou_thresh=0.35, ioa_thresh=0.60)
            tracked = tracker.update(dets_filtered)

            # 3. Lane assignment
            vis = frame.copy()
            if len(tracked) > 0:
                if enable_cp:
                    cxs, cys = contact_ref.get_contact_points_vectorized(tracked)
                else:
                    cxs = (tracked[:, 0] + tracked[:, 2]) * 0.5
                    cys = (tracked[:, 1] + tracked[:, 3]) * 0.5

                num_objs = len(tracked)
                obj_candidates = [None] * num_objs
                for lid, linfo in lanes_dict.items():
                    poly = linfo["polygon"]
                    try:
                        inside = shapely.contains_xy(poly, cxs, cys)
                    except AttributeError:
                        inside = np.array([poly.contains(Point(x, y)) for x, y in zip(cxs, cys)])

                    for idx in np.where(inside)[0]:
                        obj_candidates[idx] = lid

                for idx in range(num_objs):
                    tid = int(tracked[idx][4])
                    cls_id = int(tracked[idx][5]) if len(tracked[idx]) >= 6 else 2

                    confirmed_lane = lane_assigner.update_track_lane(
                        track_id=tid,
                        candidate_lane_id=obj_candidates[idx],
                        frame_idx=frame_idx,
                    )

                    if confirmed_lane is not None:
                        metrics_mgr.register_vehicle(lane_id=confirmed_lane, track_id=tid, vehicle_class=cls_id)

                    # Draw Bounding Box & Contact Point
                    bx1, by1, bx2, by2 = int(tracked[idx][0]), int(tracked[idx][1]), int(tracked[idx][2]), int(tracked[idx][3])
                    cname = COCO_VEHICLES.get(cls_id, "car")
                    cv.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 120), 2)

                    # Contact patch anchor (yellow circle)
                    cv.circle(vis, (int(cxs[idx]), int(cys[idx])), 5, (0, 255, 255), -1)
                    cv.circle(vis, (int(cxs[idx]), int(cys[idx])), 6, (0, 0, 0), 1)

                    tag = f"#{tid} {cname} [{confirmed_lane or '?'}]"
                    cv.putText(vis, tag, (bx1, max(15, by1 - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

            # Draw Lane Polygons
            for lid, linfo in lanes_dict.items():
                poly = linfo["polygon"]
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv.polylines(vis, [pts], True, (0, 165, 255), 2)
                first_pt = (int(pts[0][0]), max(20, int(pts[0][1])))
                cv.putText(vis, lid, first_pt, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)

            # Draw Top HUD Banner
            now = time.perf_counter()
            fps_calc = 1.0 / max(1e-4, now - t_prev)
            t_prev = now

            hud_bar = np.zeros((36, vis.shape[1], 3), dtype=np.uint8)
            hud_text = (
                f"FRAME: {frame_idx:05d} | FPS: {fps_calc:.1f} | "
                f"SCE [S]: {'ON' if enable_sce else 'OFF'} | "
                f"CONTACT-PATCH [C]: {'ON' if enable_cp else 'OFF'} | "
                f"ACTIVE TRACKS: {len(tracked)}"
            )
            cv.putText(hud_bar, hud_text, (15, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv.LINE_AA)
            display_frame = np.vstack([hud_bar, vis])

            cv.imshow(window_name, display_frame)

        key = cv.waitKey(1 if not paused else 30) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            enable_sce = not enable_sce
            print(f"Shadow Contrast Equalizer toggled: {'ON' if enable_sce else 'OFF'}")
        elif key == ord("c"):
            enable_cp = not enable_cp
            print(f"Contact-Patch Anchoring toggled: {'ON' if enable_cp else 'OFF'}")
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Harsh Shadow Visual Inspection GUI")
    parser.add_argument("--video", default="videos/dry/gettyimages-151939150-640_adpp.mp4", help="Video path")
    parser.add_argument("--config", default=None, help="Camera JSON configuration")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--device", default=None, help="Compute device (auto, cuda:0, mps, cpu)")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    args = parser.parse_args()

    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    run_gui(
        video_path=args.video,
        config_path=args.config,
        model_path=args.model,
        device=device,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
