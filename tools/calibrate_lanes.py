"""
Interactive Mouse-Driven Road Lane Calibration Tool.
Allows users to visually click 4 corners (polygon) along real road lane markings,
preview lane boundaries, and save directly to standard JSON config format.

Controls:
  - Left Mouse Click : Add lane polygon vertex (4 points per lane).
  - 'u' or Backspace : Undo last clicked point.
  - 'n' or Enter     : Complete current lane and start next lane.
  - 'd'              : Delete the last completed lane.
  - 'f' or Right     : Forward 15 frames (find a frame with clear lane markings).
  - 'b' or Left      : Backward 15 frames.
  - 'r'              : Reset all lanes.
  - 's'              : Save to JSON configuration file.
  - 'q' or ESC       : Quit.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Any
import cv2 as cv
import numpy as np

# Distinct colors for previewing different lanes
LANE_PALETTE = [
    (0, 255, 120),   # Green
    (255, 180, 0),   # Cyan/Blue
    (0, 165, 255),   # Orange
    (255, 0, 200),   # Magenta
    (0, 255, 255),   # Yellow
    (180, 100, 255), # Purple
    (50, 200, 255),  # Gold
    (255, 100, 100), # Sky blue
]


class InteractiveLaneCalibrator:
    def __init__(self, video_path: str, output_path: str, frame_idx: int = 0):
        self.video_path = video_path
        self.output_path = output_path
        self.current_frame_idx = frame_idx

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.video_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS) or 25.0

        # Calibration state
        # completed_lanes: list of {"name": str, "points": [[x, y], ...], "direction": str}
        self.completed_lanes: List[Dict[str, Any]] = []
        # current_points: list of [x, y] for current lane being drawn
        self.current_points: List[List[int]] = []

        self.current_raw_frame = None
        self.window_name = f"Road Lane Calibrator - {os.path.basename(video_path)}"

        # Display window scaling
        self.disp_w = min(1440, max(960, self.video_w))
        self.disp_h = int(self.disp_w * (self.video_h / self.video_w))

        self.scale_x = self.video_w / float(self.disp_w)
        self.scale_y = self.video_h / float(self.disp_h)

        self._load_frame(self.current_frame_idx)

    def _load_frame(self, frame_number: int) -> bool:
        frame_number = max(0, min(self.total_frames - 1, frame_number))
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            self.current_raw_frame = frame
            self.current_frame_idx = frame_number
            return True
        return False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Map display click coordinates to actual video frame coordinates
            orig_x = int(round(x * self.scale_x))
            orig_y = int(round(y * self.scale_y))

            # Clamp coordinates
            orig_x = max(0, min(self.video_w - 1, orig_x))
            orig_y = max(0, min(self.video_h - 1, orig_y))

            self.current_points.append([orig_x, orig_y])
            print(f"Added vertex {len(self.current_points)} for Lane {len(self.completed_lanes) + 1}: ({orig_x}, {orig_y})")

            # Automatically commit lane when 4 points are clicked
            if len(self.current_points) == 4:
                self._commit_current_lane()

    def _commit_current_lane(self):
        if len(self.current_points) >= 3:
            lane_idx = len(self.completed_lanes) + 1
            lane_name = f"Lane_{lane_idx}"
            self.completed_lanes.append({
                "name": lane_name,
                "direction": "N",
                "description": f"Calibrated {lane_name}",
                "points": list(self.current_points),
            })
            print(f"\n[OK] Confirmed {lane_name} with {len(self.current_points)} vertices!")
            print(f"-> Ready for Lane_{lane_idx + 1}. Click 4 points or press [S] to Save.\n")
            self.current_points = []
        else:
            print("Notice: A lane polygon requires at least 3 or 4 points.")

    def render_overlay(self) -> np.ndarray:
        if self.current_raw_frame is None:
            return np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)

        # Scale down/up raw frame to display window
        disp = cv.resize(self.current_raw_frame, (self.disp_w, self.disp_h))
        overlay = disp.copy()

        # 1. Draw Completed Lanes with Alpha Transparency
        for idx, lane in enumerate(self.completed_lanes):
            pts_orig = lane["points"]
            pts_disp = np.array(
                [[int(px / self.scale_x), int(py / self.scale_y)] for px, py in pts_orig],
                dtype=np.int32,
            )
            color = LANE_PALETTE[idx % len(LANE_PALETTE)]

            # Draw filled semi-transparent polygon
            cv.fillPoly(overlay, [pts_disp], color)
            # Draw solid border
            cv.polylines(disp, [pts_disp], True, color, 2, cv.LINE_AA)

            # Label lane name at top corner
            top_pt = min(pts_disp, key=lambda p: p[1])
            label_pos = (max(10, top_pt[0] - 20), max(25, top_pt[1] - 8))
            cv.putText(disp, lane["name"], label_pos, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(disp, lane["name"], label_pos, cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)

        # Blend overlay (alpha = 0.25)
        disp = cv.addWeighted(overlay, 0.25, disp, 0.75, 0)

        # 2. Draw Points & Edges for Current Lane being Calibrated
        curr_lane_idx = len(self.completed_lanes)
        curr_color = LANE_PALETTE[curr_lane_idx % len(LANE_PALETTE)]

        if self.current_points:
            disp_pts = [
                (int(px / self.scale_x), int(py / self.scale_y))
                for px, py in self.current_points
            ]

            # Draw lines connecting clicked points
            for i in range(len(disp_pts) - 1):
                cv.line(disp, disp_pts[i], disp_pts[i + 1], curr_color, 2, cv.LINE_AA)

            # Draw vertices with index numbers
            for i, pt in enumerate(disp_pts):
                cv.circle(disp, pt, 6, (0, 255, 255), -1)
                cv.circle(disp, pt, 7, (0, 0, 0), 1)
                cv.putText(disp, str(i + 1), (pt[0] + 8, pt[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv.LINE_AA)

        # 3. Top Status HUD Bar
        hud_bar = np.zeros((40, self.disp_w, 3), dtype=np.uint8)
        status_text = (
            f"FRAME: {self.current_frame_idx:05d}/{self.total_frames:05d} | "
            f"LANES: {len(self.completed_lanes)} | "
            f"CURRENT: Lane_{len(self.completed_lanes)+1} ({len(self.current_points)}/4 pts) | "
            f"[S]=Save [U]=Undo [F/B]=Seek [Q]=Exit"
        )
        cv.putText(hud_bar, status_text, (15, 26), cv.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1, cv.LINE_AA)

        # 4. Bottom Instructions Banner
        bottom_bar = np.zeros((30, self.disp_w, 3), dtype=np.uint8)
        inst_text = "Click 4 corners: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left along lane markings"
        cv.putText(bottom_bar, inst_text, (15, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv.LINE_AA)

        return np.vstack([hud_bar, disp, bottom_bar])

    def save_config(self) -> str:
        """Saves calibrated lane polygons into standard configuration JSON."""
        if not self.completed_lanes:
            print("Warning: No lanes to save.")
            return ""

        # Prepare lanes dictionary matching smart-traffic-vision schema
        lanes_dict = {}
        for lane in self.completed_lanes:
            lanes_dict[lane["name"]] = {
                "direction": lane["direction"],
                "description": lane["description"],
                "polygon": lane["points"],
            }

        config_data = {
            "_comment": f"=== Calibrated Road Lanes for {os.path.basename(self.video_path)} ===",
            "camera_info": {
                "camera_id": "CAM-CALIBRATED",
                "location": "Custom Intersection / Approach",
                "intersection_id": "INT-001",
                "description": f"Calibrated from {os.path.basename(self.video_path)}"
            },
            "video": {
                "path": self.video_path,
                "skip": 0,
                "max_frames": 0
            },
            "model": {
                "engine_path": "models/yolov8s.engine",
                "input_shape": [1, 3, 640, 640],
                "device": "cuda:0"
            },
            "tracker": {
                "type": "ByteTrack",
                "params": {
                    "min_conf": 0.3,
                    "track_thresh": 0.5,
                    "match_thresh": 0.7,
                    "track_buffer": 25,
                    "frame_rate": 25
                }
            },
            "classes": {
                "dict_class": {
                    "1": "bicycle",
                    "2": "car",
                    "3": "motorcycle",
                    "5": "bus",
                    "7": "truck"
                }
            },
            "output": {
                "base_dir": "./output/calibrated",
                "save_crop": False,
                "data_output_path": "output/calibrated/data.json"
            },
            "tracking": {
                "zones": []
            },
            "density": {
                "enabled": False
            },
            "processing": {
                "opencv_threads": 0,
                "use_opencl": False,
                "video_queue_size": 2
            },
            "lane_metrics": {
                "enabled": True,
                "publish_interval_frames": 50,
                "queue_speed_threshold": 2.0,
                "lanes": lanes_dict
            }
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        print("\n" + "=" * 65)
        print(f"[SUCCESS] Calibrated configuration saved successfully!")
        print(f"Target file: {os.path.abspath(self.output_path)}")
        print(f"Total lanes configured: {len(self.completed_lanes)}")
        print("=" * 65)
        print("\nYou can now test your calibrated lanes immediately:")
        print(f"1. Run GUI preview:")
        print(f"   .venv/bin/python tools/run_harsh_shadow_gui.py --video \"{self.video_path}\" --config \"{self.output_path}\"")
        print(f"2. Run Benchmark:")
        print(f"   .venv/bin/python tools/test_harsh_shadow_counter.py --video \"{self.video_path}\" --config \"{self.output_path}\"")
        print("=" * 65 + "\n")

        return self.output_path

    def run(self):
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, self.disp_w, self.disp_h + 70)
        cv.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n=======================================================")
        print(f"INTERACTIVE LANE CALIBRATOR STARTED")
        print(f"Video: {self.video_path} ({self.video_w}x{self.video_h})")
        print(f"Saving to: {self.output_path}")
        print(f"Instructions: Click 4 corners for each lane. Press 's' to Save.")
        print(f"=======================================================\n")

        while True:
            display_img = self.render_overlay()
            cv.imshow(self.window_name, display_img)

            key = cv.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("u") or key == 8:  # 'u' or Backspace: undo point
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"Undid vertex: {removed}")
            elif key == ord("n") or key == 13:  # 'n' or Enter: commit current lane
                self._commit_current_lane()
            elif key == ord("d"):  # 'd': delete last completed lane
                if self.completed_lanes:
                    deleted = self.completed_lanes.pop()
                    print(f"Deleted {deleted['name']}.")
            elif key == ord("r"):  # 'r': reset all lanes
                self.completed_lanes = []
                self.current_points = []
                print("Reset all calibrated lanes.")
            elif key == ord("f") or key == ord(" "):  # 'f' or Space: forward 15 frames
                self._load_frame(self.current_frame_idx + 15)
            elif key == ord("b"):  # 'b': backward 15 frames
                self._load_frame(self.current_frame_idx - 15)
            elif key == ord("s"):  # 's': save to config JSON
                if self.current_points:
                    self._commit_current_lane()
                self.save_config()
                # Flash confirmation on screen
                conf_hud = display_img.copy()
                cv.rectangle(conf_hud, (self.disp_w // 2 - 250, self.disp_h // 2 - 40), (self.disp_w // 2 + 250, self.disp_h // 2 + 40), (0, 180, 0), -1)
                cv.putText(conf_hud, "CONFIG SAVED SUCCESSFULLY!", (self.disp_w // 2 - 220, self.disp_h // 2 + 8), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow(self.window_name, conf_hud)
                cv.waitKey(1000)

        self.cap.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive Road Lane Calibration Tool")
    parser.add_argument("--video", default="videos/dry/gettyimages-151939150-640_adpp.mp4", help="Path to video file")
    parser.add_argument("--output", default="config/custom_lanes.json", help="Path to output JSON config")
    parser.add_argument("--frame-idx", type=int, default=0, help="Starting frame index")
    args = parser.parse_args()

    calibrator = InteractiveLaneCalibrator(
        video_path=args.video,
        output_path=args.output,
        frame_idx=args.frame_idx,
    )
    calibrator.run()


if __name__ == "__main__":
    main()
