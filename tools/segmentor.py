"""
Interactive Lane Polygon & Virtual Counting Gate Calibration Suite.

Provides a unified, user-friendly GUI tool to:
1. Select camera approach / direction (North, South, East, West, Northeast) or custom video.
2. Interactively draw lane polygons and directed virtual counting gates (Stopline, Ingress).
3. Scrub video frames (+5s / -5s) to calibrate on clear, unobscured frames.
4. Automatically persist calibrated geometry directly into the camera's config JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.lane_validation import validate_camera_lanes

# Camera Approach Presets & Configuration Paths
APPROACH_CONFIGS = {
    "north": {
        "name": "North Approach (CAM-01 / CAM-44)",
        "camera_id": "CAM-01",
        "video": "./videos/cam44_north.avi",
        "config": "./config/config_north.json",
        "lane_prefix": "N",
        "suggested_gates": [
            {"id": "GATE_N_STOPLINE", "label": "STOPLINE", "type": "stopline"},
            {"id": "GATE_N_IN", "label": "INFLOW_N", "type": "ingress"},
        ],
    },
    "south": {
        "name": "South Approach (CAM-02 / CAM-43)",
        "camera_id": "CAM-02",
        "video": "./videos/cam43_south.avi",
        "config": "./config/config_south.json",
        "lane_prefix": "S",
        "suggested_gates": [
            {"id": "GATE_S_STOPLINE", "label": "STOP_S", "type": "stopline"},
            {"id": "GATE_S_IN", "label": "INFLOW_S", "type": "ingress"},
        ],
    },
    "east": {
        "name": "East Soi Approach (CAM-03)",
        "camera_id": "CAM-03",
        "video": "./videos/cam03_east.avi",
        "config": "./config/config_east.json",
        "lane_prefix": "E",
        "suggested_gates": [
            {"id": "GATE_E_STOPLINE", "label": "STOP_E", "type": "stopline"},
            {"id": "GATE_E_IN", "label": "INFLOW_E", "type": "ingress"},
        ],
    },
    "west": {
        "name": "West Soi Approach (CAM-04 / CAM-46)",
        "camera_id": "CAM-04",
        "video": "./videos/cam46_west.avi",
        "config": "./config/config_west.json",
        "lane_prefix": "W",
        "suggested_gates": [
            {"id": "GATE_W_STOPLINE", "label": "STOP_W", "type": "stopline"},
            {"id": "GATE_W_IN", "label": "INFLOW_W", "type": "ingress"},
        ],
    },
    "northeast": {
        "name": "Northeast Elevated Corridor (CAM-05 / CAM-45)",
        "camera_id": "CAM-05",
        "video": "./videos/cam45_northeast.avi",
        "config": "./config/config_northeast.json",
        "lane_prefix": "NE",
        "suggested_gates": [
            {"id": "GATE_NE_IN", "label": "INFLOW_NE", "type": "ingress"},
            {"id": "GATE_NE_STOPLINE", "label": "STOP_NE", "type": "stopline"},
        ],
    },
}
CAMERA_PRESETS = APPROACH_CONFIGS

# Color definitions (BGR)
COLOR_LANE_FILL = (40, 200, 40)       # Translucent Green
COLOR_LANE_EDGE = (0, 255, 0)         # Bright Green
COLOR_STOPLINE = (0, 165, 255)        # Amber / Orange
COLOR_INGRESS = (255, 220, 0)         # Cyan
COLOR_EGRESS = (255, 50, 220)         # Magenta
COLOR_SELECTED = (0, 255, 255)        # Bright Yellow
COLOR_IN_PROGRESS = (0, 230, 255)     # Yellow-Orange
COLOR_HUD_BG = (20, 20, 22)           # Dark Acrylic


def compute_gate_normal(p1: Tuple[float, float], p2: Tuple[float, float], flip: bool = False) -> Tuple[float, float]:
    """
    Computes unit normal vector perpendicular to line segment p1-p2.
    Default orientation: 90 degrees clockwise perpendicular (-dy, dx).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(1e-6, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    if flip:
        nx, ny = -nx, -ny
    return (round(float(nx), 3), round(float(ny), 3))


def point_to_line_dist(p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Computes minimum Euclidean distance from point p to line segment p1-p2."""
    px, py = p
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return math.hypot(px - x1, py - y1)

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def load_config_geometry(config_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Loads lanes, gates, and entire config dict from JSON file."""
    if not os.path.exists(config_path):
        return {}, [], {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, [], {}

    if not isinstance(data, dict):
        return {}, [], {}

    lm = data.get("lane_metrics")
    lanes_raw = lm.get("lanes") if isinstance(lm, dict) else None
    lanes = copy.deepcopy(lanes_raw) if isinstance(lanes_raw, dict) else {}
    gates_raw = data.get("gates")
    gates = copy.deepcopy(gates_raw) if isinstance(gates_raw, list) else []
    return lanes, gates, data


def save_config_geometry(config_path: str, lanes: Dict[str, Any], gates: List[Dict[str, Any]]) -> bool:
    """
    Saves calibrated lanes and gates back into the JSON config file.
    Validates candidate geometry before modifying the destination or its backup.
    Performs atomic file replacement using a temporary file in the destination directory.
    """
    dest_name = os.path.basename(config_path)

    # 1. Validate candidate lane geometry before touching existing files or creating backups
    report = validate_camera_lanes(lanes, context=dest_name)
    if report.warnings:
        for w in report.warnings:
            print(f"[WARNING] Calibration warning: {w.message}", file=sys.stderr)

    if report.errors:
        for err in report.errors:
            target = f"Lane '{err.lane_id}'" if err.lane_id else "Config"
            print(f"[ERROR] Cannot save invalid calibration for {config_path}: {target} - {err.reason}", file=sys.stderr)
        return False

    if not isinstance(gates, list):
        print(f"[ERROR] Cannot save invalid calibration for {config_path}: 'gates' must be a list, got {type(gates).__name__}", file=sys.stderr)
        return False

    temp_path: Optional[str] = None
    try:
        raw_config: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    raw_config = loaded

        if not isinstance(raw_config.get("lane_metrics"), dict):
            raw_config["lane_metrics"] = {
                "enabled": True,
                "publish_interval_frames": 50,
                "queue_speed_threshold": 2.0,
            }
        else:
            raw_config["lane_metrics"]["enabled"] = True

        raw_config["lane_metrics"]["lanes"] = lanes
        raw_config["gates"] = gates

        # Ensure target directory exists
        dest_dir = os.path.dirname(os.path.abspath(config_path))
        os.makedirs(dest_dir, exist_ok=True)

        # 2. Serialize to temporary file in the destination directory
        prefix = f".tmp_{dest_name}_"
        with tempfile.NamedTemporaryFile("w", dir=dest_dir, prefix=prefix, suffix=".tmp", delete=False, encoding="utf-8") as tf:
            temp_path = tf.name
            json.dump(raw_config, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())

        # 3. Create .bak backup only after temporary file is safely written and closed
        if os.path.exists(config_path):
            backup_path = f"{config_path}.bak"
            shutil.copy2(config_path, backup_path)

        # 4. Atomically replace destination
        os.replace(temp_path, config_path)
        temp_path = None

        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config to {config_path}: {e}", file=sys.stderr)
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


class InteractiveCalibrator:
    """
    Interactive GUI Calibration Tool for Lane Polygons and Virtual Counting Gates.
    """

    def __init__(
        self,
        video_path: str,
        config_path: str,
        direction: str = "custom",
        initial_time_sec: float = 0.0,
    ):
        self.video_path = video_path
        self.config_path = config_path
        self.direction = direction.lower()
        self.current_time_sec = initial_time_sec

        self.preset = CAMERA_PRESETS.get(self.direction, {})
        self.lane_prefix = self.preset.get("lane_prefix", "L")
        self.suggested_gates = self.preset.get("suggested_gates", [])

        # Load existing config data
        self.lanes, self.gates, self.full_config = load_config_geometry(self.config_path)

        # Video source state
        self.cap: Optional[cv.VideoCapture] = None
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0
        self.current_frame: Optional[np.ndarray] = None
        self._init_video()

        # Calibration tool state
        # Modes: 'LANE' (polygon), 'GATE' (line), 'SELECT' (inspect/delete)
        self.mode: str = "LANE"
        self.active_gate_type: str = "stopline"  # 'stopline', 'ingress'
        self.flip_normal: bool = False

        self.in_progress_points: List[Tuple[int, int]] = []
        self.cursor_pos: Tuple[int, int] = (0, 0)
        self.selected_item: Optional[Tuple[str, Any]] = None  # ('lane', lane_id) or ('gate', gate_idx)

        # Notification banner state
        self.notification_msg: str = "Welcome to Interactive Calibration! Press [H] for Help."
        self.notification_color: Tuple[int, int, int] = (0, 255, 255)
        self.notification_expiry: float = time.time() + 4.0

        # Window settings
        self.window_name = f"Traffic Vision Calibrator - [{self.direction.upper()}]"

    def _init_video(self) -> None:
        """Opens video and extracts preview frame."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        self.fps = self.cap.get(cv.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_sec = self.total_frames / self.fps if self.total_frames > 0 else 0.0

        self._seek_and_read_frame(self.current_time_sec)

    def _seek_and_read_frame(self, time_sec: float) -> None:
        """Reads frame at specific timestamp."""
        if self.cap is None or not self.cap.isOpened():
            return

        time_sec = max(0.0, min(self.duration_sec, time_sec)) if self.duration_sec > 0 else max(0.0, time_sec)
        self.current_time_sec = time_sec
        frame_idx = int(time_sec * self.fps)

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame = frame
        else:
            print(f"[WARN] Failed to read frame at {time_sec:.1f}s", file=sys.stderr)

    def set_notification(self, msg: str, color: Tuple[int, int, int] = (0, 255, 255), duration: float = 3.5) -> None:
        """Displays temporary notification on the HUD."""
        self.notification_msg = msg
        self.notification_color = color
        self.notification_expiry = time.time() + duration

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handles mouse interactions in the canvas."""
        self.cursor_pos = (x, y)

        if event == cv.EVENT_LBUTTONDOWN:
            if self.mode == "LANE":
                self.in_progress_points.append((x, y))
                self.set_notification(f"Lane vertex #{len(self.in_progress_points)} added at ({x}, {y}). [C] to complete.")

            elif self.mode == "GATE":
                if len(self.in_progress_points) == 0:
                    self.in_progress_points.append((x, y))
                    self.set_notification(f"Gate Start point set at ({x}, {y}). Click End point.")
                elif len(self.in_progress_points) == 1:
                    p1 = self.in_progress_points[0]
                    p2 = (x, y)
                    self.in_progress_points.clear()
                    self._create_gate_interactive(p1, p2)

            elif self.mode == "SELECT":
                self._select_at_point((x, y))

        elif event == cv.EVENT_RBUTTONDOWN:
            # Right click: in drawing mode, cancel current points; in select mode, deselect
            if self.in_progress_points:
                self.in_progress_points.clear()
                self.set_notification("Cancelled current drawing in progress.", (0, 165, 255))
            else:
                self._select_at_point((x, y))

    def _select_at_point(self, pt: Tuple[int, int]) -> None:
        """Finds closest gate or lane polygon containing/near the click point."""
        # 1. Check gates (within 15 pixels)
        best_gate_idx = -1
        best_gate_dist = 18.0

        for idx, g in enumerate(self.gates):
            d = point_to_line_dist(pt, tuple(g["p1"]), tuple(g["p2"]))
            if d < best_gate_dist:
                best_gate_dist = d
                best_gate_idx = idx

        if best_gate_idx >= 0:
            self.selected_item = ("gate", best_gate_idx)
            g = self.gates[best_gate_idx]
            self.set_notification(f"Selected Gate: {g['gate_id']} ({g['type']}). [D] to Delete, [F] to Flip.", COLOR_SELECTED)
            return

        # 2. Check lanes (point in polygon)
        for lid, linfo in self.lanes.items():
            poly = np.array(linfo.get("polygon", []), dtype=np.int32)
            if len(poly) >= 3:
                res = cv.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), measureDist=False)
                if res >= 0:
                    self.selected_item = ("lane", lid)
                    self.set_notification(f"Selected Lane: {lid}. [D] to Delete.", COLOR_SELECTED)
                    return

        # If nothing found, deselect
        self.selected_item = None
        self.set_notification("Deselected. Click an existing shape to select it.", (180, 180, 180))

    def _create_gate_interactive(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        """Completes gate creation and prompts for ID / label."""
        normal = compute_gate_normal(p1, p2, flip=self.flip_normal)

        # Default suggestion based on active gate type & direction
        default_id = f"GATE_{self.direction.upper()}_{self.active_gate_type.upper()}"
        default_label = self.active_gate_type.upper()

        for sg in self.suggested_gates:
            if sg.get("type") == self.active_gate_type:
                # Check if this suggested ID is already used
                used_ids = {g.get("gate_id") for g in self.gates}
                if sg.get("id") not in used_ids:
                    default_id = sg.get("id")
                    default_label = sg.get("label", default_id)
                    break

        gate_id, label, g_type = self._prompt_gate_details(default_id, default_label, self.active_gate_type)
        if not gate_id:
            self.set_notification("Gate creation cancelled.", (0, 165, 255))
            return

        new_gate = {
            "gate_id": gate_id,
            "p1": [int(p1[0]), int(p1[1])],
            "p2": [int(p2[0]), int(p2[1])],
            "type": g_type,
            "direction": [normal[0], normal[1]],
            "label": label,
        }
        self.gates.append(new_gate)
        self.set_notification(f"Added Gate: {gate_id} ({g_type.upper()})! Remember to press [S] to Save.", COLOR_LANE_EDGE)

    def _create_lane_interactive(self) -> None:
        """Completes lane polygon creation."""
        if len(self.in_progress_points) < 3:
            self.set_notification("Lane requires at least 3 points! Click more points or [R] to reset.", (0, 0, 255))
            return

        # Generate default lane id: e.g. S1, S2, S3...
        used_ids = set(self.lanes.keys())
        idx = 1
        while f"{self.lane_prefix}{idx}" in used_ids:
            idx += 1
        default_lid = f"{self.lane_prefix}{idx}"

        lid = self._prompt_lane_details(default_lid)
        if not lid:
            self.set_notification("Lane creation cancelled.", (0, 165, 255))
            return

        polygon_pts = [[int(x), int(y)] for x, y in self.in_progress_points]
        self.lanes[lid] = {
            "direction": self.lane_prefix[0] if self.lane_prefix else "N",
            "description": f"{self.direction.capitalize()} Lane {lid}",
            "polygon": polygon_pts,
        }
        self.in_progress_points.clear()
        self.set_notification(f"Added Lane: {lid} ({len(polygon_pts)} points)! Remember to press [S] to Save.", COLOR_LANE_EDGE)

    def _prompt_gate_details(
        self, default_id: str, default_label: str, gate_type: str
    ) -> Tuple[Optional[str], str, str]:
        """Opens dialog to select/enter gate ID, label, and type."""
        try:
            import tkinter as tk
            from tkinter import ttk

            result: Dict[str, Any] = {"id": None, "label": default_label, "type": gate_type}

            dialog = tk.Toplevel()
            dialog.title("Configure Virtual Counting Gate")
            dialog.geometry("440x320")
            dialog.resizable(False, False)
            dialog.attributes("-topmost", True)
            dialog.grab_set()

            # Center on screen
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() - dialog.winfo_reqwidth()) // 2
            y = (dialog.winfo_screenheight() - dialog.winfo_reqheight()) // 2
            dialog.geometry(f"+{x}+{y}")

            ttk.Label(dialog, text="Virtual Counting Gate Setup", font=("Helvetica", 12, "bold")).pack(pady=10)

            # Gate Type selector
            type_frame = ttk.Frame(dialog)
            type_frame.pack(fill="x", padx=20, pady=5)
            ttk.Label(type_frame, text="Gate Type:").pack(side="left")
            type_var = tk.StringVar(value=gate_type)
            type_combo = ttk.Combobox(type_frame, textvariable=type_var, values=["stopline", "ingress"], state="readonly", width=15)
            type_combo.pack(side="right")

            # Quick suggestions buttons
            sugg_frame = ttk.LabelFrame(dialog, text="Quick Suggestions")
            sugg_frame.pack(fill="x", padx=20, pady=6)

            id_var = tk.StringVar(value=default_id)
            lbl_var = tk.StringVar(value=default_label)

            def apply_sugg(s_id: str, s_lbl: str, s_type: str):
                id_var.set(s_id)
                lbl_var.set(s_lbl)
                type_var.set(s_type)

            for sg in self.suggested_gates:
                s_id = sg["id"]
                s_lbl = sg.get("label", s_id)
                s_type = sg.get("type", "stopline")
                b = ttk.Button(sugg_frame, text=f"{s_id} ({s_type})", command=lambda i=s_id, l=s_lbl, t=s_type: apply_sugg(i, l, t))
                b.pack(side="left", padx=3, pady=4)

            # Custom Gate ID Entry
            id_frame = ttk.Frame(dialog)
            id_frame.pack(fill="x", padx=20, pady=5)
            ttk.Label(id_frame, text="Gate ID:").pack(side="left")
            id_entry = ttk.Entry(id_frame, textvariable=id_var, width=24)
            id_entry.pack(side="right")

            # Custom Label Entry
            lbl_frame = ttk.Frame(dialog)
            lbl_frame.pack(fill="x", padx=20, pady=5)
            ttk.Label(lbl_frame, text="Display Label:").pack(side="left")
            lbl_entry = ttk.Entry(lbl_frame, textvariable=lbl_var, width=24)
            lbl_entry.pack(side="right")

            # Action buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(fill="x", padx=20, pady=12)

            def on_ok():
                result["id"] = id_var.get().strip() or default_id
                result["label"] = lbl_var.get().strip() or result["id"]
                result["type"] = type_var.get()
                dialog.destroy()

            def on_cancel():
                result["id"] = None
                dialog.destroy()

            ttk.Button(btn_frame, text="Save Gate", command=on_ok).pack(side="right", padx=5)
            ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)

            dialog.wait_window()
            return result["id"], result["label"], result["type"]

        except Exception as e:
            # Fallback to defaults if Tkinter window fails
            print(f"[WARN] Dialog error ({e}), using default: {default_id}")
            return default_id, default_label, gate_type

    def _prompt_lane_details(self, default_lid: str) -> Optional[str]:
        """Opens dialog to enter Lane ID."""
        try:
            import tkinter as tk
            from tkinter import simpledialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            lid = simpledialog.askstring(
                "Lane ID",
                f"Enter Lane Identifier (e.g. {self.lane_prefix}1, {self.lane_prefix}2):",
                initialvalue=default_lid,
                parent=root,
            )
            root.destroy()
            return lid.strip() if lid else None
        except (ImportError, Exception):  # unslop-ignore: fallback when Tkinter or display is unavailable
            return default_lid

    def _draw_hud(self, canvas: np.ndarray) -> None:
        """Renders header toolbar, shortcut footer, and notification banner."""
        h, w = canvas.shape[:2]

        # 1. Top Header Bar (Acrylic Dark)
        header_h = 56
        overlay = canvas.copy()
        cv.rectangle(overlay, (0, 0), (w, header_h), COLOR_HUD_BG, -1)
        cv.line(overlay, (0, header_h), (w, header_h), (60, 60, 65), 1)

        # 2. Bottom Shortcut Bar
        footer_h = 46
        cv.rectangle(overlay, (0, h - footer_h), (w, h), COLOR_HUD_BG, -1)
        cv.line(overlay, (0, h - footer_h), (w, h - footer_h), (60, 60, 65), 1)

        cv.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Header Text
        dir_badge = f"APPROACH: {self.direction.upper()}"
        cv.putText(canvas, dir_badge, (15, 34), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv.LINE_AA)

        # Video & Time metadata
        vid_name = os.path.basename(self.video_path)
        meta_text = f"VIDEO: {vid_name} | TIME: {self.current_time_sec:.1f}s / {self.duration_sec:.1f}s"
        cv.putText(canvas, meta_text, (260, 34), cv.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv.LINE_AA)

        # Current Mode Badge
        mode_str = f"MODE: [{self.mode}]"
        if self.mode == "GATE":
            mode_str += f" ({self.active_gate_type.upper()})"
        cv.rectangle(canvas, (w - 380, 10), (w - 180, 46), (40, 40, 45), -1)
        cv.rectangle(canvas, (w - 380, 10), (w - 180, 46), COLOR_SELECTED if self.mode == "SELECT" else COLOR_LANE_EDGE, 1)
        cv.putText(canvas, mode_str, (w - 370, 33), cv.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv.LINE_AA)

        # Save Button Indicator
        cv.rectangle(canvas, (w - 165, 10), (w - 15, 46), (0, 120, 0), -1)
        cv.rectangle(canvas, (w - 165, 10), (w - 15, 46), (0, 255, 0), 1)
        cv.putText(canvas, "[S] SAVE CONFIG", (w - 153, 33), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

        # Footer Shortcuts
        shortcuts = (
            "[1] Lane Mode  |  [2] Gate Mode  |  [3] Select/Delete  |  [Tab] Gate Type  |  "
            "[F] Flip Arrow  |  [C] Complete  |  [Z] Undo  |  [N]/[P] Seek Frames  |  [S] Save  |  [Q] Exit"
        )
        cv.putText(canvas, shortcuts, (15, h - 16), cv.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv.LINE_AA)

        # Notification Banner (if active)
        if time.time() < self.notification_expiry:
            banner_y = header_h + 8
            (nw, nh), _ = cv.getTextSize(self.notification_msg, cv.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv.rectangle(canvas, (10, banner_y), (20 + nw, banner_y + nh + 14), (20, 20, 25), -1)
            cv.rectangle(canvas, (10, banner_y), (20 + nw, banner_y + nh + 14), self.notification_color, 1)
            cv.putText(
                canvas,
                self.notification_msg,
                (16, banner_y + nh + 7),
                cv.FONT_HERSHEY_SIMPLEX,
                0.52,
                self.notification_color,
                1,
                cv.LINE_AA,
            )

    def _render_scene(self) -> np.ndarray:
        """Composes complete visual frame with lanes, gates, and overlays."""
        if self.current_frame is None:
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        else:
            canvas = self.current_frame.copy()

        # 1. Render Lanes (Polygons)
        poly_overlay = canvas.copy()
        for lid, linfo in self.lanes.items():
            pts = linfo.get("polygon", [])
            if len(pts) >= 3:
                np_pts = np.array(pts, dtype=np.int32)
                is_selected = self.selected_item == ("lane", lid)
                edge_col = COLOR_SELECTED if is_selected else COLOR_LANE_EDGE
                cv.fillPoly(poly_overlay, [np_pts], COLOR_LANE_FILL)
                cv.polylines(canvas, [np_pts], isClosed=True, color=edge_col, thickness=3 if is_selected else 2, lineType=cv.LINE_AA)

                # Centroid for label pill
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                lbl = f"LANE {lid}"
                (lw, lh), _ = cv.getTextSize(lbl, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv.rectangle(canvas, (cx - lw // 2 - 4, cy - lh - 4), (cx + lw // 2 + 4, cy + 4), (10, 10, 10), -1)
                cv.rectangle(canvas, (cx - lw // 2 - 4, cy - lh - 4), (cx + lw // 2 + 4, cy + 4), edge_col, 1)
                cv.putText(canvas, lbl, (cx - lw // 2, cy), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

        cv.addWeighted(poly_overlay, 0.25, canvas, 0.75, 0, canvas)

        # 2. Render Virtual Counting Gates
        for idx, g in enumerate(self.gates):
            p1 = tuple(g["p1"])
            p2 = tuple(g["p2"])
            g_type = g.get("type", "stopline")
            is_selected = self.selected_item == ("gate", idx)

            if is_selected:
                color = COLOR_SELECTED
            elif g_type == "ingress":
                color = COLOR_INGRESS
            elif g_type == "egress":
                color = COLOR_EGRESS
            else:
                color = COLOR_STOPLINE

            # Line
            cv.line(canvas, p1, p2, color, 4 if is_selected else 3, cv.LINE_AA)
            cv.circle(canvas, p1, 5, color, -1)
            cv.circle(canvas, p2, 5, color, -1)

            # Midpoint & Normal Direction Arrow
            mx = int((p1[0] + p2[0]) * 0.5)
            my = int((p1[1] + p2[1]) * 0.5)
            nx, ny = g.get("direction", [0.0, 1.0])
            arrow_len = 36
            tip = (int(mx + nx * arrow_len), int(my + ny * arrow_len))
            cv.arrowedLine(canvas, (mx, my), tip, color, 2, tipLength=0.35)

            # Label Pill
            lbl = f"{g.get('label', g['gate_id'])} [{g_type.upper()}]"
            (gw, gh), _ = cv.getTextSize(lbl, cv.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx = max(10, min(canvas.shape[1] - gw - 10, mx - gw // 2))
            by = max(gh + 10, min(canvas.shape[0] - 10, my - 6))
            cv.rectangle(canvas, (bx - 4, by - gh - 4), (bx + gw + 4, by + 4), (15, 15, 15), -1)
            cv.rectangle(canvas, (bx - 4, by - gh - 4), (bx + gw + 4, by + 4), color, 1)
            cv.putText(canvas, lbl, (bx, by), cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)

        # 3. Render In-Progress Drawing
        if self.in_progress_points:
            for idx, pt in enumerate(self.in_progress_points):
                cv.circle(canvas, pt, 6, COLOR_IN_PROGRESS, -1)
                cv.putText(canvas, f"#{idx+1}", (pt[0] + 8, pt[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

            if len(self.in_progress_points) > 1:
                cv.polylines(
                    canvas,
                    [np.array(self.in_progress_points, dtype=np.int32)],
                    isClosed=False,
                    color=COLOR_IN_PROGRESS,
                    thickness=2,
                    lineType=cv.LINE_AA,
                )

            # Live rubber-band line to mouse cursor
            last_pt = self.in_progress_points[-1]
            cv.line(canvas, last_pt, self.cursor_pos, (0, 200, 255), 1, cv.LINE_AA)

            if self.mode == "GATE" and len(self.in_progress_points) == 1:
                # Show live normal arrow while placing gate endpoint
                p1 = self.in_progress_points[0]
                p2 = self.cursor_pos
                nx, ny = compute_gate_normal(p1, p2, flip=self.flip_normal)
                mx = int((p1[0] + p2[0]) * 0.5)
                my = int((p1[1] + p2[1]) * 0.5)
                tip = (int(mx + nx * 36), int(my + ny * 36))
                cv.arrowedLine(canvas, (mx, my), tip, (0, 255, 255), 2, tipLength=0.35)

        # 4. Render HUD Header & Footer
        self._draw_hud(canvas)
        return canvas

    def run(self) -> None:
        """Main interaction loop."""
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        # Default window size for comfortable 16:9 viewing
        cv.resizeWindow(self.window_name, 1280, 720)
        cv.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n[INFO] Starting Calibration for approach: {self.direction.upper()}")
        print(f"[INFO] Video:  {self.video_path}")
        print(f"[INFO] Config: {self.config_path}")

        try:
            while True:
                canvas = self._render_scene()
                cv.imshow(self.window_name, canvas)
                key = cv.waitKey(20) & 0xFF

                if key in (ord("q"), 27):  # 'q' or ESC
                    break

                elif key == ord("1"):
                    self.mode = "LANE"
                    self.in_progress_points.clear()
                    self.selected_item = None
                    self.set_notification("Switched to [LANE] polygon mode. Click 3+ vertices, press [C] to complete.", COLOR_LANE_EDGE)

                elif key == ord("2"):
                    self.mode = "GATE"
                    self.in_progress_points.clear()
                    self.selected_item = None
                    self.set_notification(f"Switched to [GATE] mode ({self.active_gate_type.upper()}). Click start and end points.", COLOR_STOPLINE)

                elif key == ord("3"):
                    self.mode = "SELECT"
                    self.in_progress_points.clear()
                    self.set_notification("Switched to [SELECT] mode. Click any gate or lane to inspect/delete.", COLOR_SELECTED)

                elif key in (ord("\t"), ord("t")):  # Cycle gate type
                    types = ["stopline", "ingress"]
                    cur_idx = types.index(self.active_gate_type) if self.active_gate_type in types else 0
                    self.active_gate_type = types[(cur_idx + 1) % len(types)]
                    self.set_notification(f"Active Gate Type: {self.active_gate_type.upper()}", COLOR_STOPLINE if self.active_gate_type == "stopline" else COLOR_INGRESS)

                elif key in (ord("i"), ord("I")):
                    self.active_gate_type = "ingress"
                    self.set_notification("Gate Type: INGRESS (Advance Platoon Inflow)", COLOR_INGRESS)

                elif key in (ord("o"), ord("O")):
                    self.active_gate_type = "stopline"
                    self.set_notification("Gate Type: STOPLINE (Queue Clearance / Discharge)", COLOR_STOPLINE)

                elif key in (ord("e"), ord("E")):
                    self.set_notification("Egress disabled (not needed). Use STOPLINE (discharge) or INGRESS (inflow).", COLOR_STOPLINE)

                elif key in (ord("f"), ord("F")):
                    if self.selected_item and self.selected_item[0] == "gate":
                        g_idx = self.selected_item[1]
                        g = self.gates[g_idx]
                        cur_n = g.get("direction", [0.0, 1.0])
                        g["direction"] = [-cur_n[0], -cur_n[1]]
                        self.set_notification(f"Flipped direction for gate: {g['gate_id']}", COLOR_SELECTED)
                    else:
                        self.flip_normal = not self.flip_normal
                        self.set_notification(f"Gate normal direction flipped: {'REVERSE' if self.flip_normal else 'STANDARD'}")

                elif key in (ord("c"), ord("C"), 13):  # 'c' or Enter
                    if self.mode == "LANE":
                        self._create_lane_interactive()

                elif key in (ord("z"), ord("Z"), 8):  # 'z' or Backspace (Undo)
                    if self.in_progress_points:
                        self.in_progress_points.pop()
                        self.set_notification("Undid last point.", (0, 200, 255))
                    elif self.selected_item:
                        self.selected_item = None
                        self.set_notification("Deselected item.")

                elif key in (ord("r"), ord("R")):  # Reset
                    self.in_progress_points.clear()
                    self.set_notification("Reset current drawing.", (0, 165, 255))

                elif key in (ord("d"), ord("D"), 127):  # Delete selected item
                    if self.selected_item:
                        item_type, item_ref = self.selected_item
                        if item_type == "gate" and 0 <= item_ref < len(self.gates):
                            deleted = self.gates.pop(item_ref)
                            self.set_notification(f"Deleted Gate: {deleted['gate_id']}. Press [S] to Save.", (0, 100, 255))
                        elif item_type == "lane" and item_ref in self.lanes:
                            del self.lanes[item_ref]
                            self.set_notification(f"Deleted Lane: {item_ref}. Press [S] to Save.", (0, 100, 255))
                        self.selected_item = None
                    else:
                        self.set_notification("No item selected to delete. Click an item first in [3] Select mode.", (0, 0, 255))

                elif key in (ord("n"), ord("N")):  # Next frame (+5 sec)
                    self._seek_and_read_frame(self.current_time_sec + 5.0)
                    self.set_notification(f"Stepped forward +5s (Timestamp: {self.current_time_sec:.1f}s)")

                elif key in (ord("p"), ord("P")):  # Previous frame (-5 sec)
                    self._seek_and_read_frame(max(0.0, self.current_time_sec - 5.0))
                    self.set_notification(f"Stepped backward -5s (Timestamp: {self.current_time_sec:.1f}s)")

                elif key in (ord("s"), ord("S")):  # Save config
                    success = save_config_geometry(self.config_path, self.lanes, self.gates)
                    if success:
                        self.set_notification(
                            f"[SAVED] Saved to {os.path.basename(self.config_path)} ({len(self.lanes)} lanes, {len(self.gates)} gates)!",
                            (0, 255, 0),
                            duration=5.0,
                        )
                    else:
                        self.set_notification("Failed to save config! Check console log.", (0, 0, 255))

                elif key in (ord("h"), ord("H")):  # Help
                    self.set_notification("[1]:Lane, [2]:Gate, [3]:Select, [Tab]:Type, [F]:Flip, [C]:Done, [S]:Save", (255, 255, 255), duration=5.0)

        finally:
            if self.cap:
                self.cap.release()
            cv.destroyAllWindows()

    def segment(self) -> List[List[int]]:
        """Legacy compatibility wrapper for single-lane segmentation."""
        self.run()
        if self.lanes:
            last_lane = list(self.lanes.values())[-1]
            return last_lane.get("polygon", [])
        return []


def show_launcher_gui() -> Optional[Dict[str, Any]]:
    """
    Displays an interactive launcher GUI allowing the user to select
    camera approach, video file, config file, and preview timestamp.
    """
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        print("[WARN] Tkinter unavailable; falling back to CLI.", file=sys.stderr)
        return None

    result: Dict[str, Any] = {"proceed": False}

    root = tk.Tk()
    root.title("Smart Traffic Vision - Geometry Calibrator")
    root.geometry("540x380")
    root.resizable(False, False)

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 540) // 2
    y = (root.winfo_screenheight() - 380) // 2
    root.geometry(f"+{x}+{y}")

    ttk.Label(root, text="Traffic Vision Geometry Calibration", font=("Helvetica", 14, "bold")).pack(pady=12)

    # 1. Approach Selection
    frame_dir = ttk.LabelFrame(root, text="Select Approach / Camera Direction")
    frame_dir.pack(fill="x", padx=20, pady=8)

    dir_var = tk.StringVar(value="north")
    preset_choices = [
        ("north", "North Approach (CAM-01 / CAM-44)"),
        ("south", "South Approach (CAM-02 / CAM-43)"),
        ("east", "East Soi Approach (CAM-03)"),
        ("west", "West Soi Approach (CAM-04 / CAM-46)"),
        ("northeast", "Northeast Corridor (CAM-05 / CAM-45)"),
        ("custom", "Custom Video & Config..."),
    ]

    video_var = tk.StringVar(value=CAMERA_PRESETS["north"]["video"])
    config_var = tk.StringVar(value=CAMERA_PRESETS["north"]["config"])
    time_var = tk.StringVar(value="0.0")

    def on_preset_change(*args):
        choice = dir_var.get()
        if choice in CAMERA_PRESETS:
            video_var.set(CAMERA_PRESETS[choice]["video"])
            config_var.set(CAMERA_PRESETS[choice]["config"])

    dir_combo = ttk.Combobox(
        frame_dir,
        textvariable=dir_var,
        values=[p[0] for p in preset_choices],
        state="readonly",
        width=35,
    )
    dir_combo.pack(padx=10, pady=8)
    dir_var.trace_add("write", on_preset_change)

    # 2. File Selection Frame
    frame_files = ttk.LabelFrame(root, text="File Paths")
    frame_files.pack(fill="x", padx=20, pady=8)

    # Video Row
    row_vid = ttk.Frame(frame_files)
    row_vid.pack(fill="x", padx=10, pady=4)
    ttk.Label(row_vid, text="Video:", width=8).pack(side="left")
    ttk.Entry(row_vid, textvariable=video_var, width=42).pack(side="left", padx=5)

    def browse_video():
        fn = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4 *.mkv"), ("All files", "*.*")])
        if fn:
            video_var.set(fn)
            dir_var.set("custom")

    ttk.Button(row_vid, text="Browse", width=8, command=browse_video).pack(side="right")

    # Config Row
    row_cfg = ttk.Frame(frame_files)
    row_cfg.pack(fill="x", padx=10, pady=4)
    ttk.Label(row_cfg, text="Config:", width=8).pack(side="left")
    ttk.Entry(row_cfg, textvariable=config_var, width=42).pack(side="left", padx=5)

    def browse_config():
        fn = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if fn:
            config_var.set(fn)
            dir_var.set("custom")

    ttk.Button(row_cfg, text="Browse", width=8, command=browse_config).pack(side="right")

    # Timestamp Row
    row_time = ttk.Frame(frame_files)
    row_time.pack(fill="x", padx=10, pady=4)
    ttk.Label(row_time, text="Time (s):", width=8).pack(side="left")
    ttk.Entry(row_time, textvariable=time_var, width=12).pack(side="left", padx=5)

    # 3. Action Buttons
    frame_action = ttk.Frame(root)
    frame_action.pack(fill="x", padx=20, pady=14)

    def launch():
        vid_p = video_var.get().strip()
        cfg_p = config_var.get().strip()
        if not os.path.exists(vid_p):
            messagebox.showerror("Error", f"Video file does not exist:\n{vid_p}")
            return
        try:
            sec_val = float(time_var.get().strip() or "0.0")
        except ValueError:
            sec_val = 0.0

        result["proceed"] = True
        result["direction"] = dir_var.get()
        result["video"] = vid_p
        result["config"] = cfg_p
        result["sec"] = sec_val
        root.destroy()

    ttk.Button(frame_action, text="Launch Interactive Calibrator", command=launch).pack(side="right", padx=5)
    ttk.Button(frame_action, text="Cancel", command=root.destroy).pack(side="right", padx=5)

    root.mainloop()
    return result if result.get("proceed") else None


def run_calibration(
    direction: Optional[str] = None,
    video: Optional[str] = None,
    config: Optional[str] = None,
    sec: float = 0.0,
) -> None:
    """Entry point to launch the interactive calibrator."""
    if direction and direction.lower() in CAMERA_PRESETS:
        preset = CAMERA_PRESETS[direction.lower()]
        video = video or preset["video"]
        config = config or preset["config"]
    elif not video or not config:
        # Launch UI picker
        launcher_res = show_launcher_gui()
        if not launcher_res:
            print("[INFO] Calibration launcher cancelled.")
            return
        direction = launcher_res["direction"]
        video = launcher_res["video"]
        config = launcher_res["config"]
        sec = launcher_res["sec"]

    direction = direction or "custom"
    calibrator = InteractiveCalibrator(
        video_path=video,
        config_path=config,
        direction=direction,
        initial_time_sec=sec,
    )
    calibrator.run()


# Legacy alias
RoadSegmenter = InteractiveCalibrator


def main():
    parser = argparse.ArgumentParser(description="Smart Traffic Vision - Interactive Geometry Calibration Tool")
    parser.add_argument(
        "--direction",
        "--approach",
        type=str,
        dest="direction",
        choices=["north", "south", "east", "west", "northeast", "custom"],
        default=None,
        help="Target approach / camera direction (automatically resolves video and config files)",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--config", type=str, default=None, help="Path to camera config JSON file")
    parser.add_argument("--sec", type=float, default=0.0, help="Initial timestamp in seconds to grab video frame")
    args = parser.parse_args()

    run_calibration(
        direction=args.direction,
        video=args.video,
        config=args.config,
        sec=args.sec,
    )


if __name__ == "__main__":
    main()
