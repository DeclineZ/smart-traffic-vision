"""
Production-Grade Weather & Road Surface Condition Classifier for Smart Traffic Vision.
Handles real-time environmental classification (Dry vs Wet/Rainy/Flood), Auto-Road ROI,
and Hysteresis-driven Lane Detection Mode switching.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class WeatherResult:
    """Encapsulates the state and telemetry of weather and road condition analysis."""
    state: str                 # "CLEAR" or "RAINY"
    lane_mode: str             # "DRY_STANDARD" or "WET_ADAPTIVE"
    is_rainy: bool             # True if state is RAINY
    raw_class: str             # Instant frame prediction: "clear" or "rainy"
    confidence: float          # Raw prediction confidence (0.0 - 1.0)
    rain_ratio: float          # Fraction of recent window voting rainy (0.0 - 1.0)
    window_count: Tuple[int, int]  # (rainy_votes, total_samples)
    latency_ms: float          # Model inference latency in milliseconds
    frame_idx: int             # Frame number
    roi_box: Tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) of road ROI


class WeatherClassifier:
    """
    Production-ready Weather & Road Condition perception engine.
    Applies Auto-Road Focus to ignore sky/billboard distractions and
    Hysteresis-based state transitions for stable Lane Controller operations.
    """

    def __init__(
        self,
        model_path: str | Path = "models/weather_yolov8n_cls.pt",
        device: str | None = None,
        check_interval: int = 5,
        window_size: int = 10,
        rain_threshold: float = 0.60,
        dry_threshold: float = 0.30,
        auto_road_roi: bool = True,
        top_crop_ratio: float = 0.30,
        custom_roi: Tuple[int, int, int, int] | None = None,
    ):
        """
        Args:
            model_path: Path to trained YOLOv8-cls weights.
            device: 'mps', '0' (CUDA), or 'cpu'. Auto-detected if None.
            check_interval: Run inference every N frames to optimize compute.
            window_size: Sliding window sample count for temporal smoothing.
            rain_threshold: Rain vote ratio required to enter RAINY state (default 60%).
            dry_threshold: Rain vote ratio required to return to CLEAR state (default 30%).
            auto_road_roi: If True, crops the upper portion of the frame to focus on road pavement.
            top_crop_ratio: Fraction of top frame to discard (default 0.30 = top 30% sky/buildings removed).
            custom_roi: Optional explicit (x1, y1, x2, y2) bounding box.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Weather model weights not found at: {self.model_path.resolve()}")

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = "0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = YOLO(str(self.model_path.resolve()))
        self.class_names = self.model.names  # {0: 'clear', 1: 'rainy'}

        self.check_interval = check_interval
        self.window_size = window_size
        self.rain_threshold = rain_threshold
        self.dry_threshold = dry_threshold
        self.auto_road_roi = auto_road_roi
        self.top_crop_ratio = max(0.0, min(0.7, top_crop_ratio))
        self.custom_roi = custom_roi

        # State tracking
        self.history: Deque[int] = collections.deque(maxlen=window_size)
        self.current_state = "CLEAR"
        self.current_raw_class = "clear"
        self.current_raw_conf = 1.0
        self.last_latency_ms = 0.0
        self.frame_counter = 0

    def get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extracts the road inspection ROI to focus on pavement condition."""
        h, w = frame.shape[:2]
        if self.custom_roi is not None:
            x1, y1, x2, y2 = self.custom_roi
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
        elif self.auto_road_roi:
            # Crop lower portion where road and standing water are situated
            y_start = int(h * self.top_crop_ratio)
            return frame[y_start:h, 0:w], (0, y_start, w, h)
        return frame, (0, 0, w, h)

    def update(self, frame: np.ndarray, frame_idx: int | None = None) -> WeatherResult:
        """
        Processes a video frame and returns the updated weather state and lane mode.
        Runs inference according to check_interval, and applies hysteresis state transitions.
        """
        self.frame_counter = frame_idx if frame_idx is not None else (self.frame_counter + 1)
        roi_img, roi_box = self.get_roi(frame)

        # Run inference on schedule or first frame
        if self.frame_counter % self.check_interval == 0 or len(self.history) == 0:
            t0 = time.time()
            res = self.model.predict(roi_img, imgsz=224, device=self.device, verbose=False)[0]
            self.last_latency_ms = (time.time() - t0) * 1000.0

            top1_id = int(res.probs.top1)
            self.current_raw_conf = float(res.probs.top1conf)
            self.current_raw_class = self.class_names[top1_id].lower()

            is_rain_sample = 1 if self.current_raw_class == "rainy" else 0
            self.history.append(is_rain_sample)

            # Hysteresis State Machine
            rain_count = sum(self.history)
            total_count = len(self.history)
            rain_ratio = rain_count / total_count if total_count > 0 else 0.0

            if self.current_state == "CLEAR":
                if rain_ratio >= self.rain_threshold:
                    self.current_state = "RAINY"
            else:  # current_state == "RAINY"
                if rain_ratio <= self.dry_threshold:
                    self.current_state = "CLEAR"

        rain_votes = sum(self.history)
        total_votes = len(self.history)
        ratio = rain_votes / total_votes if total_votes > 0 else 0.0
        is_rain = (self.current_state == "RAINY")
        lane_mode = "WET_ADAPTIVE" if is_rain else "DRY_STANDARD"

        return WeatherResult(
            state=self.current_state,
            lane_mode=lane_mode,
            is_rainy=is_rain,
            raw_class=self.current_raw_class,
            confidence=self.current_raw_conf,
            rain_ratio=ratio,
            window_count=(rain_votes, total_votes),
            latency_ms=self.last_latency_ms,
            frame_idx=self.frame_counter,
            roi_box=roi_box,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        result: WeatherResult,
        show_roi: bool = True,
        camera_id: str = "CAM_LIVE",
    ) -> np.ndarray:
        """Draws the Smart Traffic Weather & Road Condition HUD on the frame."""
        h, w = frame.shape[:2]
        output = frame.copy()

        # 1. Optional Road ROI boundary indicator
        if show_roi and result.roi_box is not None:
            x1, y1, x2, y2 = result.roi_box
            if y1 > 0:  # Draw top divider line of road zone
                cv2.line(output, (0, y1), (w, y1), (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    output,
                    "[AUTO ROAD INSPECTION ZONE]",
                    (w - 240, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. HUD Panel Box
        panel_w, panel_h = 520, 195
        x0, y0 = 20, 20
        x1_panel, y1_panel = x0 + panel_w, y0 + panel_h

        overlay = output.copy()
        cv2.rectangle(overlay, (x0, y0), (x1_panel, y1_panel), (15, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

        is_rain = result.is_rainy
        theme_color = (40, 60, 235) if is_rain else (50, 210, 80)

        # Border
        cv2.rectangle(output, (x0, y0), (x1_panel, y1_panel), theme_color, 2)
        cv2.line(output, (x0, y0 + 40), (x1_panel, y0 + 40), theme_color, 1)

        # Header Title
        cv2.putText(
            output,
            f"SMART TRAFFIC - ROAD CONDITION MONITOR [{camera_id}]",
            (x0 + 15, y0 + 26),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # State Badge
        badge_text = f"SURFACE STATE: {result.state}"
        cv2.putText(
            output,
            badge_text,
            (x0 + 15, y0 + 75),
            cv2.FONT_HERSHEY_DUPLEX,
            0.80,
            theme_color,
            2,
            cv2.LINE_AA,
        )

        # Lane Mode text
        lane_text = (
            "LANE CONTROLLER: [WET / ADAPTIVE MODE]"
            if is_rain
            else "LANE CONTROLLER: [STANDARD DRY MODE]"
        )
        cv2.putText(
            output,
            lane_text,
            (x0 + 15, y0 + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 230, 100) if is_rain else (180, 240, 180),
            1,
            cv2.LINE_AA,
        )

        # Instant prediction & confidence
        rain_votes, total_votes = result.window_count
        rain_pct = result.rain_ratio * 100
        stats_text = (
            f"Instant: {result.raw_class.upper()} ({result.confidence*100:.1f}%) | "
            f"Latency: {result.latency_ms:.1f}ms | Frame #{result.frame_idx}"
        )
        cv2.putText(
            output, stats_text, (x0 + 15, y0 + 133), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Smoothing Progress Bar
        bar_x, bar_y = x0 + 15, y0 + 150
        bar_w, bar_h = 240, 10
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * (rain_pct / 100.0))
        if fill_w > 0:
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), theme_color, -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

        cv2.putText(
            output,
            f"Buffer: {rain_votes}/{total_votes} ({rain_pct:.0f}%)",
            (bar_x + bar_w + 10, bar_y + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        # Status footer line
        status_msg = "Road condition stable" if (rain_pct in (0, 100)) else "Evaluating road transition..."
        cv2.putText(
            output,
            f"Telemetry: {status_msg}",
            (x0 + 15, y0 + 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        return output
