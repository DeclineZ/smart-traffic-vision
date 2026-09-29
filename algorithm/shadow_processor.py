"""
Shadow Processor & Contact-Patch Refiner for Harsh Shadow Vehicle Counting.
Provides:
  - ShadowContrastEqualizer: Adaptive L-channel CLAHE and localized shadow contrast booster.
  - ContactPatchRefiner: Bottom tire-ground contact point computation & lateral cast shadow trimming.
  - ShadowLaneAssigner: Temporal hysteresis and single-lane exclusivity debouncing.
"""

from __future__ import annotations
import cv2 as cv
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point, Polygon


class ShadowContrastEqualizer:
    """
    Adaptive Shadow Contrast Equalizer (SCE).
    Identifies underexposed shadow zones (cast by bridges, flyovers, buildings, trees)
    and applies localized L-channel CLAHE + adaptive gamma compensation to boost dark vehicle
    contrast without overexposing or blowing out sunlit asphalt.
    """

    def __init__(
        self,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 0.70,
        shadow_thresh: float = 75.0,
        sun_thresh: float = 145.0,
        dynamic_trigger: bool = True,
        contrast_variance_threshold: float = 85.0,
    ):
        """
        Args:
            clip_limit: CLAHE threshold for contrast limiting.
            tile_grid_size: Grid size for histogram equalization.
            gamma: Power-law exponent for shadow brightening (< 1.0 brightens shadows).
            shadow_thresh: Luminance threshold (0-255) below which a pixel is considered shadow.
            sun_thresh: Luminance threshold above which a pixel is considered direct sunlight.
            dynamic_trigger: If True, evaluates scene contrast and activates enhancement
                             only when harsh shadow conditions are detected.
            contrast_variance_threshold: P90 - P10 luminance difference threshold to activate enhancement.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma
        self.shadow_thresh = shadow_thresh
        self.sun_thresh = sun_thresh
        self.dynamic_trigger = dynamic_trigger
        self.contrast_variance_threshold = contrast_variance_threshold

        self.clahe = cv.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        # Precompute gamma lookup table for 8-bit channel
        inv_gamma = self.gamma
        self.gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

        self._frame_counter = 0
        self._is_shadow_scene_active = True

    def detect_harsh_shadows(self, l_channel: np.ndarray) -> bool:
        """
        Fast illumination dynamic range assessment using downsampled percentile statistics.
        Returns True if the scene exhibits high-contrast harsh shadows.
        """
        # Downsample 4x for microsecond evaluation
        small_l = cv.resize(l_channel, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
        p10, p90 = np.percentile(small_l, [10, 90])
        dynamic_range = p90 - p10

        # Harsh shadow condition: deep dark regions alongside bright direct sunlight
        has_harsh_contrast = (dynamic_range > self.contrast_variance_threshold) and (p10 < self.shadow_thresh)
        return bool(has_harsh_contrast)

    def enhance(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Applies localized shadow contrast enhancement to BGR image.

        Returns:
            enhanced_bgr: Enhanced BGR image (or original if harsh shadows not active).
            is_active: Whether shadow enhancement was actively applied.
        """
        if bgr_image is None or bgr_image.size == 0:
            return bgr_image, False

        # Convert to CIELAB color space (L = lightness, A/B = chromaticity)
        lab = cv.cvtColor(bgr_image, cv.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        self._frame_counter += 1
        # Check scene contrast every 15 frames if dynamic trigger enabled
        if self.dynamic_trigger:
            if self._frame_counter % 15 == 1 or self._frame_counter <= 3:
                self._is_shadow_scene_active = self.detect_harsh_shadows(l_chan)

            if not self._is_shadow_scene_active:
                return bgr_image, False

        # 1. Apply CLAHE directly to L-channel to preserve high-frequency edge gradients
        l_clahe = self.clahe.apply(l_chan)

        # 2. Soft shadow mask computation to smoothly blend enhancement
        # Mask = 1.0 in deep shadows (L <= shadow_thresh), smoothly decays to 0.0 at sun_thresh
        shadow_mask = np.clip(
            (self.sun_thresh - l_chan.astype(np.float32)) / (self.sun_thresh - self.shadow_thresh + 1e-5),
            0.0,
            1.0,
        )

        # Fast soft blending: L_out = L_orig * (1 - Mask) + L_clahe * Mask
        l_out = (l_chan.astype(np.float32) * (1.0 - shadow_mask) + l_clahe.astype(np.float32) * shadow_mask).astype(np.uint8)

        lab[:, :, 0] = l_out
        enhanced_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        return enhanced_bgr, True


COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Common indoor/household classes YOLO confuses with dark vehicles under extreme pillar shadow
SHADOW_CONFUSED_CLASSES = {
    4: "airplane", 8: "boat", 13: "bench", 28: "suitcase", 56: "chair",
    57: "couch", 59: "bed", 62: "tv", 67: "cell phone", 72: "refrigerator"
}


def remap_shadow_detection(cls_id: int, bbox: np.ndarray | List[float]) -> int:
    """
    Resolves YOLO classification confusion in deep shadows and bridge pillar occlusions.
    If a vehicle-sized object (w >= 25, h >= 25) is misclassified under shadow,
    remaps it to class 2 ('car').
    """
    if cls_id in COCO_VEHICLES:
        return cls_id
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if cls_id in SHADOW_CONFUSED_CLASSES and w >= 25 and h >= 25:
        return 2  # Remap to car
    return cls_id


class ContactPatchRefiner:
    """
    Refines detected vehicle bounding boxes and computes the true ground contact patch.
    1. Replaces naive geometric center (cx, cy) with the bottom tire-road contact point.
    2. Analyzes lateral gradient and shadow profiles to trim cast shadow wings.
    """

    def __init__(self, ground_offset_ratio: float = 0.05, trim_shadow_wings: bool = True):
        """
        Args:
            ground_offset_ratio: Vertical offset from the bottom of the bounding box
                                 (e.g., 0.05 anchors point 5% above bottom edge at the tire level).
            trim_shadow_wings: Whether to detect and trim cast shadow shelves extending laterally.
        """
        self.ground_offset_ratio = ground_offset_ratio
        self.trim_shadow_wings = trim_shadow_wings

    def get_contact_point(self, bbox: np.ndarray | List[float]) -> Tuple[float, float]:
        """
        Computes the bottom tire-road contact anchor point:
        (x_contact, y_contact) = ((x1 + x2) / 2, y2 - ground_offset_ratio * h)
        """
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        h = max(1.0, y2 - y1)
        x_contact = (x1 + x2) * 0.5
        y_contact = y2 - (self.ground_offset_ratio * h)
        return (x_contact, y_contact)

    def get_contact_points_vectorized(self, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized computation of contact patch anchors for N bounding boxes.
        Returns:
            xs: Array of x-coordinates (N,)
            ys: Array of y-coordinates (N,)
        """
        if len(bboxes) == 0:
            return np.empty(0), np.empty(0)
        xs = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        heights = np.maximum(1.0, bboxes[:, 3] - bboxes[:, 1])
        ys = bboxes[:, 3] - (self.ground_offset_ratio * heights)
        return xs, ys

    def trim_lateral_cast_shadow(
        self,
        bbox: np.ndarray | List[float],
        gray_frame: np.ndarray,
        sobel_thresh: float = 25.0,
    ) -> np.ndarray:
        """
        Inspects the bottom 20% strip of the bounding box.
        If a low-gradient cast shadow shelf extends to the left or right beyond the tires,
        trims the box back to the mechanical vehicle boundary.
        """
        if not self.trim_shadow_wings or gray_frame is None:
            return np.array(bbox)

        x1, y1, x2, y2 = int(round(bbox[0])), int(round(bbox[1])), int(round(bbox[2])), int(round(bbox[3]))
        h, w = gray_frame.shape[:2]

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 20 or box_h < 20:
            return np.array(bbox)

        # Inspect bottom 20% strip where tires and ground shadow lie
        strip_y1 = max(y1, int(y2 - 0.20 * box_h))
        strip = gray_frame[strip_y1:y2, x1:x2]
        if strip.size == 0 or strip.shape[1] < 10:
            return np.array(bbox)

        # Compute horizontal Sobel gradient (edges of vehicle tires vs uniform shadow)
        sobel_x = cv.Sobel(strip, cv.CV_32F, 1, 0, ksize=3)
        edge_energy = np.mean(np.abs(sobel_x), axis=0)  # 1D energy profile across width

        # A shadow wing typically has very low edge energy (< sobel_thresh)
        # Check left wing (up to 25% of box width)
        left_limit = int(0.25 * box_w)
        left_trim = 0
        for i in range(left_limit):
            if edge_energy[i] < sobel_thresh:
                left_trim = i
            else:
                break

        # Check right wing (up to 25% of box width from the right)
        right_limit = int(0.25 * box_w)
        right_trim = 0
        for i in range(right_limit):
            col_idx = box_w - 1 - i
            if edge_energy[col_idx] < sobel_thresh:
                right_trim = i
            else:
                break

        refined = np.array(bbox, dtype=float).copy()
        # Only trim if substantial wing is detected (> 8% of width) but leaves at least 60% of vehicle intact
        if left_trim > int(0.08 * box_w) and (box_w - left_trim - right_trim) > int(0.60 * box_w):
            refined[0] = x1 + left_trim
        if right_trim > int(0.08 * box_w) and (box_w - left_trim - right_trim) > int(0.60 * box_w):
            refined[2] = x2 - right_trim

        return refined


class ShadowLaneAssigner:
    """
    Debounced Single-Lane Assigner with Temporal Hysteresis.
    Guarantees:
      1. Single-Lane Exclusivity: A track ID is owned by at most ONE lane per snapshot interval.
      2. Temporal Debounce (Hysteresis): A vehicle must be detected inside a new lane for K consecutive
         frames before a lane transition is granted, completely eliminating boundary shadow chatter.
    """

    def __init__(self, hysteresis_frames: int = 3, max_history: int = 30):
        """
        Args:
            hysteresis_frames: Number of consecutive frames a vehicle's contact point must
                               reside in a candidate lane before confirming the lane change.
            max_history: Maximum history buffer per track ID.
        """
        self.hysteresis_frames = hysteresis_frames
        self.max_history = max_history

        # track_id -> deque of recent candidate lane IDs: deque([(frame_idx, lane_id)])
        self.track_candidates: Dict[int, deque] = {}
        # track_id -> confirmed lane_id
        self.confirmed_lanes: Dict[int, str] = {}
        # track_id -> last observed frame_idx
        self.last_seen: Dict[int, int] = {}

    def update_track_lane(
        self,
        track_id: int,
        candidate_lane_id: Optional[str],
        frame_idx: int,
    ) -> Optional[str]:
        """
        Updates lane assignment for a track ID using temporal hysteresis.

        Args:
            track_id: Vehicle unique tracking ID.
            candidate_lane_id: Lane polygon containing the vehicle's contact point (or None).
            frame_idx: Current frame index.

        Returns:
            confirmed_lane: The stably confirmed lane for this vehicle (or None).
        """
        self.last_seen[track_id] = frame_idx

        if track_id not in self.track_candidates:
            self.track_candidates[track_id] = deque(maxlen=self.max_history)

        history = self.track_candidates[track_id]
        history.append((frame_idx, candidate_lane_id))

        # If vehicle was never assigned a lane yet, initialize if candidate valid
        if track_id not in self.confirmed_lanes:
            if candidate_lane_id is not None:
                # Count consecutive occurrences of candidate_lane_id at the end of history
                consecutive = sum(
                    1 for _, l_id in reversed(history) if l_id == candidate_lane_id
                )
                if consecutive >= self.hysteresis_frames:
                    self.confirmed_lanes[track_id] = candidate_lane_id
            return self.confirmed_lanes.get(track_id)

        # Vehicle already has a confirmed lane
        current_lane = self.confirmed_lanes[track_id]
        if candidate_lane_id == current_lane or candidate_lane_id is None:
            return current_lane

        # Candidate is different from current confirmed lane: require K consecutive frames
        recent_matches = 0
        for _, l_id in reversed(history):
            if l_id == candidate_lane_id:
                recent_matches += 1
            else:
                break

        if recent_matches >= self.hysteresis_frames:
            # Transfer confirmed ownership
            self.confirmed_lanes[track_id] = candidate_lane_id

        return self.confirmed_lanes[track_id]

    def purge_inactive_tracks(self, current_frame: int, max_idle_frames: int = 150) -> None:
        """Removes memory for tracks not seen for max_idle_frames."""
        inactive_ids = [
            tid for tid, lframe in self.last_seen.items() if (current_frame - lframe) > max_idle_frames
        ]
        for tid in inactive_ids:
            self.track_candidates.pop(tid, None)
            self.confirmed_lanes.pop(tid, None)
            self.last_seen.pop(tid, None)
