"""
Temporal Track Class Voting & Anti-Flicker Filter for Multi-Camera Vehicle Tracking.
Smooths vehicle classification across consecutive video frames using confidence-weighted
temporal aggregation, recency decay, and hysteresis margins to prevent rapid class switching
(e.g. pickup truck / รถกระบะ oscillating between car and truck).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class TrackClassVotingFilter:
    """
    Stabilizes object classification for tracked vehicles across time.

    Features:
    - Bounded temporal sliding window (deque) per active track
    - Confidence-weighted vote accumulation
    - Exponential recency decay (fresh observations slightly favored)
    - Pickup truck ambiguity resolver (prior bias favoring car over truck)
    - Hysteresis switching threshold to prevent single-frame label flipping
    - Isolated multi-camera stream namespaces with automatic dead-track pruning
    """

    def __init__(
        self,
        num_streams: int = 1,
        window_size: int = 15,
        car_truck_bias: float = 1.15,
        hysteresis_margin: float = 1.20,
        decay_factor: float = 0.96,
        car_cls_id: int = 0,
        truck_cls_id: int = 3,
        enabled: bool = True,
    ):
        """
        Args:
            num_streams: Number of parallel camera streams.
            window_size: Number of recent detection observations to retain per track.
            car_truck_bias: Multiplier weight for 'car' when resolving pickup truck ambiguity.
            hysteresis_margin: Multiplier threshold required for a competing class to overtake current class.
            decay_factor: Per-step exponential decay applied to older observations (1.0 = equal weight).
            car_cls_id: Class ID representing 'car' (default: 0 in Thai model, 2 in COCO).
            truck_cls_id: Class ID representing 'truck' (default: 3 in Thai model, 7 in COCO).
            enabled: If False, bypasses smoothing and returns raw instantaneous class.
        """
        self.num_streams = max(1, num_streams)
        self.window_size = max(1, window_size)
        self.car_truck_bias = max(1.0, car_truck_bias)
        self.hysteresis_margin = max(1.0, hysteresis_margin)
        self.decay_factor = min(1.0, max(0.5, decay_factor))
        self.car_cls_id = car_cls_id
        self.truck_cls_id = truck_cls_id
        self.enabled = enabled

        # Per-stream track state: stream_idx -> {track_id -> {"history": deque, "current_cls": int}}
        self.stream_tracks: List[Dict[int, Dict[str, any]]] = [{} for _ in range(self.num_streams)]

    def update(
        self,
        cam_idx: int,
        track_id: int,
        raw_cls_id: int,
        conf: float,
    ) -> int:
        """
        Records a new detection observation for a tracked vehicle and returns its smoothed class ID.

        Args:
            cam_idx: Camera stream index.
            track_id: Tracking identity integer assigned by SORT / Kalman filter.
            raw_cls_id: Instantaneous class ID from the latest YOLO detection.
            conf: Detection confidence score (0.0 to 1.0).

        Returns:
            Stabilized class ID.
        """
        if not self.enabled or cam_idx >= len(self.stream_tracks):
            return raw_cls_id

        tracks = self.stream_tracks[cam_idx]
        if track_id not in tracks:
            tracks[track_id] = {
                "history": deque(maxlen=self.window_size),
                "current_cls": raw_cls_id,
            }

        t_state = tracks[track_id]
        t_state["history"].append((raw_cls_id, float(conf)))

        # If only 1 observation, adopt it directly
        if len(t_state["history"]) == 1:
            t_state["current_cls"] = raw_cls_id
            return raw_cls_id

        # Accumulate weighted votes across temporal history
        hist = list(t_state["history"])
        n_obs = len(hist)
        class_scores: Dict[int, float] = {}

        for age_idx, (c_id, c_conf) in enumerate(hist):
            # Most recent observation has age 0
            age = n_obs - 1 - age_idx
            decay = self.decay_factor ** age
            weight = c_conf * decay

            # Apply pickup-as-car bias when resolving car vs truck ambiguity
            if c_id == self.car_cls_id:
                weight *= self.car_truck_bias

            class_scores[c_id] = class_scores.get(c_id, 0.0) + weight

        # Determine leading candidate class
        best_cls = max(class_scores.keys(), key=lambda c: class_scores[c])
        current_cls = t_state["current_cls"]

        # Hysteresis check: require best_cls to surpass current_cls by hysteresis_margin
        if best_cls != current_cls:
            current_score = class_scores.get(current_cls, 0.0)
            best_score = class_scores.get(best_cls, 0.0)
            if best_score >= current_score * self.hysteresis_margin:
                t_state["current_cls"] = best_cls
        else:
            t_state["current_cls"] = best_cls

        return t_state["current_cls"]

    def get_class(self, cam_idx: int, track_id: int, fallback: int) -> int:
        """
        Retrieves the smoothed class ID for a track without adding an observation
        (e.g., during frame-skipping intervals).
        """
        if not self.enabled or cam_idx >= len(self.stream_tracks):
            return fallback

        tracks = self.stream_tracks[cam_idx]
        if track_id in tracks:
            return tracks[track_id]["current_cls"]
        return fallback

    def prune(self, cam_idx: int, active_track_ids: Set[int]) -> None:
        """
        Removes memory for tracks that are no longer active to prevent memory growth.
        """
        if cam_idx >= len(self.stream_tracks):
            return

        tracks = self.stream_tracks[cam_idx]
        if len(tracks) > len(active_track_ids) + 50:
            stale_ids = [tid for tid in tracks if tid not in active_track_ids]
            for tid in stale_ids:
                del tracks[tid]

    def reset(self, cam_idx: Optional[int] = None) -> None:
        """Resets voting memory for a specific stream or all streams."""
        if cam_idx is not None and cam_idx < len(self.stream_tracks):
            self.stream_tracks[cam_idx].clear()
        else:
            for stream_map in self.stream_tracks:
                stream_map.clear()
