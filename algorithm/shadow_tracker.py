"""
Shadow-Resilient Vehicle Tracker.
Combines:
  1. Two-Tier Confidence Association (ByteTrack principle) for shadow-contrast dips.
  2. Velocity Direction Consistency (VDC) & Observation-Centric Recovery (OC-SORT).
  3. Inertial Kalman Coasting across Deep Shadow Gaps (e.g., flyovers/overpasses).
  4. Class ID Temporal Smoothing & Persistence.
"""

from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional, Dict, Any

from .utils import (
    linear_assignment,
    iou_batch,
    convert_bbox_to_z,
    convert_x_to_bbox,
    k_previous_obs,
    speed_direction,
    speed_direction_batch,
)


class ShadowKalmanBoxTracker:
    """
    Kalman tracker with observation memory and velocity direction history,
    optimized for tracking vehicles through harsh illumination discontinuities.
    """
    count = 0

    def __init__(self, bbox: np.ndarray, delta_t: int = 3, cls_id: int = 2, min_hits: int = 2):
        """
        Args:
            bbox: [x1, y1, x2, y2, score, ...]
            delta_t: Number of frames to look back for velocity vector estimation.
            cls_id: Detected class ID (e.g. 2 for car, 5 for bus).
            min_hits: Minimum hits required to confirm track.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])
        self.time_since_update = 0
        self.id = ShadowKalmanBoxTracker.count
        ShadowKalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.min_hits = min_hits
        self.is_confirmed = False

        self.last_observation = np.array([-1, -1, -1, -1, -1], dtype=float)
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

        # Class vote memory
        self.class_history: List[int] = [cls_id]

    def update(self, bbox: Optional[np.ndarray], cls_id: Optional[int] = None):
        """
        Updates state with new observation.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if (self.age - dt) in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox[:4])

            self.last_observation = bbox[:5] if len(bbox) >= 5 else np.append(bbox[:4], 1.0)
            self.observations[self.age] = self.last_observation
            self.history_observations.append(self.last_observation)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.hit_streak >= self.min_hits or self.hits >= self.min_hits:
                self.is_confirmed = True
            self.kf.update(convert_bbox_to_z(bbox[:4]))

            if cls_id is not None:
                self.class_history.append(int(cls_id))
                if len(self.class_history) > 30:
                    self.class_history.pop(0)
        else:
            self.kf.update(bbox)

    def predict(self) -> np.ndarray:
        """Advances state and returns predicted [x1, y1, x2, y2]."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Returns current bbox estimate [x1, y1, x2, y2]."""
        return convert_x_to_bbox(self.kf.x)

    def get_dominant_class(self) -> int:
        """Returns most frequent class ID observed for this vehicle."""
        if not self.class_history:
            return 2  # default car
        c = Counter(self.class_history)
        return c.most_common(1)[0][0]


class ShadowResilientTracker:
    """
    Shadow-Resilient Multi-Object Tracker.
    Maintains vehicle tracks through abrupt shadow transitions and bridge occlusions.
    """

    def __init__(
        self,
        det_thresh: float = 0.40,
        min_conf: float = 0.15,
        max_age: int = 30,
        max_coast_frames: int = 12,
        min_hits: int = 2,
        iou_threshold: float = 0.30,
        delta_t: int = 3,
        inertia: float = 0.20,
    ):
        """
        Args:
            det_thresh: Primary high-confidence detection threshold (Tier 1).
            min_conf: Secondary low-confidence detection threshold for shadow recovery (Tier 2).
            max_age: Maximum frames to retain tracker before deletion.
            max_coast_frames: Maximum frames a track is allowed to coast through deep shadow
                              and still be reported as an active track.
            min_hits: Minimum detection hits before track is confirmed.
            iou_threshold: Spatial overlap threshold for association.
            delta_t: Frame interval for velocity direction estimation.
            inertia: Velocity Direction Consistency weight.
        """
        self.det_thresh = det_thresh
        self.min_conf = min_conf
        self.max_age = max_age
        self.max_coast_frames = max_coast_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia

        self.trackers: List[ShadowKalmanBoxTracker] = []
        self.frame_count = 0
        ShadowKalmanBoxTracker.count = 0

    def update(self, dets: np.ndarray = np.empty((0, 6))) -> np.ndarray:
        """
        Update tracker with detections.

        Args:
            dets: Array of detections [x1, y1, x2, y2, score, class_id] or [x1, y1, x2, y2, score]

        Returns:
            np.ndarray of shape (N, 6): [x1, y1, x2, y2, track_id, class_id]
        """
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6))

        self.frame_count += 1

        # Standardize detections to 6 columns: [x1, y1, x2, y2, score, cls_id]
        if dets.shape[1] == 5:
            # Default cls = 2 (car)
            cls_col = np.full((len(dets), 1), 2.0)
            dets = np.hstack([dets, cls_col])

        scores = dets[:, 4] if len(dets) > 0 else np.array([])

        # Tier 1: High confidence detections
        tier1_mask = scores >= self.det_thresh if len(scores) > 0 else np.array([], dtype=bool)
        dets_tier1 = dets[tier1_mask] if len(dets) > 0 else np.empty((0, 6))

        # Tier 2: Low confidence detections (vehicles underexposed inside harsh shadows)
        tier2_mask = (scores >= self.min_conf) & (scores < self.det_thresh) if len(scores) > 0 else np.array([], dtype=bool)
        dets_tier2 = dets[tier2_mask] if len(dets) > 0 else np.empty((0, 6))

        # Predict new locations for all existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0) if len(to_del) > 0 else trks

        # Prepare velocities and observations for OC-SORT association
        velocities = np.array([
            trk.velocity if trk.velocity is not None else np.array((0.0, 0.0))
            for trk in self.trackers
        ])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([
            k_previous_obs(trk.observations, trk.age, self.delta_t)
            for trk in self.trackers
        ])

        # --- Round 1: High-confidence matching with Velocity Direction Consistency ---
        if len(self.trackers) > 0 and len(dets_tier1) > 0:
            # Compute velocity direction consistency cost
            Y, X = speed_direction_batch(dets_tier1[:, :5], k_observations)
            inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
            inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
            inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)

            diff_angle_cos = np.clip(inertia_X * X + inertia_Y * Y, -1.0, 1.0)
            diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

            valid_mask = np.ones(k_observations.shape[0])
            if k_observations.shape[1] >= 5:
                valid_mask[np.where(k_observations[:, 4] < 0)] = 0
            else:
                valid_mask[np.where(k_observations[:, 0] < 0)] = 0
            valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

            scores_rep = np.repeat(dets_tier1[:, 4][:, np.newaxis], len(self.trackers), axis=1)
            angle_cost = (valid_mask * diff_angle).T * scores_rep * self.inertia

            iou_matrix = iou_batch(dets_tier1[:, :4], trks[:, :4])
            total_affinity = iou_matrix + angle_cost

            if min(total_affinity.shape) > 0:
                matched_indices = linear_assignment(-total_affinity)
            else:
                matched_indices = np.empty((0, 2), dtype=int)

            matched_1 = []
            unmatched_dets_1 = []
            unmatched_trks_1 = []

            for d in range(len(dets_tier1)):
                if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                    unmatched_dets_1.append(d)

            for t in range(len(self.trackers)):
                if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                    unmatched_trks_1.append(t)

            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.iou_threshold:
                    unmatched_dets_1.append(m[0])
                    unmatched_trks_1.append(m[1])
                else:
                    matched_1.append(m)

            matched_1 = np.array(matched_1) if len(matched_1) else np.empty((0, 2), dtype=int)
            unmatched_dets_1 = np.array(unmatched_dets_1, dtype=int)
            unmatched_trks_1 = np.array(unmatched_trks_1, dtype=int)
        else:
            matched_1 = np.empty((0, 2), dtype=int)
            unmatched_dets_1 = np.arange(len(dets_tier1))
            unmatched_trks_1 = np.arange(len(self.trackers))

        # Update Tier-1 matches
        for m in matched_1:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(
                bbox=dets_tier1[det_idx, :5],
                cls_id=int(dets_tier1[det_idx, 5]),
            )

        # --- Round 2: Low-confidence BYTE association (Recover shadow-crushed vehicles) ---
        unmatched_trks_2 = unmatched_trks_1.copy()
        if len(dets_tier2) > 0 and len(unmatched_trks_1) > 0:
            u_trks = trks[unmatched_trks_1]
            iou_low = iou_batch(dets_tier2[:, :4], u_trks[:, :4])

            if min(iou_low.shape) > 0 and iou_low.max() >= self.iou_threshold:
                match_indices_low = linear_assignment(-iou_low)
                matched_trks_in_u = []
                for m in match_indices_low:
                    d_idx, u_idx = m[0], m[1]
                    if iou_low[d_idx, u_idx] >= self.iou_threshold:
                        actual_trk_idx = unmatched_trks_1[u_idx]
                        self.trackers[actual_trk_idx].update(
                            bbox=dets_tier2[d_idx, :5],
                            cls_id=int(dets_tier2[d_idx, 5]),
                        )
                        matched_trks_in_u.append(actual_trk_idx)
                unmatched_trks_2 = np.setdiff1d(unmatched_trks_1, np.array(matched_trks_in_u))

        # --- Round 3: Observation-Centric Recovery (OCR) ---
        if len(unmatched_dets_1) > 0 and len(unmatched_trks_2) > 0:
            left_dets = dets_tier1[unmatched_dets_1]
            left_last_boxes = last_boxes[unmatched_trks_2]
            iou_ocr = iou_batch(left_dets[:, :4], left_last_boxes[:, :4])

            if min(iou_ocr.shape) > 0 and iou_ocr.max() >= self.iou_threshold:
                rematched = linear_assignment(-iou_ocr)
                to_remove_dets = []
                to_remove_trks = []
                for m in rematched:
                    d_sub, t_sub = m[0], m[1]
                    if iou_ocr[d_sub, t_sub] >= self.iou_threshold:
                        actual_det = unmatched_dets_1[d_sub]
                        actual_trk = unmatched_trks_2[t_sub]
                        self.trackers[actual_trk].update(
                            bbox=dets_tier1[actual_det, :5],
                            cls_id=int(dets_tier1[actual_det, 5]),
                        )
                        to_remove_dets.append(actual_det)
                        to_remove_trks.append(actual_trk)
                unmatched_dets_1 = np.setdiff1d(unmatched_dets_1, np.array(to_remove_dets))
                unmatched_trks_2 = np.setdiff1d(unmatched_trks_2, np.array(to_remove_trks))

        # Update remaining unmatched trackers with None (advance Kalman filter)
        for t_idx in unmatched_trks_2:
            self.trackers[t_idx].update(None)

        # Initialize new trackers for remaining unmatched high-confidence detections
        for d_idx in unmatched_dets_1:
            trk = ShadowKalmanBoxTracker(
                bbox=dets_tier1[d_idx, :5],
                delta_t=self.delta_t,
                cls_id=int(dets_tier1[d_idx, 5]),
                min_hits=self.min_hits,
            )
            self.trackers.append(trk)

        # --- Assemble output tracks with Inertial Velocity Coasting ---
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # Retrieve latest position
            if trk.last_observation.sum() < 0:
                box = trk.get_state()[0]
            else:
                # If currently detected, use last observation; if coasting through shadow, use Kalman state
                if trk.time_since_update == 0:
                    box = trk.last_observation[:4]
                else:
                    box = trk.get_state()[0]

            is_confirmed = trk.is_confirmed or (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            is_active_or_coasting = (trk.time_since_update <= self.max_coast_frames)

            if is_confirmed and is_active_or_coasting:
                # Format: [x1, y1, x2, y2, track_id, class_id]
                track_id = trk.id + 1
                dom_cls = trk.get_dominant_class()
                ret.append(np.array([box[0], box[1], box[2], box[3], track_id, dom_cls]))

            # Prune dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 6))
