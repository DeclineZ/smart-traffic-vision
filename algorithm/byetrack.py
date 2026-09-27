"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
Reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864)

Engineered for real-time edge/server deployment in smart traffic vision.
Key advantages over legacy SORT:
1. Retains low-confidence detections (D_low) to maintain tracks through
   occlusions, shadows, and distant perspective entry.
2. Two-stage hierarchical IoU association eliminates track fragmentation.
3. Stable Kalman filter bounds prevent bounding box distortion/stretching.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from filterpy.kalman import KalmanFilter

try:
    from .utils import linear_assignment, iou_batch
except (ImportError, ValueError):
    from utils import linear_assignment, iou_batch


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single Object Tracklet with Kalman Filter motion estimation."""

    _count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def __init__(self, tlbr: np.ndarray, score: float, class_id: int = 0):
        # tlbr: [x1, y1, x2, y2]
        self._tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.frame_id = 0
        self.tracklet_len = 0
        self.time_since_update = 0

    @property
    def tlbr(self) -> np.ndarray:
        """Returns [x1, y1, x2, y2] bounding box."""
        if self.mean is None:
            return self._tlbr.copy()
        # Convert Kalman mean [cx, cy, s, r] to [x1, y1, x2, y2]
        cx, cy, s, r = self.mean[:4]
        s = max(1.0, float(s))
        r = float(np.clip(r, 0.1, 10.0))
        w = np.sqrt(s * r)
        h = s / max(1e-6, w)
        return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # Initialize Kalman state [cx, cy, s, r, 0, 0, 0, 0]
        w = max(1.0, self._tlbr[2] - self._tlbr[0])
        h = max(1.0, self._tlbr[3] - self._tlbr[1])
        cx = self._tlbr[0] + w * 0.5
        cy = self._tlbr[1] + h * 0.5
        s = w * h
        r = w / max(1e-6, h)

        self.mean = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32)

        self.tracklet_len = 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Re-activate a lost track with a new matched observation."""
        self.update_kalman(new_track.tlbr)
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
        self.score = new_track.score
        self.class_id = new_track.class_id
        if new_id:
            self.track_id = self.next_id()

    def predict(self):
        """Advance Kalman state by 1 time step with safety clamping."""
        if self.mean is None:
            return

        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0  # Constant velocity model

        Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01]).astype(np.float32)

        self.mean = np.dot(F, self.mean)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q

        # Safety check against numerical divergence
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.covariance)):
            w = max(1.0, self._tlbr[2] - self._tlbr[0])
            h = max(1.0, self._tlbr[3] - self._tlbr[1])
            cx = self._tlbr[0] + w * 0.5
            cy = self._tlbr[1] + h * 0.5
            s = max(10.0, w * h)
            r = float(np.clip(w / max(1.0, h), 0.1, 10.0))
            self.mean = np.array([cx, cy, s, r, 0, 0, 0, 0], dtype=np.float32)
            self.covariance = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32)
        else:
            # Safety clamps to prevent Kalman aspect ratio explosion
            self.mean[2] = max(10.0, float(self.mean[2]))  # scale (area)
            self.mean[3] = float(np.clip(self.mean[3], 0.1, 10.0))  # aspect ratio

        self.time_since_update += 1

    def update_kalman(self, measurement: np.ndarray):
        """Update Kalman filter state with new detection observation."""
        w = max(1.0, measurement[2] - measurement[0])
        h = max(1.0, measurement[3] - measurement[1])
        cx = measurement[0] + w * 0.5
        cy = measurement[1] + h * 0.5
        s = w * h
        r = w / max(1e-6, h)

        z = np.array([cx, cy, s, r], dtype=np.float32)
        H = np.zeros((4, 8), dtype=np.float32)
        H[:4, :4] = np.eye(4, dtype=np.float32)

        R = np.diag([1, 1, 10, 10]).astype(np.float32)

        y = z - np.dot(H, self.mean)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = np.dot(np.dot(self.covariance, H.T), S_inv)

        self.mean = self.mean + np.dot(K, y)
        I = np.eye(8, dtype=np.float32)
        self.covariance = np.dot(I - np.dot(K, H), self.covariance)

        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.covariance)):
            self.mean = np.array([cx, cy, max(10.0, s), float(np.clip(r, 0.1, 10.0)), 0, 0, 0, 0], dtype=np.float32)
            self.covariance = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float32)
        else:
            self.mean[2] = max(10.0, float(self.mean[2]))
            self.mean[3] = float(np.clip(self.mean[3], 0.1, 10.0))
        self._tlbr = measurement.copy()

    def update(self, new_track: STrack, frame_id: int):
        """Update state with matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0

        self.update_kalman(new_track._tlbr)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTrack:
    """
    ByteTrack Multi-Camera Object Tracker.
    Drop-in replacement for legacy SORT with identical numpy input/output format.
    """

    def __init__(
        self,
        track_thresh: float = 0.40,
        low_thresh: float = 0.10,
        match_thresh: float = 0.70,
        max_age: int = 30,
        min_hits: int = 2,
    ):
        """
        Args:
            track_thresh: Threshold separating high-confidence and low-confidence detections.
            low_thresh: Minimum confidence to consider a detection in secondary association.
            match_thresh: Maximum IoU distance (1 - IoU) allowed for association.
            max_age: Maximum consecutive frames a lost track is retained before deletion.
            min_hits: Number of consecutive detections required before activating a new track.
        """
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.frame_id = 0
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.unconfirmed_stracks: List[STrack] = []

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Update tracker with detections from current frame.

        Args:
            dets: Array of shape (N, 5) or (N, 6)
                  Format: [[x1, y1, x2, y2, score], ...] or
                          [[x1, y1, x2, y2, score, class_id], ...]

        Returns:
            tracked_objs: Array of shape (M, 6)
                  Format: [[x1, y1, x2, y2, track_id, class_id], ...]
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        elif dets.shape[1] == 5:
            # Append default class_id = 0 if not provided
            dets = np.column_stack([dets, np.zeros(len(dets), dtype=np.float32)])

        scores = dets[:, 4]
        remain_inds = scores >= self.track_thresh
        low_inds = (scores >= self.low_thresh) & (~remain_inds)

        dets_high = dets[remain_inds]
        dets_low = dets[low_inds]

        # 1. Create STrack objects
        detections_high = [
            STrack(d[:4], d[4], int(d[5]))
            for d in dets_high
        ]
        detections_low = [
            STrack(d[:4], d[4], int(d[5]))
            for d in dets_low
        ]

        # 2. Predict Kalman locations for all existing tracks
        for track in (self.tracked_stracks + self.lost_stracks + self.unconfirmed_stracks):
            track.predict()

        # 3. First Association: D_high with (tracked_stracks + lost_stracks)
        strack_pool = self.tracked_stracks + self.lost_stracks
        dists = self._iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection = self._linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 4. Second Association: D_low with remaining unmatched tracked tracks
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = self._iou_distance(r_tracked_stracks, detections_low)
        matches, u_track_2, _ = self._linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 5. Deal with unconfirmed tracks using remaining D_high
        detections_remaining = [detections_high[i] for i in u_detection]
        dists = self._iou_distance(self.unconfirmed_stracks, detections_remaining)
        matches, u_unconfirmed, u_detection_final = self._linear_assignment(dists, thresh=0.7)

        kept_unconfirmed: List[STrack] = []
        for itracked, idet in matches:
            track = self.unconfirmed_stracks[itracked]
            track.update(detections_remaining[idet], self.frame_id)
            if track.tracklet_len >= self.min_hits:
                track.is_activated = True
                activated_stracks.append(track)
            else:
                kept_unconfirmed.append(track)

        for it in u_unconfirmed:
            track = self.unconfirmed_stracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 6. Initialize new tracks from unmatched high-confidence detections
        for inew in u_detection_final:
            track = detections_remaining[inew]
            if track.score >= self.track_thresh:
                track.activate(KalmanFilter(dim_x=8, dim_z=4), self.frame_id)
                if self.min_hits <= 1:
                    track.is_activated = True
                    activated_stracks.append(track)
                else:
                    track.is_activated = False
                    kept_unconfirmed.append(track)

        # 7. Update lost tracks lifecycle
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        # Merge track lists
        self.unconfirmed_stracks = kept_unconfirmed

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked and t.is_activated
        ]
        seen_ids = set()
        merged_tracked = []
        for t in (self.tracked_stracks + activated_stracks + refind_stracks):
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                merged_tracked.append(t)
        self.tracked_stracks = merged_tracked

        self.lost_stracks = [
            t for t in self.lost_stracks if t.state == TrackState.Lost
        ]
        seen_lost = set()
        merged_lost = []
        for t in (self.lost_stracks + lost_stracks):
            if t.track_id not in seen_ids and t.track_id not in seen_lost:
                seen_lost.add(t.track_id)
                merged_lost.append(t)
        self.lost_stracks = merged_lost

        # 8. Build output array in standard [x1, y1, x2, y2, track_id, class_id] format
        output_tracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                box = track.tlbr
                output_tracks.append([
                    box[0], box[1], box[2], box[3],
                    float(track.track_id),
                    float(track.class_id),
                ])

        if len(output_tracks) == 0:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(output_tracks, dtype=np.float32)

    def _iou_distance(self, atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
        """Computes cost matrix based on IoU distance: Cost = 1 - IoU."""
        if len(atracks) == 0 or len(btracks) == 0:
            return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

        a_boxes = np.array([t.tlbr for t in atracks], dtype=np.float32)
        b_boxes = np.array([t.tlbr for t in btracks], dtype=np.float32)
        ious = iou_batch(a_boxes, b_boxes)
        return 1.0 - ious

    def _linear_assignment(
        self, cost_matrix: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Performs Hungarian linear sum assignment and filters cost > thresh."""
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                list(range(cost_matrix.shape[0])),
                list(range(cost_matrix.shape[1])),
            )

        matches_raw = linear_assignment(cost_matrix)
        matched_indices = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))

        for m in matches_raw:
            r, c = int(m[0]), int(m[1])
            if cost_matrix[r, c] <= thresh:
                matched_indices.append([r, c])
                if r in unmatched_a:
                    unmatched_a.remove(r)
                if c in unmatched_b:
                    unmatched_b.remove(c)

        if len(matched_indices) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matched_indices, dtype=int)

        return matches, unmatched_a, unmatched_b
