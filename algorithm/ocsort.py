"""
OC-SORT: Observation-Centric SORT
Simplified implementation for Jetson Nano compatibility.
Compatible with numpy 1.19.4, Python 3.6.9, and filterpy.
"""
from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter

# Support both relative and absolute imports
try:
    from .utils import (
        linear_assignment, 
        iou_batch,
        convert_bbox_to_z, 
        convert_x_to_bbox,
        k_previous_obs,
        speed_direction,
        speed_direction_batch
    )
except (ImportError, ValueError):
    from utils import (
        linear_assignment, 
        iou_batch,
        convert_bbox_to_z, 
        convert_x_to_bbox,
        k_previous_obs,
        speed_direction,
        speed_direction_batch
    )

np.random.seed(0)


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    using velocity direction consistency.
    
    Args:
        detections: Array of detections in format [[x1,y1,x2,y2,score], ...]
        trackers: Array of predicted tracker positions [[x1,y1,x2,y2,0], ...]
        iou_threshold: Minimum IOU for match
        velocities: Velocity vectors for each tracker
        previous_obs: Previous observations for each tracker
        vdc_weight: Weight for velocity direction consistency (inertia)
        
    Returns:
        matches, unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    # Calculate velocity direction consistency
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    # Create validity mask for observations
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    
    # Calculate IOU matrix
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)
    
    # Angle difference cost weighted by validity and detection scores
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    # Perform linear assignment
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    # Find unmatched detections and trackers
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    Extends SORT's KalmanBoxTracker with observation history and velocity tracking.
    """
    count = 0
    
    def __init__(self, bbox, delta_t=3):
        """
        Initialises a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2, score, ...]
            delta_t: Number of frames to look back for velocity estimation
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # OC-SORT specific attributes
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2, score, ...] or None
        """
        if bbox is not None:
            # Estimate velocity if we have previous observations
            if self.last_observation.sum() >= 0:  # has previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                
                # Calculate velocity direction using observations delta_t steps away
                self.velocity = speed_direction(previous_box, bbox)
            
            # Store new observations
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)


class OcSort(object):
    """
    OC-SORT tracker with observation-centric association.
    """
    def __init__(self, det_thresh=0.6, max_age=30, min_hits=3, 
                 iou_threshold=0.3, delta_t=3, asso_func="iou", 
                 inertia=0.2, use_byte=False):
        """
        Sets key parameters for OC-SORT
        
        Args:
            det_thresh: Detection confidence threshold for primary association
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits to start outputting track
            iou_threshold: Minimum IOU for association
            delta_t: Number of frames for velocity estimation
            asso_func: Association function (only "iou" supported in simplified version)
            inertia: Weight for velocity direction consistency
            use_byte: Whether to use BYTE association for low confidence detections
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = iou_batch  # simplified version only supports IOU
        self.inertia = inertia
        self.use_byte = use_byte
        KalmanBoxTracker.count = 0

    def update(self, dets=np.empty((0, 5)), min_conf=0.1):
        """
        Update tracker with detections from current frame.
        
        Args:
            dets: Detections array [[x1,y1,x2,y2,score], ...] or empty array
            min_conf: Minimum confidence for BYTE association (if use_byte=True)
            
        Returns:
            Array of active tracks [[x1,y1,x2,y2,track_id], ...]
        """
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5))

        self.frame_count += 1
        
        # Split detections by confidence for BYTE
        scores = dets[:, 4] if len(dets) > 0 else np.array([])
        remain_inds = scores > self.det_thresh if len(scores) > 0 else np.array([], dtype=bool)
        dets_first = dets[remain_inds] if len(dets) > 0 else np.empty((0, 5))
        
        dets_second = np.empty((0, 5))
        if self.use_byte and len(dets) > 0:
            inds_low = scores > min_conf
            inds_high = scores < self.det_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = dets[inds_second]

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Prepare velocity and observation data for association
        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) 
             for trk in self.trackers]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) 
             for trk in self.trackers]
        )

        # First round of association with high confidence detections
        matched, unmatched_dets, unmatched_trks = associate(
            dets_first, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )
        
        for m in matched:
            self.trackers[m[1]].update(dets_first[m[0], :])

        # Second round: BYTE association with low confidence detections
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Third round: Observation-Centric Recovery (OCR)
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets_first[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_first[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Update unmatched trackers with None
        for m in unmatched_trks:
            self.trackers[m].update(None)

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_first[i, :], delta_t=self.delta_t)
            self.trackers.append(trk)
        
        # Output tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # Use last observation if available, otherwise use Kalman prediction
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
