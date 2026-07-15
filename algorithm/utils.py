"""
Utility functions for tracking algorithms.
Compatible with numpy 1.19.4 and Python 3.6.9
"""
from __future__ import print_function
import numpy as np


def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using scipy or lap.
    
    Args:
        cost_matrix: Cost matrix for assignment
        
    Returns:
        Array of matched indices [[det_idx, trk_idx], ...]
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    
    Args:
        bb_test: Detections array of shape (N, 4+)
        bb_gt: Trackers array of shape (M, 4+)
        
    Returns:
        IOU matrix of shape (N, M)
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + 
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2, ...]
        
    Returns:
        Array of shape (4, 1) with [cx, cy, scale, ratio]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)  # add small epsilon to avoid division by zero
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    
    Args:
        x: State vector [cx, cy, scale, ratio, ...]
        score: Optional confidence score
        
    Returns:
        Bounding box [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if score is None:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score]).reshape((1, 5))


def k_previous_obs(observations, cur_age, k):
    """
    Get the observation from k frames ago.
    
    Args:
        observations: Dictionary mapping age to bbox observations
        cur_age: Current age of the tracker
        k: Number of frames to look back (delta_t)
        
    Returns:
        Previous observation bbox or [-1, -1, -1, -1, -1] if not found
    """
    if len(observations) == 0:
        return np.array([-1, -1, -1, -1, -1])
    
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    """
    Calculate normalized velocity direction between two bboxes.
    
    Args:
        bbox1: Previous bbox [x1, y1, x2, y2, ...]
        bbox2: Current bbox [x1, y1, x2, y2, ...]
        
    Returns:
        Normalized velocity vector [dy, dx]
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
    return speed / norm


def speed_direction_batch(dets, tracks):
    """
    Calculate velocity direction for batch of detections and tracks.
    
    Args:
        dets: Detections array of shape (N, 4+) 
        tracks: Previous observations array of shape (M, 4+)
        
    Returns:
        dy, dx: Direction components of shape (M, N)
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx
