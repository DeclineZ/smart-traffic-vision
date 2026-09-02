# Tracking Algorithms

Standalone multi-object tracking (MOT) implementations used across the traffic vision pipelines.

## Available Algorithms

### SORT (Simple Online and Realtime Tracking)
- Implemented in `sort.py` via class `Sort`.
- Uses a 7-state Kalman Filter $[u, v, s, r, \dot{u}, \dot{v}, \dot{s}]$ for constant-velocity bounding box motion estimation.
- Matches tracks to detections using IOU cost matrices and the Hungarian assignment algorithm.
- Lightweight and fast, suitable for real-time edge processing across 1 to 8+ parallel camera streams.

### OC-SORT (Observation-Centric SORT)
- Implemented in `ocsort.py` via class `OcSort`.
- Uses Velocity Direction Consistency (VDC) to improve track association during turns.
- Uses Observation-Centric Recovery (OCR) to recover track IDs when vehicles re-emerge from occlusions.
- Runs a secondary association step for lower-confidence detections.

## Usage

### Basic SORT Tracker

```python
import numpy as np
from algorithm.sort import Sort

tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

# Update with detections: [[x1, y1, x2, y2, score], ...]
detections = np.array([
    [100, 150, 220, 310, 0.85],
    [450, 200, 580, 420, 0.92]
])

# Output tracks: [[x1, y1, x2, y2, track_id], ...]
tracked_objects = tracker.update(detections)
```

### OC-SORT Tracker

```python
import numpy as np
from algorithm.ocsort import OcSort

tracker = OcSort(
    det_thresh=0.20,
    max_age=30,
    min_hits=3,
    iou_threshold=0.30
)

# Output tracks: [[x1, y1, x2, y2, track_id, cls_id, conf], ...]
tracked_objects = tracker.update(detections)
```

## Directory Files

- `sort.py`: Standard SORT tracking implementation.
- `ocsort.py`: Observation-Centric SORT tracking implementation.
- `utils.py`: Shared linear assignment (Hungarian algorithm) and IOU computation helpers.
