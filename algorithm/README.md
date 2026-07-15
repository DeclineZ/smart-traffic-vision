# Tracking Algorithms

This directory contains implementations of object tracking algorithms compatible with Jetson Nano (Python 3.6.9, numpy 1.19.4, filterpy).

## Algorithms

### SORT (Simple Online and Realtime Tracking)
- **File**: `sort.py`
- **Description**: Classic SORT tracker using Kalman Filter and Hungarian algorithm
- **Key Features**:
  - Kalman Filter for motion prediction
  - IOU-based association
  - Simple and fast

### OC-SORT (Observation-Centric SORT)
- **File**: `ocsort.py`
- **Description**: Enhanced SORT with observation-centric improvements
- **Key Features**:
  - Velocity Direction Consistency (VDC) for better association
  - Observation-Centric Recovery (OCR) using last observations
  - Optional BYTE association for low confidence detections
  - Better handling of occlusions and missed detections

## Files

- **sort.py** - Original SORT implementation
- **ocsort.py** - OC-SORT implementation with observation-centric features
- **utils.py** - Shared utility functions for both trackers
- **test_ocsort.py** - Test script comparing SORT and OC-SORT
- **todo.md** - Implementation checklist (completed)

## Usage

### Basic SORT Usage

```python
from sort import Sort

tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

# Update with detections from each frame
detections = np.array([[x1, y1, x2, y2, score], ...])
tracks = tracker.update(detections)

# tracks format: [[x1, y1, x2, y2, track_id], ...]
```

### OC-SORT Usage

```python
from ocsort import OcSort

tracker = OcSort(
    det_thresh=0.6,      # Detection confidence threshold
    max_age=30,          # Max frames to keep alive a track
    min_hits=3,          # Minimum hits before reporting track
    iou_threshold=0.3,   # IOU threshold for matching
    delta_t=3,           # Frames to look back for velocity
    inertia=0.2,         # Weight for velocity consistency
    use_byte=True        # Enable BYTE association
)

# Update with detections from each frame
detections = np.array([[x1, y1, x2, y2, score], ...])
tracks = tracker.update(detections, min_conf=0.1)

# tracks format: [[x1, y1, x2, y2, track_id], ...]
```

## Key Differences: SORT vs OC-SORT

| Feature | SORT | OC-SORT |
|---------|------|---------|
| Association | IOU only | IOU + Velocity Direction |
| Occlusion Handling | Basic | Enhanced with OCR |
| Low Confidence Dets | Ignored | Optional BYTE association |
| Observation History | Not tracked | Full history maintained |
| Velocity Estimation | Kalman only | Observation-based + Kalman |

## Parameters

### Common Parameters (both SORT and OC-SORT)
- **max_age**: Maximum number of frames to keep a track alive without detections (default: 1 for SORT, 30 for OC-SORT)
- **min_hits**: Minimum number of detection hits before a track is reported (default: 3)
- **iou_threshold**: Minimum IOU for matching detections to tracks (default: 0.3)

### OC-SORT Specific Parameters
- **det_thresh**: Confidence threshold for primary detections (default: 0.6)
  - Detections above this threshold are used in first association round
- **delta_t**: Number of frames to look back for velocity estimation (default: 3)
  - Larger values = smoother velocity but less responsive
- **inertia**: Weight for velocity direction consistency (default: 0.2)
  - Range: 0.0 to 1.0
  - Higher values = more influence from velocity direction
- **use_byte**: Enable BYTE association for low confidence detections (default: False)
  - Helps recover tracks with temporarily low detection scores
- **min_conf**: Minimum confidence for BYTE association (default: 0.1)
  - Only used when use_byte=True

## Testing

Run the test script to compare SORT and OC-SORT:

```bash
cd algorithm
python test_ocsort.py
```

This will:
1. Generate synthetic moving objects
2. Track them with both SORT and OC-SORT
3. Compare tracking results over multiple frames
4. Show the difference in track counts and IDs

## Utility Functions

The `utils.py` module provides shared functions:

- **linear_assignment(cost_matrix)**: Solve linear assignment problem
- **iou_batch(bb_test, bb_gt)**: Batch IOU calculation
- **convert_bbox_to_z(bbox)**: Convert [x1,y1,x2,y2] to Kalman state [cx,cy,s,r]
- **convert_x_to_bbox(x, score)**: Convert Kalman state to bbox
- **k_previous_obs(observations, cur_age, k)**: Get observation k frames ago
- **speed_direction(bbox1, bbox2)**: Calculate normalized velocity between bboxes
- **speed_direction_batch(dets, tracks)**: Batch velocity calculation

## Compatibility

All implementations are compatible with:
- Python 3.6.9 (Jetson Nano default)
- numpy 1.19.4
- scipy (for linear_sum_assignment)
- filterpy (for KalmanFilter)

Key compatibility notes:
- No f-strings (uses `.format()`)
- No type hints in function signatures (only in docstrings)
- No walrus operator (`:=`)
- Only basic numpy operations supported by 1.19.4

## Performance Tips

### For High Frame Rate (>30 FPS)
```python
tracker = OcSort(
    det_thresh=0.7,
    max_age=20,
    min_hits=2,
    delta_t=2,
    inertia=0.3
)
```

### For Low Frame Rate (<15 FPS)
```python
tracker = OcSort(
    det_thresh=0.6,
    max_age=40,
    min_hits=3,
    delta_t=5,
    inertia=0.1
)
```

### For Crowded Scenes
```python
tracker = OcSort(
    det_thresh=0.7,
    iou_threshold=0.2,
    use_byte=True,
    inertia=0.3
)
```

### For Fast Moving Objects
```python
tracker = OcSort(
    delta_t=2,
    inertia=0.4,
    max_age=30
)
```

## References

- SORT: [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- OC-SORT: [Observation-Centric SORT](https://arxiv.org/abs/2203.14360)
- BYTE: [ByteTrack](https://arxiv.org/abs/2110.06864)

## License

Implementations adapted from official repositories with compatibility modifications for Jetson Nano.
