# Lane Polygon Calibration Guide

Step-by-step guide for marking and calibrating perspective lane polygons on camera feeds for queue and flow counting.

## Overview

Lane polygons define the spatial zones on the road where vehicles are detected, counted, and classified as queued or moving. Calibrating these boundaries matches the perspective angle and optical zoom of each intersection camera.

## Using the Calibration Tool

Launch the interactive segmentor via the unified CLI:

```bash
python main.py calibrate --video videos/cam03_east.avi --sec 0 --n 1
```

Or run the tool script directly:

```bash
python tools/segmentor.py --video videos/cam03_east.avi --sec 0 --n 1
```

### Command Options

| Argument | Description |
| :--- | :--- |
| `--video` | Path to local video file or RTSP stream URL. |
| `--sec` | Video timestamp in seconds to capture preview frame (default: `0.0`). |
| `--n` | Number of consecutive lanes to calibrate in one session. |

## Interactive Calibration Workflow

1. A calibration window will display the video frame.
2. **Left-Click** around the road lane boundary in perimeter order (clockwise or counter-clockwise).
3. The tool renders red vertex markers and green perimeter lines as points are added.
| Key | Action |
| :--- | :--- |
| `r` | Reset all points to start over. |
| `s` | Print current coordinates array to console. |
| `q` / `ESC` | Save and finish the current lane calibration. |

```
Coordinate Array Output Example:
"polygon": [[240, 580], [620, 580], [890, 1080], [120, 1080]]
```

## Updating Intersection Configurations

Copy the coordinate array into the `"polygon"` field of the corresponding lane in `config/`:

- North approach: `config/config_north.json` (`N1`, `N2`, `N3`)
- South approach: `config/config_south.json` (`S1`, `S2`, `S3`)
- East approach: `config/config_east.json` (`E1`)
- West approach: `config/config_west.json` (`W1`, `W2`)

### Config Example

```json
"lanes": {
  "E1": {
    "direction": "E",
    "polygon": [
      [240, 580],
      [620, 580],
      [890, 1080],
      [120, 1080]
    ]
  }
}
```
