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

## Validating Lane Geometry

Before running the multi-camera pipeline, validate lane configurations to ensure all polygons are non-self-intersecting, have positive area, and check for overlapping zones:

```bash
# Validate all bundled intersection configurations
python -m tools.validate_calibration

# Or validate specific configuration files
python -m tools.validate_calibration --configs config/config_north.json config/config_south.json
```

- **Errors** (e.g. self-intersecting lines, duplicate points with zero area, non-finite coordinates) prevent the runner from starting and must be corrected.
- **Warnings** report positive-area overlaps between lanes in the same camera that require operator review. Overlaps do not prevent saving or execution, but indicate calibration that should be checked against physical lane markings.

## Measurement conventions the polygons must match

- **Lane membership uses the road-contact point**: the bottom-centre of each vehicle box (`lane_metrics.anchor: "bottom_center"`, the default). Draw lanes on the road surface where tyres touch, extending to the stop line. A polygon drawn around vehicle bodies misses vehicles whose bottom edge falls outside it. Polygons calibrated with the older box-centre convention can be kept temporarily with `"anchor": "center"`, but redraw them.
- **Overlaps**: a vehicle counts in the first lane (configuration order) whose polygon contains its point. The validator warns about every positive-area overlap; remove them unless deliberate.
- **Range**: end lanes where vehicles are still large enough to detect reliably. A lane reaching the vanishing point reports a lower bound there.
- **Gates** need an explicit approach, `"target_dir": "N" | "S" | "E" | "W"` (the approach whose traffic the gate measures), and an ID unique across all cameras. The segmentor fills `target_dir` from the gate ID (`GATE_N_STOPLINE` → `N`) or the camera's approach, and refuses to save gates without one.
- **Camera IDs** (`camera_info.camera_id`) must be registered in the controller's `cameras` table (INT-001 uses `INT-001-CAM-N/S/E/W/NE`, see smart-traffic-sys migration 012). Lane IDs must be listed in the intersection's `topology.lanes`.

## Revisions, reference frames and rollback

Each save from the segmentor:

1. validates lanes and gates; nothing is written if they are invalid;
2. records `calibration.revision`, `saved_by` (set `CALIBRATION_OPERATOR` to your name), `saved_at`, the frame `resolution` and a `reference_image` (the frame you calibrated on, in `config/reference/`);
3. archives the saved file under `config/history/<config name>/<revision>.json`, and keeps the previous file as `.bak`.

The runner reads these at startup:

- frames whose resolution differs from `calibration.resolution` are not used, and the camera is reported `resolution_mismatch`;
- every 30 s the live view is matched to the reference image (ORB features with a RANSAC fit, so passing vehicles are ignored). A shift of more than 12 px in three consecutive checks marks the camera `camera_shifted`, and its lanes become invalid until the view is restored or the camera is recalibrated. On the recordings the measured shift of the fixed cameras stayed within 4.2 px. A night view rarely matches a daytime reference, so the check is inconclusive (never flagged) at night; add a night reference if night shift detection is needed.

Calibrate from a live frame at commissioning so the reference matches the installed camera. The current reference images were taken from the recordings.

```bash
.venv\Scripts\python.exe -m tools.calibration_history list config/config_north.json
.venv\Scripts\python.exe -m tools.calibration_history restore config/config_north.json <revision>
```

A restore validates the archived revision before replacing the config. **Restart the runner after any save or restore**: calibration is read only at startup, and the revision in use is published in each payload (`cameras[].calibrationRevision`).

