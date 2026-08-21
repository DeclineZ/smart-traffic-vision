# 🚦 Smart Traffic Vision — Multi-Camera Tracking & MQTT Streaming

An AI-powered multi-camera traffic monitoring and tracking subsystem. It processes 4 simultaneous camera feeds (**North, South, East, West**), detects and tracks vehicles in real-time using YOLO and SORT, calculates lane-by-lane queued and moving vehicle counts, and streams atomic telemetry to an MQTT broker for adaptive traffic signal control.

---

## 🌟 Key Features

- **Multi-Camera Orchestration**: Runs 4 video streams in parallel with shared GPU memory optimization.
- **YOLO & SORT Tracking**: Robust vehicle detection and tracking across cars, motorcycles, buses, and trucks.
- **Lane-by-Lane Queue & Flow Metrics**: Point-in-polygon geometry checks to categorize vehicles as `queued` vs `moving`.
- **Integrated MQTT Publisher**: Publishes aggregated intersection payloads (`INT-001`) to `traffic/counts` every 2 seconds.
- **2×2 Live HUD Window**: Synchronized visual grid with colored bounding boxes, track IDs, class badges, confidence scores, and lane boundary polygons.
- **GPU Accelerated**: Optimized for NVIDIA GPUs (RTX series, Jetson, and CUDA 12+).

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Make sure you have [uv](https://github.com/astral-sh/uv) or Python 3.11+ installed:

```bash
# Clone the repository (refactor/websocket-to-mqtt branch)
cd smart-traffic-vision

# Install dependencies
uv pip install -r requirements.txt
```

> **GPU Note**: For CUDA GPU acceleration, ensure PyTorch with CUDA is installed (e.g. `uv pip install torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cu128`).

---

### 2. Start the MQTT Broker
The system requires an MQTT broker listening on port `1883`.

- **Option A (Production / Mosquitto)**:
  ```bash
  mosquitto -v -p 1883
  ```
- **Option B (Lightweight Built-in Python Broker)**:
  ```bash
  uv run python run_broker.py
  ```

---

### 3. Launch the Multi-Camera Vision Runner

Run all 4 intersection feeds with real-time visual display:

```bash
# Standard run (YOLOv8 Nano, auto-device)
uv run python run_multi_camera.py --all --display

# High-accuracy run (YOLOv8 Medium with 0.15 threshold for smaller motorbikes)
uv run python run_multi_camera.py --all --display --model yolov8m.pt --conf 0.15

# Headless mode (maximum FPS, no GUI window — ideal for background servers)
uv run python run_multi_camera.py --all --model yolov8s.pt
```

---

## ⚙️ CLI Options & Arguments (`run_multi_camera.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--all` | Flag | `True` | Run all 4 camera streams simultaneously (`north`, `south`, `east`, `west`). |
| `--camera` | `str` | `all` | Select a single camera approach: `north`, `south`, `east`, `west`, or `all`. |
| `--display` | Flag | `False` | Opens the 2×2 live OpenCV monitoring window with visual overlays. |
| `--model` | `str` | `yolov8n.pt` | YOLO weights path/name (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`). |
| `--conf` | `float` | `0.20` | Confidence detection threshold (e.g. `0.15` for small/distant motorcycles). |
| `--device` | `str` | `auto` | Compute device (`cuda:0`, `cuda`, `cpu`). |
| `--fps` | `float` | `25.0` | Target processing frame rate per stream. |
| `--pub-interval`| `float` | `2.0` | MQTT broadcast interval in seconds. |

---

## 🎯 Lane & Polygon Calibration

To customize or fine-tune lane polygons to match road geometry:

```bash
# Run the polygon segmentor on any camera video
uv run python main.py segment --video videos/cam03_east.avi --n 1
```

1. **Left-Click** corner points around the lane boundaries in **clockwise or counter-clockwise perimeter order**.
2. Press **`q`** when finished to output the coordinate array.
3. Paste the coordinates into the `"polygon"` field in the corresponding JSON config:
   - `config/config_north.json` (`N1`, `N2`, `N3`)
   - `config/config_south.json` (`S1`, `S2`, `S3`)
   - `config/config_east.json` (`E1`)
   - `config/config_west.json` (`W1`, `W2`)

---

## 📡 MQTT Telemetry Contract

The multi-camera runner publishes consolidated JSON snapshots to `traffic/counts` every 2 seconds:

```json
{
  "intersection_id": "INT-001",
  "camera_id": "MULTI-CAM",
  "frame_idx": 42,
  "timestamp": 1787280724.668,
  "lanes": [
    {
      "lane_id": "N1",
      "direction": "N",
      "count": 5,
      "queued_count": 2,
      "moving_count": 3,
      "vehicles": {
        "queued": { "cars": 2, "motorbike": 0 },
        "moving": { "cars": 3, "motorbike": 0 }
      }
    },
    { "lane_id": "S1", "direction": "S", "count": 8, "vehicles": { ... } },
    { "lane_id": "E1", "direction": "E", "count": 2, "vehicles": { ... } },
    { "lane_id": "W1", "direction": "W", "count": 1, "vehicles": { ... } }
  ]
}
```

---

## 📂 Project Structure

```
smart-traffic-vision/
├── config/
│   ├── config_north.json       # North approach lane definitions & MQTT settings
│   ├── config_south.json       # South approach lane definitions
│   ├── config_east.json        # East approach lane definitions
│   └── config_west.json        # West approach lane definitions
├── trt_pipeline/
│   ├── payload.py              # JSON payload schemas & metrics aggregation
│   ├── publisher.py            # Thread-safe MQTT client wrapper
│   ├── tools.py                # Config loader & system utilities
│   └── tracking.py             # SORT Kalman filter tracking
├── videos/                     # 1080p sample intersection recordings
├── run_multi_camera.py         # 🌟 Main entrypoint for 4-camera tracking & streaming
├── run_broker.py               # Lightweight local MQTT broker (for testing)
└── main.py                     # Legacy single-camera & polygon segmentor CLI
```