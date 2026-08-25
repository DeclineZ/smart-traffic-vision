# Smart Traffic Vision — Multi-Camera Tracking & MQTT Streaming

An AI-powered multi-camera traffic monitoring and tracking subsystem. It processes 4 simultaneous camera feeds (**North, South, East, West**), detects and tracks vehicles in real-time using YOLO and SORT, calculates lane-by-lane queued and moving vehicle counts, and streams atomic telemetry to an MQTT broker for adaptive traffic signal control.

---

## Key Features

- **Multi-Camera Orchestration**: Runs 4 video streams in parallel with shared GPU memory optimization.
- **YOLO & SORT Tracking**: Robust vehicle detection and tracking across cars, motorcycles, buses, and trucks.
- **Lane-by-Lane Queue & Flow Metrics**: Point-in-polygon geometry checks to categorize vehicles as `queued` vs `moving`.
- **Integrated MQTT Publisher**: Publishes aggregated intersection payloads (`INT-001`) to `traffic/counts` every 2 seconds.
- **2×2 Live HUD Window**: Synchronized visual grid with colored bounding boxes, track IDs, class badges, confidence scores, and lane boundary polygons.

---

## Quick Start Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone <repo-url>
cd smart-traffic-vision

# Create Python virtual environment (Python 3.10 - 3.12)
python -m venv .venv

# Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / Ubuntu:
source .venv/bin/activate
```

---

### 2. Install PyTorch with CUDA (Required for GPU Acceleration)

Install the build matching your GPU:

* **For Most NVIDIA GPUs (RTX 30-series / 40-series / GTX 16-series - CUDA 12.4 — Recommended)**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

* **For Latest GPUs (RTX 50-series Blackwell - CUDA 12.8)**:
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

* **Verify GPU is detected** (should print `True`):
  ```bash
  python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only')"
  ```

---

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```
*(Or if using [uv](https://github.com/astral-sh/uv) for fast installs: `uv pip install -r requirements.txt`).*

---

### 4. Start the MQTT Broker (Optional for Local Telemetry)
The system publishes traffic counts to an MQTT broker on port `1883`:

```bash
mosquitto -v -p 1883
```

---

### 5. Launch the Multi-Camera Vision Runner or Hardware Benchmark

```bash
# A. Run 4-Camera Vision Tracking with Live 2x2 Display Window:
python run_multi_camera.py --all --display

# B. Run the Multi-Camera Hardware Benchmark & Sizing Suite:
python benchmark_hardware.py --streams 4 8 --mode batched --display --save-plots
```

---

## CLI Options & Arguments (`run_multi_camera.py`)

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

## Lane & Polygon Calibration

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

## MQTT Telemetry Contract

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

## Hardware Benchmarking & Sizing Suite

To benchmark multi-camera performance (1 to 8+ feeds) and profile GPU/VRAM/CPU resource consumption on your hardware:

```bash
# Run 8-camera batched benchmark with frame skipping and visual performance charts
python benchmark_hardware.py --streams 1 2 4 8 --duration 8 --models yolov8n.pt yolov8s.pt --mode batched --frame-skips 0 1 --save-plots
```

For complete benchmarking instructions, platform guides (RTX PCs, Servers, Jetson Orin), and hardware procurement specifications, see the **[Hardware Benchmark & Sizing Guide](docs/HARDWARE_BENCHMARK_GUIDE.md)**.

---

## Project Structure

```
smart-traffic-vision/
├── docs/
│   └── HARDWARE_BENCHMARK_GUIDE.md # Comprehensive multi-camera hardware sizing & benchmark guide
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
├── benchmark_hardware.py       # Comprehensive multi-camera hardware benchmark & sizing suite
├── run_multi_camera.py         # Main entrypoint for 4-camera tracking & streaming
└── main.py                     # Legacy single-camera & polygon segmentor CLI
```