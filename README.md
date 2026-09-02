# Smart Traffic Vision: Multi-Camera Tracking & MQTT Streaming

Multi-camera traffic monitoring pipeline for intersection management. It ingests 1 to 8+ video feeds, detects and tracks vehicles with YOLO and SORT, calculates lane queue and movement counts through point-in-polygon tests, and publishes structured JSON telemetry to MQTT for adaptive signal controllers.

## Overview

- Ingests multiple video streams or RTSP feeds in parallel.
- Detects cars, motorcycles, buses, and trucks using YOLO models.
- Tracks persistent vehicle trajectories with SORT across frames.
- Classifies vehicles as queued or moving based on speed and lane boundaries.
- Emits consolidated MQTT payloads per intersection window.
- Displays an adaptive multi-camera grid preview with lane overlays and tracking IDs.

## Setup

### 1. Create Virtual Environment

```bash
git clone <repo-url>
cd smart-traffic-vision

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA

Install the package built for your GPU:

For CUDA 12.4 (RTX 30 / 40 series, GTX 16 series):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

For CUDA 12.8 (RTX 50 series):
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify GPU availability:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MQTT Broker (Optional for Local Testing)

To run a local Mosquitto broker on port 1883:

```bash
mosquitto -v -p 1883
```

## Running the Pipeline

### Multi-Camera Traffic Runner

Run 4 cameras with live visual grid:
```bash
python run_multi_camera.py --num-cams 4 --display
```

Run 8 camera feeds:
```bash
python run_multi_camera.py --num-cams 8 --display
```

Run specific directional cameras:
```bash
python run_multi_camera.py --cameras north south east west --display
```

Run custom video files or RTSP streams:
```bash
python run_multi_camera.py --videos videos/cam44_north.avi videos/cam43_south.avi --display
```

### Hardware Benchmark Suite

Profile GPU, VRAM, CPU load, and stage latencies across stream counts:
```bash
python benchmark_hardware.py --streams 1 2 4 8 --duration 8 --models yolov8n.pt --mode both --save-plots
```

See [Hardware Benchmark Guide](docs/HARDWARE_BENCHMARK_GUIDE.md) for full benchmark configuration options and sizing metrics.

## CLI Options (`run_multi_camera.py`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--num-cams` | `4` | Number of camera feeds to spawn (1 to 8+). |
| `--cameras` | `None` | Named camera feeds to run (`north`, `south`, `east`, `west`, or `all`). |
| `--configs` | `None` | Custom JSON configuration paths per feed. |
| `--videos` | `None` | Video source file paths or RTSP URLs. |
| `--model` | `yolov8n.pt` | YOLO weights path (`yolov8n.pt`, `yolov8s.pt`, etc.). |
| `--device` | `auto` | Compute device (`cuda:0`, `cpu`). |
| `--conf` | `0.20` | Detection confidence threshold. |
| `--fps` | `25.0` | Target processing FPS per feed. |
| `--pub-interval` | `2.0` | MQTT broadcast interval in seconds. |
| `--mqtt-broker` | `mqtt://localhost:1883` | Target MQTT broker URL. |
| `--mqtt-topic` | `traffic/counts` | Target MQTT topic. |
| `--intersection-id` | `INT-001` | Intersection identifier for emitted payloads. |
| `--display` | `False` | Opens the adaptive multi-camera HUD window. |

## Calibrating Lane Polygons

To adjust lane boundaries to camera perspectives:

```bash
python main.py segment --video videos/cam03_east.avi --n 1
```

1. Click perimeter points around the lane boundary in sequential order.
2. Press `q` when complete to print coordinates.
3. Update the `"polygon"` field in the corresponding config:
   - `config/config_north.json` (`N1`, `N2`, `N3`)
   - `config/config_south.json` (`S1`, `S2`, `S3`)
   - `config/config_east.json` (`E1`)
   - `config/config_west.json` (`W1`, `W2`)

## MQTT Telemetry Schema

Aggregated payloads publish to `traffic/counts`:

```json
{
  "intersectionId": "INT-001",
  "cameraId": "MULTI-CAM",
  "timestamp": "2026-09-02T03:00:00.000Z",
  "meta": {
    "frameId": "frame_42",
    "active_cameras": ["CAM_01", "CAM_02", "CAM_03", "CAM_04"]
  },
  "lanes": [
    {
      "laneId": "N1",
      "direction": "N",
      "count": 5,
      "queuedCount": 2,
      "movingCount": 3,
      "vehicles": {
        "queued": { "cars": 2, "motorbike": 0 },
        "moving": { "cars": 3, "motorbike": 0 }
      }
    }
  ]
}
```

## Repository Structure

```
smart-traffic-vision/
├── docs/
│   └── HARDWARE_BENCHMARK_GUIDE.md # Hardware benchmarking and stream sizing guide
├── config/
│   ├── config_north.json           # North lane polygons and camera configuration
│   ├── config_south.json           # South lane polygons
│   ├── config_east.json            # East lane polygons
│   └── config_west.json            # West lane polygons
├── trt_pipeline/
│   ├── payload.py                  # Payload builder and lane aggregation
│   ├── publisher.py                # Thread-safe MQTT client wrapper
│   └── tools.py                    # Config loader and system utilities
├── algorithm/
│   └── sort.py                     # SORT tracker implementation
├── videos/                         # Sample 1080p traffic clips
├── benchmark_hardware.py           # Multi-camera hardware benchmarking suite
├── run_multi_camera.py             # Multi-camera tracking and MQTT streaming runner
└── main.py                         # Single-camera runner and polygon calibration utility
```