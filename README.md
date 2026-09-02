# Smart Traffic Vision: Multi-Camera Tracking & MQTT Streaming

Multi-camera traffic monitoring and hardware benchmarking pipeline for adaptive signal control. Ingests 1 to 8+ video feeds, detects and tracks vehicles in real-time with YOLO and SORT, calculates lane-by-lane queued and moving vehicle counts, and publishes structured JSON telemetry to MQTT.

## System Architecture

```
Camera Feeds (1..N) -> YOLO Detector -> SORT Tracker -> Lane Analytics -> Lane Metrics -> MQTT Publisher -> Traffic Controller
```

### Core Components

- Ingests multiple video files, RTSP streams, or live network cameras via independent camera worker instances.
- Detects vehicle classes (`car`, `motorcycle`, `bus`, `truck`, `bicycle`) using thread-synchronized YOLO models.
- Tracks persistent trajectories across frames using SORT (7-state Kalman Filter and Hungarian assignment).
- Evaluates vehicle centroids against polygonal lane boundaries and classifies movement as queued or flowing.
- Emits aggregated intersection telemetry snapshots at regular intervals (default: 2.0s) over MQTT.

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

```bash
mosquitto -v -p 1883
```

## Quick Commands

### A. Run Multi-Camera Traffic Vision

Launch 4 camera feeds with live HUD window:
```bash
python main.py run --num-cams 4 --display
```

Launch 8 camera feeds:
```bash
python main.py run --num-cams 8 --display
```

Run specific directional cameras:
```bash
python main.py run --cameras north south east west --display
```

Run custom video files or RTSP streams:
```bash
python main.py run --videos videos/cam44_north.avi videos/cam43_south.avi --display
```

*(You can also run `python run_multi_camera.py` directly).*

### B. Run Hardware Benchmarking Suite

Profile GPU, VRAM, CPU load, and stage latencies across stream counts:
```bash
python main.py benchmark --streams 1 2 4 8 --duration 8 --models yolov8n.pt --mode both --save-plots
```

*(You can also run `python benchmark_hardware.py` directly).*

### C. Calibrate Lane Polygons

Mark perspective lane polygon boundaries on video frames:
```bash
python main.py calibrate --video videos/cam03_east.avi --sec 0 --n 1
```

*(You can also use the alias `python main.py segment` or run `python tools/segmentor.py` directly).*

## Telemetry Payload Schema

Aggregated snapshots publish to `traffic/counts` every 2 seconds:

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

## Specialized Documentation

- **[Hardware Benchmark Guide](docs/HARDWARE_BENCHMARK_GUIDE.md)**: Hardware profiling, latency stages, sizing evaluation, and telemetry export.
- **[Lane Calibration Guide](docs/CALIBRATION_GUIDE.md)**: Perspective calibration guide for defining road lane boundaries.
- **[Tracking Algorithms](algorithm/README.md)**: Details on SORT and OC-SORT tracker implementations.

## Repository Layout

```
smart-traffic-vision/
├── docs/                           # Benchmark and calibration guides
├── config/                         # Production lane polygon JSON configurations
├── trt_pipeline/                   # Core payload builder, MQTT publisher, and stream utilities
├── algorithm/                      # Standalone SORT and OC-SORT tracking implementations
├── tools/                          # Interactive calibration tools and optional detectors
│   ├── calibration/                # Calibration helpers
│   ├── face_detector/              # Optional face detection utilities
│   └── segmentor.py                # Lane polygon calibration tool
├── tests/                          # Automated unit tests
├── benchmark/
│   ├── notebooks/                  # Experimental analysis & evaluation notebooks
│   └── hardware-results/           # Benchmark telemetry reports (JSON, CSV, MD, PNG)
├── main.py                         # Unified CLI dispatcher (run, benchmark, calibrate/segment)
├── run_multi_camera.py             # Multi-camera traffic runner
└── benchmark_hardware.py           # Multi-camera hardware benchmark suite
```