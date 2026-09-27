# Smart Traffic Vision: Multi-Camera Tracking & MQTT Streaming

Multi-camera traffic monitoring and hardware benchmarking pipeline for adaptive signal control. Ingests 1 to 8+ video feeds, detects and tracks vehicles in real-time with YOLO and ByteTrack, stabilizes classifications with temporal class voting, calculates lane-by-lane queued vehicle counts, measures live stopline discharge rates and upstream platoon inflow, and publishes structured JSON telemetry to MQTT.

## System Architecture

```
Camera Feeds (1..N) 
  └─► YOLO Detector 
        └─► ByteTrack Tracker 
              └─► Temporal Class Voting Filter 
                    ├─► Vectorized Lane Polygons (Queued & Moving Counts)
                    └─► Virtual Counting Gates (Live Discharge & Platoon Inflow)
                          └─► MQTT Telemetry Publisher ──► Traffic Controller (smart-traffic-sys)
```

### Core Components

- **Multi-Camera Ingestion:** Handles multiple local video files in synchronized lockstep mode (0% drift) or live RTSP streams with jitter-absorbing ring buffers.
- **YOLO Detection:** Detects vehicle classes (`car`, `motorcycle`, `bus`, `truck`, `three_wheeler`) with PyTorch or TensorRT FP16 execution.
- **ByteTrack Multi-Object Tracking:** Two-stage association ($D_{\text{high}} \ge 0.40$, $0.10 \le D_{\text{low}} < 0.40$) maintaining robust track identities through occlusions with zero bounding-box deformation.
- **Temporal Class Voting Filter:** Smooths classification over a 15-frame sliding window with confidence weighting, exponential decay, and hysteresis margins to eliminate car vs. pickup truck (รถกระบะ) flickering.
- **Vectorized Spatial Lane Analytics:** Evaluates vehicle road contact points against polygonal lane boundaries using `shapely.contains_xy`.
- **Virtual Counting Gates:** Directed 2D tripwires measuring **Stopline** clearance flux ($\text{cars/s}$) for adaptive green timing and **Ingress** flow for advance platoon arrival warnings.
- **MQTT Telemetry Publisher:** Emits standardized intersection snapshots every 2.0 seconds over MQTT to topic `traffic/counts`.

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

Launch all 5 production intersection cameras with live HUD:
```bash
python main.py run --cameras all --display
```

Run with custom target frame rate:
```bash
python main.py run --cameras all --display --fps 25
```

Run with higher inference resolution (for distant vehicle detail):
```bash
python main.py run --cameras all --display --imgsz 960
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

### B. Interactive Calibration Suite (Lanes & Counting Gates)

Launch the interactive calibration menu:
```bash
python main.py calibrate
```

Or calibrate a specific approach directly:
```bash
python main.py calibrate --approach north
python main.py calibrate --approach south
python main.py calibrate --approach northeast
```

#### Calibration GUI Controls:
* **Timeline Video Scrubbing:**
  * `[` / `]`: Jump $-5\text{s}$ / $+5\text{s}$ through the video
  * `Space`: Pause / Resume playback to freeze on a clear frame
* **Modes:**
  * `[1]`: **LANE Mode** — Left-click vertices to draw lane polygons. Press `C` or `Enter` to name and confirm.
  * `[2]`: **GATE Mode** — Click 2 points to place a directed counting gate:
    * `O`: **Stopline** (Queue Clearance / Discharge Rate)
    * `I`: **Ingress** (Upstream Platoon Inflow)
    * `Tab` / `T`: Cycle gate type
    * `F`: Flip normal direction vector
  * `[3]`: **SELECT Mode** — Click any lane or gate to inspect or delete (`D` / `Delete`).
* **Save Config:** Press `S` to save calibrated geometry directly into `config/config_<approach>.json` (creates automatic `.bak` backup).

### C. Run Hardware Benchmarking Suite

Profile GPU, VRAM, CPU load, and stage latencies across stream counts:
```bash
python main.py benchmark --streams 1 2 4 8 --duration 8 --models yolov8n.pt --mode both --save-plots
```

*(You can also run `python benchmark_hardware.py` directly).*

## Telemetry Payload Schema

Aggregated snapshots publish to `traffic/counts` every 2 seconds:

```json
{
  "intersectionId": "INT-001",
  "cameraId": "MULTI-CAM",
  "timestamp": "2026-09-22T12:00:00.000Z",
  "meta": {
    "frameId": "frame_100",
    "active_cameras": ["north", "south", "east", "west", "northeast"],
    "fps": 24.8,
    "mode": "batched_production"
  },
  "lanes": [
    {
      "laneId": "N1",
      "direction": "N",
      "count": 5,
      "queuedCount": 4,
      "movingCount": 1,
      "vehicles": {
        "queued": { "cars": 3, "motorbike": 1 },
        "moving": { "cars": 1, "motorbike": 0 }
      }
    }
  ],
  "traffic_flow": {
    "by_direction": {
      "N": {
        "inflow": 12,
        "cleared": 14,
        "live_discharge_rate": 0.85
      },
      "S": {
        "inflow": 0,
        "cleared": 9,
        "live_discharge_rate": 0.60
      },
      "E": {
        "inflow": 0,
        "cleared": 0,
        "live_discharge_rate": 0.00
      },
      "W": {
        "inflow": 0,
        "cleared": 8,
        "live_discharge_rate": 0.55
      }
    }
  }
}
```

### How the Controller Consumes This:
* **`lanes`:** Immediate queue count ($Q$) per approach for phase selection.
* **`by_direction[dir].live_discharge_rate`:** Dynamically replaces static saturation flow ($0.5\text{ cars/s}$) to calculate exact green time: $\text{Green} = \frac{Q}{\text{Discharge Rate}} + \text{Reaction Time}$.
* **`by_direction[dir].inflow`:** Advance warning of arriving vehicle platoons from upstream cameras.

## Repository Layout

```
smart-traffic-vision/
├── docs/                           # Benchmark and calibration guides
├── config/                         # Production lane polygon & gate JSON configurations
│   ├── config_north.json
│   ├── config_south.json
│   ├── config_east.json
│   ├── config_west.json
│   └── config_northeast.json
├── trt_pipeline/                   # Production vision pipeline modules
│   ├── display.py                  # Asynchronous GUI preview & NVENC video worker
│   ├── gates.py                    # Virtual counting gates & directional flow engine
│   ├── payload.py                  # Standardized MQTT JSON payload builder
│   ├── publisher.py                # Asynchronous MQTT publisher
│   ├── stream.py                   # Double-buffered stream ingestion ring worker
│   └── voter.py                    # Temporal class voting & anti-flicker filter
├── algorithm/                      # Multi-object tracking engines
│   ├── byetrack.py                 # Primary high-performance ByteTrack tracker
│   └── sort.py                     # Legacy SORT tracker fallback
├── tools/                          # Interactive calibration tools
│   └── segmentor.py                # Interactive Lane Polygon & Gate Calibration Suite
├── tests/                          # Automated unit test suite (35+ tests)
├── benchmark/                      # Experimental analysis & evaluation
├── main.py                         # Unified CLI dispatcher (run, benchmark, calibrate)
├── run_multi_camera.py             # Multi-camera production pipeline runner
└── benchmark_hardware.py           # Multi-camera hardware benchmark suite
```