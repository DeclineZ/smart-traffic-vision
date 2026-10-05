# Smart Traffic Vision: Multi-Camera Tracking & MQTT Streaming

Multi-camera traffic measurement for adaptive signal control. It ingests camera streams (or recordings), detects and tracks vehicles with YOLO and ByteTrack, and measures current per-lane occupancy split into queued / moving / unknown vehicles plus interval stop-line and upstream gate crossings. Every lane reading carries its observation time and validity, and results are published to MQTT in the contract described in [docs/CONTROLLER_CONTRACT.md](docs/CONTROLLER_CONTRACT.md).

**Vision-only release:** `--output-mode controller` now projects measurements into
the existing `smart-traffic-sys/main` format, using its registered counting source
`CAM-01`. It sends complete, fresh stop-line occupancy and withholds the entire
snapshot when any required camera is unusable. No controller code or migration
is required. CLI output defaults to shadow. See
[main compatibility and startup](docs/CONTROLLER_MAIN_COMPATIBILITY.md) for producer
cutover, failure behavior and pilot limits.

Production readiness status: [docs/PRODUCTION_READINESS_REVIEW.md](docs/PRODUCTION_READINESS_REVIEW.md) (see "Status after remediation"). Deployment: [deploy/README.md](deploy/README.md).

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

- **Multi-Camera Ingestion:** Each live camera has its own reader thread with bounded open/read timeouts, reconnect with backoff and newest-frame delivery. Every frame carries its capture time, and frames older than `--max-frame-age` are dropped. A missing or slow camera never blocks the others; its lanes are published as invalid (unknown), not zero. Recordings replay in lockstep with media timestamps (`--replay-fps`).
- **YOLO Detection:** Detects vehicle classes (`car`, `motorcycle`, `bus`, `truck`, `three_wheeler`) with PyTorch or TensorRT FP16 execution.
- **ByteTrack Multi-Object Tracking:** Two-stage association ($D_{\text{high}} \ge 0.40$, $0.10 \le D_{\text{low}} < 0.40$) with threshold-aware assignment. The detector runs at `--conf 0.10` so the low-confidence stage receives input. Track class and confidence come from the detection matched to the track. Thresholds are not yet tuned on labelled clips.
- **Temporal Class Voting Filter:** Smooths classification over a 15-frame sliding window with confidence weighting, exponential decay, and hysteresis margins to eliminate car vs. pickup truck (รถกระบะ) flickering.
- **Lane Occupancy and Queue State:** The road-contact point (bottom-centre of the box) is assigned to at most one lane (first matching polygon, boundary included). Queued/moving state uses time-based, perspective-normalised speed with hysteresis (`trt_pipeline/motion.py`).
- **Virtual Counting Gates:** Directed tripwires with an explicit approach (`target_dir`) and globally unique IDs, bounded continuity across short occlusions, and per-interval counts with observed coverage. Directions without a gate are reported as not instrumented (`null`).
- **MQTT Publisher:** Publishes to `traffic/counts/shadow` every 1 s (`--pub-interval`), with `schemaVersion`, `sessionId` and `sequence`. Retained shadow health goes to `traffic/health/shadow/<source-id>` with a last will. Supports TLS (`mqtts://`, `--mqtt-tls-*`) and username/password. Occupancy is latest-only: nothing is queued while disconnected.

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
python main.py run --cameras north south --videos videos/cam44_north.avi videos/cam43_south.avi --display
```

*(You can also run `python run_multi_camera.py` directly).*

Send live counts to the existing controller main (stop its mock counts for this
intersection when switching producers):

```bash
python main.py run --cameras all --sources-file /path/to/sources.json --output-mode controller --controller-config config/controller_main.json --mqtt-broker mqtt://controller-host:1883 --health-file /path/to/health.json --display
```

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

The full definition is in [docs/CONTROLLER_CONTRACT.md](docs/CONTROLLER_CONTRACT.md). A real example produced from the five INT-001 recordings is in [tests/fixtures/vision_payload_v2.json](tests/fixtures/vision_payload_v2.json).

Key rules for consumers:

* `count = queuedCount + movingCount + unknownStateCount`. Queue-based timing must use `queuedCount`; `count` includes moving traffic.
* `valid: false` lanes have `null` counts. They are unknown, never zero.
* `timestamp` is when the cameras observed the traffic, not when the message was sent.
* `traffic_flow.interval` holds per-interval gate counts with `observedSec`. `null` means the direction has no gate.

### Existing main and future rich-schema consumers

The current main adapter sends occupancy in the existing `count` field and
excludes upstream regions. It suppresses incomplete snapshots so main's stale
fallback can take over at the next scheduled decision. The rules below concern
an optional future schema 2.0 consumer; they are not dependencies of the main
adapter. See [the main contract](docs/CONTROLLER_MAIN_COMPATIBILITY.md).

* Phase selection and green time use `queuedCount` (per-direction totals, or the busiest lane for max-pressure).
* Skipping an empty phase uses occupancy (`count`), and never treats an unknown approach as empty.
* Any approach without fully valid lanes, or data older than `FRESHNESS_MAX_AGE_MS`, puts the intersection on its fixed-time fallback.
* Stop-line departures feed a measured saturation-discharge estimate. It is recorded with every decision but used for green time only when `MP_USE_MEASURED_DISCHARGE=true`, because it is not yet validated.

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
│   ├── stream.py                   # Live camera reader (timeouts, reconnect) and file replay source
│   ├── motion.py                   # Time-based queued/moving classifier
│   ├── camera_health.py            # Black / frozen / blurred / shifted view detection
│   ├── lane_validation.py          # Calibration geometry validation
│   └── voter.py                    # Temporal class voting & anti-flicker filter
├── algorithm/                      # Multi-object tracking engines
│   ├── byetrack.py                 # Primary high-performance ByteTrack tracker
│   └── sort.py                     # Legacy SORT tracker fallback
├── tools/
│   ├── segmentor.py                # Interactive lane/gate calibration (versioned saves)
│   ├── validate_calibration.py     # Offline lane and gate validation
│   ├── calibration_history.py      # List / roll back calibration revisions
│   ├── trt_parity.py               # TensorRT vs PyTorch detection and lane-count parity
│   ├── soak.py                     # Long-run resource/latency test with camera outage injection
│   ├── eval_counts.py              # End-to-end count accuracy against labelled clips
│   ├── diagnostic_export.py        # Incident bundle (payloads, calibration, hashes, logs)
│   └── healthcheck.py              # Exit status for service supervisors
├── deploy/                         # systemd units, env/sources templates, commissioning checklist
├── tests/                          # Unit and runner tests (python -m unittest discover -s tests)
├── benchmark/                      # Experimental analysis & evaluation
├── main.py                         # Unified CLI dispatcher (run, benchmark, calibrate)
├── run_multi_camera.py             # Multi-camera production pipeline runner
└── benchmark_hardware.py           # Multi-camera hardware benchmark suite
```
