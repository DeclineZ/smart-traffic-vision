# Smart Traffic Vision

Multi-camera vehicle detection and tracking for adaptive traffic control. The runner measures current lane occupancy, queued/moving/unknown vehicles, and directed gate crossings. Camera health and observation times travel with the measurements.

The default model is `models/yolo26s_thai_traffic_manual_v1_640.engine`, a TensorRT build of `yolo26s_thai_traffic_manual_v1_640.pt` (the runner falls back to the `.pt` when no engine has been built on the machine). Engines are GPU-specific: build one with `export_trt.py` and check it with `tools.trt_parity` on the deployment hardware. ByteTrack is the default tracker.

## Guides

| Guide | Use |
|---|---|
| [Existing controller integration](docs/CONTROLLER_MAIN_COMPATIBILITY.md) | Wire format, producer cutover, freshness and fallback limitations |
| [Measurement contract](docs/CONTROLLER_CONTRACT.md) | Rich shadow/local payload fields and measurement meanings |
| [Calibration](docs/CALIBRATION_GUIDE.md) | Lane polygons, gates, references and rollback |
| [Validation and hardware](docs/HARDWARE_BENCHMARK_GUIDE.md) | Model parity, count accuracy, transport and soak checks |
| [Deployment](deploy/README.md) | Services, credentials, monitoring and maintenance |
| [Future system-owner changes](docs/SYSTEM_OWNER_CHANGE_PLAN.md) | Proposed controller/backend work requiring the owners' approval |

## Setup

Use a Python version supported by your CUDA/PyTorch build and a virtual environment. The current development lock was verified with Python 3.13.5:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux, activate with `source .venv/bin/activate`. Install CUDA-capable PyTorch and torchvision appropriate for the target GPU first, then install the application dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Live Stream Simulation & Dual-Camera Workflow

To simulate IP cameras using local recordings with real-time network streaming and playback controls:

### 1. Start the Network Camera Streamer
Streams video continuously over UDP (MPEG-TS) and automatically populates `sources.json`:

```powershell
.\.venv\Scripts\python.exe stream_cameras.py
```

#### Streamer Keyboard Controls:
| Key | Action | Description |
| :---: | :--- | :--- |
| **`[SPACE]`** | **Pause / Play** | Freezes video playback while keeping network stream alive |
| **`[→]` / `[←]`** | **Forward / Rewind 5s** | Skip +5s or -5s across all camera feeds |
| **`[↑]` / `[↓]`** | **Fast Forward / Rewind 15s** | Skip +15s or -15s |
| **`[R]`** | **Restart Video** | Reset playback to `0:00` |
| **`[Q]`** / Esc | **Quit** | Stop all video streams |

*(To stream all 8 cameras for a 4-way intersection, run `stream_cameras.py --all-8`).*

---

### 2. Run the Multi-Camera Vision Algorithm
Processes streams, tracks vehicles, manages Virtual Carts, and outputs controller payloads:

```powershell
.\.venv\Scripts\python.exe run_multi_camera.py `
  --cameras north north_upstream `
  --sources-file sources.json `
  --model models/yolo26s_thai_traffic.pt `
  --output-mode controller `
  --controller-config config/controller_1approach.json `
  --record-payloads live_counts.jsonl `
  --display
```

---

### 3. Inspect Live Outputs & Virtual Cart Buffer

#### Watch Clean Controller Payloads:
Filters out diagnostic logs and displays the exact wire payload sent to the signal controller:
```powershell
.\.venv\Scripts\python.exe watch_counts.py
```
```json
{
  "intersectionId": "INT-001",
  "cameraId": "CAM-01",
  "timestamp": "2026-11-03T12:35:00.000Z",
  "lanes": [
    { "laneId": "N1", "direction": "N", "count": 3 },
    { "laneId": "N2", "direction": "N", "count": 2 },
    { "laneId": "N3", "direction": "N", "count": 1 }
  ],
  "outgoingfull": { "N": false, "E": false, "S": false, "W": false }
}
```

#### Inspect Virtual Cart Buffer & Inflow per Lane:
Monitors real visible cars, virtual cart tokens in flight, inflow rates, and queue freeze state:
```powershell
.\.venv\Scripts\python.exe watch_buffer.py
```

#### Inspect Detailed In-Memory Tokens & Remaining TTL:
Inspects individual vehicle IDs and remaining lifespan in memory (`Zone B: 100m -> 55m`) to verify tokens do not exceed per-lane TTL:
```powershell
.\.venv\Scripts\python.exe watch_buffer.py --memory
```

---

## Calibration

Launch the interactive polygon and counting gate calibration GUI:

```powershell
# Calibrate North Stop-line (Cam 1)
.\.venv\Scripts\python.exe tools/segmentor.py --direction north

# Calibrate North Upstream (Cam 2)
.\.venv\Scripts\python.exe tools/segmentor.py --direction north_upstream
```

- **`[1]`**: Draw lane polygons (Click vertices, press `[C]` to complete).
- **`[2]`**: Draw directed gates (`stopline` or `ingress`).
- **`[3]`**: Select / inspect / delete (`[D]`) existing shapes.
- **`[S]`**: Save directly to config JSON.

---

## Developer Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tools.validate_calibration
```
