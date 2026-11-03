# Smart Traffic Vision

Multi-camera vehicle detection and tracking for adaptive traffic control. The runner measures current lane occupancy, queued/moving/unknown vehicles, directed gate crossings, and downstream exit congestion (`outgoingfull`). Camera health and observation times travel with the measurements.

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

### Build TensorRT Engine

The runner defaults to `models/yolo26s_thai_traffic_manual_v1_640.engine`. Build it on the target GPU with dynamic batching (1..8) and FP32 precision:

```powershell
.\.venv\Scripts\python.exe export_trt.py --model models/yolo26s_thai_traffic_manual_v1_640.pt --batch 8 --no-half
```

To verify detection parity between PyTorch weights and the exported TensorRT engine:

```powershell
.\.venv\Scripts\python.exe -m tools.trt_parity --model models/yolo26s_thai_traffic_manual_v1_640.pt --engine models/yolo26s_thai_traffic_manual_v1_640.engine --frames 40
```

---


This launches 4 dedicated windows automatically:
1. **Camera Streamer (UDP):** Streams dual video feeds on `udp://127.0.0.1:8554` and `8555`.
2. **Multi-Camera AI Runner:** Runs TensorRT inference, ByteTrack, spillback engine, and displays detection overlay.
3. **Wire Counts Watcher (`watch_counts.py`):** Real-time JSON controller payloads and `outgoingfull` states.
4. **Buffer & Memory Watcher (`watch_buffer.py --memory`):** Real-time token tracking, inflow rates, and TTL memory.

---

## Manual Multi-Camera Workflow (Step-by-Step)

If you prefer to start each component manually in separate terminal tabs:

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
  --model models/yolo26s_thai_traffic_manual_v1_640.engine `
  --output-mode controller `
  --controller-config config/controller_1approach.json `
  --record-payloads live_counts.jsonl `
  --display
```

*(If not using UDP streamer, replace `--sources-file sources.json` with `--videos ./videos/cam44_north.avi ./videos/cam43_south.avi`).*

---

### 3. Inspect Live Outputs & Monitors

#### A. Watch Clean Controller Payloads (`watch_counts.py`)
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

#### B. Inspect Virtual Cart Buffer & Inflow per Lane:
Monitors real visible cars, virtual cart tokens in flight, inflow rates, and queue freeze state:
```powershell
.\.venv\Scripts\python.exe watch_buffer.py
```

#### C. Inspect Detailed In-Memory Tokens & Remaining TTL:
Inspects individual vehicle IDs and remaining lifespan in memory (`Zone B: 100m -> 55m`) to verify tokens do not exceed per-lane TTL:
```powershell
.\.venv\Scripts\python.exe watch_buffer.py --memory
```

---

## Interactive Calibration & Outgoing Exit Roads

Launch the interactive polygon and counting gate calibration GUI:

```powershell
# Calibrate North Stop-line (Main Cam 1)
.\.venv\Scripts\python.exe tools/segmentor.py --direction north

# Calibrate North Upstream (Upstream Cam 2)
.\.venv\Scripts\python.exe tools/segmentor.py --direction north_upstream

# Or launch with UI camera picker:
python main.py calibrate
```

### Calibrator Controls & Modes:
| Key | Mode / Action | Description |
| :---: | :--- | :--- |
| **`[1]`** | **Lane Mode** | Click 3+ vertices to define a stop-line queue lane (`N1`, `N2`, `S1`, etc.). Press `[C]` to finish. |
| **`[2]`** | **Gate Mode** | Click start and end points for virtual counting line. Press `[Tab]` to cycle type (`stopline` / `ingress`). |
| **`[3]`** | **Select Mode** | Click any existing lane, gate, or outgoing road to inspect or delete (`[D]`). |
| **`[4]`** | **Outgoing Mode** | Click 3–4 corners for outgoing departure road (`OUT_N`, `OUT_NU`). Press `[C]` to finish. |
| **`[C]` / Enter** | **Complete** | Finish current polygon or line and enter identifier. |
| **`[F]`** | **Flip Normal** | Reverse gate counting arrow direction. |
| **`[Z]` / Backspace** | **Undo** | Undo last clicked vertex. |
| **`[R]`** | **Reset** | Clear in-progress drawing points. |
| **`[N]` / `[P]`** | **Seek Video** | Step forward / backward by $\pm 5\text{s}$ to find an unobstructed frame. |
| **`[S]`** | **Save** | Save geometry directly to config JSON with atomic backup. |
| **`[Q]` / Esc** | **Quit** | Close calibrator. |

### How `outgoingfull` Works:
- Outgoing exit roads are configured with `"role": "outgoing"` (e.g. `OUT_N` on Main, `OUT_NU` on Upstream).
- When vehicles are stopped/queued on the exit road for $\ge 3.0\text{ seconds}$, the engine sets `"outgoingfull": { "N": true, ... }`.
- Once exit flow resumes and stopped vehicles clear, it automatically resets back to `false`.
- **Dual Camera Coverage (Main + Upstream):** If both cameras have an outgoing polygon configured for direction `N`, they operate as a logical **OR**—detecting bottlenecks either right outside the intersection box or 100m downstream.

---

## Developer Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tools.validate_calibration
```
