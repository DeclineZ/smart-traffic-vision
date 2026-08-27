# Hardware Benchmark & Sizing Guide

Benchmarking suite for testing multi-camera traffic vision pipelines on edge hardware. Profiles CPU, GPU, VRAM, and per-stage latencies to determine real-world stream limits.

---

## 1. Quick Start Commands

Make sure your virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
cd smart-traffic-vision
```

### Quick Sanity Test (~15s)
Verifies that CUDA, TensorRT, and video ingestion work across 1-2 streams:

```powershell
python benchmark_hardware.py --streams 1 2 --duration 5 --models yolov8n.engine --mode batched --target-fps 25.0
```

### Full Benchmark
Evaluates 1, 2, 4, and 8 camera feeds with TensorRT models, frame skipping, ByteTrack, NVENC, and live HUD display:

```powershell
python benchmark_hardware.py `
    --streams 1 2 4 8 `
    --models yolov8n.engine yolov8s.engine `
    --mode batched `
    --frame-skips 1 2 `
    --tracker bytetrack `
    --conf 0.15 `
    --buffer-size 2 `
    --target-fps 25.0 `
    --nvenc `
    --display `
    --duration 8 `
    --save-plots
```

---

## 2. CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--streams` | `1 2 4 8` | Camera feed counts to test sequentially. |
| `--models` | `yolov8n.engine yolov8s.engine` | YOLO model checkpoints (`.pt`) or TensorRT engines (`.engine`). |
| `--mode` | `batched` | Pipeline mode: `batched` (centralized batching) or `threaded`. |
| `--frame-skips` | `0` | Skip intervals: `0` (every frame), `1` (every 2nd), `2` (every 3rd). |
| `--tracker` | `bytetrack` | Tracker: `bytetrack`, `botsort`, or `sort`. |
| `--buffer-size` | `2` | Camera queue size (2 absorbs inference bursts without latency buildup). |
| `--target-fps` | `25.0` | Target camera frame rate. |
| `--conf` | `0.15` | YOLO detection confidence threshold. |
| `--display` | `False` | Opens multi-camera preview window (runs asynchronously with 0ms loop overhead). |
| `--nvenc` | `False` | Uses hardware NVENC video encoding for preview streaming. |
| `--duration` | `10.0` | Test duration in seconds per configuration. |
| `--save-plots` | `False` | Generates 4-panel analysis charts (PNG). |
| `--out-dir` | `benchmark/hardware-results` | Directory where benchmark artifacts are saved. |

---

## 3. Pipeline Optimizations on this Branch

Key improvements implemented in `benchmark_hardware.py`:

- **Asynchronous Display Worker (`AsyncDisplayWorker`)**: Resizing, polyline drawing, grid stitching, and `cv.imshow()` run in a background daemon thread with a 1-slot buffer. This decoupled the UI completely, cutting loop display delay from 15.3 ms to 0.00 ms.
- **Jitter Ring Buffer (`--buffer-size 2`)**: Absorbs transient inference spikes on alternating skip frames. Prevents false frame drops and stabilizes E2E latency at ~47 ms.
- **TensorRT FP16 Batched Inference**: Combines all active camera feeds into a single `[B, 3, 640, 640]` forward pass on Tensor Cores, avoiding multi-threaded CUDA lock contention.
- **NVENC Hardware Acceleration**: Preview encoding uses NVIDIA silicon encoder (`h264_nvenc`) with < 1% chip load.

---

## 4. Benchmark Results & Sizing (RTX 5060 Laptop GPU)

Results from the full 16-run matrix test across 1–8 streams:

| Cameras | Target FPS | Total FPS | FPS / Cam | Drops | GPU % | CPU % | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 Cam** | 25.0 | 24.4 | 24.4 | 0.0% | 4% | 10% | Production Ready |
| **2 Cams** | 50.0 | 48.5 | 24.2 | 0.0% | 5% | 12% | Production Ready |
| **4 Cams** | 100.0 | 94.0 | 23.5 | 1.2% | 9% | 17% | Production Ready |
| **8 Cams** | 200.0 | 171.9 | 21.5 | 6.6% | 17% | 32% | Production Ready |

### Model Selection: Nano vs. Small
- **Throughput is identical**: Both `yolov8n.engine` and `yolov8s.engine` sustain ~171 Total FPS at 8 cameras because TensorRT forward pass is under 3 ms.
- **VRAM & Power**: `yolov8n.engine` uses only **2.5 GB VRAM** (vs. 6.8 GB for Small) and draws **18W** (vs. 24W).
- **Recommendation**: Standardize on `yolov8n.engine` for 4–8 camera edge deployments.

---

## 5. Output Files

All runs save timestamped artifacts to `benchmark/hardware-results/`:
- `BENCHMARK_REPORT_<timestamp>.md`: Summary report with hardware specs and feasibility metrics.
- `benchmark_summary_<timestamp>.csv`: Raw tabular data for spreadsheet analysis.
- `benchmark_results_<timestamp>.json`: Time-series hardware telemetry (CPU, GPU %, VRAM, power, thermals).
- `benchmark_analysis_<timestamp>.png`: 4-panel scaling and latency breakdown plots.
