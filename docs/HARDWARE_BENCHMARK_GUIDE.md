# Multi-Camera Hardware Benchmark and Sizing Guide

Benchmarking suite to profile throughput, hardware resource usage, and per-stage latency across 1 to 8+ parallel camera feeds on edge systems.

## Quick Start

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Quick Sanity Test

Tests 1 and 2 streams across both threaded and batched pipelines:

```powershell
python benchmark_hardware.py --streams 1 2 --duration 3 --models yolov8n.pt --mode both
```

### Standard Multi-Camera Benchmark

Evaluates scaling from 1 to 8 streams with frame skipping, terminal reporting, and PNG summary charts:

```powershell
python benchmark_hardware.py `
    --streams 1 2 4 8 `
    --models yolov8n.pt yolov8s.pt `
    --mode batched `
    --frame-skips 0 1 `
    --target-fps 15.0 `
    --duration 8 `
    --save-plots
```

### Live Visual HUD Benchmark

Includes real-time OpenCV window rendering to measure UI drawing overhead:

```powershell
python benchmark_hardware.py `
    --streams 4 8 `
    --models yolov8n.pt `
    --mode batched `
    --display `
    --duration 10
```

### Uncapped Maximum Throughput Stress Test

Removes RTSP ingestion pacing to measure peak raw system capacity:

```powershell
python benchmark_hardware.py `
    --streams 4 8 `
    --models yolov8n.pt `
    --mode batched `
    --unpaced `
    --duration 10
```

## Pipeline Architectures

### Threaded Mode (`--mode threaded`)
Each camera stream runs inside a dedicated worker thread with an RTSP stream simulator and SORT tracker. Model forward passes are synchronized via a thread mutex to guarantee CUDA memory safety. This mode reflects independent process architectures but introduces CPU lock contention when scaling past 4 streams on a single GPU.

### Batched Mode (`--mode batched`)
Frames from all active camera streams are collected and combined into a single batch tensor `[B, 3, H, W]` before running inference in one forward pass. Detections are fanned out to independent SORT trackers and lane polygon evaluators. This maximizes GPU compute saturation and achieves higher throughput.

## CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--streams` | `1 2 4 8` | Stream counts to test sequentially. |
| `--models` | `yolov8n.pt yolov8s.pt` | Model checkpoints or engine paths to test. |
| `--mode` | `threaded` | Pipeline architecture: `threaded`, `batched`, or `both`. |
| `--frame-skips` | `0` | Skip intervals: `0` (every frame), `1` (every 2nd frame), `2` (every 3rd frame). |
| `--duration` | `10.0` | Test duration in seconds per configuration. |
| `--target-cams` | `8` | Target camera count for feasibility sizing. |
| `--target-fps` | `15.0` | Target frame rate per camera feed. |
| `--display` | `False` | Opens multi-camera preview window to measure rendering overhead. |
| `--unpaced` | `False` | Disables stream pacing for uncapped stress testing. |
| `--imgsz` | `640` | Input image size for inference. |
| `--videos` | Default 4 clips | Video file paths or RTSP stream URLs. |
| `--out-dir` | `benchmark/hardware-results` | Directory where benchmark artifacts are saved. |
| `--keep-latest` | `3` | Number of historical benchmark result sets to retain. |
| `--save-plots` | `False` | Generates 4-panel analysis charts in PNG format. |

## Output Metrics Reference

### Throughput and Loss

| Metric | Meaning |
| :--- | :--- |
| `FPS/Cam` | Average frames processed per second for each camera feed. |
| `Total FPS` | Combined system throughput across all camera feeds. |
| `Drops (%)` | Percentage of frames dropped by the stream simulator due to backpressure. Target is under 5%. |

### Stage Latencies

| Stage | Measured Operation |
| :--- | :--- |
| `Decode` | Frame ingestion and color decoding. |
| `Preprocess` | Tensor formatting and normalization. |
| `Inference` | GPU model forward pass. |
| `Tracking` | SORT Kalman filter prediction and Hungarian association. |
| `Analytics` | Point-in-polygon lane boundary containment tests. |
| `Render` | Bounding box and preview HUD drawing. |
| `E2E P50` | Median end-to-end frame latency from capture to output. |

### Hardware Profiling

| Metric | Monitored Subsystem |
| :--- | :--- |
| `GPU Util %` | Percentage of active GPU compute cores. |
| `GPU Memory Bus %` | PCIe and memory controller saturation. |
| `Peak VRAM (MB)` | Maximum dedicated GPU memory allocated during the test. |
| `CPU System %` | Overall host CPU load. |
| `Per-Core CPU %` | Individual core load distribution to spot single-thread Python bottlenecks. |

## Feasibility Sizing Evaluation

The benchmark evaluates whether the hardware meets production criteria for target deployments:

1. Target throughput: Checks if the system sustains `target_cameras * target_fps` with under 5% frame drops.
2. Safe headroom: Confirms GPU and CPU load stay below 85% to absorb traffic surges.
3. Bottleneck diagnosis: Isolates whether limits stem from GPU compute cores, VRAM capacity, host CPU decoding, or PCIe bus bandwidth.

## Generated Artifacts

Each benchmark run writes timestamped files to `benchmark/hardware-results/`:

- `BENCHMARK_REPORT_<timestamp>.md`: Formatted summary table, latency breakdown, core distribution, and sizing verdict.
- `benchmark_summary_<timestamp>.csv`: Tabular metric row per tested configuration for spreadsheet export.
- `benchmark_results_<timestamp>.json`: Raw time-series telemetry and system metadata.
- `benchmark_analysis_<timestamp>.png`: Generated with `--save-plots`, showing throughput scaling, GPU/CPU loads, memory demand, and stage latencies.
