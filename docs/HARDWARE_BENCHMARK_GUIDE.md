# Hardware Benchmark Guide

`benchmark_hardware.py` profiles CPU, RAM, GPU, VRAM, and stage latencies across multiple video streams. It tests real-time throughput scaling (1 to 8+ cameras) and evaluates whether a machine can support production intersection workloads.

## Execution Modes

### Quick Verification Test
Runs a fast check across 1 and 2 streams to confirm CUDA, models, and ingestion pipelines work:

```bash
python benchmark_hardware.py --streams 1 2 --duration 4 --models yolov8n.pt
```

### Full Benchmark Run
Runs scaling tests across multiple streams, models, and frame skip settings:

```bash
python benchmark_hardware.py --streams 1 2 4 8 --models yolov8n.pt yolov8s.pt --mode both --frame-skips 0 1 --target-fps 25.0 --duration 8 --display --save-plots
```

## CLI Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--streams` | `int ...` | `1 2 4 8` | Stream counts to test sequentially. |
| `--models` | `str ...` | `yolov8n.pt yolov8s.pt` | Model checkpoints or engine files. |
| `--mode` | `str` | `threaded` | Pipeline architecture: `threaded`, `batched`, or `both`. |
| `--frame-skips` | `int ...` | `0` | Inference cadence: `0` (every frame), `1` (every 2nd frame), `2` (every 3rd frame). |
| `--display` | `flag` | `False` | Opens OpenCV window to measure visual rendering overhead. |
| `--duration` | `float` | `10.0` | Test duration in seconds per configuration. |
| `--target-cams` | `int` | `8` | Target stream count for sizing evaluation. |
| `--target-fps` | `float` | `15.0` | Target frame rate per stream (e.g. 15.0, 25.0). |
| `--unpaced` | `flag` | `False` | Disables frame pacing for uncapped stress testing. |
| `--imgsz` | `int` | `640` | Input image size (`640`, `480`, etc.). |
| `--videos` | `str ...` | Default 4 AVI videos | Video file paths or RTSP URLs. |
| `--out-dir` | `str` | `benchmark/hardware-results` | Output directory for results. |
| `--keep-latest` | `int` | `3` | Number of historical runs to retain. |
| `--save-plots` | `flag` | `False` | Generates summary PNG charts. |

## Pipeline Modes

- **Threaded (`--mode threaded`)**: Each stream runs in its own worker thread with sequential model calls. Suffers from GPU lock contention at higher stream counts (4–8 cameras).
- **Batched (`--mode batched`)**: Ingests frames across all active streams into a single tensor `[B, 3, H, W]` and runs one forward pass. Significantly higher throughput on GPUs.

## Testing Live RTSP Streams

To benchmark physical network cameras:

```bash
python benchmark_hardware.py \
  --videos \
    rtsp://admin:pass@192.168.1.101:554/stream1 \
    rtsp://admin:pass@192.168.1.102:554/stream1 \
  --mode batched \
  --frame-skips 0 1 \
  --target-fps 25.0
```

## Interpreting Output Metrics

### Performance & Latency
- **FPS/Cam**: Average frames processed per second for each camera feed.
- **Total FPS**: Combined throughput across all streams ($\text{FPS/Cam} \times N$).
- **Drops (%)**: Percentage of frames dropped due to pipeline backpressure. Safe production threshold is $< 5\%$.
- **Latency (ms)**:
  - `Decode`: Video demuxing and frame capture.
  - `Inference`: Model forward pass.
  - `Tracking`: Tracker update and state prediction.
  - `E2E P50`: Median end-to-end frame latency. At 25 FPS, the target budget is $40\text{ ms}$; at 15 FPS, $66.7\text{ ms}$.

### Feasibility Verdict
- `[A+] / [A] GOOD / EXCELLENT`: Target FPS sustained with safe headroom and $< 5\%$ drops.
- `[B-] BORDERLINE`: Meets target FPS but operates near compute or memory limits.
- `[C / FAIL] INSUFFICIENT`: Cannot sustain target FPS or experiences high frame drop rates.

## Output Files

Each run writes to `benchmark/hardware-results/`:
- `BENCHMARK_REPORT_<timestamp>.md`: Markdown summary table, latency breakdown, and sizing verdict.
- `benchmark_summary_<timestamp>.csv`: Summary metrics row per test configuration.
- `benchmark_results_<timestamp>.json`: Raw run data, per-core CPU usage, and time-series telemetry.
- `benchmark_analysis_<timestamp>.png`: Generated when `--save-plots` is passed (throughput scaling, resource utilization, memory, and stage latencies).
