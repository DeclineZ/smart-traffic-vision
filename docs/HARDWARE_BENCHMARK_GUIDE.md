# Multi-Camera Hardware Benchmark & Sizing Guide

This guide explains how to use the automated hardware benchmarking suite (`benchmark_hardware.py`) to profile CPU, RAM, GPU, VRAM, and stage-by-stage latencies on any machine, test real-time scaling across 1 to 8+ camera feeds, and evaluate hardware feasibility for smart traffic vision deployments.

---

## Table of Contents
- [1. Overview & Architecture](#1-overview--architecture)
- [2. Benchmark Execution Modes](#2-benchmark-execution-modes)
- [3. CLI Flags & Configuration Options](#3-cli-flags--configuration-options)
- [4. How the Benchmark Works Under the Hood](#4-how-the-benchmark-works-under-the-hood)
- [5. Benchmarking Real RTSP Cameras](#5-benchmarking-real-rtsp-cameras)
- [6. How to Read & Interpret Benchmark Results](#6-how-to-read--interpret-benchmark-results)
- [7. 8-Camera Feasibility & Hardware Procurement Specs](#7-8-camera-feasibility--hardware-procurement-specs)
- [8. Generated Output Artifacts](#8-generated-output-artifacts)

---

## 1. Overview & Architecture

When deploying computer vision models (like YOLOv8 + SORT) across multiple 1080p camera feeds at traffic intersections, real-time performance depends on multiple hardware subsystems:

```mermaid
flowchart TD
    subgraph Ingestion [1. Camera Ingestion & RTSP Simulation]
        C1[Camera 1 RTSP / Video] --> S1[RTSP Simulator: Paced Buffer]
        C2[Camera 2 RTSP / Video] --> S2[RTSP Simulator: Paced Buffer]
        CN[Camera N RTSP / Video] --> SN[RTSP Simulator: Paced Buffer]
    end

    subgraph Pipeline [2. Vision Pipeline Architecture]
        subgraph Mode A [Threaded Workers]
            S1 --> W1[Worker 1: YOLO + SORT + Analytics]
            S2 --> W2[Worker 2: YOLO + SORT + Analytics]
            SN --> WN[Worker N: YOLO + SORT + Analytics]
        end
        subgraph Mode B [Batched Pipeline - Recommended]
            S1 & S2 & SN --> B1[Batch Collector: Bx3xHxW Tensor]
            B1 --> B2[Single GPU Forward Pass]
            B2 --> B3[Fan-out: Parallel SORT + Analytics]
        end
    end

    subgraph Telemetry [3. Zero-Overhead Hardware Profiler]
        NVML[Direct NVML C-Bindings] --> GPU[GPU % / Bus % / VRAM / Temp / Power]
        PSUTIL[psutil System Monitor] --> CPU[Process CPU / Per-Core / System RAM]
    end

    subgraph Reporting [4. Sizing & Reporting Engine]
        O1[8-Camera Feasibility Grade]
        O2[Stage Latency Matrix: Decode/Inference/SORT/E2E]
        O3[Exports: Markdown, CSV, JSON, PNG Plots]
    end

    Mode A --> Reporting
    Mode B --> Reporting
    Telemetry --> Reporting
```

### Why this benchmark is accurate:
- **Zero-Overhead GPU Profiling**: Uses direct C-level NVML bindings (`pynvml`) sampled every 50ms without launching slow external subprocesses.
- **RTSP Network Simulation**: Paces video frames at real camera frame rates (e.g. 15 or 25 FPS) with bounded ring buffers (`maxsize=1`). If inference lags behind, frames are dropped, exposing real-world backpressure.
- **Nanosecond Stage Breakdown**: Measures exact times spent in Video Decode, Preprocessing, GPU Inference, SORT Tracking, and Polygon Analytics.
- **Frame Skipping Support**: Evaluates running YOLO every $N$ frames (`skip=1`, `skip=2`) with Kalman filter motion prediction on skipped frames.

---

## 2. Benchmark Execution Modes

Ensure your virtual environment is active with dependencies installed before running:

### A. Quick Verification Test (~15 seconds)
A fast sanity check to verify that your GPU, CUDA environment, and RTSP ingestion pipelines are running properly:

```bash
python benchmark_hardware.py --streams 1 2 --duration 4 --models yolov8n.pt
```

---

### B. Full Production Hardware Sizing Benchmark (The Works)
The comprehensive benchmark that tests **everything** to determine whether your machine can run 8 cameras smoothly at 25 FPS:
- Tests scaling across **1, 2, 4, and 8 camera feeds**.
- Compares **Batched vs. Threaded** pipeline architectures (`--mode both`).
- Evaluates **Frame Skipping** optimization (`--frame-skips 0 1`).
- Opens the **Live Multi-Camera Visual HUD** (`--display`) to measure real UI drawing overhead.
- Targets **25.0 FPS per camera** (`--target-fps 25.0`).
- Generates **4-panel performance visualization charts** (`--save-plots`).

```bash
python benchmark_hardware.py --streams 1 2 4 8 --models yolov8n.pt yolov8s.pt --mode both --frame-skips 0 1 --target-fps 25.0 --duration 8 --display --save-plots
```

---

## 3. CLI Flags & Configuration Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--streams` | `int ...` | `1 2 4 8` | List of parallel camera stream counts to test sequentially. |
| `--models` | `str ...` | `yolov8n.pt yolov8s.pt` | YOLO model checkpoints to benchmark. |
| `--mode` | `str` | `threaded` | Vision pipeline architecture: `threaded` (distributed workers), `batched` (centralized tensor batching), or `both`. |
| `--frame-skips`| `int ...` | `0` | Frame skip intervals: `0` (every frame), `1` (every 2nd frame), `2` (every 3rd frame). |
| `--display` | `Flag` | `False` | Opens live multi-camera OpenCV visual HUD window with bounding boxes, labels, and lane polygons (measures UI rendering CPU/GPU overhead). |
| `--duration` | `float` | `10.0` | Test duration in seconds per individual benchmark run. |
| `--target-cams`| `int` | `8` | Target camera count used for procurement calculations and verdict. |
| `--target-fps` | `float` | `15.0` | Target frame rate per camera feed (e.g. 15.0, 20.0, 25.0 FPS). |
| `--unpaced` | `Flag` | `False` | Disables RTSP stream pacing to run in uncapped max-throughput stress mode. |
| `--imgsz` | `int` | `640` | Input inference image resolution (`640`, `480`, `1280`). |
| `--videos` | `str ...` | Default 4 AVI videos | Custom list of video files or RTSP stream URLs. |
| `--out-dir` | `str` | `benchmark/hardware-results` | Directory where JSON, CSV, Markdown, and PNG reports are stored. |
| `--keep-latest`| `int` | `3` | Maximum number of historical benchmark result runs to retain in results folder (automatically prunes older saves). |
| `--save-plots` | `Flag` | `False` | Generates 4-panel performance visualization charts (PNG). |

---

## 4. How the Benchmark Works Under the Hood

### 1. Baseline Calibration
Before running tests, the tool samples the idle machine for 1.5 seconds to establish baseline CPU, RAM, VRAM, and power draw. This ensures metrics isolate the exact cost of the vision pipeline.

### 2. Threaded Mode vs. Batched Mode
- **Threaded Mode (`--mode threaded`)**: Each camera runs in its own thread, calling `model(frame)` sequentially. While intuitive, Python's GIL and serial GPU kernel dispatches create contention at high camera counts (4–8 cameras).
- **Batched Mode (`--mode batched`)**: Frames from all active cameras are grouped into a single batch tensor `[B, 3, H, W]` and executed in **one single forward pass on the GPU Tensor Cores**. Detections are then fanned out to per-camera SORT trackers. This typically provides **3x–4x higher throughput** on modern GPUs.

### 3. Stage-by-Stage Latency Profiling
For every single processed frame across all cameras, exact timestamps are logged:
- **`Decode`**: Camera ingestion, packet decoding, and queue retrieval.
- **`Preprocess`**: Letterboxing, tensor formatting, and CUDA memory copy.
- **`Inference`**: GPU forward pass (measured with explicit CUDA synchronization).
- **`SORT Tracking`**: Kalman filter state prediction and Hungarian bipartite matching.
- **`Polygon Analytics`**: Shapely point-in-polygon checks and queue speed calculations.
- **`End-to-End Latency`**: Time elapsed from camera capture timestamp to output emission.

---

## 5. Benchmarking Real RTSP Cameras

To benchmark physical IP cameras over your local network:

```bash
python benchmark_hardware.py \
  --videos \
    rtsp://admin:pass@192.168.1.101:554/h264Preview_01_main \
    rtsp://admin:pass@192.168.1.102:554/h264Preview_01_main \
    rtsp://admin:pass@192.168.1.103:554/h264Preview_01_main \
    rtsp://admin:pass@192.168.1.104:554/h264Preview_01_main \
  --mode batched \
  --frame-skips 0 1 \
  --target-fps 20.0 \
  --save-plots
```

The benchmark will measure network ingestion latency, frame drops caused by network congestion or processing bottlenecks, and overall throughput.

---

## 6. How to Read & Interpret Benchmark Results

When a benchmark run completes, you will see three summary tables in your terminal and in the generated Markdown report:

### 1. Scalability & Resource Demand Table
```
Model      | Streams | Mode     | Skip | FPS/Cam   | Total FPS | GPU %  | VRAM(MB) | CPU %  | Drops
yolov8n.pt | 4       | batched  | 0    | 13.8      | 55.1      | 12.8   | 2341     | 21.5   | 2.4%
yolov8n.pt | 8       | batched  | 0    | 12.5      | 99.8      | 14.1   | 2422     | 25.6   | 8.2%
yolov8n.pt | 8       | batched  | 1    | 12.6      | 100.5     | 6.6    | 2459     | 21.0   | 7.1%
```
- **FPS/Cam**: Average frames processed per second for each individual camera feed.
- **Total FPS**: Aggregate throughput across all streams ($FPS_{cam} \times N$).
- **Drops (%)**: Percentage of camera frames dropped due to processing backpressure. Safe production threshold is $< 5.0\%$.
- **GPU % / CPU %**: Average compute utilization across the run.
- **VRAM (MB)**: Peak dedicated GPU memory consumed.

### 2. Stage Latency Breakdown Table
```
Model / Streams           | Decode (ms)  | Inference (ms)  | SORT (ms)  | E2E P50 (ms)
yolov8n.pt (8 cams)       | 1.55         | 4.50            | 1.70       | 80.51
```
- **E2E P50 (ms)**: Median end-to-end frame latency. For a 15 FPS stream, your latency budget is $\frac{1000}{15} = 66.7\text{ ms}$.

### 3. Sizing Verdict & Bottleneck Diagnosis
The tool provides an automated verdict grade:
- **`[A+] EXCELLENT`**: Production-ready with $>30\%$ compute and memory headroom, drops $<1\%$.
- **`[A] GOOD`**: Production-ready, meets target FPS with safe margins.
- **`[B-] BORDERLINE`**: Operates near hardware limits; frame drops or thermal throttling likely under peak traffic.
- **`[C / FAIL] INSUFFICIENT`**: Hardware cannot sustain target FPS without excessive frame drops or lag.

---

## 7. 8-Camera Feasibility Sizing Rules

### Sizing Rules of Thumb:
1. **Throughput Target**: 8 Cameras @ 15 FPS = **120 Aggregate FPS** (or 8 @ 20 FPS = **160 Aggregate FPS**).
2. **VRAM Formula**:
   $$\text{Total VRAM} = \text{Base Model (800 MB)} + (8 \times 200\text{ MB RTSP Buffers}) + 1.5\text{ GB OS Headroom} \approx \mathbf{3.9\text{ GB}}$$
   *(An 8 GB GPU provides abundant headroom).*
3. **CPU Cores Formula**:
   $$\text{CPU Threads} = (8 \text{ streams} \times 1.0\text{ thread/decode}) + 4\text{ threads (MQTT + DB + Controller)} \approx \mathbf{12+\text{ Logical Threads}}$$

---

## 8. Generated Output Artifacts

All benchmark runs automatically export timestamped reports to `benchmark/hardware-results/`:

1. **`BENCHMARK_REPORT_<timestamp>.md`**: Executive markdown report with baseline hardware specs, scaling matrices, and feasibility verdicts.
2. **`benchmark_summary_<timestamp>.csv`**: Flat CSV data file ready for importing into Excel, Google Sheets, or Pandas.
3. **`benchmark_results_<timestamp>.json`**: Structured JSON containing full high-frequency time-series telemetry (CPU, GPU %, VRAM MB, power, temp over time).
4. **`benchmark_analysis_<timestamp>.png`**: 4-panel Matplotlib visualization chart:
   - Panel 1: Throughput scaling (Streams vs. FPS/Camera)
   - Panel 2: Resource demand (GPU % and CPU % vs. Streams)
   - Panel 3: Memory footprint (VRAM MB and RAM MB vs. Streams)
   - Panel 4: Stage Latency Breakdown bar chart.
