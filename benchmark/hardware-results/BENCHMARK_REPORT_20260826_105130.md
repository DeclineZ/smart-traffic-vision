# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-26 10:51:30

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **5.6 FPS/cam** (Total: **45.0 FPS**)
- **GPU Utilization & Headroom:** **18.6%** load (Headroom: **81.4%**)
- **VRAM Footprint & Free Space:** **2882 MB** / 8151 MB (Headroom: **5.15 GB**)
- **CPU Utilization & Headroom:** **42.1%** load (Headroom: **57.9%**)
- **Frame Drop Rate:** **56.38%**
- **Max Safe Stream Capacity:** **16 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 8 | threaded | 0 | NO | **5.6** | **45.0** | 18.6% | 2882 | 42.1% | 3040 | 56.4% |
| `yolov8n.pt` | 8 | batched | 0 | NO | **12.2** | **97.4** | 18.8% | 2757 | 34.1% | 3048 | 10.1% |

## 4. Per-Core CPU Load Distribution (yolov8n.pt - 8 Streams, BATCHED)

| Core | Load % | Core | Load % | Core | Load % | Core | Load % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core 0** | 64.8% | **Core 1** | 26.8% | **Core 2** | 44.3% | **Core 3** | 47.9% |
| **Core 4** | 20.1% | **Core 5** | 27.9% | **Core 6** | 19.6% | **Core 7** | 24.2% |
| **Core 8** | 16.7% | **Core 9** | 33.7% | **Core 10** | 41.3% | **Core 11** | 50.1% |
| **Core 12** | 23.4% | **Core 13** | 39.2% | **Core 14** | 18.6% | **Core 15** | 28.9% |

- **Peak Single-Core Load:** `64.8%`
- **Minimum Core Load:** `16.7%`
- **Core Load Imbalance (Std Dev):** `13.3%` (Exposes single-thread Python bottleneck)
