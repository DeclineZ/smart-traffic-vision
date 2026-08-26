# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-26 10:56:23

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **0.4 FPS/cam** (Total: **3.4 FPS**)
- **GPU Utilization & Headroom:** **8.1%** load (Headroom: **91.9%**)
- **VRAM Footprint & Free Space:** **2622 MB** / 8151 MB (Headroom: **5.40 GB**)
- **CPU Utilization & Headroom:** **91.9%** load (Headroom: **8.1%**)
- **Frame Drop Rate:** **98.36%**
- **Max Safe Stream Capacity:** **7 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** Host CPU Multithreading / Video Decoding Backpressure (91.9% load)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 4 | threaded | 0 | NO | **1.4** | **5.4** | 10.3% | 2606 | 87.1% | 2408 | 97.9% |
| `yolov8n.pt` | 8 | threaded | 0 | NO | **0.4** | **3.4** | 8.1% | 2622 | 91.9% | 3156 | 98.4% |
| `yolov8n.pt` | 4 | batched | 0 | NO | **2.4** | **9.7** | 10.7% | 2675 | 88.1% | 2440 | 96.3% |
| `yolov8n.pt` | 8 | batched | 0 | NO | **0.7** | **5.2** | 7.0% | 2759 | 93.6% | 3178 | 97.4% |

## 4. Per-Core CPU Load Distribution (yolov8n.pt - 8 Streams, BATCHED)

| Core | Load % | Core | Load % | Core | Load % | Core | Load % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core 0** | 96.1% | **Core 1** | 90.8% | **Core 2** | 94.9% | **Core 3** | 94.3% |
| **Core 4** | 92.4% | **Core 5** | 92.9% | **Core 6** | 92.4% | **Core 7** | 92.0% |
| **Core 8** | 93.7% | **Core 9** | 91.5% | **Core 10** | 97.5% | **Core 11** | 96.7% |
| **Core 12** | 92.8% | **Core 13** | 92.6% | **Core 14** | 92.8% | **Core 15** | 94.0% |

- **Peak Single-Core Load:** `97.5%`
- **Minimum Core Load:** `90.8%`
- **Core Load Imbalance (Std Dev):** `1.8%` (Exposes single-thread Python bottleneck)
