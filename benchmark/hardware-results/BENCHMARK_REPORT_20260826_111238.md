# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-26 11:12:38

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **4.3 FPS/cam** (Total: **34.8 FPS**)
- **GPU Utilization & Headroom:** **16.8%** load (Headroom: **83.2%**)
- **VRAM Footprint & Free Space:** **2727 MB** / 8151 MB (Headroom: **5.30 GB**)
- **CPU Utilization & Headroom:** **47.0%** load (Headroom: **53.0%**)
- **Frame Drop Rate:** **80.17%**
- **Max Safe Stream Capacity:** **14 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 4 | threaded | 0 | NO | **15.0** | **59.9** | 21.2% | 2711 | 35.0% | 2343 | 36.8% |
| `yolov8n.pt` | 8 | threaded | 0 | NO | **4.3** | **34.8** | 16.8% | 2727 | 47.0% | 3070 | 80.2% |
| `yolov8n.pt` | 4 | batched | 0 | NO | **23.4** | **93.5** | 19.1% | 2781 | 29.4% | 2360 | 2.3% |
| `yolov8n.pt` | 8 | batched | 0 | NO | **11.3** | **90.2** | 17.9% | 2868 | 41.9% | 3091 | 50.7% |

## 4. Per-Core CPU Load Distribution (yolov8n.pt - 8 Streams, BATCHED)

| Core | Load % | Core | Load % | Core | Load % | Core | Load % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core 0** | 69.6% | **Core 1** | 28.3% | **Core 2** | 56.2% | **Core 3** | 48.0% |
| **Core 4** | 30.0% | **Core 5** | 32.1% | **Core 6** | 31.5% | **Core 7** | 25.8% |
| **Core 8** | 40.8% | **Core 9** | 32.2% | **Core 10** | 62.2% | **Core 11** | 46.3% |
| **Core 12** | 48.5% | **Core 13** | 37.1% | **Core 14** | 33.0% | **Core 15** | 31.5% |

- **Peak Single-Core Load:** `69.6%`
- **Minimum Core Load:** `25.8%`
- **Core Load Imbalance (Std Dev):** `12.7%` (Exposes single-thread Python bottleneck)
