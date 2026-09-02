# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-09-02 10:02:44

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **13.0 FPS/cam** (Total: **26.0 FPS**)
- **GPU Utilization & Headroom:** **15.1%** load (Headroom: **84.9%**)
- **VRAM Footprint & Free Space:** **1911 MB** / 8151 MB (Headroom: **6.09 GB**)
- **CPU Utilization & Headroom:** **25.1%** load (Headroom: **74.9%**)
- **Frame Drop Rate:** **0.00%**
- **Max Safe Stream Capacity:** **6 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 1 | threaded | 0 | NO | **12.0** | **12.0** | 10.6% | 1931 | 18.3% | 1795 | 3.7% |
| `yolov8n.pt` | 2 | threaded | 0 | NO | **13.0** | **26.0** | 15.1% | 1911 | 25.1% | 1954 | 0.0% |
| `yolov8n.pt` | 1 | batched | 0 | NO | **14.3** | **14.3** | 8.3% | 1888 | 18.5% | 1817 | 0.0% |
| `yolov8n.pt` | 2 | batched | 0 | NO | **13.5** | **26.9** | 9.5% | 1892 | 19.9% | 1973 | 1.6% |

## 4. Per-Core CPU Load Distribution (yolov8n.pt - 2 Streams, BATCHED)

| Core | Load % | Core | Load % | Core | Load % | Core | Load % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core 0** | 44.4% | **Core 1** | 13.9% | **Core 2** | 28.9% | **Core 3** | 27.0% |
| **Core 4** | 10.4% | **Core 5** | 10.6% | **Core 6** | 8.9% | **Core 7** | 5.4% |
| **Core 8** | 21.1% | **Core 9** | 12.0% | **Core 10** | 40.6% | **Core 11** | 21.8% |
| **Core 12** | 27.4% | **Core 13** | 10.2% | **Core 14** | 9.1% | **Core 15** | 13.5% |

- **Peak Single-Core Load:** `44.4%`
- **Minimum Core Load:** `5.4%`
- **Core Load Imbalance (Std Dev):** `11.4%` (Exposes single-thread Python bottleneck)
