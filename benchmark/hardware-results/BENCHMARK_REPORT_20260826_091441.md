# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-26 09:14:41

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **6.5 FPS/cam** (Total: **52.3 FPS**)
- **GPU Utilization & Headroom:** **15.9%** load (Headroom: **84.1%**)
- **VRAM Footprint & Free Space:** **1919 MB** / 8151 MB (Headroom: **6.09 GB**)
- **CPU Utilization & Headroom:** **34.9%** load (Headroom: **65.1%**)
- **Frame Drop Rate:** **52.32%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 4 | threaded | 0 | YES | **14.3** | **57.1** | 15.9% | 1965 | 21.4% | 2311 | 0.2% |
| `yolov8n.pt` | 8 | threaded | 0 | YES | **6.5** | **52.3** | 15.9% | 1919 | 34.9% | 3067 | 52.3% |
| `yolov8s.pt` | 4 | threaded | 0 | YES | **14.3** | **57.2** | 22.4% | 1998 | 19.8% | 1402 | 0.0% |
| `yolov8s.pt` | 8 | threaded | 0 | YES | **7.0** | **56.0** | 22.7% | 2030 | 30.1% | 2183 | 49.6% |