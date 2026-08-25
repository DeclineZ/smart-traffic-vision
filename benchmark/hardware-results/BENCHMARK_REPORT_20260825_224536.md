# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-25 22:45:36

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **13.2 FPS/cam** (Total: **105.7 FPS**)
- **GPU Utilization & Headroom:** **14.3%** load (Headroom: **85.7%**)
- **VRAM Footprint & Free Space:** **2495 MB** / 8151 MB (Headroom: **5.52 GB**)
- **CPU Utilization & Headroom:** **24.4%** load (Headroom: **75.6%**)
- **Frame Drop Rate:** **6.14%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 4 | batched | 0 | YES | **14.0** | **56.1** | 8.7% | 2403 | 15.8% | 2336 | 2.6% |
| `yolov8n.pt` | 8 | batched | 0 | YES | **13.2** | **105.7** | 14.3% | 2495 | 24.4% | 3051 | 6.1% |
| `yolov8s.pt` | 4 | batched | 0 | YES | **14.1** | **56.4** | 15.5% | 2502 | 17.7% | 2403 | 2.1% |
| `yolov8s.pt` | 8 | batched | 0 | YES | **11.4** | **90.9** | 25.3% | 2673 | 28.4% | 3123 | 18.9% |