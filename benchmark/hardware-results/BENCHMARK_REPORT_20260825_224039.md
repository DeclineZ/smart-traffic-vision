# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-25 22:40:39

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **9.6 FPS/cam** (Total: **77.0 FPS**)
- **GPU Utilization & Headroom:** **17.2%** load (Headroom: **82.8%**)
- **VRAM Footprint & Free Space:** **2539 MB** / 8151 MB (Headroom: **5.48 GB**)
- **CPU Utilization & Headroom:** **38.0%** load (Headroom: **62.0%**)
- **Frame Drop Rate:** **28.46%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 4 | batched | 0 | YES | **13.3** | **53.1** | 13.2% | 2447 | 25.6% | 2360 | 5.3% |
| `yolov8n.pt` | 8 | batched | 0 | YES | **9.6** | **77.0** | 17.2% | 2539 | 38.0% | 3090 | 28.5% |