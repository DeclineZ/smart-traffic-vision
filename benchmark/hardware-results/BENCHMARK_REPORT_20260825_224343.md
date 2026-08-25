# Smart Traffic Vision - Multi-Camera Hardware Benchmark & Sizing Report

**Generated On:** 2026-08-25 22:43:43

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **5.2 FPS/cam** (Total: **41.6 FPS**)
- **GPU Utilization & Headroom:** **5.9%** load (Headroom: **94.1%**)
- **VRAM Footprint & Free Space:** **2508 MB** / 8151 MB (Headroom: **5.51 GB**)
- **CPU Utilization & Headroom:** **37.7%** load (Headroom: **62.3%**)
- **Frame Drop Rate:** **76.54%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Skip | Display | FPS/Cam | Total FPS | GPU % | VRAM (MB) | CPU % | RAM (MB) | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.pt` | 1 | batched | 0 | YES | **24.1** | **24.1** | 9.2% | 2358 | 18.4% | 1821 | 1.3% |
| `yolov8n.pt` | 2 | batched | 0 | YES | **23.7** | **47.3** | 8.4% | 2367 | 17.6% | 2001 | 1.6% |
| `yolov8n.pt` | 4 | batched | 0 | YES | **13.4** | **53.5** | 8.0% | 2424 | 23.5% | 2378 | 42.9% |
| `yolov8n.pt` | 8 | batched | 0 | YES | **5.2** | **41.6** | 5.9% | 2508 | 37.7% | 3104 | 76.5% |
| `yolov8n.pt` | 1 | batched | 1 | YES | **16.4** | **16.4** | 2.3% | 2462 | 10.0% | 1860 | 32.7% |
| `yolov8n.pt` | 2 | batched | 1 | YES | **16.0** | **31.9** | 8.4% | 2466 | 15.0% | 2034 | 33.7% |
| `yolov8n.pt` | 4 | batched | 1 | YES | **14.7** | **59.0** | 5.3% | 2587 | 20.5% | 2375 | 36.0% |
| `yolov8n.pt` | 8 | batched | 1 | YES | **9.0** | **71.8** | 10.2% | 2525 | 34.8% | 3109 | 59.8% |
| `yolov8s.pt` | 1 | batched | 0 | YES | **16.3** | **16.3** | 7.7% | 2512 | 12.0% | 1913 | 33.3% |
| `yolov8s.pt` | 2 | batched | 0 | YES | **15.6** | **31.2** | 11.1% | 2510 | 13.7% | 2086 | 35.2% |
| `yolov8s.pt` | 4 | batched | 0 | YES | **12.8** | **51.0** | 15.5% | 2560 | 18.4% | 2438 | 45.6% |
| `yolov8s.pt` | 8 | batched | 0 | YES | **5.5** | **44.2** | 13.0% | 2718 | 32.4% | 3160 | 75.0% |
| `yolov8s.pt` | 1 | batched | 1 | YES | **16.4** | **16.4** | 3.4% | 2716 | 13.1% | 1930 | 32.7% |
| `yolov8s.pt` | 2 | batched | 1 | YES | **15.9** | **31.8** | 8.9% | 2717 | 12.5% | 2100 | 33.9% |
| `yolov8s.pt` | 4 | batched | 1 | YES | **15.3** | **61.1** | 11.3% | 2803 | 20.9% | 2431 | 35.0% |
| `yolov8s.pt` | 8 | batched | 1 | YES | **8.3** | **66.7** | 13.2% | 2864 | 33.9% | 3168 | 62.7% |