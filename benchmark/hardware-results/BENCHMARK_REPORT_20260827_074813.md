# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 07:48:13

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **15.4 FPS/cam** (Total: **123.2 FPS**)
- **GPU Utilization & Headroom:** **13.5%** load (Headroom: **86.5%**)
- **VRAM Footprint & Free Space:** **6334 MB** / 8151 MB (Headroom: **1.77 GB**)
- **CPU Utilization & Headroom:** **36.7%** load (Headroom: **63.3%**)
- **Frame Drop Rate:** **35.47%**
- **Max Safe Stream Capacity:** **9 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **15.4** | **123.2** | 13.5% | 0.8% | 6334 | 36.7% | 35.5% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (8 cams) | batched | **68.6%** | 20.4% | **C0:** 69%, **C1:** 54%, **C2:** 57%, **C3:** 39%, **C4:** 27%, **C5:** 23%, **C6:** 22%, **C7:** 20%, **C8:** 32%, **C9:** 22%, **C10:** 56%, **C11:** 36%, **C12:** 40%, **C13:** 29%, **C14:** 26%, **C15:** 22% |