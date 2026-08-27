# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 07:46:56

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **16.6 FPS/cam** (Total: **132.5 FPS**)
- **GPU Utilization & Headroom:** **13.4%** load (Headroom: **86.6%**)
- **VRAM Footprint & Free Space:** **6158 MB** / 8151 MB (Headroom: **1.95 GB**)
- **CPU Utilization & Headroom:** **30.3%** load (Headroom: **69.7%**)
- **Frame Drop Rate:** **28.14%**
- **Max Safe Stream Capacity:** **9 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 8 | batched | bytetrack | NO | **16.6** | **132.5** | 13.4% | 0.0% | 6158 | 30.3% | 28.1% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (8 cams) | batched | **62.8%** | 14.2% | **C0:** 63%, **C1:** 52%, **C2:** 40%, **C3:** 46%, **C4:** 20%, **C5:** 16%, **C6:** 17%, **C7:** 18%, **C8:** 20%, **C9:** 20%, **C10:** 33%, **C11:** 47%, **C12:** 26%, **C13:** 30%, **C14:** 14%, **C15:** 16% |