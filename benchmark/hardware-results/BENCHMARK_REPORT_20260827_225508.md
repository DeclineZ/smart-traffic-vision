# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 22:55:08

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **17.1 FPS/cam** (Total: **136.9 FPS**)
- **GPU Utilization & Headroom:** **17.2%** load (Headroom: **82.8%**)
- **VRAM Footprint & Free Space:** **6670 MB** / 8151 MB (Headroom: **1.45 GB**)
- **CPU Utilization & Headroom:** **46.0%** load (Headroom: **54.0%**)
- **Frame Drop Rate:** **26.54%**
- **Max Safe Stream Capacity:** **9 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** VRAM Capacity Limit (6670 MB / 8151 MB used)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **17.1** | **136.9** | 17.2% | 0.2% | 6670 | 46.0% | 26.5% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (8 cams) | batched | **68.7%** | 29.9% | **C0:** 69%, **C1:** 43%, **C2:** 65%, **C3:** 64%, **C4:** 36%, **C5:** 35%, **C6:** 32%, **C7:** 30%, **C8:** 37%, **C9:** 35%, **C10:** 65%, **C11:** 57%, **C12:** 48%, **C13:** 42%, **C14:** 32%, **C15:** 34% |