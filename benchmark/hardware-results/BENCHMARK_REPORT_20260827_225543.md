# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 22:55:43

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **22.2 FPS/cam** (Total: **177.5 FPS**)
- **GPU Utilization & Headroom:** **17.2%** load (Headroom: **82.8%**)
- **VRAM Footprint & Free Space:** **6545 MB** / 8151 MB (Headroom: **1.57 GB**)
- **CPU Utilization & Headroom:** **43.6%** load (Headroom: **56.4%**)
- **Frame Drop Rate:** **5.10%**
- **Max Safe Stream Capacity:** **9 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **22.2** | **177.5** | 17.2% | 0.3% | 6545 | 43.6% | 5.1% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (8 cams) | batched | **63.1%** | 28.0% | **C0:** 63%, **C1:** 34%, **C2:** 63%, **C3:** 59%, **C4:** 36%, **C5:** 32%, **C6:** 32%, **C7:** 28%, **C8:** 40%, **C9:** 34%, **C10:** 59%, **C11:** 58%, **C12:** 42%, **C13:** 39%, **C14:** 35%, **C15:** 32% |