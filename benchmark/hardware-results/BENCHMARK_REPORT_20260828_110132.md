# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-28 11:01:32

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **22.1 FPS/cam** (Total: **176.5 FPS**)
- **GPU Utilization & Headroom:** **13.2%** load (Headroom: **86.8%**)
- **VRAM Footprint & Free Space:** **6868 MB** / 8151 MB (Headroom: **1.25 GB**)
- **CPU Utilization & Headroom:** **32.1%** load (Headroom: **67.9%**)
- **Frame Drop Rate:** **5.63%**
- **Max Safe Stream Capacity:** **8 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** VRAM Capacity Limit (6868 MB / 8151 MB used)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 1 | batched | bytetrack | YES | **24.2** | **24.2** | 4.4% | 0.0% | 6775 | 8.4% | 0.0% |
| `yolov8s.engine` | 2 | batched | bytetrack | YES | **24.2** | **48.3** | 5.9% | 0.5% | 6800 | 10.4% | 0.2% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **23.7** | **94.8** | 10.3% | 1.0% | 6820 | 15.1% | 1.3% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **22.1** | **176.5** | 13.2% | 0.4% | 6868 | 32.1% | 5.6% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (1 cams) | batched | **18.3%** | 2.7% | **C0:** 18%, **C1:** 6%, **C2:** 17%, **C3:** 11%, **C4:** 3%, **C5:** 5%, **C6:** 3%, **C7:** 3%, **C8:** 6%, **C9:** 4%, **C10:** 15%, **C11:** 13%, **C12:** 8%, **C13:** 6%, **C14:** 4%, **C15:** 4% |
| `yolov8s.engine` (2 cams) | batched | **23.7%** | 4.0% | **C0:** 24%, **C1:** 7%, **C2:** 24%, **C3:** 13%, **C4:** 6%, **C5:** 4%, **C6:** 4%, **C7:** 4%, **C8:** 7%, **C9:** 7%, **C10:** 21%, **C11:** 11%, **C12:** 12%, **C13:** 6%, **C14:** 7%, **C15:** 5% |
| `yolov8s.engine` (4 cams) | batched | **29.6%** | 4.2% | **C0:** 25%, **C1:** 9%, **C2:** 30%, **C3:** 23%, **C4:** 8%, **C5:** 7%, **C6:** 4%, **C7:** 6%, **C8:** 11%, **C9:** 12%, **C10:** 28%, **C11:** 25%, **C12:** 18%, **C13:** 12%, **C14:** 9%, **C15:** 8% |
| `yolov8s.engine` (8 cams) | batched | **55.9%** | 16.1% | **C0:** 42%, **C1:** 26%, **C2:** 34%, **C3:** 56%, **C4:** 20%, **C5:** 26%, **C6:** 16%, **C7:** 21%, **C8:** 21%, **C9:** 32%, **C10:** 49%, **C11:** 50%, **C12:** 29%, **C13:** 39%, **C14:** 19%, **C15:** 25% |