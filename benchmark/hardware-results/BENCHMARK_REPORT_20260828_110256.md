# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-28 11:02:56

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
- **GPU Utilization & Headroom:** **13.1%** load (Headroom: **86.9%**)
- **VRAM Footprint & Free Space:** **3317 MB** / 8151 MB (Headroom: **4.72 GB**)
- **CPU Utilization & Headroom:** **30.7%** load (Headroom: **69.3%**)
- **Frame Drop Rate:** **5.02%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_ws1.engine` | 1 | batched | bytetrack | YES | **24.2** | **24.2** | 3.4% | 0.4% | 3272 | 6.4% | 0.0% |
| `yolov8s_ws1.engine` | 2 | batched | bytetrack | YES | **24.2** | **48.4** | 5.2% | 0.2% | 3285 | 9.6% | 0.2% |
| `yolov8s_ws1.engine` | 4 | batched | bytetrack | YES | **23.7** | **94.7** | 8.5% | 0.5% | 3296 | 17.0% | 1.3% |
| `yolov8s_ws1.engine` | 8 | batched | bytetrack | YES | **22.2** | **177.5** | 13.1% | 0.5% | 3317 | 30.7% | 5.0% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s_ws1.engine` (1 cams) | batched | **18.5%** | 1.7% | **C0:** 18%, **C1:** 2%, **C2:** 16%, **C3:** 6%, **C4:** 2%, **C5:** 2%, **C6:** 2%, **C7:** 2%, **C8:** 3%, **C9:** 3%, **C10:** 14%, **C11:** 13%, **C12:** 7%, **C13:** 3%, **C14:** 2%, **C15:** 2% |
| `yolov8s_ws1.engine` (2 cams) | batched | **27.1%** | 1.8% | **C0:** 24%, **C1:** 7%, **C2:** 26%, **C3:** 12%, **C4:** 3%, **C5:** 4%, **C6:** 2%, **C7:** 2%, **C8:** 5%, **C9:** 3%, **C10:** 27%, **C11:** 8%, **C12:** 11%, **C13:** 5%, **C14:** 4%, **C15:** 3% |
| `yolov8s_ws1.engine` (4 cams) | batched | **35.1%** | 6.4% | **C0:** 26%, **C1:** 12%, **C2:** 35%, **C3:** 24%, **C4:** 9%, **C5:** 8%, **C6:** 7%, **C7:** 6%, **C8:** 15%, **C9:** 12%, **C10:** 32%, **C11:** 23%, **C12:** 23%, **C13:** 13%, **C14:** 12%, **C15:** 10% |
| `yolov8s_ws1.engine` (8 cams) | batched | **53.3%** | 15.4% | **C0:** 45%, **C1:** 24%, **C2:** 32%, **C3:** 53%, **C4:** 20%, **C5:** 21%, **C6:** 15%, **C7:** 17%, **C8:** 22%, **C9:** 30%, **C10:** 44%, **C11:** 50%, **C12:** 29%, **C13:** 35%, **C14:** 19%, **C15:** 26% |