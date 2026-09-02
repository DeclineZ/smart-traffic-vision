# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-09-01 21:04:01

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[A] - GOOD (Production Ready)**

- **Target Throughput:** 8 Cameras @ 15.0 FPS = **120 Total FPS**
- **Achieved Throughput:** **13.7 FPS/cam** (Total: **109.6 FPS**)
- **GPU Utilization & Headroom:** **27.0%** load (Headroom: **73.0%**)
- **VRAM Footprint & Free Space:** **4221 MB** / 8151 MB (Headroom: **3.84 GB**)
- **CPU Utilization & Headroom:** **27.6%** load (Headroom: **72.4%**)
- **Frame Drop Rate:** **3.47%**
- **Max Safe Stream Capacity:** **15 Camera Streams** @ 15.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8m.engine` | 4 | batched | bytetrack | YES | **14.1** | **56.5** | 15.2% | 0.5% | 4347 | 18.4% | 0.9% |
| `yolov8m.engine` | 8 | batched | bytetrack | YES | **13.7** | **109.6** | 27.0% | 0.5% | 4221 | 27.6% | 3.5% |
| `yolov8m.engine` | 4 | batched | bytetrack | YES | **14.4** | **57.7** | 14.8% | 0.5% | 4534 | 26.0% | 0.6% |
| `yolov8m.engine` | 8 | batched | bytetrack | YES | **13.9** | **111.1** | 18.6% | 0.6% | 4238 | 28.3% | 2.6% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8m.engine` (4 cams) | batched | **37.2%** | 4.9% | **C0:** 32%, **C1:** 10%, **C2:** 37%, **C3:** 26%, **C4:** 9%, **C5:** 7%, **C6:** 7%, **C7:** 5%, **C8:** 18%, **C9:** 14%, **C10:** 36%, **C11:** 25%, **C12:** 20%, **C13:** 18%, **C14:** 10%, **C15:** 12% |
| `yolov8m.engine` (8 cams) | batched | **50.5%** | 13.4% | **C0:** 40%, **C1:** 20%, **C2:** 41%, **C3:** 35%, **C4:** 16%, **C5:** 16%, **C6:** 13%, **C7:** 15%, **C8:** 24%, **C9:** 26%, **C10:** 50%, **C11:** 37%, **C12:** 29%, **C13:** 28%, **C14:** 19%, **C15:** 21% |
| `yolov8m.engine` (4 cams) | batched | **46.7%** | 12.5% | **C0:** 39%, **C1:** 20%, **C2:** 47%, **C3:** 32%, **C4:** 14%, **C5:** 16%, **C6:** 12%, **C7:** 13%, **C8:** 22%, **C9:** 23%, **C10:** 41%, **C11:** 36%, **C12:** 24%, **C13:** 28%, **C14:** 17%, **C15:** 21% |
| `yolov8m.engine` (8 cams) | batched | **49.3%** | 12.9% | **C0:** 37%, **C1:** 19%, **C2:** 34%, **C3:** 49%, **C4:** 19%, **C5:** 20%, **C6:** 13%, **C7:** 14%, **C8:** 23%, **C9:** 26%, **C10:** 40%, **C11:** 43%, **C12:** 29%, **C13:** 33%, **C14:** 20%, **C15:** 23% |