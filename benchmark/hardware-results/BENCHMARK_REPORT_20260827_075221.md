# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 07:52:21

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **10.7 FPS/cam** (Total: **85.2 FPS**)
- **GPU Utilization & Headroom:** **5.9%** load (Headroom: **94.1%**)
- **VRAM Footprint & Free Space:** **6342 MB** / 8151 MB (Headroom: **1.77 GB**)
- **CPU Utilization & Headroom:** **39.0%** load (Headroom: **61.0%**)
- **Frame Drop Rate:** **54.20%**
- **Max Safe Stream Capacity:** **9 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **20.6** | **82.3** | 9.6% | 0.7% | 6304 | 28.0% | 13.7% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **10.7** | **85.2** | 5.9% | 0.1% | 6342 | 39.0% | 54.2% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **22.9** | **91.8** | 10.2% | 1.4% | 6348 | 22.7% | 4.8% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **17.1** | **137.1** | 11.9% | 0.5% | 6348 | 30.5% | 26.3% |
| `yolov8n.engine` | 4 | batched | bytetrack | YES | **23.3** | **93.2** | 7.5% | 1.0% | 3055 | 21.2% | 2.1% |
| `yolov8n.engine` | 8 | batched | bytetrack | YES | **17.7** | **141.7** | 9.0% | 0.7% | 3065 | 33.2% | 23.9% |
| `yolov8n.engine` | 4 | batched | bytetrack | YES | **23.4** | **93.6** | 5.5% | 0.8% | 3053 | 21.3% | 2.0% |
| `yolov8n.engine` | 8 | batched | bytetrack | YES | **19.1** | **152.7** | 6.3% | 0.8% | 3065 | 30.2% | 18.1% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s.engine` (4 cams) | batched | **77.6%** | 13.8% | **C0:** 76%, **C1:** 78%, **C2:** 32%, **C3:** 28%, **C4:** 17%, **C5:** 17%, **C6:** 22%, **C7:** 26%, **C8:** 15%, **C9:** 15%, **C10:** 23%, **C11:** 19%, **C12:** 19%, **C13:** 14%, **C14:** 15%, **C15:** 19% |
| `yolov8s.engine` (8 cams) | batched | **74.1%** | 26.1% | **C0:** 74%, **C1:** 66%, **C2:** 38%, **C3:** 37%, **C4:** 38%, **C5:** 38%, **C6:** 40%, **C7:** 43%, **C8:** 29%, **C9:** 30%, **C10:** 36%, **C11:** 34%, **C12:** 27%, **C13:** 26%, **C14:** 28%, **C15:** 28% |
| `yolov8s.engine` (4 cams) | batched | **68.6%** | 11.0% | **C0:** 69%, **C1:** 49%, **C2:** 32%, **C3:** 23%, **C4:** 12%, **C5:** 12%, **C6:** 11%, **C7:** 12%, **C8:** 15%, **C9:** 14%, **C10:** 32%, **C11:** 20%, **C12:** 16%, **C13:** 15%, **C14:** 11%, **C15:** 11% |
| `yolov8s.engine` (8 cams) | batched | **61.0%** | 15.0% | **C0:** 61%, **C1:** 44%, **C2:** 42%, **C3:** 47%, **C4:** 17%, **C5:** 18%, **C6:** 15%, **C7:** 17%, **C8:** 22%, **C9:** 19%, **C10:** 54%, **C11:** 28%, **C12:** 33%, **C13:** 26%, **C14:** 19%, **C15:** 17% |
| `yolov8n.engine` (4 cams) | batched | **60.5%** | 6.9% | **C0:** 60%, **C1:** 45%, **C2:** 21%, **C3:** 32%, **C4:** 8%, **C5:** 9%, **C6:** 7%, **C7:** 11%, **C8:** 11%, **C9:** 14%, **C10:** 37%, **C11:** 27%, **C12:** 15%, **C13:** 17%, **C14:** 7%, **C15:** 9% |
| `yolov8n.engine` (8 cams) | batched | **60.7%** | 15.3% | **C0:** 61%, **C1:** 44%, **C2:** 45%, **C3:** 49%, **C4:** 15%, **C5:** 22%, **C6:** 16%, **C7:** 17%, **C8:** 22%, **C9:** 27%, **C10:** 48%, **C11:** 51%, **C12:** 29%, **C13:** 36%, **C14:** 18%, **C15:** 21% |
| `yolov8n.engine` (4 cams) | batched | **58.2%** | 7.3% | **C0:** 58%, **C1:** 48%, **C2:** 19%, **C3:** 43%, **C4:** 9%, **C5:** 10%, **C6:** 7%, **C7:** 10%, **C8:** 9%, **C9:** 11%, **C10:** 28%, **C11:** 32%, **C12:** 12%, **C13:** 17%, **C14:** 8%, **C15:** 8% |
| `yolov8n.engine` (8 cams) | batched | **61.9%** | 14.6% | **C0:** 62%, **C1:** 39%, **C2:** 56%, **C3:** 34%, **C4:** 18%, **C5:** 16%, **C6:** 16%, **C7:** 17%, **C8:** 26%, **C9:** 17%, **C10:** 52%, **C11:** 35%, **C12:** 34%, **C13:** 20%, **C14:** 17%, **C15:** 15% |