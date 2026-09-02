# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-09-01 17:26:10

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[A] - GOOD (Production Ready)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **23.1 FPS/cam** (Total: **185.1 FPS**)
- **GPU Utilization & Headroom:** **18.5%** load (Headroom: **81.5%**)
- **VRAM Footprint & Free Space:** **3829 MB** / 8151 MB (Headroom: **4.22 GB**)
- **CPU Utilization & Headroom:** **35.8%** load (Headroom: **64.2%**)
- **Frame Drop Rate:** **3.24%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo11s.engine` | 4 | batched | bytetrack | YES | **24.1** | **96.4** | 10.2% | 0.9% | 3795 | 18.5% | 0.3% |
| `yolo11s.engine` | 8 | batched | bytetrack | YES | **23.1** | **185.1** | 18.5% | 0.9% | 3829 | 35.8% | 3.2% |
| `yolo11s.engine` | 4 | batched | bytetrack | YES | **24.3** | **97.2** | 7.2% | 0.6% | 3795 | 15.9% | 0.2% |
| `yolo11s.engine` | 8 | batched | bytetrack | YES | **23.5** | **188.3** | 12.9% | 0.7% | 3829 | 32.1% | 1.8% |
| `yolo11m.engine` | 4 | batched | bytetrack | YES | **24.2** | **96.8** | 23.1% | 0.6% | 4350 | 19.5% | 0.5% |
| `yolo11m.engine` | 8 | batched | bytetrack | YES | **19.8** | **158.2** | 35.0% | 0.7% | 4384 | 35.1% | 17.4% |
| `yolo11m.engine` | 4 | batched | bytetrack | YES | **24.2** | **96.8** | 15.6% | 0.5% | 4350 | 18.2% | 0.5% |
| `yolo11m.engine` | 8 | batched | bytetrack | YES | **23.5** | **188.3** | 27.9% | 0.5% | 4384 | 30.9% | 1.8% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolo11s.engine` (4 cams) | batched | **42.8%** | 5.6% | **C0:** 41%, **C1:** 12%, **C2:** 43%, **C3:** 27%, **C4:** 8%, **C5:** 9%, **C6:** 6%, **C7:** 6%, **C8:** 13%, **C9:** 10%, **C10:** 39%, **C11:** 24%, **C12:** 22%, **C13:** 14%, **C14:** 8%, **C15:** 6% |
| `yolo11s.engine` (8 cams) | batched | **59.3%** | 17.3% | **C0:** 58%, **C1:** 23%, **C2:** 56%, **C3:** 43%, **C4:** 28%, **C5:** 22%, **C6:** 21%, **C7:** 17%, **C8:** 37%, **C9:** 25%, **C10:** 59%, **C11:** 39%, **C12:** 46%, **C13:** 29%, **C14:** 32%, **C15:** 24% |
| `yolo11s.engine` (4 cams) | batched | **43.8%** | 5.0% | **C0:** 33%, **C1:** 14%, **C2:** 34%, **C3:** 24%, **C4:** 7%, **C5:** 7%, **C6:** 5%, **C7:** 5%, **C8:** 11%, **C9:** 7%, **C10:** 44%, **C11:** 16%, **C12:** 18%, **C13:** 12%, **C14:** 7%, **C15:** 5% |
| `yolo11s.engine` (8 cams) | batched | **57.6%** | 17.0% | **C0:** 50%, **C1:** 22%, **C2:** 37%, **C3:** 57%, **C4:** 20%, **C5:** 22%, **C6:** 17%, **C7:** 18%, **C8:** 25%, **C9:** 29%, **C10:** 36%, **C11:** 58%, **C12:** 28%, **C13:** 37%, **C14:** 21%, **C15:** 27% |
| `yolo11m.engine` (4 cams) | batched | **42.8%** | 4.9% | **C0:** 43%, **C1:** 14%, **C2:** 26%, **C3:** 37%, **C4:** 9%, **C5:** 8%, **C6:** 6%, **C7:** 5%, **C8:** 12%, **C9:** 14%, **C10:** 31%, **C11:** 37%, **C12:** 18%, **C13:** 21%, **C14:** 9%, **C15:** 12% |
| `yolo11m.engine` (8 cams) | batched | **58.6%** | 17.8% | **C0:** 59%, **C1:** 23%, **C2:** 53%, **C3:** 42%, **C4:** 28%, **C5:** 20%, **C6:** 21%, **C7:** 18%, **C8:** 37%, **C9:** 25%, **C10:** 57%, **C11:** 39%, **C12:** 44%, **C13:** 30%, **C14:** 28%, **C15:** 24% |
| `yolo11m.engine` (4 cams) | batched | **36.8%** | 5.9% | **C0:** 35%, **C1:** 16%, **C2:** 37%, **C3:** 26%, **C4:** 9%, **C5:** 8%, **C6:** 6%, **C7:** 6%, **C8:** 12%, **C9:** 12%, **C10:** 36%, **C11:** 24%, **C12:** 20%, **C13:** 17%, **C14:** 9%, **C15:** 9% |
| `yolo11m.engine` (8 cams) | batched | **55.8%** | 14.0% | **C0:** 51%, **C1:** 20%, **C2:** 34%, **C3:** 56%, **C4:** 20%, **C5:** 20%, **C6:** 14%, **C7:** 18%, **C8:** 24%, **C9:** 28%, **C10:** 39%, **C11:** 54%, **C12:** 30%, **C13:** 33%, **C14:** 19%, **C15:** 22% |