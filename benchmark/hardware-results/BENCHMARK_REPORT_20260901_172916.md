# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-09-01 17:29:16

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[A] - GOOD (Production Ready)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **22.6 FPS/cam** (Total: **180.4 FPS**)
- **GPU Utilization & Headroom:** **12.7%** load (Headroom: **87.3%**)
- **VRAM Footprint & Free Space:** **3823 MB** / 8151 MB (Headroom: **4.23 GB**)
- **CPU Utilization & Headroom:** **33.7%** load (Headroom: **66.3%**)
- **Frame Drop Rate:** **3.69%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolo11s.engine` | 4 | batched | bytetrack | YES | **23.7** | **94.8** | 8.7% | 0.9% | 3813 | 21.3% | 0.7% |
| `yolo11s.engine` | 8 | batched | bytetrack | YES | **22.6** | **180.4** | 12.7% | 0.6% | 3823 | 33.7% | 3.7% |
| `yolo11m.engine` | 4 | batched | bytetrack | YES | **23.9** | **95.4** | 16.9% | 0.7% | 4343 | 21.5% | 0.7% |
| `yolo11m.engine` | 8 | batched | bytetrack | YES | **22.6** | **180.5** | 28.4% | 0.7% | 4386 | 32.7% | 3.8% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolo11s.engine` (4 cams) | batched | **47.9%** | 7.5% | **C0:** 41%, **C1:** 19%, **C2:** 48%, **C3:** 23%, **C4:** 11%, **C5:** 12%, **C6:** 8%, **C7:** 8%, **C8:** 16%, **C9:** 16%, **C10:** 34%, **C11:** 29%, **C12:** 24%, **C13:** 20%, **C14:** 12%, **C15:** 11% |
| `yolo11s.engine` (8 cams) | batched | **59.1%** | 18.9% | **C0:** 49%, **C1:** 25%, **C2:** 53%, **C3:** 41%, **C4:** 25%, **C5:** 23%, **C6:** 19%, **C7:** 19%, **C8:** 32%, **C9:** 29%, **C10:** 59%, **C11:** 37%, **C12:** 38%, **C13:** 32%, **C14:** 25%, **C15:** 24% |
| `yolo11m.engine` (4 cams) | batched | **42.7%** | 7.5% | **C0:** 35%, **C1:** 16%, **C2:** 43%, **C3:** 30%, **C4:** 11%, **C5:** 11%, **C6:** 8%, **C7:** 9%, **C8:** 17%, **C9:** 15%, **C10:** 38%, **C11:** 33%, **C12:** 23%, **C13:** 22%, **C14:** 12%, **C15:** 14% |
| `yolo11m.engine` (8 cams) | batched | **55.9%** | 16.3% | **C0:** 45%, **C1:** 23%, **C2:** 40%, **C3:** 56%, **C4:** 21%, **C5:** 25%, **C6:** 16%, **C7:** 19%, **C8:** 25%, **C9:** 30%, **C10:** 46%, **C11:** 52%, **C12:** 30%, **C13:** 37%, **C14:** 23%, **C15:** 25% |