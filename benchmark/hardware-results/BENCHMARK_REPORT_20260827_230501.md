# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-27 23:05:01

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **21.4 FPS/cam** (Total: **170.9 FPS**)
- **GPU Utilization & Headroom:** **9.6%** load (Headroom: **90.4%**)
- **VRAM Footprint & Free Space:** **2537 MB** / 8151 MB (Headroom: **5.48 GB**)
- **CPU Utilization & Headroom:** **31.6%** load (Headroom: **68.4%**)
- **Frame Drop Rate:** **7.32%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8n.engine` | 1 | batched | bytetrack | YES | **24.1** | **24.1** | 4.0% | 0.7% | 2508 | 9.7% | 0.0% |
| `yolov8n.engine` | 2 | batched | bytetrack | YES | **24.0** | **48.1** | 4.5% | 0.0% | 2493 | 11.7% | 0.0% |
| `yolov8n.engine` | 4 | batched | bytetrack | YES | **23.5** | **94.0** | 6.4% | 0.8% | 2504 | 16.0% | 1.2% |
| `yolov8n.engine` | 8 | batched | bytetrack | YES | **21.4** | **170.9** | 9.6% | 0.4% | 2537 | 31.6% | 7.3% |
| `yolov8n.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 3.2% | 0.3% | 2514 | 9.7% | 0.0% |
| `yolov8n.engine` | 2 | batched | bytetrack | YES | **23.8** | **47.5** | 2.7% | 0.0% | 2516 | 11.0% | 0.0% |
| `yolov8n.engine` | 4 | batched | bytetrack | YES | **23.5** | **93.9** | 5.6% | 0.8% | 2525 | 17.1% | 1.5% |
| `yolov8n.engine` | 8 | batched | bytetrack | YES | **21.4** | **171.2** | 6.9% | 0.5% | 2535 | 33.9% | 6.3% |
| `yolov8s.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 8.5% | 0.9% | 6738 | 10.2% | 0.0% |
| `yolov8s.engine` | 2 | batched | bytetrack | YES | **24.1** | **48.2** | 7.0% | 0.3% | 6744 | 12.8% | 0.0% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **23.4** | **93.4** | 12.4% | 0.7% | 6753 | 18.2% | 1.8% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **21.3** | **170.3** | 19.4% | 0.5% | 6763 | 33.0% | 7.5% |
| `yolov8s.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 6.2% | 0.8% | 6742 | 8.7% | 0.0% |
| `yolov8s.engine` | 2 | batched | bytetrack | YES | **24.2** | **48.5** | 4.7% | 0.2% | 6752 | 12.2% | 0.0% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **23.5** | **94.0** | 9.4% | 0.6% | 6766 | 18.5% | 1.2% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **21.5** | **171.9** | 16.8% | 0.4% | 6773 | 31.7% | 6.6% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8n.engine` (1 cams) | batched | **27.5%** | 2.0% | **C0:** 28%, **C1:** 11%, **C2:** 23%, **C3:** 8%, **C4:** 5%, **C5:** 3%, **C6:** 2%, **C7:** 2%, **C8:** 6%, **C9:** 3%, **C10:** 24%, **C11:** 12%, **C12:** 11%, **C13:** 5%, **C14:** 4%, **C15:** 4% |
| `yolov8n.engine` (2 cams) | batched | **32.1%** | 2.5% | **C0:** 30%, **C1:** 9%, **C2:** 32%, **C3:** 10%, **C4:** 5%, **C5:** 4%, **C6:** 4%, **C7:** 2%, **C8:** 9%, **C9:** 5%, **C10:** 30%, **C11:** 10%, **C12:** 13%, **C13:** 6%, **C14:** 7%, **C15:** 4% |
| `yolov8n.engine` (4 cams) | batched | **39.8%** | 4.6% | **C0:** 33%, **C1:** 13%, **C2:** 40%, **C3:** 21%, **C4:** 7%, **C5:** 7%, **C6:** 5%, **C7:** 5%, **C8:** 9%, **C9:** 10%, **C10:** 34%, **C11:** 20%, **C12:** 17%, **C13:** 13%, **C14:** 8%, **C15:** 7% |
| `yolov8n.engine` (8 cams) | batched | **58.0%** | 16.8% | **C0:** 53%, **C1:** 21%, **C2:** 35%, **C3:** 58%, **C4:** 18%, **C5:** 24%, **C6:** 17%, **C7:** 19%, **C8:** 24%, **C9:** 26%, **C10:** 45%, **C11:** 50%, **C12:** 25%, **C13:** 36%, **C14:** 17%, **C15:** 26% |
| `yolov8n.engine` (1 cams) | batched | **26.5%** | 2.1% | **C0:** 26%, **C1:** 11%, **C2:** 19%, **C3:** 15%, **C4:** 4%, **C5:** 3%, **C6:** 2%, **C7:** 2%, **C8:** 5%, **C9:** 5%, **C10:** 22%, **C11:** 12%, **C12:** 7%, **C13:** 7%, **C14:** 4%, **C15:** 3% |
| `yolov8n.engine` (2 cams) | batched | **26.7%** | 1.9% | **C0:** 27%, **C1:** 7%, **C2:** 25%, **C3:** 14%, **C4:** 4%, **C5:** 4%, **C6:** 2%, **C7:** 2%, **C8:** 7%, **C9:** 6%, **C10:** 24%, **C11:** 21%, **C12:** 11%, **C13:** 8%, **C14:** 5%, **C15:** 4% |
| `yolov8n.engine` (4 cams) | batched | **36.1%** | 6.2% | **C0:** 34%, **C1:** 11%, **C2:** 32%, **C3:** 22%, **C4:** 10%, **C5:** 8%, **C6:** 6%, **C7:** 7%, **C8:** 14%, **C9:** 10%, **C10:** 36%, **C11:** 25%, **C12:** 18%, **C13:** 13%, **C14:** 10%, **C15:** 8% |
| `yolov8n.engine` (8 cams) | batched | **54.9%** | 20.2% | **C0:** 51%, **C1:** 28%, **C2:** 55%, **C3:** 42%, **C4:** 25%, **C5:** 24%, **C6:** 20%, **C7:** 22%, **C8:** 32%, **C9:** 25%, **C10:** 51%, **C11:** 39%, **C12:** 40%, **C13:** 31%, **C14:** 26%, **C15:** 24% |
| `yolov8s.engine` (1 cams) | batched | **27.7%** | 3.2% | **C0:** 28%, **C1:** 10%, **C2:** 19%, **C3:** 17%, **C4:** 4%, **C5:** 4%, **C6:** 4%, **C7:** 3%, **C8:** 4%, **C9:** 6%, **C10:** 18%, **C11:** 12%, **C12:** 8%, **C13:** 8%, **C14:** 3%, **C15:** 5% |
| `yolov8s.engine` (2 cams) | batched | **31.4%** | 3.2% | **C0:** 31%, **C1:** 10%, **C2:** 17%, **C3:** 31%, **C4:** 5%, **C5:** 6%, **C6:** 4%, **C7:** 3%, **C8:** 7%, **C9:** 8%, **C10:** 20%, **C11:** 20%, **C12:** 9%, **C13:** 14%, **C14:** 5%, **C15:** 6% |
| `yolov8s.engine` (4 cams) | batched | **42.8%** | 5.2% | **C0:** 39%, **C1:** 16%, **C2:** 43%, **C3:** 21%, **C4:** 7%, **C5:** 8%, **C6:** 6%, **C7:** 5%, **C8:** 14%, **C9:** 10%, **C10:** 40%, **C11:** 22%, **C12:** 19%, **C13:** 14%, **C14:** 8%, **C15:** 8% |
| `yolov8s.engine` (8 cams) | batched | **54.8%** | 14.9% | **C0:** 53%, **C1:** 23%, **C2:** 54%, **C3:** 33%, **C4:** 24%, **C5:** 18%, **C6:** 18%, **C7:** 15%, **C8:** 37%, **C9:** 21%, **C10:** 55%, **C11:** 46%, **C12:** 43%, **C13:** 28%, **C14:** 30%, **C15:** 19% |
| `yolov8s.engine` (1 cams) | batched | **26.0%** | 1.8% | **C0:** 26%, **C1:** 7%, **C2:** 21%, **C3:** 12%, **C4:** 3%, **C5:** 2%, **C6:** 3%, **C7:** 2%, **C8:** 4%, **C9:** 2%, **C10:** 22%, **C11:** 10%, **C12:** 8%, **C13:** 5%, **C14:** 3%, **C15:** 2% |
| `yolov8s.engine` (2 cams) | batched | **29.5%** | 2.8% | **C0:** 30%, **C1:** 12%, **C2:** 24%, **C3:** 24%, **C4:** 5%, **C5:** 4%, **C6:** 3%, **C7:** 3%, **C8:** 6%, **C9:** 8%, **C10:** 26%, **C11:** 16%, **C12:** 9%, **C13:** 10%, **C14:** 3%, **C15:** 4% |
| `yolov8s.engine` (4 cams) | batched | **35.7%** | 7.4% | **C0:** 36%, **C1:** 14%, **C2:** 32%, **C3:** 30%, **C4:** 10%, **C5:** 10%, **C6:** 7%, **C7:** 8%, **C8:** 10%, **C9:** 14%, **C10:** 32%, **C11:** 28%, **C12:** 19%, **C13:** 18%, **C14:** 10%, **C15:** 10% |
| `yolov8s.engine` (8 cams) | batched | **60.9%** | 13.6% | **C0:** 47%, **C1:** 24%, **C2:** 53%, **C3:** 38%, **C4:** 28%, **C5:** 20%, **C6:** 18%, **C7:** 14%, **C8:** 33%, **C9:** 20%, **C10:** 61%, **C11:** 34%, **C12:** 40%, **C13:** 23%, **C14:** 26%, **C15:** 19% |