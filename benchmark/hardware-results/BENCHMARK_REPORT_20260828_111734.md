# Smart Traffic Vision - Production Multi-Camera Benchmark & Sizing Report

**Generated On:** 2026-08-28 11:17:34

## 1. System Hardware Baseline

- **GPU Model:** NVIDIA GeForce RTX 5060 Laptop GPU
- **CPU:** 8 Physical Cores / 16 Logical Threads
- **System RAM:** 15.3 GB
- **Dedicated VRAM:** 8.0 GB
- **PyTorch / CUDA:** 2.12.0.dev20260408+cu128 (CUDA: 12.8)

## 2. 8-Camera Sizing Feasibility Verdict

> **Verdict:** **[C / FAIL] - INSUFFICIENT (Cannot sustain target 8-camera FPS without frame drops or severe latency)**

- **Target Throughput:** 8 Cameras @ 25.0 FPS = **200 Total FPS**
- **Achieved Throughput:** **21.7 FPS/cam** (Total: **173.9 FPS**)
- **GPU Utilization & Headroom:** **18.7%** load (Headroom: **81.3%**)
- **VRAM Footprint & Free Space:** **3305 MB** / 8151 MB (Headroom: **4.73 GB**)
- **CPU Utilization & Headroom:** **30.9%** load (Headroom: **69.1%**)
- **Frame Drop Rate:** **5.88%**
- **Max Safe Stream Capacity:** **17 Camera Streams** @ 25.0 FPS
- **Identified Bottlenecks:** None (Hardware operates with healthy margin across all subsystems)

## 3. Detailed Results Matrix

| Model | Streams | Mode | Tracker | NVENC | FPS/Cam | Total FPS | GPU % | NVENC % | VRAM (MB) | CPU % | Drops |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `yolov8s_ws1.engine` | 1 | batched | bytetrack | YES | **24.2** | **24.2** | 5.4% | 0.0% | 3229 | 8.5% | 0.0% |
| `yolov8s_ws1.engine` | 2 | batched | bytetrack | YES | **24.1** | **48.2** | 7.6% | 0.8% | 3246 | 10.2% | 0.0% |
| `yolov8s_ws1.engine` | 4 | batched | bytetrack | YES | **23.4** | **93.8** | 13.1% | 0.7% | 3263 | 17.3% | 1.6% |
| `yolov8s_ws1.engine` | 8 | batched | bytetrack | YES | **21.7** | **173.9** | 18.7% | 0.4% | 3305 | 30.9% | 5.9% |
| `yolov8s_ws1.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 3.4% | 0.0% | 3260 | 8.5% | 0.0% |
| `yolov8s_ws1.engine` | 2 | batched | bytetrack | YES | **24.1** | **48.1** | 6.8% | 0.8% | 3264 | 9.8% | 0.0% |
| `yolov8s_ws1.engine` | 4 | batched | bytetrack | YES | **23.5** | **93.9** | 8.6% | 0.1% | 3273 | 14.3% | 1.5% |
| `yolov8s_ws1.engine` | 8 | batched | bytetrack | YES | **21.8** | **174.3** | 12.5% | 0.4% | 3308 | 29.8% | 5.8% |
| `yolov8s.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 8.3% | 0.2% | 6817 | 8.1% | 0.0% |
| `yolov8s.engine` | 2 | batched | bytetrack | YES | **24.1** | **48.1** | 10.5% | 0.6% | 6820 | 10.1% | 0.0% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **23.4** | **93.4** | 13.1% | 0.6% | 6831 | 18.1% | 1.9% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **21.3** | **170.7** | 20.7% | 0.6% | 6866 | 32.9% | 7.0% |
| `yolov8s.engine` | 1 | batched | bytetrack | YES | **24.4** | **24.4** | 6.7% | 0.2% | 6817 | 6.2% | 0.0% |
| `yolov8s.engine` | 2 | batched | bytetrack | YES | **24.2** | **48.4** | 6.8% | 0.0% | 6822 | 11.8% | 0.0% |
| `yolov8s.engine` | 4 | batched | bytetrack | YES | **23.4** | **93.8** | 10.2% | 1.0% | 6831 | 15.2% | 1.6% |
| `yolov8s.engine` | 8 | batched | bytetrack | YES | **21.9** | **174.9** | 14.6% | 0.7% | 6866 | 27.0% | 5.5% |

## 4. Individual CPU Core Utilization & Thread Balance

| Model / Setup | Mode | Peak Core % | Min Core % | Per-Core Utilization Distribution (Core 0 to N) |
| :--- | :---: | :---: | :---: | :--- |
| `yolov8s_ws1.engine` (1 cams) | batched | **23.9%** | 1.6% | **C0:** 24%, **C1:** 8%, **C2:** 20%, **C3:** 9%, **C4:** 3%, **C5:** 4%, **C6:** 2%, **C7:** 2%, **C8:** 4%, **C9:** 4%, **C10:** 15%, **C11:** 14%, **C12:** 12%, **C13:** 5%, **C14:** 4%, **C15:** 3% |
| `yolov8s_ws1.engine` (2 cams) | batched | **31.2%** | 2.5% | **C0:** 29%, **C1:** 7%, **C2:** 31%, **C3:** 13%, **C4:** 3%, **C5:** 4%, **C6:** 3%, **C7:** 2%, **C8:** 6%, **C9:** 4%, **C10:** 22%, **C11:** 9%, **C12:** 11%, **C13:** 5%, **C14:** 4%, **C15:** 3% |
| `yolov8s_ws1.engine` (4 cams) | batched | **31.8%** | 5.9% | **C0:** 31%, **C1:** 14%, **C2:** 31%, **C3:** 26%, **C4:** 10%, **C5:** 9%, **C6:** 6%, **C7:** 7%, **C8:** 12%, **C9:** 11%, **C10:** 26%, **C11:** 32%, **C12:** 20%, **C13:** 15%, **C14:** 11%, **C15:** 9% |
| `yolov8s_ws1.engine` (8 cams) | batched | **56.9%** | 12.9% | **C0:** 45%, **C1:** 23%, **C2:** 37%, **C3:** 55%, **C4:** 18%, **C5:** 24%, **C6:** 13%, **C7:** 16%, **C8:** 20%, **C9:** 33%, **C10:** 39%, **C11:** 57%, **C12:** 31%, **C13:** 35%, **C14:** 16%, **C15:** 24% |
| `yolov8s_ws1.engine` (1 cams) | batched | **19.4%** | 2.9% | **C0:** 19%, **C1:** 8%, **C2:** 14%, **C3:** 17%, **C4:** 3%, **C5:** 4%, **C6:** 3%, **C7:** 4%, **C8:** 4%, **C9:** 4%, **C10:** 10%, **C11:** 17%, **C12:** 7%, **C13:** 8%, **C14:** 3%, **C15:** 3% |
| `yolov8s_ws1.engine` (2 cams) | batched | **27.0%** | 3.4% | **C0:** 27%, **C1:** 7%, **C2:** 18%, **C3:** 14%, **C4:** 5%, **C5:** 4%, **C6:** 3%, **C7:** 3%, **C8:** 6%, **C9:** 5%, **C10:** 17%, **C11:** 20%, **C12:** 8%, **C13:** 7%, **C14:** 4%, **C15:** 3% |
| `yolov8s_ws1.engine` (4 cams) | batched | **46.1%** | 4.4% | **C0:** 22%, **C1:** 8%, **C2:** 46%, **C3:** 12%, **C4:** 7%, **C5:** 7%, **C6:** 4%, **C7:** 4%, **C8:** 12%, **C9:** 8%, **C10:** 36%, **C11:** 17%, **C12:** 16%, **C13:** 11%, **C14:** 7%, **C15:** 5% |
| `yolov8s_ws1.engine` (8 cams) | batched | **50.6%** | 17.1% | **C0:** 43%, **C1:** 21%, **C2:** 41%, **C3:** 51%, **C4:** 21%, **C5:** 19%, **C6:** 17%, **C7:** 18%, **C8:** 24%, **C9:** 24%, **C10:** 46%, **C11:** 39%, **C12:** 33%, **C13:** 26%, **C14:** 23%, **C15:** 21% |
| `yolov8s.engine` (1 cams) | batched | **20.4%** | 2.2% | **C0:** 18%, **C1:** 7%, **C2:** 9%, **C3:** 20%, **C4:** 3%, **C5:** 3%, **C6:** 2%, **C7:** 4%, **C8:** 4%, **C9:** 7%, **C10:** 16%, **C11:** 11%, **C12:** 6%, **C13:** 7%, **C14:** 3%, **C15:** 3% |
| `yolov8s.engine` (2 cams) | batched | **25.3%** | 2.7% | **C0:** 24%, **C1:** 5%, **C2:** 13%, **C3:** 20%, **C4:** 4%, **C5:** 4%, **C6:** 4%, **C7:** 3%, **C8:** 5%, **C9:** 6%, **C10:** 25%, **C11:** 17%, **C12:** 9%, **C13:** 7%, **C14:** 4%, **C15:** 5% |
| `yolov8s.engine` (4 cams) | batched | **40.3%** | 7.0% | **C0:** 34%, **C1:** 13%, **C2:** 40%, **C3:** 18%, **C4:** 10%, **C5:** 11%, **C6:** 7%, **C7:** 7%, **C8:** 12%, **C9:** 11%, **C10:** 34%, **C11:** 36%, **C12:** 21%, **C13:** 12%, **C14:** 10%, **C15:** 8% |
| `yolov8s.engine` (8 cams) | batched | **58.0%** | 17.8% | **C0:** 47%, **C1:** 24%, **C2:** 57%, **C3:** 39%, **C4:** 24%, **C5:** 18%, **C6:** 20%, **C7:** 18%, **C8:** 28%, **C9:** 26%, **C10:** 58%, **C11:** 38%, **C12:** 43%, **C13:** 28%, **C14:** 27%, **C15:** 21% |
| `yolov8s.engine` (1 cams) | batched | **21.7%** | 0.9% | **C0:** 22%, **C1:** 8%, **C2:** 5%, **C3:** 17%, **C4:** 2%, **C5:** 1%, **C6:** 1%, **C7:** 2%, **C8:** 2%, **C9:** 2%, **C10:** 12%, **C11:** 10%, **C12:** 4%, **C13:** 5%, **C14:** 2%, **C15:** 2% |
| `yolov8s.engine` (2 cams) | batched | **28.2%** | 4.6% | **C0:** 24%, **C1:** 9%, **C2:** 14%, **C3:** 28%, **C4:** 7%, **C5:** 8%, **C6:** 5%, **C7:** 5%, **C8:** 6%, **C9:** 9%, **C10:** 18%, **C11:** 22%, **C12:** 8%, **C13:** 10%, **C14:** 5%, **C15:** 6% |
| `yolov8s.engine` (4 cams) | batched | **30.6%** | 7.2% | **C0:** 24%, **C1:** 10%, **C2:** 28%, **C3:** 27%, **C4:** 9%, **C5:** 8%, **C6:** 8%, **C7:** 7%, **C8:** 9%, **C9:** 10%, **C10:** 26%, **C11:** 31%, **C12:** 15%, **C13:** 12%, **C14:** 8%, **C15:** 9% |
| `yolov8s.engine` (8 cams) | batched | **45.2%** | 11.5% | **C0:** 37%, **C1:** 17%, **C2:** 45%, **C3:** 35%, **C4:** 20%, **C5:** 15%, **C6:** 12%, **C7:** 12%, **C8:** 28%, **C9:** 19%, **C10:** 45%, **C11:** 44%, **C12:** 33%, **C13:** 25%, **C14:** 21%, **C15:** 17% |