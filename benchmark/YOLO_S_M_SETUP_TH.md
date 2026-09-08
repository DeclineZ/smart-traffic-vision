# ชุด YOLOv8 / YOLO11 / YOLO26 ขนาด s และ m

โมเดลทางการ 6 รุ่นอยู่ใน `traffic/models` แต่ละรุ่นมี `.pt`, `.engine`, `.onnx` และ `.build.json` หลังสร้างเสร็จ

| โมเดล | ชื่อไฟล์ฐาน | PyTorch ID | TensorRT ID |
|---|---|---|---|
| YOLOv8s | yolov8s | v8s_fp16_pt | v8s_fp16_engine |
| YOLOv8m | yolov8m | v8m_fp16_pt | v8m_fp16_engine |
| YOLO11s | yolo11s | v11s_fp16_pt | v11s_fp16_engine |
| YOLO11m | yolo11m | v11m_fp16_pt | v11m_fp16_engine |
| YOLO26s | yolo26s | v26s_fp16_pt | v26s_fp16_engine |
| YOLO26m | yolo26m | v26m_fp16_pt | v26m_fp16_engine |

ใช้ Config [models_yolo_s_m.yaml](models_yolo_s_m.yaml) ซึ่งแยกจาก Config ตัวอย่างเดิม และใช้ Ultralytics **8.4.142** จาก `benchmark/runtime-yolo26` โดย Runner เลือก runtime นี้ให้อัตโนมัติ ไม่ต้อง Upgrade Ultralytics ใน `.venv` เดิม

Engine สร้างบน RTX 4060, TensorRT 10.14.1.48.post1, Batch 1, ขนาด 640×640, FP16 (`quantize=16`), Static shape, Workspace 2 GiB และ `nms=False` ไฟล์ `.build.json` เก็บแหล่งดาวน์โหลด, SHA-256 ของน้ำหนัก/Engine, จำนวน Parameters และ Environment ที่สร้าง

Config ใช้ `.pt` FP16 เช่นเดียวกับ Engine FP16 แต่ Engine อาจรับ/ส่ง Tensor แบบ FP32 ขณะที่ใช้ FP16 ภายใน จึงต้องอ่าน Input dtype ประกอบจาก Artifact metadata

YOLOv8/YOLO11 เป็น Anchor-free และใช้ NMS ภายนอก ส่วน YOLO26 คง One-to-one / NMS-free head ตามโมเดล ไม่แปลงให้เป็น NMS แบบรุ่นเก่า เปรียบเทียบความเร็วหลักด้วย Detection FPS; ไม่มีการใช้ IoU threshold เพื่อ Suppress กล่องสำหรับ YOLO26 NMS-free

## คำสั่งใช้งาน

รันจากโฟลเดอร์โปรเจกต์:

```powershell
cd "C:\Sea traffic\traffic"
```

ตรวจไฟล์ทั้ง 12 รายการ:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolo_s_m.yaml --mode validate
```

Smoke test กับวิดีโอกลางวัน/กลางคืนทั้ง 8 คลิป:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolo_s_m.yaml --mode performance --smoke
```

Benchmark เต็ม 3 รอบ:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolo_s_m.yaml --mode performance
```

ชุดนี้มี 12 Artifacts × 8 คลิป × 3 รอบ = **288 model/video runs** จึงใช้เวลามากกว่าชุดเริ่มต้น เลือกบางรุ่นได้ เช่น:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolo_s_m.yaml --mode performance --group night --models v26s_fp16_pt v26s_fp16_engine
```

Count-check ต้องกรอก `frame` และ `truth` ใน Config **ชุดนี้** ก่อน โดยอ้างอิงขั้นตอนใน [คู่มือหลัก](SETUP_AND_RUN_TH.md):

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolo_s_m.yaml --mode count-check
```

หากต้องการอ้างอิงโมเดลอื่นสำหรับ Difference map ให้เปลี่ยน `reference_model` ค่าเริ่มต้นคือ `v8s_fp16_pt` เมื่อไม่ได้เลือก Reference ระบบจะข้าม Difference map แต่ยังสร้าง Heat map ปกติ

## การเตรียมใหม่บนเครื่องอื่น

ติดตั้ง CUDA PyTorch/TensorRT ตามคู่มือหลัก จากนั้นติดตั้ง runtime แยก:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --target benchmark/runtime-yolo26 ultralytics==8.4.142
.\.venv\Scripts\python.exe benchmark/provision_models.py --download-only
.\.venv\Scripts\python.exe benchmark/provision_models.py
```

สคริปต์สร้างทีละโมเดลและข้าม Engine ที่มีอยู่แล้ว เลือกสร้างเฉพาะรุ่นได้ด้วย `--build yolo26m` หรือใช้ `--check` ตรวจ Predict ทั้ง `.pt/.engine` โดยไม่ Export ใหม่ หากเปลี่ยน GPU/ซอฟต์แวร์ อย่าอาศัย Engine เก่าโดยไม่ตรวจความเข้ากันได้

แหล่งข้อมูล: [YOLO26 ทางการ](https://docs.ultralytics.com/models/yolo26/) และ [TensorRT Export](https://docs.ultralytics.com/integrations/tensorrt/)
