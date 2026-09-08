# YOLOv7 และ YOLOv7-X แบบ Engine

ไฟล์ต้นทางอยู่ใน `traffic/models/yolov7.pt` และ `yolov7x.pt` Config `models_yolov7_engines.yaml` ลงทะเบียนเฉพาะ Engine 2 รุ่น ไม่มี PyTorch benchmark

## สร้าง Engine ด้วยตัวเอง

```powershell
cd "C:\Sea traffic\traffic"
.\.venv\Scripts\python.exe benchmark/provision_yolov7.py --model yolov7
.\.venv\Scripts\python.exe benchmark/provision_yolov7.py --model yolov7x
```

รันทีละคำสั่ง รอเสร็จแล้วจึงเริ่มตัวต่อไป ได้ models/yolov7.engine และ models/yolov7x.engine พร้อม .build.json ใช้ TensorRT FP16, batch 1, static 640x640, workspace 2 GiB สคริปต์จะไม่เขียนทับ Engine ที่มีอยู่

การ Export แปลงผล YOLOv7 เป็น xywh และ class score คูณ objectness เพื่อใช้ postprocessing ของ Ultralytics ใน Benchmark เดิม ไม่มี NMS ฝังใน Engine การแปลงนี้ไม่ได้เปลี่ยนโมเดลเป็น anchor-free ต้องตรวจผลก่อนนำไปจัดอันดับ

## ตรวจการทำงาน

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolov7_engines.yaml --mode performance --smoke --videos day_north day_south night_north night_south --output benchmark/model-results/v7-smoke
```

## Performance เต็ม 4 คลิป 1 รอบ

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_yolov7_engines.yaml --mode performance --videos day_north day_south night_north night_south --repeats 1 --prefetch 4 --output benchmark/model-results/v7-performance
```

รวม 2 jobs / 8 งานย่อยรายคลิป Config คัดลอกรายการวิดีโอและ truth จากชุดเดิม ณ วันที่สร้าง หากแก้ truth ภายหลังให้แก้ Config นี้ด้วย

สถานะเตรียมไฟล์: ดาวน์โหลดต้นทางครบแล้ว การสร้าง YOLOv7 ที่เริ่มไว้ถูกหยุดตามคำขอผู้ใช้ ยังไม่ได้สร้าง Engine ทั้งสองจนเสร็จหรือทดสอบผล TensorRT จริง
