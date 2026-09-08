# รวมกล่อง ตรวจ Ground Truth และวัด Accuracy

## 1. รันโหมดใหม่

```powershell
cd "C:\Sea traffic\traffic"
.\.venv\Scripts\python.exe benchmark/model_benchmark.py `
  --config benchmark/models_count_20frames.yaml `
  --mode ensemble-review `
  --videos day_east day_west night_east night_west `
  --output benchmark/model-results/ensemble-20frames
```

รัน Engine ที่เลือกทั้งหมด (ค่าเริ่มต้นใน Config นี้ 8 รุ่น) บน 20 ภาพ โดยไม่ใช้ truth จำนวนรถเดิม เก็บ predictions ที่ confidence >= 0.001 ตาม max_det ใน Config เพื่อคำนวณ AP ภายหลัง รวมข้อเสนอเฉพาะ confidence >= 0.25

กล่องคลาสเดียวกันที่ IoU >= 0.5 กับกล่องรวมจะรวมตำแหน่งแบบถ่วงน้ำหนักด้วย confidence เรียง confidence สูงก่อน โมเดลหนึ่งมีสิทธิ์สนับสนุนได้ครั้งเดียวต่อกลุ่ม กล่องไม่ทับกับรุ่นอื่นยังอยู่ ทุกกล่องเก็บชื่อโมเดลที่สนับสนุน ไม่รวมรถกับมอเตอร์ไซค์เข้าด้วยกัน

## 2. เปิด review.html

เปิด `benchmark/model-results/ensemble-20frames/review.html` ใน Browser ใช้งาน local file ได้

- เลือกภาพจากรายการ Green = หลายโมเดลสนับสนุน Orange = โมเดลเดียว
- Select box: คลิกกล่องหรือรายการ แล้วแก้ x1/y1/x2/y2 และคลาส กด Apply
- Delete selected box: ลบกล่องผิด
- Draw new box: ลากกล่องเพิ่มรถที่ทุกโมเดลพลาด เลือกคลาสก่อนวาด
- Confirm image reviewed: ยืนยันเมื่อคุณตรวจครบทั้งภาพ รวมกรณีภาพว่าง
- Download: หากยังยืนยันไม่ครบจะดาวน์โหลด draft นำกลับมาเปิดด้วย Load saved review ได้
- ยืนยันครบแล้ว Download จะได้ reviewed_truth.json บันทึกไว้ในโฟลเดอร์ผลลัพธ์นี้ การแก้กล่องจะยกเลิกการยืนยันภาพนั้นจนกว่าจะยืนยันอีกครั้ง

ตรวจจริงทุกภาพ อย่ากดยืนยันอัตโนมัติ เพราะโมเดลอาจพลาดพร้อมกันและกล่องรวมไม่ใช่คำตอบจริงโดยตัวมันเอง

## 3. วัด Accuracy จาก predictions ที่บันทึกไว้

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py `
  --mode ensemble-score `
  --output benchmark/model-results/ensemble-20frames `
  --truth-file benchmark/model-results/ensemble-20frames/reviewed_truth.json
```

ไม่รัน GPU ซ้ำ สร้าง accuracy.html, accuracy.csv, accuracy.json แยก Car/Motorcycle และ Day/Night/All พร้อมอันดับ mAP50:95

Precision/Recall ใช้ confidence ตาม Config และ IoU 0.5 AP ใช้ confidence ไล่ลำดับ, 101 recall points และ IoU 0.50 ถึง 0.95 ก้าวละ 0.05 ไม่มี crowd/ignore/area logic ของ COCO เต็มรูปแบบ คลาสที่ไม่มี GT ไม่รวมใน mAP และรายงาน AP เป็น null การเก็บ predictions จำกัดที่ confidence 0.001 และ max_det ใน Config

ข้อมูลนี้เป็น human-reviewed model-assisted ground truth สำหรับภาพที่เลือก การทำ annotation อิสระจาก predictions เหมาะกว่าสำหรับข้อสรุปทางวิชาการสุดท้าย
