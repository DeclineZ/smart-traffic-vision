# Count test: East/West กลางวันและกลางคืน วิดีโอละ 5 ภาพ

ใช้ `models_count_20frames.yaml` แยกจาก Config เดิม มี Engine 8 รุ่น: YOLOv8 s/m, YOLO11 s/m, YOLO26 s/m, YOLOv7 และ YOLOv7-X

เลือก 4 วิดีโอ day_east, day_west, night_east, night_west วิดีโอละ frame 450, 1350, 2250, 3150, 4050 (ประมาณ 3, 9, 15, 21, 27 วินาที ที่ 150 FPS) รวม 20 ภาพ แต่ละภาพมี ID แยก เช่น day_east_f450

## 1. ดูภาพและกรอกจำนวนจริง

ภาพถูกเตรียมใน `benchmark/model-results/count-20frames-east-west` เปิดภาพ PNG แล้วแก้ `selected_frames.json` ในโฟลเดอร์เดียวกัน เฉพาะ truth ของแต่ละภาพ:

```json
"truth": {"car": 12, "motorcycle": 3}
```

12 และ 3 เป็นตัวอย่างเท่านั้น กรอกจำนวนจริงของคุณครบ 20 ภาพ ใช้ 0 เฉพาะเมื่อไม่มีวัตถุคลาสนั้นจริง ห้ามเปลี่ยน video, frame, path หรือ Hash โปรแกรมตรวจภาพ PNG และเฟรมที่ Decode ตรงกับ Hash ก่อนเปรียบเทียบ

ไม่ต้องแก้ truth ใน YAML เพราะการรันนี้รับจำนวนจริงจาก JSON ด้วย --truth-file

## 2. รัน Engine ทั้ง 8 รุ่น

```powershell
cd "C:\Sea traffic\traffic"
.\.venv\Scripts\python.exe benchmark/model_benchmark.py `
  --config benchmark/models_count_20frames.yaml `
  --mode count-check `
  --videos day_east day_west night_east night_west `
  --truth-file benchmark/model-results/count-20frames-east-west/selected_frames.json `
  --output benchmark/model-results/count-20frames-all-engines
```

ได้ 8 models x 20 images = 160 คู่โมเดล/ภาพ ที่ threshold หลัก มี confidence sweep 0.10, 0.25, 0.50, 0.75 ด้วย แต่ไม่นับเป็นภาพทดสอบใหม่ ไม่ต้อง repeats หลายรอบสำหรับ count error

เปิด `benchmark/model-results/count-20frames-all-engines/index.html` ดูภาพ bounding box, counts รายภาพ, MAE รถและมอเตอร์ไซค์, WAPE และอันดับ lowest_count_MAE แยก Day/Night/All จำนวนตรงกันไม่ได้รับประกันว่าตรวจถูกทุกกล่อง จึงไม่ใช่ mAP

## เปลี่ยนภาพที่เลือก

แก้ `count_frames` ของแต่ละวิดีโอใน YAML แล้ว Prepare ลงโฟลเดอร์ใหม่:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --config benchmark/models_count_20frames.yaml --mode prepare --videos day_east day_west night_east night_west --output benchmark/model-results/count-20frames-v2
```

กรอก truth ใน JSON ใหม่และใช้ --truth-file ชี้ไฟล์ใหม่นั้น รูปแบบนี้ยังรองรับ Config เดิมที่มี frame เดียวต่อวิดีโอ
