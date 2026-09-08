# คู่มือตั้งค่า เพิ่มโมเดล และรัน YOLO Benchmark ทีละขั้นตอน

สำหรับชุด **YOLOv8s/m, YOLO11s/m, YOLO26s/m** ที่มี Engine คู่กัน อ่าน [คู่มือชุด S/M และ Config เฉพาะ](YOLO_S_M_SETUP_TH.md) และเพิ่ม `--config benchmark/models_yolo_s_m.yaml` ในคำสั่งรัน

คู่มือนี้ใช้กับโปรเจกต์ `C:\Sea traffic\traffic` บน Windows / PowerShell และคำสั่ง `benchmark/model_benchmark.py`

เริ่มจากขั้นตอน 1–4 เพื่อเตรียมระบบและโมเดล จากนั้นเลือกโหมดที่ต้องการ โดย **Performance ไม่ต้องมีจำนวนรถจริง** ส่วน **Count-check ต้องเลือกภาพและนับจำนวนจริงก่อน**

## สารบัญ

1. [เตรียมข้อมูลและเข้าโฟลเดอร์โปรเจกต์](#step-1)
2. [ตรวจ Environment](#step-2)
3. [เพิ่มโมเดล .pt และ .engine](#step-3)
4. [ตั้งค่าวิดีโอกลางวันและกลางคืน](#step-4)
5. [ตรวจโมเดลด้วย Validate](#step-5)
6. [รัน Performance และ Heat map](#step-6)
7. [เลือกเฟรมและเตรียมจำนวนจริง](#step-7)
8. [รัน Count-check](#step-8)
9. [รันทั้งสองโหมดด้วย All](#step-9)
10. [เปิดรายงานและอ่านตัวชี้วัด](#step-10)
11. [แก้ปัญหาและทดสอบระบบ](#step-11)
12. [สรุปคำสั่งที่ใช้บ่อย](#step-12)

<a id="step-1"></a>
## 1. เตรียมข้อมูลและเข้าโฟลเดอร์โปรเจกต์

| ข้อมูล | Performance | Count-check |
|---|---|---|
| โมเดล `.pt` หรือ `.engine` | ต้องมี | ใช้โมเดลเดียวกัน |
| วิดีโอกลางวัน 4 คลิป | ต้องมีเมื่อเลือก `day` หรือ `all` | ใช้เลือกภาพ |
| วิดีโอกลางคืน 4 คลิป | ต้องมีเมื่อเลือก `night` หรือ `all` | ใช้เลือกภาพ |
| ช่วงเวลา 30 วินาทีของแต่ละคลิป | ต้องเลือก | ภาพที่เลือกต้องอยู่ในช่วงนี้ |
| ภาพที่เลือกเอง 1 ภาพต่อคลิป | ไม่ต้องมี | ต้องมี |
| จำนวนรถยนต์และมอเตอร์ไซค์จริงในภาพ | ไม่ต้องมี | ต้องกรอก |
| กล่อง Ground truth | ไม่ต้องมี | ไม่ต้องมีสำหรับการวัด Count error |

เปิด PowerShell แล้วรัน:

```powershell
cd "C:\Sea traffic\traffic"
```

**ทุกคำสั่งในคู่มือนี้ให้รันจากโฟลเดอร์นี้** ไม่ต้อง Activate Environment เพราะเรียก Python ใน `.venv` โดยตรง

ไฟล์สำคัญ:

```text
traffic/
├── .venv/Scripts/python.exe
├── models/                         ← วางโมเดลที่เพิ่มเองได้ที่นี่
├── videos/                         ← วิดีโอกลางวัน
├── videos_night/                   ← วิดีโอกลางคืน
└── benchmark/
    ├── models.yaml                 ← ตั้งค่ารายการโมเดลและวิดีโอ
    ├── model_benchmark.py           ← คำสั่งรัน
    ├── requirements-benchmark.txt
    └── model-results/              ← ผลลัพธ์แต่ละรัน
```

<a id="step-2"></a>
## 2. ตรวจ Environment

### 2.1 ใช้ Environment ที่มีในโปรเจกต์ก่อน

```powershell
Test-Path .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe --version
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"
```

ค่าที่ต้องตรวจ:

- `Test-Path` เป็น `True`
- `nvidia-smi` แสดง GPU NVIDIA และ Driver
- `CUDA available` เป็น `True`
- หากต้องใช้ `.engine` ให้ตรวจ TensorRT เพิ่ม:

```powershell
.\.venv\Scripts\python.exe -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
```

Environment ที่เคยทดสอบกับเครื่องนี้: Python 3.11, PyTorch `2.9.1+cu126`, Ultralytics `8.3.240`, TensorRT `10.14.1.48.post1` และ RTX 4060

ตัวเลข CUDA จาก `nvidia-smi` คือความสามารถของ Driver ส่วน `torch.version.cuda` คือ CUDA runtime ของ PyTorch จึงไม่จำเป็นต้องแสดงเลขเดียวกัน

### 2.2 หากไม่มี `.venv`

ต้องมี Python 3.11 และ NVIDIA Driver ที่รองรับก่อน จากนั้นสร้าง Environment **เฉพาะเมื่อยังไม่มี `.venv`**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r benchmark/requirements-benchmark.txt
```

สำหรับ Engine ที่ใช้ Environment รุ่นเดียวกับเครื่องพัฒนา:

```powershell
.\.venv\Scripts\python.exe -m pip install tensorrt-cu12==10.14.1.48.post1
```

ชุดคำสั่งนี้ระบุรุ่นที่เคยทดสอบ ไม่ได้ทำให้ Engine ทุกไฟล์เข้ากันได้อัตโนมัติ หาก Engine มาจาก GPU/TensorRT คนละรุ่น ให้ใช้ Environment ที่ตรงกับการ Export หรือสร้าง Engine ใหม่จากต้นฉบับภายนอกเครื่องมือ Benchmark นี้

ติดตั้งเสร็จให้ตรวจตามข้อ 2.1 อีกครั้ง ไม่ต้องติดตั้งซ้ำหาก Environment เดิมทำงานได้แล้ว

<a id="step-3"></a>
## 3. เพิ่มโมเดล `.pt` และ `.engine`

### 3.1 วางไฟล์โมเดล

ตัวอย่าง:

```text
traffic/models/my_model.pt
traffic/models/my_model.engine
```

ไม่จำเป็นต้องมีทั้งคู่ หากมีเฉพาะ `.pt` หรือเฉพาะ `.engine` ให้เพิ่มเฉพาะรายการที่มีจริง

### 3.2 เพิ่มรายการใน YAML

เปิด [models.yaml](models.yaml) แล้วเพิ่มรายการใต้ `models:` ที่มีอยู่ โดยจัดระดับช่องว่างให้เหมือนรายการเดิม **อย่าเพิ่มหัวข้อ `models:` ซ้ำอีกหัวข้อ**

ตัวอย่างด้านล่างเป็นโมเดล Custom ที่ใช้ Class 0 = รถยนต์ และ Class 1 = มอเตอร์ไซค์ ต้องเปลี่ยนเลขให้ตรงกับโมเดลของคุณ:

```yaml
  - id: my_model_pt
    name: My model PyTorch
    version: custom-v1
    path: ../models/my_model.pt
    backend: ultralytics
    pair_id: my_model
    precision: fp32
    input_size: 640
    classes: {car: 0, motorcycle: 1}

  - id: my_model_engine
    name: My model TensorRT
    version: custom-v1
    path: ../models/my_model.engine
    backend: ultralytics
    pair_id: my_model
    precision: fp16
    input_size: 640
    classes: {car: 0, motorcycle: 1}
```

| ค่า | วิธีกรอก |
|---|---|
| `id` | ต้องไม่ซ้ำ ใช้ตัวอักษรอังกฤษ ตัวเลข `_` หรือ `-` ใช้ ID นี้กับ `--models` |
| `name` | ชื่อที่ต้องการแสดงในรายงาน |
| `version` | รุ่นโมเดลหรือรุ่นการ Train เช่น `YOLOv8n` หรือ `custom-v1` |
| `path` | อ้างอิงจากโฟลเดอร์ที่เก็บ YAML ไม่ใช่จาก PowerShell |
| `backend` | เวอร์ชันนี้ใช้ `ultralytics` |
| `pair_id` | ค่าเดียวกันเฉพาะ `.pt/.engine` ที่มาจากน้ำหนักเดียวกัน ไม่จำเป็นต้องกรอกสำหรับโมเดลที่ไม่ต้องการจับคู่ |
| `precision` | `.pt` ใช้ `fp32` หรือ `fp16`; `.engine` ต้องตรงกับค่าที่ Export เช่น `fp32`, `fp16`, `int8` |
| `input_size` | ใช้ `640` |
| `classes` | Class ID จริงของรถยนต์และมอเตอร์ไซค์ |
| `parameters` | ระบุได้ถ้าทราบ โดยเฉพาะ Engine ที่ไม่มี `.pt` คู่กัน หากไม่ทราบให้เว้นไว้ |

ตัวอย่างการแปลเส้นทาง:

| ไฟล์จริง | `path` เมื่อ YAML อยู่ใน `traffic/benchmark` |
|---|---|
| `traffic/yolov8n.pt` | `../yolov8n.pt` |
| `traffic/models/my_model.pt` | `../models/my_model.pt` |
| `traffic/models/my_model.engine` | `../models/my_model.engine` |

โมเดล COCO โดยทั่วไปใช้:

```yaml
    classes: {car: 2, motorcycle: 3}
```

ตรวจชื่อคลาสของ `.pt` ที่เพิ่มได้ด้วย:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; print(YOLO('models/my_model.pt').names)"
```

การเปลี่ยน `precision` ใน YAML ไม่ใช่การแปลง Engine จาก FP32 เป็น FP16 ตัวอย่างโมเดล Engine เดิมของโปรเจกต์เป็น FP32 จึงไม่ควรเปลี่ยนตามตัวอย่าง FP16 โดยไม่ได้ Export ใหม่

### 3.3 ตั้งโมเดลอ้างอิงสำหรับ Difference map

เลือก ID ที่มีอยู่ใน `models:` เช่น:

```yaml
reference_model: v8n_pt
```

ถ้าเอาโมเดลเดิมออกจากรายการ ให้เปลี่ยน `reference_model` ด้วย หากรันเฉพาะบางโมเดลด้วย `--models` และไม่ได้เลือกโมเดลอ้างอิง ระบบจะข้าม Difference map พร้อมแจ้งเหตุผล แต่ยังสร้าง Heat map ปกติได้

### 3.4 ขอบเขตการรองรับ

รองรับโมเดล Detection ที่ Ultralytics โหลดได้ และ Engine แบบ Ultralytics export ที่มี Metadata โดย Engine ต้องรองรับ Batch 1 และ 640×640 หากเป็น Dynamic engine ช่วง Profile ต้องครอบคลุมอินพุตนี้

Legacy YOLOv7, Custom raw engines, Segmentation และ OBB อาจต้องเพิ่ม Adapter ไม่สามารถรับประกันว่าไฟล์ทุกชนิดที่ลงท้าย `.pt/.engine` จะเปิดได้ เครื่องมือนี้ไม่ดาวน์โหลดโมเดลหรือสร้าง Engine อัตโนมัติ

<a id="step-4"></a>
## 4. ตั้งค่าวิดีโอกลางวันและกลางคืน

Config ต้องมี 4 กล้องในกลุ่ม `day` และ 4 กล้องในกลุ่ม `night` โดยใช้ Camera ID ชุดเดียวกัน รายการเริ่มต้นมีดังนี้:

| Camera | Video ID กลางวัน | Video ID กลางคืน | ชื่อไฟล์ |
|---|---|---|---|
| north | `day_north` | `night_north` | `cam44_north.avi` |
| south | `day_south` | `night_south` | `cam43_south.avi` |
| east | `day_east` | `night_east` | `cam03_east.avi` |
| west | `day_west` | `night_west` | `cam46_west.avi` |

กลางวันอยู่ใน `videos/` กลางคืนอยู่ใน `videos_night/` สามารถเปลี่ยนไฟล์ในรายการได้ แต่เวอร์ชันนี้ยังไม่รองรับเพิ่มเกิน 4 คลิปต่อกลุ่ม

ตัวอย่างรายการวิดีโอแบบหลายบรรทัด ใช้แทนรายการเดิมที่มี ID เดียวกัน:

```yaml
  - id: day_north
    camera: north
    group: day
    path: ../videos/cam44_north.avi
    start_seconds: 60
    frame: null
    truth: {car: null, motorcycle: null}
```

รายการนี้เลือกช่วงวินาทีที่ **60–90** ของไฟล์ต้นฉบับ ทุกโมเดลใช้ช่วงเดียวกัน

- `start_seconds: 0` = เริ่มจากต้นไฟล์
- ความยาวแต่ละช่วงเป็น 30 วินาที ไม่ต้องตัดหรือเขียนวิดีโอใหม่
- คงความละเอียดและ FPS ต้นฉบับ จนถึงขั้นตอน Letterbox เข้าโมเดล
- ถ้าไฟล์รายงาน 150 FPS จะได้ 4,500 เฟรมต่อช่วง 30 วินาที ระบบไม่เปลี่ยนเป็น 30 FPS ให้เอง
- สำหรับ Performance สามารถเว้น `frame` และ `truth` เป็น `null` ได้

ค่าหลักใต้ `settings:` คือ Confidence 0.25, NMS IoU 0.70, Warm-up 30 ครั้ง และ 3 รอบวัดจริง ควรใช้ค่าเดียวกันในการเปรียบเทียบทุกโมเดล ขนาดภาพ 640×640, Batch 1 และ Square letterbox เป็นเงื่อนไขของเครื่องมือนี้

หากต้องการสรุปตามเขตถนน สามารถเพิ่ม `regions` ภายในรายการวิดีโอได้ เช่นภาพ 1920×1080:

```yaml
    regions:
      - id: near_road
        points: [[0, 700], [1920, 700], [1920, 1080], [0, 1080]]
```

ใช้พิกัด Pixel ของภาพต้นฉบับ และปรับ Polygon ให้ตรงกับถนนจริงของกล้องนั้น ข้อมูลนี้ไม่บังคับ

<a id="step-5"></a>
## 5. ตรวจโมเดลด้วย Validate

ตรวจทุกรายการใน Config:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode validate
```

ตรวจเฉพาะโมเดลที่เพิ่มเอง:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode validate --models my_model_pt my_model_engine
```

ต้องเพิ่ม ID เหล่านี้ตามขั้นตอน 3 ก่อนใช้คำสั่ง หากใช้เฉพาะโมเดลเดิม ให้แทนด้วย `v8n_pt v8n_engine` เป็นต้น

Validate ตรวจข้อมูลวิดีโอและทดลอง Predict หนึ่งภาพกับแต่ละโมเดล เพื่อตรวจโหลดไฟล์ ขนาดอินพุต Batch และผลลัพธ์ **ไม่ใช่การ Benchmark เต็ม** หากมีรายการล้มเหลว ให้ดู `worker.log` และ `failed.json` ในโฟลเดอร์ผลลัพธ์ก่อนรันต่อ

<a id="step-6"></a>
## 6. โหมด Performance: ความเร็ว ทรัพยากร และ Heat map

### 6.1 ทดสอบสั้นก่อน

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --smoke
```

Smoke ใช้ 8 เฟรมต่อคลิป, Warm-up 3 ครั้ง และ 1 รอบ ผลติดป้าย `SMOKE TEST` ใช้ตรวจว่าโปรแกรมทำงานได้เท่านั้น ห้ามนำไปสรุปอันดับประสิทธิภาพจริง

### 6.2 รันเต็มทุกโมเดลและทั้งสองกลุ่ม

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance
```

เมื่อใช้ค่าเริ่มต้น 4 รายการโมเดล จะมี 4 โมเดล × 8 คลิป × 3 รอบ = **96 model/video runs** แต่ละรอบอ่านทุกเฟรมในช่วง 30 วินาที ไม่มีการข้ามเฟรม

ระบบรันโมเดลเรียงกันและสลับลำดับระหว่างรอบ ไม่ใช่การจำลอง 8 กล้องประมวลผลพร้อมกัน

### 6.3 เลือกเฉพาะกลุ่มหรือโมเดล

เฉพาะกลางวัน:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --group day
```

เฉพาะกลางคืนและคู่โมเดลที่ต้องการ:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --group night --models v8n_pt v8n_engine
```

กำหนดชื่อโฟลเดอร์ผลลัพธ์เอง:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --output benchmark/model-results/my-first-full-run
```

`--output` ต้องเป็นโฟลเดอร์ใหม่ หากรันครั้งถัดไปให้ใช้ชื่อใหม่หรือไม่ระบุ เพื่อให้ระบบตั้งชื่อตามวันและเวลา

<a id="step-7"></a>
## 7. เลือกเฟรมและเตรียมจำนวนจริงสำหรับ Count-check

### 7.1 เลือกภาพเอง

เลือกหนึ่งเฟรมจากแต่ละคลิปที่จะทดสอบ รวม 8 ภาพสำหรับ `all` หรือ 4 ภาพสำหรับ `day`/`night`

`frame` คือเลขเฟรมของ **ไฟล์ต้นฉบับ เริ่มจาก 0** ไม่ใช่เลขเฟรมที่เริ่มนับใหม่หลัง `start_seconds` และภาพต้องอยู่ในช่วง 30 วินาทีที่เลือก

ตัวอย่างสำหรับไฟล์ที่รายงาน 150 FPS:

| ช่วงที่เลือก | ช่วงเลขเฟรมที่ใช้ได้ | ตัวอย่างเฟรมกึ่งกลาง |
|---|---|---:|
| 0–30 วินาที | 0–4499 | 2250 |
| 60–90 วินาที | 9000–13499 | 11250 |

ตัวเลขนี้ใช้เฉพาะตัวอย่าง 150 FPS ตรวจ FPS จริงของแต่ละไฟล์จาก `dataset.json` ที่ได้หลัง Validate หรือ Smoke ก่อนเลือกเลข

### 7.2 ทดลองดึงภาพโดยไม่แก้ YAML

สำหรับคลิปที่เริ่มวินาที 0 และมีเฟรม 2250 อยู่ในช่วงที่เลือก:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode prepare --frame day_north=2250
```

เลือกหลายภาพในคำสั่งเดียวได้ ตัวอย่างต่อไปนี้ใช้ได้เมื่อเลขเฟรมอยู่ในช่วงของแต่ละคลิป:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode prepare --frame day_north=2250 --frame night_north=2250
```

คำสั่งจะแสดงโฟลเดอร์ที่บันทึก PNG เปิดภาพเพื่อตรวจว่าเป็นจังหวะที่ต้องการ หากไม่ใช่ให้เลือกเลขใหม่แล้วรันอีกครั้ง ไม่มีหน้าต่างเลือกเฟรมหรือ Slider ในตัว Runner

`--frame` เปลี่ยนค่าเฉพาะการเรียกครั้งนั้น **ไม่เขียนกลับ YAML** หลังเลือกเสร็จต้องกรอกเลขเดียวกันใน Config

Prepare จะสร้างภาพสำหรับทุกรายการในกลุ่มที่มี `frame` กรอกแล้ว รวมทั้งค่าที่ Override ด้วย `--frame` หากต้องการจำกัด Day/Night ให้เพิ่ม `--group`

### 7.3 กรอกเลขเฟรมแล้วสร้างภาพครบกลุ่ม

แก้แต่ละรายการใน YAML เช่น:

```yaml
    start_seconds: 0
    frame: 2250
    truth: {car: null, motorcycle: null}
```

เมื่อเลือกครบแล้ว:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode prepare
```

ไฟล์ภาพจะมีชื่อ เช่น `day_north_frame_2250.png` และมี `selected_frames.json` ระบุแหล่งที่มา หมายเลขเฟรม และ Hash

### 7.4 นับจำนวนจริงจาก PNG

เปิดภาพที่ระบบสร้าง แล้วนับรถยนต์กับรถจักรยานยนต์ที่เห็นจริง กรอกตัวเลขลง YAML เช่น:

```yaml
    frame: 2250
    truth: {car: 10, motorcycle: 3}
```

**10 และ 3 เป็นตัวอย่าง ไม่ใช่จำนวนจริงของวิดีโอในโปรเจกต์** ต้องแทนด้วยจำนวนที่คุณนับเอง

ใช้กติกาเดียวกันทุกภาพ:

- นับรถแต่ละคันครั้งเดียวในภาพนั้น
- นับรถที่ถูกบังบางส่วนเมื่อยังระบุได้ว่าเป็นรถคลาสใด
- มอเตอร์ไซค์กับผู้ขับขี่เป็นมอเตอร์ไซค์หนึ่งคัน
- ไม่รวมรถบัส รถบรรทุก จักรยาน หรือคนในสองคลาสนี้
- หากภาพกำกวมมาก ควรเลือกภาพใหม่ก่อนดูผลของโมเดล หรือบันทึกกติกาการนับให้ชัดเจน
- `0` หมายถึงตรวจแล้วว่าไม่มีรถคลาสนั้นจริง ส่วน `null` หมายถึงยังไม่ได้กรอก

ทางเลือกเพิ่มเติม: คัดลอก `frame_sha256` จาก `selected_frames.json` มาใส่ `truth_frame_sha256` ในรายการวิดีโอ เพื่อให้ระบบตรวจว่าภาพยังตรงกับตอนนับ ห้ามใช้ Hash ของไฟล์ PNG แทน เพราะค่าที่ต้องการเป็น Hash ของ Pixel ที่ถอดรหัสแล้ว

<a id="step-8"></a>
## 8. โหมด Count-check: เปรียบเทียบจำนวนรถ

เมื่อ Frame และ Truth ครบทั้ง 8 คลิป:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode count-check
```

หากเตรียมเฉพาะกลางวันครบ 4 คลิป:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode count-check --group day
```

เฉพาะกลางคืนและบางโมเดล:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode count-check --group night --models v8n_pt v8n_engine
```

ทุกโมเดลตรวจภาพเดียวกัน ระบบสร้างภาพกล่องตรวจจับ ชื่อคลาส Confidence และจำนวนที่พบเทียบกับจำนวนจริง

ใช้ Confidence ใน `settings.conf` เป็นผลหลัก โดยค่าเริ่มต้นคือ 0.25 พร้อมสำรวจ 0.10, 0.25, 0.50 และ 0.75 สำหรับโมเดลที่ปรับได้ Engine ที่ฝัง NMS จะมีข้อจำกัดในการปรับ Threshold และแยกออกจากอันดับที่ต้องใช้เงื่อนไขตรงกัน

### สูตร Count error

| ตัวชี้วัด | ความหมาย |
|---|---|
| Signed error | จำนวนที่โมเดลนับ − จำนวนจริง; ติดลบคือนับขาด บวกคือนับเกิน |
| Absolute error | ค่าสัมบูรณ์ของ Signed error |
| MAE | ค่าเฉลี่ย Absolute error ต่อภาพ |
| Bias | ค่าเฉลี่ย Signed error |
| Exact-count match rate | สัดส่วนภาพที่นับตรงทั้งหมดสำหรับคลาสนั้น |
| WAPE | ผลรวม Absolute error ÷ ผลรวมจำนวนจริง × 100 |

คำนวณแยกรถยนต์และมอเตอร์ไซค์เสมอ เช่น รถยนต์จริง 10 ตรวจได้ 8 และมอเตอร์ไซค์จริง 3 ตรวจได้ 5 จะมี Absolute error รวม 2 + 2 = **4** ไม่ใช่ 0 แม้จำนวนรถรวมจะเท่ากัน

อันดับ Count error ใช้ค่าเฉลี่ยของผลรวม Absolute error สองคลาสต่อภาพ ค่าน้อยกว่าดีกว่า และค่าที่เท่ากันให้ครองอันดับร่วม หากผลรวมจำนวนจริงของคลาสใดเป็น 0 ให้ WAPE เป็น `N/A` แต่ยังรายงาน MAE ได้

จำนวนที่ตรงกันไม่ได้แปลว่าทุกกล่องถูกต้อง โมเดลอาจพลาดรถจริงหนึ่งคันและตรวจผิดเพิ่มหนึ่งคัน จึงไม่มี Precision, Recall หรือ mAP จากข้อมูลจำนวนอย่างเดียว ภาพ 8 ภาพเป็น Spot check ไม่ใช่ชุดประเมินความแม่นยำขนาดใหญ่

<a id="step-9"></a>
## 9. รันทั้งสองโหมดด้วย All

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode all
```

หรือเลือกเฉพาะกลางวันและคู่โมเดล:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode all --group day --models v8n_pt v8n_engine
```

หาก Frame/Truth ยังไม่ครบกลุ่มที่เลือก ระบบจะข้าม Count-check ทั้งกลุ่มพร้อมบอกว่าคลิปใดยังขาด และยังรัน Performance ได้ ไม่มีการเติมจำนวนจริงให้อัตโนมัติ

การใช้ `--smoke` กับ `all` จะลดงาน Performance และ Warm-up แต่ยังต้องกรอก Frame/Truth สำหรับ Count-check และไม่เปลี่ยนภาพที่คุณเลือก

<a id="step-10"></a>
## 10. เปิดรายงานและอ่านตัวชี้วัด

### 10.1 หาโฟลเดอร์ผลลัพธ์

เมื่อไม่กำหนด `--output` จะบันทึกใน:

```text
traffic/benchmark/model-results/<วันและเวลา>/
```

โปรแกรมแสดงตำแหน่ง `index.html` เมื่อเสร็จ เปิดไฟล์นี้ด้วย Browser ได้โดยตรงแบบออฟไลน์

| ไฟล์ | ใช้ดูอะไร |
|---|---|
| `index.html` | รายงานหลัก ตาราง ภาพ และตัวกรองภาพตามโมเดล/กล้อง/Day-Night/ชนิดภาพ |
| `REPORT.md` | สรุปวิธีวัด เงื่อนไข สถานะ และตาราง |
| `comparison.csv` | ผลรวมและค่าเฉลี่ยต่อโมเดล แยก Day/Night/รวม |
| `per_video_pass.csv` | รายละเอียดทุกคลิปและทุกรอบ |
| `per_video_average.csv` | ค่าเฉลี่ยแต่ละคลิปจากรอบที่วัด |
| `count_per_frame.csv` | จำนวนจริง ผลตรวจ และ Count error ของภาพที่เลือก |
| `count_summary.csv` | Count error แยกคลาส/กลุ่ม/Confidence |
| `paired_comparison.csv` | Speedup และความต่างของคู่ `.pt/.engine` |
| `export_agreement.csv` | ความสอดคล้องของกล่องระหว่างไฟล์คู่กัน |
| `rankings.csv` | อันดับแยกตามตัวชี้วัดและกลุ่ม Precision |
| `regions.csv` | ผลตาม Polygon ถนน ถ้ามีการตั้งค่า |
| `contact_*.png` | ภาพเปรียบเทียบ Heat map หลายโมเดลบนคลิปเดียวกัน |
| `resolved_config.json` | Config ที่ใช้จริง รวมค่า Override |
| `dataset.json` | FPS ความละเอียด ช่วงเฟรม และ Hash ของวิดีโอ |
| `run.json` | ลำดับการรัน สถานะ และข้อผิดพลาด |
| `requirements-lock.txt` | รุ่นแพ็กเกจของ Environment ที่รัน |

ไฟล์ CSV บางรายการจะไม่มีหากโหมดนั้นไม่ได้รันหรือไม่มีผลสำเร็จ

ภายในผลของแต่ละโมเดล/รอบ มี Environment, Artifact metadata และ Log ส่วนภายในแต่ละคลิปมี `frames.npz`, `telemetry.csv`, `timeline.png` และในรอบแรกมี `heat.npz`, `predictions.npz`, `background.png`

### 10.2 ตัวชี้วัด Performance

| ตัวชี้วัด | วิธีอ่าน |
|---|---|
| Frames | จำนวนที่อ่านและตรวจจริง; `frames_all_passes` รวมการทดสอบซ้ำ |
| Decode time | เวลาถอดรหัสวิดีโอ แสดงแยกจากการตรวจจับ |
| Preprocess time | Letterbox/แปลงข้อมูล/ส่งอินพุตเข้า GPU |
| Inference time/FPS | เวลาทำ Model forward และอัตราเฟรมจากเวลานี้ |
| Postprocess time | ขั้นตอนหลัง Forward รวม NMS ภายนอกโมเดลตาม Backend |
| Detection time/FPS | ครบขั้นตอนตรวจจับ รวมส่งกล่องกลับ CPU แต่ไม่รวม Decode |
| P95/P99 | ค่า Latency ที่ 95%/99% ของเฟรมไม่เกินค่านี้ ใช้ดูการกระตุก |
| Pass FPS standard deviation | ความแปรปรวนของ FPS ระหว่างรอบ; Smoke รอบเดียวไม่มีค่านี้ |
| Budget exceed percentage | สัดส่วนเฟรมที่ตรวจจับเกินงบเวลาเป้าหมาย เช่น 33.3 ms ที่ 30 FPS |
| CPU/RAM/GPU utilization | ทรัพยากรที่ Sample ในช่วง Measurement loop |
| VRAM | ค่าก่อนโหลด หลังโหลด หลัง Warm-up และ Peak ที่ Sample ได้ |
| Detection counts | จำนวนกล่องสะสมตามเฟรม แยกรถยนต์/มอเตอร์ไซค์ ไม่ใช่จำนวนรถไม่ซ้ำ |
| File size/Parameters | ขนาดไฟล์และจำนวน Parameters; Engine อาจอ้างอิง `.pt` คู่กันหรือข้อมูลที่กรอก |

FPS รวมคำนวณจากจำนวนเฟรมทั้งหมดหารเวลาทั้งหมด ไม่ใช่เฉลี่ย FPS ตรง ๆ เวลาโหลดโมเดล Warm-up Hash การสร้างภาพและเขียนไฟล์ไม่รวมใน Inference/Detection time

อันดับความเร็วหลักใช้ Detection FPS ส่วน Forward FPS เป็นข้อมูลแยกต่างหาก Engine ที่ฝัง NMS มีขอบเขตงาน Forward ต่างกัน จึงไม่ควรเทียบตรง ๆ กับ Forward ที่ไม่มี NMS

VRAM ของ TensorRT ไม่สามารถใช้ PyTorch allocated memory แทนได้ บน Windows WDDM ค่า Process VRAM อาจเป็น `N/A` จึงมี Device VRAM และผลต่างจากก่อนโหลดเป็นข้อมูลประกอบ ซึ่งรวม Desktop/โปรแกรมอื่นด้วย Peak เป็นค่าสูงสุดที่ Sample พบและอาจไม่เห็นการเพิ่มขึ้นช่วงสั้น ๆ

CPU/RAM/GPU utilization ครอบคลุมช่วง Loop ที่มี Decode, Hash และเก็บผล ไม่ใช่เฉพาะ Model forward เพียงขั้นตอนเดียว ค่า Mean resource ใน Summary เป็นค่าเฉลี่ยของ Mean แต่ละคลิป

### 10.3 Heat map และ Difference map

- Heat map ใช้จุดกึ่งกลางด้านล่างของกล่องในพิกัดภาพต้นฉบับ
- สร้างแยกรถยนต์ มอเตอร์ไซค์ และรวม โดยใช้เฉพาะรอบแรก ไม่สะสมซ้ำสามรอบ
- ปรับค่าด้วยจำนวนเฟรม และใช้สเกลสีร่วมกันระหว่างโมเดลบนคลิป/คลาสเดียวกัน
- คลิก Heat map ในรายงานเพื่อเปิด Viewer แล้วชี้/คลิกตำแหน่งเพื่อดูจำนวนและ Density ที่ Pixel จริง ค่าที่ชี้เป็นค่าดิบ ส่วนภาพ Overlay ผ่านการ Smooth
- Difference map สีแดงหมายถึงตรวจมากกว่าโมเดลอ้างอิง สีน้ำเงินหมายถึงน้อยกว่า
- แยก Day/Night และแต่ละกล้อง ไม่รวมพิกัดคนละมุมกล้องเป็นภาพเดียว

รถจอดอาจทำให้ตำแหน่งนั้นมี Density สูง แผนที่นี้บอกว่าโมเดลตรวจพบที่ไหนบ่อย ไม่ใช่หลักฐานว่าตรวจถูกหรือพลาดที่ไหน

### 10.4 การเปรียบเทียบ `.pt/.engine`

ไฟล์ที่มี `pair_id` ตรงกันจะมี Speedup, ความต่างของหน่วยความจำ/จำนวนที่ตรวจพบ และการจับคู่กล่องแบบหนึ่งต่อหนึ่งในคลาสเดียวกันที่ IoU ≥ 0.5

การจับคู่เรียกว่า Export agreement ไม่ใช่ความแม่นยำกับ Ground truth ทั้งสองไฟล์อาจเห็นตรงกันแต่ตรวจผิดเหมือนกันได้

แยกผลที่ใช้ Precision เดียวกันกับผล Deployment ที่ Precision ต่างกัน เช่น `.pt` FP32 เทียบ `.engine` FP16 ทั้งนี้ Precision ที่ประกาศไม่ได้ยืนยันรูปแบบคำนวณทุก Kernel: Tactics/TF32 ใน Engine ที่ Compile แล้วเปลี่ยนย้อนหลังผ่าน YAML ไม่ได้

ระบบแยกอันดับความเร็ว VRAM CPU RAM และ Count error โดยไม่มีคะแนนผู้ชนะความเร็ว/ความแม่นยำรวมค่าเดียว

<a id="step-11"></a>
## 11. แก้ปัญหาและทดสอบระบบ

### 11.1 ปัญหาที่พบบ่อย

| อาการ | สิ่งที่ควรตรวจ |
|---|---|
| หา Python ไม่พบ | เข้า `C:\Sea traffic\traffic` และตรวจ `.venv/Scripts/python.exe` |
| `CUDA available: False` | ตรวจ NVIDIA Driver และ PyTorch แบบ CUDA; Runner ไม่สลับไป CPU อัตโนมัติ |
| โมเดลไม่อยู่ในรายการ | ใช้ค่า `id` กับ `--models` ไม่ใช่ชื่อไฟล์ และเพิ่มรายการใน YAML ก่อน |
| หาไฟล์ไม่พบ | `path` ใน YAML อ้างอิงจากที่อยู่ YAML |
| YAML อ่านไม่ได้ | ใช้ Space จัดระดับให้ตรง ห้ามมี `models:`/`videos:` ซ้ำหลายหัวข้อ |
| Engine โหลดไม่ได้ | ตรวจ GPU/TensorRT/Metadata ของ Engine และดู `worker.log` |
| Precision mismatch | กรอก Precision ให้ตรงกับ Export จริง ไม่ใช่ค่าที่อยากให้ Engine ใช้ |
| Fixed engine batch ไม่ใช่ 1 | ใช้ Engine Batch 1 หรือ Dynamic profile ที่รองรับ Batch 1 |
| Count-check unavailable | กรอก `frame` และ `truth` ให้ครบทุกคลิปในกลุ่มที่เลือก |
| เลขเฟรมอยู่นอกช่วง | ใช้เลขต้นฉบับเริ่มจาก 0 และตรวจ `start_seconds`/FPS จาก `dataset.json` |
| Annotated image hash ไม่ตรง | ตรวจไฟล์และเฟรมใหม่ ถ้าจะเปลี่ยนภาพต้องนับและอัปเดต Truth/Hash ใหม่ |
| ไม่มี Difference map | ตรวจว่า `reference_model` อยู่ในรายการและถูกเลือกใน `--models` |
| Output directory exists | เปลี่ยนชื่อโฟลเดอร์ หรือไม่ระบุ `--output` |
| บางโมเดลไม่มีอันดับ | ดู `complete`, `comparable`, `run.json` และ `failed.json`; ผลที่ไม่ครบหรือเงื่อนไขต่างกันถูกตัดออก |

### 11.2 หยุดและสร้างรายงานใหม่

กด `Ctrl+C` เพื่อหยุด Worker ระบบเก็บ Checkpoint ของคลิปที่เสร็จแล้ว แต่ **ไม่ Resume Inference อัตโนมัติ** หากต้องการรันซ้ำให้เลือกโมเดลแล้วใช้โฟลเดอร์ผลลัพธ์ใหม่

สร้างรายงานจากผลเดิมโดยไม่ทำ Inference เพิ่ม:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode report --output benchmark/model-results/my-first-full-run
```

เปลี่ยนชื่อท้ายเส้นทางให้ตรงกับโฟลเดอร์จริง โหมด `report` เป็นกรณีที่ใช้โฟลเดอร์ผลลัพธ์เดิมได้

### 11.3 รันชุดทดสอบโปรแกรม

Unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_benchmark.py -v
```

รวม GPU integration tests:

```powershell
$env:BENCHMARK_GPU_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_benchmark.py -v
Remove-Item Env:BENCHMARK_GPU_TESTS
```

Integration tests ใช้ไฟล์โมเดลตัวอย่างของโปรเจกต์ และสร้างวิดีโอสังเคราะห์ว่างที่มีจำนวนจริงเป็น 0 เพื่อทดสอบระบบ ไม่ได้สร้าง Ground truth ให้กับวิดีโอจราจรของคุณ

<a id="step-12"></a>
## 12. สรุปคำสั่งที่ใช้บ่อย

เริ่มจากโฟลเดอร์นี้เสมอ:

```powershell
cd "C:\Sea traffic\traffic"
```

| ต้องการทำอะไร | ค่าต่อท้าย `model_benchmark.py` |
|---|---|
| ตรวจว่าโมเดลโหลดได้ | `--mode validate` |
| ทดสอบระบบแบบสั้น | `--mode performance --smoke` |
| รัน Performance เต็ม | `--mode performance` |
| สร้างภาพที่เลือกไว้นับจริง | `--mode prepare` |
| ทดลองดึงภาพหนึ่งเฟรม | `--mode prepare --frame day_north=2250` |
| เปรียบเทียบกับจำนวนจริง | `--mode count-check` |
| รันทั้งสองโหมด | `--mode all` |
| เลือกเฉพาะกลางคืน | เพิ่ม `--group night` |
| เลือกเฉพาะคู่โมเดล | เพิ่ม `--models v8n_pt v8n_engine` |
| ใช้ Config อีกไฟล์ | เพิ่ม `--config benchmark/my_models.yaml` |
| ตั้งชื่อโฟลเดอร์ผลใหม่ | เพิ่ม `--output benchmark/model-results/run-02` |

ตัวอย่างคำสั่งเต็ม:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --group night --models v8n_pt v8n_engine
```

ลำดับใช้งานที่แนะนำ: **ตรวจ Environment → เพิ่มโมเดล → ตั้งช่วงวิดีโอ → Validate → Smoke → Performance เต็ม → เลือกภาพ → กรอก Truth → Count-check → เปิด `index.html`**

รายละเอียดวิธีวัดเพิ่มเติมอยู่ใน [README_MODEL_BENCHMARK_TH.md](README_MODEL_BENCHMARK_TH.md) และรายการแพ็กเกจอยู่ใน [requirements-benchmark.txt](requirements-benchmark.txt)
