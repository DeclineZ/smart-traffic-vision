# Benchmark YOLO: กลางวัน/กลางคืน, .pt/.engine

**เริ่มใช้งานทีละขั้นตอน:** อ่าน [คู่มือตั้งค่า เพิ่มโมเดล และรันทุกโหมด](SETUP_AND_RUN_TH.md) ซึ่งรวมการตรวจ Environment, ตัวอย่าง YAML สำหรับ `.pt/.engine`, ตั้งค่า Day/Night, เลือกเฟรม, กรอกจำนวนจริง, คำสั่งทุกโหมด และวิธีอ่านผลไว้ครบแล้ว

เครื่องมือนี้แยกจาก `benchmark_hardware.py` และระบบจราจรจริง ใช้การตรวจจับรถยนต์และรถจักรยานยนต์เท่านั้น ไม่มี Tracking หรือการนับรถไม่ซ้ำ

## เริ่มใช้งาน

เปิด PowerShell ที่ `C:\Sea traffic\traffic` ใช้ Python ใน `.venv` ที่มีอยู่:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode validate
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance --smoke
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode performance
```

`validate` ทดลองโหลดและ Predict หนึ่งภาพด้วยแต่ละโมเดล เพื่อตรวจ Batch 1 / 640×640 และรูปแบบผลลัพธ์ ไม่ใช่ผล Benchmark

`--smoke` ใช้เพียง 8 เฟรมต่อวิดีโอ, Warm-up 3 ครั้ง, 1 รอบ ผลติดป้าย SMOKE TEST ห้ามนำไปสรุปประสิทธิภาพจริง

รันปกติใช้ 8 คลิป × 30 วินาที × 3 รอบ × จำนวนโมเดล ค่าปริยาย 4 โมเดลรวม 96 model/video runs อาจใช้เวลานาน ข้อมูลตรวจจับรอบแรกและภาพ Heat map ใช้พื้นที่เพิ่มเติม

## การตั้งค่า

แก้ `models.yaml` โดยเส้นทางไฟล์อ้างอิงจากโฟลเดอร์ของ YAML เลือกกลุ่มด้วย `--group day`, `night` หรือ `all` เลือกโมเดลด้วย `--models v8n_pt v8n_engine` ใช้ `--output` เพื่อเลือกโฟลเดอร์ผลลัพธ์ใหม่ หากไม่ระบุจะสร้างโฟลเดอร์ตามเวลาใน `benchmark/model-results/`

โมเดลแต่ละรายการต้องมี `id`, `name`, `version`, `path`, `backend: ultralytics`, `precision`, `input_size: 640`, `classes: {car: ..., motorcycle: ...}` เพิ่ม `pair_id` เดียวกันเฉพาะ `.pt`/`.engine` ที่มาจากน้ำหนักเดียวกัน Pair ID เป็นข้อมูลที่ผู้ใช้ยืนยัน ไม่สามารถพิสูจน์ต้นกำเนิดจากชื่อไฟล์ได้

ตัวอย่างโมเดล Custom สองคลาส:

```yaml
- id: custom_pt
  name: My traffic detector
  version: custom-v1
  path: ../models/best.pt
  backend: ultralytics
  pair_id: custom_v1
  precision: fp32
  input_size: 640
  classes: {car: 0, motorcycle: 1}
```

รองรับเฉพาะ Detection ที่ Ultralytics โหลดและแปลงผลลัพธ์เป็นกล่องได้ ไฟล์ Legacy YOLOv7, Custom raw engines, Segmentation/OBB ต้องมี Adapter เพิ่ม ไม่ใช่ทุก `.pt/.engine` ที่จะเปิดได้ทันที Interface อยู่ใน `bench_runtime.DetectionAdapter` และทะเบียน `ADAPTERS` ต้องเพิ่มชื่อที่อนุญาตใน Config validator เมื่อติดตั้ง Adapter

Engine ต้องเป็น Ultralytics export ที่มี Metadata และรองรับ Batch 1, 640×640 Dynamic engine ใช้ได้หาก Profile ครอบคลุมอินพุตนี้ การรันนี้ไม่สร้าง Engine หรือดาวน์โหลดโมเดลเอง

ไฟล์ Engine ที่มีอยู่ตอนพัฒนาเป็น FP32 Dynamic export, Batch สูงสุดตาม Profile ต้องตรวจด้วย `validate` ห้ามเปลี่ยน Precision ใน YAML เพื่ออ้างว่า Engine เป็น FP16

รายการวิดีโอแบ่ง Day/Night กลุ่มละ 4 กล้อง กำหนด `start_seconds` แยกคลิปได้ ทุก Segment ยาว 30 วินาที หาก AVI รายงาน 150 FPS จะประมวลผล 4,500 เฟรมต่อ Segment ไม่เปลี่ยน FPS ให้เอง และไม่ยืนยันว่า FPS ในไฟล์ตรงกับจังหวะจับภาพของกล้อง

เพิ่มพื้นที่ถนนได้เป็นพิกัดภาพต้นฉบับ เช่น:

```yaml
regions:
  - id: near_road
    points: [[0, 700], [1920, 700], [1920, 1080], [0, 1080]]
```

## เตรียมภาพและจำนวนจริงสำหรับ Count-check

1. เลือกหนึ่งเฟรมจากแต่ละช่วงวิดีโอด้วยตนเอง
2. กรอก `frame` เป็นเลขเฟรมของไฟล์ต้นฉบับ เริ่มจาก 0 และต้องอยู่ใน Segment ที่เลือก
3. สร้างภาพ PNG เพื่อเปิดนับจำนวนจริง

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode prepare --frame day_north=2250
```

เพิ่ม `--frame VIDEO_ID=FRAME` ได้หลายครั้ง หรือกรอกเลขครบใน YAML แล้วใช้ `--mode prepare` คำสั่ง Override ไม่เขียนทับ YAML ให้คัดลอกเลขที่เลือกกลับลง Config ภาพที่ได้และ `selected_frames.json` ระบุ Hash/เฟรมให้ตรวจสอบ

4. เปิด PNG และนับรถยนต์กับรถจักรยานยนต์จริง กรอก `truth: {car: 10, motorcycle: 3}` ตามภาพของคุณ ตัวเลขนี้เป็นตัวอย่างเท่านั้น ค่า 0 ใช้เฉพาะเมื่อไม่มีรถคลาสนั้นจริง ส่วน `null` หมายถึงยังไม่กรอก
   สามารถคัดลอก `frame_sha256` จาก `selected_frames.json` มาใส่ `truth_frame_sha256` ในรายการวิดีโอ เพื่อให้ระบบหยุดหากไฟล์หรือภาพเปลี่ยนจากตอนนับจริง
5. เมื่อครบกลุ่มที่เลือกแล้วรัน:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode count-check
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode all
```

หากไม่มี Frame/Truth ครบทุกคลิปของกลุ่ม ระบบจะระบุ Count-check ว่ายังไม่พร้อม `--mode all` ยังคงรัน Performance ได้ ไม่มีการเติมจำนวนจริงให้เอง

Count-check ใช้ Confidence หลัก 0.25 และสำรวจ 0.10/0.25/0.50/0.75 ด้วย รายงาน Signed error, Absolute error, MAE, Bias, Exact-match rate และ WAPE แยกคลาส/กลางวัน/กลางคืน/รวม ถ้าตัวหาร WAPE เป็น 0 ให้ N/A อันดับนับใช้ค่าเฉลี่ยผลรวม Absolute error สองคลาสต่อภาพ ห้ามหักล้างข้อผิดพลาดข้ามคลาส

ข้อมูลจำนวนอย่างเดียวไม่รองรับ Precision, Recall, mAP และการบอกว่ารถคันไหนพลาด การนับตรงอาจเกิดจากพลาดรถจริงหนึ่งคันและตรวจผิดเพิ่มหนึ่งคัน ภาพ 8 ภาพเป็น Spot check เท่านั้น ไม่ใช่ชุดประเมินความแม่นยำขนาดใหญ่

## อ่านผล

- `index.html`: เปิดใน Browser ได้โดยตรง มีตารางและตัวกรองภาพตามโมเดล/กล้อง/Day-Night/ชนิดภาพ ใช้ได้ออฟไลน์
- `REPORT.md`: สรุปวิธีวัด สถานะ เงื่อนไข และตาราง
- `comparison.csv`: ค่าเฉลี่ยถ่วงตามเฟรม, FPS รวม, Percentile, ความแปรปรวนระหว่างรอบ, CPU/RAM/GPU/VRAM
- `per_video_pass.csv`, `per_video_average.csv`: ผลแต่ละคลิป/รอบและค่าเฉลี่ยคลิป
- `count_per_frame.csv`, `count_summary.csv`: จำนวนจริงและผลตรวจพร้อม Count error
- `paired_comparison.csv`, `export_agreement.csv`: Speedup/จำนวนที่ต่างกัน และการจับคู่กล่องคลาสเดียวกันแบบหนึ่งต่อหนึ่งที่ IoU ≥ 0.5 เป็น Agreement ไม่ใช่ความถูกต้องกับ Ground truth
- `rankings.csv`: แยกอันดับ Deployment รวม Precision, กลุ่ม Precision เดียวกัน และ Count error
- `contact_*.png`: ภาพเปรียบเทียบ Heat map โมเดลบนคลิปเดียวกัน
- ภายในผลแต่ละคลิปมี `frames.npz`, `telemetry.csv`, `timeline.png` และในรอบแรกมี `heat.npz`, `predictions.npz`, `background.png`
- คลิก Heat map ใน HTML เพื่อดูค่า Density ที่พิกัด Pixel จริง ค่าที่ชี้เป็นค่าดิบ ส่วนภาพเป็นแบบ Smooth สเกลสีเหมือนกันระหว่างโมเดลต่อคลิป/คลาส
- Difference map: สีแดงตรวจมากกว่า Reference, สีน้ำเงินน้อยกว่า Reference แผนที่ต่างกล้องไม่รวมพิกัดเข้าด้วยกัน
- `resolved_config.json`, `dataset.json`, `run.json`, `requirements-lock.txt`: ข้อมูลทำซ้ำและ Hash; Environment/Artifact metadata อยู่ในผลของแต่ละ Process

Heat map ใช้จุดกลางล่างของกล่องจากรอบแรกเท่านั้น จำนวนเป็นการตรวจซ้ำตามเฟรม รถจอดอาจทำให้หนาแน่นสูง ไม่ใช่จำนวนรถไม่ซ้ำหรือพื้นที่ที่โมเดลตรวจได้แน่นอน

Inference time เป็น Model forward จาก CUDA-synchronized profiler ส่วน Detection time รวม Preprocess, Forward, Postprocess และส่งกล่องกลับ CPU ไม่รวม Decode, โหลด, Warm-up, Hash, วิเคราะห์ภาพ หรือเขียนไฟล์ Embedded NMS รวมใน Forward จึงไม่ปะปนกับกลุ่มที่ปรับ Threshold ได้ อันดับ Forward ไม่รวมโมเดล NMS-free ที่ขอบเขตงานต่างกัน

VRAM เก็บก่อนโหลด หลังโหลด และหลัง Warm-up พร้อม Sample ระหว่างรัน NVML Process VRAM บน WDDM อาจ N/A จึงมี Device memory และผลต่างจากก่อนโหลดเป็นข้อมูลประกอบแบบ Provisional ซึ่งรวม Desktop/โปรแกรมอื่น ใช้ PyTorch allocated memory แทน TensorRT ไม่ได้ ค่า Telemetry CPU/GPU/RAM ครอบคลุมช่วง Measurement loop รวม Decode/Hash/เก็บผล ไม่ใช่เฉพาะ Model call ค่า Mean resource ใน Summary เป็นค่าเฉลี่ย Mean ของแต่ละคลิป

## ความล้มเหลวและการทดสอบ

แต่ละ Artifact/รอบเป็น Process ใหม่ ทำงานเรียงกัน ไม่มีการรันหลายโมเดลบน GPU พร้อมกัน ผิดพลาดจะเก็บ `failed.json` และ `worker.log` แล้วทำงานรายการต่อไป ผลที่ไม่ครบถูกตัดออกจากอันดับ กด Ctrl+C เพื่อหยุด Worker และเก็บ Checkpoint รายงานสร้างใหม่ได้โดยไม่ทำ Inference:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py --mode report --output benchmark/model-results/ชื่อรันเดิม
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_benchmark.py -v
```

ระบบไม่ Resume Inference อัตโนมัติ ให้รันรายการโมเดลที่ต้องการใหม่ในโฟลเดอร์ใหม่ ส่วนการสร้างรายงานจาก Checkpoint ทำซ้ำได้ ไม่ลบผลเก่า

อ้างอิง: https://docs.ultralytics.com/modes/predict/ และ https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__v1__t.html

Dependencies สำหรับสร้าง Environment ใหม่อยู่ใน `requirements-benchmark.txt` โดย CUDA PyTorch และ TensorRT ต้องติดตั้งให้ตรง GPU/Engine เอง การเปรียบเทียบ Precision อ้างอิงค่าที่ตั้ง/Export; Engine ที่ Compile แล้วอาจใช้ TF32 หรือ Tactics ต่างจาก PyTorch และเปลี่ยนย้อนหลังผ่าน YAML ไม่ได้ Peak ทรัพยากรเป็นค่าสูงสุดที่ Sample ได้ อาจพลาดการเพิ่มขึ้นสั้น ๆ

ชุดทดสอบ GPU เพิ่มเติม (สร้างวิดีโอว่างชั่วคราวที่รู้จำนวนจริงเป็น 0 ไม่ใช้แทน Ground truth ของวิดีโอคุณ):

```powershell
$env:BENCHMARK_GPU_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_model_benchmark.py -v
```
