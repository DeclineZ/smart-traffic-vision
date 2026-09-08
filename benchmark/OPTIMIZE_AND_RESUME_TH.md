# รันทดสอบชุดเล็กและอ่านวิดีโอล่วงหน้า

รันจาก `C:\Sea traffic\traffic` หลังหยุด Benchmark อื่นบน GPU แล้ว:

```powershell
.\.venv\Scripts\python.exe benchmark/model_benchmark.py `
  --config benchmark/models_yolo_s_m.yaml `
  --mode performance `
  --models v8s_fp16_engine v8m_fp16_engine v11s_fp16_engine v11m_fp16_engine v26s_fp16_engine v26m_fp16_engine `
  --videos day_north day_south night_north night_south `
  --repeats 1 --prefetch 4 `
  --output benchmark/model-results/engine-north-south
```

รวม 6 jobs แต่ละ job มี 4 คลิป รวม 24 งานย่อยรายคลิป ประมวลผลทุกเฟรมเหมือนเดิม

`--prefetch 4` ให้ CPU อ่านและ Hash ล่วงหน้าสูงสุด 4 เฟรมในคิว ไม่ทิ้งเฟรม ค่าเริ่มต้น `0` ทำงานตามลำดับแบบเดิม ใช้ค่าเดียวกันทุกโมเดลในชุดเปรียบเทียบ เพราะการทำงานพร้อมกันอาจกระทบ latency และ CPU/RAM

เพิ่ม `--smoke` เพื่อทดสอบ 8 เฟรมต่อคลิปก่อน ต้องใช้ output ใหม่ ผล Smoke ใช้ตรวจระบบเท่านั้น

## หยุดและทำต่อ

กด Ctrl+C ครั้งเดียวแล้วรอจบการบันทึก จากนั้นเรียกคำสั่งเดิมพร้อม `--resume` และ output เดิม ระบบตรวจ Config, ตัวเลือก, Hash ของโมเดล/วิดีโอ และโค้ดให้ตรงกันก่อนทำต่อ

ข้าม job ที่เสร็จ และใช้คลิปที่มี result.json พร้อมไฟล์ผลครบแล้ว คลิปที่ยังไม่เสร็จเริ่มใหม่และทำ warm-up งานที่รันต่อมี resume_history ใน run.json ผลเดิมก่อนเพิ่มฟีเจอร์นี้ไม่มี resume_signature.json จึงใช้ resume ไม่ได้

อย่าเรียก Resume ขณะ run เดิมยังทำงานอยู่ และอย่าแก้ไฟล์ผลลัพธ์ด้วยตนเอง ระบบตรวจไฟล์ผลที่ต้องมีแต่ไม่ได้ตรวจ hash ของทุกไฟล์ผล

## อ่านเวลา

result.json และ per_video_pass.csv เพิ่ม processing_seconds/FPS (รวมงานอ่านและตรวจจับในลูป), hash_ms, clip_seconds (รวมเปิดคลิปและ warm-up ไปจนบันทึก artifacts) และ prefetch

Decode และ Hash ทำงานซ้อนกับ Detection เมื่อเปิด prefetch จึงห้ามบวกเวลาเหล่านี้เป็น wall time ให้ดู processing_seconds โดยตรง Pure inference ยังรายงานแยกเช่นเดิม ETA ประมาณจากคลิปที่เสร็จจริงของ job นั้น

การเปรียบเทียบความเร็วควรรันชุดเดียวกันด้วย prefetch 0 และ 4 แยก output เมื่อ GPU ว่าง ตรวจ decoded_frames_sha256 และจำนวนเฟรมให้ตรงกัน และเปรียบเทียบ latency P95 ด้วย ยังไม่รับประกันว่าจะเร็วขึ้นจนกว่าจะทดสอบจริง
