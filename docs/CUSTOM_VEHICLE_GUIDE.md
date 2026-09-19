# Thai Multi-Class Traffic Vision & Custom Vehicle Guide

Comprehensive guide to mining, curating, and fine-tuning Ultralytics YOLO26s using YOLO26x co-training for the 5-Class COCO-aligned Thai traffic standard:
- **`car`** (0): Sedans, hatchbacks, taxis, SUVs, PPVs, and commuter passenger vans (Toyota Commuter / HiAce - COCO aligned)
- **`motorcycle`** (1): Scooters, underbones, commuter bikes, big bikes, delivery motorbikes (Grab, Lineman, Shopee)
- **`bus`** (2): BMTA city buses (ขสมก. ครีมแดง / ปรับอากาศ), EV Thai Smile Bus, intercity tour coaches, double-deckers
- **`truck`** (3): Standard กระบะ (Hilux, D-Max), high-cage รถคอก, สองแถว (songthaew), 6/10-wheelers, and articulated 18-wheelers (รถพ่วง)
- **`three_wheeler`** (4): ตุ๊กตุ๊ก (tuk-tuk) and ซาเล้ง (saleng / cargo sidecar tricycle)

---

## 1. Directory Structure

Staging and dataset folders are organized under `data/`:

```text
data/
├── saleng/
│   ├── seeds/                  # Reference crops of ซาเล้ง
│   ├── mined_candidates/       # Auto-extracted crops & manifest.json
│   └── verified_hits/          # User-confirmed true positive crops (71 crops)
├── pickup/
│   ├── seeds/                  # Reference crops of กระบะ, รถคอก, สองแถว
│   ├── mined_candidates/
│   └── verified_hits/          # User-confirmed true positive crops (319 crops)
├── truck_trailer/
│   ├── seeds/                  # Reference crops of 18-wheeler / รถพ่วง
│   ├── mined_candidates/
│   └── verified_hits/          # User-confirmed true positive crops (95 crops)
├── van/
│   ├── seeds/                  # Reference crops of รถตู้โดยสาร
│   ├── mined_candidates/
│   └── verified_hits/          # User-confirmed true positive crops (165 crops)
├── bus/
│   ├── seeds/                  # Reference crops of BMTA, EV bus, coach
│   ├── mined_candidates/
│   └── verified_hits/          # User-confirmed true positive crops (76 crops)
├── tuktuk/
│   └── verified_hits/          # Pre-verified tuk-tuk hits (264 crops)
└── multiclass_dataset/         # Final compiled 5-class YOLO training dataset
    ├── images/{train,val}
    ├── labels/{train,val}
    └── data.yaml
```

---

## 2. Seed Capturing Guidelines

When screenshotting reference seeds from CCTV:
- **Zero-Margin Tight Crop**: Crop tightly around the vehicle body with zero excess road asphalt or yellow zebra markings. This prevents DINOv2 from biasing toward road texture instead of vehicle geometry.
- **Angle Diversity**: Capture ~1/3 front-quarter, ~1/3 side profile, and ~1/3 rear-quarter.
- **Lighting**: Include daylight, dusk, and nighttime CCTV shots.

---

## 3. Mining Candidates with DINOv2 & Dual-Stream Proposals

Mine candidates using `tools/mine_vehicles_from_seeds.py`:

```bash
.venv/Scripts/python.exe tools/mine_vehicles_from_seeds.py \
    --target-class saleng \
    --yolo-model yolo26s.pt \
    --sim-threshold 0.70 \
    --fast
```

---

## 4. Human Verification Workflow

1. Open `data/<target-class>/mined_candidates/crops/` in Windows File Explorer.
2. Select genuine true-positive instances.
3. Cut/copy selected crops directly into `data/<target-class>/verified_hits/`.

---

## 5. Compile Master 5-Class Dataset with YOLO26x Co-Training Teacher

Compile the master dataset:

```bash
.venv/Scripts/python.exe tools/compile_multiclass_dataset.py \
    --teacher-model yolo26x.pt \
    --teacher-conf 0.60 \
    --iomin-suppress 0.65 \
    --saleng-boost 4 \
    --bus-boost 3 \
    --truck-boost 2 \
    --roadway-crop
```

### Key Protections Enforced During Compilation:
1. **IoMin $\ge 0.65$ Suppression**: Suppresses teacher model detections of tractor cabs nested inside 18-wheeler articulated trailers.
2. **Cross-Category Anti-Corruption**: Automatically rejects conflicting teacher predictions (e.g. teacher labeling pickups as car, vans as truck, or 3-wheelers as motorcycle).
3. **Minority Class Oversampling**: Boosts `saleng` ($4\times$), `bus` ($3\times$), and heavy `truck_trailer` ($2\times$) to overcome the car/motorcycle class dominance.
4. **Temporal Event Clustering**: Partitions transits (< 3.0s temporal gap) into discrete events so frames from the same vehicle transit never leak across train and validation splits.
5. **Night-Boost & Synthetic IR**: Real nighttime frames are jitter-boosted $3\times$, and daylight frames are synthesized into monochrome IR to ensure round-the-clock detection accuracy.

---

## 6. Fine-Tune YOLO26s on the Unified Dataset

Train YOLO26s with Copy-Paste, Mixup, and Cosine Annealing:

```bash
.venv/Scripts/python.exe train_yolo26s.py
```

The script:
- Fine-tunes `yolo26s.pt` with automatic weight remapping for the 5 classes.
- Uses `copy_paste=0.35` and `mixup=0.10` to balance rare vehicles (`saleng`, `bus`, `truck`).
- Disables mosaic augmentation in the last 5 epochs for bounding box stabilization (`close_mosaic=5`).
- Evaluates per-class Precision, Recall, and mAP@0.5 across all 5 classes.
- Exports best weights directly to `models/yolo26s_thai_traffic.pt`.

---

## 7. Multi-Camera Edge Deployment

Run the fine-tuned model across all 4 intersection approaches (RTX 5060 Laptop GPU):

```bash
.venv/Scripts/python.exe run_multi_camera.py \
    --model models/yolo26s_thai_traffic.pt \
    --conf 0.15 \
    --display
```
