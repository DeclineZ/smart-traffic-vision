# YOLO26s Thai Traffic Baseline Evaluation Report (Batch 3)

- **Date**: 2026-09-27T15:36:54.044190+00:00
- **Evaluated Checkpoint**: `E:\Work\Projects\AdaptiveTrafficControl\smart-traffic-vision\models\yolo26s_thai_traffic.pt`
- **Checkpoint SHA256**: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`
- **Device**: `NVIDIA GeForce RTX 5060 Laptop GPU` (CUDA: True)
- **Frameworks**: PyTorch `2.12.0.dev20260408+cu128`, Ultralytics `8.4.124`, OpenCV `4.14.0`
- **Benchmark Dataset**: `data/eval_snapshot_v1` (42 frames, 1174 verified ground-truth instances)

---

> [!IMPORTANT]
> **Scientific Integrity & Benchmark Non-Independence Notice**
> This evaluation benchmark consists of 42 diagnostic frames with documented exposure lineage:
> - **Nearby Training Diagnostic**: 7 frames (<= 3.0s temporal distance from training footage).
> - **Unproven Checkpoint Exposure**: 35 candidate frames where model weight training absence cannot be definitively proven.
> 
> This benchmark measures internal model capability, failure modes, and slice disparities. It **MUST NOT** be cited as an independent test set.

---

## 1. Ultralytics Standard Benchmark AP Metrics (conf=0.001, iou=0.6)

Evaluated via native Ultralytics `YOLO.val()` with low confidence integration threshold (`conf=0.001`) to calculate complete Precision-Recall curves. Per-class AP is mapped strictly through returned class IDs:

| Vehicle Class | Class ID | Images | GT Instances | AP50 | AP50-95 | Best-F1 Precision | Best-F1 Recall | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **All Classes** | — | **42** | **1174** | **0.4644** | **0.3222** | **0.6013** | **0.4940** | **Benchmark Overall** |
| `car` | 0 | 42 | 986 | 0.5558 | 0.3210 | 0.8379 | 0.4442 | Dense Support |
| `motorcycle` | 1 | 37 | 132 | 0.4742 | 0.2237 | 0.8390 | 0.4167 | Moderate Support |
| `bus` | 2 | 16 | 21 | 0.3011 | 0.2351 | 0.4720 | 0.4762 | Low Support (21 GT) |
| `truck` | 3 | 13 | 18 | 0.3104 | 0.2645 | 0.1953 | 0.5556 | Low Support (18 GT) |
| `three_wheeler` | 4 | 14 | 17 | 0.6805 | 0.5664 | 0.6623 | 0.5774 | Low Support (17 GT) |

---

## 2. Operational Diagnostics at Fixed Threshold (conf=0.25, matching IoU >= 0.5)

Diagnostics evaluated at operational threshold `conf=0.25`:

| Vehicle Class | Class ID | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Overall Operational** | — | **1174** | **617** | **79.6%** | **41.8%** | **54.8%** | Operational Baseline |
| `car` | 0 | 986 | 483 | 85.7% | 42.0% | 56.4% | Dense Support |
| `motorcycle` | 1 | 132 | 59 | 86.4% | 38.6% | 53.4% | Moderate Support |
| `bus` | 2 | 21 | 21 | 38.1% | 38.1% | 38.1% | Low Support |
| `truck` | 3 | 18 | 42 | 23.8% | 55.6% | 33.3% | Low Support |
| `three_wheeler` | 4 | 17 | 12 | 66.7% | 47.1% | 55.2% | Low Support |

### 6x6 Object Confusion Matrix (Matching IoU >= 0.50)
Rows represent authoritative **Human Ground Truth**, columns represent **Baseline Model Predictions**:

| GT / Pred | car (0) | motorcycle (1) | bus (2) | truck (3) | three_wheeler (4) | Background (FN) | Total GT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **car (0)** | **414** | 2 | 8 | 24 | 1 | **537** | 986 |
| **motorcycle (1)** | 0 | **51** | 0 | 0 | 2 | **79** | 132 |
| **bus (2)** | 1 | 0 | **8** | 2 | 0 | **10** | 21 |
| **truck (3)** | 2 | 0 | 0 | **10** | 0 | **6** | 18 |
| **three_wheeler (4)** | 2 | 0 | 0 | 0 | **8** | **7** | 17 |
| **Background (FP)** | 64 | 6 | 5 | 6 | 1 | — | **82** |

---

## 3. Diagnostic Slices Breakdown

### Slice A: Ground-Truth Object Size Recall (Matched on Full Frame First)
*Formula*: `scale = 640 / max(img_w, img_h)`, `area = (bw * img_w * scale) * (bh * img_h * scale)`.
*Note*: Full-frame matching is performed first to guarantee that slightly larger or shifted detections successfully match small ground-truth vehicles. Size-specific precision is omitted because false-positive attribution across sizes is ambiguous.

| Size Category | Area Definition | GT Count | Detected (TP) | Missed (FN) | Class Disagree | Loc Error | GT Recall | Support Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Small** | < 32^2 px (< 1024 px^2) | 948 | 331 | 561 | 16 | 40 | **34.9%** | Dense support (948 GT instances) |
| **Medium** | 32^2 <= area <= 96^2 (1024 to 9216 px^2) | 180 | 127 | 30 | 18 | 5 | **70.6%** | Dense support (180 GT instances) |
| **Large** | > 96^2 px (> 9216 px^2) | 46 | 33 | 2 | 10 | 1 | **71.7%** | Dense support (46 GT instances) |

### Slice B: Lighting Disparity (Real Day vs Real Night)

| Condition | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Real Day** | 34 | 938 | 501 | 78.0% | 41.7% | 54.3% | Dense Support |
| **Real Night** | 8 | 236 | 116 | 86.2% | 42.4% | 56.8% | Moderate Support |

### Slice C: Camera Domain Disparity (Holdout Northeast vs In-Distribution)

| Camera Group | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Support Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **In-Distribution Cameras** | 34 | 948 | 489 | 83.8% | 43.2% | 57.1% | Dense Support |
| **Holdout Northeast (cam45)** | 8 | 226 | 128 | 63.3% | 35.8% | 45.8% | Moderate Support |

### Slice D: Exposure Lineage Disparity

| Exposure Group | Frames | GT Count | Model Preds | Precision | Recall | F1 Score | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Nearby Training (<= 3.0s)** | 7 | 277 | 132 | 87.1% | 41.5% | 56.2% | Known Training Proximity |
| **Unproven Checkpoint Exposure** | 35 | 897 | 485 | 77.5% | 41.9% | 54.4% | Unproven Lineage Candidate |

---

## 4. Primary Baseline Failure Modes & Empirical Observations

1. **Small Vehicle Omission**:
   - For vehicles letterboxed under 32x32 pixels, ground-truth recall is **34.9%** (561 pure omissions). Distant vehicles in queued lanes are frequently missed.
2. **Truck vs Light Vehicle Confusion (Poor Truck Precision)**:
   - Operational truck precision is only **23.8%** (32 false alarms out of 42 truck predictions). The model frequently misidentifies pickup-based songthaews, passenger vans, and delivery pickups as commercial trucks.
3. **Motorcycle Omission in Congestion**:
   - At operational threshold (conf=0.25), motorcycle recall is **38.6%** (81 false negatives out of 132 GT).
4. **Generalization Gap on Holdout Camera**:
   - Precision drops from 83.8% on in-distribution cameras to **63.3%** on `cam45_northeast`.

---

## 5. Potential Experimental Directions to Test (Future Iterations)

The following directions represent empirical hypotheses to evaluate in future model training passes, not established requirements:
- **Multi-Scale Tiling Inference**: Test high-resolution slicing (e.g. SAHI / tile inference) on distant traffic queues to evaluate if small vehicle recall can be improved without generating excessive false alarms.
- **Morphological Feature Distillation**: Test loss reweighting or fine-tuning with hard negatives to differentiate ordinary pickups and vans (class 0) from medium trucks (class 3).
- **Domain Adaptive Night Augmentation**: Test low-light glare simulation and contrast augmentation to evaluate night recall recovery.

---

## 6. Visual Error Overlays

All 42 visual overlays are saved in `output/baseline_eval/overlays/`:
- **Green**: True Positive (GT matched with correct prediction)
- **Red**: Missed Vehicle (False Negative: GT omitted by baseline)
- **Orange**: False Alarm (False Positive: spurious baseline detection)
- **Purple**: Class Disagreement (Spatial overlap >= 0.50, but wrong class assigned)
- **Cyan**: Localization Error (0.10 <= IoU < 0.50)

Representative diagnostic frames:
- Daytime Approach: `output/baseline_eval/overlays/cam03_east_f003720_overlay.jpg`
- Real Night Approach: `output/baseline_eval/overlays/cam43_south_night_f002800_overlay.jpg`
- Northeast Camera: `output/baseline_eval/overlays/cam45_northeast_f063055_overlay.jpg`
- Small / Distant Vehicles: `output/baseline_eval/overlays/cam43_south_f046440_overlay.jpg`
