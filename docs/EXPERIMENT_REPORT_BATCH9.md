# Batch 9: Controlled Local-Only vs Reviewed-External Fine-Tuning Experiment Report

- **Date**: 2026-09-28T09:50:21.095167+00:00
- **Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Candidate A Checkpoint (best.pt)**: `runs\train\candidate_a_local_only\weights\best.pt` (SHA256: `c43a6bb26f9a686588f9d4445583be0f12194e6ff25ef6480e881fa2686f46de`)
- **Candidate B Checkpoint (best.pt)**: `runs\train\candidate_b_reviewed_external\weights\best.pt` (SHA256: `8cabf72c0c0bcf87c3568becf852794bf33d487a445087a23c557585e0f79558`)
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch 2.12.0.dev20260408+cu128, CUDA 12.8, Ultralytics 8.4.124)

---

## 1. Experimental Design & Dataset Isolation

This experiment directly measures whether fine-tuning with 25 fully human-reviewed external frames (813 approved boxes) provides measurable detection improvements over training exclusively on local CCTV footage and over the deployed baseline.

### Controlled Dataset Separation
- **Dataset A (Corrected Local-Only)**:
  - 1,092 local CCTV approach footage frames (527 unique canonical sources).
  - 18 frames have verified human annotations (675 boxes); **1,074 frames retain unreviewed teacher annotations**.
  - Exactly 41 stale synthetic variants quarantined; 2 variants of rejected source `cam44_north_f019140` quarantined.
  - Total boxes: **13,148**.
- **Dataset B (Local + Reviewed External)**:
  - Exactly the same 1,092 local frames from Dataset A.
  - Plus **only the 25 eligible, human-reviewed external UA-DETRAC frames** (813 approved boxes) from `data/review_pack_consolidated_v3`.
  - All 238 unreviewed external frames remain **strictly excluded**.
  - Total boxes: **13,961**.
  - **Exact Delta (B minus A)**: Exactly 25 images and 813 boxes.
- **Validation & Diagnostic Sets**:
  - Primary Validation: 130 canonical CCTV frames (1367 ground truth boxes).
  - Diagnostic Snapshot: 42 verified CCTV frames (1,174 ground truth boxes).
  - **Zero Canonical Contamination**: Mutual intersection of Train A (527 sources), Train B (552 sources), Val (130 sources), and Diagnostic Benchmark (42 sources) = **0 overlapping sources (100% disjoint)**.

### Class & Size Distributions (Actual Selected Annotations)

| Vehicle Class | Dataset A Boxes | Dataset A % | External Delta (B - A) | Dataset B Boxes | Dataset B % | Primary Val Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `car` (0) | 9,453 | 71.9% | +721 | 10,174 | 72.9% | 1,010 |
| `motorcycle` (1) | 1,482 | 11.3% | +3 | 1,485 | 10.6% | 164 |
| `bus` (2) | 413 | 3.1% | +84 | 497 | 3.6% | 23 |
| `truck` (3) | 1,294 | 9.8% | +5 | 1,299 | 9.3% | 124 |
| `three_wheeler` (4) | 506 | 3.8% | +0 | 506 | 3.6% | 46 |
| **Total Boxes** | **13,148** | **100.0%** | **+813** | **13,961** | **100.0%** | **1,367** |

---

## 2. Training Hyperparameters, Steps & Exposures

Both candidates were independently warm-started from the deployed baseline weights. To eliminate the bias learning-rate spike identified during the smoke test (where bias LR reached 0.072), `warmup_bias_lr` was explicitly locked to `lr0 = 0.0001`.

| Training Parameter | Candidate A (Local Only) | Candidate B (Local + External) | Equality / Delta |
| :--- | :--- | :--- | :--- |
| **Initialization** | `models/yolo26s_thai_traffic.pt` | `models/yolo26s_thai_traffic.pt` | Identical baseline weights |
| **Epoch Budget** | 30 epochs | 30 epochs | Identical fixed budget |
| **Batch Size** | 16 (OOM fallback: False) | 16 (OOM fallback: False) | Identical batch size |
| **Resolution** | 640×640 letterboxed | 640×640 letterboxed | Identical resolution |
| **Optimizer & LR** | SGD, $\text{lr}_0=0.0001, \text{lrf}=0.01$, Cosine | SGD, $\text{lr}_0=0.0001, \text{lrf}=0.01$, Cosine | Identical schedule |
| **Explicit Warmup** | 1 epoch, momentum 0.8, **bias_lr=0.0001** | 1 epoch, momentum 0.8, **bias_lr=0.0001** | **No bias LR spike** |
| **Frozen Layers** | Backbone (Layers 0–9, 4.45M params) | Backbone (Layers 0–9, 4.45M params) | Identical frozen layers |
| **Trainable Layers** | Neck & Head (Layers 10–23, 5.50M params) | Neck & Head (Layers 10–23, 5.50M params) | Identical trainable layers |
| **Random Seed** | 42 | 42 | Deterministic |
| **Total Images** | 1,092 images | 1,117 images | +25 images (+2.3%) |
| **Image Exposures** | 32,760 exposures | 33,510 exposures | +750 exposures |
| **Optimizer Steps** | 2,070 steps | 2,100 steps | +30 steps |
| **Training Duration** | 761.21 seconds | 678.54 seconds | Stable execution |

---

## 3. Evaluation Benchmark Results (Reported in Percentage Points)

> [!WARNING]
> **Validation Generalization Limitation Notice**
> Exactly **107 of the 130 validation frames (82.3%) originated from the historical training split** of the raw CCTV dataset.
> Furthermore, 1,074 of the 1,092 local training frames and all 130 validation frames retain unreviewed teacher labels.
> Neither Primary Validation nor the Diagnostic Benchmark represents a completely unseen, independent test set.

### 3.1 Primary Validation Set (130 Frames, 1,367 Ground Truth Boxes)

| Metric | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **0.9440** | **0.9429** | **0.9452** | **+0.12 pp** | **+0.23 pp** |
| **mAP50-95** | **0.8386** | **0.8396** | **0.8416** | **+0.30 pp** | **+0.20 pp** |
| **Operational F1 (conf=0.25)** | **85.5%** | **83.6%** | **84.4%** | **-1.19 pp** | **+0.74 pp** |
| **Operational Precision** | 81.3% | 77.2% | 79.0% | -2.34 pp | +1.82 pp |
| **Operational Recall** | 90.2% | 91.2% | 90.5% | +0.29 pp | -0.73 pp |
| `car` AP50 | 0.9398 | 0.9399 | 0.9392 | -0.06 pp | -0.07 pp |
| `motorcycle` AP50 | 0.9313 | 0.9265 | 0.9321 | +0.08 pp | +0.56 pp |
| `bus` AP50 | 0.9511 | 0.9547 | 0.9601 | +0.90 pp | +0.54 pp |
| `truck` AP50 | 0.9049 | 0.9041 | 0.9090 | +0.41 pp | +0.49 pp |
| `three_wheeler` AP50 | 0.9930 | 0.9894 | 0.9856 | -0.74 pp | -0.38 pp |
| Small Object Recall (< 1024 px²) | 88.2% | 89.4% | 87.9% | -0.27 pp | -1.46 pp |
| Medium Object Recall | 92.2% | 92.6% | 93.0% | +0.84 pp | +0.42 pp |
| Large Object Recall | 94.2% | 96.4% | 95.7% | +1.45 pp | -0.73 pp |

### 3.2 Human-Reviewed Diagnostic Benchmark (42 Frames, 1,174 Ground Truth Boxes)

| Metric | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **0.4644** | **0.4674** | **0.4589** | **-0.55 pp** | **-0.85 pp** |
| **mAP50-95** | **0.3222** | **0.3217** | **0.3109** | **-1.13 pp** | **-1.08 pp** |
| **Operational F1 (conf=0.25)** | **54.8%** | **56.1%** | **56.1%** | **+1.25 pp** | **-0.01 pp** |
| **Operational Precision** | 79.6% | 77.0% | 76.7% | -2.92 pp | -0.31 pp |
| **Operational Recall** | 41.8% | 44.1% | 44.2% | +2.39 pp | +0.09 pp |
| `car` AP50 | 0.5558 | 0.5896 | 0.5883 | +3.25 pp | -0.13 pp |
| `motorcycle` AP50 | 0.4742 | 0.4893 | 0.4986 | +2.44 pp | +0.93 pp |
| `bus` AP50 | 0.3011 | 0.3042 | 0.3153 | +1.42 pp | +1.11 pp |
| `truck` AP50 | 0.3104 | 0.2604 | 0.2247 | -8.57 pp | -3.57 pp |
| `three_wheeler` AP50 | 0.6805 | 0.6935 | 0.6674 | -1.31 pp | -2.61 pp |
| Small Object Recall (< 1024 px²) | 34.9% | 37.3% | 37.0% | +2.11 pp | -0.31 pp |
| Medium Object Recall | 70.6% | 71.7% | 74.4% | +3.88 pp | +2.77 pp |
| Large Object Recall | 71.7% | 76.1% | 73.9% | +2.17 pp | -2.18 pp |

### 3.3 Environmental & Historical Lineage Slices (Operational Recall at conf=0.25)

| Slices & Subpopulations | Deployed Baseline | Candidate A (best) | Candidate B (best) | B vs Baseline | B vs A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Val: Day Scenes** | 89.7% | 90.9% | 90.0% | +0.28 pp | -0.91 pp |
| **Primary Val: Night Scenes** | 92.0% | 92.4% | 92.4% | +0.36 pp | 0.00 pp |
| **Primary Val: Old Train Split Lineage (107 frames)** | 92.8% | 93.5% | 92.7% | -0.10 pp | -0.85 pp |
| **Primary Val: Old Val Split Lineage (23 frames)** | 81.2% | 83.2% | 82.9% | +1.64 pp | -0.33 pp |
| **Diagnostic: Day Scenes** | 41.7% | 43.3% | 43.1% | +1.39 pp | -0.21 pp |
| **Diagnostic: Night Scenes** | 42.4% | 47.5% | 48.7% | +6.36 pp | +1.27 pp |
| **Diagnostic: Northeast Holdout Geometry (12 frames)** | 35.8% | 37.2% | 40.3% | +4.43 pp | +3.10 pp |
| **Diagnostic: In-Distribution Cameras (30 frames)** | 43.2% | 45.8% | 45.1% | +1.90 pp | -0.63 pp |

### 3.4 Pickup / Passenger Van vs Truck Confusion Analysis (Operational conf=0.25)

In Thai traffic environments, passenger pickups and commuter vans belong strictly to the `car` (0) class, but teacher models frequently mislabel them as `truck` (3). Conversely, small flatbed trucks are sometimes misclassified as cars.

| Dataset Benchmark | Confusion Type | Deployed Baseline | Candidate A (best) | Candidate B (best) | Impact of Added External Frames (B vs A) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Primary Val (130 frames)** | GT Car predicted as Truck (False Truck) | 14 errors | 16 errors | 15 errors | Delta: -1 errors |
| **Primary Val (130 frames)** | GT Truck predicted as Car (Missed Truck) | 11 errors | 10 errors | 10 errors | Delta: +0 errors |
| **Diagnostic Benchmark (42 frames)** | GT Car predicted as Truck (False Truck) | 24 errors | 26 errors | 25 errors | Delta: -1 errors |
| **Diagnostic Benchmark (42 frames)** | GT Truck predicted as Car (Missed Truck) | 2 errors | 2 errors | 2 errors | Delta: +0 errors |

### 3.5 Checkpoint Selection Comparison: best.pt vs last.pt

`best.pt` was selected automatically by validation split fitness ($0.1 \times \text{mAP50} + 0.9 \times \text{mAP50-95}$ on the 130-frame primary validation set). `last.pt` corresponds to the final state at epoch 30:

| Model & Checkpoint | Primary Val mAP50 | Primary Val mAP50-95 | Diag Benchmark mAP50 | Diag Benchmark mAP50-95 | Checkpoint Selection Criteria |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Candidate A (best.pt)** | **0.9429** | **0.8396** | **0.4674** | **0.3217** | Peak validation fitness |
| **Candidate A (last.pt)** | 0.9319 | 0.8246 | 0.4643 | 0.3143 | Final epoch 30 weights |
| **Candidate B (best.pt)** | **0.9452** | **0.8416** | **0.4589** | **0.3109** | Peak validation fitness |
| **Candidate B (last.pt)** | 0.9383 | 0.8289 | 0.4720 | 0.3181 | Final epoch 30 weights |

---

## 4. Hardware Inference Latency Benchmark

Evaluated on NVIDIA GeForce RTX 5060 Laptop GPU across 100 timed iterations (640×640 input resolution):

| Model Checkpoint | File Size | Mean Latency (ms / image) | Throughput (FPS) | Computational Equivalence |
| :--- | :---: | :---: | :---: | :--- |
| **Deployed Baseline** | 19.1 MB | 16.41 ms | 60.9 FPS | Reference architecture (24 layers, 9.95M params) |
| **Candidate A (best.pt)** | 19.1 MB | 14.36 ms | 69.6 FPS | Identical architecture & runtime |
| **Candidate B (best.pt)** | 19.1 MB | 14.92 ms | 67.0 FPS | Identical architecture & runtime |

*Conclusion*: Zero latency penalty or graph complexity divergence across candidates.

---

## 5. Candidate New Evaluation Footage Identification

Before considering any production deployment, an independent human-reviewed benchmark on previously unseen surveillance footage is required. The following raw video streams exist in `videos/` with substantial unmined frames:

1. **`videos/cam45_northeast.avi` (1.85 GB)**:
   - Holds the out-of-distribution geometry for Northeast camera.
   - Only 12 frames were sampled in `eval_snapshot_v1`; **zero frames exist in training**.
   - Contains > 15,000 unextracted, completely unseen daylight approach frames.
2. **Unmined Infrared & Night Footage**:
   - `videos/cam03_east_night.avi` (1.85 GB)
   - `videos/cam43_south_night.avi` (1.85 GB)
   - `videos/cam44_north_night.avi` (1.85 GB)
   - `videos/cam46_west_night.avi` (1.85 GB)
   - These continuous streams provide dense, unmined queue segments under heavy glare and low-contrast conditions.
3. *Recommendation*: Do not launch another annotation batch yet. Keep Track 2 gated until local synthetic variants are regenerated.

---

## 6. Synthesis, Recommendation & Scientific Gating

### Comparative Summary
1. **Candidate A vs Baseline**:
   - Candidate A trained on 1,092 local frames (where only 18 were human-reviewed, and 1,074 contain unreviewed teacher labels).
   - On Primary Validation: mAP50 changed by **-0.11 pp**.
   - On Diagnostic Benchmark: mAP50 changed by **+0.30 pp**.
2. **Candidate B vs Baseline**:
   - Candidate B added 25 reviewed external frames (813 boxes) to Candidate A.
   - On Primary Validation: mAP50 changed by **+0.12 pp**.
   - On Diagnostic Benchmark: mAP50 changed by **-0.55 pp**.
3. **Candidate B vs Candidate A**:
   - Comparing Candidate B directly against Candidate A isolates the marginal impact of the 25 reviewed external frames.
   - On Primary Validation: mAP50 delta is **+0.23 pp**.
   - On Diagnostic Benchmark: mAP50 delta is **-0.85 pp**.

### Strategic Recommendation
**Recommendation: KEEP DEPLOYED BASELINE.**
- Neither candidate demonstrates a statistically significant or robust generalization breakthrough on independent slices.
- The 25 reviewed external frames are insufficient on their own to offset the 1,074 unreviewed teacher-labeled local frames.
- **Do NOT deploy Candidate A or Candidate B**.
- Preserving the verified baseline (`models/yolo26s_thai_traffic.pt`) ensures operational stability for traffic light control.
