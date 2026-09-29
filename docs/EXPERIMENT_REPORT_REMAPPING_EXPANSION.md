# External-Data Remapping & Expansion Experiment Report

- **Date**: 2026-09-29T07:10:31.273183+00:00
- **Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Candidate R25 Checkpoint (best.pt)**: `E:\Work\Projects\AdaptiveTrafficControl\smart-traffic-vision\runs\train\candidate_r25_remapped_external\weights\best.pt` (SHA256: `fc0ec4f7a8ef2f70dcc20e193c40fd27a3bd80125f734b0f1a5c934232b86136`)
- **Candidate R100 Checkpoint (best.pt)**: `E:\Work\Projects\AdaptiveTrafficControl\smart-traffic-vision\runs\train\candidate_r100_expanded_external\weights\best.pt` (SHA256: `3827f93ca0b5b80f6a3ac76c1bcc387cae300a29151d45b5eefa44bb4984afb8`)
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (PyTorch 2.12.0.dev20260408+cu128, CUDA 12.8, Ultralytics 8.4.124)

---

## 1. Experimental Rationale & Taxonomy Mapping

### Hypothesis
External car/van/truck distinctions conflict with our light-vehicle taxonomy (where pickup trucks and commuter vans belong strictly to `car` (0), while `truck` (3) is reserved for heavy commercial freight trucks).
This experiment evaluates:
1. **Remapping Hypothesis (B25 vs R25)**: Merging external car, van, and truck categories into `car` (0) while retaining `bus` (2) reduces taxonomy confusion without sacrificing heavy truck detection.
2. **Expansion Hypothesis (R25 vs R100)**: Expanding the external training set from 25 to 100 frames via high-capacity teacher completion (`models/yolo26x.pt` with multi-scale tiling) improves vehicle generalization and small-object detection.

> [!NOTE]
> This is an explicitly authorized experimental remapping, not a claim that all external vehicles are identical. The teacher model (`yolo26x.pt`) lacks a dedicated `three_wheeler` class; this limitation is tracked.

### External Mapping Rules (Applied Strictly to Copied External Data)
- **Source UA-DETRAC Annotations**:
  - `bus` (0) $\to$ `bus` (2)
  - `car` (1), `truck` (2), `van` (3) $\to$ `car` (0)
- **YOLO26x Teacher Proposals**:
  - COCO `car` (2), `truck` (7) $\to$ `car` (0)
  - COCO `bus` (5) $\to$ `bus` (2)
  - COCO `motorcycle` (3) $\to$ `motorcycle` (1)
  - Non-vehicle COCO classes discarded.
- **Source Artifact Protection**:
  - Local training annotations, validation sets, and diagnostic ground truth are **never remapped**.
  - Exactly 5 truck boxes in the 25 reviewed external frames were remapped from class 3 to class 0 in the copied experiment files.

---

## 2. Dataset Composition & Disjointness Verification

| Partition | Local Frames | External Frames | Total Frames | Total Bounding Boxes | Canonical Sources | External Label Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dataset A (Batch 9)** | 1,092 | 0 | 1,092 | 13,148 | 527 | N/A (Local CCTV only) |
| **Dataset B (Batch 9)** | 1,092 | 25 | 1,117 | 13,961 | 552 | 25 human-reviewed (original taxonomy) |
| **Candidate R25** | 1,092 | 25 | 1,117 | 13,961 | 552 | 25 human-reviewed (**remapped**: 5 trucks $\to$ car) |
| **Candidate R100** | 1,092 | 100 | 1,192 | 15,832 | 627 | 25 reviewed + 75 teacher-completed |
| **Primary Val Benchmark** | 130 | 0 | 130 | 1,367 | 130 | Verified CCTV (100% disjoint) |
| **Diagnostic Benchmark** | 42 | 0 | 42 | 1,174 | 42 | Verified CCTV (100% disjoint) |

- **Strict Nested Property**: The 25 reviewed external frames are strictly a subset of the 100 external frames in R100.
- **Canonical Disjointness**: $\text{Train R100} \cap \text{Primary Val} = \emptyset$; $\text{Train R100} \cap \text{Diagnostic Benchmark} = \emptyset$. Mutual canonical overlap is **0.0%**.

---

## 3. Training Configurations & Convergence

Both candidates were independently warm-started from `models/yolo26s_thai_traffic.pt` matching Batch 9:
- **Optimizer**: SGD, $\text{lr}_0=0.0001, \text{lrf}=0.01$, Cosine schedule
- **Explicit Warmup**: 1 epoch, momentum 0.8, **warmup_bias_lr=0.0001** (locks bias LR to avoid instability)
- **Frozen Layers**: Layers 0–9 (backbone frozen, 4.45M params)
- **Trainable Layers**: Layers 10–23 (neck & head, 5.50M params)
- **Batch Size**: 16 (0 OOM fallbacks)
- **Resolution**: 640×640 letterboxed

| Metric | Candidate R25 | Candidate R100 | Delta (R100 vs R25) |
| :--- | :---: | :---: | :---: |
| **Total Images** | 1,117 | 1,192 | +75 images (+6.7%) |
| **Total Boxes** | 13,961 | 15,832 | +1871 boxes |
| **Optimizer Steps** | 2100 steps | 2250 steps | +150 steps |
| **Training Duration** | 657.00 s | 686.56 s | +29.56 s |

---

## 4. Evaluation Benchmark Results

### 4.1 Primary Validation Benchmark (130 Frames, 1,367 Ground Truth Boxes)

| Metric | Deployed Baseline | Batch 9 Cand A (best) | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 (Remap) | R100 vs R25 (Expand) | R100 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **0.9440** | **0.9429** | **0.9452** | **0.9448** | **0.9430** | **-0.04 pp** | **-0.19 pp** | **-0.11 pp** |
| **mAP50-95** | **0.8386** | **0.8396** | **0.8416** | **0.8423** | **0.8373** | **+0.07 pp** | **-0.50 pp** | **-0.13 pp** |
| **Operational F1** | **85.5%** | **83.6%** | **84.4%** | **84.7%** | **83.7%** | **+0.30 pp** | **-0.97 pp** | **-1.86 pp** |
| **Operational Precision** | 81.3% | 77.2% | 79.0% | 79.6% | 77.6% | +0.64 pp | -2.02 pp | -3.72 pp |
| **Operational Recall** | 90.2% | 91.2% | 90.5% | 90.3% | 90.8% | -0.15 pp | +0.44 pp | +0.58 pp |
| **Agnostic F1** | 87.6% | 85.6% | 86.3% | 86.6% | 85.9% | +0.31 pp | -0.66 pp | -1.71 pp |
| **Agnostic Precision** | 83.3% | 79.0% | 80.8% | 81.4% | 79.7% | +0.65 pp | -1.76 pp | -3.64 pp |
| **Agnostic Recall** | 92.4% | 93.4% | 92.5% | 92.4% | 93.2% | -0.15 pp | +0.81 pp | +0.81 pp |
| `car` AP50 | 0.9398 | 0.9399 | 0.9392 | 0.9367 | 0.9398 | -0.25 pp | +0.31 pp | 0.00 pp |
| `motorcycle` AP50 | 0.9313 | 0.9265 | 0.9321 | 0.9324 | 0.9336 | +0.03 pp | +0.12 pp | +0.23 pp |
| `bus` AP50 | 0.9511 | 0.9547 | 0.9601 | 0.9579 | 0.9461 | -0.22 pp | -1.18 pp | -0.50 pp |
| `truck` AP50 | 0.9049 | 0.9041 | 0.9090 | 0.9107 | 0.9061 | +0.17 pp | -0.46 pp | +0.12 pp |
| `three_wheeler` AP50 | 0.9930 | 0.9894 | 0.9856 | 0.9864 | 0.9892 | +0.08 pp | +0.28 pp | -0.38 pp |
| Small Recall (< 1024 px²) | 88.2% | 89.4% | 87.9% | 87.5% | 89.5% | -0.40 pp | +1.99 pp | +1.32 pp |
| Medium Recall | 92.2% | 92.6% | 93.0% | 93.0% | 91.8% | 0.00 pp | -1.26 pp | -0.42 pp |
| Large Recall | 94.2% | 96.4% | 95.7% | 96.4% | 94.2% | +0.73 pp | -2.18 pp | 0.00 pp |
| FP / Frame | 1.88 | 2.53 | 2.25 | 2.14 | 2.43 | -0.11 | +0.29 | +0.55 |

---

### 4.2 Human-Reviewed Diagnostic Benchmark (42 Frames, 1,174 Ground Truth Boxes)

| Metric | Deployed Baseline | Batch 9 Cand A (best) | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 (Remap) | R100 vs R25 (Expand) | R100 vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mAP50** | **0.4644** | **0.4674** | **0.4589** | **0.4611** | **0.4598** | **+0.23 pp** | **-0.14 pp** | **-0.46 pp** |
| **mAP50-95** | **0.3222** | **0.3217** | **0.3109** | **0.3123** | **0.3156** | **+0.15 pp** | **+0.33 pp** | **-0.66 pp** |
| **Operational F1** | **54.8%** | **56.1%** | **56.1%** | **56.3%** | **55.9%** | **+0.18 pp** | **-0.41 pp** | **+1.02 pp** |
| **Operational Precision** | 79.6% | 77.0% | 76.7% | 76.8% | 77.4% | +0.18 pp | +0.54 pp | -2.20 pp |
| **Operational Recall** | 41.8% | 44.1% | 44.2% | 44.4% | 43.7% | +0.17 pp | -0.68 pp | +1.88 pp |
| **Agnostic F1** | 59.7% | 61.0% | 60.7% | 60.8% | 60.9% | +0.08 pp | +0.06 pp | +1.12 pp |
| **Agnostic Precision** | 86.7% | 83.7% | 83.0% | 83.0% | 84.3% | +0.03 pp | +1.27 pp | -2.40 pp |
| **Agnostic Recall** | 45.6% | 48.0% | 47.9% | 48.0% | 47.6% | +0.09 pp | -0.35 pp | +2.04 pp |
| `car` AP50 | 0.5558 | 0.5896 | 0.5883 | 0.5875 | 0.5692 | -0.08 pp | -1.83 pp | +1.34 pp |
| `motorcycle` AP50 | 0.4742 | 0.4893 | 0.4986 | 0.5030 | 0.4598 | +0.44 pp | -4.32 pp | -1.44 pp |
| `bus` AP50 | 0.3011 | 0.3042 | 0.3153 | 0.3153 | 0.3006 | 0.00 pp | -1.47 pp | -0.05 pp |
| `truck` AP50 | 0.3104 | 0.2604 | 0.2247 | 0.2335 | 0.2865 | +0.88 pp | +5.30 pp | -2.39 pp |
| `three_wheeler` AP50 | 0.6805 | 0.6935 | 0.6674 | 0.6665 | 0.6829 | -0.09 pp | +1.64 pp | +0.24 pp |
| Small Recall (< 1024 px²) | 34.9% | 37.3% | 37.0% | 37.1% | 37.1% | +0.10 pp | 0.00 pp | +2.21 pp |
| Medium Recall | 70.6% | 71.7% | 74.4% | 74.4% | 71.1% | 0.00 pp | -3.33 pp | +0.55 pp |
| Large Recall | 71.7% | 76.1% | 73.9% | 76.1% | 71.7% | +2.18 pp | -4.35 pp | 0.00 pp |
| FP / Frame | 0.86 | 1.29 | 1.48 | 1.50 | 1.05 | +0.02 | -0.45 | +0.19 |

---

### 4.3 Environmental & Historical Lineage Slices (Operational Recall at conf=0.25)

| Slices & Subpopulations | Deployed Baseline | Batch 9 Cand B (best) | Candidate R25 (best) | Candidate R100 (best) | R25 vs B25 | R100 vs R25 | R100 vs Base |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Val: Day Scenes** | 89.7% | 90.0% | 90.1% | 90.6% | +0.09 pp | +0.46 pp | +0.83 pp |
| **Primary Val: Night Scenes** | 92.0% | 92.4% | 91.3% | 91.6% | -1.09 pp | +0.37 pp | -0.36 pp |
| **Primary Val: Old Train Split (107 frames)** | 92.8% | 92.7% | 92.3% | 93.4% | -0.37 pp | +1.12 pp | +0.65 pp |
| **Primary Val: Old Val Split (23 frames)** | 81.2% | 82.9% | 83.5% | 81.6% | +0.66 pp | -1.97 pp | +0.33 pp |
| **Diagnostic: Day Scenes** | 41.7% | 43.1% | 43.3% | 43.3% | +0.21 pp | 0.00 pp | +1.60 pp |
| **Diagnostic: Night Scenes** | 42.4% | 48.7% | 48.7% | 45.3% | 0.00 pp | -3.39 pp | +2.97 pp |
| **Diagnostic: Northeast Holdout (12 frames)** | 35.8% | 40.3% | 39.8% | 36.7% | -0.45 pp | -3.09 pp | +0.89 pp |
| **Diagnostic: In-Dist Cams (30 frames)** | 43.2% | 45.1% | 45.5% | 45.4% | +0.31 pp | -0.10 pp | +2.11 pp |

---

### 4.4 Pickup / Passenger Van vs Truck Confusion Analysis (Operational conf=0.25)

| Benchmark | Confusion Type | Baseline | Batch 9 B25 | Cand R25 | Cand R100 | Delta (R25 vs B25) | Delta (R100 vs R25) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Val (130 frames)** | GT Car pred as Truck (False Truck) | 14 | 15 | 16 | 16 | +1 | +0 |
| **Primary Val (130 frames)** | GT Truck pred as Car (Missed Truck) | 11 | 10 | 9 | 11 | -1 | +2 |
| **Diagnostic Benchmark (42 frames)** | GT Car pred as Truck (False Truck) | 24 | 25 | 25 | 25 | +0 | +0 |
| **Diagnostic Benchmark (42 frames)** | GT Truck pred as Car (Missed Truck) | 2 | 2 | 2 | 2 | +0 | +0 |

---

### 4.5 Checkpoint Selection: best.pt vs last.pt

| Model Checkpoint | Primary Val mAP50 | Primary Val mAP50-95 | Diag Benchmark mAP50 | Diag Benchmark mAP50-95 | Checkpoint Selection Criteria |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Candidate R25 (best.pt)** | **0.9448** | **0.8423** | **0.4611** | **0.3123** | Peak validation fitness ($0.1 \text{mAP50} + 0.9 \text{mAP50-95}$) |
| **Candidate R25 (last.pt)** | 0.9382 | 0.8283 | 0.4678 | 0.3142 | Final epoch 30 weights |
| **Candidate R100 (best.pt)** | **0.9430** | **0.8373** | **0.4598** | **0.3156** | Peak validation fitness ($0.1 \text{mAP50} + 0.9 \text{mAP50-95}$) |
| **Candidate R100 (last.pt)** | 0.9324 | 0.8226 | 0.4624 | 0.3101 | Final epoch 30 weights |

---

## 5. Hardware Inference Latency Benchmark

Evaluated on NVIDIA GeForce RTX 5060 Laptop GPU across 100 timed iterations (640×640 input resolution):

| Model Checkpoint | File Size | Mean Latency (ms / image) | Throughput (FPS) | Computational Equivalence |
| :--- | :---: | :---: | :---: | :--- |
| **Deployed Baseline** | 19.1 MB | 14.15 ms | 70.7 FPS | Reference architecture (24 layers, 9.95M params) |
| **Candidate R25 (best.pt)** | 19.1 MB | 13.73 ms | 72.8 FPS | Identical architecture & runtime |
| **Candidate R100 (best.pt)** | 19.1 MB | 14.05 ms | 71.2 FPS | Identical architecture & runtime |

---

## 6. Synthesis, Distinguishing Conclusions & Scientific Gating

### Distinguishing Conclusions
1. **Original B25 vs Candidate R25 (Testing Remapping Alone)**:
   - Evaluates whether remapping the external truck annotations into `car` (0) helps alignment.
   - On Primary Validation: mAP50 delta is **-0.04 pp**; mAP50-95 delta is **+0.07 pp**.
   - On Diagnostic Benchmark: mAP50 delta is **+0.23 pp**; mAP50-95 delta is **+0.15 pp**.
2. **Candidate R25 vs Candidate R100 (Testing Expansion with Machine-Completed Labels)**:
   - Evaluates the effect of adding 75 pseudo-labeled external frames.
   - Volume and annotation quality are **not independently isolated** in this comparison (both data volume and teacher noise increase together).
   - On Primary Validation: mAP50 delta is **-0.19 pp**; mAP50-95 delta is **-0.50 pp**.
   - On Diagnostic Benchmark: mAP50 delta is **-0.14 pp**; mAP50-95 delta is **+0.33 pp**.

### Validation Lineage & Label Quality Caveats
- **Lineage Exposure**: 107 of the 130 Primary Validation frames originated from the historical training split of the CCTV dataset.
- **Machine Label Status**: All 75 expanded frames in R100 are machine-labeled via YOLO26x; automated completion does not guarantee exhaustive ground truth.

### Strategic Recommendation
**Recommendation**: Maintain deployed baseline (`models/yolo26s_thai_traffic.pt`).
Neither R25 nor R100 demonstrates an unequivocal generalization breakthrough across both benchmarks. Do NOT deploy Candidate R25 or Candidate R100.
