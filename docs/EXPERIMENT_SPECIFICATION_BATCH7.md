# Batch 7: Experiment Specification & Training Readiness Review

- **Status**: Proposal for Review (Do Not Train Yet)
- **Baseline Model**: [`models/yolo26s_thai_traffic.pt`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/models/yolo26s_thai_traffic.pt) (SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`, Preserved as Immutable Reference)
- **Candidate Training Manifests**: [`data/training_manifests_v6/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6)
- **Consolidated Review Pack**: [`data/review_pack_consolidated_v3/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3)

---

## 1. Executive Summary & Strategy Overview

Following human review of the 44-frame sample and the complete resolution of the continuation pack, **43 frames** (18 local CCTV + 25 external UA-DETRAC) are fully verified, complete, and training-eligible. Exactly 1 frame ([`cam44_north_f019140`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/images/cam44_north_f019140.jpg)) remains rejected from the pilot review due to intractable queue occlusions, and **0 frames are blocked**.

To maintain scientific rigor and prevent premature overfitting or training on unreviewed noisy labels, we divide the next phase into two distinct, non-overlapping experiment tracks:

1. **Track 1: Short Pipeline Smoke Test** (Eligible Reviewed Data Only, 5 Epochs)
   - Verifies pipeline mechanics, tensor shapes, loss stability, checkpoint creation, and benchmark logging.
   - Strictly bounded; avoids overfitting on the 43-frame sample (avoiding 30–50 epoch defaults).
2. **Track 2: Meaningful Improvement Experiment** (Manifest A vs Manifest B, 100 Epochs with Early Stopping)
   - Conditioned on explicit quality gates (regenerating stale local synthetic variants and unreviewed label noise protocols).
   - Evaluates whether balanced external data (263 frames) improves vehicle detection and queue tail recall without hurting minority classes.

> [!CAUTION]
> **Do Not Promote Unreviewed External Labels**:
> The 238 unreviewed external candidate frames in Manifest B retain original source labels with known omissions (distant queue vehicles) and unverified van/truck classifications. They must not be treated as complete or deployed without explicit noise safeguards.

---

## 2. Track 1: Short Pipeline Smoke Test Specification

### 2.1 Objective & Rationale
Verify end-to-end execution of the YOLO training pipeline, data loading from consolidated manifests, gradient computation, and validation hooks using only the **43 explicitly eligible, human-reviewed frames**. Running 30–50 epochs on 43 frames would cause severe memorization; Track 1 is strictly capped at **5 epochs**.

### 2.2 Dataset Membership & Splits
- **Training Set (Eligible Reviewed Subset)**:
  - 43 images (18 local CCTV + 25 external UA-DETRAC) from `data/review_pack_consolidated_v3/images/`.
  - Labels from `data/review_pack_consolidated_v3/annotations/labels/`.
  - 100% of images verified on local disk; 100% of bounding boxes synchronized.
- **Validation Set**:
  - Primary Unaugmented Validation: 130 CCTV frames (1,367 boxes) from `data/training_manifests_v6/primary_validation_manifest.json`.
- **Diagnostic Benchmark**:
  - 42-frame evaluation snapshot (`data/eval_snapshot_v1`).
  - *Exposure Caveat*: Internal diagnostic tool with known/unproven exposure, not an independent test set.

### 2.3 Class Support (Track 1 Training Set - Actual Selected Annotations)
| Class ID | Class Name | Box Count | Percentage | Aspect-Preserving Size Breakdown |
| :---: | :--- | :---: | :---: | :--- |
| 0 | `car` | 1,252 | 84.1% | 750 small, 431 medium, 71 large |
| 1 | `motorcycle` | 106 | 7.1% | 94 small, 12 medium, 0 large |
| 2 | `bus` | 99 | 6.7% | 14 small, 35 medium, 50 large |
| 3 | `truck` | 24 | 1.6% | 9 small, 14 medium, 1 large |
| 4 | `three_wheeler` | 7 | 0.5% | 3 small, 4 medium, 0 large |
| **Total** | **All Classes** | **1,488** | **100.0%** | **870 small, 496 medium, 122 large** |

### 2.4 Initialization, Hyperparameters & Training Budget
- **Initialization**: Fine-tune warm start from [`models/yolo26s_thai_traffic.pt`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/models/yolo26s_thai_traffic.pt) (verified SHA256: `cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`).
- **Epoch Budget**: **5 epochs** (batch size 8).
- **Optimizer**: SGD, initial learning rate $\eta_0 = 10^{-4}$, weight decay $0.0005$, momentum $0.937$.
- **Backbone Freezing**: Freeze backbone (first 10 layers, `freeze=10`) to test head adaptation without catastrophic forgetting.
- **Image Resolution**: 640×640 (aspect-preserving letterbox).
- **Candidate Output Path**: `runs/train/smoke_test_batch8/` (never overwriting baseline weights).

### 2.5 Success & Exit Criteria
1. Loss decreases consistently across all 5 epochs with zero NaN/Inf anomalies.
2. Best checkpoint `runs/train/smoke_test_batch7/weights/best.pt` saved successfully.
3. Diagnostic evaluation pass executes and outputs per-class AP without tensor dimension errors.

---

## 3. Track 2: Meaningful Improvement Experiment Specification

### 3.1 Objective & Scientific Hypothesis
Determine whether augmenting local surveillance data with sequence-stratified, balanced external UA-DETRAC frames ($\text{cap} = 263$) significantly improves general vehicle detection and small-vehicle queue recall, without degrading performance on local Thai minority classes (`motorcycle`, `three_wheeler`).

- **Null Hypothesis ($H_0$)**: Manifest B does not improve small-vehicle recall or mAP50 on primary validation by $\ge 1.0\%$ over Manifest A, or degrades motorcycle/three-wheeler mAP50 by $> 1.0\%$.
- **Alternative Hypothesis ($H_1$)**: Manifest B improves small-vehicle recall by $\ge 3.0\%$ while maintaining or improving all minority-class metrics.

### 3.2 Prerequisites & Quality Gates Before Launching Track 2
Track 2 must remain gated until the following preparatory actions are completed:
1. **Gate 1 (Blocked Frames)**: **RESOLVED**. All proposals on `MVI_20063_img00769` and `cam44_north_f148440` are resolved, and both frames are verified.
2. **Gate 2 (Regenerate Stale Local Variants)**: Apply approved annotations to regenerate the 41 quarantined synthetic variants (`truckboost`, `nightboost`, etc.) for the 18 modified local frames.
3. **Gate 3 (External Data Noise Protocol)**: Decide between:
   - *Option A*: Conduct a targeted 30-frame review of the remaining external candidate sequences.
   - *Option B*: Apply soft loss-weighting or label smoothing to unreviewed external candidate boxes during training.

### 3.3 Dataset Membership & Comparison Configurations

| Parameter | Control: Manifest A (Local-Only V6) | Treatment: Manifest B (Local + External V6) |
| :--- | :--- | :--- |
| **Manifest Path** | [`data/training_manifests_v6/manifest_a_local_only.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_a_local_only.json) | [`data/training_manifests_v6/manifest_b_local_plus_external.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_b_local_plus_external.json) |
| **Local Training Images** | 1,092 images (527 unique canonical sources) | 1,092 images (527 unique canonical sources) |
| **External Training Images** | 0 | 263 images (stratified sample, ratio 0.5) |
| **Total Training Images** | **1,092** | **1,355** |
| **Primary Validation Set** | 130 unaugmented images (1,367 boxes) | 130 unaugmented images (1,367 boxes) |
| **Diagnostic Benchmark** | 42 frames (`eval_snapshot_v1`) | 42 frames (`eval_snapshot_v1`) |
| **Candidate Output Path** | `runs/train/candidate_manifest_a_v6/` | `runs/train/candidate_manifest_b_v6/` |

### 3.4 Class & Size Support Breakdown (Manifests V6)

#### Training Set Class Support
| Class Name | Manifest A Boxes | Manifest A % | Manifest B Boxes | Manifest B % | External Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `car` (0) | 9,762 | 74.2% | 12,266 | 73.6% | +2,504 |
| `motorcycle` (1) | 2,752 | 20.9% | 2,752 | 16.5% | 0 (Preserved) |
| `bus` (2) | 185 | 1.4% | 1,029 | 6.2% | +844 |
| `truck` (3) | 398 | 3.0% | 545 | 3.3% | +147 |
| `three_wheeler` (4) | 51 | 0.4% | 51 | 0.3% | 0 (Preserved) |
| **Total Boxes** | **13,148** | **100.0%** | **16,661** | **100.0%** | **+3,513** |

#### Training Set Size Distribution (True Aspect-Preserving Dimensions)
| Size Bucket | Manifest A Boxes | Manifest A % | Manifest B Boxes | Manifest B % |
| :--- | :---: | :---: | :---: | :---: |
| `small` (< 1024 px²) | 7,262 | 55.2% | 8,078 | 48.5% |
| `medium` (1024–9216 px²) | 4,581 | 34.8% | 6,380 | 38.3% |
| `large` (> 9216 px²) | 1,305 | 9.9% | 2,185 | 13.1% |

### 3.5 Proposed Training Budget & Optimization Settings
- **Initialization**: Warm start from [`weights/yolo26s_thai_traffic.pt`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/weights/yolo26s_thai_traffic.pt).
- **Epoch Budget**: Maximum **100 epochs** with **early stopping patience = 15 epochs** based on validation mAP50-95.
- **Batch Size**: 16 (effective batch size 32 with gradient accumulation steps = 2).
- **Learning Rate**: Cosine decay scheduler with $\eta_{\text{max}} = 10^{-3}$, $\eta_{\text{min}} = 10^{-5}$, warmup for 3 epochs.
- **Data Augmentation**: Mosaic ($p=0.5$), MixUp ($p=0.15$), HSV jitter (h=0.015, s=0.7, v=0.4), horizontal flip ($p=0.5$).
- **Seed**: Deterministic seed `42` across both runs.

### 3.6 Evaluation Metrics & Comparison Protocol
1. **Primary Evaluation (130 Unaugmented Local Frames)**:
   - Overall mAP50 and mAP50-95.
   - Per-class AP50 for `car`, `motorcycle`, `bus`, `truck`, `three_wheeler`.
   - Precision and Recall at confidence threshold = 0.25.
2. **Diagnostic Evaluation (42 Benchmark Frames)**:
   - Recall on distant queue tails (small vehicles).
   - Nighttime glare false-positive rate.
   - Confusion matrix analysis: truck vs pickup/van.
3. **Decision Criteria for Promotion**:
   - Manifest B must beat Manifest A on overall mAP50 by $\ge 0.5\%$.
   - Motorcycle and three-wheeler AP must not drop by $> 1.0\%$.
   - Small-vehicle recall on diagnostic benchmark must improve by $\ge 3.0\%$.

---

## 4. Summary of Readiness & Actionable Checklist

| Action Item | Status | Tool / Command |
| :--- | :---: | :--- |
| **Lineage & Hash Verification** | **VERIFIED** | 44 canonical sources verified (0 collisions, strict precedence). |
| **Roboflow External Variant Linking** | **VERIFIED** | 25 eligible external records linked with exact SHA256 matches. |
| **YOLO Label & Inventory Agreement** | **VERIFIED** | Line-for-line, box-for-box agreement asserted across all 43 records. |
| **Aspect-Preserving Size Correction** | **VERIFIED** | Corrected 117 medium $\to$ small and 9 large $\to$ medium local boxes. |
| **Canonical Resampling Guard** | **VERIFIED** | All variants of rejected source (`cam44_north_f019140`) quarantined. |
| **Continuation Frame Proposals** | **ALL ACCEPTED & VERIFIED** | `MVI_20063_img00769` and `cam44_north_f148440` verified; 0 blocked frames. |
| **Execute Track 1 Pipeline Smoke Test** | **READY TO PROCEED** | Awaiting user approval to run 5-epoch smoke test. |
