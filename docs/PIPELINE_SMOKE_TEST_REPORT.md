# YOLO26s Thai Traffic Pipeline Smoke Test Report (Batch 8)

- **Date**: 2026-09-28T02:22:35.251336+00:00
- **Run Identifier**: `smoke_test_batch8`
- **Purpose**: Verify end-to-end training pipeline mechanics, loss stability, checkpoint creation, and diagnostic benchmark evaluation.
- **Candidate Checkpoint**: `E:\Work\Projects\AdaptiveTrafficControl\smart-traffic-vision\runs\train\smoke_test_batch8\weights\best.pt`
- **Candidate SHA256**: `22c3128690e9bb06ad4513d8bfee0a6eef74b37dcdedf981b0f71a16b73213b5`
- **Verified Baseline Checkpoint**: `models/yolo26s_thai_traffic.pt` (`cc579a0387668e204ba79372e9f2380ad6531659fff5fd1d78d7cf282b6b1c2c`)
- **Device**: `NVIDIA GeForce RTX 5060 Laptop GPU` (CUDA: True)
- **Frameworks**: PyTorch `2.12.0.dev20260408+cu128`, Ultralytics `8.4.124`, OpenCV `4.14.0`

---

> [!IMPORTANT]
> **Strict Pipeline Smoke Test Notice**
> This execution is strictly a pipeline sanity verification test across 5 epochs on 43 reviewed frames.
> It **MUST NOT** be cited as evidence of improved generalized vehicle detection or immunity to overfitting.
> Full external-data training (Track 2) remains **gated** pending variant regeneration and label noise safeguards.

---

## 1. Materialized Smoke Test Dataset & Split Separation

A dedicated, isolated dataset was materialized at [`data/smoke_test_dataset_v1/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/smoke_test_dataset_v1) to guarantee zero directory-glob contamination:
- **Training Set (Eligible Reviewed Data Only)**:
  - Total Images: **43** (18 local CCTV + 25 external UA-DETRAC).
  - Excluded Rejected Frames: Exactly 1 frame ([`cam44_north_f019140`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/images/cam44_north_f019140.jpg)) quarantined.
  - Total Approved Boxes: **1,488** (100% geometry validated).
- **Validation Set (Primary Unaugmented CCTV)**:
  - Total Images: **130** (130 clean canonical CCTV frames).
  - Total Ground Truth Boxes: **1,367**.
- **Canonical Source Separation**:
  - Mutual intersection of Train (43), Val (130), and Diagnostic Benchmark (42) = **0 frames (completely disjoint)**.

### Actual Selected Annotations: Class & Size Support

#### Training Set (43 Frames, 1,488 Boxes)
| Class Name | Class ID | Box Count | % of Split | Aspect-Preserving Size Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| `car` | 0 | 1252 | 84.1% | 750 small, 431 medium, 71 large |
| `motorcycle` | 1 | 106 | 7.1% | 94 small, 12 medium, 0 large |
| `bus` | 2 | 99 | 6.7% | 14 small, 35 medium, 50 large |
| `truck` | 3 | 24 | 1.6% | 9 small, 14 medium, 1 large |
| `three_wheeler` | 4 | 7 | 0.5% | 3 small, 4 medium, 0 large |
| **All Classes** | — | **1488** | **100.0%** | **870 small, 496 medium, 122 large** |

#### Validation Set (130 Frames, 1,367 Boxes)
| Class Name | Class ID | Box Count | % of Split | Aspect-Preserving Size Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| `car` | 0 | 1010 | 73.9% | 595 small, 328 medium, 87 large |
| `motorcycle` | 1 | 164 | 12.0% | 132 small, 31 medium, 1 large |
| `bus` | 2 | 23 | 1.7% | 1 small, 10 medium, 12 large |
| `truck` | 3 | 124 | 9.1% | 16 small, 89 medium, 19 large |
| `three_wheeler` | 4 | 46 | 3.4% | 10 small, 17 medium, 19 large |
| **All Classes** | — | **1367** | **100.0%** | **754 small, 475 medium, 138 large** |

---

## 2. Model Architecture & Layer Freezing Specification

- **Total Architecture Modules**: 24 modules (0 to 23).
- **Total Parameters**: 9,951,734
- **Frozen Parameters (`freeze=10`)**: **4,451,008 (44.7%)**
  - Frozen layers 0 through 9: Backbone feature extraction (Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF) + DFL.
- **Trainable Parameters**: **5,500,726 (55.3%)**
  - Trainable layers 10 through 23: Multi-scale PAN-FPN Neck (C2PSA, Upsample, Concat, C3k2) and Detection Head.

---

## 3. Training Execution & Loss Stability

- **Optimizer**: SGD ($	ext{lr}_0 = 0.0001$, momentum = $0.937$, weight decay = $0.0005$)
- **Warm-start Checkpoint**: `models/yolo26s_thai_traffic.pt`
- **Batch Size**: 8 (Initial: 8, OOM Fallback Triggered: False)
- **Resolution**: 640×640 letterboxed
- **Seed**: 42 (deterministic)
- **Training Duration**: 24.77 seconds (5 epochs)

### Epoch Loss Progression Table
| Epoch | Train Box | Train Cls | Train L1 | Val Box | Val Cls | Val L1 | Val mAP50 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.7907 | 2.1536 | 0.0053 | 0.5696 | 0.3769 | 0.0020 | 0.9429 |
| 2 | 1.6649 | 2.1252 | 0.0045 | 0.5855 | 0.3897 | 0.0021 | 0.9395 |
| 3 | 1.6462 | 2.3378 | 0.0051 | 0.5855 | 0.3897 | 0.0021 | 0.9395 |
| 4 | 1.5056 | 2.0029 | 0.0051 | 0.5894 | 0.3963 | 0.0021 | 0.9383 |
| 5 | 1.5493 | 2.1256 | 0.0049 | 0.5976 | 0.4057 | 0.0022 | 0.9367 |

*Observations*:
- Losses and metrics remained strictly finite across all 5 epochs with 0 NaN/Inf anomalies.
- Checkpoints saved successfully to `runs/train/smoke_test_batch8/weights/best.pt` and `last.pt`.
- Baseline checkpoint `models/yolo26s_thai_traffic.pt` remained completely unmodified.

---

## 4. Candidate vs Baseline Diagnostic Benchmark Evaluation

Both checkpoints evaluated on `data/eval_snapshot_v1` (42 frames, 1,174 ground-truth instances) using identical operational and AP evaluation settings:

### Primary Metrics Comparison
| Evaluation Metric | Baseline Reference | Candidate (Batch 8) | Delta | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Ultralytics mAP50** | **0.4644** | **0.4584** | **-0.0060** | Stable |
| **Ultralytics mAP50-95** | **0.3222** | **0.3173** | **-0.0048** | Stable |
| **Operational Precision (conf=0.25)** | **79.6%** | **80.3%** | **+0.7%** | Stable |
| **Operational Recall (conf=0.25)** | **41.8%** | **41.9%** | **+0.1%** | Stable |
| **Operational F1 Score** | **54.8%** | **55.1%** | **+0.2%** | Stable |

### Per-Class AP50 Comparison
| Vehicle Class | Baseline AP50 | Candidate AP50 | Delta | Diagnostic Support Status |
| :--- | :---: | :---: | :---: | :--- |
| `car` (0) | 0.5558 | 0.5555 | -0.0003 | Dense Support (986 GT) |
| `motorcycle` (1) | 0.4742 | 0.4751 | +0.0009 | Moderate Support (132 GT) |
| `bus` (2) | 0.3011 | 0.2965 | -0.0046 | Low Support (21 GT) |
| `truck` (3) | 0.3104 | 0.2820 | -0.0284 | Low Support (18 GT) |
| `three_wheeler` (4) | 0.6805 | 0.6830 | +0.0025 | Low Support (17 GT) |

### Ground-Truth Object Size Recall (conf=0.25, Matching IoU >= 0.50)
| Size Category | Baseline Recall | Candidate Recall | Delta | Benchmark Support |
| :--- | :---: | :---: | :---: | :--- |
| **Small** (< 1024 px²) | 34.9% | 34.8% | -0.1% | 948 GT boxes |
| **Medium** (1024–9216 px²) | 70.6% | 71.7% | +1.1% | 180 GT boxes |
| **Large** (> 9216 px²) | 71.7% | 71.7% | +0.0% | 46 GT boxes |

---

## 5. Acceptance & Quality Gates Verification

| Verification Item | Acceptance Requirement | Result | Evidence |
| :--- | :--- | :---: | :--- |
| **Baseline Path & Hash** | `models/yolo26s_thai_traffic.pt` matching `cc579a0...` | **PASSED** | Verified byte-for-byte before and after run |
| **Dataset Membership** | Exactly 43 eligible reviewed frames; 1 rejected excluded | **PASSED** | 43 images & 1,488 approved boxes in `train` |
| **Canonical Source Isolation** | 0% overlap between train, val, and eval sources | **PASSED** | Disjoint sets asserted across all 215 sources |
| **Loss & Gradient Health** | Finite losses across all 5 epochs, zero NaN/Inf | **PASSED** | Verified from `runs/train/smoke_test_batch8/results.csv` |
| **Checkpoint Generation** | Valid, non-empty candidate checkpoints | **PASSED** | `best.pt` generated with unique SHA256 |
| **Diagnostic Evaluation** | Benchmark run with candidate mode | **PASSED** | 42 frames evaluated; candidate report generated |
| **Asset Immutability** | Baseline weights, review packs, manifests unchanged | **PASSED** | All reference hashes verified identical |

---

## 6. Next Steps & Gating Constraints

- **Do NOT deploy candidate model**: This 5-epoch smoke test was conducted on 43 frames exclusively to validate execution mechanics.
- **Do NOT run full Manifest A/B training yet**: Full external data training remains strictly gated until:
  1. The 41 quarantined synthetic variants for modified local frames are regenerated.
  2. The external data noise protocol is confirmed.
