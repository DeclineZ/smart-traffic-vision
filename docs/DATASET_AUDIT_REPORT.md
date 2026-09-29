# Thai Traffic Vision & YOLO26s Dataset Audit Report (Batch 1)

> **Audit Execution Date**: 2026-09-27  
> **Reference Resolution**: 640×640 (Aspect-Preserving Letterbox Scaling)  
> **Edge Rounding Tolerance**: ±1e-03 (Documented Coordinate Precision)  
> **Preserved Standards**: Thai 5-Class COCO-aligned specification (`0: car`, `1: motorcycle`, `2: bus`, `3: truck`, `4: three_wheeler`).

---

## 1. Executive Summary

This audit provides an exhaustive, read-only baseline verification of all local and external data sources for the upcoming YOLO26s fine-tuning cycle. All statements below are derived strictly from measured results:

| Metric / Component | Verified Baseline | Audit Finding |
| :--- | :--- | :--- |
| **External UA-DETRAC Image Records** | 16,584 | **Independently reproduced**: Exactly 16,584 image records. |
| **External UA-DETRAC Source Frames** | 9,716 | **Independently reproduced**: Exactly 9,716 canonical source frames. |
| **External Sequence Leakage** | Leakage Audit | **Cross-Split Leakage**: 100 of 100 sequences appear across splits. |
| **Compiled Dataset Total Images** | 1,397 | **1,278 train** + **119 val** images across measured approaches. |
| **Compiled Dataset Total Boxes** | 16,261 | **14,908 train** + **1,353 val** boxes across 5 classes. |
| **Local Split Integrity** | Measured Metrics | **0 exact frame leaks**, **0 temporal leaks (<= 3.0s)**. Minimum measured temporal separation: 480 frames (3.20s @ 150.0 FPS in cam43_south). |
| **Local Staging Verified Hits** | 990 crops | **989 matched**, **1 unmatched** across indexed manifests. |
| **Stashed Unmatched Crops** | 5 crops | Located in `data/unmatched_crops_stash`. |
| **Holdout Camera Approach** | cam45_northeast | **Sampled in evaluation manifest**; 0 images in current training split (checkpoint lineage unproven). |

> [!CAUTION]
> **Scientific Label Quality Disclaimer**: Filename-based checks and manifest indexes audit *dataset mechanics, pipeline lineage, and split integrity*. They do **NOT** constitute proof of visual label correctness, bounding box tightness, or ground-truth annotation accuracy. Visual quality requires visual inspection against raw pixel footage.

> [!NOTE]
> **Vehicle Independence Context**: Passing a 3.0-second temporal proximity check prevents adjacent-frame video burst leakage, but does **not** prove vehicle-level independence without trajectory tracking. Vehicles in queues, red-light stops, or dense platoons may persist across minutes.

---

## 2. Automated External Dataset Audit: UA-DETRAC NDJSON

- **NDJSON File**: `data\usdetrac\ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson`
- **Total Image Records**: **16,584**
- **Total Canonical Source Frames (pre-Roboflow hash)**: **9,716**
- **Total Unique Video Sequences**: **100** (`MVI_xxxxx`)
- **Malformed Records Detected**: **0**

### 2.1 Split and Box Distribution (Source Classes)

| Split | Images | Source Bus (0) | Source Car (1) | Source Truck (2) | Source Van (3) | Total Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **train** | 13,665 | 2,117 | 103,221 | 9,610 | 10,579 | 125,527 |
| **val** | 2,437 | 393 | 18,365 | 1,633 | 1,842 | 22,233 |
| **test** | 482 | 67 | 3,575 | 343 | 377 | 4,362 |

### 2.2 Source-Class vs Target-Class Distinction

| Source Class (UA-DETRAC) | Source Class ID | Target 5-Class Name | Target Class ID | Semantic Alignment Notes |
| :--- | :---: | :--- | :---: | :--- |
| `bus` | `0` | **`bus`** | `2` | Aligned (transit buses, coaches). |
| `car` | `1` | **`car`** | `0` | Aligned (sedans, SUVs, hatchbacks). |
| `truck` | `2` | **`truck`** | `3` | Aligned (commercial trucks). |
| `van` | `3` | **`car`** *(unresolved)* | `0` | Guide aligns passenger commuter vans to `0: car`; code coerces vans to car; external keeps distinct `3: van`. |
| *N/A* | *N/A* | **`motorcycle`** | `1` | Not present in UA-DETRAC. |
| *N/A* | *N/A* | **`three_wheeler`** | `4` | Not present in UA-DETRAC. |

### 2.3 Cross-Split Sequence Leakage Finding

Roboflow augmentation hashes (`.rf.<hash>`) disguise frame-level splitting. When stripping augmentation hashes to uncover underlying sequence provenance:

| Split Pair | Common Source Frames | Common Sequences (`MVI_xxxxx`) | Sequence Overlap Rate |
| :--- | :---: | :---: | :---: |
| **test vs train** | 0 | **100 / 100** | **100.0% Overlap** |
| **test vs val** | 0 | **100 / 100** | **100.0% Overlap** |
| **train vs val** | 0 | **100 / 100** | **100.0% Overlap** |

> [!WARNING]
> **External Sequence Leakage**: Every single video sequence in UA-DETRAC is present in all three splits (`train`, `val`, `test`). Model validation on UA-DETRAC val/test evaluates memorization of vehicles in known traffic streams rather than generalization to new sequences.

### 2.4 External Object-Size Distribution (Aspect-Preserving Resize at 640×640)

| Source Class | Total Boxes | Small (< 32² px) | Medium (32² - 96² px) | Large (> 96² px) | Median Size (W × H) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **car** | 125,161 | 13.6% (16,971) | 69.4% (86,837) | 17.1% (21,353) | 53.9 × 58.0 px |
| **truck** | 11,586 | 10.3% (1,194) | 60.4% (6,997) | 29.3% (3,395) | 58.7 × 76.0 px |
| **van** | 12,798 | 0.6% (71) | 27.2% (3,475) | 72.3% (9,252) | 120.0 × 129.0 px |
| **bus** | 2,577 | 0.6% (15) | 42.8% (1,103) | 56.6% (1,459) | 103.5 × 134.8 px |

---

## 3. Automated Compiled Dataset Audit: data/multiclass_dataset

### 3.1 Overview and Pairing Integrity

- **Total Images**: **1,397**
- **Total Labels**: **1,397** (Unpaired images/labels: **0**)
- **Total Boxes**: **16,261** across 5 classes
- **Malformed Records**: **0**
- **Conflicting Labels (IoU > 0.5 with conflicting classes)**: **0**
- **Measured Image Dimensions**: {(1920, 1080): 1397} (Unknown dimensions: 0)

### 3.2 Breakdown by Split and Class

| Class ID | Class Name | Train Boxes | Val Boxes | Total Boxes | Train Share | Val Share |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0` | **car** | 10,623 | 945 | 11,568 | 71.3% | 69.8% |
| `1` | **motorcycle** | 1,762 | 169 | 1,931 | 11.8% | 12.5% |
| `2` | **bus** | 434 | 39 | 473 | 2.9% | 2.9% |
| `3` | **truck** | 1,494 | 139 | 1,633 | 10.0% | 10.3% |
| `4` | **three_wheeler** | 595 | 61 | 656 | 4.0% | 4.5% |
| **Total** | | **14,908** | **1,353** | **16,261** | 100.0% | 100.0% |

### 3.3 Provenance and Variant Breakdown

Distinguishes original curated frames from synthetic oversampling variants and replay frames:

| Provenance Variant | Train Images | Val Images | Train Boxes | Val Boxes | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`busboost`** | 70 | 0 | 968 | 0 | Targeted 3× photometric jitter for Bus minority class |
| **`nightboost`** | 172 | 0 | 2,124 | 0 | 3× photometric jitter for real nighttime CCTV frames |
| **`original_curated`** | 524 | 107 | 6,011 | 1,221 | Human-curated CCTV frame base hits |
| **`replay`** | 44 | 12 | 455 | 132 | Negative/positive background frames sampled from raw CCTV feeds |
| **`salengboost`** | 141 | 0 | 1,554 | 0 | Targeted 4× photometric jitter for minority Saleng class |
| **`synth_ir_night`** | 40 | 0 | 488 | 0 | Synthetic monochrome infrared converted daytime frames |
| **`truckboost`** | 116 | 0 | 1,768 | 0 | Targeted 3× photometric jitter for Heavy Truck trailer class |
| **`tuktukboost`** | 171 | 0 | 1,540 | 0 | Targeted 2× photometric jitter for Tuk-Tuk class |

### 3.4 Camera Approach and Day/Night Lighting Distribution

Distinguishes authentic real night footage from synthetic monochrome IR augmentation:

| Camera | Real Day | Real Night | Synthetic IR | Total Images | Total Boxes | Split Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`cam03_east`** | 138 | 7 | 3 | 148 | 1,375 | 132 train, 16 val |
| **`cam43_south`** | 209 | 17 | 4 | 230 | 3,922 | 198 train, 32 val |
| **`cam44_north`** | 224 | 280 | 15 | 519 | 7,981 | 492 train, 27 val |
| **`cam46_west`** | 407 | 75 | 18 | 500 | 2,983 | 456 train, 44 val |

### 3.5 Cross-Split Overlap & Leakage Analysis

- **Exact Source-Frame Overlap between Train and Val**: **0 frames**.
- **Temporal Proximity Leaks (<= 3.0s between train and val frames)**: **0 pairs**.
- **Minimum Measured Temporal Separation**: **480 video frames (3.20s @ 150.0 FPS)** in `cam43_south` between val frame 121860 and train frame 121380.
- **Camera Approach Overlap**: ['cam03_east', 'cam43_south', 'cam44_north', 'cam46_west'].
- **Unseen Camera Approach (`cam45_northeast`)**: 0 images in train, 0 images in val.

### 3.6 Object-Size Distribution (Aspect-Preserving Resize at 640×640)

| Class | Split | Total Boxes | Small (< 32² px) | Medium (32² - 96² px) | Large (> 96² px) | Median Size (W × H) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **car** | train | 10,623 | 58.5% (6214) | 32.7% (3473) | 8.8% (936) | 34.2 × 24.6 px |
| **motorcycle** | train | 1,762 | 79.4% (1399) | 20.1% (354) | 0.5% (9) | 20.0 × 24.6 px |
| **bus** | train | 434 | 8.8% (38) | 62.9% (273) | 28.3% (123) | 52.2 × 43.3 px |
| **truck** | train | 1,494 | 20.4% (305) | 58.0% (867) | 21.6% (322) | 58.5 × 48.7 px |
| **three_wheeler** | train | 595 | 27.4% (163) | 56.6% (337) | 16.0% (95) | 42.7 × 44.0 px |
| **car** | val | 945 | 59.6% (563) | 30.3% (286) | 10.2% (96) | 34.1 × 24.3 px |
| **motorcycle** | val | 169 | 65.1% (110) | 33.1% (56) | 1.8% (3) | 20.1 × 24.4 px |
| **bus** | val | 39 | 20.5% (8) | 38.5% (15) | 41.0% (16) | 81.0 × 64.3 px |
| **truck** | val | 139 | 22.3% (31) | 59.0% (82) | 18.7% (26) | 57.4 × 46.3 px |
| **three_wheeler** | val | 61 | 27.9% (17) | 57.4% (35) | 14.8% (9) | 43.7 × 45.3 px |

---

## 4. Automated Local Staging & Mining Manifest Audit

| Staging Category | Seeds | Manifest Candidate Targets | Verified Hits | Matched Hits | Unmatched Hits |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`saleng`** | 30 | 276 | 71 | 71 | **0** |
| **`pickup`** | 73 | 483 | 319 | 318 | **1** |
| **`truck_trailer`** | 40 | 404 | 95 | 95 | **0** |
| **`van`** | 27 | 322 | 165 | 165 | **0** |
| **`bus`** | 43 | 437 | 76 | 76 | **0** |
| **`tuktuk`** | 29 | 9,292 | 264 | 264 | **0** |
| **`songthaew`** | 10 | 206 | 0 | 0 | **0** |

### 4.1 Programmatic Unmatched Verified Crop Diagnosis

Found **1 unmatched crop** in `verified_hits`:
- **File**: `crop_cam46_west_f001800_yolo_7_3_sim0.77.jpg` in `data/pickup/verified_hits/`
  - **Diagnostic Finding**: Frame key 'cam46_west_f001800' exists in manifest with targets: ['crop_cam46_west_f001800_yolo_2_0_sim0.74.jpg', 'crop_cam46_west_f001800_yolo_2_1_sim0.77.jpg', 'crop_cam46_west_f001800_yolo_7_2_sim0.82.jpg', 'crop_cam46_west_f001800_yolo_2_3_sim0.68.jpg', 'crop_cam46_west_f001800_yolo_5_4_sim0.82.jpg', 'crop_cam46_west_f001800_yolo_7_5_sim0.77.jpg', 'crop_cam46_west_f001800_yolo_7_0_sim0.72.jpg', 'crop_cam46_west_f001800_yolo_5_1_sim0.72.jpg', 'crop_cam46_west_f001800_yolo_7_0_sim0.75.jpg', 'crop_cam46_west_f001800_yolo_5_1_sim0.75.jpg', 'crop_cam46_west_f001800_yolo_2_0_sim0.72.jpg', 'crop_cam46_west_f001800_yolo_2_1_sim0.77.jpg', 'crop_cam46_west_f001800_yolo_7_2_sim0.77.jpg']

### 4.2 Stashed Crops in `data/unmatched_crops_stash`

Found **5 crops** in `data/unmatched_crops_stash`:
- `crop_cam44_north_f141480_b23_sim0.69.jpg` (Frame key `cam44_north_f141480` indexed in manifest: False)
- `crop_cam44_north_f141480_b7_sim0.69.jpg` (Frame key `cam44_north_f141480` indexed in manifest: False)
- `crop_cam44_north_f141540_b17_sim0.66.jpg` (Frame key `cam44_north_f141540` indexed in manifest: False)
- `crop_cam44_north_f141540_b33_sim0.66.jpg` (Frame key `cam44_north_f141540` indexed in manifest: False)
- `crop_cam44_north_f141540_b7_sim0.66.jpg` (Frame key `cam44_north_f141540` indexed in manifest: False)

### 4.3 Cross-Category Overlap and Duplicate Hits

- **Verified Hit Folder Duplication**: **1 crop** present across multiple verified folders:
  - `crop_cam44_north_night_f004320_yolo_2_0_sim0.71.jpg` is present in ['pickup', 'van'].
- **Candidate Mining Duplication**: **28 candidate crops** appear across multiple candidate manifests.

---

## 5. Automated Evaluation Sampling Manifest & Unified Candidate Registry

- **Reviewable Manifest File**: `docs/eval_sampling_manifest.json` (Tracked in Git)
- **Runtime Manifest File**: `data/eval_sampling_manifest.json`
- **Total Candidates Registered**: **135 frames** (Unique canonical source IDs)
- **Total Selected Samples**: **42 frames**
- **Classes Covered**: ['bus', 'car', 'motorcycle', 'three_wheeler', 'truck']
- **Classes Missing**: []
- **Cameras Covered**: ['cam03_east', 'cam43_south', 'cam44_north', 'cam45_northeast', 'cam46_west']
- **Cameras Missing**: []
- **Real Night CCTV Frames**: 8
- **Small Object Frames (< 32² px evidenced)**: 26
- **Exposure Breakdown**: {'unproven_checkpoint_exposure': 35, 'nearby_training_exposure': 7}
- **Annotation Status Breakdown**: {'unreviewed_candidate_and_teacher_annotations': 26, 'missing_unannotated': 8, 'unreviewed_historical_predictions': 8}
- **Clean Input Status Breakdown**: {'existing_raw_image': 26, 'requires_video_extraction': 16}
- **Quantitative Benchmark Eligibility**: {'eligible_for_quantitative_eval_count': 0, 'ineligible_unreviewed_count': 42, 'policy_note': 'Unreviewed annotations are strictly ineligible for final quantitative evaluation scoring until verified by human review. Clean image inputs must be verified on disk or extracted from source video.'}

---

## 6. Manual Specification vs. Executable Code Review

> *Note: This section documents architectural and taxonomy discrepancies identified via manual code and guide review, kept strictly separate from the automated data measurements above.*

### 6.1 Pickup and Songthaew Subtype Class Mapping [CRITICAL]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md (line 7) explicitly classifies กระบะ (Hilux, D-Max), รถคอก, and สองแถว (songthaew) under Class 3: truck. train_yolo26s.py docstring (line 7) also states: '3: truck (pickup, delivery box pickup, flatbed, 6/10/18-wheeler)'.
- **Executable Implementation**: tools/compile_multiclass_dataset.py (lines 48-59, 370-377) maps both 'pickup' and 'songthaew' to class_id: 0 (car), citing 1.0 Passenger Car Equivalent (PCE) traffic controller alignment. Teacher detections predicting truck/bus on pickup proposals are forcibly coerced to class 0.
- **Engineering Impact**: Discrepancy directly splits semantic labeling: the guide and training docstring define pickups as trucks, while dataset compilation forces them into the car class. Requires definitive engineering alignment.

### 6.2 Teacher Model Confidence Policy [HIGH]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md (line 86) specifies running the compiler with `--teacher-conf 0.60`.
- **Executable Implementation**: tools/compile_multiclass_dataset.py (line 747) defaults `--teacher-conf` to 0.25. Inside run_compilation() (lines 381-388), it applies hardcoded per-class cutoffs: conf < 0.25 (car, motorcycle) and conf < 0.35 (bus, truck), overriding the guide command-line.
- **Engineering Impact**: A higher teacher confidence cutoff (e.g. 0.60) reduces candidate false positives but significantly degrades recall for small/distant vehicles, and cannot mathematically guarantee zero false positive labels. A lower cutoff (0.25) admits distant perspective traffic but admits more background false alarms into training labels.

### 6.3 IoMin Suppression Threshold for Articulated Trailers [MEDIUM]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md (lines 87, 95) specifies `--iomin-suppress 0.65` to suppress tractor cabs nested inside 18-wheeler articulated trailers.
- **Executable Implementation**: tools/compile_multiclass_dataset.py (line 749) sets CLI default to `--iomin-suppress 0.85`, and deduplicate_boxes() calls (lines 101, 398, 543, 580) hardcode `iomin_thresh=0.85`.
- **Engineering Impact**: IoMin of 0.85 requires a smaller cab box to overlap by 85% before suppression, leaving cab-in-trailer double detections unsuppressed when overlap is between 65% and 85%.

### 6.4 Tiny-Box Discard Filtering [MEDIUM]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md contains no mention of bounding box size cutoffs or minimum dimension filtering.
- **Executable Implementation**: tools/compile_multiclass_dataset.py (line 591) silently filters out any box where: `(nx2 - nx1) < 10 or (ny2 - ny1) < 10` native CCTV pixels (1920x1080).
- **Engineering Impact**: 10 pixels in native 1920x1080 is only ~3.3 pixels under 640-scale letterbox resizing. Distant motorcycles and vehicles near intersection horizons are dropped silently.

### 6.5 Copy-Paste and Mixup Augmentations [HIGH]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md (line 113) specifies fine-tuning YOLO26s with `copy_paste=0.35` and `mixup=0.10` to balance rare vehicles.
- **Executable Implementation**: train_yolo26s.py (lines 39-40) sets `copy_paste=0.10` and `mixup=0.0`. Furthermore, Ultralytics YOLO Copy-Paste requires polygon segmentation masks; because multiclass_dataset only contains bounding boxes (class xc yc w h), Ultralytics silently disables Copy-Paste during training.
- **Engineering Impact**: Copy-Paste does not execute during training regardless of parameter setting unless segment masks exist. Mixup is disabled in code despite guide documentation.

### 6.6 Validation Metric Logging Name [LOW]
- **Guide Specification**: docs/CUSTOM_VEHICLE_GUIDE.md (line 115) states evaluation reports per-class Precision, Recall, and mAP@0.5.
- **Executable Implementation**: train_yolo26s.py (line 80, 83) accesses `val_res.box.maps[idx]` and labels it `mAP@0.5`. In Ultralytics YOLO DetMetrics, `box.maps` contains per-class mAP@0.5:0.95 (mAP50-95), not mAP@0.5 (which is indexed via `box.all_ap[:, 0]`).
- **Engineering Impact**: Logging mislabels mAP@0.5:0.95 as mAP@0.5, reporting artificially lower per-class numbers under an mAP@0.5 heading.

---

## 7. Unresolved Architectural Decisions for User Review

The following engineering decisions remain open and require explicit user determination before Batch 2 (compiler and training pipeline updates):

1. **Pickup & Songthaew Subtype Taxonomy**:
   - Option A: Retain compiler behavior (`0: car`, 1.0 PCE passenger transport alignment).
   - Option B: Align with guide text (`3: truck`, classifying Hilux, รถคอก, and สองแถว under commercial transport).
   - *Action*: Update docs or compiler code to establish uniform taxonomy.
2. **Teacher Co-Annotation Confidence Policy**:
   - Enforcing strict `conf >= 0.60` reduces background false positives but drops distant vehicles (and cannot guarantee zero false positives).
   - Retaining calibrated `conf >= 0.25` admits distant traffic but admits teacher hallucinations into training labels.
3. **Tiny-Box Filtering Resolution Limit**:
   - Determine whether `(nx2 - nx1) >= 10` native pixels cutoff should be lowered or parameterized to avoid dropping distant motorcycles.
4. **Copy-Paste Augmentation Resolution**:
   - Ultralytics YOLO Copy-Paste requires polygon segmentation masks. Multiclass dataset currently has bounding boxes only.
   - Either generate segment polygons (via SAM or polygon annotations) or remove `copy_paste=0.35` recommendation from the guide.
5. **Holdout Evaluation Protocol for `cam45_northeast`**:
   - Establish whether the holdout frames from `cam45_northeast` should be human-annotated to create an official holdout test benchmark.

---
*Report generated deterministically by `tools/audit_dataset.py`.*
