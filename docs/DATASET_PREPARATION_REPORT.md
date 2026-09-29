# Thai Traffic Vision & YOLO26s Dataset Preparation Report (Batch 4 - Corrected)

- **Date**: 2026-09-27T16:52:38.215481+00:00
- **Report Status**: Candidate Manifests & Leakage Safeguards Established (**Pre-Training Gate**)
- **Random Seed**: `42` (Deterministic sequence splitting & stratified sampling)
- **External Image Cap**: **264 frames** (0.5 per unique local training frame; 528 unique local frames)
- **Target Taxonomy**: Thai 5-Class COCO-aligned morphology (`0: car`, `1: motorcycle`, `2: bus`, `3: truck`, `4: three_wheeler`)

---

## 1. Executive Summary & Comparative Matrix

This audit prepares a controlled comparison between **Manifest A (Local-Only)** and **Manifest B (Local + Capped External UA-DETRAC)** to expand visual training variety while strictly preventing external cars from overwhelming Thai motorcycles, three-wheelers, and local conditions:

| Metric / Attribute | Manifest A (Local-Only) | Manifest B (Local + External) | Delta (B vs A) | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Total Training Images** | **1135** | **1399** | +264 (264 external) | Controlled external expansion |
| **Unique Canonical Training Frames** | **528** | **792** | +264 | Independent video frames |
| **Total Training Bounding Boxes** | **13697** | **16926** | +3229 (+23.6%) | Preserves dense annotations |
| **Total Validation Images** | **130** | **130** | **0 (Identical)** | **Validation is strictly identical** |
| **Validation Bounding Boxes** | **1367** | **1367** | **0 (Identical)** | **Zero validation leakage** |
| **External Sequence Overlap** | **0.0%** | **0.0%** | 0.0% | Whole MVI sequence splitting |
| **Diagnostic Benchmark Overlap** | **0 frames** | **0 frames** | 0 | 42 eval frames strictly excluded |

---

## 2. Per-Class Box Proportions & Class Imbalance Guardrails

The external cap is calculated strictly as $\lfloor 528 \times 0.5 \rfloor = 264$ frames (using unique local frames, NOT synthetic variants). This guarantees that external passenger cars do not submerge local minority classes:

| Vehicle Class | Class ID | Manifest A Boxes | Manifest A Share | Manifest B Boxes | Manifest B Share | External Added | Share Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `car` | 0 | 9743 | 71.13% | 12551 | 74.15% | +2808 | +3.0% |
| `motorcycle` | 1 | 1621 | 11.83% | 1621 | 9.58% | +0 | -2.2% |
| `bus` | 2 | 419 | 3.06% | 756 | 4.47% | +337 | +1.4% |
| `truck` | 3 | 1399 | 10.21% | 1483 | 8.76% | +84 | -1.5% |
| `three_wheeler` | 4 | 515 | 3.76% | 515 | 3.04% | +0 | -0.7% |
| **Total** | — | **13697** | **100.0%** | **16926** | **100.0%** | **+3229** | — |

> [!NOTE]
> **Minority Class Protection**: Local motorcycle boxes (1621) and three-wheeler boxes (515) remain fully preserved without box deletion. External data adds valuable truck and bus variety without collapsing minority representation.

---

## 3. Local Split & Contiguous Temporal Block Definitions

Local surveillance video recordings are split into explicit contiguous temporal blocks (80% timeline for train, exclusion buffer zone, remaining timeline for val) with all canonical frame variants kept together:

| Recording | Evidenced FPS | Strategy | Train Block [Start-End] | Buffer Zone Excluded | Val Block [Start-End] | Measured Separation |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `cam03_east` | 150.0 FPS | Contiguous Blocks | [0, 156189] (105 imgs) | 0 sources (0 imgs) | [157149, 495202] (28 imgs) | 960 frames (6.4s) |
| `cam03_east_night` | 150.0 FPS | Contiguous Blocks | [100, 302789] (6 imgs) | 0 sources (0 imgs) | [363327, 363327] (1 imgs) | 60538 frames (403.59s) |
| `cam43_south` | 150.0 FPS | Contiguous Blocks | [100, 159780] (178 imgs) | 0 sources (0 imgs) | [164700, 557739] (31 imgs) | 4920 frames (32.8s) |
| `cam43_south_night` | 150.0 FPS | Contiguous Blocks | [100, 244412] (9 imgs) | 0 sources (0 imgs) | [305490, 366568] (2 imgs) | 61078 frames (407.19s) |
| `cam44_north` | 150.0 FPS | Contiguous Blocks | [0, 151200] (201 imgs) | 1 sources (2 imgs) | [153120, 495924] (32 imgs) | 1920 frames (12.8s) |
| `cam44_north_night` | 150.0 FPS | Contiguous Blocks | [0, 122256] (247 imgs) | 0 sources (0 imgs) | [134520, 549802] (33 imgs) | 12264 frames (81.76s) |
| `cam46_west` | 150.0 FPS | Contiguous Blocks | [100, 118567] (323 imgs) | 1 sources (1 imgs) | [120007, 496354] (97 imgs) | 1440 frames (9.6s) |
| `cam46_west_night` | 150.0 FPS | Contiguous Blocks | [0, 182506] (66 imgs) | 1 sources (4 imgs) | [242899, 667798] (5 imgs) | 60393 frames (402.62s) |

- **Minimum Measured Temporal Separation**: **960 frames (6.40s @ 150.0 FPS in `cam03_east`)**.
- *Scientific Disclaimer*: Passing a temporal proximity check prevents frame-burst leakage, but does NOT prove vehicle-level independence without trajectory tracking, nor does it remove historical checkpoint training exposure.

---

## 4. Split & Exclusion Audit (Safeguards Enforced)

All records excluded from training candidates are tracked in `split_and_exclusion_manifest.json`:

| Exclusion Category | Records Excluded | Scientific Rationale |
| :--- | :---: | :--- |
| **`diagnostic_evaluation_benchmark_leakage`** | **26** | Excludes the 42 authoritative evaluation benchmark frames (and variants) from training/validation to prevent circular benchmarking. |
| **`temporal_buffer_violation`** | **7** | Frames falling within the temporal exclusion buffer between train and val blocks. |
| **`reserved_northeast_holdout`** | **0** | Reserves `cam45_northeast` approach footage from training. |
| **`external_val_sequence_reserved`** | **2409** | Sequence-level partition: 15 MVI sequences reserved exclusively for external validation. |
| **`external_test_sequence_reserved`** | **2272** | Sequence-level partition: 15 MVI sequences reserved exclusively for external testing. |
| **`unknown_provenance_quarantined`** | **0** | Quarantines any frames with unproven lineage. |

### Whole-Sequence Partitioning (UA-DETRAC)
- **Train Sequences**: 70 sequences (MVI_20012, MVI_20032, MVI_20035, MVI_20051, MVI_20052...)
- **Validation Sequences**: 15 sequences (MVI_20011, MVI_39781, MVI_39811, MVI_39851, MVI_40714...)
- **Test Sequences**: 15 sequences (MVI_20033, MVI_20034, MVI_20064, MVI_39031, MVI_39051...)
- **Cross-Split Sequence Leakage**: **0 sequences (0.0% overlap)**.

### Achieved External Stratified Coverage
- **Sequences Represented**: 70 / 70 (100.0%)
- **Frames with Small Boxes (< 32² px)**: 155 frames (58.7%)
- **Frames with Medium Boxes**: 261 frames (98.9%)
- **Frames with Large Boxes**: 245 frames (92.8%)
- **Frames with Source Vans (class 3)**: 189 frames
- **Frames with Source Trucks (class 2)**: 81 frames
- **Frames with Source Buses (class 0)**: 213 frames

---

## 5. Annotation Uncertainty & Taxonomy Warnings

1. **Semantic Compatibility of External Vans & Trucks**:
   - `van -> 0 (car)`: In Thailand, commuter passenger vans (Toyota Commuter/HiAce) belong to class 0. However, truck-based cargo vans might border class 3.
   - `truck -> 3 (truck)`: Medium and heavy commercial trucks belong to class 3. In external datasets, light utility trucks or flatbeds might be labeled truck, whereas Thai morphology treats light pickups as class 0.
2. **Missing Foreground Object Bias**:
   - UA-DETRAC does **NOT** annotate motorcycles, bicycles, or three-wheelers.
   - The absence of motorcycle labels in external images does **not** prove motorcycles are absent. Training on unannotated motorcycles with standard detection loss would treat them as negative background and penalize motorcycle recall.
3. **Missing External Image Files**:
   - The UA-DETRAC image files are currently **unresolved on disk** (only NDJSON metadata exists in `data/usdetrac/`). They are explicitly tracked as `unresolved_not_downloaded` and must not be treated as locally available image files.

---

## 6. Bounded Visual Spot-Check Plan (Up to 50 Unique Canonical Frames)

A dedicated manifest of **44 unique canonical source frames** is established at `data/training_manifests_v2/visual_spot_check_manifest.json`:

| Category | Target Quota | Achieved Frames | Shortfall | Key Inspection Objective |
| :--- | :---: | :---: | :---: | :--- |
| **external_vans** | 10 | 10 | 0 | Verify passenger commuter van morphology vs commercial cargo trucks. |
| **external_trucks** | 10 | 10 | 0 | Verify medium/heavy commercial chassis vs light pickup flatbeds. |
| **external_dense_small** | 10 | 10 | 0 | Inspect tiny box bounds and verify absence of unannotated motorcycles in background. |
| **local_teacher_completed** | 10 | 10 | 0 | Check background boxes generated by COCO YOLO26x teacher for false alarms. |
| **local_night_congestion** | 10 | 10 | 0 | Verify dense motorcycle and tuk-tuk queue annotations under headlight glare. |

> [!IMPORTANT]
> **Pre-Training Gate**: MANDATORY GATE: The training datasets must NOT be considered training-ready or exported to training pipelines until these spot-check frames are visually reviewed and external image files are resolved on disk.

---

## 7. Reproducibility & Commands

To regenerate these manifests deterministically from source data:
```bash
python tools/prepare_training_manifests.py \
  --config config/training_manifest_config.json \
  --local-dir data/multiclass_dataset \
  --ndjson-path data/usdetrac/ua-detrac-dataset-10kv1-2024-11-14-3-44pmyolov11.ndjson \
  --eval-snapshot data/eval_snapshot_v1/manifest.json \
  --output-dir data/training_manifests_v2 \
  --videos-dir videos \
  --report-md docs/DATASET_PREPARATION_REPORT.md \
  --seed 42 \
  --external-ratio 0.5
```
