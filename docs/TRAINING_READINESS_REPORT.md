# Batch 7: Annotation Consolidation & Training-Readiness Report

- **Date**: 2026-09-28T01:57:50.677780+00:00
- **Consolidated Pack**: `data/review_pack_consolidated_v3`
- **Candidate Manifests**: `data/training_manifests_v6`
- **Authoritative Review Status**: **43 Verified**, **1 Rejected** across 44 Canonical Sources
- **Training Eligibility**: **43 Eligible**, **0 Blocked (Pending Proposals)**, **1 Rejected**

---

## 1. Executive Summary & Review Lineage Reconciliation

All 44 canonical training-review frames across `review_pack_v2`, `review_pack_pilot_v1`, and `review_pack_continuation_v1` have been reconciled with strict precedence:

| Source Pack | Frames Contributed | Precedence Applied | Verified | Rejected | Training Eligible | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`review_pack_v2` (Retained)** | **20** | Direct retention of previously verified baseline | 20 | 0 | 20 | 11 external UA-DETRAC, 9 local CCTV. |
| **`review_pack_pilot_v1`** | **6** | Human review decisions supersede v2 drafts | 5 | 1 | 5 | 1 local frame (`cam44_north_f019140`) rejected. |
| **`review_pack_continuation_v1`** | **18** | Human review decisions supersede v2 drafts | 18 | 0 | 18 | All 18 frames verified and training-eligible. |
| **Total Consolidated** | **44** | **Strict Precedence (0 Collisions, 0 Overlaps)** | **43** | **1** | **43** | **100% accounted for exactly once**. |

### Rejected Frame Enforcements (1 Frame)
1. **`cam44_north_f019140`** (Local Daytime Queue): Rejected due to intractable motorcycle queue density / occlusions. Excluded from training along with its variant `cam44_north_f019140_tuktukboost_1`.

> [!IMPORTANT]
> **Negative Example Safeguard**: Rejection does **NOT** generate empty negative label files. Both canonical sources and their derived variants are completely quarantined from candidate training splits. Pending proposals on rejected frames require no further review.

---

## 2. Approved Annotations, Training Eligibility & Size-Bin Correction

User review status has been strictly separated from training eligibility:
- A frame is **training-eligible** if and only if `review_status == "verified"` **AND** `pending_proposals_count == 0` **AND** `unresolved_conflicts_count == 0`.
- **43 frames** are fully eligible and exported to `annotations/labels/*.txt`.
- **0 frames** blocked due to pending proposals.
- Export class totals and proposal statistics are counted **only over the 43 exported eligible frames**; rejected frames are quarantined.

| Metric | Count | Provenance & Handling |
| :--- | :---: | :--- |
| **Total Exported Verified Frames** | **43** | 25 external UA-DETRAC, 18 local CCTV approach frames |
| **Total Approved Bounding Boxes** | **1488** | Synchronized line-for-line in `annotations/labels/*.txt` |
| **Accepted Teacher Proposals (Exported)** | **383** | Recovered background queue vehicles and distant oncoming traffic |
| **Human Manually Added Boxes (Exported)** | **246** | Drawn by human reviewers following the Small Vehicle Scan Checklist |
| **Teacher Conflict Corrections (Pickup/Van $\to$ Car)** | **10** | Teacher classified pickups/vans as truck; human corrected to car |
| **Rejected Teacher Proposals (Exported Frames)** | **86** | False positive halos, glare reflections, and duplicate boxes discarded |

### Per-Class Approved Box Counts (Exported Eligible Frames Only)
| Class ID | Class Name | Box Count | Percentage |
| :---: | :--- | :---: | :---: |
| 0 | `car` | 1252 | 84.1% |
| 1 | `motorcycle` | 106 | 7.1% |
| 2 | `bus` | 99 | 6.7% |
| 3 | `truck` | 24 | 1.6% |
| 4 | `three_wheeler` | 7 | 0.5% |

### Size-Bin Calculation Fix (Non-Square Image Dimensions)
Size buckets for reviewed local records pass actual image dimensions (1920×1080) instead of a naive 640×640 assumption into `compute_aspect_preserving_size`. Normalized coordinates $[xc, yc, bw, bh]$ remain completely unaltered:
- **117 medium $\to$ small corrections** across all 18 local CCTV frames (111 across the original 17).
- **9 large $\to$ medium corrections** across local CCTV frames.
- **Resulting Manifest A Size Distribution**: 7,262 small, 4,581 medium, 1,305 large boxes.

### Blocked & Rejected Quarantine Summary (Reported Separately)
| Frame ID | Review Status | Training Eligibility | Boxes / Proposals Status | Required Action |
| :--- | :---: | :---: | :--- | :--- |
| **`cam44_north_f019140`** | `rejected` | **EXCLUDED** | 51 unapproved boxes quarantined | No further action (rejected source). |

---

## 3. Dataset Lineage & Candidate Manifests V6

The candidate training manifests were updated to reflect human review decisions, box-level stale variant detection, and exact external record linking:

| Split / Attribute | Manifest A (Local-Only V6) | Manifest B (Local + External V6) | Delta vs V3 | Integrity Safeguard |
| :--- | :---: | :---: | :---: | :--- |
| **Local Training Images** | **1092** | **1092** | -43 images | Excludes 2 rejected frames/variants + 41 stale variants |
| **Unique Local Canonical Sources** | **527** | **527** | -1 source | Reduced from 528 to 527 by excluding 1 rejected source |
| **External Training Images** | **0** | **263** | -1 image | Recalculated cap: $\lfloor 527 \times 0.5 \rfloor = 263$ (was 264) |
| **Total Candidate Training Images** | **1092** | **1355** | -44 images | Fully synchronized across A and B |
| **Total Candidate Training Boxes** | **13148** | **16661** | — | Dynamically derived from final selected records |
| **Primary Validation Images** | **130** | **130** | **0 (Identical)** | **Byte-for-byte identical; 0 validation leakage** |
| **Stale Variants Excluded** | **41** | **41** | +41 | Excludes unregenerated synthetic copies of all 18 modified frames |

### Canonical Source-Level Exclusions (Resampling Guard)
Exclusions for rejected sources (`cam44_north_f019140`) are enforced at the **canonical source level**. Resampling cannot select any variant or descendant of an unresolved source into Manifest B.

---

## 4. Honest Annotation Readiness & Benchmark Status

> [!WARNING]
> **Evaluation Benchmark Status Correction**:
> `data/eval_snapshot_v1` (42 frames) is **NOT** an independent evaluation benchmark. It is an **internal diagnostic benchmark** with known or unproven exposure to similar sequence environments. It must not be cited as an uncompromised external benchmark.

| Dataset Partition | Total Frames | Human-Reviewed & Complete | Unreviewed / Pending Labels | Status & Readiness Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Reviewed Sample (Batch 6 & 7)** | **44** | **43 (97.7%)** | 0 blocked, 1 rejected | **43 Frames Complete**: 18 local and 25 external frames ready for training / smoke testing. |
| **Local CCTV Dataset (`multiclass_dataset`)** | **1,397** | **18 canonical** | ~1,354 frames | **Partially Reviewed**: 509 local training sources remain teacher-completed without human review. 41 synthetic variants stale. |
| **External Selection (UA-DETRAC)** | **263** | **25 canonical** | 238 frames | **Incomplete Labels**: 238 external frames retain uncorrected source labels (missing small background cars; images not yet downloaded locally). |
| **Quarantined / Stale Records** | **43** | **0** | 43 excluded | **Quarantined**: 2 rejected records + 41 stale variants excluded from candidate training. |

---

## 5. Experiment Specification Summary

See detailed specification in [`docs/EXPERIMENT_SPECIFICATION_BATCH7.md`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/EXPERIMENT_SPECIFICATION_BATCH7.md).

Two distinct experiment tracks have been prepared:
1. **Track 1: Short Pipeline Smoke Test**
   - **Goal**: Verify training pipeline, loss stability, checkpoint saving, and eval metrics without overfitting.
   - **Dataset**: 43 eligible reviewed frames (18 local + 25 external); 130 primary val images.
   - **Budget**: 5 epochs (batch size 8, warm start from `models/yolo26s_thai_traffic.pt`).
   - **Path**: `runs/train/smoke_test_batch7/` (baseline weights preserved).
2. **Track 2: Meaningful Improvement Experiment (Gated)**
   - **Prerequisites**: Regenerate 41 local synthetic variants.
   - **Dataset**: Manifest A (1,092 local frames) vs Manifest B (1,092 local + 263 external frames).
   - **Budget**: 100 epochs with early stopping (patience = 15).
   - **Path**: `runs/train/candidate_manifest_a_v6/` and `runs/train/candidate_manifest_b_v6/`.

---

## 6. Output Files & Artifacts

- **Consolidated Review Pack**: [`data/review_pack_consolidated_v3/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3)
- **Approved YOLO Labels (43 Frames)**: [`data/review_pack_consolidated_v3/annotations/labels/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/annotations/labels)
- **Pack Manifest**: [`data/review_pack_consolidated_v3/manifest.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/review_pack_consolidated_v3/manifest.json)
- **Candidate Manifests V6**: [`data/training_manifests_v6/`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6)
- **Local-Only Manifest**: [`data/training_manifests_v6/manifest_a_local_only.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_a_local_only.json)
- **Local + External Manifest**: [`data/training_manifests_v6/manifest_b_local_plus_external.json`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/data/training_manifests_v6/manifest_b_local_plus_external.json)
- **Experiment Specification**: [`docs/EXPERIMENT_SPECIFICATION_BATCH7.md`](file:///e:/Work/Projects/AdaptiveTrafficControl/smart-traffic-vision/docs/EXPERIMENT_SPECIFICATION_BATCH7.md)
