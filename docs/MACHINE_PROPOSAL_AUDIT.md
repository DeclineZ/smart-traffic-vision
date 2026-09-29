# Machine Proposal Quality Assessment Report (Batch 3)

## 1. Executive Summary

This audit assesses the quality of the initial automated machine proposals (`review_pack/proposals/*.txt`) compared against authoritative human ground truth across the **26 frames** where machine proposals originally existed:

- **Total Initial Machine Proposals**: 324
- **Total Verified Human Ground Truth Boxes**: 624
- **Proposal Precision (Standard Detection)**: **89.5%** (290 TP / 324 proposals)
- **Proposal Recall (Standard Detection)**: **46.5%** (290 TP / 624 human ground truth)
- **Proposal F1 Score**: **61.2%**

---

## 2. Granular Proposal Error Taxonomy (One-to-One Matching)

To avoid conflating classification and boundary fit errors with pure omissions or hallucinations, the matching errors are decomposed as follows:

| Category | Count | % of Reference | Description |
| :--- | :---: | :---: | :--- |
| **Correct Matches (True Positives)** | **290** | 46.5% of human GT | Spatial overlap (IoU >= 0.50) with correct vehicle class |
| **Unmatched Ground Truth (Pure Omissions)** | **304** | 48.7% of human GT | Human-verified vehicles completely missed by proposals |
| **Unmatched Proposals (Spurious Detections)** | **4** | 1.2% of proposals | Proposals with no ground truth vehicle in vicinity |
| **Class Disagreements** | **28** | 4.5% of human GT | Overlapped GT (IoU >= 0.50) but assigned wrong class |
| **Localization Errors** | **2** | 0.3% of human GT | Correct class match with moderate overlap (0.10 <= IoU < 0.50) |

*Reconciliation*:
- Human Ground Truth: 290 (correct) + 304 (omissions) + 28 (class mismatch) + 2 (loc error) = **624 total GT**.
- Machine Proposals: 290 (correct) + 4 (spurious) + 28 (class mismatch) + 2 (loc error) = **324 total proposals**.

### Class Disagreement Breakdown
- `car -> truck`: 18 instances
- `three_wheeler -> car`: 1 instances
- `car -> bus`: 8 instances
- `motorcycle -> three_wheeler`: 1 instances

---

## 3. Potential Experimental Directions for Teacher Generation (Future Iterations)

The following directions represent empirical hypotheses to evaluate before scaling up teacher-assisted pseudo-labeling:
1. **Targeted Human Verification on Night Footage**: Test human-in-the-loop review for nighttime footage where contrast degradation causes disproportionate omission.
2. **Morphology Disambiguation Rules**: Evaluate geometric prior filters (aspect ratio, height) to reduce `car -> truck` and `car -> bus` misclassifications.
3. **Multi-Scale Feature Tiling**: Test tile-based inference to evaluate if small vehicle proposal recall can be enhanced.
4. **Empirical Confidence Threshold Sweeps**: Calibrate teacher candidate thresholds against verified validation frames rather than using raw uncalibrated proposals.
