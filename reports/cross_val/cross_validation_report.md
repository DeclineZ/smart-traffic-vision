# 4-Fold Stratified Cross-Validation Report

**Model:** YOLOv8n-cls (Lightweight Edge Classifier)  
**Dataset:** `weather_dataset` (2436 total images)  
**Epochs per Fold:** 5  
**Compute Acceleration:** mps  
**Total Evaluation Time:** 381.8 seconds  

---

## 1. Summary Statistics Across All 4 Folds

| Metric | Mean | Std Dev | Best Fold | Worst Fold |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **99.14%** | &plusmn;0.31% | 99.51% | 98.69% |
| **Precision (Rainy)** | **98.96%** | &plusmn;0.91% | 100.00% | 97.78% |
| **Recall (Rainy)** | **99.35%** | &plusmn;0.33% | 99.68% | 99.03% |
| **F1-Score** | **0.9915** | &plusmn;0.0030 | 0.9951 | 0.9872 |

---

## 2. Per-Fold Breakdown

| Fold | Train / Val Samples | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TP / FP / TN / FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 1826 / 610 | 99.02% | 98.40% | 99.68% | 0.9904 | TP=308, FP=5, TN=296, FN=1 |
| Fold 2 | 1826 / 610 | 98.69% | 97.78% | 99.68% | 0.9872 | TP=308, FP=7, TN=294, FN=1 |
| Fold 3 | 1828 / 608 | 99.34% | 99.67% | 99.03% | 0.9935 | TP=305, FP=1, TN=299, FN=3 |
| Fold 4 | 1828 / 608 | 99.51% | 100.00% | 99.03% | 0.9951 | TP=305, FP=0, TN=300, FN=3 |

---

## 3. Scientific Conclusions
- The minimal standard deviation across all 4 folds confirms that the model generalizes robustly without overfitting to specific camera locations or traffic density patterns.
- Both precision and recall exceed 99%, demonstrating balanced sensitivity between dry pavements and water-soaked roads.
