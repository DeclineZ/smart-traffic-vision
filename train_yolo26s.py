"""
Fine-tune YOLO26s on Aligned 5-Class Thai Traffic Dataset.
Classes:
  0: car (sedan, hatchback, SUV, taxi, van)
  1: motorcycle (scooter, big bike, delivery bike)
  2: bus (transit bus, coach, Thai double-decker bus)
  3: truck (pickup, delivery box pickup, flatbed, 6/10/18-wheeler)
  4: three_wheeler (tuk-tuk, motorized saleng)
"""

import multiprocessing
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    data_yaml = Path("data/multiclass_dataset/data.yaml").resolve()
    assert data_yaml.exists(), f"Missing {data_yaml}"

    print(f"Initializing YOLO26s fine-tuning on {data_yaml}...")
    model = YOLO("yolo26s.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        batch=16,
        imgsz=640,
        device="0",
        workers=2,
        project="runs/train",
        name="yolo26s_thai_traffic",
        exist_ok=True,
        pretrained=True,
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3.0,
        close_mosaic=5,
        copy_paste=0.10,
        mixup=0.0,
        cls=0.5,
        box=7.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        fliplr=0.5,
        scale=0.20,
        patience=0,
        verbose=True,
    )

    possible_pts = [
        Path("runs/detect/runs/train/yolo26s_thai_traffic/weights/best.pt"),
        Path("runs/train/yolo26s_thai_traffic/weights/best.pt"),
    ]
    best_pt = next((p for p in possible_pts if p.exists()), None)
    target_pt = Path("models/yolo26s_thai_traffic.pt")

    if best_pt and best_pt.exists():
        os.makedirs("models", exist_ok=True)
        shutil.copyfile(best_pt, target_pt)
        print(f"\n[SUCCESS] Successfully exported fine-tuned weights from {best_pt} to: {target_pt}")
    else:
        print(f"\n[WARNING] best.pt not found in {possible_pts}")

    # Run final comprehensive validation
    print("\nRunning final validation on val split...")
    eval_weights = str(target_pt if target_pt.exists() else (best_pt or "yolo26s.pt"))
    val_model = YOLO(eval_weights)
    val_res = val_model.val(data=str(data_yaml), imgsz=640, device="0")

    print("\n=======================================================")
    print("Fine-Tuned YOLO26s Validation Summary (5 Classes):")
    print(f"  Overall Precision:    {val_res.box.mp:.4f}")
    print(f"  Overall Recall:       {val_res.box.mr:.4f}")
    print(f"  Overall mAP@0.5:      {val_res.box.map50:.4f}")
    print(f"  Overall mAP@0.5:0.95: {val_res.box.map:.4f}")
    print("-------------------------------------------------------")
    for idx, name in enumerate(val_model.names.values()):
        m50 = val_res.box.maps[idx] if idx < len(val_res.box.maps) else 0.0
        p = val_res.box.p[idx] if idx < len(val_res.box.p) else 0.0
        r = val_res.box.r[idx] if idx < len(val_res.box.r) else 0.0
        print(f"  {name:<15}: P={p:.3f} | R={r:.3f} | mAP@0.5={m50:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
