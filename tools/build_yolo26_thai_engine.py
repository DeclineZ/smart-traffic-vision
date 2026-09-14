"""Export the Thai-traffic YOLO26s checkpoint as a fixed-batch TensorRT engine."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "benchmark" / "runtime-yolo26"
WEIGHTS = ROOT / "models" / "yolo26s_thai_traffic.pt"
ENGINE = WEIGHTS.with_suffix(".engine")

sys.path.insert(0, str(RUNTIME))
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "benchmark" / "runtime-settings")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    from ultralytics import YOLO
    import tensorrt
    import torch

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    model = YOLO(str(WEIGHTS), task="detect")
    if model.task != "detect":
        raise ValueError("The Thai traffic checkpoint must be a detection model")
    names = dict(model.names)
    expected = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "three_wheeler"}
    if names != expected:
        raise ValueError(f"Unexpected class mapping: {names}")
    start = time.perf_counter()
    exported = model.export(format="engine", imgsz=640, batch=1, dynamic=False,
                            quantize=16, workspace=2, device=0, nms=False,
                            simplify=False, opset=17)
    if Path(exported).resolve() != ENGINE.resolve() or not ENGINE.is_file():
        raise FileNotFoundError(f"TensorRT export did not create {ENGINE}")
    manifest = {
        "model": "yolo26s_thai_traffic",
        "weights_sha256": sha256(WEIGHTS),
        "engine_sha256": sha256(ENGINE),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "precision": "fp16",
        "batch": 1,
        "input_size": 640,
        "classes": names,
        "ultralytics": __import__("ultralytics").__version__,
        "torch": torch.__version__,
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "build_seconds": time.perf_counter() - start,
        "export": str(exported),
        "nms": False,
        "end2end": True,
    }
    ENGINE.with_suffix(".build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {ENGINE}", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
