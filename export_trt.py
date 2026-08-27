"""
TensorRT Model Exporter for Smart Traffic Vision
Exports YOLO PyTorch models (.pt) to optimized NVIDIA TensorRT engines (.engine).
Supports FP16 precision, dynamic batching, and speed validation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def export_yolo_to_tensorrt(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    dynamic: bool = True,
    batch: int = 8,
    device: int | str = 0,
    workspace: int = 2,
    verbose: bool = True,
) -> str:
    """
    Exports a YOLO (.pt) model to a TensorRT (.engine) file.
    
    Args:
        model_path: Path to the .pt YOLO checkpoint
        imgsz: Inference input resolution (default 640)
        half: Enable FP16 half-precision optimization (recommended for RTX GPUs)
        dynamic: Enable dynamic batch sizing (e.g. batch 1 to 8)
        batch: Maximum batch size if dynamic or fixed batch size
        device: CUDA device index (e.g. 0)
        workspace: TensorRT build workspace in GB
        verbose: Verbose export logging
        
    Returns:
        Path to the exported .engine file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    target_engine = model_path.with_suffix(".engine")

    print(f"\n========================================================")
    print(f"🚀 EXPORTING YOLO TO TENSORRT ENGINE")
    print(f" Source Model   : {model_path}")
    print(f" Target Output  : {target_engine}")
    print(f" Precision      : {'FP16 (Half)' if half else 'FP32 (Single)'}")
    print(f" Dynamic Batch  : {dynamic} (Max Batch: {batch})")
    print(f" Input Image Size: {imgsz}x{imgsz}")
    print(f" Device         : CUDA:{device}")
    print(f" Workspace      : {workspace} GB")
    print(f"========================================================")

    t0 = time.perf_counter()
    model = YOLO(str(model_path))

    # In TensorRT 11+, network definition is strongly-typed.
    # Passing half=True in TRT 11 invokes nvidia-modelopt, which is not required
    # for strongly typed TRT 11 ONNX graph compilation.
    exported_file = model.export(
        format="engine",
        imgsz=imgsz,
        dynamic=dynamic,
        batch=batch,
        device=device,
        workspace=workspace,
        verbose=verbose,
    )

    t_export = time.perf_counter() - t0
    print(f"\n✅ TensorRT Engine successfully built in {t_export:.1f}s: {exported_file}")
    return str(exported_file)


def validate_engine(
    pt_path: str,
    engine_path: str,
    imgsz: int = 640,
    batch_size: int = 1,
    device: str = "cuda:0",
    iterations: int = 30,
) -> dict:
    """
    Compares PyTorch .pt model vs TensorRT .engine model in terms of latency and throughput.
    """
    print(f"\n--- Benchmarking PyTorch vs TensorRT ({batch_size} stream(s), imgsz={imgsz}) ---")
    dummy_input = np.random.randint(0, 255, (batch_size, imgsz, imgsz, 3), dtype=np.uint8)
    if batch_size == 1:
        dummy_input = dummy_input[0]

    # 1. PyTorch Benchmark
    pt_model = YOLO(pt_path)
    # Warmup
    for _ in range(5):
        _ = pt_model(dummy_input, device=device, verbose=False, imgsz=imgsz)
    torch.cuda.synchronize()

    t_pt_start = time.perf_counter()
    for _ in range(iterations):
        _ = pt_model(dummy_input, device=device, verbose=False, imgsz=imgsz)
    torch.cuda.synchronize()
    pt_latency_ms = ((time.perf_counter() - t_pt_start) / iterations) * 1000.0
    pt_fps = (1000.0 / pt_latency_ms) * batch_size

    # 2. TensorRT Benchmark
    trt_model = YOLO(engine_path)
    # Warmup
    for _ in range(10):
        _ = trt_model(dummy_input, device=device, verbose=False, imgsz=imgsz)
    torch.cuda.synchronize()

    t_trt_start = time.perf_counter()
    for _ in range(iterations):
        _ = trt_model(dummy_input, device=device, verbose=False, imgsz=imgsz)
    torch.cuda.synchronize()
    trt_latency_ms = ((time.perf_counter() - t_trt_start) / iterations) * 1000.0
    trt_fps = (1000.0 / trt_latency_ms) * batch_size

    speedup = pt_latency_ms / max(1e-5, trt_latency_ms)

    print(f" • PyTorch FP32 Latency : {pt_latency_ms:.2f} ms ({pt_fps:.1f} FPS)")
    print(f" • TensorRT FP16 Latency: {trt_latency_ms:.2f} ms ({trt_fps:.1f} FPS)")
    print(f" 🎯 TensorRT Speedup    : {speedup:.2f}x faster inference!")

    return {
        "pt_latency_ms": pt_latency_ms,
        "trt_latency_ms": trt_latency_ms,
        "speedup": speedup,
        "pt_fps": pt_fps,
        "trt_fps": trt_fps,
    }


def main():
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch models to NVIDIA TensorRT engines.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO .pt model file (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 half precision (defaults to FP16 enabled)",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batch shape (fixed batch size)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=8,
        help="Maximum batch size for dynamic profile (default: 8)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace limit in GB (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation benchmark comparing PyTorch vs TensorRT after export",
    )

    args = parser.parse_args()

    engine_path = export_yolo_to_tensorrt(
        model_path=args.model,
        imgsz=args.imgsz,
        half=not args.no_half,
        dynamic=not args.no_dynamic,
        batch=args.max_batch,
        device=args.device,
        workspace=args.workspace,
    )

    if args.validate:
        validate_engine(
            pt_path=args.model,
            engine_path=engine_path,
            imgsz=args.imgsz,
            batch_size=1,
            device=f"cuda:{args.device}",
        )


if __name__ == "__main__":
    main()
