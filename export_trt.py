"""
TensorRT Model Exporter for Smart Traffic Vision.
Exports YOLO PyTorch checkpoints (.pt) to optimized NVIDIA TensorRT engines (.engine).
Configured for FP16 precision, dynamic batching, and production edge deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


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
        half: Enable FP16 half-precision optimization (recommended for modern NVIDIA GPUs)
        dynamic: Enable dynamic batch sizing (e.g. batch 1 to N)
        batch: Maximum batch size for dynamic batching
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

    print("\n" + "=" * 60)
    print("[EXPORT] EXPORTING YOLO TO TENSORRT ENGINE (FP16)")
    print(f" Source Model    : {model_path}")
    print(f" Target Output   : {target_engine}")
    print(f" Precision       : {'FP16 (Half)' if half else 'FP32 (Single)'}")
    print(f" Dynamic Batch   : {dynamic} (Max Batch: {batch})")
    print(f" Resolution      : {imgsz}x{imgsz}")
    print(f" Device          : CUDA:{device}")
    print(f" Workspace       : {workspace} GB")
    print("=" * 60 + "\n")

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT export, but torch.cuda.is_available() returned False.")

    if half:
        try:
            import tensorrt as trt
            trt_major = int(trt.__version__.split(".")[0])
        except Exception:
            trt_major = 0
        if trt_major >= 11:
            try:
                import modelopt.onnx  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "TensorRT 11+ is strongly typed: Ultralytics bakes FP16 into the ONNX graph with NVIDIA "
                    "ModelOpt. Install it (pip install nvidia-modelopt[onnx]) or export FP32 with --no-half."
                )

    t0 = time.perf_counter()
    model = YOLO(str(model_path))

    # Trigger export via Ultralytics engine builder
    exported_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        batch=batch,
        device=device,
        workspace=workspace,
        verbose=verbose,
    )

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Export completed successfully in {elapsed:.1f} seconds!")
    print(f" Output Engine : {exported_path}")
    print("=" * 60 + "\n")

    return str(exported_path)


def export_fixed_size_engine(
    model_path: str,
    imgsz: int = 960,
    batch: int = 8,
    device: int | str = 0,
    workspace: int = 4,
) -> str:
    """
    FP32 engine with dynamic batch (1..batch) and height/width bounded at imgsz.

    Ultralytics' dynamic export lets height/width grow to max(2, workspace) x imgsz, so at 960 px
    with 8 cameras TensorRT plans for 8 x 1920x1920 (or larger) inputs and asks for more scratch
    memory than an 8 GB GPU has. Inference never exceeds imgsz on the long side (16:9 cameras are
    letterboxed to e.g. 544x960), so the profile stops there. The file keeps Ultralytics' metadata
    header, so YOLO(engine) loads it unchanged.
    """
    import ast
    import json

    import onnx
    import tensorrt as trt
    from ultralytics import YOLO

    model_path = Path(model_path)
    onnx_path = YOLO(str(model_path)).export(format="onnx", imgsz=imgsz, dynamic=True, batch=batch,
                                             simplify=True, device=device)
    meta = {}
    for prop in onnx.load(onnx_path).metadata_props:
        try:
            meta[prop.key] = ast.literal_eval(prop.value)
        except (ValueError, SyntaxError):
            meta[prop.key] = prop.value
    meta["batch"] = batch
    meta["imgsz"] = [imgsz, imgsz]
    if isinstance(meta.get("args"), dict):
        meta["args"].update({"batch": batch, "dynamic": True, "workspace": workspace})

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace * (1 << 30)))
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, min=(1, 3, 32, 32), opt=(batch, 3, imgsz, imgsz), max=(batch, 3, imgsz, imgsz))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed; see the TensorRT log above")
    engine_path = model_path.with_suffix(".engine")
    header = json.dumps(meta).encode()
    with open(engine_path, "wb") as f:
        f.write(len(header).to_bytes(4, byteorder="little", signed=True))
        f.write(header)
        f.write(serialized)
    print(f"[SUCCESS] Bounded engine (batch 1-{batch}, up to {imgsz}x{imgsz}, FP32) built in "
          f"{time.perf_counter() - t0:.1f}s -> {engine_path}")
    return str(engine_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt checkpoint to a TensorRT .engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="models/yolo26s_thai_traffic.pt", help="Path to .pt model weights")
    parser.add_argument("--batch", type=int, default=8, help="Max batch size for dynamic batching (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640)")
    parser.add_argument("--workspace", type=int, default=2, help="TensorRT build workspace in GB (default: 2)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 half-precision (use FP32)")
    parser.add_argument("--fixed-batch", action="store_true", help="Disable dynamic batching")
    parser.add_argument("--fixed-size", action="store_true",
                        help="Dynamic batch (1..--batch) with height/width capped at imgsz, FP32. Needed for 960 px "
                             "with 8 cameras on 8 GB GPUs, where the default dynamic export runs out of memory")
    return parser


def write_engine_metadata(engine_path: str, source_model: str, **settings) -> str:
    """Records what the engine was built from, so deployments can verify provenance."""
    import hashlib
    import json
    import platform

    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    meta = {
        "engine": os.path.basename(engine_path),
        "engineSha256": sha256(engine_path),
        "sourceModel": os.path.basename(source_model),
        "sourceSha256": sha256(source_model),
        "settings": settings,
        "builtAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
    }
    try:
        import tensorrt, torch
        meta["tensorrt"] = tensorrt.__version__
        meta["torch"] = torch.__version__
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    out = str(Path(engine_path).with_suffix(".engine.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fixed_size:
        if not args.no_half or args.fixed_batch:
            parser.error("--fixed-size builds an FP32 dynamic-batch engine: add --no-half, drop --fixed-batch")
        engine = export_fixed_size_engine(args.model, imgsz=args.imgsz, batch=args.batch, device=args.device,
                                          workspace=args.workspace)
    else:
        engine = export_yolo_to_tensorrt(
            model_path=args.model,
            imgsz=args.imgsz,
            half=not args.no_half,
            dynamic=not args.fixed_batch,
            batch=args.batch,
            device=args.device,
            workspace=args.workspace,
        )
    meta = write_engine_metadata(engine, args.model, imgsz=args.imgsz, half=not args.no_half,
                                 dynamic=not args.fixed_batch, batch=args.batch, fixed_size=args.fixed_size)
    print(f"Engine metadata: {meta}")
    print("Engines are specific to this GPU model and TensorRT version; rebuild on the deployment hardware.")


if __name__ == "__main__":
    main()
