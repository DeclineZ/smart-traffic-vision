import tensorrt as trt
import os

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
JETSON_PATH = os.path.dirname(SCRIPT_PATH)
ROOT_PATH = os.path.dirname(JETSON_PATH)
MODEL_DIR = os.path.join(ROOT_PATH, "models")

ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "yolov7-tiny.onnx")
ENGINE_PATH = os.path.join(MODEL_DIR, "yolov7-tiny.engine")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # OLD SYNTAX for TRT 8.2.1
    # Set workspace to 1GB (Safe for Nano)
    config.max_workspace_size = 1 << 30

    # FP16
    if builder.platform_has_fast_fp16:
        print("✔ FP16 enabled")
        config.set_flag(trt.BuilderFlag.FP16)

    # Parse ONNX
    print(f"⏳ Loading and parsing {onnx_path}...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    print("⏳ Building TensorRT engine (this takes 10-15 mins)...")

    # OLD SYNTAX for TRT 8.2.1
    engine = builder.build_engine(network, config)

    if engine is None:
        print("❌ Engine build failed. Check if your ONNX has dynamic shapes.")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"✅ Engine saved: {engine_path}")
    return engine

if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ Error: {ONNX_MODEL_PATH} not found!")
    else:
        build_engine(ONNX_MODEL_PATH, ENGINE_PATH)
