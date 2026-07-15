import argparse
from pipeline import Pipeline
from pathlib import Path
import os
import sys

# Setup path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from trt_pipeline.tools import get_logger

logger = get_logger("RunPipeline")

def resolve_config_path(config_arg, root_dir, logger):
    """
    Resolve config file path with multiple fallbacks.
    Tries: exact path -> add .json -> root_dir/config/ -> ./config/
    """
    config_arg = str(config_arg)
    logger.info(f"Searching for config: {config_arg} with root_dir: {root_dir}")
    
    # Try 1: Use as-is (full path or relative)
    if os.path.exists(config_arg):
        logger.info(f"Config file found at provided path: {config_arg}")
        return os.path.abspath(config_arg)

    # Try 2: Add .json extension
    if os.path.exists(f"{config_arg}.json"):
        logger.info(f"Config file found with .json extension: {config_arg}.json")
        return os.path.abspath(f"{config_arg}.json")
    
    # Try 3: Look in root_dir/config/
    config_in_root = os.path.join(root_dir, "config", f"{config_arg}.json")
    logger.info(f"Trying: {config_in_root}")
    if os.path.exists(config_in_root):
        logger.info(f"✓ Config file found in root_dir/config/: {config_in_root}")
        return config_in_root
    
    # Try 4: Look in ./config/ (current working directory)
    config_in_cwd = os.path.join("config", f"{config_arg}.json")
    logger.info(f"Trying: {config_in_cwd}")
    if os.path.exists(config_in_cwd):
        logger.info(f"✓ Config file found in current working directory config/: {config_in_cwd}")
        return os.path.abspath(config_in_cwd)
    
    logger.error(f"Config file NOT FOUND! Checked: {config_in_root}, {config_in_cwd}")
    return config_in_root

def main():
    # Get script location and work backwards to find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))  # JETSON/src
    jetson_dir = os.path.dirname(script_dir)  # JETSON
    root_dir = os.path.dirname(jetson_dir)  # traffic (project root)
    
    logger.info(f"Root directory resolved to: {root_dir}")

    parser = argparse.ArgumentParser(description="Run Jetson TRT Pipeline with SORT")
    parser.add_argument("--config", required=True, help="Config filename (e.g., 'config_south_jetson2')")
    parser.add_argument("--engine", default=os.path.join(root_dir, "models", "yolov7-tiny.engine"), help="Path to TensorRT engine")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped images")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, root_dir, logger)
    logger.info(f"Using config file: {config_path}")

    pipeline = Pipeline(
        config_path=config_path,
        engine_path=args.engine,
        save_crop=args.save_crop,
        root_dir=root_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
