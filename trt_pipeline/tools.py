import json
import logging
import os
from typing import Optional


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def engine_settings(model_path: str) -> dict:
    """Build settings of a TensorRT engine (imgsz, batch, dynamic, half) from the .engine.json sidecar
    export_trt.py writes. Empty for non-engines or when the sidecar is missing."""
    if not str(model_path).endswith(".engine"):
        return {}
    try:
        with open(f"{model_path}.json", "r", encoding="utf-8") as f:
            settings = json.load(f)["settings"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def engine_imgsz(model_path: str) -> Optional[int]:
    """Input size a TensorRT engine was built for, or None if unknown."""
    try:
        return int(engine_settings(model_path)["imgsz"])
    except (KeyError, TypeError, ValueError):
        return None
