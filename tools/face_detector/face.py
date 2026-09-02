"""
face_detector.py

Face detection using OpenCV DNN (or Haar cascade fallback) on a live RTSP stream
or any video source, following the same structure as TrafficTracker.

Usage:
    python face_detector.py --config config_face_detection.json

Dependencies:
    pip install opencv-python numpy
    
    Optional (for GPU util logging):
    pip install pynvml
"""

import cv2 as cv
import numpy as np
import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trt_pipeline.video_stream import (
    VideoStream,
    AsyncImageSaver,
    capture_one_frame,
    is_network_stream,
    prepare_video_source,
)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

class FaceDetectorModel:
    """
    Wraps either:
      1. OpenCV DNN face detector  (res10_300x300_ssd_iter_140000.caffemodel)
      2. Haar cascade fallback     (haarcascade_frontalface_default.xml)

    The DNN model gives much better accuracy on RTSP / low-quality streams.
    Download weights once with:
        wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
        wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    """

    DNN_PROTOTXT   = "deploy.prototxt"
    DNN_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.logger = get_logger("FaceDetectorModel")
        self._net   = None
        self._haar  = None
        self._mode  = None
        self._load()

    def _load(self):
        # Try DNN first
        if os.path.exists(self.DNN_PROTOTXT) and os.path.exists(self.DNN_CAFFEMODEL):
            self._net  = cv.dnn.readNetFromCaffe(self.DNN_PROTOTXT, self.DNN_CAFFEMODEL)
            self._mode = "dnn"
            self.logger.info("Loaded OpenCV DNN face detector (Caffe SSD)")
            return

        # Fall back to Haar cascade bundled with OpenCV
        cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self._haar = cv.CascadeClassifier(cascade_path)
            self._mode = "haar"
            self.logger.warning(
                "DNN model files not found – falling back to Haar cascade. "
                "For better accuracy download deploy.prototxt + "
                "res10_300x300_ssd_iter_140000.caffemodel into the working directory."
            )
            return

        raise RuntimeError(
            "No face detector available. "
            "Install OpenCV with data files or provide DNN model weights."
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
            { "bbox": (x1, y1, x2, y2), "confidence": float }
        """
        if self._mode == "dnn":
            return self._detect_dnn(frame_bgr)
        return self._detect_haar(frame_bgr)

    def _detect_dnn(self, frame_bgr: np.ndarray) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        blob = cv.dnn.blobFromImage(
            cv.resize(frame_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append({"bbox": (x1, y1, x2, y2), "confidence": confidence})
        return results

    def _detect_haar(self, frame_bgr: np.ndarray) -> list[dict]:
        gray  = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        results = []
        for (x, y, fw, fh) in faces:
            results.append({
                "bbox":       (x, y, x + fw, y + fh),
                "confidence": 1.0,   # Haar doesn't return confidence
            })
        return results


# ─────────────────────────── annotator helper ────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and confidence scores onto a copy of the frame."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"face {conf:.2f}"
        (lw, lh), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(out, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 255, 0), -1)
        cv.putText(out, label, (x1, y1 - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


# ─────────────────────────── main tracker class ──────────────────────────────
class FaceDetector:
    def __init__(self, config_path: str):
        self.logger = get_logger("FaceDetector")
        self.config = load_config(config_path)

        # ── video ──────────────────────────────────────────────────────────
        video_cfg         = self.config["video"]
        self.video_source = prepare_video_source(video_cfg["path"])
        self.frame_stride = max(1, int(video_cfg.get("skip", 1)))
        self.max_frames   = video_cfg.get("max_frames")   # None = no limit
        self.logger.info(f"Video source : {self.video_source}")
        self.logger.info(f"Frame stride : {self.frame_stride}")

        # ── output dirs ────────────────────────────────────────────────────
        output_cfg          = self.config["output"]
        self.base_dir       = output_cfg["base_dir"]
        self.save_crop      = output_cfg.get("save_crop", True)
        self.save_annotated = output_cfg.get("save_annotated", True)

        self.crops_dir      = os.path.join(self.base_dir, "faces")
        self.annotated_dir  = os.path.join(self.base_dir, "annotated")
        os.makedirs(self.crops_dir,     exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

        # ── processing ─────────────────────────────────────────────────────
        proc_cfg              = self.config.get("processing", {})
        self.conf_threshold   = float(proc_cfg.get("confidence_threshold", 0.5))
        self.queue_size       = int(proc_cfg.get("video_queue_size", 2))

        cv.setNumThreads(proc_cfg.get("opencv_threads", 0))
        cv.ocl.setUseOpenCL(proc_cfg.get("use_opencl", False))

        # ── model ──────────────────────────────────────────────────────────
        self.model = FaceDetectorModel(confidence_threshold=self.conf_threshold)

        # ── async image saver ──────────────────────────────────────────────
        self.image_saver = AsyncImageSaver()

        # ── counters ───────────────────────────────────────────────────────
        self.total_detections = 0
        self.total_faces_saved = 0

        self.logger.info("FaceDetector initialized successfully")

    # ─────────────────────────── run ─────────────────────────────────────────
    def run(self):
        # Test network stream before spinning up thread
        if is_network_stream(self.video_source):
            self.logger.info("Testing network stream connection…")
            if capture_one_frame(self.video_source) is None:
                self.logger.error("Failed to connect to network stream. Aborting.")
                return
            self.logger.info("Network stream connection successful!")

        stream = VideoStream(
            video_path=self.video_source,
            skip=self.frame_stride,
            queue_size=self.queue_size,
        )

        total_start = time.perf_counter()
        frame_count = 0

        try:
            while True:
                item = stream.read()
                if item is None:
                    break

                frame_idx, frame_bgr = item
                frame_count += 1

                # ── progress log ──────────────────────────────────────────
                if frame_idx % 10 == 0:
                    print(f"\rframe: {frame_idx}  faces saved: {self.total_faces_saved}",
                          end="", flush=True)

                if is_network_stream(self.video_source) and frame_idx % 100 == 0:
                    self.logger.info(
                        f"Processed {frame_idx} frames | "
                        f"Detections so far: {self.total_detections} | "
                    )

                # ── max_frames guard ──────────────────────────────────────
                if self.max_frames and frame_idx >= self.max_frames:
                    self.logger.info(f"Reached max_frames limit: {self.max_frames}")
                    break

                # ── detect faces ──────────────────────────────────────────
                detections = self.model.detect(frame_bgr)
                self.total_detections += len(detections)

                # ── save outputs ──────────────────────────────────────────
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                if self.save_crop and detections:
                    for det_idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det["bbox"]
                        crop      = frame_bgr[y1:y2, x1:x2]
                        conf_str  = f"{det['confidence']:.2f}".replace(".", "p")
                        save_path = os.path.join(
                            self.crops_dir,
                            f"frame{frame_idx:06d}_{ts}_face{det_idx}_conf{conf_str}.jpg",
                        )
                        self.image_saver.save(save_path, crop)
                        self.total_faces_saved += 1

                if self.save_annotated and detections:
                    annotated  = draw_detections(frame_bgr, detections)
                    annot_path = os.path.join(
                        self.annotated_dir,
                        f"frame{frame_idx:06d}_{ts}.jpg",
                    )
                    self.image_saver.save(annot_path, annotated)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            stream.stop()
            self.image_saver.stop()

        # ── final summary ─────────────────────────────────────────────────
        total_time = time.perf_counter() - total_start
        print()   # newline after \r progress
        self.logger.info(
            f"Done. Total time : {total_time:.2f}s | "
            f"Frames processed: {frame_count} | "
            + (f"FPS: {frame_count / total_time:.2f} | " if frame_count > 0 else "")
            + f"Total detections: {self.total_detections} | "
            f"Faces saved: {self.total_faces_saved}"
        )
        self.logger.info(f"Crops     → {self.crops_dir}")
        self.logger.info(f"Annotated → {self.annotated_dir}")


# ─────────────────────────── entry point ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detector on RTSP / video file")
    parser.add_argument(
        "--config",
        default="config_face_detection.json",
        help="Path to JSON config file",
    )
    args = parser.parse_args()
    FaceDetector(config_path=args.config).run()