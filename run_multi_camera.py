"""
Multi-camera traffic measurement runner.

Pipeline per camera: capture -> (batched) YOLO -> ByteTrack -> class voting ->
lane occupancy with queued/moving/unknown state -> virtual gates.
Every publish interval one payload (schema 2.0, docs/CONTROLLER_CONTRACT.md)
describes all cameras, with per-camera status and per-lane validity.

Design rules:
* Cameras are independent. A missing or slow camera never blocks the others;
  its lanes are published as invalid (unknown), never as zero.
* Each frame carries its capture time. Frames older than ``max_frame_age_s``
  are not processed, and lanes whose last observation is older than
  ``max_observation_age_s`` are invalid.
* Motion, gate continuity and coverage use per-camera observation clocks, not
  processing FPS.
* A camera discontinuity (reconnect, file loop) resets that camera's tracker,
  motion, voting and gate history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import shapely

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trt_pipeline.camera_health import HealthSettings, ImageHealthMonitor
from trt_pipeline.controller_main import MainControllerAdapter
from trt_pipeline.gates import GateFlowManager, VirtualGate
from trt_pipeline.lane_validation import validate_config_file
from trt_pipeline.motion import MotionStateClassifier, QueueSettings
from trt_pipeline.payload import LaneMetricsManager, PayloadBuilder, invalidate_lane, iso_utc
from trt_pipeline.stream import FileReplaySource, FramePacket, StreamBufferWorker, is_live_source, redact_source
from trt_pipeline.voter import TrackClassVotingFilter

logger = logging.getLogger("ProductionMultiCameraRunner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def initial_config(config_path: str) -> dict:
    """Loads JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS = {
    "north": "config/config_north.json",
    "south": "config/config_south.json",
    "east": "config/config_east.json",
    "west": "config/config_west.json",
    "northeast": "config/config_northeast.json",
}
DEFAULT_MODEL = "models/yolo26s_thai_traffic.pt"

# Per-camera JSON sections this runner does not read. Settings come from the CLI.
IGNORED_CONFIG_SECTIONS = ("model", "tracker", "mqtt", "processing", "tracking", "density", "output", "classes")
IGNORED_LANE_METRIC_KEYS = ("publish_interval_frames", "queue_speed_threshold", "enabled")

COCO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VALID_STATUSES = ("ok", "degraded")
# queue:    lane is part of the approach's stop-line queue (used for signal timing)
# upstream: lane further back on the approach (arrivals / spill-back context only);
#           kept out of queue totals so two cameras on one approach never add up
LANE_ROLES = ("queue", "upstream")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class CameraRuntime:
    """Mutable state for one camera."""

    idx: int
    name: str
    camera_id: str
    config_path: str
    source: Any
    lanes: Dict[str, Dict[str, Any]]
    metrics: LaneMetricsManager
    motion: MotionStateClassifier
    tracker: Any
    health: ImageHealthMonitor
    anchor: str = "bottom_center"
    expected_resolution: Optional[Tuple[int, int]] = None
    calibration_revision: Optional[str] = None
    epoch: int = 0
    frames_seen: int = 0
    inferences: int = 0
    epoch_inferences: int = 0
    stale_frames_dropped: int = 0
    last_frame: Optional[np.ndarray] = None
    last_tracked: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    last_obs_wall: Optional[float] = None
    last_obs_mono: Optional[float] = None
    last_obs_t: Optional[float] = None
    frame_invalid_reason: Optional[str] = None
    health_invalid: bool = False
    maintenance_active: bool = False
    fps_window: List[float] = field(default_factory=list)


class BatchedCameraPipeline:
    """Orchestrates independent camera sources, one shared model and per-camera analytics."""

    def __init__(
        self,
        camera_names: List[str],
        config_paths: List[str],
        video_sources: List[Any],
        model_path: str,
        device: str = "cuda:0",
        conf: float = 0.10,
        target_fps: float = 25.0,
        skip_frames: int = 1,
        buffer_size: int = 1,
        display: bool = False,
        nvenc_out: Optional[str] = None,
        pub_interval: float = 1.0,
        mqtt_broker: Optional[str] = None,
        mqtt_topic: Optional[str] = None,
        intersection_id: str = "INT-001",
        voting_window: int = 15,
        pickup_bias: float = 1.15,
        enable_voting: bool = True,
        tracker_type: str = "byetrack",
        imgsz: int = 640,
        source_id: Optional[str] = None,
        track_thresh: float = 0.40,
        low_thresh: float = 0.10,
        match_thresh: float = 0.70,
        track_max_age: int = 30,
        max_frame_age_s: float = 1.0,
        max_observation_age_s: float = 1.5,
        replay_fps: Optional[float] = None,
        health_topic: Optional[str] = None,
        mqtt_tls: Optional[Dict[str, Any]] = None,
        model_sha256: Optional[str] = None,
        record_path: Optional[str] = None,
        health_file: Optional[str] = None,
        max_consecutive_errors: int = 50,
        maintenance_file: Optional[str] = None,
        model: Any = None,
        publisher: Any = None,
        sources: Optional[List[Any]] = None,
        output_mode: str = "shadow",
        controller_config: Optional[str] = None,
        allow_replay_controller: bool = False,
    ):
        if not (len(camera_names) == len(config_paths) == len(video_sources)):
            raise ValueError("camera_names, config_paths and video_sources must have the same length")
        self.num_streams = len(camera_names)
        self.camera_names = list(camera_names)
        self.config_paths = list(config_paths)
        self.video_sources = list(video_sources)
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self.target_fps = float(target_fps)
        self.skip_frames = max(0, int(skip_frames))
        self.display = display
        self.nvenc_out = nvenc_out
        self.pub_interval = float(pub_interval)
        self.intersection_id = intersection_id
        self.source_id = source_id or f"VISION-{intersection_id}"
        if output_mode not in ("shadow", "controller"):
            raise ValueError("output_mode must be 'shadow' or 'controller'")
        self.output_mode = output_mode
        if output_mode == "shadow":
            self.mqtt_topic = mqtt_topic or os.getenv("VISION_SHADOW_TOPIC") or "traffic/counts/shadow"
            self.health_topic = health_topic or f"traffic/health/shadow/{self.source_id}"
            live_topic = os.getenv("TRAFFIC_COUNTS_TOPIC") or os.getenv("MQTT_TOPIC") or "traffic/counts"
            live_health = self.health_topic.startswith("traffic/health/") and len(self.health_topic.split("/")) == 3
            if self.mqtt_topic in ("traffic/counts", live_topic) or live_health:
                raise ValueError("Live controller topics require --output-mode controller")
        else:
            self.mqtt_topic = mqtt_topic or os.getenv("TRAFFIC_COUNTS_TOPIC") or os.getenv("MQTT_TOPIC") or "traffic/counts"
            self.health_topic = health_topic or f"traffic/health/{self.source_id}"
        self.tracker_type = tracker_type.lower()
        self.imgsz = int(imgsz)
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.max_frame_age_s = float(max_frame_age_s)
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.batch_wait_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.04

        if self.conf > self.low_thresh:
            logger.warning(
                f"Detector confidence {self.conf} is above the tracker's low threshold {self.low_thresh}: "
                f"ByteTrack's second association stage will receive no detections."
            )

        # 1. Calibration preflight (before any model, capture or network resources)
        all_errors, all_warnings, raw_configs, lane_dicts = [], [], [], []
        for idx, cfg_path in enumerate(self.config_paths):
            context_str = f"{self.camera_names[idx]} ({os.path.basename(cfg_path)})"
            report = validate_config_file(cfg_path, context=context_str)
            all_errors.extend(report.errors)
            all_warnings.extend(report.warnings)
            if report.is_valid and report.raw_config is not None:
                raw_configs.append(report.raw_config)
                raw_lanes = report.raw_config.get("lane_metrics", {}).get("lanes", {})
                lane_dict = {}
                for lane_id, poly in report.valid_polygons.items():
                    l_info = raw_lanes.get(lane_id, {}) if isinstance(raw_lanes, dict) else {}
                    lane_dict[lane_id] = {"direction": l_info.get("direction"), "polygon": poly,
                                          "role": l_info.get("role", "queue")}
                    if lane_dict[lane_id]["direction"] not in ("N", "S", "E", "W"):
                        all_errors.append(_ConfigError(context_str, lane_id,
                                                       "lane 'direction' must be one of N, S, E, W"))
                    if lane_dict[lane_id]["role"] not in LANE_ROLES:
                        all_errors.append(_ConfigError(context_str, lane_id,
                                                       f"lane 'role' must be one of {LANE_ROLES}"))
                lane_dicts.append(lane_dict)

        for warn in all_warnings:
            logger.warning(f"Calibration warning [{warn.context}]: {warn.message}")

        seen_lanes: Dict[str, str] = {}
        for idx, ld in enumerate(lane_dicts):
            for lane_id in ld:
                if lane_id in seen_lanes:
                    all_errors.append(_ConfigError(self.camera_names[idx], lane_id,
                                                   f"duplicate lane ID (also defined for {seen_lanes[lane_id]})"))
                seen_lanes[lane_id] = self.camera_names[idx]

        if all_errors:
            error_details = "\n".join(
                f"  - [{e.context}] Lane '{e.lane_id}': {e.reason}" if e.lane_id else f"  - [{e.context}] Config: {e.reason}"
                for e in all_errors
            )
            msg = f"Fatal calibration validation error(s) detected during startup:\n{error_details}"
            logger.error(msg)
            raise ValueError(msg)

        self.camera_configs = raw_configs
        self.lane_configs = lane_dicts
        self._log_ignored_settings()
        self.controller_adapter = None
        self.controller_delivery = None
        if self.output_mode == "controller":
            self.controller_adapter = MainControllerAdapter(
                controller_config or os.path.join(REPO_ROOT, "config", "controller_main.json"),
                self.intersection_id, self.lane_configs, self.pub_interval)
            self.controller_delivery = {
                "profile": "smart-traffic-sys/main", "state": "starting", "blockers": ["no_snapshot"],
                "cameraId": self.controller_adapter.camera_id, "accepted": 0, "suppressed": 0,
                "lastAcceptedAt": None,
            }
        self.last_payload = None

        # 2. Gates (validated: unique IDs, explicit direction and type)
        camera_ids = [self._camera_id_for(i) for i in range(self.num_streams)]
        self.gate_manager = GateFlowManager(camera_names=self.camera_names, camera_ids=camera_ids)
        for idx, cfg in enumerate(self.camera_configs):
            for g_cfg in cfg.get("gates", []) or []:
                try:
                    self.gate_manager.add_gate(VirtualGate(
                        gate_id=g_cfg["gate_id"],
                        cam_idx=idx,
                        p1=g_cfg["p1"],
                        p2=g_cfg["p2"],
                        gate_type=g_cfg.get("type", "stopline"),
                        direction_vec=g_cfg.get("direction"),
                        label=g_cfg.get("label", g_cfg["gate_id"]),
                        target_dir=g_cfg.get("target_dir"),
                    ))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid gate in {self.config_paths[idx]}: {exc}") from exc

        # 3. Per-camera runtime
        self.cams: List[CameraRuntime] = []
        for idx in range(self.num_streams):
            cfg = self.camera_configs[idx]
            lm = cfg.get("lane_metrics", {}) if isinstance(cfg.get("lane_metrics"), dict) else {}
            anchor = lm.get("anchor", "bottom_center")
            if anchor not in ("bottom_center", "center"):
                raise ValueError(f"{self.config_paths[idx]}: lane_metrics.anchor must be 'bottom_center' or 'center'")
            calib = cfg.get("calibration", {}) if isinstance(cfg.get("calibration"), dict) else {}
            res = calib.get("resolution")
            reference = None
            if calib.get("reference_image"):
                ref_path = os.path.join(os.path.dirname(os.path.abspath(self.config_paths[idx])), calib["reference_image"])
                if os.path.exists(ref_path):
                    import cv2 as cv
                    reference = cv.imread(ref_path)
                else:
                    logger.warning(f"[{self.camera_names[idx]}] calibration reference image missing: {ref_path}")
            self.cams.append(CameraRuntime(
                idx=idx,
                name=self.camera_names[idx],
                camera_id=camera_ids[idx],
                config_path=self.config_paths[idx],
                source=None,
                lanes=self.lane_configs[idx],
                metrics=LaneMetricsManager(self.lane_configs[idx], camera_id=camera_ids[idx]),
                motion=MotionStateClassifier(QueueSettings.from_config(lm.get("queue"))),
                tracker=self._make_tracker(track_max_age),
                health=ImageHealthMonitor(reference=reference, settings=HealthSettings.from_config(cfg.get("camera_health"))),
                anchor=anchor,
                expected_resolution=(int(res[0]), int(res[1])) if isinstance(res, (list, tuple)) and len(res) == 2 else None,
                calibration_revision=calib.get("revision"),
            ))
        # Compatibility views used by display/tests
        self.metrics_managers = [c.metrics for c in self.cams]
        self.trackers = [c.tracker for c in self.cams]

        # 4. Sources
        if sources is not None:
            if len(sources) != self.num_streams:
                raise ValueError("sources must match camera count")
            self.replay_mode = all(isinstance(s, FileReplaySource) for s in sources)
            for cam, src in zip(self.cams, sources):
                cam.source = src
        else:
            live = [is_live_source(s) for s in self.video_sources]
            if any(live) and not all(live):
                raise ValueError("Mixing recorded files and live streams in one run is not supported")
            self.replay_mode = not any(live)
            for cam, src in zip(self.cams, self.video_sources):
                cam.source = (FileReplaySource(cam.name, src, replay_fps=replay_fps) if self.replay_mode
                              else StreamBufferWorker(name=cam.name, source=src, target_fps=self.target_fps))
        self.is_file_mode = self.replay_mode
        if (self.controller_adapter and not allow_replay_controller
                and (self.replay_mode or any(isinstance(c.source, FileReplaySource) for c in self.cams))):
            raise ValueError("Recorded footage requires shadow output; --allow-replay-controller is only for isolated tests")

        # 5. Model
        self.model_sha256 = None
        self._model_warmed = model is not None
        self.model_warmup_sec = 0.0 if model is not None else None
        if model is not None:
            self.model = model
            raw_names = getattr(model, "names", None) or {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "three_wheeler"}
        else:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model weights not found: {self.model_path}")
            self.model_sha256 = sha256_file(self.model_path)
            if model_sha256 and self.model_sha256.lower() != model_sha256.lower():
                raise ValueError(f"Model SHA-256 mismatch: expected {model_sha256}, got {self.model_sha256}")
            logger.info(f"Loading model '{self.model_path}' (sha256={self.model_sha256}) onto '{self.device}'")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path, task="detect")
            raw_names = getattr(self.model, "names", None)

        if isinstance(raw_names, dict):
            model_names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, list):
            model_names = {i: str(v) for i, v in enumerate(raw_names)}
        else:
            model_names = COCO_CLASSES
        traffic_keywords = {"car", "motorcycle", "bus", "truck", "three_wheeler", "tuktuk", "bicycle"}
        if len(model_names) <= 15 and any(v.lower() in traffic_keywords for v in model_names.values()):
            self.class_names = model_names
            self.target_classes = None
        else:
            self.class_names = COCO_CLASSES
            self.target_classes = list(COCO_CLASSES.keys())
        self.default_car_cls = next((k for k, v in self.class_names.items() if v.lower() == "car"), 0)
        truck_cls_id = next((k for k, v in self.class_names.items() if v.lower() == "truck"), 3)
        self.class_voter = TrackClassVotingFilter(
            num_streams=self.num_streams,
            window_size=voting_window,
            car_truck_bias=pickup_bias,
            car_cls_id=self.default_car_cls,
            truck_cls_id=truck_cls_id,
            enabled=enable_voting,
        )

        # 6. Display (optional)
        self.display_worker = None
        if self.display or self.nvenc_out:
            from trt_pipeline.display import AsyncDisplayWorker, NVENCVideoWriter, is_nvenc_available
            nvenc_writer = None
            if self.nvenc_out:
                if is_nvenc_available():
                    cols = min(4, self.num_streams)
                    rows = (self.num_streams + cols - 1) // cols
                    nvenc_writer = NVENCVideoWriter(self.nvenc_out, width=cols * 480, height=rows * 270, fps=self.target_fps)
                else:
                    logger.warning("Hardware NVENC is not available on this host.")
            if self.display or nvenc_writer:
                self.display_worker = AsyncDisplayWorker(
                    display=self.display,
                    nvenc_writer=nvenc_writer,
                    window_name=f"Smart Traffic Vision ({self.num_streams} cameras)",
                    class_names=self.class_names,
                )

        # 7. Delivery
        if publisher is not None:
            self.publisher = publisher
        else:
            from trt_pipeline.publisher import MQTTPublisher
            tls = mqtt_tls or {}
            self.publisher = MQTTPublisher(
                broker_url=mqtt_broker, topic=self.mqtt_topic, qos=0 if self.controller_adapter else 1,
                client_id=f"vision_{self.output_mode}_{self.source_id}",
                health_topic=self.health_topic,
                tls_ca=tls.get("ca"), tls_cert=tls.get("cert"), tls_key=tls.get("key"),
                tls_insecure=bool(tls.get("insecure")),
            )
        self.payload_builder = PayloadBuilder(intersection_id=self.intersection_id, camera_id=self.source_id)

        self.recorder = _make_recorder(record_path) if record_path else None
        self.health_file = health_file
        self.maintenance_file = maintenance_file
        self._maintenance: Dict[str, str] = {}
        self._maintenance_mtime: Optional[float] = None

        self.running = False
        self.batch_idx = 0
        self.total_processed_batches = 0
        self.total_inferred_batches = 0
        self.consecutive_errors = 0
        self.total_errors = 0
        self.started_wall = time.time()
        self._last_pub_mono = time.monotonic()
        self._current_fps = 0.0
        self._fps_count = 0
        self._fps_mono = time.monotonic()

    # ------------------------------------------------------------------ setup helpers
    def _camera_id_for(self, idx: int) -> str:
        info = self.camera_configs[idx].get("camera_info", {}) if idx < len(self.camera_configs) else {}
        cid = info.get("camera_id") if isinstance(info, dict) else None
        return str(cid) if cid else self.camera_names[idx]

    def _make_tracker(self, max_age: int):
        min_hits = 1 if self.skip_frames > 0 else 2
        if self.tracker_type == "byetrack":
            from algorithm.byetrack import ByteTrack
            return ByteTrack(track_thresh=self.track_thresh, low_thresh=self.low_thresh,
                             match_thresh=self.match_thresh, max_age=max_age, min_hits=min_hits)
        from algorithm.sort import Sort
        return Sort(max_age=max_age, min_hits=min_hits, iou_threshold=0.3)

    def _log_ignored_settings(self) -> None:
        for idx, cfg in enumerate(self.camera_configs):
            ignored = [k for k in IGNORED_CONFIG_SECTIONS if k in cfg]
            lm = cfg.get("lane_metrics", {}) if isinstance(cfg.get("lane_metrics"), dict) else {}
            ignored += [f"lane_metrics.{k}" for k in IGNORED_LANE_METRIC_KEYS if k in lm]
            if ignored:
                logger.info(f"[{self.camera_names[idx]}] config keys not used by this runner "
                            f"(settings come from the command line): {', '.join(ignored)}")

    def effective_settings(self) -> Dict[str, Any]:
        """Everything that determines the measurements, logged at startup and sent in payload meta."""
        return {
            "sourceId": self.source_id,
            "outputMode": self.output_mode,
            "countsTopic": self.mqtt_topic,
            "healthTopic": self.health_topic,
            "controllerProfile": None if self.controller_adapter is None else {
                "config": self.controller_adapter.config_path,
                "cameraId": self.controller_adapter.camera_id,
                "expectedLanes": self.controller_adapter.lanes,
                "countKind": "occupancy",
                "maxObservationAgeMs": self.controller_adapter.max_age_ms,
                "freshnessMs": self.controller_adapter.freshness_ms,
            },
            "intersectionId": self.intersection_id,
            "mode": "replay" if self.replay_mode else "live",
            "model": {"path": self.model_path, "sha256": self.model_sha256, "imgsz": self.imgsz,
                      "warmupSec": self.model_warmup_sec,
                      "conf": self.conf, "device": self.device},
            "tracker": {"type": self.tracker_type, "trackThresh": self.track_thresh,
                        "lowThresh": self.low_thresh, "matchThresh": self.match_thresh},
            "skipFrames": self.skip_frames,
            "publishIntervalSec": self.pub_interval,
            "maxFrameAgeSec": self.max_frame_age_s,
            "maxObservationAgeSec": self.max_observation_age_s,
            "cameras": [{
                "name": c.name, "cameraId": c.camera_id, "config": c.config_path,
                "source": redact_source(self.video_sources[c.idx]), "anchor": c.anchor,
                "calibrationRevision": c.calibration_revision,
                "expectedResolution": c.expected_resolution,
                "queueUnits": c.motion.settings.units,
            } for c in self.cams],
        }

    # ------------------------------------------------------------------ lifecycle
    def warmup(self) -> None:
        """Initialize inference before camera frames can age during a cold GPU setup."""
        if self._model_warmed:
            return
        if self.health_file:
            _write_json_atomic(self.health_file, {"status": "starting", "reason": "model_warmup",
                               "sourceId": self.source_id, "updatedAt": iso_utc(time.time()),
                               "validCameras": 0, "totalCameras": self.num_streams})
        logger.info("Warming model before camera intake starts")
        started = time.monotonic()
        frames = []
        for cam in self.cams:
            width, height = cam.expected_resolution or (self.imgsz, self.imgsz)
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
        kwargs = {"verbose": False, "device": self.device, "conf": self.conf, "imgsz": self.imgsz}
        if self.target_classes is not None:
            kwargs["classes"] = self.target_classes
        results = self.model(frames, **kwargs)
        if len(results) != self.num_streams:
            raise RuntimeError("Model warmup returned an incomplete camera batch")
        self.model_warmup_sec = round(time.monotonic() - started, 3)
        self._model_warmed = True
        logger.info(f"Model warmup completed in {self.model_warmup_sec}s")

    def start(self) -> None:
        self.running = True
        self.warmup()
        if not self.running:
            return
        self.publisher.start()
        for cam in self.cams:
            cam.source.start()
        if self.display_worker:
            self.display_worker.start()
        logger.info("Effective settings: " + json.dumps(self.effective_settings()))

    def stop(self) -> None:
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        self.running = False
        for cam in getattr(self, "cams", []):
            try:
                if cam.source is not None:
                    cam.source.stop()
            except Exception as exc:
                logger.warning(f"[{cam.name}] error stopping source: {exc}")
        if getattr(self, "display_worker", None):
            self.display_worker.stop()
        if getattr(self, "publisher", None):
            try:
                self.publisher.stop({"status": "offline", "reason": "stopped", "sourceId": self.source_id})
            except TypeError:
                self.publisher.stop()
        logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------ per-camera state
    def _reset_camera(self, cam: CameraRuntime, reset_health: bool = True) -> None:
        """Discontinuity: track IDs, motion and crossings from the previous epoch must not carry over."""
        cam.tracker = self._make_tracker(getattr(cam.tracker, "max_age", 30))
        self.trackers[cam.idx] = cam.tracker
        cam.motion.reset()
        cam.metrics.reset()
        if reset_health:
            cam.health.reset()
        cam.last_tracked = np.empty((0, 6))
        cam.last_obs_wall = None
        cam.last_obs_mono = None
        cam.last_obs_t = None
        cam.frames_seen = 0
        cam.epoch_inferences = 0
        self.class_voter.reset(cam_idx=cam.idx)
        self.gate_manager.reset_camera(cam.idx)
        logger.info(f"[{cam.name}] observation discontinuity (epoch {cam.epoch}): temporal state reset")

    def _anchors(self, cam: CameraRuntime, objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = (objs[:, 0] + objs[:, 2]) * 0.5
        ys = objs[:, 3] if cam.anchor == "bottom_center" else (objs[:, 1] + objs[:, 3]) * 0.5
        return xs, ys

    def _evaluate_lanes(self, cam: CameraRuntime, objs: np.ndarray, t: float) -> None:
        """
        Rebuilds the camera's current lane occupancy from one observed frame.
        Each track is assigned to at most one lane: the first configured lane whose
        polygon covers its anchor point (boundary included).
        """
        metrics = cam.metrics
        metrics.reset()
        if objs is None or len(objs) == 0:
            return

        xs, ys = self._anchors(cam, objs)
        heights = np.maximum(1.0, objs[:, 3] - objs[:, 1])
        states = {}
        for i in range(len(objs)):
            tid = int(objs[i, 4])
            states[tid] = cam.motion.update(tid, (float(xs[i]), float(ys[i])), float(heights[i]), t)

        assigned = np.zeros(len(objs), dtype=bool)
        seen: set = set()
        for lane_id, linfo in cam.lanes.items():
            poly = linfo["polygon"]
            if poly is None or poly.is_empty:
                continue
            inside = shapely.intersects_xy(poly, xs, ys)
            for i in np.where(inside & ~assigned)[0]:
                tid = int(objs[i, 4])
                if tid in seen:
                    continue
                assigned[i] = True
                seen.add(tid)
                cls_id = int(objs[i, 5]) if objs.shape[1] >= 6 else self.default_car_cls
                metrics.register_vehicle(lane_id=lane_id, track_id=tid,
                                         vehicle_class=self.class_names.get(cls_id, "car"),
                                         state=states[tid])
            if assigned.all():
                break

    def _track(self, cam: CameraRuntime, dets: np.ndarray) -> np.ndarray:
        """Runs the tracker and returns (M, 6) [x1, y1, x2, y2, track_id, voted_class]."""
        idx = cam.idx
        if self.tracker_type == "byetrack":
            out = cam.tracker.update(dets)
            if len(out) == 0:
                return np.empty((0, 6))
            res = out[:, :6].astype(float).copy()
            for i, row in enumerate(out):
                # class/score of the detection the tracker matched this frame
                res[i, 5] = self.class_voter.update(cam_idx=idx, track_id=int(row[4]),
                                                    raw_cls_id=int(row[5]), conf=float(row[6]) if len(row) > 6 else 1.0)
            return res

        hi = dets[dets[:, 4] >= self.track_thresh] if len(dets) else np.empty((0, 6))
        out = cam.tracker.update(hi[:, :5] if len(hi) else np.empty((0, 5)))
        if len(out) == 0:
            return np.empty((0, 6))
        res = np.zeros((len(out), 6))
        res[:, :5] = out[:, :5]
        matched = _match_tracks_to_dets(out[:, :4], hi[:, :4]) if len(hi) else {}
        for i, row in enumerate(out):
            tid = int(row[4])
            if i in matched:
                d = hi[matched[i]]
                res[i, 5] = self.class_voter.update(cam_idx=idx, track_id=tid, raw_cls_id=int(d[5]), conf=float(d[4]))
            else:
                res[i, 5] = self.class_voter.get_class(cam_idx=idx, track_id=tid, fallback=self.default_car_cls)
        return res

    @staticmethod
    def _extract_dets(result: Any) -> np.ndarray:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6))
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = boxes.cls.cpu().numpy().reshape(-1, 1)
            return np.hstack([xyxy, conf, cls]).astype(float)
        except AttributeError:
            rows = []
            for box in boxes:
                x1, y1, x2, y2 = np.asarray(box.xyxy[0].cpu().numpy(), dtype=float)
                rows.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
            return np.array(rows, dtype=float) if rows else np.empty((0, 6))

    # ------------------------------------------------------------------ main loop
    def _collect_packets(self) -> Dict[int, FramePacket]:
        packets: Dict[int, FramePacket] = {}
        if self.replay_mode:
            for cam in self.cams:
                p = cam.source.poll()
                if p is not None:
                    packets[cam.idx] = p
            return packets
        deadline = time.monotonic() + self.batch_wait_s
        while self.running:
            for cam in self.cams:
                if cam.idx not in packets:
                    p = cam.source.poll()
                    if p is not None:
                        packets[cam.idx] = p
            if len(packets) == self.num_streams or time.monotonic() >= deadline:
                break
            time.sleep(0.002)
        return packets

    def step(self) -> None:
        """One iteration: take available frames, infer on due cameras, update analytics, maybe publish."""
        self._load_maintenance()
        packets = self._collect_packets()
        now_wall = time.time()
        due: List[Tuple[CameraRuntime, FramePacket]] = []

        for idx, p in packets.items():
            cam = self.cams[idx]
            if cam.maintenance_active:
                continue
            if p.epoch != cam.epoch:
                first = cam.epoch == 0
                cam.epoch = p.epoch
                if not first:
                    self._reset_camera(cam)
            frame_age = p.age_s(now_wall)
            if frame_age > self.max_frame_age_s:
                cam.stale_frames_dropped += 1
                continue
            captured_mono = p.captured_mono if p.captured_mono is not None else time.monotonic() - frame_age
            if cam.last_obs_mono is not None and captured_mono - cam.last_obs_mono > self.max_observation_age_s:
                self._reset_camera(cam)
            h, w = p.frame.shape[:2]
            if cam.expected_resolution and (w, h) != cam.expected_resolution:
                if cam.frame_invalid_reason != "resolution_mismatch":
                    self._reset_camera(cam)
                    logger.error(f"[{cam.name}] frame {w}x{h} does not match calibration "
                                 f"{cam.expected_resolution[0]}x{cam.expected_resolution[1]}")
                cam.frame_invalid_reason = "resolution_mismatch"
                cam.last_frame = p.frame
                continue
            cam.frame_invalid_reason = None
            cam.health.check(p.frame, time.monotonic())
            cam.last_frame = p.frame
            if cam.health.verdict():
                if not cam.health_invalid:
                    self._reset_camera(cam, reset_health=False)
                cam.health_invalid = True
                continue
            cam.health_invalid = False
            cam.frames_seen += 1
            if (cam.frames_seen - 1) % (self.skip_frames + 1) == 0:
                due.append((cam, p))

        if due:
            self.batch_idx += 1
            infer_kwargs = {"verbose": False, "device": self.device, "conf": self.conf, "imgsz": self.imgsz}
            if self.target_classes is not None:
                infer_kwargs["classes"] = self.target_classes
            results = self.model([p.frame for _, p in due], **infer_kwargs)
            self.total_inferred_batches += 1
            self._fps_count += 1

            for (cam, p), result in zip(due, results):
                dets = self._extract_dets(result)
                objs = self._track(cam, dets)
                self.class_voter.prune(cam_idx=cam.idx, active_track_ids={int(o[4]) for o in objs})
                cam.motion.prune(p.obs_t)
                self._evaluate_lanes(cam, objs, p.obs_t)
                self.gate_manager.update_tracks(cam.idx, objs, now=p.obs_t)
                cam.last_tracked = objs
                cam.last_obs_wall = p.captured_wall
                cam.last_obs_mono = p.captured_mono if p.captured_mono is not None else time.monotonic() - p.age_s()
                cam.last_obs_t = p.obs_t
                cam.inferences += 1
                cam.epoch_inferences += 1

        self.total_processed_batches = self.batch_idx
        mono = time.monotonic()
        if mono - self._fps_mono >= 1.0:
            self._current_fps = self._fps_count / (mono - self._fps_mono)
            self._fps_count = 0
            self._fps_mono = mono

        if self.display_worker:
            self._submit_display()

        if mono - self._last_pub_mono >= self.pub_interval:
            self._last_pub_mono = mono
            self.publish_once()

    # ------------------------------------------------------------------ reporting
    def camera_status(self, cam: CameraRuntime, now_wall: Optional[float] = None) -> Dict[str, Any]:
        now_wall = now_wall if now_wall is not None else time.time()
        src = cam.source.status() if cam.source is not None else {"state": "unknown"}
        age = None
        if cam.last_obs_mono is not None:
            age = max(0.0, time.monotonic() - cam.last_obs_mono)
        elif cam.last_obs_wall is not None:
            age = max(0.0, now_wall - cam.last_obs_wall)
        reason = None
        if cam.name in self._maintenance or cam.camera_id in self._maintenance:
            status = "maintenance"
            reason = self._maintenance.get(cam.name) or self._maintenance.get(cam.camera_id) or "maintenance"
        elif cam.frame_invalid_reason:
            status, reason = "invalid", cam.frame_invalid_reason
        elif cam.health.verdict():
            status, reason = "invalid", cam.health.verdict()
        elif age is None:
            status = src.get("state") if src.get("state") in ("offline", "reconnecting") else "starting"
            reason = status
        elif age > self.max_observation_age_s:
            state = src.get("state")
            status = state if state in ("offline", "reconnecting") else "stale"
            reason = status
        elif cam.epoch_inferences < cam.tracker.min_hits:
            status, reason = "starting", "tracker_warmup"
        elif cam.health.degraded():
            status, reason = "degraded", cam.health.degraded()
        else:
            status = "ok"
        return {
            "cameraId": cam.camera_id,
            "name": cam.name,
            "status": status,
            "reason": reason,
            "observedAt": iso_utc(cam.last_obs_wall) if cam.last_obs_wall else None,
            "ageMs": None if age is None else int(age * 1000),
            "calibrationRevision": cam.calibration_revision,
            "inferences": cam.inferences,
            "staleFramesDropped": cam.stale_frames_dropped,
            "framesDropped": src.get("framesDropped"),
            "reconnects": src.get("reconnects"),
            "epoch": cam.epoch,
            "trackedVehicles": int(len(cam.last_tracked)),
            "motionTracks": len(cam.motion),
            "shiftPx": cam.health.flags.get("shiftPx"),
        }

    def _load_maintenance(self) -> None:
        """
        Operator maintenance mode: a JSON file {camera name or ID: reason}. Listed
        cameras are published as status "maintenance" (lanes unknown) until removed.
        Re-read whenever the file changes; a missing file means no maintenance.
        """
        if not self.maintenance_file:
            return
        try:
            mtime = os.path.getmtime(self.maintenance_file)
        except FileNotFoundError:
            if self._maintenance:
                logger.info("Maintenance file removed: all cameras back in service")
            self._maintenance, self._maintenance_mtime = {}, None
            self._apply_maintenance()
            return
        except OSError as exc:
            self._maintenance = {c.name: "maintenance_config_error" for c in self.cams}
            self._maintenance_mtime = None
            self._apply_maintenance()
            logger.error(f"Cannot inspect maintenance file {self.maintenance_file}; measurements disabled: {exc}")
            return
        if mtime == self._maintenance_mtime:
            return
        try:
            with open(self.maintenance_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            known = {c.name for c in self.cams} | {c.camera_id for c in self.cams}
            if set(data) - known or any(not isinstance(v, str) or not v.strip() for v in data.values()):
                raise ValueError("maintenance entries require a known camera name/ID and a nonempty reason")
            self._maintenance = data
            self._maintenance_mtime = mtime
            logger.warning(f"Maintenance mode: {self._maintenance or 'none'}")
        except (OSError, ValueError) as exc:
            self._maintenance = {c.name: "maintenance_config_error" for c in self.cams}
            self._maintenance_mtime = mtime
            logger.error(f"Invalid maintenance file {self.maintenance_file}; measurements disabled: {exc}")
        self._apply_maintenance()

    def _apply_maintenance(self) -> None:
        for cam in self.cams:
            active = cam.name in self._maintenance or cam.camera_id in self._maintenance
            if active != cam.maintenance_active:
                self._reset_camera(cam)
                cam.maintenance_active = active

    def build_payload(self, now_wall: Optional[float] = None) -> Dict[str, Any]:
        now_wall = now_wall if now_wall is not None else time.time()
        self._load_maintenance()
        cam_statuses = [self.camera_status(c, now_wall) for c in self.cams]
        lanes: List[Dict[str, Any]] = []
        valid_obs = []
        for cam, st in zip(self.cams, cam_statuses):
            valid = st["status"] in VALID_STATUSES
            if valid:
                valid_obs.append(cam.last_obs_wall)
            for lane in cam.metrics.snapshot():
                lane["observedAt"] = st["observedAt"]
                if valid:
                    lane["valid"] = True
                    lane["invalidReason"] = None
                    lanes.append(lane)
                else:
                    lanes.append(invalidate_lane(lane, st["reason"] or st["status"]))

        meta = {
            "fps": round(self._current_fps, 1),
            "outputMode": self.output_mode,
            "mode": "replay" if self.replay_mode else "live",
            "skip_frames": self.skip_frames,
            "active_cameras": self.camera_names,
            "model": {"path": os.path.basename(self.model_path), "sha256": self.model_sha256},
            "anchor": {c.camera_id: c.anchor for c in self.cams},
            "queueUnits": {c.camera_id: c.motion.settings.units for c in self.cams},
            "uptimeSec": int(now_wall - self.started_wall),
        }
        return self.payload_builder.build(
            frame_idx=self.batch_idx,
            lanes_snapshot=lanes,
            meta=meta,
            traffic_flow=self.gate_manager.get_mqtt_telemetry(
                now_wall, camera_validity={c.idx: st["status"] in VALID_STATUSES for c, st in zip(self.cams, cam_statuses)}),
            cameras=cam_statuses,
            observed_at=min(valid_obs) if valid_obs else now_wall,
            published_at=now_wall,
        )

    def publish_once(self) -> Dict[str, Any]:
        payload = self.build_payload()
        self.last_payload = payload
        wire_payload = payload
        if self.controller_adapter:
            wire_payload = self.controller_adapter.build(payload, time.time())
            previous = (self.controller_delivery["state"], self.controller_delivery["blockers"])
            if wire_payload is None:
                self.controller_delivery["state"] = "suppressed"
                self.controller_delivery["blockers"] = list(self.controller_adapter.blockers)
                self.controller_delivery["suppressed"] += 1
                delivered = False
            else:
                delivered = bool(self.publisher.publish(wire_payload))
                self.controller_delivery["state"] = "publishing" if delivered else "disconnected"
                self.controller_delivery["blockers"] = [] if delivered else ["broker_delivery_failed"]
                if delivered:
                    self.controller_delivery["accepted"] += 1
                    self.controller_delivery["lastAcceptedAt"] = payload["publishedAt"]
            if previous != (self.controller_delivery["state"], self.controller_delivery["blockers"]):
                logger.warning(f"Controller-main delivery: {self.controller_delivery['state']}; "
                               f"{', '.join(self.controller_delivery['blockers']) or 'complete stop-line occupancy'}")
        else:
            delivered = bool(self.publisher.publish(payload))
        if delivered or self.controller_adapter:
            # Main carries no gates, so each local snapshot closes its interval.
            # Shadow retries retain intervals until delivery succeeds.
            self.gate_manager.reset_interval()
        if self.recorder:
            record = {"delivered": delivered, "payload": payload}
            if self.controller_adapter:
                record["controllerPayload"] = wire_payload
                record["controllerDelivery"] = self.controller_delivery
            self.recorder.info(json.dumps(record))
        health = self.health_summary(payload)
        if hasattr(self.publisher, "publish_health"):
            try:
                self.publisher.publish_health(health)
            except Exception as exc:
                logger.debug(f"health publish failed: {exc}")
        if self.health_file:
            _write_json_atomic(self.health_file, health)
        return payload

    def health_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cams = payload.get("cameras", [])
        valid = sum(1 for c in cams if c["status"] in VALID_STATUSES)
        stats = self.publisher.get_stats() if hasattr(self.publisher, "get_stats") else {}
        return {
            "status": "online" if valid == len(cams) else ("degraded" if valid else "no_valid_cameras"),
            "sourceId": self.source_id,
            "intersectionId": self.intersection_id,
            "sessionId": payload.get("sessionId"),
            "sequence": payload.get("sequence"),
            "updatedAt": payload.get("publishedAt"),
            "validCameras": valid,
            "totalCameras": len(cams),
            "cameras": [{k: c[k] for k in ("cameraId", "name", "status", "reason", "ageMs")} for c in cams],
            "delivery": {k: v for k, v in stats.items() if k != "lastAckWall"} if isinstance(stats, dict) else {},
            "controllerDelivery": dict(self.controller_delivery) if self.controller_delivery is not None else None,
            "errors": {"total": self.total_errors, "consecutive": self.consecutive_errors},
        }

    def _submit_display(self) -> None:
        now_wall = time.time()
        statuses = [self.camera_status(c, now_wall) for c in self.cams]
        annotations = []
        for cam in self.cams:
            ann = {}
            for o in cam.last_tracked:
                rec = cam.metrics.lane_of(int(o[4]))
                if rec:
                    ann[int(o[4])] = rec
            annotations.append(ann)
        acc = self.gate_manager.get_corridor_accounting()
        stats_str = (f"BATCH {self.batch_idx:06d} | INFER FPS {self._current_fps:.1f} | "
                     f"GATES IN {acc['total_inflow']} STOP {acc['total_stopline_cleared']} | "
                     f"CAMS OK {sum(s['status'] in VALID_STATUSES for s in statuses)}/{len(statuses)}")
        if self.controller_delivery:
            stats_str += f" | CONTROLLER {self.controller_delivery['state'].upper()}"
        gates_render = []
        for c_i in range(self.num_streams):
            gates_render.append([{
                "p1": g.p1, "p2": g.p2, "normal": g.normal, "count": g.count,
                "label": g.label, "type": g.gate_type, "flash": (now_wall - g.last_flash_ts < 0.4),
            } for g in self.gate_manager.gates_by_cam.get(c_i, [])])
        self.display_worker.submit(
            frames=[c.last_frame for c in self.cams],
            tracked_list=[c.last_tracked for c in self.cams],
            cam_names=self.camera_names,
            lane_configs=self.lane_configs,
            header_stats=stats_str,
            gates=gates_render,
            annotations=annotations,
            cam_status=statuses,
        )
        if not self.display_worker.poll_window():
            logger.info("User requested exit from preview window (pressed 'q').")
            self.running = False

    def run(self) -> None:
        """Starts all workers and runs until stopped. Startup is inside the cleanup boundary."""
        try:
            self.start()
            logger.info(f"Pipeline running: {self.num_streams} cameras | frame skipping 1-in-{self.skip_frames + 1}")
            while self.running:
                t0 = time.perf_counter()
                try:
                    self.step()
                    self.consecutive_errors = 0
                except Exception:  # unslop-ignore: logged retry boundary with a bounded fatal-error threshold
                    self.total_errors += 1
                    self.consecutive_errors += 1
                    logger.exception(f"Error in processing loop ({self.consecutive_errors} consecutive)")
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        raise RuntimeError("Too many consecutive processing errors; exiting for supervisor restart")
                    time.sleep(min(1.0, 0.05 * self.consecutive_errors))
                if self.replay_mode and self.target_fps > 0:
                    remaining = 1.0 / self.target_fps - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught. Shutting down pipeline...")
        finally:
            self.stop()


class _ConfigError:
    def __init__(self, context: str, lane_id: Optional[str], reason: str):
        self.context, self.lane_id, self.reason = context, lane_id, reason


def _match_tracks_to_dets(track_boxes: np.ndarray, det_boxes: np.ndarray, min_iou: float = 0.3) -> Dict[int, int]:
    """One-to-one IoU matching of output tracks to detections (for trackers that do not report classes)."""
    from scipy.optimize import linear_sum_assignment
    from algorithm.utils import iou_batch

    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return {}
    ious = iou_batch(track_boxes, det_boxes)
    rows, cols = linear_sum_assignment(-ious)
    return {int(r): int(c) for r, c in zip(rows, cols) if ious[r, c] >= min_iou}


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".health-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:  # unslop-ignore: clean up the temporary file and propagate the original write failure
        try:
            os.remove(tmp)
        except OSError as exc:
            logger.warning(f"Could not remove temporary health file {tmp}: {exc}")
        raise


def _make_recorder(path: str) -> logging.Logger:
    rec = logging.getLogger(f"PayloadRecorder:{path}")
    rec.propagate = False
    rec.setLevel(logging.INFO)
    if not rec.handlers:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        h = logging.handlers.RotatingFileHandler(path, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))
        rec.addHandler(h)
    return rec


# ---------------------------------------------------------------------- CLI
def build_pipeline_args() -> argparse.ArgumentParser:
    env = os.environ.get
    parser = argparse.ArgumentParser(
        description="Smart Traffic Vision - multi-camera measurement runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("cameras and sources")
    src.add_argument("--cameras", nargs="+", default=None,
                     help=f"Camera names ({' '.join(DEFAULT_CONFIGS)} or 'all'). Default: all, or derived from --configs")
    src.add_argument("--configs", nargs="+", default=None, help="Calibration JSON per camera (same order as --cameras)")
    src.add_argument("--videos", nargs="+", default=None,
                     help="Source per camera (file or rtsp://). Prefer --sources-file for credentials")
    src.add_argument("--sources-file", default=env("VISION_SOURCES_FILE"),
                     help="JSON {camera_name: source_url}. Keeps camera credentials out of the process list")
    src.add_argument("--replay-fps", type=float, default=None,
                     help="True recording frame rate for file replay (default: file metadata)")

    mdl = parser.add_argument_group("model and tracking")
    mdl.add_argument("--model", default=DEFAULT_MODEL, help="YOLO .pt or TensorRT .engine path")
    mdl.add_argument("--model-sha256", default=env("VISION_MODEL_SHA256"), help="Refuse to start unless the model matches")
    mdl.add_argument("--device", default=None, help="'cuda:0', 'cpu' (default: cuda:0 if available)")
    mdl.add_argument("--allow-cpu", action="store_true", help="Permit running without a CUDA GPU (development only)")
    mdl.add_argument("--conf", type=float, default=0.10,
                     help="Detector confidence; must be <= --low-thresh so ByteTrack's low-confidence stage gets input")
    mdl.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    mdl.add_argument("--tracker", choices=["byetrack", "sort"], default="byetrack")
    mdl.add_argument("--track-thresh", type=float, default=0.40, help="Min detection confidence to start a track")
    mdl.add_argument("--low-thresh", type=float, default=0.10, help="Min confidence for ByteTrack's second stage")
    mdl.add_argument("--match-thresh", type=float, default=0.70, help="Max IoU distance for association")
    mdl.add_argument("--track-max-age", type=int, default=30, help="Inferred frames a lost track is kept")
    mdl.add_argument("--voting-window", type=int, default=15)
    mdl.add_argument("--pickup-bias", type=float, default=1.15)
    mdl.add_argument("--no-voting", action="store_true")

    run = parser.add_argument_group("timing")
    run.add_argument("--fps", type=float, default=25.0, help="Max processing iterations per second")
    run.add_argument("--skip-frames", type=int, default=1, help="Infer on 1 of every N+1 frames per camera")
    run.add_argument("--max-frame-age", type=float, default=1.0, help="Drop frames older than this (s) before inference")
    run.add_argument("--max-observation-age", type=float, default=1.5, help="Lanes invalid when last observation is older (s)")
    run.add_argument("--pub-interval", type=float, default=1.0, help="Publish interval (s)")

    out = parser.add_argument_group("delivery and identity")
    out.add_argument("--intersection-id", default=env("VISION_INTERSECTION_ID", "INT-001"))
    out.add_argument("--source-id", default=env("VISION_SOURCE_ID"),
                     help="Diagnostic vision source ID (default VISION-<intersection>)")
    out.add_argument("--mqtt-broker", default=None, help="Broker URL (default: $MQTT_URL or mqtt://localhost:1883)")
    out.add_argument("--output-mode", choices=("shadow", "controller"), default=env("VISION_OUTPUT_MODE", "shadow"),
                     help="Shadow publishes full diagnostics; controller sends compatible stop-line counts to system main")
    out.add_argument("--controller-config", default=env("VISION_CONTROLLER_CONFIG"),
                     help="Existing controller-main identity, required lanes and freshness profile (default config/controller_main.json)")
    out.add_argument("--allow-replay-controller", action="store_true",
                     help="Allow recordings on the controller output for isolated tests (never use with field signals)")
    out.add_argument("--mqtt-topic", default=None, help="Counts topic (default traffic/counts/shadow in shadow mode)")
    out.add_argument("--health-topic", default=None, help="Retained health topic (isolated in shadow mode)")
    out.add_argument("--mqtt-tls-ca", default=None)
    out.add_argument("--mqtt-tls-cert", default=None)
    out.add_argument("--mqtt-tls-key", default=None)
    out.add_argument("--mqtt-tls-insecure", action="store_true", help="Skip hostname verification (testing only)")

    ops = parser.add_argument_group("operations")
    ops.add_argument("--display", action="store_true", help="Show the live overlay window")
    ops.add_argument("--nvenc", default=None, help="Record the overlay grid with NVENC (e.g. out.mp4)")
    ops.add_argument("--log-file", default=env("VISION_LOG_FILE"), help="Rotating log file")
    ops.add_argument("--record-payloads", default=env("VISION_RECORD_PAYLOADS"),
                     help="Rotating JSONL of every payload (for incident review)")
    ops.add_argument("--health-file", default=env("VISION_HEALTH_FILE"),
                     help="JSON health file rewritten each publish (for watchdogs)")
    ops.add_argument("--maintenance-file", default=env("VISION_MAINTENANCE_FILE"),
                     help='JSON {"camera": "reason"}; listed cameras are reported as under maintenance (lanes unknown)')
    ops.add_argument("--check-config", action="store_true",
                     help="Validate configuration, sources and model, print effective settings, and exit")
    return parser


def resolve_run_plan(args: argparse.Namespace) -> Tuple[List[str], List[str], List[Any]]:
    """Strictly resolves camera names, configs and sources. Never falls back silently."""
    def repo_path(p: str) -> str:
        return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(REPO_ROOT, p)

    if args.cameras and "all" in args.cameras:
        if len(args.cameras) != 1:
            raise ValueError("'all' cannot be combined with other camera names")
        names = list(DEFAULT_CONFIGS)
    elif args.cameras:
        names = list(args.cameras)
    elif args.configs:
        names = [os.path.splitext(os.path.basename(p))[0].replace("config_", "") for p in args.configs]
    else:
        names = list(DEFAULT_CONFIGS)

    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate camera names: {names}")

    if args.configs:
        if len(args.configs) != len(names):
            raise ValueError(f"{len(args.configs)} configs given for {len(names)} cameras")
        configs = [repo_path(p) for p in args.configs]
    else:
        unknown = [n for n in names if n not in DEFAULT_CONFIGS]
        if unknown:
            raise ValueError(f"Unknown camera name(s) {unknown}; known: {list(DEFAULT_CONFIGS)} (or pass --configs)")
        configs = [repo_path(DEFAULT_CONFIGS[n]) for n in names]
    missing = [c for c in configs if not os.path.exists(c)]
    if missing:
        raise FileNotFoundError(f"Config file(s) not found: {missing}")

    sources_map: Dict[str, Any] = {}
    if args.sources_file:
        with open(args.sources_file, "r", encoding="utf-8") as f:
            sources_map = json.load(f)
        if not isinstance(sources_map, dict):
            raise ValueError("--sources-file must contain a JSON object {camera_name: source}")
    if args.videos and sources_map:
        raise ValueError("Use either --videos or --sources-file, not both")
    if args.videos and len(args.videos) != len(names):
        raise ValueError(f"{len(args.videos)} sources given for {len(names)} cameras")

    sources: List[Any] = []
    for i, (name, cfg_path) in enumerate(zip(names, configs)):
        if args.videos:
            src = args.videos[i]
        elif sources_map:
            if name not in sources_map:
                raise ValueError(f"--sources-file has no entry for camera '{name}'")
            src = sources_map[name]
        else:
            src = initial_config(cfg_path).get("video", {}).get("path")
            if not src:
                raise ValueError(f"No source for camera '{name}': give --videos/--sources-file or set video.path")
            if not is_live_source(src):
                src = repo_path(src)
        if isinstance(src, str) and "://" not in src and not os.path.exists(src):
            raise FileNotFoundError(f"Source for camera '{name}' not found: {src}")
        sources.append(src)
    return names, configs, sources


def configure_file_logging(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h = logging.handlers.RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logger.addHandler(h)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_pipeline_args()
    args = parser.parse_args(argv)
    if args.log_file:
        configure_file_logging(args.log_file)

    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    if device.startswith("cpu") and not args.allow_cpu:
        logger.error("No CUDA GPU available. Refusing to run on CPU in production (use --allow-cpu for development).")
        return 2
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error(f"Requested device {device} but CUDA is not available.")
        return 2
    model_path = args.model if os.path.isabs(args.model) or os.path.exists(args.model) else os.path.join(REPO_ROOT, args.model)

    try:
        names, configs, sources = resolve_run_plan(args)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Configuration error: {exc}")
        return 2

    tls = {"ca": args.mqtt_tls_ca, "cert": args.mqtt_tls_cert, "key": args.mqtt_tls_key,
           "insecure": args.mqtt_tls_insecure}
    try:
        pipeline = BatchedCameraPipeline(
            camera_names=names,
            config_paths=configs,
            video_sources=sources,
            model_path=model_path,
            device=device,
            conf=args.conf,
            target_fps=args.fps,
            skip_frames=args.skip_frames,
            display=args.display,
            nvenc_out=args.nvenc,
            pub_interval=args.pub_interval,
            mqtt_broker=args.mqtt_broker,
            mqtt_topic=args.mqtt_topic,
            intersection_id=args.intersection_id,
            voting_window=args.voting_window,
            pickup_bias=args.pickup_bias,
            enable_voting=not args.no_voting,
            tracker_type=args.tracker,
            imgsz=args.imgsz,
            source_id=args.source_id,
            track_thresh=args.track_thresh,
            low_thresh=args.low_thresh,
            match_thresh=args.match_thresh,
            track_max_age=args.track_max_age,
            max_frame_age_s=args.max_frame_age,
            max_observation_age_s=args.max_observation_age,
            replay_fps=args.replay_fps,
            health_topic=args.health_topic,
            mqtt_tls=tls,
            model_sha256=args.model_sha256,
            record_path=args.record_payloads,
            health_file=args.health_file,
            maintenance_file=args.maintenance_file,
            output_mode=args.output_mode,
            controller_config=args.controller_config,
            allow_replay_controller=args.allow_replay_controller,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error(f"Startup error: {exc}")
        return 2

    if args.check_config:
        print(json.dumps(pipeline.effective_settings(), indent=2))
        return 0

    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received. Stopping pipeline...")
        pipeline.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        pipeline.run()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
