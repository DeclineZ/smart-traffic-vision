"""Shared fakes for runner tests: no GPU, camera, broker or model weights required."""

import json
import os
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np

from trt_pipeline.stream import FramePacket

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def write_config(directory: str, name: str, lanes: Dict[str, dict], gates: Optional[list] = None,
                 camera_id: Optional[str] = None, extra: Optional[dict] = None) -> str:
    cfg = {
        "camera_info": {"camera_id": camera_id or f"CAM-{name.upper()}"},
        "lane_metrics": {"lanes": lanes},
        "gates": gates or [],
    }
    if extra:
        cfg.update(extra)
    path = os.path.join(directory, f"config_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


class FakeSource:
    """Hands out queued packets; poll() returns None once the queue is empty (camera silent)."""

    def __init__(self, name="cam", shape=(100, 100)):
        self.name = name
        self.shape = shape
        self.queue: List[FramePacket] = []
        self.seq = 0
        self.epoch = 1
        self.state = "ok"
        self.started = False
        self.stopped = False

    def push(self, obs_t: float, captured_wall: Optional[float] = None, epoch: Optional[int] = None,
             frame: Optional[np.ndarray] = None):
        self.seq += 1
        if epoch is not None:
            self.epoch = epoch
        if frame is None:
            frame = np.random.randint(0, 255, (*self.shape, 3), dtype=np.uint8)
        self.queue.append(FramePacket(frame=frame, seq=self.seq, epoch=self.epoch,
                                      captured_wall=captured_wall if captured_wall is not None else time.time(),
                                      obs_t=obs_t))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def poll(self):
        return self.queue.pop(0) if self.queue else None

    def status(self):
        return {"state": self.state, "framesDropped": 0, "reconnects": 0}


class FakeBoxes:
    def __init__(self, dets: np.ndarray):
        self._d = np.asarray(dets, dtype=float).reshape(-1, 6)

    def __len__(self):
        return len(self._d)

    class _T:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    @property
    def xyxy(self):
        return self._T(self._d[:, :4])

    @property
    def conf(self):
        return self._T(self._d[:, 4])

    @property
    def cls(self):
        return self._T(self._d[:, 5])


class FakeResult:
    def __init__(self, dets):
        self.boxes = FakeBoxes(dets)


class FakeModel:
    """Returns scripted detections per call: script is a list of per-frame det arrays, consumed in order."""

    names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "three_wheeler"}

    def __init__(self, dets_fn=None):
        self.calls = []
        self.dets_fn = dets_fn or (lambda frame_index, call: np.empty((0, 6)))

    def __call__(self, frames, **kwargs):
        call = len(self.calls)
        self.calls.append(len(frames))
        return [FakeResult(self.dets_fn(i, call)) for i in range(len(frames))]


class FakePublisher:
    def __init__(self, succeed=True):
        self.succeed = succeed
        self.payloads = []
        self.health = []

    def start(self):
        pass

    def stop(self, *a):
        pass

    def publish(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.succeed

    def publish_health(self, h):
        self.health.append(h)
        return True

    def get_stats(self):
        return {"sent": len(self.payloads)}


def make_pipeline(test_dir, cameras: Dict[str, dict], model=None, publisher=None, **kwargs):
    """cameras: name -> {"lanes": {...}, "gates": [...], "extra": {...}}"""
    from run_multi_camera import BatchedCameraPipeline

    names = list(cameras)
    configs = [write_config(test_dir, n, c["lanes"], c.get("gates"), extra=c.get("extra")) for n, c in cameras.items()]
    sources = [FakeSource(n) for n in names]
    kwargs.setdefault("skip_frames", 0)
    kwargs.setdefault("pub_interval", 3600)
    p = BatchedCameraPipeline(
        camera_names=names,
        config_paths=configs,
        video_sources=[f"fake://{n}" for n in names],
        model_path="unused.pt",
        device="cpu",
        model=model or FakeModel(),
        publisher=publisher or FakePublisher(),
        sources=sources,
        **kwargs,
    )
    p.running = True
    p.batch_wait_s = 0.0
    return p, sources


def tmpdir():
    return tempfile.mkdtemp(prefix="stv_test_")
