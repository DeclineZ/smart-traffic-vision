"""
Shared helpers for the manual labeling workflow.

Stdlib only: the labeling server imports this module, and an annotator should be
able to run the server with a bare Python install (no torch / OpenCV).

Pack layout (data/<pack>/):
    frames.json                     frame list, split, assignee, selection metadata
    images/<frame_id>.jpg           full-resolution frames extracted from videos/
    prelabels/<frame_id>.json       machine boxes the labeler starts from
    work/<annotator>/<frame_id>.json        that annotator's labels (one file per frame)
    work/<annotator>/reviews/<frame_id>.json  review notes left on a partner's frame
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ["car", "motorcycle", "bus", "truck", "three_wheeler"]
CAR, MOTORCYCLE, BUS, TRUCK, THREE_WHEELER = range(5)

# Frames from these datasets must never land in the new train/val sets (or near them).
EXISTING_DATASETS = ["data/multiclass_dataset", "data/eval_snapshot_v1"]

FRAME_ID_RE = re.compile(r"^(cam\d+_[a-z]+(?:_night)?)_f(\d+)")

VALID_STATUSES = ("todo", "in_progress", "done", "skip")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_frame_id(camera: str, frame_idx: int) -> str:
    return f"{camera}_f{frame_idx:07d}"


def parse_frame_id(name: str) -> Optional[Tuple[str, int]]:
    """'cam43_south_night_f0712345.jpg' -> ('cam43_south_night', 712345)."""
    m = FRAME_ID_RE.match(Path(name).name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write via temp file + rename so a crash or Drive sync never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def xyxy_to_yolo(box: Sequence[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = (max(0.0, min(float(v), lim)) for v, lim in zip(box, (img_w, img_h, img_w, img_h)))
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h)


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    return [(xc - w / 2) * img_w, (yc - h / 2) * img_h, (xc + w / 2) * img_w, (yc + h / 2) * img_h]


def read_yolo_labels(path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        boxes.append({"cls": cls, "xyxy": yolo_to_xyxy(*map(float, parts[1:5]), img_w, img_h)})
    return boxes


# --------------------------------------------------------------------------- pack access


class Pack:
    """Read/write access to a manual labeling pack directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.meta = read_json(self.root / "frames.json")
        if self.meta is None:
            raise FileNotFoundError(f"{self.root / 'frames.json'} not found - is this a labeling pack?")
        self.frames: List[Dict[str, Any]] = self.meta["frames"]
        self.by_id = {f["id"]: f for f in self.frames}
        self.annotators: List[str] = self.meta["annotators"]

    # paths
    def image_path(self, frame_id: str) -> Path:
        return self.root / "images" / f"{frame_id}.jpg"

    def prelabel_path(self, frame_id: str) -> Path:
        return self.root / "prelabels" / f"{frame_id}.json"

    def work_dir(self, annotator: str) -> Path:
        return self.root / "work" / annotator

    def label_path(self, annotator: str, frame_id: str) -> Path:
        return self.work_dir(annotator) / f"{frame_id}.json"

    def review_path(self, reviewer: str, frame_id: str) -> Path:
        return self.work_dir(reviewer) / "reviews" / f"{frame_id}.json"

    # assignment
    def is_assigned(self, frame_id: str, annotator: str) -> bool:
        f = self.by_id.get(frame_id)
        return f is not None and (f["assignee"] == "*" or f["assignee"] == annotator)

    def frames_for(self, annotator: str) -> List[Dict[str, Any]]:
        return [f for f in self.frames if f["assignee"] in ("*", annotator)]

    def owners_of(self, frame_id: str) -> List[str]:
        f = self.by_id[frame_id]
        return list(self.annotators) if f["assignee"] == "*" else [f["assignee"]]

    # records
    def load_label(self, annotator: str, frame_id: str) -> Optional[Dict[str, Any]]:
        return read_json(self.label_path(annotator, frame_id))

    def load_prelabel(self, frame_id: str) -> Dict[str, Any]:
        return read_json(self.prelabel_path(frame_id), default={"boxes": []})

    def load_reviews_on(self, frame_id: str) -> List[Dict[str, Any]]:
        """Review notes any annotator left on this frame (present only if their work folder was copied in)."""
        out = []
        work = self.root / "work"
        if not work.exists():
            return out
        for d in sorted(p for p in work.iterdir() if p.is_dir()):
            rec = read_json(d / "reviews" / f"{frame_id}.json")
            if rec:
                out.append(rec)
        return out

    def status_of(self, annotator: str, frame_id: str) -> str:
        rec = self.load_label(annotator, frame_id)
        return rec.get("status", "todo") if rec else "todo"


def validate_boxes(boxes: Iterable[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Normalise incoming boxes from the UI; drop degenerate ones.

    An unresolved pre-label marker ("check" + "note") is kept so an in-progress frame
    reopens with its "?" boxes still highlighted. Merge ignores it.
    """
    clean = []
    for b in boxes:
        cls = int(b["cls"])
        if not 0 <= cls < len(CLASS_NAMES):
            raise ValueError(f"bad class id {cls}")
        x1, y1, x2, y2 = (float(v) for v in b["xyxy"])
        x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
        y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        box = {"cls": cls, "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}
        if b.get("check"):
            box["check"] = True
            box["note"] = str(b.get("note", ""))[:200]
        clean.append(box)
    return clean
