"""
Virtual counting gates and interval flow measurement.

A gate is a directed line segment. A track is counted when the road-contact
point (bottom centre of its box) moves across the line in the gate's forward
direction. Each track is counted at most once per gate.

Measurements are reported per publication interval: event counts, the interval
length, and how many seconds of that interval the gate's camera was actually
observed. A direction with no gate is reported as *not instrumented* (null), and
a gate whose camera was not observed for most of the interval is reported as
invalid, so missing coverage is never confused with zero vehicles.

All durations use each camera's observation clock (``t`` seconds), supplied by
the runner. Wall-clock time is only used for labelling interval boundaries.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

VALID_DIRECTIONS = ("N", "S", "E", "W")
GATE_TYPES = ("stopline", "ingress", "egress")


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Tests if points A, B, C are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Returns True if line segment p1-p2 intersects line segment p3-p4."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


class VirtualGate:
    """Directed 2D tripwire."""

    def __init__(
        self,
        gate_id: str,
        cam_idx: int,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        gate_type: str = "stopline",  # 'stopline' (departures), 'ingress' (arrivals), 'egress'
        direction_vec: Optional[Tuple[float, float]] = None,
        label: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        target_dir: Optional[str] = None,
        min_motion_px: float = 2.0,
        crossed_ttl_s: float = 120.0,
    ):
        self.gate_id = gate_id
        self.cam_idx = cam_idx
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.gate_type = gate_type.lower()
        if self.gate_type not in GATE_TYPES:
            raise ValueError(f"Gate '{gate_id}': type must be one of {GATE_TYPES}, got '{gate_type}'")
        self.label = label or gate_id
        self.color = color
        self.min_motion_px = float(min_motion_px)
        self.crossed_ttl_s = float(crossed_ttl_s)

        if target_dir is None:
            raise ValueError(f"Gate '{gate_id}': target_dir is required (one of {VALID_DIRECTIONS})")
        self.target_dir = str(target_dir).upper()
        if self.target_dir not in VALID_DIRECTIONS:
            raise ValueError(f"Gate '{gate_id}': target_dir must be one of {VALID_DIRECTIONS}, got '{target_dir}'")

        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        line_len = float(np.hypot(dx, dy))
        if line_len < 1.0:
            raise ValueError(f"Gate '{gate_id}': endpoints must be at least 1 px apart")

        if direction_vec is not None:
            nx, ny = float(direction_vec[0]), float(direction_vec[1])
            n_len = max(1e-6, float(np.hypot(nx, ny)))
            self.normal = (nx / n_len, ny / n_len)
        else:
            # Default normal: 90 degrees clockwise perpendicular (-dy, dx)
            self.normal = (-dy / line_len, dx / line_len)

        self.midpoint = ((self.p1[0] + self.p2[0]) * 0.5, (self.p1[1] + self.p2[1]) * 0.5)

        self.count = 0  # cumulative for this session
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.interval_count = 0
        self.interval_class_counts: Dict[int, int] = defaultdict(int)
        self.crossed_track_ids: Dict[int, float] = {}  # track_id -> observation time of crossing
        self.recent_crossings: deque[Dict[str, Any]] = deque(maxlen=200)
        self.last_flash_ts: float = 0.0

    def check_crossing(
        self,
        track_id: int,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        class_id: int = 0,
        now: Optional[float] = None,
    ) -> bool:
        """True if the step p_prev -> p_curr is a new forward crossing of this gate."""
        if track_id in self.crossed_track_ids:
            return False

        move_x = p_curr[0] - p_prev[0]
        move_y = p_curr[1] - p_prev[1]
        if np.hypot(move_x, move_y) < self.min_motion_px:
            return False  # sub-pixel jitter on the line is not a crossing

        if not segments_intersect(p_prev, p_curr, self.p1, self.p2):
            return False

        dot = move_x * self.normal[0] + move_y * self.normal[1]
        if dot <= 0.0:
            return False

        ts = now if now is not None else time.monotonic()
        self.crossed_track_ids[track_id] = ts
        self.count += 1
        self.class_counts[class_id] += 1
        self.interval_count += 1
        self.interval_class_counts[class_id] += 1
        self.last_flash_ts = time.time()

        self.recent_crossings.append({
            "gate_id": self.gate_id,
            "track_id": track_id,
            "class_id": class_id,
            "t": ts,
            "wall": self.last_flash_ts,
        })
        return True

    def prune_stale_tracks(self, active_track_ids: Set[int], now: Optional[float] = None) -> None:
        """Forgets crossed IDs that are no longer active and crossed more than ``crossed_ttl_s`` ago."""
        if now is None:
            return
        self.crossed_track_ids = {
            tid: t for tid, t in self.crossed_track_ids.items()
            if tid in active_track_ids or now - t <= self.crossed_ttl_s
        }

    def reset_tracks(self) -> None:
        """Called after a camera discontinuity; track IDs from the previous epoch are meaningless."""
        self.crossed_track_ids.clear()

    def reset_interval(self) -> None:
        self.interval_count = 0
        self.interval_class_counts = defaultdict(int)


class GateFlowManager:
    """Gates across all cameras plus per-camera observation coverage."""

    def __init__(
        self,
        camera_names: Optional[List[str]] = None,
        camera_ids: Optional[List[str]] = None,
        max_gap_s: float = 1.0,
        min_coverage: float = 0.8,
    ):
        self.camera_names = camera_names or []
        self.camera_ids = camera_ids or list(self.camera_names)
        self.max_gap_s = float(max_gap_s)
        self.min_coverage = float(min_coverage)
        self.gates: Dict[str, VirtualGate] = {}
        self.gates_by_cam: Dict[int, List[VirtualGate]] = defaultdict(list)

        # (cam_idx, track_id) -> (point, observation time)
        self.track_prev_pts: Dict[Tuple[int, int], Tuple[Tuple[float, float], float]] = {}
        self._cam_last_t: Dict[int, float] = {}
        self._cam_observed_s: Dict[int, float] = defaultdict(float)
        self._invalid_interval_cameras: Set[int] = set()

        self.interval_start_wall = time.time()
        self.interval_start_mono = time.monotonic()

    def add_gate(self, gate: VirtualGate) -> None:
        if gate.gate_id in self.gates:
            raise ValueError(f"Duplicate gate_id '{gate.gate_id}': gate IDs must be unique across all cameras")
        self.gates[gate.gate_id] = gate
        self.gates_by_cam[gate.cam_idx].append(gate)

    def reset_camera(self, cam_idx: int) -> None:
        """Drops temporal state for one camera after a discontinuity (reconnect, file loop)."""
        for key in [k for k in self.track_prev_pts if k[0] == cam_idx]:
            del self.track_prev_pts[key]
        self._cam_last_t.pop(cam_idx, None)
        self._invalid_interval_cameras.add(cam_idx)
        for gate in self.gates_by_cam.get(cam_idx, []):
            gate.reset_tracks()

    def update_tracks(
        self,
        cam_idx: int,
        tracked_objs: np.ndarray,
        now: Optional[float] = None,
    ) -> List[str]:
        """
        Evaluates one observed frame of a camera. Call it for every inferred frame,
        including frames with no vehicles, so observation coverage is measured.

        Args:
            cam_idx: Camera index
            tracked_objs: (M, 5+) [x1, y1, x2, y2, track_id, class_id, ...]
            now: Observation time in seconds on this camera's clock
        """
        t = float(now) if now is not None else time.monotonic()

        # Coverage is measured on the local monotonic clock, the same clock as the
        # interval window. Only continuous observation counts; a long gap does not.
        mono = time.monotonic()
        last = self._cam_last_t.get(cam_idx)
        if last is not None and 0 < mono - last <= self.max_gap_s:
            self._cam_observed_s[cam_idx] += max(0.0, mono - max(last, self.interval_start_mono))
        self._cam_last_t[cam_idx] = mono

        crossed: List[str] = []
        cam_gates = self.gates_by_cam.get(cam_idx, [])
        if not cam_gates:
            return crossed

        objs = tracked_objs if tracked_objs is not None else np.empty((0, 6))
        active: Set[int] = set()
        for obj in objs:
            ox1, oy1, ox2, oy2, track_id = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), int(obj[4])
            cls_id = int(obj[5]) if len(obj) >= 6 else 0
            active.add(track_id)
            curr_pt = ((ox1 + ox2) * 0.5, oy2)  # bottom-centre road contact point
            key = (cam_idx, track_id)

            prev = self.track_prev_pts.get(key)
            if prev is not None and t - prev[1] <= self.max_gap_s:
                for gate in cam_gates:
                    if gate.check_crossing(track_id, prev[0], curr_pt, cls_id, now=t):
                        crossed.append(gate.gate_id)
            self.track_prev_pts[key] = (curr_pt, t)

        # Missing tracks keep their last point only within the bounded continuity gap.
        for key in [k for k, (_, kt) in self.track_prev_pts.items()
                    if k[0] == cam_idx and k[1] not in active and t - kt > self.max_gap_s]:
            del self.track_prev_pts[key]

        for gate in cam_gates:
            gate.prune_stale_tracks(active, now=t)

        return crossed

    # ------------------------------------------------------------------ reporting
    def interval_report(self, now_wall: Optional[float] = None,
                        camera_validity: Optional[Dict[int, bool]] = None) -> Dict[str, Any]:
        """Interval measurements since the last ``reset_interval()``."""
        from .payload import iso_utc

        end_wall = now_wall if now_wall is not None else time.time()
        window_s = max(1e-3, time.monotonic() - self.interval_start_mono)

        gates_out = []
        for g in self.gates.values():
            observed = min(window_s, self._cam_observed_s.get(g.cam_idx, 0.0))
            coverage = observed / window_s
            valid = (coverage >= self.min_coverage and g.cam_idx not in self._invalid_interval_cameras
                     and (camera_validity is None or camera_validity.get(g.cam_idx, False)))
            gates_out.append({
                "gateId": g.gate_id,
                "type": g.gate_type,
                "direction": g.target_dir,
                "cameraId": self.camera_ids[g.cam_idx] if g.cam_idx < len(self.camera_ids) else str(g.cam_idx),
                "count": g.interval_count if valid else None,
                "observedSec": round(observed, 3),
                "coverage": round(coverage, 3),
                "valid": valid,
            })

        by_dir: Dict[str, Dict[str, Any]] = {}
        for d in VALID_DIRECTIONS:
            entry: Dict[str, Any] = {}
            for kind, gtype in (("arrivals", "ingress"), ("departures", "stopline")):
                gs = [g for g in gates_out if g["direction"] == d and g["type"] == gtype]
                if not gs:
                    entry[kind] = None  # not instrumented: unknown, not zero
                    continue
                valid = all(g["valid"] for g in gs)
                observed = min(g["observedSec"] for g in gs)
                count = sum(g["count"] for g in gs) if valid else None
                entry[kind] = {
                    "count": count if valid else None,
                    "observedSec": observed,
                    "ratePerSec": round(count / observed, 4) if valid and observed > 0 else None,
                    "valid": valid,
                }
            by_dir[d] = entry

        return {
            "windowStart": iso_utc(self.interval_start_wall),
            "windowEnd": iso_utc(end_wall),
            "windowSec": round(window_s, 3),
            "gates": gates_out,
            "by_direction": by_dir,
        }

    def get_corridor_accounting(self) -> Dict[str, Any]:
        """Session-cumulative totals for the operator HUD (not a controller input)."""
        inflow = sum(g.count for g in self.gates.values() if g.gate_type == "ingress")
        stopline = sum(g.count for g in self.gates.values() if g.gate_type == "stopline")
        return {
            "total_inflow": inflow,
            "total_stopline_cleared": stopline,
            "gate_counts": {gid: g.count for gid, g in self.gates.items()},
        }

    def reset_interval(self) -> None:
        """Starts a new measurement interval (call only after a successful publish)."""
        self.interval_start_wall = time.time()
        self.interval_start_mono = time.monotonic()
        self._cam_observed_s = defaultdict(float)
        self._invalid_interval_cameras.clear()
        for g in self.gates.values():
            g.reset_interval()

    def get_mqtt_telemetry(self, now_wall: Optional[float] = None,
                           camera_validity: Optional[Dict[int, bool]] = None) -> Dict[str, Any]:
        """
        Flow block for the payload. ``by_direction`` keeps the v1 field names for
        instrumented directions only (interval counts, not cumulative); new readers
        use ``interval``.
        """
        report = self.interval_report(now_wall, camera_validity)
        legacy: Dict[str, Dict[str, Any]] = {}
        for d, entry in report["by_direction"].items():
            arr, dep = entry["arrivals"], entry["departures"]
            if arr is None and dep is None:
                continue
            legacy[d] = {
                "inflow": arr["count"] if arr else None,
                "cleared": dep["count"] if dep else None,
                "live_discharge_rate": dep["ratePerSec"] if dep else None,
            }
        return {"by_direction": legacy, "interval": report}
