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
from shapely.geometry import Point, Polygon

from .corridor import CorridorQueueEstimator, is_motorcycle

VALID_DIRECTIONS = ("N", "S", "E", "W")
GATE_TYPES = ("stopline", "ingress", "egress", "handover")


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


def get_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Returns (x, y) intersection of segment p1-p2 and p3-p4 if they intersect, else None."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-9:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
        return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
    return None


class VirtualGate:
    """Directed 2D tripwire with lane classification and anti-jitter protections."""

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
        lane_id: Optional[str] = None,
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
        self.lane_id = str(lane_id) if lane_id else None
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
        self.lane_counts: Dict[str, int] = defaultdict(int)
        self.interval_count = 0
        self.interval_class_counts: Dict[int, int] = defaultdict(int)
        self.interval_lane_counts: Dict[str, int] = defaultdict(int)

        self.crossed_track_ids: Dict[int, float] = {}  # track_id -> observation time of crossing
        self.recent_crossings: deque[Dict[str, Any]] = deque(maxlen=200)
        # Spatial-temporal suppression for ID-switches on the gate line:
        # deque of (crossing_point, timestamp, class_id, lane_id)
        self.recent_crossing_points: deque[Tuple[Tuple[float, float], float, int, Optional[str]]] = deque(maxlen=50)
        self.last_flash_ts: float = 0.0

    def check_crossing(
        self,
        track_id: int,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        class_id: int = 0,
        now: Optional[float] = None,
        lane_id: Optional[str] = None,
    ) -> bool:
        """
        True if the step p_prev -> p_curr is a new forward crossing of this gate.
        Assigns the crossing to lane_id or self.lane_id.
        """
        if track_id in self.crossed_track_ids:
            return False

        move_x = p_curr[0] - p_prev[0]
        move_y = p_curr[1] - p_prev[1]
        dist = np.hypot(move_x, move_y)
        if dist < self.min_motion_px:
            return False  # sub-pixel jitter on the line is not a crossing

        if not segments_intersect(p_prev, p_curr, self.p1, self.p2):
            return False

        dot = move_x * self.normal[0] + move_y * self.normal[1]
        if dot <= 0.0:
            return False

        ts = now if now is not None else time.monotonic()
        cross_pt = get_segment_intersection(p_prev, p_curr, self.p1, self.p2) or p_curr

        # EDGE CASE 4 & 5: Spatial-temporal cooldown against ID switches right on the line
        # If an ID switched while crossing, a new ID crossing within 0.5s at almost the exact point
        # in a subsequent frame is the same vehicle; remember it as crossed without double-incrementing.
        for prev_pt, prev_ts, prev_cls, _ in self.recent_crossing_points:
            if 0.0 < ts - prev_ts <= 0.5 and prev_cls == class_id:
                if np.hypot(cross_pt[0] - prev_pt[0], cross_pt[1] - prev_pt[1]) < 15.0:
                    self.crossed_track_ids[track_id] = ts
                    return False

        assigned_lane = lane_id or self.lane_id or "unknown"
        self.crossed_track_ids[track_id] = ts
        self.count += 1
        self.class_counts[class_id] += 1
        self.lane_counts[assigned_lane] += 1
        self.interval_count += 1
        self.interval_class_counts[class_id] += 1
        self.interval_lane_counts[assigned_lane] += 1
        self.last_flash_ts = time.time()

        self.recent_crossing_points.append((cross_pt, ts, class_id, assigned_lane))
        self.recent_crossings.append({
            "gate_id": self.gate_id,
            "track_id": track_id,
            "class_id": class_id,
            "lane_id": assigned_lane,
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
        # Also clean up crossing points older than 5 seconds
        while self.recent_crossing_points and now - self.recent_crossing_points[0][1] > 5.0:
            self.recent_crossing_points.popleft()

    def reset_tracks(self) -> None:
        """Called after a camera discontinuity; track IDs from the previous epoch are meaningless."""
        self.crossed_track_ids.clear()
        self.recent_crossing_points.clear()

    def reset_interval(self) -> None:
        self.interval_count = 0
        self.interval_class_counts = defaultdict(int)
        self.interval_lane_counts = defaultdict(int)


class GateFlowManager:
    """Gates across all cameras plus per-camera observation coverage and corridor estimation."""

    def __init__(
        self,
        camera_names: Optional[List[str]] = None,
        camera_ids: Optional[List[str]] = None,
        max_gap_s: float = 1.0,
        min_coverage: float = 0.8,
        lane_mappings: Optional[Dict[str, str]] = None,
    ):
        self.camera_names = camera_names or []
        self.camera_ids = camera_ids or list(self.camera_names)
        self.max_gap_s = float(max_gap_s)
        self.min_coverage = float(min_coverage)
        self.gates: Dict[str, VirtualGate] = {}
        self.gates_by_cam: Dict[int, List[VirtualGate]] = defaultdict(list)
        self.camera_lanes: Dict[int, Dict[str, Dict[str, Any]]] = {}

        # Corridor Queue Estimator
        self.corridor_estimator = CorridorQueueEstimator(lane_mappings=lane_mappings)

        # (cam_idx, track_id) -> (point, observation time)
        self.track_prev_pts: Dict[Tuple[int, int], Tuple[Tuple[float, float], float]] = {}
        # (cam_idx, track_id) -> (last_assigned_lane, observation time)
        self.track_lane_history: Dict[Tuple[int, int], Tuple[str, float]] = {}

        self._cam_last_t: Dict[int, float] = {}
        self._cam_observed_s: Dict[int, float] = defaultdict(float)
        self._invalid_interval_cameras: Set[int] = set()

        self.interval_start_wall = time.time()
        self.interval_start_mono = time.monotonic()

    def set_camera_lanes(self, cam_idx: int, lanes: Dict[str, Dict[str, Any]]) -> None:
        """Registers calibrated lane polygons for a camera index."""
        self.camera_lanes[cam_idx] = lanes
        for lane_id, info in lanes.items():
            self.corridor_estimator.register_lane(
                lane_id=lane_id,
                direction=info.get("direction", "N"),
                role=info.get("role", "queue"),
                max_cap_0_100m=info.get("max_cap_0_100m"),
            )

    def add_gate(self, gate: VirtualGate) -> None:
        if gate.gate_id in self.gates:
            raise ValueError(f"Duplicate gate_id '{gate.gate_id}': gate IDs must be unique across all cameras")
        self.gates[gate.gate_id] = gate
        self.gates_by_cam[gate.cam_idx].append(gate)

    def reset_camera(self, cam_idx: int) -> None:
        """Drops temporal state for one camera after a discontinuity (reconnect, file loop)."""
        for key in [k for k in self.track_prev_pts if k[0] == cam_idx]:
            del self.track_prev_pts[key]
        for key in [k for k in self.track_lane_history if k[0] == cam_idx]:
            del self.track_lane_history[key]
        self._cam_last_t.pop(cam_idx, None)
        self._invalid_interval_cameras.add(cam_idx)
        for gate in self.gates_by_cam.get(cam_idx, []):
            gate.reset_tracks()

    def resolve_lane_for_crossing(
        self,
        cam_idx: int,
        gate: VirtualGate,
        track_id: int,
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        cross_pt: Tuple[float, float],
        t: float,
    ) -> Optional[str]:
        """
        Resolves which lane a gate crossing belongs to using a 3-tier fallback:
        1. Explicit gate lane_id
        2. Recent track lane assignment from LaneMetricsManager
        3. Point-in-polygon containment or lateral distance along gate line
        """
        if gate.lane_id:
            return gate.lane_id

        # Tier 2: Check recent lane assignment from this track
        hist = self.track_lane_history.get((cam_idx, track_id))
        if hist is not None and t - hist[1] <= 2.0:
            return hist[0]

        # Tier 3: Test intersection point and endpoints against lane polygons
        lanes = self.camera_lanes.get(cam_idx, {})
        pt_cross = Point(cross_pt[0], cross_pt[1])
        pt_curr = Point(p_curr[0], p_curr[1])
        pt_prev = Point(p_prev[0], p_prev[1])

        for lid, linfo in lanes.items():
            poly = linfo.get("polygon")
            if poly is not None and (poly.contains(pt_cross) or poly.contains(pt_curr) or poly.contains(pt_prev)):
                return lid

        # Tier 4: Lateral projection fallback (minimum distance to lane polygon)
        best_lane = None
        min_dist = float("inf")
        for lid, linfo in lanes.items():
            poly = linfo.get("polygon")
            if poly is not None:
                d = poly.distance(pt_cross)
                if d < min_dist:
                    min_dist = d
                    best_lane = lid

        if best_lane and min_dist <= 60.0:  # Within 60px of lane polygon
            return best_lane

        return None

    def update_tracks(
        self,
        cam_idx: int,
        tracked_objs: np.ndarray,
        now: Optional[float] = None,
        track_lanes: Optional[Dict[int, str]] = None,
    ) -> List[str]:
        """
        Evaluates one observed frame of a camera.

        Args:
            cam_idx: Camera index
            tracked_objs: (M, 5+) [x1, y1, x2, y2, track_id, class_id, ...]
            now: Observation time in seconds on this camera's clock
            track_lanes: Optional mapping of track_id -> assigned lane_id from LaneMetricsManager
        """
        t = float(now) if now is not None else time.monotonic()

        # Update track lane history if provided
        if track_lanes:
            for tid, lid in track_lanes.items():
                self.track_lane_history[(cam_idx, tid)] = (lid, t)

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
                    cross_pt = get_segment_intersection(prev[0], curr_pt, gate.p1, gate.p2) or curr_pt
                    lane_id = self.resolve_lane_for_crossing(cam_idx, gate, track_id, prev[0], curr_pt, cross_pt, t)
                    if gate.check_crossing(track_id, prev[0], curr_pt, cls_id, now=t, lane_id=lane_id):
                        crossed.append(gate.gate_id)
                        # Feed crossing to Corridor Queue Estimator
                        self.corridor_estimator.record_gate_crossing(
                            gate_type=gate.gate_type,
                            direction=gate.target_dir,
                            lane_id=lane_id,
                            vehicle_class=cls_id,
                        )
            self.track_prev_pts[key] = (curr_pt, t)

        # Missing tracks keep their last point only within the bounded continuity gap.
        for key in [k for k, (_, kt) in self.track_prev_pts.items()
                    if k[0] == cam_idx and k[1] not in active and t - kt > self.max_gap_s]:
            del self.track_prev_pts[key]
        for key in [k for k, (_, kt) in self.track_lane_history.items()
                    if k[0] == cam_idx and k[1] not in active and t - kt > self.max_gap_s]:
            del self.track_lane_history[key]

        for gate in cam_gates:
            gate.prune_stale_tracks(active, now=t)

        return crossed

    # ------------------------------------------------------------------ reporting
    def interval_report(
        self,
        now_wall: Optional[float] = None,
        camera_validity: Optional[Dict[int, bool]] = None,
    ) -> Dict[str, Any]:
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
                "laneCounts": dict(g.interval_lane_counts) if valid else {},
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

        # Get enriched telemetry from Corridor Queue Estimator
        corridor_data = self.corridor_estimator.get_telemetry(interval_sec=window_s)

        return {
            "windowStart": iso_utc(self.interval_start_wall),
            "windowEnd": iso_utc(end_wall),
            "windowSec": round(window_s, 3),
            "gates": gates_out,
            "by_direction": by_dir,
            "corridor_by_direction": corridor_data["by_direction"],
            "by_lane": corridor_data["by_lane"],
            "outgoingfull": corridor_data.get("outgoingfull", {"N": False, "E": False, "S": False, "W": False}),
            "motorcycle_count_cam1": corridor_data.get("motorcycle_count_cam1", 0),
            "alarm_flags": corridor_data.get("alarm_flags", {
                "emergency_spillback_override": False,
                "target_override_lanes": [],
                "target_override_lane": None,
            }),
        }

    def get_corridor_accounting(self) -> Dict[str, Any]:
        """Session-cumulative totals for the operator HUD."""
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
        self.corridor_estimator.reset_interval()

    def get_mqtt_telemetry(
        self,
        now_wall: Optional[float] = None,
        camera_validity: Optional[Dict[int, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Flow block for the payload. Contains backward-compatible v1 fields as well
        as enriched by_lane and motorcycle direction aggregation.
        """
        report = self.interval_report(now_wall, camera_validity)
        legacy: Dict[str, Dict[str, Any]] = {}
        for d, entry in report["by_direction"].items():
            arr, dep = entry["arrivals"], entry["departures"]
            if arr is None and dep is None:
                continue
            corr_d = report["corridor_by_direction"].get(d, {})
            legacy[d] = {
                "inflow": arr["count"] if arr else None,
                "cleared": dep["count"] if dep else None,
                "live_discharge_rate": dep["ratePerSec"] if dep else None,
                "motorcycles": corr_d.get("motorcycles"),
                "cars_total": corr_d.get("cars_total"),
                "avg_speed_kmh": corr_d.get("avg_speed_kmh", 0.0),
                "flow_rate": corr_d.get("flow_rate", 0.0),
                "max_pressure": corr_d.get("max_pressure", 0.0),
                "spillback_active": corr_d.get("spillback_active", False),
                "is_approach_empty": corr_d.get("is_approach_empty", False),
            }
        return {
            "by_direction": legacy,
            "by_lane": report["by_lane"],
            "outgoingfull": report["outgoingfull"],
            "motorcycle_count_cam1": report["motorcycle_count_cam1"],
            "alarm_flags": report["alarm_flags"],
            "interval": report,
        }
