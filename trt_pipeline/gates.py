"""
Virtual Counting Gates & Cumulative Inflow-Outflow Accounting Engine.
Implements robust 2D directed tripwires that track vehicle passage events,
eliminating front-to-rear appearance mismatch and providing exact flow rates,
corridor queue accumulation, and turning movements.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2 as cv
import numpy as np


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
    """
    Directed 2D Virtual Tripwire Gate.
    Detects when a vehicle trajectory crosses the line in the designated forward direction.
    """

    def __init__(
        self,
        gate_id: str,
        cam_idx: int,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        gate_type: str = "stopline",  # 'stopline' (cleared/discharge) or 'ingress' (platoon inflow)
        direction_vec: Optional[Tuple[float, float]] = None,
        label: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 255),
        target_dir: Optional[str] = None,
    ):
        self.gate_id = gate_id
        self.cam_idx = cam_idx
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.gate_type = gate_type.lower()
        self.label = label or gate_id
        self.color = color

        if target_dir is not None:
            self.target_dir = str(target_dir).upper()
        else:
            # Deduce approach direction from gate_id (e.g. GATE_N_IN -> 'N')
            parts = gate_id.upper().split('_')
            dirs = {'N', 'S', 'E', 'W'}
            found = next((p for p in parts if p in dirs), None)
            self.target_dir = found or "N"

        # Compute line vector & normal direction vector
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        line_len = max(1e-6, np.hypot(dx, dy))

        if direction_vec is not None:
            nx, ny = float(direction_vec[0]), float(direction_vec[1])
            n_len = max(1e-6, np.hypot(nx, ny))
            self.normal = (nx / n_len, ny / n_len)
        else:
            # Default normal: 90 degrees clockwise perpendicular (-dy, dx)
            self.normal = (-dy / line_len, dx / line_len)

        self.midpoint = ((self.p1[0] + self.p2[0]) * 0.5, (self.p1[1] + self.p2[1]) * 0.5)

        self.count = 0
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.crossed_track_ids: Set[int] = set()
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
        """
        Checks if the trajectory from p_prev to p_curr crosses this gate line.
        Ensures directionality (must move forward along normal vector) and de-duplicates.
        """
        if track_id in self.crossed_track_ids:
            return False

        # Segment intersection test
        if not segments_intersect(p_prev, p_curr, self.p1, self.p2):
            return False

        # Directional dot product check
        move_x = p_curr[0] - p_prev[0]
        move_y = p_curr[1] - p_prev[1]
        dot = move_x * self.normal[0] + move_y * self.normal[1]

        # Require motion to be generally aligned with the gate normal (> 0)
        if dot <= 0.0:
            return False

        # Valid forward crossing detected!
        ts = now if now is not None else time.time()
        self.crossed_track_ids.add(track_id)
        self.count += 1
        self.class_counts[class_id] += 1
        self.last_flash_ts = ts

        self.recent_crossings.append({
            "gate_id": self.gate_id,
            "track_id": track_id,
            "class_id": class_id,
            "timestamp": ts,
        })
        return True

    def prune_stale_tracks(self, active_track_ids: Set[int]) -> None:
        """Keeps crossed track history clean while retaining recent IDs to prevent double counting."""
        if len(self.crossed_track_ids) > 500:
            # Keep tracks that crossed recently or are still active
            recent_ids = set(c["track_id"] for c in self.recent_crossings)
            self.crossed_track_ids = (self.crossed_track_ids & active_track_ids) | recent_ids


class GateFlowManager:
    """
    Coordinates Virtual Counting Gates across all camera streams.
    Calculates live corridor inflow-outflow balance, discharge flow rate,
    and turning movement distributions.
    """

    def __init__(self, camera_names: Optional[List[str]] = None):
        self.camera_names = camera_names or []
        self.gates: Dict[str, VirtualGate] = {}
        self.gates_by_cam: Dict[int, List[VirtualGate]] = defaultdict(list)

        # Vehicle previous positions: (cam_idx, track_id) -> (x, y)
        self.track_prev_pts: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Aggregate flow metrics
        self.interval_start_time = time.time()
        self.interval_stopline_clearances = 0
        self.interval_dir_clearances: Dict[str, int] = defaultdict(int)

    def add_gate(self, gate: VirtualGate) -> None:
        """Registers a VirtualGate."""
        self.gates[gate.gate_id] = gate
        self.gates_by_cam[gate.cam_idx].append(gate)

    def update_tracks(
        self,
        cam_idx: int,
        tracked_objs: np.ndarray,
        now: Optional[float] = None,
    ) -> List[str]:
        """
        Evaluates all tracked vehicles in a camera against its registered gates.

        Args:
            cam_idx: Camera index
            tracked_objs: Array of shape (M, 5+) [x1, y1, x2, y2, track_id, ...]
            now: Current timestamp

        Returns:
            List of gate_ids that were crossed in this frame
        """
        ts = now if now is not None else time.time()
        crossed_gates_this_frame: List[str] = []
        cam_gates = self.gates_by_cam.get(cam_idx, [])
        if not cam_gates or tracked_objs is None or len(tracked_objs) == 0:
            return crossed_gates_this_frame

        active_lids = set()
        for obj in tracked_objs:
            ox1, oy1, ox2, oy2, track_id = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), int(obj[4])
            cls_id = int(obj[6]) if len(obj) >= 7 else (int(obj[5]) if len(obj) >= 6 else 0)
            active_lids.add(track_id)

            # Use bottom-center contact point with road
            cx = (ox1 + ox2) * 0.5
            cy = oy2
            curr_pt = (cx, cy)
            key = (cam_idx, track_id)

            if key in self.track_prev_pts:
                prev_pt = self.track_prev_pts[key]
                for gate in cam_gates:
                    if gate.check_crossing(track_id, prev_pt, curr_pt, cls_id, now=ts):
                        crossed_gates_this_frame.append(gate.gate_id)
                        if gate.gate_type == "stopline":
                            self.interval_stopline_clearances += 1
                            self.interval_dir_clearances[gate.target_dir] += 1

            self.track_prev_pts[key] = curr_pt

        # Cleanup disappeared tracks
        dead_keys = [k for k in self.track_prev_pts if k[0] == cam_idx and k[1] not in active_lids]
        for k in dead_keys:
            self.track_prev_pts.pop(k, None)

        for gate in cam_gates:
            gate.prune_stale_tracks(active_lids)

        return crossed_gates_this_frame

    def get_corridor_accounting(self) -> Dict[str, Any]:
        """
        Computes the Inflow-Clearance balance and live discharge rate,
        both aggregated and segmented by approach direction (N, S, E, W).
        """
        inflow = sum(g.count for g in self.gates.values() if g.gate_type == "ingress")
        stopline = sum(g.count for g in self.gates.values() if g.gate_type == "stopline")
        corridor_queue = max(0, inflow - stopline)

        now = time.time()
        dt = max(0.1, now - self.interval_start_time)
        discharge_rate = self.interval_stopline_clearances / dt

        # Directional breakdown for signal controller
        directions = {"N", "S", "E", "W"}
        for g in self.gates.values():
            directions.add(g.target_dir)

        by_direction: Dict[str, Dict[str, Any]] = {}
        for d in sorted(directions):
            d_inflow = sum(g.count for g in self.gates.values() if g.target_dir == d and g.gate_type == "ingress")
            d_cleared = sum(g.count for g in self.gates.values() if g.target_dir == d and g.gate_type == "stopline")
            d_rate = self.interval_dir_clearances[d] / dt
            by_direction[d] = {
                "inflow": d_inflow,
                "cleared": d_cleared,
                "live_discharge_rate": round(d_rate, 2),
            }

        return {
            "total_inflow": inflow,
            "total_stopline_cleared": stopline,
            "corridor_queue": corridor_queue,
            "discharge_rate_cars_per_sec": round(discharge_rate, 2),
            "by_direction": by_direction,
            "gate_counts": {gid: g.count for gid, g in self.gates.items()},
        }

    def reset_interval(self) -> None:
        """Resets periodic rate counters."""
        self.interval_start_time = time.time()
        self.interval_stopline_clearances = 0
        self.interval_dir_clearances.clear()

    def get_mqtt_telemetry(self) -> Dict[str, Any]:
        """
        Formats streamlined flow telemetry per approach direction for the controller:
        - Adaptive flow: live_discharge_rate
        - Platoon tracking: inflow
        - Cumulative cleared count: cleared
        """
        accounting = self.get_corridor_accounting()

        return {
            "by_direction": accounting["by_direction"],
        }
