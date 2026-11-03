"""
Dual-Camera Smart Traffic Counting & Virtual Cart Handover System.

Implements the full specification from TRAFFIC_SYSTEM_LOGIC_FULL.md:
- Zone A: Upstream Detection Zone (Cam 2, e.g. 100m+ to 100m):
    * Real-time vehicle count N_vis_Cam2[lane] (cars/trucks/buses/three-wheelers, excluding motorcycles).
    * Stopping detection at Cam 2 (speed < 8 km/h or queued state).
- 100m Check-in Line (Upstream Ingress Gate / Tripwire):
    * Enqueues a Virtual Token into the per-lane Virtual Memory Cart (FIFO deque) with initial age=0, ttl=8.5s.
    * Motorcycle Exclusion Rule: Motorcycles skip token enqueueing at Cam 2.
- Zone B: Blind Spot Zone (100m to 55m) & Virtual Memory Carts:
    * Adaptive Per-Lane TTL with Freeze Mechanism:
      If any vehicle in Zone C (Cam 1, 0-55m) of that lane is stopped (speed < 8 km/h / queued),
      is_lane_frozen[lane] = True and TTL countdown is paused (d(TTL)/dt = 0) while age continues to increment.
      Otherwise TTL decrements by dt, auto-expiring tokens with TTL <= 0 (e.g. vehicles that turned into a side soi at 70m).
- Zone C: Intersection Approach Zone (Cam 1, 55m to 0m) & Handover Boundary:
    * Real-time vehicle count N_vis_Cam1[lane] (excluding motorcycles) and independent motorcycle count bikes_cam1.
    * 4-tier Priority FIFO Handover & Adjacent Lane Borrowing when a non-motorcycle vehicle enters Cam 1 (55m):
        1. Exact vehicle class match in target lane cart with age >= 1.0s (or any age if exact match).
        2. Any vehicle class in target lane cart with age >= 0.7s (or oldest in target lane cart).
        3. Adjacent lane borrowing (±1 lane in same direction, even if that adjacent lane is frozen).
        4. Fallback borrowing: oldest token from any remaining lane in the same direction.
- Dual-Verification Spillback Override Mode:
    * spillback[lane] = stopped_cam1[lane] AND stopped_cam2[lane].
    * Normal Hybrid Demand: Demand[lane] = N_vis_Cam1[lane] + N_vis_Cam2[lane] + Buffer[lane].
    * Spillback Override Demand: Demand[lane] = MaxCap_0_100m[lane] (default 15) + N_vis_Cam2[lane].
- Per-lane speed & flow rate metrics for adaptive / Max-Pressure signal control.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

# Vehicle categorization
MOTORCYCLE_CLASSES = {"motorcycle", "motorbike", "bicycle", "bike"}
CAR_CLASSES = {"car", "truck", "bus", "three_wheeler", "tuktuk", "van"}


def is_motorcycle(class_name_or_id: Any) -> bool:
    if isinstance(class_name_or_id, int):
        # 1: motorcycle in custom Thai model & COCO bicycle, 3: COCO motorcycle
        return class_name_or_id in (1, 3)
    name = str(class_name_or_id).lower().strip().replace("-", "_").replace(" ", "_")
    return name in MOTORCYCLE_CLASSES


def normalize_v_type(class_name_or_id: Any) -> str:
    if is_motorcycle(class_name_or_id):
        return "motorcycle"
    if isinstance(class_name_or_id, int):
        return {0: "car", 2: "car", 4: "three_wheeler", 5: "bus", 7: "truck"}.get(class_name_or_id, "car")
    name = str(class_name_or_id).lower().strip().replace("-", "_").replace(" ", "_")
    if name in ("tuktuk", "three_wheeler"):
        return "three_wheeler"
    if name in ("car", "truck", "bus", "van"):
        return name
    return "car"


@dataclass
class VirtualToken:
    """Represents a vehicle traversing the blind-spot zone (100m -> 55m)."""

    id: int
    lane_id: str
    direction: str
    v_type: str
    age: float = 0.0
    ttl: float = 8.5
    created_mono: float = field(default_factory=time.monotonic)


@dataclass
class LaneCorridorState:
    lane_id: str
    direction: str
    target_lane_id: str  # Downstream stopline lane identity (e.g. N1)
    lane_index: int = 1  # Numeric index within direction (e.g. 1 for N1, 2 for N2)
    max_capacity_0_100m: int = 15  # Spec default: 15 cars (0-100m) or 9 for pocket lane
    max_capacity: int = 40

    # Virtual Memory Cart (Zone B: 100m -> 55m)
    cart: Deque[VirtualToken] = field(default_factory=deque)
    is_frozen: bool = False
    spillback: bool = False

    # Zone C (Cam 1: 0-55m) & Zone A (Cam 2: 100m+) visual counts (non-motorcycle)
    vis_cam1: int = 0
    vis_cam2: int = 0
    queued_cam1: int = 0
    moving_cam1: int = 0
    queued_cam2: int = 0
    moving_cam2: int = 0
    stopped_cam1: bool = False
    stopped_cam2: bool = False

    # Motorcycles in Cam 1 (Zone C)
    vis_motorcycles_cam1: int = 0
    vis_motorcycles_cam2: int = 0

    # Speed metrics
    avg_speed_cam1: float = 0.0
    avg_speed_cam2: float = 0.0
    avg_speed_kmh: float = 0.0

    # Flow & Handover counters
    total_inflow: int = 0
    total_outflow: int = 0
    interval_inflow: int = 0
    interval_outflow: int = 0
    handover_exact_count: int = 0
    handover_borrowed_count: int = 0
    expired_tokens_count: int = 0
    last_borrowed_from: Optional[str] = None
    last_borrow_mono: Optional[float] = None

    # Track IDs already handed over in Cam 1 (Zone C) so each arriving vehicle pulls at most 1 token
    seen_cam1_tracks: Dict[int, float] = field(default_factory=dict)
    # Track IDs seen in Cam 2 (Zone A) for automatic check-in when crossing 100m or leaving Zone A
    seen_cam2_tracks: Dict[int, str] = field(default_factory=dict)

    # Final computed demand & legacy compatibility aliases
    demand_count: int = 0
    fused_waiting_cars: int = 0
    visual_queued_cars: int = 0
    visual_motorcycles: int = 0
    last_empty_mono: Optional[float] = None


def _parse_lane_index(lane_id: str) -> int:
    digits = "".join(ch for ch in lane_id if ch.isdigit())
    return int(digits) if digits else 1


class CorridorQueueEstimator:
    """
    Dual-Camera Virtual Cart & Queue Fusion Engine.
    Implements per-lane Virtual Memory Carts, Adaptive Per-Lane TTL Freeze,
    4-tier Priority FIFO Handover with Adjacent Borrowing, Dual-Verification Spillback Override,
    and Motorcycle Cam 1 Only Exclusion Policy.
    """

    def __init__(
        self,
        lane_mappings: Optional[Dict[str, str]] = None,
        max_lane_capacity: int = 40,
        default_cap_0_100m: int = 15,
        default_ttl_s: float = 8.5,
        freeze_speed_kmh: float = 8.0,
        drift_reset_duration_s: float = 3.0,
        saturation_flow_per_sec: float = 0.5,  # ~1800 veh/hr/lane (1 car every 2.0s)
    ):
        """
        Args:
            lane_mappings: Mapping from upstream lane IDs to downstream stopline lane IDs,
                           e.g. {"NE1": "N1", "NE2": "N2", "NE3": "N3"}.
            max_lane_capacity: Absolute physical ceiling for a lane.
            default_cap_0_100m: Max capacity between 0m and 100m for Spillback Override (spec: 15 cars).
            default_ttl_s: Initial TTL in seconds for tokens dropped at 100m check-in line (spec: 8.0-8.5s).
            freeze_speed_kmh: Speed threshold (km/h) below which a vehicle is considered stopped (spec: 8.0 km/h).
            drift_reset_duration_s: Sustained empty duration for drift cleanup.
            saturation_flow_per_sec: Nominal saturation flow rate (veh/s) per lane for Max-Pressure calculation.
        """
        self.lane_mappings = dict(lane_mappings or {})
        self.reverse_mappings: Dict[str, List[str]] = defaultdict(list)
        for up_lid, down_lid in self.lane_mappings.items():
            if up_lid not in self.reverse_mappings[down_lid]:
                self.reverse_mappings[down_lid].append(up_lid)

        self.max_lane_capacity = max_lane_capacity
        self.default_cap_0_100m = default_cap_0_100m
        self.default_ttl_s = float(default_ttl_s)
        self.freeze_speed_kmh = float(freeze_speed_kmh)
        self.drift_reset_duration_s = float(drift_reset_duration_s)
        self.saturation_flow_per_sec = float(saturation_flow_per_sec)

        self.next_token_id: int = 1
        self._last_step_mono: Optional[float] = None

        # Per stop-line lane state: lane_id -> LaneCorridorState
        self.lanes: Dict[str, LaneCorridorState] = {}
        # Upstream lane IDs registered as upstream role
        self.upstream_lanes: Set[str] = set()
        # Outgoing lane IDs registered as outgoing role: lane_id -> direction
        self.outgoing_lanes: Dict[str, str] = {}

        # Direction-level motorcycle pools (Cam 1 only policy + gate flow tracking)
        self.moto_pools: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "cumulative_inflow": 0,
            "cumulative_outflow": 0,
            "interval_inflow": 0,
            "interval_outflow": 0,
            "waiting_count": 0,
            "visual_present": 0,
            "last_empty_mono": None,
        })

        # Direction-level cars aggregate
        self.car_pools: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "cumulative_inflow": 0,
            "cumulative_outflow": 0,
            "interval_inflow": 0,
            "interval_outflow": 0,
            "waiting_count": 0,
        })

        # Outgoing road full status per direction (N, E, S, W -> bool)
        self.outgoing_full: Dict[str, bool] = {"N": False, "E": False, "S": False, "W": False}

    def register_lane(
        self,
        lane_id: str,
        direction: str,
        target_lane_id: Optional[str] = None,
        role: str = "queue",
        max_cap_0_100m: Optional[int] = None,
    ) -> None:
        if role == "outgoing":
            self.outgoing_lanes[lane_id] = direction.upper()
            return
        if role == "upstream" or lane_id in self.lane_mappings:
            self.upstream_lanes.add(lane_id)
            target = target_lane_id or self.lane_mappings.get(lane_id, lane_id)
            self.lane_mappings[lane_id] = target
            if lane_id not in self.reverse_mappings[target]:
                self.reverse_mappings[target].append(lane_id)
        else:
            target = target_lane_id or self.lane_mappings.get(lane_id, lane_id)

        if target not in self.lanes:
            self.lanes[target] = LaneCorridorState(
                lane_id=target,
                direction=direction.upper(),
                target_lane_id=target,
                lane_index=_parse_lane_index(target),
                max_capacity_0_100m=int(max_cap_0_100m if max_cap_0_100m is not None else self.default_cap_0_100m),
                max_capacity=self.max_lane_capacity,
            )

    # ------------------------------------------------------------------ Virtual Cart Operations
    def enqueue_token(
        self,
        lane_id: str,
        vehicle_class: Any = "car",
        ttl: Optional[float] = None,
        now_mono: Optional[float] = None,
    ) -> Optional[VirtualToken]:
        """
        Drops a Virtual Token into the target lane's blind-spot memory cart (100m Check-in Line).
        Enforces the Motorcycle Exclusion Rule: motorcycles never drop tokens into the cart.
        """
        if is_motorcycle(vehicle_class):
            return None

        target_lane = self.lane_mappings.get(lane_id, lane_id)
        if target_lane not in self.lanes:
            return None

        lst = self.lanes[target_lane]
        v_type = normalize_v_type(vehicle_class)
        token = VirtualToken(
            id=self.next_token_id,
            lane_id=target_lane,
            direction=lst.direction,
            v_type=v_type,
            age=0.0,
            ttl=float(ttl if ttl is not None else self.default_ttl_s),
            created_mono=now_mono if now_mono is not None else time.monotonic(),
        )
        self.next_token_id += 1
        lst.cart.append(token)
        self._recompute_lane_demand(lst)
        return token

    def handover_token(
        self,
        lane_id: str,
        vehicle_class: Any = "car",
        require_min_age: bool = True,
        now_mono: Optional[float] = None,
    ) -> Optional[VirtualToken]:
        """
        4-Tier Priority FIFO Handover & Adjacent Lane Borrowing at the 55m Cam 1 boundary:
          1. Exact vehicle class match in target lane cart with age >= 1.0s (or any age if not require_min_age).
          2. Any vehicle class in target lane cart with age >= 0.7s (or oldest in target lane cart).
          3. Adjacent lane borrowing (±1 lane index in the same direction, even if frozen).
          4. Fallback borrowing: oldest token from any other lane in the same direction.
        Motorcycles are excluded and never consume cart tokens.
        """
        if is_motorcycle(vehicle_class):
            return None

        target_lane = self.lane_mappings.get(lane_id, lane_id)
        if target_lane not in self.lanes:
            return None

        now = now_mono if now_mono is not None else time.monotonic()
        lst = self.lanes[target_lane]
        v_type = normalize_v_type(vehicle_class)

        # Priority 1: Exact vehicle type match in own lane cart (age >= 1.0s preferred, then any age)
        for min_age in ((1.0, 0.0) if require_min_age else (0.0,)):
            for tk in list(lst.cart):
                if tk.v_type == v_type and tk.age >= min_age:
                    lst.cart.remove(tk)
                    lst.handover_exact_count += 1
                    self._recompute_lane_demand(lst)
                    return tk

        # Priority 2: Any vehicle type in own lane cart (age >= 0.7s preferred, then FIFO oldest)
        for min_age in ((0.7, 0.0) if require_min_age else (0.0,)):
            for tk in list(lst.cart):
                if tk.age >= min_age:
                    lst.cart.remove(tk)
                    lst.handover_exact_count += 1
                    self._recompute_lane_demand(lst)
                    return tk

        # Priority 3: Adjacent Lane Borrowing (±1 lane index in same direction, even if frozen!)
        same_dir_lanes = [
            other for other in self.lanes.values()
            if other.direction == lst.direction and other.lane_id != target_lane and len(other.cart) > 0
        ]
        adj_lanes = [
            other for other in same_dir_lanes
            if abs(other.lane_index - lst.lane_index) == 1
        ]
        adj_lanes.sort(key=lambda o: len(o.cart), reverse=True)
        for adj in adj_lanes:
            tk = adj.cart.popleft()
            lst.handover_borrowed_count += 1
            lst.last_borrowed_from = adj.lane_id
            lst.last_borrow_mono = now
            self._recompute_lane_demand(adj)
            self._recompute_lane_demand(lst)
            return tk

        # Priority 4: Fallback Borrowing (oldest token across any remaining lane in same direction)
        if same_dir_lanes:
            same_dir_lanes.sort(key=lambda o: o.cart[0].age if o.cart else -1.0, reverse=True)
            chosen = same_dir_lanes[0]
            tk = chosen.cart.popleft()
            lst.handover_borrowed_count += 1
            lst.last_borrowed_from = chosen.lane_id
            lst.last_borrow_mono = now
            self._recompute_lane_demand(chosen)
            self._recompute_lane_demand(lst)
            return tk

        return None

    def _tick_carts(self, dt: float) -> None:
        """
        Updates Virtual Memory Carts across all lanes by elapsed time dt:
        - If lane is frozen (stopped vehicle at Cam 1), TTL is paused (d(TTL)/dt = 0) while age increases.
        - If lane is moving freely, TTL decreases by dt and tokens with TTL <= 0 auto-expire (midblock soi turn-in).
        """
        if dt <= 0.0:
            return
        for lst in self.lanes.values():
            lst.is_frozen = bool(lst.stopped_cam1)
            if not lst.is_frozen:
                remaining = deque()
                while lst.cart:
                    tk = lst.cart.popleft()
                    tk.age += dt
                    tk.ttl -= dt
                    if tk.ttl > 0.0:
                        remaining.append(tk)
                    else:
                        lst.expired_tokens_count += 1
                lst.cart = remaining
            else:
                for tk in lst.cart:
                    tk.age += dt

    def _recompute_lane_demand(self, lst: LaneCorridorState) -> None:
        """
        Computes final per-lane vehicle demand according to the specification formulas:
        - Spillback Override Mode (stopped at Cam 1 AND stopped at Cam 2):
            Demand = MaxCap(0-100m: 15) + N_vis_Cam2
        - Normal & Blind-Spot Buffer Mode:
            Demand = N_vis_Cam1 + N_vis_Cam2 + Buffer (len(cart))
        Clamped to [N_vis_Cam1, max_capacity].
        """
        lst.is_frozen = bool(lst.stopped_cam1)
        lst.spillback = bool(lst.stopped_cam1 and lst.stopped_cam2)
        if lst.spillback:
            raw_demand = lst.max_capacity_0_100m + lst.vis_cam2
        else:
            raw_demand = lst.vis_cam1 + lst.vis_cam2 + len(lst.cart)

        lst.demand_count = max(lst.vis_cam1, min(lst.max_capacity, raw_demand))
        lst.fused_waiting_cars = lst.demand_count

    # ------------------------------------------------------------------ Gate & Visual Integration
    def record_gate_crossing(
        self,
        gate_type: str,
        direction: str,
        lane_id: Optional[str],
        vehicle_class: Any,
        count: int = 1,
        now_mono: Optional[float] = None,
    ) -> None:
        """
        Records a vehicle crossing an ingress (100m check-in), handover (55m), or stopline (0m) gate.
        """
        d = direction.upper()
        is_moto = is_motorcycle(vehicle_class)

        target_lane = None
        if lane_id:
            target_lane = self.lane_mappings.get(lane_id, lane_id)

        if is_moto:
            # Motorcycle Policy: Cam 1 only for queue/demand; track gate flow at direction level
            pool = self.moto_pools[d]
            if gate_type == "ingress":
                pool["cumulative_inflow"] += count
                pool["interval_inflow"] += count
            elif gate_type == "stopline":
                pool["cumulative_outflow"] += count
                pool["interval_outflow"] += count
                pool["waiting_count"] = max(0, pool["waiting_count"] - count)
            return

        c_pool = self.car_pools[d]
        if gate_type == "ingress":
            c_pool["cumulative_inflow"] += count
            c_pool["interval_inflow"] += count
            c_pool["waiting_count"] += count

            if target_lane and target_lane in self.lanes:
                lane_st = self.lanes[target_lane]
                lane_st.total_inflow += count
                lane_st.interval_inflow += count
                for _ in range(count):
                    self.enqueue_token(target_lane, vehicle_class=vehicle_class, now_mono=now_mono)

        elif gate_type == "handover":
            if target_lane and target_lane in self.lanes:
                for _ in range(count):
                    self.handover_token(target_lane, vehicle_class=vehicle_class, require_min_age=False, now_mono=now_mono)

        elif gate_type == "stopline":
            c_pool["cumulative_outflow"] += count
            c_pool["interval_outflow"] += count
            c_pool["waiting_count"] = max(0, c_pool["waiting_count"] - count)

            if target_lane and target_lane in self.lanes:
                lane_st = self.lanes[target_lane]
                lane_st.total_outflow += count
                lane_st.interval_outflow += count
                # If a car exits stopline without having triggered 55m handover earlier,
                # consume a token from cart or adjacent cart so tokens never linger.
                if lane_st.vis_cam1 == 0:
                    for _ in range(count):
                        self.handover_token(target_lane, vehicle_class=vehicle_class, require_min_age=False, now_mono=now_mono)
                self._recompute_lane_demand(lane_st)

    def update_visual_occupancy(
        self,
        lane_visuals: Dict[str, Dict[str, Any]],
        now_mono: Optional[float] = None,
        camera_validity: Optional[Dict[str, bool]] = None,
        dt: Optional[float] = None,
    ) -> None:
        """
        Per-cycle calculation & decision logic (Executed every frame / evaluation step Δt):
          1. Separate Cam 1 (stopline, 0-55m) and Cam 2 (upstream, 100m+) visual counts (excluding motorcycles).
          2. Detect stopped vehicles (Speed < 8 km/h or queued state) at Cam 1 and Cam 2 per lane.
          3. Perform automatic 55m FIFO Handover when new vehicle tracks appear in Cam 1.
          4. Freeze TTL for lanes with stopped vehicles at Cam 1; tick down & expire tokens for moving lanes.
          5. Evaluate Dual-Verification Spillback Override (stopped_cam1 AND stopped_cam2).
          6. Compute Final Demand per lane.
        """
        now = now_mono if now_mono is not None else time.monotonic()
        if dt is None:
            dt = max(0.0, min(1.0, now - self._last_step_mono)) if self._last_step_mono is not None else 0.0
        self._last_step_mono = now

        dir_visual_motos: Dict[str, int] = defaultdict(int)

        # Reset transient visual counters for lanes present in update
        updated_targets: Set[str] = set()
        cam1_data_by_target: Dict[str, Dict[str, Any]] = {}
        cam2_data_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        outgoing_stopped_by_dir: Dict[str, bool] = defaultdict(bool)
        seen_outgoing_dirs: Set[str] = set()

        for lane_id, vis in lane_visuals.items():
            if vis.get("role") == "outgoing" or lane_id in self.outgoing_lanes:
                out_dir = str(vis.get("direction") or self.outgoing_lanes.get(lane_id, "N")).upper()
                seen_outgoing_dirs.add(out_dir)
                q_c = int(vis.get("cars_queued", 0))
                if bool(vis.get("has_stopped", q_c > 0)) or int(vis.get("cars_total", 0)) >= self.default_cap_0_100m:
                    outgoing_stopped_by_dir[out_dir] = True
                continue
            is_upstream = (
                vis.get("role") == "upstream"
                or lane_id in self.upstream_lanes
                or (lane_id in self.lane_mappings and self.lane_mappings[lane_id] != lane_id)
            )
            target_lane = self.lane_mappings.get(lane_id, lane_id)
            if target_lane not in self.lanes:
                continue
            updated_targets.add(target_lane)
            if is_upstream:
                cam2_data_by_target[target_lane].append(vis)
            else:
                cam1_data_by_target[target_lane] = vis

        for target_lane in updated_targets:
            lst = self.lanes[target_lane]

            # --- Cam 1 (Zone C: 0-55m) ---
            c1 = cam1_data_by_target.get(target_lane)
            if c1 is not None:
                q_cars = int(c1.get("cars_queued", 0))
                t_cars = int(c1.get("cars_total", q_cars))
                m_cars = int(c1.get("cars_moving", max(0, t_cars - q_cars)))
                m_bikes = int(c1.get("motorcycles", 0))
                avg_spd = float(c1.get("avg_speed_kmh", 0.0))
                has_stopped = bool(c1.get("has_stopped", q_cars > 0))

                # Automatic 55m Handover when new non-motorcycle tracks enter Cam 1
                car_tracks: Dict[int, str] = c1.get("car_tracks") or {}
                for tid, v_cls in car_tracks.items():
                    if tid not in lst.seen_cam1_tracks:
                        lst.seen_cam1_tracks[tid] = now
                        self.handover_token(target_lane, vehicle_class=v_cls, require_min_age=True, now_mono=now)
                    else:
                        lst.seen_cam1_tracks[tid] = now

                # Prune old seen Cam 1 tracks (> 30s)
                lst.seen_cam1_tracks = {
                    tid: ts for tid, ts in lst.seen_cam1_tracks.items()
                    if tid in car_tracks or (now - ts <= 30.0)
                }

                lst.vis_cam1 = t_cars
                lst.queued_cam1 = q_cars
                lst.moving_cam1 = m_cars
                lst.vis_motorcycles_cam1 = m_bikes
                lst.stopped_cam1 = has_stopped
                lst.avg_speed_cam1 = avg_spd
                lst.visual_queued_cars = q_cars
                lst.visual_motorcycles = m_bikes
                dir_visual_motos[lst.direction] += m_bikes

            # --- Cam 2 (Zone A: 100m+) ---
            c2_list = cam2_data_by_target.get(target_lane)
            if c2_list:
                q_cars2 = sum(int(x.get("cars_queued", 0)) for x in c2_list)
                t_cars2 = sum(int(x.get("cars_total", int(x.get("cars_queued", 0)))) for x in c2_list)
                m_cars2 = sum(int(x.get("cars_moving", max(0, int(x.get("cars_total", 0)) - int(x.get("cars_queued", 0))))) for x in c2_list)
                m_bikes2 = sum(int(x.get("motorcycles", 0)) for x in c2_list)
                has_stopped2 = any(bool(x.get("has_stopped", int(x.get("cars_queued", 0)) > 0)) for x in c2_list)
                speeds2 = [float(x.get("avg_speed_kmh", 0.0)) for x in c2_list if float(x.get("avg_speed_kmh", 0.0)) > 0]
                avg_spd2 = sum(speeds2) / len(speeds2) if speeds2 else 0.0

                lst.vis_cam2 = t_cars2
                lst.queued_cam2 = q_cars2
                lst.moving_cam2 = m_cars2
                lst.vis_motorcycles_cam2 = m_bikes2
                lst.stopped_cam2 = has_stopped2
                lst.avg_speed_cam2 = avg_spd2
            elif target_lane not in self.reverse_mappings:
                lst.vis_cam2 = 0
                lst.queued_cam2 = 0
                lst.moving_cam2 = 0
                lst.stopped_cam2 = False
                lst.avg_speed_cam2 = 0.0

            # Combined lane average speed (weighted by visible cars in Cam 1 & Cam 2)
            total_vis = lst.vis_cam1 + lst.vis_cam2
            if total_vis > 0:
                lst.avg_speed_kmh = round(
                    (lst.avg_speed_cam1 * lst.vis_cam1 + lst.avg_speed_cam2 * lst.vis_cam2) / total_vis,
                    2,
                )
            elif lst.stopped_cam1 or lst.stopped_cam2:
                lst.avg_speed_kmh = 0.0
            else:
                lst.avg_speed_kmh = round(max(lst.avg_speed_cam1, lst.avg_speed_cam2), 2)

        # Tick TTL & Freeze mechanism for all carts
        self._tick_carts(dt)

        # Recompute demand & handle empty-corridor auto-zero drift protection
        for lst in self.lanes.values():
            if lst.vis_cam1 == 0 and lst.vis_cam2 == 0 and lst.vis_motorcycles_cam1 == 0:
                if lst.last_empty_mono is None:
                    lst.last_empty_mono = now
                elif now - lst.last_empty_mono >= self.drift_reset_duration_s and not lst.is_frozen:
                    lst.cart.clear()
            else:
                lst.last_empty_mono = None

            self._recompute_lane_demand(lst)

        # Outgoing road full status per direction (default False; not implemented yet)
        for d in ("N", "E", "S", "W"):
            self.outgoing_full.setdefault(d, False)

        # Update direction-level motorcycle pools (Cam 1 only policy)
        for d in set(list(self.moto_pools.keys()) + [l.direction for l in self.lanes.values()]):
            pool = self.moto_pools[d]
            vis_moto = dir_visual_motos.get(d, 0)
            pool["visual_present"] = vis_moto
            pool["waiting_count"] = vis_moto
            if vis_moto == 0:
                if pool["last_empty_mono"] is None:
                    pool["last_empty_mono"] = now
            else:
                pool["last_empty_mono"] = None

    def step(
        self,
        dt: float,
        cam1_cars: List[Dict[str, Any]],
        cam2_cars: List[Dict[str, Any]],
        cam1_range_m: float = 55.0,
        cam2_range_m: float = 40.0,
    ) -> Dict[str, Any]:
        """
        Direct reference step interface matching Section 7 of TRAFFIC_SYSTEM_LOGIC_FULL.md.
        Accepts per-vehicle dicts with keys:
          'lane' (int 0..2 or lane_id str), 'x_meters' (float), 'class' (str),
          'speed_kmh' (float), 'stopped_time' (float),
          'crossed_100m_trigger' (bool), 'crossed_55m_trigger' (bool).
        """
        ordered_lanes = sorted(self.lanes.keys())

        def resolve_lid(ln_key: Any) -> str:
            if isinstance(ln_key, str) and ln_key in self.lanes:
                return ln_key
            if isinstance(ln_key, str) and ln_key in self.lane_mappings:
                return self.lane_mappings[ln_key]
            if isinstance(ln_key, int) and 0 <= ln_key < len(ordered_lanes):
                return ordered_lanes[ln_key]
            lid_str = f"L{int(ln_key) + 1}" if isinstance(ln_key, int) else str(ln_key)
            if lid_str not in self.lanes:
                self.register_lane(lid_str, "N")
                ordered_lanes.append(lid_str)
            return lid_str

        # Ensure at least 3 lanes exist if none were registered
        if not ordered_lanes:
            for i in range(3):
                self.register_lane(f"L{i+1}", "N")
            ordered_lanes = sorted(self.lanes.keys())

        for lst in self.lanes.values():
            lst.vis_cam1 = 0
            lst.vis_cam2 = 0
            lst.vis_motorcycles_cam1 = 0
            lst.stopped_cam1 = False
            lst.stopped_cam2 = False

        bikes_cam1 = 0
        speed_sums: Dict[str, List[float]] = defaultdict(list)

        # 1. Filter YOLO Visible Counts
        for c in cam1_cars:
            lid = resolve_lid(c["lane"])
            lst = self.lanes[lid]
            x_m = float(c.get("x_meters", 0.0))
            if 0.0 <= x_m <= cam1_range_m:
                if is_motorcycle(c.get("class", "car")):
                    bikes_cam1 += 1
                    lst.vis_motorcycles_cam1 += 1
                else:
                    lst.vis_cam1 += 1
                    spd = float(c.get("speed_kmh", 0.0))
                    speed_sums[lid].append(spd)
                    if spd < self.freeze_speed_kmh or float(c.get("stopped_time", 0.0)) > 1.5:
                        lst.stopped_cam1 = True

        for c in cam2_cars:
            lid = resolve_lid(c["lane"])
            lst = self.lanes[lid]
            if not is_motorcycle(c.get("class", "car")):
                x_m = float(c.get("x_meters", 110.0))
                spd = float(c.get("speed_kmh", 0.0))
                if 100.0 < x_m <= (100.0 + cam2_range_m):
                    lst.vis_cam2 += 1
                    speed_sums[lid].append(spd)
                if 95.0 <= x_m <= 115.0:
                    if spd < self.freeze_speed_kmh or float(c.get("stopped_time", 0.0)) > 2.0:
                        lst.stopped_cam2 = True

        # 2. Check for Upstream 100m Crossings (Check-in)
        for c in cam2_cars:
            if c.get("crossed_100m_trigger", False):
                if not is_motorcycle(c.get("class", "car")):
                    lid = resolve_lid(c["lane"])
                    self.enqueue_token(lid, vehicle_class=c.get("class", "car"))
                c["crossed_100m_trigger"] = False

        # 3. Per-Lane Freeze & TTL Management
        self._tick_carts(dt)

        # 4. Check for Cam 1 55m Arrivals (Handover)
        for c in cam1_cars:
            if c.get("crossed_55m_trigger", False):
                if not is_motorcycle(c.get("class", "car")):
                    lid = resolve_lid(c["lane"])
                    self.handover_token(lid, vehicle_class=c.get("class", "car"), require_min_age=False)
                c["crossed_55m_trigger"] = False

        # 5 & 6. Spillback Assessment & Final Demand Computation
        for lid in ordered_lanes:
            lst = self.lanes[lid]
            spds = speed_sums.get(lid, [])
            lst.avg_speed_kmh = round(sum(spds) / len(spds), 2) if spds else 0.0
            self._recompute_lane_demand(lst)

        return {
            "n_vis_cam1": [self.lanes[lid].vis_cam1 for lid in ordered_lanes],
            "n_vis_cam2": [self.lanes[lid].vis_cam2 for lid in ordered_lanes],
            "bikes_cam1": bikes_cam1,
            "cart_tokens": [len(self.lanes[lid].cart) for lid in ordered_lanes],
            "is_lane_frozen": [self.lanes[lid].is_frozen for lid in ordered_lanes],
            "spillback_lanes": [self.lanes[lid].spillback for lid in ordered_lanes],
            "final_demand": [self.lanes[lid].demand_count for lid in ordered_lanes],
            "avg_speed_kmh": [self.lanes[lid].avg_speed_kmh for lid in ordered_lanes],
            "outgoingfull": dict(self.outgoing_full),
        }

    def reset_interval(self) -> None:
        """Resets per-interval counters (call after successful publication)."""
        for l in self.lanes.values():
            l.interval_inflow = 0
            l.interval_outflow = 0
        for p in self.moto_pools.values():
            p["interval_inflow"] = 0
            p["interval_outflow"] = 0
        for p in self.car_pools.values():
            p["interval_inflow"] = 0
            p["interval_outflow"] = 0

    def get_telemetry(self, interval_sec: float = 1.0) -> Dict[str, Any]:
        """
        Generates enriched by_direction, by_lane, and alarm_flags telemetry for payload building.
        Includes avg_speed_kmh, flow_rate, pressure, cart_tokens, is_frozen, and spillback.
        """
        w_sec = max(1e-3, float(interval_sec))
        by_direction: Dict[str, Any] = {}
        spillback_lanes: List[str] = []

        for d in sorted(set(list(self.moto_pools.keys()) + [l.direction for l in self.lanes.values()])):
            m_pool = self.moto_pools[d]
            c_pool = self.car_pools[d]
            m_cleared = m_pool["interval_outflow"]
            c_cleared = c_pool["interval_outflow"]

            dir_lanes = [l for l in self.lanes.values() if l.direction == d]
            dir_waiting_cars = sum(l.demand_count for l in dir_lanes)
            max_lane_demand = max((l.demand_count for l in dir_lanes), default=0)
            dir_vis_motos = sum(l.vis_motorcycles_cam1 for l in dir_lanes)
            m_pool["visual_present"] = dir_vis_motos

            # Average speed across lanes in this direction
            active_speeds = [l.avg_speed_kmh for l in dir_lanes if (l.vis_cam1 + l.vis_cam2) > 0]
            dir_avg_speed = round(sum(active_speeds) / len(active_speeds), 2) if active_speeds else 0.0
            dir_flow_rate = round(c_cleared / w_sec, 3)
            dir_inflow_rate = round(c_pool["interval_inflow"] / w_sec, 3)

            # Max-Pressure metric: max_lane_demand * saturation_flow_per_sec
            dir_pressure = round(max_lane_demand * self.saturation_flow_per_sec, 3)

            by_direction[d] = {
                "motorcycles": {
                    "waiting_count": dir_vis_motos,
                    "inflow": m_pool["interval_inflow"],
                    "cleared": m_cleared,
                    "discharge_rate": round(m_cleared / w_sec, 2),
                    "visual_present": dir_vis_motos,
                },
                "cars_total": {
                    "waiting_count": dir_waiting_cars,
                    "max_lane_demand": max_lane_demand,
                    "inflow": c_pool["interval_inflow"],
                    "cleared": c_cleared,
                    "discharge_rate": round(c_cleared / w_sec, 2),
                    "inflow_rate": dir_inflow_rate,
                },
                "avg_speed_kmh": dir_avg_speed,
                "flow_rate": dir_flow_rate,
                "max_pressure": dir_pressure,
                "spillback_active": any(l.spillback for l in dir_lanes),
                "outgoing_full": bool(self.outgoing_full.get(d, False)),
                "is_approach_empty": (dir_waiting_cars == 0 and dir_vis_motos == 0),
            }

        by_lane: Dict[str, Any] = {}
        for lid, lst in self.lanes.items():
            if lst.spillback:
                spillback_lanes.append(lid)
            outflow_rate = round(lst.interval_outflow / w_sec, 3)
            inflow_rate = round(lst.interval_inflow / w_sec, 3)
            lane_pressure = round(lst.demand_count * self.saturation_flow_per_sec, 3)
            by_lane[lid] = {
                "direction": lst.direction,
                "demand_count": lst.demand_count,
                "waiting_cars": lst.demand_count,
                "vis_cam1": lst.vis_cam1,
                "vis_cam2": lst.vis_cam2,
                "cart_tokens": len(lst.cart),
                "is_frozen": lst.is_frozen,
                "spillback": lst.spillback,
                "override_capacity": lst.max_capacity_0_100m if lst.spillback else None,
                "avg_speed_kmh": lst.avg_speed_kmh,
                "flow_rate": outflow_rate,
                "inflow_rate": inflow_rate,
                "saturation_flow_rate": self.saturation_flow_per_sec,
                "max_pressure": lane_pressure,
                "inflow": lst.interval_inflow,
                "outflow": lst.interval_outflow,
                "visual_queued_cars": lst.visual_queued_cars,
                "visual_motorcycles": lst.vis_motorcycles_cam1,
                "handover_exact": lst.handover_exact_count,
                "handover_borrowed": lst.handover_borrowed_count,
                "expired_tokens": lst.expired_tokens_count,
                "is_lane_empty": (lst.demand_count == 0 and lst.vis_motorcycles_cam1 == 0),
            }

        return {
            "by_direction": by_direction,
            "by_lane": by_lane,
            "outgoingfull": {d: bool(self.outgoing_full.get(d, False)) for d in ("N", "E", "S", "W")},
            "motorcycle_count_cam1": sum(l.vis_motorcycles_cam1 for l in self.lanes.values()),
            "alarm_flags": {
                "emergency_spillback_override": len(spillback_lanes) > 0,
                "target_override_lanes": spillback_lanes,
                "target_override_lane": spillback_lanes[0] if spillback_lanes else None,
            },
        }

