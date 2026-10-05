"""
Traffic count payload builder and lane metrics manager.

Canonical contract: docs/CONTROLLER_CONTRACT.md (schemaVersion 2.0). Rich readers
must honour ``valid`` and ``unknownStateCount``. Existing controller-main wire
output is projected separately by MainControllerAdapter; this payload cannot
be sent directly to that receiver when a required reading is unknown.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from shapely.geometry import Polygon

SCHEMA_VERSION = "2.0"
LANE_STATES = ("queued", "moving", "unknown")
VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "three_wheeler")


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def classify_vehicle(
    class_identifier: int | str,
    class_names: dict[int, str] | None = None,
) -> str:
    """
    Map class ID or class name to controller vehicle category ('cars' or 'motorbike').

    Supports:
      - Custom domain models (e.g. Thai Traffic: 0: car, 1: motorcycle, 2: bus, 3: truck, 4: three_wheeler)
      - Standard COCO class IDs (1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck)
      - Class names:
          'motorcycle', 'motorbike', 'bicycle', 'bike' -> 'motorbike'
          'three_wheeler', 'tuktuk', 'car', 'bus', 'truck', 'van' -> 'cars'
    """
    if class_names and isinstance(class_identifier, int) and class_identifier in class_names:
        class_identifier = class_names[class_identifier]

    if isinstance(class_identifier, int):
        # Fallback for standard COCO integer classes when class_names dictionary is omitted
        if class_identifier in (1, 3):
            return "motorbike"
        return "cars"

    name = str(class_identifier).lower().strip().replace("-", "_").replace(" ", "_")
    if name in ("motorcycle", "motorbike", "bicycle", "bike"):
        return "motorbike"
    # Three-wheelers (tuk-tuks, saleng) occupy standard lane queues and are aggregated with 'cars'
    return "cars"


def normalize_class_name(class_identifier: int | str) -> str:
    """Raw detector class for the per-class breakdown; unknown names map to 'car'."""
    if isinstance(class_identifier, int):
        return {1: "motorcycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}.get(class_identifier, "car")
    name = str(class_identifier).lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {"motorbike": "motorcycle", "bicycle": "motorcycle", "bike": "motorcycle", "tuktuk": "three_wheeler"}
    name = aliases.get(name, name)
    return name if name in VEHICLE_CLASSES else "car"


class LaneMetricsManager:
    """
    Current lane occupancy for one camera.

    The runner rebuilds it on every evaluated frame: ``reset()`` then one
    ``register_vehicle()`` per track. A track is held in exactly one
    lane/category/state at a time.
    """

    def __init__(self, lane_config: dict[str, dict[str, Any]], camera_id: Optional[str] = None):
        """
        Args:
            lane_config: Dictionary mapping lane_id to lane config, e.g.:
                {
                    "N1": {"direction": "N", "polygon": Polygon(...)},
                    "N2": {"direction": "N", "polygon": Polygon(...)}
                }
            camera_id: Registered ID of the camera that measures these lanes.
        """
        self.camera_id = camera_id
        self.lanes: dict[str, dict[str, Any]] = {}
        self._tracks: dict[int, tuple[str, str, str, str]] = {}
        self._init_lanes(lane_config)

    def _init_lanes(self, lane_config: dict[str, dict[str, Any]]) -> None:
        for lane_id, cfg in lane_config.items():
            polygon = cfg.get("polygon")
            if polygon is not None and not isinstance(polygon, Polygon):
                polygon = Polygon(polygon)

            self.lanes[lane_id] = {
                "laneId": lane_id,
                "direction": cfg.get("direction", lane_id[0] if lane_id else "N"),
                "role": cfg.get("role", "queue"),
                "polygon": polygon,
                "vehicles": self._empty_vehicle_state(),
                "classes": {c: set() for c in VEHICLE_CLASSES},
            }

    def _empty_vehicle_state(self) -> dict[str, dict[str, set[int]]]:
        return {state: {"cars": set(), "motorbike": set()} for state in LANE_STATES}

    def register_vehicle(
        self,
        lane_id: str,
        track_id: int,
        vehicle_class: int | str,
        is_queued: bool = False,
        state: Optional[str] = None,
    ) -> None:
        """
        Register a vehicle in a lane for the current evaluation.

        Args:
            lane_id: Target lane ID (e.g. 'N1')
            track_id: Unique tracking ID for the vehicle
            vehicle_class: Model class ID (e.g. 2) or name (e.g. 'car')
            is_queued: Legacy flag; used only when ``state`` is not given
            state: 'queued', 'moving' or 'unknown'
        """
        if lane_id not in self.lanes:
            return

        tid = int(track_id)
        category = classify_vehicle(vehicle_class)  # 'cars' or 'motorbike'
        raw_class = normalize_class_name(vehicle_class)
        state_key = state if state in LANE_STATES else ("queued" if is_queued else "moving")

        # If track was previously registered in any lane/category/state, remove it first
        if tid in self._tracks:
            old_lane_id, old_category, old_state_key, old_class = self._tracks[tid]
            if old_lane_id in self.lanes:
                self.lanes[old_lane_id]["vehicles"][old_state_key][old_category].discard(tid)
                self.lanes[old_lane_id]["classes"][old_class].discard(tid)

        lane = self.lanes[lane_id]
        lane["vehicles"][state_key][category].add(tid)
        lane["classes"][raw_class].add(tid)
        self._tracks[tid] = (lane_id, category, state_key, raw_class)

    def lane_of(self, track_id: int) -> Optional[tuple[str, str]]:
        """(lane_id, state) the track currently contributes to, or None."""
        rec = self._tracks.get(int(track_id))
        return (rec[0], rec[2]) if rec else None

    def reset(self) -> None:
        """Clear current occupancy before rebuilding it from the latest evaluation."""
        self._tracks.clear()
        for lane in self.lanes.values():
            lane["vehicles"] = self._empty_vehicle_state()
            lane["classes"] = {c: set() for c in VEHICLE_CLASSES}

    def snapshot(self) -> list[dict[str, Any]]:
        """
        Current per-lane occupancy. Invariant:
        count == queuedCount + movingCount + unknownStateCount.
        """
        result = []
        for data in self.lanes.values():
            v = data["vehicles"]
            q_cars, q_bikes = len(v["queued"]["cars"]), len(v["queued"]["motorbike"])
            m_cars, m_bikes = len(v["moving"]["cars"]), len(v["moving"]["motorbike"])
            u_cars, u_bikes = len(v["unknown"]["cars"]), len(v["unknown"]["motorbike"])
            lane: dict[str, Any] = {
                "laneId": data["laneId"],
                "direction": data["direction"],
                "role": data["role"],
                "count": q_cars + q_bikes + m_cars + m_bikes + u_cars + u_bikes,
                "queuedCount": q_cars + q_bikes,
                "movingCount": m_cars + m_bikes,
                "unknownStateCount": u_cars + u_bikes,
                "vehicles": {
                    "queued": {"cars": q_cars, "motorbike": q_bikes},
                    "moving": {"cars": m_cars, "motorbike": m_bikes},
                    "unknown": {"cars": u_cars, "motorbike": u_bikes},
                },
                "classes": {c: len(ids) for c, ids in data["classes"].items()},
            }
            if self.camera_id is not None:
                lane["cameraId"] = self.camera_id
            result.append(lane)
        return result


def invalidate_lane(lane: dict[str, Any], reason: str) -> dict[str, Any]:
    """Marks a lane snapshot as unknown: counts become null, never zero."""
    out = dict(lane)
    for key in ("count", "queuedCount", "movingCount", "unknownStateCount"):
        out[key] = None
    out["vehicles"] = None
    out["classes"] = None
    out["valid"] = False
    out["invalidReason"] = reason
    return out


class PayloadBuilder:
    """
    Constructs the traffic-count message. Each builder instance is one publishing
    session: ``sessionId`` changes on restart and ``sequence`` increases by one per
    message, so receivers can drop duplicates and detect restarts.
    """

    def __init__(self, intersection_id: str, camera_id: str, session_id: Optional[str] = None):
        self.intersection_id = intersection_id
        self.camera_id = camera_id  # registered vision source ID (kept as cameraId for v1 readers)
        self.session_id = session_id or str(uuid.uuid4())
        self.sequence = 0

    def build(
        self,
        frame_idx: int,
        lanes_snapshot: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
        traffic_flow: dict[str, Any] | None = None,
        cameras: list[dict[str, Any]] | None = None,
        observed_at: float | None = None,
        published_at: float | None = None,
    ) -> dict[str, Any]:
        """
        Build the payload.

        ``timestamp`` is the observation time the lane data describes (the oldest
        observation among valid cameras), so the controller's freshness check
        measures the age of the traffic seen, not the age of the message.
        """
        import time as _time

        published = published_at if published_at is not None else _time.time()
        observed = observed_at if observed_at is not None else published
        self.sequence += 1

        merged_meta = {"frameId": f"frame_{frame_idx}"}
        if meta:
            merged_meta.update(meta)

        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "intersectionId": self.intersection_id,
            "cameraId": self.camera_id,
            "sourceId": self.camera_id,
            "sessionId": self.session_id,
            "sequence": self.sequence,
            "timestamp": iso_utc(observed),
            "observedAt": iso_utc(observed),
            "publishedAt": iso_utc(published),
            "meta": merged_meta,
            "lanes": [lane if "valid" in lane else {**lane, "valid": True, "invalidReason": None}
                      for lane in lanes_snapshot],
        }
        if cameras is not None:
            payload["cameras"] = cameras
        if traffic_flow:
            payload["traffic_flow"] = traffic_flow

        return payload

    def to_json(
        self,
        frame_idx: int,
        lanes_snapshot: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
        traffic_flow: dict[str, Any] | None = None,
    ) -> str:
        """Build and serialize payload to JSON string."""
        return json.dumps(self.build(frame_idx, lanes_snapshot, meta, traffic_flow))
