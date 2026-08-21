"""
Traffic count payload builder and lane metrics manager.
Conforms to the schema expected by the smart-traffic-controller-js decision engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from shapely.geometry import Polygon


def classify_vehicle(class_identifier: int | str) -> str:
    """
    Map class ID or class name to controller vehicle category ('cars' or 'motorbike').

    Standard mappings:
      - 1 (bicycle), 3 (motorcycle), 'motorbike', 'motorcycle', 'bicycle', 'bike' -> 'motorbike'
      - 2 (car), 5 (bus), 7 (truck), 'car', 'bus', 'truck', 'van' -> 'cars'
    """
    if isinstance(class_identifier, int):
        if class_identifier in (1, 3):
            return "motorbike"
        return "cars"

    name = str(class_identifier).lower().strip()
    if name in ("motorcycle", "motorbike", "bicycle", "bike"):
        return "motorbike"
    return "cars"


class LaneMetricsManager:
    """
    Tracks and aggregates vehicle counts per lane for a given aggregation interval.
    Maintains queued vs moving vehicle state for 'cars' and 'motorbike'.
    """

    def __init__(self, lane_config: dict[str, dict[str, Any]]):
        """
        Args:
            lane_config: Dictionary mapping lane_id to lane config, e.g.:
                {
                    "N1": {"direction": "N", "polygon": Polygon(...)},
                    "N2": {"direction": "N", "polygon": Polygon(...)}
                }
        """
        self.lanes: dict[str, dict[str, Any]] = {}
        self._init_lanes(lane_config)

    def _init_lanes(self, lane_config: dict[str, dict[str, Any]]) -> None:
        for lane_id, cfg in lane_config.items():
            polygon = cfg.get("polygon")
            if polygon is not None and not isinstance(polygon, Polygon):
                polygon = Polygon(polygon)

            self.lanes[lane_id] = {
                "laneId": lane_id,
                "direction": cfg.get("direction", lane_id[0] if lane_id else "N"),
                "polygon": polygon,
                "vehicles": self._empty_vehicle_state(),
            }

    def _empty_vehicle_state(self) -> dict[str, dict[str, set[int]]]:
        return {
            "queued": {
                "cars": set(),
                "motorbike": set(),
            },
            "moving": {
                "cars": set(),
                "motorbike": set(),
            },
        }

    def register_vehicle(
        self,
        lane_id: str,
        track_id: int,
        vehicle_class: int | str,
        is_queued: bool = False,
    ) -> None:
        """
        Register a vehicle detection in a lane for the current window.

        Args:
            lane_id: Target lane ID (e.g. 'N1')
            track_id: Unique tracking ID for the vehicle
            vehicle_class: Model class ID (e.g. 2) or name (e.g. 'car')
            is_queued: True if the vehicle is stationary/queued, False if moving
        """
        if lane_id not in self.lanes:
            return

        category = classify_vehicle(vehicle_class)  # 'cars' or 'motorbike'
        state_key = "queued" if is_queued else "moving"
        other_key = "moving" if is_queued else "queued"

        vehicles = self.lanes[lane_id]["vehicles"]

        # If it was previously marked in the opposite state in this interval,
        # update it to the latest detected state
        if track_id in vehicles[other_key][category]:
            vehicles[other_key][category].discard(track_id)

        vehicles[state_key][category].add(int(track_id))

    def reset(self) -> None:
        """Reset internal vehicle counters for the next interval."""
        for lane_id in self.lanes:
            self.lanes[lane_id]["vehicles"] = self._empty_vehicle_state()

    def snapshot(self) -> list[dict[str, Any]]:
        """
        Generate a snapshot of lane metrics formatted for the controller.

        Returns:
            list[dict]: Array of lane objects matching controller schema:
                [
                    {
                        "laneId": "N1",
                        "direction": "N",
                        "vehicles": {
                            "queued": {"cars": 2, "motorbike": 1},
                            "moving": {"cars": 0, "motorbike": 0}
                        }
                    },
                    ...
                ]
        """
        result = []
        for lane_id, data in self.lanes.items():
            queued = data["vehicles"]["queued"]
            moving = data["vehicles"]["moving"]

            q_cars = len(queued["cars"])
            q_bikes = len(queued["motorbike"])
            m_cars = len(moving["cars"])
            m_bikes = len(moving["motorbike"])
            total_count = q_cars + q_bikes + m_cars + m_bikes

            result.append({
                "laneId": data["laneId"],
                "direction": data["direction"],
                "count": total_count,
                "queuedCount": q_cars + q_bikes,
                "movingCount": m_cars + m_bikes,
                "vehicles": {
                    "queued": {
                        "cars": q_cars,
                        "motorbike": q_bikes,
                    },
                    "moving": {
                        "cars": m_cars,
                        "motorbike": m_bikes,
                    },
                },
            })
        return result


class PayloadBuilder:
    """
    Constructs standardized JSON payloads for MQTT transmission to the controller.
    """

    def __init__(self, intersection_id: str, camera_id: str):
        self.intersection_id = intersection_id
        self.camera_id = camera_id

    def build(
        self,
        frame_idx: int,
        lanes_snapshot: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build full payload matching controller's basicValidate expectations:
          - intersectionId (str)
          - cameraId (str)
          - timestamp (ISO 8601 UTC with ms)
          - meta (dict with frameId)
          - lanes (list of lane objects)
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        merged_meta = {"frameId": f"frame_{frame_idx}"}
        if meta:
            merged_meta.update(meta)

        return {
            "intersectionId": self.intersection_id,
            "cameraId": self.camera_id,
            "timestamp": ts,
            "meta": merged_meta,
            "lanes": lanes_snapshot,
        }

    def to_json(
        self,
        frame_idx: int,
        lanes_snapshot: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Build and serialize payload to JSON string."""
        return json.dumps(self.build(frame_idx, lanes_snapshot, meta))