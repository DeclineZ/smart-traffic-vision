"""Project vision measurements into the existing smart-traffic-sys/main contract.

Main reads lane.count as its traffic input and cannot represent an unknown lane.
Only complete, fresh stop-line occupancy snapshots are eligible for delivery.
Suppressing an incomplete snapshot lets main's existing freshness check expire;
it must never receive a missing/unknown lane as a measured zero.
"""

from datetime import datetime
import json
import math

from .payload import iso_utc


class MainControllerAdapter:
    def __init__(self, config_path, intersection_id, lane_configs, pub_interval):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict) or config.get("intersection_id") != intersection_id:
            raise ValueError("Controller profile must name the selected intersection")
        self.camera_id = config.get("camera_id")
        self.intersection_id = intersection_id
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("Controller profile requires its existing registered counting camera_id")
        self.lanes = config.get("expected_lanes")
        if (not isinstance(self.lanes, dict) or not self.lanes
                or any(not isinstance(lid, str) or not lid or direction not in ("N", "E", "S", "W")
                       for lid, direction in self.lanes.items())):
            raise ValueError("Controller profile requires expected_lanes as laneId -> direction")
        self.max_age_ms = config.get("max_observation_age_ms", 750)
        self.freshness_ms = config.get("controller_freshness_ms", 2000)
        for value in (self.max_age_ms, self.freshness_ms, pub_interval):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Controller age limits and publication interval must be positive and finite")
        if self.max_age_ms + pub_interval * 1000 >= self.freshness_ms:
            raise ValueError("Controller freshness budget needs room for source age, publication interval and delivery")
        actual = {lid: info["direction"] for lanes in lane_configs for lid, info in lanes.items()
                  if info.get("role", "queue") == "queue"}
        if actual != self.lanes:
            raise ValueError(f"Stop-line calibration must match controller profile: missing={sorted(set(self.lanes) - set(actual))}, "
                             f"extra={sorted(set(actual) - set(self.lanes))}, "
                             f"wrong_direction={sorted(lid for lid in actual.keys() & self.lanes.keys() if actual[lid] != self.lanes[lid])}")
        self.config_path = config_path
        self.blockers = []

    def build(self, payload, now_wall):
        """Return the wire payload, or None when any required reading is unknown."""
        self.blockers = []
        if payload.get("intersectionId") != self.intersection_id:
            self.blockers.append("wrong_intersection")
            return None
        by_id = {}
        for lane in payload["lanes"]:
            lid = lane["laneId"]
            if lid in by_id:
                self.blockers.append(f"duplicate_lane:{lid}")
            if lid not in self.lanes and lane.get("role", "queue") != "upstream":
                self.blockers.append(f"unexpected_stopline_lane:{lid}")
            by_id[lid] = lane
        lanes, observations = [], []
        for lid, direction in self.lanes.items():
            lane = by_id.get(lid)
            if lane is None:
                self.blockers.append(f"missing_lane:{lid}")
                continue
            if (lane.get("direction") != direction or lane.get("role", "queue") != "queue"
                    or not lane.get("valid", False)):
                self.blockers.append(f"{lid}:{lane.get('invalidReason') or 'invalid_lane'}")
                continue
            count = lane.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                self.blockers.append(f"{lid}:invalid_count")
                continue
            try:
                observed = datetime.fromisoformat(lane["observedAt"].replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    raise ValueError("observation time needs a timezone")
                observed = observed.timestamp()
            except (KeyError, ValueError, TypeError, AttributeError):
                self.blockers.append(f"{lid}:invalid_timestamp")
                continue
            age_ms = (now_wall - observed) * 1000
            if age_ms < 0 or age_ms > self.max_age_ms:
                self.blockers.append(f"{lid}:observation_clock_or_age")
                continue
            observations.append(observed)
            lanes.append({"laneId": lid, "direction": direction, "count": count})
        if self.blockers:
            return None
        return {
            "intersectionId": payload["intersectionId"],
            "cameraId": self.camera_id,
            "timestamp": iso_utc(min(observations)),
            "meta": {
                "frameId": payload["meta"]["frameId"],
                "countKind": "occupancy",
                "visionSourceId": payload["sourceId"],
                "visionSessionId": payload["sessionId"],
                "visionSequence": payload["sequence"],
                "physicalCameraIds": sorted({by_id[lid]["cameraId"] for lid in self.lanes}),
            },
            "lanes": lanes,
        }
