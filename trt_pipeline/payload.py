import json
from datetime import datetime
from zoneinfo import ZoneInfo

# lanes structure
# {
#     "laneId": "N1",
#     "direction": "N",
#     "vehicles": {
#     "density": {
#         // pressure at current frame
#         scores: 0.77,    // [0, 1]
#         level: 4         // [1, 2, 3, 4, 5]
#     },
#     "confident": {
#         // every max frames (50 frames -> 2 seconds)
#         "cars": 0.8,     // [0, 1] mean
#         "motorbike": 0.7 // [0, 1] mean
#     }
#     "moving": {
#         // every max frames (50 frames -> 2 seconds)
#         "cars": 0,       // count
#         "motorbike": 0   // count
#     }
#     }
# },

class LaneMetricsManager:
    def __init__(self, lane_config):
        self.lanes = {}
        self._init_lanes(lane_config)

    def _init_lanes(self, lane_config):
        """
        Initialize full internal lane structure.
        """
        for lane_id, cfg in lane_config.items():
            self.lanes[lane_id] = {
                "direction": cfg["direction"],
                "polygon": cfg["polygon"],
                "vehicles": self._empty_vehicle_state()
            }

    def _empty_vehicle_state(self):
        return {
            "density": {
                "score": 0.0,
                "level": 0
            },
            "confident": {
                "car": [],
                "motorbike": []
            },
            "moving": {
                "car": [],       # list of track ids seen this interval
                "motorbike": []
            },
        }

    def update_vehicle(self, lane_id, vehicle_type, confidence, vehicle_id):
        self.lanes[lane_id]["vehicles"]["confident"][vehicle_type].append(float(confidence))
        self.lanes[lane_id]["vehicles"]["moving"][vehicle_type].append(int(vehicle_id))

    def reset(self):
        for lane_id in self.lanes:
            self.lanes[lane_id]["vehicles"] = self._empty_vehicle_state()

    def snapshot(self, density):
        result = {}

        for lane_id, data in self.lanes.items():
            vehicles = data["vehicles"]

            car_conf = vehicles["confident"]["car"]
            bike_conf = vehicles["confident"]["motorbike"]

            avg_car = sum(car_conf) / len(car_conf) if car_conf else 0.0
            avg_bike = sum(bike_conf) / len(bike_conf) if bike_conf else 0.0

            if 0 <= density < 0.2: level = 1
            elif 0.2 <= density < 0.4: level = 2
            elif 0.4 <= density < 0.6: level = 3
            elif 0.6 <= density < 0.8: level = 4
            else: level = 5

            car_ids = vehicles["moving"]["car"]
            bike_ids = vehicles["moving"]["motorbike"]

            result[lane_id] = {
                "direction": data["direction"],
                "vehicles": {
                    "density": {
                        "score": round(density, 3),
                        "level": level,
                    },
                    "confident": {
                        "car": round(avg_car, 3),
                        "motorbike": round(avg_bike, 3),
                    },
                    "moving": {
                        "car": {
                            "count": len(car_ids),
                            "ids": car_ids,
                        },
                        "motorbike": {
                            "count": len(bike_ids),
                            "ids": bike_ids,
                        },
                    }
                }
            }

        return result

class PayloadBuilder:
    def __init__(self, intersection_id, camera_id, timezone="Asia/Bangkok"):
        self.intersection_id = intersection_id
        self.camera_id = camera_id
        self.timezone = timezone

    def build(self, frame_idx, lanes_snapshot):
        return {
            "intersectionId": self.intersection_id,
            "cameraId": self.camera_id,
            "timestamp": datetime.now(
                ZoneInfo(self.timezone)
            ).isoformat(timespec="milliseconds"),
            "meta": {
                "frameId": f"frame_{frame_idx}"
            },
            "lanes": lanes_snapshot
        }