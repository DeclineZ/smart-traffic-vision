import torch
from trt_model import TRTModel
from video_stream import VideoStream, AsyncImageSaver, capture_one_frame, letterbox, is_network_stream, prepare_video_source
import numpy as np
import cv2 as cv
import os
import time
import json
from tools import (
    get_logger, cleanup, initial_config, initial_lane_data, to_original_coords,
    parse_zones, side_of_line, save_lane_data, RoadDensityEstimator
)
from shapely.geometry import Point, Polygon
from shapely import contains
from payload import LaneMetricsManager, PayloadBuilder
from publisher import WSPublisher

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

def _get_gpu_util() -> dict:
    if not _GPU_AVAILABLE:
        return {}
    try:
        return {
            f"gpu:{i}": pynvml.nvmlDeviceGetUtilizationRates(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            ).gpu
            for i in range(pynvml.nvmlDeviceGetCount())
        }
    except Exception:
        return {}


class TrafficTracker:
    def __init__(self, config_path):
        self.logger = get_logger("TrafficTracker")
        self.config = initial_config(config_path)

        device_name = self.config["model"].get("device", "cuda:0")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")

        self.frame_stride = max(1, int(self.config["video"].get("skip", 1)))
        self.max_frames   = self.config["video"].get("max_frames")  # None = no limit
        self.video_source = prepare_video_source(self.config["video"]["path"])
        self.logger.info(f"Video source: {self.video_source}")

        model_config = self.config["model"]
        self.model = TRTModel(
            engine_path=model_config["engine_path"],
            input_shape=tuple(model_config.get("input_shape", [1, 3, 640, 640])),
            device=self.device,
        )

        tracker_config = self.config["tracker"]
        tracker_type   = tracker_config["type"]
        tracker_params = tracker_config.get("params", {})

        from boxmot import ByteTrack, BotSort, DeepOcSort
        tracker_map = {"ByteTrack": ByteTrack, "BotSort": BotSort, "DeepOCSort": DeepOcSort}
        if tracker_type not in tracker_map:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        self.tracker = tracker_map[tracker_type](**tracker_params)

        self.dict_class = {int(k): v for k, v in self.config["classes"]["dict_class"].items()}

        self.save_crop        = self.config["output"].get("save_crop", False)
        self.data_output_path = self.config["output"].get(
            "data_output_path",
            os.path.join(self.config["output"]["base_dir"], "data.json")
        )

        self.tracking_zone = parse_zones(self.config["tracking"]["zones"])
        self.lane_data     = initial_lane_data(self.tracking_zone, self.dict_class)
        self.track_memory  = {}

        metrics_config  = self.config["lane_metrics"]
        camera_info     = self.config["camera_info"]
        self.publish_interval_frames = metrics_config.get("publish_interval_frames", 50)
        self.payload_builder = PayloadBuilder(
            intersection_id=camera_info["intersection_id"],
            camera_id=camera_info["camera_id"]
        )

        self.lane_config = {
            lane_id: {
                "direction": lane_info.get("direction", "N"),
                "polygon":   Polygon(lane_info["polygon"])
            }
            for lane_id, lane_info in metrics_config.get("lanes", {}).items()
        }
        self.metrics = LaneMetricsManager(self.lane_config)

        density_config        = self.config["density"]
        self.density_estimator = None
        if density_config.get("enabled", False):
            zone_coords     = density_config.get("zone_coordinates")
            base_image_path = density_config.get("base_road_image")
            if zone_coords and base_image_path and os.path.exists(base_image_path):
                self.density_estimator = RoadDensityEstimator(
                    base_image=cv.imread(base_image_path),
                    coordinates=zone_coords
                )
            else:
                self.logger.warning(f"Base road image not found: {base_image_path}")

        ws_config         = self.config.get("websocket", {})
        self.ws_publisher = None
        if ws_config.get("enabled", False):
            self.ws_publisher = WSPublisher(
                uri=ws_config.get("uri", "redis://localhost:6379"),
                topic=ws_config.get("topic", "traffic"),
            )
            self.ws_publisher.start()
            self.logger.info(f"WSPublisher → topic: {ws_config.get('topic', 'traffic')}")

        self._vehicle_counts: dict[str, int] = {v: 0 for v in self.dict_class.values()}

        self.image_saver = AsyncImageSaver()
        self.save_dir    = {}
        base_output      = self.config["output"]["base_dir"]

        for lane_name in self.lane_data.keys():
            for cls in self.dict_class.values():
                for lane_id in self.lane_config.keys():
                    path = os.path.join(base_output, lane_name, cls, lane_id)
                    os.makedirs(path, exist_ok=True)
                    self.save_dir[(lane_name, cls, lane_id)] = path

        self.logger.info("TrafficTracker initialized successfully")

    def run(self):
        processing_config = self.config["processing"]
        cv.setNumThreads(processing_config.get("opencv_threads", 0))
        cv.ocl.setUseOpenCL(processing_config.get("use_opencl", False))

        if is_network_stream(self.video_source):
            self.logger.info("Testing network stream connection...")
            if capture_one_frame(self.video_source) is None:
                self.logger.error("Failed to connect to network stream. Aborting.")
                return
            self.logger.info("Network stream connection successful!")

        stream = VideoStream(
            video_path=self.video_source,
            skip=self.frame_stride,
            queue_size=processing_config.get("video_queue_size", 2),
        )

        total_start = time.perf_counter()
        frame_count = 0

        while True:
            item = stream.read()
            if item is None:
                break

            frame_idx, frame_bgr = item
            frame_count += 1

            if frame_idx % 10 == 0:
                print(f"\rframe: {frame_idx}", end="", flush=True)

            if is_network_stream(self.video_source) and frame_idx % 100 == 0:
                self.logger.info(f"Processed {frame_idx} frames from network stream")

            if self.max_frames and frame_idx > self.max_frames:
                self.logger.info(f"Reached max_frames limit: {self.max_frames}")
                break

            img_rgb   = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            img_lb, ratio, (dw, dh) = letterbox(img_rgb, new_shape=(640, 640), auto=False)
            img_chw   = np.ascontiguousarray(img_lb.transpose(2, 0, 1), dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)

            _, outputs = self.model.infer(input_tensor)
            num     = int(outputs["num_dets"][0])
            boxes   = outputs["det_boxes"][0][:num].cpu().numpy()
            scores  = outputs["det_scores"][0][:num].cpu().numpy()
            classes = outputs["det_classes"][0][:num].cpu().numpy()
            dets    = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=-1)

            results = self.tracker.update(dets, frame_bgr)

            h, w = frame_bgr.shape[:2]
            for x1, y1, x2, y2, track_id, conf, class_id, _ in results:
                class_id = int(class_id)
                track_id = int(track_id)

                if class_id not in self.dict_class:
                    continue

                cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
                cxo, cyo = to_original_coords(cx, cy, dw, dh, ratio)
                x1o, y1o = to_original_coords(x1, y1, dw, dh, ratio)
                x2o, y2o = to_original_coords(x2, y2, dw, dh, ratio)

                x1c = int(max(0, min(w, x1o)))
                y1c = int(max(0, min(h, y1o)))
                x2c = int(max(0, min(w, x2o)))
                y2c = int(max(0, min(h, y2o)))

                if x2c <= x1c or y2c <= y1c:
                    continue

                centroid      = Point(cxo, cyo)
                current_point = (cxo, cyo)

                if track_id not in self.track_memory:
                    self.track_memory[track_id] = current_point
                    continue

                prev_point = self.track_memory[track_id]

                for lane_name, lane in self.lane_data.items():
                    if track_id in lane["cross_ids"]:
                        continue

                    prev_side = side_of_line(prev_point, lane["line"][0], lane["line"][1])
                    curr_side = side_of_line(current_point, lane["line"][0], lane["line"][1])

                    if contains(lane["polygon"], centroid) and prev_side * curr_side < 0 and curr_side < 0:
                        lane["cross_ids"].add(track_id)
                        lane["count_cls"][class_id] += 1
                        lane["cross_obj"].append({
                            "id":       track_id,
                            "frame":    frame_idx,
                            "class_id": class_id,
                            "bbox":     (x1c, y1c, x2c, y2c),
                        })
                        self._vehicle_counts[self.dict_class[class_id]] += 1

                        for lane_id, lane_cfg in self.metrics.lanes.items():
                            if contains(lane_cfg["polygon"], centroid):
                                vehicle_type = "motorbike" if class_id == 3 else "car"
                                self.metrics.update_vehicle(lane_id, vehicle_type, conf, track_id) # add track_id to moving list

                                if self.save_crop:
                                    crop = frame_bgr[y1c:y2c, x1c:x2c]
                                    save_path = os.path.join(
                                        self.save_dir[(lane_name, self.dict_class[class_id], lane_id)],
                                        f"frame_{frame_idx}_id_{track_id}.jpg"
                                    )
                                    self.image_saver.save(save_path, crop)
                                break

                self.track_memory[track_id] = current_point

            if frame_idx % self.publish_interval_frames == 0:
                density        = self.density_estimator.calculate(frame_bgr) if self.density_estimator else 0
                lanes_snapshot = self.metrics.snapshot(density)
                self.metrics.reset()

                if self.ws_publisher:
                    self.ws_publisher.publish({
                        "frame":          frame_idx,
                        "vehicle_counts": dict(self._vehicle_counts),
                        "density":        density,
                        "gpu":            _get_gpu_util(),
                        "timestamp":      time.time(),
                    })

                self._vehicle_counts = {v: 0 for v in self.dict_class.values()}

        total_time = time.perf_counter() - total_start
        self.logger.info(
            f"Total time: {total_time:.2f}s | Frames: {frame_count}"
            + (f" | FPS: {frame_count / total_time:.2f}" if frame_count > 0 else "")
        )

        cleanup()
        self.image_saver.stop()

        if self.ws_publisher:
            self.ws_publisher.stop()

        save_lane_data(self.lane_data, os.path.join(self.config["output"]["base_dir"], "lane_data.json"))
        self.logger.info("Outputs saved successfully.")

if __name__ == "__main__":
    TrafficTracker(config_path="./config/config_south_1.json").run()