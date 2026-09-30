"""Offline audit probes. No camera connections or MQTT publishing."""
import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity
import run_multi_camera as runner
from algorithm.byetrack import ByteTrack
from trt_pipeline.payload import LaneMetricsManager
from trt_pipeline.stream import StreamBufferWorker
from trt_pipeline.gates import VirtualGate, GateFlowManager

runner.np, runner.shapely, runner.Point = np, shapely, Point
evidence = {}

files = [ROOT / 'main.py', ROOT / 'run_multi_camera.py', ROOT / 'export_trt.py']
for folder in ('algorithm', 'trt_pipeline', 'tools', 'tests'):
    files.extend((ROOT / folder).rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
evidence['syntax_files_passed'] = len(files)

lanes = {'N1': {'direction': 'N', 'polygon': Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])}}
m = LaneMetricsManager(lanes)
m.register_vehicle('N1', 1, 'car', True)
m.register_vehicle('N1', 1, 'motorcycle', True)
evidence['same_id_changed_category'] = m.snapshot()[0]
m.reset()
m.register_vehicle('N1', 1, 'car', True)
# Next observation is an empty frame; the production runner returns without removal.
p = runner.BatchedCameraPipeline.__new__(runner.BatchedCameraPipeline)
p.metrics_managers = [m]
p._evaluate_vectorized_lanes(0, np.empty((0, 6)), 2)
evidence['departed_vehicle_remains_until_publish_reset'] = m.snapshot()[0]

p.track_histories = [{}]
p.queue_speed_thresholds = [2.0]
speed_cases = {}
for fps in (10, 30):
    p.track_histories = [{}]
    for frame in range(15):
        queued = p._is_queued(0, 1, (30 * frame / fps, 0), frame)
    speed_cases[str(fps)] = queued
evidence['same_30_pixels_per_second_queued_by_fps'] = speed_cases

p.lane_configs = [lanes]
p.metrics_managers = [LaneMetricsManager(lanes)]
p.class_names = {0: 'car'}
p.default_car_cls = 0
p.track_histories = [{}]
p._evaluate_vectorized_lanes(0, np.array([[20, -50, 40, 30, 1, 0]]), 1)
evidence['bottom_center_inside_but_centroid_outside_count'] = p.metrics_managers[0].snapshot()[0]['count']

worker = StreamBufferWorker('offline', 'unused')
worker.queue.put((1, np.array([1])))
worker.queue.put((2, np.array([2])))
evidence['get_latest_returns_timestamp'] = worker.get_frame()[0]

p = runner.BatchedCameraPipeline.__new__(runner.BatchedCameraPipeline)
p.start = lambda: setattr(p, 'running', True)
p.stop = lambda: setattr(p, 'running', False)
p.num_streams, p.skip_frames, p.is_file_mode = 2, 1, False
p.publisher = Mock()
def missing_frame(timeout):
    p.running = False
    return None
p.stream_workers = [SimpleNamespace(get_frame=lambda timeout: (time.perf_counter(), np.zeros((4,4,3)))), SimpleNamespace(get_frame=missing_frame)]
p.run()
evidence['one_missing_camera_publish_calls'] = p.publisher.publish.call_count

tracker = ByteTrack()
matches, ua, ub = tracker._linear_assignment(np.array([[0.1, 0.6], [0.6, 0.8]]), 0.7)
evidence['assignment_feasible_two_matches_actual'] = matches.tolist()

gm = GateFlowManager()
gm.add_gate(VirtualGate('GATE_N_STOP', 0, (0, 5), (20, 5), direction_vec=(0, 1)))
gm.update_tracks(0, np.array([[5,-3,7,0,1,0]]))
gm.update_tracks(0, np.array([[30,-3,32,0,2,0]]))
gm.update_tracks(0, np.array([[5,7,7,10,1,0]]))
evidence['gate_crossing_after_one_missing_observation'] = gm.gates['GATE_N_STOP'].count

configs = []
frames = []
for path in sorted((ROOT / 'config').glob('config_*.json')):
    cfg = json.loads(path.read_text())
    cap = cv2.VideoCapture(str(ROOT / cfg['video']['path']))
    ok, frame = cap.read()
    row = {'config': path.name, 'video_read': ok, 'fps': cap.get(cv2.CAP_PROP_FPS), 'shape': list(frame.shape) if ok else None, 'lanes': {}, 'gates': [g['gate_id'] for g in cfg.get('gates', [])]}
    cap.release()
    polys = {}
    for lane_id, info in cfg['lane_metrics']['lanes'].items():
        poly = Polygon(info['polygon'])
        polys[lane_id] = poly
        row['lanes'][lane_id] = {'valid': poly.is_valid, 'reason': explain_validity(poly), 'area': poly.area}
    row['overlaps'] = []
    ids = list(polys)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if polys[a].is_valid and polys[b].is_valid:
                area = polys[a].intersection(polys[b]).area
                if area > 0:
                    row['overlaps'].append({'lanes': [a,b], 'area': area})
    configs.append(row)
    if ok:
        frames.append(frame)
evidence['config_checks'] = configs

import export_trt
try:
    export_trt.main()
except Exception as exc:
    evidence['export_cli_error'] = f'{type(exc).__name__}: {exc}'

import torch
import ultralytics
from ultralytics import YOLO
model = YOLO(str(ROOT / 'models/yolo26s_thai_traffic.pt'))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
evidence['model'] = {'names': model.names, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__, 'device': device}
results = model(frames, device=device, imgsz=640, conf=0.20, verbose=False)
evidence['model']['five_frame_smoke'] = [{'detections': len(r.boxes), 'finite_boxes': bool(torch.isfinite(r.boxes.xyxy).all()), 'speed_ms': r.speed} for r in results]

out = Path(__file__).with_name('evidence.json')
out.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps(evidence, indent=2))
