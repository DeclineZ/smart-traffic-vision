"""GPU adapter, isolated measurement worker and optional NVML telemetry."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Protocol

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT

from bench_core import CLASSES, read_json, save_json, save_csv, sha256, stats, completed_clip
from bench_prefetch import FrameReader


def environment():
    packages = {}
    for name in ('torch', 'ultralytics', 'opencv-python', 'tensorrt', 'tensorrt_cu12', 'numpy', 'psutil', 'nvidia-ml-py'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                cpu = winreg.QueryValueEx(k, 'ProcessorNameString')[0]
        except OSError:
            pass
    try:
        smi = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        smi = None
    return dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                cpu=cpu, logical_cpus=psutil.cpu_count(), ram_GiB=psutil.virtual_memory().total / 2**30,
                os=platform.platform(), python=platform.python_version(), packages=packages,
                pytorch_cuda_runtime=torch.version.cuda, nvidia_smi=smi,
                cuda_note='nvidia-smi CUDA means driver capability, not the PyTorch runtime.')


class Telemetry:
    """Sample outside latency accounting. Device values include all desktop processes."""
    def __init__(self, index):
        self.process = psutil.Process()
        self.nv = self.handle = None
        self.unavailable = None
        try:
            import pynvml
            pynvml.nvmlInit()
            # Match the selected CUDA device by UUID; do not assume NVML/CUDA indices agree.
            props = torch.cuda.get_device_properties(index)
            uuid = str(props.uuid)
            if not uuid.startswith('GPU-'):
                uuid = 'GPU-' + uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            self.nv = pynvml
        except Exception as e:
            self.unavailable = f'NVML unavailable: {type(e).__name__}: {e}'
        self.rows = []
        self.stop_event = threading.Event()

    def optional(self, name, *args):
        try:
            return getattr(self.nv, name)(self.handle, *args)
        except Exception:
            return None

    def sample(self):
        r = dict(monotonic_s=time.perf_counter(), process_cpu_percent=self.process.cpu_percent(),
                 process_ram_MiB=self.process.memory_info().rss / 2**20,
                 system_cpu_percent=psutil.cpu_percent(), system_ram_percent=psutil.virtual_memory().percent,
                 device_vram_MiB=None, process_vram_MiB=None, gpu_percent=None,
                 temperature_C=None, graphics_clock_MHz=None, power_W=None)
        if self.nv:
            memory = self.optional('nvmlDeviceGetMemoryInfo')
            util = self.optional('nvmlDeviceGetUtilizationRates')
            r['device_vram_MiB'] = memory.used / 2**20 if memory else None
            r['gpu_percent'] = util.gpu if util else None
            r['temperature_C'] = self.optional('nvmlDeviceGetTemperature', 0)
            r['graphics_clock_MHz'] = self.optional('nvmlDeviceGetClockInfo', 0)
            power = self.optional('nvmlDeviceGetPowerUsage')
            r['power_W'] = power / 1000 if power is not None else None
            procs = self.optional('nvmlDeviceGetComputeRunningProcesses')
            if procs is not None:
                for p in procs:
                    if p.pid == os.getpid() and isinstance(p.usedGpuMemory, int) and p.usedGpuMemory < 2**60:
                        r['process_vram_MiB'] = p.usedGpuMemory / 2**20
        return r

    def start(self, interval):
        self.rows = [self.sample()]
        self.stop_event.clear()
        def loop():
            while not self.stop_event.wait(interval):
                self.rows.append(self.sample())
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.rows.append(self.sample())
        result = {}
        for key in self.rows[0]:
            if key == 'monotonic_s':
                continue
            values = [r[key] for r in self.rows[1:] if r[key] is not None]
            result[key + '_mean'] = float(np.mean(values)) if values else None
            result[key + '_peak'] = float(max(values)) if values else None
        return result


class DetectionAdapter(Protocol):
    info: dict
    def predict(self, frame: np.ndarray, conf: float | None = None) -> tuple[np.ndarray, dict]: ...


def engine_metadata(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        if not 0 < n < 1_000_000:
            raise ValueError('Engine has no Ultralytics metadata; a custom adapter is required')
        try:
            return json.loads(f.read(n).decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError('Unrecognized engine metadata/output schema; custom adapter required') from e


class UltralyticsAdapter:
    def __init__(self, artifact, settings):
        self.a, self.s = artifact, settings
        self.device = torch.device(settings['device'])
        self.engine = Path(artifact['path']).suffix == '.engine'
        meta = engine_metadata(artifact['path']) if self.engine else {}
        export = meta.get('args', {})
        if meta and (meta.get('task') != 'detect' or meta.get('imgsz') != [640, 640]):
            raise ValueError('Engine must be a 640x640 detection export')
        embedded = bool(export.get('nms', artifact.get('export', {}).get('nms', False)))
        if self.engine and 'quantize' in export:
            quantize = export['quantize']
            precision = {None: 'fp32', 32: 'fp32', 16: 'fp16', 8: 'int8'}.get(quantize)
            if artifact['precision'] != precision:
                raise ValueError(f'Precision mismatch: config {artifact["precision"]}, export quantize={quantize}')
        if self.engine and 'half' in export and 'int8' in export:
            precision = 'int8' if export['int8'] else 'fp16' if export['half'] else 'fp32'
            if artifact['precision'] != precision:
                raise ValueError(f'Precision mismatch: config {artifact["precision"]}, export {precision}')
        if self.engine and not export.get('dynamic', False) and meta.get('batch') != 1:
            raise ValueError(f'Fixed engine batch {meta.get("batch")} cannot benchmark batch 1')
        self.model = YOLO(artifact['path'], task='detect')
        if self.model.task != 'detect':
            raise ValueError('Only axis-aligned detection models are supported')
        params = (sum(p.numel() for p in self.model.model.parameters()) if not self.engine
                  else artifact.get('parameters', meta.get('parameters')))
        self.info = dict(id=artifact['id'], name=artifact['name'], version=artifact['version'],
                         format='engine' if self.engine else 'pt', precision=artifact['precision'],
                         pair_id=artifact.get('pair_id'), parameters=params,
                         parameters_source='checkpoint before fusion' if not self.engine else 'manifest' if params else None,
                         file_MiB=Path(artifact['path']).stat().st_size / 2**20,
                         sha256=sha256(artifact['path']), export_metadata=meta,
                         math_mode='PyTorch TF32 disabled' if not self.engine else 'Compiled engine tactics; TF32 policy not recoverable from export metadata',
                         embedded_nms=embedded, comparable=not embedded,
                         limitation='Embedded NMS cannot be retuned/reversed; separate cohort' if embedded else None)
        self.kw = dict(imgsz=640, rect=False, batch=1, device=settings['device'],
                       half=artifact['precision'] == 'fp16', conf=settings['conf'], iou=settings['iou'],
                       classes=list(artifact['classes'].values()), max_det=settings['max_det'],
                       agnostic_nms=False, augment=False, verbose=False, save=False, show=False)
        if 'quantize' in DEFAULT_CFG_DICT:
            self.kw.pop('half')
            self.kw['quantize'] = 16 if artifact['precision'] == 'fp16' else None
            # Modern runtime defaults to external NMS even for YOLO26 checkpoints.
            # Preserve the native one-to-one head where available, matching exports.
            self.kw['nms'] = False
        self.mapping = {v: CLASSES.index(k) for k, v in artifact['classes'].items()}
        self.initialized = False
        # Force actual backend loading here, so the post-load baseline precedes warm-up.
        self.model.predictor = self.model._smart_load('predictor')(
            overrides=self.model.overrides | self.kw | {'mode': 'predict', 'task': 'detect'})
        self.model.predictor.setup_model(model=self.model.model, verbose=False)

    def predict(self, frame, conf=None):
        with torch.inference_mode():
            result = self.model.predict(frame, **(self.kw | ({'conf': conf} if conf is not None else {})))[0]
            boxes = result.boxes.data.detach().cpu().numpy().copy()
        if not self.initialized:
            names = self.model.names
            for label, i in self.a['classes'].items():
                if i not in names:
                    raise ValueError(f'Class ID {i} absent in model names: {names}')
            self.info['model_names'] = names
            self.info['mapped_names'] = {k: names[v] for k, v in self.a['classes'].items()}
            self.info['nms_free_architecture'] = bool(getattr(self.model.predictor.model, 'end2end', False)) and not self.info['embedded_nms']
            if self.engine:
                backend = self.model.predictor.model
                engine = backend.model
                shape = tuple(engine.get_tensor_shape('images'))
                if -1 in shape:
                    low, opt, high = engine.get_tensor_profile_shape('images', 0)
                    required = (1, 3, 640, 640)
                    if not all(l <= d <= h for l, d, h in zip(low, required, high)):
                        raise ValueError(f'Batch-one input outside engine profile: {low}, {high}')
                    self.info['input_profile'] = [list(low), list(opt), list(high)]
                elif shape != (1, 3, 640, 640):
                    raise ValueError(f'Unsupported fixed engine shape {shape}')
                self.info['input_dtype'] = str(engine.get_tensor_dtype('images'))
            self.initialized = True
        if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
            raise ValueError('Adapter expected finite Nx6 xyxy/conf/class detections')
        if len(boxes):
            boxes = boxes[np.isin(boxes[:, 5], list(self.mapping))]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0])
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
            boxes[:, 5] = [self.mapping[int(x)] for x in boxes[:, 5]]
        return boxes, dict(result.speed)


ADAPTERS = {'ultralytics': UltralyticsAdapter}


def open_at(video, frame_index):
    cap = cv2.VideoCapture(video['path'])
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {video["id"]}')
    # Avoid keyframe seek ambiguity: advance sequentially from frame zero.
    for _ in range(frame_index):
        if not cap.grab():
            cap.release()
            raise ValueError(f'Cannot reach frame {frame_index}: {video["id"]}')
    return cap


def draw_overlay(frame, boxes, title):
    image = frame.copy()
    colors = [(30, 210, 30), (0, 140, 255)]
    for x1, y1, x2, y2, conf, cls in boxes:
        k = int(cls)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[k], 2)
        cv2.putText(image, f'{CLASSES[k]} {conf:.2f}', (int(x1), max(20, int(y1))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[k], 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(image, title, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return image


def save_overlay(path, frame, boxes, title):
    if not cv2.imwrite(str(path), draw_overlay(frame, boxes, title)):
        raise OSError(f'Cannot write image: {path}')


def worker(job_path):
    job = read_json(job_path)
    root = Path(job['output'])
    root.mkdir(parents=True, exist_ok=True)
    s, a = job['config']['settings'], job['artifact']
    torch.cuda.set_device(s['device'])
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(s['seed'])
    np.random.seed(s['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    env = environment()
    env['gpu'] = torch.cuda.get_device_name()
    env['gpu_total_MiB'] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**20
    save_json(root / 'environment.json', env)
    telemetry = Telemetry(torch.cuda.current_device())
    before = telemetry.sample()
    backend = ADAPTERS[a['backend']](a, s)
    after_load = telemetry.sample()
    rows = []
    show_preview = s.get('show', False)
    preview_window = 'Live inference - Q/Esc hides preview'
    if job['mode'] == 'validate':
        v = job['videos'][0]
        cap = open_at(v, v['start_frame'])
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError('Cannot read validation frame')
        backend.predict(image)
        save_json(root / 'artifact.json', backend.info)
        save_json(root / 'memory_baselines.json', dict(before_model_load=before, after_model_load=after_load, after_load_and_warmup=telemetry.sample()))
        save_json(root / 'done.json', dict(mode='validate', compatible=True))
        return
    for v in job['videos']:
        out = root / v['id']
        out.mkdir(exist_ok=True)
        if job['mode'] == 'performance':
            if job.get('resume') and (out / 'result.json').is_file():
                existing = completed_clip(out, v, a['id'], job['pass'])
                if existing is not None:
                    rows.append(existing)
                    save_json(root / 'checkpoint.json', rows)
                    print(f'Reusing completed clip: {v["id"]}', flush=True)
                    continue
            clip_start = time.perf_counter()
            cap = open_at(v, v['start_frame'])
            first_start = time.perf_counter()
            ok, first = cap.read()
            first_decode_ms = (time.perf_counter() - first_start) * 1000
            if not ok:
                raise ValueError(f'Cannot read first frame: {v["id"]}')
            for _ in range(s['warmup']):
                backend.predict(first)
            torch.cuda.synchronize()
            loaded = telemetry.sample()
            save_json(root / 'artifact.json', backend.info)
            save_json(root / 'memory_baselines.json', dict(before_model_load=before, after_model_load=after_load, after_load_and_warmup=loaded,
                        nvml_warning=telemetry.unavailable,
                        note='Process VRAM may be unavailable under WDDM. Device readings include desktop load.'))
            timings, saved, counts = [], [], []
            first_boxes = np.empty((0, 6), dtype=np.float32)
            hash_times = []
            heat = np.zeros((2, v['height'], v['width']), np.uint32) if job['pass'] == 0 else None
            region_counts = {r['id']: [0, 0] for r in v.get('regions', [])}
            telemetry.start(s['telemetry_interval'])
            measured_start = time.perf_counter()
            reader = FrameReader(cap, v['selected_frames'], first, first_decode_ms, s.get('prefetch', 0))
            try:
                if show_preview:
                    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(preview_window, 1280, 720)
                for i, (frame, decode_ms, hash_ms) in enumerate(reader):
                    hash_times.append(hash_ms)
                    if frame.shape[:2] != (v['height'], v['width']):
                        raise ValueError('Source resolution changed')
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    boxes, speed = backend.predict(frame)
                    torch.cuda.synchronize()
                    call_ms = (time.perf_counter() - t) * 1000
                    if i == 0:
                        first_boxes = boxes.copy()
                    timings.append([decode_ms, speed['preprocess'], speed['inference'], speed['postprocess'], call_ms])
                    counts.append([int((boxes[:, 5] == k).sum()) for k in range(2)])
                    if heat is not None:
                        saved.append(boxes)
                    if show_preview:
                        title = (f'{a["id"]} / {v["id"]} / {i + 1}/{v["selected_frames"]} '
                                 f'/ inference {speed["inference"]:.1f} ms')
                        cv2.imshow(preview_window, draw_overlay(frame, boxes, title))
                        key = cv2.waitKey(1) & 0xff
                        if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                            show_preview = False
                            cv2.destroyAllWindows()
                    if (i + 1) % 1000 == 0:
                        elapsed = time.perf_counter() - measured_start
                        eta = elapsed / (i + 1) * (v['selected_frames'] - i - 1)
                        print(f'{a["id"]} pass {job["pass"] + 1} {v["id"]}: {i + 1}/{v["selected_frames"]}; clip ETA {eta:.0f}s', flush=True)
            finally:
                reader.close()
                resource = telemetry.stop()
                cap.release()
                if show_preview:
                    cv2.destroyAllWindows()
            measured_seconds = time.perf_counter() - measured_start
            values = np.asarray(timings)
            analysis_start = time.perf_counter()
            if heat is not None:
                for boxes in saved:
                    if not len(boxes):
                        continue
                    x = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, v['width'] - 1)
                    y = np.clip(boxes[:, 3].astype(int), 0, v['height'] - 1)
                    np.add.at(heat, (boxes[:, 5].astype(int), y, x), 1)
                    for region in v.get('regions', []):
                        polygon = np.asarray(region['points'], np.float32)
                        for xx, yy, k in zip(x, y, boxes[:, 5].astype(int)):
                            if cv2.pointPolygonTest(polygon, (float(xx), float(yy)), False) >= 0:
                                region_counts[region['id']][k] += 1
            row = dict(model=a['id'], video=v['id'], group=v['group'], camera=v['camera'],
                       pass_index=job['pass'], frames=len(timings), source_fps=v['fps'],
                       decoded_frames_sha256=reader.digest.hexdigest(), resource=resource,
                       prefetch=s.get('prefetch', 0), hash_ms=stats(np.asarray(hash_times)),
                       processing_seconds=measured_seconds, processing_fps=len(timings) / measured_seconds,
                       before_load=before, after_load=after_load, after_warmup=loaded,
                       analysis_seconds=time.perf_counter() - analysis_start, comparable=backend.info['comparable'],
                       precision=a['precision'], car_detections=sum(x[0] for x in counts),
                       motorcycle_detections=sum(x[1] for x in counts), region_counts=region_counts)
            for idx, name in enumerate(('decode_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'detection_ms')):
                row[name] = stats(values[:, idx])
            row['inference_fps'] = len(values) * 1000 / values[:, 2].sum()
            row['detection_fps'] = len(values) * 1000 / values[:, 4].sum()
            row['target_budget_exceed_percent'] = float((values[:, 4] > 1000 / s['target_fps']).mean() * 100)
            row['source_budget_exceed_percent'] = float((values[:, 4] > 1000 / v['fps']).mean() * 100)
            write_start = time.perf_counter()
            np.savez_compressed(out / 'frames.npz', timings=values, counts=np.asarray(counts))
            save_csv(out / 'telemetry.csv', telemetry.rows)
            if heat is not None:
                np.savez_compressed(out / 'heat.npz', counts=heat)
                offsets = np.r_[0, np.cumsum([len(b) for b in saved])]
                np.savez_compressed(out / 'predictions.npz', boxes=np.concatenate(saved), offsets=offsets)
                if not cv2.imwrite(str(out / 'background.png'), first):
                    raise OSError('Cannot save heat-map background')
                save_overlay(out / 'boxes_first.png', first, first_boxes,
                             f'{a["id"]} / {v["id"]} / first frame boxes')
            row['artifact_write_seconds'] = time.perf_counter() - write_start
            row['clip_seconds'] = time.perf_counter() - clip_start
            row['resumed_run'] = bool(job.get('resume'))
            save_json(out / 'result.json', row)
            rows.append(row)
            save_json(root / 'checkpoint.json', rows)
            print(f'Complete: {a["id"]}/{v["id"]}, {row["detection_fps"]:.1f} detection FPS', flush=True)
        else:
            cap = open_at(v, v['frame'])
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f'Cannot decode selected frame: {v["id"]}')
            frame_hash = hashlib.sha256(frame.data).hexdigest()
            if v.get('truth_frame_sha256') and v['truth_frame_sha256'] != frame_hash:
                raise ValueError(f'{v["id"]}: selected frame no longer matches the annotated image hash')
            for _ in range(s['warmup']):
                backend.predict(frame)
            save_json(root / 'artifact.json', backend.info)
            thresholds = sorted(set([0.10, 0.25, 0.50, 0.75, s['conf']])) if backend.info['comparable'] else [s['conf']]
            if job['mode'] == 'ensemble-review':
                boxes, _ = backend.predict(frame, .001)
                record = dict(model=a['id'], video=v['id'], frame=v['frame'], frame_sha256=frame_hash,
                              width=frame.shape[1], height=frame.shape[0], boxes=boxes.tolist())
                save_json(out / 'detections.json', record)
                if not cv2.imwrite(str(out / 'source.png'), frame):
                    raise OSError('Cannot save review source')
                rows.append(record)
                save_json(root / 'checkpoint.json', rows)
                print(f'Ensemble predictions: {a["id"]}/{v["id"]}', flush=True)
                continue
            for conf in thresholds:
                boxes, _ = backend.predict(frame, conf)
                predicted = {k: int((boxes[:, 5] == i).sum()) for i, k in enumerate(CLASSES)}
                r = dict(model=a['id'], video=v['id'], camera=v['camera'], group=v['group'], frame=v['frame'],
                         confidence=conf, primary=conf == s['conf'], truth=v['truth'], predicted=predicted,
                         frame_sha256=frame_hash, comparable=backend.info['comparable'])
                r['signed_error'] = {k: predicted[k] - v['truth'][k] for k in CLASSES}
                r['absolute_error'] = {k: abs(r['signed_error'][k]) for k in CLASSES}
                stem = f'conf_{conf:.4f}'
                save_json(out / f'{stem}.json', r | {'boxes': boxes.tolist()})
                title = f"{a['id']} conf={conf:.2f} car {predicted['car']}/{v['truth']['car']} motorcycle {predicted['motorcycle']}/{v['truth']['motorcycle']} (prediction/truth)"
                save_overlay(out / f'{stem}.png', frame, boxes, title)
                rows.append(r)
            save_json(root / 'checkpoint.json', rows)
            print(f'Count check: {a["id"]}/{v["id"]}', flush=True)
    save_json(root / 'done.json', {'mode': job['mode'], 'rows': len(rows)})
