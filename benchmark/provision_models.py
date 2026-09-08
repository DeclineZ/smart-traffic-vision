"""Download official weights and build fixed-batch FP16 TensorRT engines, sequentially."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'benchmark' / 'runtime-yolo26'
MODELS = ('yolov8s', 'yolov8m', 'yolo11s', 'yolo11m', 'yolo26s', 'yolo26m')
sys.path.insert(0, str(RUNTIME))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'benchmark' / 'runtime-settings')
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)


def checksum(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--build', choices=MODELS)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    import ultralytics
    from ultralytics import YOLO
    import torch
    import numpy as np
    import tensorrt
    print('Runtime', ultralytics.__version__, 'Torch', torch.__version__, 'TensorRT', tensorrt.__version__, flush=True)
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    names = [a.build] if a.build else MODELS
    for name in names:
        weights = folder / f'{name}.pt'
        url = f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt'
        if not weights.exists():
            if a.check:
                raise FileNotFoundError(weights)
            partial = weights.with_suffix('.download')
            print('Downloading', url, flush=True)
            urllib.request.urlretrieve(url, partial)
            partial.replace(weights)
        if a.download_only:
            print(name, weights.stat().st_size, checksum(weights), flush=True)
            continue
        engine = weights.with_suffix('.engine')
        if not a.check and not engine.exists():
            print('Building', name, 'FP16 batch=1 imgsz=640 workspace=2 GiB', flush=True)
            start = time.perf_counter()
            model = YOLO(str(weights))
            params = sum(x.numel() for x in model.model.parameters())
            result = model.export(format='engine', imgsz=640, batch=1, dynamic=False,
                                  quantize=16, workspace=2, device=0, nms=False, simplify=False, opset=17)
            manifest = dict(model=name, weights_sha256=checksum(weights), engine_sha256=checksum(engine),
                            source_url=url, parameters=params, precision='fp16', batch=1, input_size=640,
                            ultralytics=ultralytics.__version__, torch=torch.__version__,
                            tensorrt=tensorrt.__version__, cuda=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(), build_seconds=time.perf_counter() - start,
                            export=str(result), nms=False, end2end=name.startswith('yolo26'))
            engine.with_suffix('.build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if not engine.exists():
            raise FileNotFoundError(engine)
        for path in (weights, engine):
            model = YOLO(str(path), task='detect')
            r = model.predict(np.zeros((360, 640, 3), dtype=np.uint8), imgsz=640, rect=False,
                              device=0, quantize=16, nms=False, classes=[2, 3], verbose=False)[0]
            print('PASS', path.name, 'boxes', len(r.boxes), 'end2end', getattr(model.predictor.model, 'end2end', None), flush=True)
            del model, r
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
