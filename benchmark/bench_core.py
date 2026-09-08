"""Configuration, reproducibility and metrics shared by benchmark modes (no GPU imports)."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

CLASSES = ('car', 'motorcycle')
DEFAULTS = dict(imgsz=640, conf=0.25, iou=0.70, max_det=300, warmup=30,
                repeats=3, device='cuda:0', target_fps=30, seed=0, telemetry_interval=0.1,
                batch=1, rect=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'ID must contain only letters, digits, underscores or hyphens: {value!r}')
    return value


def load_config(path):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(c, dict):
        raise ValueError('Config must be a YAML mapping')
    if c.get('runtime_path'):
        c['runtime_path'] = str((path.parent / c['runtime_path']).resolve())
    s = c['settings'] = DEFAULTS | c.get('settings', {})
    if s['imgsz'] != 640:
        raise ValueError('This comparison requires 640x640 input')
    if s['batch'] != 1 or s['rect'] is not False:
        raise ValueError('This comparison requires batch=1 and rect=false')
    if type(s['seed']) is not int or not 0 <= s['seed'] < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    for key in ('warmup', 'repeats', 'max_det'):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('conf', 'iou'):
        if not isinstance(s[key], (int, float)) or not 0 < s[key] < 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if not re.fullmatch(r'cuda:\d+', str(s['device'])):
        raise ValueError('Use an explicit GPU such as cuda:0; no silent CPU fallback')
    for key in ('target_fps', 'telemetry_interval'):
        if not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f'{key} must be positive')
    for collection in ('models', 'videos'):
        if not c.get(collection):
            raise ValueError(f'No {collection} registered')
        seen = set()
        for item in c[collection]:
            name = safe_id(item['id'])
            if name in seen:
                raise ValueError(f'Duplicate {collection} ID: {name}')
            seen.add(name)
            item['path'] = str((path.parent / item['path']).resolve())
    for m in c['models']:
        if m.get('backend', 'ultralytics') != 'ultralytics':
            raise ValueError(f"{m['id']}: backend requires an adapter; supported: ultralytics")
        m.setdefault('backend', 'ultralytics')
        m.setdefault('name', m['id'])
        m.setdefault('version', 'unspecified')
        if Path(m['path']).suffix not in ('.pt', '.engine'):
            raise ValueError(f"{m['id']}: only .pt and .engine are supported")
        if m.get('precision') not in ('fp32', 'fp16', 'int8'):
            raise ValueError(f"{m['id']}: declare precision fp32/fp16/int8")
        if Path(m['path']).suffix == '.pt' and m['precision'] == 'int8':
            raise ValueError('INT8 PyTorch requires a separate adapter')
        if m.get('input_size', 640) != 640:
            raise ValueError(f"{m['id']}: input must be 640")
        ids = m.get('classes', {})
        if set(ids) != set(CLASSES) or any(type(v) is not int or v < 0 for v in ids.values()) or len(set(ids.values())) != 2:
            raise ValueError(f"{m['id']}: classes must map car and motorcycle to distinct nonnegative IDs")
        if m.get('pair_id'):
            safe_id(m['pair_id'])
        if m.get('parameters') is not None and (type(m['parameters']) is not int or m['parameters'] < 1):
            raise ValueError('parameters must be a positive integer or null')
    cameras = {}
    for v in c['videos']:
        if v.get('group') not in ('day', 'night'):
            raise ValueError('video group must be day or night')
        camera = safe_id(v['camera'])
        cameras.setdefault(v['group'], []).append(camera)
        v.setdefault('start_seconds', 0)
        if not math.isfinite(v['start_seconds']) or v['start_seconds'] < 0:
            raise ValueError('start_seconds must be finite and nonnegative')
        if v.get('duration_seconds', 30) != 30:
            raise ValueError('Every registered segment must be 30 seconds')
        v['duration_seconds'] = 30
        f = v.get('frame')
        if f is not None and (type(f) is not int or f < 0):
            raise ValueError('frame must be a zero-based integer or null')
        for label, count in (v.get('truth') or {}).items():
            if label not in CLASSES or (count is not None and (type(count) is not int or count < 0)):
                raise ValueError('truth must contain nonnegative integer car/motorcycle counts or null')
        for region in v.get('regions', []):
            safe_id(region['id'])
            pts = np.asarray(region['points'], dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
                raise ValueError('Region points must be at least three finite original-pixel x,y pairs')
    if set(cameras) != {'day', 'night'} or any(len(x) != 4 or len(set(x)) != 4 for x in cameras.values()):
        raise ValueError('Register exactly four distinct cameras in each of day and night')
    if set(cameras['day']) != set(cameras['night']):
        raise ValueError('Day and night camera IDs must match')
    c.setdefault('reference_model', c['models'][0]['id'])
    if c['reference_model'] not in {m['id'] for m in c['models']}:
        raise ValueError('reference_model is not registered')
    return c


def truth_ready(v):
    t = v.get('truth') or {}
    return v.get('frame') is not None and all(type(t.get(k)) is int and t[k] >= 0 for k in CLASSES)


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {k: None for k in ('mean', 'median', 'std', 'p95', 'p99', 'total')}
    return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), total=float(a.sum()))


def count_metrics(pairs):
    result = {}
    for label in CLASSES:
        truth = np.array([p['truth'][label] for p in pairs], dtype=float)
        pred = np.array([p['predicted'][label] for p in pairs], dtype=float)
        error = pred - truth
        result[label] = dict(mae=float(np.abs(error).mean()), bias=float(error.mean()),
                             exact_match_rate=float((error == 0).mean()),
                             wape_percent=float(np.abs(error).sum() / truth.sum() * 100) if truth.sum() else None)
    result['total_mae'] = sum(result[k]['mae'] for k in CLASSES)
    return result


def heat_points(boxes, height, width):
    """Nx6 normalized detections: original-pixel xyxy, confidence, canonical class 0/1."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 6)
    h = np.zeros((2, height, width), dtype=np.uint32)
    if len(b):
        x = np.clip(np.floor((b[:, 0] + b[:, 2]) / 2).astype(int), 0, width - 1)
        y = np.clip(np.floor(b[:, 3]).astype(int), 0, height - 1)
        np.add.at(h, (b[:, 5].astype(int), y, x), 1)
    return h


def box_agreement(a, b, threshold=0.5):
    """Deterministic descending-IoU one-to-one same-class agreement, not GT accuracy."""
    a = np.asarray(a, dtype=float).reshape(-1, 6)
    b = np.asarray(b, dtype=float).reshape(-1, 6)
    candidates = []
    for i, x in enumerate(a):
        if not len(b):
            break
        lo = np.maximum(x[:2], b[:, :2])
        hi = np.minimum(x[2:4], b[:, 2:4])
        inter = np.prod(np.maximum(0, hi - lo), axis=1)
        area = np.prod(np.maximum(0, x[2:4] - x[:2]))
        other = np.prod(np.maximum(0, b[:, 2:4] - b[:, :2]), axis=1)
        ious = inter / np.maximum(area + other - inter, 1e-12)
        candidates.extend((float(ious[j]), i, j) for j in range(len(b))
                          if x[5] == b[j, 5] and ious[j] >= threshold)
    left, right, matches = set(), set(), []
    for iou, i, j in sorted(candidates, key=lambda v: (-v[0], v[1], v[2])):
        if i not in left and j not in right:
            left.add(i)
            right.add(j)
            matches.append((iou, abs(float(a[i, 4] - b[j, 4]))))
    return dict(matched=len(matches), unmatched_a=len(a) - len(matches), unmatched_b=len(b) - len(matches),
                sum_iou=sum(v[0] for v in matches), sum_confidence_delta=sum(v[1] for v in matches))
