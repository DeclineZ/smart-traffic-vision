"""CLI: reproducible day/night performance, preparation, count checks and reports."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bench_core import load_config, read_json, save_json, sha256, truth_ready


def inspect_videos(videos, smoke=False):
    import cv2
    import math
    out = []
    for v in videos:
        if not Path(v['path']).is_file():
            raise ValueError(f'Missing video {v["path"]}')
        cap = cv2.VideoCapture(v['path'])
        if not cap.isOpened():
            raise ValueError(f'Cannot open {v["id"]}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
        cap.release()
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f'Invalid video metadata: {v["id"]}')
        start, count = round(v['start_seconds'] * fps), round(30 * fps)
        if count < 1 or (n > 0 and start + count > n):
            raise ValueError(f'Selected 30-second segment exceeds video: {v["id"]}')
        if v.get('frame') is not None and not start <= v['frame'] < start + count:
            raise ValueError(f'{v["id"]}: selected frame must be in [{start}, {start + count - 1}]')
        for region in v.get('regions', []):
            if any(not (0 <= x <= width and 0 <= y <= height) for x, y in region['points']):
                raise ValueError(f'{v["id"]}: region coordinates outside source image')
        out.append(v | dict(fps=fps, width=width, height=height, reported_frames=n,
                            start_frame=start, selected_frames=min(8, count) if smoke else count,
                            full_segment_frames=count, sha256=sha256(v['path'])))
    return out


def prepare(videos, output):
    import cv2
    # Preparation does not load CUDA or models.
    selected = [v for v in videos if v.get('frame') is not None]
    if not selected:
        raise ValueError('Set frame in YAML or use --frame day_north=2250 (zero-based original frame)')
    records = []
    for v in selected:
        cap = cv2.VideoCapture(v['path'])
        try:
            for _ in range(v['frame']):
                if not cap.grab():
                    raise ValueError(f'Cannot reach {v["id"]} frame {v["frame"]}')
            ok, image = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f'Cannot decode {v["id"]}')
        path = output / f'{v["id"]}_frame_{v["frame"]}.png'
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Cannot write {path}')
        records.append(dict(video=v['id'], frame=v['frame'], path=path.name,
                            source_sha256=v['sha256'], frame_sha256=hashlib.sha256(image.data).hexdigest(),
                            truth=v.get('truth'), instruction='Count this exact image and enter counts plus frame in YAML.'))
    save_json(output / 'selected_frames.json', records)


def check_frame_identity(output):
    refs = {}
    for p in sorted(output.glob('performance/*/pass_*/**/result.json')):
        r = read_json(p)
        key = r['video']
        sig = (r['frames'], r['decoded_frames_sha256'])
        if key in refs and refs[key] != sig:
            raise ValueError(f'Decoded frame mismatch: {p}; refusing comparison')
        refs[key] = sig
    for p in sorted(output.glob('count-check/*/pass_*/checkpoint.json')):
        for r in read_json(p):
            key = ('count', r['video'])
            sig = (r['frame'], r['frame_sha256'])
            if key in refs and refs[key] != sig:
                raise ValueError(f'Selected frame mismatch: {p}')
            refs[key] = sig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('models.yaml'))
    p.add_argument('--mode', choices=['performance', 'count-check', 'all', 'prepare', 'validate', 'report'], default='all')
    p.add_argument('--group', choices=['day', 'night', 'all'], default='all')
    p.add_argument('--models', nargs='+', help='Registered model IDs; defaults to all')
    p.add_argument('--frame', action='append', default=[], metavar='VIDEO_ID=FRAME', help='Override selected original zero-based frame')
    p.add_argument('--output', type=Path, help='New run directory; existing only with --mode report')
    p.add_argument('--smoke', action='store_true', help='8 frames/video, 3 warmups, 1 pass; never use for conclusions')
    p.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.job:
        try:
            job_config = read_json(a.job)['config']
            if job_config.get('runtime_path'):
                sys.path.insert(0, job_config['runtime_path'])
                os.environ['YOLO_CONFIG_DIR'] = str(Path(__file__).parent / 'runtime-settings')
                Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
            from bench_runtime import worker
            worker(a.job)
        except Exception as e:
            job = read_json(a.job)
            save_json(Path(job['output']) / 'failed.json', dict(error=str(e), traceback=traceback.format_exc()))
            raise
        return
    if a.mode == 'report':
        if not a.output or not (a.output / 'run.json').is_file():
            p.error('--mode report requires --output pointing at an existing run')
        from bench_report import make_report
        check_frame_identity(a.output)
        make_report(a.output)
        return
    config = load_config(a.config)
    if config.get('runtime_path'):
        sys.path.insert(0, config['runtime_path'])
    for entry in a.frame:
        name, sep, number = entry.partition('=')
        matches = [v for v in config['videos'] if v['id'] == name]
        if not sep or not matches or not number.isdigit():
            p.error(f'Invalid frame override {entry}')
        matches[0]['frame'] = int(number)
    models = config['models']
    if a.models:
        unknown = set(a.models) - {m['id'] for m in models}
        if unknown:
            p.error(f'Unregistered model IDs: {unknown}')
        models = [m for m in models if m['id'] in a.models]
    videos = [v for v in config['videos'] if a.group == 'all' or v['group'] == a.group]
    if a.smoke:
        config['settings'].update(warmup=3, repeats=1)
    output = (a.output or Path(__file__).parent / 'model-results' / time.strftime('%Y%m%d-%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'resolved_config.json', config)
    print('Inspecting video metadata and hashing source files...', flush=True)
    videos = inspect_videos(videos, a.smoke)
    save_json(output / 'dataset.json', videos)
    if a.mode == 'prepare':
        prepare(videos, output)
        print(f'Prepared source frames: {output}')
        return
    requested = ['performance', 'count-check'] if a.mode == 'all' else [a.mode]
    skipped = []
    if config['reference_model'] not in {m['id'] for m in models}:
        skipped.append(dict(mode='difference-map', reason='Reference model not selected', reference=config['reference_model']))
    if 'count-check' in requested:
        missing = [v['id'] for v in videos if not truth_ready(v)]
        if missing:
            skipped.append(dict(mode='count-check', reason='Missing selected frames or human counts', videos=missing))
            requested.remove('count-check')
            print(f'Count-check unavailable: enter frame + truth for {", ".join(missing)}', flush=True)
    run = dict(smoke=a.smoke, selected_models=[m['id'] for m in models], selected_videos=[v['id'] for v in videos],
               settings=config['settings'], requested_mode=a.mode, skipped=skipped,
               command=subprocess.list2cmdline([sys.executable, *sys.argv]), jobs=[], failures=[],
               source_hashes={f.name: sha256(f) for f in Path(__file__).parent.glob('bench*.py')})
    run['source_hashes'][Path(__file__).name] = sha256(__file__)
    save_json(output / 'run.json', run)
    lock = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata['Name'])
    (output / 'requirements-lock.txt').write_text('\n'.join(lock), encoding='utf-8')
    jobs_dir = output / 'jobs'
    jobs_dir.mkdir()
    try:
        for mode in requested:
            repeats = config['settings']['repeats'] if mode == 'performance' else 1
            for pass_index in range(repeats):
                order = models[pass_index % len(models):] + models[:pass_index % len(models)]
                for artifact in order:
                    root = output / mode / artifact['id'] / f'pass_{pass_index + 1}'
                    root.mkdir(parents=True)
                    job = dict(mode=mode, artifact=artifact, config=config, videos=videos,
                               output=str(root), **{'pass': pass_index})
                    job_path = jobs_dir / f'{mode}_{artifact["id"]}_{pass_index + 1}.json'
                    save_json(job_path, job)
                    record = dict(mode=mode, model=artifact['id'], pass_index=pass_index, output=str(root.relative_to(output)), status='running')
                    run['jobs'].append(record)
                    save_json(output / 'run.json', run)
                    print(f'Running {mode}: {artifact["id"]}, pass {pass_index + 1}/{repeats}', flush=True)
                    # Each worker is isolated. A file avoids pipe deadlock and preserves diagnostics.
                    with (root / 'worker.log').open('w', encoding='utf-8') as log:
                        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--job', str(job_path)],
                                                stdout=log, stderr=subprocess.STDOUT)
                        try:
                            while proc.poll() is None:
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    print(f'  still running: {artifact["id"]} ({mode}); log {root / "worker.log"}', flush=True)
                        except KeyboardInterrupt:
                            proc.terminate()
                            proc.wait()
                            record['status'] = 'interrupted'
                            raise
                    record['status'] = 'complete' if proc.returncode == 0 and (root / 'done.json').exists() else 'failed'
                    if record['status'] == 'failed':
                        error = read_json(root / 'failed.json')['error'] if (root / 'failed.json').exists() else f'Worker exit {proc.returncode}'
                        run['failures'].append(record | {'error': error})
                        print(f'FAILED: {error}', flush=True)
                    check_frame_identity(output)
                    save_json(output / 'run.json', run)
    except KeyboardInterrupt:
        run['interrupted'] = True
        print('Interrupted; completed checkpoints preserved.', flush=True)
    except Exception as e:
        run['invalid_comparison'] = str(e)
        raise
    finally:
        save_json(output / 'run.json', run)
    from bench_report import make_report
    make_report(output)
    print(f'Report: {output / "index.html"}', flush=True)
    if run['failures'] or run.get('interrupted') or (a.mode == 'count-check' and skipped):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
