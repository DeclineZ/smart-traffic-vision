"""CLI: reproducible day/night performance, preparation, count checks and reports."""
from __future__ import annotations
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bench_core import load_config, read_json, save_json, sha256, truth_ready, completed_clip, expand_count_frames


def lock_run(output):
    """Hold an OS lock until process exit; crashes release it automatically."""
    handle = (output / 'run.lock').open('a+b')
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ValueError('This output directory is already in use by another run')
    atexit.register(handle.close)
    return handle


def inspect_videos(videos, smoke=False):
    import cv2
    import math
    out = []
    source_hashes = {}
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
        start, count = round(v['start_seconds'] * fps), round(v.get('duration_seconds', 30) * fps)
        if 'start_frame_override' in v:
            start = v['start_frame_override']
            v = v | dict(start_seconds=start / fps)
        if count < 1 or (n > 0 and start + count > n):
            raise ValueError(f'Selected {v.get("duration_seconds", 30):g}-second segment exceeds video: {v["id"]}')
        if v.get('frame') is not None and not start <= v['frame'] < start + count:
            raise ValueError(f'{v["id"]}: selected frame must be in [{start}, {start + count - 1}]')
        for region in v.get('regions', []):
            if any(not (0 <= x <= width and 0 <= y <= height) for x, y in region['points']):
                raise ValueError(f'{v["id"]}: region coordinates outside source image')
        if v['path'] not in source_hashes:
            source_hashes[v['path']] = sha256(v['path'])
        out.append(v | dict(fps=fps, width=width, height=height, reported_frames=n,
                            start_frame=start, selected_frames=min(8, count) if smoke else count,
                            full_segment_frames=count, sha256=source_hashes[v['path']]))
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
        records.append(dict(video=v['id'], source_video=v.get('source_video', v['id']), frame=v['frame'], path=path.name,
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
    p.add_argument('--mode', choices=['performance', 'count-check', 'ensemble-review', 'ensemble-score', 'all', 'prepare', 'validate', 'report'], default='all')
    p.add_argument('--group', choices=['day', 'night', 'all'], default='all')
    p.add_argument('--models', nargs='+', help='Registered model IDs; defaults to all')
    p.add_argument('--videos', nargs='+', help='Video IDs, intersected with --group')
    p.add_argument('--repeats', type=int, help='Override measurement repeats')
    p.add_argument('--prefetch', type=int, default=0, help='Decode queue depth 0..16; 0 is sequential')
    p.add_argument('--show', action='store_true', help='Show live bounding boxes during performance inference; Q/Esc hides preview')
    p.add_argument('--resume', action='store_true', help='Resume matching performance run at completed clip boundaries')
    p.add_argument('--truth-file', type=Path, help='Prepared selected_frames.json with manually entered counts')
    p.add_argument('--frame', action='append', default=[], metavar='VIDEO_ID=FRAME', help='Override selected original zero-based frame')
    p.add_argument('--output', type=Path, help='New run directory; existing with --resume or --mode report')
    p.add_argument('--smoke', action='store_true', help='8 frames/video, 3 warmups, 1 pass; never use for conclusions')
    p.add_argument('--duration-seconds', type=float, help='Override selected video segment duration for performance runs')
    p.add_argument('--start-frame', type=int, help='Start performance clips at this zero-based original frame')
    p.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.mode == 'ensemble-score':
        if not a.output or not a.truth_file:
            p.error('ensemble-score requires --output review directory and --truth-file reviewed JSON')
        from bench_ensemble import score
        score(a.output.resolve(), a.truth_file.resolve())
        return
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
    if a.videos:
        if set(a.videos) - {v['id'] for v in config['videos']}:
            p.error('Unknown video ID')
        videos = [v for v in videos if v['id'] in a.videos]
    if a.start_frame is not None:
        if a.start_frame < 0 or a.mode != 'performance':
            p.error('--start-frame requires a nonnegative frame and --mode performance')
        for v in videos:
            v['start_frame_override'] = a.start_frame
            v['frame'] = None
    if a.duration_seconds is not None:
        if a.duration_seconds <= 0:
            p.error('--duration-seconds must be positive')
        for v in videos:
            v['duration_seconds'] = a.duration_seconds
    if not videos:
        p.error('No videos selected')
    if a.mode == 'all' and any('count_frames' in v for v in videos):
        p.error('Use --mode count-check or performance separately with count_frames')
    if a.mode in ('prepare', 'count-check', 'ensemble-review'):
        videos = expand_count_frames(videos)
    if a.mode == 'ensemble-review':
        if any(v.get('frame') is None for v in videos):
            p.error('ensemble-review requires selected frame/count_frames for every selected video')
        models = [m for m in models if Path(m['path']).suffix == '.engine']
        if not models:
            p.error('No engines selected')
    if a.truth_file and a.mode != 'count-check':
        p.error('--truth-file requires --mode count-check')
    if not 0 <= a.prefetch <= 16 or (a.repeats is not None and a.repeats < 1):
        p.error('prefetch must be 0..16 and repeats must be positive')
    config['settings']['prefetch'] = a.prefetch
    if a.show and a.mode not in ('performance', 'all'):
        p.error('--show requires --mode performance or all')
    config['settings']['show'] = a.show
    if a.repeats is not None:
        config['settings']['repeats'] = a.repeats
    if a.resume and (a.mode != 'performance' or not a.output):
        p.error('--resume requires --mode performance and --output')
    if a.smoke:
        config['settings'].update(warmup=3, repeats=1)
    output = (a.output or Path(__file__).parent / 'model-results' / time.strftime('%Y%m%d-%H%M%S')).resolve()
    if not a.resume:
        output.mkdir(parents=True, exist_ok=False)
    run_lock = lock_run(output)
    print('Inspecting video metadata and hashing source files...', flush=True)
    videos = inspect_videos(videos, a.smoke)
    if a.truth_file:
        records = read_json(a.truth_file)
        by_id = {r['video']: r for r in records}
        if len(by_id) != len(records):
            p.error('Duplicate image IDs in truth file')
        for v in videos:
            r = by_id.get(v['id'])
            if r is None or r['frame'] != v['frame'] or r['source_sha256'] != v['sha256']:
                p.error(f'Truth file does not match selected source/frame: {v["id"]}')
            import cv2
            image = cv2.imread(str(a.truth_file.resolve().parent / r['path']))
            if image is None or hashlib.sha256(image.data).hexdigest() != r['frame_sha256']:
                p.error(f'Prepared image missing or modified: {v["id"]}')
            v['truth'] = r['truth']
            v['truth_frame_sha256'] = r['frame_sha256']
            if not truth_ready(v):
                p.error(f'Enter nonnegative integer car and motorcycle counts for {v["id"]}; use 0 only if actually absent')
    signature = dict(config=config, videos=videos, models={m['id']: sha256(m['path']) for m in models},
                     smoke=a.smoke, mode=a.mode,
                     code={f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')})
    if a.resume:
        if read_json(output / 'resume_signature.json') != signature:
            p.error('Resume rejected: configuration, selection, code, models or videos changed')
    else:
        save_json(output / 'resume_signature.json', signature)
        save_json(output / 'resolved_config.json', config)
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
    if a.resume:
        previous = read_json(output / 'run.json')
        run['jobs'] = previous['jobs']
        run['resume_history'] = previous.get('resume_history', []) + [dict(time=time.time(), command=run['command'])]
    save_json(output / 'run.json', run)
    lock = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata['Name'])
    (output / 'requirements-lock.txt').write_text('\n'.join(lock), encoding='utf-8')
    jobs_dir = output / 'jobs'
    jobs_dir.mkdir(exist_ok=a.resume)
    total_jobs = len(models) * sum(config['settings']['repeats'] if m == 'performance' else 1 for m in requested)
    print(f'Planned: {total_jobs} jobs / {total_jobs * len(videos)} clip runs', flush=True)
    try:
        for mode in requested:
            repeats = config['settings']['repeats'] if mode == 'performance' else 1
            for pass_index in range(repeats):
                order = models[pass_index % len(models):] + models[:pass_index % len(models)]
                for artifact in order:
                    root = output / mode / artifact['id'] / f'pass_{pass_index + 1}'
                    root.mkdir(parents=True, exist_ok=a.resume)
                    record = next((r for r in run['jobs'] if r['mode'] == mode and r['model'] == artifact['id'] and r['pass_index'] == pass_index), None)
                    if a.resume and record and record['status'] == 'complete' and (root / 'done.json').is_file() and all(completed_clip(root / v['id'], v, artifact['id'], pass_index) is not None for v in videos):
                        print(f'Skipping completed job: {artifact["id"]}, pass {pass_index + 1}', flush=True)
                        continue
                    job = dict(mode=mode, artifact=artifact, config=config, videos=videos,
                               output=str(root), resume=a.resume, **{'pass': pass_index})
                    job_path = jobs_dir / f'{mode}_{artifact["id"]}_{pass_index + 1}.json'
                    save_json(job_path, job)
                    if record is None:
                        record = dict(mode=mode, model=artifact['id'], pass_index=pass_index, output=str(root.relative_to(output)))
                        run['jobs'].append(record)
                    record['status'] = 'running'
                    save_json(output / 'run.json', run)
                    print(f'Running {mode}: {artifact["id"]}, pass {pass_index + 1}/{repeats}', flush=True)
                    # Each worker is isolated. A file avoids pipe deadlock and preserves diagnostics.
                    with (root / 'worker.log').open('a' if a.resume else 'w', encoding='utf-8') as log:
                        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--job', str(job_path)],
                                                stdout=log, stderr=subprocess.STDOUT)
                        try:
                            while proc.poll() is None:
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    print(f'  still running: {artifact["id"]} ({mode}); log {root / "worker.log"}', flush=True)
                                    checkpoint = root / 'checkpoint.json'
                                    if checkpoint.is_file():
                                        completed = read_json(checkpoint)
                                        seconds = [r['clip_seconds'] for r in completed if 'clip_seconds' in r]
                                        if seconds:
                                            eta = sum(seconds) / len(seconds) * (len(videos) - len(completed))
                                            print(f'  {len(completed)}/{len(videos)} clips saved; job ETA ~{eta / 60:.1f} min (completed-clip estimate)', flush=True)
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
    if a.mode == 'ensemble-review':
        if run['failures'] or run.get('interrupted'):
            raise SystemExit('Ensemble incomplete; inspect worker logs. Review was not generated.')
        from bench_ensemble import make_review
        make_review(output)
        print(f'Review: {output / "review.html"}', flush=True)
        return
    from bench_report import make_report
    make_report(output)
    print(f'Report: {output / "index.html"}', flush=True)
    if run['failures'] or run.get('interrupted') or (a.mode == 'count-check' and skipped):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
