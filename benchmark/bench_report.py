"""Offline HTML/Markdown/CSV reports, shared-scale heat maps and paired comparisons."""
from __future__ import annotations
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np

from bench_core import CLASSES, box_agreement, count_metrics, read_json, save_csv, save_json, stats


NOTES = """Detection FPS includes preprocessing, forward, postprocessing and transfer of normalized boxes to CPU;
decode, warm-up, loading, hashing, analysis and writing are outside that interval. Stage times come from
Ultralytics' synchronized profiler; forward includes embedded NMS for engines exported that way.
Runtime sampling includes decode/hash/analysis activity: CPU, GPU utilization, power and RSS describe the
measurement loop, not exclusively the model call. Device VRAM includes other applications. WDDM may
not expose process VRAM. Missing readings are N/A, never zero. Device memory changes are provisional.
Heat maps count repeated bottom-center detections per source frame, not unique vehicles or detectability.
Export agreement is not ground-truth accuracy. Count checks evaluate counts only; missed and spurious
boxes can cancel within a class. Precision, Recall and mAP are unavailable without bounding-box labels.
One selected frame per clip is a small spot check. Confidence sweeps are exploratory, not threshold tuning
on a held-out test set. No combined accuracy/speed winner is computed. FP32/FP16/INT8 rows represent
different deployment settings; same-precision comparisons are identified separately.
Source seconds use reported FPS; capture cadence is not independently verified. Timed repeats describe
run-to-run variation, not independent observations of each frame. No simultaneous-camera claim is made."""
NOTES += "\nVRAM/resource peaks are sampled and may miss brief transients. Precision cohorts use declared precision; compiled TensorRT tactics/TF32 cannot be changed after export. PyTorch TF32 is disabled."


def number(value):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def table(rows, columns, markdown=False):
    if not rows:
        return 'No completed eligible results.'
    if markdown:
        lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        return '\n'.join(lines + ['| ' + ' | '.join(number(r.get(k)).replace('|', '/') for k in columns) + ' |' for r in rows])
    head = '<tr>' + ''.join(f'<th>{html.escape(k)}</th>' for k in columns) + '</tr>'
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(number(r.get(k)))}</td>' for k in columns) + '</tr>' for r in rows)
    return '<div class="scroll"><table>' + head + body + '</table></div>'


def plot_timelines(path, rows, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arr = np.load(path / 'frames.npz')
    t, counts = arr['timings'], arr['counts']
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t[:, 4], linewidth=.6, label='Complete detection')
    axes[0].plot(t[:, 2], linewidth=.6, label='Forward')
    axes[0].axhline(1000 / target, color='red', linestyle='--', label=f'{target:g} FPS budget')
    axes[0].set_ylabel('ms')
    axes[0].legend()
    for k, label in enumerate(CLASSES):
        axes[1].plot(counts[:, k], linewidth=.7, label=label)
    axes[1].legend()
    axes[1].set_ylabel('Detections')
    axes[1].set_xlabel('Selected segment frame (zero-based)')
    fig.suptitle(f'{rows["model"]} / {rows["video"]} / pass {rows["pass_index"] + 1}')
    fig.tight_layout()
    fig.savefig(path / 'timeline.png', dpi=120)
    plt.close(fig)


def optional_stat(rows, key, method):
    v = [r['resource'].get(key) for r in rows if r['resource'].get(key) is not None]
    return float(max(v) if method == 'max' else np.mean(v)) if v else None


def aggregate(rows, root, expected_videos, repeats):
    out = []
    for model in dict.fromkeys(r['model'] for r in rows):
        for group in ('day', 'night', 'all'):
            selected = [v for v in expected_videos if group == 'all' or v['group'] == group]
            if not selected:
                continue
            g = [r for r in rows if r['model'] == model and (group == 'all' or r['group'] == group)]
            if not g:
                continue
            expected = {(v['id'], i) for v in selected for i in range(repeats)}
            complete = {(r['video'], r['pass_index']) for r in g} == expected
            arrays = [np.load(root / r['_path'] / 'frames.npz')['timings'] for r in g]
            t = np.concatenate(arrays)
            frame_count = len(t)
            r = dict(model=model, group=group, complete=complete,
                     comparable=all(x['comparable'] for x in g), precision=g[0]['precision'],
                     measured_passes=repeats, video_runs=len(g), frames_all_passes=frame_count,
                     unique_source_frames=sum(v['selected_frames'] for v in selected),
                     inference_seconds=float(t[:, 2].sum() / 1000), inference_ms=float(t[:, 2].mean()),
                     inference_fps=float(frame_count * 1000 / t[:, 2].sum()),
                     detection_seconds=float(t[:, 4].sum() / 1000), detection_ms=float(t[:, 4].mean()),
                     detection_fps=float(frame_count * 1000 / t[:, 4].sum()),
                     detection_p50_ms=float(np.median(t[:, 4])), detection_p95_ms=float(np.percentile(t[:, 4], 95)),
                     detection_p99_ms=float(np.percentile(t[:, 4], 99)),
                     mean_video_detection_ms=float(np.mean([x['detection_ms']['mean'] for x in g])),
                     car_detections_all_passes=sum(x['car_detections'] for x in g),
                     motorcycle_detections_all_passes=sum(x['motorcycle_detections'] for x in g),
                     car_detections_canonical_pass=sum(x['car_detections'] for x in g if x['pass_index'] == 0),
                     motorcycle_detections_canonical_pass=sum(x['motorcycle_detections'] for x in g if x['pass_index'] == 0))
            for col, idx in [('decode', 0), ('preprocess', 1), ('postprocess', 3)]:
                r[col + '_seconds'] = float(t[:, idx].sum() / 1000)
            target = read_json(root / 'run.json')['settings']['target_fps']
            r['target_budget_exceed_percent'] = float((t[:, 4] > 1000 / target).mean() * 100)
            r['source_budget_exceed_percent'] = sum(x['source_budget_exceed_percent'] * x['frames'] for x in g) / frame_count
            pass_fps = []
            for i in range(repeats):
                p = [x for x in g if x['pass_index'] == i]
                if {x['video'] for x in p} == {v['id'] for v in selected}:
                    pass_fps.append(sum(x['frames'] for x in p) * 1000 / sum(x['detection_ms']['total'] for x in p))
            r['pass_fps_mean'] = float(np.mean(pass_fps)) if pass_fps else None
            r['pass_fps_sample_std'] = float(np.std(pass_fps, ddof=1)) if len(pass_fps) > 1 else None
            for metric in ('process_ram_MiB', 'process_vram_MiB', 'device_vram_MiB', 'process_cpu_percent',
                           'system_cpu_percent', 'system_ram_percent', 'gpu_percent', 'temperature_C', 'graphics_clock_MHz', 'power_W'):
                r[metric + '_peak'] = optional_stat(g, metric + '_peak', 'max')
                r[metric + '_mean'] = optional_stat(g, metric + '_mean', 'mean')
            deltas = [x['resource']['device_vram_MiB_peak'] - x['before_load']['device_vram_MiB'] for x in g
                      if x['resource'].get('device_vram_MiB_peak') is not None and x['before_load'].get('device_vram_MiB') is not None]
            r['device_vram_increase_over_preload_MiB'] = max(deltas) if deltas else None
            out.append(r)
    return out


def create_heats(root, rows, reference):
    gallery, regional = [], []
    canonical = [r for r in rows if r['pass_index'] == 0 and (root / r['_path'] / 'heat.npz').exists()]
    for video in dict.fromkeys(r['video'] for r in canonical):
        group = [r for r in canonical if r['video'] == video]
        maps, backgrounds = {}, {}
        maxima = [0., 0., 0.]
        for r in group:
            path = root / r['_path']
            raw = np.load(path / 'heat.npz')['counts']
            scaled = raw.astype(np.float32) / r['frames']
            sigma = max(1., raw.shape[1] / 1080 * 8)
            smooth = [cv2.GaussianBlur(x, (0, 0), sigma) for x in (scaled[0], scaled[1], scaled.sum(axis=0))]
            maps[r['model']] = (raw, smooth)
            backgrounds[r['model']] = cv2.imread(str(path / 'background.png'))
            maxima = [max(m, float(x.max())) for m, x in zip(maxima, smooth)]
        contact = [[], [], []]
        for r in group:
            path = root / r['_path']
            raw, smooth = maps[r['model']]
            sparse = []
            for channel in raw:
                yy, xx = np.nonzero(channel)
                sparse.append([[int(x), int(y), int(channel[y, x])] for y, x in zip(yy, xx)])
            links = []
            for k, label in enumerate((*CLASSES, 'combined')):
                ratio = smooth[k] / max(maxima[k], 1e-12)
                color = cv2.applyColorMap(np.uint8(np.clip(ratio * 255, 0, 255)), cv2.COLORMAP_TURBO)
                alpha = (ratio > 0.001).astype(np.float32)[..., None] * .60
                bg = backgrounds[r['model']]
                overlay = np.uint8(bg * (1 - alpha) + color * alpha)
                filename = f'heat_{label}.png'
                cv2.imwrite(str(path / filename), overlay)
                panel = cv2.resize(overlay, (480, round(480 * bg.shape[0] / bg.shape[1])))
                cv2.rectangle(panel, (0, 0), (480, 27), (0, 0, 0), -1)
                cv2.putText(panel, f'{r["model"]} / {label}', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                contact[k].append(panel)
                links.append(f'<figure><img src="{filename}" data-channel="{k}"><figcaption>{label}: shared smoothed scale 0..{maxima[k]:.6g} detections/frame/pixel</figcaption></figure>')
            viewer = f'''<!doctype html><meta charset="utf-8"><title>{html.escape(video)} heat density</title>
<style>body{{font:16px sans-serif}}img{{max-width:100%;cursor:crosshair}}#value{{position:sticky;top:0;background:white;padding:10px}}</style>
<h1>{html.escape(r['model'])} / {html.escape(video)}</h1><p>Repeated bottom-center detections. Hover/click for ORIGINAL-pixel raw density; images are smoothed.</p>
<div id="value">Select an image point</div>{''.join(links)}<script>
const raw={json.dumps(sparse)}, width={raw.shape[2]}, height={raw.shape[1]}, frames={r['frames']};
const maps=raw.map(a=>new Map(a.map(([x,y,n])=>[y*width+x,n])));
for(const img of document.querySelectorAll('img')){{ const show=e=>{{const b=img.getBoundingClientRect();
const x=Math.min(width-1,Math.max(0,Math.floor((e.clientX-b.left)*width/b.width)));
const y=Math.min(height-1,Math.max(0,Math.floor((e.clientY-b.top)*height/b.height)));
const k=Number(img.dataset.channel), counts=maps.map(m=>m.get(y*width+x)||0), n=k===2?counts[0]+counts[1]:counts[k];
document.getElementById('value').textContent=`x=${{x}}, y=${{y}}: ${{n}} detections; ${{(n/frames).toFixed(6)}} per frame`;}};
img.onmousemove=show;img.onclick=show;}}
</script>'''
            (path / 'heat_viewer.html').write_text(viewer, encoding='utf-8')
            gallery.append(dict(kind='heat', model=r['model'], video=video, group=r['group'], camera=r['camera'],
                                image=str((path / 'heat_combined.png').relative_to(root)).replace('\\', '/'),
                                link=str((path / 'heat_viewer.html').relative_to(root)).replace('\\', '/')))
            for region, counts in r.get('region_counts', {}).items():
                regional.append(dict(model=r['model'], video=video, region=region,
                                     car_detections=counts[0], motorcycle_detections=counts[1],
                                     cars_per_frame=counts[0] / r['frames'], motorcycles_per_frame=counts[1] / r['frames']))
        for k, label in enumerate((*CLASSES, 'combined')):
            cv2.imwrite(str(root / f'contact_{video}_{label}.png'), np.concatenate(contact[k], axis=1))
        if reference in maps:
            for r in group:
                if r['model'] == reference:
                    continue
                path = root / r['_path']
                arrays = [maps[r['model']][1][k] - maps[reference][1][k] for k in range(3)]
                for k, label in enumerate((*CLASSES, 'combined')):
                    # Signed per-frame smoothed density difference; shared symmetric scale.
                    v = arrays[k] / max(maxima[k], 1e-12)
                    white = np.ones((*v.shape, 3), dtype=np.float32) * 255
                    positive, negative = np.clip(v, 0, 1), np.clip(-v, 0, 1)
                    white[:, :, 0] *= (1 - positive)
                    white[:, :, 1] *= (1 - np.abs(v))
                    white[:, :, 2] *= (1 - negative)
                    cv2.imwrite(str(path / f'difference_{label}_vs_{reference}.png'), white.astype(np.uint8))
                np.savez_compressed(path / f'difference_vs_{reference}.npz', car=arrays[0], motorcycle=arrays[1], combined=arrays[2])
                gallery.append(dict(kind='difference', model=r['model'], video=video, group=r['group'], camera=r['camera'],
                                    image=f'{r["_path"]}/difference_combined_vs_{reference}.png',
                                    link=f'{r["_path"]}/difference_combined_vs_{reference}.png'))
    save_csv(root / 'regions.csv', regional)
    return gallery


def paired_results(root, summaries, artifacts, videos):
    pairs, agreement_rows = [], []
    for b in artifacts.values():
        if b['format'] != 'engine' or not b.get('pair_id'):
            continue
        for a in artifacts.values():
            if a['format'] != 'pt' or a.get('pair_id') != b['pair_id']:
                continue
            for group in ('day', 'night', 'all'):
                x = next((r for r in summaries if r['model'] == a['id'] and r['group'] == group and r['complete']), None)
                y = next((r for r in summaries if r['model'] == b['id'] and r['group'] == group and r['complete']), None)
                if x and y:
                    pairs.append(dict(pt=a['id'], engine=b['id'], group=group,
                                      comparison='same precision backend' if a['precision'] == b['precision'] else 'different precision deployment',
                                      thresholds_comparable=x['comparable'] and y['comparable'],
                                      detection_speedup=y['detection_fps'] / x['detection_fps'],
                                      forward_speedup=y['inference_fps'] / x['inference_fps'],
                                      device_peak_difference_MiB=(y['device_vram_MiB_peak'] - x['device_vram_MiB_peak']) if y['device_vram_MiB_peak'] is not None and x['device_vram_MiB_peak'] is not None else None,
                                      car_count_difference=y['car_detections_canonical_pass'] - x['car_detections_canonical_pass'],
                                      motorcycle_count_difference=y['motorcycle_detections_canonical_pass'] - x['motorcycle_detections_canonical_pass']))
            for video in videos:
                left = root / 'performance' / a['id'] / 'pass_1' / video['id'] / 'predictions.npz'
                right = root / 'performance' / b['id'] / 'pass_1' / video['id'] / 'predictions.npz'
                if not left.is_file() or not right.is_file():
                    continue
                aa, bb = np.load(left), np.load(right)
                a_boxes, a_offsets = aa['boxes'], aa['offsets']
                b_boxes, b_offsets = bb['boxes'], bb['offsets']
                aa.close()
                bb.close()
                if len(a_offsets) != len(b_offsets):
                    raise ValueError('Pair has different frame counts')
                total = dict(matched=0, unmatched_a=0, unmatched_b=0, sum_iou=0., sum_confidence_delta=0.)
                for i in range(len(a_offsets) - 1):
                    match = box_agreement(a_boxes[a_offsets[i]:a_offsets[i + 1]], b_boxes[b_offsets[i]:b_offsets[i + 1]])
                    for k in total:
                        total[k] += match[k]
                agreement_rows.append(dict(pt=a['id'], engine=b['id'], video=video['id'], **total,
                                           mean_iou=total['sum_iou'] / total['matched'] if total['matched'] else None,
                                           mean_confidence_difference=total['sum_confidence_delta'] / total['matched'] if total['matched'] else None))
    save_csv(root / 'paired_comparison.csv', pairs)
    save_csv(root / 'export_agreement.csv', agreement_rows)
    return pairs, agreement_rows


def make_report(root):
    root = Path(root)
    start = time.perf_counter()
    run, config, videos = (read_json(root / f) for f in ('run.json', 'resolved_config.json', 'dataset.json'))
    if run.get('invalid_comparison'):
        raise ValueError(run['invalid_comparison'])
    rows, checks, artifacts, gallery = [], [], {}, []
    for path in sorted(root.glob('**/artifact.json')):
        a = read_json(path)
        artifacts[a['id']] = a
    for a in artifacts.values():
        if a['parameters'] is None and a.get('pair_id'):
            source = next((b for b in artifacts.values() if b['format'] == 'pt' and b.get('pair_id') == a['pair_id']), None)
            if source:
                a['parameters'] = source['parameters']
                a['parameters_source'] = f'Paired checkpoint {source["id"]}, pairing declared by manifest'
    for path in sorted(root.glob('performance/*/pass_*/*/result.json')):
        r = read_json(path)
        r['_path'] = path.parent.relative_to(root).as_posix()
        rows.append(r)
        plot_timelines(path.parent, r, run['settings']['target_fps'])
        gallery.append(dict(kind='timeline', model=r['model'], video=r['video'], group=r['group'], camera=r['camera'],
                            image=f'{r["_path"]}/timeline.png', link=f'{r["_path"]}/timeline.png'))
    summaries = aggregate(rows, root, videos, run['settings']['repeats'])
    # A failed/interrupted worker cannot enter rankings, even if some clip checkpoints survived.
    for r in summaries:
        jobs = [j for j in run['jobs'] if j['mode'] == 'performance' and j['model'] == r['model']]
        r['complete'] = r['complete'] and len(jobs) == run['settings']['repeats'] and all(j['status'] == 'complete' for j in jobs)
    flat_rows = []
    for r in rows:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for key in ('decode_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'detection_ms', 'resource', 'before_load', 'after_load', 'after_warmup'):
            if key not in r:
                continue
            flat.update({f'{key}_{k}': v for k, v in r[key].items()})
        flat_rows.append(flat)
    save_csv(root / 'per_video_pass.csv', flat_rows)
    save_csv(root / 'comparison.csv', summaries)
    averages = []
    for model in artifacts:
        for v in videos:
            g = [r for r in rows if r['model'] == model and r['video'] == v['id']]
            if not g:
                continue
            frames = sum(r['frames'] for r in g)
            averages.append(dict(model=model, video=v['id'], group=v['group'], measured_passes=len(g),
                                 frames_per_pass=frames / len(g), inference_seconds_mean=np.mean([r['inference_ms']['total'] / 1000 for r in g]),
                                 inference_ms=sum(r['inference_ms']['total'] for r in g) / frames,
                                 inference_fps=frames * 1000 / sum(r['inference_ms']['total'] for r in g),
                                 detection_ms=sum(r['detection_ms']['total'] for r in g) / frames,
                                 detection_fps=frames * 1000 / sum(r['detection_ms']['total'] for r in g),
                                 mean_car_detections=np.mean([r['car_detections'] for r in g]),
                                 mean_motorcycle_detections=np.mean([r['motorcycle_detections'] for r in g])))
    save_csv(root / 'per_video_average.csv', averages)
    if rows:
        gallery.extend(create_heats(root, rows, config['reference_model']))
    for path in sorted(root.glob('count-check/*/pass_1/checkpoint.json')):
        records = read_json(path)
        checks.extend(records)
        for r in records:
            if r['primary']:
                image = (path.parent / r['video'] / f'conf_{r["confidence"]:.4f}.png').relative_to(root).as_posix()
                gallery.append(dict(kind='count', model=r['model'], video=r['video'], group=r['group'], camera=r['camera'], image=image, link=image))
    count_summary = []
    for model in dict.fromkeys(r['model'] for r in checks):
        for group in ('day', 'night', 'all'):
            for conf in sorted(set(r['confidence'] for r in checks)):
                g = [r for r in checks if r['model'] == model and r['confidence'] == conf and (group == 'all' or r['group'] == group)]
                if not g:
                    continue
                expected = {v['id'] for v in videos if group == 'all' or v['group'] == group}
                complete = {r['video'] for r in g} == expected and all(j['status'] == 'complete' for j in run['jobs'] if j['mode'] == 'count-check' and j['model'] == model)
                m = count_metrics(g)
                row = dict(model=model, group=group, confidence=conf, images=len(g), complete=complete,
                           comparable=all(r['comparable'] for r in g), primary=conf == config['settings']['conf'], total_mae=m['total_mae'])
                row.update({f'{label}_{k}': value for label in CLASSES for k, value in m[label].items()})
                count_summary.append(row)
    save_csv(root / 'count_per_frame.csv', checks)
    save_csv(root / 'count_summary.csv', count_summary)
    pairs, agreement = paired_results(root, summaries, artifacts, videos)
    rankings = []
    for group in ('day', 'night', 'all'):
        eligible = [r for r in summaries if r['complete'] and r['comparable'] and r['group'] == group]
        for cohort in ['deployment_all_precisions'] + sorted({r['precision'] for r in eligible}):
            candidates = [r for r in eligible if cohort == 'deployment_all_precisions' or r['precision'] == cohort]
            for label, metric, desc in [('fastest_detection', 'detection_fps', True), ('fastest_forward', 'inference_fps', True),
                                        ('lowest_process_VRAM', 'process_vram_MiB_peak', False),
                                        ('lowest_device_VRAM_increase_provisional', 'device_vram_increase_over_preload_MiB', False),
                                        ('lowest_process_RAM', 'process_ram_MiB_peak', False), ('lowest_process_CPU', 'process_cpu_percent_mean', False)]:
                g = [r for r in candidates if r.get(metric) is not None]
                if label == 'fastest_forward':
                    g = [r for r in g if not artifacts[r['model']].get('nms_free_architecture')]
                for r in sorted(g, key=lambda x: x[metric], reverse=desc):
                    rank = 1 + sum((x[metric] > r[metric] if desc else x[metric] < r[metric]) for x in g)
                    rankings.append(dict(group=group, cohort=cohort, metric=label, rank=rank, model=r['model'], value=r[metric]))
        g = [r for r in count_summary if r['group'] == group and r['primary'] and r['complete'] and r['comparable']]
        for r in sorted(g, key=lambda x: x['total_mae']):
            rankings.append(dict(group=group, cohort='count_spot_check', metric='lowest_count_MAE',
                                 rank=1 + sum(x['total_mae'] < r['total_mae'] for x in g), model=r['model'], value=r['total_mae']))
    save_csv(root / 'rankings.csv', rankings)
    save_json(root / 'report_data.json', dict(run=run, models=list(artifacts.values()), per_video=rows,
                                            comparison=summaries, count_summary=count_summary, rankings=rankings,
                                            paired_comparison=pairs, agreement=agreement, gallery=gallery))
    columns = ['model', 'group', 'complete', 'precision', 'inference_ms', 'inference_fps', 'detection_fps',
               'detection_p95_ms', 'pass_fps_sample_std', 'device_vram_MiB_peak']
    counts_columns = ['model', 'group', 'confidence', 'complete', 'total_mae', 'car_mae', 'motorcycle_mae', 'car_wape_percent', 'motorcycle_wape_percent']
    rank_columns = ['group', 'cohort', 'metric', 'rank', 'model', 'value']
    title = 'SMOKE TEST — NOT A PERFORMANCE CONCLUSION' if run['smoke'] else 'YOLO day/night benchmark'
    status = json.dumps(dict(skipped=run['skipped'], failures=run['failures'], interrupted=run.get('interrupted', False)), indent=2)
    environments = [read_json(p) for p in sorted(root.glob('**/environment.json'))]
    env = environments[0] if environments else {}
    markdown = f'# {title}\n\n## Status\n\n```json\n{status}\n```\n\n## Method\n\n{NOTES}\n\n## Settings\n\n```json\n{json.dumps(run["settings"], indent=2)}\n```\n\n## Comparison\n\n{table(summaries, columns, True)}\n\n## Count check\n\n{table(count_summary, counts_columns, True)}\n\n## Rankings\n\n{table(rankings, rank_columns, True)}\n\n## Environment\n\n```json\n{json.dumps(env, indent=2)}\n```\n\nSee index.html for images and filters; CSV/JSON files for every metric and per-video averages.\n'
    (root / 'REPORT.md').write_text(markdown, encoding='utf-8')
    cards = []
    for g in gallery:
        attrs = ' '.join(f'data-{k}="{html.escape(str(g[k]), quote=True)}"' for k in ('model', 'group', 'camera', 'kind'))
        cards.append(f'<article {attrs}><h3>{html.escape(g["model"])} / {html.escape(g["video"])} / {g["kind"]}</h3><a href="{html.escape(g["link"], quote=True)}"><img loading="lazy" src="{html.escape(g["image"], quote=True)}"></a></article>')
    filters = ''
    for field in ('model', 'group', 'camera', 'kind'):
        options = ''.join(f'<option>{html.escape(v)}</option>' for v in sorted({g[field] for g in gallery}))
        filters += f'<label>{field} <select id="{field}"><option value="">all</option>{options}</select></label> '
    assets = [p for pattern in ('*.csv', '*.json', 'contact_*.png') for p in sorted(root.glob(pattern))]
    downloads = ' · '.join(f'<a href="{p.name}">{html.escape(p.name)}</a>' for p in assets)
    paircols = ['pt', 'engine', 'group', 'comparison', 'thresholds_comparable', 'detection_speedup', 'device_peak_difference_MiB']
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{title}</title>
<style>article[hidden]{{display:none!important}}</style>
<style>body{{font:15px system-ui;margin:24px;background:#f5f7fb;color:#182438}}table{{border-collapse:collapse;background:white}}td,th{{padding:8px;border:1px solid #ccd4df;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}article{{display:inline-block;vertical-align:top;width:min(600px,100%);margin:8px}}img{{width:100%}}pre{{white-space:pre-wrap}}select{{padding:8px}}.filters{{position:sticky;top:0;background:#f5f7fb;padding:10px}}a{{color:#164db5}}</style>
<h1>{title}</h1><h2>Run status</h2><pre>{html.escape(status)}</pre><h2>Method and limitations</h2><pre>{html.escape(NOTES)}</pre>
<h2>Settings</h2><pre>{html.escape(json.dumps(run['settings'], indent=2))}</pre>
<h2>Models</h2>{table(list(artifacts.values()), ['id','version','format','precision','file_MiB','parameters','parameters_source','comparable'])}
<h2>Performance</h2>{table(summaries, columns)}<h2>Per-video averages</h2>{table(averages, ['model','video','measured_passes','inference_ms','detection_ms','detection_fps','mean_car_detections','mean_motorcycle_detections'])}
<h2>Count spot check</h2>{table(count_summary, counts_columns)}<h2>Paired exports</h2>{table(pairs,paircols)}
<h2>Rankings</h2>{table(rankings, rank_columns)}<h2>Files / contact sheets</h2><p>{downloads}</p>
<h2>Visual comparisons</h2><p>Difference maps: red = more detections than reference; blue = fewer. Heat images link to original-pixel density viewers.</p>
<div class="filters">{filters}</div>{''.join(cards)}<h2>Environment</h2><pre>{html.escape(json.dumps(env, indent=2))}</pre>
<p>References: <a href="https://docs.ultralytics.com/modes/predict/">Ultralytics prediction timing</a>; <a href="https://docs.nvidia.com/deploy/nvml-api/structnvmlProcessInfo__v1__t.html">NVML WDDM memory limits</a>.</p>
<script>function filter(){{for(const card of document.querySelectorAll('article')){{card.hidden=!['model','group','camera','kind'].every(k=>!document.getElementById(k).value||card.dataset[k]===document.getElementById(k).value);}}}}for(const s of document.querySelectorAll('select'))s.onchange=filter;</script></html>'''
    (root / 'index.html').write_text(doc, encoding='utf-8')
    save_json(root / 'report_generation.json', dict(seconds=time.perf_counter() - start, excluded_from_inference=True))
