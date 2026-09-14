"""Model-assisted box review and explicit 101-point interpolated AP evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from bench_core import read_json, save_json, save_csv


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x*y
    return intersection / max(1e-12, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def fuse(predictions, conf=.25, threshold=.5):
    clusters = []
    candidates = [(model, b) for model, boxes in predictions.items() for b in boxes if b[4] >= conf]
    for model, b in sorted(candidates, key=lambda x: (-x[1][4], x[0])):
        matches = [(iou(b, c['box']), i) for i, c in enumerate(clusters)
                   if c['class_id'] == int(b[5]) and model not in c['models']]
        overlap, index = max(matches, default=(0, -1))
        if overlap >= threshold:
            c = clusters[index]
            weight = c['weight'] + b[4]
            c['box'] = [(x*c['weight']+y*b[4])/weight for x, y in zip(c['box'], b[:4])]
            c['weight'] = weight
            c['models'].append(model)
        else:
            clusters.append(dict(box=b[:4], class_id=int(b[5]), models=[model], weight=b[4]))
    return [dict(box=c['box'], class_id=c['class_id'], models=c['models']) for c in clusters]


def make_review(root):
    root = Path(root)
    run, videos = read_json(root/'run.json'), read_json(root/'dataset.json')
    frames = []
    for v in videos:
        predictions, records = {}, []
        for model in run['selected_models']:
            record = read_json(root/'ensemble-review'/model/'pass_1'/v['id']/'detections.json')
            predictions[model] = record['boxes']
            records.append(record)
        if len({r['frame_sha256'] for r in records}) != 1:
            raise ValueError('Models received different source images')
        r = records[0]
        frames.append(dict(id=v['id'], group=v['group'], camera=v['camera'], frame=v['frame'],
                           frame_sha256=r['frame_sha256'], width=r['width'], height=r['height'],
                           image=f'ensemble-review/{run["selected_models"][0]}/pass_1/{v["id"]}/source.png',
                           reviewed=False, boxes=fuse(predictions), predictions=predictions))
    bundle = dict(schema=1, status='proposals_not_ground_truth', proposal_conf=.25, merge_iou=.5,
                  prediction_conf=.001, max_det=run['settings']['max_det'], operating_conf=run['settings']['conf'],
                  models=run['selected_models'], frames=frames)
    bundle['review_id'] = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    save_json(root/'ensemble.json', bundle)
    template = Path(__file__).with_name('ensemble_review.html').read_text(encoding='utf-8')
    (root/'review.html').write_text(template.replace('__DATA__', json.dumps(bundle).replace('<', '\\u003c')), encoding='utf-8')


def evaluate(predictions, truth, cls, threshold, conf=0):
    gt = {k: [b['box'] for b in boxes if b['class_id'] == cls] for k, boxes in truth.items()}
    n = sum(map(len, gt.values()))
    entries = sorted([(b[4], key, b) for key, boxes in predictions.items() for b in boxes
                      if int(b[5]) == cls and b[4] >= conf], key=lambda x: -x[0])
    used = {k: set() for k in gt}
    tp = []
    for _, key, box in entries:
        matches = [(iou(box, g), i) for i, g in enumerate(gt[key]) if i not in used[key]]
        overlap, idx = max(matches, default=(0, -1))
        hit = overlap >= threshold
        if hit:
            used[key].add(idx)
        tp.append(int(hit))
    cumulative = np.cumsum(tp)
    recall = cumulative / n if n else np.zeros(len(tp))
    precision = cumulative / np.arange(1, len(tp)+1)
    ap = float(np.mean([max(precision[recall >= t], default=0) for t in np.linspace(0, 1, 101)])) if n else None
    hits = int(sum(tp))
    return dict(ap=ap, precision=hits/len(tp) if tp else 0., recall=hits/n if n else None,
                tp=hits, fp=len(tp)-hits, fn=n-hits, ground_truth=n)


def score(root, truth_path):
    import cv2
    bundle, reviewed = read_json(root/'ensemble.json'), read_json(truth_path)
    if reviewed.get('review_id') != bundle['review_id'] or reviewed.get('status') != 'human_reviewed':
        raise ValueError('Export confirmed annotations from the matching review.html first')
    frames = reviewed['frames']
    mapping = {f['id']: f for f in frames}
    if len(mapping) != len(frames) or set(mapping) != {f['id'] for f in bundle['frames']}:
        raise ValueError('Reviewed image IDs do not match')
    for original in bundle['frames']:
        f = mapping[original['id']]
        image = cv2.imread(str(root/original['image']))
        if image is None or hashlib.sha256(image.data).hexdigest() != original['frame_sha256']:
            raise ValueError('Source image was modified')
        if f.get('reviewed') is not True or f.get('frame_sha256') != original['frame_sha256']:
            raise ValueError(f'Image not confirmed or hash mismatch: {original["id"]}')
        for b in f['boxes']:
            box = b['box']
            if type(b['class_id']) is not int or b['class_id'] not in (0,1) or len(box) != 4 or not np.isfinite(box).all():
                raise ValueError('Invalid annotation')
            if not (0 <= box[0] < box[2] <= original['width'] and 0 <= box[1] < box[3] <= original['height']):
                raise ValueError('Box outside image or zero area')
    rows = []
    for group in ('all', 'night', 'day'):
        selected = [f for f in bundle['frames'] if group == 'all' or f['group'] == group]
        if not selected:
            continue
        truth = {f['id']: mapping[f['id']]['boxes'] for f in selected}
        for model in bundle['models']:
            predictions = {f['id']: f['predictions'][model] for f in selected}
            class_rows = []
            for cls, label in enumerate(('car', 'motorcycle')):
                aps = [evaluate(predictions, truth, cls, float(t))['ap'] for t in np.linspace(.5,.95,10)]
                operating = evaluate(predictions, truth, cls, .5, bundle['operating_conf'])
                row = dict(model=model, group=group, label=label, images=len(selected), **operating,
                           AP50=aps[0], AP50_95=float(np.mean(aps)) if aps[0] is not None else None)
                rows.append(row)
                class_rows.append(row)
            supported = [r for r in class_rows if r['ground_truth']]
            total_tp = sum(r['tp'] for r in class_rows)
            total_fp = sum(r['fp'] for r in class_rows)
            total_fn = sum(r['fn'] for r in class_rows)
            total_pred = total_tp + total_fp
            total_gt = total_tp + total_fn
            total_precision = total_tp / total_pred if total_pred else 0.0
            total_recall = total_tp / total_gt if total_gt else None
            combined_ap50 = float(np.mean([r['AP50'] for r in supported])) if supported else None
            combined_ap = float(np.mean([r['AP50_95'] for r in supported])) if supported else None
            rows.append(dict(model=model, group=group, label='all', images=len(selected),
                             precision=total_precision, recall=total_recall,
                             f1=(2 * total_precision * total_recall / (total_precision + total_recall)) if total_recall is not None and total_precision + total_recall else 0.0,
                             tp=total_tp, fp=total_fp, fn=total_fn, ground_truth=total_gt,
                             AP50=combined_ap50, AP50_95=combined_ap,
                             mAP50=combined_ap50, mAP50_95=combined_ap))
    for group in ('all', 'night', 'day'):
        ranked = sorted([r for r in rows if r['group']==group and r['label']=='all' and r['mAP50_95'] is not None], key=lambda r:-r['mAP50_95'])
        for r in ranked:
            r['rank'] = 1 + sum(x['mAP50_95'] > r['mAP50_95'] for x in ranked)
    save_csv(root/'accuracy.csv', rows)
    save_json(root/'accuracy.json', dict(metrics=rows, truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        method='101-point interpolated AP at IoU .50:.05:.95; no crowd/ignore/area evaluation; absent GT classes excluded from mAP; P/R at configured confidence and IoU .5',
        limitation='Human-reviewed model-assisted labels on selected frames; independent annotation preferred for final claims. Predictions capped at configured max_det and confidence .001.'))
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ensemble accuracy</title>
<style>body{font:15px system-ui;margin:24px;background:#f4f7fb;color:#182438}table{border-collapse:collapse;background:#fff;width:100%;margin:10px 0 28px}th,td{border:1px solid #ccd4df;padding:7px;text-align:right;white-space:nowrap}th{cursor:pointer;background:#e9eef5}th:first-child,td:first-child{text-align:left}.controls{position:sticky;top:0;background:#f4f7fb;padding:12px 0;z-index:2}select{padding:7px;margin-right:8px}.group{margin-top:28px}small{color:#52627a}</style>
<h1>Accuracy against human-reviewed boxes</h1><p>Model-assisted annotations; AP uses 101 recall points. Precision/Recall use operating confidence and IoU 0.5. Click a column heading to sort; use filters to compare models or classes.</p><p><b>Why a value can be —:</b> AP/mAP is undefined when the selected group has no ground-truth objects for that class. Class rows show AP for that class; the <code>all</code> row shows combined precision, recall, F1, AP50 and AP50:95, with mAP values averaged across classes that have ground truth. A blank metric is not the same as zero.</p>
<div class="controls"><label>Group <select id="group"><option value="all">All</option><option value="night">Night</option><option value="day">Day</option></select></label><label>Model <select id="model"><option value="">all</option></select></label><label>Class <select id="label"><option value="">all</option><option>all</option><option>car</option><option>motorcycle</option></select></label><label>Sort by <select id="sort"><option value="mAP50_95">mAP50:95</option><option value="mAP50">mAP50</option><option value="AP50_95">AP50:95</option><option value="AP50">AP50</option><option value="precision">Precision</option><option value="recall">Recall</option><option value="f1">F1</option><option value="rank">Rank</option><option value="model">Model</option></select></label><label>Direction <select id="direction"><option value="desc">High to low</option><option value="asc">Low to high</option></select></label><button id="reset">Reset sort</button></div><div id="tables"></div>
<script>const rows=__ROWS__,cols=['model','group','label','images','precision','recall','f1','AP50','AP50_95','mAP50','mAP50_95','rank'];let sortKey='mAP50_95',descending=true;const $=id=>document.getElementById(id);[...new Set(rows.map(r=>r.model))].sort().forEach(m=>$('model').add(new Option(m,m)));
function fmt(v){return v==null?'—':typeof v==='number'?(Math.abs(v)<2?v.toFixed(4):v.toFixed(2)):v}function render(){const group=$('group').value,model=$('model').value,label=$('label').value;sortKey=$('sort').value;descending=$('direction').value==='desc';let selected=rows.filter(r=>(r.group===group)&&(!model||r.model===model)&&(!label||r.label===label));selected.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return-1;return (x<y?-1:x>y?1:0)*(descending?-1:1)});const heading=cols.map(c=>`<th data-key="${c}">${c}${sortKey===c?(descending?' ↓':' ↑'):''}</th>`).join('');let body=selected.map(r=>'<tr>'+cols.map(c=>`<td>${fmt(r[c])}</td>`).join('')+'</tr>').join('');$('tables').innerHTML=`<section class="group"><h2>${group==='all'?'All':group[0].toUpperCase()+group.slice(1)} <small>${selected.length} rows</small></h2><table><thead><tr>${heading}</tr></thead><tbody>${body||'<tr><td colspan="12">No matching results</td></tr>'}</tbody></table></section>`;document.querySelectorAll('th[data-key]').forEach(th=>th.onclick=()=>{const k=th.dataset.key;$('sort').value=k;if(sortKey===k)descending=!descending;else descending=!['model','group','label'].includes(k);$('direction').value=descending?'desc':'asc';render()})}['group','model','label','sort','direction'].forEach(id=>$(id).onchange=render);$('reset').onclick=()=>{$('sort').value='mAP50_95';$('direction').value='desc';render()};render();</script></html>'''.replace('__ROWS__', payload)
    (root/'accuracy.html').write_text(html, encoding='utf-8')
    print(f'Accuracy report: {root / "accuracy.html"}')
