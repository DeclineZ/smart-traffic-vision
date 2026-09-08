"""CPU-only regression tests; optional GPU integration fixture generator below."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

BENCH = Path(__file__).resolve().parents[1] / 'benchmark'
sys.path.insert(0, str(BENCH))
from bench_core import box_agreement, count_metrics, heat_points, load_config, save_json, stats, truth_ready
from model_benchmark import check_frame_identity, inspect_videos
from bench_report import aggregate, table


class MetricTests(unittest.TestCase):
    def test_class_errors_cannot_cancel(self):
        result = count_metrics([{'truth': {'car': 10, 'motorcycle': 3}, 'predicted': {'car': 8, 'motorcycle': 5}}])
        self.assertEqual(result['total_mae'], 4)
        self.assertEqual(result['car']['bias'], -2)
        self.assertEqual(result['motorcycle']['bias'], 2)

    def test_zero_truth_and_absent_truth_are_distinct(self):
        self.assertFalse(truth_ready({'frame': 0, 'truth': {'car': None, 'motorcycle': 0}}))
        self.assertTrue(truth_ready({'frame': 0, 'truth': {'car': 0, 'motorcycle': 0}}))
        r = count_metrics([{'truth': {'car': 0, 'motorcycle': 0}, 'predicted': {'car': 2, 'motorcycle': 0}}])
        self.assertIsNone(r['car']['wape_percent'])
        self.assertEqual(r['car']['mae'], 2)
        self.assertEqual(r['motorcycle']['exact_match_rate'], 1)

    def test_wape_uses_sum_absolute_errors(self):
        r = count_metrics([{'truth': {'car': 10, 'motorcycle': 1}, 'predicted': {'car': n, 'motorcycle': 1}}
                           for n in (8, 12, 9, 11)])
        self.assertEqual(r['car']['wape_percent'], 15)
        self.assertEqual(r['car']['bias'], 0)
        self.assertEqual(r['car']['mae'], 1.5)

    def test_heat_bottom_center_and_boundary(self):
        boxes = [[2, 1, 6, 8, .8, 0], [2, 1, 6, 8, .5, 0], [8, 2, 12, 12, .6, 1]]
        heat = heat_points(boxes, 10, 10)
        self.assertEqual(heat[0, 8, 4], 2)
        self.assertEqual(heat[1, 9, 9], 1)
        self.assertEqual(heat.sum(), 3)
        self.assertEqual(heat_points([], 10, 10).sum(), 0)

    def test_agreement_one_to_one_class_aware(self):
        a = [[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .9, 1]]
        b = [[0, 0, 10, 10, .7, 0]]
        r = box_agreement(a, b)
        self.assertEqual(r['matched'], 1)
        self.assertEqual(r['unmatched_a'], 2)
        self.assertAlmostEqual(r['sum_confidence_delta'], .2)
        self.assertEqual(box_agreement([], [])['matched'], 0)

    def test_stats_percentiles(self):
        r = stats([1, 2, 3, 4])
        self.assertEqual(r['mean'], 2.5)
        self.assertEqual(r['total'], 10)
        self.assertAlmostEqual(r['p95'], 3.85)

    def test_html_escapes_model_names(self):
        rendered = table([{'model': '<script>alert(1)</script>'}], ['model'])
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((BENCH / 'models.yaml').read_text(encoding='utf-8'))

    def load(self, change=None):
        c = copy.deepcopy(self.config)
        if change:
            change(c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump(c), encoding='utf-8')
            return load_config(path)

    def test_default_eight_videos(self):
        c = self.load()
        self.assertEqual(len(c['videos']), 8)
        self.assertTrue(all(not truth_ready(v) for v in c['videos']))
        self.assertTrue(all(Path(m['path']).is_absolute() for m in c['models']))

    def test_isolated_runtime_path(self):
        c = self.load(lambda c: c.update(runtime_path='runtime-yolo26'))
        self.assertTrue(Path(c['runtime_path']).is_absolute())
        self.assertEqual(Path(c['runtime_path']).name, 'runtime-yolo26')

    def test_s_m_suite_registers_six_pairs(self):
        c = load_config(BENCH / 'models_yolo_s_m.yaml')
        self.assertEqual(len(c['models']), 12)
        self.assertEqual(len({m['pair_id'] for m in c['models']}), 6)
        self.assertTrue(all(m['precision'] == 'fp16' for m in c['models']))

    def test_duplicate_id_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load(lambda c: c['models'].append(copy.deepcopy(c['models'][0])))

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.load(lambda c: c['models'][0].update(id='../escape'))

    def test_custom_class_mapping(self):
        c = self.load(lambda c: c['models'][0].update(classes={'car': 0, 'motorcycle': 1}))
        self.assertEqual(c['models'][0]['classes']['car'], 0)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            self.load(lambda c: c['settings'].update(conf=1.1))

    def test_bad_engine_size(self):
        with self.assertRaises(ValueError):
            self.load(lambda c: c['models'][1].update(input_size=1280))

    def test_unknown_adapter(self):
        with self.assertRaisesRegex(ValueError, 'adapter'):
            self.load(lambda c: c['models'][0].update(backend='legacy'))

    def test_truth_must_be_integer(self):
        with self.assertRaises(ValueError):
            self.load(lambda c: c['videos'][0].update(truth={'car': 1.5, 'motorcycle': 0}))

    def test_missing_video(self):
        with self.assertRaisesRegex(ValueError, 'Missing video'):
            inspect_videos([{'id': 'missing', 'path': 'does-not-exist.avi'}])


class ReproducibilityTests(unittest.TestCase):
    def test_mismatched_hashes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, digest in [('a', 'abc'), ('b', 'xyz')]:
                save_json(root / 'performance' / model / 'pass_1' / 'day_north' / 'result.json',
                          {'video': 'day_north', 'frames': 8, 'decoded_frames_sha256': digest})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                check_frame_identity(root)

    def test_aggregate_weighted_fps_and_incomplete_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root / 'run.json', {'settings': {'target_fps': 30}})
            rows = []
            for i, latency in enumerate((10., 20.)):
                path = root / str(i)
                path.mkdir()
                values = np.tile([1., 2., latency / 2, 1., latency], (2, 1))
                np.savez(path / 'frames.npz', timings=values)
                rows.append(dict(model='m', video='v', group='day', pass_index=i, _path=str(i),
                                 comparable=True, precision='fp32', frames=2, car_detections=0,
                                 motorcycle_detections=0, detection_ms={'mean': latency, 'total': latency * 2},
                                 resource={}, source_budget_exceed_percent=0, before_load={}))
            out = aggregate(rows, root, [{'id': 'v', 'group': 'day', 'selected_frames': 2}], 3)
            self.assertFalse(out[0]['complete'])
            self.assertAlmostEqual(out[0]['detection_fps'], 4000 / 60)
            self.assertIsNone(out[0]['process_vram_MiB_peak'])


@unittest.skipUnless(os.environ.get('BENCHMARK_GPU_TESTS') == '1', 'Set BENCHMARK_GPU_TESTS=1 for local PT/Engine integration')
class GPUIntegrationTests(unittest.TestCase):
    def test_prepare_and_count_known_empty_synthetic_frames(self):
        import cv2
        # These generated blank clips have known zero objects. No real-video truth is invented.
        with tempfile.TemporaryDirectory(prefix='yolo_benchmark_test_') as tmp:
            root = Path(tmp)
            config = load_config(BENCH / 'models.yaml')
            for v in config['videos']:
                path = root / (v['id'] + '.avi')
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 1, (64, 48))
                self.assertTrue(writer.isOpened())
                for _ in range(30):
                    writer.write(np.zeros((48, 64, 3), np.uint8))
                writer.release()
                v.update(path=str(path), frame=2, start_seconds=0, truth={'car': 0, 'motorcycle': 0})
            cfg = root / 'fixture.yaml'
            cfg.write_text(yaml.safe_dump(config), encoding='utf-8')
            prepared = root / 'prepared'
            command = [sys.executable, str(BENCH / 'model_benchmark.py'), '--config', str(cfg)]
            result = subprocess.run(command + ['--mode', 'prepare', '--output', str(prepared)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(read_json_local(prepared / 'selected_frames.json')), 8)
            output = root / 'count_result'
            result = subprocess.run(command + ['--mode', 'count-check', '--smoke', '--output', str(output)], capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = read_json_local(output / 'report_data.json')
            self.assertFalse(report['run']['failures'])
            self.assertEqual(len(report['run']['jobs']), 4)
            self.assertTrue(all(r['complete'] for r in report['count_summary']))
            self.assertTrue(all(r['car_wape_percent'] is None for r in report['count_summary']))
            self.assertTrue(all(r['total_mae'] == 0 for r in report['count_summary']))
            self.assertEqual(len(list(output.glob('count-check/*/pass_1/*/conf_*.png'))), 128)
            check_frame_identity(output)
            self.assertTrue((output / 'index.html').is_file())
            # Exercise real three-pass scheduling and fresh PT/engine subprocesses on short,
            # 30-second 1-FPS synthetic clips (not the user's camera dataset).
            config['settings'].update(warmup=3, repeats=3)
            cfg.write_text(yaml.safe_dump(config), encoding='utf-8')
            perf_output = root / 'three_pass_performance'
            result = subprocess.run(command + ['--mode', 'performance', '--models', 'v8n_pt', 'v8n_engine',
                                               '--output', str(perf_output)], capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            perf = read_json_local(perf_output / 'report_data.json')
            self.assertEqual([j['model'] for j in perf['run']['jobs']],
                             ['v8n_pt', 'v8n_engine', 'v8n_engine', 'v8n_pt', 'v8n_pt', 'v8n_engine'])
            self.assertEqual(len(perf['per_video']), 48)
            self.assertTrue(all(r['frames'] == 30 for r in perf['per_video']))
            self.assertTrue(all(r['complete'] and r['pass_fps_sample_std'] is not None for r in perf['comparison']))
            check_frame_identity(perf_output)
            # Wrong annotation-image hash must fail visibly and produce no accuracy ranking.
            config['videos'][0]['truth_frame_sha256'] = '0' * 64
            cfg.write_text(yaml.safe_dump(config), encoding='utf-8')
            bad_output = root / 'wrong_truth_image'
            result = subprocess.run(command + ['--mode', 'count-check', '--smoke', '--models', 'v8n_pt',
                                               '--group', 'day', '--output', str(bad_output)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            failed = read_json_local(bad_output / 'run.json')
            self.assertIn('annotated image hash', failed['failures'][0]['error'])
            self.assertFalse(read_json_local(bad_output / 'report_data.json')['rankings'])

    def test_engine_metadata_rejects_fixed_batch_and_wrong_precision(self):
        from bench_runtime import UltralyticsAdapter
        config = load_config(BENCH / 'models.yaml')
        artifact = copy.deepcopy(config['models'][1])
        with tempfile.TemporaryDirectory(prefix='yolo_engine_metadata_') as tmp:
            path = Path(tmp) / 'bad.engine'
            artifact['path'] = str(path)
            meta = dict(task='detect', imgsz=[640, 640], batch=8,
                        args=dict(dynamic=False, half=False, int8=False, nms=False))
            blob = json.dumps(meta).encode('utf-8')
            path.write_bytes(len(blob).to_bytes(4, 'little') + blob)
            with self.assertRaisesRegex(ValueError, 'Fixed engine batch'):
                UltralyticsAdapter(artifact, config['settings'])
            meta['args']['half'] = True
            blob = json.dumps(meta).encode('utf-8')
            path.write_bytes(len(blob).to_bytes(4, 'little') + blob)
            with self.assertRaisesRegex(ValueError, 'Precision mismatch'):
                UltralyticsAdapter(artifact, config['settings'])


def read_json_local(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
