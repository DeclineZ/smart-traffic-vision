import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from bench_core import expand_count_frames, truth_ready, load_config


class CountFramesTests(unittest.TestCase):
    def test_independent_truth_and_stable_ids(self):
        source = dict(id='day_east', count_frames=[450, 1350], frame=99,
                      truth=dict(car=40, motorcycle=0))
        samples = expand_count_frames([source])
        self.assertEqual([v['id'] for v in samples], ['day_east_f450', 'day_east_f1350'])
        self.assertTrue(all(not truth_ready(v) for v in samples))
        samples[0]['truth']['car'] = 2
        self.assertIsNone(samples[1]['truth']['car'])
        self.assertEqual(source['truth']['car'], 40)

    def test_existing_single_frame_preserved(self):
        source = dict(id='day_west', frame=42, truth=dict(car=2, motorcycle=0))
        self.assertEqual(expand_count_frames([source]), [source])

    def test_suite_has_20_images_and_eight_engines(self):
        c = load_config(Path(__file__).resolve().parents[1] / 'benchmark/models_count_20frames.yaml')
        selected = [v for v in c['videos'] if v['camera'] in ('east', 'west')]
        self.assertEqual(len(expand_count_frames(selected)), 20)
        self.assertEqual(len(c['models']), 8)
        self.assertTrue(all(m['path'].endswith('.engine') for m in c['models']))
