import hashlib
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from bench_prefetch import FrameReader
from bench_core import completed_clip, save_json


class Capture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame


class PrefetchTests(unittest.TestCase):
    def test_resume_requires_complete_matching_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = dict(id='north', selected_frames=8)
            save_json(folder / 'result.json', dict(model='test', video='north', pass_index=0, frames=8))
            self.assertIsNone(completed_clip(folder, video, 'test', 0))
            for name in ('frames.npz', 'telemetry.csv', 'heat.npz', 'predictions.npz', 'background.png'):
                (folder / name).touch()
            self.assertIsNotNone(completed_clip(folder, video, 'test', 0))
            self.assertIsNone(completed_clip(folder, video, 'other', 0))
            self.assertIsNone(completed_clip(folder, dict(id='north', selected_frames=9), 'test', 0))

    def test_order_and_hash_match_sequential(self):
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(20)]
        expected = hashlib.sha256(b''.join(f.tobytes() for f in frames)).hexdigest()
        for depth in (0, 1, 4):
            reader = FrameReader(Capture(frames[1:]), len(frames), frames[0], .1, depth)
            try:
                received = list(reader)
                self.assertEqual([int(x[0][0, 0, 0]) for x in received], list(range(20)))
                self.assertEqual(reader.digest.hexdigest(), expected)
            finally:
                reader.close()

    def test_decode_error_propagates(self):
        reader = FrameReader(Capture([]), 2, np.zeros((2, 2, 3), np.uint8), 0, 1)
        try:
            with self.assertRaisesRegex(ValueError, 'Early decode'):
                list(reader)
        finally:
            reader.close()

    def test_cancel_full_queue(self):
        frame = np.zeros((2, 2, 3), np.uint8)
        reader = FrameReader(Capture([frame] * 100), 101, frame, 0, 1)
        reader.close()
        self.assertFalse(reader.thread.is_alive())
