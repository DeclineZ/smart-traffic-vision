"""Camera ingestion: reconnect with backoff, epochs, latest-only delivery, replay timestamps, redaction."""

import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.stream import FileReplaySource, StreamBufferWorker, redact_source


class FakeCap:
    """Scripted capture: yields `frames` good frames, then fails forever."""

    def __init__(self, frames=5, opened=True, delay=0.0):
        self.left = frames
        self.total = frames
        self.opened = opened
        self.delay = delay
        self.released = False
        self.pos = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.left <= 0:
            return False, None
        self.left -= 1
        self.pos += 1
        return True, np.full((4, 4, 3), self.pos, dtype=np.uint8)

    def get(self, prop):
        return 25.0

    def set(self, prop, val):  # seek to start: the file can be read again
        self.pos = 0
        self.left = self.total
        return True

    def release(self):
        self.released = True


def wait_until(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class TestStreamBufferWorker(unittest.TestCase):
    def test_reconnects_after_read_failures_and_starts_new_epoch(self):
        caps = []

        def factory():
            caps.append(FakeCap(frames=3, delay=0.005))
            return caps[-1]

        w = StreamBufferWorker("cam", "rtsp://x", capture_factory=factory, reconnect_initial_s=0.01,
                               max_consecutive_failures=2)
        w.start()
        try:
            self.assertTrue(wait_until(lambda: w.reconnects >= 2))
            self.assertGreaterEqual(w.epoch, 3)
            self.assertTrue(caps[0].released)
        finally:
            w.stop()

    def test_open_failure_backs_off_and_does_not_block_start(self):
        attempts = []

        def factory():
            attempts.append(time.time())
            return FakeCap(opened=False)

        w = StreamBufferWorker("cam", "rtsp://x", capture_factory=factory, reconnect_initial_s=0.05,
                               reconnect_max_s=0.2)
        t0 = time.time()
        w.start()
        self.assertLess(time.time() - t0, 0.5)  # start() returns immediately
        try:
            time.sleep(0.6)
            self.assertEqual(w.status()["state"], "offline")
            self.assertIsNone(w.poll())
            gaps = np.diff(attempts)
            self.assertTrue(all(g >= 0.04 for g in gaps))
            self.assertLess(len(attempts), 12)  # backoff, not a busy loop
        finally:
            w.stop()

    def test_latest_frame_only_and_drop_count(self):
        gate = threading.Event()

        class Cap(FakeCap):
            def read(inner):
                if inner.left <= 0:
                    gate.set()
                    time.sleep(0.01)
                    return True, None
                return FakeCap.read(inner)

        w = StreamBufferWorker("cam", "rtsp://x", capture_factory=lambda: Cap(frames=10),
                               max_consecutive_failures=10 ** 6)
        w.start()
        try:
            self.assertTrue(gate.wait(2.0))
            p = w.poll()
            self.assertIsNotNone(p)
            self.assertEqual(int(p.frame[0, 0, 0]), 10)   # newest frame, not the oldest
            self.assertEqual(w.frames_dropped, 9)
            self.assertIsNone(w.poll())                     # nothing newer
            self.assertLessEqual(abs(time.time() - p.captured_wall), 2.0)
        finally:
            w.stop()


class TestFileReplaySource(unittest.TestCase):
    def test_media_time_and_loop_epoch(self):
        src = FileReplaySource("cam", "video.avi", replay_fps=10.0, capture_factory=lambda: FakeCap(frames=3))
        src.start()
        packets = [src.poll() for _ in range(3)]
        self.assertEqual([p.obs_t for p in packets], [0.0, 0.1, 0.2])
        self.assertEqual({p.epoch for p in packets}, {1})
        p = src.poll()  # end of file: loops back to the start
        self.assertEqual(p.epoch, 2)
        self.assertEqual(p.obs_t, 0.0)

    def test_unreadable_file_reports_offline(self):
        src = FileReplaySource("cam", "missing.avi", replay_fps=10.0,
                               capture_factory=lambda: FakeCap(opened=False))
        src.start()
        self.assertIsNone(src.poll())
        self.assertEqual(src.status()["state"], "offline")


class TestRedaction(unittest.TestCase):
    def test_credentials_hidden(self):
        self.assertEqual(redact_source("rtsp://admin:secret@10.0.0.5:554/s1"), "rtsp://***@10.0.0.5:554/s1")
        self.assertEqual(redact_source("videos/a.avi"), "videos/a.avi")


if __name__ == "__main__":
    unittest.main()
