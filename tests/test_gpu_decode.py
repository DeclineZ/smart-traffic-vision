"""
GPU decoding support: frames delivered downscaled by the decoder must produce the same
measurements as full-resolution frames. No GPU, ffmpeg or camera required.
"""

import os
import shutil
import subprocess
import sys
import time
import unittest
import unittest.mock
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.helpers import FakeModel, FakePublisher, make_pipeline, tmpdir
from tests.test_camera_health import FAST, scene
from trt_pipeline import nvdec
from trt_pipeline.camera_health import ImageHealthMonitor
from trt_pipeline.stream import FileReplaySource, StreamBufferWorker

H264_BANNER = (
    "Input #0, avi, from 'videos/cam44_north.avi':\n"
    "  Duration: 02:17:44.79, start: 0.000000, bitrate: 1767 kb/s\n"
    "  Stream #0:0: Video: h264 (Main) (H264 / 0x34363248), yuv420p(progressive), 1920x1080, 1767 kb/s, "
    "150 fps, 25 tbr, 150 tbn\n"
)
FULL_RANGE_BANNER = (
    "  Stream #0:0: Video: h264 (Main) (H264 / 0x34363248), yuvj420p(pc, progressive), 1920x1080 "
    "[SAR 1:1 DAR 16:9], 1736 kb/s, 150 fps, 25 tbr, 150 tbn"
)
RTSP_BANNER = "  Stream #0:0: Video: hevc (Main), yuv420p(tv, bt709), 2560x1440, 25 fps, 25 tbr, 90k tbn\n"


class TestBannerAndSizing(unittest.TestCase):
    def test_parse_banner(self):
        self.assertEqual(nvdec.parse_banner(H264_BANNER), ("h264", 1920, 1080, 150.0, False))
        self.assertEqual(nvdec.parse_banner(FULL_RANGE_BANNER), ("h264", 1920, 1080, 150.0, True))
        self.assertEqual(nvdec.parse_banner(RTSP_BANNER), ("hevc", 2560, 1440, 25.0, False))
        with self.assertRaises(OSError):
            nvdec.parse_banner("Input #0, mp3, from 'a.mp3':\n  Stream #0:0: Audio: mp3, 44100 Hz\n")

    def test_output_size_downscales_only_and_keeps_aspect(self):
        self.assertEqual(nvdec.output_size(1920, 1080, 640), (640, 360))
        self.assertEqual(nvdec.output_size(1920, 1080, 960), (960, 540))
        self.assertEqual(nvdec.output_size(2560, 1440, 640), (640, 360))
        self.assertEqual(nvdec.output_size(1280, 720, 1920), (1280, 720))  # never upscales
        w, h = nvdec.output_size(1920, 1080, 641)
        self.assertEqual((w % 2, h % 2), (0, 0))

    def test_interp_matches_cv2_inter_linear(self):
        self.assertEqual(nvdec.resize_interp(3.0), "nearest")   # 1920 -> 640
        self.assertEqual(nvdec.resize_interp(2.0), "bilinear")  # 1920 -> 960
        self.assertEqual(nvdec.resize_interp(4.0), "bilinear")
        self.assertEqual(nvdec.resize_interp(2560 / 640.0), "bilinear")
        self.assertEqual(nvdec.resize_interp(1.5), "bilinear")

    def test_full_range_tables_map_to_video_range(self):
        self.assertEqual((int(nvdec._Y_FULL_TO_VIDEO[0]), int(nvdec._Y_FULL_TO_VIDEO[255])), (16, 235))
        self.assertEqual((int(nvdec._C_FULL_TO_VIDEO[0]), int(nvdec._C_FULL_TO_VIDEO[128]),
                          int(nvdec._C_FULL_TO_VIDEO[255])), (16, 128, 240))


class TestDecoderSelection(unittest.TestCase):
    def test_auto_falls_back_to_opencv_and_nvdec_is_strict(self):
        missing = os.path.join(tmpdir(), "no-ffmpeg.exe")
        sentinel = object()
        with patch.dict(os.environ, {"VISION_FFMPEG": missing}), \
                patch("trt_pipeline.stream.open_capture", return_value=sentinel) as opener:
            self.assertIs(nvdec.make_capture("video.avi", "auto"), sentinel)
            self.assertIs(nvdec.make_capture("video.avi", "opencv"), sentinel)
            self.assertEqual(opener.call_count, 2)
            with self.assertRaises(OSError):
                nvdec.make_capture("video.avi", "nvdec")

    def test_device_index_never_uses_ffmpeg(self):
        with patch("trt_pipeline.stream.open_capture", return_value="cv") as opener, \
                patch.object(nvdec, "NvdecCapture", side_effect=AssertionError("must not be used")):
            self.assertEqual(nvdec.make_capture(0, "auto"), "cv")
        opener.assert_called_once()


# Stands in for the ffmpeg child: writes 64x36 NV12 frames to stdout following a script of
# frame counts and pauses, then exits (stream end) or hangs (stall).
FAKE_FFMPEG = """
import sys, time
frame = bytes(64 * 36 * 3 // 2)
for step in sys.argv[1].split(","):
    if step == "hang":
        time.sleep(60)
    elif step.startswith("sleep"):
        time.sleep(float(step[5:]))
    else:
        for _ in range(int(step)):
            sys.stdout.buffer.write(frame)
            sys.stdout.buffer.flush()
"""


class TestNvdecStreamEnd(unittest.TestCase):
    def setUp(self):
        self.script = None
        real_popen = subprocess.Popen

        def popen(cmd, **kwargs):  # every ffmpeg start, including restarts, runs FAKE_FFMPEG
            return real_popen([sys.executable, "-c", FAKE_FFMPEG, self.script], **kwargs)

        for patcher in (patch.object(nvdec, "probe", return_value=("h264", 64, 36, 25.0, False)),
                        patch.object(nvdec.subprocess, "Popen", popen)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_capture(self, script, read_timeout_s=3.0):
        """NvdecCapture whose ffmpeg is FAKE_FFMPEG running ``script``."""
        self.script = script
        return nvdec.NvdecCapture("rtsp://camera.invalid/stream", out_width=64, open_timeout_s=10.0,
                                  read_timeout_s=read_timeout_s, exe="ffmpeg")

    def test_ended_stream_keeps_failing_without_waiting(self):
        cap = self.fake_capture("3")
        try:
            self.assertEqual([cap.read()[0] for _ in range(4)], [True, True, True, False])
            started = time.monotonic()
            self.assertFalse(cap.read()[0])
            self.assertFalse(cap.read()[0])
            self.assertLess(time.monotonic() - started, 1.0, "later reads must not wait for the read timeout")
        finally:
            cap.release()

    def test_stall_stops_ffmpeg_and_never_delivers_late_frames(self):
        cap = self.fake_capture("3,sleep2,5,hang", read_timeout_s=0.5)
        try:
            proc = cap._proc
            self.assertEqual([cap.read()[0] for _ in range(3)], [True, True, True])
            started = time.monotonic()
            self.assertFalse(cap.read()[0])
            self.assertLess(time.monotonic() - started, 1.5)
            self.assertIsNotNone(proc.poll(), "a stalled ffmpeg must be stopped, not left to its own timeout")
            time.sleep(2.0)  # when the late frames would have arrived
            self.assertFalse(cap.read()[0])
        finally:
            cap.release()

    def test_rewind_restarts_an_ended_stream(self):
        cap = self.fake_capture("2")
        try:
            self.assertEqual([cap.read()[0] for _ in range(3)], [True, True, False])
            self.assertTrue(cap.set(nvdec.cv.CAP_PROP_POS_FRAMES, 0))
            self.assertEqual([cap.read()[0] for _ in range(3)], [True, True, False])
        finally:
            cap.release()

    def test_release_kills_ffmpeg(self):
        cap = self.fake_capture("1,hang")
        proc = cap._proc
        cap.release()
        self.assertIsNotNone(proc.poll())

    def test_worker_reconnects_promptly_after_stream_end(self):
        # The runner's live worker needs 3 failed reads; an ended stream must fail all of them at once.
        procs = []

        def factory():
            cap = self.fake_capture("5")
            procs.append(cap._proc)
            return cap

        def wait_for(condition, timeout_s):
            deadline = time.monotonic() + timeout_s
            while not condition() and time.monotonic() < deadline:
                time.sleep(0.01)
            return condition()

        worker = StreamBufferWorker("cam", "rtsp://camera.invalid/stream", reconnect_initial_s=5.0,
                                    capture_factory=factory)
        worker.start()
        try:
            self.assertTrue(wait_for(lambda: worker.frames_ingested >= 5, 15.0))
            ended = time.monotonic()
            self.assertTrue(wait_for(lambda: worker.state == "reconnecting", 15.0))
            self.assertLess(time.monotonic() - ended, 1.0, "an ended stream must be given up at once")
        finally:
            worker.stop()
        self.assertTrue(all(p.poll() is not None for p in procs), "every ffmpeg child is gone after stop")


class ScaledCap:
    """Capture that delivers frames downscaled 3x, like NvdecCapture."""

    scale = 3.0

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((36, 64, 3), np.uint8)

    def get(self, prop):
        return 25.0

    def set(self, prop, value):
        return True

    def release(self):
        pass


class TestScaleThroughPipeline(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_replay_packets_carry_capture_scale(self):
        src = FileReplaySource("cam", "video.avi", replay_fps=25.0, capture_factory=ScaledCap)
        src.start()
        self.assertEqual(src.poll().scale, 3.0)

    def test_downscaled_frame_measures_like_full_resolution(self):
        # Lane in camera pixels (300x300 camera). The decoder delivers a 100x100 frame (scale 3), the
        # model sees a vehicle whose road contact point is at (50, 50) in frame pixels = (150, 150) camera px.
        lane = [[120, 120], [180, 120], [180, 180], [120, 180]]
        frame_box = np.array([[45, 40, 55, 50, 0.9, 0]], dtype=float)
        model = FakeModel(lambda i, call: frame_box.copy())
        p, sources = make_pipeline(self.dir, {"north": {"lanes": {"N1": {"direction": "N", "polygon": lane}}}},
                                   model=model, publisher=FakePublisher())
        p.cams[0].tracker.min_hits = 1
        p.cams[0].expected_resolution = (300, 300)
        sources[0].shape = (100, 100)
        sources[0].push(obs_t=0.0)
        sources[0].queue[-1].scale = 3.0
        p.step()
        cam = p.cams[0]
        self.assertIsNone(cam.frame_invalid_reason, "a 100x100 frame at scale 3 is the 300x300 camera")
        self.assertEqual(cam.metrics.snapshot()[0]["count"], 1)
        np.testing.assert_allclose(cam.last_tracked[0, :4], [135, 120, 165, 150], atol=1.0)
        self.assertEqual(cam.last_frame_scale, 3.0)

    def test_wrong_camera_resolution_still_detected_when_downscaled(self):
        p, sources = make_pipeline(self.dir, {"north": {"lanes": {"N1": {"direction": "N",
                                                                         "polygon": [[0, 0], [9, 0], [9, 9]]}}}},
                                   publisher=FakePublisher())
        p.cams[0].expected_resolution = (1920, 1080)
        sources[0].shape = (360, 640)  # 640x360 at scale 2 is a 1280x720 camera, not the calibrated 1080p
        sources[0].push(obs_t=0.0)
        sources[0].queue[-1].scale = 2.0
        p.step()
        self.assertEqual(p.cams[0].frame_invalid_reason, "resolution_mismatch")


class TestNvencWriterSize(unittest.TestCase):
    def test_writer_adopts_rendered_grid_size(self):
        """The preview grid carries a header banner; raw video at the wrong size is sheared garbage."""
        from trt_pipeline.display import NVENCVideoWriter

        procs = []

        def fake_popen(cmd, **kw):
            proc = unittest.mock.MagicMock()
            proc.cmd = cmd
            procs.append(proc)
            return proc

        with patch("trt_pipeline.nvdec.ffmpeg_exe", return_value="ffmpeg"), \
                patch("trt_pipeline.display.subprocess.Popen", side_effect=fake_popen):
            w = NVENCVideoWriter("out.mkv", width=1920, height=540)
            w.write(np.zeros((568, 1440, 3), np.uint8))
            w.write(np.zeros((540, 1920, 3), np.uint8))  # later odd-sized frame is fitted, not restarted
        self.assertEqual(len(procs), 2)
        self.assertIn("1440x568", procs[1].cmd)
        written = [c.args[0] for c in procs[1].stdin.write.call_args_list]
        self.assertEqual([len(b) for b in written], [1440 * 568 * 3] * 2)


class TestShiftInCameraPixels(unittest.TestCase):
    def test_shift_is_reported_in_camera_pixels_for_downscaled_frames(self):
        ref = scene()
        moved = scene(shift=(40, 25))
        full = ImageHealthMonitor(reference=ref, settings=FAST).measure_shift(moved)
        half = moved[::2, ::2].copy()
        down = ImageHealthMonitor(reference=ref, settings=FAST).measure_shift(half, frame_scale=2.0)
        self.assertIsNotNone(full)
        self.assertIsNotNone(down)
        self.assertAlmostEqual(down[0], full[0], delta=4.0)
        unscaled = ImageHealthMonitor(reference=ref, settings=FAST).measure_shift(half)
        self.assertLess(unscaled[0], full[0] * 0.7, "without the scale the shift would be under-reported")


if __name__ == "__main__":
    unittest.main()
