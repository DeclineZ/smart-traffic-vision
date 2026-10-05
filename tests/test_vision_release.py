"""Vision-only release regressions; no broker, camera, model weights or GPU."""

import shutil
import os
import time
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.helpers import SQUARE, FakeModel, make_pipeline, tmpdir
from trt_pipeline.camera_health import HealthSettings, ImageHealthMonitor
from trt_pipeline.gates import GateFlowManager, VirtualGate
from trt_pipeline.motion import MotionStateClassifier, UNKNOWN
from trt_pipeline.stream import StreamBufferWorker
from tests.test_stream import FakeCap, wait_until


class TestVisionRelease(unittest.TestCase):
    def setUp(self):
        self.directory = tmpdir()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def pipeline(self, **kwargs):
        return make_pipeline(self.directory, {
            "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}}
        }, **kwargs)

    def test_shadow_default_ignores_legacy_topic_environment(self):
        with patch.dict("os.environ", {"TRAFFIC_COUNTS_TOPIC": "traffic/counts", "MQTT_TOPIC": "traffic/counts"}):
            pipeline, _ = self.pipeline()
        self.assertEqual(pipeline.mqtt_topic, "traffic/counts/shadow")
        self.assertEqual(pipeline.health_topic, "traffic/health/shadow/VISION-INT-001")
        self.assertEqual(pipeline.effective_settings()["outputMode"], "shadow")

    def test_live_topics_require_explicit_controller_mode(self):
        with self.assertRaises(ValueError):
            self.pipeline(mqtt_topic="traffic/counts")
        with self.assertRaises(ValueError):
            self.pipeline(health_topic="traffic/health/VISION-INT-001")

    def test_reconnect_forces_inference_on_first_fresh_frame(self):
        model = FakeModel(lambda i, call: np.array([[40, 30, 60, 50, .9, 0]]))
        pipeline, sources = self.pipeline(skip_frames=1, model=model)
        sources[0].push(obs_t=0.0)
        pipeline.step()
        sources[0].push(obs_t=0.0, epoch=2)
        pipeline.step()
        lane = pipeline.build_payload()["lanes"][0]
        self.assertEqual(model.calls, [1, 1])
        self.assertTrue(lane["valid"])
        self.assertEqual(lane["count"], 1)

    def test_stale_reconnect_packet_cannot_reuse_old_observation(self):
        pipeline, sources = self.pipeline(skip_frames=1)
        sources[0].push(obs_t=0.0)
        pipeline.step()
        sources[0].push(obs_t=0.0, epoch=2, captured_wall=time.time() - 5)
        pipeline.step()
        lane = pipeline.build_payload()["lanes"][0]
        self.assertFalse(lane["valid"])
        self.assertIsNone(lane["count"])
        self.assertIsNone(lane["observedAt"])

    def test_tracker_warmup_is_unknown_before_confirmation(self):
        model = FakeModel(lambda i, call: np.array([[40, 30, 60, 50, .9, 0]]))
        pipeline, sources = self.pipeline(skip_frames=0, model=model)
        sources[0].push(obs_t=0.0)
        pipeline.step()
        lane = pipeline.build_payload()["lanes"][0]
        self.assertFalse(lane["valid"])
        self.assertIsNone(lane["count"])
        sources[0].push(obs_t=0.1)
        pipeline.step()
        self.assertEqual(pipeline.build_payload()["lanes"][0]["count"], 1)

    def test_obstructed_camera_never_reports_valid_zero(self):
        pipeline, sources = self.pipeline(skip_frames=1)
        sources[0].push(obs_t=0.0, frame=np.full((100, 100, 3), 160, np.uint8))
        pipeline.step()
        lane = pipeline.build_payload()["lanes"][0]
        self.assertFalse(lane["valid"])
        self.assertIsNone(lane["count"])

    def test_invalid_maintenance_file_disables_measurements(self):
        path = os.path.join(self.directory, "maintenance.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"typo": "cleaning"}')
        pipeline, sources = self.pipeline(skip_frames=1, maintenance_file=path)
        sources[0].push(obs_t=0)
        pipeline.step()
        payload = pipeline.build_payload()
        self.assertEqual(payload["cameras"][0]["reason"], "maintenance_config_error")
        self.assertIsNone(payload["lanes"][0]["count"])
        self.assertEqual(pipeline.model.calls, [])

    def test_wall_clock_backstep_does_not_keep_silent_camera_fresh(self):
        pipeline, sources = self.pipeline(skip_frames=1)
        with patch("run_multi_camera.time.monotonic", return_value=100):
            sources[0].push(obs_t=0, captured_wall=1000)
            sources[0].queue[-1].captured_mono = 100
            with patch("run_multi_camera.time.time", return_value=1000):
                pipeline.step()
        with patch("run_multi_camera.time.monotonic", return_value=110):
            lane = pipeline.build_payload(now_wall=900)["lanes"][0]
        self.assertFalse(lane["valid"])
        self.assertIsNone(lane["count"])

    def test_packet_age_uses_monotonic_clock_when_wall_clock_moves(self):
        from trt_pipeline.stream import FramePacket
        packet = FramePacket(np.zeros((4, 4, 3)), 1, 1, 1000, 100, captured_mono=100)
        with patch("trt_pipeline.stream.time.monotonic", return_value=103):
            self.assertEqual(packet.age_s(now_wall=900), 3)

    def test_health_write_failure_is_visible_and_preserves_original(self):
        from run_multi_camera import _write_json_atomic
        path = os.path.join(self.directory, "health.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"intact": true}')
        with patch("run_multi_camera.os.replace", side_effect=PermissionError("denied")), self.assertRaises(PermissionError):
            _write_json_atomic(path, {"status": "online"})
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"intact": true}')
        self.assertEqual(os.listdir(self.directory), ["health.json"])


class TestHealthRecovery(unittest.TestCase):
    def test_noisy_bright_obstruction_is_invalid_until_sustained_recovery(self):
        monitor = ImageHealthMonitor(settings=HealthSettings(check_interval_s=0))
        rng = np.random.default_rng(12)
        for t in range(20):
            blocked = np.clip(160 + rng.normal(0, 1, (180, 320, 3)), 0, 255).astype(np.uint8)
            monitor.check(blocked, float(t))
            self.assertEqual(monitor.verdict(), "low_detail")
        healthy = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
        monitor.check(healthy, 20.0)
        self.assertEqual(monitor.verdict(), "low_detail")
        monitor.check(healthy, 22.1)
        self.assertIsNone(monitor.verdict())

    def test_shift_confirmations_restart_after_inconclusive_check(self):
        monitor = ImageHealthMonitor(settings=HealthSettings(check_interval_s=0, shift_check_interval_s=0))
        monitor._ref_desc = np.zeros((1, 32), np.uint8)
        frame = np.random.default_rng(4).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        with patch.object(monitor, "measure_shift", side_effect=[(50, 100), (50, 100), None, (50, 100)]):
            for t in range(4):
                monitor.check(frame, float(t))
        self.assertIsNone(monitor.verdict())

    def test_confirmed_shift_survives_reconnect_until_view_is_restored(self):
        monitor = ImageHealthMonitor(settings=HealthSettings(check_interval_s=0, shift_check_interval_s=0))
        monitor._ref_desc = np.zeros((1, 32), np.uint8)
        frame = np.random.default_rng(4).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        with patch.object(monitor, "measure_shift", return_value=(50, 100)):
            for t in range(3):
                monitor.check(frame, float(t))
            self.assertEqual(monitor.verdict(), "camera_shifted")
            monitor.reset()
            monitor.check(frame, 4.0)
            self.assertEqual(monitor.verdict(), "camera_shifted")

    def test_health_settings_reject_nonfinite_values_and_unknown_keys(self):
        for config in ({"obstruction_std": float("nan")}, {"recovery_duration_s": -1},
                       {"shift_confirmations": 0}, {"typo": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                HealthSettings.from_config(config)


class TestTensorRTAcceptance(unittest.TestCase):
    def test_agreement_covers_extra_engine_boxes_and_wrong_classes(self):
        from tools.trt_parity import detection_agreement
        self.assertEqual(detection_agreement(10, 20, 10, 0), (1, .5, 0))
        self.assertEqual(detection_agreement(0, 0, 0, 0), (1, 1, 1))
        self.assertEqual(detection_agreement(0, 1, 0, 0), (0, 0, 0))

    def test_actual_tool_rejects_wrong_classes_and_extra_engine_detections(self):
        from tools import trt_parity
        from shapely.geometry import Polygon
        frames = {c: [np.zeros((100, 100, 3), np.uint8)] for c in trt_parity.CAMERAS}
        lanes = {c: {"L1": Polygon(SQUARE)} for c in trt_parity.CAMERAS}
        pt = np.array([[40, 30, 60, 50, .9, 0]])
        cases = [np.array([[40, 30, 60, 50, .9, 1]]),
                 np.array([[40, 30, 60, 50, .9, 0], [240, 230, 260, 250, .9, 0]])]
        for engine in cases:
            def backend(model, batches, kw):
                dets = engine if model == "test.engine" else pt
                return [[dets for frame in batch] for batch in batches], [1]
            with self.subTest(engine=engine.tolist()), \
                 patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=lambda path, **kw: path)}), \
                 patch.object(trt_parity, "load_frames", return_value=(frames, lanes)), \
                 patch.object(trt_parity, "run_backend", side_effect=backend), patch("builtins.print"):
                self.assertEqual(trt_parity.main(["--engine", "test.engine", "--frames", "1"]), 1)


class TestCaptureExceptions(unittest.TestCase):
    def test_backend_read_and_metadata_exceptions_reconnect(self):
        for failing_method in ("read", "get", "isOpened"):
            with self.subTest(method=failing_method):
                broken = FakeCap()
                setattr(broken, failing_method, lambda *args: (_ for _ in ()).throw(RuntimeError("backend failure")))
                healthy = FakeCap(frames=100000, delay=.01)
                captures = iter([broken, healthy])
                worker = StreamBufferWorker("cam", "rtsp://x", capture_factory=lambda: next(captures),
                                            reconnect_initial_s=.01, max_consecutive_failures=1)
                worker.start()
                try:
                    self.assertTrue(wait_until(lambda: worker.frames_ingested > 0, timeout=.5))
                    self.assertTrue(broken.released)
                    self.assertTrue(worker.thread.is_alive())
                finally:
                    worker.stop()

    def test_replay_backend_failure_is_offline_and_releases_capture(self):
        from trt_pipeline.stream import FileReplaySource
        cap = FakeCap()
        cap.read = Mock(side_effect=RuntimeError("read failed"))
        source = FileReplaySource("cam", "file.avi", capture_factory=lambda: cap)
        source.start()
        self.assertIsNone(source.poll())
        self.assertEqual(source.status()["state"], "offline")
        self.assertTrue(cap.released)


class TestFlowValidity(unittest.TestCase):
    def manager(self):
        with patch("trt_pipeline.gates.time.monotonic", return_value=0):
            manager = GateFlowManager(camera_ids=["CAM-N"])
        manager.add_gate(VirtualGate("G_N", 0, (0, 50), (100, 50), target_dir="N", gate_type="stopline"))
        return manager

    def test_no_coverage_is_unknown_at_individual_gate(self):
        manager = self.manager()
        self.assertIsNone(manager.interval_report()["gates"][0]["count"])

    def test_long_gap_does_not_credit_blind_time(self):
        manager = self.manager()
        for t in (0, 3):
            with patch("trt_pipeline.gates.time.monotonic", return_value=t):
                manager.update_tracks(0, np.empty((0, 6)), now=t)
        with patch("trt_pipeline.gates.time.monotonic", return_value=3):
            gate = manager.interval_report()["gates"][0]
        self.assertEqual(gate["observedSec"], 0)
        self.assertFalse(gate["valid"])

    def test_coverage_does_not_cross_publication_boundary(self):
        manager = self.manager()
        with patch("trt_pipeline.gates.time.monotonic", return_value=0):
            manager.update_tracks(0, np.empty((0, 6)), now=0)
        with patch("trt_pipeline.gates.time.monotonic", return_value=.5):
            manager.reset_interval()
        with patch("trt_pipeline.gates.time.monotonic", return_value=1):
            manager.update_tracks(0, np.empty((0, 6)), now=1)
        with patch("trt_pipeline.gates.time.monotonic", return_value=1.5):
            report = manager.interval_report()
        self.assertEqual(report["gates"][0]["observedSec"], .5)

    def test_current_camera_invalidity_overrides_previous_coverage(self):
        manager = self.manager()
        manager._cam_observed_s[0] = 1
        with patch("trt_pipeline.gates.time.monotonic", return_value=1):
            report = manager.interval_report(camera_validity={0: False})
        self.assertFalse(report["gates"][0]["valid"])
        self.assertIsNone(report["gates"][0]["count"])

    def test_discontinuity_invalidates_the_whole_interval(self):
        manager = self.manager()
        manager._cam_observed_s[0] = 1
        manager.reset_camera(0)
        with patch("trt_pipeline.gates.time.monotonic", return_value=1):
            self.assertFalse(manager.interval_report(camera_validity={0: True})["gates"][0]["valid"])
            manager.reset_interval()
        for t in (1, 1.5):
            with patch("trt_pipeline.gates.time.monotonic", return_value=t):
                manager.update_tracks(0, np.empty((0, 6)), now=t)
        with patch("trt_pipeline.gates.time.monotonic", return_value=1.5):
            self.assertTrue(manager.interval_report(camera_validity={0: True})["gates"][0]["valid"])


class TestMotionDiscontinuity(unittest.TestCase):
    def test_long_unobserved_gap_restarts_motion_evidence(self):
        classifier = MotionStateClassifier()
        for t in np.arange(0, 4, .1):
            classifier.update(1, (50, 50), 20, float(t))
        self.assertEqual(classifier.state_of(1), "queued")
        self.assertEqual(classifier.update(1, (50, 50), 20, 20), UNKNOWN)


class TestSoakChecks(unittest.TestCase):
    def test_latency_history_is_bounded(self):
        from tools.soak import LatencySample
        sample = LatencySample(limit=100)
        for i in range(20000):
            sample.add(i)
        self.assertEqual(sample.count, 20000)
        self.assertEqual(len(sample.values), 100)
        self.assertGreater(max(sample.values), 10000)

    def test_broker_wrapper_still_observes_failed_payload(self):
        from tools.soak import SinkPublisher
        publisher = Mock()
        publisher.publish.return_value = False
        observer = SinkPublisher(publisher)
        payload = {"sequence": 1}
        self.assertFalse(observer.publish(payload))
        self.assertIs(observer.last, payload)
        publisher.publish.assert_called_once_with(payload)

    def test_outage_checks_all_lanes_and_recovery(self):
        from tools.soak import observe_outages
        outage = {"camera": "north", "start": 0, "duration": 10, "invalidSeen": False,
                  "knownDuringOutage": False, "zeroDuringOutage": False, "recoveredSeen": False,
                  "othersValidDuringOutage": True, "otherCameraProblems": []}
        payload = {"cameras": [{"name": "north", "cameraId": "CAM-N", "status": "stale"}],
                   "lanes": [{"cameraId": "CAM-N", "valid": False, "count": None, "queuedCount": None}]}
        observe_outages(payload, 3, [outage], 2)
        self.assertTrue(outage["invalidSeen"])
        payload["lanes"].append({"cameraId": "CAM-N", "valid": True, "count": 4, "queuedCount": 1})
        observe_outages(payload, 4, [outage], 2)
        self.assertTrue(outage["knownDuringOutage"])
        payload["lanes"] = payload["lanes"][1:]
        payload["cameras"][0]["status"] = "ok"
        observe_outages(payload, 13, [outage], 2)
        self.assertTrue(outage["recoveredSeen"])

    def test_source_selection_reaches_real_run_plan(self):
        from tools.soak import build_args
        from run_multi_camera import resolve_run_plan
        args = build_args().parse_args(["--cameras", "north", "--videos", "rtsp://camera/live"])
        names, _, sources = resolve_run_plan(args)
        self.assertEqual(names, ["north"])
        self.assertEqual(sources, ["rtsp://camera/live"])
