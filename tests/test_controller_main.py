"""Legacy-main projection and real runner regressions, without a broker or GPU."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from tests.helpers import SQUARE, FakePublisher, make_pipeline, tmpdir, write_config, FakeModel
from trt_pipeline.controller_main import MainControllerAdapter
from trt_pipeline.payload import iso_utc


class TestMainAdapter(unittest.TestCase):
    def setUp(self):
        self.directory = tmpdir()
        self.path = os.path.join(self.directory, "main.json")
        self.expected = {f"{d}{i}": d for d in "NESW" for i in range(1, 4)}
        self.config = {"intersection_id": "INT-001", "camera_id": "CAM-01",
                       "expected_lanes": self.expected, "max_observation_age_ms": 750,
                       "controller_freshness_ms": 2000}
        self.lane_configs = [{lid: {"direction": direction, "role": "queue"}
                              for lid, direction in self.expected.items()},
                             {"NE1": {"direction": "N", "role": "upstream"}}]
        self.write_profile()
        self.adapter = MainControllerAdapter(self.path, "INT-001", self.lane_configs, 1)
        self.now = 1700000000.0
        self.payload = {"intersectionId": "INT-001", "sourceId": "VISION-INT-001",
                        "sessionId": "test-session", "sequence": 1, "meta": {"frameId": 1},
                        "lanes": [{"laneId": lid, "direction": direction, "count": 7,
                                   "queuedCount": 1, "movingCount": 5, "unknownCount": 1,
                                   "valid": True, "role": "queue", "cameraId": f"CAM-{direction}",
                                   "observedAt": iso_utc(self.now - .1)}
                                  for lid, direction in self.expected.items()]}
        self.payload["lanes"].append({"laneId": "NE1", "direction": "N", "count": 99,
                                      "valid": False, "role": "upstream", "cameraId": "CAM-NE",
                                      "observedAt": None})

    def tearDown(self):
        shutil.rmtree(self.directory)

    def write_profile(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    def test_occupancy_all_twelve_lanes_and_registered_counting_source(self):
        result = self.adapter.build(self.payload, self.now)
        self.assertEqual(result["cameraId"], "CAM-01")
        self.assertEqual(result["timestamp"], iso_utc(self.now - .1))
        self.assertEqual(result["lanes"], [{"laneId": lid, "direction": direction, "count": 7}
                                           for lid, direction in self.expected.items()])
        self.assertEqual(result["meta"]["physicalCameraIds"], ["CAM-E", "CAM-N", "CAM-S", "CAM-W"])
        self.assertEqual(self.payload["lanes"][0]["queuedCount"], 1)

    def test_one_bad_required_lane_suppresses_entire_snapshot(self):
        changes = ({"valid": False, "count": None}, {"direction": "S"}, {"role": "upstream"},
                   {"count": -1}, {"count": True}, {"count": 1.5},
                   {"observedAt": None}, {"observedAt": "2023-11-14T22:13:20"},
                   {"observedAt": iso_utc(self.now - .751)}, {"observedAt": iso_utc(self.now + .001)})
        for change in changes:
            with self.subTest(change=change):
                payload = copy.deepcopy(self.payload)
                payload["lanes"][0].update(change)
                self.assertIsNone(self.adapter.build(payload, self.now))
                self.assertTrue(self.adapter.blockers)

    def test_missing_duplicate_extra_and_wrong_intersection_suppress(self):
        for mode in ("missing", "duplicate", "extra", "wrong_site"):
            with self.subTest(mode=mode):
                payload = copy.deepcopy(self.payload)
                if mode == "missing":
                    payload["lanes"].pop(0)
                elif mode == "duplicate":
                    payload["lanes"].append(copy.deepcopy(payload["lanes"][0]))
                elif mode == "extra":
                    payload["lanes"].append({"laneId": "N4", "role": "queue"})
                else:
                    payload["intersectionId"] = "INT-002"
                self.assertIsNone(self.adapter.build(payload, self.now))

    def test_oldest_required_observation_and_measured_zero(self):
        self.payload["lanes"][0].update(count=0, observedAt=iso_utc(self.now - .6))
        result = self.adapter.build(self.payload, self.now)
        self.assertEqual(result["timestamp"], iso_utc(self.now - .6))
        self.assertEqual(result["lanes"][0]["count"], 0)

    def test_profile_rejects_partial_wrong_direction_and_bad_budget(self):
        for mode in ("missing", "wrong_direction", "site", "camera", "budget", "nan"):
            with self.subTest(mode=mode):
                config = copy.deepcopy(self.config)
                lanes = copy.deepcopy(self.lane_configs)
                if mode == "missing":
                    lanes[0].pop("N3")
                elif mode == "wrong_direction":
                    lanes[0]["N3"]["direction"] = "S"
                elif mode == "site":
                    config["intersection_id"] = "INT-002"
                elif mode == "camera":
                    config["camera_id"] = ""
                elif mode == "budget":
                    config["controller_freshness_ms"] = 1750
                else:
                    config["max_observation_age_ms"] = float("nan")
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(config, f)
                with self.assertRaises(ValueError):
                    MainControllerAdapter(self.path, "INT-001", lanes, 1)


class TestMainRunner(unittest.TestCase):
    def setUp(self):
        self.directory = tmpdir()
        self.profile = os.path.join(self.directory, "main.json")
        with open(self.profile, "w", encoding="utf-8") as f:
            json.dump({"intersection_id": "INT-001", "camera_id": "CAM-01",
                       "expected_lanes": {f"{d}1": d for d in "NESW"}}, f)
        self.cameras = {name: {"lanes": {f"{d}1": {"direction": d, "polygon": SQUARE}}}
                        for name, d in zip(("north", "east", "south", "west"), "NESW")}
        self.cameras["northeast"] = {"lanes": {"NE1": {"direction": "N", "polygon": SQUARE, "role": "upstream"}}}

    def tearDown(self):
        shutil.rmtree(self.directory)

    def pipeline(self, **kwargs):
        return make_pipeline(self.directory, self.cameras, pub_interval=1,
                             output_mode="controller", controller_config=self.profile, **kwargs)

    def observe(self, pipeline, sources, indices=range(5), obs_t=0):
        for offset in (0, .02):
            for i in indices:
                sources[i].push(obs_t=obs_t + offset)
            pipeline.step()

    def test_real_publish_failure_suppression_health_and_recovery(self):
        health_path = os.path.join(self.directory, "health.json")
        publisher = FakePublisher()
        pipeline, sources = self.pipeline(publisher=publisher, health_file=health_path)
        self.observe(pipeline, sources)
        payload = pipeline.publish_once()
        self.assertEqual(pipeline.mqtt_topic, "traffic/counts")
        self.assertEqual(len(publisher.payloads[-1]["lanes"]), 4)
        self.assertEqual(len(payload["lanes"]), 5)
        self.assertTrue(all(lane["count"] == 0 for lane in publisher.payloads[-1]["lanes"]))
        pipeline.cams[1].last_obs_mono = time.monotonic() - 10
        pipeline.cams[1].last_obs_wall = time.time() - 10
        before = len(publisher.payloads)
        canonical = pipeline.publish_once()
        self.assertEqual(len(publisher.payloads), before)
        self.assertIsNone(next(l for l in canonical["lanes"] if l["laneId"] == "E1")["count"])
        with open(health_path, encoding="utf-8") as f:
            health = json.load(f)
        self.assertEqual(health["controllerDelivery"]["state"], "suppressed")
        self.assertEqual(len(publisher.health), 2)
        self.observe(pipeline, sources, obs_t=.1)
        pipeline.publish_once()
        self.assertEqual(len(publisher.payloads), before + 1)
        self.assertEqual(pipeline.controller_delivery["state"], "publishing")

    def test_upstream_outage_does_not_block_stopline_counts(self):
        pipeline, sources = self.pipeline()
        self.observe(pipeline, sources, indices=range(4))
        canonical = pipeline.publish_once()
        self.assertIsNone(canonical["lanes"][-1]["count"])
        self.assertEqual(pipeline.controller_delivery["state"], "publishing")
        self.assertEqual(len(pipeline.publisher.payloads[-1]["lanes"]), 4)

    def test_broker_failure_not_claimed_as_success(self):
        pipeline, sources = self.pipeline(publisher=FakePublisher(succeed=False))
        self.observe(pipeline, sources)
        pipeline.publish_once()
        self.assertEqual(pipeline.controller_delivery["state"], "disconnected")
        self.assertEqual(pipeline.controller_delivery["accepted"], 0)

    def test_operator_health_check_reports_blocked_output_with_healthy_cameras(self):
        from tools.healthcheck import check
        path = os.path.join(self.directory, "health.json")
        for state in ("suppressed", "disconnected"):
            with self.subTest(state=state):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"validCameras": 5, "totalCameras": 5, "cameras": [],
                               "controllerDelivery": {"state": state, "blockers": ["source_age_or_broker"]}}, f)
                code, message = check(path, 10)
                self.assertEqual(code, 1)
                self.assertIn(state, message)
                self.assertEqual(check(path, 10, stale_only=True)[0], 0)

    def test_required_camera_maintenance_suppresses_counts(self):
        maintenance = os.path.join(self.directory, "maintenance.json")
        with open(maintenance, "w", encoding="utf-8") as f:
            json.dump({"east": "lens cleaning"}, f)
        pipeline, sources = self.pipeline(maintenance_file=maintenance)
        self.observe(pipeline, sources)
        pipeline.publish_once()
        self.assertEqual(pipeline.publisher.payloads, [])
        self.assertEqual(pipeline.controller_delivery["state"], "suppressed")
        self.assertIn("E1:lens cleaning", pipeline.controller_delivery["blockers"])

    def test_default_profile_requires_all_stopline_calibrations(self):
        with self.assertRaisesRegex(ValueError, "Stop-line calibration"):
            make_pipeline(self.directory, self.cameras, pub_interval=1, output_mode="controller")

    def test_recorded_footage_requires_explicit_isolated_opt_in(self):
        from run_multi_camera import BatchedCameraPipeline
        from trt_pipeline.stream import FileReplaySource
        from tests.test_stream import FakeCap
        names = list(self.cameras)
        configs = [write_config(self.directory, n, self.cameras[n]["lanes"]) for n in names]
        kwargs = dict(camera_names=names, config_paths=configs, video_sources=["file.avi"] * 5,
                      sources=[FileReplaySource(n, "file.avi", capture_factory=FakeCap) for n in names],
                      model_path="unused.pt", model=FakeModel(), publisher=FakePublisher(), pub_interval=1,
                      output_mode="controller", controller_config=self.profile)
        with self.assertRaisesRegex(ValueError, "Recorded footage"):
            BatchedCameraPipeline(**kwargs)
        pipeline = BatchedCameraPipeline(**kwargs, allow_replay_controller=True)
        self.assertTrue(pipeline.replay_mode)

    def test_main_publisher_uses_qos_zero(self):
        with patch("trt_pipeline.publisher.MQTTPublisher") as publisher:
            # A real constructor with source/model fakes never opens a connection.
            from run_multi_camera import BatchedCameraPipeline
            from tests.helpers import FakeSource
            names = list(self.cameras)
            configs = [write_config(self.directory, n, self.cameras[n]["lanes"]) for n in names]
            BatchedCameraPipeline(camera_names=names, config_paths=configs,
                                  video_sources=["rtsp://test"] * 5, sources=[FakeSource(n) for n in names],
                                  model_path="unused.pt", model=FakeModel(), pub_interval=1, output_mode="controller",
                                  controller_config=self.profile)
        self.assertEqual(publisher.call_args.kwargs["qos"], 0)


class TestCommittedMain(unittest.TestCase):
    def test_soak_rejects_wrong_controller_behavior_during_both_outage_roles(self):
        from tools.soak import observe_outages
        for role, state, failed in (("queue", "publishing", True), ("queue", "suppressed", False),
                                    ("upstream", "suppressed", True), ("upstream", "publishing", False)):
            with self.subTest(role=role, state=state):
                outage = {"camera": "north", "start": 0, "duration": 10, "invalidSeen": False,
                          "knownDuringOutage": False, "zeroDuringOutage": False, "recoveredSeen": False,
                          "othersValidDuringOutage": True, "otherCameraProblems": []}
                payload = {"cameras": [{"name": "north", "cameraId": "CAM-N", "status": "stale"}],
                           "lanes": [{"cameraId": "CAM-N", "role": role, "valid": False,
                                      "count": None, "queuedCount": None}]}
                observe_outages(payload, 3, [outage], 2, {"state": state})
                self.assertTrue(outage["controllerCheckSeen"])
                self.assertEqual(outage["controllerProjectionFailed"], failed)

    def test_shipped_profile_matches_real_calibrations(self):
        root = Path(__file__).resolve().parents[1]
        lanes = [json.loads((root / "config" / f"config_{name}.json").read_text(encoding="utf-8"))["lane_metrics"]["lanes"]
                 for name in ("north", "east", "south", "west", "northeast")]
        adapter = MainControllerAdapter(root / "config/controller_main.json", "INT-001", lanes, 1)
        self.assertEqual(len(adapter.lanes), 9)
        self.assertNotIn("NE1", adapter.lanes)

    def test_real_model_wire_fixture_is_accepted_by_main(self):
        root = Path(__file__).parent
        payload = json.loads((root / "fixtures/vision_payload_main.json").read_text(encoding="utf-8"))
        result = subprocess.run(["node", str(root / "controller_main_harness.cjs")],
                                input=json.dumps({"payload": payload}), text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["received"]["laneCount"], 9)
        self.assertEqual(report["received"]["totalCars"], sum(l["count"] for l in payload["lanes"]))
        self.assertEqual(report["storedLaneCounts"], {l["laneId"]: l["count"] for l in payload["lanes"]})
        self.assertEqual(report["fallback"]["decision"]["strategy"], "FIXED_CYCLE")

    def test_adapter_output_through_real_intake_aggregation_decision_and_stale_fallback(self):
        fixture = TestMainAdapter()
        fixture.setUp()
        try:
            counts = {"N": [1, 2, 17], "E": [3, 4, 5], "S": [2, 2, 2], "W": [1, 1, 1]}
            for lane in fixture.payload["lanes"][:12]:
                lane["count"] = counts[lane["direction"]][int(lane["laneId"][-1]) - 1]
            payload = fixture.adapter.build(fixture.payload, fixture.now)
            self.assertIsNotNone(payload)
            harness = Path(__file__).with_name("controller_main_harness.cjs")
            for controller in ("MAXPRESSURE_SWITCHING_LOSS", "QUEUE_BASED", "SIMPLE_CYCLE_BASED", "FIXED_CYCLE"):
                with self.subTest(controller=controller):
                    result = subprocess.run(["node", str(harness)], input=json.dumps({"payload": payload, "controller": controller}),
                                            text=True, capture_output=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["received"]["totalCars"], 41)
                    self.assertEqual(report["received"]["laneCount"], 12)
                    self.assertEqual(report["camera"]["directionTotals"], {"N": 20, "E": 12, "S": 6, "W": 3})
                    self.assertEqual(report["aggregate"]["laneMaxByDirection"], {"N": 17, "E": 5, "S": 2, "W": 1})
                    self.assertEqual(report["storedLaneCounts"]["N3"], 17)
                    self.assertNotIn("NE1", report["storedLaneCounts"])
                    self.assertEqual(report["adaptive"]["decision"]["strategy"], controller)
                    self.assertEqual(report["adaptive"]["decision"]["phase"], "N_GO")
                    self.assertEqual(report["fallback"]["decision"]["strategy"], "FIXED_CYCLE")
        finally:
            fixture.tearDown()

    def test_main_snapshot_integrity_and_local_commit_when_available(self):
        root = Path(__file__).parent / "fixtures/controller_main"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        system = Path(__file__).resolve().parents[2] / "smart-traffic-sys"
        git = shutil.which("git")
        for name, expected in manifest["files"].items():
            data = (root / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)
            if git and (system / ".git").exists():
                result = subprocess.run([git, "-c", "safe.directory=" + system.as_posix(), "-C", str(system),
                                         "show", manifest["revision"] + ":" + name], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout, data, name)
