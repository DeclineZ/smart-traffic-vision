"""
Runner-level lane occupancy and payload behaviour, exercised through the real
BatchedCameraPipeline with fake sources/model/publisher (no GPU, camera or broker).
"""

import os
import shutil
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.helpers import SQUARE, FakeModel, FakePublisher, make_pipeline, tmpdir


def box(x, y, w=20, h=20, conf=0.9, cls=0):
    """Detection whose bottom-centre (road contact point) is at (x, y)."""
    return [x - w / 2, y - h, x + w / 2, y, conf, cls]


class ScriptedModel(FakeModel):
    """Detections chosen by the current scenario dict: cam index -> list of boxes."""

    def __init__(self):
        super().__init__()
        self.scene = {}
        self.order = []

    def __call__(self, frames, **kwargs):
        self.calls.append(len(frames))
        out = []
        from tests.helpers import FakeResult
        for i in range(len(frames)):
            cam = self.order[i] if i < len(self.order) else 0
            out.append(FakeResult(np.array(self.scene.get(cam, []), dtype=float).reshape(-1, 6)))
        return out


class RunnerTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir()
        self.model = ScriptedModel()
        self.pub = FakePublisher()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def pipeline(self, cameras, **kw):
        p, sources = make_pipeline(self.dir, cameras, model=self.model, publisher=self.pub, **kw)
        for cam in p.cams:
            cam.tracker.min_hits = 1  # confirm tracks on first sight so single-frame scenarios are observable
        return p, sources

    def frame(self, p, sources, t, cams=None, **push_kw):
        cams = list(range(len(sources))) if cams is None else cams
        for c in cams:
            sources[c].push(obs_t=t, **push_kw)
        self.model.order = cams
        p.step()

    @staticmethod
    def lane(payload, lane_id):
        return next(l for l in payload["lanes"] if l["laneId"] == lane_id)


class TestLaneOccupancy(RunnerTestBase):
    def one_cam(self, **kw):
        return self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}}}, **kw)

    def test_nonempty_then_empty_evaluation_clears_occupancy(self):
        p, s = self.one_cam()
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 1)
        self.model.scene = {0: []}
        self.frame(p, s, 0.1)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 0)

    def test_track_moves_outside_polygons_disappears(self):
        p, s = self.one_cam()
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)
        self.model.scene = {0: [box(52, 99), ]}
        self.frame(p, s, 0.1)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 1)
        self.model.scene = {0: [box(55, 160)]}  # contact point now below the lane
        self.frame(p, s, 0.2)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 0)

    def test_overlapping_polygons_first_lane_wins_and_count_once(self):
        p, s = self.pipeline({"north": {"lanes": {
            "N1": {"direction": "N", "polygon": [[0, 0], [60, 0], [60, 100], [0, 100]]},
            "N2": {"direction": "N", "polygon": [[40, 0], [100, 0], [100, 100], [40, 100]]},
        }}})
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)
        snap = {l["laneId"]: l["count"] for l in p.cams[0].metrics.snapshot()}
        self.assertEqual(snap, {"N1": 1, "N2": 0})

    def test_boundary_point_is_counted(self):
        p, s = self.one_cam()
        self.model.scene = {0: [box(50, 100)]}  # contact point exactly on the bottom edge
        self.frame(p, s, 0.0)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 1)

    def test_bottom_centre_anchor_vs_centre(self):
        # Box centre inside the lane but road contact point outside: not in the lane by default.
        lanes = {"N1": {"direction": "N", "polygon": [[0, 0], [100, 0], [100, 60], [0, 60]]}}
        p, s = self.pipeline({"north": {"lanes": lanes}})
        self.model.scene = {0: [box(50, 80, h=40)]}  # centre y=60..., bottom y=80
        self.frame(p, s, 0.0)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 0)

        p2, s2 = self.pipeline({"north": {"lanes": lanes, "extra": {}}})
        p2.cams[0].anchor = "center"
        self.frame(p2, s2, 0.0)
        self.assertEqual(p2.cams[0].metrics.snapshot()[0]["count"], 1)

    def test_skipped_frames_do_not_update_occupancy_or_motion(self):
        p, s = self.one_cam(skip_frames=1)
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)  # inferred
        self.assertEqual(len(self.model.calls), 1)
        tid = next(iter(p.cams[0].motion._tracks))
        hist_before = len(p.cams[0].motion._tracks[tid].samples)
        self.frame(p, s, 0.04)  # skipped: no inference, no new motion sample
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(len(p.cams[0].motion._tracks[tid].samples), hist_before)
        self.assertEqual(p.cams[0].metrics.snapshot()[0]["count"], 1)

    def test_publication_does_not_consume_occupancy(self):
        p, s = self.one_cam()
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)
        payload = p.publish_once()
        lane = self.lane(payload, "N1")
        self.assertEqual(lane["count"], 1)
        self.assertEqual(lane["unknownStateCount"], 1)  # no motion history yet
        retained = p.cams[0].metrics.snapshot()[0]
        self.assertEqual(retained["count"], 1)
        payload2 = p.publish_once()
        self.assertEqual(self.lane(payload2, "N1")["count"], 1)

    def test_tracker_class_is_used_not_nearest_neighbour(self):
        # Two overlapping detections of different classes; each track must keep its own class.
        p, s = self.one_cam(enable_voting=False)
        self.model.scene = {0: [box(40, 50, cls=0), box(60, 50, cls=1)]}
        for i in range(3):
            self.frame(p, s, i * 0.1)
        classes = p.cams[0].metrics.snapshot()[0]["classes"]
        self.assertEqual(classes["car"], 1)
        self.assertEqual(classes["motorcycle"], 1)


class TestQueueStatesThroughRunner(RunnerTestBase):
    def test_stopped_vehicle_becomes_queued_moving_vehicle_moving(self):
        p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]}}}})
        for i in range(40):
            t = i * 0.1
            self.model.scene = {0: [box(100, 500), box(300 + 40 * t, 500)]}  # 2nd: 2 box-heights/s
            self.frame(p, s, t)
        lane = p.cams[0].metrics.snapshot()[0]
        self.assertEqual(lane["queuedCount"], 1)
        self.assertEqual(lane["movingCount"], 1)


class TestCameraIndependence(RunnerTestBase):
    def two_cams(self, **kw):
        return self.pipeline({
            "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
            "south": {"lanes": {"S1": {"direction": "S", "polygon": SQUARE}}},
        }, **kw)

    def test_missing_camera_does_not_block_healthy_camera(self):
        p, s = self.two_cams(max_observation_age_s=0.5)
        self.model.scene = {0: [box(50, 50)], 1: [box(50, 50)]}
        self.frame(p, s, 0.0)
        # south goes silent; north keeps delivering
        for i in range(1, 4):
            self.frame(p, s, i * 0.1, cams=[0])
        p.cams[1].last_obs_wall = time.time() - 5  # south's last observation is old
        p.cams[1].last_obs_mono = time.monotonic() - 5
        payload = p.publish_once()
        n1, s1 = self.lane(payload, "N1"), self.lane(payload, "S1")
        self.assertTrue(n1["valid"])
        self.assertEqual(n1["count"], 1)
        self.assertFalse(s1["valid"])
        self.assertIsNone(s1["count"])  # unknown, never zero
        self.assertIsNone(s1["queuedCount"])
        cams = {c["name"]: c["status"] for c in payload["cameras"]}
        self.assertEqual(cams, {"north": "ok", "south": "stale"})

    def test_camera_never_seen_is_invalid(self):
        p, s = self.two_cams()
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0, cams=[0])
        payload = p.publish_once()
        self.assertFalse(self.lane(payload, "S1")["valid"])
        self.assertEqual(self.lane(payload, "S1")["invalidReason"], "starting")

    def test_stale_frame_is_not_processed(self):
        p, s = self.two_cams(max_frame_age_s=0.5)
        s[0].push(obs_t=0.0, captured_wall=time.time() - 3)
        p.step()
        self.assertEqual(self.model.calls, [])
        self.assertEqual(p.cams[0].stale_frames_dropped, 1)

    def test_resolution_mismatch_invalidates_camera(self):
        p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}},
                                        "extra": {"calibration": {"resolution": [1920, 1080]}}}})
        self.frame(p, s, 0.0)  # fake frames are 100x100
        self.assertEqual(self.model.calls, [])
        payload = p.publish_once()
        self.assertFalse(self.lane(payload, "N1")["valid"])
        self.assertEqual(payload["cameras"][0]["reason"], "resolution_mismatch")

    def test_epoch_change_resets_camera_state(self):
        p, s = self.two_cams()
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0, cams=[0])
        self.frame(p, s, 0.1, cams=[0])
        old_tracker = p.cams[0].tracker
        self.assertGreater(len(p.cams[0].motion), 0)
        s[0].push(obs_t=0.0, epoch=2)  # reconnect / file loop
        self.model.order = [0]
        self.model.scene = {0: []}
        p.step()
        self.assertIsNot(p.cams[0].tracker, old_tracker)
        self.assertEqual(len(p.cams[0].motion), 0)


class TestPayloadContract(RunnerTestBase):
    def test_schema_session_sequence_and_invariants(self):
        p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE},
                                                   "N2": {"direction": "N", "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]]}}}})
        self.model.scene = {0: [box(50, 50), box(60, 80, cls=1)]}
        self.frame(p, s, 0.0)
        a, b = p.publish_once(), p.publish_once()
        self.assertEqual(a["schemaVersion"], "2.0")
        self.assertEqual(a["sessionId"], b["sessionId"])
        self.assertEqual(b["sequence"], a["sequence"] + 1)
        self.assertEqual(a["cameraId"], "VISION-INT-001")
        for lane in a["lanes"]:
            self.assertEqual(lane["count"], lane["queuedCount"] + lane["movingCount"] + lane["unknownStateCount"])
            self.assertEqual(lane["count"], sum(lane["classes"].values()))
            self.assertEqual(lane["cameraId"], "CAM-NORTH")
            self.assertTrue(lane["valid"])
        self.assertLessEqual(a["observedAt"], a["publishedAt"])

    def test_failed_publish_keeps_gate_interval(self):
        gate = {"gate_id": "G_N", "p1": [0, 50], "p2": [100, 50], "type": "stopline",
                "direction": [0, 1], "target_dir": "N"}
        self.pub.succeed = False
        with patch("trt_pipeline.gates.time.monotonic", return_value=0) as clock:
            p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}, "gates": [gate]}})
            for i, y in enumerate([36, 44, 52, 60]):
                clock.return_value = i * .1
                self.model.scene = {0: [box(50, y)]}
                self.frame(p, s, i * .1)
            clock.return_value = .31
            first = p.publish_once()
            self.assertEqual(first["traffic_flow"]["interval"]["gates"][0]["count"], 1)
            clock.return_value = .32
            second = p.publish_once()  # failed publication must retain the crossing
            self.assertEqual(second["traffic_flow"]["interval"]["gates"][0]["count"], 1)
            self.pub.succeed = True
            clock.return_value = .33
            p.publish_once()
            for t in (.4, .5):
                clock.return_value = t
                self.frame(p, s, t)
            clock.return_value = .51
            third = p.publish_once()
            self.assertEqual(third["traffic_flow"]["interval"]["gates"][0]["count"], 0)

    def test_health_published_with_each_payload(self):
        p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}}})
        self.frame(p, s, 0.0)
        p.publish_once()
        self.assertEqual(self.pub.health[-1]["status"], "online")
        self.assertEqual(self.pub.health[-1]["validCameras"], 1)


class TestStartupValidation(RunnerTestBase):
    def test_duplicate_gate_ids_rejected(self):
        g = {"gate_id": "G1", "p1": [0, 50], "p2": [100, 50], "type": "stopline", "target_dir": "N"}
        with self.assertRaises(ValueError):
            self.pipeline({
                "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}, "gates": [g]},
                "south": {"lanes": {"S1": {"direction": "S", "polygon": SQUARE}}, "gates": [g]},
            })

    def test_gate_without_direction_rejected(self):
        g = {"gate_id": "G1", "p1": [0, 50], "p2": [100, 50], "type": "stopline"}
        with self.assertRaises(ValueError):
            self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}, "gates": [g]}})

    def test_lane_without_valid_direction_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline({"north": {"lanes": {"N1": {"direction": "X", "polygon": SQUARE}}}})

    def test_duplicate_lane_ids_across_cameras_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline({
                "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
                "south": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
            })

    def test_missing_model_file_fails(self):
        from run_multi_camera import BatchedCameraPipeline
        from tests.helpers import FakeSource, write_config
        cfg = write_config(self.dir, "north", {"N1": {"direction": "N", "polygon": SQUARE}})
        with self.assertRaises(FileNotFoundError):
            BatchedCameraPipeline(["north"], [cfg], ["fake://x"], model_path=os.path.join(self.dir, "missing.pt"),
                                  publisher=FakePublisher(), sources=[FakeSource()])


class TestRolesAndMaintenance(RunnerTestBase):
    def test_lane_role_published_and_validated(self):
        p, s = self.pipeline({"northeast": {"lanes": {"NE1": {"direction": "N", "role": "upstream", "polygon": SQUARE}}}})
        self.frame(p, s, 0.0)
        self.assertEqual(p.publish_once()["lanes"][0]["role"], "upstream")
        with self.assertRaises(ValueError):
            self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "role": "stopline", "polygon": SQUARE}}}})

    def test_maintenance_file_marks_camera_unknown_until_removed(self):
        import json as _json
        mfile = os.path.join(self.dir, "maintenance.json")
        p, s = self.pipeline({"north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}}},
                             maintenance_file=mfile)
        self.model.scene = {0: [box(50, 50)]}
        self.frame(p, s, 0.0)
        self.assertTrue(self.lane(p.publish_once(), "N1")["valid"])
        with open(mfile, "w", encoding="utf-8") as f:
            _json.dump({"north": "lens cleaning"}, f)
        payload = p.publish_once()
        self.assertFalse(self.lane(payload, "N1")["valid"])
        self.assertIsNone(self.lane(payload, "N1")["count"])
        self.assertEqual(payload["cameras"][0]["status"], "maintenance")
        self.assertEqual(payload["cameras"][0]["reason"], "lens cleaning")
        os.remove(mfile)
        self.assertFalse(self.lane(p.publish_once(), "N1")["valid"])
        self.frame(p, s, .1)
        self.frame(p, s, .2)  # fresh tracker confirmation after maintenance
        self.assertTrue(self.lane(p.publish_once(), "N1")["valid"])


class TestRunPlan(unittest.TestCase):
    def parse(self, *argv):
        from run_multi_camera import build_pipeline_args
        return build_pipeline_args().parse_args(list(argv))

    def test_default_is_all_five_cameras(self):
        from run_multi_camera import resolve_run_plan
        names, configs, sources = resolve_run_plan(self.parse())
        self.assertEqual(names, ["north", "south", "east", "west", "northeast"])
        self.assertEqual(len(sources), 5)

    def test_source_count_must_match(self):
        from run_multi_camera import resolve_run_plan
        with self.assertRaises(ValueError):
            resolve_run_plan(self.parse("--cameras", "north", "south", "--videos", "rtsp://a/1"))

    def test_unknown_camera_rejected(self):
        from run_multi_camera import resolve_run_plan
        with self.assertRaises(ValueError):
            resolve_run_plan(self.parse("--cameras", "north_2"))

    def test_sources_file_must_cover_every_camera(self):
        from run_multi_camera import resolve_run_plan
        d = tmpdir()
        try:
            path = os.path.join(d, "sources.json")
            with open(path, "w") as f:
                f.write('{"north": "rtsp://user:pw@10.0.0.1/stream"}')
            names, _, sources = resolve_run_plan(self.parse("--cameras", "north", "--sources-file", path))
            self.assertEqual(sources, ["rtsp://user:pw@10.0.0.1/stream"])
            with self.assertRaises(ValueError):
                resolve_run_plan(self.parse("--cameras", "north", "south", "--sources-file", path))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
