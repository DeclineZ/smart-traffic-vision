"""
Stress tests and adversarial edge-case probes for the `dev` branch.

Covers:
1. Thai-model class ID vs COCO class ID mismatch in GateFlowManager -> CorridorQueueEstimator.
2. CorridorQueueEstimator stale/health-invalid/maintenance camera ghost-state retention.
3. Stale Cam 2 without health_invalid keeps feeding old metrics into CorridorQueueEstimator.
4. Payload schema 2.0 count invariant violation when motorcycles + Cam 2/cart co-exist.
5. VirtualToken ttl_remaining_sec double-subtraction in telemetry.
6. GateFlowManager.get_mqtt_telemetry flow_rate window scaling mismatch.
7. Handover Priority 1 (age 0.0s) hijacking Priority 2 (age >= 0.7s) against docstring spec.
8. Cam 1 tracker ID switch silently draining Virtual Cart tokens.
9. stream_cameras.py Windows-only msvcrt import crash on macOS/Linux.
10. VirtualGate endpoint crossing discrepancy (segments_intersect vs get_segment_intersection).
11. High-volume randomized fuzzing & soak tests (ByteTrack, MotionStateClassifier, CorridorQueueEstimator).
12. End-to-end 6-camera (with opt-in north_upstream) stress with real config files.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time
import unittest
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.byetrack import ByteTrack
from tests.helpers import SQUARE, FakeModel, FakePublisher, make_pipeline, tmpdir
from trt_pipeline.corridor import CorridorQueueEstimator
from trt_pipeline.gates import GateFlowManager, VirtualGate, get_segment_intersection, segments_intersect
from trt_pipeline.motion import MOVING, QUEUED, UNKNOWN, MotionStateClassifier, QueueSettings
from trt_pipeline.stream import FramePacket


class TestDevLoopholesAndBugs(unittest.TestCase):
    def test_01_thai_model_truck_class_id_3_misclassified_as_motorcycle_at_gate(self):
        """
        BUG CHECK: In Thai traffic models (0:car, 1:motorcycle, 2:bus, 3:truck, 4:three_wheeler),
        GateFlowManager.update_tracks passes raw integer cls_id=3 (truck) to
        CorridorQueueEstimator.record_gate_crossing, where is_motorcycle(3) returns True!
        """
        gm = GateFlowManager(
            camera_names=["northeast", "north"],
            camera_ids=["CAM-NE", "CAM-N"],
            lane_mappings={"NE1": "N1"},
        )
        gm.set_camera_lanes(0, {"NE1": {"direction": "N", "role": "upstream"}})
        gm.set_camera_lanes(1, {"N1": {"direction": "N", "role": "queue"}})
        gm.add_gate(VirtualGate(
            "GATE_N_IN", cam_idx=0, p1=(0, 50), p2=(100, 50),
            gate_type="ingress", direction_vec=(0, 1), target_dir="N", lane_id="NE1",
        ))

        gm.update_tracks(0, np.array([[40, 20, 60, 40, 101, 3]]), now=1.0)
        gm.update_tracks(0, np.array([[40, 40, 60, 60, 101, 3]]), now=1.1)

        self.assertEqual(gm.gates["GATE_N_IN"].count, 1)
        self.assertEqual(
            len(gm.corridor_estimator.lanes["N1"].cart), 1,
            "Thai-model truck (class_id=3) was misclassified as motorcycle and skipped cart enqueue!",
        )
        self.assertEqual(gm.corridor_estimator.moto_pools["N"]["cumulative_inflow"], 0)

    def test_02_cam2_health_invalid_leaves_stale_counts_and_spillback_in_corridor(self):
        """
        BUG CHECK: If Cam 2 (NE1) goes health-invalid/maintenance, update_visual_occupancy
        omits NE1 from lane_vis. Because N1 is in reverse_mappings, lst.vis_cam2 and
        lst.stopped_cam2 are never cleared and keep triggering fake spillback!
        """
        d = tmpdir()
        try:
            cameras = {
                "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
                "northeast": {
                    "lanes": {"NE1": {"direction": "N", "target_lane": "N1", "role": "upstream", "polygon": SQUARE}},
                    "extra": {"camera_health": {"check_interval_s": 0.01}},
                },
            }
            pub = FakePublisher()
            det = np.array([[20.0, 20.0, 60.0, 60.0, 0.9, 0]], dtype=float)
            pipeline, sources = make_pipeline(
                d, cameras, publisher=pub, pub_interval=1.0,
                model=FakeModel(lambda idx, call: det),
            )
            for c in pipeline.cams:
                c.tracker.min_hits = 1

            # Step over >1.5s so MotionStateClassifier transitions stopped car -> QUEUED
            for t in (0.1, 0.8, 1.5, 2.2):
                sources[0].push(obs_t=t)
                sources[1].push(obs_t=t)
                pipeline.step()

            n1 = pipeline.gate_manager.corridor_estimator.lanes["N1"]
            self.assertTrue(n1.stopped_cam1)
            self.assertTrue(n1.stopped_cam2)
            self.assertTrue(n1.spillback)

            # Black frame on northeast triggers health_invalid ("no_signal")
            time.sleep(0.02)
            sources[1].push(frame=np.zeros((100, 100, 3), dtype=np.uint8), obs_t=2.8)
            sources[0].push(obs_t=2.8)
            pipeline.step()

            self.assertTrue(pipeline.cams[1].health_invalid)
            self.assertFalse(n1.spillback, "Fake spillback persisted after Cam 2 became health_invalid!")
            self.assertEqual(n1.vis_cam2, 0, "Stale Cam 2 visual count persisted after Cam 2 became health_invalid!")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_03_stale_cam2_without_health_invalid_keeps_feeding_metrics(self):
        """
        BUG CHECK: When Cam 2 stops sending packets (goes stale/offline, NOT health_invalid),
        run_multi_camera.py checks `not c.health_invalid and c.frame_invalid_reason is None and not c.maintenance_active`,
        so it continues feeding Cam 2's frozen `c.metrics` into CorridorQueueEstimator indefinitely!
        """
        d = tmpdir()
        try:
            cameras = {
                "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
                "northeast": {
                    "lanes": {"NE1": {"direction": "N", "target_lane": "N1", "role": "upstream", "polygon": SQUARE}},
                    "extra": {"camera_health": {"check_interval_s": 0.01}},
                },
            }
            pub = FakePublisher()
            det = np.array([[20.0, 20.0, 60.0, 60.0, 0.9, 0]], dtype=float)
            pipeline, sources = make_pipeline(
                d, cameras, publisher=pub, pub_interval=1.0,
                max_observation_age_s=0.2,
                model=FakeModel(lambda idx, call: det),
            )
            for c in pipeline.cams:
                c.tracker.min_hits = 1

            for t in (0.1, 0.8, 1.5, 2.2):
                sources[0].push(obs_t=t)
                sources[1].push(obs_t=t)
                pipeline.step()

            # Simulate Cam 2 going silent for > max_observation_age_s
            time.sleep(0.25)
            sources[0].push(obs_t=2.8)
            pipeline.step()

            st_ne = pipeline.camera_status(pipeline.cams[1])
            self.assertEqual(st_ne["status"], "stale")
            n1 = pipeline.gate_manager.corridor_estimator.lanes["N1"]
            self.assertEqual(n1.vis_cam2, 0, "Stale Cam 2 still fed old car counts into CorridorQueueEstimator!")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_04_payload_count_invariant_and_motorcycle_drop(self):
        """
        BUG CHECK: In run_multi_camera.py, when has_dual_cam_or_cart flips True,
        lane['count'] is replaced by demand_count (cars only), which:
        1) Violates schema 2.0 invariant: count == queuedCount + movingCount + unknownStateCount
        2) Drops motorcycles from lane['count'] when an upstream car appears!
        """
        d = tmpdir()
        try:
            cameras = {
                "north": {"lanes": {"N1": {"direction": "N", "polygon": SQUARE}}},
                "northeast": {
                    "lanes": {"NE1": {"direction": "N", "target_lane": "N1", "role": "upstream", "polygon": SQUARE}},
                    "extra": {"camera_health": {"check_interval_s": 0.01}},
                },
            }
            pub = FakePublisher()
            dets_n = np.array([
                [10, 10, 30, 30, 0.9, 1],
                [35, 10, 55, 30, 0.9, 1],
                [60, 10, 80, 30, 0.9, 1],
                [20, 50, 50, 80, 0.9, 0],
            ], dtype=float)
            dets_ne = np.array([[20, 20, 50, 50, 0.9, 0]], dtype=float)

            phase = {"ne": False}

            def dets_fn(idx, call):
                if idx == 0:
                    return dets_n
                return dets_ne if phase["ne"] else np.empty((0, 6))

            pipeline, sources = make_pipeline(
                d, cameras, publisher=pub, pub_interval=1.0,
                model=FakeModel(dets_fn),
            )
            for c in pipeline.cams:
                c.tracker.min_hits = 1

            sources[0].push(obs_t=0.1)
            sources[1].push(obs_t=0.1)
            pipeline.step()
            p1 = pipeline.build_payload()
            n1_before = next(l for l in p1["lanes"] if l["laneId"] == "N1")
            self.assertEqual(n1_before["count"], 4)

            phase["ne"] = True
            sources[0].push(obs_t=0.2)
            sources[1].push(obs_t=0.2)
            pipeline.step()
            p2 = pipeline.build_payload()
            n1_after = next(l for l in p2["lanes"] if l["laneId"] == "N1")

            self.assertEqual(
                n1_after["count"],
                n1_after["queuedCount"] + n1_after["movingCount"] + n1_after["unknownStateCount"],
                f"Schema 2.0 invariant broken: count={n1_after['count']} vs "
                f"q+m+u={n1_after['queuedCount'] + n1_after['movingCount'] + n1_after['unknownStateCount']}",
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_05_token_ttl_remaining_sec_double_subtraction(self):
        """
        BUG CHECK: _tick_carts decrements tk.ttl -= dt AND increments tk.age += dt,
        then get_telemetry() computes t.ttl - t.age (double-counting elapsed time!).
        """
        est = CorridorQueueEstimator(default_ttl_s=8.5)
        est.register_lane("N1", "N")
        est.enqueue_token("N1", "car", ttl=8.5)
        est.update_visual_occupancy({"N1": {"cars_total": 0, "cars_queued": 0, "has_stopped": False}}, now_mono=1.0, dt=4.5)

        tel = est.get_telemetry()
        tok = tel["by_lane"]["N1"]["tokens_detail"][0]
        self.assertAlmostEqual(tok["ttl_remaining_sec"], 4.0, places=2)

    def test_06_mqtt_telemetry_flow_rate_uses_wrong_interval(self):
        """
        BUG CHECK: GateFlowManager.get_mqtt_telemetry calls corridor_estimator.get_telemetry()
        without interval_sec=window_s, so flow_rate is divided by 1.0s instead of window_s!
        """
        gm = GateFlowManager(camera_names=["north"], camera_ids=["CAM-N"])
        gm.set_camera_lanes(0, {"N1": {"direction": "N", "role": "queue"}})
        gm.corridor_estimator.record_gate_crossing("stopline", "N", "N1", "car", count=5)
        gm.interval_start_mono = time.monotonic() - 5.0
        tel = gm.get_mqtt_telemetry(camera_validity={0: True})
        self.assertLess(
            tel["by_lane"]["N1"]["flow_rate"], 1.2,
            f"flow_rate used 1.0s default instead of ~5.0s window: {tel['by_lane']['N1']['flow_rate']}",
        )

    def test_07_handover_priority_1_vs_2_age_rule(self):
        """
        BUG CHECK: Docstring says Priority 1 is exact class with age >= 1.0s,
        Priority 2 is any class with age >= 0.7s (oldest FIFO).
        Code puts min_age=0.0 inside Priority 1, letting a brand-new 0.1s car hijack
        handover before an old 5.0s truck in the same cart!
        """
        est = CorridorQueueEstimator()
        est.register_lane("N1", "N")
        t_truck = est.enqueue_token("N1", "truck", ttl=8.5)
        t_truck.age = 5.0
        t_car_new = est.enqueue_token("N1", "car", ttl=8.5)
        t_car_new.age = 0.1

        handed = est.handover_token("N1", "car", require_min_age=True)
        self.assertEqual(
            handed.id, t_truck.id,
            "0.1s brand-new car token hijacked handover ahead of 5.0s token in FIFO!",
        )

    def test_08_cam1_id_switch_drains_virtual_cart(self):
        """
        BUG CHECK: If a vehicle already inside Cam 1 undergoes a tracker ID switch,
        update_visual_occupancy sees a new track ID in car_tracks and immediately calls handover_token,
        draining tokens from the blind-spot cart (or borrowing from adjacent frozen lanes)!
        """
        est = CorridorQueueEstimator()
        est.register_lane("N1", "N")
        est.register_lane("N2", "N")

        # Establish track 10 in N1 and track 20 in N2 first
        est.update_visual_occupancy({
            "N1": {"cars_total": 1, "cars_queued": 1, "has_stopped": True, "car_tracks": {10: "car"}},
            "N2": {"cars_total": 1, "cars_queued": 1, "has_stopped": True, "car_tracks": {20: "car"}},
        }, now_mono=0.9, dt=0.1)

        # Now 1 car enters blind spot in N2 (1 token in N2 cart)
        est.enqueue_token("N2", "car", ttl=8.5)
        self.assertEqual(len(est.lanes["N2"].cart), 1)

        # Track 10 in N1 flickers to track 11 (ID switch while cars_total stays 1!)
        est.update_visual_occupancy({
            "N1": {"cars_total": 1, "cars_queued": 1, "has_stopped": True, "car_tracks": {11: "car"}},
            "N2": {"cars_total": 1, "cars_queued": 1, "has_stopped": True, "car_tracks": {20: "car"}},
        }, now_mono=1.1, dt=0.2)

        self.assertEqual(
            len(est.lanes["N2"].cart), 1,
            "Tracker ID switch inside N1 stole/drained N2's blind-spot cart token!",
        )

    def test_09_stream_cameras_import_on_non_windows(self):
        """BUG CHECK: stream_cameras.py unconditionally imports Windows-only `msvcrt`."""
        if sys.platform != "win32":
            with self.assertRaises(ModuleNotFoundError):
                import stream_cameras  # noqa: F401

    def test_10_virtual_gate_endpoint_crossing_inconsistency(self):
        """
        BUG CHECK: segments_intersect() uses strict ccw inequality and returns False
        when a vehicle crosses directly through a gate endpoint (e.g. (100, 200)),
        whereas get_segment_intersection() reports a valid intersection point.
        """
        p1, p2 = (100.0, 200.0), (400.0, 200.0)
        prev_pt, curr_pt = (100.0, 190.0), (100.0, 210.0)
        inter_pt = get_segment_intersection(prev_pt, curr_pt, p1, p2)
        self.assertIsNotNone(inter_pt)
        self.assertTrue(
            segments_intersect(prev_pt, curr_pt, p1, p2),
            "segments_intersect missed a crossing passing directly through gate endpoint p1!",
        )


class TestHighVolumeStressAndFuzz(unittest.TestCase):
    def test_fuzz_corridor_estimator_10k_steps(self):
        """
        Fuzzes CorridorQueueEstimator for 10,000 steps with random vehicles, dt spikes,
        out-of-order/zero dt, borrowing, spillback toggles, and empty resets.
        Verifies bounded state, non-negative counts, and capacity invariants.
        """
        rng = random.Random(42)
        est = CorridorQueueEstimator(
            lane_mappings={"NE1": "N1", "NE2": "N2", "NE3": "N3", "NU1": "N1", "NU2": "N2", "NU3": "N3"},
            max_lane_capacity=40,
            default_cap_0_100m=15,
            default_ttl_s=8.5,
        )
        for i in (1, 2, 3):
            est.register_lane(f"N{i}", "N", role="queue")
            est.register_lane(f"NE{i}", "N", target_lane_id=f"N{i}", role="upstream")
            est.register_lane(f"NU{i}", "N", target_lane_id=f"N{i}", role="upstream")
        est.register_lane("OUT_N", "N", role="outgoing")

        t = 0.0
        next_tid = 1
        active_c1 = {f"N{i}": {} for i in (1, 2, 3)}
        for step in range(10000):
            dt = rng.choice([0.0, 0.04, 0.1, 0.5, 2.0])
            t += dt

            for lid in ("NE1", "NE2", "NE3"):
                if rng.random() < 0.15:
                    vcls = rng.choice(["car", "truck", "bus", "motorcycle", "three_wheeler", 0, 1, 2, 3, 4])
                    est.record_gate_crossing("ingress", "N", lid, vcls, count=1, now_mono=t)

            vis = {}
            for lid in ("N1", "N2", "N3"):
                if rng.random() < 0.2 and len(active_c1[lid]) < 12:
                    tid = next_tid
                    next_tid += 1
                    active_c1[lid][tid] = rng.choice(["car", "truck", "bus"])
                if active_c1[lid] and rng.random() < 0.2:
                    gone = rng.choice(list(active_c1[lid].keys()))
                    del active_c1[lid][gone]
                    est.record_gate_crossing("stopline", "N", lid, "car", count=1, now_mono=t)

                n_cars = len(active_c1[lid])
                q_cars = rng.randint(0, n_cars) if n_cars else 0
                vis[lid] = {
                    "role": "queue",
                    "cars_queued": q_cars,
                    "cars_total": n_cars,
                    "motorcycles": rng.randint(0, 5),
                    "has_stopped": q_cars > 0,
                    "avg_speed_kmh": 0.0 if q_cars > 0 else rng.uniform(10.0, 50.0),
                    "car_tracks": dict(active_c1[lid]),
                }
            for ulid in ("NE1", "NE2", "NE3"):
                ucars = rng.randint(0, 6)
                uq = rng.randint(0, ucars)
                vis[ulid] = {
                    "role": "upstream",
                    "cars_queued": uq,
                    "cars_total": ucars,
                    "has_stopped": uq > 0,
                    "avg_speed_kmh": 0.0 if uq > 0 else 30.0,
                }
            vis["OUT_N"] = {
                "role": "outgoing",
                "direction": "N",
                "cars_total": rng.randint(0, 16),
                "cars_queued": rng.randint(0, 4),
                "has_stopped": rng.random() < 0.3,
            }

            est.update_visual_occupancy(vis, now_mono=t, dt=dt)

            for lid, lst in est.lanes.items():
                self.assertGreaterEqual(lst.demand_count, lst.vis_cam1)
                self.assertLessEqual(lst.demand_count, est.max_lane_capacity)
                self.assertGreaterEqual(len(lst.seen_cam1_tracks), 0)

            if step % 25 == 0:
                tel = est.get_telemetry(interval_sec=1.0)
                self.assertIn("by_lane", tel)
                est.reset_interval()

    def test_fuzz_byetrack_and_motion_5k_frames(self):
        """
        Stress tests ByteTrack + MotionStateClassifier over 5,000 frames with rapid births,
        deaths, NaN-free outputs, bounded memory, and zero crashes on degenerate boxes.
        """
        rng = random.Random(123)
        tracker = ByteTrack(track_thresh=0.4, low_thresh=0.1, match_thresh=0.7, max_age=30, min_hits=2)
        motion = MotionStateClassifier(QueueSettings())

        t = 0.0
        boxes = []
        for _frame in range(5000):
            t += 0.04
            if len(boxes) < 30 and rng.random() < 0.3:
                x = rng.uniform(50, 1500)
                y = rng.uniform(50, 900)
                boxes.append([x, y, x + rng.uniform(20, 120), y + rng.uniform(20, 120), rng.uniform(0.15, 0.95), rng.randint(0, 4)])
            kept = []
            for b in boxes:
                if rng.random() > 0.05:
                    dx = rng.uniform(-5, 5)
                    dy = rng.uniform(-8, 8)
                    b[0] += dx
                    b[2] += dx
                    b[1] += dy
                    b[3] += dy
                    b[4] = rng.uniform(0.11, 0.98)
                    kept.append(b)
            boxes = kept
            dets = np.array(boxes, dtype=float) if boxes else np.empty((0, 6))
            out = tracker.update(dets)
            for row in out:
                tid = int(row[4])
                cx = float((row[0] + row[2]) * 0.5)
                cy = float(row[3])
                bh = float(max(1.0, row[3] - row[1]))
                st = motion.update(tid, (cx, cy), bh, t)
                self.assertIn(st, (QUEUED, MOVING, UNKNOWN))
            motion.prune(t)
            self.assertLessEqual(len(motion), 200)

    def test_six_camera_real_configs_startup_and_step(self):
        """
        Stress tests loading ALL 6 real bundled configs (5 standard + opt-in north_upstream)
        into BatchedCameraPipeline, stepping 20 frames, publishing in controller mode.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        names = ["north", "south", "east", "west", "northeast", "north_upstream"]
        configs = [os.path.join(repo_root, "config", f"config_{n}.json") for n in names]
        ctrl_cfg = os.path.join(repo_root, "config", "controller_main.json")

        sources = [_DummyReplaySource(n, width=1920, height=1080) for n in names]
        from run_multi_camera import BatchedCameraPipeline

        det_box = np.array([[900.0, 500.0, 960.0, 560.0, 0.9, 0]], dtype=float)
        pipeline = BatchedCameraPipeline(
            camera_names=names,
            config_paths=configs,
            video_sources=names,
            model_path="unused.pt",
            device="cpu",
            output_mode="controller",
            controller_config=ctrl_cfg,
            allow_replay_controller=True,
            model=FakeModel(lambda idx, call: det_box),
            publisher=FakePublisher(),
            sources=sources,
            pub_interval=0.5,
            skip_frames=0,
        )
        pipeline.running = True
        pipeline.batch_wait_s = 0.0
        for c in pipeline.cams:
            c.tracker.min_hits = 1

        for step_i in range(20):
            for s in sources:
                s.push(obs_t=step_i * 0.05)
            pipeline.step()

        payload = pipeline.publish_once()
        self.assertIsNotNone(payload)
        self.assertEqual(len(pipeline.publisher.payloads), 1)
        wire = pipeline.publisher.payloads[-1]
        self.assertEqual(len(wire["lanes"]), 9)


class _DummyReplaySource:
    def __init__(self, name: str, width: int = 1920, height: int = 1080):
        self.name = name
        self.width = width
        self.height = height
        self.q = deque()
        self.epoch = 1
        self.seq = 0

    def start(self):
        pass

    def stop(self):
        pass

    def status(self):
        return {"state": "playing", "framesDropped": 0, "reconnects": 0}

    def push(self, obs_t: float):
        self.seq += 1
        frame = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
        frame[::4, ::4] = 200
        self.q.append(FramePacket(
            frame=frame,
            seq=self.seq,
            captured_wall=time.time(),
            captured_mono=time.monotonic(),
            obs_t=obs_t,
            epoch=self.epoch,
            scale=1.0,
        ))

    def poll(self):
        return self.q.popleft() if self.q else None


if __name__ == "__main__":
    unittest.main()
