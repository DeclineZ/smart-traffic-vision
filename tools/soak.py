"""
Soak test: run the real pipeline (model, trackers, analytics, payloads) for a
fixed duration and record resource use and timing.

Measures, sampled every --sample-sec:
  * process RSS and GPU memory
  * per-camera motion-history and gate-history sizes (must stay bounded)
  * step latency p50/p95/p99 and observation age at publish
  * payload validity per camera

Optional fault injection (--outage CAM:START:DURATION, seconds from start)
stops a camera's frames for a while, and checks that its lanes become invalid
(never zero) and that other cameras keep publishing valid data.

    .venv\\Scripts\\python.exe -m tools.soak --minutes 15 --model models/yolo26s_thai_traffic.engine \\
        --outage east:300:60 --out docs/soak/soak-15min.json

Publishing goes to an in-memory sink unless --mqtt-broker is given.
Broker runs use the shadow topic by default. Use --sources-file for field RTSP
cameras. Fault injection withholds decoded frames; it does not test RTSP
transport reconnection. Physically disconnect cameras as a separate check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_multi_camera as rmc  # noqa: E402


class SinkPublisher:
    def __init__(self, inner=None):
        self.count = 0
        self.last = None
        self.inner = inner

    def start(self):
        if self.inner:
            self.inner.start()

    def stop(self, *a):
        if self.inner:
            self.inner.stop(*a)

    def publish(self, payload):
        self.count += 1
        self.last = payload
        return self.inner.publish(payload) if self.inner else True

    def publish_health(self, h):
        return self.inner.publish_health(h) if self.inner else True

    def get_stats(self):
        return self.inner.get_stats() if self.inner else {"sent": self.count}


class LatencySample:
    """Reservoir sample: latency recording cannot grow the process during a long soak."""

    def __init__(self, limit=10000):
        self.limit = limit
        self.values = []
        self.count = 0
        self.rng = np.random.default_rng(0)

    def add(self, value):
        self.count += 1
        if len(self.values) < self.limit:
            self.values.append(value)
        else:
            slot = int(self.rng.integers(self.count))
            if slot < self.limit:
                self.values[slot] = value


def observe_outages(payload, elapsed, outages, grace_s, controller_delivery=None):
    statuses = {c["name"]: c for c in payload["cameras"]}
    unavailable = {o["camera"] for o in outages
                   if o["start"] <= elapsed < o["start"] + o["duration"] + grace_s}
    for outage in outages:
        status = statuses[outage["camera"]]
        lanes = [l for l in payload["lanes"] if l.get("cameraId") == status["cameraId"]]
        if outage["start"] + grace_s <= elapsed < outage["start"] + outage["duration"]:
            unknown = bool(lanes) and all(not l["valid"] and l["count"] is None and l["queuedCount"] is None for l in lanes)
            outage["invalidSeen"] |= unknown
            outage["knownDuringOutage"] |= not unknown
            outage["zeroDuringOutage"] |= any(l["valid"] and l["count"] == 0 for l in lanes)
            if controller_delivery is not None:
                required = any(l.get("role", "queue") == "queue" for l in lanes)
                suppressed = controller_delivery["state"] == "suppressed"
                outage["controllerCheckSeen"] = True
                outage["controllerProjectionFailed"] = outage.get("controllerProjectionFailed", False) or (required != suppressed)
            for name, camera in statuses.items():
                if name not in unavailable and camera["status"] not in rmc.VALID_STATUSES:
                    outage["othersValidDuringOutage"] = False
                    problem = f"{name}:{camera['status']}:{camera.get('reason')}"
                    if problem not in outage["otherCameraProblems"]:
                        outage["otherCameraProblems"].append(problem)
        if elapsed >= outage["start"] + outage["duration"] + grace_s:
            outage["recoveredSeen"] |= (status["status"] in rmc.VALID_STATUSES and bool(lanes)
                                         and all(l["valid"] and l["count"] is not None for l in lanes))


class OutageSource:
    """Wraps a source and withholds its frames during [start, start+duration)."""

    def __init__(self, inner, t0, start, duration):
        self.inner, self.t0, self.start_s, self.dur = inner, t0, start, duration

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def poll(self):
        p = self.inner.poll()
        el = time.monotonic() - self.t0
        return None if self.start_s <= el < self.start_s + self.dur else p


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        rmc.logger.warning("psutil is unavailable; RSS measurements will be missing")
        return None


def gpu_mb():
    try:
        import torch
        return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
    except ImportError:
        rmc.logger.warning("PyTorch is unavailable; GPU memory measurements will be missing")
        return None


def build_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--model", default="models/yolo26s_thai_traffic.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cameras", nargs="+", default=["all"])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--sources-file", default=os.getenv("VISION_SOURCES_FILE"))
    ap.add_argument("--replay-fps", type=float, default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--skip-frames", type=int, default=1)
    ap.add_argument("--pub-interval", type=float, default=1.0)
    ap.add_argument("--sample-sec", type=float, default=30)
    ap.add_argument("--outage", action="append", default=[], help="camera:start_s:duration_s")
    ap.add_argument("--mqtt-broker", default=None)
    ap.add_argument("--mqtt-topic", default=None)
    ap.add_argument("--output-mode", choices=("shadow", "controller"), default="shadow")
    ap.add_argument("--controller-config", default=None)
    ap.add_argument("--allow-replay-controller", action="store_true", help="Use legacy output with recorded footage only in isolated tests")
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    ap = build_args()
    args = ap.parse_args(argv)
    for field in ("minutes", "sample_sec", "fps", "pub_interval"):
        if not np.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            ap.error(f"--{field.replace('_', '-')} must be positive and finite")
    if args.skip_frames < 0:
        ap.error("--skip-frames must be nonnegative")

    names, configs, sources = rmc.resolve_run_plan(args)
    grace_s = max(2.0, 2 * args.pub_interval)
    outages = []
    for spec in args.outage:
        try:
            cam, start, dur = spec.split(":")
            start, dur = float(start), float(dur)
            if cam not in names or not np.isfinite(start + dur) or start < 0 or dur <= grace_s + args.pub_interval:
                raise ValueError()
            if start + dur + grace_s + args.pub_interval >= args.minutes * 60:
                raise ValueError()
        except ValueError:
            ap.error(f"Invalid outage '{spec}': use a selected camera and leave time to observe failure and recovery")
        outages.append({"camera": cam, "start": start, "duration": dur, "invalidSeen": False,
                        "knownDuringOutage": False, "zeroDuringOutage": False, "recoveredSeen": False,
                        "othersValidDuringOutage": True, "otherCameraProblems": []})
    pub = None if args.mqtt_broker else SinkPublisher()
    p = rmc.BatchedCameraPipeline(names, configs, sources, model_path=args.model, device=args.device,
                                  pub_interval=args.pub_interval, publisher=pub, mqtt_broker=args.mqtt_broker,
                                  mqtt_topic=args.mqtt_topic, replay_fps=args.replay_fps,
                                  target_fps=args.fps, skip_frames=args.skip_frames,
                                  output_mode=args.output_mode, controller_config=args.controller_config,
                                  allow_replay_controller=args.allow_replay_controller)
    if pub is None:
        p.publisher = SinkPublisher(p.publisher)
    t0 = time.monotonic()
    for outage in outages:
        idx = names.index(outage["camera"])
        p.cams[idx].source = OutageSource(p.cams[idx].source, t0, outage["start"], outage["duration"])

    step_ms, samples, publishes = LatencySample(), [], 0
    unexpected_problems = set()
    fatal_error = None
    next_sample = 0.0
    last_seq = 0
    end = t0 + args.minutes * 60
    try:
        p.start()
        while time.monotonic() < end:
            s0 = time.perf_counter()
            p.step()
            step_ms.add((time.perf_counter() - s0) * 1000)
            el = time.monotonic() - t0
            last = p.last_payload
            if last and last["sequence"] != last_seq:
                last_seq = last["sequence"]
                publishes += 1
                observe_outages(last, el, outages, grace_s, p.controller_delivery)
                if el >= 10:
                    expected = {o["camera"] for o in outages if o["start"] <= el < o["start"] + o["duration"] + grace_s}
                    for camera in last["cameras"]:
                        if camera["name"] not in expected and camera["status"] not in rmc.VALID_STATUSES:
                            unexpected_problems.add(f"{camera['name']}:{camera['status']}:{camera.get('reason')}")
            if el >= next_sample:
                next_sample += args.sample_sec
                cam_ages = [c["ageMs"] for c in (last or {}).get("cameras", []) if c["ageMs"] is not None]
                samples.append({
                    "t": round(el, 1),
                    "rssMB": rss_mb(),
                    "gpuMB": gpu_mb(),
                    "motionTracks": {c.name: len(c.motion) for c in p.cams},
                    "gatePrevPoints": len(p.gate_manager.track_prev_pts),
                    "crossedIds": sum(len(g.crossed_track_ids) for g in p.gate_manager.gates.values()),
                    "maxObservationAgeMs": max(cam_ages) if cam_ages else None,
                    "inferences": p.total_inferred_batches,
                })
                print(json.dumps(samples[-1]))
            if p.replay_mode:
                rem = 1.0 / p.target_fps - (time.perf_counter() - s0)
                if rem > 0:
                    time.sleep(rem)
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {rmc.redact_source(exc)}"
    finally:
        p.stop()

    rss = [s["rssMB"] for s in samples if s["rssMB"]]
    half = len(rss) // 2
    report = {
        "minutes": args.minutes,
        "model": args.model,
        "publishes": publishes,
        "wirePublishes": p.publisher.count,
        "controllerDelivery": p.controller_delivery,
        "lastControllerPayload": p.publisher.last if p.controller_adapter else None,
        "inferenceBatches": p.total_inferred_batches,
        "errors": p.total_errors + int(fatal_error is not None),
        "fatalError": fatal_error,
        "settings": p.effective_settings(),
        "unexpectedCameraProblems": sorted(unexpected_problems),
        "stepCount": step_ms.count,
        "latencySampleCount": len(step_ms.values),
        "stepMs": {q: round(float(np.percentile(step_ms.values, q)), 2) for q in (50, 95, 99)} if step_ms.values else None,
        "rssMB": {"start": rss[0] if rss else None, "end": rss[-1] if rss else None,
                  "maxFirstHalf": max(rss[:half]) if half else None, "maxSecondHalf": max(rss[half:]) if rss else None},
        "maxMotionTracksPerCamera": max(max(s["motionTracks"].values()) for s in samples) if samples else None,
        "maxGatePrevPoints": max(s["gatePrevPoints"] for s in samples) if samples else None,
        "outages": outages,
        "samples": samples,
    }
    ok = (report["errors"] == 0 and publishes > 0 and not unexpected_problems
          and all(o["invalidSeen"] and not o["knownDuringOutage"] and o["othersValidDuringOutage"]
                  and o["recoveredSeen"] for o in outages)
          and (p.controller_adapter is None or (p.controller_delivery["accepted"] > 0
               and all(o.get("controllerCheckSeen") and not o.get("controllerProjectionFailed") for o in outages))))
    report["passed"] = ok
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
