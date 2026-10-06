"""Exercise real RTSP and MQTT recovery on loopback only; never uses field URLs.

Requires ffmpeg, an explicit MediaMTX binary and the optional amqtt test package.
RTSP uses a recording; MQTT uses the real-model main fixture. No signal receiver
or database is started. Results do not certify camera/NVR or deployment latency.
"""

import argparse
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
import queue
import socket
import subprocess
import time
import uuid

from amqtt.broker import Broker
import paho.mqtt.client as mqtt

from trt_pipeline.payload import iso_utc
from trt_pipeline.publisher import MQTTPublisher
from trt_pipeline.stream import StreamBufferWorker

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_until(check, label, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        await asyncio.sleep(.05)
    raise RuntimeError(f"Timed out: {label}")


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


async def exercise(args, report):
    rtsp_port, mqtt_port = free_port(), free_port()
    artifacts = ROOT / ".venv/network-smoke" / ("run-" + uuid.uuid4().hex[:8])
    artifacts.mkdir(parents=True)
    config = artifacts / "mediamtx.yml"
    config.write_text(f"logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp_port}\nrtspTransports: [tcp]\n"
                      "rtmp: false\nhls: false\nwebrtc: false\nsrt: false\nmoq: false\n"
                      "api: false\nmetrics: false\npprof: false\nplayback: false\npaths:\n  all_others: {}\n", encoding="utf-8")
    url = f"rtsp://127.0.0.1:{rtsp_port}/camera"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server_log = (artifacts / "server.log").open("w", encoding="utf-8")
    feeder_log = (artifacts / "feeder.log").open("w", encoding="utf-8")
    server = feeder = worker = publisher = subscriber = broker = None
    messages = queue.Queue()
    ready = False
    broker_config = {"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{mqtt_port}"}},
                     "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}}

    def feed():
        return subprocess.Popen([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                                 "-i", str(Path(args.video).resolve()), "-an", "-vf", "scale=640:360",
                                 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                                 "-g", "25", "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", url],
                                stdout=feeder_log, stderr=feeder_log, creationflags=flags)

    def connected(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe("traffic/counts", qos=0)

    def subscribed(client, userdata, mid, reason_codes, properties):
        nonlocal ready
        ready = True

    def disconnected(client, userdata, flags, reason_code, properties):
        nonlocal ready
        ready = False

    def received(client, userdata, message):
        messages.put(json.loads(message.payload))

    def drain():
        out = []
        while not messages.empty():
            out.append(messages.get_nowait())
        return out

    try:
        server = subprocess.Popen([str(Path(args.mediamtx).resolve()), str(config.resolve())],
                                  cwd=artifacts, stdout=server_log, stderr=server_log, creationflags=flags)
        await asyncio.sleep(.5)
        if server.poll() is not None:
            raise RuntimeError(f"MediaMTX exited; inspect {artifacts}")
        feeder = feed()
        worker = StreamBufferWorker("loopback", url, open_timeout_s=5, read_timeout_s=5,
                                    max_consecutive_failures=1, reconnect_initial_s=.25, reconnect_max_s=1)
        worker.start()
        first = await wait_until(worker.poll, "first decoded RTSP frame")
        await asyncio.sleep(1.2)
        newest = await wait_until(lambda: (p if (p := worker.poll()) and p.age_s() < .75 else None), "latest decoded frame")
        assert newest.seq > first.seq + 5 and newest.age_s() < .75, f"RTSP latest frame: seq {first.seq}->{newest.seq}, age {newest.age_s()}"
        assert worker.frames_dropped > 0, "RTSP queue did not discard old frames"
        stop_child(feeder)
        await wait_until(lambda: worker.state in ("offline", "reconnecting"), "RTSP loss detected")
        await wait_until(lambda: worker.status()["lastFrameAgeSec"] > 1.5, "unusable RTSP observation age")
        old_epoch = worker.epoch
        feeder = feed()
        recovered = await wait_until(lambda: (p if (p := worker.poll()) and p.epoch > old_epoch else None),
                                     "RTSP reconnect epoch", timeout=20)
        assert recovered.age_s() < .75, f"RTSP recovery age {recovered.age_s()}"
        report["rtsp"] = {"framesIngested": worker.frames_ingested, "framesDropped": worker.frames_dropped,
                          "reconnects": worker.reconnects, "firstEpoch": first.epoch, "recoveredEpoch": recovered.epoch,
                          "latestAgeMs": round(recovered.age_s() * 1000, 1), "decodedResolution": list(recovered.frame.shape[:2]),
                          "latestOnlyPassed": True, "outageAndRecoveryPassed": True}

        broker = Broker(broker_config)
        await broker.start()
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="smoke-subscriber-" + uuid.uuid4().hex[:8])
        subscriber.on_connect, subscriber.on_subscribe = connected, subscribed
        subscriber.on_disconnect, subscriber.on_message = disconnected, received
        subscriber.connect("127.0.0.1", mqtt_port, keepalive=5)
        subscriber.loop_start()
        publisher = MQTTPublisher(broker_url=f"mqtt://127.0.0.1:{mqtt_port}", topic="traffic/counts", qos=0,
                                  client_id="smoke-vision-" + uuid.uuid4().hex[:8], keepalive=5)
        publisher.start()
        await wait_until(lambda: ready and publisher.is_connected, "MQTT clients ready")
        fixture = json.loads((ROOT / "tests/fixtures/vision_payload_main.json").read_text(encoding="utf-8"))
        fixture["timestamp"], fixture["meta"]["visionSequence"] = iso_utc(time.time()), 1
        assert publisher.publish(fixture)
        delivered = await wait_until(lambda: drain(), "first received wire message")
        assert delivered[0] == fixture
        await broker.shutdown()
        broker = None
        await wait_until(lambda: not publisher.is_connected and not ready, "MQTT loss detected")
        obsolete = copy.deepcopy(fixture)
        obsolete["meta"]["visionSequence"] = 999
        assert not publisher.publish(obsolete)
        broker = Broker(broker_config)
        await broker.start()
        await wait_until(lambda: ready and publisher.is_connected, "MQTT reconnect", timeout=20)
        await asyncio.sleep(.5)
        assert not drain(), "Disconnected snapshot was replayed"
        fixture["timestamp"], fixture["meta"]["visionSequence"] = iso_utc(time.time()), 2
        assert publisher.publish(fixture)
        resumed = await wait_until(lambda: drain(), "post-reconnect wire message")
        assert resumed[0] == fixture
        controller = subprocess.run(["node", str(ROOT / "tests/controller_main_harness.cjs")],
                                    input=json.dumps({"payload": resumed[0]}), text=True, capture_output=True, timeout=15)
        if controller.returncode:
            raise RuntimeError(controller.stderr)
        accepted = json.loads(controller.stdout)
        report["mqtt"] = {"receivedSequences": [delivered[0]["meta"]["visionSequence"], resumed[0]["meta"]["visionSequence"]],
                          "outageAndRecoveryPassed": True, "disconnectedSnapshotDropped": True,
                          "laneCount": accepted["received"]["laneCount"], "totalVehicles": accepted["received"]["totalCars"],
                          "mainRevision": accepted["revision"], "fallbackStrategy": accepted["fallback"]["decision"]["strategy"]}
    finally:
        if worker:
            started = time.monotonic()
            await asyncio.to_thread(worker.stop)
            report["readerShutdownSec"] = round(time.monotonic() - started, 3)
            report["readerStopped"] = not worker.thread.is_alive()
        if publisher:
            await asyncio.to_thread(publisher.stop)
        if subscriber:
            subscriber.disconnect()
            await asyncio.to_thread(subscriber.loop_stop)
        if broker:
            await asyncio.wait_for(broker.shutdown(), timeout=5)
        stop_child(feeder)
        stop_child(server)
        server_log.close()
        feeder_log.close()
        report["artifacts"] = str(artifacts.relative_to(ROOT))
        if worker and worker.thread.is_alive():
            raise RuntimeError("RTSP reader did not stop")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video", default=str(ROOT / "videos/cam44_north.avi"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    report = {"scope": "isolated loopback RTSP reader and MQTT transport; real-model fixture, no inference",
              "startedAt": iso_utc(time.time()), "passed": False}
    try:
        asyncio.run(exercise(args, report))
        report["passed"] = True
    except Exception as exc:  # unslop-ignore: write failure evidence, then exit nonzero
        report["error"] = f"{type(exc).__name__}: {exc}"
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
