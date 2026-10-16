"""Exercise real RTSP and MQTT recovery on loopback only; never uses field URLs.

Requires ffmpeg, an explicit MediaMTX binary and the optional amqtt test package.
RTSP uses a recording read through the runner's decoder (--decoder, as in main.py run);
MQTT uses the real-model main fixture. The camera is reached through a loopback relay
that can stop forwarding with the connection left open, like a hung camera. No signal
receiver or database is started. Results do not certify camera/NVR or deployment latency.
"""

import argparse
import asyncio
import copy
from functools import partial
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
import psutil

from trt_pipeline.nvdec import NvdecCapture, ffmpeg_exe, make_capture
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


def decoder_processes():
    """Running children of this process that are the GPU decoder's ffmpeg (not the feeder's)."""
    exe = os.path.normcase(os.path.abspath(ffmpeg_exe() or ""))
    found = []
    for child in psutil.Process().children(recursive=True):
        try:
            if os.path.normcase(child.exe()) == exe and child.is_running():
                found.append(child.pid)
        except psutil.Error:
            pass
    return found


class StallRelay:
    """Loopback TCP relay to the RTSP server. ``stall()`` stops forwarding but keeps every
    connection open, so only a read timeout can notice it (a hung camera, not a closed one)."""

    def __init__(self, upstream_port):
        self.upstream_port = upstream_port
        self.flowing = asyncio.Event()
        self.flowing.set()
        self.writers = set()
        self.server = None
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    def stall(self):
        self.flowing.clear()

    def resume(self):
        self.flowing.set()

    async def _accept(self, reader, writer):
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", self.upstream_port)
        except OSError:
            writer.close()
            return
        await asyncio.gather(self._forward(reader, upstream_writer), self._forward(upstream_reader, writer))

    async def _forward(self, reader, writer):
        self.writers.add(writer)
        try:
            while data := await reader.read(65536):
                await self.flowing.wait()
                writer.write(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()

    async def close(self):
        self.flowing.set()
        self.server.close()
        for writer in list(self.writers):
            writer.close()


async def exercise(args, report):
    rtsp_port, mqtt_port = free_port(), free_port()
    artifacts = ROOT / ".venv/network-smoke" / ("run-" + uuid.uuid4().hex[:8])
    artifacts.mkdir(parents=True)
    config = artifacts / "mediamtx.yml"
    config.write_text(f"logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp_port}\nrtspTransports: [tcp]\n"
                      "rtmp: false\nhls: false\nwebrtc: false\nsrt: false\nmoq: false\n"
                      "api: false\nmetrics: false\npprof: false\nplayback: false\npaths:\n  all_others: {}\n", encoding="utf-8")
    publish_url = f"rtsp://127.0.0.1:{rtsp_port}/camera"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    server_log = (artifacts / "server.log").open("w", encoding="utf-8")
    feeder_log = (artifacts / "feeder.log").open("w", encoding="utf-8")
    server = feeder = worker = publisher = subscriber = broker = relay = None
    messages = queue.Queue()
    ready = False
    broker_config = {"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{mqtt_port}"}},
                     "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}}

    def feed():
        scale = ["-vf", f"scale={args.feed_width}:-2"] if args.feed_width else []
        return subprocess.Popen([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                                 "-i", str(Path(args.video).resolve()), "-an", *scale,
                                 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                                 "-g", "25", "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", publish_url],
                                stdout=feeder_log, stderr=feeder_log, creationflags=flags)

    async def recovered_after(epoch, label, timeout=20):
        return await wait_until(lambda: (p if (p := worker.poll()) and p.epoch > epoch and p.age_s() < .75 else None),
                                label, timeout=timeout)

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
        relay = StallRelay(rtsp_port)
        await relay.start()
        url = f"rtsp://127.0.0.1:{relay.port}/camera"
        # Built as the runner builds a live camera (run_multi_camera.py), with its default timeouts and
        # failure count; only the reconnect backoff is shortened to keep the smoke quick.
        worker = StreamBufferWorker("loopback", url, capture_factory=partial(make_capture, url, args.decoder, args.imgsz),
                                    reconnect_initial_s=.25, reconnect_max_s=1)
        worker.start()
        first = await wait_until(worker.poll, "first decoded RTSP frame", timeout=30)
        decoder_used = type(worker.cap).__name__
        expected_children = 1 if isinstance(worker.cap, NvdecCapture) else 0
        children = {"connected": len(decoder_processes())}
        await asyncio.sleep(1.2)
        newest = await wait_until(lambda: (p if (p := worker.poll()) and p.age_s() < .75 else None), "latest decoded frame")
        assert newest.seq > first.seq + 5 and newest.age_s() < .75, f"RTSP latest frame: seq {first.seq}->{newest.seq}, age {newest.age_s()}"
        assert worker.frames_dropped > 0, "RTSP queue did not discard old frames"

        # Camera goes away (connection closed by the server).
        stop_child(feeder)
        lost_at = time.monotonic()
        await wait_until(lambda: worker.state in ("offline", "reconnecting"), "RTSP loss detected")
        loss_detect_s = time.monotonic() - lost_at
        await wait_until(lambda: worker.status()["lastFrameAgeSec"] > 1.5, "unusable RTSP observation age")
        old_epoch = worker.epoch
        feeder = feed()
        recovered = await recovered_after(old_epoch, "RTSP reconnect epoch")
        await asyncio.sleep(1)
        children["afterReconnect"] = len(decoder_processes())

        # Camera hangs: the connection stays open but no data arrives.
        stalled_cap = worker.cap
        relay.stall()
        stalled_at = time.monotonic()
        await wait_until(lambda: worker.state in ("offline", "reconnecting"), "RTSP stall detected", timeout=30)
        stall_detect_s = time.monotonic() - stalled_at
        stall_error = getattr(stalled_cap, "last_error", None)
        await asyncio.sleep(2)
        stalled_epoch = worker.epoch
        relay.resume()
        resumed = await recovered_after(stalled_epoch, "RTSP reconnect after stall", timeout=30)
        resumed_age_s = resumed.age_s()
        await asyncio.sleep(1)
        children["afterStall"] = len(decoder_processes())

        # A stall must end through the transport timeout, well before three read backstops pass.
        assert stall_detect_s < worker.read_timeout_s + 5, f"RTSP stall took {stall_detect_s:.1f}s to detect"
        assert all(n == expected_children for n in children.values()), f"decoder processes {children}"
        report["rtsp"] = {"decoder": args.decoder, "decoderUsed": decoder_used,
                          "framesIngested": worker.frames_ingested, "framesDropped": worker.frames_dropped,
                          "reconnects": worker.reconnects, "firstEpoch": first.epoch, "recoveredEpoch": recovered.epoch,
                          "stallRecoveredEpoch": resumed.epoch,
                          "lossDetectSec": round(loss_detect_s, 2), "stallDetectSec": round(stall_detect_s, 2),
                          "stallDecoderError": stall_error,
                          "latestAgeMs": round(resumed_age_s * 1000, 1), "decodedResolution": list(resumed.frame.shape[:2]),
                          "frameScale": resumed.scale, "decoderProcesses": children,
                          "latestOnlyPassed": True, "outageAndRecoveryPassed": True, "stallAndRecoveryPassed": True}

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
            report["decoderProcessesAfterStop"] = len(decoder_processes())
        if relay:
            await relay.close()
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
        if report.get("decoderProcessesAfterStop"):
            raise RuntimeError(f"{report['decoderProcessesAfterStop']} decoder process(es) left running after stop")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--video", default=str(ROOT / "videos/cam44_north.avi"))
    parser.add_argument("--feed-width", type=int, default=0, help="Restream at this width (default: the recording's)")
    parser.add_argument("--decoder", choices=("auto", "nvdec", "opencv"), default="auto", help="As main.py run --decoder")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector input width the GPU decoder resizes to")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    report = {"scope": "isolated loopback RTSP reader (runner decoder, close and stall faults) and MQTT transport; real-model fixture, no inference",
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
