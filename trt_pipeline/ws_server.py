"""
WebSocket bridge: subscribes to MQTT topics and forwards to browser clients.
Accepts SUBSCRIBE and LIST_TOPICS messages from browser clients.

Default MQTT topic subscription: traffic/#
Relays raw or wrapped MQTT envelopes to WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("WSBridge")

MQTT_URL = os.getenv("MQTT_URL", "mqtt://localhost:1883")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "traffic/#")
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", 8765))

# topic -> set of WebSocket clients
_subscribers: dict[str, set[WebSocketServerProtocol]] = {}
# topic -> last cached message
_last_msg: dict[str, str] = {}
# Thread-safe event loop reference
_loop: asyncio.AbstractEventLoop | None = None


def _add_subscriber(topic: str, ws: WebSocketServerProtocol) -> None:
    _subscribers.setdefault(topic, set()).add(ws)


def _remove_subscriber(ws: WebSocketServerProtocol) -> None:
    for subs in _subscribers.values():
        subs.discard(ws)


async def _broadcast(topic: str, raw: str) -> None:
    _last_msg[topic] = raw
    subs = _subscribers.get(topic, set())
    # Also forward to wildcard subscribers
    wildcard_subs = _subscribers.get("traffic/#", set()) | _subscribers.get("#", set())
    all_subs = subs | wildcard_subs

    if not all_subs:
        return
    await asyncio.gather(*[ws.send(raw) for ws in all_subs], return_exceptions=True)


def on_mqtt_message(client, userdata, msg):
    """Callback from Paho MQTT thread when a message is received."""
    topic = msg.topic
    payload_str = msg.payload.decode("utf-8", errors="ignore")

    envelope = json.dumps({
        "type": "UPDATE",
        "topic": topic,
        "payload": json.loads(payload_str) if payload_str.startswith("{") else payload_str,
        "timestamp": time.time(),
    })

    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(topic, envelope), _loop)


def start_mqtt_listener() -> mqtt.Client:
    """Initialize and start background MQTT client subscription."""
    parsed = urlparse(MQTT_URL if "://" in MQTT_URL else f"mqtt://{MQTT_URL}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 1883
    username = parsed.username or os.getenv("MQTT_USERNAME")
    password = parsed.password or os.getenv("MQTT_PASSWORD")

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"ws_bridge_{os.getpid()}",
        )
    except AttributeError:
        client = mqtt.Client(client_id=f"ws_bridge_{os.getpid()}")

    if username:
        client.username_pw_set(username, password)

    def on_connect(client, userdata, flags, reason_code, properties=None):
        rc = getattr(reason_code, "value", reason_code)
        if rc == 0:
            logger.info(f"MQTT Connected → subscribing to {MQTT_TOPIC}")
            client.subscribe(MQTT_TOPIC)
        else:
            logger.error(f"MQTT connection failed with code: {reason_code}")

    client.on_connect = on_connect
    client.on_message = on_mqtt_message

    client.connect_async(host, port, keepalive=60)
    client.loop_start()
    return client


async def ws_handler(ws: WebSocketServerProtocol):
    remote = ws.remote_address
    logger.info(f"Client connected: {remote}")

    connect_env = json.dumps({
        "type": "CONNECT",
        "topic": "system",
        "payload": {"topics": list(_last_msg.keys())},
        "timestamp": time.time(),
    })
    await ws.send(connect_env)

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "SUBSCRIBE":
                topic = msg.get("topic", "")
                if not topic:
                    continue
                _add_subscriber(topic, ws)
                logger.info(f"{remote} subscribed to '{topic}'")
                if topic in _last_msg:
                    await ws.send(_last_msg[topic])

            elif msg.get("type") == "LIST_TOPICS":
                await ws.send(json.dumps({
                    "type": "TOPICS",
                    "payload": list(_last_msg.keys()),
                    "timestamp": time.time(),
                }))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _remove_subscriber(ws)
        logger.info(f"Client disconnected: {remote}")


async def main():
    global _loop
    _loop = asyncio.get_running_loop()

    logger.info(f"Starting WS bridge on ws://{WS_HOST}:{WS_PORT}")
    logger.info(f"MQTT Broker URL: {MQTT_URL}")

    mqtt_client = start_mqtt_listener()

    try:
        async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
            await asyncio.Future()  # run forever
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())