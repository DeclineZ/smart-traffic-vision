"""
WebSocket bridge: subscribes to Redis channels and forwards to browser clients.
Also accepts SUBSCRIBE messages from browsers to register interest in topics.

Redis channel naming: traffic:<topic>
Each message on the channel is a JSON-serialized envelope:
  { "type": "UPDATE"|"HEARTBEAT", "topic": str, "payload": dict, "timestamp": float }
"""
import asyncio
import json
import logging
import os
import time
import redis.asyncio as aioredis
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("WSBridge")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WS_HOST   = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT   = int(os.getenv("WS_PORT", 8765))
CHANNEL_PREFIX = "traffic:"

# topic -> set of WebSocket clients
_subscribers: dict[str, set[WebSocketServerProtocol]] = {}
# topic -> last cached message (for replay on subscribe)
_last_msg: dict[str, str] = {}


def _add_subscriber(topic: str, ws: WebSocketServerProtocol):
    _subscribers.setdefault(topic, set()).add(ws)


def _remove_subscriber(ws: WebSocketServerProtocol):
    for subs in _subscribers.values():
        subs.discard(ws)


async def _broadcast(topic: str, raw: str):
    _last_msg[topic] = raw
    subs = _subscribers.get(topic, set())
    if not subs:
        return
    await asyncio.gather(*[ws.send(raw) for ws in subs], return_exceptions=True)


async def redis_listener():
    """Subscribes to all traffic:* channels on Redis and fans out to WS clients."""
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.psubscribe(f"{CHANNEL_PREFIX}*")
    logger.info(f"Redis listener subscribed to {CHANNEL_PREFIX}*")

    async for message in pubsub.listen():
        if message["type"] != "pmessage":
            continue
        channel: str = message["channel"]          # e.g. "traffic:south_1"
        topic = channel.removeprefix(CHANNEL_PREFIX)
        raw   = message["data"]
        await _broadcast(topic, raw)


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
                # Replay last cached message so client isn't blank
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
    logger.info(f"Starting WS bridge on ws://{WS_HOST}:{WS_PORT}")
    logger.info(f"Redis: {REDIS_URL}")
    await asyncio.gather(
        redis_listener(),
        websockets.serve(ws_handler, WS_HOST, WS_PORT),
    )


if __name__ == "__main__":
    asyncio.run(main())