"""
WSPublisher — publishes TrafficTracker payloads to Redis.
Runs in a background thread; never blocks the main tracking loop.
"""
import json
import logging
import threading
import time
import redis

logger = logging.getLogger("WSPublisher")

CHANNEL_PREFIX = "traffic:"


class WSPublisher:
    def __init__(self, uri: str, topic: str):
        """
        Args:
            uri:   Redis URL, e.g. "redis://localhost:6379"
            topic: Channel suffix, e.g. "south_1" → publishes to "traffic:south_1"
        """
        self.redis_url = uri
        self.topic     = topic
        self.channel   = f"{CHANNEL_PREFIX}{topic}"
        self._client: redis.Redis | None = None
        self._lock     = threading.Lock()

    def start(self):
        self._client = redis.from_url(self.redis_url, decode_responses=True)
        try:
            self._client.ping()
            logger.info(f"WSPublisher connected → channel: {self.channel}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"WSPublisher failed to connect to Redis: {e}")
            self._client = None

    def stop(self):
        if self._client:
            self._client.close()
            self._client = None

    def publish(self, payload: dict):
        if not self._client:
            return
        envelope = json.dumps({
            "type":      "UPDATE",
            "topic":     self.topic,
            "payload":   payload,
            "timestamp": time.time(),
        })
        try:
            with self._lock:
                self._client.publish(self.channel, envelope)
        except redis.exceptions.RedisError as e:
            logger.warning(f"Publish failed: {e}")