"""
MQTTPublisher — Publishes traffic detection and count payloads to an MQTT broker.
Uses paho-mqtt with background networking loop to prevent blocking video processing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import paho.mqtt.client as mqtt

logger = logging.getLogger("MQTTPublisher")


class MQTTPublisher:
    """
    Thread-safe MQTT publisher for real-time traffic count streaming.
    Supports URI strings ('mqtt://user:pass@host:port') and explicit connection params.
    """

    def __init__(
        self,
        broker_url: str = "mqtt://localhost:1883",
        topic: str = "traffic/counts",
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        qos: int = 1,
        keepalive: int = 60,
    ):
        """
        Initialize the MQTT publisher.

        Args:
            broker_url: MQTT URL (e.g. 'mqtt://localhost:1883' or 'mqtt://user:pass@broker:1883')
            topic: Destination MQTT topic (e.g. 'traffic/counts')
            host: Override broker host
            port: Override broker port
            username: Optional MQTT username
            password: Optional MQTT password
            client_id: Optional MQTT client ID
            qos: MQTT QoS level (0, 1, or 2, default 1)
            keepalive: Keepalive interval in seconds (default 60)
        """
        # Parse broker URL with environment variable fallbacks
        env_url = os.getenv("MQTT_URL")
        target_url = env_url or broker_url

        parsed = urlparse(target_url if "://" in target_url else f"mqtt://{target_url}")

        self.host = host or parsed.hostname or "localhost"
        self.port = port or parsed.port or 1883
        self.topic = os.getenv("TRAFFIC_COUNTS_TOPIC") or os.getenv("MQTT_TOPIC") or topic
        self.username = username or parsed.username or os.getenv("MQTT_USERNAME") or None
        self.password = password or parsed.password or os.getenv("MQTT_PASSWORD") or None
        self.qos = int(qos)
        self.keepalive = int(keepalive)

        # Generate unique client ID if not provided
        self.client_id = client_id or f"smart_traffic_vision_{os.getpid()}_{int(time.time())}"

        self._is_connected = False
        self._lock = threading.Lock()

        # Initialize paho MQTT client compatible with v1 and v2 API versions
        try:
            # paho-mqtt v2 API
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        except AttributeError:
            # paho-mqtt v1 API fallback
            self._client = mqtt.Client(client_id=self.client_id)

        if self.username:
            self._client.username_pw_set(self.username, self.password)

        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        def on_connect(client, userdata, flags, reason_code, properties=None):
            rc = getattr(reason_code, "value", reason_code)
            if rc == 0:
                with self._lock:
                    self._is_connected = True
                logger.info(f"Connected to MQTT broker at {self.host}:{self.port} (topic: {self.topic})")
            else:
                logger.error(f"MQTT connection failed with code: {reason_code}")

        def on_disconnect(client, userdata, flags, reason_code=None, properties=None):
            with self._lock:
                self._is_connected = False
            logger.warning(f"Disconnected from MQTT broker: {reason_code}")

        def on_publish(client, userdata, mid, reason_code=None, properties=None):
            logger.debug(f"Message {mid} published to {self.topic}")

        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect
        self._client.on_publish = on_publish

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    def start(self) -> None:
        """Connect to broker and start background network processing loop."""
        try:
            logger.info(f"Connecting to MQTT broker at {self.host}:{self.port}...")
            self._client.connect_async(self.host, self.port, keepalive=self.keepalive)
            self._client.loop_start()
        except Exception as e:
            logger.error(f"Failed to initialize MQTT connection: {e}")

    def stop(self) -> None:
        """Stop background network loop and disconnect cleanly."""
        try:
            self._client.loop_stop()
            self._client.disconnect()
            with self._lock:
                self._is_connected = False
            logger.info("MQTT publisher stopped cleanly.")
        except Exception as e:
            logger.warning(f"Error while stopping MQTT client: {e}")

    def publish(self, payload: dict[str, Any] | str, topic: str | None = None) -> bool:
        """
        Publish payload to target topic.

        Args:
            payload: Dictionary (will be serialized to JSON) or pre-formatted JSON string
            topic: Target topic override (defaults to self.topic)

        Returns:
            bool: True if message was queued for delivery, False otherwise
        """
        target_topic = topic or self.topic
        message = json.dumps(payload) if isinstance(payload, dict) else str(payload)

        try:
            info = self._client.publish(target_topic, message, qos=self.qos)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"MQTT publish returned status code: {info.rc}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to publish MQTT message: {e}")
            return False

    def __enter__(self) -> MQTTPublisher:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# Backward compatibility alias
WSPublisher = MQTTPublisher