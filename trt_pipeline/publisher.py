"""
MQTTPublisher -- publishes traffic payloads and source health to an MQTT broker.

* Explicit constructor arguments take precedence over environment variables,
  which take precedence over built-in defaults.
* ``mqtts://`` or any TLS file option enables TLS with certificate verification.
* A retained last-will marks the source offline on the health topic if the
  process or network dies; a retained "online" health message replaces it after
  connecting.
* Occupancy is latest-only: nothing is queued while disconnected, and the paho
  in-flight queue is bounded so a broker outage cannot build a backlog of old
  measurements that would later be replayed as current.
* Sent/acknowledged/failed counters and the last acknowledgement time are kept
  for health reporting. A broker acknowledgement only proves the broker has the
  message, not that the controller validated or used it.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import paho.mqtt.client as mqtt

logger = logging.getLogger("MQTTPublisher")

DEFAULT_BROKER = "mqtt://localhost:1883"
DEFAULT_TOPIC = "traffic/counts"


class MQTTPublisher:
    """Thread-safe MQTT publisher for real-time traffic count streaming."""

    def __init__(
        self,
        broker_url: str | None = None,
        topic: str | None = None,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        qos: int = 1,
        keepalive: int = 30,
        health_topic: str | None = None,
        tls_ca: str | None = None,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        tls_insecure: bool = False,
        max_inflight: int = 10,
    ):
        target_url = broker_url or os.getenv("MQTT_URL") or DEFAULT_BROKER
        parsed = urlparse(target_url if "://" in target_url else f"mqtt://{target_url}")

        self.scheme = (parsed.scheme or "mqtt").lower()
        self.host = host or parsed.hostname or "localhost"
        self.port = port or parsed.port or (8883 if self.scheme == "mqtts" else 1883)
        self.topic = topic or os.getenv("TRAFFIC_COUNTS_TOPIC") or os.getenv("MQTT_TOPIC") or DEFAULT_TOPIC
        self.username = username or parsed.username or os.getenv("MQTT_USERNAME") or None
        self.password = password or parsed.password or os.getenv("MQTT_PASSWORD") or None
        self.qos = int(qos)
        self.keepalive = int(keepalive)
        self.health_topic = health_topic

        self.tls_ca = tls_ca or os.getenv("MQTT_TLS_CA") or None
        self.tls_cert = tls_cert or os.getenv("MQTT_TLS_CERT") or None
        self.tls_key = tls_key or os.getenv("MQTT_TLS_KEY") or None
        self.tls_insecure = bool(tls_insecure)
        self.use_tls = self.scheme == "mqtts" or bool(self.tls_ca or self.tls_cert)

        self.client_id = client_id or f"smart_traffic_vision_{os.getpid()}_{int(time.time())}"

        self._is_connected = False
        self._lock = threading.Lock()
        self._warned_no_conn = False
        self.stats = {
            "sent": 0,
            "acked": 0,
            "failed": 0,
            "skippedDisconnected": 0,
            "connects": 0,
            "disconnects": 0,
            "lastAckWall": None,
            "lastSentWall": None,
        }

        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        except AttributeError:
            self._client = mqtt.Client(client_id=self.client_id)

        if self.username:
            self._client.username_pw_set(self.username, self.password)
        if self.use_tls:
            self._client.tls_set(
                ca_certs=self.tls_ca,
                certfile=self.tls_cert,
                keyfile=self.tls_key,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            if self.tls_insecure:
                self._client.tls_insecure_set(True)
                logger.warning("MQTT TLS hostname verification disabled (--mqtt-tls-insecure)")
        if self.health_topic:
            self._client.will_set(
                self.health_topic,
                json.dumps({"status": "offline", "reason": "connection_lost"}),
                qos=1,
                retain=True,
            )
        try:
            self._client.max_inflight_messages_set(max_inflight)
            self._client.max_queued_messages_set(max_inflight)
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        except Exception:
            pass

        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        def on_connect(client, userdata, flags, reason_code, properties=None):
            rc = getattr(reason_code, "value", reason_code)
            if rc == 0:
                with self._lock:
                    self._is_connected = True
                    self._warned_no_conn = False
                    self.stats["connects"] += 1
                logger.info(f"Connected to MQTT broker at {self.host}:{self.port} (topic: {self.topic}, tls={self.use_tls})")
            else:
                logger.error(f"MQTT connection refused: {reason_code}")

        def on_disconnect(client, userdata, flags, reason_code=None, properties=None):
            with self._lock:
                self._is_connected = False
                self.stats["disconnects"] += 1
            logger.warning(f"Disconnected from MQTT broker: {reason_code}")

        def on_publish(client, userdata, mid, reason_code=None, properties=None):
            with self._lock:
                self.stats["acked"] += 1
                self.stats["lastAckWall"] = time.time()

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
            logger.info(f"Connecting to MQTT broker at {self.host}:{self.port} (tls={self.use_tls})...")
            self._client.connect_async(self.host, self.port, keepalive=self.keepalive)
            self._client.loop_start()
        except Exception as e:
            logger.error(f"Failed to initialize MQTT connection: {e}")

    def stop(self, final_health: Optional[dict] = None) -> None:
        """Publishes a final retained health message (if configured), then disconnects."""
        try:
            if self.health_topic and self.is_connected:
                info = self._client.publish(
                    self.health_topic,
                    json.dumps(final_health or {"status": "offline", "reason": "stopped"}),
                    qos=1,
                    retain=True,
                )
                try:
                    info.wait_for_publish(timeout=2.0)
                except Exception:
                    pass
            self._client.loop_stop()
            self._client.disconnect()
            with self._lock:
                self._is_connected = False
            logger.info("MQTT publisher stopped cleanly.")
        except Exception as e:
            logger.warning(f"Error while stopping MQTT client: {e}")

    def publish(self, payload: dict[str, Any] | str, topic: str | None = None, retain: bool = False) -> bool:
        """
        Hands a message to the client. Returns False (and queues nothing) while
        disconnected, so a caller can keep interval counters for the next attempt.
        """
        target_topic = topic or self.topic
        message = json.dumps(payload) if isinstance(payload, dict) else str(payload)

        if not self.is_connected:
            with self._lock:
                self.stats["skippedDisconnected"] += 1
                warn = not self._warned_no_conn
                self._warned_no_conn = True
            if warn:
                logger.warning(f"MQTT broker {self.host}:{self.port} not connected; dropping messages until reconnected.")
            return False

        try:
            info = self._client.publish(target_topic, message, qos=self.qos, retain=retain)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                with self._lock:
                    self.stats["failed"] += 1
                logger.warning(f"MQTT publish returned status code: {info.rc} ({mqtt.error_string(info.rc)})")
                return False
            with self._lock:
                self.stats["sent"] += 1
                self.stats["lastSentWall"] = time.time()
            return True
        except Exception as e:
            with self._lock:
                self.stats["failed"] += 1
            logger.error(f"Failed to publish MQTT message: {e}")
            return False

    def publish_health(self, health: dict[str, Any]) -> bool:
        if not self.health_topic:
            return False
        return self.publish(health, topic=self.health_topic, retain=True)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self.stats)
        out["connected"] = self.is_connected
        out["lastAckAgeSec"] = None if out["lastAckWall"] is None else round(time.time() - out["lastAckWall"], 3)
        return out

    def __enter__(self) -> MQTTPublisher:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# Backward compatibility alias
WSPublisher = MQTTPublisher
