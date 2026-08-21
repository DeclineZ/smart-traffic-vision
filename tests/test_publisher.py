"""
Unit tests for trt_pipeline.publisher:
- MQTTPublisher initialization and URL parsing
- Safe publishing & payload serialization
- Context manager & lifecycle handling
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trt_pipeline.publisher import MQTTPublisher


class TestMQTTPublisher(unittest.TestCase):
    def test_default_initialization(self):
        pub = MQTTPublisher()
        self.assertEqual(pub.host, "localhost")
        self.assertEqual(pub.port, 1883)
        self.assertEqual(pub.topic, "traffic/counts")
        self.assertEqual(pub.qos, 1)

    def test_url_parsing(self):
        pub = MQTTPublisher(
            broker_url="mqtt://admin:secret123@192.168.1.100:8883",
            topic="test/topic",
            qos=2,
        )
        self.assertEqual(pub.host, "192.168.1.100")
        self.assertEqual(pub.port, 8883)
        self.assertEqual(pub.username, "admin")
        self.assertEqual(pub.password, "secret123")
        self.assertEqual(pub.topic, "test/topic")
        self.assertEqual(pub.qos, 2)

    def test_env_var_override(self):
        with patch.dict(os.environ, {
            "MQTT_URL": "mqtt://envuser:envpass@mqtt.server.internal:1884",
            "TRAFFIC_COUNTS_TOPIC": "traffic/counts_test",
        }):
            pub = MQTTPublisher()
            self.assertEqual(pub.host, "mqtt.server.internal")
            self.assertEqual(pub.port, 1884)
            self.assertEqual(pub.username, "envuser")
            self.assertEqual(pub.password, "envpass")
            self.assertEqual(pub.topic, "traffic/counts_test")

    def test_publish_dict_payload(self):
        pub = MQTTPublisher(topic="traffic/counts")
        mock_publish_info = MagicMock()
        mock_publish_info.rc = 0
        pub._client.publish = MagicMock(return_value=mock_publish_info)

        payload = {
            "intersectionId": "INT-001",
            "cameraId": "CAM-01",
            "lanes": [],
        }

        success = pub.publish(payload)
        self.assertTrue(success)
        pub._client.publish.assert_called_once()

        # Check published string argument
        call_args = pub._client.publish.call_args
        topic_arg = call_args[0][0]
        payload_arg = call_args[0][1]

        self.assertEqual(topic_arg, "traffic/counts")
        deserialized = json.loads(payload_arg)
        self.assertEqual(deserialized["intersectionId"], "INT-001")

    def test_publish_error_handling(self):
        pub = MQTTPublisher()
        pub._client.publish = MagicMock(side_effect=Exception("Network error"))

        # Should handle exception and return False without crashing
        success = pub.publish({"key": "val"})
        self.assertFalse(success)

    def test_context_manager(self):
        with patch.object(MQTTPublisher, "start") as mock_start, \
             patch.object(MQTTPublisher, "stop") as mock_stop:
            with MQTTPublisher() as pub:
                self.assertIsNotNone(pub)
            mock_start.assert_called_once()
            mock_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
