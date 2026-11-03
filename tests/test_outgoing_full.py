import unittest
from trt_pipeline.corridor import CorridorQueueEstimator


class TestOutgoingFull(unittest.TestCase):
    def test_outgoing_full_lifecycle(self):
        estimator = CorridorQueueEstimator(drift_reset_duration_s=60.0)

        # Register normal stopline lane N1 and outgoing road OUT_N
        estimator.register_lane("N1", direction="N", role="queue")
        estimator.register_lane("OUT_N", direction="N", role="outgoing")

        # Initial state: all outgoing roads are not full
        telemetry = estimator.get_telemetry()
        self.assertEqual(
            telemetry["outgoingfull"],
            {"N": False, "E": False, "S": False, "W": False},
        )

        now = 1000.0

        # Step 1: Normal moving traffic on OUT_N (no stopped cars)
        estimator.update_visual_occupancy(
            {
                "N1": {"cars_total": 2, "cars_queued": 1, "has_stopped": True},
                "OUT_N": {
                    "role": "outgoing",
                    "direction": "N",
                    "cars_total": 2,
                    "cars_queued": 0,
                    "has_stopped": False,
                },
            },
            now_mono=now,
        )
        self.assertFalse(estimator.outgoing_full["N"])

        # Step 2: Traffic stops on OUT_N at t = 1001.0 (e.g. downstream jam)
        now = 1001.0
        estimator.update_visual_occupancy(
            {
                "OUT_N": {
                    "role": "outgoing",
                    "direction": "N",
                    "cars_total": 3,
                    "cars_queued": 2,
                    "has_stopped": True,
                },
            },
            now_mono=now,
        )
        # Not yet 3.0 seconds, so still False
        self.assertFalse(estimator.outgoing_full["N"])

        # Step 3: Still stopped at t = 1003.5 (2.5 seconds elapsed)
        now = 1003.5
        estimator.update_visual_occupancy(
            {
                "OUT_N": {
                    "role": "outgoing",
                    "direction": "N",
                    "cars_total": 3,
                    "cars_queued": 2,
                    "has_stopped": True,
                },
            },
            now_mono=now,
        )
        self.assertFalse(estimator.outgoing_full["N"])

        # Step 4: Still stopped at t = 1004.1 (3.1 seconds elapsed) -> outgoingfull["N"] MUST be True
        now = 1004.1
        estimator.update_visual_occupancy(
            {
                "OUT_N": {
                    "role": "outgoing",
                    "direction": "N",
                    "cars_total": 3,
                    "cars_queued": 2,
                    "has_stopped": True,
                },
            },
            now_mono=now,
        )
        self.assertTrue(estimator.outgoing_full["N"])
        telemetry = estimator.get_telemetry()
        self.assertEqual(
            telemetry["outgoingfull"],
            {"N": True, "E": False, "S": False, "W": False},
        )

        # Step 5: Downstream jam clears, cars moving on OUT_N at t = 1005.0
        now = 1005.0
        estimator.update_visual_occupancy(
            {
                "OUT_N": {
                    "role": "outgoing",
                    "direction": "N",
                    "cars_total": 1,
                    "cars_queued": 0,
                    "has_stopped": False,
                },
            },
            now_mono=now,
        )
        self.assertFalse(estimator.outgoing_full["N"])
        telemetry = estimator.get_telemetry()
        self.assertEqual(
            telemetry["outgoingfull"],
            {"N": False, "E": False, "S": False, "W": False},
        )


if __name__ == "__main__":
    unittest.main()
