import unittest

from validate_f1b0_timing_receipt import TimingReceiptError, validate_receipt


def receipt() -> dict:
    semantic = {
        field: index for index, field in enumerate(
            (
                "capturedAt",
                "receivedAt",
                "queuedAt",
                "startedAt",
                "completedAt",
                "publishedAt",
                "availableAt",
                "consumedAt",
            ),
            start=1,
        )
    }
    semantic.update(
        {
            "clockDomain": "ANDROID_ELAPSED_REALTIME_NANOS",
            "dropReason": "NONE",
            "detectorBackend": "qualcomm_qnn_htp",
            "backendRouteReason": "supported",
        }
    )
    geometry = {
        field: index for index, field in enumerate(
            (
                "previousObservationAt",
                "currentObservationAt",
                "geometryQueuedAt",
                "geometryStartedAt",
                "geometryCompletedAt",
                "geometryPublishedAt",
                "geometryAvailableAt",
                "geometryConsumedAt",
            ),
            start=1,
        )
    }
    geometry.update(
        {
            "clockDomain": "ANDROID_ELAPSED_REALTIME_NANOS",
            "dropReason": "NONE",
            "abstainReason": "NONE",
        }
    )
    return {
        "schema": "blindassist_dual_loop_f1b0_timing_baseline_v1",
        "effect_outputs_accessed": False,
        "alerts_invoked": False,
        "semantic_results": [dict(semantic) for _ in range(20)],
        "geometry_results": [dict(geometry) for _ in range(20)],
    }


class ValidateF1b0TimingReceiptTest(unittest.TestCase):
    def test_accepts_causal_complete_traces(self) -> None:
        self.assertEqual("READY", validate_receipt(receipt())["terminal"])

    def test_rejects_future_semantic_availability(self) -> None:
        value = receipt()
        value["semantic_results"][0]["availableAt"] = 2
        with self.assertRaises(TimingReceiptError):
            validate_receipt(value)


if __name__ == "__main__":
    unittest.main()
