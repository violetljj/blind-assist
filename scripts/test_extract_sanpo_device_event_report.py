from __future__ import annotations

import unittest

import extract_sanpo_device_event_report as converter


class ExtractSanpoDeviceEventReportTest(unittest.TestCase):
    def test_builds_hash_bound_gate_input_from_complete_continuous_benchmark(self) -> None:
        model_sha = "a" * 64
        result = converter.build_report(
            self.benchmark(model_sha),
            model_id="segmentation_candidate",
            model_sha256=model_sha,
            benchmark_sha256="b" * 64,
        )

        self.assertEqual(converter.SCHEMA, result["schema"])
        self.assertEqual(model_sha, result["model_sha256"])
        self.assertEqual(0.9, result["metrics"]["event_recall"])
        self.assertEqual(0.5, result["metrics"]["critical_miss_rate"])
        self.assertEqual(0.4, result["metrics"]["false_alerts_per_minute"])
        self.assertEqual(72.0, result["metrics"]["p95_latency_ms"])

    def test_rejects_missing_duration_and_hash_mismatch(self) -> None:
        model_sha = "a" * 64
        missing_duration = self.benchmark(model_sha)
        del missing_duration["models"][0]["app_detector"]["blindassist_metrics"]["sequenceDurationMs"]
        with self.assertRaisesRegex(ValueError, "sequenceDurationMs"):
            converter.build_report(
                missing_duration,
                model_id="segmentation_candidate",
                model_sha256=model_sha,
                benchmark_sha256="b" * 64,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            converter.build_report(
                self.benchmark("c" * 64),
                model_id="segmentation_candidate",
                model_sha256=model_sha,
                benchmark_sha256="b" * 64,
            )

    @staticmethod
    def benchmark(model_sha: str) -> dict:
        return {
            "device_under_test": "instrumentation-connected-device",
            "models": [
                {
                    "id": "segmentation_candidate",
                    "model_asset_sha256": model_sha,
                    "app_detector": {
                        "total_ms": {"p95": 72.0},
                        "blindassist_metrics": {
                            "eventAlertCount": 10,
                            "criticalEventCount": 2,
                            "criticalEventMissCount": 1,
                            "eventAlertRecall": 0.9,
                            "falseAlertsPerMinute": 0.4,
                            "postEventClearanceRate": 1.0,
                            "deliveredRepeatedAlertRate": 0.0,
                            "sequenceDurationMs": 12_000,
                            "falseAlertCount": 0,
                            "deliveredAlertCount": 9,
                            "deliveredRepeatedAlertCount": 0,
                            "suppressedDuplicateAttemptCount": 3,
                            "eventRegenerationCount": 0,
                        },
                    },
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
