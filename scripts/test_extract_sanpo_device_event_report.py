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
        self.assertEqual(converter.DECISION_KERNEL_CONTRACT, result["provenance"]["decision_kernel_contract_id"])
        self.assertEqual(converter.FEEDBACK_ADAPTER, result["provenance"]["feedback_adapter"])

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

    def test_rejects_legacy_or_non_shared_decision_benchmark(self) -> None:
        model_sha = "a" * 64
        legacy = self.benchmark(model_sha)
        del legacy["schema"]
        with self.assertRaisesRegex(ValueError, "benchmark schema"):
            converter.build_report(
                legacy,
                model_id="segmentation_candidate",
                model_sha256=model_sha,
                benchmark_sha256="b" * 64,
            )

        wrong_kernel = self.benchmark(model_sha)
        wrong_kernel["decision_kernel_contract_id"] = "legacy_manual_feedback"
        with self.assertRaisesRegex(ValueError, "decision kernel"):
            converter.build_report(
                wrong_kernel,
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
            "schema": converter.BENCHMARK_SCHEMA,
            "decision_kernel_contract_id": converter.DECISION_KERNEL_CONTRACT,
            "risk_metric_semantics": "shared_production_stable_risk_v1",
            "feedback_adapter": converter.FEEDBACK_ADAPTER,
            "alert_profile": converter.ALERT_PROFILE,
            "synthetic_clock_frame_step_ms": converter.SYNTHETIC_CLOCK_FRAME_STEP_MS,
            "device_under_test": "instrumentation-connected-device",
            "models": [
                {
                    "id": "segmentation_candidate",
                    "model_asset_sha256": model_sha,
                    "app_detector": {
                        "decision_kernel_contract_id": converter.DECISION_KERNEL_CONTRACT,
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
