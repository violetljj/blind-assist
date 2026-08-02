#!/usr/bin/env python3
"""Tests for D45 four-distance metric-depth aggregation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from aggregate_stage_c_d45_phone_metric_depth_canary import (
    ControlPlaneError,
    EXPECTED_BASELINE_APP_SHA256,
    MAX_RECEIPT_BYTES,
    _atomic_nonoverwriting_write,
    evaluate_payloads,
    load_receipt,
)


def receipt(
    distance: int,
    *,
    depth_offset: float = 0.1,
    latency_ms: float = 40.0,
    accepted: int = 20,
    exact_person: int = 25,
    terminal: str = "DISTANCE_MEASUREMENT_OBSERVED",
) -> dict:
    depths = [distance + depth_offset] * accepted
    errors = [abs(depth - distance) for depth in depths]
    error = errors[0] if errors else None
    relative = error / distance if error is not None else None
    return {
        "schema": "blindassist_hftf_d45_person_measurement_canary_v1",
        "run_id": f"manual-{distance}m-test",
        "terminal": terminal,
        "reference_distance": {
            "meters": distance,
            "definition": "PERSON_TORSO_PLANE_TO_CAMERA_OPTICAL_CENTER",
            "source": "OPERATOR_DECLARED_INSTRUMENTATION_ARGUMENT",
        },
        "device": {
            "manufacturer": "Samsung",
            "brand": "samsung",
            "model": "SM-S9280",
            "device": "e3q",
            "build_fingerprint": "test/fingerprint",
            "android_sdk_int": 36,
            "android_release": "16",
        },
        "build": {
            "target_apk_bytes": 100,
            "target_apk_sha256": "a" * 64,
            "instrumentation_apk_bytes": 50,
            "instrumentation_apk_sha256": "b" * 64,
        },
        "source": {
            "arcore_sdk_dependency_version": "1.33.0",
            "camera_id": "0",
            "detector_rotation_degrees": 90,
        },
        "detector": {
            "model_asset": "yolo11n_fp16_320.tflite",
            "model_sha256": (
                "00edb41a528b0a7e709c4af8ce3e6854"
                "91492c4539274804e5cfc17a1a867cd2"
            ),
            "backend": "cpu_xnnpack",
            "selection_contract": "EXACTLY_ONE_PERSON_DETECTION",
        },
        "coverage": {
            "accepted_measurement_count": accepted,
            "exact_single_person_frame_count": exact_person,
            "accepted_person_coverage": accepted / exact_person,
        },
        "metric_error": {
            "accepted_observation_count": accepted,
            "absolute_error_median_m": error,
            "absolute_error_p90_m": error,
            "relative_error_median": relative,
        },
        "bounded_measurement_values": {
            "optical_axis_depth_m": depths,
            "source_to_measurement_latency_ms": [latency_ms] * accepted,
            "maximum_value_count": 1_800,
        },
        "history": {
            "eligible_window_count": 10,
            "available_forecast_count": 6,
            "availability": 0.6,
        },
        "evidence_boundary": {
            "benchmark_only": True,
            "app_runtime_involved": False,
            "risk_feedback_invocation_count": 0,
            "raster_pixels_persisted": False,
            "camera_images_persisted": False,
            "person_boxes_persisted": False,
            "event_outcome_evaluated": False,
            "navigation_output_issued": False,
            "production_authorized": False,
        },
    }


class D45MetricDepthAggregateTests(unittest.TestCase):
    def test_supports_exact_four_distance_contract(self) -> None:
        report = evaluate_payloads(
            [receipt(distance) for distance in (1, 2, 3, 5)],
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(
            report["scientific_terminal"],
            "D45_PHONE_METRIC_DEPTH_SOURCE_SUPPORTED_DEVELOPMENT_ONLY",
        )
        self.assertTrue(report["frozen_gate"]["supported"])
        self.assertAlmostEqual(report["metrics"]["accepted_person_coverage"], 0.8)

    def test_missing_distance_is_not_a_scientific_terminal(self) -> None:
        report = evaluate_payloads(
            [receipt(distance) for distance in (1, 2, 3)],
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(report["evaluation_status"], "INCOMPLETE_DISTANCE_SET")
        self.assertIsNone(report["scientific_terminal"])
        self.assertEqual(report["missing_distances_m"], [5.0])

    def test_not_evaluable_source_is_not_algorithm_failure(self) -> None:
        payloads = [receipt(distance) for distance in (1, 2, 3, 5)]
        payloads[2]["terminal"] = "NOT_EVALUABLE_NO_RAW_DEPTH_OBSERVATIONS"
        payloads[2]["source"]["camera_id"] = None
        payloads[2]["source"]["detector_rotation_degrees"] = None
        report = evaluate_payloads(
            payloads,
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(
            report["scientific_terminal"],
            "D45_PHONE_METRIC_DEPTH_SOURCE_NOT_EVALUABLE",
        )

    def test_frozen_measurement_gate_failure_is_not_supported(self) -> None:
        report = evaluate_payloads(
            [
                receipt(distance, depth_offset=1.5, latency_ms=175.0)
                for distance in (1, 2, 3, 5)
            ],
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(
            report["scientific_terminal"],
            "D45_PHONE_METRIC_DEPTH_SOURCE_NOT_SUPPORTED",
        )
        self.assertFalse(
            report["frozen_gate"]["checks"]["p90_absolute_error_lte_1_00_m"]
        )
        self.assertFalse(report["frozen_gate"]["checks"]["latency_p95_lte_150_ms"])

    def test_cross_build_receipts_are_control_plane_rejected(self) -> None:
        payloads = [receipt(distance) for distance in (1, 2, 3, 5)]
        payloads[3]["build"]["target_apk_sha256"] = "c" * 64
        report = evaluate_payloads(
            payloads,
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(report["evaluation_status"], "CONTROL_PLANE_INPUT_REJECTED")
        self.assertIsNone(report["scientific_terminal"])

    def test_recomputed_metric_mismatch_is_control_plane_rejected(self) -> None:
        payloads = [receipt(distance) for distance in (1, 2, 3, 5)]
        corrupted = copy.deepcopy(payloads[0])
        corrupted["metric_error"]["absolute_error_median_m"] = 9.0
        payloads[0] = corrupted
        report = evaluate_payloads(
            payloads,
            baseline_app_sha256=EXPECTED_BASELINE_APP_SHA256,
        )
        self.assertEqual(report["evaluation_status"], "CONTROL_PLANE_INPUT_REJECTED")
        self.assertIsNone(report["scientific_terminal"])

    def test_baseline_mismatch_does_not_emit_measurement_terminal(self) -> None:
        report = evaluate_payloads(
            [receipt(distance) for distance in (1, 2, 3, 5)],
            baseline_app_sha256="f" * 64,
        )
        self.assertEqual(
            report["evaluation_status"],
            "CONTROL_PLANE_BASELINE_MISMATCH",
        )
        self.assertIsNone(report["scientific_terminal"])

    def test_duplicate_json_key_is_rejected_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "duplicate.json")
            path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
            with self.assertRaisesRegex(ControlPlaneError, "duplicate JSON key"):
                load_receipt(path)

    def test_oversized_receipt_is_rejected_before_json_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "oversized.json")
            path.write_bytes(b" " * (MAX_RECEIPT_BYTES + 1))
            with self.assertRaisesRegex(ControlPlaneError, "receipt size"):
                load_receipt(path)

    def test_final_report_write_is_nonoverwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "report.json")
            _atomic_nonoverwriting_write(path, "first\n")
            with self.assertRaisesRegex(FileExistsError, "non-overwriting"):
                _atomic_nonoverwriting_write(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")


if __name__ == "__main__":
    unittest.main()
