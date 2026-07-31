from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .validate_model_selection_closeout import (
    CloseoutValidationError,
    assert_identity_sets_equal,
    assert_mapping,
    component_summary,
    evaluator_mask_ids,
    gate,
    normalize_confusion,
    validate_runtime,
)


class ModelSelectionCloseoutValidationTests(unittest.TestCase):
    def test_confusion_metrics_are_recomputed_from_counts(self) -> None:
        result = normalize_confusion(
            {
                "tp": 6,
                "fp": 2,
                "fn": 4,
                "tn": 8,
                "predicted_pixels": 8,
                "truth_pixels": 10,
            }
        )

        self.assertEqual(result["pixel_count"], 20)
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["recall"], 0.6)
        self.assertAlmostEqual(result["false_positive_area_fraction"], 0.1)

    def test_confusion_rejects_inconsistent_derived_counts(self) -> None:
        with self.assertRaises(CloseoutValidationError):
            normalize_confusion(
                {
                    "tp": 6,
                    "fp": 2,
                    "fn": 4,
                    "tn": 8,
                    "predicted_pixels": 7,
                    "truth_pixels": 10,
                }
            )

    def test_optional_missing_field_does_not_hide_a_mismatch(self) -> None:
        assert_mapping({}, {"pixel_count": 20}, context="frame", optional_missing=frozenset({"pixel_count"}))
        with self.assertRaises(CloseoutValidationError):
            assert_mapping(
                {"pixel_count": 19},
                {"pixel_count": 20},
                context="frame",
                optional_missing=frozenset({"pixel_count"}),
            )

    def test_gate_boundaries_are_inclusive(self) -> None:
        self.assertTrue(gate(0.05, 0.05, ">=")["threshold_satisfied"])
        self.assertTrue(gate(3.0, 3.0, "<=")["threshold_satisfied"])
        self.assertFalse(gate(3.01, 3.0, "<=")["threshold_satisfied"])

    def test_component_summary_rejects_impossible_counts(self) -> None:
        invalid = {
            "predicted_component_count": 1,
            "truth_component_count": 1,
            "hit_predicted_component_count": 2,
            "hit_truth_component_count": 1,
            "false_activation_component_count": -1,
        }
        with self.assertRaises(CloseoutValidationError):
            component_summary([invalid])

    def test_trace_identity_mismatch_fails_closed(self) -> None:
        actual = {("session-a", 1, "image-a")}
        frozen = {("session-a", 1, "image-b")}
        with self.assertRaises(CloseoutValidationError):
            assert_identity_sets_equal(actual, frozen, context="candidate")

    def test_formal_mask_uses_evaluator_grayscale_decode(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.new("RGB", (1, 1), color=(4, 0, 0)).save(path)
            self.assertEqual(evaluator_mask_ids(path), [1])

    def test_runtime_receipt_never_claims_independent_recompute(self) -> None:
        stage = {"count": 200, "mean": 1.0, "p50": 1.0, "p90": 1.0, "p95": 1.0, "min": 1.0, "max": 1.0}
        runtime = {
            "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
            "status": "RUNTIME_BENCHMARK_COMPLETE",
            "model_sha256": "model",
            "runtime_contract": {"threads": 4, "warmup_frames": 20, "measured_frames": 200},
            "corpus": {"truth_pixels_read": False},
            "runtime": {
                name: dict(stage)
                for name in (
                    "preprocess",
                    "tflite_inference",
                    "output_dequantize_argmax",
                    "component_extraction",
                    "fusion_operator",
                    "total_increment",
                )
            },
        }

        result = validate_runtime(runtime, model_sha256="model")

        self.assertEqual(result["integrity_status"], "VALID_AGGREGATE_ONLY")
        self.assertEqual(result["summary_recompute_status"], "NOT_EVALUABLE")


if __name__ == "__main__":
    unittest.main()
