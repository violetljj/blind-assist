from __future__ import annotations

import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage_c_f0_1_heldout_result as independent
from evaluate_stage_c_f0_1_heldout import (
    PREDICTION_SCHEMA,
    _effect_decision,
    _metrics,
    _ordered_join_key_sha256,
    _probability_matrix,
    _truth_matrix,
    _validate_exact_sets,
    evaluate,
)
from materialize_stage_c_f0_1_heldout_package import _split_record
from predict_stage_c_f0_1_heldout import _matrix, predict
from validate_stage_c_f0_1_heldout_package import _validate_label


def _label() -> dict:
    known = np.ones((2, 6, 6), dtype=int)
    risk = np.zeros((2, 6, 6), dtype=int)
    risk[0, 0, 0] = 1
    risk[1, 0, 0] = 1
    return {
        "known_target": known.tolist(),
        "risk_target_nullable": risk.tolist(),
    }


def _contract() -> dict:
    sources = ("source-a", "source-b", "source-c")
    checkpoints = [
        {"seed": seed, "arm": arm, "sha256": str(index) * 64}
        for index, (seed, arm) in enumerate(
            (
                (17, "SF_CURRENT"),
                (17, "SF_FUTURE"),
                (17, "HIST_FUTURE"),
                (29, "SF_CURRENT"),
                (29, "SF_FUTURE"),
                (29, "HIST_FUTURE"),
                (43, "SF_CURRENT"),
                (43, "SF_FUTURE"),
                (43, "HIST_FUTURE"),
            ),
            start=1,
        )
    ]
    return {
        "heldout_source_contract": {"source_order": list(sources)},
        "checkpoint_contract": {"checkpoints": checkpoints},
        "metric_contract": {
            "arm_target_mapping": {
                "SF_CURRENT": "current",
                "SF_FUTURE": "future",
                "HIST_FUTURE": "future",
            }
        },
        "student_effect_gates": {
            "primary_median_seed_micro_f1_delta_minimum": 0.03,
            "median_seed_recall_delta_minimum": -0.02,
            "median_seed_false_positive_rate_delta_maximum": 0.02,
            "median_seed_f1_delta_body_minimum": 0.0,
            "median_seed_f1_delta_head_minimum": 0.0,
            "worst_source_median_seed_f1_delta_minimum": -0.02,
            "sf_current_median_seed_micro_f1_minimum": 0.6,
        },
    }


class StageCF01HeldoutExecutionTest(unittest.TestCase):
    def test_materialized_inference_input_has_no_truth(self) -> None:
        record = {
            "sample_id": "sample",
            "session_id": "source",
            "role": "heldout",
            "target_fps": 10.0,
            "anchor_timeline_index": 8,
            "anchor_source_frame_index": 16,
            "history_rgb": [{"relative_time_s": 0.0}],
            "labels": {"current": _label(), "future": _label()},
        }
        inference, truth = _split_record(record)
        self.assertNotIn("labels", inference)
        self.assertNotIn("history_rgb", truth)
        self.assertEqual({"current", "future"}, set(truth["labels"]))

    def test_unknown_null_mask_is_exact(self) -> None:
        label = _label()
        label["known_target"][0][0][0] = 0
        label["risk_target_nullable"][0][0][0] = None
        known, risk = _validate_label(label)
        self.assertFalse(known[0, 0, 0])
        self.assertFalse(risk[0, 0, 0])
        label["risk_target_nullable"][0][0][0] = 0
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            _validate_label(label)

    def test_binary_truth_rejects_json_type_coercion(self) -> None:
        for invalid in (256, 257, "1", True, 1.0):
            with self.subTest(invalid=repr(invalid)):
                label = _label()
                label["known_target"][0][0][0] = invalid
                with self.assertRaisesRegex(ValueError, "exact JSON integers"):
                    _validate_label(label)
                with self.assertRaisesRegex(ValueError, "exact JSON integers"):
                    _truth_matrix(label)
                with self.assertRaisesRegex(ValueError, "exact JSON integers"):
                    independent._decode_truth(label)
        for invalid in (True, 1.0):
            with self.subTest(risk=repr(invalid)):
                label = _label()
                label["risk_target_nullable"][0][0][0] = invalid
                with self.assertRaisesRegex(ValueError, "binary"):
                    _validate_label(label)
                with self.assertRaisesRegex(ValueError, "binary"):
                    _truth_matrix(label)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    independent._decode_truth(label)

    def test_probability_shape_range_and_finite(self) -> None:
        matrix = torch.full((2, 6, 6), 0.5)
        self.assertEqual((2, 6, 6), np.asarray(_matrix(matrix)).shape)
        with self.assertRaisesRegex(ValueError, "finite"):
            _matrix(torch.full((2, 6, 6), math.nan))
        with self.assertRaisesRegex(ValueError, "invalid"):
            _probability_matrix(np.full((2, 6, 6), 1.1))
        for invalid in (True, "0.5"):
            values = np.full((2, 6, 6), 0.5, dtype=object)
            values[0, 0, 0] = invalid
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "JSON numbers"):
                    _probability_matrix(values.tolist())
                with self.assertRaisesRegex(ValueError, "JSON numbers"):
                    independent._decode_probability(values.tolist())

    def test_confusion_metric_zero_denominator_contract(self) -> None:
        metric = _metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        self.assertEqual(0.0, metric["f1"])
        self.assertEqual(0.0, metric["recall"])
        self.assertEqual(0.0, metric["false_positive_rate"])

    def test_exact_cartesian_join_rejects_duplicate(self) -> None:
        contract = _contract()
        truth = []
        for source in contract["heldout_source_contract"]["source_order"]:
            for anchor in range(8, 21):
                sample_id = f"hftf_f0_1_heldout_{source}_{anchor:02d}"
                truth.append(
                    {
                        "schema": "blindassist_hftf_f0_1_heldout_truth",
                        "sample_id": sample_id,
                        "session_id": source,
                        "anchor_timeline_index": anchor,
                        "anchor_source_frame_index": anchor * 2,
                        "labels": {
                            "current": _label(),
                            "future": _label(),
                        },
                    }
                )
        predictions = []
        index = 0
        for checkpoint in contract["checkpoint_contract"]["checkpoints"]:
            for truth_record in truth:
                predictions.append(
                    {
                        "schema": PREDICTION_SCHEMA,
                        "prediction_index": index,
                        "seed": checkpoint["seed"],
                        "arm": checkpoint["arm"],
                        "checkpoint_sha256": checkpoint["sha256"],
                        "sample_id": truth_record["sample_id"],
                        "session_id": truth_record["session_id"],
                        "anchor_timeline_index": truth_record[
                            "anchor_timeline_index"
                        ],
                        "risk_probability": np.full(
                            (2, 6, 6), 0.25
                        ).tolist(),
                        "known_probability": np.full(
                            (2, 6, 6), 0.75
                        ).tolist(),
                    }
                )
                index += 1
        truth_by_sample, prediction_by_key = _validate_exact_sets(
            contract, predictions, truth
        )
        self.assertEqual(39, len(truth_by_sample))
        self.assertEqual(351, len(prediction_by_key))
        self.assertEqual(
            _ordered_join_key_sha256(predictions),
            independent._ordered_join_key_sha256(predictions),
        )
        predictions[-1] = predictions[-2]
        with self.assertRaisesRegex(ValueError, "Cartesian"):
            _validate_exact_sets(contract, predictions, truth)

    def test_worst_source_operator_is_source_then_seed_median_then_min(
        self,
    ) -> None:
        contract = _contract()
        values = {
            17: {"source-a": 0.10, "source-b": -0.01, "source-c": 0.04},
            29: {"source-a": 0.20, "source-b": -0.02, "source-c": 0.05},
            43: {"source-a": -0.50, "source-b": -0.03, "source-c": 0.06},
        }
        runs = []
        for seed in (17, 29, 43):
            for arm in ("SF_CURRENT", "SF_FUTURE", "HIST_FUTURE"):
                base = 0.7 if arm == "SF_CURRENT" else 0.5
                delta = 0.04 if arm == "HIST_FUTURE" else 0.0
                runs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "risk_micro": {
                            "f1": base + delta,
                            "recall": 0.7,
                            "false_positive_rate": 0.1,
                        },
                        "risk_by_height": {
                            height: {"f1": base + delta}
                            for height in ("body", "head")
                        },
                        "risk_by_source": {
                            source: {
                                "f1": (
                                    0.5
                                    + (
                                        values[seed][source]
                                        if arm == "HIST_FUTURE"
                                        else 0.0
                                    )
                                )
                            }
                            for source in (
                                "source-a",
                                "source-b",
                                "source-c",
                            )
                        },
                    }
                )
        aggregate, _ = _effect_decision(contract, runs)
        self.assertAlmostEqual(
            -0.02,
            aggregate["worst_source_median_seed_f1_delta"],
        )

    def test_independent_metric_matches_primary_on_counts(self) -> None:
        counts = {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
        self.assertEqual(_metrics(counts), independent._metric(counts))

    def test_independent_future_denominator_gate_fails_closed(self) -> None:
        runs = [
            {
                "arm": arm,
                "risk_micro": {
                    "positive_truth_count": 1,
                    "negative_truth_count": 1,
                },
                "risk_by_height": {
                    "body": {
                        "positive_truth_count": 1,
                        "negative_truth_count": 1,
                    },
                    "head": {
                        "positive_truth_count": 1,
                        "negative_truth_count": 0,
                    },
                },
                "risk_by_source": {
                    source: {
                        "positive_truth_count": 1,
                        "negative_truth_count": 1,
                    }
                    for source in ("source-a", "source-b", "source-c")
                },
            }
            for arm in ("SF_FUTURE", "HIST_FUTURE")
        ]
        with self.assertRaisesRegex(ValueError, "denominator"):
            independent._validate_future_denominators(runs)

    def test_process_signatures_enforce_truth_output_firewall(self) -> None:
        prediction_parameters = set(inspect.signature(predict).parameters)
        join_parameters = set(inspect.signature(evaluate).parameters)
        self.assertNotIn("truth_path", prediction_parameters)
        self.assertNotIn("teacher_receipts_path", prediction_parameters)
        self.assertNotIn("checkpoints_root", join_parameters)
        self.assertNotIn("inference_inputs_path", join_parameters)
        for filename in (
            "evaluate_stage_c_f0_1_heldout.py",
            "validate_stage_c_f0_1_heldout_result.py",
        ):
            source = (Path(__file__).resolve().parent / filename).read_text(
                encoding="utf-8"
            )
            self.assertNotIn(
                "from predict_stage_c_f0_1_heldout import",
                source,
            )


if __name__ == "__main__":
    unittest.main()
