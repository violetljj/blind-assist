from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_temporal_392_clip_p1_r0_1 import TRANSITIONS, evaluate


def protocol() -> dict:
    return {
        "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_protocol",
        "clip_p1": {
            "minimum_evaluable_clips": 32,
            "minimum_video_parents": 8,
            "bootstrap": {"draws": 100, "seed": 20260805},
            "noninferiority_tolerances": {
                "minimum_pixel_coverage_drop": 0.01,
                "metric_abs_rel_absolute_increase": 0.02,
                "scale_aligned_abs_rel_absolute_increase": 0.01,
                "clearance_mae_increase_m": 0.025,
                "clearance_delta_mae_increase_m": 0.015,
                "clearance_delta_pair_coverage_drop": 0.01,
                "transition_macro_f1_drop": 0.05,
                "transition_exact_agreement_drop": 0.05,
                "ground_recovery_rate_drop": 0.01,
                "false_clear_rate_increase": 0.01,
                "invalid_to_known_rate_increase": 0.01,
                "valid_to_unknown_rate_increase": 0.02,
            },
            "absolute_gates": {
                "maximum_clearance_mae_m": 0.25,
                "maximum_clearance_delta_mae_m": 0.15,
                "maximum_false_clear_rate": 0.05,
                "maximum_invalid_to_known_rate": 0.05,
                "maximum_valid_to_unknown_rate": 0.10,
            },
        },
    }


def ledger() -> dict:
    rows = []
    truth_states = ("CLEAR", "OCCUPIED", "OCCUPIED", "CLEAR")
    clearances = (2.0, 1.5, 1.0, 1.4)
    for clip_index in range(32):
        parent = f"parent-{clip_index % 8}"
        for frame_index in range(4):
            external_invalid = frame_index == 2
            truth_state = truth_states[frame_index]
            truth = {
                "clearance_m": [clearances[frame_index]] * 3,
                "clearance_valid": [True] * 3,
                "geometry_state": [truth_state] * 3,
                "external_abstain_target": [external_invalid] * 3,
            }
            prediction = {
                "paired_pixel_coverage": 1.0,
                "metric_abs_rel": 0.10,
                "scale_aligned_abs_rel": 0.05,
                "clearance_m": [clearances[frame_index]] * 3,
                "geometry_state": [truth_state] * 3,
                "external_abstain": [external_invalid] * 3,
            }
            rows.append(
                {
                    "clip_id": f"clip-{clip_index}",
                    "parent_id": parent,
                    "frame_index": frame_index,
                    "timestamp_ns": 1_000_000_000 + frame_index * 100_000_000,
                    "truth": truth,
                    "baseline": copy.deepcopy(prediction),
                    "candidate": copy.deepcopy(prediction),
                }
            )
    return {
        "schema": "blindassist_dav2_temporal_392_clip_p1_r0_1_opened_ledger",
        "identity_manifest_sha256": "A" * 64,
        "sealed_bundle_sha256": "B" * 64,
        "rows": rows,
    }


class ClipP1EvaluatorTest(unittest.TestCase):
    def test_identity_candidate_passes_every_gate(self) -> None:
        result = evaluate(protocol(), ledger())
        self.assertTrue(result["passed"])
        self.assertEqual(result["undefined_metrics"], [])
        self.assertTrue(all(result["gates"].values()))

    def test_clearance_delta_regression_fails(self) -> None:
        value = ledger()
        for row in value["rows"]:
            if row["frame_index"] % 2:
                row["candidate"]["clearance_m"] = [3.0] * 3
        result = evaluate(protocol(), value)
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["clearance_delta_noninferiority"])

    def test_undefined_invalid_denominator_fails_closed(self) -> None:
        value = ledger()
        for row in value["rows"]:
            row["truth"]["external_abstain_target"] = [False] * 3
            row["baseline"]["external_abstain"] = [False] * 3
            row["candidate"]["external_abstain"] = [False] * 3
        result = evaluate(protocol(), value)
        self.assertFalse(result["passed"])
        self.assertIn("invalid_to_known_rate", result["undefined_metrics"])

    def test_one_nonfinite_frame_metric_cannot_be_dropped(self) -> None:
        value = ledger()
        value["rows"][0]["candidate"]["metric_abs_rel"] = float("nan")
        result = evaluate(protocol(), value)
        self.assertFalse(result["passed"])
        self.assertIn("metric_abs_rel_median", result["undefined_metrics"])


if __name__ == "__main__":
    unittest.main()
