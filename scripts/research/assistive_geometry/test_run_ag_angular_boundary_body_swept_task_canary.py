from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ag_angular_boundary_body_swept_task_canary import (
    comparison_metrics,
    replace_observed_boundary,
    task_gain_gates,
)


class AngularBoundaryBodySweptTaskCanaryTest(unittest.TestCase):
    def test_replace_observed_boundary_preserves_completion(self) -> None:
        prediction = {
            "factor_identity": {"source": "fixture"},
            "obstacle_boundary_evidence": {
                "boundary_probability_hw": [[0.2, 0.3], [0.4, 0.5]],
            },
        }
        observed = np.asarray([[True, False], [False, True]])
        learned = np.zeros((8, 8), dtype=np.float32)
        learned[:4, :4] = 0.7
        learned[:4, 4:] = 0.8
        learned[4:, :4] = 0.9
        learned[4:, 4:] = 1.0
        output = replace_observed_boundary(
            prediction,
            observed,
            learned,
            variant="fixture",
            checkpoint_sha256="A" * 64,
        )
        np.testing.assert_allclose(
            output["obstacle_boundary_evidence"]["boundary_probability_hw"],
            [[0.7, 0.3], [0.4, 1.0]],
            atol=1e-7,
            rtol=0.0,
        )

    def test_comparison_metrics_keeps_unknown_out_of_negative_class(self) -> None:
        reference = [{
            ("left", 1.0): "CLEAR_OBSERVED",
            ("center", 1.0): "OCCUPIED_OBSERVED",
            ("right", 1.0): "UNKNOWN",
        }]
        candidate = [{
            ("left", 1.0): "CLEAR_OBSERVED",
            ("center", 1.0): "UNKNOWN",
            ("right", 1.0): "CLEAR_OBSERVED",
        }]
        metrics = comparison_metrics(reference, candidate)
        self.assertEqual(metrics["reference_known_count"], 2)
        self.assertEqual(metrics["correct_reference_known_count"], 1)
        self.assertEqual(metrics["abstain_on_reference_known_count"], 1)
        self.assertEqual(metrics["spurious_definite_count"], 1)
        self.assertEqual(metrics["unsafe_clear_count"], 1)

    def test_task_gate_requires_strict_safe_gain(self) -> None:
        r20 = {
            "exact_match_count": 80,
            "correct_reference_known_count": 20,
            "abstain_on_reference_known_count": 5,
        }
        r21 = {
            "exact_match_count": 81,
            "correct_reference_known_count": 21,
            "abstain_on_reference_known_count": 4,
            "unsafe_clear_count": 0,
            "spurious_definite_count": 0,
            "candidate_known_count": 25,
        }
        self.assertTrue(all(task_gain_gates(r20, r21).values()))
        r21["exact_match_count"] = 80
        r21["correct_reference_known_count"] = 20
        r21["abstain_on_reference_known_count"] = 5
        self.assertFalse(
            task_gain_gates(r20, r21)["R21TASK_C11_STRICT_BODY_SWEPT_TASK_GAIN_VS_R20"]
        )


if __name__ == "__main__":
    unittest.main()
