from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from evaluate_bonn_rgbd_consumed import associate_unique_nearest, sample_causal
from evaluate_arkitscenes_rgbd_consumed import dense_truth
from validate_bonn_rgbd_result import recompute, terminal


class BonnRgbdEvaluationTest(unittest.TestCase):
    def test_association_is_bounded_and_unique(self) -> None:
        rgb = [(1.0, Path("r0")), (1.01, Path("r1")), (1.10, Path("r2"))]
        depth = [(1.005, Path("d0")), (1.095, Path("d1"))]
        pairs = associate_unique_nearest(rgb, depth, 0.02)
        self.assertEqual(["r0", "r2"], [str(row[1]) for row in pairs])
        self.assertEqual(["d0", "d1"], [str(row[3]) for row in pairs])

    def test_sampling_is_causal_from_last_selected_pair(self) -> None:
        pairs = [(value, Path("r"), value, Path("d")) for value in (0, .1, .21, .4, .42)]
        selected = sample_causal(pairs, 5.0)
        self.assertEqual([0, .21, .42], [row[0] for row in selected])

    def test_validator_recomputes_accuracy_and_opposite_error(self) -> None:
        def row(index: int, truth: str, candidate: str) -> dict:
            return {
                "sequence_id": "s",
                "video_id": "v",
                "role": "validation",
                "frame_index": index,
                "truth_score": {"status": "VALID"},
                "candidate_score": {"status": "VALID"},
                "truth_decision": {"status": "VALID", "label": truth},
                "candidate_decision": {"status": "VALID", "label": candidate},
            }

        rows = [
            row(0, "RELATIVELY_OPEN_LEFT", "RELATIVELY_OPEN_LEFT"),
            row(1, "RELATIVELY_OPEN_LEFT", "RELATIVELY_OPEN_RIGHT"),
            row(2, "AMBIGUOUS", "AMBIGUOUS"),
        ]
        summary = recompute(rows)[0]
        self.assertEqual(2, summary["truth_directional_support"])
        self.assertEqual(0.5, summary["directional_accuracy"])
        self.assertEqual(0.5, summary["opposite_direction_rate"])
        self.assertEqual(2 / 3, summary["exact_decision_agreement"])
        self.assertEqual("v", summary["video_id"])
        self.assertEqual("validation", summary["role"])

    def test_terminal_fails_closed_on_truth_support(self) -> None:
        summary = {
            "truth_score_coverage": 1.0,
            "truth_directional_support": 9,
            "candidate_execution_coverage": 1.0,
            "recommendation_coverage": 1.0,
            "directional_accuracy": 1.0,
            "opposite_direction_rate": 0.0,
        }
        gates = {
            "minimum_truth_score_coverage_each_sequence": 0.5,
            "minimum_directional_truth_support_each_sequence": 10,
            "minimum_candidate_execution_coverage_each_sequence": 0.95,
            "minimum_recommendation_coverage_each_sequence": 0.5,
            "minimum_directional_accuracy_worst_sequence": 0.6,
            "minimum_directional_accuracy_macro": 0.75,
            "maximum_macro_opposite_direction_rate": 0.05,
        }
        self.assertEqual(
            "SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT",
            terminal([summary], gates),
        )

    def test_arkit_truth_reconstruction_requires_source_support(self) -> None:
        contract = {
            "confidence_value": 2,
            "minimum_depth_m": 0.25,
            "maximum_depth_m": 6.0,
            "minimum_source_valid_fraction_per_frame": 0.5,
        }
        depth = np.full((16, 16), 1000, dtype=np.uint16)
        confidence = np.full((16, 16), 2, dtype=np.uint8)
        confidence[:, :32 // 5] = 0
        dense, fraction = dense_truth(depth, confidence, contract)
        self.assertIsNotNone(dense)
        self.assertGreater(fraction, 0.5)
        self.assertTrue(np.allclose(dense, 1.0))
        confidence[:, 6:] = 0
        dense, _ = dense_truth(depth, confidence, contract)
        self.assertIsNone(dense)


if __name__ == "__main__":
    unittest.main()
