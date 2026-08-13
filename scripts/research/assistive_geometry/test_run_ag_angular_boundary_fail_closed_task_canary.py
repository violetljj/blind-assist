from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ag_angular_boundary_fail_closed_task_canary import (
    bind_boundary_as_one_sided_uncertainty,
    component_edge,
    fail_closed_task_gates,
)


class AngularBoundaryFailClosedTaskCanaryTest(unittest.TestCase):
    @staticmethod
    def prediction() -> dict:
        return {
            "factor_identity": {"learned_final_task_head": False},
            "obstacle_boundary_evidence": {
                "obstacle_evidence_probability_hw": [[0.9, 0.0], [0.0, 0.0]],
                "boundary_probability_hw": [[0.9, 0.0], [0.0, 0.0]],
                "boundary_localization_sigma_px_hw": [[0.5, 0.5], [0.5, 0.5]],
            },
        }

    def test_component_edge_is_not_component_interior(self) -> None:
        mask = np.ones((5, 5), dtype=np.bool_)
        edge = component_edge(mask)
        self.assertEqual(int(edge.sum()), 16)
        self.assertFalse(bool(edge[2, 2]))

    def test_absent_boundary_never_negates_obstacle_and_increases_sigma(self) -> None:
        output, receipt = bind_boundary_as_one_sided_uncertainty(
            self.prediction(),
            np.ones((2, 2), dtype=np.bool_),
            np.zeros((8, 8), dtype=np.float32),
            variant="fixture",
            checkpoint_sha256="A" * 64,
        )
        evidence = output["obstacle_boundary_evidence"]
        self.assertAlmostEqual(evidence["boundary_probability_hw"][0][0], 0.9, places=6)
        self.assertEqual(evidence["boundary_localization_sigma_px_hw"][0][0], 12.0)
        self.assertGreaterEqual(
            receipt["minimum_boundary_minus_obstacle_on_positive_blocks"], 0.0
        )
        self.assertGreater(receipt["minimum_sigma_increment_on_obstacle_blocks"], 0.0)

    def test_aligned_boundary_keeps_sigma_floor(self) -> None:
        learned = np.zeros((8, 8), dtype=np.float32)
        learned[:4, :4] = 1.0
        output, receipt = bind_boundary_as_one_sided_uncertainty(
            self.prediction(),
            np.ones((2, 2), dtype=np.bool_),
            learned,
            variant="fixture",
            checkpoint_sha256="A" * 64,
        )
        self.assertEqual(
            output["obstacle_boundary_evidence"]["boundary_localization_sigma_px_hw"][0][0],
            0.5,
        )
        self.assertEqual(receipt["minimum_sigma_increment_on_obstacle_blocks"], 0.0)

    def test_gate_requires_strict_safe_gain(self) -> None:
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
        self.assertTrue(all(fail_closed_task_gates(r20, r21).values()))
        r21["unsafe_clear_count"] = 1
        self.assertFalse(fail_closed_task_gates(r20, r21)["R21FC_C07_NO_UNSAFE_CLEAR"])


if __name__ == "__main__":
    unittest.main()
