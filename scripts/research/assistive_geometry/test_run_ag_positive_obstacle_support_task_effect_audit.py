from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ag_positive_obstacle_support_task_effect_audit import (
    replace_positive_factors,
    select_bottleneck,
)


class PositiveObstacleSupportTaskEffectAuditTest(unittest.TestCase):
    @staticmethod
    def prediction() -> dict:
        return {
            "factor_identity": {"learned_final_task_head": False},
            "depth_scale": {"sentinel": "unchanged"},
            "support_surface": {"support_probability_hw": [[0.1, 0.2], [0.3, 0.4]]},
            "obstacle_boundary_evidence": {
                "obstacle_evidence_probability_hw": [[0.5, 0.6], [0.7, 0.8]],
                "boundary_probability_hw": [[0.9, 0.9], [0.9, 0.9]],
            },
        }

    def test_one_factor_replacement_preserves_depth_boundary_and_completion(self) -> None:
        learned = np.zeros((8, 8), dtype=np.float32)
        learned[:4, :4] = 0.9
        learned[:4, 4:] = 0.8
        learned[4:, :4] = 0.7
        learned[4:, 4:] = 0.6
        source = self.prediction()
        output, receipt = replace_positive_factors(
            source,
            np.asarray([[True, False], [False, True]]),
            learned_support_probability=learned,
            learned_obstacle_probability=None,
            arm="support",
            checkpoint_sha256="A" * 64,
        )
        np.testing.assert_allclose(
            output["support_surface"]["support_probability_hw"],
            [[0.9, 0.2], [0.3, 0.6]],
            atol=1e-7,
            rtol=0.0,
        )
        self.assertEqual(output["depth_scale"], source["depth_scale"])
        self.assertEqual(
            output["obstacle_boundary_evidence"]["boundary_probability_hw"],
            source["obstacle_boundary_evidence"]["boundary_probability_hw"],
        )
        self.assertTrue(receipt["support_replaced"])
        self.assertFalse(receipt["obstacle_replaced"])

    def test_bottleneck_prefers_obstacle_when_it_creates_more_unsafe_clear(self) -> None:
        def row(unsafe: int, abstain: int, exact: int) -> dict:
            return {
                "unsafe_clear_count": unsafe,
                "abstain_on_reference_known_count": abstain,
                "exact_match_count": exact,
            }

        selected = select_bottleneck(
            {
                "learned_support_only": row(0, 4, 90),
                "learned_obstacle_only": row(3, 1, 80),
                "learned_support_plus_obstacle": row(3, 5, 70),
            }
        )
        self.assertEqual(selected["primary_bottleneck"], "POSITIVE_OBSTACLE_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
