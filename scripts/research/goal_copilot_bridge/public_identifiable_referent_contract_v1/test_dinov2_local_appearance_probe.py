from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as sut,
)


class Dinov2LocalAppearanceProbeTest(unittest.TestCase):
    def test_relative_bbox_maps_object_into_crop(self) -> None:
        np.testing.assert_allclose(
            [0.25, 0.25, 0.75, 0.75],
            sut._relative_bbox([0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.5, 0.6]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_patch_mask_uses_patch_centers(self) -> None:
        mask = sut._patch_mask([0.25, 0.25, 0.75, 0.75])
        self.assertEqual((sut.PATCH_COUNT,), mask.shape)
        self.assertEqual(64, int(mask.sum()))

    def test_symmetric_score_is_permutation_invariant(self) -> None:
        reference = np.zeros((sut.PATCH_COUNT, sut.FEATURE_DIM), dtype="float32")
        candidate_a = np.zeros_like(reference)
        candidate_b = np.zeros_like(reference)
        reference[:, 0] = 1.0
        candidate_a[:, 0] = 1.0
        candidate_b[:, 1] = 1.0
        mask = np.ones(sut.PATCH_COUNT, dtype=bool)
        score_a = float(sut.symmetric_local_score(reference, candidate_a, mask, mask)["symmetric_score"])
        score_b = float(sut.symmetric_local_score(reference, candidate_b, mask, mask)["symmetric_score"])
        self.assertEqual("A", sut._winner(score_a, score_b))
        self.assertEqual("B", sut._winner(score_b, score_a))
        self.assertEqual(1.0, score_a)
        self.assertEqual(0.0, score_b)

    def test_score_config_rejects_private_target_mapping(self) -> None:
        with self.assertRaises(sut.LocalAppearanceProbeError):
            sut._assert_score_config_blind({"target_position": "A"})

    def test_history_categories_are_physical_outcome_based(self) -> None:
        oracle_report = {
            "rows": [
                {"observation_id": "stable-target", "stratum": "HISTORICAL_WRONG", "evaluation": "TARGET_SELECTED"},
                {
                    "observation_id": "stable-distractor",
                    "stratum": "HISTORICAL_WRONG",
                    "evaluation": "DISTRACTOR_SELECTED",
                },
                {"observation_id": "order", "stratum": "HISTORICAL_WRONG", "evaluation": "TARGET_SELECTED"},
            ]
        }
        counterbalance_report = {
            "rows": [
                {"observation_id": "stable-target", "evaluation": "TARGET_SELECTED"},
                {"observation_id": "stable-distractor", "evaluation": "DISTRACTOR_SELECTED"},
                {"observation_id": "order", "evaluation": "DISTRACTOR_SELECTED"},
            ]
        }
        self.assertEqual(
            {
                "order": "ORDER_SENSITIVE",
                "stable-distractor": "STABLE_DISTRACTOR",
                "stable-target": "ROBUST_TARGET",
            },
            sut._history_categories(oracle_report, counterbalance_report),
        )


if __name__ == "__main__":
    unittest.main()
