from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    nearid_hard_negative_unary_v0 as sut,
)


class NearIdentityHardNegativeUnaryV0Test(unittest.TestCase):
    def test_object_routes_are_frozen_within_split(self) -> None:
        self.assertEqual(2, sut._next_same_category(1))
        self.assertEqual(1, sut._next_same_category(5))
        self.assertEqual(6, sut._ordinary_object(1, "train"))
        self.assertEqual(1, sut._ordinary_object(16, "train"))
        self.assertEqual(41, sut._ordinary_object(36, "test"))
        self.assertEqual(36, sut._ordinary_object(46, "test"))

    def test_frame_selection_uses_frozen_floor_quantile(self) -> None:
        members = {(1, 1): [f"root/s1/o1/C_01_01_{index:03d}.png" for index in range(10)]}
        self.assertTrue(sut._choose_member(members, 1, 1, 0.25).endswith("002.png"))
        self.assertTrue(sut._choose_member(members, 1, 1, 0.50).endswith("004.png"))
        self.assertTrue(sut._choose_member(members, 1, 1, 0.75).endswith("006.png"))

    def test_threshold_respects_predeclared_false_accept_limit(self) -> None:
        scores = [float(index) for index in range(60)]
        threshold = sut.select_absence_threshold(scores)
        self.assertEqual(3, sum(score >= threshold for score in scores))
        self.assertGreater(threshold, 56.0)
        self.assertLess(threshold, 57.0)

    def test_unary_decision_is_candidate_permutation_invariant(self) -> None:
        scores = {"A": 0.8, "B": 0.2}
        swapped = {"A": scores["B"], "B": scores["A"]}
        self.assertEqual("A", sut._decision(scores, 0.5))
        self.assertEqual("B", sut._decision(swapped, 0.5))
        self.assertEqual("NONE", sut._decision(scores, 0.9))

    def test_near_identity_loss_rewards_declared_order(self) -> None:
        import torch

        anchor = torch.tensor([[1.0, 0.0]])
        good = sut.near_identity_loss(
            anchor,
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.7, np.sqrt(0.51)]]),
            torch.tensor([[0.0, 1.0]]),
        )
        bad = sut.near_identity_loss(
            anchor,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.2, np.sqrt(0.96)]]),
            torch.tensor([[0.7, np.sqrt(0.51)]]),
        )
        self.assertLess(float(good), float(bad))


if __name__ == "__main__":
    unittest.main()
