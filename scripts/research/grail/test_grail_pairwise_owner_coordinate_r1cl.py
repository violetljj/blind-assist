#!/usr/bin/env python3
"""Focused mechanics tests for GRAIL-R1C-L."""

from __future__ import annotations

import math
import unittest

import torch

from collect_grail_pairwise_owner_coordinate_r1cl import valid_bins, view_ordinals
from grail_pairwise_owner_coordinate_r1cl import (
    exchange_consistency_loss, predicted_slot_modes, slot_marginalized_loss, slot_mode_correct,
)


def _view(view_id: str, centroids: list[tuple[str, list[float]]]) -> dict:
    return {"view_id": view_id, "members": [
        {"object_id": object_id, "centroid": centroid} for object_id, centroid in centroids
    ]}


class R1CLMechanicsTest(unittest.TestCase):
    def test_preserved_slot_permutation_accepts_positive_cosine_bins(self) -> None:
        reference = _view("r", [("a", [10, 10]), ("b", [30, 10])])
        query = _view("q", [("a", [8, 12]), ("b", [28, 12])])
        bins, modes = valid_bins(reference, query)
        self.assertEqual(modes, ["PRESERVE"])
        self.assertTrue(bins)
        self.assertTrue(all(math.cos(math.radians(index * 10)) > 0 for index in bins))

    def test_flipped_slot_permutation_accepts_negative_cosine_bins(self) -> None:
        reference = _view("r", [("a", [10, 10]), ("b", [30, 10])])
        query = _view("q", [("a", [30, 12]), ("b", [10, 12])])
        bins, modes = valid_bins(reference, query)
        self.assertEqual(modes, ["FLIP"])
        self.assertTrue(all(math.cos(math.radians(index * 10)) < 0 for index in bins))

    def test_marginalized_loss_rewards_total_valid_mass(self) -> None:
        valid = torch.zeros(1, 36, dtype=torch.bool)
        valid[0, :3] = True
        good = torch.zeros(1, 36); good[0, :3] = 4
        bad = torch.zeros(1, 36); bad[0, 10:13] = 4
        self.assertLess(slot_marginalized_loss(good, valid), slot_marginalized_loss(bad, valid))

    def test_exchange_inverse_is_zero_for_exact_inverse_distribution(self) -> None:
        forward = torch.randn(2, 36)
        inverse = torch.tensor([(-index) % 36 for index in range(36)])
        reverse = forward.index_select(-1, inverse)
        self.assertLess(float(exchange_consistency_loss(forward, reverse)), 1e-6)

    def test_slot_mode_decision_uses_probability_mass_not_argmax(self) -> None:
        logits = torch.full((1, 36), -10.0)
        logits[0, 0] = 2.0
        logits[0, 10:18] = 1.0
        self.assertFalse(bool(predicted_slot_modes(logits)[0]))
        valid = torch.zeros(1, 36, dtype=torch.bool); valid[0, 10:18] = True
        self.assertTrue(bool(slot_mode_correct(logits, valid)[0]))

    def test_vertical_group_is_symmetry_marginalized(self) -> None:
        reference = _view("r", [("a", [10, 10]), ("b", [10, 30])])
        query = _view("q", [("a", [12, 10]), ("b", [12, 30])])
        _, modes = valid_bins(reference, query)
        self.assertEqual(modes, ["PRESERVE", "FLIP"])
        self.assertEqual(view_ordinals(reference, 1), view_ordinals(reference, -1))

    def test_partial_visibility_uses_shared_physical_siblings(self) -> None:
        reference = _view("r", [("a", [10, 10]), ("b", [20, 10]), ("c", [30, 10])])
        query = _view("q", [("a", [8, 12]), ("c", [28, 12])])
        bins, modes = valid_bins(reference, query)
        self.assertTrue(bins)
        self.assertEqual(modes, ["PRESERVE"])

    def test_partial_visibility_rejects_single_shared_sibling(self) -> None:
        reference = _view("r", [("a", [10, 10]), ("b", [20, 10])])
        query = _view("q", [("a", [8, 12]), ("c", [18, 12])])
        self.assertEqual(valid_bins(reference, query), ([], []))


if __name__ == "__main__":
    unittest.main()
