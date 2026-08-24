from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    spatial_layout_identity_verification_v0 as sut,
)


class SpatialLayoutIdentityVerificationV0Test(unittest.TestCase):
    def test_official_evaluation_sequence_ids_are_frozen(self) -> None:
        self.assertEqual(1, sut.REFERENCE_VIDEO)
        self.assertEqual(4, sut.CANDIDATE_VIDEO)

    def test_frame_selection_is_numeric_and_frozen(self) -> None:
        rows = [(index, f"frame-{index}.png") for index in range(10)]
        self.assertEqual("frame-2.png", sut._choose_member(rows, 0.25))
        self.assertEqual("frame-4.png", sut._choose_member(rows, 0.50))
        self.assertEqual("frame-6.png", sut._choose_member(rows, 0.75))

    def test_layout_score_is_direction_invariant(self) -> None:
        rng = np.random.default_rng(20260824)
        reference = rng.normal(size=(256, 384)).astype(np.float32)
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        candidate = reference.copy()
        forward = sut.spatial_layout_score(reference, candidate)
        reverse = sut.spatial_layout_score(candidate, reference)
        self.assertAlmostEqual(float(forward["score"]), float(reverse["score"]), places=12)
        self.assertGreater(float(forward["score"]), 0.95)

    def test_layout_penalizes_scrambled_spatial_correspondence(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(256, 384)).astype(np.float32)
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        coherent = sut.spatial_layout_score(reference, reference.copy())
        permutation = rng.permutation(256)
        scrambled = sut.spatial_layout_score(reference, reference[permutation])
        self.assertGreater(float(coherent["score"]), float(scrambled["score"]))

    def test_candidate_ranking_is_permutation_equivariant(self) -> None:
        self.assertEqual("A", sut._winner(0.8, 0.2))
        self.assertEqual("B", sut._winner(0.2, 0.8))
        self.assertEqual("TIE", sut._winner(0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
