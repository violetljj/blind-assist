#!/usr/bin/env python3
"""Focused CPU tests for multi-Teacher factor-label regrading."""

from __future__ import annotations

import unittest

import numpy as np

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    PROVENANCE_TEACHER,
    TIER_A_SOURCE,
    TIER_B_ANCHORED,
    TIER_C_TEACHER,
    TIER_UNKNOWN,
)
from build_ag_st_multiteacher_factor_labels import (
    regrade_teacher_labels,
    robust_observed_scale,
    teacher_pair_quality,
)


class MultiTeacherFactorLabelTests(unittest.TestCase):
    def test_observed_scale_uses_only_positive_overlap(self) -> None:
        observed = np.asarray([[2.0, 4.0], [0.0, np.nan]], dtype=np.float32)
        prediction = np.asarray([[1.0, 2.0], [9.0, 3.0]], dtype=np.float32)
        scale, support = robust_observed_scale(
            observed, prediction, minimum_support=2
        )
        self.assertEqual(support, 2)
        self.assertAlmostEqual(scale, 2.0)

    def test_pair_quality_is_symmetric_and_invalid_is_zero(self) -> None:
        first = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        second = np.asarray([[1.0, 2.2, 3.0]], dtype=np.float32)
        first_valid = np.asarray([[True, True, True]])
        second_valid = np.asarray([[True, True, False]])
        d12, q12, valid12 = teacher_pair_quality(
            first, first_valid, second, second_valid
        )
        d21, q21, valid21 = teacher_pair_quality(
            second, second_valid, first, first_valid
        )
        np.testing.assert_allclose(d12, d21, equal_nan=True)
        np.testing.assert_allclose(q12, q21)
        np.testing.assert_array_equal(valid12, valid21)
        self.assertAlmostEqual(float(q12[0, 0]), 1.0)
        self.assertLess(float(q12[0, 1]), 1.0)
        self.assertEqual(float(q12[0, 2]), 0.0)

    def test_regrade_preserves_source_and_rejects_disagreement(self) -> None:
        source = np.asarray([[True, True, False, False]])
        source_tiers = np.asarray(
            [[TIER_A_SOURCE, TIER_B_ANCHORED, TIER_C_TEACHER, TIER_C_TEACHER]],
            dtype=np.uint8,
        )
        source_provenance = np.asarray(
            [[
                PROVENANCE_SOURCE_NATIVE,
                PROVENANCE_SOURCE_NATIVE,
                PROVENANCE_TEACHER,
                PROVENANCE_TEACHER,
            ]],
            dtype=np.uint8,
        )
        primary_valid = np.asarray([[True, True, True, True]])
        base_quality = np.asarray([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
        anchor_quality = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        multiview_valid = np.asarray([[True, True, True, True]])
        pair_quality = np.asarray([[0.0, 0.0, 0.8, 0.01]], dtype=np.float32)
        pair_valid = np.asarray([[False, False, True, True]])
        tiers, provenance, scores = regrade_teacher_labels(
            source,
            source_tiers,
            source_provenance,
            primary_valid,
            base_quality,
            anchor_quality,
            multiview_valid,
            pair_quality,
            pair_valid,
        )
        self.assertEqual(int(tiers[0, 0]), TIER_A_SOURCE)
        self.assertEqual(int(provenance[0, 0]), PROVENANCE_SOURCE_NATIVE)
        self.assertGreater(float(scores[0, 0]), 0.9)
        self.assertAlmostEqual(float(scores[0, 1]), 0.90)
        self.assertGreater(int(tiers[0, 2]), TIER_UNKNOWN)
        self.assertEqual(int(provenance[0, 2]), PROVENANCE_TEACHER)
        self.assertEqual(int(tiers[0, 3]), TIER_UNKNOWN)
        self.assertEqual(int(provenance[0, 3]), 0)
        self.assertEqual(float(scores[0, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
