#!/usr/bin/env python3
"""Focused CPU tests for the TUM cross-source SuperTeacher runner."""

from __future__ import annotations

import unittest

import numpy as np

from build_ag_st_factor_labels import PROVENANCE_SOURCE_NATIVE, PROVENANCE_TEACHER
from run_ag_st_tum_cross_source import (
    build_depth_label_payload,
    cross_source_passes,
)


class TumCrossSourceTests(unittest.TestCase):
    def test_cross_source_gate_requires_coverage_and_error_separation(self) -> None:
        evaluation = {
            "overall": {
                "coverage": 0.6,
                "accepted": {"count": 60, "mae_m": 0.02},
                "rejected": {"count": 40, "mae_m": 0.10},
            },
            "parents": [{}, {}, {}],
            "evaluable_parent_count": 3,
            "accepted_lower_risk_parent_count": 2,
        }
        self.assertTrue(cross_source_passes(evaluation))
        evaluation["overall"]["rejected"]["mae_m"] = 0.01
        self.assertFalse(cross_source_passes(evaluation))

    def test_depth_materialization_keeps_source_and_rejects_low_quality_hole(self) -> None:
        shape = (1, 3)
        record = {
            "source_valid": np.asarray([[True, False, False]]),
            "truth_depth_m": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "primary_valid": np.ones(shape, dtype=np.bool_),
            "pair_valid": np.ones(shape, dtype=np.bool_),
            "combined_quality": np.asarray([[0.0, 0.8, 0.1]], dtype=np.float32),
            "multiview_valid": np.ones(shape, dtype=np.bool_),
            "anchor_quality": np.ones(shape, dtype=np.float32),
            "primary_depth_m": np.asarray([[9.0, 2.1, 3.2]], dtype=np.float32),
            "secondary_depth_m": np.asarray([[8.0, 2.0, 2.7]], dtype=np.float32),
            "pair_relative_disagreement": np.asarray(
                [[0.1, 0.05, 0.17]], dtype=np.float32
            ),
            "pair_quality": np.asarray([[0.6, 0.8, 0.4]], dtype=np.float32),
            "intrinsics": np.eye(3, dtype=np.float64),
            "camera_to_world": np.eye(4, dtype=np.float64),
        }
        label = build_depth_label_payload(record)
        np.testing.assert_array_equal(
            label["metric_depth_valid_hw"], [[True, True, False]]
        )
        self.assertEqual(float(label["metric_depth_m_hw"][0, 0]), 1.0)
        self.assertAlmostEqual(float(label["metric_depth_m_hw"][0, 1]), 2.1, places=6)
        self.assertTrue(np.isnan(label["metric_depth_m_hw"][0, 2]))
        self.assertEqual(
            int(label["provenance_code_hw"][0, 0]), PROVENANCE_SOURCE_NATIVE
        )
        self.assertEqual(
            int(label["provenance_code_hw"][0, 1]), PROVENANCE_TEACHER
        )
        self.assertFalse(np.any(label["support_valid_hw"]))
        self.assertFalse(np.any(label["boundary_evidence_valid_hw"]))


if __name__ == "__main__":
    unittest.main()
