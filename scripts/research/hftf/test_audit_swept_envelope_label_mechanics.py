from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_swept_envelope_label_mechanics import (
    STATE_RISK,
    STATE_SAFE,
    STATE_UNKNOWN,
    _ground_support,
    _structural_canaries,
    _swept_prism_counts,
    _swept_prism_probes_world,
    _tri_state_field,
)


class SweptEnvelopeLabelMechanicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        self.theta_edges = np.radians(
            np.asarray([-45.0, 0.0, 45.0])
        )
        self.distance_edges = np.asarray([0.0, 1.0, 2.0])
        self.bands = [
            (0.05, 0.35),
            (0.35, 1.35),
            (1.35, 2.05),
        ]

    def test_all_frozen_structural_canaries_pass(self) -> None:
        checks = _structural_canaries()
        self.assertTrue(all(checks.values()), checks)

    def test_swept_counts_are_height_specific_and_monotone(self) -> None:
        points = np.asarray(
            [
                [0.5, 1.5, 1.5],
                [0.0, 0.3, 1.5 * np.tan(np.radians(22.5))],
                [0.1, 0.8, 1.8],
            ]
        )
        narrow, _ = _swept_prism_counts(
            points,
            np.zeros(3, dtype=bool),
            self.basis,
            self.theta_edges,
            self.distance_edges,
            self.bands,
            np.asarray([0.1, 0.1, 0.1]),
        )
        wide, _ = _swept_prism_counts(
            points,
            np.zeros(3, dtype=bool),
            self.basis,
            self.theta_edges,
            self.distance_edges,
            self.bands,
            np.asarray([0.3, 0.4, 0.28]),
        )
        self.assertTrue(np.all(wide >= narrow))
        self.assertGreater(wide[:, :, 1].sum(), narrow[:, :, 1].sum())
        self.assertGreater(wide[:, :, 2].sum(), 0)
        self.assertEqual(1, wide[:, :, 2].sum())

    def test_probe_shape_is_nine_per_swept_prism(self) -> None:
        probes = _swept_prism_probes_world(
            self.basis,
            self.theta_edges,
            self.distance_edges,
            self.bands,
            np.asarray([0.3, 0.4, 0.28]),
        )
        self.assertEqual((12, 3, 9), probes.shape)

    def test_missing_ground_is_unknown_not_risk(self) -> None:
        known, risk, _ = _ground_support(
            np.zeros((3, 0)),
            self.basis,
            np.radians(np.asarray([-15.0, 15.0])),
            np.asarray([0.0, 2.0]),
            half_width_m=0.3,
            section_count=5,
            section_half_length_m=0.2,
            minimum_points_per_section=3,
            minimum_supported_sections=4,
            maximum_step_rise_m=0.18,
            maximum_drop_m=0.15,
        )
        self.assertFalse(known.any())
        self.assertEqual(0.0, float(risk.max()))

    def test_numeric_zero_is_not_safe_without_known_support(self) -> None:
        known = np.asarray([False, True, True])
        risk = np.asarray([0.0, 0.0, 0.5])
        state = _tri_state_field(known, risk)
        np.testing.assert_array_equal(
            np.asarray([STATE_UNKNOWN, STATE_SAFE, STATE_RISK]),
            state,
        )


if __name__ == "__main__":
    unittest.main()
