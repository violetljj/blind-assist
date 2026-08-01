from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_swept_envelope_label_mechanics import (  # noqa: E402
    _swept_prism_counts,
)
from run_stage_c_g0_signed_clearance_mechanics import (  # noqa: E402
    _box_support_equivalent_clearance,
    _mask_unknown_targets,
    _signed_clearance_field,
)


class StageCG0SignedClearanceMechanicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        self.theta = np.radians(np.asarray([-15.0, 15.0]))
        self.distance = np.asarray([0.0, 2.0])
        self.bands = [(0.35, 1.35), (1.35, 2.05)]
        self.widths = np.asarray([0.4, 0.28])
        self.kwargs = {
            "order_statistic": 2,
            "final_edge_atol_m": 1e-12,
            "final_edge_rtol": 0.0,
            "clip_min_m": -0.5,
            "clip_max_m": 1.0,
        }

    def clearance(
        self, points: list[list[float]], widths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        _, clipped, counts = _signed_clearance_field(
            np.asarray(points, dtype=np.float64).T,
            self.basis,
            self.theta,
            self.distance,
            self.bands,
            self.widths if widths is None else widths,
            **self.kwargs,
        )
        return clipped, counts

    def test_second_order_zero_threshold_matches_two_point_support(self) -> None:
        one, one_count = self.clearance([[1.0, 0.0, 0.8]])
        two, two_count = self.clearance(
            [[1.0, 0.0, 0.8], [1.2, 0.1, 0.9]]
        )
        self.assertEqual(1, one_count[0, 0, 0])
        self.assertGreater(one[0, 0, 0], 0.0)
        self.assertEqual(2, two_count[0, 0, 0])
        self.assertLessEqual(two[0, 0, 0], 0.0)

    def test_half_open_longitudinal_boundary_is_not_duplicated(self) -> None:
        along = np.asarray([1.0])
        cross = np.asarray([0.0])
        height = np.asarray([0.8])
        left_inside = np.asarray([False])
        left = _box_support_equivalent_clearance(
            along,
            cross,
            height,
            left_inside,
            distance_lower=0.0,
            distance_upper=1.0,
            height_lower=0.35,
            height_upper=1.35,
            half_width=0.4,
        )
        right_inside = np.asarray([True])
        right = _box_support_equivalent_clearance(
            along,
            cross,
            height,
            right_inside,
            distance_lower=1.0,
            distance_upper=2.0,
            height_lower=0.35,
            height_upper=1.35,
            half_width=0.4,
        )
        self.assertGreater(left[0], 0.0)
        self.assertLess(right[0], 0.0)

    def test_exact_boundary_tie_uses_smallest_float64_sign(self) -> None:
        along = np.asarray([1.0, 1.0])
        cross = np.asarray([0.0, 0.0])
        height = np.asarray([0.8, 0.8])
        proxy = _box_support_equivalent_clearance(
            along,
            cross,
            height,
            np.asarray([False, True]),
            distance_lower=0.0,
            distance_upper=1.0,
            height_lower=0.35,
            height_upper=1.35,
            half_width=0.4,
        )
        self.assertEqual(
            np.nextafter(0.0, np.inf),
            proxy[0],
        )
        self.assertEqual(
            np.nextafter(0.0, -np.inf),
            proxy[1],
        )

    def test_final_distance_isclose_semantics_are_support_equivalent(
        self,
    ) -> None:
        points = np.asarray(
            [[8.0 + 5e-13, 8.0 + 5e-13], [0.0, 0.1], [0.8, 0.9]]
        )
        _, clipped, counts = _signed_clearance_field(
            points,
            self.basis,
            self.theta,
            np.asarray([0.0, 4.0, 8.0]),
            self.bands,
            self.widths,
            **self.kwargs,
        )
        self.assertEqual(2, counts[0, 1, 0])
        self.assertLess(clipped[0, 1, 0], 0.0)

    def test_membership_counts_match_frozen_teacher_mechanics(self) -> None:
        points = np.asarray(
            [
                [0.0, 1.0, 2.0, 2.0 + 5e-13, 1.5],
                [0.0, 0.0, 0.0, 0.1, 0.4],
                [0.35, 0.8, 1.35, 0.9, 2.05],
            ]
        )
        distance = np.asarray([0.0, 1.0, 2.0])
        _, _, proxy_counts = _signed_clearance_field(
            points,
            self.basis,
            self.theta,
            distance,
            self.bands,
            self.widths,
            **self.kwargs,
        )
        teacher_counts, _ = _swept_prism_counts(
            points,
            np.zeros(points.shape[1], dtype=bool),
            self.basis,
            self.theta,
            distance,
            self.bands,
            self.widths,
        )
        np.testing.assert_array_equal(teacher_counts, proxy_counts)

    def test_fewer_than_order_statistic_uses_positive_infinity_preclip(
        self,
    ) -> None:
        raw, clipped, counts = _signed_clearance_field(
            np.asarray([[1.0], [0.0], [0.8]]),
            self.basis,
            self.theta,
            self.distance,
            self.bands,
            self.widths,
            **self.kwargs,
        )
        self.assertEqual(1, counts[0, 0, 0])
        self.assertTrue(np.isposinf(raw[0, 0, 0]))
        self.assertEqual(1.0, clipped[0, 0, 0])

    def test_unknown_target_and_safe_state_remain_null(self) -> None:
        target, safe = _mask_unknown_targets(
            np.asarray([[-0.1, 0.2]]),
            np.asarray([[True, False]]),
        )
        self.assertEqual(-0.1, target[0, 0])
        self.assertIsNone(target[0, 1])
        self.assertFalse(safe[0, 0])
        self.assertIsNone(safe[0, 1])

    def test_wider_envelope_cannot_increase_clearance(self) -> None:
        points = [[1.0, 0.35, 0.8], [1.2, 0.36, 0.9]]
        narrow, _ = self.clearance(points, np.asarray([0.2, 0.2]))
        wide, _ = self.clearance(points, np.asarray([0.4, 0.4]))
        self.assertLessEqual(wide[0, 0, 0], narrow[0, 0, 0])

    def test_height_layers_are_separate(self) -> None:
        body, _ = self.clearance(
            [[1.0, 0.0, 0.8], [1.2, 0.1, 0.9]]
        )
        head, _ = self.clearance(
            [[1.0, 0.0, 1.7], [1.2, 0.1, 1.8]]
        )
        self.assertLessEqual(body[0, 0, 0], 0.0)
        self.assertGreater(body[0, 0, 1], 0.0)
        self.assertGreater(head[0, 0, 0], 0.0)
        self.assertLessEqual(head[0, 0, 1], 0.0)

    def test_invalid_or_nonfinite_points_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self.clearance([[1.0, 0.0, float("nan")]])
        with self.assertRaisesRegex(ValueError, "3xN"):
            _signed_clearance_field(
                np.zeros((2, 3)),
                self.basis,
                self.theta,
                self.distance,
                self.bands,
                self.widths,
                **self.kwargs,
            )


if __name__ == "__main__":
    unittest.main()
