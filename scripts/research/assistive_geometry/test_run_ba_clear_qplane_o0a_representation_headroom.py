import json
from pathlib import Path
import unittest

import numpy as np

from scripts.research.assistive_geometry.run_ba_clear_qplane_o0a_representation_headroom import (
    DEFAULT_PROTOCOL,
    apply_plane_residual,
    fit_ridge,
    soft_interval,
    summarize_records,
    support_and_evaluation_masks,
)
from scripts.research.assistive_geometry.replay_ba_clear_qplane_o0a_query_decomposition import (
    summarize_per_query,
)


REPRESENTATION = {
    "corrected_depth_clip_m": [0.05, 20.0],
    "epsilon_rho_m_inverse": 0.2,
    "maximum_fit_pixels": 20000,
    "minimum_fit_pixels": 32,
    "query_mask_far_margin_m": 0.25,
    "query_mask_lateral_margin_m": 0.2,
    "query_mask_near_margin_m": 0.1,
    "ridge_lambda": 1e-8,
    "support_fit_forward_range_m": [0.2, 3.0],
    "support_fit_horizon_decay_m": 0.75,
    "support_fit_minimum_query_weight": 0.02,
    "support_plane_tolerance_m": 0.045,
}


class QPlaneRepresentationHeadroomTest(unittest.TestCase):
    def test_frozen_protocol_preserves_o0a_authority_boundary(self) -> None:
        protocol = json.loads(Path(DEFAULT_PROTOCOL).read_text(encoding="utf-8"))
        self.assertEqual(protocol["status"], "FROZEN_REPRESENTATION_AUDIT_ONLY")
        self.assertFalse(protocol["authority"]["training_authorized"])
        self.assertFalse(protocol["authority"]["fresh_outcome_authorized"])
        self.assertFalse(protocol["authority"]["android_qnn_htp_authorized"])
        self.assertFalse(
            protocol["representation"]["dense_corrected_depth_persistence_allowed"]
        )
        self.assertEqual(
            protocol["fit_evaluation_isolation"][
                "required_fit_evaluation_pixel_overlap"
            ],
            0,
        )

    def test_ridge_recovers_three_parameter_geometry_residual(self) -> None:
        rng = np.random.default_rng(7)
        basis = rng.normal(size=(24, 20, 3))
        expected = np.asarray([0.04, -0.015, 0.025])
        target = np.einsum("...j,j->...", basis, expected)
        theta, full_count, used_count = fit_ridge(
            basis,
            target,
            np.ones(target.shape, dtype=bool),
            REPRESENTATION,
        )
        np.testing.assert_allclose(theta, expected, atol=1e-8)
        self.assertEqual(full_count, target.size)
        self.assertEqual(used_count, target.size)

    def test_fit_support_and_obstacle_evaluation_cells_are_disjoint(self) -> None:
        height, width = 80, 100
        intrinsics = np.asarray(
            [[60.0, 0.0, 49.5], [0.0, 60.0, 29.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rows = np.arange(height, dtype=np.float64)[:, None]
        ray_y = (rows - intrinsics[1, 2]) / intrinsics[1, 1]
        depth = np.zeros((height, width), dtype=np.float64)
        floor = ray_y > 0.0
        depth[floor.repeat(width, axis=1)] = np.repeat(
            1.2 / ray_y, width, axis=1
        )[floor.repeat(width, axis=1)]
        depth[55:70, 45:55] = 1.0
        support, local_support, evaluation, local_weight = support_and_evaluation_masks(
            depth,
            intrinsics,
            np.asarray([0.0, -1.0, 0.0]),
            "center",
            2.0,
            REPRESENTATION,
        )
        self.assertGreater(int(np.sum(support)), 100)
        self.assertGreater(int(np.sum(local_support)), 32)
        self.assertGreater(int(np.sum(evaluation)), 0)
        self.assertEqual(int(np.sum(local_support & evaluation)), 0)
        self.assertGreater(float(np.sum(local_weight[local_support])), 0.0)

    def test_query_application_is_temporary_and_keeps_base_unchanged(self) -> None:
        base = np.full((8, 10), 2.0, dtype=np.float64)
        before = base.copy()
        intrinsics = np.asarray(
            [[8.0, 0.0, 4.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        mask = np.zeros(base.shape, dtype=np.float64)
        mask[:, :5] = 1.0
        corrected = apply_plane_residual(
            base,
            intrinsics,
            np.asarray([0.0, -1.0, 0.0]),
            [0.05, 0.0, 0.0],
            mask,
            REPRESENTATION,
        )
        np.testing.assert_array_equal(base, before)
        self.assertFalse(np.shares_memory(base, corrected))
        self.assertTrue(np.isfinite(corrected).all())
        self.assertTrue(np.all(corrected[:, :5] < base[:, :5]))
        np.testing.assert_allclose(corrected[:, 5:], base[:, 5:])

    def test_unknown_candidate_is_not_counted_as_negative_outcome(self) -> None:
        def value(known: bool, occupied: bool | None, clearance: float | None):
            return {
                "known": known,
                "occupied": occupied,
                "clearance_m": clearance,
            }

        records = [
            {
                "arms": {
                    "A5_SOURCE_DEPTH_ORACLE": value(True, True, 0.5),
                    "A4_QUERY_LOCAL_RAY_PLANE": value(False, None, None),
                }
            },
            {
                "arms": {
                    "A5_SOURCE_DEPTH_ORACLE": value(True, False, 2.5),
                    "A4_QUERY_LOCAL_RAY_PLANE": value(True, True, 1.5),
                }
            },
        ]
        summary = summarize_records(records, "A4_QUERY_LOCAL_RAY_PLANE")
        self.assertEqual(summary["truth_known_decisions"], 2)
        self.assertEqual(summary["known_decisions"], 1)
        self.assertEqual(summary["coverage"], 0.5)
        self.assertEqual(summary["false_block_count"], 1)
        self.assertEqual(summary["false_clear_count"], 0)

    def test_soft_interval_honors_asymmetric_near_and_far_margins(self) -> None:
        values = np.asarray([0.10, 0.15, 0.20, 2.00, 2.125, 2.25])
        weights = soft_interval(values, 0.2, 2.0, 0.1, 0.25)
        np.testing.assert_allclose(weights, [0.0, 0.5, 1.0, 1.0, 0.5, 0.0])

    def test_reporting_replay_keeps_band_horizon_queries_separate(self) -> None:
        value = {"known": True, "occupied": False, "clearance_m": 2.5}
        occupied = {"known": True, "occupied": True, "clearance_m": 0.5}
        records = [
            {
                "band": "left",
                "horizon_m": horizon,
                "arms": {
                    arm: value
                    for arm in (
                        "A0_FROZEN_DEPTHART",
                        "A1_GLOBAL_SCALE",
                        "A2_GLOBAL_AFFINE",
                        "A3_GLOBAL_RAY_PLANE",
                        "A4_QUERY_LOCAL_RAY_PLANE",
                        "NC_SHUFFLED_QUERY",
                        "NC_WRONG_GRAVITY",
                        "NC_WRONG_K",
                    )
                }
                | {"A5_SOURCE_DEPTH_ORACLE": occupied},
            }
            for horizon in (1.0, 1.5)
        ]
        summary = summarize_per_query(records)
        self.assertEqual(set(summary), {"left@1.0m", "left@1.5m"})
        self.assertEqual(
            summary["left@1.0m"]["A4_QUERY_LOCAL_RAY_PLANE"][
                "false_clear_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
