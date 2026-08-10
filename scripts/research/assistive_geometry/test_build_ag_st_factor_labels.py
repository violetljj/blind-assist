from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ag_st_factor_labels import (  # noqa: E402
    PROVENANCE_SOURCE_NATIVE,
    PROVENANCE_TEACHER,
    TIER_A_SOURCE,
    TIER_B_ANCHORED,
    TIER_C_TEACHER,
    TIER_UNKNOWN,
    assign_quality_tiers,
    compute_dense_normals,
    compute_geometric_factors,
    depth_uncertainty_proxy,
    projective_depth_residual,
    propagated_anchor_signal,
)


class AgStFactorLabelFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = np.asarray(
            [[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_anchor_signal_does_not_read_zeroed_hidden_values(self) -> None:
        observed = np.asarray([[1.0, 0.0, 0.0, 2.0]], dtype=np.float32)
        teacher = np.asarray([[1.1, 1.2, 1.8, 2.2]], dtype=np.float32)
        first = propagated_anchor_signal(observed, teacher)
        changed_hidden_reference = np.asarray([[1.0, 0.0, 0.0, 2.0]], dtype=np.float32)
        second = propagated_anchor_signal(changed_hidden_reference, teacher)
        for first_value, second_value in zip(first, second):
            np.testing.assert_array_equal(first_value, second_value)
        self.assertTrue(np.all(np.isfinite(first[0])))

    def test_identity_reprojection_has_zero_residual(self) -> None:
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        pose = np.eye(4, dtype=np.float64)
        residual, residual_valid = projective_depth_residual(
            depth,
            valid,
            self.intrinsics,
            pose,
            depth,
            valid,
            self.intrinsics,
            pose,
        )
        self.assertTrue(np.all(residual_valid))
        self.assertLess(float(np.max(np.abs(residual[residual_valid]))), 1e-6)

    def test_dense_normals_are_unit_and_camera_facing(self) -> None:
        depth = np.full((8, 8), 1.5, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        normals, normal_valid = compute_dense_normals(
            depth,
            valid,
            self.intrinsics,
        )
        self.assertEqual(36, int(normal_valid.sum()))
        values = normals[normal_valid]
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(values[:, 2] < 0))

    def test_source_priority_and_teacher_unknown_tiers(self) -> None:
        source = np.asarray([[True, True, False, False, False]])
        sensor = np.asarray([[2, 1, 0, 0, 0]], dtype=np.uint8)
        teacher = np.asarray([[True, True, True, True, True]])
        quality = np.asarray([[0.0, 0.0, 0.80, 0.40, 0.10]], dtype=np.float32)
        anchor = np.asarray([[0.0, 0.0, 0.90, 0.50, 0.10]], dtype=np.float32)
        multiview = np.asarray([[False, False, True, True, True]])
        tiers, provenance, _ = assign_quality_tiers(
            source,
            sensor,
            teacher,
            quality,
            anchor,
            multiview,
        )
        np.testing.assert_array_equal(
            tiers,
            np.asarray([[TIER_A_SOURCE, TIER_B_ANCHORED, TIER_B_ANCHORED, TIER_C_TEACHER, TIER_UNKNOWN]]),
        )
        np.testing.assert_array_equal(
            provenance,
            np.asarray([[PROVENANCE_SOURCE_NATIVE, PROVENANCE_SOURCE_NATIVE, PROVENANCE_TEACHER, PROVENANCE_TEACHER, 0]]),
        )

    def test_source_uncertainty_ignores_teacher_residual(self) -> None:
        depth = np.asarray([[1.0, 1.0]], dtype=np.float32)
        tiers = np.asarray([[TIER_B_ANCHORED, TIER_B_ANCHORED]], dtype=np.uint8)
        provenance = np.asarray([[PROVENANCE_SOURCE_NATIVE, PROVENANCE_TEACHER]], dtype=np.uint8)
        sigma = depth_uncertainty_proxy(
            depth,
            tiers,
            provenance,
            np.asarray([[0.9, 0.9]], dtype=np.float32),
            np.asarray([[0.5, 0.5]], dtype=np.float32),
            np.asarray([[0.5, 0.5]], dtype=np.float32),
            np.asarray([[True, True]]),
        )
        self.assertAlmostEqual(0.025, float(sigma[0, 0]), places=6)
        self.assertGreater(float(sigma[0, 1]), float(sigma[0, 0]))

    def test_support_and_boundary_derive_from_metric_geometry(self) -> None:
        depth = np.full((32, 32), 1.50, dtype=np.float32)
        depth[:, 24:] = 2.00
        valid = np.ones_like(depth, dtype=np.bool_)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        factors = compute_geometric_factors(
            depth,
            valid,
            np.asarray([[120.0, 0.0, 15.5], [0.0, 120.0, 15.5], [0.0, 0.0, 1.0]]),
            pose,
            np.full_like(depth, 0.95),
            np.full_like(depth, TIER_A_SOURCE, dtype=np.uint8),
            np.full_like(depth, PROVENANCE_SOURCE_NATIVE, dtype=np.uint8),
            np.full_like(depth, 0.03),
        )
        self.assertTrue(bool(factors["support_plane_valid"]))
        self.assertGreater(float(np.max(factors["boundary_probability_pseudo_hw"][:, 23:25])), 0.80)
        self.assertTrue(np.all((factors["support_probability_pseudo_hw"] >= 0) & (factors["support_probability_pseudo_hw"] <= 1)))
        self.assertTrue(np.all((factors["obstacle_evidence_truth_hw"] >= 0) & (factors["obstacle_evidence_truth_hw"] <= 1)))

    def test_continuous_steep_plane_is_not_a_boundary(self) -> None:
        height = width = 24
        intrinsics = np.asarray(
            [[120.0, 0.0, 11.5], [0.0, 120.0, 11.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        theta = np.deg2rad(80.0)
        up_camera = np.asarray([np.sin(theta), 0.0, -np.cos(theta)], dtype=np.float64)
        rows, columns = np.indices((height, width), dtype=np.float64)
        rays = np.stack(
            (
                (columns - intrinsics[0, 2]) / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) / intrinsics[1, 1],
                np.ones((height, width), dtype=np.float64),
            ),
            axis=-1,
        )
        depth = (-1.0 / np.einsum("...i,i->...", rays, up_camera)).astype(np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        cosine = float(np.cos(theta))
        sine = float(np.sin(theta))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            [[-cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, -cosine]],
            dtype=np.float64,
        )
        factors = compute_geometric_factors(
            depth,
            valid,
            intrinsics,
            pose,
            np.full_like(depth, 0.95),
            np.full_like(depth, TIER_A_SOURCE, dtype=np.uint8),
            np.full_like(depth, PROVENANCE_SOURCE_NATIVE, dtype=np.uint8),
            np.full_like(depth, 0.03),
        )
        interior = factors["boundary_probability_pseudo_hw"][2:-2, 2:-2]
        self.assertLess(float(np.percentile(interior, 95.0)), 0.05)

    def test_derived_normal_inherits_weakest_local_provenance(self) -> None:
        depth = np.full((32, 32), 1.5, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        tiers = np.full_like(depth, TIER_A_SOURCE, dtype=np.uint8)
        provenance = np.full_like(depth, PROVENANCE_SOURCE_NATIVE, dtype=np.uint8)
        tiers[16, 16] = TIER_C_TEACHER
        provenance[16, 16] = PROVENANCE_TEACHER
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        factors = compute_geometric_factors(
            depth,
            valid,
            np.asarray([[120.0, 0.0, 15.5], [0.0, 120.0, 15.5], [0.0, 0.0, 1.0]]),
            pose,
            np.full_like(depth, 0.95),
            tiers,
            provenance,
            np.full_like(depth, 0.03),
        )
        self.assertEqual(TIER_C_TEACHER, int(factors["normal_quality_tier_hw"][16, 16]))
        self.assertEqual(PROVENANCE_TEACHER, int(factors["normal_provenance_code_hw"][16, 16]))


if __name__ == "__main__":
    unittest.main()
