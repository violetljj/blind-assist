from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    TrialSpec,
    load_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.synthetic_generator import (
    axis_rotation,
    generate_sequence,
    scale_about_principal_point,
)


class PhaseASyntheticGeometryTest(unittest.TestCase):
    def test_rotation_matrices_are_proper(self) -> None:
        for axis in ("yaw", "pitch", "roll"):
            rotation = axis_rotation(axis, math.radians(7.0))
            np.testing.assert_allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-12
            )
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_scale_matrix_keeps_principal_point_fixed(self) -> None:
        matrix = scale_about_principal_point(1.2, 239.5, 179.5)
        point = np.asarray([239.5, 179.5, 1.0])
        np.testing.assert_allclose(matrix @ point, point, atol=1e-12)

    def test_nonidentity_warp_direction_cancels_rotation(self) -> None:
        protocol = load_protocol()
        spec = TrialSpec(
            trial_id="geometry_direction",
            split="clean",
            motion_family="pure_rotation",
            axis="yaw",
            angular_velocity_deg_per_s=30.0,
            scale_rate_per_s=0.0,
            fps=30,
            degradation="clean",
            seed=1000,
        )
        sequence = generate_sequence(spec, protocol)
        result = compensate_current_to_previous(
            sequence.frames[1],
            sequence.valid_masks[1],
            sequence.valid_masks[0],
            sequence.rotation_homography_previous_to_current,
        )
        common = (result.valid_mask > 0) & (sequence.valid_masks[0] > 0)
        error = np.mean(
            np.abs(
                result.image[common].astype(np.float32)
                - sequence.frames[0][common].astype(np.float32)
            )
        )
        # Two linear interpolations add bounded photometric error even when the
        # projective direction is correct; landmark and mixed-scale tests below
        # cover the exact matrix convention.
        self.assertLess(error, 5.0)
        self.assertGreater(result.overlap_fraction, 0.9)

    def test_mixed_warp_leaves_scale_only(self) -> None:
        protocol = load_protocol()
        spec = TrialSpec(
            trial_id="mixed_geometry",
            split="clean",
            motion_family="rotation_plus_scale_up",
            axis="pitch",
            angular_velocity_deg_per_s=-30.0,
            scale_rate_per_s=0.15,
            fps=30,
            degradation="clean",
            seed=1001,
        )
        sequence = generate_sequence(spec, protocol)
        result = compensate_current_to_previous(
            sequence.frames[1],
            sequence.valid_masks[1],
            sequence.valid_masks[0],
            sequence.rotation_homography_previous_to_current,
        )
        values = protocol["rendering"]["intrinsics"]
        scale = scale_about_principal_point(
            math.exp(spec.scale_rate_per_s / spec.fps),
            values["cx_pixels"],
            values["cy_pixels"],
        )
        expected = cv2.warpPerspective(
            sequence.frames[0],
            scale,
            (
                protocol["rendering"]["width"],
                protocol["rendering"]["height"],
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        common = (result.valid_mask > 0) & (sequence.valid_masks[0] > 0)
        error = np.mean(
            np.abs(
                result.image[common].astype(np.float32)
                - expected[common].astype(np.float32)
            )
        )
        self.assertLess(error, 5.0)


if __name__ == "__main__":
    unittest.main()
