from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as discovery,
)
from scripts.research.egomotion_compensated_looming.rotation_compensation_mechanism_audit_r1 import (
    source_coordinate_audit,
    synthetic_direction_audit,
)


class RotationCompensationMechanismAuditR1Test(unittest.TestCase):
    def test_official_wxyz_rotation(self) -> None:
        half = math.radians(10.0) / 2.0
        quaternion = np.asarray(
            (math.cos(half), 0.0, math.sin(half), 0.0)
        )
        rotation = discovery.quaternion_rotation_wxyz(quaternion)
        expected = np.asarray(
            (
                (math.cos(math.radians(10.0)), 0.0, math.sin(math.radians(10.0))),
                (0.0, 1.0, 0.0),
                (-math.sin(math.radians(10.0)), 0.0, math.cos(math.radians(10.0))),
            )
        )
        np.testing.assert_allclose(rotation, expected, rtol=0.0, atol=1e-12)

    def test_wxyz_is_not_silently_xyzw(self) -> None:
        quaternion = np.asarray((0.9, 0.1, 0.2, 0.3), dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        self.assertFalse(
            np.allclose(
                discovery.quaternion_rotation_wxyz(quaternion),
                discovery.quaternion_rotation_xyzw(quaternion),
            )
        )

    def test_synthetic_correct_reverse_and_raw_matrix(self) -> None:
        result = synthetic_direction_audit.run_audit()
        self.assertEqual(result["case_count"], 6)
        self.assertTrue(result["all_pass"])
        for row in result["rows"]:
            self.assertLess(row["correct_mae"], row["raw_mae"])
            self.assertLess(row["correct_mae"], row["reverse_mae"])

    def test_coordinate_candidates_include_direct_and_reverse(self) -> None:
        previous = np.asarray((1.0, 0.0, 0.0, 0.0))
        current = synthetic_direction_audit.axis_quaternion_wxyz(
            "yaw_y", 10.0
        )
        candidates = source_coordinate_audit.candidate_homographies(
            previous, current
        )
        direct = candidates["official_wxyz_direct"]
        reverse = candidates["official_wxyz_reverse"]
        np.testing.assert_allclose(
            direct @ reverse,
            np.eye(3),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(
            set(candidates),
            {
                "identity_no_rotation",
                "official_wxyz_direct",
                "official_wxyz_reverse",
                "legacy_xyzw_direct",
                "negative_z_basis_conjugated",
                "t_cam_imu_rotation_conjugated",
            },
        )

    def test_pair_geometry_applies_official_camera_imu_basis(self) -> None:
        previous_pose = (
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
        )
        current_quaternion = (
            synthetic_direction_audit.axis_quaternion_wxyz(
                "pitch_x", 10.0
            )
        )
        current_pose = (np.zeros(3), current_quaternion)
        matrix, _, _ = discovery.pair_geometry(
            previous_pose,
            current_pose,
            1.0,
            quaternion_component_order="wxyz",
            pose_to_camera_rotation=discovery.T_CAM_IMU_ROTATION,
        )
        source_relative = (
            discovery.quaternion_rotation_wxyz(current_quaternion).T
        )
        expected_rotation = (
            discovery.T_CAM_IMU_ROTATION
            @ source_relative
            @ discovery.T_CAM_IMU_ROTATION.T
        )
        expected = (
            discovery.INTRINSIC
            @ expected_rotation
            @ np.linalg.inv(discovery.INTRINSIC)
        )
        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-12)

    def test_undistort_valid_mask_excludes_out_of_bounds_map(self) -> None:
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        maps = discovery.build_undistort_maps(720, 1280)
        image, valid = discovery.preprocess_frame_with_mask(
            frame, 0.5, maps
        )
        self.assertEqual(image.shape, (640, 360))
        self.assertEqual(valid.shape, image.shape)
        fraction = float(np.mean(valid > 0))
        self.assertGreater(fraction, 0.95)
        self.assertLess(fraction, 0.99)

    def test_raw_preprocess_mask_is_full(self) -> None:
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        _, valid = discovery.preprocess_frame_with_mask(frame, 0.5, None)
        self.assertTrue(np.all(valid == 255))


if __name__ == "__main__":
    unittest.main()
