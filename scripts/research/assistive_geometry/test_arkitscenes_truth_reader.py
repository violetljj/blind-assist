import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_trajectory,
    rotate_intrinsics_upright,
    upright_to_source_basis,
)


def synthetic_floor_with_center_obstacle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 192, 256
    fx = fy = 128.0
    cx, cy = 127.5, 95.5
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    rows, columns = np.mgrid[0:height, 0:width]
    ray_y = (rows - cy) / fy
    depth = np.zeros((height, width), dtype=np.float32)
    floor = ray_y > 0
    depth[floor] = 1.5 / ray_y[floor]
    depth[(depth < 0.25) | (depth > 6.0)] = 0.0
    depth[55:180, 112:144] = 1.0
    confidence = np.full((height, width), 2, dtype=np.uint8)
    return depth, confidence, intrinsics


class ArkitScenesTruthReaderTest(unittest.TestCase):
    def test_millimetres_convert_to_metres_and_zero_stays_zero(self) -> None:
        raw = np.asarray([[0, 1000, 2500]], dtype=np.uint16)
        np.testing.assert_allclose(depth_mm_to_metres(raw), [[0.0, 1.0, 2.5]])

    def test_pose_interpolation_uses_official_inverse_convention(self) -> None:
        trajectory = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, math.pi / 2, -1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        transform, receipt = interpolate_camera_to_world(trajectory, 0.05, 0.25)
        self.assertAlmostEqual(0.5, receipt["fraction"])
        self.assertAlmostEqual(math.sqrt(0.5), transform[0, 0], places=6)
        self.assertTrue(np.all(np.isfinite(transform)))

    def test_rotation_and_k_preserve_projection_for_all_quadrants(self) -> None:
        width, height = 256, 192
        source_k = np.asarray([[210.0, 0.0, 128.0], [0.0, 211.0, 96.0], [0.0, 0.0, 1.0]])
        upright_point = np.asarray([0.2, -0.1, 2.0])
        for index in range(4):
            basis = upright_to_source_basis(index)
            source_point = basis @ upright_point
            source_pixel = source_k @ source_point
            source_pixel /= source_pixel[2]
            upright_k, output_size = rotate_intrinsics_upright(source_k, width, height, index)
            upright_pixel = upright_k @ upright_point
            upright_pixel /= upright_pixel[2]
            if index == 0:
                expected = source_pixel[:2]
            elif index == 1:
                expected = np.asarray([height - 1 - source_pixel[1], source_pixel[0]])
            elif index == 2:
                expected = np.asarray([width - 1 - source_pixel[0], height - 1 - source_pixel[1]])
            else:
                expected = np.asarray([source_pixel[1], width - 1 - source_pixel[0]])
            np.testing.assert_allclose(upright_pixel[:2], expected, atol=1e-9)
            self.assertEqual(output_size, (width, height) if index in (0, 2) else (height, width))

    def test_canonicalize_rotates_all_registered_modalities_together(self) -> None:
        rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        depth = np.arange(6, dtype=np.uint16).reshape(2, 3)
        confidence = np.arange(6, dtype=np.uint8).reshape(2, 3) % 3
        intrinsics = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[2, :3] = [-1.0, 0.0, 0.0]
        result = canonicalize_frame(rgb, depth, confidence, intrinsics, pose)
        self.assertEqual(1, result["rotation_index"])
        np.testing.assert_array_equal(result["depth_raw_mm"], np.rot90(depth, -1))
        np.testing.assert_array_equal(result["confidence"], np.rot90(confidence, -1))

    def test_gravity_ground_and_center_obstacle_are_derived(self) -> None:
        depth, confidence, intrinsics = synthetic_floor_with_center_obstacle()
        result = derive_assistive_truth(
            depth,
            confidence,
            intrinsics,
            np.asarray([0.0, -1.0, 0.0]),
        )
        self.assertEqual("VALID", result["status"])
        self.assertAlmostEqual(1.5, result["ground_plane"]["camera_height_m"], delta=0.05)
        self.assertTrue(np.all(result["ground_probability"][result["ground_valid"]] == 1.0))
        self.assertLess(int(np.sum(result["ground_valid"])), int(np.sum(result["depth_valid"])))
        self.assertLess(result["bands"]["center"]["clearance_m"], 1.1)
        self.assertTrue(result["bands"]["center"]["occupied_by_horizon"]["1.0"])
        self.assertFalse(result["bands"]["left"]["occupied_by_horizon"]["1.0"])
        self.assertFalse(result["bands"]["right"]["occupied_by_horizon"]["1.0"])

    def test_low_confidence_or_missing_ground_fails_closed(self) -> None:
        depth = np.ones((20, 20), dtype=np.float32)
        confidence = np.zeros((20, 20), dtype=np.uint8)
        intrinsics = np.asarray([[20.0, 0.0, 10.0], [0.0, 20.0, 10.0], [0.0, 0.0, 1.0]])
        result = derive_assistive_truth(depth, confidence, intrinsics, np.asarray([0.0, -1.0, 0.0]))
        self.assertEqual("UNKNOWN", result["status"])
        self.assertIn("UNKNOWN_INSUFFICIENT_VALID_DEPTH", result["unknown_reasons"])

    def test_malformed_trajectory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.traj"
            path.write_text("0 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seven fields"):
                parse_trajectory(path)


if __name__ == "__main__":
    unittest.main()
