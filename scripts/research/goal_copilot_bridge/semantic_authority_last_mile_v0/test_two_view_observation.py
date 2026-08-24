import unittest
import tempfile
from pathlib import Path

import numpy as np

from .observation import CameraIntrinsics, RgbEpisodeInput, RgbEpisodeTruth
from .two_view_observation import (
    SourceCameraPose,
    SourcePoseTwoViewBoundaryProvider,
    detect_vertical_lines,
    oracle_pixel_lines,
    triangulate_aperture,
)
from .materialize_rgb_cohort import _frame_pose, _project_aperture_to_view, _trajectory


class SourcePoseTwoViewBoundaryTest(unittest.TestCase):
    def test_materializer_reads_official_world_to_camera_trajectory_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = root / "lowres_wide.traj"
            trajectory_path.write_text(
                "0.0 0 0 0 -1 -2 -3\n1.0 0 0 0 -2 -2 -3\n",
                encoding="utf-8",
            )
            frame = root / "fixture_0.0.png"
            position, rotation = _frame_pose(frame, _trajectory(trajectory_path))
            np.testing.assert_allclose(position, [1.0, 2.0, 3.0], atol=1e-9)
            np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)

    def test_materializer_accepts_aperture_visible_in_second_view(self) -> None:
        intrinsics = {"width": 320, "height": 240, "fx": 200.0, "fy": 200.0, "cx": 160.0, "cy": 120.0}
        truth = {"left_x_px": 100.0, "right_x_px": 220.0, "range_m": 2.0}
        projected = _project_aperture_to_view(
            np.zeros(3), np.eye(3), np.asarray([0.24, 0.0, 0.1]), np.eye(3), intrinsics, truth
        )
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertGreater(projected[0], 0.03 * intrinsics["width"])
        self.assertLess(projected[1], 0.97 * intrinsics["width"])

    def test_materializer_rejects_aperture_outside_second_view(self) -> None:
        intrinsics = {"width": 320, "height": 240, "fx": 200.0, "fy": 200.0, "cx": 160.0, "cy": 120.0}
        truth = {"left_x_px": 100.0, "right_x_px": 220.0, "range_m": 2.0}
        projected = _project_aperture_to_view(
            np.zeros(3), np.eye(3), np.asarray([1.8, 0.0, 0.0]), np.eye(3), intrinsics, truth
        )
        self.assertIsNone(projected)

    def test_two_endpoint_line_fit_preserves_vertical_geometry(self) -> None:
        image = np.zeros((192, 256, 3), dtype=np.uint8)
        image[:, :32] = 25
        image[:, 32:] = 220
        lines = detect_vertical_lines(image, None)
        boundary = min(lines, key=lambda line: abs(line.x_at(96.0) - 31.5))
        self.assertLess(abs(boundary.x_at(0.0) - boundary.x_at(191.0)), 2.0)
        self.assertLess(abs(boundary.x_at(96.0) - 31.5), 2.0)

    def test_oracle_pixels_recover_metric_aperture(self) -> None:
        episode_input = RgbEpisodeInput(
            "fixture",
            "ROOM_SIGN",
            (),
            CameraIntrinsics(256, 192, 200.0, 200.0, 128.0, 96.0),
            0.24,
            1,
            (),
        )
        truth = RgbEpisodeTruth(
            "fixture",
            0.0,
            1.0,
            2.5,
            ((0.0, 0.0, 0.0), (0.24, 0.0, 0.0)),
            0.0,
            (88.0, 168.0),
        )
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        pose_a = SourceCameraPose((0.0, 0.0, 0.0), identity)
        pose_b = SourceCameraPose((0.24, 0.0, 0.0), identity)
        first, second = oracle_pixel_lines(episode_input, truth, pose_a, pose_b)
        intrinsic = np.asarray([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])

        geometry = triangulate_aperture(*first, *second, pose_a, pose_b, intrinsic, 105.0)

        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertAlmostEqual(geometry.center_x_m, 0.0, places=6)
        self.assertAlmostEqual(geometry.width_m, 1.0, places=6)
        self.assertAlmostEqual(geometry.range_m, 2.5, places=6)
        self.assertGreater(geometry.confidence, 0.8)

    def test_automatic_arm_cannot_receive_evaluator_truth(self) -> None:
        episode_input = RgbEpisodeInput(
            "fixture", "ROOM_SIGN", (), CameraIntrinsics(1, 1, 1.0, 1.0, 0.0, 0.0), 0.24, 0, ()
        )
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        pose = SourceCameraPose((0.0, 0.0, 0.0), identity)
        provider = SourcePoseTwoViewBoundaryProvider(episode_input, None, pose, pose, "b2")
        self.assertIsNone(provider.truth)
        with self.assertRaisesRegex(ValueError, "requires evaluator boundary truth"):
            SourcePoseTwoViewBoundaryProvider(episode_input, None, pose, pose, "b0")
        truth = RgbEpisodeTruth("fixture", 0.0, 1.0, 2.0, ((0.0, 0.0, 0.0),), 0.0, (1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "cannot receive evaluator truth"):
            SourcePoseTwoViewBoundaryProvider(episode_input, truth, pose, pose, "b2")


if __name__ == "__main__":
    unittest.main()
