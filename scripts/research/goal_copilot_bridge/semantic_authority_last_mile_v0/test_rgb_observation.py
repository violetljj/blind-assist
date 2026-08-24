import inspect
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .observation import CameraIntrinsics, ExactAnchorObservation, RgbEpisodeInput, RgbEpisodeTruth
from .rgb_observation import RgbObservationProvider


class RgbObservationProviderTest(unittest.TestCase):
    def test_provider_surface_cannot_receive_truth(self) -> None:
        parameters = inspect.signature(RgbObservationProvider).parameters
        self.assertEqual(set(parameters), {"episode_input", "depth_estimator"})
        self.assertNotIn("truth", RgbEpisodeInput.__dataclass_fields__)
        self.assertIn("aperture_center_x_m", RgbEpisodeTruth.__dataclass_fields__)

    def test_boundary_flow_depth_channels_produce_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(4)
            first = rng.integers(35, 105, size=(192, 256, 3), dtype=np.uint8)
            cv2.line(first, (72, 25), (72, 165), (245, 245, 245), 4)
            cv2.line(first, (160, 25), (160, 165), (245, 245, 245), 4)
            cv2.rectangle(first, (36, 60), (64, 88), (230, 230, 230), -1)
            transform = np.float32([[1, 0, 2], [0, 1, 0]])
            second = cv2.warpAffine(first, transform, (256, 192), borderMode=cv2.BORDER_REFLECT)
            paths = (root / "first.png", root / "second.png")
            cv2.imwrite(str(paths[0]), first)
            cv2.imwrite(str(paths[1]), second)
            depth = np.full((192, 256), 2.0, dtype=np.float32)
            depth[:, 76:157] = 3.0
            episode_input = RgbEpisodeInput(
                episode_id="fixture",
                kind="QR_ENTRANCE",
                rgb_frames=paths,
                intrinsics=CameraIntrinsics(256, 192, 212.0, 212.0, 128.0, 96.0),
                commanded_baseline_m=0.24,
                active_parallax_frame_index=1,
                exact_anchor_observations=(
                    ExactAnchorObservation(0, "qr::fixture", (36, 60, 64, 88)),
                    ExactAnchorObservation(1, "qr::fixture", (38, 60, 66, 88)),
                ),
            )
            observation = RgbObservationProvider(episode_input, depth_estimator=lambda _: depth).observe()
            self.assertTrue(observation.visible)
            self.assertIsNotNone(observation.center_x_m)
            self.assertGreater(observation.boundary_confidence, 0.45)
            self.assertGreater(observation.flow_confidence, 0.35)
            self.assertGreater(observation.depth_consistency, 0.45)
            self.assertGreater(observation.geometry_confidence, 0.35)


if __name__ == "__main__":
    unittest.main()
