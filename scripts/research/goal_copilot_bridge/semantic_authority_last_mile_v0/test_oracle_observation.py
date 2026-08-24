import unittest

from .observation import CameraIntrinsics, ExactAnchorObservation, RgbEpisodeInput, RgbEpisodeTruth
from .oracle_observation import OracleApertureObservationProvider


class OracleApertureObservationProviderTest(unittest.TestCase):
    def test_all_oracle_values_are_evaluator_truth(self) -> None:
        episode_input = RgbEpisodeInput(
            episode_id="fixture",
            kind="ROOM_SIGN",
            rgb_frames=(),
            intrinsics=CameraIntrinsics(256, 192, 212.0, 212.0, 128.0, 96.0),
            commanded_baseline_m=0.24,
            active_parallax_frame_index=1,
            exact_anchor_observations=(ExactAnchorObservation(0, "anchor", (1, 2, 3, 4)),),
        )
        truth = RgbEpisodeTruth("fixture", -0.31, 0.94, 2.7, ((0.0, 0.0, 0.0), (0.24, 0.0, 0.0)), -0.31)
        provider = OracleApertureObservationProvider(episode_input, truth)

        observation = provider.observe()

        self.assertEqual(observation.center_x_m, -0.31)
        self.assertEqual(observation.width_m, 0.94)
        self.assertEqual(observation.range_m, 2.7)
        self.assertEqual(observation.geometry_confidence, 1.0)
        self.assertEqual(provider.diagnostics["source_camera_positions_m"][1][0], 0.24)

    def test_rejects_truth_for_a_different_episode(self) -> None:
        episode_input = RgbEpisodeInput(
            "input", "ROOM_SIGN", (), CameraIntrinsics(1, 1, 1.0, 1.0, 0.0, 0.0), 0.24, 0, ()
        )
        truth = RgbEpisodeTruth("truth", 0.0, 1.0, 2.0, ((0.0, 0.0, 0.0),), 0.0)
        with self.assertRaisesRegex(ValueError, "input/truth episode mismatch"):
            OracleApertureObservationProvider(episode_input, truth)


if __name__ == "__main__":
    unittest.main()
