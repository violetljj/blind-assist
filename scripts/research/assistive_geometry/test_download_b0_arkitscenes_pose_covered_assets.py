import tempfile
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.download_b0_arkitscenes_pose_covered_assets import (
    pose_covered_common_stems,
    read_trajectory_bounds,
)


class DownloadB0ArkitScenesPoseCoveredAssetsTest(unittest.TestCase):
    def test_pose_covered_window_skips_pretrajectory_frames(self) -> None:
        maps = {
            name: {f"9_{index / 10:.1f}": f"{name}/{index}.png" for index in range(20)}
            for name in ("rgb", "depth", "confidence")
        }
        selected = pose_covered_common_stems(maps, 3, 0.5, 1.5)
        self.assertEqual(["9_0.5", "9_0.6", "9_0.7"], selected)

    def test_insufficient_pose_covered_window_fails(self) -> None:
        maps = {
            name: {f"9_{index / 10:.1f}": f"{name}/{index}.png" for index in range(5)}
            for name in ("rgb", "depth", "confidence")
        }
        with self.assertRaisesRegex(ValueError, "pose-covered"):
            pose_covered_common_stems(maps, 3, 0.4, 0.5)

    def test_trajectory_bounds_require_strict_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.traj"
            path.write_text(
                "1.0 0 0 0 0 0 0\n1.0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_trajectory_bounds(path)


if __name__ == "__main__":
    unittest.main()
