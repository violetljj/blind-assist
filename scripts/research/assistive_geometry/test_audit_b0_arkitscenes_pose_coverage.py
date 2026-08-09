import tempfile
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.audit_b0_arkitscenes_pose_coverage import (
    audit_video,
    read_trajectory_timestamps,
)


class AuditB0ArkitScenesPoseCoverageTest(unittest.TestCase):
    def test_pose_covered_video_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / "lowres_wide"
            rgb.mkdir()
            (root / "lowres_wide.traj").write_text(
                "1.0 0 0 0 0 0 0\n2.0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            video = {
                "role": "TRAIN",
                "visit_id": "1",
                "video_id": "2",
                "selected_frame_count": 2,
                "selected_frame_stems": ["2_1.1", "2_1.9"],
                "extracted": {"lowres_wide": [{"path": str(rgb / "2_1.1.png")}]},
            }
            result = audit_video(video)
            self.assertTrue(result["all_selected_frames_pose_covered"])
            self.assertEqual(2, result["pose_covered_frame_count"])

    def test_frame_before_trajectory_fails_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / "lowres_wide"
            rgb.mkdir()
            (root / "lowres_wide.traj").write_text(
                "1.0 0 0 0 0 0 0\n2.0 0 0 0 0 0 0\n",
                encoding="utf-8",
            )
            video = {
                "role": "DEVELOPMENT",
                "visit_id": "3",
                "video_id": "4",
                "selected_frame_count": 2,
                "selected_frame_stems": ["4_0.9", "4_1.1"],
                "extracted": {"lowres_wide": [{"path": str(rgb / "4_0.9.png")}]},
            }
            result = audit_video(video)
            self.assertFalse(result["all_selected_frames_pose_covered"])
            self.assertEqual(1, result["pose_covered_frame_count"])

    def test_malformed_trajectory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.traj"
            path.write_text("1.0 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "7 fields"):
                read_trajectory_timestamps(path)


if __name__ == "__main__":
    unittest.main()
