from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("rgb_timeline", SCRIPTS / "acquire_sanpo_rgb_timeline_candidate.py")
assert SPEC and SPEC.loader
rgb_timeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rgb_timeline)


class SanpoRgbTimelineTest(unittest.TestCase):
    def test_selects_sparse_ten_second_window_without_masks(self) -> None:
        objects = [{"name": f"prefix/{index:06d}.png", "generation": "1", "size": "1"} for index in range(150, 286)]
        selected = rgb_timeline.select_rgb_window(objects, source_fps=15.0, target_fps=1.0, start_frame=150, frame_count=10)
        self.assertEqual([150, 165, 180, 195, 210, 225, 240, 255, 270, 285], [item[0] for item in selected])

    def test_rejects_incomplete_window(self) -> None:
        objects = [{"name": f"prefix/{index:06d}.png"} for index in range(150, 200)]
        with self.assertRaisesRegex(rgb_timeline.CandidateError, "only"):
            rgb_timeline.select_rgb_window(objects, source_fps=15.0, target_fps=1.0, start_frame=150, frame_count=10)

    def test_allows_partial_output_but_rejects_completed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "images" / "0000_000150.png").write_bytes(b"verified partial")
            rgb_timeline.assert_output_can_resume(root)
            (root / "candidate_spec.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(rgb_timeline.CandidateError, "completed"):
                rgb_timeline.assert_output_can_resume(root)


if __name__ == "__main__":
    unittest.main()
