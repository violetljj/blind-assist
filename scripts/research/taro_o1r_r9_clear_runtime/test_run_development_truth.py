from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r9_clear_runtime import run_development_truth as runner


class DevelopmentTruthRunnerTests(unittest.TestCase):
    def test_remaining_roster_is_exact_and_selected_disjoint(self) -> None:
        frames, sources, receipts = runner.load_development_rows()
        self.assertEqual(len(frames), runner.DEVELOPMENT_FRAME_COUNT)
        self.assertEqual(len(sources), len(receipts))
        counts = {(frame.parent_id, frame.video_id) for frame in frames}
        self.assertEqual(len(counts), runner.DEVELOPMENT_PARENT_COUNT)


if __name__ == "__main__":
    unittest.main()
