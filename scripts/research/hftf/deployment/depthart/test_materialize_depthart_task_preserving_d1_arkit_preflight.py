from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d1_arkit_preflight import (
    continuous_window,
    select_final,
)


class DepthArtD1BodyPreflightTest(unittest.TestCase):
    def test_continuous_window_skips_earlier_broken_run(self) -> None:
        stems = ["x_0.0", "x_0.1", "x_1.0", "x_1.1", "x_1.2"]
        self.assertEqual(["x_1.0", "x_1.1", "x_1.2"], continuous_window(stems, 3, 0.5))

    def test_continuous_window_fails_when_short(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous"):
            continuous_window(["x_0.0", "x_1.0"], 2, 0.5)

    def test_first_eligible_reserve_replaces_failed_primary(self) -> None:
        videos = [
            {"role": "PRIMARY", "frozen_order": index, "visit_id": f"p{index}", "video_id": f"pv{index}", "eligible": index != 2}
            for index in range(1, 9)
        ] + [
            {"role": "RESERVE", "frozen_order": index, "visit_id": f"r{index}", "video_id": f"rv{index}", "eligible": index >= 3}
            for index in range(1, 9)
        ]
        selected, replacements = select_final(videos)
        self.assertEqual(8, len(selected))
        self.assertEqual("rv3", replacements[0]["reserve_video_id"])
        self.assertIn({"visit_id": "r3", "video_id": "rv3"}, selected)

    def test_insufficient_reserve_cannot_fake_complete_roster(self) -> None:
        videos = [
            {"role": "PRIMARY", "frozen_order": index, "visit_id": f"p{index}", "video_id": f"pv{index}", "eligible": index == 1}
            for index in range(1, 9)
        ] + [
            {"role": "RESERVE", "frozen_order": index, "visit_id": f"r{index}", "video_id": f"rv{index}", "eligible": False}
            for index in range(1, 9)
        ]
        selected, _ = select_final(videos)
        self.assertEqual(1, len(selected))


if __name__ == "__main__":
    unittest.main()
