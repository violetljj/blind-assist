import unittest

import numpy as np

from evaluate_consumed_tartanground import summarize_arm, up_optical_from_pose


class EvaluateConsumedTartanGroundTest(unittest.TestCase):
    def test_ned_to_optical_up_mapping(self) -> None:
        pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(up_optical_from_pose(pose), [0.0, -1.0, 0.0])

    def test_summary_counts_unknown_and_false_clear(self) -> None:
        records = [
            {
                "parent_id": "p",
                "anchor_frame_id": 0,
                "truth": {band: 0.5 for band in ("left", "center", "right")},
                "candidate": {band: 3.0 for band in ("left", "center", "right")},
            },
            {
                "parent_id": "p",
                "anchor_frame_id": 1,
                "truth": {band: 0.5 for band in ("left", "center", "right")},
                "candidate": None,
            },
        ]
        summary = summarize_arm(records, "candidate")["parent_macro"]
        self.assertEqual(0.5, summary["known_coverage"])
        self.assertEqual(1.0, summary["false_clear_rate"])


if __name__ == "__main__":
    unittest.main()
