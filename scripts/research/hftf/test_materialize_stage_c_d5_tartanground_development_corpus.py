import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d5_tartanground_development_corpus import (
    centered_even_block_start,
    label_record,
    member_frame_id,
)


class TartanGroundDevelopmentCorpusTest(unittest.TestCase):
    def test_centered_block_is_even_and_bounded(self):
        start = centered_even_block_start(609)
        self.assertEqual(start % 2, 0)
        self.assertGreaterEqual(start, 0)
        self.assertLess(start + 80, 609)

    def test_short_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            centered_even_block_start(80)

    def test_member_frame_id(self):
        self.assertEqual(
            member_frame_id("image_lcam_front/000123_lcam_front.png"),
            123,
        )
        self.assertIsNone(member_frame_id("directory/"))

    def test_nullable_label_is_height_first_and_preserves_unknown(self):
        known = np.zeros((6, 6, 3), dtype=bool)
        risk = np.zeros((6, 6, 3), dtype=np.float64)
        known[1, 2, 0] = True
        risk[1, 2, 0] = 0.75

        record = label_record(known, risk)

        self.assertEqual(record["known_target"][0][1][2], 1)
        self.assertEqual(
            record["risk_score_target_nullable"][0][1][2],
            0.75,
        )
        self.assertIsNone(
            record["risk_score_target_nullable"][1][1][2]
        )


if __name__ == "__main__":
    unittest.main()
