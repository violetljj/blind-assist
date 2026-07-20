import unittest

import build_public_video_bangkok_matched_counterfactual_pair as subject


class BangkokMatchedCounterfactualPairTest(unittest.TestCase):
    def test_episode_rows_are_inclusive_and_parent_bound(self) -> None:
        rows = subject.episode_rows("pair", "source", "safe_lateral", 0, 300000, 302000)
        self.assertEqual([300000, 301000, 302000], [row["timestamp_ms"] for row in rows])
        self.assertTrue(all(row["parent_source_id"] == "source" for row in rows))
        self.assertTrue(all(row["provisional_binary_label"] == 0 for row in rows))

    def test_episode_rows_reject_non_second_boundary(self) -> None:
        with self.assertRaises(ValueError):
            subject.episode_rows("pair", "source", "risk", 1, 328001, 339000)


if __name__ == "__main__":
    unittest.main()
