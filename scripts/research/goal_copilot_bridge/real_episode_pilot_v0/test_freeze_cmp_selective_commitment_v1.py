import unittest

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.freeze_cmp_selective_commitment_v1 import (
    assign_splits,
)


class CmpSelectiveCommitmentFreezeTest(unittest.TestCase):
    def test_assign_splits_is_deterministic_and_disjoint(self) -> None:
        candidates = [
            {"rgb_sha256": f"rgb-{index:03d}", "fresh_rank_sha256": f"{121 - index:03d}"}
            for index in range(122)
        ]

        first = assign_splits(candidates)
        second = assign_splits(list(reversed(candidates)))

        self.assertEqual(first, second)
        self.assertEqual([len(first[name]) for name in ("development", "confirmation", "reserve")], [32, 64, 26])
        hashes = [item["rgb_sha256"] for rows in first.values() for item in rows]
        self.assertEqual(len(hashes), len(set(hashes)))


if __name__ == "__main__":
    unittest.main()
