from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_yoloe_tiled_rescue import deduplicate, two_by_two_tiles


class Pa1MechanicsTest(unittest.TestCase):
    def test_two_by_two_overlap_geometry(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tiles = two_by_two_tiles(image)
        self.assertEqual([(112, 56), (112, 56), (112, 56), (112, 56)], [(tile.shape[1], tile.shape[0]) for tile, _, _ in tiles])
        self.assertEqual([(0, 0), (88, 0), (0, 44), (88, 44)], [(x, y) for _, x, y in tiles])

    def test_global_dedup_is_score_ordered_and_audited(self) -> None:
        candidates = [
            {"candidate_id": "low", "proposal_score": 0.2, "bbox_xyxy": [0, 0, 10, 10]},
            {"candidate_id": "high", "proposal_score": 0.9, "bbox_xyxy": [0, 0, 10, 10]},
            {"candidate_id": "other", "proposal_score": 0.1, "bbox_xyxy": [20, 20, 30, 30]},
        ]
        kept, decisions = deduplicate(candidates)
        self.assertEqual(["high", "other"], [row["candidate_id"] for row in kept])
        decision = {row["candidate_id"]: row for row in decisions}
        self.assertEqual("high", decision["low"]["suppressed_by_candidate_id"])
        self.assertTrue(decision["other"]["retained"])


if __name__ == "__main__":
    unittest.main()
