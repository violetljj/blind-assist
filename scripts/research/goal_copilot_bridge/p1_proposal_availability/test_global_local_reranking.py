from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_global_local_reranking import global_rank_and_nms, map_local_box


def candidate(box, score, parent_rank, local_rank=1):
    return {
        "bbox_xyxy": box, "score": score, "parent_region_rank": parent_rank,
        "local_provider_rank": local_rank, "label": "door", "parent_semantic_supported": False,
        "source": "test",
    }


class GlobalLocalRerankingTest(unittest.TestCase):
    def test_maps_crop_local_box_back_to_full_frame(self) -> None:
        self.assertEqual([110.0, 220.0, 150.0, 280.0], map_local_box([10, 20, 50, 80], [100, 200, 300, 400]))

    def test_global_score_can_promote_late_parent(self) -> None:
        rows = [candidate([0, 0, 20, 20], 0.2, 1), candidate([40, 40, 60, 60], 0.8, 9)]
        self.assertEqual(9, global_rank_and_nms(rows)[0]["parent_region_rank"])

    def test_class_agnostic_nms_removes_duplicate_local_boxes(self) -> None:
        rows = [candidate([0, 0, 100, 100], 0.9, 1), candidate([5, 5, 105, 105], 0.8, 7), candidate([200, 200, 250, 250], 0.7, 2)]
        retained = global_rank_and_nms(rows)
        self.assertEqual([1, 2], [row["parent_region_rank"] for row in retained])

    def test_ties_are_deterministic(self) -> None:
        rows = [candidate([40, 0, 50, 10], 0.5, 3), candidate([0, 0, 10, 10], 0.5, 2, 2), candidate([20, 0, 30, 10], 0.5, 2, 1)]
        retained = global_rank_and_nms(rows)
        self.assertEqual([(2, 1), (2, 2), (3, 1)], [(row["parent_region_rank"], row["local_provider_rank"]) for row in retained])


if __name__ == "__main__":
    unittest.main()
