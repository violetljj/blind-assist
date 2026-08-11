import unittest

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d2_support_pool import select


class D2SupportPoolPlannerTest(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_excluded(self) -> None:
        rows = [
            {"visit_id": "100001", "video_id": "200001", "fold": "Training"},
            {"visit_id": "100001", "video_id": "200002", "fold": "Training"},
            {"visit_id": "100002", "video_id": "200003", "fold": "Training"},
            {"visit_id": "100003", "video_id": "200004", "fold": "Validation"},
            {"visit_id": "100004", "video_id": "200005", "fold": "Training"},
        ]
        selected = select(rows, {"200003"}, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["visit_id"] for row in selected}), 2)
        self.assertNotIn("200003", {row["video_id"] for row in selected})
        self.assertTrue(all(row["role"] == "D2_SOURCE_SUPPORT_POOL_ONLY" for row in selected))

    def test_selection_fails_when_capacity_is_insufficient(self) -> None:
        rows = [{"visit_id": "100001", "video_id": "200001", "fold": "Training"}]
        with self.assertRaisesRegex(ValueError, "only 1 unique eligible visits"):
            select(rows, set(), 2)


if __name__ == "__main__":
    unittest.main()
