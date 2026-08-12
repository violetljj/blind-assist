import unittest

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d3_fresh_metadata_roster import (
    select,
)


class D3FreshMetadataRosterPlannerTest(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_excluded(self) -> None:
        rows = [
            {"visit_id": "100001", "video_id": "200001", "fold": "Training"},
            {"visit_id": "100001", "video_id": "200002", "fold": "Training"},
            {"visit_id": "100002", "video_id": "200003", "fold": "Training"},
            {"visit_id": "100003", "video_id": "200004", "fold": "Validation"},
            {"visit_id": "100004", "video_id": "200005", "fold": "Training"},
            {"visit_id": "100005", "video_id": "200006", "fold": "Training"},
        ]
        first = select(rows, {"200003"}, 3)
        second = select(reversed(rows), {"200003"}, 3)
        self.assertEqual(first, second)
        self.assertEqual(len({row["visit_id"] for row in first}), 3)
        self.assertNotIn("200003", {row["video_id"] for row in first})
        self.assertTrue(
            all(row["role"] == "D3_METADATA_CANDIDATE_POOL_ONLY" for row in first)
        )

    def test_selection_excludes_visit_and_session_identity(self) -> None:
        rows = [
            {"visit_id": "100001", "video_id": "200001", "fold": "Training"},
            {"visit_id": "100002", "video_id": "200002", "fold": "Training"},
            {"visit_id": "100003", "video_id": "200003", "fold": "Training"},
        ]
        selected = select(rows, {"100001", "200002"}, 1)
        self.assertEqual(selected[0]["visit_id"], "100003")

    def test_selection_fails_when_capacity_is_insufficient(self) -> None:
        rows = [{"visit_id": "100001", "video_id": "200001", "fold": "Training"}]
        with self.assertRaisesRegex(ValueError, "only 1 unique eligible visits"):
            select(rows, set(), 2)


if __name__ == "__main__":
    unittest.main()
