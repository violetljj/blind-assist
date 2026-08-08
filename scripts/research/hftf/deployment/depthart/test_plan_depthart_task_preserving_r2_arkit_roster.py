import unittest

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_r2_arkit_roster import (
    select,
)


class PlanDepthArtTaskPreservingR2ArkitRosterTest(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_excluded(self) -> None:
        rows = [
            {"visit_id": str(100000 + index // 2), "video_id": str(200000 + index), "fold": "Validation"}
            for index in range(12)
        ]
        rows.append({"visit_id": "999999", "video_id": "888888", "fold": "Training"})
        first = select(rows, {"100001", "200008"}, 4)
        second = select(reversed(rows), {"100001", "200008"}, 4)
        self.assertEqual(first, second)
        self.assertEqual(len({row["visit_id"] for row in first}), 4)
        self.assertNotIn("100001", {row["visit_id"] for row in first})
        self.assertNotIn("200008", {row["video_id"] for row in first})

    def test_insufficient_unique_visits_fails(self) -> None:
        rows = [{"visit_id": "100000", "video_id": "200000", "fold": "Validation"}]
        with self.assertRaisesRegex(ValueError, "only 1"):
            select(rows, set(), 2)


if __name__ == "__main__":
    unittest.main()
