import unittest

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d1_arkit_roster import (
    select,
)


class PlanDepthArtTaskPreservingD1ArkitRosterTest(unittest.TestCase):
    def rows(self) -> list[dict[str, str]]:
        return [
            {"video_id": f"{41000000 + index}", "visit_id": f"{380000 + index}", "fold": fold}
            for index, fold in enumerate(
                ("Training", "Training", "Validation", "Training", "Training", "Training")
            )
        ]

    def test_selects_training_primary_and_reserve_without_excluded_identity(self) -> None:
        rows = self.rows()
        excluded = {rows[0]["visit_id"]}
        primary, reserve = select(rows, excluded, primary_count=2, reserve_count=1)
        selected = primary + reserve
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(reserve), 1)
        self.assertTrue(all(row["fold"] == "Training" for row in selected))
        self.assertNotIn(rows[0]["visit_id"], {row["visit_id"] for row in selected})
        self.assertEqual(len({row["visit_id"] for row in selected}), 3)

    def test_selection_is_deterministic(self) -> None:
        rows = self.rows()
        first = select(rows, set(), primary_count=2, reserve_count=2)
        second = select(reversed(rows), set(), primary_count=2, reserve_count=2)
        self.assertEqual(first, second)

    def test_fails_when_unique_training_visits_are_insufficient(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique eligible visits"):
            select(self.rows(), set(), primary_count=5, reserve_count=1)


if __name__ == "__main__":
    unittest.main()
