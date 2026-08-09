import unittest

from scripts.research.assistive_geometry.plan_b0_arkitscenes_rosters import (
    select_rosters,
)


class PlanB0ArkitScenesRostersTest(unittest.TestCase):
    def rows(self) -> list[dict[str, str]]:
        rows = [
            {
                "visit_id": str(100000 + index),
                "video_id": str(200000 + index),
                "fold": "Training",
            }
            for index in range(12)
        ]
        rows.extend(
            {
                "visit_id": str(300000 + index),
                "video_id": str(400000 + index),
                "fold": "Validation",
            }
            for index in range(6)
        )
        return rows

    def test_roles_are_deterministic_disjoint_and_excluded(self) -> None:
        specs = (("TRAIN", "Training", 4), ("DEVELOPMENT", "Training", 3), ("CONFIRMATION", "Validation", 2))
        first = select_rosters(self.rows(), {"100001", "400001"}, specs)
        second = select_rosters(reversed(self.rows()), {"100001", "400001"}, specs)
        self.assertEqual(first, second)
        all_rows = [row for rows in first.values() for row in rows]
        self.assertEqual(len({row["visit_id"] for row in all_rows}), 9)
        self.assertEqual(len({row["video_id"] for row in all_rows}), 9)
        self.assertNotIn("100001", {row["visit_id"] for row in all_rows})
        self.assertNotIn("400001", {row["video_id"] for row in all_rows})

    def test_insufficient_unique_visits_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT"):
            select_rosters(
                self.rows(),
                set(),
                (("TRAIN", "Training", 10), ("DEVELOPMENT", "Training", 4)),
            )

if __name__ == "__main__":
    unittest.main()
