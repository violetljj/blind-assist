import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.plan_depthart_task_preserving_d3r1_fresh_metadata_roster import (
    ROLE,
    binomial_tail_probability,
    load_python_pool_identity_ids,
    select,
)


class D3R1FreshMetadataRosterPlannerTest(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_excludes_both_ids(self) -> None:
        rows = [
            {"visit_id": "100001", "video_id": "200001", "fold": "Training"},
            {"visit_id": "100001", "video_id": "200002", "fold": "Training"},
            {"visit_id": "100002", "video_id": "200003", "fold": "Training"},
            {"visit_id": "100003", "video_id": "200004", "fold": "Validation"},
            {"visit_id": "100004", "video_id": "200005", "fold": "Training"},
            {"visit_id": "100005", "video_id": "200006", "fold": "Training"},
        ]
        first = select(rows, {"100002", "200005"}, 2)
        second = select(reversed(rows), {"100002", "200005"}, 2)
        self.assertEqual(first, second)
        self.assertEqual(2, len({row["visit_id"] for row in first}))
        self.assertTrue(all(row["role"] == ROLE for row in first))
        selected_ids = {
            value for row in first for value in (row["visit_id"], row["video_id"])
        }
        self.assertFalse(selected_ids & {"100002", "200005"})

    def test_selection_fails_when_capacity_is_insufficient(self) -> None:
        rows = [
            {"visit_id": "100001", "video_id": "200001", "fold": "Training"}
        ]
        with self.assertRaisesRegex(ValueError, "only 1 unique eligible visits"):
            select(rows, set(), 2)

    def test_127_is_minimal_for_frozen_planning_heuristic(self) -> None:
        lower_bound = 0.3150044506435995
        self.assertLess(binomial_tail_probability(126, lower_bound, 32), 0.95)
        self.assertAlmostEqual(
            0.9502686917714296,
            binomial_tail_probability(127, lower_bound, 32),
            places=12,
        )

    def test_literal_concurrent_pool_is_read_without_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.py"
            path.write_text(
                "raise RuntimeError('must not execute')\n"
                f"EXPECTED_POOL = ((\"100001\", \"200001\", \"{'A' * 64}\"), "
                f"(\"100002\", \"200002\", \"{'B' * 64}\"))\n",
                encoding="utf-8",
            )
            self.assertEqual(
                {"100001", "200001", "100002", "200002"},
                load_python_pool_identity_ids(
                    path,
                    "EXPECTED_POOL",
                    2,
                    "0E42C51E65A676438DEB8EA866BAF9B6CB85899F46019633A3D7DB1158BBA0B3",
                ),
            )


if __name__ == "__main__":
    unittest.main()
