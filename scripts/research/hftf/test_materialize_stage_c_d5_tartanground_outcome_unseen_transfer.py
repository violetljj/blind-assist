import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d5_tartanground_outcome_unseen_transfer import (
    REQUIRED_ARCHIVES,
    select_parents,
)


class TartanGroundOutcomeUnseenTransferTest(unittest.TestCase):
    def test_selection_excludes_used_and_is_hash_deterministic(self):
        def row(environment, trajectory="P1000", complete=True):
            archives = (
                {name: {} for name in REQUIRED_ARCHIVES}
                if complete
                else {"metadata.zip": {}}
            )
            return {
                "environment": environment,
                "parent_id": f"{environment}/Data_diff/{trajectory}",
                "trajectory_id": trajectory,
                "archive_urls": archives,
            }

        archive_map = {
            "parents": [
                row("Used"),
                row("A"),
                row("B"),
                row("C"),
                row("D"),
                row("WrongTrajectory", "P1001"),
                row("Incomplete", complete=False),
            ]
        }

        first = select_parents(archive_map, {"Used"}, count=3)
        second = select_parents(archive_map, {"Used"}, count=3)

        self.assertEqual(first, second)
        self.assertEqual(3, len(first))
        self.assertNotIn(
            "Used",
            {item["environment"] for item in first},
        )
        self.assertNotIn(
            "WrongTrajectory",
            {item["environment"] for item in first},
        )
        self.assertNotIn(
            "Incomplete",
            {item["environment"] for item in first},
        )


if __name__ == "__main__":
    unittest.main()
