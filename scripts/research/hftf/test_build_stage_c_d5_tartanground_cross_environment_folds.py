import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_stage_c_d5_tartanground_cross_environment_folds import (
    assign_folds,
    family,
)


class TartanGroundCrossEnvironmentFoldsTest(unittest.TestCase):
    def test_day_and_night_watermill_stay_in_same_fold(self):
        assignments = assign_folds(
            [
                "WaterMillDay",
                "WaterMillNight",
                "A",
                "B",
                "C",
                "D",
            ]
        )

        self.assertEqual(
            assignments["WaterMillDay"],
            assignments["WaterMillNight"],
        )
        self.assertEqual(family("WaterMillDay"), "WaterMill")

    def test_assignments_are_deterministic_and_cover_input(self):
        environments = [f"E{index}" for index in range(12)]

        first = assign_folds(environments)
        second = assign_folds(list(reversed(environments)))

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(environments))
        counts = [
            sum(value == fold for value in first.values())
            for fold in range(3)
        ]
        self.assertEqual(counts, [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
