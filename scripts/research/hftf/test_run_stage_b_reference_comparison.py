from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_b_reference_comparison import (
    GROUND_NOT_EVALUABLE,
    GROUND_STOP,
    OBSTACLE_STOP,
    SOURCE_NOT_EVALUABLE,
    SUPPORTED,
    _decide_terminal,
)


class StageBReferenceComparisonTest(unittest.TestCase):
    def test_ordered_terminal_source_first(self) -> None:
        self.assertEqual(
            SOURCE_NOT_EVALUABLE,
            _decide_terminal(False, True, 3, True),
        )

    def test_ordered_terminal_obstacle_second(self) -> None:
        self.assertEqual(
            OBSTACLE_STOP,
            _decide_terminal(True, False, 3, True),
        )

    def test_no_ground_opportunity_is_partial(self) -> None:
        self.assertEqual(
            GROUND_NOT_EVALUABLE,
            _decide_terminal(True, True, 0, False),
        )

    def test_ground_failure_stops_full_stage_b(self) -> None:
        self.assertEqual(
            GROUND_STOP,
            _decide_terminal(True, True, 2, False),
        )

    def test_full_stage_b_requires_every_gate(self) -> None:
        self.assertEqual(
            SUPPORTED,
            _decide_terminal(True, True, 2, True),
        )


if __name__ == "__main__":
    unittest.main()
