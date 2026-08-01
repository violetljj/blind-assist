from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_r4_obstacle_reference_comparison as module
from run_r4_obstacle_reference_comparison import (
    GAIN_STOP,
    GAIN_SUPPORTED,
    SOURCE_NOT_EVALUABLE,
    _terminal,
)


class R4ObstacleReferenceComparisonTest(unittest.TestCase):
    def test_ordered_terminals(self) -> None:
        self.assertEqual(SOURCE_NOT_EVALUABLE, _terminal(False, True))
        self.assertEqual(GAIN_STOP, _terminal(True, False))
        self.assertEqual(GAIN_SUPPORTED, _terminal(True, True))

    def test_runner_has_no_ground_component(self) -> None:
        source = inspect.getsource(module)
        self.assertNotIn("_ground_support", source)
        self.assertNotIn("_session_ground_comparison", source)


if __name__ == "__main__":
    unittest.main()
