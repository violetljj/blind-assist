import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("dual_loop_depth_evaluate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluateHelpersTest(unittest.TestCase):
    def test_rank_uses_average_for_ties(self) -> None:
        self.assertEqual(MODULE.rank([3.0, 1.0, 1.0, 2.0]), [3.0, 0.5, 0.5, 2.0])

    def test_spearman_preserves_and_reverses_monotonic_order(self) -> None:
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(MODULE.spearman([1, 2, 3], [6, 5, 4]), -1.0)

    def test_key_is_stable_at_frozen_area_precision(self) -> None:
        self.assertEqual(MODULE.key(7, 0.12345678901234), (7, "0.123456789012"))


if __name__ == "__main__":
    unittest.main()
