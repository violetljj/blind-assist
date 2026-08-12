from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r10_clear_runtime import run_top8_selection as base
from scripts.research.taro_o1r_r10_clear_runtime import run_top8_selection_r1 as recovery


class Top8SelectionR1Tests(unittest.TestCase):
    def test_r0_failure_is_exact_pre_selection_stop(self) -> None:
        recovery._verify_r0_failure()
        root = base._repo_path(recovery.R0_ROOT)
        self.assertEqual(
            {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()},
            {"execution-receipt.json", "failure.json", "manifest.json"},
        )

    def test_r1_has_fresh_identity_and_root(self) -> None:
        self.assertNotEqual(recovery.LOCK_SCHEMA, base.LOCK_SCHEMA)
        self.assertNotEqual(recovery.LOCK_ID, base.LOCK_ID)
        self.assertNotEqual(recovery.OUTPUT_ROOT, base.OUTPUT_ROOT)
        self.assertFalse(base._repo_path(recovery.OUTPUT_ROOT).exists())
        self.assertIn("R10_SELECTION_R0_FAILURE", recovery.EXPECTED_BINDINGS)
        self.assertIn("R10_SELECTION_BASE_TEST", recovery.EXPECTED_BINDINGS)

    def test_canonical_fraction_matches_round12(self) -> None:
        self.assertEqual(base._canonical_fraction(1, 9), round(1 / 9, 12))
        self.assertEqual(base._canonical_fraction(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
