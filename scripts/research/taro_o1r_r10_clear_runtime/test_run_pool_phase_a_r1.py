from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as base
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a_r1 as runner


class PoolPhaseAR1Tests(unittest.TestCase):
    def test_import_does_not_mutate_r0_runtime(self) -> None:
        self.assertEqual(base.OUTPUT_ROOT, "artifacts.local/evidence/taro/o1r-r10-fresh-pool-phase-a-r0")
        self.assertEqual(base.LOCK_ID, "TARO_O1R_R10_FRESH_POOL_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK")
        self.assertNotEqual(runner.OUTPUT_ROOT, base.OUTPUT_ROOT)

    def test_exact_dependency_only_r0_failure_is_admitted(self) -> None:
        runner._verify_r0_failure()

    def test_r1_binding_set_adds_base_and_failure_lineage(self) -> None:
        self.assertNotIn("R10_PHASE_A_RUNNER", runner.EXPECTED_BINDINGS)
        self.assertIn("R10_PHASE_A_BASE_RUNTIME", runner.EXPECTED_BINDINGS)
        self.assertIn("R10_PHASE_A_R0_EXECUTION_LOCK", runner.EXPECTED_BINDINGS)
        self.assertIn("R10_PHASE_A_R0_FAILURE", runner.EXPECTED_BINDINGS)
        self.assertIn("R10_PHASE_A_R1_RUNNER", runner.EXPECTED_BINDINGS)


if __name__ == "__main__":
    unittest.main()
