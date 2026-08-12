from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r8_clear_runtime import run_truth_owned_fallback_canary as runner


class TruthOwnedFallbackCanaryRunnerTests(unittest.TestCase):
    def test_consumed_sparse_v1_is_admitted_as_failure(self) -> None:
        result, completion = runner._verify_v1_failure()
        self.assertFalse(result["passed"])
        self.assertEqual(result["old_occupied_reclassified_clear"], 36)
        self.assertEqual(completion["faro_payload_reads"], {"highres_depth": 133})


if __name__ == "__main__":
    unittest.main()
