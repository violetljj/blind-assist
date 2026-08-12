from __future__ import annotations

import unittest

from scripts.research.taro_o1r_r8_clear_runtime import clear_enrichment
from scripts.research.taro_o1r_r8_clear_runtime import run_top8_selection as runner


class Top8SelectionTests(unittest.TestCase):
    def test_phase_a_manifest_cardinality_is_exact(self) -> None:
        self.assertEqual(runner.PHASE_A_FILE_COUNT, 808)
        self.assertEqual(runner.QUERY_COUNT, 3618)

    def test_selector_public_api_is_truth_blind(self) -> None:
        clear_enrichment.assert_public_api_truth_blind()
        self.assertFalse(runner.EXPECTED_AUTHORITY["faro_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["truth_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["label_read"])
        self.assertFalse(runner.EXPECTED_AUTHORITY["outcome_read"])


if __name__ == "__main__":
    unittest.main()
