from __future__ import annotations

import unittest
from pathlib import Path

from .analyze_result import RESULT_SCHEMA, analyze
from .protocol import PROTOCOL_ID


class L10MB1AnalyzeResultTest(unittest.TestCase):
    def test_result_schema_is_bound_to_successor_protocol(self) -> None:
        self.assertEqual(RESULT_SCHEMA, "l10m_b1_matched_search_result_v2")
        self.assertIn("V2-FRESH-SUCCESSOR", PROTOCOL_ID)

    def test_completed_fresh_run_uses_strict_equivalence_rule(self) -> None:
        run = Path("artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875")
        if not run.exists():
            self.skipTest("formal local result fixture is unavailable")
        result = analyze(run)
        self.assertEqual(result["secondary"]["equivalent_seed_count"], 2)
        self.assertEqual(result["scientific_verdict"], "B1_INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
