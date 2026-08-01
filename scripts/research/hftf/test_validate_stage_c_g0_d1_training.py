from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_stage_c_g0_d1_training import (  # noqa: E402
    CORPUS_VALIDATION_CHECKS,
    _arm_directory,
    _corpus_checks_pass,
    _expected_run_roots,
    _selection_key,
)


class StageCG0D1TrainingValidationTest(unittest.TestCase):
    def test_arm_directory_is_canonical(self) -> None:
        self.assertEqual(
            "signed-clearance-current",
            _arm_directory("SIGNED_CLEARANCE_CURRENT"),
        )

    def test_expected_run_roots_cover_exact_twelve(self) -> None:
        roots = _expected_run_roots(Path("root"))
        self.assertEqual(12, len(roots))
        self.assertIn(
            ("phase-b", 43, "DIRECT_RISK_CURRENT"),
            roots,
        )

    def test_clearance_mae_only_breaks_exact_risk_tie(self) -> None:
        better_mae = {
            "risk_source_macro_f1": 0.5,
            "risk_worst_source_f1": 0.4,
            "risk_micro": {"f1": 0.6},
            "clearance_source_macro_mae_m": {"overall": 0.1},
        }
        worse_mae = {
            **better_mae,
            "clearance_source_macro_mae_m": {"overall": 0.2},
        }
        self.assertGreater(
            _selection_key(
                "SIGNED_CLEARANCE_CURRENT", better_mae, 2
            ),
            _selection_key(
                "SIGNED_CLEARANCE_CURRENT", worse_mae, 1
            ),
        )

    def test_direct_selection_ignores_clearance_key(self) -> None:
        metrics = {
            "risk_source_macro_f1": 0.5,
            "risk_worst_source_f1": 0.4,
            "risk_micro": {"f1": 0.6},
        }
        self.assertGreater(
            _selection_key("DIRECT_RISK_CURRENT", metrics, 1),
            _selection_key("DIRECT_RISK_CURRENT", metrics, 2),
        )

    def test_corpus_validation_checks_reject_empty_or_partial(self) -> None:
        self.assertFalse(_corpus_checks_pass({}))
        self.assertFalse(
            _corpus_checks_pass(
                {
                    key: True
                    for key in list(CORPUS_VALIDATION_CHECKS)[:-1]
                }
            )
        )
        self.assertTrue(
            _corpus_checks_pass(
                {key: True for key in CORPUS_VALIDATION_CHECKS}
            )
        )


if __name__ == "__main__":
    unittest.main()
