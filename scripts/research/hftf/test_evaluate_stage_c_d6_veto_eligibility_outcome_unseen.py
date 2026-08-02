import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d6_veto_eligibility_outcome_unseen import (
    mean_metric,
    paired_environment_metric,
)


class VetoEligibilityOutcomeUnseenTest(unittest.TestCase):
    def test_mean_metric_is_environment_macro(self):
        rows = {
            "large": {"candidate": {"auroc": 0.9}},
            "small": {"candidate": {"auroc": 0.5}},
        }

        result = mean_metric(rows, "candidate", "auroc")

        self.assertAlmostEqual(result, 0.7)

    def test_mean_metric_rejects_missing_class_metric(self):
        rows = {"only": {"candidate": {"auroc": None}}}

        with self.assertRaises(ValueError):
            mean_metric(rows, "candidate", "auroc")

    def test_paired_metric_excludes_same_undefined_environment(self):
        rows = {
            "usable": {
                "candidate": {"auroc": 0.7},
                "baseline_inverse_risk_confidence": {"auroc": 0.6},
            },
            "one_class": {
                "candidate": {"auroc": None},
                "baseline_inverse_risk_confidence": {"auroc": None},
            },
        }

        result = paired_environment_metric(rows, "auroc")

        self.assertAlmostEqual(result["delta"], 0.1)
        self.assertEqual(
            result["evaluable_environments"],
            ["usable"],
        )
        self.assertEqual(
            result["not_evaluable_environments"],
            ["one_class"],
        )


if __name__ == "__main__":
    unittest.main()
