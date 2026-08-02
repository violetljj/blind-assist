import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_stage_c_d6_veto_eligibility_ranking import macro_metric


class VetoEligibilityRankingSummaryTest(unittest.TestCase):
    def test_macro_metric_weights_environments_equally(self):
        environments = {
            "large": {"candidate": {"auroc": 0.9}},
            "small": {"candidate": {"auroc": 0.5}},
        }

        self.assertAlmostEqual(
            macro_metric(environments, "candidate", "auroc"),
            0.7,
        )


if __name__ == "__main__":
    unittest.main()
