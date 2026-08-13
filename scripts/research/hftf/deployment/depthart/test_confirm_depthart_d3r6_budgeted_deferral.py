from __future__ import annotations

import unittest

from scripts.research.hftf.deployment.depthart.confirm_depthart_d3r6_budgeted_deferral import (
    next_fresh_plan,
)


class NextFreshPlanTest(unittest.TestCase):
    def test_rejects_prior_roster_drift_before_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior model-output roster drift"):
            next_fresh_plan(
                {"processed": []},
                {"processed": []},
                {"processed_extension": []},
                {"candidate_role_split": {"TRAIN": {"identities": []}, "DEVELOPMENT": {"identities": []}}},
                {"dataset": {"identities": []}},
            )


if __name__ == "__main__":
    unittest.main()
