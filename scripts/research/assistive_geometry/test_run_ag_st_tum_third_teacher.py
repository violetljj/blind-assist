import unittest

import numpy as np

from plan_ag_st_tum_third_teacher_cohort import parent_roles
from run_ag_st_tum_third_teacher import (
    combine_third_teacher_quality,
    heldout_promotes,
    select_variant_on_fit,
)


def _evaluation(coverage: float, accepted_mae: float, rejected_mae: float) -> dict:
    return {
        "overall": {
            "coverage": coverage,
            "accepted": {"count": 100, "mae_m": accepted_mae},
            "rejected": {"count": 100, "mae_m": rejected_mae},
        },
        "parents": [{"parent_id": str(index)} for index in range(3)],
        "evaluable_parent_count": 3,
        "accepted_lower_risk_parent_count": 3,
    }


class TumThirdTeacherTest(unittest.TestCase):
    def test_new_parent_roles_are_four_plus_three(self) -> None:
        roles = parent_roles()
        self.assertEqual(sum(value == "fit" for value in roles.values()), 4)
        self.assertEqual(sum(value == "evaluation" for value in roles.values()), 3)
        self.assertEqual(len(roles), 7)

    def test_union_expands_and_consensus_contracts(self) -> None:
        geometry = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        two = np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        two_valid = np.asarray([True, True, False])
        third_pair = np.asarray([0.25, 0.81, 0.64], dtype=np.float32)
        third_valid = np.asarray([True, True, True])
        variants = combine_third_teacher_quality(
            geometry, two, two_valid, third_pair, third_valid
        )
        union, union_valid = variants["three_union"]
        consensus, consensus_valid = variants["three_consensus"]
        np.testing.assert_allclose(union, [0.8, 0.9, 0.8], atol=1e-6)
        np.testing.assert_array_equal(union_valid, [True, True, True])
        np.testing.assert_allclose(consensus[:2], np.sqrt([0.4, 0.18]), atol=1e-6)
        np.testing.assert_array_equal(consensus_valid, [True, True, False])
        self.assertEqual(float(consensus[2]), 0.0)

    def test_fit_selects_union_only_when_no_regret(self) -> None:
        evaluations = {
            "two_teacher": _evaluation(0.50, 0.08, 0.25),
            "three_union": _evaluation(0.60, 0.07, 0.24),
            "three_consensus": _evaluation(0.30, 0.06, 0.30),
        }
        self.assertEqual(select_variant_on_fit(evaluations)["selected"], "three_union")
        evaluations["three_union"] = _evaluation(0.60, 0.09, 0.24)
        self.assertEqual(
            select_variant_on_fit(evaluations)["selected"], "three_consensus"
        )

    def test_heldout_rejects_post_fit_regression(self) -> None:
        evaluations = {
            "two_teacher": _evaluation(0.50, 0.08, 0.25),
            "three_union": _evaluation(0.60, 0.09, 0.24),
            "three_consensus": _evaluation(0.30, 0.07, 0.30),
        }
        self.assertFalse(heldout_promotes("three_union", evaluations))
        self.assertTrue(heldout_promotes("three_consensus", evaluations))


if __name__ == "__main__":
    unittest.main()
