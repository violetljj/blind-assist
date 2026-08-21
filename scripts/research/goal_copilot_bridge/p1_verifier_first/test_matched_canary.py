from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_verifier_first.run_matched_canary import (
    _bbox_proxy,
    adjudicate,
)


def arm(*, precision, coverage, wrong, true_reacq, false_reacq=0, multiview=0):
    return {
        "identity_precision": {"value": precision},
        "identity_coverage": {"value": coverage},
        "wrong_identity_commitments": wrong,
        "true_same_instance_reacquisition": true_reacq,
        "wrong_instance_reacquisition": false_reacq,
        "multiview_contributed_true_reacquisition": multiview,
    }


class MatchedCanaryContractTest(unittest.TestCase):
    def gates(self, *, parity=True, poisoning=True):
        return {
            "candidate_availability_exact_parity": parity,
            "memory_poisoning_events_zero": poisoning,
            "post_initialization_gt_reads_zero": True,
            "added_candidates_zero": True,
        }

    def test_signal_requires_precision_coverage_and_true_reacquisition(self):
        baseline = arm(precision=0.60, coverage=0.90, wrong=40, true_reacq=0, false_reacq=3)
        amrm = arm(precision=0.95, coverage=0.75, wrong=3, true_reacq=2, false_reacq=0, multiview=1)
        self.assertEqual(
            "P1_AMRM0_IDENTITY_VALUE_SIGNAL_ESTABLISHED",
            adjudicate(baseline, amrm, self.gates()),
        )

    def test_perfect_precision_at_five_percent_coverage_is_abstention_only(self):
        baseline = arm(precision=0.60, coverage=0.90, wrong=40, true_reacq=0)
        amrm = arm(precision=1.0, coverage=0.05, wrong=0, true_reacq=1)
        self.assertEqual(
            "P1_AMRM0_ABSTENTION_ONLY_NO_SIGNAL",
            adjudicate(baseline, amrm, self.gates()),
        )

    def test_poisoning_or_parity_failure_is_not_evaluable(self):
        baseline = arm(precision=0.60, coverage=0.90, wrong=40, true_reacq=0)
        amrm = arm(precision=0.95, coverage=0.75, wrong=3, true_reacq=2)
        self.assertEqual(
            "P1_AMRM0_MEMORY_POISONING_FAIL",
            adjudicate(baseline, amrm, self.gates(poisoning=False)),
        )

    def test_verifier_only_gain_does_not_establish_multiview_value(self):
        baseline = arm(precision=0.60, coverage=0.90, wrong=40, true_reacq=0, false_reacq=3)
        amrm = arm(precision=0.95, coverage=0.75, wrong=3, true_reacq=2, false_reacq=0, multiview=0)
        self.assertEqual(
            "P1_AMRM0_VERIFIER_ONLY_SIGNAL_NO_MULTIVIEW_VALUE",
            adjudicate(
                baseline,
                amrm,
                self.gates(),
            ),
        )

    def test_bbox_proxy_is_public_and_deterministic(self):
        initial = [0.0, 0.0, 10.0, 10.0]
        self.assertEqual(("MEDIUM", "FRONTAL"), _bbox_proxy(initial, initial))
        self.assertEqual(("LARGE", "RIGHT"), _bbox_proxy([20.0, 0.0, 40.0, 20.0], initial))
        self.assertEqual(("SMALL", "LEFT"), _bbox_proxy([-20.0, 0.0, -15.0, 5.0], initial))


if __name__ == "__main__":
    unittest.main()
