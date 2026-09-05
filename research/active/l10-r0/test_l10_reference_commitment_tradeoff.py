import unittest

from l10_reference_commitment_tradeoff import analyze


def episode(identifier, baseline="CORRECT", verified="UNKNOWN", oracle=None):
    baseline = baseline if baseline == "UNKNOWN" else baseline + "_BINDING"
    verified = verified if verified == "UNKNOWN" else verified + "_BINDING"
    controls = [] if oracle is None else [{"kind": "TARGET_ORACLE", "verifier": {"accepted": oracle}}]
    return {"episode_id": identifier, "arms": {
        "FIXED_SWEEP": {"outcome": baseline, "extra_observation_count": 3},
        "TRIGGERED_ACTIVE": {"outcome": baseline, "extra_observation_count": 2},
        "TRIGGERED_VERIFIED": {"outcome": verified, "extra_observation_count": 2},
    }, "diagnostic_controls": controls}


class CommitmentTradeoffTest(unittest.TestCase):
    def result(self, episodes):
        return analyze({"episodes": episodes, "decision": "FROZEN_GATE_NOT_MET", "reference_setup_views": 4})

    def test_zero_commits_is_not_perfect_precision_or_error_reduction(self):
        value = self.result([episode("a", oracle=True), episode("b", oracle=False)])
        self.assertIsNone(value["arms"]["TRIGGERED_VERIFIED"]["commit_precision"])
        self.assertEqual(0.0, value["correct_retention"])
        self.assertIsNone(value["wrong_commit_reduction"])
        self.assertEqual("NOT_EVALUABLE_NO_BASELINE_ERRORS", value["wrong_commit_reduction_status"])
        self.assertEqual(2, value["online_views_saved_by_triggering"])
        self.assertEqual(4, value["separate_reference_setup_views"])
        self.assertEqual(1, value["lost_correct_attribution"]["target_box_not_supported"])
        self.assertEqual(1, value["lost_correct_attribution"]["target_box_supported_but_runtime_commit_rejected"])

    def test_missing_oracle_is_not_negative_support(self):
        value = self.result([episode("a")])
        self.assertEqual({"target_support_not_evaluable": 1}, value["lost_correct_attribution"])

    def test_wrong_baseline_reduction_uses_paired_error_opportunities(self):
        value = self.result([episode("a", baseline="WRONG"), episode("b", verified="CORRECT")])
        self.assertEqual(1.0, value["wrong_commit_reduction"])
        self.assertEqual(1.0, value["correct_retention"])

    def test_duplicate_episodes_cannot_inflate_denominators(self):
        with self.assertRaises(ValueError):
            self.result([episode("a"), episode("a")])


if __name__ == "__main__":
    unittest.main()
