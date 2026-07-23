from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import validate_evidence_maturity_v2 as validator


class EvidenceMaturityV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = cls.repo / "configs/ustrf_route_target_evidence_maturity_v2.json"
        cls.standard = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_exact_standard_passes(self) -> None:
        result = validator.validate_standard(self.repo, self.standard)
        self.assertEqual(result["decision"], "VALID_EVIDENCE_MATURITY_STANDARD_V2")
        self.assertEqual(result["r1_decision_preserved"], "DATA_BLOCKED_STOP_SOURCE_SEARCH")
        self.assertEqual(result["current_authority"], "L0_ENGINEERING_DIAGNOSTIC")
        self.assertFalse(result["r2_candidate_execution_authority"])

    def test_r1_failure_cannot_be_rewritten(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["r1_preservation"]["r1_failure_must_not_be_rewritten_as_r2_pass"] = False
        with self.assertRaisesRegex(RuntimeError, "R1 preservation weakened"):
            validator.validate_standard(self.repo, changed)

    def test_recall_does_not_require_terminal_clear(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["metric_evaluability"]["event_recall"]["terminal_clear_required"] = True
        with self.assertRaisesRegex(RuntimeError, "metric contract drifted: event_recall"):
            validator.validate_standard(self.repo, changed)

    def test_empty_critical_denominator_cannot_be_zero_miss(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["metric_evaluability"]["critical_miss"][
            "empty_critical_denominator_is_not_zero_miss"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "metric contract drifted: critical_miss"):
            validator.validate_standard(self.repo, changed)

    def test_right_censored_clearance_cannot_be_imputed(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["censoring_policy"]["censored_never_imputed_as_success_or_zero_latency"] = False
        with self.assertRaisesRegex(RuntimeError, "censoring imputation opened"):
            validator.validate_standard(self.repo, changed)

    def test_pre_clear_events_cannot_enter_clearance_censor_denominator(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["censoring_policy"]["clearance_censor_fraction_denominator"] = (
            "all_events_with_truth_clear_or_post_alertable_clearance_observation_started"
        )
        with self.assertRaisesRegex(RuntimeError, "censor fraction denominator drifted"):
            validator.validate_standard(self.repo, changed)

    def test_provenance_family_cannot_be_randomly_bootstrapped_with_two_families(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["confidence_policy"]["cluster_bootstrap"][
            "provenance_family_random_resampling_enabled"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "cluster bootstrap contract drifted"):
            validator.validate_standard(self.repo, changed)

    def test_l1_cannot_select_winner(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][1]["candidate_winner_allowed"] = True
        with self.assertRaisesRegex(RuntimeError, "L1 winner opened"):
            validator.validate_standard(self.repo, changed)

    def test_powered_support_cannot_bypass_confidence_bound(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["status_taxonomy"][
            "evaluable_powered_never_implies_bound_sufficient_or_gate_pass"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "powered support can bypass"):
            validator.validate_standard(self.repo, changed)

    def test_performance_threshold_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["threshold_profile"]["event_recall_min"] = 0.8
        with self.assertRaisesRegex(RuntimeError, "R1 performance threshold drifted"):
            validator.validate_standard(self.repo, changed)

    def test_opened_r1_data_cannot_enter_confirmation(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["existing_data_disposition"][0]["maximum_level"] = "L3_OFFLINE_CONFIRMATION"
        with self.assertRaisesRegex(RuntimeError, "seen R1 data authority drifted"):
            validator.validate_standard(self.repo, changed)

    def test_source_search_budget_cannot_be_unbounded(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["source_acquisition_budget"]["maximum_new_source_families_per_round"] = 99
        with self.assertRaisesRegex(RuntimeError, "source family budget is unbounded"):
            validator.validate_standard(self.repo, changed)

    def test_l3_requires_sufficient_critical_bound(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][3]["minimum_critical_events_for_zero_miss_rate_bound"] = 5
        with self.assertRaisesRegex(RuntimeError, "L3 critical confidence-bound floor weakened"):
            validator.validate_standard(self.repo, changed)

    def test_metric_denominator_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["metric_evaluability"]["event_recall"]["denominator"] = "all_frames"
        with self.assertRaisesRegex(RuntimeError, "metric contract drifted: event_recall"):
            validator.validate_standard(self.repo, changed)

    def test_adaptation_rule_cannot_be_deleted(self) -> None:
        changed = copy.deepcopy(self.standard)
        del changed["adaptation_policy"]["between_round_change_requires_new_protocol_version"]
        with self.assertRaisesRegex(RuntimeError, "adaptation rule roster drifted"):
            validator.validate_standard(self.repo, changed)

    def test_l3_provenance_family_floor_cannot_be_one(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][3]["minimum_provenance_families"] = 1
        with self.assertRaisesRegex(RuntimeError, "L3 provenance-family floor weakened"):
            validator.validate_standard(self.repo, changed)

    def test_l4_canvas_parity_requirement_cannot_be_deleted(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][4]["entry_requirements"].remove(
            "android_canvas_raw_tensor_parity"
        )
        with self.assertRaisesRegex(RuntimeError, "L4 entry requirements drifted"):
            validator.validate_standard(self.repo, changed)

    def test_rejected_source_cannot_gain_l1_authority(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["existing_data_disposition"][2]["maximum_level"] = (
            "L1_EXPLORATORY_METRIC_PROFILE"
        )
        with self.assertRaisesRegex(RuntimeError, "rejected R1 sources gained exploratory authority"):
            validator.validate_standard(self.repo, changed)

    def test_partial_support_counts_are_evidence_bound(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["existing_data_disposition"][1]["known_partial_support"][
            "complete_terminal_clear_events"
        ] = 200
        with self.assertRaisesRegex(RuntimeError, "R1 complete-clear support count drifted"):
            validator.validate_standard(self.repo, changed)

    def test_current_authority_cannot_jump_to_selection(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["current_authority"]["highest_authorized_level"] = "L2_CANDIDATE_SELECTION"
        with self.assertRaisesRegex(RuntimeError, "current authority overclaimed"):
            validator.validate_standard(self.repo, changed)

    def test_l2_family_concentration_cap_is_locked(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][2]["maximum_single_family_share"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "L2 family concentration cap weakened"):
            validator.validate_standard(self.repo, changed)

    def test_l2_relative_claim_pair_floor_is_locked(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][2]["minimum_matched_pairs_if_relative_claim"] = 0
        with self.assertRaisesRegex(RuntimeError, "L2 relative-claim matched-pair floor weakened"):
            validator.validate_standard(self.repo, changed)

    def test_l2_selection_is_one_shot(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][2]["selection_runs_per_candidate"] = 99
        with self.assertRaisesRegex(RuntimeError, "L2 one-shot selection contract drifted"):
            validator.validate_standard(self.repo, changed)

    def test_l2_cannot_claim_safety_effectiveness(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][2]["safety_effectiveness_claim_allowed"] = True
        with self.assertRaisesRegex(RuntimeError, "L2 gained safety-effectiveness authority"):
            validator.validate_standard(self.repo, changed)

    def test_l3_cluster_ci_cannot_be_disabled(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][3]["cluster_confidence_intervals_required"] = False
        with self.assertRaisesRegex(RuntimeError, "L3 cluster confidence intervals disabled"):
            validator.validate_standard(self.repo, changed)

    def test_l3_cannot_claim_human_safety(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][3]["human_safety_claim_allowed"] = True
        with self.assertRaisesRegex(RuntimeError, "L3 gained human-safety authority"):
            validator.validate_standard(self.repo, changed)

    def test_l4_cannot_bypass_separate_production_review(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["maturity_levels"][4][
            "production_promotion_requires_separate_protocol_and_review"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "L4 bypasses separate production review"):
            validator.validate_standard(self.repo, changed)

    def test_source_budget_override_requires_blind_preregistration(self) -> None:
        changed = copy.deepcopy(self.standard)
        changed["source_acquisition_budget"][
            "budget_override_requires_new_candidate_blind_preregistration"
        ] = False
        with self.assertRaisesRegex(
            RuntimeError,
            "source budget override bypasses candidate-blind preregistration",
        ):
            validator.validate_standard(self.repo, changed)


if __name__ == "__main__":
    unittest.main()
