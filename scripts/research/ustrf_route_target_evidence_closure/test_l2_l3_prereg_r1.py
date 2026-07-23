from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import validate_l2_l3_prereg_r1 as validator


class L2L3PreregR1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.l2 = json.loads((cls.repo / validator.L2_CONFIG).read_text(encoding="utf-8"))
        cls.l3 = json.loads((cls.repo / validator.L3_TEMPLATE).read_text(encoding="utf-8"))
        cls.l2_schema = json.loads(
            (cls.repo / validator.L2_SCHEMA).read_text(encoding="utf-8")
        )
        cls.l3_schema = json.loads(
            (cls.repo / validator.L3_SCHEMA).read_text(encoding="utf-8")
        )

    def validate(self, l2: dict | None = None, l3: dict | None = None) -> dict:
        return validator.validate_contracts(
            self.repo,
            l2 or self.l2,
            l3 or self.l3,
            self.l2_schema,
            self.l3_schema,
        )

    def test_exact_contracts_pass_without_execution(self) -> None:
        result = self.validate()
        self.assertEqual(result["decision"], "VALID_L2_L3_PREREG_R1")
        self.assertFalse(result["new_data_or_candidate_execution"])
        self.assertFalse(result["l3_executable"])
        self.assertIsNone(result["l3_candidate_id"])

    def test_l2_must_freeze_before_candidate_outputs(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["candidate_outputs_visible_at_freeze"] = True
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_cannot_authorize_execution_now(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["execution_authorized_now"] = True
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_required_metric_cannot_be_removed(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["required_metrics"].remove("evidence_age")
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_performance_gate_cannot_be_relaxed(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["performance_gates"]["event_recall"]["threshold"] = 0.8
        with self.assertRaisesRegex(RuntimeError, "L2 performance gates drifted"):
            self.validate(l2=changed)

    def test_l2_primary_metric_is_frozen(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["selection_rule"]["primary_metric"]["metric"] = "clearance"
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_tie_break_order_is_frozen(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["selection_rule"]["tie_break_order"].reverse()
        with self.assertRaisesRegex(RuntimeError, "L2 tie-break order drifted"):
            self.validate(l2=changed)

    def test_l2_candidate_order_is_frozen(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["candidate_execution"]["candidate_order"].reverse()
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_is_one_shot(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["candidate_execution"]["runs_per_candidate"] = 2
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_total_support_floor_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["support_floors"]["totals"]["critical_events"] = 4
        with self.assertRaisesRegex(RuntimeError, "L2 total support floors drifted"):
            self.validate(l2=changed)

    def test_l2_per_family_floor_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["support_floors"]["per_family"]["negative_exposure_minutes"] = 4.9
        with self.assertRaisesRegex(RuntimeError, "L2 per-family support floors drifted"):
            self.validate(l2=changed)

    def test_l2_family_share_cap_cannot_be_relaxed(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["support_floors"]["maximum_single_family_share"] = 0.8
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_source_budget_cannot_expand(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["source_acquisition_budget"]["maximum_new_source_families"] = 3
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_source_qualification_must_be_candidate_blind(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["source_qualification"]["candidate_blind"] = False
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_download_cannot_be_claimed_in_this_stage(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["source_acquisition_budget"][
            "download_or_materialization_performed_in_this_stage"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_hard_veto_cannot_be_removed(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["hard_vetoes"].remove("unresolved_person_active_alert")
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_all_required_metrics_must_be_powered(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["support_floors"]["all_required_metrics_must_be_evaluable_powered"] = False
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_worst_source_gate_cannot_be_disabled(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["source_wise_gate"][
            "decidable_worst_source_gate_required_for_every_required_metric"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_promotion_veto_must_equal_zero(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["source_wise_gate"]["promotion_veto_count_must_equal"] = 1
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_role_cannot_use_seen_data(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["role_isolation"]["allowed_roles"] = ["fresh_selection", "seen"]
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_selection_semantics_cannot_promote_to_confirmation(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["selection_rule"][
            "allowed_selection_decision"
        ] = "OFFLINE_CONFIRMED_RESEARCH_CANDIDATE"
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l2_candidate_id_remains_unbound(self) -> None:
        changed = copy.deepcopy(self.l2)
        changed["candidate_execution"]["candidate_id"] = "C1_CAUSAL_ROUTE_RELATION_FSM"
        with self.assertRaisesRegex(RuntimeError, "L2 schema validation failed"):
            self.validate(l2=changed)

    def test_l3_template_cannot_be_executable(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["executable"] = True
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_candidate_id_must_remain_null(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["candidate_id"] = "C1_CAUSAL_ROUTE_RELATION_FSM"
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_requires_future_independent_l2_pass(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["generation_gate"]["requires_independent_l2_pass"] = False
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_cannot_claim_current_l2_pass(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["generation_gate"]["current_l2_pass_available"] = True
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_session_floor_is_exactly_six(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["sessions"] = 5
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_scenario_strata_and_loso_folds_are_frozen(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["minimum_scenario_strata"] = 4
        changed["lockbox_floors"]["loso_folds"] = 5
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_matched_pair_floor_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["complete_positive_negative_matched_pairs"] = 59
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_repeat_and_regeneration_floors_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["complete_repeat_events"] = 59
        changed["lockbox_floors"]["complete_regeneration_intervals"] = 59
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_critical_floor_cannot_be_lowered(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["minimum_critical_events"] = 58
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_family_share_cap_cannot_be_relaxed(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"]["maximum_single_family_share"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_each_metric_requires_two_families(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["lockbox_floors"][
            "each_required_metric_minimum_provenance_families"
        ] = 1
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_bootstrap_seed_and_iterations_are_frozen(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["statistical_contract"]["bootstrap_iterations"] = 9999
        changed["statistical_contract"]["bootstrap_seed"] = 1
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_worst_family_sentinel_cannot_be_disabled(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["statistical_contract"]["worst_family_sentinel_required"] = False
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_loso_worst_session_sentinel_cannot_be_disabled(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["statistical_contract"][
            "loso_worst_session_sentinel_required"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_l3_cannot_reuse_l2_selection_data(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["role_isolation"]["strictly_disjoint_from_l2_selection_data"] = False
        with self.assertRaisesRegex(RuntimeError, "L3 schema validation failed"):
            self.validate(l3=changed)

    def test_schema_rejects_unknown_root_property(self) -> None:
        changed = copy.deepcopy(self.l3)
        changed["run_now"] = True
        with self.assertRaisesRegex(RuntimeError, "additional property run_now"):
            self.validate(l3=changed)


if __name__ == "__main__":
    unittest.main()
