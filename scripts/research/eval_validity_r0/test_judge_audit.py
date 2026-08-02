from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from .common import PROTOCOL_ID, sha256_json
from .judge_audit import (
    CAUSAL_TEMPORAL_EVIDENCE_WINDOW,
    EVENT_LEDGER_SCHEMA,
    JUDGE_CONTRACT_SCHEMA,
    ORACLE_SCHEMA,
    PAIR_SCHEMA,
    PRIMITIVE_OBSERVATION_POLICY,
    PRIMITIVE_POLICY_VERSION,
    REVIEW_MAP_SCHEMA,
    REVIEW_SCHEMA,
    RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
    VISIBILITY_EVIDENCE_WINDOW,
    VISIBILITY_POLICY_VERSION,
    _contract,
    run_judge_audit,
)


CATEGORIES = (
    "known_front_obstacle",
    "unknown_object",
    "roadside_nonblocking",
    "head_on_approach",
    "lateral_crossing",
    "camera_motion_only",
    "wide_corridor",
    "insufficient_evidence",
)
PRIMITIVE_FIELDS = ("visibility", "path_relation", "motion_relation", "phase", "route_certainty", "evidence_quality")


def contract() -> dict:
    return {
        "schema_version": JUDGE_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "CALIBRATION_BURNED",
        "cohort_role": "CALIBRATION_BURNED",
        "minimum_events": 8,
        "maximum_events": 12,
        "minimum_contiguous_frames": 20,
        "minimum_event_duration_ms": 1000,
        "minimum_pre_alert_duration_ms": 200,
        "minimum_active_duration_ms": 500,
        "minimum_passed_clear_duration_ms": 500,
        "required_coverage_min_counts": {category: 1 for category in CATEGORIES},
        "required_coverage_min_source_sessions": {category: 1 for category in CATEGORIES},
        "maximum_event_share_per_session": 0.2,
        "yolo_box_similarity_threshold": 0.9,
        "minimum_pair_distance_scale_similarity": 0.8,
        "minimum_pair_position_similarity": 0.8,
        "minimum_pair_visibility_similarity": 0.8,
        "minimum_counterfactual_pairs": 1,
        "maximum_counterfactual_pairs": 2,
        "boundary_tolerance_frames": 2,
        "minimum_event_consistency": 0.8,
        "minimum_boundary_consistency": 0.8,
        "minimum_primitive_consistency": 0.8,
        "minimum_primitive_field_consistency": 0.8,
        "minimum_derived_actionability_consistency": 0.8,
        "minimum_derived_clearance_consistency": 0.8,
        "minimum_primitive_to_derived_determinism": 1.0,
        "maximum_unknown_event_rate": 0.5,
        "metadata_blind_forbidden_tokens": ["blocking", "parallel", "negative", "approach", "curb", "unknown-object", "roadside"],
        "primitive_observation_policy": copy.deepcopy(PRIMITIVE_OBSERVATION_POLICY),
        "derived_actionability_policy": {
            "version": "primitive_to_actionability_v1",
            "reviewer_may_submit_action_labels": False,
            "unknown_if_any_unresolved_primitive": True,
        },
        "counterfactual_pair_policy": {
            "version": "counterfactual_selection_v2",
            "selection_stage": "AFTER_REVIEWS_SEALED",
            "yolo_role": "SELECTION_ONLY",
            "yolo_visible_to_reviewers": False,
            "yolo_used_for_truth": False,
            "primitive_labels_visible_to_pair_builder": False,
            "derived_labels_visible_to_pair_builder": False,
            "reviewed_event_phase_visible_to_pair_builder": False,
            "reviewed_motion_relation_visible_to_pair_builder": False,
            "selection_time_slot_source": "fixed_sampling_slot",
            "reviewed_event_phase_field": "reviewed_event_phase",
            "reviewed_motion_relation_field": "reviewed_motion_relation",
            "ordering_rule": "yolo_box_similarity_desc,distance_scale_similarity_desc,position_similarity_desc,visibility_similarity_desc,event_a_id_asc,event_b_id_asc,selection_time_slot_asc",
            "selection_fields": ["yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot"],
            "below_minimum_terminal": "NOT_EVALUABLE",
        },
        "discovery_arm_policy": {
            "allowed_arms": ["source_mask", "random_continuous_rgb", "motion_temporal_change", "metadata_only_normal"],
            "source_mask_arm": "source_mask",
            "independent_arms": ["random_continuous_rgb", "motion_temporal_change", "metadata_only_normal"],
            "minimum_distinct_arms_formal": 2,
            "minimum_independent_arms_formal": 1,
            "minimum_distinct_arms_calibration": 1,
        },
        "native_information_ceiling_gates": {
            "truth_mask_adapter": {"blocking_vs_nonblocking_accuracy": 0.8},
            "truth_depth": {"clearance_order_accuracy": 0.8},
            "truth_geometry": {"corridor_occupancy_accuracy": 0.8},
            "truth_trajectory": {"future_path_intersection_accuracy": 0.8},
        },
        "oracle_opportunity_required_fields": [
            "eligible_event_ids",
            "eligible_for_native_task",
            "eligible_for_system_chain",
            "required_inputs",
            "expected_improvement_dimension",
            "not_evaluable_reason",
        ],
        "shared_execution": {
            "decision_kernel_contract_id": "judge-test-kernel",
            "risk_config_id": "judge-test-risk",
            "feedback_profile": "TEST",
            "clock": "frozen-source-frame",
            "reset": "event-start",
        },
        "retrospective_comparison_policy": {
            "adjudicates_causal_truth": False,
            "gated": False,
        },
    }


def ledger() -> dict:
    conditions = {
        "known_front_obstacle": "BLOCKING_PATH",
        "unknown_object": "BLOCKING_PATH",
        "roadside_nonblocking": "NON_BLOCKING_PATH",
        "head_on_approach": "BLOCKING_PATH",
        "lateral_crossing": "NON_BLOCKING_PATH",
        "camera_motion_only": "NON_BLOCKING_PATH",
        "wide_corridor": "NON_BLOCKING_PATH",
        "insufficient_evidence": "UNKNOWN",
    }
    items = []
    for index, category in enumerate(CATEGORIES, start=1):
        items.append({
            "event_id": f"event-{index}",
            "source_session_id": f"session-{index}",
            "discovery_arm": "source_mask",
            "frame_indices": list(range(20)),
            "frame_timestamps_ms": [frame * 100 for frame in range(20)],
            "coverage": [category],
            "physical_condition": conditions[category],
            "evidence_sufficiency": "INSUFFICIENT" if category == "insufficient_evidence" else "SUFFICIENT",
            "label_provenance": {
                "truth_constructible_without_yolo": True,
                "yolo_boxes_used": False,
                "oracle_outputs_used": False,
                "model_outputs_visible": False,
            },
        })
    return {
        "schema_version": EVENT_LEDGER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "cohort_role": "CALIBRATION_BURNED",
        "status": "PRIMITIVE_REVIEW_FACTS_PENDING",
        "items": items,
    }


def review_map() -> dict:
    return {
        "schema_version": REVIEW_MAP_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "items": [{"review_item_id": f"r-{index:02d}", "parent_event_id": f"event-{index}"} for index in range(1, 9)],
    }


def _primitive_for(category: str, frame: int) -> dict[str, str]:
    if category == "insufficient_evidence":
        return {
            "visibility": "NOT_EVALUABLE",
            "path_relation": "AMBIGUOUS",
            "motion_relation": "STATIC_OR_UNCLEAR",
            "phase": "UNKNOWN",
            "route_certainty": "UNKNOWN",
            "evidence_quality": "INSUFFICIENT",
        }
    path = "BLOCKING_PATH" if category in {"known_front_obstacle", "unknown_object", "head_on_approach"} else "NON_BLOCKING_PATH"
    quality = "CAMERA_ROTATION" if category == "camera_motion_only" else "CLEAR"
    phase = "BEFORE_INTRUSION" if frame < 4 else ("CURRENT_INTRUSION" if frame < 11 else "PASSED_CLEAR")
    return {
        "visibility": "EVALUABLE",
        "path_relation": path,
        "motion_relation": "APPROACHING" if category == "head_on_approach" else ("LATERAL_PASS" if category == "lateral_crossing" else "STATIC_OR_UNCLEAR"),
        "phase": phase,
        "route_certainty": "SINGLE_PLAUSIBLE_ROUTE",
        "evidence_quality": quality,
    }


def review(role: str, view: str = "CAUSAL") -> dict:
    items = []
    for index, category in enumerate(CATEGORIES, start=1):
        observations = []
        for frame in range(20):
            primitive = _primitive_for(category, frame)
            observations.append({"frame_index": frame, **primitive})
        items.append({"review_item_id": f"r-{index:02d}", "primitive_observations": observations})
    return {
        "schema_version": REVIEW_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "reviewer_role": role,
        "view": view,
        "future_frame_access": view == "RETROSPECTIVE",
        "primitive_policy_version": PRIMITIVE_POLICY_VERSION,
        "visibility_evidence_window": VISIBILITY_EVIDENCE_WINDOW,
        "field_evidence_windows": copy.deepcopy(PRIMITIVE_OBSERVATION_POLICY["field_evidence_windows"]),
        "temporal_fields_evidence_window": CAUSAL_TEMPORAL_EVIDENCE_WINDOW if view == "CAUSAL" else RETROSPECTIVE_TEMPORAL_EVIDENCE_WINDOW,
        "sealed_before_pair_selection": True,
        "isolated_context": True,
        "metadata_blind": True,
        "other_review_visible_before_submission": False,
        "model_output_visible": False,
        "candidate_metadata_visible": False,
        "selection_reason_visible": False,
        "semantic_bucket_visible": False,
        "source_session_visible": False,
        "items": items,
    }


def pairs(reviews: list[dict]) -> dict:
    return {
        "schema_version": PAIR_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "pair_builder": {
            "stage": "AFTER_REVIEWS_SEALED",
            "review_bundle_sealed": True,
            "review_bundle_sha256": sha256_json(sorted(reviews, key=lambda review: review["reviewer_role"])),
            "yolo_role": "SELECTION_ONLY",
            "yolo_visible_to_reviewers": False,
            "yolo_used_for_truth": False,
            "primitive_labels_visible_to_pair_builder": False,
            "derived_labels_visible_to_pair_builder": False,
            "reviewed_event_phase_visible_to_pair_builder": False,
            "reviewed_motion_relation_visible_to_pair_builder": False,
            "pair_freeze_rule_version": "counterfactual_selection_v2",
            "selection_time_slot_source": "fixed_sampling_slot",
            "ordering_rule": "yolo_box_similarity_desc,distance_scale_similarity_desc,position_similarity_desc,visibility_similarity_desc,event_a_id_asc,event_b_id_asc,selection_time_slot_asc",
            "candidate_pair_universe_sha256": sha256_json({"candidate_pairs": ["event-1:event-3:5"]}),
            "candidate_pair_universe_count": 1,
            "eligible_pair_count_before_label_access": 1,
            "selection_fields": ["yolo_box_similarity", "distance_scale_similarity", "position_similarity", "visibility_similarity", "selection_time_slot"],
            "pair_count_frozen": True,
            "below_minimum_terminal": "NOT_EVALUABLE",
        },
        "items": [{
            "pair_id": "p-01",
            "pair_rank": 1,
            "event_a_id": "event-1",
            "event_b_id": "event-3",
            "yolo_box_similarity": 0.97,
            "distance_scale_similarity": 0.95,
            "position_similarity": 0.95,
            "visibility_similarity": 0.95,
            "selection_time_slot": 5,
            "comparison_frame_index_a": 5,
            "comparison_frame_index_b": 5,
        }],
    }


def oracle() -> dict:
    system_metrics = {
        "current_yolo": {"positive_hits": 2, "critical_misses": 1, "false_alert_events": 1, "cleared_positives": 2, "response_delay_frames": 3},
        "truth_mask_adapter": {"positive_hits": 3, "critical_misses": 0, "false_alert_events": 1, "cleared_positives": 3, "response_delay_frames": 2},
        "truth_depth": {"positive_hits": 3, "critical_misses": 0, "false_alert_events": 1, "cleared_positives": 3, "response_delay_frames": 2},
        "truth_geometry": {"positive_hits": 3, "critical_misses": 0, "false_alert_events": 1, "cleared_positives": 3, "response_delay_frames": 2},
        "truth_trajectory": {"positive_hits": 3, "critical_misses": 0, "false_alert_events": 1, "cleared_positives": 3, "response_delay_frames": 2},
    }
    native_metrics = {
        "truth_mask_adapter": {"blocking_vs_nonblocking_accuracy": 0.95},
        "truth_depth": {"clearance_order_accuracy": 0.95},
        "truth_geometry": {"corridor_occupancy_accuracy": 0.95},
        "truth_trajectory": {"future_path_intersection_accuracy": 0.95},
    }
    system_opportunity = {
        "eligible_event_ids": ["event-1", "event-4"],
        "eligible_for_native_task": True,
        "eligible_for_system_chain": True,
        "required_inputs": ["declared_native_input"],
        "expected_improvement_dimension": ["event_recall", "clearance"],
        "not_evaluable_reason": None,
    }
    native_opportunity = {
        "eligible_event_ids": ["event-1", "event-4"],
        "eligible_for_native_task": True,
        "eligible_for_system_chain": True,
        "required_inputs": ["declared_native_input"],
        "expected_improvement_dimension": ["native_physical_discrimination"],
        "not_evaluable_reason": None,
    }
    return {
        "schema_version": ORACLE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "shared_execution": contract()["shared_execution"],
        "system_chain": {
            arm: {"available": True, "metrics": metrics, **({"opportunity": system_opportunity} if arm != "current_yolo" else {})}
            for arm, metrics in system_metrics.items()
        },
        "native_information_ceiling": {
            arm: {
                "available": True,
                "opportunity": native_opportunity,
                "metrics": metrics,
            }
            for arm, metrics in native_metrics.items()
        },
    }


def refresh_pair_review_hash(values: list) -> None:
    values[4]["pair_builder"]["review_bundle_sha256"] = sha256_json(sorted(values[3], key=lambda review: review["reviewer_role"]))


class JudgeAuditTest(unittest.TestCase):
    def inputs(self) -> tuple[dict, dict, dict, list[dict], dict, dict]:
        reviews = [review("CAUSAL_A"), review("CAUSAL_B"), review("RETROSPECTIVE_C", "RETROSPECTIVE")]
        return contract(), ledger(), review_map(), reviews, pairs(reviews), oracle()

    def test_primitive_first_four_tests_pass_as_burned_pilot(self) -> None:
        result = run_judge_audit(*self.inputs())
        self.assertEqual("VALID_BURNED_CALIBRATION_PILOT", result["status"])
        self.assertEqual("PASS", result["tests"]["test_1_primitive_yolo_free_truth"]["status"])
        self.assertEqual("PASS", result["tests"]["test_2_physical_counterfactual"]["status"])
        self.assertEqual("PASS", result["tests"]["test_3_oracle_two_path_discrimination"]["status"])
        self.assertEqual("PASS", result["tests"]["test_4_blind_review_stability"]["status"])

    def test_stability_reports_primitive_and_derived_layers_separately(self) -> None:
        result = run_judge_audit(*self.inputs())
        stability = result["tests"]["test_4_blind_review_stability"]
        for field in ("path_relation", "motion_relation", "phase", "route_certainty", "evidence_quality"):
            self.assertIn(field, stability["primitive_field_consistency"])
            self.assertIn(field, stability["primitive_classwise_agreement"])
        self.assertIn("derived_actionability_consistency", stability)
        self.assertIn("derived_clearance_consistency", stability)
        self.assertIn("unknown_event_union_rate", stability)
        self.assertIn("unknown_event_intersection_rate", stability)
        self.assertEqual(1.0, stability["primitive_to_derived_determinism"])
        self.assertFalse(stability["retrospective_adjudicates_causal_truth"])

    def test_direct_action_label_is_rejected(self) -> None:
        values = list(self.inputs())
        values[3] = copy.deepcopy(values[3])
        values[3][0]["items"][0]["event_reminder_now"] = "YES"
        with self.assertRaisesRegex(ValueError, "direct action or metadata field is forbidden"):
            run_judge_audit(*values)

    def test_visibility_review_must_declare_current_frame_boundary(self) -> None:
        values = list(self.inputs())
        values[3] = copy.deepcopy(values[3])
        values[3][0].pop("visibility_evidence_window")
        with self.assertRaisesRegex(ValueError, "visibility evidence window mismatch"):
            run_judge_audit(*values)

    def test_visibility_contract_cannot_collapse_into_quality_field(self) -> None:
        values = list(self.inputs())
        values[0] = copy.deepcopy(values[0])
        values[0]["primitive_observation_policy"]["visibility"]["independent_of"] = ["actionability"]
        with self.assertRaisesRegex(ValueError, "visibility dependency boundary mismatch"):
            run_judge_audit(*values)

    def test_semantic_review_id_is_rejected(self) -> None:
        values = list(self.inputs())
        values[2] = copy.deepcopy(values[2])
        values[2]["items"][0]["review_item_id"] = "blocking-r-01"
        with self.assertRaisesRegex(ValueError, "review id leaks semantic metadata"):
            run_judge_audit(*values)

    def test_pair_selection_requires_sealed_blind_review_bundle(self) -> None:
        values = list(self.inputs())
        values[4] = copy.deepcopy(values[4])
        values[4]["pair_builder"]["stage"] = "BEFORE_REVIEWS_SEALED"
        with self.assertRaisesRegex(ValueError, "occurred before review sealing"):
            run_judge_audit(*values)

    def test_reviewer_phase_cannot_be_a_selection_field(self) -> None:
        value = contract()
        value["counterfactual_pair_policy"]["selection_fields"][-1] = "phase_match"
        with self.assertRaisesRegex(ValueError, "pair selection fields mismatch"):
            _contract(value)

    def test_pair_builder_cannot_see_reviewed_phase(self) -> None:
        values = list(self.inputs())
        values[4] = copy.deepcopy(values[4])
        values[4]["pair_builder"]["reviewed_event_phase_visible_to_pair_builder"] = True
        with self.assertRaisesRegex(ValueError, "forbidden visibility/role flag reviewed_event_phase_visible_to_pair_builder"):
            run_judge_audit(*values)

    def test_pair_comparison_frame_is_bound_to_selection_time_slot(self) -> None:
        values = list(self.inputs())
        values[4] = copy.deepcopy(values[4])
        values[4]["items"][0]["selection_time_slot"] = 6
        with self.assertRaisesRegex(ValueError, "comparison frame is not the frozen selection slot"):
            run_judge_audit(*values)

    def test_yolo_dependency_is_a_hard_failure(self) -> None:
        values = list(self.inputs())
        values[1] = copy.deepcopy(values[1])
        values[1]["items"][0]["label_provenance"]["yolo_boxes_used"] = True
        result = run_judge_audit(*values)
        self.assertEqual("STOP_JUDGE_AUDIT_FAILED", result["status"])
        self.assertEqual("FAIL", result["tests"]["test_1_primitive_yolo_free_truth"]["status"])

    def test_same_box_same_primitive_phase_is_a_hard_failure(self) -> None:
        values = list(self.inputs())
        values[3] = copy.deepcopy(values[3])
        for packet in values[3][:2]:
            for observation in packet["items"][0]["primitive_observations"]:
                if observation["frame_index"] == 5:
                    observation["phase"] = "BEFORE_INTRUSION"
            for observation in packet["items"][2]["primitive_observations"]:
                if observation["frame_index"] == 5:
                    observation["phase"] = "BEFORE_INTRUSION"
        refresh_pair_review_hash(values)
        result = run_judge_audit(*values)
        self.assertEqual("STOP_JUDGE_AUDIT_FAILED", result["status"])
        self.assertEqual("FAIL", result["tests"]["test_2_physical_counterfactual"]["status"])

    def test_unknown_pair_is_not_counted_as_physical_failure(self) -> None:
        values = list(self.inputs())
        values[3] = copy.deepcopy(values[3])
        for packet in values[3][:2]:
            for observation in packet["items"][2]["primitive_observations"]:
                if observation["frame_index"] == 5:
                    observation["evidence_quality"] = "INSUFFICIENT"
        refresh_pair_review_hash(values)
        result = run_judge_audit(*values)
        self.assertEqual("NOT_EVALUABLE_JUDGE_AUDIT_INPUTS", result["status"])
        self.assertEqual("NOT_EVALUABLE", result["tests"]["test_2_physical_counterfactual"]["status"])
        self.assertEqual([], result["tests"]["test_2_physical_counterfactual"]["failures"])

    def test_missing_truth_depth_is_not_negative_evidence(self) -> None:
        values = list(self.inputs())
        values[5] = copy.deepcopy(values[5])
        unavailable_opportunity = {
            "eligible_event_ids": [],
            "eligible_for_native_task": False,
            "eligible_for_system_chain": False,
            "required_inputs": [],
            "expected_improvement_dimension": [],
            "not_evaluable_reason": "no native depth sidecar",
        }
        values[5]["system_chain"]["truth_depth"] = {"available": False, "not_evaluable_reason": "no native depth sidecar", "opportunity": unavailable_opportunity}
        values[5]["native_information_ceiling"]["truth_depth"] = {"available": False, "not_evaluable_reason": "no native depth sidecar", "opportunity": unavailable_opportunity}
        result = run_judge_audit(*values)
        self.assertEqual("NOT_EVALUABLE_JUDGE_AUDIT_INPUTS", result["status"])
        self.assertEqual("NOT_EVALUABLE", result["tests"]["test_3_oracle_two_path_discrimination"]["status"])

    def test_native_pass_but_system_no_improvement_flags_stack_not_metric(self) -> None:
        values = list(self.inputs())
        values[5] = copy.deepcopy(values[5])
        for arm in ("truth_mask_adapter", "truth_depth", "truth_geometry", "truth_trajectory"):
            values[5]["system_chain"][arm]["metrics"] = values[5]["system_chain"]["current_yolo"]["metrics"].copy()
        result = run_judge_audit(*values)
        self.assertEqual("FLAG_EVALUATION_STACK_CEILING_SUSPECTED", result["status"])
        self.assertEqual("FLAG_EVALUATION_STACK_CEILING_SUSPECTED", result["tests"]["test_3_oracle_two_path_discrimination"]["status"])

    def test_empty_oracle_opportunity_is_not_accepted(self) -> None:
        values = list(self.inputs())
        values[5] = copy.deepcopy(values[5])
        values[5]["system_chain"]["truth_mask_adapter"]["opportunity"]["eligible_event_ids"] = []
        with self.assertRaisesRegex(ValueError, "available opportunity must have events"):
            run_judge_audit(*values)

    def test_discontinuous_positive_phase_fails_time_gate(self) -> None:
        values = list(self.inputs())
        values[3] = copy.deepcopy(values[3])
        for packet in values[3][:2]:
            for observation in packet["items"][0]["primitive_observations"]:
                if observation["frame_index"] in {6, 7}:
                    observation["phase"] = "BEFORE_INTRUSION"
        refresh_pair_review_hash(values)
        result = run_judge_audit(*values)
        self.assertEqual("STOP_JUDGE_AUDIT_FAILED", result["status"])
        self.assertIn("event-1", result["tests"]["test_1_primitive_yolo_free_truth"]["positive_time_coverage_failures"])

    def test_missing_boundary_counts_as_blind_disagreement(self) -> None:
        values = list(self.inputs())
        values[0] = copy.deepcopy(values[0])
        values[0]["minimum_boundary_consistency"] = 0.95
        values[3] = copy.deepcopy(values[3])
        for observation in values[3][1]["items"][0]["primitive_observations"]:
            if observation["frame_index"] >= 4:
                observation["phase"] = "BEFORE_INTRUSION"
        refresh_pair_review_hash(values)
        result = run_judge_audit(*values)
        self.assertEqual("STOP_JUDGE_AUDIT_FAILED", result["status"])
        stability = result["tests"]["test_4_blind_review_stability"]
        self.assertLess(stability["boundary_consistency"], 0.95)

    def test_out_of_range_cohort_is_held_before_interpretation(self) -> None:
        values = list(self.inputs())
        values[0] = copy.deepcopy(values[0])
        values[0]["minimum_events"] = 50
        values[0]["maximum_events"] = 50
        result = run_judge_audit(*values)
        self.assertEqual("HOLD_JUDGE_AUDIT_COHORT", result["status"])

    def test_formal_cohort_requires_an_independent_discovery_arm(self) -> None:
        values = list(self.inputs())
        values[0] = copy.deepcopy(values[0])
        values[0]["mode"] = "FORMAL_FROZEN"
        values[0]["cohort_role"] = "FORMAL_AUDIT"
        values[1] = copy.deepcopy(values[1])
        values[1]["cohort_role"] = "FORMAL_AUDIT"
        result = run_judge_audit(*values)
        self.assertEqual("HOLD_JUDGE_AUDIT_COHORT", result["status"])
        self.assertFalse(result["coverage"]["discovery_mix_passed"])

    def test_mode_and_cohort_role_must_match(self) -> None:
        value = contract()
        value["cohort_role"] = "FORMAL_AUDIT"
        with self.assertRaisesRegex(ValueError, "mode/cohort role mismatch"):
            _contract(value)

    def test_checked_in_formal_contract_is_primitive_first(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        path = repository / "docs" / "research" / "dual-loop" / "JUDGE_AUDIT_R0_CONTRACT_2026-08-02.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        _contract(value)
        self.assertEqual("FORMAL_FROZEN", value["mode"])
        self.assertFalse(value["derived_actionability_policy"]["reviewer_may_submit_action_labels"])
        self.assertEqual(8, value["minimum_counterfactual_pairs"])
        self.assertEqual(0.8, value["minimum_pair_position_similarity"])
        self.assertFalse(value["retrospective_comparison_policy"]["adjudicates_causal_truth"])


if __name__ == "__main__":
    unittest.main()
