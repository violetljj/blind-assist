#!/usr/bin/env python3
"""Validate the evidence-maturity-driven route-target R2 standard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_ustrf_route_target_evidence_maturity_standard_v2"
LEVEL_IDS = [
    "L0_ENGINEERING_DIAGNOSTIC",
    "L1_EXPLORATORY_METRIC_PROFILE",
    "L2_CANDIDATE_SELECTION",
    "L3_OFFLINE_CONFIRMATION",
    "L4_ANDROID_SHADOW",
]
METRIC_IDS = {
    "event_recall",
    "critical_miss",
    "repeat_within_observation",
    "clearance",
    "event_regeneration_after_clear",
    "false_alerts_per_minute",
    "evidence_age",
    "unknown_or_stale_active_alert",
}
TRUTH_INVARIANTS = {
    "truth_eligibility_windows_and_denominators_frozen_before_candidate_or_app_outputs",
    "person_identity_unique_and_continuous_over_each_scored_interval",
    "identity_ambiguity_or_mismatch_is_unknown",
    "candidate_route_uses_current_and_past_inputs_only",
    "future_route_is_annotation_only",
    "missing_occluded_stale_or_unknown_never_clears",
    "candidate_output_cannot_change_truth_quarantine_or_denominator",
    "full_sequence_single_pass_without_scoring_window_reset",
    "source_wise_results_required_and_pooled_cannot_override_source_failure",
    "unknown_route_or_unresolved_person_active_alert_is_promotion_veto",
    "android_claim_requires_canonical_canvas_raw_tensor_evidence",
}
THRESHOLD_KEYS = {
    "event_recall_min",
    "critical_miss_max",
    "false_alerts_per_minute_max",
    "clearance_rate_min",
    "clearance_p95_ms_max",
    "repeat_alert_count_max",
    "event_regeneration_count_max",
    "evidence_age_p95_ms_max",
    "active_alert_on_unknown_route_frames_max",
}
EXPECTED_METRIC_CONTRACTS = {
    "event_recall": {
        "eligible_when": [
            "same_person_identity_continuous_from_onset_through_alertable_deadline",
            "causal_route_known_through_alertable_deadline",
        ],
        "terminal_clear_required": False,
        "right_censored_after_alertable_deadline_allowed": True,
        "denominator": "eligible_alertable_truth_events",
    },
    "critical_miss": {
        "eligible_when": [
            "critical_interval_truth_observed",
            "same_person_identity_continuous_through_critical_interval",
            "causal_route_known_through_critical_interval",
        ],
        "terminal_clear_required": False,
        "empty_critical_denominator_is_not_zero_miss": True,
        "denominator": "eligible_critical_truth_events",
    },
    "repeat_within_observation": {
        "eligible_when": [
            "first_delivery_observed",
            "same_active_episode_continuously_observed_until_observation_end",
        ],
        "terminal_clear_required": False,
        "right_censored_result_label": "observed_repeat_only",
        "any_observed_repeat_is_immediate_fail": True,
        "right_censored_zero_repeat_result_status": "estimate_only",
        "complete_active_episode_required_for_pass": True,
        "descriptive_denominator": "eligible_delivered_events_observed_after_first_delivery",
        "gate_denominator": "eligible_delivered_events_with_complete_active_episode",
    },
    "clearance": {
        "eligible_when": [
            "truth_terminal_clear_observed_for_same_person",
            "post_clear_followup_available",
        ],
        "terminal_clear_required": True,
        "assessment_horizon_ms": 1500.0,
        "success_definition": "candidate_clear_latency_ms_at_most_assessment_horizon",
        "failure_definition": "full_assessment_horizon_observed_without_candidate_clear",
        "right_censored_definition": "truth_clear_observed_but_followup_shorter_than_assessment_horizon",
        "pre_clear_observation_end_state": "not_evaluable_pre_clear",
        "right_censored_before_terminal_clear_is_not_failure_or_success": True,
        "identity_loss_censoring_is_unresolved_for_promotion": True,
        "undefined_clearance_rate_lcb_or_p95_or_survival_quantile_sets_bound_sufficient_false": True,
        "denominator": "eligible_truth_clear_events",
    },
    "event_regeneration_after_clear": {
        "eligible_when": [
            "truth_terminal_clear_observed",
            "post_clear_new_onset_observation_available",
        ],
        "terminal_clear_required": True,
        "post_clear_observation_horizon_ms": 2000.0,
        "complete_followup_required_for_pass": True,
        "right_censored_zero_regeneration_result_status": "estimate_only",
        "any_observed_regeneration_is_immediate_fail": True,
        "denominator": "eligible_post_clear_observation_intervals",
    },
    "false_alerts_per_minute": {
        "eligible_when": [
            "causal_route_known",
            "all_route_relevant_person_truth_resolved",
        ],
        "terminal_clear_required": False,
        "matched_negative_window_required": False,
        "matched_negative_role": "paired_causal_comparison_only",
        "numerator": "all_full_sequence_deliveries_not_attributed_to_an_eligible_active_truth_event_and_person_including_wrong_person_adjacent_safe_receding_cleared_and_outside_matched_windows",
        "unknown_or_unattributable_delivery": "separate_promotion_veto_not_silently_dropped",
        "denominator": "full_sequence_scorable_negative_exposure_minutes",
    },
    "evidence_age": {
        "eligible_when": [
            "capture_and_consuming_timestamps_bound",
        ],
        "terminal_clear_required": False,
        "denominator": "all_replay_frames_with_bound_evidence_timestamps",
    },
    "unknown_or_stale_active_alert": {
        "eligible_when": [
            "candidate_replay_executed_with_route_validity_state",
        ],
        "terminal_clear_required": False,
        "denominator": "all_replay_frames",
        "any_positive_count_is_promotion_veto": True,
    },
}
ADAPTATION_KEYS = {
    "within_round_semantics_denominators_thresholds_and_tie_breaks_frozen",
    "between_round_change_requires_new_protocol_version",
    "change_rationale_must_not_use_candidate_scores_from_target_lockbox",
    "opened_data_loses_selection_confirmation_and_shadow_lockbox_authority_after_semantic_change",
    "outcome_unseen_materialization_or_transport_fix_may_use_hash_bound_amendment",
    "source_inclusion_may_use_candidate_blind_truth_quality_and_support_counts_only",
    "not_evaluable_must_not_be_recoded_as_zero_safe_clear_or_pass",
}
L4_ENTRY_REQUIREMENTS = {
    "android_canvas_raw_tensor_parity",
    "detector_and_decision_kernel_hash_parity",
    "clock_and_route_age_contract_pass",
    "no_user_feedback_or_production_routing",
    "device_metric_geometry_admission_for_any_metric_distance_claim",
}
L2_SELECTION_CONTRACT = {
    "each_required_metric_point_gate_must_pass": True,
    "each_required_metric_worst_source_gate_must_pass": True,
    "promotion_veto_count_must_equal": 0,
    "bound_sufficient_required": False,
    "result_status_when_any_required_bound_is_insufficient": "estimate_only",
    "gate_result_required_for_selection": "pass",
    "selection_decision": "PROVISIONAL_SELECTION_FOR_FRESH_CONFIRMATION_ONLY",
    "l2_pass_definition": "all_required_metrics_powered_point_and_worst_source_gates_pass_zero_promotion_veto",
    "l3_entry_requires": "l2_provisional_selection_plus_fresh_confirmation_lockbox",
}
SOURCE_WISE_POLICY = {
    "insufficient_source_support_is_not_source_failure": True,
    "hard_vetoes_apply_per_source_regardless_of_support": [
        "any_observed_critical_miss",
        "unknown_route_active_alert",
        "unresolved_person_active_alert",
        "truth_or_identity_invariant_violation",
    ],
    "ordinary_rate_source_pass_or_fail_requires_source_specific_floor": True,
    "underpowered_source_status": "insufficient_support",
    "aggregate_advancement_still_requires_minimum_independent_families_and_family_share_cap": True,
    "pooled_result_cannot_override_an_evaluable_source_failure_or_hard_veto": True,
}
METRIC_INFERENCE_POLICY = {
    "event_recall": "exact_binomial_plus_cluster_bound_required",
    "critical_miss": "exact_binomial_plus_cluster_bound_required",
    "false_alerts_per_minute": "poisson_working_model_plus_cluster_bound_required",
    "clearance": "exact_binomial_survival_quantile_plus_cluster_bound_required",
    "repeat_within_observation": "hard_veto_plus_complete_denominator_floor_bound_not_applicable",
    "event_regeneration_after_clear": "hard_veto_plus_complete_denominator_floor_bound_not_applicable",
    "unknown_or_stale_active_alert": "hard_veto_over_all_replay_frames_bound_not_applicable",
    "evidence_age": "point_and_worst_session_engineering_gate_bound_not_applicable",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bound(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"{label} hash mismatch")
    return path


def validate_standard(repo: Path, standard: dict[str, Any]) -> dict[str, Any]:
    require(standard.get("schema") == SCHEMA, "evidence maturity schema mismatch")
    require(
        standard.get("status") == "active_governance_only_candidates_not_run",
        "R2 status must remain governance-only before metric masks freeze",
    )

    preservation = standard["r1_preservation"]
    r1_path = resolve_bound(repo, preservation["preregistration"], "R1 preregistration")
    resolve_bound(repo, preservation["result"], "R1 result")
    r1 = load_json(r1_path)
    require(
        preservation["r1_decision_remains"] == "DATA_BLOCKED_STOP_SOURCE_SEARCH",
        "R1 stopped decision was rewritten",
    )
    for key in (
        "r1_candidate_outputs_remain_unrun",
        "r1_failure_must_not_be_rewritten_as_r2_pass",
        "r1_opened_data_may_only_be_seen_or_exploratory",
        "r1_opened_data_may_not_be_r2_selection_confirmation_or_shadow_lockbox",
    ):
        require(preservation[key] is True, f"R1 preservation weakened: {key}")

    invariants = standard["truth_invariants"]
    require(set(invariants) == TRUTH_INVARIANTS, "truth invariant roster drifted")
    require(all(invariants.values()), "truth invariant disabled")

    taxonomy = standard["status_taxonomy"]
    require(
        taxonomy["support_statuses"]
        == ["not_evaluable", "evaluable_underpowered", "evaluable_powered"],
        "support status taxonomy drifted",
    )
    require(
        taxonomy["result_statuses"] == ["not_tested", "estimate_only", "pass", "fail"],
        "result status taxonomy drifted",
    )
    require(taxonomy["empty_denominator_status"] == "not_evaluable", "empty denominator can pass")
    require(taxonomy["empty_denominator_value"] is None, "empty denominator gained a numeric value")
    require(
        taxonomy["zero_numerator_with_empty_denominator_is_never_pass"] is True,
        "zero numerator with empty denominator can pass",
    )
    require(
        taxonomy["partial_metric_evidence_does_not_inherit_global_failure"] is True,
        "partial metric evidence is erased by a global failure",
    )
    require(
        taxonomy["support_status_is_relative_to_declared_maturity_level"] is True,
        "metric support status lost its declared-level context",
    )
    require(
        taxonomy["evaluable_powered_never_implies_bound_sufficient_or_gate_pass"] is True,
        "powered support can bypass confidence or performance gates",
    )

    result_contract = standard["metric_result_contract"]
    required_result_fields = {
        "support_status",
        "result_status",
        "numerator",
        "denominator",
        "value",
        "ci_method",
        "ci_lower",
        "ci_upper",
        "bound_sufficient",
        "gate_result",
        "ineligible_reason_counts",
    }
    require(
        set(result_contract["required_fields"]) == required_result_fields,
        "metric result contract fields drifted",
    )

    metrics = standard["metric_evaluability"]
    require(set(metrics) == METRIC_IDS, "metric evaluability roster drifted")
    for metric_id, expected in EXPECTED_METRIC_CONTRACTS.items():
        require(metrics[metric_id] == expected, f"metric contract drifted: {metric_id}")

    censoring = standard["censoring_policy"]
    require(censoring["censored_never_imputed_as_success_or_zero_latency"] is True, "censoring imputation opened")
    require(
        censoring["full_clearance_horizon_observed_without_candidate_clear_is_failure"] is True,
        "observed clearance timeout can be censored",
    )
    require(
        censoring["identity_loss_censoring_may_not_enter_optimistic_survival_estimate"] is True,
        "identity-loss censoring can enter optimistic survival analysis",
    )
    require(
        censoring["competing_route_change_is_separate_event_not_clearance_success_by_default"]
        is True,
        "competing route change can count as clearance success",
    )
    require(
        censoring["competing_route_change_treatment_frozen_by_truth_event_type_before_candidate_output"]
        is True,
        "competing route change treatment is post-hoc",
    )
    require(
        censoring["unregistered_competing_route_change_is_unresolved_for_promotion"] is True,
        "unregistered competing route change can promote",
    )
    require(
        censoring["clearance_censor_fraction_numerator"]
        == "post_truth_clear_administrative_plus_identity_loss_plus_unregistered_competing_route_change",
        "censor fraction numerator drifted",
    )
    require(
        censoring["clearance_censor_fraction_denominator"]
        == "events_with_truth_terminal_clear_observed",
        "censor fraction denominator drifted",
    )
    require(
        censoring["terminal_clear_observability_rate"]
        == "events_with_truth_terminal_clear_observed_divided_by_recall_eligible_events",
        "terminal-clear observability contract drifted",
    )
    require(
        censoring["clearance_rate_lcb_and_p95_or_survival_quantile_required_for_confirmation"]
        is True,
        "clearance confirmation bounds are incomplete",
    )
    require(
        censoring["undefined_clearance_bound_sets_bound_sufficient_false"] is True,
        "undefined clearance bound can confirm",
    )
    require(censoring["maximum_censor_fraction_for_selection"] <= 0.2, "selection censor cap weakened")
    require(
        censoring["maximum_censor_fraction_for_confirmation_or_shadow"] <= 0.1,
        "confirmation censor cap weakened",
    )

    thresholds = standard["threshold_profile"]
    r1_thresholds = r1["selection_gate_each_source"]
    require(THRESHOLD_KEYS.issubset(thresholds), "R2 threshold profile incomplete")
    for key in THRESHOLD_KEYS:
        require(thresholds[key] == r1_thresholds[key], f"R1 performance threshold drifted: {key}")
    require(
        thresholds["performance_thresholds_are_not_relaxed_by_lower_maturity"] is True,
        "lower maturity can relax performance thresholds",
    )

    confidence = standard["confidence_policy"]
    require(confidence["confidence_level"] == 0.95, "confidence level drifted")
    require(confidence["poisson_rate_bound_is_working_model_only"] is True, "Poisson bound overclaimed")
    require(
        confidence["event_level_exact_bounds_are_necessary_not_sufficient_for_confirmation"]
        is True,
        "event-level exact bounds can bypass cluster evidence",
    )
    require(
        confidence["cluster_aware_ci_and_worst_session_required_for_confirmation"] is True,
        "cluster-aware confirmation gate disabled",
    )
    require(
        confidence["cluster_bootstrap"]
        == {
            "iterations": 10000,
            "seed": 20260723,
            "stratify_by": "provenance_family",
            "resample_unit": "session_within_family",
            "minimum_sessions_per_family": 3,
            "worst_family_sentinel_required": True,
            "provenance_family_random_resampling_enabled": False,
            "minimum_families_before_random_family_resampling": 5,
            "loso_worst_session_required": True,
        },
        "cluster bootstrap contract drifted",
    )
    require(
        confidence["undefined_or_degenerate_cluster_ci_sets_bound_sufficient_false"] is True,
        "undefined cluster CI can confirm",
    )
    require(confidence["zero_failure_recall_lcb_0_90_minimum_n"] >= 29, "recall bound floor weakened")
    require(confidence["zero_critical_miss_ucb_0_05_minimum_n"] >= 59, "critical bound floor weakened")
    require(
        confidence["zero_false_alert_poisson_ucb_0_50_per_minute_minimum_exposure_minutes"]
        >= 5.99,
        "false-alert bound exposure weakened",
    )
    require(
        confidence["insufficient_bound_sets_bound_sufficient_false_and_cannot_confirm"] is True,
        "insufficient confidence bound can confirm",
    )
    require(
        standard["metric_inference_policy"] == METRIC_INFERENCE_POLICY,
        "metric inference policy drifted",
    )

    levels = standard["maturity_levels"]
    require([level["id"] for level in levels] == LEVEL_IDS, "maturity level order drifted")
    by_id = {level["id"]: level for level in levels}
    l0 = by_id["L0_ENGINEERING_DIAGNOSTIC"]
    l1 = by_id["L1_EXPLORATORY_METRIC_PROFILE"]
    l2 = by_id["L2_CANDIDATE_SELECTION"]
    l3 = by_id["L3_OFFLINE_CONFIRMATION"]
    l4 = by_id["L4_ANDROID_SHADOW"]
    require(l0["candidate_execution_allowed"] is False, "L0 candidate execution opened")
    require(l0["candidate_winner_allowed"] is False, "L0 winner opened")
    require(l1["all_metrics_need_not_be_evaluable_together"] is True, "L1 restored all-or-nothing admission")
    require(l1["underpowered_nonzero_support_may_be_reported"] is True, "L1 hides underpowered evidence")
    require(l1["candidate_winner_allowed"] is False, "L1 winner opened")
    require(l1["android_shadow_allowed"] is False, "L1 Android shadow opened")
    require(l2["minimum_independent_session_families"] >= 2, "L2 independent-family floor weakened")
    require(l2["maximum_single_family_share"] <= 0.7, "L2 family concentration cap weakened")
    require(l2["minimum_total_event_recall_events"] >= 20, "L2 recall floor weakened")
    require(l2["minimum_total_critical_events"] >= 5, "L2 critical floor weakened")
    require(l2["minimum_total_clearance_events"] >= 15, "L2 clearance floor weakened")
    require(l2["minimum_total_complete_repeat_events"] >= 15, "L2 repeat floor weakened")
    require(
        l2["minimum_total_complete_regeneration_intervals"] >= 15,
        "L2 regeneration floor weakened",
    )
    require(l2["minimum_total_negative_exposure_minutes"] >= 20.0, "L2 negative exposure weakened")
    require(
        l2["minimum_matched_pairs_if_relative_claim"] >= 10,
        "L2 relative-claim matched-pair floor weakened",
    )
    require(
        l2["minimum_per_family_support_for_rate_gate"]
        == {
            "event_recall_events": 5,
            "critical_events": 1,
            "clearance_events": 3,
            "complete_repeat_events": 3,
            "complete_regeneration_intervals": 3,
            "negative_exposure_minutes": 5.0,
        },
        "L2 per-family support floor drifted",
    )
    require(
        l2["all_selected_replay_frames_require_bound_timestamps_and_route_validity"] is True,
        "L2 timestamp or route-validity denominator incomplete",
    )
    require(set(l2["required_metrics"]) == METRIC_IDS, "L2 required metric roster drifted")
    require(l2["each_required_metric_must_be_evaluable_powered"] is True, "L2 accepts underpowered metrics")
    require(l2["selection_runs_per_candidate"] == 1, "L2 one-shot selection contract drifted")
    require(l2["selection_contract"] == L2_SELECTION_CONTRACT, "L2 selection contract drifted")
    require(l2["android_shadow_allowed"] is False, "L2 directly opens Android shadow")
    require(
        l2["safety_effectiveness_claim_allowed"] is False,
        "L2 gained safety-effectiveness authority",
    )
    require(l3["minimum_sessions"] >= 6, "L3 session floor weakened")
    require(l3["minimum_provenance_families"] >= 2, "L3 provenance-family floor weakened")
    require(l3["minimum_positive_events"] >= 60, "L3 positive floor weakened")
    require(l3["minimum_matched_negative_episodes"] >= 60, "L3 negative episode floor weakened")
    require(l3["minimum_complete_matched_pairs"] >= 60, "L3 complete-pair floor weakened")
    require(
        l3["paired_positive_and_negative_must_remain_in_same_fold"] is True,
        "L3 matched pair can split across folds",
    )
    require(l3["minimum_complete_repeat_events"] >= 60, "L3 repeat floor weakened")
    require(
        l3["minimum_complete_regeneration_intervals"] >= 60,
        "L3 regeneration floor weakened",
    )
    require(
        l3["all_confirmation_replay_frames_require_bound_timestamps_and_route_validity"]
        is True,
        "L3 timestamp or route-validity denominator incomplete",
    )
    require(l3["minimum_scenario_strata"] >= 5, "L3 scenario-strata floor weakened")
    require(l3["loso_fold_count"] >= 6, "L3 LOSO floor weakened")
    require(l3["maximum_single_family_share"] <= 0.6, "L3 family concentration cap weakened")
    require(
        l3["minimum_critical_events_for_zero_miss_rate_bound"] >= 59,
        "L3 critical confidence-bound floor weakened",
    )
    require(
        set(l3["required_bound_metrics_must_be_sufficient"])
        == {"event_recall", "critical_miss", "false_alerts_per_minute", "clearance"},
        "L3 required-bound metric roster drifted",
    )
    require(
        set(l3["hard_veto_or_engineering_metrics_use_complete_denominator_and_worst_session_gate"])
        == {
            "repeat_within_observation",
            "event_regeneration_after_clear",
            "unknown_or_stale_active_alert",
            "evidence_age",
        },
        "L3 hard-veto or engineering metric roster drifted",
    )
    require(
        l3["each_required_metric_must_draw_from_at_least_two_families"] is True,
        "L3 metrics can rely on one provenance family",
    )
    require(
        l3["cluster_confidence_intervals_required"] is True,
        "L3 cluster confidence intervals disabled",
    )
    require(l3["android_shadow_execution_allowed"] is False, "L3 bypasses Android admission")
    require(l3["human_safety_claim_allowed"] is False, "L3 gained human-safety authority")
    require(l4["requires_l3_pass"] is True, "L4 no longer requires L3")
    require(set(l4["entry_requirements"]) == L4_ENTRY_REQUIREMENTS, "L4 entry requirements drifted")
    l4_floors = l4["completion_floors"]
    require(l4_floors["independent_runs"] >= 10, "L4 run floor weakened")
    require(l4_floors["event_recall_events"] >= 60, "L4 recall floor weakened")
    require(l4_floors["critical_events"] >= 59, "L4 critical floor weakened")
    require(l4_floors["complete_repeat_events"] >= 60, "L4 repeat floor weakened")
    require(l4_floors["clearance_events"] >= 60, "L4 clearance floor weakened")
    require(
        l4_floors["complete_regeneration_intervals"] >= 60,
        "L4 regeneration floor weakened",
    )
    require(l4_floors["negative_exposure_minutes"] >= 120.0, "L4 exposure weakened")
    require(
        l4["each_required_metric_shadow_gate_and_bound_sufficient_required"] is True,
        "L4 accepts incomplete metric gates",
    )
    require(
        l4["partial_metric_insufficiency_stops_only_that_metric_but_global_l4_complete_requires_all"]
        is True,
        "L4 partial evidence semantics drifted",
    )
    require(l4["human_outcome_or_production_authority"] is False, "L4 gained production authority")
    require(
        l4["production_promotion_requires_separate_protocol_and_review"] is True,
        "L4 bypasses separate production review",
    )

    require(standard["source_wise_policy"] == SOURCE_WISE_POLICY, "source-wise policy drifted")
    adaptation = standard["adaptation_policy"]
    require(set(adaptation) == ADAPTATION_KEYS, "adaptation rule roster drifted")
    require(all(adaptation.values()), "adaptation or anti-post-hoc rule disabled")
    budget = standard["source_acquisition_budget"]
    require(0 < budget["maximum_new_source_families_per_round"] <= 2, "source family budget is unbounded")
    require(0 < budget["maximum_canaries_per_source"] <= 2, "source canary budget is unbounded")
    require(budget["default_maximum_download_bytes_per_round"] > 0, "download budget is unbounded")
    require(
        budget["maximum_consecutive_ineligible_source_families_before_stop"] <= 2,
        "source futility stop weakened",
    )
    require(budget["full_materialization_before_metric_specific_canary_feasibility"] is False, "blind full download opened")
    require(budget["candidate_specific_source_search"] is False, "candidate-specific source search opened")
    require(
        budget["budget_override_requires_new_candidate_blind_preregistration"] is True,
        "source budget override bypasses candidate-blind preregistration",
    )
    require(budget["current_r1_source_search_state"] == "stopped", "R1 source search resumed")

    dispositions = {row["id"]: row for row in standard["existing_data_disposition"]}
    require(len(dispositions) == 3, "existing data disposition roster drifted")
    require(
        dispositions["r1_seen_lilocbench_15_positive_15_negative"]["maximum_level"]
        == "L1_EXPLORATORY_METRIC_PROFILE",
        "seen R1 data authority drifted",
    )
    replacement = dispositions["r1_crowdbot_first_and_replacement_materializations"]
    require(
        replacement["maximum_level"] == "L1_EXPLORATORY_METRIC_PROFILE",
        "replacement R1 data authority drifted",
    )
    require(
        dispositions["r1_rejected_0327_navwareset_revel_and_external_sources"]["maximum_level"]
        == "L0_ENGINEERING_DIAGNOSTIC",
        "rejected R1 sources gained exploratory authority",
    )
    require(
        "r2_confirmation" in dispositions["r1_seen_lilocbench_15_positive_15_negative"]["forbidden"],
        "seen data entered R2 confirmation",
    )
    require(
        replacement[
            "right_censored_event_proposals_require_new_candidate_blind_local_eligibility_audit"
        ]
        is True,
        "censored R1 events were accepted without local eligibility audit",
    )
    partial_path = resolve_bound(repo, replacement["known_partial_support_evidence"], "R1 partial support")
    partial = load_json(partial_path)
    require(partial["candidate_outputs_executed"] is False, "R1 partial support saw candidate outputs")
    require(
        partial["decision"] == "STOP_NO_CANDIDATE_EXECUTION_REPLACEMENT_HOLDOUT_ADMISSION_0_OF_2",
        "R1 partial support decision drifted",
    )
    source_rows = partial["sources"]
    observed_support = {
        "complete_terminal_clear_events": sum(row["accepted_positive_event_count"] for row in source_rows),
        "critical_events": sum(row["accepted_critical_event_count"] for row in source_rows),
        "quarantined_events": partial["fusion_totals"]["quarantined_event_count"],
        "scorable_negative_exposure_minutes": sum(
            row["scorable_negative_exposure_minutes"] for row in source_rows
        ),
    }
    declared_support = replacement["known_partial_support"]
    require(
        declared_support["complete_terminal_clear_events"]
        == observed_support["complete_terminal_clear_events"],
        "R1 complete-clear support count drifted",
    )
    require(
        declared_support["critical_events"] == observed_support["critical_events"],
        "R1 critical support count drifted",
    )
    require(
        declared_support["quarantined_events"] == observed_support["quarantined_events"],
        "R1 quarantined-event count drifted",
    )
    require(
        abs(
            declared_support["scorable_negative_exposure_minutes"]
            - observed_support["scorable_negative_exposure_minutes"]
        )
        < 1e-12,
        "R1 negative-exposure support drifted",
    )

    decisions = standard["decision_states"]
    require(decisions["partial_support"] == "PARTIAL_METRIC_EVIDENCE", "partial support state drifted")
    require(
        decisions["insufficient_support_after_budget"] == "STOP_DATA_COLLECTION_AT_CURRENT_LEVEL",
        "bounded insufficient-support stop drifted",
    )
    require(decisions["global_data_blocked_may_not_erase_partial_metric_evidence"] is True, "global block erases evidence")
    authority = standard["current_authority"]
    require(authority["highest_authorized_level"] == "L0_ENGINEERING_DIAGNOSTIC", "current authority overclaimed")
    require(authority["r2_metric_eligibility_masks_frozen"] is False, "R2 masks falsely marked frozen")
    require(authority["r2_candidate_outputs_executed"] is False, "R2 candidates falsely marked run")
    require(authority["r2_candidate_selection_authority"] is False, "R2 selection authority opened")
    require(authority["android_shadow"] == "closed", "Android shadow opened")
    require(authority["production"] == "not_authorized", "production authority opened")

    return {
        "decision": "VALID_EVIDENCE_MATURITY_STANDARD_V2",
        "r1_decision_preserved": preservation["r1_decision_remains"],
        "maturity_level_count": len(levels),
        "metric_count": len(metrics),
        "current_authority": authority["highest_authorized_level"],
        "r2_candidate_execution_authority": False,
        "android_shadow": authority["android_shadow"],
        "production": authority["production"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/ustrf_route_target_evidence_maturity_v2.json",
    )
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    result = validate_standard(repo, load_json(repo / args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
