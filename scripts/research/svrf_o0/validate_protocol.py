#!/usr/bin/env python3
"""Validate that SVRF-O0 protocol and executable constants remain aligned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import ARM_IDS, BEST_SINGLE_RULE, EvaluationPolicy


EXPECTED_STATUS = "FROZEN_PREOUTCOME_BLOCKED_ON_SELECTIVE_MATERIALIZATION_AND_TRUTH_WRITER_LOCK"


def validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("schema") != "blindassist.svrf_o0.protocol.v1":
        raise ValueError("SVRF-O0 protocol schema mismatch")
    if protocol.get("status") != EXPECTED_STATUS:
        raise ValueError("SVRF-O0 top-level status drift")
    if tuple(protocol["arms"]["all_ids"]) != ARM_IDS:
        raise ValueError("SVRF-O0 arm identity drift")
    if protocol["candidate_input_firewall"]["rgb_only"] is not True:
        raise ValueError("SVRF-O0 must remain RGB-only")
    if protocol["candidate_input_firewall"].get("candidate_intrinsics_policy") != "FIXED_CANONICAL_OR_RGB_DERIVED":
        raise ValueError("SVRF-O0 candidate intrinsics policy drift")
    if protocol["candidate_input_firewall"].get("source_native_camera_intrinsics") != "FORBIDDEN_CANDIDATE_INPUT_EVALUATOR_ONLY":
        raise ValueError("SVRF-O0 source-native intrinsics firewall drift")
    source = protocol["fresh_source_contract"]
    if source.get("exact_source_roster_selected") is not True or source.get("source_lock") != "SVRF_O0_A2D2_SPRING_SOURCE_LOCK_2026-08-15.json":
        raise ValueError("SVRF-O0 source-lock binding drift")
    if source.get("archive_access_capability_lock") != "SVRF_O0_ARCHIVE_ACCESS_CAPABILITY_LOCK_2026-08-15.json":
        raise ValueError("SVRF-O0 archive capability binding drift")
    if source.get("stream_index_execution_lock") != "SVRF_O0_STREAM_INDEX_EXECUTION_LOCK_2026-08-15.json":
        raise ValueError("SVRF-O0 stream-index execution binding drift")
    if any(protocol["authority"].values()):
        raise ValueError("SVRF-O0 pre-outcome authority must remain closed")
    truth = protocol.get("truth_unknown_contract", {})
    if truth.get("valid_status") != "VALID_TRUTH" or truth.get("unknown_status_prefix") != "UNKNOWN_":
        raise ValueError("SVRF-O0 truth status contract drift")
    if truth.get("all_locked_identities_retained") is not True:
        raise ValueError("SVRF-O0 truth UNKNOWN denominator drift")
    coverage = truth.get("coverage_definitions", {})
    if coverage.get("winner_rule_coverage") != "joint_evaluable_coverage":
        raise ValueError("SVRF-O0 winner coverage definition drift")
    policy = EvaluationPolicy()
    winner = protocol["winner_rule"]
    if winner.get("best_single_rule") != BEST_SINGLE_RULE:
        raise ValueError("SVRF-O0 best-single comparator drift")
    expected = {
        "minimum_parent_count": policy.minimum_parent_count,
        "minimum_source_count": policy.minimum_source_count,
        "a3_parent_macro_coverage_min": policy.a3_parent_macro_coverage_min,
        "a3_worst_parent_coverage_min": policy.a3_worst_parent_coverage_min,
        "a3_approach_macro_f1_min": policy.a3_approach_macro_f1_min,
        "a3_parent_macro_spearman_min": policy.a3_parent_macro_spearman_min,
        "a3_parent_macro_pairwise_ranking_min": policy.a3_parent_macro_pairwise_ranking_min,
        "a3_parent_macro_false_clear_max": policy.a3_parent_macro_false_clear_max,
        "a3_parent_macro_false_block_max": policy.a3_parent_macro_false_block_max,
        "a3_worst_parent_false_block_max": policy.a3_worst_parent_false_block_max,
        "matched_false_clear_absolute_gain_min": policy.matched_false_clear_absolute_gain_min,
        "matched_parent_improvement_count_min": policy.matched_parent_improvement_count_min,
        "matched_source_improvement_count_min": policy.matched_source_improvement_count_min,
        "matched_coverage_delta_min": policy.matched_coverage_delta_min,
        "negative_control_macro_f1_degradation_min": policy.negative_control_macro_f1_degradation_min,
        "minimum_parents_with_core_metric_support": policy.minimum_parents_with_core_metric_support,
    }
    if any(winner.get(key) != value for key, value in expected.items()):
        raise ValueError("SVRF-O0 executable winner-rule drift")
    support = protocol.get("metric_support_contract", {})
    if support.get("all_eight_parents_required_for_core_metrics") is not True:
        raise ValueError("SVRF-O0 all-parent metric support drift")
    if support.get("matched_high_risk_support_minimum") != {
        "parents": policy.matched_parent_improvement_count_min,
        "sources": policy.matched_source_improvement_count_min,
    }:
        raise ValueError("SVRF-O0 matched high-risk support drift")
    preflight = protocol.get("selective_materialization_preflight_contract", {})
    if preflight.get("outcome_metrics_forbidden") is not True:
        raise ValueError("SVRF-O0 preflight outcome firewall drift")
    if "canonical F-drive junction target" not in preflight.get("archive_policy", ""):
        raise ValueError("SVRF-O0 large-data storage policy drift")
    negative = protocol.get("negative_control_lock_contract", {})
    if "positive-yaw 5-degree" not in negative.get("N2", ""):
        raise ValueError("SVRF-O0 N2 transform identity drift")
    if "preserve RGB/depth identities" not in negative.get("N3", ""):
        raise ValueError("SVRF-O0 N3 support-preservation drift")
    if protocol["activation_contract"].get("outcome_access_authorized") is not False:
        raise ValueError("SVRF-O0 outcome access opened before preflight")
    if protocol.get("preflight_execution_authority") != {
        "a2d2_stream_index_execution_authorized": True,
        "spring_range_manifest_execution_authorized": True,
        "selected_payload_materialization_authorized": False,
        "truth_writer_execution_authorized": False,
        "candidate_run_authorized": False,
        "outcome_access_authorized": False,
    }:
        raise ValueError("SVRF-O0 preflight execution authority drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    print("SVRF_O0_PROTOCOL_VALID")


if __name__ == "__main__":
    main()
