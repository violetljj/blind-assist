#!/usr/bin/env python3
"""Validate that SVRF-O0 protocol and executable constants remain aligned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import ARM_IDS, BEST_SINGLE_RULE, EvaluationPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "blindassist.svrf_o0.protocol.v1":
        raise ValueError("SVRF-O0 protocol schema mismatch")
    if tuple(protocol["arms"]["all_ids"]) != ARM_IDS:
        raise ValueError("SVRF-O0 arm identity drift")
    if protocol["candidate_input_firewall"]["rgb_only"] is not True:
        raise ValueError("SVRF-O0 must remain RGB-only")
    if any(protocol["authority"].values()):
        raise ValueError("SVRF-O0 pre-outcome authority must remain closed")
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
    }
    if any(winner.get(key) != value for key, value in expected.items()):
        raise ValueError("SVRF-O0 executable winner-rule drift")
    print("SVRF_O0_PROTOCOL_VALID")


if __name__ == "__main__":
    main()
