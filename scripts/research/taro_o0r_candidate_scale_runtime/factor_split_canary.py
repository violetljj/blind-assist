"""Post-hoc TARO canary for a fixed factor-wise R5 successor policy.

The policy is intentionally simple and truth-independent once selected:
SUPPORT and BOUNDARY use the frozen R5 selected branch, while
QUERY_CLEARANCE always uses the R1 baseline block.  Because the policy was
formed after reading R5 outcomes, this module emits landscape evidence only;
it has no PASS/FAIL or promotion authority.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor


SCHEMA = "blindassist.taro.o0r.r6_factor_split_posthoc_canary.v1"
POLICY_ID = "R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1"
CLAIM_CEILING = "POST_HOC_LANDSCAPE_ONLY_REQUIRES_UNTOUCHED_PARENT_DISJOINT_CONFIRMATION"


def _parent_macro(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row["effects"].get(field)
        if r5._finite(value):
            by_frame.setdefault((row["parent_id"], row["physical_frame_id"]), []).append(float(value))
    by_parent: dict[str, list[float]] = {}
    for (parent, _), values in by_frame.items():
        by_parent.setdefault(parent, []).append(float(np.median(np.asarray(values, dtype=np.float64))))
    parent_values = []
    for parent in sorted({row["parent_id"] for row in rows}):
        values = by_parent.get(parent, [])
        parent_values.append(
            {
                "parent_id": parent,
                "paired_frame_count": len(values),
                "median_frame_effect": float(np.median(np.asarray(values, dtype=np.float64))) if values else None,
            }
        )
    usable = [row["median_frame_effect"] for row in parent_values if row["median_frame_effect"] is not None]
    return {
        "parents_with_metric": len(usable),
        "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None,
        "parent_values": parent_values,
    }


def evaluate_factor_split_landscape(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_parent_frame_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows = [r5.validate_query_record(dict(row)) for row in records]
    expected_counts = dict(expected_parent_frame_counts or zip((parent for parent, _ in r5.R5_ROSTER), r5.EXPECTED_PARENT_FRAME_COUNTS))
    expected_frames = sum(expected_counts.values())
    expected_queries = expected_frames * 9
    frames = {row["physical_frame_id"] for row in rows}
    parents = {row["parent_id"] for row in rows}
    keys = {(row["physical_frame_id"], row["query_id"]) for row in rows}
    r5.require(
        len(rows) == len(keys) == expected_queries
        and len(frames) == expected_frames
        and parents == set(expected_counts),
        "R6_FACTOR_SPLIT_COHORT_DRIFT",
        "factor-split canary cohort differs",
    )
    observed = Counter(row["parent_id"] for row in rows)
    r5.require(all(observed[parent] == count * 9 for parent, count in expected_counts.items()), "R6_FACTOR_SPLIT_PARENT_COUNT_DRIFT", "factor-split parent counts differ")

    effect_rows: list[dict[str, Any]] = []
    baseline_extraction = 0
    composite_extraction = 0
    baseline_known = 0
    composite_known = 0
    boundary_recovered = 0
    boundary_lost = 0
    for row in rows:
        baseline = row["baseline"]
        support_boundary = row["selected_hybrid"]
        baseline_extraction += bool(baseline["extraction_evaluable"])
        composite_extraction += bool(support_boundary["extraction_evaluable"])
        baseline_query_known = bool(baseline["query_point_clearance"]["evaluable"])
        baseline_known += baseline_query_known
        composite_known += baseline_query_known
        boundary_recovered += not baseline["boundary"]["evaluable"] and support_boundary["boundary"]["evaluable"]
        boundary_lost += baseline["boundary"]["evaluable"] and not support_boundary["boundary"]["evaluable"]
        effect_rows.append(
            {
                "parent_id": row["parent_id"],
                "physical_frame_id": row["physical_frame_id"],
                "effects": {
                    "height_error_reduction_vs_baseline_m": source_factor._difference(baseline, support_boundary, "support", "height_abs_error_m"),
                    "normal_error_reduction_vs_baseline_rad": source_factor._difference(baseline, support_boundary, "support", "normal_angular_error_rad"),
                },
            }
        )
    height = _parent_macro(effect_rows, "height_error_reduction_vs_baseline_m")
    normal = _parent_macro(effect_rows, "normal_error_reduction_vs_baseline_rad")
    height_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in height["parent_values"]}
    normal_by_parent = {row["parent_id"]: row["median_frame_effect"] for row in normal["parent_values"]}
    jointly_positive = sum(
        r5._finite(height_by_parent[parent]) and float(height_by_parent[parent]) > 0.0
        and r5._finite(normal_by_parent[parent]) and float(normal_by_parent[parent]) > 0.0
        for parent in parents
    )
    denominator_defined = height["parents_with_metric"] == normal["parents_with_metric"] == len(expected_counts)
    gate_landscape = [
        {"id": "EXACT_COHORT_AND_LINEAGE", "would_pass": True},
        {"id": "PARENT_METRIC_DENOMINATORS", "would_pass": denominator_defined},
        {"id": "HEIGHT_PARENT_MACRO_POSITIVE", "would_pass": denominator_defined and float(height["median_of_parent_medians"]) > 0.0},
        {"id": "NORMAL_PARENT_MACRO_POSITIVE", "would_pass": denominator_defined and float(normal["median_of_parent_medians"]) > 0.0},
        {"id": "ALL_PARENTS_JOINTLY_POSITIVE", "would_pass": jointly_positive == len(expected_counts)},
        {"id": "EXTRACTION_COVERAGE_NO_REGRET", "would_pass": composite_extraction >= baseline_extraction},
        {"id": "QUERY_KNOWN_COVERAGE_NO_REGRET", "would_pass": composite_known >= baseline_known},
    ]
    return r5._seal(
        {
            "schema": SCHEMA,
            "policy_id": POLICY_ID,
            "policy_blocks": {
                "SUPPORT": "R5_SELECTED_SOURCE_ONLY_BRANCH",
                "BOUNDARY": "R5_SELECTED_SOURCE_ONLY_BRANCH",
                "QUERY_CLEARANCE": "ALWAYS_R1_BASELINE",
            },
            "post_hoc": True,
            "claim_ceiling": CLAIM_CEILING,
            "promotion_allowed": False,
            "requires_untouched_confirmation": True,
            "parent_count": len(expected_counts),
            "physical_frame_count": expected_frames,
            "query_record_count": expected_queries,
            "baseline_extraction_evaluable_query_count": baseline_extraction,
            "composite_extraction_evaluable_query_count": composite_extraction,
            "baseline_query_known_count": baseline_known,
            "composite_query_known_count": composite_known,
            "boundary_evaluability_recovered_vs_baseline": boundary_recovered,
            "boundary_evaluability_lost_vs_baseline": boundary_lost,
            "height_error_reduction_vs_baseline_parent_macro_m": height,
            "normal_error_reduction_vs_baseline_parent_macro_rad": normal,
            "parents_jointly_positive_height_and_normal": jointly_positive,
            "gate_landscape": gate_landscape,
            "all_gate_landscape_would_pass": all(row["would_pass"] for row in gate_landscape),
            "pass_fail_terminal_absent": True,
        }
    )


__all__ = ["CLAIM_CEILING", "POLICY_ID", "SCHEMA", "evaluate_factor_split_landscape"]
