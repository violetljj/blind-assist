#!/usr/bin/env python3
"""Evaluate the frozen D2 Development baseline-versus-head quality payload."""

from __future__ import annotations

import hashlib
from typing import Any

from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d1_quality import (
    METRICS,
    aggregate,
    finite,
    flatten,
    ge,
    le,
    require,
)


SCHEMA = "blindassist_depthart_task_preserving_d2_development_quality_payload_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_BASELINE_AND_FROZEN_HEAD_QUALITY"


def evaluate(protocol: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    require(payload.get("schema") == SCHEMA, "payload schema mismatch")
    require(payload.get("protocol_id") == PROTOCOL_ID, "payload protocol id mismatch")
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 1200, "D2 Development requires exactly 1200 frames")
    ordered_sessions = protocol.get("development_scope")
    require(isinstance(ordered_sessions, list) and len(ordered_sessions) == 4,
            "protocol requires four ordered Development sessions")
    for session_index, expected in enumerate(ordered_sessions):
        block = rows[session_index * 300:(session_index + 1) * 300]
        require(all(str(row.get("parent_id")) == str(expected["visit_id"]) for row in block),
                "ordered parent mapping drift")
        require(all(str(row.get("session_id")) == str(expected["video_id"]) for row in block),
                "ordered session mapping drift")
        require([row.get("frame_index") for row in block] == list(range(300)),
                "ordered frame indices drift")
        stems_digest = hashlib.sha256(
            ("\n".join(str(row.get("frame_id")) for row in block) + "\n").encode()
        ).hexdigest().upper()
        require(stems_digest == expected["frame_stems_sha256"], "frozen frame stem schedule drift")

    cells, bands = flatten(rows)
    require(len({row["parent_id"] for row in cells}) == 4, "D2 requires exactly four parents")
    require(len({row["session_id"] for row in cells}) == 4, "D2 requires exactly four sessions")
    baseline = aggregate(cells, bands, "reference")
    candidate = aggregate(cells, bands, "candidate")
    gates = protocol["quality_gates"]
    noninferiority_gates = gates["noninferiority_against_baseline"]
    base, cand = baseline["pooled"], candidate["pooled"]

    aggregate_complete = all(
        finite(arm[name][metric])
        for arm in (baseline, candidate)
        for name in ("pooled", "parent_macro", "session_macro", "worst_parent")
        for metric in METRICS
    )
    strata_complete = (
        len(baseline["by_grid"]) == len(candidate["by_grid"]) == 9
        and all(
            finite(grid[metric])
            for arm in (baseline, candidate)
            for grid in arm["by_grid"].values()
            for metric in METRICS
        )
    )
    absolute = {
        "known_coverage": ge(cand["known_coverage"], gates["known_coverage_min"]),
        "clearance_mae_m": le(cand["clearance_mae_m"], gates["clearance_mae_m_max"]),
        "false_clear_all_known": le(cand["false_clear_all_known"], gates["false_clear_all_known_max"]),
        "false_clear_given_occupied_finite": finite(cand["false_clear_given_occupied"]),
        "false_block_given_clear": le(cand["false_block_given_clear"], gates["false_block_given_clear_max"]),
        "temporal_clearance_delta_mae_m": le(
            cand["temporal_clearance_delta_mae_m"], gates["temporal_clearance_delta_mae_m_max"]
        ),
        "geometry_transition_agreement": ge(
            cand["geometry_transition_agreement"], gates["geometry_transition_agreement_min"]
        ),
        "valid_to_unknown_rate": le(cand["valid_to_unknown_rate"], gates["valid_to_unknown_rate_max"]),
        "worst_parent_false_clear_all_known": le(
            candidate["worst_parent"]["false_clear_all_known"],
            gates["worst_parent_false_clear_all_known_max"],
        ),
    }
    noninferiority = {
        "known_coverage": ge(
            cand["known_coverage"], base["known_coverage"] - noninferiority_gates["known_coverage_decrease_max"]
        ) if finite(base["known_coverage"]) else False,
        "clearance_mae_m": le(
            cand["clearance_mae_m"], base["clearance_mae_m"] + noninferiority_gates["clearance_mae_m_increase_max"]
        ) if finite(base["clearance_mae_m"]) else False,
        "false_clear_all_known": le(
            cand["false_clear_all_known"],
            base["false_clear_all_known"] + noninferiority_gates["false_clear_all_known_increase_max"],
        ) if finite(base["false_clear_all_known"]) else False,
        "false_clear_given_occupied": le(
            cand["false_clear_given_occupied"],
            base["false_clear_given_occupied"]
            + noninferiority_gates["false_clear_given_occupied_increase_max"],
        ) if finite(base["false_clear_given_occupied"]) else False,
        "false_block_given_clear": le(
            cand["false_block_given_clear"],
            base["false_block_given_clear"] + noninferiority_gates["false_block_given_clear_increase_max"],
        ) if finite(base["false_block_given_clear"]) else False,
        "temporal_clearance_delta_mae_m": le(
            cand["temporal_clearance_delta_mae_m"],
            base["temporal_clearance_delta_mae_m"]
            + noninferiority_gates["temporal_clearance_delta_mae_m_increase_max"],
        ) if finite(base["temporal_clearance_delta_mae_m"]) else False,
        "geometry_transition_agreement": ge(
            cand["geometry_transition_agreement"],
            base["geometry_transition_agreement"]
            - noninferiority_gates["geometry_transition_agreement_decrease_max"],
        ) if finite(base["geometry_transition_agreement"]) else False,
        "valid_to_unknown_rate": le(
            cand["valid_to_unknown_rate"],
            base["valid_to_unknown_rate"] + noninferiority_gates["valid_to_unknown_rate_increase_max"],
        ) if finite(base["valid_to_unknown_rate"]) else False,
    }
    passed = aggregate_complete and strata_complete and all(absolute.values()) and all(noninferiority.values())
    return {
        "schema": "blindassist_depthart_task_preserving_d2_development_quality_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "FAIL",
        "terminal": (
            "D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_PASS_IDENTITY_DISJOINT_FEASIBILITY_ONLY"
            if passed else "D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_STOP"
        ),
        "counts": {
            "frames": len(rows), "cells": len(cells), "band_clearances": len(bands),
            "parents": 4, "sessions": 4,
        },
        "baseline": baseline,
        "candidate": candidate,
        "gates": {
            "aggregation_complete": aggregate_complete,
            "required_strata_complete": strata_complete,
            "absolute": absolute,
            "noninferiority": noninferiority,
        },
        "authority": {
            "identity_disjoint_development_feasibility": passed,
            "r2_candidate_lock": False,
            "r2_access": False,
            "performance": False,
            "android_default": False,
            "production": False,
            "safety": False,
        },
    }
