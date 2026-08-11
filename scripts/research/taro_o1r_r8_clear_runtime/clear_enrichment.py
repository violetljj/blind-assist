#!/usr/bin/env python3
"""Source-only parent scoring for a fresh definite-clear negative-control pool."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Mapping, Sequence

from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


SELECTOR_ID = "R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_PARENT_ENRICHMENT_V1"
SELECTION_SALT = "TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_FINAL8_V1"
MINIMUM_FAR_VISIBLE_ANCHORS = 9
MINIMUM_FAR_FRACTION_AT_2_5M = 0.8
POOL_PARENT_COUNT = 24
SELECTED_PARENT_COUNT = 8


class ClearEnrichmentError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ClearEnrichmentError(code, message)


def query_is_clear_negative_control_candidate(feature: Mapping[str, Any]) -> bool:
    """Return a source-only enrichment flag; this never emits CLEAR."""

    state, _ = positive_occupancy_factor.state_from_feature(feature)
    if state != "UNKNOWN" or feature.get("query_receipt") is None:
        return False
    fractions = feature.get("far_fractions")
    require(isinstance(fractions, list) and len(fractions) == 3, "R8_CLEAR_FRACTION_INVALID", "source far-fraction grid invalid")
    return bool(
        feature.get("positive_obstacle_veto") is False
        and int(feature.get("far_valid_anchor_count", 0)) >= MINIMUM_FAR_VISIBLE_ANCHORS
        and float(fractions[0]) >= MINIMUM_FAR_FRACTION_AT_2_5M
    )


def score_parent(source_frame_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one parent without any FARO, truth, label, or outcome argument."""

    require(bool(source_frame_records), "R8_CLEAR_PARENT_EMPTY", "parent source records are empty")
    rows = [r7_canary.validate_source_frame_record(dict(row)) for row in source_frame_records]
    parent = str(rows[0]["parent_id"])
    video = str(rows[0]["video_id"])
    require(all(row["parent_id"] == parent and row["video_id"] == video for row in rows), "R8_CLEAR_PARENT_IDENTITY_DRIFT", "source records mix parent identities")
    eligible = available = 0
    for row in rows:
        for feature in row["query_features"]:
            available += feature["query_receipt"] is not None
            eligible += query_is_clear_negative_control_candidate(feature)
    tie = hashlib.sha256(f"{SELECTION_SALT}:{parent}:{video}".encode("ascii")).hexdigest().upper()
    return {
        "selector_id": SELECTOR_ID,
        "parent_id": parent,
        "video_id": video,
        "frame_count": len(rows),
        "query_count": len(rows) * 9,
        "available_query_count": int(available),
        "eligible_query_count": int(eligible),
        "eligible_fraction_of_available": float(eligible / available) if available else 0.0,
        "tie_break_sha256": tie,
        "faro_reads": 0,
        "truth_reads": 0,
        "clear_output_emitted": False,
    }


def select_final_parents(parent_scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(parent_scores) == POOL_PARENT_COUNT, "R8_CLEAR_POOL_COUNT_DRIFT", "source-only pool must contain 24 parents")
    scores = [dict(row) for row in parent_scores]
    identities = [(row.get("parent_id"), row.get("video_id")) for row in scores]
    require(len(set(identities)) == POOL_PARENT_COUNT, "R8_CLEAR_POOL_IDENTITY_DRIFT", "source-only pool identities are not unique")
    required = {"selector_id", "parent_id", "video_id", "frame_count", "query_count", "available_query_count", "eligible_query_count", "eligible_fraction_of_available", "tie_break_sha256", "faro_reads", "truth_reads", "clear_output_emitted"}
    for row in scores:
        require(set(row) == required and row["selector_id"] == SELECTOR_ID and row["faro_reads"] == row["truth_reads"] == 0 and row["clear_output_emitted"] is False, "R8_CLEAR_SCORE_INVALID", "parent score fields/firewall drift")
    ranked = sorted(scores, key=lambda row: (-int(row["eligible_query_count"]), -float(row["eligible_fraction_of_available"]), str(row["tie_break_sha256"])))
    return ranked[:SELECTED_PARENT_COUNT]


def assert_public_api_truth_blind() -> None:
    for function in (score_parent, select_final_parents):
        names = inspect.signature(function).parameters
        require(not any(token in name.lower() for name in names for token in ("faro", "truth", "label", "outcome")), "R8_CLEAR_API_TRUTH_SIDE", "selector public API has a result-side parameter")


__all__ = [
    "ClearEnrichmentError",
    "SELECTOR_ID",
    "assert_public_api_truth_blind",
    "query_is_clear_negative_control_candidate",
    "score_parent",
    "select_final_parents",
]
