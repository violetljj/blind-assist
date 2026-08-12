#!/usr/bin/env python3
"""Truth-blind R11 weak-distal-component abstention candidate."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


SCHEMA = "blindassist.taro.o1r.r11_strong_component_abstention_bundle.v1"
QUERY_SCHEMA = "blindassist.taro.o1r.r11_strong_component_abstention_query.v1"
FACTOR_ID = "R11_R7_WEAK_DISTAL_COMPONENT_ABSTENTION_V1"
BASE_FACTOR_ID = r7_positive.FACTOR_ID
STRONGER_CELLS_ANY = (
    (3, 0, 2),  # at least 16 confidence-2 pixels
    (0, 1, 2),  # at least 0.15 m component height
    (0, 0, 1),  # component begins no farther than 1.5 m
)
EXPECTED_FEATURE_KEYS = {
    "far_fractions",
    "far_valid_anchor_count",
    "grid_index",
    "observed_support_points",
    "occupied_hits",
    "positive_obstacle_veto",
    "query_id",
    "query_receipt",
    "r6_state",
    "reason_codes",
}
FROZEN_ALGORITHM = {
    "factor_id": FACTOR_ID,
    "base_factor_id": BASE_FACTOR_ID,
    "base_positive_cell": {
        "pixel_index": r7_positive.PIXEL_INDEX,
        "height_index": r7_positive.HEIGHT_INDEX,
        "forward_index": r7_positive.FORWARD_INDEX,
        "minimum_connected_confidence2_pixels": 2,
        "minimum_component_max_height_m": 0.08,
        "maximum_component_min_forward_m": 2.0,
    },
    "stronger_cells_any": [
        {"pixel_index": 3, "height_index": 0, "forward_index": 2, "minimum_connected_confidence2_pixels": 16, "minimum_component_max_height_m": 0.08, "maximum_component_min_forward_m": 2.0},
        {"pixel_index": 0, "height_index": 1, "forward_index": 2, "minimum_connected_confidence2_pixels": 2, "minimum_component_max_height_m": 0.15, "maximum_component_min_forward_m": 2.0},
        {"pixel_index": 0, "height_index": 0, "forward_index": 1, "minimum_connected_confidence2_pixels": 2, "minimum_component_max_height_m": 0.08, "maximum_component_min_forward_m": 1.5},
    ],
    "preserve_r6_occupied": True,
    "weak_base_positive_becomes": "UNKNOWN",
    "candidate_positive_subset_of_base_positive": True,
    "clear_output_allowed": False,
    "unknown_is_negative": False,
    "training_steps": 0,
}


class AbstentionCandidateError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AbstentionCandidateError(code, message)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R11_ABSTENTION_SEAL_COLLISION", "caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R11_ABSTENTION_RECORD_INVALID", "record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and observed == adapter.canonical_sha256(record),
        "R11_ABSTENTION_SEAL_DRIFT",
        "record schema or seal drift",
    )
    record["content_sha256"] = observed
    return record


def _validate_hits(value: Any) -> list[list[list[bool]]]:
    require(isinstance(value, list) and len(value) == 4, "R11_ABSTENTION_GRID_INVALID", "occupied grid pixel axis drift")
    for pixel in value:
        require(isinstance(pixel, list) and len(pixel) == 3, "R11_ABSTENTION_GRID_INVALID", "occupied grid height axis drift")
        for height in pixel:
            require(
                isinstance(height, list) and len(height) == 3 and all(isinstance(item, bool) for item in height),
                "R11_ABSTENTION_GRID_INVALID",
                "occupied grid forward axis or value drift",
            )
    return value


def state_from_feature(feature: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Apply the frozen candidate without accepting truth, label, outcome, or FARO inputs."""

    require(isinstance(feature, Mapping) and set(feature) == EXPECTED_FEATURE_KEYS, "R11_ABSTENTION_FEATURE_SURFACE", "feature surface drift or result-side field supplied")
    base_state = feature.get("r6_state")
    require(base_state in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R11_ABSTENTION_BASE_STATE", "R6 base state invalid")
    if base_state == "OCCUPIED_OBSERVED":
        return "OCCUPIED_OBSERVED", ["R6_PRIOR_OCCUPIED_PRESERVED"]
    if feature.get("query_receipt") is None:
        require(
            feature.get("occupied_hits") is None
            and feature.get("positive_obstacle_veto") is None
            and feature.get("far_fractions") is None,
            "R11_ABSTENTION_UNAVAILABLE_QUERY_DRIFT",
            "unavailable query gained source evidence",
        )
        return "UNKNOWN", ["SOURCE_QUERY_FRAME_UNAVAILABLE", "CLEAR_OUTPUT_DISABLED"]
    hits = _validate_hits(feature.get("occupied_hits"))
    base_positive = bool(hits[r7_positive.PIXEL_INDEX][r7_positive.HEIGHT_INDEX][r7_positive.FORWARD_INDEX])
    stronger_any = any(bool(hits[pixel][height][forward]) for pixel, height, forward in STRONGER_CELLS_ANY)
    require(not stronger_any or base_positive, "R11_ABSTENTION_SUBSET_DRIFT", "stronger positive is not a subset of the frozen base positive")
    require(feature.get("positive_obstacle_veto") == base_positive, "R11_ABSTENTION_VETO_DRIFT", "positive veto is not bound to the frozen base positive")
    if stronger_any:
        return "OCCUPIED_OBSERVED", ["FROZEN_R7_ADJACENT_MARGIN_POSITIVE"]
    if base_positive:
        return "UNKNOWN", ["R11_WEAK_COMPONENT_ABSTAINED", "CLEAR_OUTPUT_DISABLED"]
    return "UNKNOWN", ["NO_FROZEN_MARGIN_POSITIVE_EVIDENCE", "CLEAR_OUTPUT_DISABLED"]


def build_abstention_bundle(source_frame_record: Mapping[str, Any]) -> dict[str, Any]:
    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    rows = []
    abstained = 0
    base_positive_count = 0
    for feature in source["query_features"]:
        state, reasons = state_from_feature(feature)
        hits = feature.get("occupied_hits")
        base_positive = feature["r6_state"] == "OCCUPIED_OBSERVED" or (
            isinstance(hits, list)
            and bool(hits[r7_positive.PIXEL_INDEX][r7_positive.HEIGHT_INDEX][r7_positive.FORWARD_INDEX])
        )
        base_positive_count += int(base_positive)
        abstained += int(base_positive and state == "UNKNOWN")
        rows.append(
            _seal(
                {
                    "schema": QUERY_SCHEMA,
                    "factor_id": FACTOR_ID,
                    "base_factor_id": BASE_FACTOR_ID,
                    "physical_frame_id": source["physical_frame_id"],
                    "query_id": feature["query_id"],
                    "grid_index": feature["grid_index"],
                    "source_frame_record_sha256": source["content_sha256"],
                    "state": state,
                    "reason_codes": reasons,
                    "clear_output_allowed": False,
                    "unknown_is_negative": False,
                }
            )
        )
    counts = Counter(row["state"] for row in rows)
    return validate_abstention_bundle(
        _seal(
            {
                "schema": SCHEMA,
                "factor_id": FACTOR_ID,
                "base_factor_id": BASE_FACTOR_ID,
                "physical_frame_id": source["physical_frame_id"],
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "source_frame_record_sha256": source["content_sha256"],
                "frozen_algorithm": FROZEN_ALGORITHM,
                "query_results": rows,
                "base_positive_count": base_positive_count,
                "candidate_positive_count": int(counts["OCCUPIED_OBSERVED"]),
                "abstained_base_positive_count": abstained,
                "state_counts": {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "candidate_positive_subset_of_base_positive": True,
                "clear_output_allowed": False,
                "unknown_is_negative": False,
                "truth_inputs": 0,
                "training_steps": 0,
                "network_requests": 0,
                "claim_status": "DEVELOPMENT_CANDIDATE_REQUIRES_FRESH_CONFIRMATION",
            }
        )
    )


def validate_abstention_bundle(value: Any) -> dict[str, Any]:
    bundle = _validate_seal(value, SCHEMA)
    require(
        bundle.get("factor_id") == FACTOR_ID
        and bundle.get("base_factor_id") == BASE_FACTOR_ID
        and bundle.get("frozen_algorithm") == FROZEN_ALGORITHM,
        "R11_ABSTENTION_IDENTITY_DRIFT",
        "factor identity or frozen algorithm drift",
    )
    require(
        bundle.get("candidate_positive_subset_of_base_positive") is True
        and bundle.get("clear_output_allowed") is False
        and bundle.get("unknown_is_negative") is False
        and bundle.get("truth_inputs") == bundle.get("training_steps") == bundle.get("network_requests") == 0
        and bundle.get("claim_status") == "DEVELOPMENT_CANDIDATE_REQUIRES_FRESH_CONFIRMATION",
        "R11_ABSTENTION_AUTHORITY_DRIFT",
        "candidate exceeded development-only authority",
    )
    rows = bundle.get("query_results")
    require(isinstance(rows, list) and len(rows) == 9, "R11_ABSTENTION_QUERY_COUNT", "bundle must retain nine queries")
    counts: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        row = _validate_seal(raw, QUERY_SCHEMA)
        require(
            row.get("factor_id") == FACTOR_ID
            and row.get("base_factor_id") == BASE_FACTOR_ID
            and row.get("physical_frame_id") == bundle["physical_frame_id"]
            and row.get("source_frame_record_sha256") == bundle["source_frame_record_sha256"]
            and row.get("grid_index") == index,
            "R11_ABSTENTION_QUERY_LINEAGE",
            "query lineage or order drift",
        )
        require(
            row.get("state") in {"OCCUPIED_OBSERVED", "UNKNOWN"}
            and row.get("clear_output_allowed") is False
            and row.get("unknown_is_negative") is False
            and isinstance(row.get("reason_codes"), list)
            and bool(row["reason_codes"]),
            "R11_ABSTENTION_QUERY_AUTHORITY",
            "query emitted a forbidden state or metadata",
        )
        counts[row["state"]] += 1
    expected_counts = {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}
    require(
        bundle.get("state_counts") == expected_counts
        and expected_counts["CLEAR_OBSERVED"] == 0
        and bundle.get("candidate_positive_count") == expected_counts["OCCUPIED_OBSERVED"]
        and bundle.get("base_positive_count") == bundle.get("candidate_positive_count") + bundle.get("abstained_base_positive_count"),
        "R11_ABSTENTION_COUNTS",
        "bundle state or abstention counts drift",
    )
    return bundle


__all__ = [
    "AbstentionCandidateError",
    "BASE_FACTOR_ID",
    "FACTOR_ID",
    "FROZEN_ALGORITHM",
    "build_abstention_bundle",
    "state_from_feature",
    "validate_abstention_bundle",
]
