#!/usr/bin/env python3
"""Frozen fail-safe TARO positive-occupancy factor: OCCUPIED or UNKNOWN only."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


SCHEMA = "blindassist.taro.o1r.r7_positive_occupancy_factor_bundle.v1"
QUERY_SCHEMA = "blindassist.taro.o1r.r7_positive_occupancy_factor_query.v1"
FACTOR_ID = "R7_CONFIDENCE2_CONNECTED_COMPONENT_POSITIVE_OCCUPANCY_V1"
PROTOCOL_SHA256 = "1419070D09951AE7251C9832EF006C329F82D1DA1C46DB8F759ABBF6ECA11A01"
PIXEL_INDEX = 0
HEIGHT_INDEX = 0
FORWARD_INDEX = 2
FROZEN_THRESHOLDS = {
    "minimum_connected_confidence2_pixels": 2,
    "minimum_component_max_height_m": 0.08,
    "maximum_component_min_forward_m": 2.0,
}


class PositiveOccupancyFactorError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PositiveOccupancyFactorError(code, message)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R7_POSITIVE_SEAL_COLLISION", "caller supplied content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R7_POSITIVE_RECORD_INVALID", "record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == schema and isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "R7_POSITIVE_RECORD_HASH_DRIFT", "record seal/schema drift")
    record["content_sha256"] = observed
    return record


def state_from_feature(feature: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Apply the frozen positive rule without accepting truth or label inputs."""

    base = str(feature.get("r6_state"))
    require(base in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R7_POSITIVE_BASE_STATE_INVALID", "R6 base state invalid")
    if base == "OCCUPIED_OBSERVED":
        return "OCCUPIED_OBSERVED", ["R6_PRIOR_OCCUPIED_PRESERVED"]
    if feature.get("query_receipt") is None:
        return "UNKNOWN", ["SOURCE_QUERY_FRAME_UNAVAILABLE"]
    hits = feature.get("occupied_hits")
    require(isinstance(hits, list) and len(hits) == 4, "R7_POSITIVE_GRID_INVALID", "occupied grid invalid")
    try:
        occupied = bool(hits[PIXEL_INDEX][HEIGHT_INDEX][FORWARD_INDEX])
    except (IndexError, TypeError) as error:
        raise PositiveOccupancyFactorError("R7_POSITIVE_GRID_INVALID", "occupied grid invalid") from error
    if occupied:
        return "OCCUPIED_OBSERVED", ["FROZEN_CONFIDENCE2_COMPONENT_POSITIVE"]
    return "UNKNOWN", ["NO_FROZEN_POSITIVE_EVIDENCE", "CLEAR_OUTPUT_DISABLED"]


def build_positive_occupancy_factor(source_frame_record: Mapping[str, Any]) -> dict[str, Any]:
    """Build nine sealed factor states. The API deliberately has no truth side."""

    source = r7_canary.validate_source_frame_record(dict(source_frame_record))
    rows = []
    for feature in source["query_features"]:
        state, reasons = state_from_feature(feature)
        rows.append(
            _seal(
                {
                    "schema": QUERY_SCHEMA,
                    "factor_id": FACTOR_ID,
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
    return validate_positive_occupancy_factor(
        _seal(
            {
                "schema": SCHEMA,
                "factor_id": FACTOR_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "physical_frame_id": source["physical_frame_id"],
                "parent_id": source["parent_id"],
                "video_id": source["video_id"],
                "timestamp_token": source["timestamp_token"],
                "source_frame_record_sha256": source["content_sha256"],
                "frozen_thresholds": FROZEN_THRESHOLDS,
                "query_results": rows,
                "state_counts": {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "clear_output_allowed": False,
                "unknown_is_negative": False,
                "threshold_fit": False,
                "training_steps": 0,
                "network_requests": 0,
                "claim_status": "EXPERIMENTAL_RESEARCH_FACTOR_NOT_PROMOTED",
            }
        )
    )


def validate_positive_occupancy_factor(value: Any) -> dict[str, Any]:
    bundle = _validate_seal(value, SCHEMA)
    require(bundle.get("factor_id") == FACTOR_ID and bundle.get("protocol_sha256") == PROTOCOL_SHA256 and bundle.get("frozen_thresholds") == FROZEN_THRESHOLDS, "R7_POSITIVE_IDENTITY_DRIFT", "factor identity/threshold drift")
    require(bundle.get("clear_output_allowed") is False and bundle.get("unknown_is_negative") is False and bundle.get("threshold_fit") is False and bundle.get("training_steps") == bundle.get("network_requests") == 0, "R7_POSITIVE_AUTHORITY_DRIFT", "factor exceeded fail-safe authority")
    rows = bundle.get("query_results")
    require(isinstance(rows, list) and len(rows) == 9, "R7_POSITIVE_QUERY_COUNT_DRIFT", "factor must retain nine queries")
    counts = Counter()
    for index, raw in enumerate(rows):
        row = _validate_seal(raw, QUERY_SCHEMA)
        require(row.get("factor_id") == FACTOR_ID and row.get("physical_frame_id") == bundle["physical_frame_id"] and row.get("source_frame_record_sha256") == bundle["source_frame_record_sha256"] and row.get("grid_index") == index, "R7_POSITIVE_QUERY_IDENTITY_DRIFT", "factor query lineage/order drift")
        require(row.get("state") in {"OCCUPIED_OBSERVED", "UNKNOWN"} and row.get("clear_output_allowed") is False and row.get("unknown_is_negative") is False and isinstance(row.get("reason_codes"), list) and bool(row["reason_codes"]), "R7_POSITIVE_QUERY_AUTHORITY_DRIFT", "factor query emitted forbidden state/metadata")
        counts[row["state"]] += 1
    expected = {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}
    require(bundle.get("state_counts") == expected and expected["CLEAR_OBSERVED"] == 0, "R7_POSITIVE_STATE_COUNT_DRIFT", "factor state counts drift")
    return bundle


__all__ = [
    "FACTOR_ID",
    "FROZEN_THRESHOLDS",
    "PositiveOccupancyFactorError",
    "build_positive_occupancy_factor",
    "state_from_feature",
    "validate_positive_occupancy_factor",
]
