#!/usr/bin/env python3
"""Evaluate the frozen DepthART R2 task-quality contract after activation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_depthart_task_preserving_deployment_r2_quality_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2"
STATES = {"CLEAR", "OCCUPIED", "UNKNOWN_GROUND"}
METRICS = (
    "known_coverage",
    "clearance_mae_m",
    "false_clear_all_known",
    "false_clear_given_occupied",
    "false_block_given_clear",
    "temporal_clearance_delta_mae_m",
    "geometry_transition_agreement",
    "valid_to_unknown_rate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _validate_decision(decision: dict[str, Any], label: str) -> None:
    _require(decision.get("state") in STATES, f"invalid {label} state")
    clearance = decision.get("clearance_m")
    if decision["state"] == "UNKNOWN_GROUND":
        _require(clearance is None, f"{label} UNKNOWN_GROUND clearance must be null")
    else:
        _require(_finite(clearance), f"{label} known clearance must be finite")


def _flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, str, str]] = set()
    expected_bands: tuple[str, ...] | None = None
    for row in rows:
        for key in ("parent_id", "session_id", "frame_id"):
            _require(isinstance(row.get(key), str) and row[key], f"missing {key}")
        _require(isinstance(row.get("timestamp_ns"), int), "timestamp_ns must be an integer")
        frame_key = (row["parent_id"], row["session_id"], row["frame_id"])
        _require(frame_key not in seen_frames, f"duplicate frame: {frame_key}")
        seen_frames.add(frame_key)
        decisions = row.get("decisions")
        _require(isinstance(decisions, list) and decisions, "frame decisions must be non-empty")
        bands = tuple(str(item.get("band")) for item in decisions)
        _require(len(bands) == len(set(bands)), f"duplicate band in frame: {frame_key}")
        if expected_bands is None:
            expected_bands = bands
        _require(bands == expected_bands, "band order or identity drift")
        for item in decisions:
            for arm in ("truth", "reference", "candidate"):
                _require(isinstance(item.get(arm), dict), f"missing {arm} decision")
                _validate_decision(item[arm], arm)
            flat.append({
                "parent_id": row["parent_id"],
                "session_id": row["session_id"],
                "frame_id": row["frame_id"],
                "timestamp_ns": row["timestamp_ns"],
                "band": item["band"],
                "truth": item["truth"],
                "reference": item["reference"],
                "candidate": item["candidate"],
            })
    _require(bool(flat), "no decisions")
    return flat


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _metrics(records: list[dict[str, Any]], arm: str) -> dict[str, float | None]:
    truth_known = [record for record in records if record["truth"]["state"] != "UNKNOWN_GROUND"]
    pred_known = [record for record in records if record[arm]["state"] != "UNKNOWN_GROUND"]
    paired_known = [
        record for record in truth_known if record[arm]["state"] != "UNKNOWN_GROUND"
    ]
    occupied = [record for record in paired_known if record["truth"]["state"] == "OCCUPIED"]
    clear = [record for record in paired_known if record["truth"]["state"] == "CLEAR"]
    false_clear = [record for record in paired_known if record["truth"]["state"] == "OCCUPIED" and record[arm]["state"] == "CLEAR"]
    false_block = [record for record in paired_known if record["truth"]["state"] == "CLEAR" and record[arm]["state"] == "OCCUPIED"]

    temporal_errors: list[float] = []
    transition_matches: list[bool] = []
    timelines: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        timelines[(record["session_id"], str(record["band"]))].append(record)
    for timeline in timelines.values():
        timeline.sort(key=lambda item: (item["timestamp_ns"], item["frame_id"]))
        for previous, current in zip(timeline, timeline[1:]):
            truth_transition = (previous["truth"]["state"], current["truth"]["state"])
            pred_transition = (previous[arm]["state"], current[arm]["state"])
            transition_matches.append(truth_transition == pred_transition)
            if all(
                decision["state"] != "UNKNOWN_GROUND"
                for decision in (previous["truth"], current["truth"], previous[arm], current[arm])
            ):
                truth_delta = current["truth"]["clearance_m"] - previous["truth"]["clearance_m"]
                pred_delta = current[arm]["clearance_m"] - previous[arm]["clearance_m"]
                temporal_errors.append(abs(pred_delta - truth_delta))

    return {
        "known_coverage": _ratio(len(pred_known), len(records)),
        "clearance_mae_m": _mean(
            abs(record[arm]["clearance_m"] - record["truth"]["clearance_m"])
            for record in paired_known
        ),
        "false_clear_all_known": _ratio(len(false_clear), len(paired_known)),
        "false_clear_given_occupied": _ratio(len(false_clear), len(occupied)),
        "false_block_given_clear": _ratio(len(false_block), len(clear)),
        "temporal_clearance_delta_mae_m": _mean(temporal_errors),
        "geometry_transition_agreement": _mean(float(value) for value in transition_matches),
        "valid_to_unknown_rate": _ratio(
            sum(record[arm]["state"] == "UNKNOWN_GROUND" for record in truth_known),
            len(truth_known),
        ),
    }


def _aggregate(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_parent[record["parent_id"]].append(record)
        by_session[record["session_id"]].append(record)
    parent_metrics = {key: _metrics(value, arm) for key, value in sorted(by_parent.items())}
    session_metrics = {key: _metrics(value, arm) for key, value in sorted(by_session.items())}

    def macro(groups: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
        return {
            metric: (
                statistics.fmean(float(group[metric]) for group in groups.values())
                if groups and all(_finite(group[metric]) for group in groups.values())
                else None
            )
            for metric in METRICS
        }

    return {
        "pooled": _metrics(records, arm),
        "parent_macro": macro(parent_metrics),
        "session_macro": macro(session_metrics),
        "worst_parent": {
            metric: (
                max(float(group[metric]) for group in parent_metrics.values())
                if parent_metrics and all(_finite(group[metric]) for group in parent_metrics.values())
                else None
            )
            for metric in METRICS
        },
        "by_parent": parent_metrics,
        "by_session": session_metrics,
    }


def _le(value: Any, ceiling: float) -> bool:
    return _finite(value) and float(value) <= ceiling


def _ge(value: Any, floor: float) -> bool:
    return _finite(value) and float(value) >= floor


def evaluate(protocol: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    _require(payload.get("schema") == SCHEMA, "payload schema mismatch")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "payload protocol id mismatch")
    rows = payload.get("rows")
    _require(isinstance(rows, list), "rows must be a list")
    records = _flatten(rows)
    reference = _aggregate(records, "reference")
    candidate = _aggregate(records, "candidate")
    gates = protocol["gates"]
    ref = reference["pooled"]
    cand = candidate["pooled"]
    completeness = all(
        _finite(aggregate[metric])
        for arm in (reference, candidate)
        for aggregate_name in ("pooled", "parent_macro", "session_macro", "worst_parent")
        for aggregate in (arm[aggregate_name],)
        for metric in METRICS
    )
    absolute = {
        "known_coverage": _ge(cand["known_coverage"], gates["known_coverage_min"]),
        "clearance_mae_m": _le(cand["clearance_mae_m"], gates["clearance_mae_m_max"]),
        "false_clear_all_known": _le(cand["false_clear_all_known"], gates["false_clear_all_known_max"]),
        "false_clear_given_occupied_finite": _finite(cand["false_clear_given_occupied"]),
        "false_block_given_clear": _le(cand["false_block_given_clear"], gates["false_block_given_clear_max"]),
        "temporal_clearance_delta_mae_m": _le(cand["temporal_clearance_delta_mae_m"], gates["temporal_clearance_delta_mae_m_max"]),
        "geometry_transition_agreement": _ge(cand["geometry_transition_agreement"], gates["geometry_transition_agreement_min"]),
        "valid_to_unknown_rate": _le(cand["valid_to_unknown_rate"], gates["valid_to_unknown_rate_max"]),
        "worst_parent_false_clear_all_known": _le(candidate["worst_parent"]["false_clear_all_known"], gates["worst_parent_false_clear_all_known_max"]),
    }
    noninferiority = {
        "known_coverage": _ge(cand["known_coverage"], ref["known_coverage"] - gates["known_coverage_decrease_max"]) if _finite(ref["known_coverage"]) else False,
        "clearance_mae_m": _le(cand["clearance_mae_m"], ref["clearance_mae_m"] + gates["clearance_mae_m_increase_max"]) if _finite(ref["clearance_mae_m"]) else False,
        "false_clear_all_known": _le(cand["false_clear_all_known"], ref["false_clear_all_known"] + gates["false_clear_all_known_increase_max"]) if _finite(ref["false_clear_all_known"]) else False,
        "false_clear_given_occupied": _le(cand["false_clear_given_occupied"], ref["false_clear_given_occupied"] + gates["false_clear_given_occupied_increase_max"]) if _finite(ref["false_clear_given_occupied"]) else False,
        "false_block_given_clear": _le(cand["false_block_given_clear"], ref["false_block_given_clear"] + gates["false_block_given_clear_increase_max"]) if _finite(ref["false_block_given_clear"]) else False,
        "temporal_clearance_delta_mae_m": _le(cand["temporal_clearance_delta_mae_m"], ref["temporal_clearance_delta_mae_m"] + gates["temporal_clearance_delta_mae_m_increase_max"]) if _finite(ref["temporal_clearance_delta_mae_m"]) else False,
        "geometry_transition_agreement": _ge(cand["geometry_transition_agreement"], ref["geometry_transition_agreement"] - gates["geometry_transition_agreement_decrease_max"]) if _finite(ref["geometry_transition_agreement"]) else False,
        "valid_to_unknown_rate": _le(cand["valid_to_unknown_rate"], ref["valid_to_unknown_rate"] + gates["valid_to_unknown_rate_increase_max"]) if _finite(ref["valid_to_unknown_rate"]) else False,
    }
    passed = completeness and all(absolute.values()) and all(noninferiority.values())
    return {
        "schema": "blindassist_depthart_task_preserving_deployment_r2_quality_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "FAIL",
        "terminal": protocol["sequence"]["quality_pass_terminal" if passed else "quality_fail_terminal"],
        "counts": {
            "frames": len(rows),
            "decisions": len(records),
            "parents": len({record["parent_id"] for record in records}),
            "sessions": len({record["session_id"] for record in records}),
        },
        "reference": reference,
        "candidate": candidate,
        "gates": {"aggregation_complete": completeness, "absolute": absolute, "noninferiority": noninferiority},
        "downstream": {
            "candidate_partition_performance": "ELIGIBLE" if passed else "NOT_AUTHORIZED",
            "strict_g4d": "NEGATIVE_TERMINAL_UNCHANGED",
            "da2_replacement": "NOT_AUTHORIZED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--activation-receipt", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    activation = _load(args.activation_receipt)
    _require(activation.get("protocol_id") == PROTOCOL_ID, "activation protocol mismatch")
    _require(activation.get("status") == "OUTCOME_ACCESS_ACTIVATED", "outcome access is not activated")
    _require(activation.get("execution_authorized") is True, "execution is not authorized")
    result = evaluate(_load(args.protocol), _load(args.input))
    result["identities"] = {
        "protocol_sha256": _sha256(args.protocol),
        "activation_receipt_sha256": _sha256(args.activation_receipt),
        "input_sha256": _sha256(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
