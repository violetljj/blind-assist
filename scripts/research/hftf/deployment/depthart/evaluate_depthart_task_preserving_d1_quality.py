#!/usr/bin/env python3
"""Evaluate the frozen DepthART D1 8x300 Development task-quality screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_depthart_task_preserving_d1_quality_payload_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN"
STATES = {"CLEAR", "OCCUPIED", "UNKNOWN_GROUND"}
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def validate_decision(decision: dict[str, Any], label: str) -> None:
    require(decision.get("state") in STATES, f"invalid {label} state")


def validate_clearance(value: dict[str, Any], label: str) -> None:
    valid = value.get("clearance_valid")
    clearance = value.get("clearance_m")
    require(isinstance(valid, bool), f"{label} clearance_valid must be boolean")
    require(finite(clearance) if valid else clearance is None, f"{label} clearance/value mismatch")


def flatten(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: list[dict[str, Any]] = []
    bands: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        for key in ("parent_id", "session_id", "frame_id"):
            require(isinstance(row.get(key), str) and row[key], f"missing {key}")
        require(isinstance(row.get("timestamp_ns"), int), "timestamp_ns must be an integer")
        require(row.get("orientation") == "portrait", "D1 frame must be display-upright portrait")
        frame_key = (row["parent_id"], row["session_id"], row["frame_id"])
        require(frame_key not in seen, f"duplicate frame: {frame_key}")
        seen.add(frame_key)
        frame_bands = row.get("bands")
        require(isinstance(frame_bands, list) and len(frame_bands) == 3, "three bands required")
        require(tuple(item.get("band") for item in frame_bands) == BANDS, "band order drift")
        for band in frame_bands:
            for arm in ("truth", "reference", "candidate"):
                require(isinstance(band.get(arm), dict), f"missing {arm} band payload")
                validate_clearance(band[arm], arm)
            band_record = {
                "parent_id": row["parent_id"], "session_id": row["session_id"],
                "frame_id": row["frame_id"], "timestamp_ns": row["timestamp_ns"],
                "band": band["band"],
                **{arm: band[arm] for arm in ("truth", "reference", "candidate")},
            }
            bands.append(band_record)
            horizon_cells = band.get("cells")
            require(isinstance(horizon_cells, list) and len(horizon_cells) == 3, "three horizons required")
            require(tuple(float(item.get("horizon_m", -1)) for item in horizon_cells) == HORIZONS,
                    "horizon order drift")
            for item in horizon_cells:
                for arm in ("truth", "reference", "candidate"):
                    require(isinstance(item.get(arm), dict), f"missing {arm} decision")
                    validate_decision(item[arm], arm)
                cells.append({
                    "parent_id": row["parent_id"], "session_id": row["session_id"],
                    "frame_id": row["frame_id"], "timestamp_ns": row["timestamp_ns"],
                    "band": band["band"], "horizon_m": float(item["horizon_m"]),
                    **{arm: item[arm] for arm in ("truth", "reference", "candidate")},
                })
    require(bool(cells) and len(cells) == len(rows) * 9, "exactly nine cells per frame required")
    require(len(bands) == len(rows) * 3, "exactly three band clearances per frame required")
    return cells, bands


def metrics(cells: list[dict[str, Any]], bands: list[dict[str, Any]], arm: str) -> dict[str, float | None]:
    truth_known = [row for row in cells if row["truth"]["state"] != "UNKNOWN_GROUND"]
    paired = [row for row in truth_known if row[arm]["state"] != "UNKNOWN_GROUND"]
    occupied = [row for row in paired if row["truth"]["state"] == "OCCUPIED"]
    clear = [row for row in paired if row["truth"]["state"] == "CLEAR"]
    false_clear = [row for row in paired if row["truth"]["state"] == "OCCUPIED" and row[arm]["state"] == "CLEAR"]
    false_block = [row for row in paired if row["truth"]["state"] == "CLEAR" and row[arm]["state"] == "OCCUPIED"]
    clearance_pairs = [row for row in bands if row["truth"]["clearance_valid"] and row[arm]["clearance_valid"]]

    temporal_errors: list[float] = []
    clearance_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bands:
        clearance_series[(row["parent_id"], row["session_id"], row["band"])].append(row)
    for timeline in clearance_series.values():
        timeline.sort(key=lambda item: (item["timestamp_ns"], item["frame_id"]))
        for previous, current in zip(timeline, timeline[1:]):
            if all(value["clearance_valid"] for value in
                   (previous["truth"], current["truth"], previous[arm], current[arm])):
                truth_delta = current["truth"]["clearance_m"] - previous["truth"]["clearance_m"]
                pred_delta = current[arm]["clearance_m"] - previous[arm]["clearance_m"]
                temporal_errors.append(abs(pred_delta - truth_delta))

    transition_matches: list[float] = []
    state_series: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        state_series[(row["parent_id"], row["session_id"], row["band"], row["horizon_m"])].append(row)
    for timeline in state_series.values():
        timeline.sort(key=lambda item: (item["timestamp_ns"], item["frame_id"]))
        for previous, current in zip(timeline, timeline[1:]):
            if all(value["state"] != "UNKNOWN_GROUND" for value in
                   (previous["truth"], current["truth"], previous[arm], current[arm])):
                transition_matches.append(float(
                    (previous["truth"]["state"], current["truth"]["state"])
                    == (previous[arm]["state"], current[arm]["state"])
                ))
    return {
        "known_coverage": ratio(len(paired), len(truth_known)),
        "clearance_mae_m": mean(abs(row[arm]["clearance_m"] - row["truth"]["clearance_m"])
                                for row in clearance_pairs),
        "false_clear_all_known": ratio(len(false_clear), len(paired)),
        "false_clear_given_occupied": ratio(len(false_clear), len(occupied)),
        "false_block_given_clear": ratio(len(false_block), len(clear)),
        "temporal_clearance_delta_mae_m": mean(temporal_errors),
        "geometry_transition_agreement": mean(transition_matches),
        "valid_to_unknown_rate": ratio(len(truth_known) - len(paired), len(truth_known)),
    }


def subset(records: list[dict[str, Any]], key: str, value: str) -> list[dict[str, Any]]:
    return [row for row in records if str(row[key]) == value]


def aggregate(cells: list[dict[str, Any]], bands: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    parents = sorted({str(row["parent_id"]) for row in cells})
    sessions = sorted({str(row["session_id"]) for row in cells})
    by_parent = {key: metrics(subset(cells, "parent_id", key), subset(bands, "parent_id", key), arm)
                 for key in parents}
    by_session = {key: metrics(subset(cells, "session_id", key), subset(bands, "session_id", key), arm)
                  for key in sessions}

    def macro(groups: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
        return {metric: (statistics.fmean(float(group[metric]) for group in groups.values())
                         if groups and all(finite(group[metric]) for group in groups.values()) else None)
                for metric in METRICS}

    higher_is_better = {"known_coverage", "geometry_transition_agreement"}
    worst_parent = {
        metric: ((min if metric in higher_is_better else max)(float(group[metric]) for group in by_parent.values())
                 if by_parent and all(finite(group[metric]) for group in by_parent.values()) else None)
        for metric in METRICS
    }
    by_grid = {}
    for band in BANDS:
        for horizon in HORIZONS:
            grid_cells = [row for row in cells if row["band"] == band and row["horizon_m"] == horizon]
            grid_bands = [row for row in bands if row["band"] == band]
            by_grid[f"{band}@{horizon:.1f}m"] = metrics(grid_cells, grid_bands, arm)
    return {
        "pooled": metrics(cells, bands, arm),
        "parent_macro": macro(by_parent),
        "session_macro": macro(by_session),
        "worst_parent": worst_parent,
        "by_parent": by_parent,
        "by_session": by_session,
        "by_grid": by_grid,
        "by_orientation": {"portrait": metrics(cells, bands, arm)},
    }


def le(value: Any, ceiling: float) -> bool:
    return finite(value) and float(value) <= ceiling


def ge(value: Any, floor: float) -> bool:
    return finite(value) and float(value) >= floor


def evaluate(protocol: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    require(payload.get("schema") == SCHEMA, "payload schema mismatch")
    require(payload.get("protocol_id") == PROTOCOL_ID, "payload protocol id mismatch")
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 2400, "D1 requires exactly 2400 frames")
    ordered_sessions = protocol.get("cohort", {}).get("ordered_sessions")
    require(isinstance(ordered_sessions, list) and len(ordered_sessions) == 8,
            "protocol requires eight ordered sessions")
    for session_index, expected in enumerate(ordered_sessions):
        block = rows[session_index * 300:(session_index + 1) * 300]
        require(all(str(row.get("parent_id")) == str(expected["visit_id"]) for row in block),
                "ordered parent mapping drift")
        require(all(str(row.get("session_id")) == str(expected["video_id"]) for row in block),
                "ordered session mapping drift")
        require([row.get("frame_index") for row in block] == list(range(300)),
                "ordered frame indices drift")
        stems_digest = hashlib.sha256(("\n".join(str(row.get("frame_id")) for row in block) + "\n").encode()).hexdigest().upper()
        require(stems_digest == expected["frame_stems_sha256"], "frozen frame stem schedule drift")
    cells, bands = flatten(rows)
    require(len({row["parent_id"] for row in cells}) == 8, "D1 requires exactly eight parents")
    require(len({row["session_id"] for row in cells}) == 8, "D1 requires exactly eight sessions")
    reference = aggregate(cells, bands, "reference")
    candidate = aggregate(cells, bands, "candidate")
    gates = protocol["gates"]
    ref, cand = reference["pooled"], candidate["pooled"]
    aggregate_complete = all(
        finite(arm[name][metric])
        for arm in (reference, candidate)
        for name in ("pooled", "parent_macro", "session_macro", "worst_parent")
        for metric in METRICS
    )
    strata_complete = all(
        finite(grid[metric])
        for arm in (reference, candidate)
        for grid in arm["by_grid"].values()
        for metric in METRICS
    ) and len(reference["by_grid"]) == len(candidate["by_grid"]) == 9
    completeness = aggregate_complete and strata_complete
    absolute = {
        "known_coverage": ge(cand["known_coverage"], gates["known_coverage_min"]),
        "clearance_mae_m": le(cand["clearance_mae_m"], gates["clearance_mae_m_max"]),
        "false_clear_all_known": le(cand["false_clear_all_known"], gates["false_clear_all_known_max"]),
        "false_clear_given_occupied_finite": finite(cand["false_clear_given_occupied"]),
        "false_block_given_clear": le(cand["false_block_given_clear"], gates["false_block_given_clear_max"]),
        "temporal_clearance_delta_mae_m": le(cand["temporal_clearance_delta_mae_m"], gates["temporal_clearance_delta_mae_m_max"]),
        "geometry_transition_agreement": ge(cand["geometry_transition_agreement"], gates["geometry_transition_agreement_min"]),
        "valid_to_unknown_rate": le(cand["valid_to_unknown_rate"], gates["valid_to_unknown_rate_max"]),
        "worst_parent_false_clear_all_known": le(candidate["worst_parent"]["false_clear_all_known"], gates["worst_parent_false_clear_all_known_max"]),
    }
    noninferiority = {
        "known_coverage": ge(cand["known_coverage"], ref["known_coverage"] - gates["known_coverage_decrease_max"]) if finite(ref["known_coverage"]) else False,
        "clearance_mae_m": le(cand["clearance_mae_m"], ref["clearance_mae_m"] + gates["clearance_mae_m_increase_max"]) if finite(ref["clearance_mae_m"]) else False,
        "false_clear_all_known": le(cand["false_clear_all_known"], ref["false_clear_all_known"] + gates["false_clear_all_known_increase_max"]) if finite(ref["false_clear_all_known"]) else False,
        "false_clear_given_occupied": le(cand["false_clear_given_occupied"], ref["false_clear_given_occupied"] + gates["false_clear_given_occupied_increase_max"]) if finite(ref["false_clear_given_occupied"]) else False,
        "false_block_given_clear": le(cand["false_block_given_clear"], ref["false_block_given_clear"] + gates["false_block_given_clear_increase_max"]) if finite(ref["false_block_given_clear"]) else False,
        "temporal_clearance_delta_mae_m": le(cand["temporal_clearance_delta_mae_m"], ref["temporal_clearance_delta_mae_m"] + gates["temporal_clearance_delta_mae_m_increase_max"]) if finite(ref["temporal_clearance_delta_mae_m"]) else False,
        "geometry_transition_agreement": ge(cand["geometry_transition_agreement"], ref["geometry_transition_agreement"] - gates["geometry_transition_agreement_decrease_max"]) if finite(ref["geometry_transition_agreement"]) else False,
        "valid_to_unknown_rate": le(cand["valid_to_unknown_rate"], ref["valid_to_unknown_rate"] + gates["valid_to_unknown_rate_increase_max"]) if finite(ref["valid_to_unknown_rate"]) else False,
    }
    passed = completeness and all(absolute.values()) and all(noninferiority.values())
    return {
        "schema": "blindassist_depthart_task_preserving_d1_quality_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if passed else "FAIL",
        "terminal": protocol["quality_pass_terminal" if passed else "quality_fail_terminal"],
        "counts": {"frames": len(rows), "cells": len(cells), "band_clearances": len(bands), "parents": 8, "sessions": 8},
        "reference": reference, "candidate": candidate,
        "gates": {"aggregation_complete": aggregate_complete, "required_strata_complete": strata_complete,
                  "absolute": absolute, "noninferiority": noninferiority},
        "downstream": {
            "single_r2_candidate_lock": "AUTHORIZED" if passed else "NOT_AUTHORIZED",
            "performance": "ELIGIBLE" if passed else "NOT_AUTHORIZED",
            "r2_cohort_access": "NONE",
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
    activation = load_json(args.activation_receipt)
    require(activation.get("protocol_id") == PROTOCOL_ID, "activation protocol mismatch")
    require(activation.get("status") == "OUTCOME_ACCESS_ACTIVATED", "outcome access is not activated")
    require(activation.get("execution_authorized") is True, "execution is not authorized")
    result = evaluate(load_json(args.protocol), load_json(args.input))
    result["identities"] = {
        "protocol_sha256": sha256(args.protocol),
        "activation_receipt_sha256": sha256(args.activation_receipt),
        "input_sha256": sha256(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "terminal": result["terminal"], "gates": result["gates"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
