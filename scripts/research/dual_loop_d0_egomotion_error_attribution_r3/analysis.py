from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Iterable

from contract import (
    DIAGNOSTIC_METRICS,
    METRIC_ORDER,
    PRESELECTED_METRICS,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    median,
)


GROUPS = ("CORRECT", "WRONG_SIGNED")
TARGETS = ("track-000", "track-001")
TRUTH_STATES = ("approaching", "receding")
BURDEN_TRANSFORMS = {
    "median_abs_sensor_approach_component_mps": "IDENTITY",
    "median_abs_person_approach_component_mps": "IDENTITY",
    "median_flow_score_mad_per_s": "IDENTITY",
    "median_surviving_tracks": "NEGATE",
}


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _burden(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if not _finite(value):
        return None
    result = float(value)
    return -result if BURDEN_TRANSFORMS.get(field) == "NEGATE" else result


def ordinary_cliff(wrong: Iterable[float], correct: Iterable[float]) -> float | None:
    wrong_values, correct_values = list(wrong), list(correct)
    if not wrong_values or not correct_values:
        return None
    total = 0
    for first in wrong_values:
        for second in correct_values:
            total += 1 if first > second else -1 if first < second else 0
    return total / (len(wrong_values) * len(correct_values))


def _weighted_cliff(
    wrong: list[tuple[float, float]], correct: list[tuple[float, float]]
) -> float | None:
    if not wrong or not correct:
        return None
    return sum(
        wrong_weight
        * correct_weight
        * (1 if wrong_value > correct_value else -1 if wrong_value < correct_value else 0)
        for wrong_value, wrong_weight in wrong
        for correct_value, correct_weight in correct
    )


def _missing_reason(row: dict[str, Any], field: str) -> str:
    value = row.get(f"{field}_missing_reason")
    return value if isinstance(value, str) and value else "UNSPECIFIED_NULL"


def _support_cell(
    rows: list[dict[str, Any]],
    field: str,
    *,
    minimum_total: int = 8,
    minimum_finite: int = 8,
    minimum_fraction: float = 0.8,
    maximum_fraction_difference: float = 0.1,
) -> dict[str, Any]:
    selected = {
        group: [row for row in rows if row["primary_error_partition"] == group]
        for group in GROUPS
    }
    total_n = {group: len(selected[group]) for group in GROUPS}
    finite_n = {
        group: sum(_burden(row, field) is not None for row in selected[group])
        for group in GROUPS
    }
    finite_fraction = {
        group: finite_n[group] / total_n[group] if total_n[group] else 0.0
        for group in GROUPS
    }
    reasons = {
        group: dict(
            sorted(
                Counter(
                    _missing_reason(row, field)
                    for row in selected[group]
                    if _burden(row, field) is None
                ).items()
            )
        )
        for group in GROUPS
    }
    difference = abs(
        finite_fraction["WRONG_SIGNED"] - finite_fraction["CORRECT"]
    )
    failures: list[str] = []
    for group in GROUPS:
        if total_n[group] < minimum_total:
            failures.append(f"{group}_TOTAL_N_LT_{minimum_total}")
        if finite_n[group] < minimum_finite:
            failures.append(f"{group}_FINITE_N_LT_{minimum_finite}")
        if finite_fraction[group] < minimum_fraction:
            failures.append(f"{group}_FINITE_FRACTION_LT_{minimum_fraction}")
    if difference > maximum_fraction_difference:
        failures.append("FINITE_FRACTION_DIFFERENCE_GT_0.1")
    return {
        "total_n": total_n,
        "finite_n": finite_n,
        "finite_fraction": finite_fraction,
        "missing_reason_counts": reasons,
        "absolute_group_finite_fraction_difference": difference,
        "evaluable": not failures,
        "not_evaluable_reasons": failures,
    }


def _delta_cell(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = {
        group: [
            value
            for row in rows
            if row["primary_error_partition"] == group
            and (value := _burden(row, field)) is not None
        ]
        for group in GROUPS
    }
    value = ordinary_cliff(values["WRONG_SIGNED"], values["CORRECT"])
    return {
        "delta": value,
        "evaluable": value is not None,
        "reason": None if value is not None else "EMPTY_FINITE_GROUP",
    }


def _component_cell(
    rows: list[dict[str, Any]], field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = _support_cell(rows, field)
    occupied = {
        group: {
            row["overlap_component_id"]
            for row in rows
            if row["primary_error_partition"] == group
        }
        for group in GROUPS
    }
    finite_components = {
        group: {
            row["overlap_component_id"]
            for row in rows
            if row["primary_error_partition"] == group
            and _burden(row, field) is not None
        }
        for group in GROUPS
    }
    fraction = {
        group: len(finite_components[group]) / len(occupied[group])
        if occupied[group]
        else 0.0
        for group in GROUPS
    }
    component_difference = abs(fraction["WRONG_SIGNED"] - fraction["CORRECT"])
    failures = list(support["not_evaluable_reasons"])
    for group in GROUPS:
        if len(finite_components[group]) < 8:
            failures.append(f"{group}_FINITE_COMPONENTS_LT_8")
        if fraction[group] < 0.8:
            failures.append(f"{group}_FINITE_COMPONENT_FRACTION_LT_0.8")
    if component_difference > 0.1:
        failures.append("FINITE_COMPONENT_FRACTION_DIFFERENCE_GT_0.1")
    support.update(
        {
            "total_occupied_components": {
                group: len(occupied[group]) for group in GROUPS
            },
            "finite_components": {
                group: len(finite_components[group]) for group in GROUPS
            },
            "finite_component_fraction": fraction,
            "absolute_group_finite_component_fraction_difference": component_difference,
            "evaluable": not failures,
            "not_evaluable_reasons": failures,
        }
    )
    weighted: dict[str, list[tuple[float, float]]] = {}
    for group in GROUPS:
        component_counts = Counter(
            row["overlap_component_id"]
            for row in rows
            if row["primary_error_partition"] == group
            and _burden(row, field) is not None
        )
        component_count = len(component_counts)
        weighted[group] = [
            (
                _burden(row, field),
                1.0
                / (
                    component_count
                    * component_counts[row["overlap_component_id"]]
                ),
            )
            for row in rows
            if row["primary_error_partition"] == group
            and _burden(row, field) is not None
        ]
    value = (
        _weighted_cliff(weighted["WRONG_SIGNED"], weighted["CORRECT"])
        if support["evaluable"]
        else None
    )
    delta = {
        "delta": value,
        "evaluable": value is not None,
        "reason": None if value is not None else "COMPONENT_SUPPORT_NOT_EVALUABLE",
    }
    return support, delta


def _block_cells(
    rows: list[dict[str, Any]], field: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    blocks = sorted({row["time_block_id_60s"] for row in rows})
    candidate: list[int] = []
    support: dict[str, Any] = {}
    per_block: dict[str, Any] = {}
    for block in blocks:
        block_rows = [row for row in rows if row["time_block_id_60s"] == block]
        totals = Counter(row["primary_error_partition"] for row in block_rows)
        if all(totals[group] >= 8 for group in GROUPS):
            candidate.append(block)
            cell = _support_cell(block_rows, field)
            support[str(block)] = cell
            delta = _delta_cell(block_rows, field)
            if not cell["evaluable"]:
                delta = {
                    "delta": None,
                    "evaluable": False,
                    "reason": "BLOCK_SUPPORT_NOT_EVALUABLE",
                }
            per_block[str(block)] = delta
    all_support = len(candidate) >= 4 and all(
        support[str(block)]["evaluable"] for block in candidate
    )
    if not all_support:
        return (
            {
                "candidate_block_ids": candidate,
                "minimum_common_evaluable_blocks": 4,
                "cells": support,
                "evaluable": False,
                "not_evaluable_reasons": (
                    ["COMMON_EVALUABLE_BLOCKS_LT_4"] if len(candidate) < 4 else []
                )
                + [
                    f"BLOCK_{block}_SUPPORT_FAILED"
                    for block in candidate
                    if not support[str(block)]["evaluable"]
                ],
            },
            {
                "delta": None,
                "evaluable": False,
                "reason": "BLOCK_SUPPORT_NOT_EVALUABLE",
            },
            per_block,
        )
    weighted: dict[str, list[tuple[float, float]]] = {}
    for group in GROUPS:
        block_counts = Counter(
            row["time_block_id_60s"]
            for row in rows
            if row["time_block_id_60s"] in candidate
            and row["primary_error_partition"] == group
            and _burden(row, field) is not None
        )
        weighted[group] = [
            (
                _burden(row, field),
                1.0 / (len(candidate) * block_counts[row["time_block_id_60s"]]),
            )
            for row in rows
            if row["time_block_id_60s"] in candidate
            and row["primary_error_partition"] == group
            and _burden(row, field) is not None
        ]
    value = _weighted_cliff(weighted["WRONG_SIGNED"], weighted["CORRECT"])
    return (
        {
            "candidate_block_ids": candidate,
            "minimum_common_evaluable_blocks": 4,
            "cells": support,
            "evaluable": True,
            "not_evaluable_reasons": [],
        },
        {"delta": value, "evaluable": True, "reason": None},
        per_block,
    )


def _map_cells(
    rows: list[dict[str, Any]],
    field: str,
    values: Iterable[Any],
    selector: Callable[[dict[str, Any]], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    support: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for value in values:
        selected = [row for row in rows if selector(row) == value]
        support[str(value)] = _support_cell(selected, field)
        delta = _delta_cell(selected, field)
        if not support[str(value)]["evaluable"]:
            delta = {
                "delta": None,
                "evaluable": False,
                "reason": "MARGINAL_SUPPORT_NOT_EVALUABLE",
            }
        deltas[str(value)] = delta
    return support, deltas


def _metric_analysis(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    role = "PRESELECTED" if field in PRESELECTED_METRICS else "DIAGNOSTIC_ONLY"
    pooled = _support_cell(rows, field)
    event_delta = _delta_cell(rows, field) if pooled["evaluable"] else {
        "delta": None,
        "evaluable": False,
        "reason": "POOLED_SUPPORT_NOT_EVALUABLE",
    }
    target_support, target_deltas = _map_cells(
        rows, field, TARGETS, lambda row: row["target_id"]
    )
    truth_support, truth_deltas = _map_cells(
        rows, field, TRUTH_STATES, lambda row: row["truth_state"]
    )
    region_values = sorted({row["anchor_region"] for row in rows})
    required_regions = [
        region
        for region in region_values
        if all(
            sum(
                row["anchor_region"] == region
                and row["primary_error_partition"] == group
                for row in rows
            )
            >= 8
            for group in GROUPS
        )
    ]
    region_support, region_deltas = _map_cells(
        rows, field, required_regions, lambda row: row["anchor_region"]
    )
    component_support, component_delta = _component_cell(rows, field)
    block_support, block_delta, common_blocks = _block_cells(rows, field)
    required_evaluable = (
        pooled["evaluable"]
        and all(cell["evaluable"] for cell in target_support.values())
        and len(required_regions) >= 2
        and all(cell["evaluable"] for cell in region_support.values())
        and all(cell["evaluable"] for cell in truth_support.values())
        and component_support["evaluable"]
        and block_support["evaluable"]
    )
    required_delta_cells = (
        [component_delta, block_delta]
        + list(target_deltas.values())
        + list(region_deltas.values())
        + list(truth_deltas.values())
        + list(common_blocks.values())
    )
    contradiction = (not required_evaluable) or any(
        cell["delta"] is not None and cell["delta"] <= -0.33
        for cell in required_delta_cells
    )
    if (
        event_delta["delta"] is not None
        and component_delta["delta"] is not None
        and event_delta["delta"] * component_delta["delta"] < 0
    ):
        contradiction = True
    robust = (
        role == "PRESELECTED"
        and required_evaluable
        and event_delta["delta"] >= 0.33
        and component_delta["delta"] >= 0.33
        and block_delta["delta"] >= 0.33
        and all(cell["delta"] > 0 for cell in target_deltas.values())
        and len(region_deltas) >= 2
        and sum(cell["delta"] > 0 for cell in region_deltas.values()) >= 2
        and not any(cell["delta"] <= -0.33 for cell in region_deltas.values())
        and all(cell["delta"] >= 0.33 for cell in truth_deltas.values())
        and len(common_blocks) >= 4
        and all(cell["delta"] > -0.33 for cell in common_blocks.values())
        and event_delta["delta"] * component_delta["delta"] >= 0
    )
    if role == "DIAGNOSTIC_ONLY":
        required_evaluable = False
        robust = False
        contradiction = False
    return {
        "field": field,
        "role": role,
        "burden_transform": BURDEN_TRANSFORMS.get(field, "IDENTITY"),
        "missingness": {
            "pooled": pooled,
            "targets": target_support,
            "regions": {
                "required_region_ids": required_regions,
                "cells": region_support,
            },
            "truth_states": truth_support,
            "component": component_support,
            "blocks": block_support,
        },
        "deltas": {
            "event_weighted": event_delta,
            "component_balanced": component_delta,
            "block_balanced": block_delta,
            "targets": target_deltas,
            "regions": region_deltas,
            "truth_states": truth_deltas,
            "common_blocks": common_blocks,
        },
        "globally_route_evaluable": required_evaluable,
        "robust_support": robust,
        "material_contradiction": contradiction,
    }


def analyze_event_table(
    rows: list[dict[str, Any]], event_table_sha256: str
) -> dict[str, Any]:
    metrics = {field: _metric_analysis(rows, field) for field in METRIC_ORDER}
    share_rows = [
        row
        for row in rows
        if row["primary_error_partition"] == "WRONG_SIGNED"
    ]
    finite_shares = [
        float(row["sensor_absolute_share"])
        for row in share_rows
        if _finite(row.get("sensor_absolute_share"))
    ]
    share_median = median(finite_shares)
    share_fraction = len(finite_shares) / len(share_rows) if share_rows else 0.0
    share_evaluable = len(finite_shares) >= 8 and share_fraction >= 0.8
    share_gate = {
        "field": "sensor_absolute_share",
        "group": "WRONG_SIGNED",
        "median": share_median,
        "total_n": len(share_rows),
        "finite_n": len(finite_shares),
        "finite_fraction": share_fraction,
        "evaluable": share_evaluable,
        "value": share_evaluable and share_median > 0.5,
    }
    primary = [metrics[field] for field in PRESELECTED_METRICS]
    global_route_evaluable = share_evaluable and all(
        result["globally_route_evaluable"] for result in primary
    )
    ego = metrics["median_abs_sensor_approach_component_mps"]
    person = metrics["median_abs_person_approach_component_mps"]
    temporal_direct = metrics["median_flow_score_mad_per_s"]
    temporal_support = metrics["median_surviving_tracks"]
    person_equal_or_larger = (
        person["deltas"]["event_weighted"]["delta"] is not None
        and ego["deltas"]["event_weighted"]["delta"] is not None
        and person["deltas"]["component_balanced"]["delta"] is not None
        and ego["deltas"]["component_balanced"]["delta"] is not None
        and (
            person["deltas"]["event_weighted"]["delta"]
            >= ego["deltas"]["event_weighted"]["delta"]
            or person["deltas"]["component_balanced"]["delta"]
            >= ego["deltas"]["component_balanced"]["delta"]
        )
    )
    person_competing = person["robust_support"] and person_equal_or_larger
    ego_candidate = (
        global_route_evaluable
        and share_gate["value"]
        and ego["robust_support"]
        and not person_competing
        and not ego["material_contradiction"]
    )
    temporal_contradictory = (
        temporal_direct["material_contradiction"]
        or temporal_support["material_contradiction"]
    )
    temporal_candidate = (
        global_route_evaluable
        and temporal_direct["robust_support"]
        and temporal_support["robust_support"]
        and not temporal_contradictory
    )
    if global_route_evaluable and ego_candidate and not temporal_candidate:
        provisional_exit = "EGO_CANARY_PRIORITY"
    elif global_route_evaluable and temporal_candidate and not ego_candidate:
        provisional_exit = "TEMPORAL_TREND_PRIORITY"
    else:
        provisional_exit = "NO_PRIORITY_IDENTIFIED"
    return {
        "schema_version": "blindassist.d0_analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "event_table_sha256": event_table_sha256,
        "event_count": len(rows),
        "primary_arm": "ROI_SPARSE_RADIAL_FLOW",
        "reference_arm": "BBOX_LOG_AREA_GROWTH",
        "metric_order": list(METRIC_ORDER),
        "metrics": metrics,
        "ego_share_gate": share_gate,
        "routing": {
            "global_route_evaluable": global_route_evaluable,
            "ego_contradictory": ego["material_contradiction"],
            "temporal_contradictory": temporal_contradictory,
            "person_material_support": person["robust_support"],
            "person_equal_or_larger": person_equal_or_larger,
            "person_competing": person_competing,
            "ego_candidate": ego_candidate,
            "temporal_candidate": temporal_candidate,
            "provisional_scientific_exit": provisional_exit,
        },
    }
