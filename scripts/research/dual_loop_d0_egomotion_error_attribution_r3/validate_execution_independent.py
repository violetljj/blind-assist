#!/usr/bin/env python3
"""Independent D0-R3 execution validator.

This module deliberately does not import the producer, analysis, binding, or
runner implementation.  It recomputes the frozen statistical decision from the
canonical event table and verifies the producer package before any scientific
exit can be published.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from runtime_environment import validate_runtime_manifest
except ModuleNotFoundError:
    from scripts.research.dual_loop_d0_egomotion_error_attribution_r3.runtime_environment import (
        validate_runtime_manifest,
    )


PROTOCOL_ID = "D0_EGOMOTION_ERROR_ATTRIBUTION_R3"
PROTOCOL_SHA256 = "4412390fcfb4b4588600c368d3cb36a6ece875ec3f97ea7ef8bd051886f11064"
DEPENDENCY_SHA256 = "0377944df2abdeb6044d49182e1f4bc1908b4bf8ba40eb632a091b4d2d10dc7f"
ANALYSIS_SCHEMA = "blindassist.d0_analysis.v1"
PRODUCER_RECEIPT_SCHEMA = "blindassist.d0_producer_receipt.v1"
VALIDATION_SCHEMA = "blindassist.d0_execution_validation.v1"
EXECUTION_RECEIPT_SCHEMA = "blindassist.d0_execution_receipt.v1"
IMPLEMENTATION_LOCK_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.implementation_lock.v1"
)
ACTIVATION_SCHEMA = "blindassist.d0_egomotion_error_attribution.activation.v1"
IMPLEMENTATION_REVIEW_SCHEMA = (
    "blindassist.d0_egomotion_error_attribution.implementation_review.v1"
)

CORRECT = "CORRECT"
WRONG = "WRONG_SIGNED"
GROUPS = (CORRECT, WRONG)
TRUTH_STATES = ("approaching", "receding")
METRICS = (
    ("median_abs_sensor_approach_component_mps", "IDENTITY"),
    ("median_abs_person_approach_component_mps", "IDENTITY"),
    ("median_flow_score_mad_per_s", "IDENTITY"),
    ("median_surviving_tracks", "NEGATE"),
)
METRIC_FIELDS = tuple(field for field, _ in METRICS)
EGO_FIELD, PERSON_FIELD, TEMPORAL_FIELD, TEMPORAL_SUPPORT_FIELD = METRIC_FIELDS
DIAGNOSTIC_FIELDS = (
    "median_camera_translation_speed_mps",
    "p90_camera_translation_speed_mps",
    "median_camera_angular_speed_radps",
    "p90_camera_angular_speed_radps",
    "flow_sign_flip_fraction",
    "median_detected_features",
    "minimum_surviving_tracks",
    "median_occupied_quadrants",
    "finite_flow_coverage",
    "negative_log_duration_s",
    "median_abs_log_area_rate_per_s",
    "log_area_rate_mad_per_s",
    "median_center_speed_normalized_per_s",
    "center_velocity_mad_normalized_per_s",
    "median_forward_backward_error_px",
    "reference_arm_behavior",
)
ALL_METRIC_FIELDS = METRIC_FIELDS + DIAGNOSTIC_FIELDS

MIN_TOTAL = 8
MIN_FINITE = 8
MIN_FINITE_FRACTION = 0.8
MAX_FRACTION_DIFFERENCE = 0.1
MIN_COMPONENTS = 8
MIN_REGIONS = 2
MIN_BLOCKS = 4
SUPPORT_DELTA = 0.33
CONTRADICTION_DELTA = -0.33
BLOCK_IDS = tuple(range(6))
BLOCK_ORIGIN_NS = 1708490366230709837
BLOCK_WIDTH_NS = 60_000_000_000

IDENTITY_FIELDS = (
    "event_id",
    "capture_id",
    "target_id",
    "anchor_region",
    "truth_state",
    "start_timestamp_ns",
    "end_timestamp_ns",
    "eligible_frame_count",
    "overlap_component_id",
    "time_block_id_60s",
    "primary_error_partition",
    "reference_error_partition",
)
SOURCE_FIELDS = (
    "source_pair_denominator",
    "finite_source_pair_count",
    "finite_source_pair_coverage",
    "source_missing_reason_counts",
    "median_person_approach_component_mps",
    "median_sensor_approach_component_mps",
    "median_abs_person_approach_component_mps",
    "median_abs_sensor_approach_component_mps",
    "sensor_absolute_share",
    "median_camera_translation_speed_mps",
    "p90_camera_translation_speed_mps",
    "median_camera_angular_speed_radps",
    "p90_camera_angular_speed_radps",
)
ROI_FIELDS = (
    "median_abs_log_area_rate_per_s",
    "log_area_rate_mad_per_s",
    "median_center_speed_normalized_per_s",
    "center_velocity_mad_normalized_per_s",
    "duration_s",
    "finite_flow_coverage",
    "flow_sign_flip_fraction",
    "median_flow_score_mad_per_s",
    "p90_flow_score_mad_per_s",
    "median_detected_features",
    "median_surviving_tracks",
    "minimum_surviving_tracks",
    "median_occupied_quadrants",
    "median_forward_backward_error_px",
    "abstained_pair_count",
)
REQUIRED_EVENT_FIELDS = frozenset(
    IDENTITY_FIELDS + SOURCE_FIELDS + ROI_FIELDS + ("negative_log_duration_s", "reference_arm_behavior")
)
NULLABLE_SUMMARY_FIELDS = frozenset(
    (
        "median_person_approach_component_mps",
        "median_sensor_approach_component_mps",
        "median_abs_person_approach_component_mps",
        "median_abs_sensor_approach_component_mps",
        "sensor_absolute_share",
        "median_camera_translation_speed_mps",
        "p90_camera_translation_speed_mps",
        "median_camera_angular_speed_radps",
        "p90_camera_angular_speed_radps",
        "median_abs_log_area_rate_per_s",
        "log_area_rate_mad_per_s",
        "median_center_speed_normalized_per_s",
        "center_velocity_mad_normalized_per_s",
        "flow_sign_flip_fraction",
        "median_flow_score_mad_per_s",
        "p90_flow_score_mad_per_s",
        "median_detected_features",
        "median_surviving_tracks",
        "minimum_surviving_tracks",
        "median_occupied_quadrants",
        "median_forward_backward_error_px",
        "reference_arm_behavior",
    )
)


class ValidationError(ValueError):
    """A fail-closed validation mismatch."""


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number is forbidden: {value}")


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(data: bytes, *, label: str) -> Any:
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValidationError(f"{label}: UTF-8 BOM is forbidden")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{label}: invalid UTF-8") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_pairs,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValidationError(f"{label}: invalid JSON: {exc}") from exc


def canonical_json_bytes(value: Any, *, final_lf: bool = True) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + ("\n" if final_lf else "")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label}: path missing")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValidationError(f"{label}: path escapes repository") from exc
    if not path.is_file():
        raise ValidationError(f"{label}: file missing")
    return path


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value, final_lf=False))


def _normalized_input_binding(specification: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(specification)
    if "sha256" not in binding and "content_identity_sha256" in binding:
        binding["sha256"] = binding["content_identity_sha256"]
    return binding


def _expected_implementation_paths(
    repo_root: Path, protocol: Mapping[str, Any]
) -> set[str]:
    planned = protocol["planned_implementation"]
    module_root = str(planned["module_root"]).replace("\\", "/").rstrip("/")
    module_directory = (repo_root / module_root).resolve()
    expected = {
        f"{module_root}/{path.name}"
        for path in module_directory.iterdir()
        if path.is_file() and (path.suffix == ".py" or path.name == "README.md")
    }
    expected.add(str(planned["stable_adapter"]).replace("\\", "/"))
    expected.update(
        f"{module_root}/{name}" for name in planned.get("modules", ())
    )
    return expected


def load_canonical_json(path: Path) -> Any:
    data = path.read_bytes()
    value = parse_json_bytes(data, label=str(path))
    if data != canonical_json_bytes(value):
        raise ValidationError(f"{path}: JSON is not canonical")
    return value


def load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise ValidationError(f"{path}: JSONL must be nonempty LF text with final LF")
    rows: list[dict[str, Any]] = []
    rebuilt = bytearray()
    for index, line in enumerate(data.splitlines(), start=1):
        if not line:
            raise ValidationError(f"{path}:{index}: blank JSONL row")
        row = parse_json_bytes(line, label=f"{path}:{index}")
        if not isinstance(row, dict):
            raise ValidationError(f"{path}:{index}: row must be an object")
        rows.append(row)
        rebuilt.extend(canonical_json_bytes(row))
    if bytes(rebuilt) != data:
        raise ValidationError(f"{path}: JSONL is not canonical")
    return rows


def expected_dependency_protocol_id(protocol: Mapping[str, Any]) -> str:
    scientific = protocol.get("scientific_contract_binding")
    if isinstance(scientific, Mapping):
        protocol_id = scientific.get("protocol_id")
        if isinstance(protocol_id, str) and protocol_id:
            return protocol_id
    return PROTOCOL_ID


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _deadband_state(value: float) -> str:
    if value >= 0.1:
        return "approaching"
    if value <= -0.1:
        return "receding"
    return "quasi_static"


def type7_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValidationError("quantile requires a nonempty finite sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def cliff_delta(wrong: Sequence[float], correct: Sequence[float]) -> float:
    if not wrong or not correct:
        raise ValidationError("Cliff delta requires both groups")
    numerator = 0
    for wrong_value in wrong:
        for correct_value in correct:
            numerator += (wrong_value > correct_value) - (wrong_value < correct_value)
    return numerator / (len(wrong) * len(correct))


def balanced_cliff_delta(
    rows: Sequence[Mapping[str, Any]], field: str, unit_field: str, transform: str
) -> float:
    by_group_unit: dict[str, dict[Any, list[float]]] = {
        CORRECT: defaultdict(list),
        WRONG: defaultdict(list),
    }
    for row in rows:
        group = row["primary_error_partition"]
        value = row.get(field)
        if group in GROUPS and _finite(value):
            number = float(value)
            by_group_unit[group][row[unit_field]].append(
                -number if transform == "NEGATE" else number
            )
    if any(not by_group_unit[group] for group in GROUPS):
        raise ValidationError("balanced Cliff delta has an empty group")
    weighted: dict[str, list[tuple[float, float]]] = {}
    for group in GROUPS:
        unit_count = len(by_group_unit[group])
        weighted[group] = [
            (value, 1.0 / (unit_count * len(values)))
            for values in by_group_unit[group].values()
            for value in values
        ]
    return sum(
        wrong_weight
        * correct_weight
        * ((wrong_value > correct_value) - (wrong_value < correct_value))
        for wrong_value, wrong_weight in weighted[WRONG]
        for correct_value, correct_weight in weighted[CORRECT]
    )


def _burden_values(
    rows: Iterable[Mapping[str, Any]], field: str, transform: str
) -> dict[str, list[float]]:
    values = {CORRECT: [], WRONG: []}
    for row in rows:
        group = row["primary_error_partition"]
        value = row.get(field)
        if group in GROUPS and _finite(value):
            number = float(value)
            values[group].append(-number if transform == "NEGATE" else number)
    return values


def _missing_reasons(
    rows: Iterable[Mapping[str, Any]], field: str, group: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    reason_field = f"{field}_missing_reason"
    for row in rows:
        if row["primary_error_partition"] != group or _finite(row.get(field)):
            continue
        reason = row.get(reason_field)
        if reason is None:
            reason = row.get("field_missing_reasons", {}).get(field)
        counts[str(reason) if isinstance(reason, str) and reason else "UNSPECIFIED_NULL"] += 1
    return dict(sorted(counts.items()))


def event_missingness(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    total = {
        group: sum(row["primary_error_partition"] == group for row in rows)
        for group in GROUPS
    }
    finite = {
        group: sum(
            row["primary_error_partition"] == group and _finite(row.get(field))
            for row in rows
        )
        for group in GROUPS
    }
    fractions = {
        group: finite[group] / total[group] if total[group] else None for group in GROUPS
    }
    difference = (
        abs(fractions[WRONG] - fractions[CORRECT])
        if all(fractions[group] is not None for group in GROUPS)
        else None
    )
    reasons: list[str] = []
    for group in GROUPS:
        if total[group] < MIN_TOTAL:
            reasons.append(f"{group}:TOTAL_LT_{MIN_TOTAL}")
        if finite[group] < MIN_FINITE:
            reasons.append(f"{group}:FINITE_LT_{MIN_FINITE}")
        if fractions[group] is None or fractions[group] < MIN_FINITE_FRACTION:
            reasons.append(f"{group}:FINITE_FRACTION_LT_{MIN_FINITE_FRACTION}")
    if difference is None or difference > MAX_FRACTION_DIFFERENCE:
        reasons.append(f"ABSOLUTE_GROUP_FINITE_FRACTION_DIFFERENCE_GT_{MAX_FRACTION_DIFFERENCE}")
    return {
        "total_n": total,
        "finite_n": finite,
        "finite_fraction": fractions,
        "missing_reason_counts": {
            group: _missing_reasons(rows, field, group) for group in GROUPS
        },
        "absolute_group_finite_fraction_difference": difference,
        "evaluable": not reasons,
        "not_evaluable_reasons": reasons,
    }


def component_missingness(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
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
            if row["primary_error_partition"] == group and _finite(row.get(field))
        }
        for group in GROUPS
    }
    base = event_missingness(rows, field)
    total = {group: len(occupied[group]) for group in GROUPS}
    finite = {group: len(finite_components[group]) for group in GROUPS}
    fractions = {
        group: finite[group] / total[group] if total[group] else None for group in GROUPS
    }
    difference = (
        abs(fractions[WRONG] - fractions[CORRECT])
        if all(fractions[group] is not None for group in GROUPS)
        else None
    )
    reasons: list[str] = list(base["not_evaluable_reasons"])
    for group in GROUPS:
        if total[group] == 0:
            reasons.append(f"{group}:ZERO_OCCUPIED_COMPONENTS")
        if finite[group] < MIN_COMPONENTS:
            reasons.append(f"{group}:FINITE_COMPONENTS_LT_{MIN_COMPONENTS}")
        if fractions[group] is None or fractions[group] < MIN_FINITE_FRACTION:
            reasons.append(f"{group}:FINITE_COMPONENT_FRACTION_LT_{MIN_FINITE_FRACTION}")
    if difference is None or difference > MAX_FRACTION_DIFFERENCE:
        reasons.append(
            f"ABSOLUTE_GROUP_FINITE_COMPONENT_FRACTION_DIFFERENCE_GT_{MAX_FRACTION_DIFFERENCE}"
        )
    return {
        **base,
        "total_occupied_components": total,
        "finite_components": finite,
        "finite_component_fraction": fractions,
        "absolute_group_finite_component_fraction_difference": difference,
        "evaluable": not reasons,
        "not_evaluable_reasons": reasons,
    }


def _delta_cell(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    transform: str,
    missingness: Mapping[str, Any],
) -> dict[str, Any]:
    if not missingness["evaluable"]:
        return {"delta": None, "evaluable": False, "reason": "MISSINGNESS_NOT_EVALUABLE"}
    values = _burden_values(rows, field, transform)
    return {
        "delta": cliff_delta(values[WRONG], values[CORRECT]),
        "evaluable": True,
        "reason": None,
    }


def _balanced_delta_cell(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    transform: str,
    unit_field: str,
    missingness: Mapping[str, Any],
) -> dict[str, Any]:
    if not missingness["evaluable"]:
        return {"delta": None, "evaluable": False, "reason": "MISSINGNESS_NOT_EVALUABLE"}
    return {
        "delta": balanced_cliff_delta(rows, field, unit_field, transform),
        "evaluable": True,
        "reason": None,
    }


def _subsets(
    rows: Sequence[Mapping[str, Any]], key: str, values: Iterable[Any]
) -> dict[str, list[Mapping[str, Any]]]:
    return {
        str(value): [row for row in rows if row[key] == value]
        for value in values
    }


def _required_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    values = sorted({row[key] for row in rows})
    return [
        value
        for value in values
        if all(
            sum(
                row[key] == value and row["primary_error_partition"] == group
                for row in rows
            )
            >= MIN_TOTAL
            for group in GROUPS
        )
    ]


def recompute_metric(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    transform: str,
    *,
    role: str = "PRESELECTED",
) -> dict[str, Any]:
    targets = sorted({row["target_id"] for row in rows})
    regions = _required_values(rows, "anchor_region")
    candidates = _required_values(rows, "time_block_id_60s")
    target_rows = _subsets(rows, "target_id", targets)
    region_rows = _subsets(rows, "anchor_region", regions)
    truth_rows = _subsets(rows, "truth_state", TRUTH_STATES)
    block_rows = _subsets(rows, "time_block_id_60s", candidates)

    block_cells = {
        key: event_missingness(subset, field) for key, subset in block_rows.items()
    }
    failed_blocks = [key for key, cell in block_cells.items() if not cell["evaluable"]]
    block_support = {
        "candidate_block_ids": candidates,
        "minimum_common_evaluable_blocks": MIN_BLOCKS,
        "cells": block_cells,
        "evaluable": len(candidates) >= MIN_BLOCKS and not failed_blocks,
        "not_evaluable_reasons": (
            (["COMMON_EVALUABLE_BLOCKS_LT_4"] if len(candidates) < MIN_BLOCKS else [])
            + [f"BLOCK_{key}_SUPPORT_FAILED" for key in failed_blocks]
        ),
    }
    missingness = {
        "pooled": event_missingness(rows, field),
        "targets": {
            key: event_missingness(subset, field) for key, subset in target_rows.items()
        },
        "regions": {
            "required_region_ids": regions,
            "cells": {
                key: event_missingness(subset, field)
                for key, subset in region_rows.items()
            },
        },
        "truth_states": {
            key: event_missingness(subset, field) for key, subset in truth_rows.items()
        },
        "component": component_missingness(rows, field),
        "blocks": block_support,
    }
    deltas = {
        "event_weighted": _delta_cell(
            rows, field, transform, missingness["pooled"]
        ),
        "component_balanced": _balanced_delta_cell(
            rows,
            field,
            transform,
            "overlap_component_id",
            missingness["component"],
        ),
        "block_balanced": _balanced_delta_cell(
            [row for row in rows if str(row["time_block_id_60s"]) in block_rows],
            field,
            transform,
            "time_block_id_60s",
            {
                "evaluable": bool(block_rows)
                and block_support["evaluable"]
            },
        ),
        "targets": {
            key: _delta_cell(subset, field, transform, missingness["targets"][key])
            for key, subset in target_rows.items()
        },
        "regions": {
            key: _delta_cell(
                subset, field, transform, missingness["regions"]["cells"][key]
            )
            for key, subset in region_rows.items()
        },
        "truth_states": {
            key: _delta_cell(
                subset, field, transform, missingness["truth_states"][key]
            )
            for key, subset in truth_rows.items()
        },
        "common_blocks": {
            key: _delta_cell(subset, field, transform, block_cells[key])
            for key, subset in block_rows.items()
        },
    }

    all_views_evaluable = (
        len(regions) >= MIN_REGIONS
        and len(candidates) >= MIN_BLOCKS
        and missingness["pooled"]["evaluable"]
        and missingness["component"]["evaluable"]
        and all(cell["evaluable"] for cell in missingness["targets"].values())
        and all(cell["evaluable"] for cell in missingness["regions"]["cells"].values())
        and all(cell["evaluable"] for cell in missingness["truth_states"].values())
        and block_support["evaluable"]
    )
    event_delta = deltas["event_weighted"]["delta"]
    component_delta = deltas["component_balanced"]["delta"]
    robust = bool(
        all_views_evaluable
        and event_delta is not None
        and event_delta >= SUPPORT_DELTA
        and component_delta is not None
        and component_delta >= SUPPORT_DELTA
        and deltas["block_balanced"]["delta"] is not None
        and deltas["block_balanced"]["delta"] >= SUPPORT_DELTA
        and all(cell["delta"] is not None and cell["delta"] > 0 for cell in deltas["targets"].values())
        and sum(cell["delta"] is not None and cell["delta"] > 0 for cell in deltas["regions"].values())
        >= MIN_REGIONS
        and not any(
            cell["delta"] is not None and cell["delta"] <= CONTRADICTION_DELTA
            for cell in deltas["regions"].values()
        )
        and all(
            deltas["truth_states"][state]["delta"] is not None
            and deltas["truth_states"][state]["delta"] >= SUPPORT_DELTA
            for state in TRUTH_STATES
        )
        and len(deltas["common_blocks"]) >= MIN_BLOCKS
        and not any(
            cell["delta"] is not None and cell["delta"] <= CONTRADICTION_DELTA
            for cell in deltas["common_blocks"].values()
        )
        and event_delta * component_delta > 0
    )
    required_delta_cells = [
        deltas["component_balanced"],
        deltas["block_balanced"],
        *deltas["targets"].values(),
        *deltas["regions"].values(),
        *deltas["truth_states"].values(),
        *deltas["common_blocks"].values(),
    ]
    contradiction = bool(
        not all_views_evaluable
        or any(
            cell["delta"] is not None and cell["delta"] <= CONTRADICTION_DELTA
            for cell in required_delta_cells
        )
        or (
            event_delta is not None
            and component_delta is not None
            and event_delta * component_delta < 0
        )
    )
    if role == "DIAGNOSTIC_ONLY":
        all_views_evaluable = False
        robust = False
        contradiction = False
    return {
        "field": field,
        "role": role,
        "burden_transform": transform,
        "missingness": missingness,
        "deltas": deltas,
        "globally_route_evaluable": all_views_evaluable,
        "robust_support": robust,
        "material_contradiction": contradiction,
    }


def recompute_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
    event_table_sha256: str,
) -> dict[str, Any]:
    metrics = {
        field: recompute_metric(rows, field, transform) for field, transform in METRICS
    }
    metrics.update(
        {
            field: recompute_metric(
                rows, field, "IDENTITY", role="DIAGNOSTIC_ONLY"
            )
            for field in DIAGNOSTIC_FIELDS
        }
    )
    wrong_rows = [
        row for row in rows if row["primary_error_partition"] == WRONG
    ]
    shares = [
        float(row["sensor_absolute_share"])
        for row in wrong_rows
        if _finite(row.get("sensor_absolute_share"))
    ]
    share_fraction = len(shares) / len(wrong_rows) if wrong_rows else 0.0
    share_evaluable = bool(
        len(shares) >= MIN_FINITE
        and share_fraction >= MIN_FINITE_FRACTION
    )
    share_median = type7_quantile(shares, 0.5) if shares else None
    share_gate = {
        "field": "sensor_absolute_share",
        "group": WRONG,
        "median": share_median,
        "total_n": len(wrong_rows),
        "finite_n": len(shares),
        "finite_fraction": share_fraction,
        "evaluable": share_evaluable,
        "value": bool(share_evaluable and share_median is not None and share_median > 0.5),
    }
    global_evaluable = share_evaluable and all(
        metrics[field]["globally_route_evaluable"] for field in METRIC_FIELDS
    )
    ego = metrics[EGO_FIELD]
    person = metrics[PERSON_FIELD]
    temporal = metrics[TEMPORAL_FIELD]
    temporal_support = metrics[TEMPORAL_SUPPORT_FIELD]
    person_equal = bool(
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
    person_material = person["robust_support"]
    person_competing = person_material and person_equal
    ego_contradictory = ego["material_contradiction"]
    temporal_contradictory = (
        temporal["material_contradiction"]
        or temporal_support["material_contradiction"]
    )
    ego_candidate = bool(
        global_evaluable
        and share_gate["value"]
        and ego["robust_support"]
        and not person_competing
        and not ego_contradictory
    )
    temporal_candidate = bool(
        global_evaluable
        and temporal["robust_support"]
        and temporal_support["robust_support"]
        and not temporal_contradictory
    )
    provisional_exit = scientific_exit(
        execution_valid=True,
        global_route_evaluable=global_evaluable,
        ego_candidate=ego_candidate,
        temporal_candidate=temporal_candidate,
    )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "event_table_sha256": event_table_sha256,
        "event_count": len(rows),
        "primary_arm": "ROI_SPARSE_RADIAL_FLOW",
        "reference_arm": "BBOX_LOG_AREA_GROWTH",
        "metric_order": list(ALL_METRIC_FIELDS),
        "metrics": metrics,
        "ego_share_gate": share_gate,
        "routing": {
            "global_route_evaluable": global_evaluable,
            "ego_contradictory": ego_contradictory,
            "temporal_contradictory": temporal_contradictory,
            "person_material_support": person_material,
            "person_equal_or_larger": person_equal,
            "person_competing": person_competing,
            "ego_candidate": ego_candidate,
            "temporal_candidate": temporal_candidate,
            "provisional_scientific_exit": provisional_exit,
        },
    }


def scientific_exit(
    *,
    execution_valid: bool,
    global_route_evaluable: bool,
    ego_candidate: bool,
    temporal_candidate: bool,
) -> str | None:
    if not execution_valid:
        return None
    if global_route_evaluable and ego_candidate and not temporal_candidate:
        return "EGO_CANARY_PRIORITY"
    if global_route_evaluable and temporal_candidate and not ego_candidate:
        return "TEMPORAL_TREND_PRIORITY"
    return "NO_PRIORITY_IDENTIFIED"


def _assert_exact(actual: Any, expected: Any, path: str = "$") -> None:
    if type(actual) is not type(expected):
        raise ValidationError(
            f"{path}: type mismatch {type(actual).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, dict):
        if list(actual) != list(expected):
            raise ValidationError(
                f"{path}: key/order mismatch {list(actual)!r} != {list(expected)!r}"
            )
        for key in expected:
            _assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValidationError(f"{path}: list length mismatch")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or actual != expected:
            raise ValidationError(f"{path}: float mismatch {actual!r} != {expected!r}")
    elif actual != expected:
        raise ValidationError(f"{path}: mismatch {actual!r} != {expected!r}")


def _recompute_dependency(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(bindings, key=lambda row: row["event_id"])
    intervals = sorted(
        [
        (int(row["start_timestamp_ns"]), int(row["end_timestamp_ns"]), row["event_id"])
        for row in ordered
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )
    parent = {event_id: event_id for _, _, event_id in intervals}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    cross_target = 0
    same_target = 0
    overlap_pairs: list[list[str]] = []
    by_id = {row["event_id"]: row for row in ordered}
    for left_index, (left_start, left_end, left_id) in enumerate(intervals):
        for right_start, right_end, right_id in intervals[left_index + 1 :]:
            if right_start > left_end:
                break
            if left_start <= right_end and right_start <= left_end:
                overlap_pairs.append(sorted((left_id, right_id)))
                if by_id[left_id]["target_id"] == by_id[right_id]["target_id"]:
                    same_target += 1
                else:
                    cross_target += 1
                union(left_id, right_id)
    components: dict[str, list[str]] = defaultdict(list)
    for _, _, event_id in intervals:
        components[find(event_id)].append(event_id)
    component_lists = sorted(
        (sorted(ids) for ids in components.values()),
        key=lambda ids: (
            min(int(by_id[event_id]["start_timestamp_ns"]) for event_id in ids),
            ids[0],
        ),
    )
    component_id_by_event = {
        event_id: f"component-{index:04d}"
        for index, event_ids in enumerate(component_lists)
        for event_id in event_ids
    }
    component_rows = [
        {
            "component_id": f"component-{index:04d}",
            "event_count": len(event_ids),
            "event_ids": event_ids,
            "start_timestamp_ns": min(
                int(by_id[event_id]["start_timestamp_ns"])
                for event_id in event_ids
            ),
            "end_timestamp_ns": max(
                int(by_id[event_id]["end_timestamp_ns"])
                for event_id in event_ids
            ),
        }
        for index, event_ids in enumerate(component_lists)
    ]
    size_counts = Counter(len(event_ids) for event_ids in component_lists)
    origin = min(int(row["start_timestamp_ns"]) for row in ordered)
    block_counts = Counter(
        (
            (
                int(row["start_timestamp_ns"])
                + int(row["end_timestamp_ns"])
            )
            // 2
            - origin
        )
        // BLOCK_WIDTH_NS
        for row in ordered
    )
    overlap_pairs.sort()
    return {
        "ordered": ordered,
        "cross_target_overlap_pair_count": cross_target,
        "same_target_overlap_pair_count": same_target,
        "overlap_pairs_sha256": sha256_bytes(
            canonical_json_bytes(overlap_pairs, final_lf=False)
        ),
        "component_id_by_event": component_id_by_event,
        "component_size_counts": {str(key): size_counts[key] for key in sorted(size_counts)},
        "exact_overlap_component_count": len(component_lists),
        "components": component_rows,
        "time_block": {
            "origin_timestamp_ns": origin,
            "width_ns": BLOCK_WIDTH_NS,
            "assignment": (
                "floor((floor((start_timestamp_ns + end_timestamp_ns) / 2) "
                "- origin_timestamp_ns) / width_ns)"
            ),
            "block_ids": sorted(block_counts),
            "event_counts": [
                block_counts[block_id] for block_id in sorted(block_counts)
            ],
        },
        "event_bindings_sha256": sha256_bytes(
            canonical_json_bytes(ordered, final_lf=False)
        ),
    }


def validate_event_rows(
    rows: Sequence[Mapping[str, Any]],
    dependency: Mapping[str, Any],
    *,
    expected_count: int = 469,
) -> None:
    if len(rows) != expected_count:
        raise ValidationError(f"event row count {len(rows)} != {expected_count}")
    bindings = dependency.get("event_bindings")
    if not isinstance(bindings, list) or len(bindings) != expected_count:
        raise ValidationError("dependency receipt event_bindings count mismatch")
    recomputed = _recompute_dependency(bindings)
    if [row["event_id"] for row in rows] != [
        binding["event_id"] for binding in recomputed["ordered"]
    ]:
        raise ValidationError("event table order/keyset differs from dependency receipt")
    if len({row["event_id"] for row in rows}) != len(rows):
        raise ValidationError("duplicate event_id")
    for row, binding in zip(rows, recomputed["ordered"]):
        missing = REQUIRED_EVENT_FIELDS - row.keys()
        if missing:
            raise ValidationError(
                f"{row.get('event_id', '<unknown>')}: missing fields {sorted(missing)}"
            )
        for key in (
            "event_id",
            "target_id",
            "anchor_region",
            "truth_state",
            "start_timestamp_ns",
            "end_timestamp_ns",
            "overlap_component_id",
            "time_block_id_60s",
        ):
            if row[key] != binding[key]:
                raise ValidationError(f"{row['event_id']}.{key}: dependency mismatch")
        midpoint = (row["start_timestamp_ns"] + row["end_timestamp_ns"]) // 2
        expected_block = (midpoint - BLOCK_ORIGIN_NS) // BLOCK_WIDTH_NS
        if row["time_block_id_60s"] != expected_block:
            raise ValidationError(f"{row['event_id']}: time-block mismatch")
        if row["overlap_component_id"] != recomputed["component_id_by_event"][row["event_id"]]:
            raise ValidationError(f"{row['event_id']}: overlap-component mismatch")
        if row["primary_error_partition"] not in (CORRECT, WRONG, "OTHER_INCORRECT"):
            raise ValidationError(f"{row['event_id']}: invalid primary partition")
    for key in (
        "cross_target_overlap_pair_count",
        "same_target_overlap_pair_count",
        "overlap_pairs_sha256",
        "exact_overlap_component_count",
        "component_size_counts",
        "components",
        "time_block",
        "event_bindings_sha256",
    ):
        if dependency.get(key) != recomputed[key]:
            raise ValidationError(f"dependency receipt {key} mismatch")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        value = parse_json_bytes(line, label=f"{path}:{number}")
        if not isinstance(value, dict):
            raise ValidationError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def _unique(rows: Iterable[Mapping[str, Any]], fields: Sequence[str], label: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if None in key or key in result:
            raise ValidationError(f"{label}: null or duplicate key {key}")
        result[key] = row
    return result


def _rotation_matrix(quaternion: Any) -> Any:
    import numpy as np

    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-6:
        raise ValidationError("invalid Vicon quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _valid_pose(position: Any, quaternion: Any) -> bool:
    import numpy as np

    return bool(
        np.isfinite(position).all()
        and np.isfinite(quaternion).all()
        and np.linalg.norm(position) > 1e-9
        and np.linalg.norm(quaternion) > 1e-6
    )


def _valid_interval(track: Mapping[str, Any], first: int, second: int) -> bool:
    import numpy as np

    if not _valid_pose(track["positions"][first], track["quaternions"][first]):
        return False
    if not _valid_pose(track["positions"][second], track["quaternions"][second]):
        return False
    dt = (int(track["timestamps_ns"][second]) - int(track["timestamps_ns"][first])) / 1e9
    return bool(
        0.005 <= dt <= 0.05
        and np.linalg.norm(track["positions"][second] - track["positions"][first]) / dt <= 5.0
    )


def _nearest_timestamp_index(timestamps: Any, query: int) -> int:
    import numpy as np

    right = int(np.searchsorted(timestamps, query, side="left"))
    left = min(max(right - 1, 0), len(timestamps) - 1)
    right = min(max(right, 0), len(timestamps) - 1)
    return left if abs(query - int(timestamps[left])) <= abs(int(timestamps[right]) - query) else right


def _independent_source_pair(
    image_ns: int,
    person: Mapping[str, Any],
    sensor: Mapping[str, Any],
    calibration: Any,
    frozen_signed: float,
) -> tuple[dict[str, Any] | None, str | None]:
    import numpy as np

    pt = person["timestamps_ns"]
    right = bisect.bisect_left(pt, image_ns)
    if right <= 0 or right >= len(pt):
        return None, "PERSON_BRACKET_BOUNDARY"
    p0i, p1i = right - 1, right
    if not _valid_interval(person, p0i, p1i):
        return None, "PERSON_CONTINUITY_REJECTED"
    p0t, p1t = int(pt[p0i]), int(pt[p1i])
    st = sensor["timestamps_ns"]
    s0i, s1i = _nearest_timestamp_index(st, p0t), _nearest_timestamp_index(st, p1t)
    if abs(int(st[s0i]) - p0t) > 20_000_000 or abs(int(st[s1i]) - p1t) > 20_000_000:
        return None, "PERSON_SENSOR_SYNC_REJECTED"
    delta = s1i - s0i
    if delta not in (0, 1):
        return None, "SENSOR_INDEX_DELTA_REJECTED"
    if delta == 1 and not _valid_interval(sensor, s0i, s1i):
        return None, "SENSOR_CONTINUITY_REJECTED"
    if not _valid_pose(sensor["positions"][s0i], sensor["quaternions"][s0i]) or not _valid_pose(sensor["positions"][s1i], sensor["quaternions"][s1i]):
        return None, "SENSOR_POSE_INVALID"
    dt = (p1t - p0t) / 1e9
    p0, p1 = person["positions"][p0i], person["positions"][p1i]
    s0, s1 = sensor["positions"][s0i], sensor["positions"][s1i]
    r0, r1 = p0 - s0, p1 - s1
    denominator = float(np.linalg.norm(r0) + np.linalg.norm(r1))
    if denominator <= 1e-12:
        return None, "RANGE_GEOMETRY_DEGENERATE"
    gradient = (r0 + r1) / denominator
    person_rate = float(np.dot(gradient, (p1 - p0) / dt))
    sensor_rate = float(-np.dot(gradient, (s1 - s0) / dt))
    person_approach, sensor_approach = -person_rate, -sensor_rate
    signed = -(person_rate + sensor_rate)
    if (
        abs(signed - float(frozen_signed)) > 1e-6
        or _deadband_state(signed) != _deadband_state(float(frozen_signed))
    ):
        raise ValidationError("source signed-approach/deadband closure mismatch")
    share_denominator = abs(sensor_approach) + abs(person_approach)
    share = abs(sensor_approach) / share_denominator if share_denominator >= 1e-6 else None
    rs0, rs1 = _rotation_matrix(sensor["quaternions"][s0i]), _rotation_matrix(sensor["quaternions"][s1i])
    camera0 = s0 + rs0 @ calibration[:3, 3]
    camera1 = s1 + rs1 @ calibration[:3, 3]
    translation = float(np.linalg.norm(camera1 - camera0) / dt)
    q0 = sensor["quaternions"][s0i].astype(np.float64, copy=True)
    q1 = sensor["quaternions"][s1i].astype(np.float64, copy=True)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    half_angle_cosine = min(1.0, max(-1.0, float(np.dot(q0, q1))))
    angular = 2.0 * math.acos(half_angle_cosine) / dt
    return {
        "person": person_approach,
        "sensor": sensor_approach,
        "share": share,
        "translation": translation,
        "angular": angular,
    }, None


def _independent_roi_pair(previous: Mapping[str, Any] | None, current: Mapping[str, Any], bbox: Mapping[str, Any]) -> dict[str, Any] | None:
    if previous is None:
        return None
    dt = (current["captured_at_ns"] - previous["captured_at_ns"]) / 1e9
    if (
        previous.get("track_epoch") != current.get("track_epoch")
        or current.get("history_reset") is not False
        or not 0 < dt <= 0.1
        or bbox.get("abstention_reason") is not None
        or not _finite(bbox.get("signed_approach_rate_per_s"))
    ):
        return None
    before, after = previous["roi_xywh_normalized"], current["roi_xywh_normalized"]
    if len(before) != 4 or len(after) != 4 or before[2] * before[3] <= 0 or after[2] * after[3] <= 0:
        return None
    rate = math.log((after[2] * after[3]) / (before[2] * before[3])) / dt
    if abs(rate - float(bbox["signed_approach_rate_per_s"])) > 1e-12:
        raise ValidationError("BBOX log-area closure mismatch")
    return {
        "rate": rate,
        "velocity": (
            ((after[0] + after[2] / 2) - (before[0] + before[2] / 2)) / dt,
            ((after[1] + after[3] / 2) - (before[1] + before[3] / 2)) / dt,
        ),
    }


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return type7_quantile(finite, 0.5) if finite else None


def _flow_flips(rows: Sequence[Mapping[str, Any]]) -> float | None:
    previous_index = previous_sign = None
    transitions = flips = 0
    for row in rows:
        flow = row["flow"]
        rate = flow.get("signed_approach_rate_per_s")
        if flow.get("abstention_reason") is not None or not _finite(rate):
            previous_index = previous_sign = None
            continue
        sign = 1 if rate >= 0.02 else -1 if rate <= -0.02 else 0
        if previous_index is not None and row["index"] == previous_index + 1:
            transitions += 1
            flips += sign != previous_sign
        previous_index, previous_sign = row["index"], sign
    return flips / transitions if transitions else None


def _independent_event(
    natural: Mapping[str, Any],
    binding: Mapping[str, Any],
    primary_eval: Mapping[str, Any],
    reference_eval: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    denominator = natural["eligible_frame_count"]
    pairs = [row["source"] for row in members if row["source"] is not None]
    source_reasons = Counter(row["source_reason"] for row in members if row["source_reason"])
    source_supported = len(pairs) >= 3 and len(pairs) / denominator >= 0.5
    missing: dict[str, str] = {}
    output: dict[str, Any] = {
        "event_id": natural["event_id"], "capture_id": natural["capture_id"],
        "target_id": natural["target_id"], "anchor_region": natural["anchor_region"],
        "truth_state": natural["truth_state"], "start_timestamp_ns": natural["start_timestamp_ns"],
        "end_timestamp_ns": natural["end_timestamp_ns"], "eligible_frame_count": denominator,
        "overlap_component_id": binding["overlap_component_id"], "time_block_id_60s": binding["time_block_id_60s"],
        "primary_error_partition": CORRECT if primary_eval.get("correct") is True else WRONG if primary_eval.get("wrong_signed") is True else "OTHER_INCORRECT",
        "reference_error_partition": CORRECT if reference_eval.get("correct") is True else WRONG if reference_eval.get("wrong_signed") is True else "OTHER_INCORRECT",
        "source_pair_denominator": denominator, "finite_source_pair_count": len(pairs),
        "finite_source_pair_coverage": len(pairs) / denominator,
        "source_missing_reason_counts": dict(sorted(source_reasons.items())),
    }
    source_map = {
        "median_person_approach_component_mps": ("person", False, 0.5),
        "median_sensor_approach_component_mps": ("sensor", False, 0.5),
        "median_abs_person_approach_component_mps": ("person", True, 0.5),
        "median_abs_sensor_approach_component_mps": ("sensor", True, 0.5),
        "median_camera_translation_speed_mps": ("translation", False, 0.5),
        "p90_camera_translation_speed_mps": ("translation", False, 0.9),
        "median_camera_angular_speed_radps": ("angular", False, 0.5),
        "p90_camera_angular_speed_radps": ("angular", False, 0.9),
    }
    for field, (key, absolute, quantile) in source_map.items():
        values = [abs(pair[key]) if absolute else pair[key] for pair in pairs]
        output[field] = type7_quantile(values, quantile) if source_supported else None
    shares = [pair["share"] for pair in pairs if _finite(pair["share"])]
    share_supported = (
        source_supported
        and len(shares) >= 3
        and len(shares) / denominator >= 0.5
    )
    output["sensor_absolute_share"] = (
        type7_quantile(shares, 0.5) if share_supported else None
    )
    if not source_supported:
        for field in source_map:
            missing[field] = "INSUFFICIENT_FINITE_SOURCE_PAIR_SUPPORT"
        missing["sensor_absolute_share"] = "INSUFFICIENT_FINITE_SOURCE_PAIR_SUPPORT"
    elif not share_supported:
        missing["sensor_absolute_share"] = "INSUFFICIENT_FINITE_SHARE_SUPPORT"
    roi = [row["roi"] for row in members if row["roi"] is not None]
    rates, velocities = [item["rate"] for item in roi], [item["velocity"] for item in roi]
    rate_center = _median(rates)
    vx, vy = _median(item[0] for item in velocities), _median(item[1] for item in velocities)
    output.update({
        "median_abs_log_area_rate_per_s": _median(abs(value) for value in rates),
        "log_area_rate_mad_per_s": _median(abs(value - rate_center) for value in rates) if rate_center is not None else None,
        "median_center_speed_normalized_per_s": _median(math.hypot(*value) for value in velocities),
        "center_velocity_mad_normalized_per_s": _median(math.hypot(value[0] - vx, value[1] - vy) for value in velocities) if vx is not None and vy is not None else None,
    })
    component_map = {
        "median_flow_score_mad_per_s": "score_mad_per_s",
        "median_detected_features": "detected_features",
        "median_surviving_tracks": "surviving_tracks",
        "median_occupied_quadrants": "occupied_quadrants",
        "median_forward_backward_error_px": "median_fb_error_px",
    }
    component_values = {
        field: [
            row["flow"].get("quality", {}).get("components", {}).get(component)
            for row in members
            if _finite(row["flow"].get("quality", {}).get("components", {}).get(component))
        ]
        for field, component in component_map.items()
    }
    output["quality_component_support"] = {
        field: {
            "finite_count": len(values),
            "finite_coverage": len(values) / denominator,
            "missing_count": denominator - len(values),
            "missing_reason_counts": (
                {"NONFINITE_OR_MISSING_COMPONENT": denominator - len(values)}
                if len(values) < denominator
                else {}
            ),
        }
        for field, values in component_values.items()
    }
    temporal_ok = all(len(component_values[field]) >= 3 and len(component_values[field]) / denominator >= 0.5 for field in (TEMPORAL_FIELD, TEMPORAL_SUPPORT_FIELD))
    output[TEMPORAL_FIELD] = _median(component_values[TEMPORAL_FIELD]) if temporal_ok else None
    output[TEMPORAL_SUPPORT_FIELD] = _median(component_values[TEMPORAL_SUPPORT_FIELD]) if temporal_ok else None
    if not temporal_ok:
        missing[TEMPORAL_FIELD] = "INSUFFICIENT_COUPLED_TEMPORAL_SUPPORT"
        missing[TEMPORAL_SUPPORT_FIELD] = "INSUFFICIENT_COUPLED_TEMPORAL_SUPPORT"
    output["p90_flow_score_mad_per_s"] = type7_quantile(component_values[TEMPORAL_FIELD], 0.9) if component_values[TEMPORAL_FIELD] else None
    output["median_detected_features"] = _median(component_values["median_detected_features"])
    output["minimum_surviving_tracks"] = min(component_values[TEMPORAL_SUPPORT_FIELD]) if component_values[TEMPORAL_SUPPORT_FIELD] else None
    output["median_occupied_quadrants"] = _median(component_values["median_occupied_quadrants"])
    output["median_forward_backward_error_px"] = _median(component_values["median_forward_backward_error_px"])
    finite_flow = sum(row["flow"].get("abstention_reason") is None and _finite(row["flow"].get("signed_approach_rate_per_s")) for row in members)
    duration = natural["duration_s"]
    output.update({
        "duration_s": duration, "negative_log_duration_s": -math.log(duration),
        "finite_flow_coverage": finite_flow / denominator, "flow_sign_flip_fraction": _flow_flips(members),
        "abstained_pair_count": denominator - finite_flow,
        "reference_arm_behavior": reference_eval.get("event_score_per_s"),
    })
    for field, reason in (
        ("median_abs_log_area_rate_per_s", "NO_VALID_ROI_PAIRS"),
        ("log_area_rate_mad_per_s", "NO_VALID_ROI_PAIRS"),
        ("median_center_speed_normalized_per_s", "NO_VALID_ROI_PAIRS"),
        ("center_velocity_mad_normalized_per_s", "NO_VALID_ROI_PAIRS"),
        ("flow_sign_flip_fraction", "NO_ELIGIBLE_ADJACENT_FLOW_TRANSITIONS"),
    ):
        if output[field] is None:
            missing[field] = reason
    for field in (
        "p90_flow_score_mad_per_s",
        "median_detected_features",
        "minimum_surviving_tracks",
        "median_occupied_quadrants",
        "median_forward_backward_error_px",
    ):
        if output[field] is None:
            missing[field] = "NO_FINITE_QUALITY_COMPONENT_ROWS"
    for field in NULLABLE_SUMMARY_FIELDS:
        output[f"{field}_missing_reason"] = (
            missing.get(field, "NONFINITE_OR_NULL")
            if output[field] is None
            else None
        )
    return output


def read_camera_from_marker_independent(
    calibration_path: Path,
) -> Any:
    import numpy as np
    import yaml

    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    matrix = np.asarray(calibration["T_v_c"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValidationError("invalid calibration T_v_c")
    return matrix


def independently_recompute_event_table(repo_root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    frozen = protocol["frozen_inputs"]
    paths: dict[str, Path] = {}
    for name, specification in frozen.items():
        if "path" not in specification:
            continue
        path = (repo_root / specification["path"]).resolve()
        expected = specification.get("sha256") or specification.get("content_identity_sha256")
        if expected and sha256_file(path) != expected:
            raise ValidationError(f"frozen input hash mismatch: {name}")
        paths[name] = path
    replay, truth, natural, r2 = (
        _jsonl_rows(paths[name]) for name in ("replay_input", "truth", "natural_events", "r2_producer_output")
    )
    for name, rows in (("replay_input", replay), ("truth", truth), ("natural_events", natural), ("r2_producer_output", r2)):
        if len(rows) != frozen[name]["rows"]:
            raise ValidationError(f"{name}: row count mismatch")
    evaluation = parse_json_bytes(paths["r2_evaluation"].read_bytes(), label="r2 evaluation")
    dependency = parse_json_bytes(paths["dependency_receipt"].read_bytes(), label="dependency receipt")
    replay_index = _unique(replay, ("capture_id", "target_id", "source_frame_id"), "replay")
    truth_index = _unique(truth, ("capture_id", "target_id", "source_frame_id"), "truth")
    r2_index = _unique(r2, ("arm_id", "capture_id", "target_id", "source_frame_id"), "r2")
    if set(replay_index) != set(truth_index):
        raise ValidationError("replay/truth keyset mismatch")
    for arm in ("BBOX_LOG_AREA_GROWTH", "ROI_SPARSE_RADIAL_FLOW"):
        arm_keys = {key[1:] for key in r2_index if key[0] == arm}
        if arm_keys != set(replay_index):
            raise ValidationError(f"R2 arm keyset mismatch: {arm}")
    primary_natural = {row["event_id"]: row for row in natural if row.get("primary_event_eligible") is True}
    if len(primary_natural) != 469:
        raise ValidationError("primary natural-event count mismatch")
    expected_arms = {"BBOX_LOG_AREA_GROWTH", "ROI_SPARSE_RADIAL_FLOW"}
    if set(evaluation.get("arm_summaries", {})) != expected_arms:
        raise ValidationError("evaluation arm keyset mismatch")
    for arm in expected_arms:
        events = evaluation["arm_summaries"][arm].get("events")
        if not isinstance(events, list) or len(events) != 469:
            raise ValidationError(f"evaluation event count mismatch: {arm}")
        event_ids = [event.get("event_id") for event in events]
        if len(set(event_ids)) != 469 or set(event_ids) != set(primary_natural):
            raise ValidationError(f"evaluation event keyset mismatch: {arm}")
    eval_index = {
        (arm, event["event_id"]): event
        for arm, summary in evaluation["arm_summaries"].items()
        for event in summary["events"]
    }
    if len(eval_index) != 938:
        raise ValidationError("evaluation arm/event key count mismatch")
    previous: dict[str, Mapping[str, Any]] = {}
    previous_by_key: dict[tuple[Any, ...], Mapping[str, Any] | None] = {}
    last_index: dict[str, int] = {}
    for row in replay:
        key = (row["capture_id"], row["target_id"], row["source_frame_id"])
        trow = truth_index.get(key)
        if trow is None or trow["bag_image_timestamp_ns"] != row["captured_at_ns"]:
            raise ValidationError(f"truth/replay join mismatch: {key}")
        index = trow["source_frame_index"]
        if row["target_id"] in last_index and index <= last_index[row["target_id"]]:
            raise ValidationError("replay source_frame_index order mismatch")
        previous_by_key[key] = previous.get(row["target_id"])
        previous[row["target_id"]] = row
        last_index[row["target_id"]] = index
    calibration = read_camera_from_marker_independent(
        paths["revel_calibration"]
    )
    topics = {
        "sensor": "/vicon/event_lidar/event_lidar",
        "track-000": "/vicon/helmet_green/helmet_green",
        "track-001": "/vicon/helmet_yellow/helmet_yellow",
    }
    tracks: dict[str, dict[str, Any]] = {}
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(paths["revel_dynamic_bag"]) as reader:
        for name, topic in topics.items():
            info = reader.topics.get(topic)
            if info is None or len(info.connections) != 1:
                raise ValidationError(f"Vicon topic binding mismatch: {topic}")
            connection = info.connections[0]
            timestamps, positions, quaternions = [], [], []
            for _, timestamp, raw in reader.messages(connections=[connection]):
                message = typestore.deserialize_ros1(raw, connection.msgtype).transform
                timestamps.append(timestamp)
                positions.append((message.translation.x, message.translation.y, message.translation.z))
                quaternions.append((message.rotation.x, message.rotation.y, message.rotation.z, message.rotation.w))
            tracks[name] = {
                "timestamps_ns": np.asarray(timestamps, dtype=np.int64),
                "positions": np.asarray(positions, dtype=np.float64),
                "quaternions": np.asarray(quaternions, dtype=np.float64),
            }
    truth_by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in truth:
        if row.get("event_id") in primary_natural:
            truth_by_event[row["event_id"]].append(row)
    for event_id, event in primary_natural.items():
        members = sorted(
            truth_by_event[event_id], key=lambda item: item["source_frame_index"]
        )
        expected_indices = list(
            range(event["start_source_frame_index"], event["end_source_frame_index"] + 1)
        )
        if (
            [member["source_frame_index"] for member in members] != expected_indices
            or len(members) != event["eligible_frame_count"]
            or members[0]["bag_image_timestamp_ns"] != event["start_timestamp_ns"]
            or members[-1]["bag_image_timestamp_ns"] != event["end_timestamp_ns"]
        ):
            raise ValidationError(f"natural-event membership mismatch: {event_id}")
        for member in members:
            if (
                member.get("capture_id") != event["capture_id"]
                or member.get("target_id") != event["target_id"]
                or member.get("truth_state") != event["truth_state"]
                or member.get("event_id") != event_id
                or member.get("event_anchor_region") != event["anchor_region"]
            ):
                raise ValidationError(
                    f"natural-event member identity mismatch: {event_id}"
                )
        for arm in ("BBOX_LOG_AREA_GROWTH", "ROI_SPARSE_RADIAL_FLOW"):
            evaluated = eval_index.get((arm, event_id))
            if (
                evaluated is None
                or evaluated.get("target_id") != event["target_id"]
                or evaluated.get("anchor_region") != event["anchor_region"]
                or evaluated.get("truth_state") != event["truth_state"]
                or evaluated.get("denominator_rows") != event["eligible_frame_count"]
                or evaluated.get("coverage")
                != evaluated.get("non_abstained_rows") / event["eligible_frame_count"]
            ):
                raise ValidationError(f"evaluation binding mismatch: {arm}/{event_id}")
    result: list[dict[str, Any]] = []
    for binding in dependency["event_bindings"]:
        event_id = binding["event_id"]
        event = primary_natural[event_id]
        members: list[dict[str, Any]] = []
        for trow in sorted(truth_by_event[event_id], key=lambda item: item["source_frame_index"]):
            key = (trow["capture_id"], trow["target_id"], trow["source_frame_id"])
            replay_row = replay_index[key]
            bbox = r2_index[("BBOX_LOG_AREA_GROWTH", *key)]
            flow = r2_index[("ROI_SPARSE_RADIAL_FLOW", *key)]
            source, source_reason = _independent_source_pair(
                trow["bag_image_timestamp_ns"], tracks[trow["target_id"]], tracks["sensor"],
                calibration, trow["truth_signed_approach_mps"],
            )
            members.append({
                "index": trow["source_frame_index"], "source": source, "source_reason": source_reason,
                "roi": _independent_roi_pair(previous_by_key[key], replay_row, bbox), "flow": flow,
            })
        result.append(_independent_event(
            event, binding, eval_index[("ROI_SPARSE_RADIAL_FLOW", event_id)],
            eval_index[("BBOX_LOG_AREA_GROWTH", event_id)], members,
        ))
    return result


def compare_recomputed_event_table(claimed: Sequence[Mapping[str, Any]], recomputed: Sequence[Mapping[str, Any]]) -> None:
    if len(claimed) != len(recomputed):
        raise ValidationError("recomputed event-table length mismatch")
    for claimed_row, expected_row in zip(claimed, recomputed):
        if claimed_row.get("event_id") != expected_row.get("event_id"):
            raise ValidationError("recomputed event-table order mismatch")
        _assert_exact(
            dict(claimed_row),
            dict(expected_row),
            f"$.event[{claimed_row['event_id']}]",
        )


def validate_formal_identities(
    formal_start: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Path]:
    if (
        formal_start.get("protocol_id") != PROTOCOL_ID
        or formal_start.get("protocol_sha256") != PROTOCOL_SHA256
    ):
        raise ValidationError("formal_start protocol identity mismatch")
    if "runtime_environment" in protocol:
        expected_formal_keys = {
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "activation",
            "implementation_lock",
            "repository",
            "state",
            "d0_metric_computation_pending",
            "vicon_bag_messages_opened",
            "prestart_dependency_smoke",
            "prestart_operational_probe",
        }
        if (
            set(formal_start) != expected_formal_keys
            or formal_start.get("schema_version")
            != "blindassist.d0_formal_start.v1"
            or formal_start.get("state") != "FORMAL_STARTED"
            or formal_start.get("d0_metric_computation_pending") is not True
        ):
            raise ValidationError("formal_start envelope mismatch")
        expected_probe = protocol["runtime_environment"][
            "designated_prestart_probe"
        ]
        formal_probe = formal_start.get("prestart_operational_probe")
        expected_formal_probe = {
            **expected_probe,
            "runtime_manifest_sha256": protocol["runtime_environment"][
                "manifest"
            ]["sha256"],
            "runtime_tree_sha256": protocol["runtime_environment"][
                "manifest"
            ]["tree_sha256"],
        }
        if (
            formal_start.get("vicon_bag_messages_opened") is not False
            or not isinstance(formal_probe, dict)
            or formal_probe != expected_formal_probe
        ):
            raise ValidationError("formal_start runtime probe mismatch")
        smoke = formal_start.get("prestart_dependency_smoke")
        fixture = (
            "scripts/research/"
            "dual_loop_d0_egomotion_error_attribution_r3/"
            "synthetic_calibration.yaml"
        )
        expected_smoke = {
            "status": "VALID_SYNTHETIC_RUNTIME_SMOKE",
            "imports": [
                "numpy",
                "yaml",
                "rosbags.rosbag1.Reader",
                "rosbags.typesys.Stores",
                "rosbags.typesys.get_typestore",
            ],
            "yaml_safe_load_called": True,
            "synthetic_calibration_shape": [4, 4],
            "synthetic_calibration_finite": True,
            "real_calibration_opened": False,
            "bag_messages_opened": False,
            "truth_opened": False,
            "event_rows_built": False,
            "d0_metrics_computed": False,
            "producer_calibration_parser": {
                "fixture": fixture,
                "shape": [4, 4],
                "finite": True,
                "values_retained": False,
            },
            "independent_calibration_parser": {
                "fixture": fixture,
                "shape": [4, 4],
                "finite": True,
                "values_retained": False,
            },
        }
        if smoke != expected_smoke:
            raise ValidationError("formal_start dependency smoke mismatch")
    repository = formal_start.get("repository")
    if (
        not isinstance(repository, dict)
        or not isinstance(repository.get("head"), str)
        or repository.get("head") != repository.get("origin_master")
    ):
        raise ValidationError("formal_start repository identity mismatch")

    activation_binding = formal_start.get("activation")
    if not isinstance(activation_binding, dict):
        raise ValidationError("formal_start activation identity missing")
    activation_path = _bound_file(
        repo_root, activation_binding.get("path"), label="formal_start activation"
    )
    if sha256_file(activation_path) != activation_binding.get("sha256"):
        raise ValidationError("formal_start activation hash mismatch")
    activation = load_canonical_json(activation_path)
    if (
        activation.get("schema_version") != ACTIVATION_SCHEMA
        or activation.get("protocol_id") != PROTOCOL_ID
        or activation.get("protocol_sha256") != PROTOCOL_SHA256
        or activation.get("execution_state") != "NOT_RUN"
        or activation.get("formal_output_root")
        != protocol["planned_implementation"]["formal_output_root"]
    ):
        raise ValidationError("activation identity/status mismatch")
    if not (
        activation.get("formal_execution_authorized") is True
        and activation.get("authority")
        == {
            "formal_execution_authorized": True,
            "successor_execution_authorized": False,
            "confirmation_authorized": False,
            "product_or_safety_authorized": False,
        }
    ):
        raise ValidationError("activation lacks formal execution authority")

    lock_binding = formal_start.get("implementation_lock")
    if not isinstance(lock_binding, dict):
        raise ValidationError("formal_start implementation-lock identity missing")
    lock_path = _bound_file(
        repo_root,
        lock_binding.get("path"),
        label="formal_start implementation lock",
    )
    lock_hash = sha256_file(lock_path)
    if lock_hash != lock_binding.get("sha256"):
        raise ValidationError("formal_start implementation-lock hash mismatch")
    activation_lock = activation.get("implementation_lock")
    if not isinstance(activation_lock, dict):
        raise ValidationError("activation implementation-lock binding missing")
    activation_lock_path = _bound_file(
        repo_root,
        activation_lock.get("path"),
        label="activation implementation lock",
    )
    if activation_lock_path != lock_path or activation_lock.get("sha256") != lock_hash:
        raise ValidationError("activation implementation-lock binding mismatch")
    lock = load_canonical_json(lock_path)
    lock_authority = lock.get("authority")
    if (
        lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA
        or lock.get("protocol_id") != PROTOCOL_ID
        or lock.get("implementation_status") != "FROZEN_FOR_INDEPENDENT_REVIEW"
        or lock.get("execution_state") != "NOT_RUN"
        or not isinstance(lock_authority, dict)
        or lock_authority.get("activation_authorized") is not False
        or lock_authority.get("formal_execution_authorized") is not False
        or lock_authority.get("scientific_exit_authorized") is not False
    ):
        raise ValidationError("implementation-lock identity/status mismatch")

    activation_repository = activation.get("repository")
    lock_repository = lock.get("repository")
    if (
        activation_repository != repository
        or lock_repository != repository
    ):
        raise ValidationError("repository identity mismatch across formal artifacts")
    if "runtime_environment" in protocol:
        runtime_execution = activation.get("runtime_execution")
        if not isinstance(runtime_execution, dict):
            raise ValidationError("activation runtime-execution binding missing")
        argv_contract = runtime_execution.get("argv_contract")
        if (
            runtime_execution.get("manifest")
            != protocol["runtime_environment"]["manifest"]
            or runtime_execution.get("python_executable")
            != protocol["runtime_environment"]["python_executable"]
            or not isinstance(argv_contract, dict)
            or argv_contract.get("adapter")
            != "scripts/run_dual_loop_d0_egomotion_error_attribution_r3.py"
            or argv_contract.get("command") != "produce"
        ):
            raise ValidationError(
                "activation runtime-execution identity mismatch"
            )
        if (
            _bound_file(
                repo_root,
                argv_contract.get("activation_path"),
                label="runtime activation path",
            )
            != activation_path
            or _bound_file(
                repo_root,
                argv_contract.get("implementation_lock_path"),
                label="runtime implementation lock path",
            )
            != lock_path
        ):
            raise ValidationError("activation runtime argv-path mismatch")

    review_binding = activation.get("implementation_review")
    if not isinstance(review_binding, dict):
        raise ValidationError("activation implementation-review binding missing")
    review_path = _bound_file(
        repo_root,
        review_binding.get("path"),
        label="activation implementation review",
    )
    if sha256_file(review_path) != review_binding.get("sha256"):
        raise ValidationError("activation implementation-review hash mismatch")
    review = load_canonical_json(review_path)
    review_checks = review.get("checks")
    if (
        review.get("schema_version") != IMPLEMENTATION_REVIEW_SCHEMA
        or review.get("status") != "PASS"
        or review.get("reviewer_role") != "INDEPENDENT_READ_ONLY_REVIEW"
        or review.get("protocol")
        != {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256}
        or review.get("implementation_lock")
        != {
            "path": activation_lock.get("path"),
            "sha256": lock_hash,
        }
        or review.get("repository") != repository
        or review.get("formal_execution_authorized") is not False
        or not isinstance(review_checks, list)
        or not review_checks
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in review_checks
        )
    ):
        raise ValidationError("activation implementation review is invalid")

    expected_protocol_binding = {
        "path": (
            "docs/research/dual-loop/"
            "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R3_PROTOCOL_2026-07-30.json"
        ),
        "sha256": PROTOCOL_SHA256,
    }
    if lock.get("protocol") != expected_protocol_binding:
        raise ValidationError("implementation-lock protocol binding mismatch")
    expected_predecessor = {
        name: dict(protocol["predecessor_gate"][name])
        for name in ("result", "independent_validation", "seal")
    }
    if lock.get("predecessor_bindings") != expected_predecessor:
        raise ValidationError("implementation-lock predecessor bindings mismatch")
    for name, binding in expected_predecessor.items():
        predecessor_path = _bound_file(
            repo_root,
            binding.get("path"),
            label=f"implementation-lock predecessor {name}",
        )
        if sha256_file(predecessor_path) != binding.get("sha256"):
            raise ValidationError(
                f"implementation-lock predecessor hash mismatch: {name}"
            )

    if "r1_failure_gate" in protocol:
        if (
            lock.get("scientific_contract_binding")
            != protocol["scientific_contract_binding"]
            or lock.get("r1_failure_gate") != protocol["r1_failure_gate"]
            or lock.get("runtime_environment") != protocol["runtime_environment"]
        ):
            raise ValidationError(
                "implementation-lock R1/R2 recovery binding mismatch"
            )
        r1_scientific_path = repo_root / (
            "docs/research/dual-loop/"
            "DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_PROTOCOL_2026-07-30.json"
        )
        if (
            sha256_file(r1_scientific_path)
            != protocol["scientific_contract_binding"]["sha256"]
        ):
            raise ValidationError("R1 scientific protocol binding mismatch")
        scientific_binding = protocol["scientific_contract_binding"]
        for filename, expected_hash in scientific_binding[
            "byte_identical_scientific_sources"
        ].items():
            for source_root in scientific_binding["source_lineage"]:
                source_path = _bound_file(
                    repo_root,
                    f"{source_root}/{filename}",
                    label=f"scientific source {source_root}/{filename}",
                )
                if sha256_file(source_path) != expected_hash:
                    raise ValidationError(
                        "scientific source identity mismatch: "
                        f"{source_root}/{filename}"
                    )
        failure_gate = protocol["r1_failure_gate"]
        for gate_name in (
            "result",
            "implementation_lock",
            "implementation_review",
            "activation",
            "invalid_implementation_lock_archive",
            "input_freeze_dependency_receipt",
            "formal_start",
            "failure_receipt",
            "progress",
        ):
            specification = failure_gate[gate_name]
            gate_path = _bound_file(
                repo_root,
                specification["path"],
                label=f"R1 recovery {gate_name}",
            )
            if sha256_file(gate_path) != specification["sha256"]:
                raise ValidationError(
                    f"R1 recovery hash mismatch: {gate_name}"
                )
            if gate_path.suffix == ".json":
                payload = load_canonical_json(gate_path)
                for key, expected in specification.items():
                    if key.startswith("required_"):
                        actual_key = key.removeprefix("required_")
                        if payload.get(actual_key) != expected:
                            raise ValidationError(
                                f"R1 recovery semantic mismatch: "
                                f"{gate_name}/{actual_key}"
                            )
        r1_run_root = repo_root / (
            "artifacts.local/evidence/dual-loop/"
            "d0-egomotion-error-attribution-r1/run-r1"
        )
        if sorted(
            path.name for path in r1_run_root.iterdir()
        ) != sorted(failure_gate["exact_run_file_set"]):
            raise ValidationError("R1 recovery exact run file set mismatch")
        r1_evidence_root = r1_run_root.parent
        if sorted(
            path.name for path in r1_evidence_root.iterdir()
        ) != sorted(failure_gate["exact_evidence_root_entry_set"]):
            raise ValidationError(
                "R1 recovery exact evidence root entry set mismatch"
            )
        r1_implementation_root = r1_evidence_root / "implementation"
        if sorted(
            path.name
            for path in r1_implementation_root.iterdir()
            if path.is_file()
        ) != sorted(failure_gate["exact_implementation_file_set"]):
            raise ValidationError(
                "R1 recovery exact implementation file set mismatch"
            )
        r1_input_freeze_root = r1_evidence_root / "input-freeze"
        if sorted(
            path.name
            for path in r1_input_freeze_root.iterdir()
            if path.is_file()
        ) != sorted(failure_gate["exact_input_freeze_file_set"]):
            raise ValidationError(
                "R1 recovery exact input-freeze file set mismatch"
            )
        if any(
            (r1_run_root / name).exists()
            for name in failure_gate["required_absent_outputs"]
        ):
            raise ValidationError("R1 recovery absent-output set mismatch")
        if (
            failure_gate.get("r1_rerun_authorized") is not False
            or failure_gate.get("r2_is_r1_rerun") is not False
        ):
            raise ValidationError("R1 recovery rerun boundary mismatch")

        r2_failure_gate = protocol["r2_failure_gate"]
        if lock.get("r2_failure_gate") != r2_failure_gate:
            raise ValidationError(
                "implementation-lock R2 failure binding mismatch"
            )
        for gate_name in (
            "result",
            "protocol",
            "implementation_lock",
            "implementation_review",
            "activation",
            "activation_archive",
            "implementation_lock_archive",
            "implementation_review_archive",
            "runtime_manifest",
            "formal_start",
            "progress",
            "failure_receipt",
        ):
            specification = r2_failure_gate[gate_name]
            gate_path = _bound_file(
                repo_root,
                specification["path"],
                label=f"R2 recovery {gate_name}",
            )
            if sha256_file(gate_path) != specification["sha256"]:
                raise ValidationError(
                    f"R2 recovery hash mismatch: {gate_name}"
                )
            if gate_path.suffix != ".json":
                continue
            payload = load_canonical_json(gate_path)
            for key, expected in specification.items():
                if not key.startswith("required_"):
                    continue
                actual_key = key.removeprefix("required_")
                if actual_key.startswith("probe_"):
                    actual = payload.get(
                        "prestart_operational_probe", {}
                    ).get(actual_key.removeprefix("probe_"))
                else:
                    actual = payload.get(actual_key)
                if actual != expected:
                    raise ValidationError(
                        "R2 recovery semantic mismatch: "
                        f"{gate_name}/{actual_key}"
                    )
        r2_run_root = repo_root / (
            "artifacts.local/evidence/dual-loop/"
            "d0-egomotion-error-attribution-r2/run-r2"
        )
        actual_r2_files = sorted(
            path.name for path in r2_run_root.iterdir()
        )
        if actual_r2_files != sorted(
            r2_failure_gate["exact_run_file_set"]
        ):
            raise ValidationError("R2 recovery exact run file set mismatch")
        r2_evidence_root = r2_run_root.parent
        if sorted(
            path.name for path in r2_evidence_root.iterdir()
        ) != sorted(r2_failure_gate["exact_evidence_root_entry_set"]):
            raise ValidationError(
                "R2 recovery exact evidence root entry set mismatch"
            )
        r2_implementation_root = r2_evidence_root / "implementation"
        if sorted(
            path.name
            for path in r2_implementation_root.iterdir()
            if path.is_file()
        ) != sorted(r2_failure_gate["exact_implementation_file_set"]):
            raise ValidationError(
                "R2 recovery exact implementation file set mismatch"
            )
        if any(
            (r2_run_root / name).exists()
            for name in r2_failure_gate["required_absent_outputs"]
        ):
            raise ValidationError("R2 recovery absent-output set mismatch")
        if (
            r2_failure_gate.get("r2_rerun_authorized") is not False
            or r2_failure_gate.get("r3_is_r2_rerun") is not False
        ):
            raise ValidationError("R2 recovery rerun boundary mismatch")

        runtime_specification = protocol["runtime_environment"]
        runtime_manifest_path = _bound_file(
            repo_root,
            runtime_specification["manifest"]["path"],
            label="R3 runtime environment manifest",
        )
        if (
            sha256_file(runtime_manifest_path)
            != runtime_specification["manifest"]["sha256"]
        ):
            raise ValidationError("R3 runtime manifest hash mismatch")
        runtime_validation = validate_runtime_manifest(runtime_manifest_path)
        if (
            runtime_validation.get("status") != "VALID"
            or runtime_validation.get("tree_sha256")
            != runtime_specification["manifest"]["tree_sha256"]
        ):
            raise ValidationError("R3 live runtime validation mismatch")

    expected_inputs = {
        name: _normalized_input_binding(specification)
        for name, specification in protocol["frozen_inputs"].items()
    }
    if (
        lock.get("frozen_inputs") != expected_inputs
        or lock.get("frozen_inputs_sha256") != _canonical_sha256(expected_inputs)
    ):
        raise ValidationError("implementation-lock frozen-input bindings mismatch")
    canonical_contract = protocol["planned_implementation"][
        "canonical_serialization"
    ]
    if lock.get("canonical_serialization_sha256") != _canonical_sha256(
        canonical_contract
    ):
        raise ValidationError(
            "implementation-lock canonical contract hash mismatch"
        )

    expected_paths = _expected_implementation_paths(repo_root, protocol)
    implementation_hashes = lock.get("implementation_file_hashes")
    if (
        not isinstance(implementation_hashes, dict)
        or set(implementation_hashes) != expected_paths
    ):
        raise ValidationError("implementation-lock file inventory mismatch")
    for relative, expected_hash in implementation_hashes.items():
        implementation_path = _bound_file(
            repo_root,
            relative,
            label=f"implementation-lock source {relative}",
        )
        if sha256_file(implementation_path) != expected_hash:
            raise ValidationError(
                f"implementation-lock source hash mismatch: {relative}"
            )
    return {"activation_path": activation_path, "implementation_lock_path": lock_path}


def resolve_expected_run_root(
    protocol_path: Path,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    resolved_protocol = protocol_path.resolve()
    if protocol is None:
        if sha256_file(resolved_protocol) != PROTOCOL_SHA256:
            raise ValidationError("protocol SHA-256 mismatch")
        protocol = parse_json_bytes(
            resolved_protocol.read_bytes(),
            label=str(resolved_protocol),
        )
    repo_root = resolved_protocol.parents[3]
    return (
        repo_root / protocol["planned_implementation"]["formal_output_root"]
    ).resolve()


def validate_execution_package(
    run_root: Path,
    *,
    protocol_path: Path,
    dependency_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    event_hash: str | None = None
    analysis_hash: str | None = None
    final_exit: str | None = None
    claim_ceiling: str | None = None
    try:
        if sha256_file(protocol_path) != PROTOCOL_SHA256:
            raise ValidationError("protocol SHA-256 mismatch")
        protocol = parse_json_bytes(protocol_path.read_bytes(), label=str(protocol_path))
        if run_root.resolve() != resolve_expected_run_root(
            protocol_path, protocol
        ):
            raise ValidationError("run root differs from frozen R3 namespace")
        claim_ceiling = protocol.get("claim_ceiling")
        if not isinstance(claim_ceiling, str) or not claim_ceiling:
            raise ValidationError("protocol claim ceiling missing")
        if sha256_file(dependency_path) != DEPENDENCY_SHA256:
            raise ValidationError("dependency receipt SHA-256 mismatch")
        if (run_root / "failure_receipt.json").exists():
            raise ValidationError("failure receipt exists in success namespace")
        required = (
            "formal_start.json",
            "event_table.jsonl",
            "analysis.json",
            "producer_receipt.json",
        )
        missing = [name for name in required if not (run_root / name).is_file()]
        if missing:
            raise ValidationError(f"missing producer outputs: {missing}")

        formal_start = load_canonical_json(run_root / "formal_start.json")
        repo_root = protocol_path.resolve().parents[3]
        validate_formal_identities(
            formal_start,
            protocol=protocol,
            repo_root=repo_root,
        )

        dependency = parse_json_bytes(
            dependency_path.read_bytes(),
            label=str(dependency_path),
        )
        dependency_protocol_id = expected_dependency_protocol_id(protocol)
        if (
            dependency.get("schema_version") != "blindassist.d0_dependency_receipt.v1"
            or dependency.get("status") != "VALID"
            or dependency.get("protocol_id") != dependency_protocol_id
        ):
            raise ValidationError("dependency receipt identity/status mismatch")
        forbidden_dependency_flags = (
            "old_f1b_decision_opened",
            "production_ab_trace_opened",
            "confirmation_opened",
            "candidate_output_opened",
        )
        if any(dependency.get(flag) is not False for flag in forbidden_dependency_flags):
            raise ValidationError("dependency receipt forbidden-access flag is not false")

        event_path = run_root / "event_table.jsonl"
        rows = load_canonical_jsonl(event_path)
        validate_event_rows(rows, dependency)
        recomputed_rows = independently_recompute_event_table(
            repo_root,
            protocol,
        )
        recomputed_rows = parse_json_bytes(
            canonical_json_bytes(recomputed_rows),
            label="independently recomputed event table",
        )
        compare_recomputed_event_table(rows, recomputed_rows)
        event_hash = sha256_file(event_path)

        analysis_path = run_root / "analysis.json"
        claimed_analysis = load_canonical_json(analysis_path)
        analysis_hash = sha256_file(analysis_path)
        recomputed_analysis = recompute_analysis(
            rows,
            protocol_sha256=PROTOCOL_SHA256,
            event_table_sha256=event_hash,
        )
        recomputed_analysis = parse_json_bytes(
            canonical_json_bytes(recomputed_analysis),
            label="independently recomputed analysis",
        )
        _assert_exact(claimed_analysis, recomputed_analysis, "$.analysis")

        receipt = load_canonical_json(run_root / "producer_receipt.json")
        if (
            receipt.get("schema_version") != PRODUCER_RECEIPT_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("protocol_sha256") != PROTOCOL_SHA256
            or receipt.get("status") != "PRODUCER_COMPLETE_NOT_YET_VALID"
        ):
            raise ValidationError("producer receipt identity/status mismatch")
        if receipt.get("formal_start_sha256") != sha256_file(run_root / "formal_start.json"):
            raise ValidationError("producer receipt formal_start hash mismatch")
        if receipt.get("event_table") != {
            "path": str(event_path),
            "sha256": event_hash,
            "rows": len(rows),
        }:
            raise ValidationError("producer receipt event_table descriptor mismatch")
        if receipt.get("analysis") != {
            "path": str(analysis_path),
            "sha256": analysis_hash,
        }:
            raise ValidationError("producer receipt analysis descriptor mismatch")
        forbidden = receipt.get("forbidden_access")
        if not isinstance(forbidden, dict) or forbidden != {
            "old_f1b_decision_opened": False,
            "production_ab_trace_opened": False,
            "confirmation_opened": False,
        }:
            raise ValidationError("producer receipt forbidden-access flags mismatch")
        if receipt.get("errors") != []:
            raise ValidationError("producer receipt contains errors")
        frozen = receipt.get("frozen_inputs")
        if not isinstance(frozen, dict):
            raise ValidationError("producer receipt frozen_inputs missing")
        for binding_name, contract in protocol["frozen_inputs"].items():
            observed = frozen.get(binding_name)
            if not isinstance(observed, dict):
                raise ValidationError(f"producer receipt missing frozen input {binding_name}")
            if "sha256" in contract and observed.get("sha256") != contract["sha256"]:
                raise ValidationError(f"frozen input hash mismatch: {binding_name}")
            if "rows" in contract and observed.get("rows") != contract["rows"]:
                raise ValidationError(f"frozen input row count mismatch: {binding_name}")

        routing = recomputed_analysis["routing"]
        final_exit = scientific_exit(
            execution_valid=True,
            global_route_evaluable=routing["global_route_evaluable"],
            ego_candidate=routing["ego_candidate"],
            temporal_candidate=routing["temporal_candidate"],
        )
    except Exception as exc:
        errors.append(str(exc))

    passed = not errors
    return {
        "schema_version": VALIDATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "PASS" if passed else "FAIL",
        "execution_valid": passed,
        "terminal": (
            "VALID"
            if passed
            else "EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT"
        ),
        "event_table_sha256": event_hash,
        "analysis_sha256": analysis_hash,
        "scientific_exit": final_exit if passed else None,
        "claim_ceiling": claim_ceiling,
        "errors": errors,
    }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_same_volume_temp(path: Path, payload: Mapping[str, Any]) -> Path:
    data = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("temporary publication write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return temporary


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fully fsynced same-volume temporary without overwriting."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = _write_same_volume_temp(path, payload)
    try:
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _write_same_volume_temp(path, payload)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _failure_receipt(
    result: Mapping[str, Any],
    *,
    validation_sha256: str | None,
    publication_error: str | None = None,
) -> dict[str, Any]:
    errors = list(result.get("errors", []))
    if publication_error is not None:
        errors.append(publication_error)
    return {
        "schema_version": "blindassist.d0_failure_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "status": "EXECUTION_INVALID",
        "consumed": True,
        "rerun_authorized": False,
        "scientific_exit": None,
        "execution_validation_sha256": validation_sha256,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dependency-receipt", type=Path, required=True)
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="exclusively publish execution_validation and, on PASS, execution_receipt",
    )
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    try:
        expected_run_root = resolve_expected_run_root(args.protocol.resolve())
    except Exception as error:
        print(
            canonical_json_bytes(
                {
                    "status": "PRESTART_INVALID",
                    "formal_execution_authorized": False,
                    "error": str(error),
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    if run_root != expected_run_root:
        print(
            canonical_json_bytes(
                {
                    "status": "PRESTART_INVALID",
                    "formal_execution_authorized": False,
                    "error": "run root differs from frozen R3 namespace",
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    if (
        not run_root.is_dir()
        or not (run_root / "formal_start.json").is_file()
    ):
        print(
            canonical_json_bytes(
                {
                    "status": "PRESTART_INVALID",
                    "formal_execution_authorized": False,
                    "error": (
                        "frozen R3 namespace and formal_start.json must "
                        "already exist"
                    ),
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    if args.write_results and (
        (run_root / "failure_receipt.json").exists()
        or (run_root / "execution_receipt.json").exists()
    ):
        print(
            canonical_json_bytes(
                {
                    "status": "TERMINAL_IMMUTABLE",
                    "formal_execution_authorized": False,
                    "rerun_authorized": False,
                    "error": (
                        "a terminal receipt already exists; validation "
                        "publication cannot be retried"
                    ),
                }
            ).decode("utf-8"),
            end="",
        )
        return 2
    result = validate_execution_package(
        run_root,
        protocol_path=args.protocol.resolve(),
        dependency_path=args.dependency_receipt.resolve(),
    )
    if args.write_results:
        validation_path = run_root / "execution_validation.json"
        failure_path = run_root / "failure_receipt.json"
        try:
            _exclusive_write(validation_path, result)
        except BaseException as error:
            if not failure_path.exists():
                _exclusive_write(
                    failure_path,
                    _failure_receipt(
                        result,
                        validation_sha256=None,
                        publication_error=(
                            f"execution validation publication failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ),
                )
            raise
        validation_sha = sha256_file(validation_path)
        if result["execution_valid"]:
            receipt = {
                "schema_version": EXECUTION_RECEIPT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": PROTOCOL_SHA256,
                "status": "VALID",
                "execution_validation_sha256": validation_sha,
                "scientific_exit": result["scientific_exit"],
                "claim_ceiling": result["claim_ceiling"],
            }
            try:
                _atomic_replace(
                    run_root / "progress.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "state": "VALID",
                        "scientific_exit": result["scientific_exit"],
                        "completed_event_count": 469,
                        "expected_event_count": 469,
                    },
                )
                _exclusive_write(
                    run_root / "execution_receipt.json",
                    receipt,
                )
            except BaseException as error:
                if not failure_path.exists():
                    _exclusive_write(
                        failure_path,
                        _failure_receipt(
                            result,
                            validation_sha256=validation_sha,
                            publication_error=(
                                f"valid receipt publication failed: "
                                f"{type(error).__name__}: {error}"
                            ),
                        ),
                    )
                try:
                    _atomic_replace(
                        run_root / "progress.json",
                        {
                            "protocol_id": PROTOCOL_ID,
                            "state": "EXECUTION_INVALID",
                            "scientific_exit": None,
                            "errors": [
                                "VALID_TERMINAL_PUBLICATION_FAILED"
                            ],
                        },
                    )
                except BaseException:
                    pass
                raise
        else:
            if not failure_path.exists():
                _exclusive_write(
                    failure_path,
                    _failure_receipt(
                        result,
                        validation_sha256=validation_sha,
                    ),
                )
            _atomic_replace(
                run_root / "progress.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "state": "EXECUTION_INVALID",
                    "scientific_exit": None,
                    "errors": list(result["errors"]),
                },
            )
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0 if result["execution_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
