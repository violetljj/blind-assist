#!/usr/bin/env python3
"""Evaluate current YOLO box to source-native world-body measurements."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    infer_scene_column,
)
from extract_stage_c_d29_thor_magni_object_slots import (
    DEFAULT_OUTPUT as DEFAULT_OBJECT_SLOTS,
    SCHEMA as OBJECT_SLOT_SCHEMA,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    read_scenario,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_SAMPLES,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    prepare_records,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d30_thor_magni_"
    "box_world_measurement_v0"
)
HALF_FOV_DEGREES = 50.0
DISTANCE_CAP_M = 10.0
ACCEPTED_X_ERROR = 0.25
MIN_SOURCE_PAIRS = 5
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d30-thor-magni-box-world-measurement-v0/report.json"
)


def is_person_body(body_name: str, role: str) -> bool:
    return body_name.startswith("Helmet_") and (
        role.startswith("Visitors-") or role.startswith("Carrier-")
    )


def relative_bearing_degrees(
    forward: np.ndarray,
    relative: np.ndarray,
) -> float:
    forward = np.asarray(forward, dtype=np.float64)
    relative = np.asarray(relative, dtype=np.float64)
    if forward.shape != (2,) or relative.shape != (2,):
        raise ValueError("D30 bearing requires 2D vectors")
    forward_norm = float(np.linalg.norm(forward))
    relative_norm = float(np.linalg.norm(relative))
    if forward_norm <= 0 or relative_norm <= 0:
        raise ValueError("D30 bearing vector is degenerate")
    unit_forward = forward / forward_norm
    unit_relative = relative / relative_norm
    cross = (
        unit_forward[0] * unit_relative[1]
        - unit_forward[1] * unit_relative[0]
    )
    dot = float(np.dot(unit_forward, unit_relative))
    return math.degrees(math.atan2(cross, dot))


def assign_measurements(
    box_x_signed: np.ndarray,
    body_bearing_degrees: np.ndarray,
) -> list[dict[str, Any]]:
    boxes = np.asarray(box_x_signed, dtype=np.float64)
    bearings = np.asarray(body_bearing_degrees, dtype=np.float64)
    if boxes.ndim != 1 or bearings.ndim != 1:
        raise ValueError("D30 assignments require 1D inputs")
    if len(boxes) == 0 or len(bearings) == 0:
        return []
    predicted_x = -bearings / HALF_FOV_DEGREES
    cost = np.abs(boxes[:, None] - predicted_x[None, :])
    rows, columns = linear_sum_assignment(cost)
    output = []
    for row, column in zip(rows, columns, strict=True):
        x_error = float(cost[row, column])
        output.append(
            {
                "box_index": int(row),
                "body_index": int(column),
                "box_x_signed": float(boxes[row]),
                "predicted_x_signed": float(predicted_x[column]),
                "x_error": x_error,
                "bearing_error_degrees": (
                    x_error * HALF_FOV_DEGREES
                ),
                "accepted": x_error <= ACCEPTED_X_ERROR,
            }
        )
    return output


def safe_correlation(
    x: list[float],
    y: list[float],
    kind: str,
) -> float | None:
    if len(x) < 3 or len(y) != len(x):
        return None
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    if kind == "pearson":
        value = pearsonr(x, y).statistic
    elif kind == "spearman":
        value = spearmanr(x, y).statistic
    else:
        raise ValueError(f"Unknown D30 correlation: {kind}")
    return float(value) if np.isfinite(value) else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assigned = [
        pair
        for row in rows
        for pair in row["assignments"]
    ]
    accepted = [pair for pair in assigned if pair["accepted"]]
    bearing_pearson = safe_correlation(
        [pair["box_x_signed"] for pair in assigned],
        [pair["predicted_x_signed"] for pair in assigned],
        "pearson",
    )
    distance_spearman = safe_correlation(
        [pair["box_height"] for pair in assigned],
        [1.0 / pair["body_distance_m"] for pair in assigned],
        "spearman",
    )
    eligible_nearest = sum(
        bool(row["body_distances_m"]) for row in rows
    )
    accepted_nearest = sum(
        bool(row["nearest_body_accepted"]) for row in rows
    )
    return {
        "anchors": len(rows),
        "anchors_with_boxes": sum(bool(row["box_count"]) for row in rows),
        "anchors_with_visible_bodies": sum(
            bool(row["body_count"]) for row in rows
        ),
        "anchors_with_both": sum(
            bool(row["box_count"] and row["body_count"]) for row in rows
        ),
        "assigned_pairs": len(assigned),
        "accepted_pairs": len(accepted),
        "accepted_fraction": (
            len(accepted) / len(assigned) if assigned else 0.0
        ),
        "box_assignment_fraction": (
            len(assigned)
            / sum(int(row["box_count"]) for row in rows)
            if sum(int(row["box_count"]) for row in rows)
            else 0.0
        ),
        "body_assignment_fraction": (
            len(assigned)
            / sum(int(row["body_count"]) for row in rows)
            if sum(int(row["body_count"]) for row in rows)
            else 0.0
        ),
        "nearest_body_eligible_anchors": eligible_nearest,
        "nearest_body_accepted_anchors": accepted_nearest,
        "nearest_body_accepted_coverage": (
            accepted_nearest / eligible_nearest
            if eligible_nearest
            else 0.0
        ),
        "bearing_mae_degrees": (
            float(
                np.mean(
                    [
                        pair["bearing_error_degrees"]
                        for pair in assigned
                    ]
                )
            )
            if assigned
            else None
        ),
        "box_x_predicted_x_pearson": bearing_pearson,
        "height_inverse_distance_spearman": distance_spearman,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--d8-samples", type=Path, default=DEFAULT_D8_SAMPLES)
    parser.add_argument(
        "--object-slots",
        type=Path,
        default=DEFAULT_OBJECT_SLOTS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D30 report output is non-overwriting")
    slot_report_path = args.object_slots.with_suffix(
        args.object_slots.suffix + ".json"
    )
    slot_report = json.loads(slot_report_path.read_text(encoding="utf-8"))
    if (
        slot_report["schema"] != OBJECT_SLOT_SCHEMA
        or slot_report["output"]["sha256"] != sha256(args.object_slots)
    ):
        raise ValueError("D30 object-slot binding mismatch")
    records = prepare_records(
        load_jsonl(args.samples),
        load_jsonl(args.d8_samples),
    )
    d8_by_id = {
        str(record["sample_id"]): record
        for record in load_jsonl(args.d8_samples)
    }
    payload = np.load(args.object_slots)
    sample_ids = [str(value) for value in payload["sample_ids"]]
    if sample_ids != [str(record["sample_id"]) for record in records]:
        raise ValueError("D30 object-slot ordering mismatch")
    slots = np.asarray(payload["slots"], dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)
    trajectory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for sample_index, record in enumerate(records):
        d8 = d8_by_id[str(record["sample_id"])]
        scenario_path = Path(str(d8["scenario_csv_path"]))
        camera_body = str(d8["camera_body"])
        key = (str(scenario_path.resolve()), camera_body)
        data = trajectory_cache.get(key)
        if data is None:
            data = read_scenario(
                scenario_path,
                camera_body,
                infer_scene_column(scenario_path, camera_body),
            )
            trajectory_cache[key] = data
        matches = np.flatnonzero(
            data["frames"] == int(d8["qtm_frame"])
        )
        if len(matches) != 1:
            raise ValueError("D30 QTM anchor is not unique")
        index = int(matches[0])
        before = index - 25
        after = index + 25
        velocity = (
            data["camera"][after, :2] - data["camera"][before, :2]
        ) / (data["times"][after] - data["times"][before])
        speed = float(np.linalg.norm(velocity))
        if speed < 0.25 or not np.isfinite(speed):
            raise ValueError("D30 wearer forward is invalid")
        forward = velocity / speed
        origin = data["camera"][index, :2]
        body_rows = []
        for body_name, positions in data["others"].items():
            role = str(data["roles"].get(body_name, ""))
            if not is_person_body(str(body_name), role):
                continue
            position = positions[index, :2]
            if not np.isfinite(position).all():
                continue
            relative = position - origin
            distance = float(np.linalg.norm(relative))
            if distance <= 0 or distance > DISTANCE_CAP_M:
                continue
            bearing = relative_bearing_degrees(forward, relative)
            if abs(bearing) > HALF_FOV_DEGREES:
                continue
            body_rows.append(
                {
                    "body_name": str(body_name),
                    "body_role": role,
                    "bearing_degrees": bearing,
                    "distance_m": distance,
                }
            )
        slot_rows = slots[sample_index][mask[sample_index]]
        box_x = slot_rows[:, 0] if len(slot_rows) else np.empty(0)
        assignments = assign_measurements(
            box_x,
            np.asarray(
                [row["bearing_degrees"] for row in body_rows],
                dtype=np.float64,
            ),
        )
        accepted_body_indices = {
            int(pair["body_index"])
            for pair in assignments
            if pair["accepted"]
        }
        nearest_body_index = (
            int(
                np.argmin(
                    [row["distance_m"] for row in body_rows]
                )
            )
            if body_rows
            else None
        )
        for pair in assignments:
            box_index = int(pair["box_index"])
            body_index = int(pair["body_index"])
            pair["box_height"] = float(slot_rows[box_index, 3])
            pair["box_confidence"] = float(slot_rows[box_index, 5])
            pair["body_distance_m"] = float(
                body_rows[body_index]["distance_m"]
            )
            pair["body_name"] = body_rows[body_index]["body_name"]
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "source_session_id": str(record["source_session_id"]),
                "fold": int(record["fold"]),
                "box_count": len(slot_rows),
                "body_count": len(body_rows),
                "body_distances_m": [
                    float(row["distance_m"]) for row in body_rows
                ],
                "assignments": assignments,
                "nearest_body_accepted": (
                    nearest_body_index in accepted_body_indices
                    if nearest_body_index is not None
                    else False
                ),
            }
        )
    pooled = summarize(rows)
    by_source = []
    for source in sorted(
        {str(row["source_session_id"]) for row in rows}
    ):
        source_rows = [
            row for row in rows if row["source_session_id"] == source
        ]
        summary = summarize(source_rows)
        summary["source_session_id"] = source
        by_source.append(summary)
    evaluable_sources = [
        row for row in by_source if row["assigned_pairs"] >= MIN_SOURCE_PAIRS
    ]
    source_macro = {
        "evaluable_sources": len(evaluable_sources),
        "box_x_predicted_x_pearson": float(
            np.mean(
                [
                    row["box_x_predicted_x_pearson"]
                    for row in evaluable_sources
                    if row["box_x_predicted_x_pearson"] is not None
                ]
            )
        ),
        "bearing_mae_degrees": float(
            np.mean(
                [
                    row["bearing_mae_degrees"]
                    for row in evaluable_sources
                    if row["bearing_mae_degrees"] is not None
                ]
            )
        ),
        "height_inverse_distance_spearman": float(
            np.mean(
                [
                    row["height_inverse_distance_spearman"]
                    for row in evaluable_sources
                    if row["height_inverse_distance_spearman"] is not None
                ]
            )
        ),
        "by_source": by_source,
    }
    by_fold = []
    for fold in range(5):
        summary = summarize(
            [row for row in rows if int(row["fold"]) == fold]
        )
        summary["fold"] = fold
        by_fold.append(summary)
    positive_distance_folds = sum(
        row["height_inverse_distance_spearman"] is not None
        and row["height_inverse_distance_spearman"] > 0
        for row in by_fold
    )
    checks = {
        "anchor_opportunity": pooled["anchors_with_both"] >= 300,
        "accepted_assignment_fraction": (
            pooled["accepted_fraction"] >= 0.60
        ),
        "nearest_body_coverage": (
            pooled["nearest_body_accepted_coverage"] >= 0.60
        ),
        "source_macro_bearing_pearson": (
            source_macro["box_x_predicted_x_pearson"] >= 0.50
        ),
        "source_macro_bearing_mae": (
            source_macro["bearing_mae_degrees"] <= 15.0
        ),
        "source_macro_distance_spearman": (
            source_macro["height_inverse_distance_spearman"] >= 0.30
        ),
        "positive_distance_folds": positive_distance_folds >= 3,
        "evaluable_sources": len(evaluable_sources) >= 15,
    }
    supported = all(checks.values())
    status = (
        "D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_SUPPORTED"
        if supported
        else (
            "D30_THOR_MAGNI_BOX_WORLD_MEASUREMENT_RELATION_"
            "NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development current-frame box-to-world measurement "
                "diagnostic"
            ),
            "future_body_positions_read": False,
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "object_slots_path": str(args.object_slots.resolve()),
            "object_slots_sha256": sha256(args.object_slots),
            "object_slots_report_sha256": sha256(slot_report_path),
        },
        "design": {
            "half_fov_degrees": HALF_FOV_DEGREES,
            "distance_cap_m": DISTANCE_CAP_M,
            "accepted_x_error": ACCEPTED_X_ERROR,
            "accepted_bearing_error_degrees": (
                ACCEPTED_X_ERROR * HALF_FOV_DEGREES
            ),
            "assignment": (
                "Hungarian absolute current-box-x to fixed-FOV body-bearing"
            ),
            "future_outcome_read": False,
            "person_body_rule": (
                "body name Helmet_* and role prefix Visitors- or Carrier-"
            ),
            "excluded_nonperson_examples": ["DARKO_Robot", "LO1"],
        },
        "pooled": pooled,
        "source_macro": source_macro,
        "by_fold": by_fold,
        "gate": {
            "frozen_thresholds": {
                "anchors_with_both": 300,
                "accepted_fraction": 0.60,
                "nearest_body_accepted_coverage": 0.60,
                "source_macro_box_x_pearson": 0.50,
                "source_macro_bearing_mae_degrees": 15.0,
                "source_macro_distance_spearman": 0.30,
                "positive_distance_folds": 3,
                "evaluable_sources": 15,
            },
            "positive_distance_folds": positive_distance_folds,
            "checks": checks,
            "supported": supported,
        },
        "next_action": (
            "freeze an explicit world-state filter canary"
            if supported
            else (
                "stop indirect THOR box-to-world fitting and use a source "
                "with native 2D/3D identity binding"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_hash = sha256(args.output)
    sidecar.write_text(
        f"{report_hash}  {args.output.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "status": status,
                "pooled": pooled,
                "source_macro": {
                    key: value
                    for key, value in source_macro.items()
                    if key != "by_source"
                },
                "gate": report["gate"],
                "report_sha256": report_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
