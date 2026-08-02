#!/usr/bin/env python3
"""Evaluate whether causal THOR history contains future-risk information."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_SAMPLES,
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    CORRIDOR_FORWARD_LIMIT_M,
    CORRIDOR_HALF_WIDTH_M,
    FUTURE_HORIZON_SECONDS,
    read_scenario,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d11_thor_magni_"
    "causal_kinematic_information_ceiling_v0"
)
HISTORY_SECONDS = 0.8
PROJECTION_STEP_SECONDS = 0.05
MIN_HISTORY_SPAN_SECONDS = 0.5
MIN_HISTORY_ROWS = 20


def anchor_index(
    frames: np.ndarray,
    times: np.ndarray,
    qtm_frame: int,
    qtm_time: float,
) -> int:
    candidates = np.flatnonzero(frames == qtm_frame)
    if len(candidates) == 0:
        raise ValueError(f"Missing QTM frame {qtm_frame}")
    difference = np.abs(times[candidates] - qtm_time)
    index = int(candidates[int(np.argmin(difference))])
    if abs(float(times[index]) - qtm_time) > 1e-6:
        raise ValueError(
            f"QTM frame/time binding mismatch: {qtm_frame} / {qtm_time}"
        )
    return index


def label_axis(camera: np.ndarray, times: np.ndarray, index: int) -> np.ndarray:
    before = index - 25
    after = index + 25
    if before < 0 or after >= len(times):
        raise ValueError("Missing target-frame direction support")
    delta = camera[after, :2] - camera[before, :2]
    norm = float(np.linalg.norm(delta))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid target-frame direction")
    return delta / norm


def causal_relative_velocity(
    times: np.ndarray,
    camera: np.ndarray,
    body: np.ndarray,
    index: int,
) -> np.ndarray | None:
    """Fit relative XY velocity using rows at or before the anchor only."""
    anchor_time = float(times[index])
    selection = (
        (times >= anchor_time - HISTORY_SECONDS)
        & (times <= anchor_time)
        & np.isfinite(camera[:, :2]).all(axis=1)
        & np.isfinite(body[:, :2]).all(axis=1)
    )
    selected_times = times[selection]
    if (
        len(selected_times) < MIN_HISTORY_ROWS
        or float(selected_times[-1] - selected_times[0])
        < MIN_HISTORY_SPAN_SECONDS
    ):
        return None
    relative = body[selection, :2] - camera[selection, :2]
    centered_time = selected_times - float(np.mean(selected_times))
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        return None
    centered_relative = relative - np.mean(relative, axis=0)
    velocity = np.sum(
        centered_relative * centered_time[:, None],
        axis=0,
    ) / denominator
    if not np.isfinite(velocity).all():
        return None
    return velocity


def risk_scores(
    relative_positions: list[np.ndarray],
    relative_velocities: list[np.ndarray],
    forward: np.ndarray,
) -> tuple[float, float]:
    """Return continuous proximity and corridor risk scores."""
    if len(relative_positions) == 0:
        raise ValueError("No current body geometry")
    horizon = np.arange(
        0.0,
        FUTURE_HORIZON_SECONDS + 1e-9,
        PROJECTION_STEP_SECONDS,
        dtype=np.float64,
    )
    lateral_axis = np.asarray((-forward[1], forward[0]))
    minimum_distance = math.inf
    minimum_corridor_violation = math.inf
    for position, velocity in zip(
        relative_positions,
        relative_velocities,
        strict=True,
    ):
        projected = position[None] + horizon[:, None] * velocity[None]
        distance = np.linalg.norm(projected, axis=1)
        minimum_distance = min(
            minimum_distance,
            float(np.min(distance)),
        )
        longitudinal = projected @ forward
        lateral = projected @ lateral_axis
        violation = np.maximum.reduce(
            (
                -longitudinal,
                longitudinal - CORRIDOR_FORWARD_LIMIT_M,
                np.abs(lateral) - CORRIDOR_HALF_WIDTH_M,
            )
        )
        minimum_corridor_violation = min(
            minimum_corridor_violation,
            float(np.min(violation)),
        )
    return -minimum_distance, -minimum_corridor_violation


def score_sample(
    record: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    index = anchor_index(
        data["frames"],
        data["times"],
        int(record["qtm_frame"]),
        float(record["qtm_time_seconds"]),
    )
    forward = label_axis(data["camera"], data["times"], index)
    positions = []
    static_velocities = []
    history_velocities = []
    fitted = 0
    for body in data["others"].values():
        if not (
            np.isfinite(data["camera"][index, :2]).all()
            and np.isfinite(body[index, :2]).all()
        ):
            continue
        positions.append(
            body[index, :2] - data["camera"][index, :2]
        )
        static_velocities.append(np.zeros(2, dtype=np.float64))
        velocity = causal_relative_velocity(
            data["times"],
            data["camera"],
            body,
            index,
        )
        if velocity is None:
            history_velocities.append(
                np.zeros(2, dtype=np.float64)
            )
        else:
            history_velocities.append(velocity)
            fitted += 1
    current_proximity, current_corridor = risk_scores(
        positions,
        static_velocities,
        forward,
    )
    history_proximity, history_corridor = risk_scores(
        positions,
        history_velocities,
        forward,
    )
    return {
        "sample_id": record["sample_id"],
        "fold": int(record["fold"]),
        "source_session_id": record["source_session_id"],
        "body_count": len(positions),
        "fitted_velocity_body_count": fitted,
        "label": {
            "proximity": int(
                record["target"]["future_proximity_le_1_25m"]
            ),
            "corridor": int(
                record["target"]["future_corridor_intrusion"]
            ),
        },
        "current_static": {
            "proximity": current_proximity,
            "corridor": current_corridor,
        },
        "causal_history_kinematic": {
            "proximity": history_proximity,
            "corridor": history_corridor,
        },
    }


def arm_metrics(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    return {
        target: binary_metrics(
            np.asarray([row["label"][target] for row in rows]),
            np.asarray([row[arm][target] for row in rows]),
        )
        for target in ("proximity", "corridor")
    }


def metric_value(metrics: dict[str, Any], path: str) -> float:
    target, metric = path.split(".")
    value = metrics[target][metric]
    if value is None:
        raise ValueError(f"Metric is not evaluable: {path}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = Path(str(args.output) + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite D11 result")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected 1,078 THOR samples")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["scenario_csv_path"]),
                str(record["camera_body"]),
            )
        ].append(record)

    rows = []
    for (scenario_path, camera_body), session_records in sorted(
        grouped.items()
    ):
        path = Path(scenario_path)
        expected_hashes = {
            str(record["scenario_csv_sha256"])
            for record in session_records
        }
        if expected_hashes != {sha256(path)}:
            raise ValueError(f"Scenario binding mismatch: {path}")
        data = read_scenario(
            path,
            camera_body,
            f"{camera_body} PPL_SceneFNr",
        )
        rows.extend(
            score_sample(record, data)
            for record in session_records
        )
    rows.sort(key=lambda row: row["sample_id"])

    metric_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
    )
    deltas = {path: [] for path in metric_paths}
    fold_rows = []
    for fold in range(5):
        heldout = [row for row in rows if row["fold"] == fold]
        current = arm_metrics(heldout, "current_static")
        history = arm_metrics(heldout, "causal_history_kinematic")
        fold_delta = {}
        for path in metric_paths:
            delta = (
                metric_value(history, path)
                - metric_value(current, path)
            )
            fold_delta[path] = delta
            deltas[path].append(delta)
        fold_rows.append(
            {
                "fold": fold,
                "sample_count": len(heldout),
                "source_session_count": len(
                    {row["source_session_id"] for row in heldout}
                ),
                "current_static": current,
                "causal_history_kinematic": history,
                "history_minus_current": fold_delta,
            }
        )
    aggregate = {
        path: summarize(values)
        for path, values in deltas.items()
    }
    supported = all(
        aggregate[path]["mean"] is not None
        and float(aggregate[path]["mean"]) > 0.0
        and int(aggregate[path]["positive_count"]) >= 3
        for path in metric_paths
    )
    status = (
        "D11_CAUSAL_KINEMATIC_HISTORY_INFORMATION_SUPPORTED"
        if supported
        else "D11_CAUSAL_KINEMATIC_HISTORY_INFORMATION_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development supervision information diagnostic",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "scenario_csv_count": len(grouped),
        },
        "design": {
            "current_arm": (
                "current relative XY positions held static for 2 seconds"
            ),
            "history_arm": (
                "per-body relative XY velocity fitted on anchor-minus-0.8s "
                "through anchor only, then constant-velocity projection"
            ),
            "common_coordinate_frame": (
                "the exact target-construction forward axis; common oracle "
                "frame for both arms, not a deployable input"
            ),
            "history_seconds": HISTORY_SECONDS,
            "minimum_history_span_seconds": MIN_HISTORY_SPAN_SECONDS,
            "minimum_history_rows": MIN_HISTORY_ROWS,
            "projection_horizon_seconds": FUTURE_HORIZON_SECONDS,
            "projection_step_seconds": PROJECTION_STEP_SECONDS,
            "selection": "none; deterministic source-native geometry",
            "success_gate": (
                "history-minus-current mean > 0 and at least 3/5 positive "
                "folds for proximity/corridor AUROC and AP"
            ),
        },
        "counts": {
            "samples": len(rows),
            "source_sessions": len(
                {row["source_session_id"] for row in rows}
            ),
            "folds": 5,
            "current_body_observations": sum(
                row["body_count"] for row in rows
            ),
            "fitted_velocity_body_observations": sum(
                row["fitted_velocity_body_count"] for row in rows
            ),
        },
        "folds": fold_rows,
        "aggregate_history_minus_current": aggregate,
        "next_action": (
            "test an explicit motion-aware RGB representation"
            if supported
            else (
                "revise the supervision/estimand before another video model; "
                "do not rescue the current THOR target with architecture search"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "aggregate_history_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
