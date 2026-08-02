#!/usr/bin/env python3
"""Compare current-static and history-kinematic D26 geometry oracles."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    infer_scene_column,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    FUTURE_HORIZON_SECONDS,
    FUTURE_SAMPLE_SECONDS,
    read_scenario,
)
from run_stage_c_d25_thor_magni_time_to_entry import (
    HORIZONS,
    HORIZON_NAMES,
    nested,
    summarize_delta,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    DEFAULT_SAMPLES,
    DIRECTION_DEGREES,
    DIRECTION_NAMES,
    prepare_records,
    rotate,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d27_thor_magni_"
    "kinematic_information_ceiling_v0"
)
FOLDS = tuple(range(5))
HISTORY_SECONDS = 0.4
DISTANCE_CAP_M = 10.0
DEFAULT_D26_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d26-thor-magni-counterfactual-collision-field-v0/"
    "report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d27-thor-magni-kinematic-information-ceiling-v0/"
    "report.json"
)


def oracle_score_matrices(
    records: list[dict[str, Any]],
    d8_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    d8_by_id = {str(record["sample_id"]): record for record in d8_records}
    trajectory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    static_scores = np.full(
        (len(records), len(DIRECTION_NAMES), len(HORIZONS)),
        -DISTANCE_CAP_M,
        dtype=np.float64,
    )
    history_scores = np.full_like(static_scores, -DISTANCE_CAP_M)
    current_body_observations = 0
    history_velocity_observations = 0
    for sample_index, record in enumerate(records):
        d8_record = d8_by_id[str(record["sample_id"])]
        path = Path(str(d8_record["scenario_csv_path"]))
        camera_body = str(d8_record["camera_body"])
        key = (str(path.resolve()), camera_body)
        data = trajectory_cache.get(key)
        if data is None:
            data = read_scenario(
                path,
                camera_body,
                infer_scene_column(path, camera_body),
            )
            trajectory_cache[key] = data
        matches = np.flatnonzero(
            data["frames"] == int(d8_record["qtm_frame"])
        )
        if len(matches) != 1:
            raise ValueError(
                f"D27 QTM anchor is not unique: {record['sample_id']}"
            )
        index = int(matches[0])
        before = index - 25
        after = index + 25
        velocity = (
            data["camera"][after, :2] - data["camera"][before, :2]
        ) / (data["times"][after] - data["times"][before])
        speed = float(np.linalg.norm(velocity))
        if not np.isfinite(speed) or speed < 0.25:
            raise ValueError("D27 wearer velocity is invalid")
        forward = velocity / speed
        directions = [
            rotate(forward, degrees) for degrees in DIRECTION_DEGREES
        ]
        history_index = int(
            np.searchsorted(
                data["times"],
                data["times"][index] - HISTORY_SECONDS,
                side="left",
            )
        )
        history_delta = float(
            data["times"][index] - data["times"][history_index]
        )
        bodies = []
        for positions in data["others"].values():
            current = positions[index, :2]
            if not np.isfinite(current).all():
                continue
            current_body_observations += 1
            body_velocity = np.zeros(2, dtype=np.float64)
            historical = positions[history_index, :2]
            if (
                0.35 <= history_delta <= 0.45
                and np.isfinite(historical).all()
            ):
                body_velocity = (
                    current - historical
                ) / history_delta
                history_velocity_observations += 1
            bodies.append((current, body_velocity))
        end_time = data["times"][index] + FUTURE_HORIZON_SECONDS
        future_end = int(
            np.searchsorted(data["times"], end_time, side="right")
        )
        local = np.diff(
            data["times"][max(0, index - 20): index + 21]
        )
        step = max(
            1,
            int(round(FUTURE_SAMPLE_SECONDS / np.median(local))),
        )
        static_minimum = np.full(
            (len(DIRECTION_NAMES), len(HORIZONS)),
            DISTANCE_CAP_M,
            dtype=np.float64,
        )
        history_minimum = np.full_like(
            static_minimum,
            DISTANCE_CAP_M,
        )
        origin = data["camera"][index, :2]
        for future_index in range(index, future_end, step):
            delta_time = float(
                data["times"][future_index] - data["times"][index]
            )
            eligible_horizons = [
                horizon_index
                for horizon_index, horizon in enumerate(HORIZONS)
                if delta_time <= horizon + 1e-9
            ]
            for direction_index, direction in enumerate(directions):
                candidate = origin + speed * delta_time * direction
                for current, body_velocity in bodies:
                    static_distance = float(
                        np.linalg.norm(current - candidate)
                    )
                    history_position = (
                        current + body_velocity * delta_time
                    )
                    history_distance = float(
                        np.linalg.norm(history_position - candidate)
                    )
                    for horizon_index in eligible_horizons:
                        static_minimum[
                            direction_index,
                            horizon_index,
                        ] = min(
                            static_minimum[
                                direction_index,
                                horizon_index,
                            ],
                            static_distance,
                        )
                        history_minimum[
                            direction_index,
                            horizon_index,
                        ] = min(
                            history_minimum[
                                direction_index,
                                horizon_index,
                            ],
                            history_distance,
                        )
        static_scores[sample_index] = -static_minimum
        history_scores[sample_index] = -history_minimum
    if not np.isfinite(static_scores).all() or not np.isfinite(
        history_scores
    ).all():
        raise ValueError("D27 oracle score matrix is non-finite")
    return static_scores, history_scores, {
        "current_body_observations": current_body_observations,
        "history_velocity_observations": history_velocity_observations,
    }


def evaluate_scores(
    records: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    expected_shape = (
        len(records),
        len(DIRECTION_NAMES),
        len(HORIZONS),
    )
    if scores.shape != expected_shape:
        raise ValueError("D27 oracle score shape mismatch")
    monotonicity_violations = int(
        np.sum(np.diff(scores, axis=2) < -1e-12)
    )
    by_direction = {}
    for direction_index, direction_name in enumerate(DIRECTION_NAMES):
        by_horizon = {}
        for horizon_index, horizon_name in enumerate(HORIZON_NAMES):
            target = np.asarray(
                [
                    int(record["_d26_entry_bins"][direction_index])
                    <= horizon_index
                    for record in records
                ],
                dtype=np.int64,
            )
            score = scores[:, direction_index, horizon_index]
            by_source = []
            for source in sorted(
                {
                    str(record["source_session_id"])
                    for record in records
                }
            ):
                indices = [
                    index
                    for index, record in enumerate(records)
                    if str(record["source_session_id"]) == source
                ]
                metric = binary_metrics(target[indices], score[indices])
                if metric["auroc"] is None:
                    continue
                by_source.append(
                    {
                        "source_session_id": source,
                        "eligible_count": len(indices),
                        "positive_count": int(np.sum(target[indices])),
                        "auroc": float(metric["auroc"]),
                        "average_precision": float(
                            metric["average_precision"]
                        ),
                    }
                )
            pooled = binary_metrics(target, score)
            by_horizon[horizon_name] = {
                "seconds": HORIZONS[horizon_index],
                "by_source": by_source,
                "source_macro": {
                    "auroc": float(
                        np.mean([row["auroc"] for row in by_source])
                    ),
                    "average_precision": float(
                        np.mean(
                            [
                                row["average_precision"]
                                for row in by_source
                            ]
                        )
                    ),
                    "evaluable_sources": len(by_source),
                },
                "pooled": {
                    "auroc": float(pooled["auroc"]),
                    "average_precision": float(
                        pooled["average_precision"]
                    ),
                    "eligible_count": len(target),
                    "positive_count": int(np.sum(target)),
                },
            }
        by_direction[direction_name] = {
            "source_macro_horizon_macro": {
                metric: float(
                    np.mean(
                        [
                            by_horizon[name]["source_macro"][metric]
                            for name in HORIZON_NAMES
                        ]
                    )
                )
                for metric in ("auroc", "average_precision")
            },
            "pooled_horizon_macro": {
                metric: float(
                    np.mean(
                        [
                            by_horizon[name]["pooled"][metric]
                            for name in HORIZON_NAMES
                        ]
                    )
                )
                for metric in ("auroc", "average_precision")
            },
            "by_horizon": by_horizon,
        }
    source_macro = {
        metric: float(
            np.mean(
                [
                    by_direction[name][
                        "source_macro_horizon_macro"
                    ][metric]
                    for name in DIRECTION_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision")
    }
    pooled = {
        metric: float(
            np.mean(
                [
                    by_direction[name]["pooled_horizon_macro"][metric]
                    for name in DIRECTION_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision")
    }
    truth_time = np.asarray(
        [
            [
                2.5 if value is None else float(value)
                for value in record["_d26_entry_offsets_seconds"]
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    nonredundant = np.asarray(
        [
            len(set(round(float(value), 6) for value in row)) > 1
            for row in truth_time
        ],
        dtype=bool,
    )
    predicted_direction = np.argmin(scores[:, :, -1], axis=1)
    safe_time = np.max(truth_time, axis=1)
    correct = (
        np.abs(
            truth_time[np.arange(len(records)), predicted_direction]
            - safe_time
        )
        <= 1e-6
    )
    by_source = []
    for source in sorted(
        {str(record["source_session_id"]) for record in records}
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["source_session_id"]) == source
            and bool(nonredundant[index])
        ]
        if indices:
            by_source.append(
                {
                    "source_session_id": source,
                    "eligible_count": len(indices),
                    "accuracy": float(np.mean(correct[indices])),
                }
            )
    return {
        "source_macro_direction_horizon_macro": source_macro,
        "pooled_direction_horizon_macro": pooled,
        "safe_choice": {
            "source_macro_accuracy": float(
                np.mean([row["accuracy"] for row in by_source])
            ),
            "pooled_accuracy": float(np.mean(correct[nonredundant])),
            "eligible_count": int(np.sum(nonredundant)),
            "evaluable_sources": len(by_source),
            "by_source": by_source,
        },
        "by_direction": by_direction,
        "monotonicity_violations": monotonicity_violations,
    }


def metric_paths() -> list[str]:
    result = [
        f"{scope}.{metric}"
        for scope in (
            "source_macro_direction_horizon_macro",
            "pooled_direction_horizon_macro",
        )
        for metric in ("auroc", "average_precision")
    ]
    result.extend(
        ("safe_choice.source_macro_accuracy", "safe_choice.pooled_accuracy")
    )
    for direction_name in DIRECTION_NAMES:
        result.extend(
            f"by_direction.{direction_name}.{scope}.{metric}"
            for scope in (
                "source_macro_horizon_macro",
                "pooled_horizon_macro",
            )
            for metric in ("auroc", "average_precision")
        )
        result.extend(
            (
                f"by_direction.{direction_name}.by_horizon."
                f"{horizon_name}.{scope}.{metric}"
            )
            for horizon_name in HORIZON_NAMES
            for scope in ("source_macro", "pooled")
            for metric in ("auroc", "average_precision")
        )
    return result


def build_gate(
    aggregate: dict[str, dict[str, Any]],
    monotonicity_violations: int,
) -> dict[str, Any]:
    source_auroc = aggregate[
        "source_macro_direction_horizon_macro.auroc"
    ]
    source_ap = aggregate[
        "source_macro_direction_horizon_macro.average_precision"
    ]
    pooled_auroc = aggregate[
        "pooled_direction_horizon_macro.auroc"
    ]
    pooled_ap = aggregate[
        "pooled_direction_horizon_macro.average_precision"
    ]
    safe_choice = aggregate["safe_choice.source_macro_accuracy"]
    auroc_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.auroc"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    ap_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.average_precision"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    checks = {
        "source_macro_auroc_effect": source_auroc["mean"] >= 0.020,
        "source_macro_ap_effect": source_ap["mean"] >= 0.010,
        "source_macro_auroc_positive_folds": (
            source_auroc["positive_folds"] >= 3
        ),
        "source_macro_ap_positive_folds": (
            source_ap["positive_folds"] >= 3
        ),
        "auroc_positive_directions": auroc_positive_directions >= 2,
        "ap_positive_directions": ap_positive_directions >= 2,
        "safe_choice_effect": safe_choice["mean"] >= 0.050,
        "safe_choice_positive_folds": (
            safe_choice["positive_folds"] >= 3
        ),
        "pooled_auroc_noninferiority": pooled_auroc["mean"] >= -0.005,
        "pooled_ap_noninferiority": pooled_ap["mean"] >= -0.005,
        "monotonicity_exact": monotonicity_violations == 0,
    }
    return {
        "frozen_thresholds": {
            "source_macro_auroc_mean_floor": 0.020,
            "source_macro_ap_mean_floor": 0.010,
            "positive_folds": 3,
            "positive_directions": 2,
            "safe_choice_mean_delta_floor": 0.050,
            "safe_choice_positive_folds": 3,
            "pooled_noninferiority_floor": -0.005,
            "monotonicity_violations": 0,
        },
        "direction_breadth": {
            "auroc_positive_directions": auroc_positive_directions,
            "ap_positive_directions": ap_positive_directions,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--d8-samples",
        type=Path,
        default=DEFAULT_D8_SAMPLES,
    )
    parser.add_argument(
        "--d26-report",
        type=Path,
        default=DEFAULT_D26_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    score_path = args.output.with_name("oracle_scores.npz")
    score_sidecar = score_path.with_suffix(score_path.suffix + ".sha256")
    if any(
        path.exists()
        for path in (args.output, sidecar, score_path, score_sidecar)
    ):
        raise FileExistsError("D27 outputs are non-overwriting")
    d26 = json.loads(args.d26_report.read_text(encoding="utf-8"))
    if d26["status"] != (
        "D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_"
        "INCREMENT_NOT_SUPPORTED"
    ):
        raise ValueError("D27 requires the completed D26 terminal")
    if d26["inputs"]["samples_sha256"] != sha256(args.samples):
        raise ValueError("D27 D26 samples binding mismatch")
    if d26["inputs"]["d8_samples_sha256"] != sha256(args.d8_samples):
        raise ValueError("D27 D26 D8 binding mismatch")

    d12_records = load_jsonl(args.samples)
    d8_records = load_jsonl(args.d8_samples)
    records = prepare_records(d12_records, d8_records)
    static_scores, history_scores, coverage = oracle_score_matrices(
        records,
        d8_records,
    )
    units = []
    paths = metric_paths()
    for fold in FOLDS:
        indices = [
            index
            for index, record in enumerate(records)
            if int(record["fold"]) == fold
        ]
        fold_records = [records[index] for index in indices]
        current = evaluate_scores(
            fold_records,
            static_scores[indices],
        )
        history = evaluate_scores(
            fold_records,
            history_scores[indices],
        )
        delta = {}
        for path in paths:
            cursor = delta
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = (
                nested(history, path) - nested(current, path)
            )
        units.append(
            {
                "fold": fold,
                "heldout_source_sessions": sorted(
                    {
                        str(record["source_session_id"])
                        for record in fold_records
                    }
                ),
                "current_static": current,
                "history_kinematic": history,
                "history_kinematic_minus_current_static": delta,
            }
        )
    adapted_units = [
        {
            "fold": unit["fold"],
            "history_minus_current": unit[
                "history_kinematic_minus_current_static"
            ],
        }
        for unit in units
    ]
    aggregate = {
        path: summarize_delta(adapted_units, path) for path in paths
    }
    monotonicity_violations = sum(
        int(unit[arm]["monotonicity_violations"])
        for unit in units
        for arm in ("current_static", "history_kinematic")
    )
    gate = build_gate(aggregate, monotonicity_violations)
    status = (
        "D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED"
        if gate["supported"]
        else (
            "D27_THOR_MAGNI_HISTORY_KINEMATIC_"
            "INFORMATION_CEILING_NOT_SUPPORTED"
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_path,
        sample_ids=np.asarray(
            [str(record["sample_id"]) for record in records],
            dtype="U31",
        ),
        current_static=static_scores.astype(np.float32),
        history_kinematic=history_scores.astype(np.float32),
    )
    score_digest = sha256(score_path)
    score_sidecar.write_text(
        f"{score_digest}  {score_path.name}\n",
        encoding="utf-8",
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development source-native information ceiling",
            "prediction_uses_future": False,
            "truth_uses_future": True,
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "d26_report_path": str(args.d26_report.resolve()),
            "d26_report_sha256": sha256(args.d26_report),
            "oracle_scores_path": str(score_path.resolve()),
            "oracle_scores_sha256": score_digest,
        },
        "design": {
            "current_static": (
                "other bodies remain at anchor world position"
            ),
            "history_kinematic": (
                "other-body world velocity from anchor minus 0.4 seconds "
                "to anchor, then constant-velocity extrapolation"
            ),
            "future_prediction_access": False,
            "score": (
                "negative predicted minimum synchronized distance for each "
                "candidate direction and cumulative horizon"
            ),
            "distance_cap_m": DISTANCE_CAP_M,
            "history_seconds": HISTORY_SECONDS,
            "directions_degrees": list(DIRECTION_DEGREES),
            "horizons_seconds": list(HORIZONS),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "folds": len(FOLDS),
            **coverage,
            "history_velocity_coverage": (
                coverage["history_velocity_observations"]
                / coverage["current_body_observations"]
            ),
            "monotonicity_violations": monotonicity_violations,
        },
        "gate": gate,
        "aggregate_fold_mean_history_kinematic_minus_current_static": (
            aggregate
        ),
        "units": units,
        "next_action": (
            "design an explicit object-motion RGB student"
            if gate["supported"]
            else (
                "stop the current tracked-body counterfactual route; "
                "constant-velocity history has no supported information ceiling"
            )
        ),
    }
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
                "gate": gate,
                "primary_aggregate": {
                    key: aggregate[key]
                    for key in (
                        (
                            "source_macro_direction_horizon_macro."
                            "auroc"
                        ),
                        (
                            "source_macro_direction_horizon_macro."
                            "average_precision"
                        ),
                        "safe_choice.source_macro_accuracy",
                        "pooled_direction_horizon_macro.auroc",
                        (
                            "pooled_direction_horizon_macro."
                            "average_precision"
                        ),
                    )
                },
                "coverage": coverage,
                "report_sha256": digest,
                "scores_sha256": score_digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
