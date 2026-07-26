from __future__ import annotations

from collections import Counter, defaultdict
import math
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .local_expansion import (
    LocalExpansionResult,
    fit_fixed_grid_local_affine,
)
from .protocol import TrialSpec
from .rotation_compensation import compensate_current_to_previous
from .sparse_flow import detect_fixed_grid_features, track_features
from .synthetic_generator import SyntheticSequence, generate_sequence


def _median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.median(array))


def _common_cell_expansions(
    raw: Sequence[LocalExpansionResult],
    compensated: Sequence[LocalExpansionResult],
) -> tuple[list[float], list[float], list[int]]:
    raw_values: list[float] = []
    compensated_values: list[float] = []
    indices: list[int] = []
    for index, (raw_cell, compensated_cell) in enumerate(
        zip(raw, compensated, strict=True)
    ):
        if (
            raw_cell.evaluable
            and compensated_cell.evaluable
            and raw_cell.expansion is not None
            and compensated_cell.expansion is not None
        ):
            raw_values.append(raw_cell.expansion)
            compensated_values.append(compensated_cell.expansion)
            indices.append(index)
    return raw_values, compensated_values, indices


def _sign_correct(estimate: float, truth: float, zero_band: float) -> bool:
    if truth > 0.0:
        return estimate > zero_band
    if truth < 0.0:
        return estimate < -zero_band
    return abs(estimate) <= zero_band


def run_trial(
    spec: TrialSpec,
    protocol: dict[str, Any],
    include_cell_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cv2_seed = int(spec.seed % (2**31 - 1))
    import cv2

    cv2.setRNGSeed(cv2_seed)
    cv2.setNumThreads(1)
    sequence_start = time.perf_counter_ns()
    sequence = generate_sequence(spec, protocol)
    sequence_elapsed = time.perf_counter_ns() - sequence_start
    pair_total = len(sequence.frames) - 1
    lk_parameters = protocol["sparse_lk"]
    affine_parameters = protocol["local_affine"]
    minimum_common = int(
        affine_parameters["minimum_common_evaluable_cells_per_pair"]
    )
    minimum_pair_fraction = float(
        affine_parameters["minimum_evaluable_pair_fraction_per_trial"]
    )
    warp_overlap_floor = 0.75
    pair_trace: list[dict[str, Any]] = []
    abstentions: Counter[str] = Counter()
    timings_ns: defaultdict[str, int] = defaultdict(int)
    timings_ns["synthetic_generation"] = sequence_elapsed

    for pair_index in range(pair_total):
        previous = sequence.frames[pair_index]
        current = sequence.frames[pair_index + 1]
        previous_valid = sequence.valid_masks[pair_index]
        current_valid = sequence.valid_masks[pair_index + 1]
        dt = (
            sequence.timestamps_seconds[pair_index + 1]
            - sequence.timestamps_seconds[pair_index]
        )
        if not np.isfinite(dt) or dt <= 0.0:
            abstentions["NON_POSITIVE_OR_MISSING_DT"] += 1
            pair_trace.append(
                {
                    "pair_index": pair_index,
                    "timestamp_seconds": sequence.timestamps_seconds[
                        pair_index + 1
                    ],
                    "evaluable": False,
                    "reason": "NON_POSITIVE_OR_MISSING_DT",
                }
            )
            continue

        started = time.perf_counter_ns()
        compensation = compensate_current_to_previous(
            current,
            current_valid,
            previous_valid,
            sequence.rotation_homography_previous_to_current,
        )
        timings_ns["rotation_warp"] += time.perf_counter_ns() - started
        if compensation.overlap_fraction < warp_overlap_floor:
            abstentions["ROTATION_WARP_VALID_COVERAGE_BELOW_0_75"] += 1
            pair_trace.append(
                {
                    "pair_index": pair_index,
                    "timestamp_seconds": sequence.timestamps_seconds[
                        pair_index + 1
                    ],
                    "evaluable": False,
                    "reason": "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75",
                    "warp_overlap_fraction": compensation.overlap_fraction,
                }
            )
            continue

        feature_mask = np.ascontiguousarray(previous_valid)
        started = time.perf_counter_ns()
        initial_points = detect_fixed_grid_features(
            previous, feature_mask, lk_parameters
        )
        raw_tracks = track_features(
            previous,
            current,
            initial_points,
            current_valid,
            lk_parameters,
        )
        compensated_tracks = track_features(
            previous,
            compensation.image,
            initial_points,
            compensation.valid_mask,
            lk_parameters,
        )
        timings_ns["sparse_lk"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        raw_cells = fit_fixed_grid_local_affine(
            raw_tracks, dt, previous.shape, affine_parameters
        )
        compensated_cells = fit_fixed_grid_local_affine(
            compensated_tracks, dt, previous.shape, affine_parameters
        )
        timings_ns["local_affine"] += time.perf_counter_ns() - started
        raw_values, compensated_values, common_indices = (
            _common_cell_expansions(raw_cells, compensated_cells)
        )
        if len(common_indices) < minimum_common:
            abstentions["COMMON_GRID_SUPPORT_BELOW_5_OF_9"] += 1
            for cell in raw_cells:
                if not cell.evaluable and cell.abstention_reason:
                    abstentions[f"raw:{cell.abstention_reason}"] += 1
            for cell in compensated_cells:
                if not cell.evaluable and cell.abstention_reason:
                    abstentions[f"comp:{cell.abstention_reason}"] += 1
            pair_trace.append(
                {
                    "pair_index": pair_index,
                    "timestamp_seconds": sequence.timestamps_seconds[
                        pair_index + 1
                    ],
                    "evaluable": False,
                    "reason": "COMMON_GRID_SUPPORT_BELOW_5_OF_9",
                    "common_cell_count": len(common_indices),
                    "raw_track_count": raw_tracks.valid_count,
                    "compensated_track_count": compensated_tracks.valid_count,
                    "warp_overlap_fraction": compensation.overlap_fraction,
                }
            )
            continue

        raw_median = float(np.median(raw_values))
        compensated_median = float(np.median(compensated_values))
        pair_row: dict[str, Any] = {
            "pair_index": pair_index,
            "timestamp_seconds": sequence.timestamps_seconds[pair_index + 1],
            "evaluable": True,
            "common_cell_count": len(common_indices),
            "raw_expansion_median_per_s": raw_median,
            "compensated_expansion_median_per_s": compensated_median,
            "raw_abs_expansion_median_per_s": float(
                np.median(np.abs(raw_values))
            ),
            "compensated_abs_expansion_median_per_s": float(
                np.median(np.abs(compensated_values))
            ),
            "raw_track_count": raw_tracks.valid_count,
            "compensated_track_count": compensated_tracks.valid_count,
            "warp_overlap_fraction": compensation.overlap_fraction,
            "common_cell_indices": common_indices,
        }
        if include_cell_details:
            pair_row["raw_cells"] = [cell.to_dict() for cell in raw_cells]
            pair_row["compensated_cells"] = [
                cell.to_dict() for cell in compensated_cells
            ]
        pair_trace.append(pair_row)

    evaluable_pairs = [row for row in pair_trace if row["evaluable"]]
    pair_fraction = len(evaluable_pairs) / max(pair_total, 1)
    trial_evaluable = pair_fraction >= minimum_pair_fraction
    if not trial_evaluable:
        abstentions["EVALUABLE_PAIR_FRACTION_BELOW_0_80"] += 1

    result: dict[str, Any] = {
        **spec.to_dict(),
        "planned_pair_count": pair_total,
        "evaluable_pair_count": len(evaluable_pairs),
        "evaluable_pair_fraction": pair_fraction,
        "evaluable": trial_evaluable,
        "abstention_reason": (
            None
            if trial_evaluable
            else "EVALUABLE_PAIR_FRACTION_BELOW_0_80"
        ),
        "truth_scale_rate_per_s": spec.scale_rate_per_s,
        "base_sha256": sequence.base_sha256,
        "sequence_sha256": sequence.sequence_sha256,
        "abstention_counts": dict(sorted(abstentions.items())),
        "pair_trace": pair_trace,
    }
    if trial_evaluable:
        raw_leakage = _median(
            row["raw_abs_expansion_median_per_s"] for row in evaluable_pairs
        )
        compensated_leakage = _median(
            row["compensated_abs_expansion_median_per_s"]
            for row in evaluable_pairs
        )
        raw_estimate = _median(
            row["raw_expansion_median_per_s"] for row in evaluable_pairs
        )
        compensated_estimate = _median(
            row["compensated_expansion_median_per_s"]
            for row in evaluable_pairs
        )
        assert raw_leakage is not None
        assert compensated_leakage is not None
        assert raw_estimate is not None
        assert compensated_estimate is not None
        result.update(
            {
                "raw_rotation_leakage_per_s": raw_leakage,
                "compensated_rotation_leakage_per_s": compensated_leakage,
                "paired_leakage_reduction_per_s": (
                    raw_leakage - compensated_leakage
                ),
                "raw_closing_estimate_per_s": raw_estimate,
                "compensated_closing_estimate_per_s": compensated_estimate,
                "raw_closing_error_per_s": abs(
                    raw_estimate - spec.scale_rate_per_s
                ),
                "compensated_closing_error_per_s": abs(
                    compensated_estimate - spec.scale_rate_per_s
                ),
                "compensated_minus_raw_closing_error_per_s": (
                    abs(compensated_estimate - spec.scale_rate_per_s)
                    - abs(raw_estimate - spec.scale_rate_per_s)
                ),
            }
        )
        zero_band = float(protocol["metrics"]["sign_accuracy_zero_band_per_s"])
        if spec.scale_rate_per_s != 0.0:
            result["raw_sign_correct"] = _sign_correct(
                raw_estimate, spec.scale_rate_per_s, zero_band
            )
            result["compensated_sign_correct"] = _sign_correct(
                compensated_estimate, spec.scale_rate_per_s, zero_band
            )
        else:
            result["raw_sign_correct"] = None
            result["compensated_sign_correct"] = None
        rsr_floor = float(protocol["metrics"]["rsr_denominator_floor_per_s"])
        if spec.motion_family == "pure_rotation" and raw_leakage >= rsr_floor:
            result["rsr"] = 1.0 - compensated_leakage / raw_leakage
            result["rsr_status"] = "EVALUABLE"
        elif spec.motion_family == "pure_rotation":
            result["rsr"] = None
            result["rsr_status"] = "NOT_EVALUABLE_DENOMINATOR_FLOOR"
        else:
            result["rsr"] = None
            result["rsr_status"] = "NOT_APPLICABLE"
        crr_floor = float(protocol["metrics"]["crr_denominator_floor_per_s"])
        if spec.scale_rate_per_s > 0.0 and raw_estimate >= crr_floor:
            result["crr"] = compensated_estimate / raw_estimate
            result["crr_status"] = "EVALUABLE"
        elif spec.scale_rate_per_s > 0.0:
            result["crr"] = None
            result["crr_status"] = "NOT_EVALUABLE_DENOMINATOR_FLOOR"
        else:
            result["crr"] = None
            result["crr_status"] = "NOT_APPLICABLE"
    else:
        for key in (
            "raw_rotation_leakage_per_s",
            "compensated_rotation_leakage_per_s",
            "paired_leakage_reduction_per_s",
            "raw_closing_estimate_per_s",
            "compensated_closing_estimate_per_s",
            "raw_closing_error_per_s",
            "compensated_closing_error_per_s",
            "compensated_minus_raw_closing_error_per_s",
            "raw_sign_correct",
            "compensated_sign_correct",
            "rsr",
            "crr",
        ):
            result[key] = None
        result["rsr_status"] = "NOT_EVALUABLE_TRIAL"
        result["crr_status"] = "NOT_EVALUABLE_TRIAL"

    runtime = {
        "trial_id": spec.trial_id,
        "pair_count": pair_total,
        "module_total_milliseconds": {
            key: value / 1_000_000.0 for key, value in sorted(timings_ns.items())
        },
        "total_milliseconds": sum(timings_ns.values()) / 1_000_000.0,
    }
    return result, runtime


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _cluster_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    statistic: Callable[[np.ndarray], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        if np.isfinite(number):
            by_seed[int(row["seed"])].append(number)
    seeds = sorted(by_seed)
    if not seeds:
        return math.nan, math.nan
    arrays = [np.asarray(by_seed[item], dtype=np.float64) for item in seeds]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        sample = np.concatenate([arrays[index] for index in selected])
        estimates[replicate] = statistic(sample)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _metric_summary(
    rows: Sequence[dict[str, Any]],
    key: str,
    protocol: dict[str, Any],
    statistic: Callable[[np.ndarray], float] = lambda values: float(
        np.median(values)
    ),
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "n": 0,
            "estimate": None,
            "ci95": [None, None],
        }
    low, high = _cluster_bootstrap_ci(
        rows,
        lambda row: float(row[key]),
        statistic,
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    return {
        "n": int(values.size),
        "estimate": float(statistic(values)),
        "ci95": [low, high],
    }


def _coverage_summary(
    rows: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    successes = sum(bool(row["evaluable"]) for row in rows)
    total = len(rows)
    wilson = wilson_interval(successes, total)
    cluster_low, cluster_high = _cluster_bootstrap_ci(
        rows,
        lambda row: float(bool(row["evaluable"])),
        lambda values: float(np.mean(values)),
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    return {
        "planned": total,
        "evaluable": successes,
        "point": successes / total if total else None,
        "wilson95": list(wilson),
        "cluster_bootstrap95": [cluster_low, cluster_high],
    }


def _condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["motion_family"],
        row["axis"],
        row["angular_velocity_deg_per_s"],
        row["scale_rate_per_s"],
        row["fps"],
        row["degradation"],
    )


def summarize_and_decide(
    rows: Sequence[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    expected = int(protocol["trials"]["expected_total_trials"])
    trial_ids = [row["trial_id"] for row in rows]
    if len(rows) != expected or len(set(trial_ids)) != expected:
        raise ValueError("FORMAL_TRIAL_LEDGER_INVENTORY_MISMATCH")
    gates = protocol["kill_gate_a"]
    components: dict[str, dict[str, Any]] = {}

    clean = [row for row in rows if row["split"] == "clean"]
    stress = [row for row in rows if row["split"] == "stress"]
    yaw_pitch = [
        row
        for row in clean
        if row["motion_family"] == "pure_rotation"
        and row["axis"] in {"yaw", "pitch"}
        and row["evaluable"]
    ]
    informative_floor = gates["clean_rotation_yaw_pitch"][
        "informative_raw_leakage_floor_per_s"
    ]
    informative = [
        row
        for row in yaw_pitch
        if row["raw_rotation_leakage_per_s"] >= informative_floor
    ]
    compensated = _metric_summary(
        informative, "compensated_rotation_leakage_per_s", protocol
    )
    reduction = _metric_summary(
        informative, "paired_leakage_reduction_per_s", protocol
    )
    improved = _metric_summary(
        informative,
        "paired_leakage_reduction_per_s",
        protocol,
        statistic=lambda values: float(np.mean(values > 0.0)),
    )
    per_cell_rotation: list[dict[str, Any]] = []
    for key in sorted({_condition_key(row) for row in yaw_pitch}):
        cell_rows = [row for row in yaw_pitch if _condition_key(row) == key]
        cell_comp = float(
            np.median(
                [
                    row["compensated_rotation_leakage_per_s"]
                    for row in cell_rows
                ]
            )
        )
        cell_improved = float(
            np.mean(
                [
                    row["paired_leakage_reduction_per_s"] > 0.0
                    for row in cell_rows
                ]
            )
        )
        per_cell_rotation.append(
            {
                "condition": list(key),
                "n": len(cell_rows),
                "compensated_leakage_median_per_s": cell_comp,
                "improved_fraction": cell_improved,
            }
        )
    rotation_thresholds = gates["clean_rotation_yaw_pitch"]
    rotation_pass = (
        compensated["n"] > 0
        and compensated["ci95"][1]
        <= rotation_thresholds[
            "pooled_compensated_leakage_median_ci_upper_max_per_s"
        ]
        and reduction["ci95"][0]
        >= rotation_thresholds[
            "paired_leakage_reduction_median_ci_lower_min_per_s"
        ]
        and improved["ci95"][0]
        >= rotation_thresholds["improved_trial_fraction_ci_lower_min"]
        and all(
            cell["compensated_leakage_median_per_s"]
            <= rotation_thresholds[
                "per_cell_compensated_leakage_median_max_per_s"
            ]
            and cell["improved_fraction"]
            >= rotation_thresholds["per_cell_improved_trial_fraction_min"]
            for cell in per_cell_rotation
        )
    )
    components["clean_rotation_yaw_pitch"] = {
        "pass": rotation_pass,
        "planned_evaluable": len(yaw_pitch),
        "informative": len(informative),
        "compensated_leakage": compensated,
        "paired_reduction": reduction,
        "improved_fraction": improved,
        "per_cell": per_cell_rotation,
    }

    roll = [
        row
        for row in clean
        if row["motion_family"] == "pure_rotation"
        and row["axis"] == "roll"
        and row["evaluable"]
    ]
    roll_comp = _metric_summary(
        roll, "compensated_rotation_leakage_per_s", protocol
    )
    roll_delta_rows = [
        {
            **row,
            "roll_delta": (
                row["compensated_rotation_leakage_per_s"]
                - row["raw_rotation_leakage_per_s"]
            ),
        }
        for row in roll
    ]
    roll_delta = _metric_summary(roll_delta_rows, "roll_delta", protocol)
    roll_thresholds = gates["clean_rotation_roll"]
    roll_pass = (
        roll_comp["n"] > 0
        and roll_comp["ci95"][1]
        <= roll_thresholds[
            "pooled_compensated_leakage_median_ci_upper_max_per_s"
        ]
        and roll_delta["ci95"][1]
        <= roll_thresholds[
            "paired_compensated_minus_raw_median_ci_upper_max_per_s"
        ]
    )
    components["clean_rotation_roll"] = {
        "pass": roll_pass,
        "compensated_leakage": roll_comp,
        "compensated_minus_raw": roll_delta,
    }

    scale_rows = [
        row
        for row in clean
        if row["motion_family"] == "scale" and row["evaluable"]
    ]
    mixed_rows = [
        row
        for row in clean
        if row["motion_family"] == "rotation_plus_scale_up"
        and row["evaluable"]
    ]
    scale_error = _metric_summary(
        scale_rows, "compensated_closing_error_per_s", protocol
    )
    mixed_error = _metric_summary(
        mixed_rows, "compensated_closing_error_per_s", protocol
    )
    closing_rows = scale_rows + mixed_rows
    sign_summary = _metric_summary(
        closing_rows,
        "compensated_sign_correct",
        protocol,
        statistic=lambda values: float(np.mean(values)),
    )
    sign_successes = sum(bool(row["compensated_sign_correct"]) for row in closing_rows)
    sign_wilson = wilson_interval(sign_successes, len(closing_rows))
    noninferiority = _metric_summary(
        closing_rows,
        "compensated_minus_raw_closing_error_per_s",
        protocol,
    )
    scale_lookup = {
        (row["seed"], row["fps"], row["scale_rate_per_s"]): row
        for row in scale_rows
        if abs(row["scale_rate_per_s"] - 0.15) < 1e-12
    }
    matched_rows: list[dict[str, Any]] = []
    for row in mixed_rows:
        matched = scale_lookup.get((row["seed"], row["fps"], 0.15))
        if matched is not None:
            matched_rows.append(
                {
                    **row,
                    "mixed_scale_difference": abs(
                        row["compensated_closing_estimate_per_s"]
                        - matched["compensated_closing_estimate_per_s"]
                    ),
                }
            )
    matched_difference = _metric_summary(
        matched_rows, "mixed_scale_difference", protocol
    )
    closing_thresholds = gates["clean_closing"]
    closing_pass = (
        scale_error["n"] > 0
        and mixed_error["n"] > 0
        and scale_error["ci95"][1]
        <= closing_thresholds[
            "scale_closing_error_median_ci_upper_max_per_s"
        ]
        and mixed_error["ci95"][1]
        <= closing_thresholds[
            "mixed_closing_error_median_ci_upper_max_per_s"
        ]
        and sign_summary["estimate"]
        >= closing_thresholds["sign_accuracy_point_min"]
        and sign_wilson[0]
        >= closing_thresholds["sign_accuracy_ci_lower_min"]
        and noninferiority["ci95"][1]
        <= closing_thresholds[
            "compensated_minus_raw_error_median_ci_upper_max_per_s"
        ]
        and matched_difference["ci95"][1]
        <= closing_thresholds[
            "mixed_minus_matched_scale_compensated_abs_difference_ci_upper_max_per_s"
        ]
    )
    components["clean_closing"] = {
        "pass": closing_pass,
        "scale_error": scale_error,
        "mixed_error": mixed_error,
        "sign_accuracy": {
            **sign_summary,
            "wilson95": list(sign_wilson),
        },
        "compensated_minus_raw_error": noninferiority,
        "mixed_minus_matched_scale": matched_difference,
    }

    fps_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        if row["evaluable"]:
            key = (
                row["seed"],
                row["motion_family"],
                row["axis"],
                row["angular_velocity_deg_per_s"],
                row["scale_rate_per_s"],
                row["degradation"],
            )
            fps_groups[key].append(row)
    fps_rows: list[dict[str, Any]] = []
    for group in fps_groups.values():
        if {row["fps"] for row in group} == {15, 30, 60}:
            estimates = [
                row["compensated_closing_estimate_per_s"] for row in group
            ]
            fps_rows.append(
                {
                    **group[0],
                    "fps_range_per_s": float(max(estimates) - min(estimates)),
                }
            )
    fps_summary = _metric_summary(fps_rows, "fps_range_per_s", protocol)
    fps_pass = (
        fps_summary["n"] > 0
        and fps_summary["ci95"][1]
        <= gates["fps_consistency"][
            "matched_seed_motion_expansion_range_ci_upper_max_per_s"
        ]
    )
    components["fps_consistency"] = {
        "pass": fps_pass,
        "matched_groups": len(fps_rows),
        "range": fps_summary,
    }

    clean_coverage = _coverage_summary(clean, protocol)
    stress_coverage = _coverage_summary(stress, protocol)
    clean_cell_coverages: list[float] = []
    stress_cell_coverages: list[float] = []
    for key in {_condition_key(row) for row in clean}:
        cell = [row for row in clean if _condition_key(row) == key]
        clean_cell_coverages.append(
            sum(row["evaluable"] for row in cell) / len(cell)
        )
    for key in {_condition_key(row) for row in stress}:
        cell = [row for row in stress if _condition_key(row) == key]
        stress_cell_coverages.append(
            sum(row["evaluable"] for row in cell) / len(cell)
        )
    coverage_thresholds = gates["coverage"]
    coverage_pass = (
        clean_coverage["point"] >= coverage_thresholds["clean_paired_point_min"]
        and clean_coverage["cluster_bootstrap95"][0]
        >= coverage_thresholds["clean_paired_ci_lower_min"]
        and min(clean_cell_coverages, default=0.0)
        >= coverage_thresholds["clean_per_cell_point_min"]
        and stress_coverage["point"]
        >= coverage_thresholds["stress_paired_point_min"]
        and stress_coverage["cluster_bootstrap95"][0]
        >= coverage_thresholds["stress_paired_ci_lower_min"]
        and min(stress_cell_coverages, default=0.0)
        >= coverage_thresholds["stress_per_cell_point_min"]
    )
    components["coverage"] = {
        "pass": coverage_pass,
        "clean": clean_coverage,
        "stress": stress_coverage,
        "clean_worst_cell_point": min(clean_cell_coverages, default=None),
        "stress_worst_cell_point": min(stress_cell_coverages, default=None),
    }

    profile_results: dict[str, Any] = {}
    stress_thresholds = gates["stress"]
    for profile in protocol["trials"]["stress_sentinel"]["profiles"]:
        profile_rows = [
            row for row in stress if row["degradation"] == profile
        ]
        rotation_rows = [
            row
            for row in profile_rows
            if row["motion_family"] == "pure_rotation" and row["evaluable"]
        ]
        closing_profile_rows = [
            row
            for row in profile_rows
            if row["motion_family"] != "pure_rotation" and row["evaluable"]
        ]
        leakage = _metric_summary(
            rotation_rows, "compensated_rotation_leakage_per_s", protocol
        )
        error = _metric_summary(
            closing_profile_rows,
            "compensated_closing_error_per_s",
            protocol,
        )
        sign = _metric_summary(
            closing_profile_rows,
            "compensated_sign_correct",
            protocol,
            statistic=lambda values: float(np.mean(values)),
        )
        successes = sum(
            bool(row["compensated_sign_correct"])
            for row in closing_profile_rows
        )
        sign_interval = wilson_interval(successes, len(closing_profile_rows))
        profile_pass = (
            leakage["n"] > 0
            and error["n"] > 0
            and leakage["ci95"][1]
            <= stress_thresholds[
                "per_profile_rotation_compensated_leakage_median_ci_upper_max_per_s"
            ]
            and error["ci95"][1]
            <= stress_thresholds[
                "per_profile_closing_error_median_ci_upper_max_per_s"
            ]
            and sign["estimate"]
            >= stress_thresholds["per_profile_sign_accuracy_point_min"]
            and sign_interval[0]
            >= stress_thresholds["per_profile_sign_accuracy_ci_lower_min"]
        )
        profile_results[profile] = {
            "pass": profile_pass,
            "rotation_leakage": leakage,
            "closing_error": error,
            "sign_accuracy": {
                **sign,
                "wilson95": list(sign_interval),
            },
        }
    stress_pass = all(item["pass"] for item in profile_results.values())
    components["stress"] = {
        "pass": stress_pass,
        "profiles": profile_results,
    }

    scientific_pass = all(component["pass"] for component in components.values())
    clean_core_pass = (
        components["clean_rotation_yaw_pitch"]["pass"]
        and components["clean_rotation_roll"]["pass"]
        and components["clean_closing"]["pass"]
    )
    if scientific_pass:
        verdict = "PASS"
    elif clean_core_pass:
        verdict = "REVISE"
    else:
        informative_expected = len(yaw_pitch) > 0
        verdict = "STOP" if informative_expected and len(informative) > 0 else "REVISE"
    return {
        "schema_version": "rcle.phase_a.scientific_summary.v1",
        "protocol_id": protocol["protocol_id"],
        "planned_trials": expected,
        "actual_trials": len(rows),
        "evaluable_trials": sum(bool(row["evaluable"]) for row in rows),
        "not_evaluable_trials": sum(not bool(row["evaluable"]) for row in rows),
        "components": components,
        "scientific_gate_pass": scientific_pass,
        "clean_core_pass": clean_core_pass,
        "verdict": verdict,
        "authority": "SYNTHETIC_MECHANISM_AND_IMPLEMENTATION_EVIDENCE_ONLY",
    }
