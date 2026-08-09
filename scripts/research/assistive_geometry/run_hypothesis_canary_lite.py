"""Deterministic CPU canary for Assistive Geometry research hypotheses.

This module uses synthetic geometry only.  It must not read model checkpoints,
TRAIN/DEVELOPMENT/CONFIRMATION outcomes, or device evidence.  The canary checks
mathematical invariants and counterexamples; it does not establish learnability,
novelty, real-data quality, deployment readiness, or safety.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANARY_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_HYPOTHESIS_CANARY_LITE_R0"
SCHEMA = "blindassist.assistive_geometry.hypothesis_canary_lite.v1"
SEED = 1702943


def _as_finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def discrete_hazard_cdf(logits: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interval hazard, post-interval survival, and first-contact CDF.

    If C is the first obstacle clearance, interval hazard k is
    P(C in interval k | C survived all earlier intervals).  Consequently,
    occupancy P(C <= horizon_k) is structurally non-decreasing in k.
    """

    values = _as_finite_vector(logits, name="logits")
    hazards = 1.0 / (1.0 + np.exp(-values))
    survival = np.cumprod(1.0 - hazards)
    occupancy_cdf = 1.0 - survival
    return hazards, survival, occupancy_cdf


def interval_censored_nll(logits: Iterable[float], event_bin: int | None) -> float:
    """Stable discrete-time survival NLL for an event or right censoring.

    ``event_bin=None`` means that no obstacle was observed through the last
    interval, i.e. clearance is right-censored beyond the observed range.
    """

    values = _as_finite_vector(logits, name="logits")
    if event_bin is None:
        return float(np.logaddexp(0.0, values).sum())
    if event_bin < 0 or event_bin >= values.size:
        raise ValueError("event_bin is outside the hazard grid")
    survived_prefix = np.logaddexp(0.0, values[:event_bin]).sum()
    event_term = np.logaddexp(0.0, -values[event_bin])
    return float(survived_prefix + event_term)


def profile_conditioned_clearance(
    obstacles_xz: np.ndarray,
    half_widths_m: Iterable[float],
    *,
    maximum_range_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Query first-contact clearance for nested symmetric body profiles.

    Obstacles are points ``(lateral_x, forward_z)``.  A point intrudes into a
    straight swept body tube when ``abs(x) <= half_width``.  The result is the
    first forward contact, or a right-censored maximum range when none exists.
    """

    obstacles = np.asarray(obstacles_xz, dtype=np.float64)
    widths = _as_finite_vector(half_widths_m, name="half_widths_m")
    if obstacles.ndim != 2 or obstacles.shape[1] != 2 or not np.all(np.isfinite(obstacles)):
        raise ValueError("obstacles_xz must have shape [N, 2] with finite values")
    if np.any(widths <= 0.0) or maximum_range_m <= 0.0:
        raise ValueError("body half-widths and maximum range must be positive")
    if np.any(obstacles[:, 1] < 0.0):
        raise ValueError("forward obstacle distance must be non-negative")

    clearances: list[float] = []
    censored: list[bool] = []
    for half_width in widths:
        intrusions = obstacles[
            (np.abs(obstacles[:, 0]) <= half_width)
            & (obstacles[:, 1] <= maximum_range_m),
            1,
        ]
        if intrusions.size:
            clearances.append(float(np.min(intrusions)))
            censored.append(False)
        else:
            clearances.append(float(maximum_range_m))
            censored.append(True)
    return np.asarray(clearances), np.asarray(censored, dtype=bool)


def widest_forward_path_capacity(field: np.ndarray) -> float:
    """Exact max-min capacity over forward-monotone 8-connected paths."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a non-empty finite matrix")
    previous = values[0].copy()
    for row in range(1, values.shape[0]):
        current = np.empty(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            best_prefix = float(
                np.max(previous[max(0, column - 1) : min(values.shape[1], column + 2)])
            )
            current[column] = min(float(values[row, column]), best_prefix)
        previous = current
    return float(np.max(previous))


def _soft_max(values: np.ndarray, temperature: float) -> float:
    maximum = float(np.max(values))
    return maximum + temperature * float(
        np.log(np.exp((values - maximum) / temperature).sum())
    )


def _soft_min(values: np.ndarray, temperature: float) -> float:
    return -_soft_max(-values, temperature)


def soft_widest_forward_path_capacity(field: np.ndarray, temperature: float) -> float:
    """Log-sum-exp relaxation of the forward max-min dynamic program."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("field must be a non-empty finite matrix")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    previous = values[0].copy()
    for row in range(1, values.shape[0]):
        current = np.empty(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            best_prefix = _soft_max(
                previous[max(0, column - 1) : min(values.shape[1], column + 2)],
                temperature,
            )
            current[column] = _soft_min(
                np.asarray([values[row, column], best_prefix]), temperature
            )
        previous = current
    return _soft_max(previous, temperature)


def split_conformal_lower_bound(
    calibration_prediction: np.ndarray,
    calibration_truth: np.ndarray,
    test_prediction: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """One-sided split-conformal lower bound for metric clearance."""

    prediction = _as_finite_vector(calibration_prediction, name="calibration_prediction")
    truth = _as_finite_vector(calibration_truth, name="calibration_truth")
    test = _as_finite_vector(test_prediction, name="test_prediction")
    if prediction.shape != truth.shape:
        raise ValueError("calibration prediction and truth shapes differ")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    scores = prediction - truth
    rank = min(scores.size, int(math.ceil((scores.size + 1) * (1.0 - alpha))))
    quantile = float(np.partition(scores, rank - 1)[rank - 1])
    return test - quantile, quantile


def minimum_exchangeable_units(alpha: float) -> int:
    """Smallest n for which the best-case 1/(n+1) correction fits alpha."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return int(math.ceil((1.0 / alpha) - 1.0 - 1e-12))


def _band_means(field: np.ndarray) -> list[float]:
    if field.shape[1] % 3 != 0:
        raise ValueError("field width must be divisible into three bands")
    return [float(np.mean(part)) for part in np.split(field, 3, axis=1)]


def _conformal_canary(rng: np.random.Generator) -> dict[str, Any]:
    alpha = 0.05
    threshold_m = 1.5
    calibration_count = 4000
    test_count = 20000

    calibration_truth = rng.uniform(0.4, 2.6, calibration_count)
    calibration_sigma = 0.20 + 0.12 * (calibration_truth < 1.2)
    calibration_prediction = (
        calibration_truth
        + 0.25
        + rng.normal(0.0, calibration_sigma, calibration_count)
    )
    test_truth = rng.uniform(0.4, 2.6, test_count)
    test_sigma = 0.20 + 0.12 * (test_truth < 1.2)
    test_prediction = test_truth + 0.25 + rng.normal(0.0, test_sigma, test_count)
    lower, quantile = split_conformal_lower_bound(
        calibration_prediction,
        calibration_truth,
        test_prediction,
        alpha=alpha,
    )

    point_clear = test_prediction > threshold_m
    conformal_clear = lower > threshold_m
    point_false_clear = np.mean((test_truth <= threshold_m) & point_clear)
    conformal_false_clear = np.mean((test_truth <= threshold_m) & conformal_clear)
    miscoverage = np.mean(test_truth < lower)

    shifted_truth = rng.uniform(0.4, 2.6, test_count)
    shifted_prediction = shifted_truth + 0.25 + rng.normal(0.0, 0.85, test_count)
    shifted_lower = shifted_prediction - quantile
    shifted_miscoverage = np.mean(shifted_truth < shifted_lower)

    return {
        "alpha": alpha,
        "threshold_m": threshold_m,
        "calibration_count": calibration_count,
        "test_count": test_count,
        "calibration_quantile_m": quantile,
        "point_false_clear_all_known": float(point_false_clear),
        "conformal_false_clear_all_known": float(conformal_false_clear),
        "conformal_miscoverage": float(miscoverage),
        "point_clear_fraction": float(np.mean(point_clear)),
        "conformal_clear_fraction": float(np.mean(conformal_clear)),
        "shifted_test_miscoverage": float(shifted_miscoverage),
        "iid_bound_observed": bool(miscoverage <= alpha + 0.01),
        "false_clear_subset_of_miscoverage_observed": bool(
            conformal_false_clear <= miscoverage + 1e-12
        ),
        "distribution_shift_breaks_iid_observation": bool(
            shifted_miscoverage > alpha + 0.10
        ),
        "cluster_crc_feasibility": {
            "current_calibration_parent_count": 4,
            "current_best_case_finite_sample_term": 1.0 / 5.0,
            "target_false_clear_0_08_minimum_parents": minimum_exchangeable_units(0.08),
            "target_miscoverage_0_05_minimum_parents": minimum_exchangeable_units(0.05),
            "target_0_08_feasible_with_current_four_parents": False,
        },
    }


def run_canary() -> dict[str, Any]:
    started = time.perf_counter()

    hazard_logits = np.asarray([0.4, -0.2, 0.8, -1.0], dtype=np.float64)
    hazards, survival, occupancy_cdf = discrete_hazard_cdf(hazard_logits)
    independent_occupancy = np.asarray([0.65, 0.35, 0.80, 0.60])
    independent_violations = int(np.sum(np.diff(independent_occupancy) < 0.0))
    censored_low_hazard_nll = interval_censored_nll([-3.0] * 4, None)
    censored_high_hazard_nll = interval_censored_nll([0.5] * 4, None)
    event_nll = interval_censored_nll(hazard_logits, 2)

    obstacles = np.asarray(
        [
            [0.52, 0.80],
            [0.18, 1.60],
            [-0.75, 0.60],
        ],
        dtype=np.float64,
    )
    half_widths = np.asarray([0.10, 0.25, 0.60, 0.80], dtype=np.float64)
    profile_clearance, profile_censored = profile_conditioned_clearance(
        obstacles,
        half_widths,
        maximum_range_m=3.0,
    )

    blocked = np.ones((6, 6), dtype=np.float64)
    blocked[3, :] = 0.0
    routed = np.ones((6, 6), dtype=np.float64)
    routed[1, 0:2] = 0.0
    routed[3, 2:4] = 0.0
    routed[5, 4:6] = 0.0
    blocked_band_means = _band_means(blocked)
    routed_band_means = _band_means(routed)
    blocked_capacity = widest_forward_path_capacity(blocked)
    routed_capacity = widest_forward_path_capacity(routed)
    soft_values: dict[str, float] = {}
    soft_errors: dict[str, float] = {}
    for temperature in (0.20, 0.05, 0.01):
        soft_value = soft_widest_forward_path_capacity(routed, temperature)
        key = f"tau_{temperature:.2f}"
        soft_values[key] = soft_value
        soft_errors[key] = abs(soft_value - routed_capacity)

    conformal = _conformal_canary(np.random.default_rng(SEED))

    invariants = {
        "hazard_occupancy_monotone": bool(np.all(np.diff(occupancy_cdf) >= -1e-12)),
        "right_censor_prefers_lower_hazard": bool(
            censored_low_hazard_nll < censored_high_hazard_nll
        ),
        "profile_clearance_nonincreasing_with_width": bool(
            np.all(np.diff(profile_clearance) <= 1e-12)
        ),
        "band_aggregates_equal_for_topology_counterexample": bool(
            np.allclose(blocked_band_means, routed_band_means)
        ),
        "widest_path_separates_topology_counterexample": bool(
            blocked_capacity < routed_capacity
        ),
        "soft_widest_path_converges_as_temperature_decreases": bool(
            soft_errors["tau_0.01"] < soft_errors["tau_0.20"]
        ),
        "iid_conformal_bound_observed": conformal["iid_bound_observed"],
        "distribution_shift_counterexample_observed": conformal[
            "distribution_shift_breaks_iid_observation"
        ],
    }
    status = "PASS" if all(invariants.values()) else "FAIL"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema": SCHEMA,
        "canary_id": CANARY_ID,
        "status": status,
        "profile": "CANARY_LITE",
        "research_style": "WILD_LAB",
        "evidence_role": "SYNTHETIC_CPU_MATHEMATICAL_MECHANICS_ONLY",
        "seed": SEED,
        "development_outcome_access": False,
        "confirmation_outcome_access": False,
        "model_or_checkpoint_access": False,
        "hypotheses": {
            "H1_CENSORED_CLEARANCE_SURVIVAL": {
                "hazards": hazards.tolist(),
                "survival": survival.tolist(),
                "occupancy_cdf": occupancy_cdf.tolist(),
                "independent_occupancy": independent_occupancy.tolist(),
                "independent_monotonicity_violations": independent_violations,
                "right_censored_low_hazard_nll": censored_low_hazard_nll,
                "right_censored_high_hazard_nll": censored_high_hazard_nll,
                "event_bin_2_nll": event_nll,
                "terminal": "MECHANISM_INVARIANTS_SUPPORTED_SYNTHETIC_ONLY",
            },
            "H2_PROFILE_CONDITIONED_CONFIGURATION_CLEARANCE": {
                "half_widths_m": half_widths.tolist(),
                "clearance_m": profile_clearance.tolist(),
                "right_censored": profile_censored.tolist(),
                "terminal": "PROFILE_MONOTONICITY_SUPPORTED_SYNTHETIC_ONLY",
            },
            "H3_REACHABILITY_BOTTLENECK_LOSS": {
                "blocked_band_means": blocked_band_means,
                "routed_band_means": routed_band_means,
                "blocked_widest_path_capacity": blocked_capacity,
                "routed_widest_path_capacity": routed_capacity,
                "soft_relaxation_value": soft_values,
                "soft_relaxation_absolute_error": soft_errors,
                "soft_relaxation_can_overestimate_capacity": bool(
                    soft_values["tau_0.20"] > routed_capacity
                ),
                "terminal": "TASK_COUNTEREXAMPLE_SUPPORTED_NOVELTY_COLLISION_HIGH",
            },
            "H4_ONE_SIDED_CONFORMAL_CLEARANCE": {
                **conformal,
                "terminal": "IID_MECHANISM_SUPPORTED_SHIFT_REQUIRES_ABSTENTION",
            },
        },
        "invariants": invariants,
        "elapsed_cpu_ms": elapsed_ms,
        "overall_terminal": (
            "MATH_MECHANICS_SUPPORTED_PAPER_NOVELTY_AND_LEARNABILITY_NOT_ESTABLISHED"
            if status == "PASS"
            else "MATH_MECHANICS_CANARY_FAILED"
        ),
        "claim_ceiling": (
            "Deterministic synthetic CPU mechanics only; no real-data, model-learning, "
            "Development, Confirmation, device, product, production, or safety authority."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_canary()
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
