#!/usr/bin/env python3
"""Fail-closed parent-macro statistics for TARO O0R factor headroom.

The module is deliberately pure: it reads no files, writes no artifacts, and
does not know about DepthART or truth materialization.  Its input is one row per
query and arm.  Rows for the requested baseline/candidate pair are joined by
``(parent_id, frame_id, query_id)`` before any statistic is calculated.

Required row fields::

    parent_id, frame_id, query_id, arm, mode,
    truth_value_m, truth_state, truth_known,
    value_m, interval_lower_m, interval_upper_m, state, known

``strata`` is an optional mapping of frozen stratum name to a scalar level.
Truth knownness is explicit: a geometrically known numeric clearance may still
have tristate UNKNOWN because its uncertainty interval crosses a decision
boundary. Only ``truth_known=False`` is excluded. An UNKNOWN/unknown arm is
never converted to a negative label or a zero-loss observation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
import random
from typing import Any


RESULT_SCHEMA = "blindassist.taro.o0r.factor_headroom_statistics.v1"
PRIMARY_BASELINE_ARM = "NONE"
PRIMARY_CANDIDATE_ARM = "SCALE_SUPPORT_BOUNDARY"
PRIMARY_MODE = "VALUE_ONLY_COMMON_SUPPORT"

DEFAULT_INTERVAL_ALPHA = 0.05
DEFAULT_CONFIDENCE_ALPHA = 0.05
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 271_828
DEFAULT_MINIMUM_MEANINGFUL_EFFECT_M = 0.02
DEFAULT_FALSE_CLEAR_DIFFERENCE_UCB_MAX = 0.01
DEFAULT_KNOWN_COVERAGE_DIFFERENCE_LCB_MIN = -0.02
DEFAULT_MINIMUM_FAVORABLE_PARENT_FRACTION = 0.75
DIAGNOSTIC_FAMILY_SIZE = 7
DIAGNOSTIC_ARMS = (
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
)

TRISTATES = frozenset({"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"})
OBSERVED_STATES = frozenset({"CLEAR_OBSERVED", "OCCUPIED_OBSERVED"})

_REQUIRED_ROW_FIELDS = frozenset(
    {
        "parent_id",
        "frame_id",
        "query_id",
        "arm",
        "mode",
        "truth_value_m",
        "truth_state",
        "truth_known",
        "value_m",
        "interval_lower_m",
        "interval_upper_m",
        "state",
        "known",
    }
)


class StatisticsError(ValueError):
    """Invalid statistical input or configuration."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.context = context


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise StatisticsError(code, message, **context)


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_float(value: Any, code: str, field: str) -> float:
    _require(_finite_number(value), code, f"{field} must be finite", field=field, value=value)
    return float(value)


def _probability(value: Any, code: str, field: str, *, open_upper: bool = False) -> float:
    number = _finite_float(value, code, field)
    upper_ok = number < 1.0 if open_upper else number <= 1.0
    interval = "(0, 1)" if open_upper else "(0, 1]"
    _require(0.0 < number and upper_ok, code, f"{field} must be in {interval}", field=field)
    return number


def _nonempty_string(value: Any, code: str, field: str) -> str:
    _require(isinstance(value, str) and bool(value), code, f"{field} must be a non-empty string", field=field)
    return value


def proper_interval_score(
    truth_value_m: Real,
    interval_lower_m: Real,
    interval_upper_m: Real,
    *,
    alpha: float = DEFAULT_INTERVAL_ALPHA,
) -> float:
    """Return the negatively oriented proper interval score.

    For a central ``1-alpha`` interval ``[lower, upper]`` the score is

    ``width + 2/alpha * distance outside the interval``.

    Lower is better.  TARO reports improvement as baseline score minus
    candidate score, so positive values favor the candidate arm.
    """

    alpha_value = _probability(alpha, "INTERVAL_ALPHA_INVALID", "alpha", open_upper=True)
    truth = _finite_float(truth_value_m, "INTERVAL_VALUE_INVALID", "truth_value_m")
    lower = _finite_float(interval_lower_m, "INTERVAL_VALUE_INVALID", "interval_lower_m")
    upper = _finite_float(interval_upper_m, "INTERVAL_VALUE_INVALID", "interval_upper_m")
    _require(lower <= upper, "INTERVAL_ORDER_INVALID", "interval lower bound exceeds upper bound")
    score = upper - lower
    if truth < lower:
        score += (2.0 / alpha_value) * (lower - truth)
    elif truth > upper:
        score += (2.0 / alpha_value) * (truth - upper)
    _require(math.isfinite(score) and score >= 0.0, "INTERVAL_SCORE_INVALID", "interval score is invalid")
    return float(score)


def holm_bonferroni(
    p_values: Mapping[str, Real],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply the Holm step-down correction to exactly seven diagnostics.

    The family cardinality is fixed at seven; callers cannot weaken the
    correction by supplying a smaller family size.  Both adjusted p-values and
    sequential rejection decisions are returned.
    """

    _require(isinstance(p_values, Mapping), "HOLM_INPUT_INVALID", "p_values must be a mapping")
    _require(
        len(p_values) == DIAGNOSTIC_FAMILY_SIZE,
        "HOLM_FAMILY_SIZE_INVALID",
        "Holm family cardinality differs from the frozen diagnostic family",
        expected=DIAGNOSTIC_FAMILY_SIZE,
        actual=len(p_values),
    )
    alpha_value = _probability(alpha, "HOLM_ALPHA_INVALID", "alpha", open_upper=True)
    normalized: list[tuple[str, float]] = []
    for name, value in p_values.items():
        diagnostic = _nonempty_string(name, "HOLM_NAME_INVALID", "diagnostic")
        p_value = _finite_float(value, "HOLM_P_VALUE_INVALID", diagnostic)
        _require(0.0 <= p_value <= 1.0, "HOLM_P_VALUE_INVALID", "p-value must be in [0, 1]", diagnostic=diagnostic)
        normalized.append((diagnostic, p_value))
    normalized.sort(key=lambda item: (item[1], item[0]))

    family_size = len(normalized)
    adjusted: dict[str, float] = {}
    rejected: dict[str, bool] = {}
    order: list[dict[str, Any]] = []
    running_adjusted = 0.0
    rejection_open = True
    for rank, (name, p_value) in enumerate(normalized):
        remaining = family_size - rank
        running_adjusted = max(running_adjusted, min(1.0, remaining * p_value))
        threshold = alpha_value / remaining
        reject = bool(rejection_open and p_value <= threshold)
        if not reject:
            rejection_open = False
        adjusted[name] = float(running_adjusted)
        rejected[name] = reject
        order.append(
            {
                "diagnostic": name,
                "p_value": p_value,
                "rank": rank + 1,
                "threshold": threshold,
                "adjusted_p_value": float(running_adjusted),
                "rejected": reject,
            }
        )
    return {
        "method": "HOLM_BONFERRONI_STEP_DOWN",
        "family_size": family_size,
        "alpha": alpha_value,
        "order": order,
        "adjusted_p_values": adjusted,
        "rejected": rejected,
    }


# A concise name for callers that already know the correction family.
holm_correction = holm_bonferroni


def _normalize_stratum_value(value: Any, *, row_index: int, name: str) -> str | bool | int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        _require(bool(value), "STRATUM_VALUE_INVALID", "stratum string must be non-empty", row=row_index, stratum=name)
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise StatisticsError("STRATUM_VALUE_INVALID", "stratum value must be a finite JSON scalar", row=row_index, stratum=name)


def _validate_row(value: Any, row_index: int) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "ROW_INVALID", "query row must be a mapping", row=row_index)
    missing = sorted(_REQUIRED_ROW_FIELDS - set(value))
    _require(not missing, "ROW_FIELDS_MISSING", "query row is missing required fields", row=row_index, missing=missing)

    row = dict(value)
    for field in ("parent_id", "frame_id", "query_id", "arm", "mode"):
        row[field] = _nonempty_string(row[field], "ROW_IDENTITY_INVALID", field)

    truth_state = row["truth_state"]
    _require(truth_state in TRISTATES, "TRUTH_STATE_INVALID", "truth state is not a TARO tristate", row=row_index)
    _require(isinstance(row["truth_known"], bool), "TRUTH_KNOWN_INVALID", "truth_known must be boolean", row=row_index)
    if row["truth_known"]:
        row["truth_value_m"] = _finite_float(row["truth_value_m"], "TRUTH_VALUE_INVALID", "truth_value_m")
    else:
        _require(truth_state == "UNKNOWN" and row["truth_value_m"] is None, "UNKNOWN_TRUTH_VALUE_FORBIDDEN", "truth_known=False requires UNKNOWN with no numeric value", row=row_index)

    _require(isinstance(row["known"], bool), "ARM_KNOWN_INVALID", "known must be boolean", row=row_index)
    _require(row["state"] in TRISTATES, "ARM_STATE_INVALID", "arm state is not a TARO tristate", row=row_index)
    if row["known"]:
        arm_value = _finite_float(row["value_m"], "ARM_VALUE_INVALID", "value_m")
        lower = _finite_float(row["interval_lower_m"], "ARM_INTERVAL_INVALID", "interval_lower_m")
        upper = _finite_float(row["interval_upper_m"], "ARM_INTERVAL_INVALID", "interval_upper_m")
        _require(lower <= upper, "ARM_INTERVAL_ORDER_INVALID", "arm interval lower bound exceeds upper bound", row=row_index)
        _require(
            lower - 1e-12 <= arm_value <= upper + 1e-12,
            "ARM_VALUE_OUTSIDE_INTERVAL",
            "arm point value is outside its interval",
            row=row_index,
        )
        row["value_m"] = arm_value
        row["interval_lower_m"] = lower
        row["interval_upper_m"] = upper
    else:
        _require(row["state"] == "UNKNOWN", "UNKNOWN_ARM_STATE_INVALID", "unknown arm must have UNKNOWN state", row=row_index)
        _require(
            row["value_m"] is None and row["interval_lower_m"] is None and row["interval_upper_m"] is None,
            "UNKNOWN_ARM_NUMERIC_VALUE_FORBIDDEN",
            "unknown arm cannot carry a point value or interval",
            row=row_index,
        )

    strata_value = row.get("strata", {})
    _require(isinstance(strata_value, Mapping), "STRATA_INVALID", "strata must be a mapping", row=row_index)
    strata: dict[str, str | bool | int | float] = {}
    for raw_name, raw_value in strata_value.items():
        name = _nonempty_string(raw_name, "STRATUM_NAME_INVALID", "stratum")
        strata[name] = _normalize_stratum_value(raw_value, row_index=row_index, name=name)
    row["strata"] = strata
    return row


def _same_optional_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


@dataclass(frozen=True)
class _ArmPair:
    identity: tuple[str, str, str]
    baseline: dict[str, Any]
    candidate: dict[str, Any]


def _pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    candidate_arm: str,
    mode: str,
) -> list[_ArmPair]:
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        "ROWS_INVALID",
        "rows must be a sequence",
    )
    baseline_name = _nonempty_string(baseline_arm, "ARM_NAME_INVALID", "baseline_arm")
    candidate_name = _nonempty_string(candidate_arm, "ARM_NAME_INVALID", "candidate_arm")
    mode_name = _nonempty_string(mode, "MODE_INVALID", "mode")
    _require(baseline_name != candidate_name, "ARM_PAIR_INVALID", "baseline and candidate arms must differ")

    by_arm: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        baseline_name: {},
        candidate_name: {},
    }
    for row_index, raw_row in enumerate(rows):
        row = _validate_row(raw_row, row_index)
        if row["mode"] != mode_name or row["arm"] not in by_arm:
            continue
        identity = (row["parent_id"], row["frame_id"], row["query_id"])
        _require(
            identity not in by_arm[row["arm"]],
            "DUPLICATE_ARM_QUERY",
            "duplicate arm row for one query identity",
            row=row_index,
            arm=row["arm"],
            identity=identity,
        )
        by_arm[row["arm"]][identity] = row

    baseline_ids = set(by_arm[baseline_name])
    candidate_ids = set(by_arm[candidate_name])
    _require(bool(baseline_ids or candidate_ids), "ARM_ROWS_ABSENT", "requested arm/mode rows are absent")
    _require(
        baseline_ids == candidate_ids,
        "ARM_PAIR_IDENTITY_MISMATCH",
        "baseline and candidate query identity sets differ",
        baseline_only=sorted(baseline_ids - candidate_ids),
        candidate_only=sorted(candidate_ids - baseline_ids),
    )

    pairs: list[_ArmPair] = []
    for identity in sorted(baseline_ids):
        baseline = by_arm[baseline_name][identity]
        candidate = by_arm[candidate_name][identity]
        _require(
            baseline["truth_state"] == candidate["truth_state"]
            and baseline["truth_known"] == candidate["truth_known"]
            and _same_optional_float(baseline["truth_value_m"], candidate["truth_value_m"]),
            "PAIRED_TRUTH_MISMATCH",
            "paired arms do not share identical truth",
            identity=identity,
        )
        _require(
            baseline["strata"] == candidate["strata"],
            "PAIRED_STRATA_MISMATCH",
            "paired arms do not share identical frozen strata",
            identity=identity,
        )
        pairs.append(_ArmPair(identity=identity, baseline=baseline, candidate=candidate))
    return pairs


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    output = math.fsum(float(value) for value in values) / len(values)
    _require(math.isfinite(output), "NONFINITE_AGGREGATE", "aggregate mean is non-finite")
    return output


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "BOOTSTRAP_DENOMINATOR_UNDEFINED", "bootstrap draws are empty")
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def _bootstrap_parent_mean(
    values_by_parent: Mapping[str, float],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    parent_ids = sorted(values_by_parent)
    _require(bool(parent_ids), "BOOTSTRAP_DENOMINATOR_UNDEFINED", "bootstrap parent denominator is empty")
    values = [float(values_by_parent[parent_id]) for parent_id in parent_ids]
    _require(all(math.isfinite(value) for value in values), "BOOTSTRAP_VALUE_INVALID", "bootstrap parent values are non-finite")
    rng = random.Random(seed)
    draws = sorted(
        math.fsum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    lower = _linear_quantile(draws, alpha / 2.0)
    upper = _linear_quantile(draws, 1.0 - alpha / 2.0)
    return {
        "unit": "parent",
        "parent_count": len(parent_ids),
        "replicates": replicates,
        "seed": seed,
        "two_sided_alpha": alpha,
        "quantile_method": "linear",
        "lcb": float(lower),
        "ucb": float(upper),
    }


def _exact_parent_sign_flip_p_value(values: Sequence[float]) -> float:
    """One-sided exact randomization p-value for positive parent-macro effect."""

    _require(1 <= len(values) <= 20, "DIAGNOSTIC_PARENT_COUNT_INVALID", "exact sign-flip requires one to twenty parents")
    normalized = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in normalized), "DIAGNOSTIC_EFFECT_INVALID", "diagnostic parent effects must be finite")
    observed_sum = math.fsum(normalized)
    exceed = 0
    permutations = 1 << len(normalized)
    tolerance = 1e-15 * max(1.0, abs(observed_sum))
    for mask in range(permutations):
        permuted = math.fsum(value if mask & (1 << index) else -value for index, value in enumerate(normalized))
        exceed += int(permuted >= observed_sum - tolerance)
    return exceed / permutations


def evaluate_factor_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str] = DIAGNOSTIC_ARMS,
    baseline_arm: str = PRIMARY_BASELINE_ARM,
    mode: str = PRIMARY_MODE,
    interval_alpha: float = DEFAULT_INTERVAL_ALPHA,
    family_alpha: float = DEFAULT_CONFIDENCE_ALPHA,
) -> dict[str, Any]:
    """Report the seven frozen non-NONE contrasts with exact Holm correction."""

    _require(isinstance(arms, Sequence) and not isinstance(arms, (str, bytes)), "DIAGNOSTIC_ARMS_INVALID", "diagnostic arms must be a sequence")
    arm_names = tuple(_nonempty_string(arm, "DIAGNOSTIC_ARMS_INVALID", "arm") for arm in arms)
    _require(len(arm_names) == DIAGNOSTIC_FAMILY_SIZE and len(set(arm_names)) == DIAGNOSTIC_FAMILY_SIZE and baseline_arm not in arm_names, "HOLM_FAMILY_SIZE_INVALID", "diagnostic family must contain exactly seven unique non-baseline arms")
    alpha = _probability(interval_alpha, "INTERVAL_ALPHA_INVALID", "interval_alpha", open_upper=True)
    contrasts: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for arm in arm_names:
        pairs = _pair_rows(rows, baseline_arm=baseline_arm, candidate_arm=arm, mode=mode)
        parent_ids = sorted({pair.identity[0] for pair in pairs})
        effects: dict[str, list[float]] = defaultdict(list)
        for pair in pairs:
            if not pair.baseline["truth_known"] or not (pair.baseline["known"] and pair.candidate["known"]):
                continue
            truth = float(pair.baseline["truth_value_m"])
            baseline_score = proper_interval_score(truth, pair.baseline["interval_lower_m"], pair.baseline["interval_upper_m"], alpha=alpha)
            candidate_score = proper_interval_score(truth, pair.candidate["interval_lower_m"], pair.candidate["interval_upper_m"], alpha=alpha)
            effects[pair.identity[0]].append(baseline_score - candidate_score)
        parent_effects = {parent: _mean(effects[parent]) for parent in parent_ids}
        defined = bool(parent_ids) and all(parent_effects[parent] is not None for parent in parent_ids)
        values = [float(parent_effects[parent]) for parent in parent_ids if parent_effects[parent] is not None]
        point = _mean(values) if defined else None
        p_value = _exact_parent_sign_flip_p_value(values) if defined else 1.0
        p_values[arm] = p_value
        contrasts[arm] = {
            "defined": defined,
            "parent_count": len(parent_ids),
            "paired_parent_effects": {parent: parent_effects[parent] for parent in parent_ids},
            "parent_macro_interval_score_improvement_m": point,
            "one_sided_exact_sign_flip_p_value": p_value,
        }
    holm = holm_bonferroni(p_values, alpha=family_alpha)
    for arm in arm_names:
        contrasts[arm]["holm_adjusted_p_value"] = holm["adjusted_p_values"][arm]
        contrasts[arm]["holm_rejected"] = holm["rejected"][arm]
    return {
        "schema": "blindassist.taro.o0r.factor_diagnostic_holm.v1",
        "baseline_arm": baseline_arm,
        "mode": mode,
        "arms": list(arm_names),
        "test": "ONE_SIDED_EXACT_PARENT_SIGN_FLIP_OF_PARENT_MACRO_INTERVAL_SCORE_IMPROVEMENT",
        "contrasts": contrasts,
        "holm": holm,
    }


def _configuration(
    *,
    interval_alpha: float,
    confidence_alpha: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_meaningful_effect_m: float,
    false_clear_difference_ucb_max: float,
    known_coverage_difference_lcb_min: float,
    minimum_favorable_parent_fraction: float,
) -> dict[str, Any]:
    interval_alpha_value = _probability(interval_alpha, "INTERVAL_ALPHA_INVALID", "interval_alpha", open_upper=True)
    confidence_alpha_value = _probability(confidence_alpha, "CONFIDENCE_ALPHA_INVALID", "confidence_alpha", open_upper=True)
    _require(
        isinstance(bootstrap_replicates, int)
        and not isinstance(bootstrap_replicates, bool)
        and bootstrap_replicates > 0,
        "BOOTSTRAP_REPLICATES_INVALID",
        "bootstrap_replicates must be a positive integer",
    )
    _require(
        isinstance(bootstrap_seed, int) and not isinstance(bootstrap_seed, bool) and bootstrap_seed >= 0,
        "BOOTSTRAP_SEED_INVALID",
        "bootstrap_seed must be a non-negative integer",
    )
    minimum_effect = _finite_float(minimum_meaningful_effect_m, "EFFECT_THRESHOLD_INVALID", "minimum_meaningful_effect_m")
    _require(minimum_effect >= 0.0, "EFFECT_THRESHOLD_INVALID", "minimum meaningful effect must be non-negative")
    false_clear_max = _finite_float(
        false_clear_difference_ucb_max,
        "FALSE_CLEAR_THRESHOLD_INVALID",
        "false_clear_difference_ucb_max",
    )
    coverage_min = _finite_float(
        known_coverage_difference_lcb_min,
        "KNOWN_COVERAGE_THRESHOLD_INVALID",
        "known_coverage_difference_lcb_min",
    )
    favorable_min = _finite_float(
        minimum_favorable_parent_fraction,
        "FAVORABLE_PARENT_THRESHOLD_INVALID",
        "minimum_favorable_parent_fraction",
    )
    _require(0.0 <= favorable_min <= 1.0, "FAVORABLE_PARENT_THRESHOLD_INVALID", "favorable-parent threshold must be in [0, 1]")
    return {
        "interval_alpha": interval_alpha_value,
        "confidence_alpha": confidence_alpha_value,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "minimum_meaningful_effect_m": minimum_effect,
        "false_clear_difference_ucb_max": false_clear_max,
        "known_coverage_difference_lcb_min": coverage_min,
        "minimum_favorable_parent_fraction": favorable_min,
    }


def evaluate_factor_headroom(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str = PRIMARY_BASELINE_ARM,
    candidate_arm: str = PRIMARY_CANDIDATE_ARM,
    mode: str = PRIMARY_MODE,
    interval_alpha: float = DEFAULT_INTERVAL_ALPHA,
    confidence_alpha: float = DEFAULT_CONFIDENCE_ALPHA,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_meaningful_effect_m: float = DEFAULT_MINIMUM_MEANINGFUL_EFFECT_M,
    false_clear_difference_ucb_max: float = DEFAULT_FALSE_CLEAR_DIFFERENCE_UCB_MAX,
    known_coverage_difference_lcb_min: float = DEFAULT_KNOWN_COVERAGE_DIFFERENCE_LCB_MIN,
    minimum_favorable_parent_fraction: float = DEFAULT_MINIMUM_FAVORABLE_PARENT_FRACTION,
    critical_strata: Sequence[str] = (),
    structurally_not_applicable_strata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen candidate arm against NONE with parent as the unit.

    Scientific denominator failures are returned as failed gates with ``None``
    estimates rather than silently dropping parents.  Malformed/unpaired input
    raises :class:`StatisticsError`, since no result can be attributed to it.
    """

    config = _configuration(
        interval_alpha=interval_alpha,
        confidence_alpha=confidence_alpha,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_meaningful_effect_m=minimum_meaningful_effect_m,
        false_clear_difference_ucb_max=false_clear_difference_ucb_max,
        known_coverage_difference_lcb_min=known_coverage_difference_lcb_min,
        minimum_favorable_parent_fraction=minimum_favorable_parent_fraction,
    )
    _require(
        isinstance(critical_strata, Sequence) and not isinstance(critical_strata, (str, bytes)),
        "CRITICAL_STRATA_INVALID",
        "critical_strata must be a sequence of names",
    )
    stratum_names = tuple(_nonempty_string(name, "STRATUM_NAME_INVALID", "critical_strata") for name in critical_strata)
    _require(len(set(stratum_names)) == len(stratum_names), "CRITICAL_STRATA_DUPLICATE", "critical stratum names must be unique")
    raw_not_applicable = structurally_not_applicable_strata or {}
    _require(isinstance(raw_not_applicable, Mapping), "CRITICAL_STRATA_NOT_APPLICABLE_INVALID", "structurally not-applicable strata must be a mapping")
    not_applicable = {
        _nonempty_string(name, "STRATUM_NAME_INVALID", "not_applicable_stratum"): _nonempty_string(reason, "CRITICAL_STRATA_NOT_APPLICABLE_INVALID", "reason")
        for name, reason in raw_not_applicable.items()
    }
    _require(set(not_applicable) <= set(stratum_names), "CRITICAL_STRATA_NOT_APPLICABLE_INVALID", "not-applicable stratum is not in the requested critical family")

    pairs = _pair_rows(rows, baseline_arm=baseline_arm, candidate_arm=candidate_arm, mode=mode)
    parent_ids = sorted({pair.identity[0] for pair in pairs})
    interval_improvements: dict[str, list[float]] = defaultdict(list)
    truth_known_counts: dict[str, int] = defaultdict(int)
    truth_occupied_counts: dict[str, int] = defaultdict(int)
    baseline_known_counts: dict[str, int] = defaultdict(int)
    candidate_known_counts: dict[str, int] = defaultdict(int)
    candidate_classified_counts: dict[str, int] = defaultdict(int)
    common_classified_occupied_counts: dict[str, int] = defaultdict(int)
    baseline_false_clear_counts: dict[str, int] = defaultdict(int)
    candidate_false_clear_counts: dict[str, int] = defaultdict(int)

    stratum_truth_counts: dict[tuple[str, str | bool | int | float, str], int] = defaultdict(int)
    stratum_improvements: dict[tuple[str, str | bool | int | float, str], list[float]] = defaultdict(list)
    stratum_levels: dict[str, set[str | bool | int | float]] = defaultdict(set)
    missing_stratum_rows: dict[str, int] = defaultdict(int)

    for pair in pairs:
        parent_id = pair.identity[0]
        baseline = pair.baseline
        candidate = pair.candidate
        truth_state = baseline["truth_state"]
        if not baseline["truth_known"]:
            continue
        truth_value = float(baseline["truth_value_m"])
        truth_known_counts[parent_id] += 1
        baseline_known_counts[parent_id] += int(baseline["known"])
        candidate_known_counts[parent_id] += int(candidate["known"])
        candidate_classified_counts[parent_id] += int(candidate["state"] in OBSERVED_STATES)
        if truth_state == "OCCUPIED_OBSERVED":
            truth_occupied_counts[parent_id] += 1

        improvement: float | None = None
        if baseline["known"] and candidate["known"]:
            baseline_score = proper_interval_score(
                truth_value,
                baseline["interval_lower_m"],
                baseline["interval_upper_m"],
                alpha=config["interval_alpha"],
            )
            candidate_score = proper_interval_score(
                truth_value,
                candidate["interval_lower_m"],
                candidate["interval_upper_m"],
                alpha=config["interval_alpha"],
            )
            improvement = baseline_score - candidate_score
            interval_improvements[parent_id].append(improvement)
            if (
                truth_state == "OCCUPIED_OBSERVED"
                and baseline["state"] in OBSERVED_STATES
                and candidate["state"] in OBSERVED_STATES
            ):
                common_classified_occupied_counts[parent_id] += 1
                baseline_false_clear_counts[parent_id] += int(baseline["state"] == "CLEAR_OBSERVED")
                candidate_false_clear_counts[parent_id] += int(candidate["state"] == "CLEAR_OBSERVED")

        for stratum_name in stratum_names:
            if stratum_name not in baseline["strata"]:
                missing_stratum_rows[stratum_name] += 1
                continue
            level = baseline["strata"][stratum_name]
            stratum_levels[stratum_name].add(level)
            key = (stratum_name, level, parent_id)
            stratum_truth_counts[key] += 1
            if improvement is not None:
                stratum_improvements[key].append(improvement)

    parent_rows: dict[str, dict[str, Any]] = {}
    primary_by_parent: dict[str, float] = {}
    coverage_by_parent: dict[str, float] = {}
    false_clear_by_parent: dict[str, float] = {}
    primary_denominator_defined = len(parent_ids) >= 2
    coverage_denominator_defined = bool(parent_ids)
    false_clear_denominator_defined = False
    applicable_false_clear_parents = 0

    for parent_id in parent_ids:
        truth_known = truth_known_counts[parent_id]
        primary_value = _mean(interval_improvements[parent_id])
        if truth_known <= 0 or primary_value is None:
            primary_denominator_defined = False
        else:
            primary_by_parent[parent_id] = primary_value

        if truth_known <= 0:
            coverage_denominator_defined = False
            baseline_coverage = None
            candidate_coverage = None
            coverage_difference = None
        else:
            baseline_coverage = baseline_known_counts[parent_id] / truth_known
            candidate_coverage = candidate_known_counts[parent_id] / truth_known
            coverage_difference = candidate_coverage - baseline_coverage
            coverage_by_parent[parent_id] = coverage_difference

        truth_occupied = truth_occupied_counts[parent_id]
        common_occupied = common_classified_occupied_counts[parent_id]
        if truth_occupied > 0:
            applicable_false_clear_parents += 1
        if truth_occupied > 0 and common_occupied > 0:
            baseline_false_clear = baseline_false_clear_counts[parent_id] / common_occupied
            candidate_false_clear = candidate_false_clear_counts[parent_id] / common_occupied
            false_clear_difference = candidate_false_clear - baseline_false_clear
            false_clear_by_parent[parent_id] = false_clear_difference
        else:
            baseline_false_clear = None
            candidate_false_clear = None
            false_clear_difference = None

        parent_rows[parent_id] = {
            "truth_known_queries": truth_known,
            "paired_interval_queries": len(interval_improvements[parent_id]),
            "interval_score_improvement_m": primary_value,
            "baseline_known_coverage": baseline_coverage,
            "candidate_known_coverage": candidate_coverage,
            "known_coverage_difference": coverage_difference,
            "truth_occupied_queries": truth_occupied,
            "paired_classified_occupied_queries": common_occupied,
            "baseline_false_clear_rate": baseline_false_clear,
            "candidate_false_clear_rate": candidate_false_clear,
            "false_clear_rate_difference": false_clear_difference,
        }

    false_clear_denominator_defined = (
        applicable_false_clear_parents > 0
        and len(false_clear_by_parent) == applicable_false_clear_parents
    )
    primary_denominator_defined = primary_denominator_defined and len(primary_by_parent) == len(parent_ids)
    coverage_denominator_defined = coverage_denominator_defined and len(coverage_by_parent) == len(parent_ids)

    primary_point = _mean(list(primary_by_parent.values())) if primary_denominator_defined else None
    coverage_point = _mean(list(coverage_by_parent.values())) if coverage_denominator_defined else None
    false_clear_point = _mean(list(false_clear_by_parent.values())) if false_clear_denominator_defined else None

    primary_bootstrap = (
        _bootstrap_parent_mean(
            primary_by_parent,
            replicates=config["bootstrap_replicates"],
            seed=config["bootstrap_seed"],
            alpha=config["confidence_alpha"],
        )
        if primary_denominator_defined
        else None
    )
    coverage_bootstrap = (
        _bootstrap_parent_mean(
            coverage_by_parent,
            replicates=config["bootstrap_replicates"],
            seed=config["bootstrap_seed"],
            alpha=config["confidence_alpha"],
        )
        if coverage_denominator_defined
        else None
    )
    false_clear_bootstrap = (
        _bootstrap_parent_mean(
            false_clear_by_parent,
            replicates=config["bootstrap_replicates"],
            seed=config["bootstrap_seed"],
            alpha=config["confidence_alpha"],
        )
        if false_clear_denominator_defined
        else None
    )

    favorable_parent_fraction = (
        sum(value > 0.0 for value in primary_by_parent.values()) / len(parent_ids)
        if primary_denominator_defined
        else None
    )

    leave_one_parent_out: dict[str, float] = {}
    single_parent_driver_ids: list[str] = []
    single_parent_driver_defined = primary_denominator_defined and len(parent_ids) >= 2
    if single_parent_driver_defined and primary_point is not None:
        total = sum(primary_by_parent.values())
        for parent_id in parent_ids:
            leave_one_parent_out[parent_id] = (total - primary_by_parent[parent_id]) / (len(parent_ids) - 1)
        if primary_point > config["minimum_meaningful_effect_m"]:
            single_parent_driver_ids = [
                parent_id
                for parent_id, value in leave_one_parent_out.items()
                if value <= config["minimum_meaningful_effect_m"]
            ]

    total_truth_known = sum(truth_known_counts.values())
    candidate_all_unknown = total_truth_known > 0 and sum(candidate_classified_counts.values()) == 0
    all_unknown_defined = total_truth_known > 0

    stratum_rows: list[dict[str, Any]] = []
    critical_strata_defined = True
    critical_strata_reversals: list[dict[str, Any]] = []
    for stratum_name in stratum_names:
        levels = sorted(stratum_levels[stratum_name], key=lambda item: (type(item).__name__, repr(item)))
        structural_na = stratum_name in not_applicable
        if missing_stratum_rows[stratum_name] > 0 or (len(levels) < 2 and not structural_na):
            critical_strata_defined = False
        for level in levels:
            values_by_parent: dict[str, float] = {}
            group_keys = sorted(
                (key for key in stratum_truth_counts if key[0] == stratum_name and key[1] == level),
                key=lambda key: key[2],
            )
            group_defined = bool(group_keys)
            for key in group_keys:
                parent_value = _mean(stratum_improvements[key])
                if parent_value is None:
                    group_defined = False
                else:
                    values_by_parent[key[2]] = parent_value
            group_defined = group_defined and len(values_by_parent) == len(group_keys)
            group_point = _mean(list(values_by_parent.values())) if group_defined else None
            if not group_defined:
                critical_strata_defined = False
            reversal = bool(
                not structural_na
                and
                group_defined
                and primary_point is not None
                and primary_point > 0.0
                and group_point is not None
                and group_point < 0.0
            )
            row = {
                "stratum": stratum_name,
                "level": level,
                "parent_count": len(values_by_parent),
                "truth_known_queries": sum(stratum_truth_counts[key] for key in group_keys),
                "defined": group_defined,
                "parent_macro_interval_score_improvement_m": group_point,
                "reversal": reversal,
                "structurally_not_applicable": structural_na,
                "not_applicable_reason": not_applicable.get(stratum_name),
            }
            stratum_rows.append(row)
            if reversal:
                critical_strata_reversals.append(row)
    if not stratum_names:
        critical_strata_defined = True

    failure_codes: list[str] = []
    if not primary_denominator_defined:
        failure_codes.append("PRIMARY_DENOMINATOR_UNDEFINED")
    if not false_clear_denominator_defined:
        failure_codes.append("FALSE_CLEAR_DENOMINATOR_UNDEFINED")
    if not coverage_denominator_defined:
        failure_codes.append("KNOWN_COVERAGE_DENOMINATOR_UNDEFINED")
    if not all_unknown_defined:
        failure_codes.append("ALL_UNKNOWN_DENOMINATOR_UNDEFINED")
    if not single_parent_driver_defined:
        failure_codes.append("SINGLE_PARENT_DRIVER_DENOMINATOR_UNDEFINED")
    if not critical_strata_defined:
        failure_codes.append("CRITICAL_STRATUM_DENOMINATOR_UNDEFINED")

    denominators_defined = not failure_codes
    primary_gate = bool(
        primary_bootstrap is not None
        and primary_bootstrap["lcb"] > config["minimum_meaningful_effect_m"]
    )
    false_clear_gate = bool(
        false_clear_bootstrap is not None
        and false_clear_bootstrap["ucb"] <= config["false_clear_difference_ucb_max"]
    )
    coverage_gate = bool(
        coverage_bootstrap is not None
        and coverage_bootstrap["lcb"] >= config["known_coverage_difference_lcb_min"]
    )
    favorable_gate = bool(
        favorable_parent_fraction is not None
        and favorable_parent_fraction >= config["minimum_favorable_parent_fraction"]
    )
    single_parent_gate = bool(single_parent_driver_defined and not single_parent_driver_ids)
    all_unknown_gate = bool(all_unknown_defined and not candidate_all_unknown)
    stratum_gate = bool(critical_strata_defined and not critical_strata_reversals)

    gates = {
        "denominators_defined": denominators_defined,
        "primary_lcb_gt_minimum_meaningful_effect": primary_gate,
        "false_clear_difference_ucb_noninferior": false_clear_gate,
        "known_coverage_difference_lcb_noninferior": coverage_gate,
        "minimum_favorable_parent_fraction": favorable_gate,
        "single_parent_driver_forbidden": single_parent_gate,
        "all_unknown_forbidden": all_unknown_gate,
        "critical_stratum_reversal_forbidden": stratum_gate,
    }
    gates["passed"] = all(gates.values())

    return {
        "schema": RESULT_SCHEMA,
        "comparison": {
            "baseline_arm": baseline_arm,
            "candidate_arm": candidate_arm,
            "mode": mode,
            "paired_identity": ["parent_id", "frame_id", "query_id"],
            "improvement_direction": "BASELINE_INTERVAL_SCORE_MINUS_CANDIDATE_INTERVAL_SCORE",
        },
        "configuration": config,
        "counts": {
            "input_rows": len(rows),
            "paired_queries": len(pairs),
            "parents": len(parent_ids),
            "truth_known_queries": total_truth_known,
            "truth_not_known_queries_excluded": len(pairs) - total_truth_known,
            "applicable_false_clear_parents": applicable_false_clear_parents,
        },
        "parents": parent_rows,
        "primary": {
            "defined": primary_denominator_defined,
            "proper_interval_score_alpha": config["interval_alpha"],
            "parent_macro_paired_improvement_m": primary_point,
            "bootstrap": primary_bootstrap,
            "bootstrap_lcb_m": primary_bootstrap["lcb"] if primary_bootstrap else None,
            "favorable_parent_fraction": favorable_parent_fraction,
        },
        "guardrails": {
            "false_clear_difference": {
                "defined": false_clear_denominator_defined,
                "direction": "CANDIDATE_MINUS_BASELINE",
                "denominator": "TRUTH_OCCUPIED_AND_BOTH_ARM_STATES_OBSERVED",
                "parent_macro_difference": false_clear_point,
                "bootstrap": false_clear_bootstrap,
                "bootstrap_ucb": false_clear_bootstrap["ucb"] if false_clear_bootstrap else None,
            },
            "known_coverage_difference": {
                "defined": coverage_denominator_defined,
                "direction": "CANDIDATE_MINUS_BASELINE",
                "denominator": "TRUTH_KNOWN",
                "parent_macro_difference": coverage_point,
                "bootstrap": coverage_bootstrap,
                "bootstrap_lcb": coverage_bootstrap["lcb"] if coverage_bootstrap else None,
            },
            "favorable_parents": {
                "defined": primary_denominator_defined,
                "fraction": favorable_parent_fraction,
                "criterion": "PARENT_INTERVAL_SCORE_IMPROVEMENT_GT_ZERO",
            },
            "single_parent_driver": {
                "defined": single_parent_driver_defined,
                "driver_parent_ids": single_parent_driver_ids,
                "leave_one_parent_out_improvement_m": leave_one_parent_out,
                "criterion": "FULL_EFFECT_GT_THRESHOLD_AND_LEAVE_ONE_OUT_EFFECT_LE_THRESHOLD",
            },
            "all_unknown": {
                "defined": all_unknown_defined,
                "candidate_all_unknown": candidate_all_unknown,
                "truth_known_queries": total_truth_known,
                "candidate_known_queries": sum(candidate_known_counts.values()),
                "candidate_classified_queries": sum(candidate_classified_counts.values()),
            },
            "critical_strata": {
                "requested": list(stratum_names),
                "structurally_not_applicable": dict(not_applicable),
                "evaluated": bool(stratum_names),
                "defined": critical_strata_defined,
                "rows": stratum_rows,
                "reversals": critical_strata_reversals,
                "criterion": "OVERALL_IMPROVEMENT_GT_ZERO_AND_STRATUM_IMPROVEMENT_LT_ZERO",
            },
        },
        "unknown_policy": {
            "truth_not_known_excluded": True,
            "truth_tristate_unknown_with_numeric_value_allowed": True,
            "arm_unknown_as_negative": False,
            "arm_unknown_as_zero_loss": False,
            "coverage_uses_explicit_known_boolean": True,
        },
        "failure_codes": failure_codes,
        "gates": gates,
    }


# Stable concise entry point for the future runner.
evaluate = evaluate_factor_headroom


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_ALPHA",
    "DEFAULT_FALSE_CLEAR_DIFFERENCE_UCB_MAX",
    "DEFAULT_INTERVAL_ALPHA",
    "DEFAULT_KNOWN_COVERAGE_DIFFERENCE_LCB_MIN",
    "DEFAULT_MINIMUM_FAVORABLE_PARENT_FRACTION",
    "DEFAULT_MINIMUM_MEANINGFUL_EFFECT_M",
    "DIAGNOSTIC_ARMS",
    "DIAGNOSTIC_FAMILY_SIZE",
    "PRIMARY_BASELINE_ARM",
    "PRIMARY_CANDIDATE_ARM",
    "PRIMARY_MODE",
    "RESULT_SCHEMA",
    "StatisticsError",
    "evaluate",
    "evaluate_factor_diagnostics",
    "evaluate_factor_headroom",
    "holm_bonferroni",
    "holm_correction",
    "proper_interval_score",
]
