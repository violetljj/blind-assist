"""Pure data-layer metrics for RCLE unseen external confirmation R0.

This module consumes an already ordered continuous-expansion pair ledger.  It
does not decode RGB, invoke the RCLE estimator, select windows, or write a
formal claim.  Old and R1 triggers are derived together from the same input
rows so neither path can use a different denominator.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping


THRESHOLD_PER_S = 0.01
REQUIRED_CONSECUTIVE_PAIRS = 3
BELOW_RELATIVE_REDUCTION_MIN = 0.30
POSITIVE_RETENTION_MIN = 0.90
POSITIVE_FIRST_TRIGGER_DELAY_MAX_S = 0.25
MAX_PAIR_DT_S = 0.10

POSITIVE_ROLE = "POSITIVE_APPROACH_WINDOW"
BELOW_ROLE = "BELOW_TRIGGER_REFERENCE_WINDOW"
ROLES = frozenset({POSITIVE_ROLE, BELOW_ROLE})

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"

_REQUIRED_FIELDS = frozenset(
    {
        "source_id",
        "sequence_id",
        "window_id",
        "role",
        "pair_index",
        "previous_timestamp_s",
        "current_timestamp_s",
        "evaluable",
        "compensated_expansion_median_per_s",
    }
)


WindowKey = tuple[str, str, str]


def _meets_minimum(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _meets_maximum(value: float, threshold: float) -> bool:
    return value < threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _window_key(row: Mapping[str, Any]) -> WindowKey:
    return (
        str(row["source_id"]),
        str(row["sequence_id"]),
        str(row["window_id"]),
    )


def _identity(key: WindowKey) -> dict[str, str]:
    return {
        "source_id": key[0],
        "sequence_id": key[1],
        "window_id": key[2],
    }


def _finite_number(value: Any, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field.upper()}_TYPE:row={row_index}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field.upper()}_NONFINITE:row={row_index}")
    return result


def _validate_and_copy_rows(
    pair_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in pair_rows]
    if not rows:
        raise ValueError("EMPTY_PAIR_LEDGER")

    seen_closed: set[WindowKey] = set()
    active_key: WindowKey | None = None
    prior_pair_index: int | None = None
    prior_current_timestamp: float | None = None
    roles_by_window: dict[WindowKey, str] = {}

    for row_index, row in enumerate(rows):
        missing = sorted(_REQUIRED_FIELDS.difference(row))
        if missing:
            raise ValueError(f"MISSING_FIELDS:row={row_index}:{','.join(missing)}")

        for field in ("source_id", "sequence_id", "window_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{field.upper()}_INVALID:row={row_index}")
        if row["role"] not in ROLES:
            raise ValueError(f"ROLE_INVALID:row={row_index}")
        if type(row["pair_index"]) is not int or row["pair_index"] < 0:
            raise ValueError(f"PAIR_INDEX_INVALID:row={row_index}")
        if type(row["evaluable"]) is not bool:
            raise ValueError(f"EVALUABLE_INVALID:row={row_index}")

        previous = _finite_number(
            row["previous_timestamp_s"], "previous_timestamp_s", row_index
        )
        current = _finite_number(
            row["current_timestamp_s"], "current_timestamp_s", row_index
        )
        if not previous < current:
            raise ValueError(f"TIMESTAMP_ORDER:row={row_index}")
        if current - previous > MAX_PAIR_DT_S + 1e-12:
            raise ValueError(
                f"PAIR_DT_EXCEEDS_MAX:row={row_index}:"
                f"maximum_s={MAX_PAIR_DT_S}"
            )

        expansion = row["compensated_expansion_median_per_s"]
        if row["evaluable"]:
            _finite_number(
                expansion, "compensated_expansion_median_per_s", row_index
            )
        elif expansion is not None:
            raise ValueError(f"ABSTENTION_EXPANSION_PRESENT:row={row_index}")

        key = _window_key(row)
        role = str(row["role"])
        prior_role = roles_by_window.setdefault(key, role)
        if prior_role != role:
            raise ValueError(
                "WINDOW_ROLE_DRIFT:"
                f"source={key[0]}:sequence={key[1]}:window={key[2]}"
            )

        if key != active_key:
            if active_key is not None:
                seen_closed.add(active_key)
            if key in seen_closed:
                raise ValueError(
                    "WINDOW_ROWS_NOT_CONTIGUOUS:"
                    f"source={key[0]}:sequence={key[1]}:window={key[2]}"
                )
            active_key = key
            prior_pair_index = None
            prior_current_timestamp = None

        expected_pair_index = 0 if prior_pair_index is None else prior_pair_index + 1
        if row["pair_index"] != expected_pair_index:
            raise ValueError(
                "PAIR_INDEX_NOT_CONTIGUOUS:"
                f"source={key[0]}:sequence={key[1]}:window={key[2]}:"
                f"expected={expected_pair_index}:actual={row['pair_index']}"
            )
        if (
            prior_current_timestamp is not None
            and not math.isclose(
                previous,
                prior_current_timestamp,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "PAIR_TIMESTAMPS_NOT_CONTIGUOUS:"
                f"source={key[0]}:sequence={key[1]}:window={key[2]}:"
                f"pair={row['pair_index']}"
            )
        prior_pair_index = int(row["pair_index"])
        prior_current_timestamp = current

    source_roles: dict[str, dict[str, list[WindowKey]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, role in roles_by_window.items():
        source_roles[key[0]][role].append(key)
    if len(source_roles) != 2:
        raise ValueError(f"SOURCE_COUNT:expected=2:actual={len(source_roles)}")
    for source_id, grouped in sorted(source_roles.items()):
        actual_roles = set(grouped)
        if actual_roles != ROLES:
            raise ValueError(
                f"SOURCE_ROLE_SET:source={source_id}:actual={sorted(actual_roles)}"
            )
        for role in sorted(ROLES):
            if len(grouped[role]) != 1:
                raise ValueError(
                    "SOURCE_ROLE_WINDOW_COUNT:"
                    f"source={source_id}:role={role}:actual={len(grouped[role])}"
                )
    if len(roles_by_window) != 4:
        raise ValueError(f"WINDOW_COUNT:expected=4:actual={len(roles_by_window)}")
    return rows


def derive_confirmation_rows(
    pair_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate identities/order and derive old plus R1 triggers in one pass.

    A missing source-consecutive pair must be represented by an unevaluable
    row.  A gap in ``pair_index`` is invalid input rather than an implicit
    denominator change.
    """

    rows = _validate_and_copy_rows(pair_rows)
    derived: list[dict[str, Any]] = []
    active_key: WindowKey | None = None
    streak = 0

    for row in rows:
        key = _window_key(row)
        boundary = key != active_key
        if boundary:
            active_key = key
            streak = 0

        evaluable = row["evaluable"] is True
        expansion = row["compensated_expansion_median_per_s"]
        above = bool(
            evaluable
            and float(expansion) > THRESHOLD_PER_S
        )
        old_trigger = above
        streak = streak + 1 if above else 0
        r1_trigger = bool(streak >= REQUIRED_CONSECUTIVE_PAIRS)

        if boundary:
            reset_reason: str | None = "WINDOW_BOUNDARY"
        elif not evaluable:
            reset_reason = "ABSTENTION"
        elif not above:
            reset_reason = "AT_OR_BELOW_THRESHOLD"
        else:
            reset_reason = None

        derived.append(
            {
                **row,
                "threshold_per_s": THRESHOLD_PER_S,
                "old_trigger": old_trigger,
                "consecutive_above_threshold_pair_count": streak,
                "r1_trigger": r1_trigger,
                "reset_reason": reset_reason,
            }
        )
    return derived


def _gate(
    value: float | bool | None,
    operator: str,
    threshold: float | bool,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "status": status,
        "pass": status == PASS,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _summarize_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = _window_key(rows[0])
    role = str(rows[0]["role"])
    denominator = len(rows)
    old_count = sum(row["old_trigger"] is True for row in rows)
    r1_count = sum(row["r1_trigger"] is True for row in rows)
    old_coverage = old_count / denominator
    r1_coverage = r1_count / denominator
    first_old = next(
        (
            float(row["current_timestamp_s"])
            for row in rows
            if row["old_trigger"] is True
        ),
        None,
    )
    first_r1 = next(
        (
            float(row["current_timestamp_s"])
            for row in rows
            if row["r1_trigger"] is True
        ),
        None,
    )

    gates: dict[str, dict[str, Any]]
    if role == BELOW_ROLE:
        if old_count == 0:
            gates = {
                "below_relative_reduction": _gate(
                    None,
                    ">=",
                    BELOW_RELATIVE_REDUCTION_MIN,
                    NOT_EVALUABLE,
                    "NOT_EVALUABLE_OLD_BELOW_ZERO",
                )
            }
            status = NOT_EVALUABLE
            status_reason = "NOT_EVALUABLE_OLD_BELOW_ZERO"
        else:
            reduction = 1.0 - (r1_count / old_count)
            reduction_pass = _meets_minimum(
                reduction, BELOW_RELATIVE_REDUCTION_MIN
            )
            gates = {
                "below_relative_reduction": _gate(
                    reduction,
                    ">=",
                    BELOW_RELATIVE_REDUCTION_MIN,
                    PASS if reduction_pass else FAIL,
                )
            }
            status = PASS if reduction_pass else FAIL
            status_reason = None if reduction_pass else "BELOW_REDUCTION_GATE"
    else:
        if old_count == 0:
            gates = {
                "positive_retention": _gate(
                    None,
                    ">=",
                    POSITIVE_RETENTION_MIN,
                    FAIL,
                    "FAIL_OLD_POSITIVE_ZERO",
                ),
                "positive_first_trigger_delay_s": _gate(
                    None,
                    "<=",
                    POSITIVE_FIRST_TRIGGER_DELAY_MAX_S,
                    FAIL,
                    "FAIL_OLD_POSITIVE_ZERO",
                ),
            }
            status = FAIL
            status_reason = "FAIL_OLD_POSITIVE_ZERO"
        elif first_r1 is None:
            retention = r1_count / old_count
            gates = {
                "positive_retention": _gate(
                    retention,
                    ">=",
                    POSITIVE_RETENTION_MIN,
                    (
                        PASS
                        if _meets_minimum(retention, POSITIVE_RETENTION_MIN)
                        else FAIL
                    ),
                ),
                "positive_first_trigger_delay_s": _gate(
                    None,
                    "<=",
                    POSITIVE_FIRST_TRIGGER_DELAY_MAX_S,
                    FAIL,
                    "FAIL_R1_NO_FIRST_TRIGGER",
                ),
            }
            status = FAIL
            status_reason = "FAIL_R1_NO_FIRST_TRIGGER"
        else:
            retention = r1_count / old_count
            delay = first_r1 - first_old  # first_old is non-None when old_count > 0
            retention_pass = _meets_minimum(
                retention, POSITIVE_RETENTION_MIN
            )
            delay_pass = _meets_maximum(
                delay, POSITIVE_FIRST_TRIGGER_DELAY_MAX_S
            )
            gates = {
                "positive_retention": _gate(
                    retention,
                    ">=",
                    POSITIVE_RETENTION_MIN,
                    PASS if retention_pass else FAIL,
                ),
                "positive_first_trigger_delay_s": _gate(
                    delay,
                    "<=",
                    POSITIVE_FIRST_TRIGGER_DELAY_MAX_S,
                    PASS if delay_pass else FAIL,
                ),
            }
            status = PASS if retention_pass and delay_pass else FAIL
            failed = [
                name for name, gate in gates.items() if gate["status"] == FAIL
            ]
            status_reason = None if status == PASS else "+".join(failed)

    return {
        **_identity(key),
        "identity": _identity(key),
        "role": role,
        "fixed_denominator_pair_count": denominator,
        "evaluable_pair_count": sum(row["evaluable"] is True for row in rows),
        "abstention_pair_count": sum(row["evaluable"] is False for row in rows),
        "old_trigger_count": old_count,
        "r1_trigger_count": r1_count,
        "old_trigger_coverage": old_coverage,
        "r1_trigger_coverage": r1_coverage,
        "first_old_trigger_timestamp_s": first_old,
        "first_r1_trigger_timestamp_s": first_r1,
        "gates": gates,
        "status": status,
        "status_reason": status_reason,
    }


def _pooled_diagnostics(
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        grouped[str(window["role"])].append(window)

    role_totals: dict[str, dict[str, Any]] = {}
    for role in sorted(ROLES):
        selected = grouped[role]
        denominator = sum(item["fixed_denominator_pair_count"] for item in selected)
        old_count = sum(item["old_trigger_count"] for item in selected)
        r1_count = sum(item["r1_trigger_count"] for item in selected)
        role_totals[role] = {
            "window_count": len(selected),
            "fixed_denominator_pair_count": denominator,
            "evaluable_pair_count": sum(
                item["evaluable_pair_count"] for item in selected
            ),
            "old_trigger_count": old_count,
            "r1_trigger_count": r1_count,
            "old_trigger_coverage": old_count / denominator,
            "r1_trigger_coverage": r1_count / denominator,
        }

    positive = role_totals[POSITIVE_ROLE]
    below = role_totals[BELOW_ROLE]
    below_reduction = (
        1.0
        - below["r1_trigger_count"] / below["old_trigger_count"]
        if below["old_trigger_count"] > 0
        else None
    )
    positive_retention = (
        positive["r1_trigger_count"] / positive["old_trigger_count"]
        if positive["old_trigger_count"] > 0
        else None
    )
    positive_delays = [
        item["gates"]["positive_first_trigger_delay_s"]["value"]
        for item in grouped[POSITIVE_ROLE]
    ]
    maximum_delay = (
        max(float(value) for value in positive_delays if value is not None)
        if all(value is not None for value in positive_delays)
        else None
    )
    pooled_direction = (
        positive["r1_trigger_count"] * below["fixed_denominator_pair_count"]
        > below["r1_trigger_count"] * positive["fixed_denominator_pair_count"]
    )
    return {
        "diagnostic_only": True,
        "total_pair_count": sum(
            item["fixed_denominator_pair_count"] for item in windows
        ),
        "role_totals": role_totals,
        "pooled_gates": {
            "below_relative_reduction": {
                "value": below_reduction,
                "operator": ">=",
                "threshold": BELOW_RELATIVE_REDUCTION_MIN,
            },
            "positive_retention": {
                "value": positive_retention,
                "operator": ">=",
                "threshold": POSITIVE_RETENTION_MIN,
            },
            "maximum_positive_first_trigger_delay_s": {
                "value": maximum_delay,
                "operator": "<=",
                "threshold": POSITIVE_FIRST_TRIGGER_DELAY_MAX_S,
            },
            "positive_direction": {
                "value": pooled_direction,
                "operator": "==",
                "threshold": True,
            },
        },
    }


def evaluate_confirmation(
    pair_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive pair triggers and evaluate every frozen local gate.

    Scientific outcome precedence is ``FAIL`` over ``NOT_EVALUABLE`` over
    ``PASS``.  Pooled values are returned only under ``pooled_diagnostics`` and
    never participate in the cohort decision.
    """

    derived = derive_confirmation_rows(pair_rows)
    grouped: dict[WindowKey, list[dict[str, Any]]] = defaultdict(list)
    for row in derived:
        grouped[_window_key(row)].append(row)
    windows = [_summarize_window(rows) for rows in grouped.values()]

    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for window in windows:
        by_source[str(window["source_id"])][str(window["role"])] = window

    sources: list[dict[str, Any]] = []
    for source_id in sorted(by_source):
        positive = by_source[source_id][POSITIVE_ROLE]
        below = by_source[source_id][BELOW_ROLE]
        direction = (
            positive["r1_trigger_count"] * below["fixed_denominator_pair_count"]
            > below["r1_trigger_count"] * positive["fixed_denominator_pair_count"]
        )
        direction_gate = _gate(
            direction,
            "==",
            True,
            PASS if direction else FAIL,
        )
        local_statuses = [
            positive["status"],
            below["status"],
            direction_gate["status"],
        ]
        if FAIL in local_statuses:
            status = FAIL
        elif NOT_EVALUABLE in local_statuses:
            status = NOT_EVALUABLE
        else:
            status = PASS
        sources.append(
            {
                "source_id": source_id,
                "positive_window_identity": positive["identity"],
                "below_window_identity": below["identity"],
                "role_direction": direction_gate,
                "local_window_statuses": {
                    POSITIVE_ROLE: positive["status"],
                    BELOW_ROLE: below["status"],
                },
                "all_local_gates_and": status == PASS,
                "status": status,
            }
        )

    local_statuses = [window["status"] for window in windows] + [
        source["role_direction"]["status"] for source in sources
    ]
    if FAIL in local_statuses:
        cohort_status = FAIL
        scientific_outcome = "CONFIRMATION_FAIL_STOP_AT_R1"
    elif NOT_EVALUABLE in local_statuses:
        cohort_status = NOT_EVALUABLE
        scientific_outcome = "CONFIRMATION_NOT_EVALUABLE_STOP_AT_R1"
    else:
        cohort_status = PASS
        scientific_outcome = "CONFIRMATION_PASS"

    return {
        "schema": "rcle.unseen_external_confirmation.metrics.v1",
        "threshold_per_s": THRESHOLD_PER_S,
        "required_consecutive_pairs": REQUIRED_CONSECUTIVE_PAIRS,
        "pair_rows": derived,
        "window_summaries": windows,
        "source_summaries": sources,
        "pooled_diagnostics": _pooled_diagnostics(windows),
        "all_local_gates_and": cohort_status == PASS,
        "cohort_status": cohort_status,
        "scientific_outcome": scientific_outcome,
    }


# A descriptive alias for callers that treat this module as a summarizer.
summarize_confirmation = evaluate_confirmation
