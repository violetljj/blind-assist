"""Frozen numeric and execution rules for the next geometry admission.

This is a successor template.  It does not modify or reinterpret any consumed
Floor3 evidence and it contains no RGB algorithm.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping


RELATIVE_TOLERANCE = Decimal("1e-12")
ABSOLUTE_TOLERANCE = Decimal("1e-15")
DEFAULT_WORKERS = 8
WINDOW_DURATION_SECONDS = Decimal("10")
REQUIRED_POSITIVE_WINDOWS = 2
REQUIRED_BELOW_REFERENCE_WINDOWS = 2


def as_finite_decimal(value: Any) -> Decimal:
    """Normalize JSON Decimal/float/string values without binary recasting."""
    if isinstance(value, bool) or value is None:
        raise ValueError("FINITE_NUMBER_REQUIRED")
    try:
        normalized = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("FINITE_NUMBER_REQUIRED") from error
    if not normalized.is_finite():
        raise ValueError("FINITE_NUMBER_REQUIRED")
    return normalized


def numbers_equivalent(left: Any, right: Any) -> bool:
    """Compare aggregates with the tolerance frozen before source selection."""
    if left is None or right is None:
        return left is right
    try:
        left_decimal = as_finite_decimal(left)
        right_decimal = as_finite_decimal(right)
    except ValueError:
        return False
    difference = abs(left_decimal - right_decimal)
    scale = max(abs(left_decimal), abs(right_decimal))
    return difference <= max(
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE * scale,
    )


def decimal_median(values: Iterable[Any]) -> Decimal:
    """Return a Decimal median, including the even-cardinality mean case."""
    normalized = [as_finite_decimal(value) for value in values]
    if not normalized:
        raise ValueError("MEDIAN_REQUIRES_VALUES")
    return statistics.median(normalized)


def validate_execution_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if a future formal contract drifts from the frozen funnel."""
    execution = contract.get("execution", {})
    selection = contract.get("geometry_only_selection", {})
    candidate = contract.get("candidate", {})
    numeric = contract.get("numeric_equivalence", {})
    if int(execution.get("default_workers", -1)) != DEFAULT_WORKERS:
        raise ValueError("DEFAULT_WORKERS_DRIFT")
    if Decimal(str(selection.get("window_duration_s"))) != WINDOW_DURATION_SECONDS:
        raise ValueError("WINDOW_DURATION_DRIFT")
    if int(selection.get("required_positive_windows", -1)) != REQUIRED_POSITIVE_WINDOWS:
        raise ValueError("POSITIVE_WINDOW_COUNT_DRIFT")
    if (
        int(selection.get("required_below_reference_windows", -1))
        != REQUIRED_BELOW_REFERENCE_WINDOWS
    ):
        raise ValueError("BELOW_REFERENCE_WINDOW_COUNT_DRIFT")
    if int(candidate.get("metadata_rank", -1)) != 1:
        raise ValueError("ONLY_RANK_ONE_CANDIDATE_ALLOWED")
    if Decimal(str(numeric.get("relative_tolerance"))) != RELATIVE_TOLERANCE:
        raise ValueError("RELATIVE_TOLERANCE_DRIFT")
    if Decimal(str(numeric.get("absolute_tolerance"))) != ABSOLUTE_TOLERANCE:
        raise ValueError("ABSOLUTE_TOLERANCE_DRIFT")
    if numeric.get("finite_only") is not True:
        raise ValueError("FINITE_ONLY_REQUIRED")
    candidate_id = str(candidate.get("candidate_id", "")).lower()
    if "floor3_3" in candidate_id:
        raise ValueError("FLOOR3_3_FORBIDDEN")
    if contract.get("candidate_replacement_allowed") is not False:
        raise ValueError("CANDIDATE_REPLACEMENT_MUST_BE_FALSE")
    if contract.get("post_outcome_window_addition_allowed") is not False:
        raise ValueError("POST_OUTCOME_WINDOW_ADDITION_MUST_BE_FALSE")
    if contract.get("rgb_download_before_geometry_admission") is not False:
        raise ValueError("RGB_BEFORE_GEOMETRY_MUST_BE_FALSE")


def validate_burned_fixture(
    path: Path, source_ledger: Path
) -> dict[str, Any]:
    """Exercise the known Floor3_2 even-median representation failure only."""
    fixture = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if fixture.get("schema") != "rcle.motion_diverse.burned_median_fixture.v1":
        raise ValueError("BURNED_FIXTURE_SCHEMA")
    if fixture.get("source_access") != "BURNED_REGRESSION_ONLY":
        raise ValueError("BURNED_FIXTURE_ACCESS")
    if (
        fixture.get("source_protocol")
        != "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_CROSS_SEQUENCE_HOLDOUT_R0"
    ):
        raise ValueError("BURNED_FIXTURE_PROTOCOL")
    expected_ledger_sha = str(fixture.get("source_geometry_pair_ledger_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_ledger_sha):
        raise ValueError("BURNED_FIXTURE_LEDGER_SHA256")
    ledger_bytes = source_ledger.read_bytes()
    actual_ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    if actual_ledger_sha != expected_ledger_sha:
        raise ValueError("BURNED_FIXTURE_LEDGER_IDENTITY")
    count = int(fixture.get("evaluable_value_count", 0))
    if count <= 0 or count % 2:
        raise ValueError("BURNED_FIXTURE_EVEN_COUNT")
    source_window = int(fixture["source_window"])
    ledger_values = []
    for line in ledger_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, parse_float=Decimal)
        if (
            int(row.get("window_index", -1)) == source_window
            and row.get("geometry_evaluable") is True
        ):
            ledger_values.append(
                row["geometry_signed_radial_expansion_per_s"]
            )
    if len(ledger_values) != count:
        raise ValueError("BURNED_FIXTURE_LEDGER_COUNT")
    ordered = sorted(as_finite_decimal(value) for value in ledger_values)
    actual_centers = [ordered[count // 2 - 1], ordered[count // 2]]
    values = fixture.get("center_values")
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("BURNED_FIXTURE_CENTER_VALUES")
    if [as_finite_decimal(value) for value in values] != actual_centers:
        raise ValueError("BURNED_FIXTURE_CENTER_IDENTITY")
    recomputed = decimal_median(values)
    expected = fixture.get("producer_serialized_median")
    if not numbers_equivalent(recomputed, expected):
        raise ValueError("BURNED_FIXTURE_NUMERIC_MISMATCH")
    return {
        "status": "BURNED_FIXTURE_SMOKE_PASS",
        "source_window": source_window,
        "evaluable_value_count": count,
        "source_geometry_pair_ledger_sha256": actual_ledger_sha,
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recomputed_median": str(recomputed),
        "producer_serialized_median": str(expected),
        "relative_tolerance": str(RELATIVE_TOLERANCE),
        "absolute_tolerance": str(ABSOLUTE_TOLERANCE),
        "default_workers": DEFAULT_WORKERS,
    }
