"""Validate the Android research-preview admission receipt contract.

This validator is contract-only. It does not open protected outcomes, execute a
model, run an Android benchmark, or make an admission decision from raw data.
It verifies that a claimed receipt is internally auditable and fail closed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "blindassist_android_research_preview_admission_v1"
PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

QUALITY_METRICS = (
    "false_clear",
    "false_block",
    "known_coverage",
    "clearance_error",
    "transition_consistency",
)
ANDROID_METRICS = (
    "cold_start_ms",
    "warm_start_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_memory_mb",
    "thermal_window_seconds",
)
QUALITY_DIRECTIONS = {
    "false_clear": "max",
    "false_block": "max",
    "known_coverage": "min",
    "clearance_error": "max",
    "transition_consistency": "min",
}
IDENTITIES = (
    "model_sha256",
    "export_sha256",
    "preprocess_sha256",
    "postprocess_sha256",
    "input_manifest_sha256",
)


class ValidationError(ValueError):
    """The receipt violates the frozen admission contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_binding(value: Any, name: str) -> None:
    _require(isinstance(value, dict), f"{name} binding missing")
    _require(isinstance(value.get("version"), str) and value["version"], f"{name} version missing")
    _require(_sha256(value.get("sha256")), f"{name} sha256 invalid")


def _validate_metric_set(metrics: Any, names: tuple[str, ...], context: str) -> None:
    _require(isinstance(metrics, dict), f"{context} metrics missing")
    _require(set(metrics) == set(names), f"{context} metric inventory mismatch")
    for name in names:
        _require(_finite_number(metrics[name]), f"{context}.{name} must be finite")


def _validate_records(receipt: dict[str, Any]) -> None:
    evidence = receipt["evidence"]
    records = evidence.get("parent_sessions")
    _require(isinstance(records, list) and records, "parent/session records missing")
    seen: set[tuple[str, str]] = set()
    total = 0
    for index, record in enumerate(records):
        context = f"parent_sessions[{index}]"
        _require(isinstance(record, dict), f"{context} must be an object")
        parent_id = record.get("parent_id")
        session_id = record.get("session_id")
        _require(isinstance(parent_id, str) and parent_id, f"{context} parent_id missing")
        _require(isinstance(session_id, str) and session_id, f"{context} session_id missing")
        key = (parent_id, session_id)
        _require(key not in seen, f"duplicate parent/session record: {key}")
        seen.add(key)
        denominator = record.get("denominator")
        _require(isinstance(denominator, int) and denominator > 0, f"{context} denominator invalid")
        total += denominator
        _validate_metric_set(record.get("metrics"), QUALITY_METRICS, context)

    pooled = evidence.get("pooled")
    _require(isinstance(pooled, dict), "pooled evidence missing")
    _require(pooled.get("denominator") == total, "pooled denominator does not equal session sum")
    _validate_metric_set(pooled.get("metrics"), QUALITY_METRICS, "pooled")


def _stop_conditions(receipt: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the normative terminal using FAIL > UNKNOWN > PASS precedence."""
    contract = receipt["contract"]
    observed = receipt["observed"]
    evidence = receipt["evidence"]
    fail_reasons: list[str] = []
    unknown_reasons: list[str] = []

    pooled_metrics = evidence.get("pooled", {}).get("metrics", {})
    thresholds = contract["thresholds"]["values"]
    for name, direction in QUALITY_DIRECTIONS.items():
        measured = pooled_metrics.get(name)
        threshold = thresholds[name]
        if direction == "max" and measured > threshold:
            fail_reasons.append(f"{name.upper()}_THRESHOLD")
        elif direction == "min" and measured < threshold:
            fail_reasons.append(f"{name.upper()}_THRESHOLD")

    if evidence.get("opened_after_contract_freeze") is False:
        fail_reasons.append("PRE_FREEZE_OUTCOME_ACCESS")
    if evidence.get("parent_session_disjoint") is False:
        fail_reasons.append("PARENT_SESSION_OVERLAP")

    frozen_identity = contract["identity"]
    observed_identity = observed.get("identity", {})
    for name in IDENTITIES:
        frozen_value = frozen_identity.get(name)
        observed_value = observed_identity.get(name)
        if observed_value is None:
            unknown_reasons.append(f"MISSING_OBSERVED_{name.upper()}")
        elif observed_value != frozen_value:
            fail_reasons.append(f"{name.upper()}_MISMATCH")

    android = observed.get("android", {})
    requested_backend = contract.get("supported_backend")
    observed_backend = android.get("observed_backend")
    if observed_backend is None:
        unknown_reasons.append("MISSING_OBSERVED_BACKEND")
    elif observed_backend != requested_backend:
        fail_reasons.append("BACKEND_MISMATCH")
    if android.get("fallback_observed") is True:
        fail_reasons.append("BACKEND_FALLBACK")
    elif android.get("fallback_observed") is None:
        unknown_reasons.append("MISSING_FALLBACK_EVIDENCE")

    parity = android.get("reference_parity", {})
    if not _sha256(parity.get("runtime_sha256")) or not _finite_number(
        parity.get("measured_max_abs_error")
    ):
        unknown_reasons.append("MISSING_REFERENCE_PARITY_EVIDENCE")
    elif parity["measured_max_abs_error"] > contract["thresholds"]["values"]["max_reference_parity_error"]:
        fail_reasons.append("REFERENCE_PARITY_FAILURE")

    for name in ANDROID_METRICS:
        if not _finite_number(android.get(name)):
            unknown_reasons.append(f"MISSING_{name.upper()}")

    if not evidence.get("complete", False):
        unknown_reasons.append("INCOMPLETE_EVIDENCE")

    if fail_reasons:
        return FAIL, sorted(set(fail_reasons))
    if unknown_reasons:
        return UNKNOWN, sorted(set(unknown_reasons))
    return PASS, ["ALL_RESEARCH_PREVIEW_GATES_PASSED"]


def validate(receipt: dict[str, Any]) -> None:
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    for key in (
        "contract",
        "observed",
        "evidence",
        "quality",
        "android_feasibility",
        "product_authority",
        "admission",
    ):
        _require(isinstance(receipt.get(key), dict), f"{key} missing")

    contract = receipt["contract"]
    for name in ("protocol", "thresholds", "roster"):
        _validate_binding(contract.get(name), name)
    _require(isinstance(contract["thresholds"].get("values"), dict), "frozen threshold values missing")
    for name in (*QUALITY_METRICS, "max_reference_parity_error"):
        _require(_finite_number(contract["thresholds"]["values"].get(name)), f"threshold {name} invalid")
    _require(
        isinstance(contract["roster"].get("roles"), list)
        and contract["roster"]["roles"],
        "frozen roster roles missing",
    )
    _require(isinstance(contract.get("identity"), dict), "frozen identity missing")
    for name in IDENTITIES:
        _require(_sha256(contract["identity"].get(name)), f"frozen {name} invalid")
    _require(
        isinstance(contract.get("supported_backend"), str)
        and contract["supported_backend"],
        "supported backend missing",
    )

    _validate_records(receipt)
    expected_decision, expected_reasons = _stop_conditions(receipt)
    admission = receipt["admission"]
    _require(
        admission.get("decision") == expected_decision,
        "admission decision contradicts stop conditions",
    )
    _require(
        admission.get("reason_codes") == expected_reasons,
        "admission reason codes are not deterministic",
    )
    _require(
        admission.get("terminal") is True
        and admission.get("rescue_allowed") is False,
        "terminal no-rescue rule violated",
    )

    quality = receipt["quality"]
    android = receipt["android_feasibility"]
    authority = receipt["product_authority"]
    _require(
        authority.get("decision") == "DENIED"
        and authority.get("authorized_scopes") == [],
        "product authority must remain denied",
    )
    if expected_decision == PASS:
        _require(
            quality.get("decision") == PASS and android.get("decision") == PASS,
            "PASS requires both subordinate gates",
        )
        _require(
            admission.get("authorized_scope")
            == "ANDROID_RESEARCH_PREVIEW_ONLY",
            "PASS scope invalid",
        )
    else:
        _require(admission.get("authorized_scope") == "NONE", "non-PASS cannot authorize a scope")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_android_research_preview_admission.py RECEIPT.json", file=sys.stderr)
        return 2
    try:
        receipt = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        validate(receipt)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
