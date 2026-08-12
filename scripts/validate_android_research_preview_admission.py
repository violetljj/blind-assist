"""Fail-closed validator for the Android research-preview admission receipt.

This contract-only validator does not open protected outcomes, execute a model,
run an Android benchmark, or change Android/runtime behavior.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "blindassist_android_research_preview_admission_v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "android_research_preview_admission_v1.schema.json"
PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
QUALITY_METRICS = ("false_clear", "false_block", "known_coverage", "clearance_error", "transition_consistency")
QUALITY_DIRECTIONS = {"false_clear": "max", "false_block": "max", "known_coverage": "min", "clearance_error": "max", "transition_consistency": "min"}
ANDROID_THRESHOLDS = {
    "cold_start_ms": ("max_cold_start_ms", "max"),
    "warm_start_ms": ("max_warm_start_ms", "max"),
    "latency_p50_ms": ("max_latency_p50_ms", "max"),
    "latency_p95_ms": ("max_latency_p95_ms", "max"),
    "peak_memory_mb": ("max_peak_memory_mb", "max"),
    "thermal_window_seconds": ("min_thermal_window_seconds", "min"),
}
IDENTITIES = ("model_sha256", "export_sha256", "preprocess_sha256", "postprocess_sha256", "input_manifest_sha256")


class ValidationError(ValueError):
    """The receipt violates the frozen admission contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    _require(ref.startswith("#/"), f"unsupported schema reference: {ref}")
    value: Any = root
    for token in ref[2:].split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def _is_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _finite_number(value)
    raise ValidationError(f"unsupported schema type: {expected}")


def _validate_schema(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    if "$ref" in schema:
        _validate_schema(value, _resolve_ref(root, schema["$ref"]), root, path)
        return
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _validate_schema(value, option, root, path)
                return
            except ValidationError:
                pass
        raise ValidationError(f"{path}: no anyOf branch matched")
    if "const" in schema:
        _require(value == schema["const"], f"{path}: const mismatch")
    if "enum" in schema:
        _require(value in schema["enum"], f"{path}: value is not in enum")
    expected_types = schema.get("type")
    if expected_types is not None:
        expected_types = [expected_types] if isinstance(expected_types, str) else expected_types
        _require(any(_is_type(value, expected) for expected in expected_types), f"{path}: type mismatch")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        _require(not missing, f"{path}: missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            _require(not extra, f"{path}: unexpected fields {extra}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], root, f"{path}.{key}")
    elif isinstance(value, list):
        _require(len(value) >= schema.get("minItems", 0), f"{path}: too few items")
        if "maxItems" in schema:
            _require(len(value) <= schema["maxItems"], f"{path}: too many items")
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True, allow_nan=False) for item in value]
            _require(len(normalized) == len(set(normalized)), f"{path}: duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(len(value) >= schema.get("minLength", 0), f"{path}: string is too short")
        if "pattern" in schema:
            _require(re.fullmatch(schema["pattern"], value) is not None, f"{path}: pattern mismatch")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                _require(parsed.tzinfo is not None, f"{path}: date-time must include an offset")
            except ValueError as exc:
                raise ValidationError(f"{path}: invalid date-time") from exc
    if _finite_number(value) and "minimum" in schema:
        _require(value >= schema["minimum"], f"{path}: value is below minimum")


def _validate_schema_surface(receipt: dict[str, Any]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load receipt schema: {exc}") from exc
    _validate_schema(receipt, schema, schema)


def _validate_metric_set(metrics: Any, context: str) -> None:
    _require(isinstance(metrics, dict) and set(metrics) == set(QUALITY_METRICS), f"{context} metric inventory mismatch")
    for name in QUALITY_METRICS:
        _require(_finite_number(metrics[name]) and metrics[name] >= 0, f"{context}.{name} must be non-negative and finite")


def _validate_records(receipt: dict[str, Any]) -> None:
    evidence = receipt["evidence"]
    records = evidence["parent_sessions"]
    seen: set[tuple[str, str]] = set()
    total = 0
    pooled_sums = {name: 0.0 for name in QUALITY_METRICS}
    for index, record in enumerate(records):
        context = f"parent_sessions[{index}]"
        key = (record["parent_id"], record["session_id"])
        _require(key not in seen, f"duplicate parent/session record: {key}")
        seen.add(key)
        denominator = record["denominator"]
        _require(isinstance(denominator, int) and not isinstance(denominator, bool) and denominator > 0, f"{context} denominator invalid")
        total += denominator
        _validate_metric_set(record["metrics"], f"{context}.metrics")
        _validate_metric_set(record["metric_sums"], f"{context}.metric_sums")
        for name in QUALITY_METRICS:
            expected_metric = record["metric_sums"][name] / denominator
            _require(math.isclose(record["metrics"][name], expected_metric, rel_tol=0.0, abs_tol=1e-12), f"{context}.{name} does not equal metric_sums/denominator")
            pooled_sums[name] += record["metric_sums"][name]

    pooled = evidence["pooled"]
    _require(pooled["denominator"] == total, "pooled denominator does not equal session sum")
    _validate_metric_set(pooled["metrics"], "pooled.metrics")
    for name in QUALITY_METRICS:
        expected_metric = pooled_sums[name] / total
        _require(math.isclose(pooled["metrics"][name], expected_metric, rel_tol=0.0, abs_tol=1e-12), f"pooled.{name} is not recomputed from parent/session evidence")


def _quality_decision(receipt: dict[str, Any]) -> tuple[str, list[str]]:
    metrics = receipt["evidence"]["pooled"]["metrics"]
    thresholds = receipt["contract"]["thresholds"]["values"]
    reasons = []
    for name, direction in QUALITY_DIRECTIONS.items():
        if (direction == "max" and metrics[name] > thresholds[name]) or (direction == "min" and metrics[name] < thresholds[name]):
            reasons.append(f"{name.upper()}_THRESHOLD")
    if reasons:
        return FAIL, sorted(reasons)
    return PASS, ["QUALITY_THRESHOLDS_PASSED"]


def _android_decision(receipt: dict[str, Any]) -> tuple[str, list[str]]:
    contract = receipt["contract"]
    observed = receipt["observed"]
    android = observed["android"]
    thresholds = contract["thresholds"]["values"]
    fail_reasons: list[str] = []
    unknown_reasons: list[str] = []

    for name in IDENTITIES:
        value = observed["identity"][name]
        if value is None:
            unknown_reasons.append(f"MISSING_OBSERVED_{name.upper()}")
        elif value != contract["identity"][name]:
            fail_reasons.append(f"{name.upper()}_MISMATCH")

    backend = android["observed_backend"]
    if backend is None:
        unknown_reasons.append("MISSING_OBSERVED_BACKEND")
    elif backend != contract["supported_backend"]:
        fail_reasons.append("BACKEND_MISMATCH")

    fallback_observed = android["fallback_observed"]
    fallback_backend = android["fallback_backend"]
    if fallback_observed is True:
        fail_reasons.append("BACKEND_FALLBACK")
        if not isinstance(fallback_backend, str) or not fallback_backend.strip():
            fail_reasons.append("FALLBACK_STATE_INCONSISTENT")
    elif fallback_observed is False:
        if fallback_backend is not None:
            fail_reasons.append("FALLBACK_STATE_INCONSISTENT")
    else:
        unknown_reasons.append("MISSING_FALLBACK_EVIDENCE")
        if fallback_backend is not None:
            fail_reasons.append("FALLBACK_STATE_INCONSISTENT")

    parity = android["reference_parity"]
    runtime_sha256 = parity["runtime_sha256"]
    parity_error = parity["measured_max_abs_error"]
    if runtime_sha256 is None or parity_error is None:
        unknown_reasons.append("MISSING_REFERENCE_PARITY_EVIDENCE")
    else:
        if runtime_sha256 != contract["reference_runtime_sha256"]:
            fail_reasons.append("REFERENCE_RUNTIME_SHA256_MISMATCH")
        if parity_error > thresholds["max_reference_parity_error"]:
            fail_reasons.append("REFERENCE_PARITY_FAILURE")

    for metric, (threshold_name, direction) in ANDROID_THRESHOLDS.items():
        measured = android[metric]
        if measured is None:
            unknown_reasons.append(f"MISSING_{metric.upper()}")
        elif (direction == "max" and measured > thresholds[threshold_name]) or (direction == "min" and measured < thresholds[threshold_name]):
            fail_reasons.append(f"{metric.upper()}_THRESHOLD")

    if fail_reasons:
        return FAIL, sorted(set(fail_reasons))
    if unknown_reasons:
        return UNKNOWN, sorted(set(unknown_reasons))
    return PASS, ["ANDROID_FEASIBILITY_GATES_PASSED"]


def _admission_decision(receipt: dict[str, Any], quality: tuple[str, list[str]], android: tuple[str, list[str]]) -> tuple[str, list[str]]:
    evidence = receipt["evidence"]
    fail_reasons: list[str] = []
    unknown_reasons: list[str] = []
    if quality[0] == FAIL:
        fail_reasons.extend(quality[1])
    elif quality[0] == UNKNOWN:
        unknown_reasons.extend(quality[1])
    if android[0] == FAIL:
        fail_reasons.extend(android[1])
    elif android[0] == UNKNOWN:
        unknown_reasons.extend(android[1])
    if evidence["opened_after_contract_freeze"] is False:
        fail_reasons.append("PRE_FREEZE_OUTCOME_ACCESS")
    if evidence["parent_session_disjoint"] is False:
        fail_reasons.append("PARENT_SESSION_OVERLAP")
    if not evidence["complete"]:
        unknown_reasons.append("INCOMPLETE_EVIDENCE")
    if fail_reasons:
        return FAIL, sorted(set(fail_reasons))
    if unknown_reasons:
        return UNKNOWN, sorted(set(unknown_reasons))
    return PASS, ["ALL_RESEARCH_PREVIEW_GATES_PASSED"]


def validate(receipt: dict[str, Any]) -> None:
    _validate_schema_surface(receipt)
    _validate_records(receipt)
    quality_expected = _quality_decision(receipt)
    android_expected = _android_decision(receipt)
    admission_expected = _admission_decision(receipt, quality_expected, android_expected)

    _require((receipt["quality"]["decision"], receipt["quality"]["reason_codes"]) == quality_expected, "quality decision is not derived from pooled evidence")
    _require((receipt["android_feasibility"]["decision"], receipt["android_feasibility"]["reason_codes"]) == android_expected, "Android feasibility decision is not derived from frozen controls")
    _require((receipt["admission"]["decision"], receipt["admission"]["reason_codes"]) == admission_expected, "admission decision contradicts deterministic stop conditions")
    _require(receipt["admission"]["terminal"] is True and receipt["admission"]["rescue_allowed"] is False, "terminal no-rescue rule violated")
    _require(receipt["product_authority"]["decision"] == "DENIED" and receipt["product_authority"]["authorized_scopes"] == [], "product authority must remain denied")
    expected_scope = "ANDROID_RESEARCH_PREVIEW_ONLY" if admission_expected[0] == PASS else "NONE"
    _require(receipt["admission"]["authorized_scope"] == expected_scope, "admission scope contradicts terminal decision")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate_android_research_preview_admission.py RECEIPT.json", file=sys.stderr)
        return 2
    try:
        validate(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
