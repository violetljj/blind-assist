"""Fail-closed validator for the Android research-preview admission receipt.

This contract-only validator does not open protected outcomes, execute a model,
run an Android benchmark, or change Android/runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "blindassist_android_research_preview_admission_v1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "android_research_preview_admission_v1.schema.json"
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
IDENTITIES = ("model_sha256", "export_sha256", "preprocess_sha256", "postprocess_sha256", "input_manifest_sha256", "input_shape", "state_reset_sha256")


class ValidationError(ValueError):
    """The receipt violates the frozen admission contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


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
        _require(value == schema["const"] and (not isinstance(schema["const"], bool) or isinstance(value, bool)), f"{path}: const mismatch")
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
                _require(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})", value) is not None, f"{path}: invalid RFC3339 date-time")
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                _require(parsed.tzinfo is not None, f"{path}: date-time must include an offset")
            except ValueError as exc:
                raise ValidationError(f"{path}: invalid date-time") from exc
    if _finite_number(value) and "maximum" in schema:
        _require(value <= schema["maximum"], f"{path}: value exceeds maximum")
    if _finite_number(value) and "minimum" in schema:
        _require(value >= schema["minimum"], f"{path}: value is below minimum")


def _validate_schema_surface(receipt: dict[str, Any]) -> None:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load receipt schema: {exc}") from exc
    _validate_schema(receipt, schema, schema)


def canonical_sha256(value: Any) -> str:
    """Hash UTF-8 JSON with sorted keys, compact separators, ASCII escapes, no NaN."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _same(actual: Any, expected: Any, context: str) -> None:
    if expected is None:
        _require(actual is None, f"{context}: must be null for unavailable evidence")
    else:
        _require(_finite_number(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"{context}: evidence mismatch")


def performance_summary(samples: dict) -> dict:
    def percentile(values, q):
        return sorted(values)[math.ceil(q * len(values)) - 1] if values else None
    return {
        "cold_start_ms": max(samples["cold_start_ms"], default=None),
        "warm_start_ms": max(samples["warm_start_ms"], default=None),
        "latency_p50_ms": percentile(samples["latency_ms"], .5),
        "latency_p95_ms": percentile(samples["latency_ms"], .95),
        "peak_memory_mb": max(samples["memory_mb"], default=None),
        "thermal_window_seconds": min(samples["thermal_window_seconds"], default=None),
    }


def _validate_records(receipt: dict[str, Any]) -> None:
    evidence, contract = receipt["evidence"], receipt["contract"]
    for name, payload in (("thresholds", "values"), ("roster", "members")):
        binding = contract[name]
        content = {"version": binding["version"], payload: binding[payload]}
        if name == "roster":
            content["roles"] = binding["roles"]
        _require(binding["sha256"] == canonical_sha256(content), f"{name} content hash mismatch")
    _require(evidence["frozen_contract_sha256"] == canonical_sha256(contract), "frozen contract hash mismatch")
    members = contract["roster"]["members"]
    keys = [(r["parent_id"], r["session_id"]) for r in members]
    _require(len(keys) == len(set(keys)), "duplicate roster entry")
    parent_roles, session_parents = {}, {}
    for r in members:
        parent_roles.setdefault(r["parent_id"], set()).add(r["data_role"])
        session_parents.setdefault(r["session_id"], set()).add(r["parent_id"])
    disjoint = all(len(v) == 1 for v in parent_roles.values()) and all(len(v) == 1 for v in session_parents.values())
    _require(evidence["parent_session_disjoint"] == disjoint, "disjoint summary contradicts roster")
    opened = evidence["outcome_opened_at"]
    after = opened is not None and datetime.fromisoformat(opened.replace("Z", "+00:00")) >= datetime.fromisoformat(contract["frozen_at"].replace("Z", "+00:00"))
    _require(evidence["opened_after_contract_freeze"] == after, "freeze summary contradicts timestamps")
    records = evidence["parent_sessions"]
    record_keys = [(r["parent_id"], r["session_id"]) for r in records]
    _require(len(set(record_keys)) == len(record_keys), "duplicate parent/session record")
    expected_keys = {(r["parent_id"], r["session_id"]) for r in members if r["data_role"] == "evaluation"}
    _require(set(record_keys).issubset(expected_keys), "record outside frozen evaluation roster")
    for name in QUALITY_METRICS:
        sums, denominators = [], []
        for r in records:
            n, total = r["denominators"][name], r["metric_sums"][name]
            expected = total / n if n and total is not None else None
            if name != "clearance_error" and expected is not None:
                _require(total <= n and float(total).is_integer(), f"{name}: invalid event count")
            _same(r["metrics"][name], expected, f"session {name}")
            sums.append(total); denominators.append(n)
        complete = set(record_keys) == expected_keys and all(n and total is not None for n, total in zip(denominators, sums))
        pooled_n = sum(denominators) if all(n is not None for n in denominators) else None
        _require(evidence["pooled"]["denominators"][name] == pooled_n, f"pooled {name} denominator mismatch")
        _same(evidence["pooled"]["metrics"][name], sum(sums)/pooled_n if complete else None, f"pooled {name}")
    perf = evidence["performance_sessions"]
    perf_keys = [(r["parent_id"], r["session_id"]) for r in perf]
    _require(len(perf_keys) == len(set(perf_keys)), "duplicate performance session")
    _require(len({r["device_run_id"] for r in perf}) == len(perf), "duplicate device run")
    _require(set(perf_keys).issubset(expected_keys), "performance outside frozen roster")
    all_samples = {k: [] for k in ("cold_start_ms", "warm_start_ms", "latency_ms", "memory_mb", "thermal_window_seconds")}
    for r in perf:
        derived = performance_summary(r["samples"])
        for k,v in derived.items():
            _same(r["summary"][k], v, f"performance session {k}")
        for k in all_samples:
            all_samples[k].extend(r["samples"][k])
    derived = performance_summary(all_samples)
    for k,v in derived.items():
        # Incomplete session evidence cannot be hidden by other sessions' samples.
        source = "latency_ms" if k.startswith("latency_") else "memory_mb" if k == "peak_memory_mb" else k
        complete = set(perf_keys) == expected_keys and all(r["samples"][source] for r in perf)
        _same(receipt["observed"]["android"][k], v if complete else None, f"pooled performance {k}")


def _quality_decision(receipt: dict[str, Any]) -> tuple[str, list[str]]:
    metrics = receipt["evidence"]["pooled"]["metrics"]
    thresholds = receipt["contract"]["thresholds"]["values"]
    reasons = []
    missing = []
    for name, direction in QUALITY_DIRECTIONS.items():
        if metrics[name] is None:
            missing.append(f"MISSING_{name.upper()}_DENOMINATOR_OR_SUM")
            continue
        if (direction == "max" and metrics[name] > thresholds[name]) or (direction == "min" and metrics[name] < thresholds[name]):
            reasons.append(f"{name.upper()}_THRESHOLD")
    if reasons:
        return FAIL, sorted(reasons)
    if missing:
        return UNKNOWN, sorted(missing)
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

    requested = android["requested_backend"]
    if requested is None:
        unknown_reasons.append("MISSING_REQUESTED_BACKEND")
    elif requested != contract["supported_backend"]:
        fail_reasons.append("REQUESTED_BACKEND_MISMATCH")
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

    for session in receipt["evidence"]["performance_sessions"]:
        for metric, (threshold_name, direction) in ANDROID_THRESHOLDS.items():
            measured = session["summary"][metric]
            if measured is None:
                unknown_reasons.append(f"MISSING_SESSION_{metric.upper()}")
            elif (direction == "max" and measured > thresholds[threshold_name]) or (direction == "min" and measured < thresholds[threshold_name]):
                fail_reasons.append(f"SESSION_{metric.upper()}_THRESHOLD")
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
    if evidence["outcome_opened_at"] is None:
        unknown_reasons.append("MISSING_OUTCOME_OPEN_TIME")
    elif evidence["opened_after_contract_freeze"] is False:
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
