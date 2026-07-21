#!/usr/bin/env python3
"""Convert a hash-attested continuous device benchmark row into a gate input.

This converter is deliberately fail-closed. It does not turn a generic detector
benchmark into a promotion report: the source run must contain a connected-device
marker, a matching embedded model asset SHA-256, positive event denominators, and
a finite continuous-sequence duration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "blindassist_sanpo_device_event_gate_input_v1"
BENCHMARK_SCHEMA = "blindassist_detector_ab_device_benchmark_v2"
DECISION_KERNEL_CONTRACT = "blindassist_shared_decision_kernel_v1"
FEEDBACK_ADAPTER = "planner_accept_all_v1"
ALERT_PROFILE = "STANDARD"
SYNTHETIC_CLOCK_FRAME_STEP_MS = 100
HASH_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(
    benchmark: Mapping[str, Any], *, model_id: str, model_sha256: str, benchmark_sha256: str
) -> dict[str, Any]:
    if len(model_sha256) != HASH_LENGTH or any(char not in "0123456789abcdef" for char in model_sha256):
        raise ValueError("model_sha256 must be a lowercase 64-character SHA-256")
    if benchmark.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(f"benchmark schema must be {BENCHMARK_SCHEMA}")
    if benchmark.get("decision_kernel_contract_id") != DECISION_KERNEL_CONTRACT:
        raise ValueError(f"benchmark decision kernel must be {DECISION_KERNEL_CONTRACT}")
    if benchmark.get("risk_metric_semantics") != "shared_production_stable_risk_v1":
        raise ValueError("benchmark must use shared production stable-risk metrics")
    if benchmark.get("feedback_adapter") != FEEDBACK_ADAPTER:
        raise ValueError(f"benchmark feedback adapter must be {FEEDBACK_ADAPTER}")
    if benchmark.get("alert_profile") != ALERT_PROFILE:
        raise ValueError(f"benchmark alert profile must be {ALERT_PROFILE}")
    if benchmark.get("synthetic_clock_frame_step_ms") != SYNTHETIC_CLOCK_FRAME_STEP_MS:
        raise ValueError(
            f"benchmark synthetic clock step must be {SYNTHETIC_CLOCK_FRAME_STEP_MS} ms"
        )
    if benchmark.get("device_under_test") != "instrumentation-connected-device":
        raise ValueError("benchmark is not marked as a connected-device run")
    models = benchmark.get("models")
    if not isinstance(models, list):
        raise ValueError("benchmark requires a models list")
    model = next((row for row in models if isinstance(row, Mapping) and row.get("id") == model_id), None)
    if model is None:
        raise ValueError(f"model id not found in benchmark: {model_id}")
    if model.get("model_asset_sha256") != model_sha256:
        raise ValueError("benchmark model_asset_sha256 does not match the supplied model SHA-256")
    app = model.get("app_detector")
    if not isinstance(app, Mapping):
        raise ValueError("benchmark model lacks app_detector results")
    if app.get("decision_kernel_contract_id") != DECISION_KERNEL_CONTRACT:
        raise ValueError("app_detector decision kernel contract does not match benchmark contract")
    metrics = app.get("blindassist_metrics")
    total_ms = app.get("total_ms")
    if not isinstance(metrics, Mapping) or not isinstance(total_ms, Mapping):
        raise ValueError("benchmark lacks BlindAssist metrics or total latency stats")

    event_count = _positive_int(metrics, "eventAlertCount")
    critical_event_count = _positive_int(metrics, "criticalEventCount")
    sequence_duration_ms = _positive_int(metrics, "sequenceDurationMs")
    false_alerts_per_minute = _finite_number(metrics, "falseAlertsPerMinute")
    report_metrics = {
        "event_recall": _unit_interval(metrics, "eventAlertRecall"),
        "critical_miss_rate": _nonnegative_int(metrics, "criticalEventMissCount") / critical_event_count,
        "false_alerts_per_minute": false_alerts_per_minute,
        "post_event_clearance_rate": _unit_interval(metrics, "postEventClearanceRate"),
        "repeated_alert_rate": _unit_interval(metrics, "deliveredRepeatedAlertRate"),
        "p95_latency_ms": _finite_number(total_ms, "p95"),
    }
    return {
        "schema": SCHEMA,
        "model_sha256": model_sha256,
        "report_id": f"device-benchmark:{model_id}:{benchmark_sha256[:12]}",
        "metrics": report_metrics,
        "provenance": {
            "benchmark_sha256": benchmark_sha256,
            "model_id": model_id,
            "benchmark_schema": BENCHMARK_SCHEMA,
            "decision_kernel_contract_id": DECISION_KERNEL_CONTRACT,
            "feedback_adapter": FEEDBACK_ADAPTER,
            "alert_profile": ALERT_PROFILE,
            "synthetic_clock_frame_step_ms": SYNTHETIC_CLOCK_FRAME_STEP_MS,
            "feedback_delivery_semantics": "deterministic_planner_acceptance_not_physical_device_delivery",
            "event_alert_count": event_count,
            "critical_event_count": critical_event_count,
            "sequence_duration_ms": sequence_duration_ms,
            "false_alert_count": _nonnegative_int(metrics, "falseAlertCount"),
            "delivered_alert_count": _nonnegative_int(metrics, "deliveredAlertCount"),
            "delivered_repeated_alert_count": _nonnegative_int(metrics, "deliveredRepeatedAlertCount"),
            "suppressed_duplicate_attempt_count": _nonnegative_int(metrics, "suppressedDuplicateAttemptCount"),
            "event_regeneration_count": _nonnegative_int(metrics, "eventRegenerationCount"),
        },
    }


def _positive_int(source: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_int(source, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive for a promotable event report")
    return value


def _nonnegative_int(source: Mapping[str, Any], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _finite_number(source: Mapping[str, Any], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _unit_interval(source: Mapping[str, Any], key: str) -> float:
    value = _finite_number(source, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    benchmark_path = args.benchmark_json.resolve()
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark JSON must be an object")
    report = build_report(
        payload,
        model_id=args.model_id,
        model_sha256=args.model_sha256,
        benchmark_sha256=sha256_file(benchmark_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
