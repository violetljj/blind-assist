#!/usr/bin/env python3
"""Materialize the frozen Android metric-depth dual-arm instrumentation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ARM_PREFIX = "MetricDepthDualArmR0: ARM_RESULT="
TERMINAL_PATTERN = re.compile(
    r"MetricDepthDualArmR0: TERMINALS cpu=(\S+) nnapi=(\S+)"
)
EXPECTED_ARMS = {
    ("metric3d_vit_small", "ort_cpu"),
    ("unidepth_v2_vits14_camera", "ort_cpu"),
    ("metric3d_vit_small", "ort_nnapi"),
    ("unidepth_v2_vits14_camera", "ort_nnapi"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logcat", type=Path, required=True)
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--host-parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--manufacturer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--android-release", required=True)
    parser.add_argument("--sdk", type=int, required=True)
    parser.add_argument("--abi", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_text = args.logcat.read_text(encoding="utf-8", errors="replace")
    test_text = args.test_result.read_text(encoding="utf-8", errors="replace")
    if "test_status: PASSED" not in test_text:
        raise ValueError("formal instrumentation result is not PASSED")

    arms: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        marker = line.find(ARM_PREFIX)
        if marker >= 0:
            arms.append(json.loads(line[marker + len(ARM_PREFIX) :]))
    keyed = {(arm["arm"], arm["backend"]): arm for arm in arms}
    if set(keyed) != EXPECTED_ARMS or len(arms) != len(EXPECTED_ARMS):
        raise ValueError(f"expected exactly four frozen arms, found {sorted(keyed)}")
    if any(arm.get("status") != "PASS" for arm in arms):
        raise ValueError("one or more frozen arms did not pass")

    terminal_match = TERMINAL_PATTERN.search(log_text)
    if terminal_match is None:
        raise ValueError("missing terminal line")
    cpu_terminal, nnapi_terminal = terminal_match.groups()
    if cpu_terminal != "DUALARM_ANDROID_CPU_EXECUTION_SUPPORTED":
        raise ValueError(f"unexpected CPU terminal: {cpu_terminal}")
    if nnapi_terminal != "DUALARM_ANDROID_NNAPI_EXECUTION_SUPPORTED":
        raise ValueError(f"unexpected NNAPI terminal: {nnapi_terminal}")

    for arm in arms:
        arm["derived_memory_delta"] = {
            "pss_kib": (
                arm["memory_after_runs"]["pss_kib"]
                - arm["memory_before"]["pss_kib"]
            ),
            "java_heap_used_bytes": (
                arm["memory_after_runs"]["java_heap_used_bytes"]
                - arm["memory_before"]["java_heap_used_bytes"]
            ),
            "native_heap_allocated_bytes": (
                arm["memory_after_runs"]["native_heap_allocated_bytes"]
                - arm["memory_before"]["native_heap_allocated_bytes"]
            ),
        }

    metric_cpu = keyed[("metric3d_vit_small", "ort_cpu")]
    unidepth_cpu = keyed[("unidepth_v2_vits14_camera", "ort_cpu")]
    metric_nnapi = keyed[("metric3d_vit_small", "ort_nnapi")]
    unidepth_nnapi = keyed[("unidepth_v2_vits14_camera", "ort_nnapi")]
    report = {
        "schema_version": 1,
        "terminal": "DUALARM_ANDROID_CPU_AND_NNAPI_EXECUTION_SUPPORTED",
        "scope": "deployment_runtime_only_no_quality_claim",
        "formal_instrumentation_passed": True,
        "device": {
            "serial": args.device_serial,
            "manufacturer": args.manufacturer,
            "model": args.model,
            "android_release": args.android_release,
            "sdk": args.sdk,
            "abi": args.abi,
        },
        "runtime": {
            "name": "onnxruntime-android",
            "version": "1.26.0",
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "graph_optimization": "ALL_OPT",
        },
        "cpu_terminal": cpu_terminal,
        "nnapi_terminal": nnapi_terminal,
        "arms": arms,
        "comparisons": {
            "cpu_metric3d_over_unidepth_p95_ratio": (
                metric_cpu["latency_ms"]["p95"]
                / unidepth_cpu["latency_ms"]["p95"]
            ),
            "metric3d_nnapi_vs_cpu_p95_change_fraction": (
                metric_nnapi["latency_ms"]["p95"]
                / metric_cpu["latency_ms"]["p95"]
                - 1.0
            ),
            "unidepth_nnapi_vs_cpu_p95_change_fraction": (
                unidepth_nnapi["latency_ms"]["p95"]
                / unidepth_cpu["latency_ms"]["p95"]
                - 1.0
            ),
        },
        "evidence": {
            "logcat_path": str(args.logcat.resolve()),
            "logcat_sha256": sha256(args.logcat),
            "test_result_path": str(args.test_result.resolve()),
            "test_result_sha256": sha256(args.test_result),
            "host_parity_report_path": str(args.host_parity_report.resolve()),
            "host_parity_report_sha256": sha256(args.host_parity_report),
        },
        "claim_boundary": [
            "No fresh RGB or outcome labels were read.",
            "Synthetic neutral inputs support runtime cost comparison only.",
            "NNAPI session success does not prove accelerator-only graph coverage.",
            "Neither arm is admitted for real-time use by this result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
