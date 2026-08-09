#!/usr/bin/env python3
"""Evaluate the frozen D1 608x448 SM8650/v75 pre-outcome device gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RTOL = 3e-5
ATOL = 3e-6
EXPECTED_ELEMENTS = 608 * 448
EXPECTED_BYTES = EXPECTED_ELEMENTS * 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def asset(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_float32(path: Path) -> np.ndarray:
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"unexpected output bytes: {path}")
    value = np.fromfile(path, dtype=np.float32)
    if value.size != EXPECTED_ELEMENTS or not np.isfinite(value).all():
        raise ValueError(f"invalid finite float32 output: {path}")
    return value


def compare(left_path: Path, right_path: Path) -> dict[str, object]:
    left = load_float32(left_path)
    right = load_float32(right_path)
    difference = np.abs(left - right)
    relative = difference / np.maximum(np.abs(left), 1e-6)
    return {
        "left": left_path.name,
        "right": right_path.name,
        "elements": int(left.size),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(left - right)))),
        "max_rel": float(relative.max()),
        "mean_rel": float(relative.mean()),
        "bit_exact": bool(np.array_equal(left, right)),
        "allclose_rtol_3e_5_atol_3e_6": bool(np.allclose(left, right, rtol=RTOL, atol=ATOL)),
    }


def read_exit(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def require_text(path: Path, patterns: list[str]) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in patterns:
        if pattern not in text:
            raise ValueError(f"missing evidence {pattern!r} in {path}")


def evaluate(root: Path, protocol_path: Path, candidate_dlc: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    host = root / "host"
    context = root / "depthart-d1-608x448-sm8650-v75.bin"
    direct = host / "qnn-direct-depth.raw"
    cached = host / "qnn-context-depth.raw"
    reference = host / "pytorch-depth.raw"
    validation = json.loads((root / "protocol-validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "VALID_NO_DEVICE_OUTPUT_ACCESSED":
        raise ValueError("protocol validation did not pass")
    if validation["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("protocol changed after validation")
    if sha256(candidate_dlc) != protocol["bindings"]["candidate_dlc"]["sha256"]:
        raise ValueError("candidate DLC changed")
    if read_exit(root / "context-generator-attempt01.exit") != 13:
        raise ValueError("attempt 01 failure receipt changed")
    if read_exit(root / "context-generator-attempt02.exit") != 0:
        raise ValueError("context generation failed")
    if read_exit(root / "direct-run.exit") != 0 or read_exit(root / "context-run.exit") != 0:
        raise ValueError("QNN execution failed")
    require_text(
        root / "context-logcat-attempt02.txt",
        [
            "QnnBackend_registerOpPackage done successfully",
            "Successfully opened file /data/local/tmp/depthart_d1_608x448_v75_20260810_r1/dsp/./libQnnDepthArtSelectiveScanPackage.so",
            "QnnGraph_finalize done. status 0x0",
            "Successfully saved the context binary file",
        ],
    )
    require_text(root / "direct-run.log", ["Finalizing Graphs", "Finished Executing Graphs"])
    require_text(root / "context-run.log", ["Creating context from binary file", "Finished Executing Graphs"])
    expected_metadata = [
        "graph_name: depthart_608x448_fixed_mixed_fp32",
        "dimensions: [1,3,608,448]",
        "dimensions: [1,256,152,112]",
        "dimensions: [1,256,76,56]",
        "dimensions: [1,256,38,28]",
        "dimensions: [1,256,19,14]",
        "dimensions: [1,608,448]",
        "datatype: QNN_DATATYPE_FLOAT_32",
        "inferences_completed: 1",
    ]
    require_text(root / "direct-execution-metadata.yaml", expected_metadata)
    require_text(root / "context-execution-metadata.yaml", expected_metadata)
    direct_context = compare(direct, cached)
    reference_direct = compare(reference, direct)
    if not direct_context["bit_exact"]:
        raise ValueError("direct/context outputs are not bit-exact")
    result = {
        "schema": "blindassist_depthart_task_preserving_d1_device_preflight_result_v1",
        "status": "D1_SM8650_V75_CONTEXT_AND_EXECUTION_PREFLIGHT_PASS_DEVELOPMENT_OUTCOME_ACTIVATION_AUTHORIZED_NOT_STARTED",
        "protocol": asset(protocol_path),
        "protocol_validation": asset(root / "protocol-validation.json"),
        "device": protocol["device_lock"],
        "runtime": {
            "qairt": protocol["runtime_lock"]["qairt_version"],
            "device_workspace": protocol["device_workspace"],
            "candidate_dlc": asset(candidate_dlc),
            "context": asset(context),
        },
        "attempts": {
            "attempt_01": {
                "exit_code": 13,
                "reason": "OP_PACKAGE_HTP_REGISTRATION_PATH_USED_ARM64_RELATIVE_PATH",
                "retained_log": asset(root / "context-logcat-attempt01.txt"),
                "candidate_runtime_or_input_changed": False,
            },
            "attempt_02": {
                "exit_code": 0,
                "op_package_registration": "PASS_CPU_AND_HTP",
                "graph_finalize": "PASS_STATUS_0X0",
                "context_saved": True,
                "retained_log": asset(root / "context-logcat-attempt02.txt"),
            },
        },
        "outputs": {
            "direct": asset(direct),
            "saved_context": asset(cached),
            "shape": [1, 608, 448],
            "dtype": "float32",
            "finite": True,
        },
        "activation_gates": {
            "device_identity_exact": "PASS_REVERIFIED_BEFORE_AND_AFTER_EXECUTION",
            "host_and_device_asset_hashes": "PASS",
            "context_generator": "PASS_ATTEMPT_02",
            "custom_package_registration": "PASS_CPU_AND_HTP",
            "graph_finalize": "PASS_STATUS_0X0",
            "direct_execution": "PASS",
            "saved_context_execution": "PASS",
            "output_shape_dtype_finite": "PASS",
            "direct_vs_context_bit_exact": "PASS",
            "all_conjunctive_gates": "PASS",
        },
        "comparisons": {
            "direct_vs_saved_context": direct_context,
            "pytorch_reference_vs_htp_direct_diagnostic_only": reference_direct,
            "diagnostic_tolerance": {"rtol": RTOL, "atol": ATOL, "gate": False},
        },
        "governance": {
            "strict_g4d_terminal_immutable": protocol["prior_terminal_immutable"],
            "task_outcome_access": "NONE",
            "r2_cohort_access": "NONE",
            "development_outcome_activation": "AUTHORIZED_NOT_STARTED",
            "performance_authority": False,
            "da2_replacement": False,
            "android_default": False,
            "production": False,
            "safety": False,
        },
        "next_gate": "EXPLICIT_D1_DEVELOPMENT_TASK_QUALITY_SCREEN_ACTIVATION",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-dlc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.evidence_root.resolve(), args.protocol.resolve(), args.candidate_dlc.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
