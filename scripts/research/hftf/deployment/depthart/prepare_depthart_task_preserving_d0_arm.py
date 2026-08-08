#!/usr/bin/env python3
"""Materialize one frozen DepthART D0 precision arm into a fresh evidence root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d0_preflight import (
    PROTOCOL_ID,
    validate_contract,
    verify_bindings,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def qairt_environment(qairt_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    bin_path = qairt_root / "bin" / "x86_64-windows-msvc"
    lib_path = qairt_root / "lib" / "x86_64-windows-msvc"
    python_path = qairt_root / "lib" / "python"
    common_path = python_path / "qti" / "aisw" / "converters" / "common" / "windows-x86_64"
    environment["QAIRT_SDK_ROOT"] = str(qairt_root)
    environment["QNN_SDK_ROOT"] = str(qairt_root)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(python_path), str(common_path)))
    environment["PATH"] = os.pathsep.join(
        (str(bin_path), str(lib_path), environment.get("PATH", ""))
    )
    return environment


def build_command(
    python: Path,
    protocol: dict[str, Any],
    source: dict[str, Any],
    repo_root: Path,
    arm_id: str,
    output_dlc: Path,
    calibration_list: Path | None,
) -> list[str]:
    arm = next((item for item in protocol["arms"] if item["arm_id"] == arm_id), None)
    if arm is None:
        raise ValueError(f"unknown frozen arm: {arm_id}")
    toolchain = protocol["toolchain"]
    if arm["quantizer_args"] is None:
        if calibration_list is not None:
            raise ValueError("FP16 arm must not receive a calibration list")
        return [
            str(python),
            toolchain["converter"]["path"],
            "--input_network",
            str(_resolve(repo_root, source["common_candidate_source"]["onnx"]["path"])),
            "--output_path",
            str(output_dlc),
            "--op_package_config",
            str(_resolve(repo_root, source["op_package"]["definition"]["path"])),
            "--converter_op_package_lib",
            str(_resolve(repo_root, source["op_package"]["converter_extension"]["path"])),
            *arm["converter_args"],
        ]
    if calibration_list is None or not calibration_list.is_file():
        raise ValueError(f"{arm_id} requires a frozen calibration list")
    return [
        str(python),
        toolchain["quantizer"]["path"],
        "--input_dlc",
        str(_resolve(repo_root, source["common_candidate_source"]["diagnostic_control_dlc"]["path"])),
        "--output_dlc",
        str(output_dlc),
        "--input_list",
        str(calibration_list.resolve()),
        *arm["quantizer_args"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--qairt-root", required=True, type=Path)
    parser.add_argument("--qairt-python", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("D0_FP16_R0", "D0_W8A16_R0", "D0_INT8_R0"))
    parser.add_argument("--calibration-list", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    protocol = _load(args.protocol)
    source = _load(args.source_lock)
    validate_contract(protocol, source)
    verify_bindings(repo_root, protocol, source)
    artifacts_root = (repo_root / "artifacts.local").resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    if artifacts_root not in output_root.parents:
        raise ValueError(f"output root must be below {artifacts_root}")
    if not args.qairt_python.is_file():
        raise FileNotFoundError(args.qairt_python)
    output_root.mkdir(parents=True)
    output_dlc = output_root / f"{args.arm.lower().replace('_', '-')}.dlc"
    command = build_command(
        args.qairt_python.resolve(), protocol, source, repo_root, args.arm, output_dlc,
        args.calibration_list,
    )
    result = subprocess.run(
        command,
        env=qairt_environment(args.qairt_root.resolve()),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path = output_root / "tool.log"
    log_path.write_text(
        json.dumps({"command": command, "returncode": result.returncode}, indent=2)
        + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    receipt: dict[str, Any] = {
        "schema": "blindassist_depthart_task_preserving_d0_arm_materialization_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "arm_id": args.arm,
        "status": "DLC_MATERIALIZED_CONTEXT_NOT_EVALUATED" if result.returncode == 0 else "CONVERSION_OR_QUANTIZATION_FAIL",
        "returncode": result.returncode,
        "identities": {
            "protocol_sha256": _sha256(args.protocol),
            "source_lock_sha256": _sha256(args.source_lock),
            "tool_log_sha256": _sha256(log_path),
            "calibration_list_sha256": _sha256(args.calibration_list) if args.calibration_list else None,
        },
        "dlc": (
            {"path": str(output_dlc), "bytes": output_dlc.stat().st_size, "sha256": _sha256(output_dlc)}
            if output_dlc.is_file() else None
        ),
        "downstream": "DEVICE_CONTEXT_PREFLIGHT_REQUIRED" if result.returncode == 0 else "ARM_TECHNICALLY_INELIGIBLE_UNDER_FROZEN_RECIPE",
        "authority": "D0 materialization only; no task quality, R2, performance, DA2 replacement, production or safety authority.",
    }
    (output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0 if result.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
