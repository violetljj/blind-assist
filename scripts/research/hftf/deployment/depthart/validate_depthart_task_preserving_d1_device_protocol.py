#!/usr/bin/env python3
"""Validate the frozen D1 SM8650/v75 activation protocol without device execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def check_asset(repo: Path, asset: dict[str, object]) -> None:
    path = resolve(repo, str(asset["path"]))
    if not path.is_file():
        raise ValueError(f"missing locked asset: {path}")
    if "bytes" in asset and path.stat().st_size != int(asset["bytes"]):
        raise ValueError(f"byte mismatch: {path}")
    if sha256(path) != str(asset["sha256"]):
        raise ValueError(f"SHA-256 mismatch: {path}")


def validate(repo: Path, protocol_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_EXECUTION_AUTHORIZED_NO_DEVICE_OUTPUT_ACCESSED":
        raise ValueError("unexpected protocol status")
    if protocol["prior_terminal_immutable"] != (
        "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED"
    ):
        raise ValueError("strict G4-D terminal changed")
    device = protocol["device_lock"]
    expected_device = {
        "serial": "R5CX10M8Y8X",
        "model": "SM-S9280",
        "device": "e3q",
        "soc": "SM8650",
        "abi": "arm64-v8a",
        "htp_arch": "v75",
        "android_release": "16",
        "sdk": 36,
        "build_fingerprint": "samsung/e3qzcx/e3q:16/BP4A.251205.006/S9280ZCS6DZG1:user/release-keys",
        "security_patch": "2026-07-05",
    }
    for key, expected in expected_device.items():
        if device.get(key) != expected:
            raise ValueError(f"device lock mismatch: {key}")
    bindings = protocol["bindings"]
    for name in ("d1_development_protocol", "product_aspect_technical_lock", "candidate_onnx", "candidate_dlc"):
        check_asset(repo, bindings[name])
    canary_binding = bindings["synthetic_canary"]
    check_asset(repo, canary_binding)
    check_asset(
        repo,
        {
            "path": canary_binding["generator_path"],
            "sha256": canary_binding["generator_sha256"],
        },
    )
    canary = json.loads(resolve(repo, str(canary_binding["path"])).read_text(encoding="utf-8"))
    if canary.get("height") != 608 or canary.get("width") != 448:
        raise ValueError("canary geometry mismatch")
    if canary["files"]["image"]["shape"] != [1, 3, 608, 448]:
        raise ValueError("canary image shape mismatch")
    if canary["files"]["depth"]["shape"] != [1, 608, 448]:
        raise ValueError("canary depth shape mismatch")
    for asset in protocol["runtime_lock"]["assets"]:
        check_asset(repo, asset)
    gates = protocol["activation_gates"]
    if gates["direct_and_context_output_elements"] != 608 * 448:
        raise ValueError("output element gate mismatch")
    if not gates["direct_vs_context_bit_exact"] or not gates["all_gates_conjunctive"]:
        raise ValueError("activation gates are not fail-closed")
    if protocol["reference_diagnostic"]["gate"] is not False:
        raise ValueError("raw-depth diagnostic illegally became a D1 gate")
    return {
        "schema": "blindassist_depthart_task_preserving_d1_device_protocol_validation_v1",
        "status": "VALID_NO_DEVICE_OUTPUT_ACCESSED",
        "protocol_sha256": sha256(protocol_path),
        "locked_runtime_asset_count": len(protocol["runtime_lock"]["assets"]),
        "candidate_dlc_sha256": bindings["candidate_dlc"]["sha256"],
        "canary_sha256": canary_binding["sha256"],
        "task_outcome_access": "NONE",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.repo_root.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
