#!/usr/bin/env python3
"""Validate the frozen DepthART D0 three-arm contract without running models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d0_precision_screen_protocol_v1"
SOURCE_SCHEMA = "blindassist_depthart_task_preserving_d0_source_control_lock_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN"
PRIOR_TERMINAL = (
    "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED"
)
EXPECTED_ARMS = {
    "D0_FP16_R0": ("FP16", None),
    "D0_W8A16_R0": ("W8A16", ("8", "16")),
    "D0_INT8_R0": ("INT8", ("8", "8")),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    _require(index + 1 < len(args), f"missing value after {name}")
    return args[index + 1]


def validate_contract(protocol: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema mismatch")
    _require(source.get("schema") == SOURCE_SCHEMA, "source lock schema mismatch")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id mismatch")
    _require(source.get("protocol_id") == PROTOCOL_ID, "source lock protocol id mismatch")
    _require(protocol.get("immutable_prior_terminal") == PRIOR_TERMINAL,
             "strict G4-D terminal changed")
    _require(source.get("prior_terminal_immutable") == PRIOR_TERMINAL,
             "source lock strict G4-D terminal changed")
    _require(protocol.get("data_role") ==
             "DEVELOPMENT_ONLY_CONSUMED_DATA_ALLOWED_R2_COHORT_FORBIDDEN",
             "D0 data-role boundary changed")
    _require(protocol.get("common", {}).get("strict_repair_custom_families_allowed") is False,
             "strict-repair custom families must remain excluded")
    _require(source.get("op_package", {}).get("declared_tensor_dtype") ==
             "QNN_DATATYPE_FLOAT_32", "custom-island dtype changed")

    arms = protocol.get("arms")
    _require(isinstance(arms, list) and len(arms) == 3, "exactly three D0 arms required")
    _require({arm.get("arm_id") for arm in arms} == set(EXPECTED_ARMS),
             "D0 arm identity drift")
    checked: list[str] = []
    for arm in arms:
        arm_id = arm["arm_id"]
        family, quant = EXPECTED_ARMS[arm_id]
        _require(arm.get("representation_family") == family,
                 f"{arm_id} representation drift")
        converter_args = arm.get("converter_args")
        _require(isinstance(converter_args, list), f"{arm_id} converter args missing")
        _require(_option(converter_args, "--target_backend") == "HTP",
                 f"{arm_id} backend must be HTP")
        quantizer_args = arm.get("quantizer_args")
        if quant is None:
            _require(quantizer_args is None, "FP16 must not run the integer quantizer")
            _require(_option(converter_args, "--float_bitwidth") == "16",
                     "FP16 converter bitwidth drift")
        else:
            _require(isinstance(quantizer_args, list), f"{arm_id} quantizer args missing")
            _require(_option(quantizer_args, "--weights_bitwidth") == quant[0],
                     f"{arm_id} weight bitwidth drift")
            _require(_option(quantizer_args, "--act_bitwidth") == quant[1],
                     f"{arm_id} activation bitwidth drift")
            _require("--use_per_channel_quantization" in quantizer_args,
                     f"{arm_id} per-channel weight quantization missing")
        checked.append(arm_id)

    _require(protocol.get("selection", {}).get("candidate_count_after_screen") == 1,
             "D0 must select exactly one R2 candidate")
    _require(protocol.get("selection", {}).get("r2_cohort_access_during_selection") is False,
             "R2 cohort cannot be used for D0 selection")
    _require(protocol.get("target", {}).get("cpu_fallback_allowed") is False,
             "CPU fallback must remain disabled")
    return {
        "contract_valid": True,
        "arms": checked,
        "strict_g4d_terminal_immutable": True,
        "r2_cohort_excluded_from_selection": True,
    }


def _binding_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _verify_binding(repo_root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = _binding_path(repo_root, str(binding["path"]))
    _require(path.is_file(), f"missing {label}: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256(path)
    _require(actual_bytes == binding["bytes"], f"{label} byte-size mismatch")
    _require(actual_sha256 == str(binding["sha256"]).upper(), f"{label} SHA-256 mismatch")
    return {"path": str(path.resolve()), "bytes": actual_bytes, "sha256": actual_sha256}


def verify_bindings(
    repo_root: Path, protocol: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    bindings = {
        "qairt_converter": protocol["toolchain"]["converter"],
        "qairt_quantizer": protocol["toolchain"]["quantizer"],
        "reference_checkpoint": source["reference"]["checkpoint"],
        "canonical_onnx": source["reference"]["canonical_onnx"],
        "common_candidate_onnx": source["common_candidate_source"]["onnx"],
        "diagnostic_control_dlc": source["common_candidate_source"]["diagnostic_control_dlc"],
        "diagnostic_control_context": source["common_candidate_source"]["diagnostic_control_context"],
        "converter_extension": source["op_package"]["converter_extension"],
        "op_package_definition": source["op_package"]["definition"],
        "op_package_aarch64": source["op_package"]["aarch64"],
        "op_package_v75": source["op_package"]["v75"],
    }
    return {
        name: _verify_binding(repo_root, binding, name)
        for name, binding in bindings.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = _load(args.protocol)
    source = _load(args.source_lock)
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d0_preflight_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PREFLIGHT_PASS_IMPLEMENTATION_NOT_EXECUTED",
        "contract": validate_contract(protocol, source),
        "bindings": verify_bindings(args.repo_root.resolve(), protocol, source),
        "identities": {
            "protocol_sha256": _sha256(args.protocol),
            "source_lock_sha256": _sha256(args.source_lock),
        },
        "execution_authorized": False,
        "outcome_access": "NONE",
        "authority": "Static D0 recipe and source-binding preflight only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
