"""Independent validator for the pre-R3 P4 manipulation-failure terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TERMINAL = "INTERVENTION_NOT_EVALUABLE"
PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"OBJECT:{path.name}")
    return value


def validate(
    activation_path: Path,
    producer_path: Path,
    manipulation_validation_path: Path,
    result_path: Path,
    formal_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    activation = _object(activation_path)
    producer = _object(producer_path)
    manipulation_validation = _object(manipulation_validation_path)
    result = _object(result_path)
    if (
        activation.get("protocol_id") != PROTOCOL_ID
        or activation.get("formal_execution_authorized") is not True
        or activation.get("p4_activated") is not True
    ):
        errors.append("ACTIVATION")
    if (
        producer.get("terminal") != TERMINAL
        or producer.get("r3_imported_or_executed") is not False
        or producer.get("algorithm_output_read") is not False
    ):
        errors.append("PRODUCER_TERMINAL_OR_FIREWALL")
    if (
        manipulation_validation.get("validation") != "VALID"
        or manipulation_validation.get("terminal") != TERMINAL
        or manipulation_validation.get("receipt_sha256") != _sha(producer_path)
    ):
        errors.append("MANIPULATION_VALIDATION")
    failed = [
        row for row in producer.get("subgroups", [])
        if not row.get("blur_subgroup_pass")
        or not row.get("low_texture_subgroup_pass")
    ]
    if not failed:
        errors.append("FAILED_SUBGROUP_ABSENT")
    if result.get("scientific_terminal") != TERMINAL:
        errors.append("RESULT_TERMINAL")
    if result.get("execution_state") != "COMPLETE_PRE_R3_TERMINAL":
        errors.append("RESULT_EXECUTION_STATE")
    if result.get("analysis_performed") is not False:
        errors.append("RESULT_ANALYSIS_FIREWALL")
    execution = result.get("formal_r3_execution", {})
    if any(
        execution.get(field) != 0
        for field in (
            "arms_started",
            "arms_completed",
            "main_pair_core_calls",
            "guard_pair_core_calls",
        )
    ):
        errors.append("RESULT_R3_COUNTS")
    if result.get("manipulation", {}).get("failed_subgroups") != failed:
        errors.append("FAILED_SUBGROUP_SUMMARY")
    bindings = result.get("bindings", {})
    expected_bindings = {
        "activation_lock_sha256": _sha(activation_path),
        "manipulation_producer_receipt_sha256": _sha(producer_path),
        "manipulation_independent_receipt_sha256": _sha(
            manipulation_validation_path
        ),
    }
    for field, expected in expected_bindings.items():
        if bindings.get(field) != expected:
            errors.append(f"RESULT_BINDING:{field}")
    forbidden = [
        formal_root / "run" / "success.json",
        formal_root / "run" / "claim.json",
        formal_root / "run" / "arms",
    ]
    if any(path.exists() for path in forbidden):
        errors.append("FORMAL_R3_OUTPUT_PRESENT")
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_pre_r3_terminal_independent_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "validated": not errors,
        "terminal": (
            "P4_PRE_R3_RESULT_VALID / INTERVENTION_NOT_EVALUABLE"
            if not errors
            else "P4_PRE_R3_RESULT_INVALID"
        ),
        "scientific_terminal": TERMINAL if not errors else None,
        "result_sha256": _sha(result_path),
        "activation_sha256": _sha(activation_path),
        "manipulation_producer_receipt_sha256": _sha(producer_path),
        "manipulation_independent_receipt_sha256": _sha(
            manipulation_validation_path
        ),
        "validator_sha256": _sha(Path(__file__)),
        "independence": {
            "result_assembler_imported": False,
            "formal_runner_imported": False,
            "r3_pair_core_imported": False,
            "statistical_analysis_imported": False,
        },
        "errors": errors,
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(
            (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--manipulation-receipt", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(
        args.activation.resolve(),
        args.producer_receipt.resolve(),
        args.manipulation_receipt.resolve(),
        args.result.resolve(),
        args.formal_root.resolve(),
    )
    _write_exclusive(args.receipt.resolve(), receipt)
    print(
        json.dumps(
            {
                "validated": receipt["validated"],
                "terminal": receipt["terminal"],
                "errors": receipt["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
