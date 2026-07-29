"""Assemble the fail-closed P4 result when formal manipulation fails pre-R3.

This reporting-only module grants no execution authority and never imports the
R3 pair core, formal runner, transport, or statistical analysis implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TERMINAL = "INTERVENTION_NOT_EVALUABLE"


class InvalidPreR3Terminal(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidPreR3Terminal(f"JSON_OBJECT:{path.name}")
    return value


def build_result(
    activation_path: Path,
    producer_path: Path,
    independent_path: Path,
    formal_root: Path,
) -> dict[str, Any]:
    activation = load_object(activation_path)
    producer = load_object(producer_path)
    independent = load_object(independent_path)
    if (
        activation.get("protocol_id") != PROTOCOL_ID
        or activation.get("formal_execution_authorized") is not True
        or activation.get("p4_activated") is not True
    ):
        raise InvalidPreR3Terminal("ACTIVATION")
    if (
        producer.get("protocol_id") != PROTOCOL_ID
        or producer.get("terminal") != TERMINAL
        or producer.get("r3_imported_or_executed") is not False
        or producer.get("algorithm_output_read") is not False
    ):
        raise InvalidPreR3Terminal("MANIPULATION_PRODUCER")
    if (
        independent.get("validation") != "VALID"
        or independent.get("terminal") != TERMINAL
        or independent.get("receipt_sha256") != sha256_file(producer_path)
    ):
        raise InvalidPreR3Terminal("MANIPULATION_INDEPENDENT")
    forbidden = [
        formal_root / "run" / "success.json",
        formal_root / "run" / "claim.json",
        formal_root / "run" / "arms",
        formal_root / "formal_result.json",
    ]
    present = [path.as_posix() for path in forbidden if path.exists()]
    if present:
        raise InvalidPreR3Terminal("R3_FORMAL_OUTPUT_PRESENT")
    failed = [
        row for row in producer.get("subgroups", [])
        if not row.get("blur_subgroup_pass")
        or not row.get("low_texture_subgroup_pass")
    ]
    if not failed:
        raise InvalidPreR3Terminal("NO_FAILED_MANIPULATION_SUBGROUP")
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p4_pre_r3_terminal_result.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "phase": "P4_A_RESPONSE_BLIND_FORMAL_MANIPULATION",
        "execution_state": "COMPLETE_PRE_R3_TERMINAL",
        "scientific_terminal": TERMINAL,
        "terminal_precedence_applied": True,
        "analysis_performed": False,
        "formal_r3_execution": {
            "authorized_then_blocked_by_frozen_prerequisite": True,
            "arms_started": 0,
            "arms_completed": 0,
            "main_pair_core_calls": 0,
            "guard_pair_core_calls": 0,
            "reason": "FORMAL_MAIN_MANIPULATION_BLOCK_GATE_FAILED",
        },
        "manipulation": {
            "sequence_checks": producer["counts"]["sequence_checks"],
            "frame_states": producer["counts"]["total_frame_states"],
            "failed_subgroups": failed,
        },
        "bindings": {
            "activation_lock_sha256": sha256_file(activation_path),
            "manipulation_producer_receipt_sha256": sha256_file(producer_path),
            "manipulation_independent_receipt_sha256": sha256_file(
                independent_path
            ),
            "reporting_implementation_sha256": sha256_file(Path(__file__)),
        },
        "firewall": {
            "r3_imported_or_executed": False,
            "algorithm_output_read": False,
            "strength_retuned": False,
            "seed_replaced": False,
            "threshold_or_three_pair_modified": False,
            "sequence16_android_realtime": False,
            "p4_activation_consumed": True,
        },
        "authority": (
            "SYNTHETIC_DEVELOPMENT_MECHANISM_PREREQUISITE_TERMINAL_ONLY"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = build_result(
        args.activation.resolve(),
        args.producer_receipt.resolve(),
        args.independent_receipt.resolve(),
        args.formal_root.resolve(),
    )
    write_exclusive(args.result.resolve(), result)
    print(
        json.dumps(
            {
                "execution_state": result["execution_state"],
                "scientific_terminal": result["scientific_terminal"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
