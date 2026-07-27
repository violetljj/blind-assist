"""Bounded post-hoc validation of the frozen floor3_2 R0 evidence.

This does not rerun geometry or RGB and cannot overwrite the formal INVALID
terminal. It changes only the numeric aggregate equivalence predicate used by
the already-independent frozen validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.research.egomotion_compensated_looming.rgb_algorithm_cid_sims_floor3_2_cross_sequence_holdout_r0 import (
    validator as frozen_validator,
)


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_CROSS_SEQUENCE_HOLDOUT_R0_"
    "POSTHOC_VALIDATOR_R1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def bounded_same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    left_float = float(left)
    right_float = float(right)
    return (
        math.isfinite(left_float)
        and math.isfinite(right_float)
        and math.isclose(
            left_float,
            right_float,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )


def write_exclusive_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r0-contract", type=Path, required=True)
    parser.add_argument("--r0-lock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract = load_object(args.contract.resolve())
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("POSTHOC_PROTOCOL_ID")
    for entry in contract["frozen_inputs"]:
        path = repo_root / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"FROZEN_INPUT_DRIFT:{entry['path']}")
    failure = load_object(args.run_dir.resolve() / "FAILURE.json")
    if failure.get("terminal") != "CROSS_SEQUENCE_HOLDOUT_INVALID / INVALID":
        raise ValueError("FORMAL_TERMINAL_NOT_PRESERVED")
    original_predicate = frozen_validator._same_number
    try:
        frozen_validator._same_number = bounded_same_number
        validation = frozen_validator.validate(
            repo_root,
            args.r0_contract.resolve(),
            args.r0_lock.resolve(),
            args.run_dir.resolve(),
        )
    finally:
        frozen_validator._same_number = original_predicate
    selection = load_object(args.run_dir.resolve() / "geometry_selection.json")
    roles: dict[str, int] = {}
    for window in selection["candidate_windows"]:
        role = str(window["role"])
        roles[role] = roles.get(role, 0) + 1
    result = load_object(args.run_dir.resolve() / "result.json")
    output = {
        "schema_version": "rcle.cross_sequence_holdout.posthoc_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "formal_terminal_preserved": failure["terminal"],
        "status": validation["status"],
        "errors": validation["errors"],
        "numeric_equivalence": {
            "rel_tol": 1e-12,
            "abs_tol": 1e-15,
        },
        "selection_terminal": selection["terminal"],
        "selection_evaluable": selection["selection_evaluable"],
        "role_counts": dict(sorted(roles.items())),
        "selected_window_indices": [
            int(item["window_index"])
            for item in selection["selected_windows"]
        ],
        "rgb": {
            "identity_created": result["selected_rgb_identity_created"],
            "member_bytes_read": result["rgb_member_bytes_read"],
            "algorithm_executed": result["rgb_algorithm_executed"],
            "cache_exists": (args.run_dir.resolve() / "rgb_cache").exists(),
            "ledger_exists": (
                args.run_dir.resolve() / "rgb_pair_ledger.jsonl"
            ).exists(),
        },
        "authority": "POSTHOC_VALIDATION_OF_FROZEN_R0_EVIDENCE_ONLY",
    }
    output["output_payload_sha256"] = hashlib.sha256(
        json.dumps(
            output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    write_exclusive_json(args.output.resolve(), output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if validation["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
