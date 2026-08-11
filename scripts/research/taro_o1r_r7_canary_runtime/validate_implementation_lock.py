from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


DEFAULT_LOCK = Path("docs/research/taro/TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_IMPLEMENTATION_LOCK_2026-08-12.json")
EXPECTED_BYTES = 4140
EXPECTED_SHA256 = "C29E4CBF11A6DA9ED8969A9A64EA08D1236E72E935442038C1BDA76E9E25C663"


def _binding_errors(root: Path, name: str, binding: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = root / binding["path"]
    if not path.is_file():
        return [f"{name}: missing {binding['path']}"]
    raw = path.read_bytes()
    if len(raw) != binding["bytes"]:
        errors.append(f"{name}: byte count mismatch")
    if hashlib.sha256(raw).hexdigest().upper() != binding["sha256"]:
        errors.append(f"{name}: SHA-256 mismatch")
    return errors


def validate(lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    path = lock_path.resolve()
    root = Path(__file__).resolve().parents[3]
    errors: list[str] = []
    raw = path.read_bytes()
    if len(raw) != EXPECTED_BYTES or hashlib.sha256(raw).hexdigest().upper() != EXPECTED_SHA256:
        errors.append("implementation lock identity mismatch")
    lock = json.loads(raw.decode("utf-8"))
    if lock.get("status") != "FROZEN" or lock.get("lock_id") != "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_IMPLEMENTATION_LOCK":
        errors.append("implementation lock status/id drift")
    for name, binding in lock.get("predecessor_bindings", {}).items():
        errors.extend(_binding_errors(root, name, binding))
    for name, binding in lock.get("implementation_bindings", {}).items():
        errors.extend(_binding_errors(root, name, binding))
    algorithm = lock.get("frozen_algorithm", {})
    if algorithm.get("reducer_id") != r7_canary.REDUCER_ID or algorithm.get("selection", {}).get("candidate_count") != len(r7_canary.candidate_configs()) == 972:
        errors.append("frozen R7 algorithm identity/grid drift")
    clear = algorithm.get("clear_feature", {})
    if clear.get("total_anchor_count") != r7_canary.FAR_SAMPLE_COUNT or clear.get("minimum_visible_anchor_count") != r7_canary.MINIMUM_FAR_VISIBLE_ANCHORS:
        errors.append("frozen R7 clear anchor semantics drift")
    names = inspect.signature(r7_canary.build_source_frame_record).parameters
    if any(token in name.lower() for name in names for token in ("faro", "truth", "label", "outcome")):
        errors.append("source API exposes label-side input")
    firewalls = lock.get("firewalls", {})
    if firewalls.get("source_phase_label_or_faro_input") is not False or firewalls.get("source_phase_records_sealed_and_reloaded_before_label_join") is not True or firewalls.get("unknown_as_negative") is not False:
        errors.append("R7 implementation firewall drift")
    authority = lock.get("authority", {})
    if authority.get("implementation_frozen") is not True or authority.get("scientific_execution") is not False or authority.get("promotion_authorized") is not False:
        errors.append("R7 implementation authority drift")
    if lock.get("unique_successor") != "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_ONE_SHOT_EXECUTION_LOCK":
        errors.append("R7 implementation successor drift")
    return {
        "schema": "blindassist.taro.o1r.r7_canary_implementation_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R7_CANARY_IMPLEMENTATION_VALID" if not errors else "TARO_O1R_R7_CANARY_IMPLEMENTATION_INVALID",
        "lock": lock,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate(args.lock)
    printable = dict(result)
    printable.pop("lock", None)
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
