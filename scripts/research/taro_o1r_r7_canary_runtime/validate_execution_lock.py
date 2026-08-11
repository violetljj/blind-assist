from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.taro_o1r_r7_canary_runtime.validate_implementation_lock import validate as validate_implementation


DEFAULT_LOCK = Path("docs/research/taro/TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json")
EXPECTED_BYTES = 5174
EXPECTED_SHA256 = "08565AEFA686B057CAA5D8DF8EFAE39DB17F739EFFCF5A7652CA0647BBDC2BBF"
EXPECTED_OUTPUT = Path("E:/linnan/linnan/artifacts.local/evidence/taro/o1r-r7-positive-occupancy-clear-coverage-fit-canary-r0")


def _binding_errors(root: Path, name: str, binding: dict[str, Any]) -> list[str]:
    path = (root / binding["path"]).resolve()
    if not path.is_file():
        return [f"{name}: missing {binding['path']}"]
    raw = path.read_bytes()
    errors = []
    if len(raw) != binding["bytes"]:
        errors.append(f"{name}: byte count mismatch")
    if hashlib.sha256(raw).hexdigest().upper() != binding["sha256"]:
        errors.append(f"{name}: SHA-256 mismatch")
    return errors


def validate(lock_path: Path = DEFAULT_LOCK, *, require_output_absent: bool = True) -> dict[str, Any]:
    path = lock_path.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    errors: list[str] = []
    raw = path.read_bytes()
    if len(raw) != EXPECTED_BYTES or hashlib.sha256(raw).hexdigest().upper() != EXPECTED_SHA256:
        errors.append("execution lock identity mismatch")
    lock = json.loads(raw.decode("utf-8"))
    if lock.get("status") != "FROZEN" or lock.get("lock_id") != "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_ONE_SHOT_EXECUTION_LOCK":
        errors.append("execution lock status/id drift")
    errors.extend(_binding_errors(repo_root, "implementation_lock", lock["implementation_lock"]))
    errors.extend(_binding_errors(repo_root, "runner", lock["runner"]))
    for name, binding in lock.get("input_bindings", {}).items():
        errors.extend(_binding_errors(repo_root, name, binding))
    implementation = validate_implementation(repo_root / lock["implementation_lock"]["path"])
    if not implementation["passed"]:
        errors.append("implementation lock validation failed")
    expected_argv = [
        "E:/codex-tools/venvs/riskseg-r0-py311/Scripts/python.exe",
        "-m",
        "scripts.research.taro_o1r_r7_canary_runtime.run_locked_fit_canary",
        "--execution-lock",
        "E:/linnan/linnan/docs/research/taro/TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    ]
    if lock.get("argv") != expected_argv:
        errors.append("execution argv drift")
    roots = lock.get("roots", {})
    if Path(roots.get("repo_root", "")).resolve() != repo_root:
        errors.append("repo root drift")
    output = Path(roots.get("output_root", "")).resolve()
    if output != EXPECTED_OUTPUT.resolve():
        errors.append("exclusive output root drift")
    if require_output_absent and output.exists():
        errors.append("exclusive output root already exists")
    for name in ("frame_plan_path", "source_root", "source_evidence_root", "fit_candidate_root", "eval_candidate_root", "phase_a_root"):
        if not Path(roots.get(name, "")).exists():
            errors.append(f"root missing: {name}")
    authority = lock.get("authority", {})
    if (authority.get("role"), authority.get("parent_count"), authority.get("frame_count"), authority.get("query_count")) != ("ADAPTER_FIT", 8, 211, 1899):
        errors.append("R7 execution cohort drift")
    if authority.get("source_phase_allowed_payload_roles") != ["lowres_depth", "confidence"] or authority.get("source_phase_faro_reads") != 0 or authority.get("label_phase_allowed_payload_roles") != ["highres_depth"]:
        errors.append("R7 phase payload firewall drift")
    if authority.get("source_phase_must_complete_and_reload_before_label_phase") is not True or authority.get("leave_one_parent_out") is not True or authority.get("observed_eval_parent_use") is not False:
        errors.append("R7 phase/role firewall drift")
    if authority.get("training_steps") != 0 or authority.get("network_requests") != 0 or authority.get("promotion_authorized") is not False:
        errors.append("R7 execution authority drift")
    one_shot = lock.get("one_shot", {})
    if one_shot != {"consumed_on_output_root_creation": True, "output_root_must_be_absent": True, "overwrite_allowed": False, "rerun_allowed": False}:
        errors.append("R7 one-shot semantics drift")
    budget = lock.get("resource_budget", {})
    if budget.get("maximum_elapsed_seconds") != 1800 or budget.get("maximum_evidence_bytes") != 268435456 or budget.get("maximum_rss_bytes") != 4294967296:
        errors.append("R7 resource budget drift")
    return {
        "schema": "blindassist.taro.o1r.r7_canary_execution_lock_validation.v1",
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O1R_R7_CANARY_EXECUTION_LOCK_VALID" if not errors else "TARO_O1R_R7_CANARY_EXECUTION_LOCK_INVALID",
        "lock": lock,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--allow-consumed", action="store_true")
    args = parser.parse_args()
    result = validate(args.lock, require_output_absent=not args.allow_consumed)
    printable = dict(result)
    printable.pop("lock", None)
    print(json.dumps(printable, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
