#!/usr/bin/env python3
"""Validate the frozen TARO R6 untouched confirmation executor lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import r6_factor_split


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r6_untouched_confirmation_executor_implementation_lock.v1"
LOCK_ID = "TARO_O0R_R6_UNTOUCHED_CONFIRMATION_EXECUTOR_IMPLEMENTATION_LOCK"
SUCCESSOR = "TARO_O0R_R6_UNTOUCHED_CONFIRMATION_ONE_SHOT_EXECUTION_LOCK"
EXPECTED_PREDECESSORS = {
    "R6_FACTOR_COMPOSITOR_IMPLEMENTATION_LOCK": ("docs/research/taro/TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK_2026-08-11.json", 5026, "34D1C30193183F8406D5A4CA5EF7598E7EE933B4D62008500A70523A5EE3C90B"),
    "R6_UNTOUCHED_DATA_LOCK": ("docs/research/taro/TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK_2026-08-11.json", 7043, "DEC4D54487C8A321017EAC615526242BF1A31D21CE5FE5E4333C1D252AA5FA83"),
    "R6_EXACT_FRAME_INVENTORY": ("artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0/exact-frame-plan.json", 12340, "69352D6A940111E738488AA25CFAB8A924658B8C5720D9CE7A50AC612558D6A8"),
    "R6_INVENTORY_RESULT": ("artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0/result.json", 454, "DB8144DB3114CC379744AE30D26DAB7EC9AC0EA56DE017B539585B16A7ACD341"),
    "R6_INVENTORY_MANIFEST": ("artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0/manifest.json", 695, "7F889A97618653B886525DD70BCA354C0DACE46F1939F6AB7540A8D6E34C9F19"),
}
EXPECTED_IMPLEMENTATION = {
    "R6_CONFIRMATION_CORE": ("scripts/research/taro_o0r_candidate_scale_runtime/r6_confirmation.py", 60703, "19EA4752CE42DCB7FB8384F0965894CD32C5776EE8757801993883DE996205D4"),
    "R6_CONFIRMATION_IO": ("scripts/research/taro_o0r_candidate_scale_runtime/r6_confirmation_io.py", 12915, "E924D49A0A11ADAA52B0BD3F529EF70DD43ABFB6879F7DAFA460D63E89437DC3"),
    "R6_CONFIRMATION_RUNNER": ("scripts/research/taro_o0r_candidate_scale_runtime/run_r6_untouched_confirmation.py", 16730, "648B17ADCCA0D3E56F070DB5AE5C6D85C6916268E9E330D1738221526E47023C"),
    "R6_CONFIRMATION_CORE_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r6_confirmation.py", 10084, "F17961EB3A3A01765F849D6BE5A398975D348222D135D62F6F36148B4894D8B4"),
    "R6_CONFIRMATION_RUNNER_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r6_confirmation_runner.py", 2408, "3FAC7633CD49729805E03358EB2A308E20C528D078B7D3FF0E02895F3146551B"),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _bindings(value: Any, errors: list[str], prefix: str) -> dict[str, tuple[Any, Any, Any]]:
    result = {}
    if not isinstance(value, list):
        errors.append(f"{prefix}_NOT_LIST")
        return result
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "bytes", "sha256"} or row.get("role") in result:
            errors.append(f"{prefix}_ROW_INVALID")
            continue
        result[str(row["role"])] = (row["path"], row["bytes"], row["sha256"])
    return result


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected_keys = {"schema", "lock_id", "date", "research_mode", "status", "predecessor_bindings", "exact_cohort", "frozen_algorithm", "phase_firewall", "implementation_bindings", "test_receipt", "execution_authority", "unique_successor", "claim_ceiling"}
    _require(errors, set(payload) == expected_keys, "R6_CONFIRM_IMPL_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA and payload.get("lock_id") == LOCK_ID and payload.get("status") == "EXECUTOR_IMPLEMENTATION_FROZEN_EXECUTION_FALSE", "R6_CONFIRM_IMPL_IDENTITY_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R6_CONFIRM_IMPL_SUCCESSOR_DRIFT")
    predecessors = _bindings(payload.get("predecessor_bindings"), errors, "R6_CONFIRM_IMPL_PREDECESSOR_BINDINGS")
    implementation = _bindings(payload.get("implementation_bindings"), errors, "R6_CONFIRM_IMPL_IMPLEMENTATION_BINDINGS")
    _require(errors, predecessors == EXPECTED_PREDECESSORS, "R6_CONFIRM_IMPL_PREDECESSOR_SET_DRIFT")
    _require(errors, implementation == EXPECTED_IMPLEMENTATION, "R6_CONFIRM_IMPL_IMPLEMENTATION_SET_DRIFT")

    cohort = payload.get("exact_cohort", {})
    expected_roster = [{"visit_id": parent, "video_id": video, "frame_count": count} for (parent, video), count in zip(r6.ROSTER, r6.EXPECTED_PARENT_FRAME_COUNTS)]
    _require(errors, cohort == {"roster": expected_roster, "parent_count": 8, "physical_frame_count": 120, "query_slot_count": 1080, "formation_parent_overlap_count": 0}, "R6_CONFIRM_IMPL_COHORT_DRIFT")

    algorithm = payload.get("frozen_algorithm", {})
    _require(errors, algorithm.get("analysis_role") == r6.ANALYSIS_ROLE and algorithm.get("policy_id") == r6_factor_split.POLICY_ID, "R6_CONFIRM_IMPL_POLICY_DRIFT")
    _require(errors, algorithm.get("candidate_model_id") == "depthart-s-metric-indoor-448-official-fp32" and algorithm.get("checkpoint_sha256") == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65", "R6_CONFIRM_IMPL_CANDIDATE_IDENTITY_DRIFT")
    _require(errors, algorithm.get("preprocess_id") == "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1" and algorithm.get("postprocess_id") == "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1", "R6_CONFIRM_IMPL_TRANSFORM_DRIFT")
    _require(errors, algorithm.get("support_owner") == algorithm.get("boundary_owner") == "PHASE_A_SELECTED_SUPPORT_BOUNDARY_COMPONENT" and algorithm.get("query_clearance_owner") == "ALWAYS_R1_BASELINE", "R6_CONFIRM_IMPL_FACTOR_OWNER_DRIFT")
    _require(errors, algorithm.get("source_support_membership_uses_candidate_depth") is False and algorithm.get("learned_parameters") == algorithm.get("training_steps") == algorithm.get("threshold_changes") == 0, "R6_CONFIRM_IMPL_FITTING_DRIFT")

    firewall = payload.get("phase_firewall", {})
    _require(errors, firewall.get("phase_a_payload_allowlist") == list(r6.PHASE_A_ASSET_ROLES) and firewall.get("candidate_model_inputs") == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"], "R6_CONFIRM_IMPL_PHASE_A_ALLOWLIST_DRIFT")
    _require(errors, firewall.get("all_120_candidates_before_source_decisions") is True and firewall.get("all_120_source_decisions_before_faro") is True and firewall.get("phase_a_completion_reload_before_first_faro_read") is True, "R6_CONFIRM_IMPL_PHASE_ORDER_DRIFT")
    _require(errors, firewall.get("phase_b_payload_allowlist") == ["highres_depth"] and firewall.get("support_unobservable_becomes_nine_unknown_slots") is True and firewall.get("branch_reselection_after_truth") is False and firewall.get("prior_outcome_read") is False, "R6_CONFIRM_IMPL_PHASE_B_FIREWALL_DRIFT")

    tests = payload.get("test_receipt", {})
    _require(errors, tests.get("focused_test_count") == 8 and tests.get("focused_test_failures") == 0, "R6_CONFIRM_IMPL_TEST_RECEIPT_DRIFT")
    authority = payload.get("execution_authority", {})
    _require(errors, authority.get("implementation_complete") is True, "R6_CONFIRM_IMPL_COMPLETION_DRIFT")
    for field in ("source_decode", "model_execution", "truth_scoring", "evidence_root_creation", "training", "network", "device", "product", "safety"):
        _require(errors, authority.get(field) is False, f"R6_CONFIRM_IMPL_AUTHORITY_DRIFT:{field}")

    if verify_files:
        for relative, size, expected_hash in [*EXPECTED_PREDECESSORS.values(), *EXPECTED_IMPLEMENTATION.values()]:
            path = REPO_ROOT / relative
            _require(errors, path.is_file(), f"R6_CONFIRM_IMPL_FILE_MISSING:{relative}")
            if path.is_file():
                _require(errors, path.stat().st_size == size, f"R6_CONFIRM_IMPL_FILE_BYTES_DRIFT:{relative}")
                _require(errors, _sha(path) == expected_hash, f"R6_CONFIRM_IMPL_FILE_HASH_DRIFT:{relative}")
    return errors


def validate_file(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R6_CONFIRM_IMPL_LOCK_READ_FAILED:{error}"]
    return validate_payload(payload, verify_files=verify_files) if isinstance(payload, Mapping) else ["R6_CONFIRM_IMPL_LOCK_NOT_OBJECT"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    print(json.dumps({"schema": "blindassist.taro.o0r.r6_confirmation_implementation_lock_validation_result.v1", "passed": not errors, "error_count": len(errors), "errors": errors, "terminal": "TARO_O0R_R6_CONFIRMATION_IMPLEMENTATION_LOCK_VALID" if not errors else "TARO_O0R_R6_CONFIRMATION_IMPLEMENTATION_LOCK_INVALID"}, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
