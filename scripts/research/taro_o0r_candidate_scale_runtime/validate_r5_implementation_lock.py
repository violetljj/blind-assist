#!/usr/bin/env python3
"""Validate the TARO R5 implementation lock without activating execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation_io as r5io
from scripts.research.taro_o0r_candidate_scale_runtime import validate_r5_amendment


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_IMPLEMENTATION_LOCK_2026-08-11.json"
SCHEMA = "blindassist.taro.o0r.r5_confirmation_implementation_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_IMPLEMENTATION_LOCK"
SUCCESSOR = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_ONE_SHOT_EXECUTION_LOCK"
R5_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5"

EXPECTED_PROTOCOL_BINDINGS = {
    "R5_AMENDMENT": ("docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_AMENDMENT_2026-08-11.json", 15346, "F4029F658C1617044667DCBF137F7AC2DB6FF4528EC4E48EF5DFCCB9F91CE89F"),
    "R5_TRANSFORM_REPAIR": ("docs/research/taro/TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_PRE_IMPLEMENTATION_TRANSFORM_ID_REPAIR_2026-08-11.json", 2866, "9BA0EF22CAB5B29757E012E3BAB369FF1AB3B7FF4F42C777E3D3D16018C9C6BE"),
    "R3_EXACT_FRAME_PLAN": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/exact-frame-plan.json.gz", 11228, "BBAC08A9832A6CE465E8A8C6532FFF857AF878E6E821E683B24A83981C3C946C"),
}
EXPECTED_IMPLEMENTATION_BINDINGS = {
    "R5_CORE": ("scripts/research/taro_o0r_candidate_scale_runtime/r5_confirmation.py", 79293, "E792DAF55CFAAC3348FC85261641EF8860A8CAB9F2EE69112BA62FC8367F5F7F"),
    "R5_PHASE_IO": ("scripts/research/taro_o0r_candidate_scale_runtime/r5_confirmation_io.py", 11214, "D8C51E6C0857C25B291251C4B91B1CF57DF2AB7727455484565E6AE901DAE003"),
    "R5_RUNNER": ("scripts/research/taro_o0r_candidate_scale_runtime/run_direct_apple_hybrid_adapter_fit_confirmation.py", 17744, "7795D020616AC702395D1E25749FA3EACB00DBEC9E20993A1957820676B3D890"),
    "R5_CORE_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r5_confirmation.py", 14760, "E505E50CF1C99903BDE6DAA400195FA4DA47621C5BCA7BD0F36CCC9AA62B56D9"),
    "R5_RUNNER_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_r5_confirmation_runner.py", 2062, "B5F87DF36712D46A1924441DEE0269B634E7D458C4C4C5ACD754BE661F908DB4"),
    "R5_EXECUTION_LOCK_VALIDATOR": ("scripts/research/taro_o0r_candidate_scale_runtime/validate_r5_execution_lock.py", 11246, "FF12C91807892232F4FFF04A78FCC14B995ECCCB1165F7A006A0CEDA03C48E71"),
    "R5_EXECUTION_LOCK_VALIDATOR_TEST": ("scripts/research/taro_o0r_candidate_scale_runtime/test_validate_r5_execution_lock.py", 5289, "1B529E40B498ADF67195A652120DD76B67F297B78D6860B4DAD8B4DC9D8DACB0"),
    "R5_AMENDMENT_VALIDATOR": ("scripts/research/taro_o0r_candidate_scale_runtime/validate_r5_amendment.py", 29855, "DD271A797A61DE9ACE1A1BE605E37D666271DCBB3571BCE520FB5C8F43492B91"),
    "DEPTHART_RUNTIME": ("scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py", 30462, "25E80D496DBEEBCC66CCB9772DD51508453B8EB325CDBF9192F71923E440BD09"),
    "SOURCE_ADAPTER": ("scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py", 164498, "EC9E3C2D4D122AFC5939183B9FA4E00411EC1C9BEAD324831406F159DBB7E91F"),
    "APPLE_SCALE": ("scripts/research/taro_o0r_candidate_scale_runtime/apple_scale.py", 31757, "3FE701EFB7C0C70B9AC3AF9021AAABFBCBC4C4A1DD4A8F39250A3842B9A17A71"),
    "DIRECT_APPLE_SUPPORT": ("scripts/research/taro_o0r_candidate_scale_runtime/direct_apple_support.py", 38656, "0495064DB81970FBEDCAFC3EB760AFC1AE8689C1D23D5FDFA4BA31843E7FAE93"),
    "SOURCE_FACTOR": ("scripts/research/taro_o0r_candidate_scale_runtime/source_factor.py", 43033, "20D4DBF3EE414EAE07E39C8AD8FAB9BF6CDBDBCB4BED5A879700DAB48E1101A3"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _binding_map(value: Any, errors: list[str], prefix: str) -> dict[str, tuple[Any, Any, Any]]:
    result: dict[str, tuple[Any, Any, Any]] = {}
    if not isinstance(value, list):
        errors.append(f"{prefix}_NOT_LIST")
        return result
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"role", "path", "bytes", "sha256"} or not isinstance(row.get("role"), str):
            errors.append(f"{prefix}_ROW_INVALID")
            continue
        role = row["role"]
        if role in result:
            errors.append(f"{prefix}_DUPLICATE:{role}")
            continue
        result[role] = (row["path"], row["bytes"], row["sha256"])
    return result


def _verify_bindings(errors: list[str], observed: dict[str, tuple[Any, Any, Any]], expected: dict[str, tuple[str, int, str]], prefix: str) -> None:
    _require(errors, observed == expected, f"{prefix}_SET_DRIFT")
    for role, (relative, expected_bytes, expected_sha) in expected.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            errors.append(f"{prefix}_FILE_MISSING:{role}")
            continue
        _require(errors, path.stat().st_size == expected_bytes, f"{prefix}_BYTES_DRIFT:{role}")
        _require(errors, _sha256(path) == expected_sha, f"{prefix}_HASH_DRIFT:{role}")


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema", "lock_id", "date", "research_mode", "status", "protocol_bindings",
        "implementation_bindings", "candidate_identity", "exact_cohort", "phase_firewall",
        "frozen_algorithm", "evidence_contract", "test_receipt", "execution_authority",
        "claim_ceiling", "unique_successor",
    }
    _require(errors, set(payload) == expected_keys, "R5_IMPL_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA, "R5_IMPL_SCHEMA_DRIFT")
    _require(errors, payload.get("lock_id") == LOCK_ID, "R5_IMPL_LOCK_ID_DRIFT")
    _require(errors, payload.get("date") == "2026-08-11" and payload.get("research_mode") == "WILD_LAB", "R5_IMPL_CONTEXT_DRIFT")
    _require(errors, payload.get("status") == "IMPLEMENTATION_FROZEN_EXECUTION_FALSE", "R5_IMPL_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R5_IMPL_SUCCESSOR_DRIFT")

    protocol = _binding_map(payload.get("protocol_bindings"), errors, "R5_IMPL_PROTOCOL_BINDINGS")
    implementation = _binding_map(payload.get("implementation_bindings"), errors, "R5_IMPL_CODE_BINDINGS")
    _require(errors, protocol == EXPECTED_PROTOCOL_BINDINGS, "R5_IMPL_PROTOCOL_BINDING_SET_DRIFT")
    _require(errors, implementation == EXPECTED_IMPLEMENTATION_BINDINGS, "R5_IMPL_CODE_BINDING_SET_DRIFT")

    candidate = payload.get("candidate_identity", {})
    _require(errors, candidate.get("model_id") == "depthart-s-metric-indoor-448-official-fp32", "R5_IMPL_MODEL_ID_DRIFT")
    _require(errors, candidate.get("source_commit") == depthart_commit(), "R5_IMPL_MODEL_SOURCE_DRIFT")
    _require(errors, candidate.get("checkpoint_bytes") == 32871942 and candidate.get("checkpoint_sha256") == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65", "R5_IMPL_CHECKPOINT_ID_DRIFT")
    _require(errors, candidate.get("preprocess_id") == "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1", "R5_IMPL_PREPROCESS_DRIFT")
    _require(errors, candidate.get("postprocess_id") == "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1", "R5_IMPL_POSTPROCESS_DRIFT")

    cohort = payload.get("exact_cohort", {})
    _require(errors, cohort.get("source_role") == "ADAPTER_FIT" and cohort.get("successor_role") == r5.R5_ROLE, "R5_IMPL_ROLE_DRIFT")
    _require(errors, cohort.get("parent_count") == 8 and cohort.get("physical_frame_count") == r5.EXPECTED_FRAME_COUNT and cohort.get("query_slot_count") == r5.EXPECTED_QUERY_COUNT, "R5_IMPL_COHORT_COUNT_DRIFT")
    _require(errors, cohort.get("canonical_frame_identity_sequence_sha256") == r5.EXPECTED_IDENTITY_SEQUENCE_SHA256, "R5_IMPL_COHORT_HASH_DRIFT")
    _require(errors, cohort.get("prior_eval_parent_or_truth_substitution_allowed") is False, "R5_IMPL_PRIOR_EVAL_SUBSTITUTION_ALLOWED")

    firewall = payload.get("phase_firewall", {})
    _require(errors, firewall.get("phase_a_order") == ["ALL_211_CANDIDATES_SEALED", "ALL_211_SOURCE_DECISIONS_SEALED", "PHASE_A_COMPLETION_RELOADED"], "R5_IMPL_PHASE_A_ORDER_DRIFT")
    _require(errors, firewall.get("phase_a_payload_allowlist") == ["color", "lowres_depth", "confidence"], "R5_IMPL_PHASE_A_ALLOWLIST_DRIFT")
    _require(errors, set(firewall.get("phase_a_required_zero_reads", [])) == {"FARO", "QUERY_TRUTH", "COMPACT_TRUTH", "TASK_METRIC", "PRIOR_EVAL_OUTCOME"}, "R5_IMPL_PHASE_A_ZERO_READ_DRIFT")
    _require(errors, firewall.get("phase_b_first_faro_read_requires_phase_a_completion") is True and firewall.get("branch_reselection_after_phase_a_allowed") is False, "R5_IMPL_PHASE_B_FIREWALL_DRIFT")
    _require(errors, firewall.get("prior_eval_truth_or_outcome_root_enumeration_allowed") is False, "R5_IMPL_PRIOR_EVAL_READ_ALLOWED")

    algorithm = payload.get("frozen_algorithm", {})
    _require(errors, algorithm.get("policy_id") == r5.POLICY_ID and algorithm.get("selection_field_allowlist") == ["source_support_available"], "R5_IMPL_POLICY_DRIFT")
    _require(errors, algorithm.get("direct_failure_after_selection") == "RETAIN_DIRECT_UNKNOWN_NEVER_FALL_BACK", "R5_IMPL_DIRECT_FALLBACK_DRIFT")
    _require(errors, algorithm.get("free_parameter_count") == algorithm.get("threshold_count") == algorithm.get("training_steps") == 0, "R5_IMPL_FITTING_DRIFT")

    evidence = payload.get("evidence_contract", {})
    _require(errors, evidence.get("exclusive_root") == "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5", "R5_IMPL_EVIDENCE_ROOT_DRIFT")
    for field in ("root_absent_at_lock", "root_must_be_absent_at_execution", "one_shot_consumed_on_root_creation", "candidate_native_arrays_only", "query_records_per_frame_gzip", "result_then_manifest_written_last"):
        _require(errors, evidence.get(field) is True, f"R5_IMPL_EVIDENCE_RULE_DRIFT:{field}")
    for field in ("overwrite", "rerun", "faro_geometry_persisted"):
        _require(errors, evidence.get(field) is False, f"R5_IMPL_EVIDENCE_RULE_DRIFT:{field}")

    tests = payload.get("test_receipt", {})
    _require(errors, tests.get("focused_test_count") == 22 and tests.get("focused_test_failures") == 0, "R5_IMPL_TEST_RECEIPT_DRIFT")
    _require(errors, tests.get("actual_model_inference_count") == tests.get("actual_r5_task_metric_count") == 0, "R5_IMPL_PREMATURE_OUTPUT")
    _require(errors, isinstance(tests.get("commands"), list) and len(tests["commands"]) == 2, "R5_IMPL_TEST_COMMAND_DRIFT")

    authority = payload.get("execution_authority", {})
    expected_authority = {
        "implementation_complete": True, "implementation_lock": True,
        "one_shot_execution_lock": False, "user_model_execution_authority": False,
        "depthart_inference": False, "phase_a_source_decisions": False,
        "phase_b_truth_scoring": False, "training": False, "network": False,
        "device": False, "product": False, "safety": False,
    }
    _require(errors, authority == expected_authority, "R5_IMPL_AUTHORITY_DRIFT")

    if verify_files:
        _verify_bindings(errors, protocol, EXPECTED_PROTOCOL_BINDINGS, "R5_IMPL_PROTOCOL_BINDING")
        _verify_bindings(errors, implementation, EXPECTED_IMPLEMENTATION_BINDINGS, "R5_IMPL_CODE_BINDING")
        errors.extend(validate_r5_amendment.validate_file(verify_files=True))
        errors.extend(validate_r5_amendment.validate_repair_file(verify_files=True))
        _require(errors, not R5_ROOT.exists(), "R5_IMPL_EVIDENCE_ROOT_ALREADY_EXISTS")
        source_root = Path(str(candidate.get("source_root", "")))
        checkpoint = Path(str(candidate.get("checkpoint_path", "")))
        _require(errors, source_root.is_dir() and checkpoint.is_file(), "R5_IMPL_MODEL_ASSET_MISSING")
        if source_root.is_dir():
            commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(source_root), "status", "--short"], capture_output=True, text=True).stdout.strip()
            _require(errors, commit == candidate.get("source_commit") and (not candidate.get("source_tree_clean_required") or not dirty), "R5_IMPL_MODEL_SOURCE_TREE_DRIFT")
        if checkpoint.is_file():
            _require(errors, checkpoint.stat().st_size == candidate.get("checkpoint_bytes") and _sha256(checkpoint) == candidate.get("checkpoint_sha256"), "R5_IMPL_CHECKPOINT_FILE_DRIFT")
        try:
            frames = r5io.load_exact_cohort(
                REPO_ROOT / EXPECTED_PROTOCOL_BINDINGS["R3_EXACT_FRAME_PLAN"][0],
                REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3",
                REPO_ROOT / "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3",
                verify_containers=False,
            )
        except Exception as error:
            errors.append(f"R5_IMPL_COHORT_REPLAY_FAILED:{getattr(error, 'code', type(error).__name__)}")
        else:
            _require(errors, len(frames) == r5.EXPECTED_FRAME_COUNT, "R5_IMPL_COHORT_REPLAY_COUNT_DRIFT")
    return errors


def depthart_commit() -> str:
    return "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c"


def validate_file(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"R5_IMPL_LOCK_READ_FAILED:{error}"]
    if not isinstance(payload, Mapping):
        return ["R5_IMPL_LOCK_NOT_OBJECT"]
    return validate_payload(payload, verify_files=verify_files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    result = {
        "schema": "blindassist.taro.o0r.r5_implementation_lock_validation_result.v1",
        "lock": str(args.lock),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O0R_R5_IMPLEMENTATION_LOCK_VALID" if not errors else "TARO_O0R_R5_IMPLEMENTATION_LOCK_INVALID",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
