from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CONTRACT_SHA256 = (
    "ee7285c021460b25bc3f1c1a668c7e3e4181427da291b3aa599b92fc3a2bb177"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, check_id: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check_id": check_id, "status": "PASS" if condition else "FAIL"})


def validate(repo: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    check(
        sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256,
        "exact_frozen_contract_sha256",
        checks,
    )
    check(
        contract.get("schema_version")
        == "rcle_rgb_segment_confirmation_r2_transport_repair_contract.v1",
        "schema_version",
        checks,
    )
    check(
        contract.get("protocol_id") == "RCLE_RGB_SEGMENT_CONFIRMATION_R2",
        "protocol_id",
        checks,
    )
    check(
        contract.get("status") == "DESIGN_REVIEW_REQUIRED",
        "pre_review_status",
        checks,
    )
    check(
        contract.get("current_terminal") == "DESIGN_REVIEW_REQUIRED",
        "pre_review_current_terminal",
        checks,
    )

    bindings = contract.get("r1_immutable_bindings", {})
    for name in (
        "protocol_terminal",
        "protocol_terminal_validation",
        "protocol_terminal_final_independent_review",
        "completion_audit",
        "preaccess_lock",
        "r1_transport_implementation",
    ):
        binding = bindings.get(name, {})
        path = repo / binding.get("path", "")
        check(path.is_file(), f"binding_exists:{name}", checks)
        check(
            path.is_file() and sha256_file(path) == binding.get("sha256"),
            f"binding_sha256:{name}",
            checks,
        )

    check(bindings.get("claims_consumed") is True, "r1_claims_consumed", checks)
    check(bindings.get("retry_forbidden") is True, "r1_retry_forbidden", checks)
    check(bindings.get("mutation_forbidden") is True, "r1_mutation_forbidden", checks)

    scope = contract.get("scope_lock", {})
    for key in (
        "new_source_discovery",
        "source_replacement",
        "window_reselection",
        "full_dataset_download",
        "rgb_decode",
        "rgb_algorithm_execution",
        "android_execution",
    ):
        check(scope.get(key) is False, f"scope_false:{key}", checks)

    segments = scope.get("segments", [])
    check(len(segments) == 2, "exactly_two_frozen_segments", checks)
    if len(segments) == 2:
        check(
            segments[0]
            == {
                "source_family_id": "OPENLORIS_CORRIDOR",
                "capture_id": "corridor1-1",
                "window_id": "corridor1-1:w004",
                "half_open_window_s": [
                    "1560000043.537699",
                    "1560000053.537699",
                ],
                "required_selected_frames": 300,
                "required_guard_frames": 2,
                "target_member_count": 302,
            },
            "openloris_exact_segment",
            checks,
        )
        check(
            segments[1]
            == {
                "source_family_id": "DLR_RGBD_VICON",
                "capture_id": "extreme_geometry/hexagon_01",
                "window_id": "extreme_geometry/hexagon_01:w001",
                "half_open_window_s": [
                    "1634201323.995618343",
                    "1634201333.995618343",
                ],
                "required_geometry_frames": 299,
                "required_guard_frames": 2,
                "zip_member": "extreme_geometry/hexagon_01.bag",
            },
            "dlr_exact_segment",
            checks,
        )

    diagnosis = contract.get("r1_diagnosis", {})
    openloris = diagnosis.get("openloris", {})
    dlr = diagnosis.get("dlr", {})
    check(
        openloris.get("frozen_transport_limits", {}).get("remote_byte_cap")
        == 3_947_000_000,
        "openloris_cap_unchanged",
        checks,
    )
    check(
        openloris.get("frozen_transport_limits", {}).get(
            "maximum_attempts_per_identical_range"
        )
        == 3,
        "openloris_attempt_limit_unchanged",
        checks,
    )
    check(
        openloris.get("frozen_transport_limits", {}).get("cap_increase_allowed")
        is False,
        "openloris_cap_increase_forbidden",
        checks,
    )
    check(
        openloris.get("frozen_transport_limits", {}).get(
            "target_or_guard_change_allowed"
        )
        is False,
        "openloris_target_guard_change_forbidden",
        checks,
    )
    dlr_bound = dlr.get("minimum_no_retry_full_member_hard_cap", {})
    check(
        dlr_bound
        == {
            "successful_response_bytes_upper_bound_including_local_header": 3_633_353_304,
            "size_plus_one_preauthorization_cap_lower_bound": 3_633_353_305,
            "r1_cap": 1_073_741_824,
            "cap_increase_is_new_authority": True,
            "cap_increase_authorized": False,
            "hard_cap_if_later_activated": 3_633_353_305,
            "retry_headroom": 0,
            "retry_rule": (
                "Any retry that consumes the fixed hard cap before a complete index "
                "produces NOT_EVALUABLE. The cap may not be increased during or after "
                "the claim."
            ),
        },
        "dlr_minimum_no_retry_hard_cap_exact",
        checks,
    )
    check(
        dlr_bound.get("cap_increase_authorized") is False,
        "dlr_cap_increase_not_authorized",
        checks,
    )
    dlr_requirements = set(dlr.get("minimum_index_authority_requirements", []))
    check(
        any("exact member compressed-data start" in row for row in dlr_requirements)
        and any("previously verified checkpoint" in row for row in dlr_requirements),
        "dlr_no_resume_without_checkpoint",
        checks,
    )
    check(
        any("may not decode, display, retain, or execute RGB pixel payloads" in row for row in dlr_requirements),
        "dlr_index_only_boundary",
        checks,
    )

    authority = contract.get("execution_authority", {})
    expected_authority = {
        "local_outcome_blind_implementation_and_tests": True,
        "real_source_network_access": False,
        "real_data_claim_creation": False,
        "openloris_r2_execution": False,
        "dlr_index_execution": False,
        "dlr_rgb_extraction": False,
        "rgb_algorithm_execution": False,
        "performance_qualification": False,
        "host_offline_replay": False,
        "product_or_safety_claim": False,
        "android": False,
    }
    check(authority == expected_authority, "authority_exact_closed_object", checks)
    check(
        authority.get("local_outcome_blind_implementation_and_tests") is True,
        "local_outcome_blind_work_allowed",
        checks,
    )
    for key in (
        "real_source_network_access",
        "real_data_claim_creation",
        "openloris_r2_execution",
        "dlr_index_execution",
        "dlr_rgb_extraction",
        "rgb_algorithm_execution",
        "performance_qualification",
        "host_offline_replay",
        "product_or_safety_claim",
        "android",
    ):
        check(authority.get(key) is False, f"authority_false:{key}", checks)

    legal = set(contract.get("legal_terminals", []))
    check(
        legal
        == {
            "DESIGN_REVIEW_PASS_EXECUTION_NOT_AUTHORIZED",
            "DESIGN_REVIEW_FAIL",
            "HOLD_NOT_EVALUABLE",
        },
        "legal_terminals_closed",
        checks,
    )
    check(
        "source-role confounded"
        in contract.get("claim_ceiling", {}).get("source_role_confounding", ""),
        "source_role_confounding_preserved",
        checks,
    )
    activation = contract.get("future_activation_requirements", {})
    common = set(activation.get("common", []))
    openloris_activation = set(activation.get("openloris", []))
    dlr_activation = set(activation.get("dlr", []))
    check(
        any("new exclusive one-shot R2 claim namespace" in row for row in common),
        "activation_new_claim_namespace",
        checks,
    )
    check(
        any("No outcome access occurs before claim creation" in row for row in common),
        "activation_no_outcome_before_claim",
        checks,
    )
    check(
        any("302 targets" in row and "byte cap" in row for row in openloris_activation),
        "activation_openloris_frozen_constraints",
        checks,
    )
    check(
        any("index claim and a later exact-window extraction claim must be distinct" in row for row in dlr_activation),
        "activation_dlr_index_extraction_separated",
        checks,
    )
    check(
        any("transient access to serialized RGB payload bytes" in row for row in dlr_requirements),
        "activation_dlr_transient_payload_separate_authority",
        checks,
    )

    pass_count = sum(item["status"] == "PASS" for item in checks)
    try:
        display_path = contract_path.relative_to(repo).as_posix()
    except ValueError:
        display_path = str(contract_path)
    return {
        "schema_version": "rcle_rgb_segment_confirmation_r2_contract_validation.v1",
        "contract_path": display_path,
        "contract_sha256": sha256_file(contract_path),
        "decision": "PASS" if pass_count == len(checks) else "FAIL",
        "pass_count": pass_count,
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    contract_path = args.contract
    if not contract_path.is_absolute():
        contract_path = repo / contract_path
    result = validate(repo, contract_path.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = repo / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
