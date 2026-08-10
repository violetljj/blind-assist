#!/usr/bin/env python3
"""Fail-closed static validator for the TARO O0R truth-only preflight lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LOCK_SCHEMA = "blindassist.taro.o0r.truth_only_one_shot_preflight_lock.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK"
LOCK_TERMINAL = "TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_PASS"
CONTRACT_SCHEMA = "blindassist.taro.o0r.source_adapter_contract.v1"
IMPLEMENTATION_SCHEMA = "blindassist.taro.o0r.source_adapter_implementation_lock.v1"
VALIDATOR_PATH = (
    "scripts/research/taro_o0r_truth_preflight/"
    "validate_taro_o0r_truth_preflight_lock.py"
)
EXPECTED_ARGV = [
    "E:/codex-tools/bin/blindassist-python.cmd",
    VALIDATOR_PATH,
    "--lock",
    (
        "docs/research/taro/"
        "TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json"
    ),
]
EXPECTED_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
}
EXPECTED_RUNTIME_VERSIONS = {
    "python": "3.11.9",
    "numpy": "2.1.3",
    "scipy": "1.17.1",
    "pillow": "12.2.0",
}
EXPECTED_ASSET_TEMPLATES = [
    {
        "asset": "upsampling.zip",
        "relative_path_template": "upsampling/{official_fold}/{video_id}.zip",
        "url_template": (
            "https://docs-assets.developer.apple.com/ml-research/datasets/"
            "arkitscenes/v1/upsampling/{official_fold}/{video_id}.zip"
        ),
    },
    {
        "asset": "lowres_wide_intrinsics.zip",
        "relative_path_template": (
            "raw/{official_fold}/{video_id}/lowres_wide_intrinsics.zip"
        ),
        "url_template": (
            "https://docs-assets.developer.apple.com/ml-research/datasets/"
            "arkitscenes/v1/raw/{official_fold}/{video_id}/"
            "lowres_wide_intrinsics.zip"
        ),
    },
    {
        "asset": "lowres_wide.traj",
        "relative_path_template": "raw/{official_fold}/{video_id}/lowres_wide.traj",
        "url_template": (
            "https://docs-assets.developer.apple.com/ml-research/datasets/"
            "arkitscenes/v1/raw/{official_fold}/{video_id}/lowres_wide.traj"
        ),
    },
]
EXPECTED_AUTHORIZATION = {
    "bound_receipt_coverage_status": "INSUFFICIENT_FOR_24_PARENT_BODY_ACCESS",
    "lock_validation_authorized": True,
    "head_metadata_execution_authorized": False,
    "source_body_execution_authorized": False,
    "required_before_any_network_or_body_access": [
        "NEW_TARO_ROUTE_SPECIFIC_SIGNED_24_PARENT_ASSET_RECEIPT",
        "SEPARATE_CONTENT_LENGTH_HEAD_EXECUTION_LOCK",
    ],
}
EXPECTED_AUTHORITY = {
    "preflight_lock_validation": True,
    "network_or_head_execution": False,
    "source_payload_download_or_open": False,
    "truth_materialization": False,
    "selected_source_uncertainty_fit": False,
    "depthart_inference": False,
    "factorial_execution": False,
    "training": False,
    "g0_g1_a0_a1_j0": False,
    "android_qnn_htp": False,
    "default_app": False,
    "product": False,
    "safety": False,
}
EXPECTED_KNOWN_AVAILABILITY_RISKS = [
    {
        "video_id": "47333152",
        "role": "ADAPTER_FIT",
        "risk": "OFFICIAL_DOWNLOADER_MISSING_3DOD_ASSETS_LIST_SUPPRESSES_LOWRES_WIDE_TRAJECTORY",
        "required_disposition": "FUTURE_HEAD_MUST_RETURN_200_WITH_CONTENT_LENGTH_ELSE_R0_NOT_EVALUABLE_NO_REPLACEMENT",
    }
]
EXPECTED_FAILURE_SCOPE = [
    "AUTHORIZATION_SCOPE_MISMATCH",
    "ASSET_HEAD_UNAVAILABLE",
    "CONTENT_LENGTH_MISSING_OR_OVER_BUDGET",
    "BOUND_HASH_DRIFT",
    "ROOT_COLLISION",
    "RUNNER_NOT_IMPLEMENTED",
]
EXPECTED_HEAD_FIELDS = [
    "http_status",
    "content_length_bytes",
    "etag",
    "last_modified",
    "redirect_chain",
    "transport_errors",
]
EXPECTED_ONE_SHOT_RULE = {
    "consumption_root": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r0",
    "consumed_at_lock": False,
    "root_creation_authorized": False,
    "overwrite": False,
    "rerun": False,
    "consumption_trigger": (
        "A future separately authorized truth-only execution consumes R0 when the truth "
        "evidence root is atomically created, regardless of PASS, FAIL or timeout."
    ),
    "head_only_receipt_does_not_consume_truth_one_shot": True,
    "o0r_factor_headroom_root_must_remain_absent": True,
}
EXPECTED_FAILURE_POLICY = {
    "before_consumption": "TARO_O0R_TRUTH_ONLY_PREFLIGHT_NOT_READY_EXECUTION_NOT_AUTHORIZED",
    "after_future_consumption": (
        "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE_ONE_SHOT_CONSUMED_"
        "NO_REPLACEMENT_NO_RERUN"
    ),
    "undefined_denominator": "FAIL_NOT_DROP",
    "replacement_or_role_reassignment": False,
    "depthart_or_factorial_after_failure": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def repo_root_from_validator() -> Path:
    return Path(__file__).resolve().parents[3]


def contract_parents(contract: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role in contract["selection_contract"]["role_order"]:
        for parent in contract["selection_contract"]["roles"][role]:
            rows.append(
                {
                    "role": role,
                    "visit_id": str(parent["visit_id"]),
                    "video_id": str(parent["video_id"]),
                    "official_fold": str(parent["official_fold"]),
                    "selection_rank_sha256": str(parent["selection_rank_sha256"]),
                }
            )
    return rows


def expanded_requests(lock: dict[str, Any]) -> list[dict[str, str]]:
    plan = lock["asset_plan"]
    rows: list[dict[str, str]] = []
    for parent in plan["selected_parents"]:
        fields = {
            "official_fold": parent["official_fold"],
            "video_id": parent["video_id"],
        }
        for asset in plan["asset_templates"]:
            rows.append(
                {
                    "role": parent["role"],
                    "visit_id": parent["visit_id"],
                    "video_id": parent["video_id"],
                    "official_fold": parent["official_fold"],
                    "asset": asset["asset"],
                    "relative_path": asset["relative_path_template"].format(**fields),
                    "url": asset["url_template"].format(**fields),
                }
            )
    return rows


def _append(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_document(
    lock: dict[str, Any],
    *,
    repo_root: Path,
    check_filesystem: bool = True,
) -> list[str]:
    errors: list[str] = []
    _append(errors, lock.get("schema") == LOCK_SCHEMA, "lock schema drift")
    _append(errors, lock.get("lock_id") == LOCK_ID, "lock id drift")
    _append(errors, lock.get("terminal") == LOCK_TERMINAL, "terminal drift")
    _append(
        errors,
        lock.get("status")
        == "PREFLIGHT_LOCKED_HEAD_NOT_RUN_ONE_SHOT_UNCONSUMED_EXECUTION_NOT_AUTHORIZED",
        "status drift",
    )
    _append(errors, lock.get("passed") is True, "lock must be passed")
    _append(errors, lock.get("scientific_status") == "NOT_RUN", "scientific status drift")
    _append(errors, lock.get("source_payload_opened") is False, "source payload must remain unopened")
    _append(errors, lock.get("head_requests_executed") is False, "HEAD requests must remain unexecuted")
    _append(errors, lock.get("one_shot_consumed") is False, "one-shot must remain unconsumed")
    _append(errors, lock.get("truth_materialized") is False, "truth must remain unmaterialized")
    _append(errors, lock.get("model_outputs_absent") is True, "model outputs must remain absent")
    _append(errors, lock.get("execution_authorized") is False, "execution must remain unauthorized")
    _append(
        errors,
        lock.get("implementation_commit")
        == "83ac8d9ab9637eef61d6e254822baba52ddd6c12",
        "implementation commit drift",
    )

    bindings = lock.get("bindings")
    _append(errors, isinstance(bindings, list), "bindings must be a list")
    binding_by_role: dict[str, dict[str, Any]] = {}
    if isinstance(bindings, list):
        for row in bindings:
            if not isinstance(row, dict) or not isinstance(row.get("role"), str):
                errors.append("invalid binding row")
                continue
            if row["role"] in binding_by_role:
                errors.append(f"duplicate binding role: {row['role']}")
                continue
            binding_by_role[row["role"]] = row

    required_roles = {
        "SOURCE_AND_ADAPTER_CONTRACT",
        "SOURCE_ADAPTER_IMPLEMENTATION_LOCK",
        "BOUND_AUTHORIZATION_RECEIPT",
        "ARKITSCENES_OFFICIAL_DOWNLOADER",
        "STATIC_VALIDATOR",
        "VALIDATOR_TESTS",
    }
    _append(errors, set(binding_by_role) == required_roles, "binding role set drift")
    if check_filesystem:
        for role, row in binding_by_role.items():
            path = repo_root / str(row.get("path", ""))
            if not path.is_file():
                errors.append(f"missing binding: {role}")
                continue
            _append(errors, path.stat().st_size == row.get("bytes"), f"binding bytes drift: {role}")
            _append(errors, sha256_file(path) == row.get("sha256"), f"binding sha drift: {role}")

    contract_row = binding_by_role.get("SOURCE_AND_ADAPTER_CONTRACT")
    implementation_row = binding_by_role.get("SOURCE_ADAPTER_IMPLEMENTATION_LOCK")
    contract: dict[str, Any] | None = None
    implementation: dict[str, Any] | None = None
    if contract_row:
        path = repo_root / contract_row["path"]
        if path.is_file():
            contract = json.loads(path.read_text(encoding="utf-8"))
            _append(errors, contract.get("schema") == CONTRACT_SCHEMA, "contract schema drift")
    if implementation_row:
        path = repo_root / implementation_row["path"]
        if path.is_file():
            implementation = json.loads(path.read_text(encoding="utf-8"))
            _append(
                errors,
                implementation.get("schema") == IMPLEMENTATION_SCHEMA,
                "implementation lock schema drift",
            )
            _append(errors, implementation.get("passed") is True, "implementation lock not passed")

    if contract is not None:
        expected_parents = contract_parents(contract)
        plan = lock.get("asset_plan", {})
        _append(errors, plan.get("selected_parents") == expected_parents, "selected parent roster drift")
        _append(errors, plan.get("asset_templates") == EXPECTED_ASSET_TEMPLATES, "asset template drift")
        _append(errors, plan.get("request_method") == "HEAD", "request method must be HEAD")
        _append(errors, plan.get("response_body_bytes_allowed") == 0, "response body budget must be zero")
        _append(errors, plan.get("off_host_redirect_allowed") is False, "off-host redirect must be forbidden")
        _append(errors, plan.get("future_head_fields_required") == EXPECTED_HEAD_FIELDS, "HEAD field set drift")
        _append(errors, plan.get("expected_parent_count") == 24, "expected parent count drift")
        _append(errors, plan.get("expected_asset_count_per_parent") == 3, "asset count per parent drift")
        _append(errors, plan.get("expected_request_count") == 72, "request count drift")
        if isinstance(plan.get("selected_parents"), list) and isinstance(plan.get("asset_templates"), list):
            try:
                requests = expanded_requests(lock)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"request expansion failed: {error}")
            else:
                _append(errors, len(requests) == 72, "expanded request count drift")
                _append(errors, len({row["url"] for row in requests}) == 72, "expanded URL duplicate")
                _append(
                    errors,
                    canonical_sha256(requests) == plan.get("expanded_requests_sha256"),
                    "expanded request digest drift",
                )
                _append(
                    errors,
                    all(row["official_fold"] == "Training" for row in requests),
                    "non-Training request admitted",
                )

        contract_budget = contract["resource_budget"]
        budget = lock.get("resource_budget", {})
        for key in (
            "maximum_compressed_source_bytes",
            "maximum_materialized_source_bytes",
            "truth_only_materialization_wall_seconds",
            "truth_only_peak_rss_bytes",
            "maximum_scientific_evidence_bytes",
            "training_steps",
            "device_or_android",
        ):
            _append(errors, budget.get(key) == contract_budget[key], f"resource budget drift: {key}")
        _append(errors, budget.get("head_timeout_seconds") == 20, "HEAD timeout drift")
        _append(errors, budget.get("head_retries") == 3, "HEAD retry drift")
        _append(errors, budget.get("head_workers") == 8, "HEAD worker drift")
        _append(errors, budget.get("head_response_body_bytes") == 0, "HEAD body bytes drift")
        _append(
            errors,
            budget.get("source_download_content_length_preflight_required") is True,
            "content-length preflight requirement drift",
        )
        _append(
            errors,
            budget.get("network_allowed_only_for_bound_source_assets") is True,
            "network allowlist budget drift",
        )

        expected_roots = [
            {"role": "SOURCE", "path": contract["artifact_isolation"]["future_source_root"], "exists": False},
            {"role": "WORK", "path": contract["artifact_isolation"]["future_work_root"], "exists": False},
            {
                "role": "TRUTH_EVIDENCE",
                "path": contract["artifact_isolation"]["future_truth_evidence_root"],
                "exists": False,
            },
            {
                "role": "O0R_EVIDENCE_SEALED",
                "path": contract["artifact_isolation"]["future_o0r_evidence_root"],
                "exists": False,
            },
        ]
        _append(errors, lock.get("exclusive_roots") == expected_roots, "exclusive root set drift")
        if check_filesystem:
            for row in expected_roots:
                _append(errors, not (repo_root / row["path"]).exists(), f"exclusive root exists: {row['role']}")

    _append(errors, lock.get("argv") == EXPECTED_ARGV, "argv drift")
    _append(errors, lock.get("argv_purpose") == "STATIC_LOCK_VALIDATION_ONLY", "argv purpose drift")
    _append(errors, lock.get("cwd") == "E:/linnan/linnan", "cwd drift")
    _append(errors, lock.get("required_environment") == EXPECTED_ENVIRONMENT, "environment drift")
    _append(errors, lock.get("runtime_versions") == EXPECTED_RUNTIME_VERSIONS, "runtime version drift")
    _append(
        errors,
        lock.get("reserved_truth_execution")
        == {"runner_status": "ABSENT_NOT_IMPLEMENTED", "argv": None, "call_allowed": False},
        "reserved truth execution drift",
    )
    _append(errors, lock.get("head_receipt") is None, "HEAD receipt must remain absent")
    _append(errors, lock.get("authorization_gate") == EXPECTED_AUTHORIZATION, "authorization gate drift")
    _append(errors, lock.get("execution_authority") == EXPECTED_AUTHORITY, "execution authority drift")
    _append(
        errors,
        lock.get("known_availability_risks") == EXPECTED_KNOWN_AVAILABILITY_RISKS,
        "known availability risk drift",
    )
    _append(errors, lock.get("failure_scope") == EXPECTED_FAILURE_SCOPE, "failure scope drift")
    _append(errors, lock.get("one_shot_rule") == EXPECTED_ONE_SHOT_RULE, "one-shot rule drift")
    _append(errors, lock.get("failure_policy") == EXPECTED_FAILURE_POLICY, "failure policy drift")
    static_validation = lock.get("static_validation", {})
    _append(
        errors,
        static_validation.get("validator_command") == EXPECTED_ARGV,
        "static validator command drift",
    )
    _append(errors, static_validation.get("validator_status") == "VALID", "validator status drift")
    _append(errors, static_validation.get("validator_error_count") == 0, "validator error count drift")
    _append(errors, static_validation.get("tests_run") == 8, "test count drift")
    _append(errors, static_validation.get("tests_passed") == 8, "test pass count drift")
    _append(errors, static_validation.get("tests_failed") == 0, "test failure count drift")
    _append(errors, static_validation.get("network_used") is False, "validation network drift")
    _append(errors, static_validation.get("source_payload_used") is False, "validation source-use drift")
    _append(errors, static_validation.get("artifact_root_created") is False, "validation root drift")
    _append(
        errors,
        lock.get("unique_successor", {}).get("id")
        == "TARO_O0R_ARKITSCENES_TRUTH_ONLY_MATERIALIZER_IMPLEMENTATION_LOCK",
        "unique successor drift",
    )
    _append(
        errors,
        lock.get("unique_successor", {}).get("execution_authority") is False,
        "successor execution authority must remain false",
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    repo_root = repo_root_from_validator()
    lock_path = args.lock if args.lock.is_absolute() else repo_root / args.lock
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors = validate_document(lock, repo_root=repo_root, check_filesystem=True)
    if errors:
        print(json.dumps({"status": "INVALID", "errors": errors}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "VALID", "errors": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
