#!/usr/bin/env python3
"""Independent validator for the sealed R11 source-only 48-to-24 selection."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import (
    positive_occupancy_factor as r7_positive,
)
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit
from scripts.research.taro_o1r_r11_abstention_runtime import (
    abstention_candidate,
    fresh_pool,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-top24-selection-r0"
PHASE_A_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0"
PHASE_A_TERMINAL = f"{PHASE_A_ROOT}/terminal.json"
PHASE_A_COMPLETION = f"{PHASE_A_ROOT}/phase-a-completion.json"
PHASE_A_AUDIT_ROOT = (
    "artifacts.local/evidence/taro/"
    "o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0"
)
PHASE_A_AUDIT = f"{PHASE_A_AUDIT_ROOT}/post-result-audit.json"
INVENTORY = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/exact-frame-plan.json"
LOCK_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json"
)

LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_execution_lock.v1"
EXECUTION_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_execution_receipt.v1"
SCORES_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_source_only_parent_scores.v1"
SELECTION_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_source_only_selection.v1"
RESULT_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_result.v1"
TERMINAL_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_terminal.v1"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_TOP24_SOURCE_ONLY_SELECTION_SEALED_PASS"
LOCK_ID = "TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK"

PARENT_COUNT = 48
SELECTED_PARENT_COUNT = 24
FRAME_COUNT = 1043
QUERY_COUNT = 9387
PHASE_A_FILE_COUNT = 5219
PHASE_A_PRE_TERMINAL_FILE_COUNT = 5218
FROZEN_FRAME_COUNTS = [
    20, 14, 23, 24, 29, 7, 12, 14, 10, 21, 28, 15, 11, 28, 29, 72,
    36, 14, 18, 4, 54, 32, 83, 17, 15, 16, 29, 10, 12, 34, 7, 14,
    11, 6, 9, 1, 46, 6, 27, 26, 50, 9, 11, 27, 12, 9, 28, 13,
]
EXPECTED_IDENTITIES = tuple((visit, video) for visit, video, _rank in fresh_pool.EXPECTED_POOL)

FROZEN_SELECTOR = {
    "selector_id": "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1",
    "selector_content_sha256": "67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC",
    "rule_id": "02CE016D6B0011F0",
}
FROZEN_RULE = {
    "state_policy": "UNKNOWN_ONLY",
    "minimum_far_valid_anchor_count": 6,
    "maximum_far_valid_anchor_count": 1_000_000,
    "far_fraction_index": 0,
    "maximum_far_fraction": 0.0,
    "minimum_observed_support_points": 0,
    "require_query_receipt": True,
    "require_positive_obstacle_veto_false": True,
    "require_all_occupied_hits_false": True,
    "rule_id": "02CE016D6B0011F0",
}
PHASE_A_TERMINAL_FILE_SHA256 = "C4084BDBD00ECCA5735753E20893348505AB2A19650FB334850ADCD6D5173186"
PHASE_A_TERMINAL_CONTENT_SHA256 = "596DF5960A483F45A8777B914C50412B6A4A29CBCF39581FE0984DC658F77B71"
PHASE_A_COMPLETION_FILE_SHA256 = "739F3796925B88FCE6F8443DB1979D6644F4937AC913CF60C4BFB618003CF046"
PHASE_A_COMPLETION_CONTENT_SHA256 = "ADB3D8317BC40A09266A5B5FB08D6554598AFC2D038B4E37C3C308BD7CB5EB7A"
PHASE_A_AUDIT_FILE_SHA256 = "2D80268D3E6D928F928F7C3B8AF8B14C7396D16A305521293C052DF6A378D19C"
PHASE_A_AUDIT_CONTENT_SHA256 = "1717F1601A037E43A31EDF5861D496A9AEFE0C4F561F5A8DA0886E3E21EB0824"
INVENTORY_CONTENT_SHA256 = "35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600"

EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_top24_selection",
    "--execution-lock",
    LOCK_RELATIVE,
]
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": (
        "Exact frozen R11 48-parent Training pool and 144-URL plan: zero-body HEAD, bounded source download "
        "and integrity validation, all-48 source-only Phase A, source-only top-24 selection, then FARO only "
        "for the sealed top 24; no training, device, deployment, product, safety, or redistribution authority."
    ),
}
EXPECTED_AUTHORITY = {
    "sealed_phase_a_reload": True,
    "source_only_parent_scoring": True,
    "top24_selection": True,
    "source_zip_member_payload_read": False,
    "highres_depth_member_payload_read": False,
    "faro_read": False,
    "truth_read": False,
    "label_read": False,
    "outcome_read": False,
    "model_execution": False,
    "candidate_rerun": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
    "redistribution": False,
}
EXPECTED_RESOURCE_BUDGET = {
    "maximum_wall_seconds": 7200,
    "maximum_peak_rss_bytes": 8_589_934_592,
    "maximum_evidence_bytes": 16_777_216,
}
EXPECTED_ONE_SHOT_POLICY = {
    "consumed_on_output_root_creation": True,
    "failure_does_not_restore_authority": True,
    "atomic_terminal_bundle": True,
    "success_pre_terminal_file_count": 3,
    "success_final_file_count": 4,
    "terminal_reserve_bytes": 4_194_304,
    "terminal_wall_reserve_seconds": 60,
}
EXPECTED_CLAIM_CEILING = (
    "Source-only R11 scores and a sealed top-24 parent identity; no FARO label, task "
    "effectiveness, training, deployment, product or safety evidence."
)
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": "docs/research/taro/TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R11_DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json",
    "R11_INVENTORY": INVENTORY,
    "R11_PHASE_A_EXECUTION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    "R11_PHASE_A_TERMINAL": PHASE_A_TERMINAL,
    "R11_PHASE_A_COMPLETION": PHASE_A_COMPLETION,
    "R11_PHASE_A_REPAIRED_AUDIT": PHASE_A_AUDIT,
    "R9_DEVELOPMENT_RESULT": "docs/research/taro/TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_RESULT_2026-08-12.json",
    "R9_SELECTOR_ARTIFACT": "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0/selector.json",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "CANDIDATE_SCALE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "REDUCER_INTEGRATION_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "R7_SOURCE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R7_FRESH_COHORT_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/fresh_confirmation_cohort.py",
    "R10_FRESH_POOL_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R11_FRESH_POOL_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R11_TOP24_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_top24_selection.py",
    "R11_TOP24_RUNNER_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_top24_selection.py",
    "R11_TOP24_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_top24_selection.py",
    "R11_TOP24_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_top24_selection.py",
    "R11_TOP24_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_IMPLEMENTATION_LOCK_2026-08-13.md",
}
ARTIFACT_BINDING_ROLES = {
    "R11_INVENTORY",
    "R11_PHASE_A_TERMINAL",
    "R11_PHASE_A_COMPLETION",
    "R11_PHASE_A_REPAIRED_AUDIT",
    "R9_SELECTOR_ARTIFACT",
}

ZERO_FIELDS = (
    "source_zip_member_payload_reads",
    "highres_depth_member_payload_reads",
    "faro_reads",
    "truth_reads",
    "label_reads",
    "outcome_reads",
    "model_executions",
    "training_steps",
    "network_requests",
)
SCORE_FIELDS = {
    *FROZEN_SELECTOR,
    "parent_id", "video_id", "frame_count", "query_count", "available_query_count",
    "eligible_query_count", "eligible_fraction_of_available", "source_frame_hash_sequence_sha256",
    "tie_break_sha256", *ZERO_FIELDS, "clear_output_emitted", "unknown_is_negative",
}
EXECUTION_FIELDS = {
    "schema", "execution_lock_sha256", "execution_lock_content_sha256", "started_at_utc",
    "phase_a_root", "phase_a_audit_content_sha256", "frozen_selector",
    "one_shot_consumed_on_root_creation", *ZERO_FIELDS, "content_sha256",
}
SCORES_FIELDS = {
    "schema", "selector", "phase_a_completion_sha256", "source_frame_hash_sequence_sha256",
    "parent_count", "frame_count", "query_count", "parent_scores", "ranked_parent_identities",
    "all_48_source_records_sealed_before_scoring", "all_48_parent_scores_sealed_before_faro",
    *ZERO_FIELDS, "content_sha256",
}
SELECTION_FIELDS = {
    "schema", "selector", "parent_scores_sha256", "selected_parent_count",
    "selected_parent_identities", "selected_parent_scores", "selection_sealed_before_faro",
    "read_unselected_faro", "source_reselection_after_faro", "parent_reselection_after_faro",
    "candidate_or_threshold_reselection_after_faro", "unknown_is_negative", *ZERO_FIELDS,
    "content_sha256",
}
RESULT_FIELDS = {
    "schema", "terminal", "passed", "execution_valid", "parent_count", "frame_count",
    "query_count", "selected_parent_count", "selected_parent_identities",
    "phase_a_terminal_content_sha256", "phase_a_completion_sha256",
    "phase_a_audit_content_sha256", "parent_scores_sha256", "selection_sha256",
    "all_48_source_records_sealed_before_scoring", "all_48_parent_scores_sealed_before_faro",
    "selection_sealed_before_faro", "unknown_is_negative", "phase_a_prior_file_validations",
    "phase_a_prior_bytes_validated", "phase_a_lineage_decodes", "phase_a_source_receipt_decodes",
    "source_frame_records_validated", "query_features_scored", *ZERO_FIELDS, "elapsed_seconds",
    "peak_rss_bytes", "resource_budget", "one_shot_consumed", "unique_successor",
    "claim_ceiling", "content_sha256",
}
TERMINAL_FIELDS = {
    "schema", "terminal", "passed", "execution_valid", "result", "files",
    "file_count_before_terminal", "bytes_before_terminal", "one_shot_consumed", "content_sha256",
}


class Top24ValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise Top24ValidationError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_TOP24_VALIDATION_JSON", f"JSON object required: {path}")
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "R11_TOP24_VALIDATION_JSON", f"gzip JSON object required: {path}")
    return value


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), "R11_TOP24_VALIDATION_SEAL", "sealed value must be an object")
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        "R11_TOP24_VALIDATION_SEAL",
        f"schema/content seal drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _zeros(record: Mapping[str, Any]) -> bool:
    return all(record.get(field) == 0 for field in ZERO_FIELDS)


def _file_receipt(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": materializer.sha256_file(path)}


def _commit_is_on_master(commit: Any) -> bool:
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    return all(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, ancestor],
            capture_output=True,
            check=False,
        ).returncode
        == 0
        for ancestor in ("HEAD", "refs/remotes/origin/master")
    )


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        "R11_TOP24_VALIDATION_IMPLEMENTATION_BINDING",
        f"implementation commit lacks binding: {relative}",
    )
    return completed.stdout


def _validate_bindings(lock: Mapping[str, Any]) -> None:
    bindings = lock.get("bindings")
    require(
        isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS),
        "R11_TOP24_VALIDATION_BINDINGS",
        "execution-lock binding count drift",
    )
    seen: set[str] = set()
    for row in bindings:
        require(isinstance(row, Mapping), "R11_TOP24_VALIDATION_BINDING", "binding row must be an object")
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str)
            and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R11_TOP24_VALIDATION_BINDING",
            "binding role/path drift",
        )
        path = _repo_path(str(relative))
        require(
            path.is_file()
            and row.get("bytes") == path.stat().st_size
            and row.get("sha256") == materializer.sha256_file(path),
            "R11_TOP24_VALIDATION_BINDING_HASH",
            f"binding drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(
                path.read_bytes() == _git_bytes(str(lock.get("implementation_commit")), str(relative)),
                "R11_TOP24_VALIDATION_IMPLEMENTATION_BINDING",
                f"implementation-commit binding drift: {relative}",
            )
        seen.add(role)
    require(seen == set(EXPECTED_BINDINGS), "R11_TOP24_VALIDATION_BINDINGS", "binding role set drift")


def _validate_frozen_selector_artifact() -> None:
    selector = _load_json(_repo_path(EXPECTED_BINDINGS["R9_SELECTOR_ARTIFACT"]))
    observed = selector.pop("content_sha256", None)
    require(
        observed == FROZEN_SELECTOR["selector_content_sha256"]
        and adapter.canonical_sha256(selector) == observed,
        "R11_TOP24_VALIDATION_SELECTOR_SEAL",
        "R9 selector artifact seal drift",
    )
    selector["content_sha256"] = observed
    validated = clear_enrichment_fit.validate_selector(selector)
    require(
        validated.get("selector_id") == FROZEN_SELECTOR["selector_id"]
        and validated.get("chosen_rule") == FROZEN_RULE,
        "R11_TOP24_VALIDATION_SELECTOR",
        "R9 selector identity/rule drift",
    )


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_TOP24_VALIDATION_LOCK_PATH", "lock path drift")
    lock = _validate_seal(_load_json(lock_path), LOCK_SCHEMA)
    require(
        lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False
        and lock.get("argv") == EXPECTED_ARGV
        and lock.get("phase_a_root") == PHASE_A_ROOT
        and lock.get("phase_a_repaired_audit") == PHASE_A_AUDIT
        and lock.get("output_root") == EVIDENCE_ROOT
        and lock.get("overwrite") is lock.get("rerun") is False
        and lock.get("frozen_selector") == FROZEN_SELECTOR
        and lock.get("user_authority") == EXPECTED_USER_AUTHORITY
        and lock.get("execution_authority") == EXPECTED_AUTHORITY
        and lock.get("resource_budget") == EXPECTED_RESOURCE_BUDGET
        and lock.get("one_shot_policy") == EXPECTED_ONE_SHOT_POLICY
        and lock.get("phase_a_terminal_content_sha256") == PHASE_A_TERMINAL_CONTENT_SHA256
        and lock.get("phase_a_audit_content_sha256") == PHASE_A_AUDIT_CONTENT_SHA256
        and lock.get("implementation_on_origin_master") is True
        and _commit_is_on_master(lock.get("implementation_commit")),
        "R11_TOP24_VALIDATION_LOCK",
        "execution-lock identity, authority, budget, or predecessor drift",
    )
    _validate_bindings(lock)
    _validate_frozen_selector_artifact()
    lock["_path"] = lock_path
    return lock


def _validate_phase_a_predecessor() -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_path = _repo_path(PHASE_A_TERMINAL)
    completion_path = _repo_path(PHASE_A_COMPLETION)
    audit_path = _repo_path(PHASE_A_AUDIT)
    require(
        terminal_path.is_file()
        and completion_path.is_file()
        and audit_path.is_file()
        and materializer.sha256_file(terminal_path) == PHASE_A_TERMINAL_FILE_SHA256
        and materializer.sha256_file(completion_path) == PHASE_A_COMPLETION_FILE_SHA256
        and materializer.sha256_file(audit_path) == PHASE_A_AUDIT_FILE_SHA256
        and {path.name for path in audit_path.parent.iterdir()} == {audit_path.name},
        "R11_TOP24_VALIDATION_PHASE_A_FILES",
        "sealed Phase-A predecessor file drift",
    )
    terminal = _validate_seal(
        _load_json(terminal_path),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_terminal.v1",
    )
    completion = _validate_seal(
        _load_json(completion_path),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
    )
    audit = _validate_seal(
        _load_json(audit_path),
        "blindassist.taro.o1r.r11_phase_a_validator_round12_audit.v1",
    )
    original = audit.get("original_validator_result", {})
    require(
        terminal["content_sha256"] == PHASE_A_TERMINAL_CONTENT_SHA256
        and completion["content_sha256"] == PHASE_A_COMPLETION_CONTENT_SHA256
        and audit["content_sha256"] == PHASE_A_AUDIT_CONTENT_SHA256
        and terminal.get("terminal") == "TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
        and terminal.get("passed") is terminal.get("execution_valid") is True
        and terminal.get("file_count_before_terminal") == PHASE_A_PRE_TERMINAL_FILE_COUNT
        and completion.get("parent_count") == PARENT_COUNT
        and completion.get("frame_count") == FRAME_COUNT
        and completion.get("query_count") == QUERY_COUNT
        and completion.get("source_frame_hash_sequence_sha256")
        and completion.get("unknown_is_negative") is False
        and completion.get("faro_reads") == completion.get("truth_reads") == 0
        and completion.get("highres_depth_member_payload_reads") == 0
        and audit.get("status") == "TARO_O1R_R11_PHASE_A_OFFLINE_VALIDATOR_ROUND12_REPAIR_PASS"
        and audit.get("execution_validity") == "VALID_WITH_POST_TERMINAL_NUMERIC_REPRESENTATION_REPAIR"
        and audit.get("same_sealed_phase_a_root") == PHASE_A_ROOT
        and audit.get("phase_a_root_modified") is audit.get("model_rerun") is False
        and original.get("passed") is True
        and original.get("root_file_count") == PHASE_A_FILE_COUNT,
        "R11_TOP24_VALIDATION_PHASE_A",
        "Phase-A repaired PASS binding or firewall drift",
    )
    return terminal, completion


def _frame_rows() -> list[tuple[str, str, str]]:
    inventory = _validate_seal(
        _load_json(_repo_path(INVENTORY)),
        "blindassist.taro.o1r.r11_fresh_pool_inventory.v1",
    )
    parents = inventory.get("parents")
    require(
        inventory["content_sha256"] == INVENTORY_CONTENT_SHA256
        and inventory.get("parent_count") == PARENT_COUNT
        and inventory.get("exact_pose_bounded_frame_count") == FRAME_COUNT
        and isinstance(parents, list),
        "R11_TOP24_VALIDATION_INVENTORY",
        "inventory drift",
    )
    identities: list[tuple[str, str]] = []
    counts: list[int] = []
    rows: list[tuple[str, str, str]] = []
    for parent in parents:
        identity = (str(parent.get("visit_id")), str(parent.get("video_id")))
        tokens = parent.get("frame_plan", {}).get("exact_timestamp_tokens")
        require(isinstance(tokens, list), "R11_TOP24_VALIDATION_FRAME_PLAN", "frame tokens missing")
        identities.append(identity)
        counts.append(len(tokens))
        rows.extend((identity[0], identity[1], str(token)) for token in tokens)
    require(
        identities == list(EXPECTED_IDENTITIES)
        and counts == FROZEN_FRAME_COUNTS
        and len(rows) == FRAME_COUNT,
        "R11_TOP24_VALIDATION_FRAME_PLAN",
        "frozen 48-parent/1043-frame plan drift",
    )
    return rows


def _load_phase_a_sources(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    root = _repo_path(PHASE_A_ROOT)
    for parent_id, video_id, token in _frame_rows():
        receipt = _validate_seal(
            _load_json(root / f"phase-a-sources/{parent_id}/{video_id}/{token}.json"),
            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
        )
        lineage = _validate_seal(
            _load_json_gzip(root / f"phase-a-lineage/{parent_id}/{video_id}/{token}.json.gz"),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
        )
        source = r7_canary.validate_source_frame_record(lineage.get("r7_source_frame_record"))
        base = r7_positive.validate_positive_occupancy_factor(lineage.get("r7_positive_factor_bundle"))
        candidate = abstention_candidate.validate_abstention_bundle(lineage.get("r11_abstention_bundle"))
        physical = f"{video_id}:{token}"
        require(
            receipt.get("parent_id") == source.get("parent_id") == base.get("parent_id") == candidate.get("parent_id") == parent_id
            and receipt.get("video_id") == source.get("video_id") == base.get("video_id") == candidate.get("video_id") == video_id
            and receipt.get("physical_frame_id") == source.get("physical_frame_id") == physical
            and lineage.get("source_frame_receipt_sha256") == receipt["content_sha256"]
            and source.get("source_frame_receipt_sha256") == receipt["content_sha256"]
            and base.get("source_frame_record_sha256") == candidate.get("source_frame_record_sha256") == source["content_sha256"]
            and receipt.get("highres_depth_member_payload_read") is receipt.get("faro_payload_read") is receipt.get("truth_payload_read") is False
            and lineage.get("highres_depth_member_payload_read") is lineage.get("faro_payload_read") is False
            and lineage.get("truth_inputs") == 0,
            "R11_TOP24_VALIDATION_SOURCE_LINEAGE",
            "Phase-A source/R7/R11 lineage or firewall drift",
        )
        base_rows, candidate_rows = base.get("query_results"), candidate.get("query_results")
        require(
            isinstance(base_rows, list)
            and isinstance(candidate_rows, list)
            and len(base_rows) == len(candidate_rows) == 9
            and all(
                left.get("query_id") == right.get("query_id")
                and left.get("grid_index") == right.get("grid_index") == index
                and (right.get("state") != "OCCUPIED_OBSERVED" or left.get("state") == "OCCUPIED_OBSERVED")
                for index, (left, right) in enumerate(zip(base_rows, candidate_rows, strict=True))
            ),
            "R11_TOP24_VALIDATION_FACTOR_SUBSET",
            "R11 ordered occupied subset of R7 drift",
        )
        sources.append(source)
    require(
        len(sources) == FRAME_COUNT
        and adapter.canonical_sha256([row["content_sha256"] for row in sources])
        == completion.get("source_frame_hash_sequence_sha256"),
        "R11_TOP24_VALIDATION_SOURCE_SEQUENCE",
        "Phase-A source sequence drift",
    )
    return sources


def _score_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(sources) == FRAME_COUNT, "R11_TOP24_VALIDATION_SOURCE_COUNT", "source count drift")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        require(
            source.get("source_phase_has_label_input") is False
            and source.get("training_steps") == source.get("network_requests") == 0,
            "R11_TOP24_VALIDATION_SOURCE_FIREWALL",
            "source record crosses source-only firewall",
        )
        grouped[(str(source.get("parent_id")), str(source.get("video_id")))].append(source)
    require(set(grouped) == set(EXPECTED_IDENTITIES), "R11_TOP24_VALIDATION_ROSTER", "parent roster drift")
    scores: list[dict[str, Any]] = []
    for identity in EXPECTED_IDENTITIES:
        rows = grouped[identity]
        available = eligible = 0
        hashes: list[str] = []
        for source in rows:
            features = source.get("query_features")
            digest = source.get("content_sha256")
            require(
                isinstance(features, list)
                and len(features) == 9
                and isinstance(digest, str)
                and len(digest) == 64,
                "R11_TOP24_VALIDATION_SOURCE_RECORD",
                "source feature/hash drift",
            )
            hashes.append(digest)
            for feature in features:
                require(isinstance(feature, Mapping), "R11_TOP24_VALIDATION_QUERY", "query must be an object")
                available += feature.get("query_receipt") is not None
                eligible += clear_enrichment_fit.eligible(feature, FROZEN_RULE)
        fraction = 0.0 if available == 0 else round(eligible / available, adapter.FLOAT_DECIMALS)
        scores.append(
            {
                **FROZEN_SELECTOR,
                "parent_id": identity[0],
                "video_id": identity[1],
                "frame_count": len(rows),
                "query_count": len(rows) * 9,
                "available_query_count": int(available),
                "eligible_query_count": int(eligible),
                "eligible_fraction_of_available": fraction,
                "source_frame_hash_sequence_sha256": adapter.canonical_sha256(hashes),
                "tie_break_sha256": adapter.canonical_sha256([*identity]),
                **{field: 0 for field in ZERO_FIELDS},
                "clear_output_emitted": False,
                "unknown_is_negative": False,
            }
        )
    require(
        [row["frame_count"] for row in scores] == FROZEN_FRAME_COUNTS
        and sum(row["query_count"] for row in scores) == QUERY_COUNT,
        "R11_TOP24_VALIDATION_COUNTS",
        "48-parent/1043-frame/9387-query totals drift",
    )
    return scores


def _rank(scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (copy.deepcopy(dict(row)) for row in scores),
        key=lambda row: (-int(row["eligible_query_count"]), str(row["tie_break_sha256"])),
    )


def _validate_output_receipts(root: Path, terminal: Mapping[str, Any]) -> None:
    expected = {"execution-receipt.json", "parent-scores.json", "selection.json", "terminal.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    files = terminal.get("files")
    require(
        actual == expected
        and isinstance(files, Mapping)
        and set(files) == expected - {"terminal.json"}
        and terminal.get("file_count_before_terminal") == 3,
        "R11_TOP24_VALIDATION_ROOT_SET",
        "success root is not the exact four-file terminal-last set",
    )
    total = 0
    for relative, receipt in files.items():
        path = root / relative
        require(
            isinstance(receipt, Mapping)
            and dict(receipt) == _file_receipt(path, relative),
            "R11_TOP24_VALIDATION_RECEIPT",
            f"output receipt drift: {relative}",
        )
        total += path.stat().st_size
    require(total == terminal.get("bytes_before_terminal"), "R11_TOP24_VALIDATION_BYTES", "byte ledger drift")


def validate_evidence(
    root: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    evidence_root = (root or _repo_path(EVIDENCE_ROOT)).resolve()
    require(evidence_root.is_dir(), "R11_TOP24_VALIDATION_ROOT", "selection root missing")
    terminal = _validate_seal(_load_json(evidence_root / "terminal.json"), TERMINAL_SCHEMA)
    require(
        set(terminal) == TERMINAL_FIELDS
        and terminal.get("terminal") == PASS_TERMINAL
        and terminal.get("passed") is terminal.get("execution_valid") is terminal.get("one_shot_consumed") is True,
        "R11_TOP24_VALIDATION_TERMINAL",
        "PASS terminal identity drift",
    )
    _validate_output_receipts(evidence_root, terminal)
    lock = validate_execution_lock(lock_path or _repo_path(LOCK_RELATIVE))
    _phase_a_terminal, completion = _validate_phase_a_predecessor()
    execution = _validate_seal(_load_json(evidence_root / "execution-receipt.json"), EXECUTION_SCHEMA)
    scores_record = _validate_seal(_load_json(evidence_root / "parent-scores.json"), SCORES_SCHEMA)
    selection = _validate_seal(_load_json(evidence_root / "selection.json"), SELECTION_SCHEMA)
    result = _validate_seal(terminal.get("result"), RESULT_SCHEMA)
    require(
        set(execution) == EXECUTION_FIELDS
        and execution.get("execution_lock_sha256") == materializer.sha256_file(lock["_path"])
        and execution.get("execution_lock_content_sha256") == lock["content_sha256"]
        and execution.get("phase_a_root") == PHASE_A_ROOT
        and execution.get("phase_a_audit_content_sha256") == PHASE_A_AUDIT_CONTENT_SHA256
        and execution.get("frozen_selector") == FROZEN_SELECTOR
        and execution.get("one_shot_consumed_on_root_creation") is True
        and _zeros(execution),
        "R11_TOP24_VALIDATION_EXECUTION",
        "execution receipt lock/predecessor/firewall drift",
    )

    sources = _load_phase_a_sources(completion)
    expected_scores = _score_sources(sources)
    ranked = _rank(expected_scores)
    selected = ranked[:SELECTED_PARENT_COUNT]
    require(
        set(scores_record) == SCORES_FIELDS
        and scores_record.get("selector") == FROZEN_SELECTOR
        and scores_record.get("phase_a_completion_sha256") == completion["content_sha256"]
        and scores_record.get("source_frame_hash_sequence_sha256") == completion["source_frame_hash_sequence_sha256"]
        and scores_record.get("parent_count") == PARENT_COUNT
        and scores_record.get("frame_count") == FRAME_COUNT
        and scores_record.get("query_count") == QUERY_COUNT
        and scores_record.get("parent_scores") == expected_scores
        and scores_record.get("ranked_parent_identities") == [[row["parent_id"], row["video_id"]] for row in ranked]
        and scores_record.get("all_48_source_records_sealed_before_scoring") is True
        and scores_record.get("all_48_parent_scores_sealed_before_faro") is True
        and _zeros(scores_record),
        "R11_TOP24_VALIDATION_SCORES",
        "independently recomputed eligible counts or ranking drift",
    )
    require(
        all(set(row) == SCORE_FIELDS and _zeros(row) and row.get("unknown_is_negative") is False for row in expected_scores)
        and set(selection) == SELECTION_FIELDS
        and selection.get("selector") == FROZEN_SELECTOR
        and selection.get("parent_scores_sha256") == scores_record["content_sha256"]
        and selection.get("selected_parent_count") == SELECTED_PARENT_COUNT
        and selection.get("selected_parent_identities") == [[row["parent_id"], row["video_id"]] for row in selected]
        and selection.get("selected_parent_scores") == selected
        and selection.get("selection_sealed_before_faro") is True
        and selection.get("read_unselected_faro") is False
        and selection.get("source_reselection_after_faro") is False
        and selection.get("parent_reselection_after_faro") is False
        and selection.get("candidate_or_threshold_reselection_after_faro") is False
        and selection.get("unknown_is_negative") is False
        and _zeros(selection),
        "R11_TOP24_VALIDATION_SELECTION",
        "top24/tie-break/selection firewall drift",
    )

    resource = result.get("resource_budget")
    require(
        set(result) == RESULT_FIELDS
        and result.get("terminal") == PASS_TERMINAL
        and result.get("passed") is result.get("execution_valid") is True
        and result.get("parent_count") == PARENT_COUNT
        and result.get("frame_count") == FRAME_COUNT
        and result.get("query_count") == QUERY_COUNT
        and result.get("selected_parent_count") == SELECTED_PARENT_COUNT
        and result.get("selected_parent_identities") == selection["selected_parent_identities"]
        and result.get("phase_a_terminal_content_sha256") == PHASE_A_TERMINAL_CONTENT_SHA256
        and result.get("phase_a_completion_sha256") == completion["content_sha256"]
        and result.get("phase_a_audit_content_sha256") == PHASE_A_AUDIT_CONTENT_SHA256
        and result.get("parent_scores_sha256") == scores_record["content_sha256"]
        and result.get("selection_sha256") == selection["content_sha256"]
        and result.get("all_48_source_records_sealed_before_scoring") is True
        and result.get("all_48_parent_scores_sealed_before_faro") is True
        and result.get("selection_sealed_before_faro") is True
        and result.get("unknown_is_negative") is False
        and result.get("phase_a_prior_file_validations") == PHASE_A_PRE_TERMINAL_FILE_COUNT
        and isinstance(result.get("phase_a_prior_bytes_validated"), int)
        and result["phase_a_prior_bytes_validated"] > 0
        and result.get("phase_a_lineage_decodes") == FRAME_COUNT
        and result.get("phase_a_source_receipt_decodes") == FRAME_COUNT
        and result.get("source_frame_records_validated") == FRAME_COUNT
        and result.get("query_features_scored") == QUERY_COUNT
        and _zeros(result)
        and resource == EXPECTED_RESOURCE_BUDGET
        and isinstance(result.get("elapsed_seconds"), (int, float))
        and 0 <= result["elapsed_seconds"] <= resource["maximum_wall_seconds"]
        and isinstance(result.get("peak_rss_bytes"), int)
        and 0 < result["peak_rss_bytes"] <= resource["maximum_peak_rss_bytes"]
        and result.get("one_shot_consumed") is True
        and result.get("unique_successor") == "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK"
        and result.get("claim_ceiling") == EXPECTED_CLAIM_CEILING,
        "R11_TOP24_VALIDATION_RESULT",
        "result lineage/count/resource/firewall drift",
    )
    return {
        "schema": "blindassist.taro.o1r.r11_fresh_pool_top24_selection_independent_validation.v1",
        "passed": True,
        "terminal": PASS_TERMINAL,
        "root_file_count": 4,
        "parent_count": PARENT_COUNT,
        "frame_count": FRAME_COUNT,
        "query_count": QUERY_COUNT,
        "selected_parent_count": SELECTED_PARENT_COUNT,
        "selected_parent_identities": selection["selected_parent_identities"],
        "independently_recomputed_eligible_counts": True,
        "faro_reads": 0,
        "truth_reads": 0,
        "unknown_is_negative": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_repo_path(EVIDENCE_ROOT))
    parser.add_argument("--execution-lock", type=Path, default=_repo_path(LOCK_RELATIVE))
    args = parser.parse_args(argv)
    try:
        result = validate_evidence(args.root, args.execution_lock)
    except Exception as error:  # noqa: BLE001 - validator must render fail-closed errors
        print(json.dumps({"passed": False, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
