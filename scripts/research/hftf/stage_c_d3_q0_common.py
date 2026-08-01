#!/usr/bin/env python3
"""Standard-library-only state and contract helpers for Stage C D3-Q0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import ast
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_screening_execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_Q0_INVALID_BEFORE_ANY_Q0_1_SLOT_2_MEDIA_SUPPORT_OR_TRUTH"
)
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_d3_reference_and_support_only_"
    "challenge_qualification_q0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_D2_NOT_EVALUABLE_BEFORE_ANY_D3_SOURCE_"
    "MEDIA_SUPPORT_OR_TRUTH_OUTCOME"
)
ROSTER_SCHEMA = "blindassist_hftf_stage_c_d3_q0_metadata_roster"
ROSTER_TERMINAL = "D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED"
SCREENING_ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_screening_attempt"
)
SCREENING_ATTEMPT_STATUS = (
    "D3_Q0_SCREENING_ATTEMPT_FSYNCED_BEFORE_FIRST_"
    "SLOT_CONTENT_REQUEST"
)
SLOT_ATTEMPT_SCHEMA = "blindassist_hftf_stage_c_d3_q0_1_slot_attempt"
SLOT_ATTEMPT_STATUS = (
    "D3_Q0_SLOT_ATTEMPT_FSYNCED_BEFORE_FIRST_POSE_OR_MEDIA_REQUEST"
)
AGGREGATE_ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_aggregate_attempt"
)
AGGREGATE_ATTEMPT_STATUS = (
    "AGGREGATE_ATTEMPT_FSYNCED_BEFORE_FIRST_SELECTOR_OR_FAILURE_READ"
)
SELECTOR_SCHEMA = "blindassist_hftf_stage_c_d3_q0_1_slot_selector"
SELECTOR_QUALIFIED = (
    "D3_Q0_SLOT_REFERENCE_SUPPORT_OPPORTUNITY_QUALIFIED"
)
SELECTOR_NOT_QUALIFIED = (
    "D3_Q0_SLOT_REFERENCE_SUPPORT_OPPORTUNITY_NOT_QUALIFIED"
)
FAILURE_SCHEMA = "blindassist_hftf_stage_c_d3_q0_1_slot_failure"
SLOT_FAILURE_TERMINAL = (
    "D3_QUALIFICATION_SLOT_NOT_EVALUABLE_CONSUME_SLOT_"
    "CONTINUE_FROZEN_ORDER"
)
QUALIFICATION_TERMINAL = (
    "D3_Q0_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_QUALIFIED"
)
SELECTION_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_screening_selection"
)
BUDGET_TERMINAL = (
    "D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_"
    "BUDGET_EXHAUSTED_NO_EXPANSION"
)
CANONICAL_RELATIVE_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d3-q0-1-screening-20260802"
)
SCREENING_ROOT_RELATIVE = CANONICAL_RELATIVE_ROOT
SLOT_COUNT = 40
FIRST_ACTIVE_SLOT_INDEX = 2
MAXIMUM_NEWLY_OPENED_SLOTS = 39
REQUIRED_QUALIFIED = 6
MAX_PATH_CHARS_EXCLUSIVE = 240
SESSION_ID_RE = re.compile(r"[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
STRATA = (
    ("body", 0.4),
    ("body", 0.8),
    ("head", 0.4),
    ("head", 0.8),
)
AUTHORITY_HASH_KEYS = {
    "slot_attempt_sha256",
    "description_sha256",
    "pose_sha256",
    "rgb_receipts_sha256",
    "mask_receipts_sha256",
    "depth_receipts_sha256",
    "content_index_sha256",
    "sealed_payload_sha256",
}
ROSTER_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_metadata_roster_result"
)
Q0_INVALID_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_screening_invalid_result"
)
Q0_INVALID_TERMINAL = "D3_QUALIFICATION_INVALID_STOP"
Q0_EXECUTION_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_screening_execution_contract"
)
Q0_EXECUTION_CONTRACT_STATUS = (
    "FROZEN_AFTER_D3_Q0_ROSTER_BEFORE_ANY_D3_MEDIA_SUPPORT_OR_TRUTH"
)
CARRY_FORWARD_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_prior_invalid_slot_burn"
)
CARRY_FORWARD_TERMINAL = (
    "D3_Q0_1_SLOT_1_CARRY_FORWARD_BURNED_FROM_Q0_INVALID"
)
D2_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_causal_signed_clearance_transport_result"
)
D2_RESULT_TERMINAL = (
    "D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT"
)
D2_MECHANICS_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_mechanics_execution_contract"
)
D2_MECHANICS_CONTRACT_STATUS = (
    "FROZEN_AFTER_D2_MEDIA_ACQUISITION_BEFORE_PREPROCESSOR_OR_TRUTH_OUTCOME"
)
D2_DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_transport_d2"
)
D2_DESIGN_STATUS = "FROZEN_BEFORE_D2_METADATA_SCAN_OR_SOURCE_OUTCOME"
D2_1_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_"
    "transport_clarification_d2_1"
)
D2_1_STATUS = (
    "FROZEN_AFTER_METADATA_ONLY_COHORT_LOCK_BEFORE_ANY_D2_MEDIA_"
    "POSE_CONTENT_OR_MECHANICS_OUTCOME"
)
G0_SCHEMA = "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
MECHANICS_SCHEMA = (
    "blindassist_hftf_stage_b_swept_envelope_label_mechanics_canary_d0"
)
MECHANICS_STATUS = "FROZEN_DEVELOPMENT_CANARY_RESULT_NOT_RUN"
REQUIRED_IMPLEMENTATION_KEYS = {
    "d3_common",
    "next_slot_runner",
    "screening_aggregator",
    "selected_future_blind_preprocessor",
    "sealed_effect_evaluator",
    "d2_transport_acquirer",
    "transport_dependency",
    "d2_mechanics_common",
    "d2_future_blind_preprocessor",
    "d2_effect_evaluator",
    "swept_probe_mechanics",
    "probe_visibility_mechanics",
    "exact_g0_signed_clearance_runner",
    "exact_g0_geometry_primitives",
    "frozen_sanpo_pose_and_ground_authority",
}
REQUIRED_TEST_KEYS = {"state_test", "pipeline_test"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def durable_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def durable_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(durable_json_bytes(value)).hexdigest()


def test_definition_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json_exclusive_fsync(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path = path.resolve()
    temporary = path.with_name(path.name + ".tmp")
    encoded = durable_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing non-exclusive JSON write: {path} or {temporary}"
        )
    created_temporary = False
    try:
        with temporary.open("xb") as handle:
            created_temporary = True
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(
                f"Refusing to replace existing JSON artifact: {path}"
            )
        os.replace(temporary, path)
        created_temporary = False
    except BaseException:
        if created_temporary:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def preserve_temporary_artifact(path: Path) -> Path | None:
    final = path.resolve()
    temporary = final.with_name(final.name + ".tmp")
    if not temporary.exists():
        return None
    preserved = final.with_name(final.name + ".orphan")
    if preserved.exists():
        raise FileExistsError(
            f"Refusing to overwrite preserved temporary artifact: {preserved}"
        )
    temporary.rename(preserved)
    return preserved


def _resolve(
    value: str,
    contract_path: Path,
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (repo_root() / candidate).resolve()
    local_candidate = (contract_path.parent / candidate).resolve()
    if root_candidate.exists() or not local_candidate.exists():
        return root_candidate
    return local_candidate


def _closed_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_tracked_clean(path: Path) -> None:
    relative = path.resolve().relative_to(repo_root()).as_posix()
    if not _git("ls-files", "--error-unmatch", "--", relative):
        raise ValueError(f"Execution file is not tracked: {relative}")
    if _git("diff", "--name-only", "--", relative):
        raise ValueError(f"Execution file has unstaged changes: {relative}")
    if _git("diff", "--cached", "--name-only", "--", relative):
        raise ValueError(f"Execution file has staged changes: {relative}")


def _bound_json(
    contract: dict[str, Any],
    contract_path: Path,
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = contract.get("parents", {}).get(key)
    if not isinstance(receipt, dict):
        raise ValueError(f"Missing execution-contract parent: {key}")
    _closed_keys(receipt, {"path", "sha256"}, f"parent {key}")
    path = _resolve(str(receipt["path"]), contract_path)
    if sha256(path) != receipt["sha256"]:
        raise ValueError(f"Parent hash mismatch: {key}")
    return path, load_json(path)


def _validate_roster(roster: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        roster.get("schema") != ROSTER_SCHEMA
        or roster.get("terminal") != ROSTER_TERMINAL
        or roster.get("workflow_profile") != "THESIS_DEVELOPMENT"
        or roster.get("requested_roster_slot_count") != SLOT_COUNT
        or roster.get("locked_roster_slot_count") != SLOT_COUNT
        or roster.get("exclusions", {}).get("excluded_parent_count") != 84
    ):
        raise ValueError("D3-Q0 roster identity mismatch")
    slots = roster.get("locked_slots")
    if not isinstance(slots, list) or len(slots) != SLOT_COUNT:
        raise ValueError("D3-Q0 roster must contain exactly 40 slots")
    ids = [str(slot.get("session_id", "")) for slot in slots]
    indices = [slot.get("d3_roster_slot_index") for slot in slots]
    if (
        any(SESSION_ID_RE.fullmatch(value) is None for value in ids)
        or ids != sorted(ids)
        or len(set(ids)) != SLOT_COUNT
        or indices != list(range(1, SLOT_COUNT + 1))
        or any(
            slot.get("role") != "d3_q0_locked_truth_screening_slot"
            or slot.get("media_pose_content_opened") is not False
            or slot.get("support_or_truth_computed") is not False
            or slot.get("effect_computed") is not False
            for slot in slots
        )
    ):
        raise ValueError("D3-Q0 roster slot order or firewall mismatch")
    firewall = roster.get("firewall", {})
    if not firewall or any(value is not False for value in firewall.values()):
        raise ValueError("D3-Q0 roster firewall is not closed")
    authorization = roster.get("authorization", {})
    if (
        authorization.get("freeze_qualifier_effect_execution_contract")
        is not True
        or any(
            value is not False
            for key, value in authorization.items()
            if key != "freeze_qualifier_effect_execution_contract"
        )
    ):
        raise ValueError("D3-Q0 roster authorization mismatch")
    return slots


def _validate_path_lengths(
    root: Path,
    slots: list[dict[str, Any]],
) -> None:
    paths = [root, *aggregate_paths(root).values()]
    tokens: list[str] = []
    for slot in slots:
        layout = slot_layout(root, slot)
        paths.extend(layout.values())
        token = layout["slot_root"].name.rsplit("-", 1)[-1]
        tokens.append(token)
        paths.extend(
            (
                layout["slot_root"] / "content" / "pose.csv",
                layout["slot_root"] / "content" / "p" / f"{index:02x}.json",
            )
            for index in range(13)
        )
        paths.extend(
            (
                layout["slot_root"]
                / "content"
                / "m"
                / f"{index:02x}.png",
                layout["slot_root"]
                / "content"
                / "d"
                / f"{index:02x}.f16.gz",
            )
            for index in range(2, 13)
        )
        prediction_root = root / "formal" / "predictions" / "s" / token
        for anchor in range(2, 9):
            paths.extend(
                (
                    prediction_root / f"anchor-{anchor}.json",
                    prediction_root / f"anchor-{anchor}.points.npy",
                )
            )
    paths.extend(
        (
            root / "formal" / "predictions" / "attempt.json",
            root / "formal" / "predictions" / "completion.json",
            root / "formal" / "predictions" / "failure.json",
            root / "formal" / "effect" / "attempt.json",
            root
            / "formal"
            / "effect"
            / "sealed-payload-open-once.json",
            root / "formal" / "effect" / "result.json",
            root / "formal" / "effect" / "failure.json",
        )
    )
    flattened: list[Path] = []
    for value in paths:
        if isinstance(value, tuple):
            flattened.extend(value)
        else:
            flattened.append(value)
    paths = [
        child
        for path in flattened
        for child in (
            path,
            path.with_name(path.name + ".tmp"),
            path.with_name(path.name + ".orphan"),
        )
    ]
    if any(not path.is_absolute() for path in paths):
        raise ValueError("D3-Q0 canonical path preflight received a relative path")
    path_strings = [str(path) for path in paths]
    if len(set(tokens)) != SLOT_COUNT:
        raise ValueError("D3-Q0 short slot token collision")
    session_ids = [str(slot["session_id"]) for slot in slots]
    if any(
        session_id in path
        for session_id in session_ids
        for path in path_strings
    ):
        raise ValueError("D3-Q0 canonical path exposes a full session ID")
    too_long = [
        path
        for path in path_strings
        if len(path) >= MAX_PATH_CHARS_EXCLUSIVE
    ]
    if too_long:
        raise ValueError(
            "D3-Q0 canonical layout contains path >=240 chars: "
            + too_long[0]
        )


def validate_execution_contract(
    path: Path,
    implementation_key: str,
    implementation_path: Path,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = path.resolve()
    contract = load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise ValueError("D3-Q0 screening contract identity mismatch")
    protocol_path, protocol = _bound_json(
        contract,
        contract_path,
        "d3_q0_protocol",
    )
    roster_path, roster = _bound_json(
        contract,
        contract_path,
        "metadata_roster",
    )
    roster_result_path, roster_result = _bound_json(
        contract,
        contract_path,
        "metadata_roster_result",
    )
    q0_invalid_result_path, q0_invalid_result = _bound_json(
        contract,
        contract_path,
        "q0_invalid_result",
    )
    q0_execution_contract_path, q0_execution_contract = _bound_json(
        contract,
        contract_path,
        "q0_execution_contract",
    )
    d2_result_path, d2_result = _bound_json(
        contract,
        contract_path,
        "d2_result",
    )
    d2_contract_path, d2_contract = _bound_json(
        contract,
        contract_path,
        "d2_mechanics_contract",
    )
    d2_design_path, d2_design = _bound_json(
        contract,
        contract_path,
        "d2_design",
    )
    d2_1_path, d2_1 = _bound_json(
        contract,
        contract_path,
        "d2_1_clarification",
    )
    g0_path, g0 = _bound_json(
        contract,
        contract_path,
        "g0_signed_clearance_definition",
    )
    mechanics_path, mechanics = _bound_json(
        contract,
        contract_path,
        "swept_envelope_mechanics",
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("source_pool", {}).get(
            "maximum_truth_screened_slots"
        )
        != SLOT_COUNT
        or protocol.get("source_pool", {}).get(
            "required_qualified_sources"
        )
        != REQUIRED_QUALIFIED
    ):
        raise ValueError("D3-Q0 protocol parent mismatch")
    durable_roster = roster_result.get("durable_evidence", {}).get(
        "roster", {}
    )
    durable_roster_path = _resolve(
        str(durable_roster.get("path", "")),
        roster_result_path,
    )
    if (
        roster_result.get("schema") != ROSTER_RESULT_SCHEMA
        or roster_result.get("terminal") != ROSTER_TERMINAL
        or durable_roster_path != roster_path
        or durable_roster.get("sha256") != sha256(roster_path)
        or durable_roster.get("required_terminal") != ROSTER_TERMINAL
        or roster_result.get("authorization", {}).get(
            "freeze_qualifier_and_effect_execution_contract"
        )
        is not True
        or q0_invalid_result.get("schema") != Q0_INVALID_RESULT_SCHEMA
        or q0_invalid_result.get("terminal") != Q0_INVALID_TERMINAL
        or q0_invalid_result.get("invalidity", {}).get(
            "slot_1_permanently_burned"
        )
        is not True
        or q0_invalid_result.get("successor_boundary", {}).get(
            "maximum_remaining_slots"
        )
        != MAXIMUM_NEWLY_OPENED_SLOTS
        or q0_invalid_result.get("authorization", {}).get(
            "freeze_schema_only_q0_1_successor_contract"
        )
        is not True
        or q0_execution_contract.get("schema")
        != Q0_EXECUTION_CONTRACT_SCHEMA
        or q0_execution_contract.get("status")
        != Q0_EXECUTION_CONTRACT_STATUS
        or q0_invalid_result.get("execution_contract", {}).get("sha256")
        != sha256(q0_execution_contract_path)
        or q0_invalid_result.get("execution_contract", {}).get("path")
        != str(
            q0_execution_contract_path.relative_to(
                contract_path.parents[3]
            )
        ).replace("\\", "/")
        or d2_result.get("schema") != D2_RESULT_SCHEMA
        or d2_result.get("terminal") != D2_RESULT_TERMINAL
        or d2_result.get("offline_validation_summary", {}).get(
            "effect_gates_evaluated"
        )
        is not False
        or d2_contract.get("schema") != D2_MECHANICS_CONTRACT_SCHEMA
        or d2_contract.get("status") != D2_MECHANICS_CONTRACT_STATUS
        or d2_design.get("schema") != D2_DESIGN_SCHEMA
        or d2_design.get("status") != D2_DESIGN_STATUS
        or d2_1.get("schema") != D2_1_SCHEMA
        or d2_1.get("status") != D2_1_STATUS
        or g0.get("schema") != G0_SCHEMA
        or mechanics.get("schema") != MECHANICS_SCHEMA
        or mechanics.get("status") != MECHANICS_STATUS
    ):
        raise ValueError("D3-Q0 scientific parent identity mismatch")
    slots = _validate_roster(roster)
    implementations = contract.get("implementations")
    if (
        not isinstance(implementations, dict)
        or set(implementations) != REQUIRED_IMPLEMENTATION_KEYS
        or implementation_key not in implementations
    ):
        raise ValueError("D3-Q0 implementation receipts are incomplete")
    implementation_paths: dict[str, Path] = {}
    for key, receipt in implementations.items():
        if not isinstance(receipt, dict):
            raise ValueError(f"Invalid implementation receipt: {key}")
        _closed_keys(
            receipt,
            {"path", "sha256"},
            f"implementation {key}",
        )
        implementation = _resolve(str(receipt["path"]), contract_path)
        if sha256(implementation) != receipt["sha256"]:
            raise ValueError(f"Implementation hash mismatch: {key}")
        implementation_paths[str(key)] = implementation
    if implementation_paths[implementation_key] != implementation_path.resolve():
        raise ValueError("Caller implementation path differs from receipt")
    tests = contract.get("implementation_tests")
    if not isinstance(tests, dict) or set(tests) != REQUIRED_TEST_KEYS:
        raise ValueError("D3-Q0 implementation test receipts are incomplete")
    test_paths: list[Path] = []
    for key, receipt in tests.items():
        if not isinstance(receipt, dict):
            raise ValueError(f"Invalid test receipt: {key}")
        _closed_keys(
            receipt,
            {"path", "sha256", "test_count", "tests_passed"},
            f"test {key}",
        )
        test_path = _resolve(str(receipt["path"]), contract_path)
        count = test_definition_count(test_path)
        if (
            sha256(test_path) != receipt["sha256"]
            or receipt["test_count"] != count
            or receipt["tests_passed"] != count
        ):
            raise ValueError(f"D3-Q0 test receipt mismatch: {key}")
        test_paths.append(test_path)
    d2_implementations = d2_contract.get("implementations", {})
    d2_binding_map = {
        "d2_mechanics_common": "mechanics_common",
        "d2_future_blind_preprocessor": "future_blind_preprocessor",
        "d2_effect_evaluator": "truth_effect_evaluator",
        "swept_probe_mechanics": "swept_probe_mechanics",
        "probe_visibility_mechanics": "probe_visibility_mechanics",
    }
    for d3_key, d2_key in d2_binding_map.items():
        d2_receipt = d2_implementations.get(d2_key, {})
        if (
            implementation_paths[d3_key]
            != _resolve(str(d2_receipt.get("path", "")), d2_contract_path)
            or d2_receipt.get("sha256")
            != sha256(implementation_paths[d3_key])
        ):
            raise ValueError(f"D2 implementation binding drift: {d3_key}")
    primitive_map = {
        "exact_g0_signed_clearance_runner": (
            "exact_g0_signed_clearance_runner"
        ),
        "exact_g0_geometry_primitives": "exact_g0_geometry_primitives",
        "frozen_sanpo_pose_and_ground_authority": (
            "frozen_sanpo_pose_and_ground_authority"
        ),
    }
    for d3_key, d2_key in primitive_map.items():
        receipt = d2_1.get("implementation_receipts", {}).get(d2_key, {})
        if (
            implementation_paths[d3_key]
            != _resolve(str(receipt.get("path", "")), d2_1_path)
            or receipt.get("sha256")
            != sha256(implementation_paths[d3_key])
        ):
            raise ValueError(f"D2.1 primitive binding drift: {d3_key}")
    screening = contract.get("screening", {})
    if (
        screening.get("slot_count") != SLOT_COUNT
        or screening.get("first_active_slot_index")
        != FIRST_ACTIVE_SLOT_INDEX
        or screening.get("carry_forward_burned_slot_count") != 1
        or screening.get("maximum_newly_opened_slots")
        != MAXIMUM_NEWLY_OPENED_SLOTS
        or screening.get("required_qualified_sources")
        != REQUIRED_QUALIFIED
        or screening.get("maximum_path_chars_exclusive")
        != MAX_PATH_CHARS_EXCLUSIVE
        or screening.get("order") != "lexicographic_session_id"
        or screening.get("slot_failure_consumes_slot") is not True
        or screening.get("stop_immediately_after_sixth_qualified")
        is not True
        or screening.get("slot_replacement_authorized") is not False
        or screening.get("budget_expansion_authorized") is not False
        or screening.get("manual_skip_or_reorder_authorized") is not False
    ):
        raise ValueError("D3-Q0 screening constants mismatch")
    carry_policy = contract.get("prior_invalid_carry_forward", {})
    if not isinstance(carry_policy, dict):
        raise ValueError("Q0.1 carry-forward policy must be an object")
    _closed_keys(
        carry_policy,
        {
            "burned_slot_index",
            "first_new_media_slot_index",
            "remaining_new_media_slot_count",
            "slot_1_counts_toward_original_budget",
            "slot_1_counts_as_qualified",
            "slot_1_counts_as_not_qualified",
            "slot_1_counts_as_slot_failure",
            "prior_selector_outcome_admitted",
            "prior_artifact_bytes_may_be_reopened",
            "prior_outcome_fields_may_be_imported",
            "preserve_original_indices_and_order",
            "slot_replacement_authorized",
            "budget_expansion_authorized",
        },
        "Q0.1 carry-forward policy",
    )
    if carry_policy != {
        "burned_slot_index": 1,
        "first_new_media_slot_index": FIRST_ACTIVE_SLOT_INDEX,
        "remaining_new_media_slot_count": MAXIMUM_NEWLY_OPENED_SLOTS,
        "slot_1_counts_toward_original_budget": True,
        "slot_1_counts_as_qualified": False,
        "slot_1_counts_as_not_qualified": False,
        "slot_1_counts_as_slot_failure": False,
        "prior_selector_outcome_admitted": False,
        "prior_artifact_bytes_may_be_reopened": False,
        "prior_outcome_fields_may_be_imported": False,
        "preserve_original_indices_and_order": True,
        "slot_replacement_authorized": False,
        "budget_expansion_authorized": False,
    }:
        raise ValueError("Q0.1 carry-forward policy mismatch")
    if (
        contract.get(
            "qualification_gates_each_source_height_horizon_all_required"
        )
        != protocol.get(
            "qualification_gates_each_source_height_horizon_all_required"
        )
        or contract.get("d3_effect_skeleton")
        != protocol.get(
            "d3_effect_skeleton_that_must_be_frozen_before_qualification"
        )
        or contract.get("effect_gates_all_required")
        != d2_contract.get("effect_gates_all_required")
        or contract.get("estimand") != d2_design.get("estimand")
        or contract.get("selector_allowed_fields")
        != protocol.get("qualification_reader", {}).get(
            "qualification_receipt_allowed_fields"
        )
    ):
        raise ValueError("D3-Q0 scientific gates or estimand drifted")
    sealed = contract.get("sealed_payload_firewall", {})
    if (
        sealed.get("payload_required") is not True
        or sealed.get("payload_fsync_before_selector") is not True
        or sealed.get("selector_or_aggregator_may_read_payload")
        is not False
        or sealed.get(
            "effect_may_read_only_after_selected_six_and_predictions"
        )
        is not True
        or sealed.get("future_media_second_open_authorized") is not False
        or sealed.get("candidate_arm_clearance_in_payload_authorized")
        is not False
        or sealed.get("effect_metrics_in_payload_authorized") is not False
    ):
        raise ValueError("D3-Q0 sealed payload firewall mismatch")
    root = _resolve(
        str(
            contract.get("canonical_artifacts", {}).get(
                "screening_root",
                "",
            )
        ),
        contract_path,
    )
    expected_root = (repo_root() / CANONICAL_RELATIVE_ROOT).resolve()
    if root != expected_root:
        raise ValueError("D3-Q0 screening root is noncanonical")
    authorization = contract.get("authorization", {})
    if (
        authorization.get("reference_support_screening_authorized")
        is not True
        or authorization.get(
            "future_blind_preprocessor_after_selection_authorized"
        )
        is not True
        or authorization.get(
            "sealed_effect_after_predictions_authorized"
        )
        is not True
        or any(
            authorization.get(key) is not False
            for key in (
                "effect_before_six_source_selection_authorized",
                "future_blind_before_six_source_selection_authorized",
                "sealed_payload_open_before_predictions_authorized",
                "roster_rerun_authorized",
                "slot_replacement_authorized",
                "budget_expansion_authorized",
                "rgb_student_training_authorized",
                "rgb_student_execution_authorized",
                "reserved_official_test_open_authorized",
                "research_mainline_changed",
                "default_app_changed",
                "android_changed",
                "production_authorized",
                "safety_claim_authorized",
            )
        )
    ):
        raise ValueError("D3-Q0 screening authorization mismatch")
    failure_policy = contract.get("failure_policy", {})
    if (
        failure_policy.get("slot_execution_failure_terminal")
        != SLOT_FAILURE_TERMINAL
        or failure_policy.get("qualification_budget_terminal")
        != BUDGET_TERMINAL
        or failure_policy.get("qualification_invalid_terminal")
        != "D3_QUALIFICATION_INVALID_STOP"
        or failure_policy.get("effect_recompute_mismatch_terminal")
        != (
            "D3_NOT_EVALUABLE_QUALIFICATION_RECOMPUTE_"
            "MISMATCH_NO_REPLACEMENT"
        )
        or failure_policy.get("effect_pretruth_failure_terminal")
        != (
            "D3_Q0_EFFECT_PRETRUTH_VALIDATION_FAILED_"
            "NO_RERUN_NO_REPLACEMENT"
        )
        or failure_policy.get("effect_truth_interrupted_terminal")
        != (
            "D3_Q0_SEALED_PAYLOAD_EFFECT_INTERRUPTED_"
            "NO_SECOND_OPEN_NO_REPLACEMENT"
        )
        or failure_policy.get("preprocessor_interrupted_terminal")
        != (
            "D3_Q0_FUTURE_BLIND_PREPROCESSOR_FAILED_"
            "NO_RERUN_NO_REPLACEMENT"
        )
        or failure_policy.get("orphan_attempt_consumes_slot")
        is not True
        or failure_policy.get("slot_receipt_must_bind_attempt")
        is not True
        or failure_policy.get("aggregate_attempt_before_receipt_read")
        is not True
        or failure_policy.get(
            "hard_interruption_recovery_without_input_reopen"
        )
        is not True
        or failure_policy.get(
            "temporary_artifacts_preserved_as_orphan"
        )
        is not True
        or failure_policy.get("preserve_partial_artifacts") is not True
        or failure_policy.get("slot_rerun_authorized") is not False
        or failure_policy.get("preprocessor_rerun_authorized") is not False
        or failure_policy.get("second_sealed_payload_open_authorized")
        is not False
        or failure_policy.get("effect_rerun_authorized") is not False
        or failure_policy.get(
            "premature_effect_invocation_consumes_attempt"
        )
        is not False
    ):
        raise ValueError("D3-Q0 failure policy mismatch")
    if int(contract.get("network", {}).get("retries", -1)) != 3:
        raise ValueError("D3-Q0 network retries must be exactly three")
    outcome_firewall = contract.get("outcome_firewall_at_freeze", {})
    if (
        set(outcome_firewall)
        != {
            "q0_1_slot_2_pose_opened",
            "q0_1_slot_2_depth_opened",
            "q0_1_slot_2_mask_opened",
            "q0_1_slot_2_support_computed",
            "q0_1_slot_2_future_truth_computed",
            "q0_1_slot_2_qualification_known",
            "q0_1_prediction_known",
            "q0_1_effect_known",
            "q0_slot_1_selector_outcome_imported",
        }
        or any(value is not False for value in outcome_firewall.values())
    ):
        raise ValueError("D3-Q0 outcome firewall was not frozen closed")
    _validate_path_lengths(root, slots)
    if verify_git:
        for execution_file in [
            contract_path,
            protocol_path,
            roster_result_path,
            q0_invalid_result_path,
            q0_execution_contract_path,
            d2_result_path,
            d2_contract_path,
            d2_design_path,
            d2_1_path,
            g0_path,
            mechanics_path,
            *implementation_paths.values(),
            *test_paths,
        ]:
            _require_tracked_clean(execution_file)
        if _git("rev-parse", "HEAD") != _git(
            "rev-parse",
            "origin/master",
        ):
            raise ValueError("HEAD must equal origin/master")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": sha256(contract_path),
        "protocol": protocol,
        "protocol_path": protocol_path,
        "roster_result": roster_result,
        "roster_result_path": roster_result_path,
        "q0_invalid_result": q0_invalid_result,
        "q0_invalid_result_path": q0_invalid_result_path,
        "q0_execution_contract": q0_execution_contract,
        "q0_execution_contract_path": q0_execution_contract_path,
        "carry_forward_authority": {
            "q0_protocol_sha256": sha256(protocol_path),
            "metadata_roster_sha256": sha256(roster_path),
            "q0_execution_contract_sha256": sha256(
                q0_execution_contract_path
            ),
            "q0_invalid_result_sha256": sha256(q0_invalid_result_path),
            "q0_screening_invalid_sha256": q0_invalid_result[
                "durable_evidence"
            ]["screening_invalid"]["sha256"],
        },
        "roster": roster,
        "roster_path": roster_path,
        "roster_sha256": sha256(roster_path),
        "slots": slots,
        "d2_result": d2_result,
        "d2_contract": d2_contract,
        "d2_design": d2_design,
        "d2_1": d2_1,
        "g0": g0,
        "mechanics": mechanics,
        "root": root,
        "retries": int(contract.get("network", {}).get("retries", 3)),
    }


def slot_layout(
    root: Path,
    slot_dict: dict[str, Any],
) -> dict[str, Path]:
    index = int(slot_dict.get("d3_roster_slot_index", -1))
    session_id = str(slot_dict.get("session_id", ""))
    if (
        index < 1
        or index > SLOT_COUNT
        or SESSION_ID_RE.fullmatch(session_id) is None
    ):
        raise ValueError("Invalid D3-Q0 roster slot")
    token = hashlib.sha256(session_id.encode("ascii")).hexdigest()[:12]
    slot_root = root.resolve() / f"slot-{index:02d}-{token}"
    return {
        "slot_root": slot_root,
        "attempt": slot_root / "attempt.json",
        "content_index": slot_root / "content_index.json",
        "sealed_payload": slot_root / "sealed_payload.json",
        "selector": slot_root / "selector.json",
        "failure": slot_root / "failure.json",
        "carry_forward": slot_root / "carry_forward.json",
    }


def aggregate_paths(root: Path) -> dict[str, Path]:
    resolved = root.resolve()
    return {
        "screening_attempt": resolved / "screening_attempt.json",
        "aggregate_attempt": resolved / "aggregate_attempt.json",
        "selection": resolved / "selection.json",
        "exhausted": resolved / "budget_exhausted.json",
        "invalid": resolved / "screening_invalid.json",
    }


def validate_screening_attempt(
    value: dict[str, Any],
    contract_sha256: str,
    roster_sha256: str,
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "status",
            "workflow_profile",
            "contract_sha256",
            "roster_sha256",
            "first_slot_index",
            "first_network_request_started",
            "slot_replacement_authorized",
            "budget_expansion_authorized",
        },
        "screening attempt",
    )
    if (
        value["schema"] != SCREENING_ATTEMPT_SCHEMA
        or value["status"] != SCREENING_ATTEMPT_STATUS
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["contract_sha256"] != contract_sha256
        or value["roster_sha256"] != roster_sha256
        or value["first_slot_index"] != FIRST_ACTIVE_SLOT_INDEX
        or value["first_network_request_started"] is not False
        or value["slot_replacement_authorized"] is not False
        or value["budget_expansion_authorized"] is not False
    ):
        raise ValueError("Screening attempt identity or policy mismatch")
    return value


def validate_carry_forward(
    value: dict[str, Any],
    slot: dict[str, Any],
    q0_1_contract_sha256: str,
    authority: dict[str, str],
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "terminal",
            "workflow_profile",
            "q0_1_execution_contract_sha256",
            "q0_protocol_sha256",
            "metadata_roster_sha256",
            "q0_execution_contract_sha256",
            "q0_invalid_result_sha256",
            "q0_screening_invalid_sha256",
            "original_slot_index",
            "session_id",
            "burn_reason",
            "original_attempt_durable",
            "media_support_truth_opened",
            "selector_schema_valid",
            "selector_admitted",
            "permanently_burned",
            "counts_toward_original_40_budget",
            "counts_as_qualified",
            "counts_as_not_qualified",
            "counts_as_slot_failure",
            "reopen_authorized",
            "recompute_authorized",
            "rerun_authorized",
            "replacement_authorized",
            "first_remaining_original_slot",
            "maximum_remaining_original_slots",
            "preserve_original_indices",
            "preserve_original_order",
            "sealed_payload_read",
            "invalid_selector_read",
            "outcome_fields_imported",
        },
        "Q0.1 carry-forward burn receipt",
    )
    expected_authority_keys = {
        "q0_protocol_sha256",
        "metadata_roster_sha256",
        "q0_execution_contract_sha256",
        "q0_invalid_result_sha256",
        "q0_screening_invalid_sha256",
    }
    _closed_keys(
        authority,
        expected_authority_keys,
        "Q0.1 carry-forward authority",
    )
    if any(
        not isinstance(authority[key], str)
        or SHA256_RE.fullmatch(authority[key]) is None
        or value[key] != authority[key]
        for key in expected_authority_keys
    ):
        raise ValueError("Q0.1 carry-forward authority mismatch")
    if (
        slot.get("d3_roster_slot_index") != 1
        or value["schema"] != CARRY_FORWARD_SCHEMA
        or value["terminal"] != CARRY_FORWARD_TERMINAL
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["q0_1_execution_contract_sha256"]
        != q0_1_contract_sha256
        or value["original_slot_index"] != 1
        or value["session_id"] != slot.get("session_id")
        or value["burn_reason"]
        != "SCHEMA_INVALID_AFTER_MEDIA_SUPPORT_TRUTH_OPEN"
        or value["original_attempt_durable"] is not True
        or value["media_support_truth_opened"] is not True
        or value["selector_schema_valid"] is not False
        or value["selector_admitted"] is not False
        or value["permanently_burned"] is not True
        or value["counts_toward_original_40_budget"] is not True
        or value["counts_as_qualified"] is not False
        or value["counts_as_not_qualified"] is not False
        or value["counts_as_slot_failure"] is not False
        or value["reopen_authorized"] is not False
        or value["recompute_authorized"] is not False
        or value["rerun_authorized"] is not False
        or value["replacement_authorized"] is not False
        or value["first_remaining_original_slot"] != FIRST_ACTIVE_SLOT_INDEX
        or value["maximum_remaining_original_slots"]
        != MAXIMUM_NEWLY_OPENED_SLOTS
        or value["preserve_original_indices"] is not True
        or value["preserve_original_order"] is not True
        or value["sealed_payload_read"] is not False
        or value["invalid_selector_read"] is not False
        or value["outcome_fields_imported"] is not False
    ):
        raise ValueError("Q0.1 carry-forward burn policy mismatch")
    return value


def validate_slot_attempt(
    value: dict[str, Any],
    slot: dict[str, Any],
    contract_sha256: str,
    roster_sha256: str,
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "status",
            "workflow_profile",
            "contract_sha256",
            "roster_sha256",
            "slot_index",
            "session_id",
            "internal_retries_per_request",
            "content_request_started",
            "slot_retry_authorized",
            "source_replacement_authorized",
            "candidate_arm_clearance_authorized",
            "effect_metric_authorized",
        },
        "slot attempt",
    )
    retries = value["internal_retries_per_request"]
    if (
        value["schema"] != SLOT_ATTEMPT_SCHEMA
        or value["status"] != SLOT_ATTEMPT_STATUS
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["contract_sha256"] != contract_sha256
        or value["roster_sha256"] != roster_sha256
        or value["slot_index"] != slot["d3_roster_slot_index"]
        or value["session_id"] != slot["session_id"]
        or not isinstance(retries, int)
        or isinstance(retries, bool)
        or retries != 3
        or value["content_request_started"] is not False
        or value["slot_retry_authorized"] is not False
        or value["source_replacement_authorized"] is not False
        or value["candidate_arm_clearance_authorized"] is not False
        or value["effect_metric_authorized"] is not False
    ):
        raise ValueError("Slot attempt identity or policy mismatch")
    return value


def validate_aggregate_attempt(
    value: dict[str, Any],
    contract_sha256: str,
    roster_sha256: str,
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "status",
            "workflow_profile",
            "execution_contract_sha256",
            "metadata_roster_sha256",
            "selector_or_failure_receipts_read_before_attempt",
            "sealed_payload_read",
            "rerun_authorized",
        },
        "aggregate attempt",
    )
    if (
        value["schema"] != AGGREGATE_ATTEMPT_SCHEMA
        or value["status"] != AGGREGATE_ATTEMPT_STATUS
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["execution_contract_sha256"] != contract_sha256
        or value["metadata_roster_sha256"] != roster_sha256
        or value["selector_or_failure_receipts_read_before_attempt"]
        is not False
        or value["sealed_payload_read"] is not False
        or value["rerun_authorized"] is not False
    ):
        raise ValueError("Aggregate attempt identity or policy mismatch")
    return value


def validate_selector(
    value: dict[str, Any],
    slot: dict[str, Any],
    contract_sha256: str,
    roster_sha256: str,
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "terminal",
            "workflow_profile",
            "execution_contract_sha256",
            "metadata_roster_sha256",
            "slot_index",
            "session_id",
            "source_authority_and_content_hashes",
            "strata",
            "qualified",
        },
        "selector",
    )
    index = int(slot["d3_roster_slot_index"])
    session_id = str(slot["session_id"])
    if (
        value["schema"] != SELECTOR_SCHEMA
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["execution_contract_sha256"] != contract_sha256
        or value["metadata_roster_sha256"] != roster_sha256
        or value["slot_index"] != index
        or value["session_id"] != session_id
        or not isinstance(value["qualified"], bool)
    ):
        raise ValueError("Selector identity mismatch")
    hashes = value["source_authority_and_content_hashes"]
    if not isinstance(hashes, dict):
        raise ValueError("Selector authority hashes must be an object")
    _closed_keys(hashes, AUTHORITY_HASH_KEYS, "selector authority hashes")
    if any(
        not isinstance(item, str) or SHA256_RE.fullmatch(item) is None
        for item in hashes.values()
    ):
        raise ValueError("Selector authority receipt is not SHA-256")
    strata = value["strata"]
    if not isinstance(strata, list) or len(strata) != len(STRATA):
        raise ValueError("Selector must contain exactly four strata")
    passes: list[bool] = []
    for row, (height, horizon) in zip(strata, STRATA):
        if not isinstance(row, dict):
            raise ValueError("Selector stratum must be an object")
        _closed_keys(
            row,
            {
                "height",
                "horizon_s",
                "denominator",
                "common_known_count",
                "common_known_coverage",
                "truth_risk_count",
                "truth_safe_count",
                "unknown_to_safe_violation_count",
                "gates",
                "passed",
            },
            "selector stratum",
        )
        if (
            row["height"] != height
            or float(row["horizon_s"]) != horizon
            or row["denominator"] != 252
            or not isinstance(row["passed"], bool)
        ):
            raise ValueError("Selector stratum identity mismatch")
        integer_keys = (
            "common_known_count",
            "truth_risk_count",
            "truth_safe_count",
            "unknown_to_safe_violation_count",
        )
        if any(
            not isinstance(row[key], int)
            or isinstance(row[key], bool)
            or row[key] < 0
            for key in integer_keys
        ):
            raise ValueError("Selector stratum count is invalid")
        common = row["common_known_count"]
        risk = row["truth_risk_count"]
        safe = row["truth_safe_count"]
        unknown = row["unknown_to_safe_violation_count"]
        coverage = row["common_known_coverage"]
        if (
            common > 252
            or risk + safe != common
            or not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or not math.isfinite(float(coverage))
            or not math.isclose(
                float(coverage),
                common / 252.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("Selector stratum arithmetic mismatch")
        expected_gates = {
            "coverage": common / 252.0 >= 0.1,
            "risk": risk >= 5,
            "safe": safe >= 20,
            "unknown_to_safe": unknown == 0,
        }
        gates = row["gates"]
        if not isinstance(gates, dict):
            raise ValueError("Selector stratum gates must be an object")
        _closed_keys(gates, set(expected_gates), "selector stratum gates")
        if gates != expected_gates or row["passed"] != all(
            expected_gates.values()
        ):
            raise ValueError("Selector stratum gate mismatch")
        passes.append(row["passed"])
    expected_qualified = all(passes)
    expected_terminal = (
        SELECTOR_QUALIFIED
        if expected_qualified
        else SELECTOR_NOT_QUALIFIED
    )
    if (
        value["qualified"] != expected_qualified
        or value["terminal"] != expected_terminal
    ):
        raise ValueError("Selector qualification terminal mismatch")
    return value


def validate_failure(
    value: dict[str, Any],
    slot: dict[str, Any],
    contract_sha256: str,
    roster_sha256: str,
) -> dict[str, Any]:
    _closed_keys(
        value,
        {
            "schema",
            "terminal",
            "workflow_profile",
            "execution_contract_sha256",
            "metadata_roster_sha256",
            "slot_attempt_sha256",
            "slot_index",
            "session_id",
            "error",
            "slot_consumed",
            "rerun_authorized",
            "source_replacement_authorized",
        },
        "failure",
    )
    error = value["error"]
    if not isinstance(error, dict):
        raise ValueError("Failure error must be an object")
    _closed_keys(error, {"type", "message"}, "failure error")
    if (
        value["schema"] != FAILURE_SCHEMA
        or value["terminal"] != SLOT_FAILURE_TERMINAL
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["execution_contract_sha256"] != contract_sha256
        or value["metadata_roster_sha256"] != roster_sha256
        or not isinstance(value["slot_attempt_sha256"], str)
        or SHA256_RE.fullmatch(value["slot_attempt_sha256"]) is None
        or value["slot_index"] != slot["d3_roster_slot_index"]
        or value["session_id"] != slot["session_id"]
        or not isinstance(error["type"], str)
        or not error["type"]
        or not isinstance(error["message"], str)
        or not error["message"]
        or value["slot_consumed"] is not True
        or value["rerun_authorized"] is not False
        or value["source_replacement_authorized"] is not False
    ):
        raise ValueError("Failure identity or policy mismatch")
    return value


def validate_selection(
    path_or_value: Path | str | dict[str, Any],
    slots: list[dict[str, Any]],
    contract_sha256: str,
    roster_sha256: str,
    carry_forward_authority: dict[str, str],
) -> dict[str, Any]:
    if isinstance(path_or_value, dict):
        value = path_or_value
        selection_path: Path | None = None
    elif isinstance(path_or_value, (Path, str)):
        selection_path = Path(path_or_value).resolve()
        value = load_json(selection_path)
    else:
        raise TypeError("Selection must be a JSON path or object")
    _closed_keys(
        value,
        {
            "schema",
            "terminal",
            "workflow_profile",
            "execution_contract_sha256",
            "metadata_roster_sha256",
            "aggregate_attempt_sha256",
            "consumed_slot_count",
            "newly_opened_slot_count",
            "carry_forward_burned_slot_count",
            "carry_forward_burn_receipt",
            "qualified_source_count",
            "selected_sources",
            "failure_receipt_count",
            "screening_receipts_only_read",
            "sealed_payload_read",
            "source_replacement_authorized",
            "budget_expansion_authorized",
            "screening_rerun_authorized",
        },
        "selection",
    )
    integer_keys = (
        "consumed_slot_count",
        "newly_opened_slot_count",
        "carry_forward_burned_slot_count",
        "qualified_source_count",
        "failure_receipt_count",
    )
    if (
        len(slots) != SLOT_COUNT
        or value["schema"] != SELECTION_SCHEMA
        or value["terminal"] != QUALIFICATION_TERMINAL
        or value["workflow_profile"] != "THESIS_DEVELOPMENT"
        or value["execution_contract_sha256"] != contract_sha256
        or value["metadata_roster_sha256"] != roster_sha256
        or not isinstance(value["aggregate_attempt_sha256"], str)
        or SHA256_RE.fullmatch(value["aggregate_attempt_sha256"]) is None
        or any(
            not isinstance(value[key], int)
            or isinstance(value[key], bool)
            or value[key] < 0
            for key in integer_keys
        )
        or value["qualified_source_count"] != REQUIRED_QUALIFIED
        or value["carry_forward_burned_slot_count"] != 1
        or value["consumed_slot_count"]
        != value["newly_opened_slot_count"] + 1
        or value["newly_opened_slot_count"]
        > MAXIMUM_NEWLY_OPENED_SLOTS
        or value["screening_receipts_only_read"] is not True
        or value["sealed_payload_read"] is not False
        or value["source_replacement_authorized"] is not False
        or value["budget_expansion_authorized"] is not False
        or value["screening_rerun_authorized"] is not False
    ):
        raise ValueError("Selection identity or policy mismatch")
    selected = value["selected_sources"]
    if not isinstance(selected, list) or len(selected) != REQUIRED_QUALIFIED:
        raise ValueError("Selection must contain exactly six sources")
    selected_indices: list[int] = []
    selector_paths: list[Path] = []
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("Selected source must be an object")
        _closed_keys(
            row,
            {
                "slot_index",
                "session_id",
                "selector_path",
                "selector_sha256",
                "source_authority_and_content_hashes",
            },
            "selected source",
        )
        index = row["slot_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 1
            or index > SLOT_COUNT
            or not isinstance(row["session_id"], str)
            or not isinstance(row["selector_path"], str)
            or not row["selector_path"]
            or not isinstance(row["selector_sha256"], str)
            or SHA256_RE.fullmatch(row["selector_sha256"]) is None
        ):
            raise ValueError("Selected source identity is invalid")
        slot = slots[index - 1]
        if (
            slot.get("d3_roster_slot_index") != index
            or slot.get("session_id") != row["session_id"]
        ):
            raise ValueError("Selected source differs from frozen roster")
        hashes = row["source_authority_and_content_hashes"]
        if not isinstance(hashes, dict):
            raise ValueError("Selected source authority hashes must be object")
        _closed_keys(
            hashes,
            AUTHORITY_HASH_KEYS,
            "selected source authority hashes",
        )
        if any(
            not isinstance(item, str)
            or SHA256_RE.fullmatch(item) is None
            for item in hashes.values()
        ):
            raise ValueError("Selected source authority receipt is invalid")
        selected_indices.append(index)
        selector_paths.append(Path(row["selector_path"]).resolve())
    if selected_indices != sorted(selected_indices) or len(
        set(selected_indices)
    ) != REQUIRED_QUALIFIED:
        raise ValueError("Selected sources are not in frozen slot order")
    if any(index < FIRST_ACTIVE_SLOT_INDEX for index in selected_indices):
        raise ValueError("Q0.1 selection contains permanently burned slot 1")
    root = selector_paths[0].parent.parent
    if (
        selection_path is not None
        and selection_path != aggregate_paths(root)["selection"]
    ):
        raise ValueError("Selection path is outside its screening root")
    aggregate_attempt = aggregate_paths(root)["aggregate_attempt"]
    if (
        not aggregate_attempt.is_file()
        or sha256(aggregate_attempt)
        != value["aggregate_attempt_sha256"]
    ):
        raise ValueError("Selection aggregate-attempt hash mismatch")
    validate_aggregate_attempt(
        load_json(aggregate_attempt),
        contract_sha256,
        roster_sha256,
    )
    for row, index, selector_path in zip(
        selected,
        selected_indices,
        selector_paths,
    ):
        slot = slots[index - 1]
        expected_path = slot_layout(root, slot)["selector"]
        if selector_path != expected_path or row["selector_path"] != str(
            expected_path
        ):
            raise ValueError("Selected source selector path is noncanonical")
        if sha256(expected_path) != row["selector_sha256"]:
            raise ValueError("Selected source selector hash mismatch")
        selector = validate_selector(
            load_json(expected_path),
            slot,
            contract_sha256,
            roster_sha256,
        )
        if (
            selector["qualified"] is not True
            or selector["source_authority_and_content_hashes"]
            != row["source_authority_and_content_hashes"]
        ):
            raise ValueError("Selected source selector receipt mismatch")
    state = scan_screening_state(
        root,
        slots,
        contract_sha256,
        roster_sha256,
        carry_forward_authority,
    )
    carry_receipt = value["carry_forward_burn_receipt"]
    if not isinstance(carry_receipt, dict):
        raise ValueError("Selection carry-forward receipt must be an object")
    _closed_keys(
        carry_receipt,
        {
            "slot_index",
            "session_id",
            "carry_forward_path",
            "carry_forward_sha256",
            "terminal",
        },
        "selection carry-forward receipt",
    )
    if (
        state["terminal"] != QUALIFICATION_TERMINAL
        or state["qualified_rows"] != selected
        or value["consumed_slot_count"] != state["consumed_count"]
        or value["newly_opened_slot_count"]
        != state["newly_opened_count"]
        or value["failure_receipt_count"] != len(state["failure_rows"])
        or state["carry_forward_rows"] != [carry_receipt]
    ):
        raise ValueError("Selection is not the exact first-six terminal")
    return value


def scan_screening_state(
    root: Path,
    slots: list[dict[str, Any]],
    contract_sha256: str,
    roster_sha256: str,
    carry_forward_authority: dict[str, str],
) -> dict[str, Any]:
    if len(slots) != SLOT_COUNT:
        raise ValueError("Screening state requires exactly 40 slots")
    screening_attempt = aggregate_paths(root)["screening_attempt"]
    any_slot_root = any(
        slot_layout(root, slot)["slot_root"].exists()
        for slot in slots
    )
    if not screening_attempt.exists() and not any_slot_root:
        return {
            "consumed_count": 0,
            "newly_opened_count": 0,
            "carry_forward_rows": [],
            "qualified_rows": [],
            "failure_rows": [],
            "next_slot": None,
            "interrupted_slot": None,
            "control_plane_uninitialized": True,
            "terminal": None,
        }
    if not screening_attempt.is_file():
        raise ValueError("Screening attempt is absent or non-file")
    validate_screening_attempt(
        load_json(screening_attempt),
        contract_sha256,
        roster_sha256,
    )
    burned_slot = slots[0]
    burned_layout = slot_layout(root, burned_slot)
    carry_path = burned_layout["carry_forward"]
    if not carry_path.is_file():
        raise ValueError("Q0.1 slot 1 carry-forward burn receipt is absent")
    slot_one_children = {
        child.name
        for child in burned_layout["slot_root"].iterdir()
    }
    if slot_one_children != {carry_path.name}:
        raise ValueError(
            "Q0.1 slot 1 contains artifacts beyond the carry-forward receipt"
        )
    carry = validate_carry_forward(
        load_json(carry_path),
        burned_slot,
        contract_sha256,
        carry_forward_authority,
    )
    carry_forward_rows = [
        {
            "slot_index": 1,
            "session_id": burned_slot["session_id"],
            "carry_forward_path": str(carry_path.resolve()),
            "carry_forward_sha256": sha256(carry_path),
            "terminal": carry["terminal"],
        }
    ]
    qualified_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    consumed_count = 1
    newly_opened_count = 0
    for offset, slot in enumerate(slots[1:], start=1):
        layout = slot_layout(root, slot)
        if layout["carry_forward"].exists():
            raise ValueError(
                "Q0.1 carry-forward receipt is allowed only for original slot 1"
            )
        selector_exists = layout["selector"].exists()
        failure_exists = layout["failure"].exists()
        if selector_exists and failure_exists:
            raise ValueError(
                f"Ambiguous selector and failure for slot {offset + 1}"
            )
        if not selector_exists and not failure_exists:
            later_layouts = [
                slot_layout(root, later)
                for later in slots[offset + 1 :]
            ]
            if any(
                later["slot_root"].exists()
                or later["selector"].exists()
                or later["failure"].exists()
                for later in later_layouts
            ):
                raise ValueError("Screening receipts contain a prefix gap")
            interrupted = layout["slot_root"].exists() and any(
                layout["slot_root"].iterdir()
            )
            if interrupted:
                if not layout["attempt"].is_file():
                    raise ValueError(
                        "Interrupted slot lacks a durable slot attempt"
                    )
                validate_slot_attempt(
                    load_json(layout["attempt"]),
                    slot,
                    contract_sha256,
                    roster_sha256,
                )
            return {
                "consumed_count": consumed_count,
                "newly_opened_count": newly_opened_count,
                "carry_forward_rows": carry_forward_rows,
                "qualified_rows": qualified_rows,
                "failure_rows": failure_rows,
                "next_slot": None if interrupted else slot,
                "interrupted_slot": slot if interrupted else None,
                "control_plane_uninitialized": False,
                "terminal": None,
            }
        consumed_count += 1
        newly_opened_count += 1
        if not layout["attempt"].is_file():
            raise ValueError(
                f"Consumed slot {offset + 1} lacks a durable attempt"
            )
        validate_slot_attempt(
            load_json(layout["attempt"]),
            slot,
            contract_sha256,
            roster_sha256,
        )
        attempt_sha256 = sha256(layout["attempt"])
        if selector_exists:
            selector = validate_selector(
                load_json(layout["selector"]),
                slot,
                contract_sha256,
                roster_sha256,
            )
            if (
                selector["source_authority_and_content_hashes"][
                    "slot_attempt_sha256"
                ]
                != attempt_sha256
            ):
                raise ValueError("Selector slot-attempt hash mismatch")
            if selector["qualified"]:
                qualified_rows.append(
                    {
                        "slot_index": slot["d3_roster_slot_index"],
                        "session_id": slot["session_id"],
                        "selector_path": str(layout["selector"].resolve()),
                        "selector_sha256": sha256(layout["selector"]),
                        "source_authority_and_content_hashes": selector[
                            "source_authority_and_content_hashes"
                        ],
                    }
                )
        else:
            failure = validate_failure(
                load_json(layout["failure"]),
                slot,
                contract_sha256,
                roster_sha256,
            )
            if failure["slot_attempt_sha256"] != attempt_sha256:
                raise ValueError("Failure slot-attempt hash mismatch")
            failure_rows.append(
                {
                    "slot_index": slot["d3_roster_slot_index"],
                    "session_id": slot["session_id"],
                    "failure_path": str(layout["failure"].resolve()),
                    "failure_sha256": sha256(layout["failure"]),
                    "terminal": failure["terminal"],
                }
            )
        if len(qualified_rows) == REQUIRED_QUALIFIED:
            for later in slots[offset + 1 :]:
                later_layout = slot_layout(root, later)
                if later_layout["slot_root"].exists():
                    raise ValueError(
                        "Screening artifact exists after sixth qualified source"
                    )
            return {
                "consumed_count": consumed_count,
                "newly_opened_count": newly_opened_count,
                "carry_forward_rows": carry_forward_rows,
                "qualified_rows": qualified_rows,
                "failure_rows": failure_rows,
                "next_slot": None,
                "interrupted_slot": None,
                "control_plane_uninitialized": False,
                "terminal": QUALIFICATION_TERMINAL,
            }
    return {
        "consumed_count": consumed_count,
        "newly_opened_count": newly_opened_count,
        "carry_forward_rows": carry_forward_rows,
        "qualified_rows": qualified_rows,
        "failure_rows": failure_rows,
        "next_slot": None,
        "interrupted_slot": None,
        "control_plane_uninitialized": False,
        "terminal": BUDGET_TERMINAL,
    }
