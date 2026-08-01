#!/usr/bin/env python3
"""Lock the metadata-only 40-slot roster for HFTF Stage C D3-Q0.

This program must not read RGB, mask, depth, or pose content.  It only binds
the official train split, session descriptions, object receipts/listings, and
camera metadata.  A successful roster authorizes a later separately frozen
reference-and-support qualifier/effect-skeleton contract; it does not
authorize media, support, truth, effect, or student execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from plan_stage_c_d2_official_train_metadata import (
    DEFAULT_API,
    GCS_PREFIX,
    CandidateIneligible,
    MetadataApi,
    _canonical_json_sha256,
    _parse_split_ids,
    _receipt,
    _timeline,
    _validate_intrinsics,
    camera_metadata,
    indexed_objects,
    media_url,
)


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_d3_reference_and_support_only_"
    "challenge_qualification_q0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_D2_NOT_EVALUABLE_BEFORE_ANY_D3_SOURCE_"
    "MEDIA_SUPPORT_OR_TRUTH_OUTCOME"
)
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_metadata_roster_execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_D3_Q0_BEFORE_METADATA_ROSTER_OR_D3_MEDIA_TRUTH"
)
D2_QUALIFICATION_SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_metadata_qualification"
)
D2_QUALIFICATION_TERMINAL = (
    "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
)
D2_TRACKED_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_metadata_"
    "qualification_result"
)
ROSTER_SCHEMA = "blindassist_hftf_stage_c_d3_q0_metadata_roster"
ROSTER_READY = "D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED"
ROSTER_INSUFFICIENT = (
    "D3_Q0_METADATA_ROSTER_NOT_EVALUABLE_INSUFFICIENT_ELIGIBLE_SLOTS"
)
ROSTER_FAILURE = "D3_Q0_METADATA_ROSTER_EXECUTION_FAILED_NO_RERUN"
ATTEMPT_SCHEMA = "blindassist_hftf_stage_c_d3_q0_metadata_roster_attempt"
ATTEMPT_STATUS = "ATTEMPT_OPENED_AND_FSYNCED_BEFORE_FIRST_NETWORK_REQUEST"
PLANNER_RELATIVE_PATH = (
    "scripts/research/hftf/plan_stage_c_d3_q0_metadata_roster.py"
)
TEST_RELATIVE_PATH = (
    "scripts/research/hftf/test_plan_stage_c_d3_q0_metadata_roster.py"
)
D2_PLANNER_RELATIVE_PATH = (
    "scripts/research/hftf/plan_stage_c_d2_official_train_metadata.py"
)
CANONICAL_RELATIVE_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d3-q0-metadata-roster-20260802/roster.json"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve(path: str, contract_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (repo_root() / candidate).resolve()
    contract_candidate = (contract_path.parent / candidate).resolve()
    if root_candidate.exists() or not contract_candidate.exists():
        return root_candidate
    return contract_candidate


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_definition_count(path: Path) -> int:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_tracked_clean(path: Path, label: str) -> None:
    relative = path.resolve().relative_to(repo_root()).as_posix()
    if not git("ls-files", "--error-unmatch", "--", relative):
        raise ValueError(f"{label} is not tracked")
    if git("diff", "--name-only", "--", relative):
        raise ValueError(f"{label} has unstaged changes")
    if git("diff", "--cached", "--name-only", "--", relative):
        raise ValueError(f"{label} has staged changes")


def require_pushed_state(contract_path: Path, contract: dict[str, Any]) -> None:
    paths = {
        "execution contract": contract_path,
        "metadata roster planner": resolve(
            contract["implementations"]["metadata_roster_planner"]["path"],
            contract_path,
        ),
        "metadata roster planner test": resolve(
            contract["implementation_tests"]["planner_test"]["path"],
            contract_path,
        ),
        "D2 metadata helper dependency": resolve(
            contract["implementations"]["d2_metadata_helper"]["path"],
            contract_path,
        ),
    }
    for label, path in paths.items():
        require_tracked_clean(path, label)
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/master")
    if head != origin:
        raise ValueError("HEAD must equal origin/master before metadata scan")


def bound_parent(
    contract: dict[str, Any],
    contract_path: Path,
    key: str,
) -> tuple[Path, dict[str, Any]]:
    binding = contract.get("parents", {}).get(key)
    if not isinstance(binding, dict):
        raise ValueError(f"Missing contract parent: {key}")
    path = resolve(str(binding.get("path", "")), contract_path)
    if sha256(path) != binding.get("sha256"):
        raise ValueError(f"Parent hash mismatch: {key}")
    return path, load_json(path)


def derive_exclusions(
    d2_qualification: dict[str, Any],
) -> tuple[dict[str, list[str]], set[str]]:
    exclusions = d2_qualification.get("exclusions", {})
    categories = exclusions.get("category_session_ids")
    if not isinstance(categories, dict):
        raise ValueError("D2 exclusion categories are missing")
    normalized: dict[str, list[str]] = {}
    for key, values in categories.items():
        if not isinstance(values, list):
            raise ValueError(f"D2 exclusion category is invalid: {key}")
        normalized[str(key)] = sorted(str(value) for value in values)
    d2_six = sorted(
        str(parent["session_id"])
        for parent in d2_qualification.get("qualified_parents", [])
    )
    if len(d2_six) != 6 or len(set(d2_six)) != 6:
        raise ValueError("D2 six-parent exclusion is invalid")
    normalized["d2_consumed_six_parent_cohort"] = d2_six
    flattened = [value for values in normalized.values() for value in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("D3 exclusion categories overlap")
    return normalized, set(flattened)


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise ValueError("D3 metadata roster contract identity mismatch")
    protocol_path, protocol = bound_parent(
        contract,
        contract_path,
        "d3_q0_protocol",
    )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
        or protocol.get("source_pool", {}).get(
            "maximum_truth_screened_slots"
        )
        != 40
        or protocol.get("source_pool", {}).get(
            "required_qualified_sources"
        )
        != 6
    ):
        raise ValueError("D3-Q0 protocol binding mismatch")
    protocol_authority = protocol.get(
        "execution_authority_at_this_freeze", {}
    )
    if (
        protocol_authority.get(
            "freeze_metadata_roster_qualifier_and_effect_skeleton_"
            "execution_contract_authorized"
        )
        is not True
        or protocol_authority.get("metadata_scan_authorized_now")
        is not False
        or protocol_authority.get(
            "media_or_pose_content_open_authorized_now"
        )
        is not False
        or protocol_authority.get(
            "support_or_truth_qualification_authorized_now"
        )
        is not False
        or protocol_authority.get("d3_effect_authorized_now") is not False
    ):
        raise ValueError("D3-Q0 parent authority mismatch")
    d2_path, d2 = bound_parent(
        contract,
        contract_path,
        "d2_metadata_qualification",
    )
    d2_result_path, d2_result = bound_parent(
        contract,
        contract_path,
        "d2_metadata_qualification_result",
    )
    if (
        d2.get("schema") != D2_QUALIFICATION_SCHEMA
        or d2.get("terminal") != D2_QUALIFICATION_TERMINAL
        or d2.get("firewall", {}).get("rgb_bytes_read") is not False
        or d2.get("firewall", {}).get("panoptic_mask_bytes_read")
        is not False
        or d2.get("firewall", {}).get("metric_depth_bytes_read")
        is not False
        or d2.get("firewall", {}).get("camera_pose_content_read")
        is not False
    ):
        raise ValueError("D2 metadata qualification parent mismatch")
    tracked_qualification = d2_result.get(
        "durable_evidence", {}
    ).get("qualification", {})
    if (
        d2_result.get("schema") != D2_TRACKED_RESULT_SCHEMA
        or d2_result.get("terminal") != D2_QUALIFICATION_TERMINAL
        or resolve(
            str(tracked_qualification.get("path", "")),
            contract_path,
        )
        != d2_path
        or tracked_qualification.get("sha256") != sha256(d2_path)
    ):
        raise ValueError("Tracked D2 qualification result mismatch")
    categories, excluded = derive_exclusions(d2)
    selection = contract.get("selection", {})
    split = contract.get("official_train_split", {})
    if (
        selection.get("roster_slot_count") != 40
        or selection.get("order") != "ascending_session_id"
        or selection.get("stop_after_roster_slots") is not True
        or selection.get("manual_skip_or_reorder_authorized") is not False
        or selection.get("source_attribute_ranking_authorized") is not False
        or selection.get("media_or_truth_open_authorized") is not False
        or contract.get("expected_excluded_parent_count") != len(excluded)
        or len(excluded) != 84
        or split.get("generation")
        != d2["official_train_split"]["object_receipt"]["generation"]
        or split.get("text_sha256")
        != d2["official_train_split"]["text_sha256"]
    ):
        raise ValueError("D3 metadata roster selection mismatch")
    implementations = contract.get("implementations", {})
    for key, expected_path in (
        ("metadata_roster_planner", PLANNER_RELATIVE_PATH),
        ("d2_metadata_helper", D2_PLANNER_RELATIVE_PATH),
    ):
        binding = implementations.get(key, {})
        path = resolve(str(binding.get("path", "")), contract_path)
        if (
            path.relative_to(repo_root()).as_posix() != expected_path
            or sha256(path) != binding.get("sha256")
        ):
            raise ValueError(f"Implementation binding mismatch: {key}")
    test_binding = contract.get("implementation_tests", {}).get(
        "planner_test", {}
    )
    test_path = resolve(str(test_binding.get("path", "")), contract_path)
    if (
        test_path.relative_to(repo_root()).as_posix() != TEST_RELATIVE_PATH
        or sha256(test_path) != test_binding.get("sha256")
        or test_definition_count(test_path)
        != contract["implementation_tests"]["test_count"]
        or contract["implementation_tests"]["test_count"]
        != contract["implementation_tests"]["tests_passed"]
    ):
        raise ValueError("D3 planner test receipt mismatch")
    canonical = contract.get("canonical_artifacts", {})
    expected_output = (repo_root() / CANONICAL_RELATIVE_OUTPUT).resolve()
    if (
        resolve(str(canonical.get("roster_result", "")), contract_path)
        != expected_output
        or resolve(str(canonical.get("attempt", "")), contract_path)
        != expected_output.parent / "attempt.json"
        or resolve(str(canonical.get("failure", "")), contract_path)
        != expected_output.parent / "failure.json"
    ):
        raise ValueError("D3 canonical artifact paths mismatch")
    authorization = contract.get("authorization", {})
    if (
        authorization.get("metadata_roster_scan_authorized") is not True
        or authorization.get(
            "freeze_qualifier_effect_execution_contract_on_success"
        )
        is not True
        or any(
            authorization.get(key) is not False
            for key in (
                "media_or_pose_content_open_authorized",
                "support_or_truth_qualification_authorized",
                "d3_effect_authorized",
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
        raise ValueError("D3 metadata roster authorization mismatch")
    retries = int(contract.get("network", {}).get("retries", -1))
    if retries != 3:
        raise ValueError("D3 metadata retries must be exactly 3")
    failure_policy = contract.get("failure_policy", {})
    if (
        failure_policy.get("rerun_authorized") is not False
        or failure_policy.get(
            "append_or_replace_roster_slots_authorized"
        )
        is not False
        or failure_policy.get(
            "preserve_attempt_and_failure_artifacts"
        )
        is not True
        or failure_policy.get(
            "media_pose_support_truth_or_effect_must_remain_unopened"
        )
        is not True
        or any(
            value is not False
            for value in contract.get(
                "outcome_firewall_at_freeze", {}
            ).values()
        )
        or len(contract.get("outcome_firewall_at_freeze", {})) != 7
    ):
        raise ValueError("D3 metadata failure/firewall policy mismatch")
    if verify_git:
        require_pushed_state(contract_path, contract)
    return {
        "contract": contract,
        "contract_path": contract_path.resolve(),
        "protocol_path": protocol_path,
        "protocol": protocol,
        "d2_path": d2_path,
        "d2": d2,
        "d2_result_path": d2_result_path,
        "d2_result": d2_result,
        "exclusion_categories": categories,
        "excluded": excluded,
        "split_generation": str(split["generation"]),
        "split_sha256": str(split["text_sha256"]),
        "retries": retries,
        "roster_slot_count": 40,
    }


def scan_metadata_roster(
    ordered_session_ids: list[str],
    excluded: set[str],
    qualifier: Any,
    *,
    roster_slot_count: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for session_id in ordered_session_ids:
        if session_id in excluded:
            ledger.append(
                {
                    "session_id": session_id,
                    "metadata_eligible": False,
                    "reason": "frozen_exclusion_union",
                }
            )
            continue
        try:
            candidate = qualifier(session_id, len(locked) + 1)
        except CandidateIneligible as error:
            ledger.append(
                {
                    "session_id": session_id,
                    "metadata_eligible": False,
                    "reason": str(error),
                }
            )
            continue
        row = dict(candidate)
        row["role"] = "d3_q0_locked_truth_screening_slot"
        row["d3_roster_slot_index"] = len(locked) + 1
        row["media_pose_content_opened"] = False
        row["support_or_truth_computed"] = False
        row["effect_computed"] = False
        locked.append(row)
        ledger.append(row)
        if len(locked) == roster_slot_count:
            break
    return locked, ledger


def selected_modality_receipt(
    objects: list[dict[str, Any]],
    suffix: str,
    label: str,
    selected_frames: list[int],
) -> dict[str, Any]:
    try:
        indexed = indexed_objects(objects, suffix)
    except ValueError as error:
        raise CandidateIneligible(str(error)) from error
    required = set(selected_frames)
    if len(selected_frames) != 13 or not required.issubset(indexed):
        raise CandidateIneligible(
            f"{label} requires exact selected 13-frame timeline"
        )
    receipts: list[dict[str, Any]] = []
    for frame_index in selected_frames:
        item = indexed[frame_index]
        name = str(item.get("name", ""))
        receipt = _receipt(
            item,
            name,
            f"{label} selected frame {frame_index}",
        )
        receipts.append(
            {
                "frame_index": frame_index,
                "name": receipt["name"],
                "generation": receipt["generation"],
                "size": receipt["size"],
                "md5_base64": receipt["md5_base64"],
            }
        )
    return {
        "required_frame_count": 13,
        "required_frame_indices": selected_frames,
        "required_frame_receipts_sha256": _canonical_json_sha256(
            receipts
        ),
        "required_frame_receipts": receipts,
    }


def qualify_candidate_exact13_unchecked(
    session_id: str,
    rank: int,
    retries: int,
    api: MetadataApi,
) -> dict[str, Any]:
    prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
    description_name = f"{prefix}/description.json"
    pose_name = f"{prefix}/camera_chest/camera_poses.csv"
    description_receipt = _receipt(
        api.get_object(description_name, retries),
        description_name,
        "description",
    )
    pose_receipt = _receipt(
        api.get_object(pose_name, retries),
        pose_name,
        "camera pose",
    )
    description = api.fetch_json(
        media_url(
            description_name,
            description_receipt["generation"],
        ),
        retries,
    )
    if description.get("session_type") != "synthetic":
        raise CandidateIneligible(
            "description session_type is not synthetic"
        )
    try:
        source_fps, dimensions = camera_metadata(
            description,
            "camera_chest",
            "left",
        )
    except ValueError as error:
        raise CandidateIneligible(str(error)) from error
    camera = _validate_intrinsics(dimensions)
    selected_frames = _timeline(source_fps)
    modality_specs = {
        "rgb": ("video_frames", ".png"),
        "mask": ("segmentation_masks", ".png"),
        "depth": ("depth_maps", ".float16.gz"),
    }
    modalities: dict[str, dict[str, Any]] = {}
    for label, (folder, suffix) in modality_specs.items():
        objects = api.list_objects(
            f"{prefix}/camera_chest/left/{folder}/",
            retries,
        )
        modalities[label] = selected_modality_receipt(
            objects,
            suffix,
            label,
            selected_frames,
        )
    return {
        "session_id": session_id,
        "official_split": "train",
        "role": "d3_q0_locked_truth_screening_slot",
        "metadata_eligible": True,
        "metadata_eligible_rank": rank,
        "source_fps": source_fps,
        "normalized_target_fps": 5.0,
        "selected_source_frames": selected_frames,
        "description_object": description_receipt,
        "camera_pose_object_receipt": pose_receipt,
        "camera_pose_content_read": False,
        "camera": camera,
        "media_object_listing_receipts": modalities,
        "rgb_mask_depth_bytes_read": False,
        "geometry_teacher_outcome_read": False,
        "student_outcome_read": False,
    }


def qualify_candidate_exact13(
    session_id: str,
    rank: int,
    retries: int,
    api: MetadataApi,
) -> dict[str, Any]:
    try:
        return qualify_candidate_exact13_unchecked(
            session_id,
            rank,
            retries,
            api,
        )
    except CandidateIneligible:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise CandidateIneligible(
            f"metadata unavailable or invalid after {retries} retries: "
            f"{error}"
        ) from error


def plan(
    contract_path: Path,
    retries: int,
    *,
    api: MetadataApi = DEFAULT_API,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or validate_contract(contract_path, verify_git=True)
    if retries != context["retries"]:
        raise ValueError("Retries differ from frozen D3 contract")
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
    )
    split_object = api.get_object(split_name, retries)
    split_receipt = _receipt(
        split_object,
        split_name,
        "official train split",
    )
    if str(split_receipt["generation"]) != context["split_generation"]:
        raise ValueError("Official train split generation drift")
    split_text = api.fetch_text(
        media_url(split_name, split_receipt["generation"]),
        retries,
    )
    if sha256_text(split_text) != context["split_sha256"]:
        raise ValueError("Official train split SHA-256 drift")
    ordered_ids = _parse_split_ids(split_text)
    locked, ledger = scan_metadata_roster(
        ordered_ids,
        context["excluded"],
        lambda session_id, rank: qualify_candidate_exact13(
            session_id,
            rank,
            retries,
            api,
        ),
        roster_slot_count=context["roster_slot_count"],
    )
    terminal = (
        ROSTER_READY
        if len(locked) == context["roster_slot_count"]
        else ROSTER_INSUFFICIENT
    )
    planner_path = repo_root() / PLANNER_RELATIVE_PATH
    if (
        sha256(planner_path)
        != context["contract"]["implementations"][
            "metadata_roster_planner"
        ]["sha256"]
    ):
        raise ValueError("D3 metadata planner changed during execution")
    return {
        "schema": ROSTER_SCHEMA,
        "terminal": terminal,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "evidence_role": (
            "METADATA_ONLY_D3_TRUTH_SCREENING_ROSTER_"
            "NOT_MEDIA_SUPPORT_TRUTH_OR_EFFECT_EVIDENCE"
        ),
        "bindings": {
            "execution_contract": {
                "path": str(contract_path.resolve()),
                "sha256": sha256(contract_path),
            },
            "d3_q0_protocol": {
                "path": str(context["protocol_path"]),
                "sha256": sha256(context["protocol_path"]),
            },
            "d2_metadata_qualification": {
                "path": str(context["d2_path"]),
                "sha256": sha256(context["d2_path"]),
            },
            "d2_metadata_qualification_result": {
                "path": str(context["d2_result_path"]),
                "sha256": sha256(context["d2_result_path"]),
            },
            "metadata_roster_planner": {
                "path": str(planner_path.resolve()),
                "sha256": sha256(planner_path),
            },
        },
        "official_train_split": {
            "object_receipt": split_receipt,
            "text_sha256": context["split_sha256"],
            "session_count": len(ordered_ids),
            "selection_order": "ascending_session_id",
            "input_split_order_used_for_selection": False,
        },
        "exclusions": {
            "category_counts": {
                key: len(values)
                for key, values in context[
                    "exclusion_categories"
                ].items()
            },
            "excluded_parent_count": len(context["excluded"]),
            "category_session_ids": context["exclusion_categories"],
        },
        "requested_roster_slot_count": context["roster_slot_count"],
        "locked_roster_slot_count": len(locked),
        "locked_slots": locked,
        "scan_ledger": ledger,
        "firewall": {
            "rgb_bytes_read": False,
            "panoptic_mask_bytes_read": False,
            "metric_depth_bytes_read": False,
            "camera_pose_content_read": False,
            "support_masks_computed": False,
            "future_truth_opened": False,
            "effect_computed": False,
            "reserved_official_test_opened": False,
        },
        "authorization": {
            "freeze_qualifier_effect_execution_contract": (
                terminal == ROSTER_READY
            ),
            "roster_rerun_authorized": False,
            "roster_slot_replacement_authorized": False,
            "media_or_pose_content_open_authorized": False,
            "support_or_truth_qualification_authorized": False,
            "d3_effect_authorized": False,
            "rgb_student_training_authorized": False,
            "rgb_student_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }


def write_json_exclusive_fsync(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def open_attempt(
    output: Path,
    context: dict[str, Any],
    retries: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=False)
    attempt_path = output.parent / "attempt.json"
    write_json_exclusive_fsync(
        attempt_path,
        {
            "schema": ATTEMPT_SCHEMA,
            "status": ATTEMPT_STATUS,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "execution_contract_sha256": sha256(
                context["contract_path"]
            ),
            "internal_retries_per_metadata_request": retries,
            "first_network_request_started": False,
            "rerun_authorized": False,
            "media_or_pose_content_open_authorized": False,
            "support_or_truth_qualification_authorized": False,
            "d3_effect_authorized": False,
        },
    )
    return attempt_path


def require_canonical_output(path: Path) -> Path:
    expected = (repo_root() / CANONICAL_RELATIVE_OUTPUT).resolve()
    if path.resolve() != expected:
        raise ValueError("D3 metadata roster output is noncanonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--retries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_canonical_output(args.output)
    context: dict[str, Any] | None = None
    attempt_opened_this_run = False
    try:
        context = validate_contract(
            args.execution_contract,
            verify_git=True,
        )
        if args.retries != context["retries"]:
            raise ValueError("Retries differ from frozen D3 contract")
        open_attempt(output, context, args.retries)
        attempt_opened_this_run = True
        result = plan(
            args.execution_contract,
            args.retries,
            context=context,
        )
        write_json_exclusive_fsync(output, result)
        print(json.dumps({"terminal": result["terminal"]}))
        return 0 if result["terminal"] == ROSTER_READY else 2
    except Exception as error:
        if (
            context is not None
            and attempt_opened_this_run
            and output.parent.exists()
        ):
            failure_path = output.parent / "failure.json"
            if not failure_path.exists() and not output.exists():
                write_json_exclusive_fsync(
                    failure_path,
                    {
                        "schema": ROSTER_SCHEMA,
                        "terminal": ROSTER_FAILURE,
                        "workflow_profile": "THESIS_DEVELOPMENT",
                        "execution_contract_sha256": sha256(
                            context["contract_path"]
                        ),
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        "rerun_authorized": False,
                        "media_or_pose_content_opened": False,
                        "support_or_truth_computed": False,
                        "effect_computed": False,
                    },
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
