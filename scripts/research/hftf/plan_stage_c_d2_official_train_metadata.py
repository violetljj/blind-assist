#!/usr/bin/env python3
"""Qualify six new official-train parents for HFTF Stage C D2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acquire_sanpo_synthetic_replay import (  # noqa: E402
    camera_metadata,
    indexed_objects,
)
from build_sanpo_sequence_evalset import (  # noqa: E402
    GCS_PREFIX,
    fetch_json,
    fetch_text,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    object_inventory,
)


DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_transport_d2"
)
DESIGN_STATUS = "FROZEN_BEFORE_D2_METADATA_SCAN_OR_SOURCE_OUTCOME"
T0_SCHEMA = (
    "blindassist_hftf_stage_c_t0_consumed_development_transport_result"
)
T0_TERMINAL = "T0_SANPO_SHORT_PATH_CONSUMED_PACKAGE_EQUIVALENT"
EXECUTION_CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_metadata_"
    "qualification_execution_contract"
)
EXECUTION_CONTRACT_STATUS = (
    "FROZEN_AFTER_T0_BEFORE_D2_METADATA_SCAN_OR_NEW_MEDIA"
)
SOURCE_PLAN_SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
)
SOURCE_PLAN_TERMINAL = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
F0_LEDGER_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_body_head_source_pool_burn_ledger_f0"
)
R4_LEDGER_SCHEMA = "blindassist_hftf_r4_source_pool_burn_ledger"
R3_1_LEDGER_SCHEMA = "blindassist_hftf_r3_1_source_pool_burn_ledger"
F0_1_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_heldout_effect_result_f0_1"
)
F0_1_RESULT_TERMINAL = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_"
    "STUDENT_SIGNAL_NOT_SUPPORTED_STOP"
)
F0_PLAN_SCHEMA = "blindassist_hftf_stage_c_f0_sanpo_inventory_plan"
F0_PLAN_TERMINAL = "F0_SANPO_FIXED_SOURCE_INVENTORY_READY"

SCHEMA = (
    "blindassist_hftf_stage_c_d2_official_train_metadata_qualification"
)
READY = "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
INSUFFICIENT = "STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT"
CANONICAL_RELATIVE_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-official-train-metadata-qualification-20260802/"
    "qualification.json"
)
PLANNER_RELATIVE_PATH = (
    "scripts/research/hftf/"
    "plan_stage_c_d2_official_train_metadata.py"
)
PLANNER_TEST_RELATIVE_PATH = (
    "scripts/research/hftf/"
    "test_plan_stage_c_d2_official_train_metadata.py"
)
SESSION_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateIneligible(ValueError):
    """The source metadata does not satisfy the frozen eligibility rule."""


@dataclass(frozen=True)
class MetadataApi:
    """The only network-capable operations available to this planner."""

    get_object: Callable[[str, int], dict[str, Any]]
    fetch_text: Callable[[str, int], str]
    fetch_json: Callable[[str, int], dict[str, Any]]
    list_objects: Callable[[str, int], list[dict[str, Any]]]


DEFAULT_API = MetadataApi(
    get_object=get_gcs_object,
    fetch_text=fetch_text,
    fetch_json=fetch_json,
    list_objects=list_gcs_objects,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(
    repo_root: Path,
    relative_to: Path,
    raw_value: str,
) -> Path:
    raw = Path(raw_value)
    if raw.is_absolute():
        return raw.resolve()
    if raw.parts and raw.parts[0] in {
        "artifacts.local",
        "docs",
        "scripts",
    }:
        return (repo_root / raw).resolve()
    return (relative_to / raw).resolve()


def _bound_json(
    repo_root: Path,
    relative_to: Path,
    receipt: dict[str, Any],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _resolve(repo_root, relative_to, str(receipt["path"]))
    if _sha256(path) != str(receipt["sha256"]):
        raise ValueError(f"D2 metadata parent hash mismatch: {label}")
    return path, _load_json(path)


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["session_id"]) for record in records]


def _unique_ids(
    values: list[str],
    expected_count: int,
    label: str,
) -> set[str]:
    if (
        len(values) != expected_count
        or len(values) != len(set(values))
        or any(SESSION_ID_RE.fullmatch(value) is None for value in values)
    ):
        raise ValueError(f"D2 {label} source set is invalid")
    return set(values)


def _validate_self_hash(expected_sha256: str) -> str:
    actual = _sha256(Path(__file__).resolve())
    if expected_sha256 != actual:
        raise ValueError("D2 metadata planner implementation hash mismatch")
    return actual


def _require_tracked_clean_path(
    path: Path,
    repo_root: Path,
    label: str,
) -> None:
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            f"D2 {label} must be inside the repository"
        ) from error

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"D2 {label} is not committed and clean: "
                + (result.stderr.strip() or "git verification failed")
            )
        return result.stdout.strip()

    relative_text = relative.as_posix()
    git("ls-files", "--error-unmatch", "--", relative_text)
    git("diff", "--quiet", "--", relative_text)
    git("diff", "--cached", "--quiet", "--", relative_text)


def _require_committed_pushed_contract(
    contract_path: Path,
    repo_root: Path,
) -> None:
    try:
        relative = contract_path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            "D2 execution contract must be inside the repository"
        ) from error
    if (
        relative.parts[:3] != ("docs", "research", "hftf")
        or contract_path.suffix.lower() != ".json"
    ):
        raise ValueError(
            "D2 execution contract must be a tracked HFTF JSON document"
        )
    _require_tracked_clean_path(
        contract_path,
        repo_root,
        "execution contract",
    )

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                "D2 execution contract is not committed and pushed: "
                + (result.stderr.strip() or "git verification failed")
            )
        return result.stdout.strip()

    if git("rev-parse", "HEAD") != git("rev-parse", "origin/master"):
        raise ValueError(
            "D2 execution contract HEAD is not equal to origin/master"
        )


def _validate_historical_burns(
    repo_root: Path,
    f0_ledger_path: Path,
    f0_ledger: dict[str, Any],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    if (
        f0_ledger.get("schema") != F0_LEDGER_SCHEMA
        or f0_ledger.get("status") != "FROZEN_BEFORE_F0_SOURCE_OUTCOME"
    ):
        raise ValueError("D2 F0 burn ledger identity mismatch")
    r4_receipt = f0_ledger["parent_r4_burn_ledger"]
    r4_path, r4 = _bound_json(
        repo_root,
        f0_ledger_path.parent,
        r4_receipt,
        "r4_burn_ledger",
    )
    if (
        r4.get("schema") != R4_LEDGER_SCHEMA
        or r4.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("D2 R4 burn ledger identity mismatch")
    r4_ids = [str(value) for value in r4["burned_session_ids"]]
    r4_set = _unique_ids(
        r4_ids,
        int(r4["burned_session_count"]),
        "R4 burned",
    )
    r3_path, r3 = _bound_json(
        repo_root,
        r4_path.parent,
        r4["parent_r3_1_burn_ledger"],
        "r3_1_burn_ledger",
    )
    if (
        r3.get("schema") != R3_1_LEDGER_SCHEMA
        or r3.get("status")
        != "FROZEN_BEFORE_R3_1_QUALIFICATION"
    ):
        raise ValueError("D2 R3.1 burn ledger identity mismatch")
    r3_ids = [str(value) for value in r3["burned_session_ids"]]
    r3_set = _unique_ids(
        r3_ids,
        int(r3["burned_session_count"]),
        "R3.1 burned",
    )
    if not r3_set.issubset(r4_set):
        raise ValueError("D2 R4 burn ledger dropped an R3.1 burn")
    additional = [
        str(value)
        for value in f0_ledger[
            "additional_r4_outcome_open_session_ids"
        ]
    ]
    additional_set = _unique_ids(
        additional,
        len(additional),
        "F0 additional burned",
    )
    if r4_set & additional_set:
        raise ValueError("D2 F0 burn union contains duplicates")
    effective = r4_set | additional_set
    if len(effective) != int(
        f0_ledger["effective_burned_session_count"]
    ):
        raise ValueError("D2 effective historical burn count mismatch")
    receipts = {
        "f0_burn_ledger": {
            "path": str(f0_ledger_path.resolve()),
            "sha256": _sha256(f0_ledger_path),
        },
        "r4_burn_ledger": {
            "path": str(r4_path),
            "sha256": _sha256(r4_path),
        },
        "r3_1_burn_ledger": {
            "path": str(r3_path),
            "sha256": _sha256(r3_path),
        },
    }
    return effective, receipts


def _validate_train_split_binding(
    repo_root: Path,
    source_plan_path: Path,
    source_plan: dict[str, Any],
) -> tuple[str, str, dict[str, dict[str, Any]]]:
    protocol_path = _resolve(
        repo_root,
        source_plan_path.parent,
        str(source_plan["protocol_path"]),
    )
    if _sha256(protocol_path) != str(source_plan["protocol_sha256"]):
        raise ValueError("D2 G0 source-plan protocol hash mismatch")
    protocol = _load_json(protocol_path)
    f0_plan_path, f0_plan = _bound_json(
        repo_root,
        protocol_path.parent,
        protocol["parents"]["f0_inventory_plan"],
        "f0_inventory_plan",
    )
    if (
        f0_plan.get("schema") != F0_PLAN_SCHEMA
        or f0_plan.get("terminal") != F0_PLAN_TERMINAL
    ):
        raise ValueError("D2 F0 inventory-plan identity mismatch")
    generation = str(f0_plan["split_object_generation"])
    split_sha256 = str(f0_plan["split_text_sha256"])
    if not generation or re.fullmatch(r"[0-9a-f]{64}", split_sha256) is None:
        raise ValueError("D2 official-train split receipt is invalid")
    receipts = {
        "g0_source_plan_protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "f0_inventory_plan": {
            "path": str(f0_plan_path),
            "sha256": _sha256(f0_plan_path),
        },
    }
    return generation, split_sha256, receipts


def _validate_parent_chain(
    design_path: Path,
    t0_result_path: Path,
    expected_planner_sha256: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (
        repo_root or Path(__file__).resolve().parents[3]
    ).resolve()
    design = _load_json(design_path)
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("status") != DESIGN_STATUS
        or design.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise ValueError("D2 design identity mismatch")
    selection = design["metadata_only_source_qualification"]
    if (
        selection.get("official_split") != "train"
        or selection.get("deterministic_order")
        != (
            "ascending_session_id_from_generation_and_sha256_bound_"
            "official_train_split"
        )
        or int(selection.get("requested_parent_count", -1)) != 6
        or selection.get("new_parent_sessions_only") is not True
        or selection.get("source_replacement_after_scan") is not False
        or selection.get("insufficient_parent_terminal") != INSUFFICIENT
    ):
        raise ValueError("D2 metadata selection contract mismatch")
    actual_planner_sha256 = _validate_self_hash(
        expected_planner_sha256
    )

    source_plan_path, source_plan = _bound_json(
        repo_root,
        design_path.parent,
        design["parents"]["g0_source_plan"],
        "g0_source_plan",
    )
    if (
        source_plan.get("schema") != SOURCE_PLAN_SCHEMA
        or source_plan.get("terminal") != SOURCE_PLAN_TERMINAL
        or source_plan.get("all_roles_parent_session_disjoint")
        is not True
    ):
        raise ValueError("D2 G0 source plan identity mismatch")
    roles = source_plan["roles"]
    development = _unique_ids(
        _ids(roles["development_reuse"]),
        9,
        "G0-D1 development",
    )
    closed_d1 = _unique_ids(
        _ids(roles["one_shot_fresh_evaluation"]),
        3,
        "closed G0-D1 fresh cohort",
    )
    reserved_test = _unique_ids(
        _ids(roles["reserved_fresh_heldout"]),
        3,
        "G0 reserved official-test",
    )
    if development & closed_d1:
        raise ValueError("D2 G0 train role sets overlap")
    if any(
        item.get("official_split") != "train"
        for item in (
            roles["development_reuse"]
            + roles["one_shot_fresh_evaluation"]
        )
    ) or any(
        item.get("official_split") != "test"
        for item in roles["reserved_fresh_heldout"]
    ):
        raise ValueError("D2 G0 source-plan split roles drifted")

    f0_ledger_path, f0_ledger = _bound_json(
        repo_root,
        design_path.parent,
        design["parents"]["f0_burn_ledger"],
        "f0_burn_ledger",
    )
    historical, ledger_receipts = _validate_historical_burns(
        repo_root,
        f0_ledger_path,
        f0_ledger,
    )
    f0_1_path, f0_1_result = _bound_json(
        repo_root,
        design_path.parent,
        design["parents"]["f0_1_consumed_official_test_result"],
        "f0_1_consumed_official_test_result",
    )
    if (
        f0_1_result.get("schema") != F0_1_RESULT_SCHEMA
        or f0_1_result.get("terminal") != F0_1_RESULT_TERMINAL
    ):
        raise ValueError("D2 F0.1 result identity mismatch")
    consumed_test = _unique_ids(
        [
            str(value)
            for value in f0_1_result["burn_and_authorization"][
                "official_test_parent_sessions_consumed_for_f0_1_effect"
            ]
        ],
        3,
        "F0.1 consumed official-test",
    )
    category_sets = (
        historical,
        development,
        closed_d1,
        consumed_test,
        reserved_test,
    )
    for index, left in enumerate(category_sets):
        for right in category_sets[index + 1 :]:
            if left & right:
                raise ValueError("D2 exclusion categories overlap")
    excluded = set().union(*category_sets)
    if len(excluded) != 78:
        raise ValueError("D2 exclusion union must contain 78 parents")

    t0_result = _load_json(t0_result_path)
    if (
        t0_result.get("schema") != T0_SCHEMA
        or t0_result.get("terminal") != T0_TERMINAL
        or t0_result.get("workflow_profile") != "THESIS_DEVELOPMENT"
        or t0_result["authorization"].get(
            "freeze_d2_metadata_qualification_implementation_contract"
        )
        is not True
        or t0_result["authorization"].get(
            "execute_d2_metadata_scan_now"
        )
        is not False
    ):
        raise ValueError("D2 T0 result identity or authorization mismatch")
    if str(t0_result["source"]["session_id"]) not in development:
        raise ValueError("D2 T0 source was not consumed development")
    t0_contract_path, t0_contract = _bound_json(
        repo_root,
        t0_result_path.parent,
        t0_result["contract"],
        "t0_contract",
    )
    d2_receipt = t0_contract["parents"]["d2_design"]
    bound_design_path = _resolve(
        repo_root,
        t0_contract_path.parent,
        str(d2_receipt["path"]),
    )
    if (
        bound_design_path != design_path.resolve()
        or str(d2_receipt["sha256"]) != _sha256(design_path)
    ):
        raise ValueError("D2 T0 contract design binding mismatch")
    source_receipt = t0_contract["parents"]["g0_source_plan"]
    bound_source_path = _resolve(
        repo_root,
        t0_contract_path.parent,
        str(source_receipt["path"]),
    )
    if (
        bound_source_path != source_plan_path
        or str(source_receipt["sha256"]) != _sha256(source_plan_path)
    ):
        raise ValueError("D2 T0 contract source-plan binding mismatch")

    split_generation, split_sha256, split_receipts = (
        _validate_train_split_binding(
            repo_root,
            source_plan_path,
            source_plan,
        )
    )
    return {
        "design": design,
        "selection": selection,
        "excluded": excluded,
        "exclusion_categories": {
            "historical_burned_or_consumed": sorted(historical),
            "g0_d1_development": sorted(development),
            "closed_g0_d1_fresh_cohort": sorted(closed_d1),
            "consumed_f0_1_official_test": sorted(consumed_test),
            "g0_reserved_official_test": sorted(reserved_test),
        },
        "split_generation": split_generation,
        "split_sha256": split_sha256,
        "bindings": {
            "d2_design": {
                "path": str(design_path.resolve()),
                "sha256": _sha256(design_path),
            },
            "t0_result": {
                "path": str(t0_result_path.resolve()),
                "sha256": _sha256(t0_result_path),
            },
            "t0_contract": {
                "path": str(t0_contract_path),
                "sha256": _sha256(t0_contract_path),
            },
            "g0_source_plan": {
                "path": str(source_plan_path),
                "sha256": _sha256(source_plan_path),
            },
            "f0_1_consumed_official_test_result": {
                "path": str(f0_1_path),
                "sha256": _sha256(f0_1_path),
            },
            **ledger_receipts,
            **split_receipts,
            "metadata_planner": {
                "path": PLANNER_RELATIVE_PATH,
                "sha256": actual_planner_sha256,
            },
        },
    }


def _contract_parent(
    contract: dict[str, Any],
    key: str,
    expected_path: Path,
    expected_sha256: str,
    repo_root: Path,
    contract_root: Path,
    required_terminal: str | None = None,
) -> None:
    receipt = contract["parents"][key]
    actual_path = _resolve(
        repo_root,
        contract_root,
        str(receipt["path"]),
    )
    if actual_path != expected_path.resolve():
        raise ValueError(f"D2 execution contract path mismatch: {key}")
    if str(receipt["sha256"]) != expected_sha256:
        raise ValueError(f"D2 execution contract hash mismatch: {key}")
    if (
        required_terminal is not None
        and str(receipt.get("required_terminal")) != required_terminal
    ):
        raise ValueError(
            f"D2 execution contract terminal mismatch: {key}"
        )


def _validate_frozen_inputs(
    execution_contract_path: Path,
    repo_root: Path | None = None,
    verify_tracked_contract: bool = True,
) -> dict[str, Any]:
    repo_root = (
        repo_root or Path(__file__).resolve().parents[3]
    ).resolve()
    if verify_tracked_contract:
        _require_committed_pushed_contract(
            execution_contract_path,
            repo_root,
        )
    contract = _load_json(execution_contract_path)
    if (
        contract.get("schema") != EXECUTION_CONTRACT_SCHEMA
        or contract.get("status") != EXECUTION_CONTRACT_STATUS
        or contract.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise ValueError("D2 metadata execution contract identity mismatch")
    implementation = contract["implementations"]["metadata_planner"]
    if (
        Path(str(implementation.get("path", ""))).as_posix()
        != PLANNER_RELATIVE_PATH
        or implementation.get("metadata_network_execution_authorized")
        is not True
    ):
        raise ValueError(
            "D2 metadata execution contract planner authorization mismatch"
        )
    expected_planner_sha256 = str(implementation["sha256"])
    implementation_test = contract["implementation_tests"][
        "metadata_planner_test"
    ]
    if (
        Path(str(implementation_test.get("path", ""))).as_posix()
        != PLANNER_TEST_RELATIVE_PATH
        or str(implementation_test.get("sha256", ""))
        != _sha256(repo_root / PLANNER_TEST_RELATIVE_PATH)
        or int(contract["implementation_tests"].get("test_count", -1))
        != 14
        or int(contract["implementation_tests"].get("tests_passed", -1))
        != 14
    ):
        raise ValueError(
            "D2 metadata planner test receipt mismatch"
        )
    if verify_tracked_contract:
        _require_tracked_clean_path(
            repo_root / PLANNER_RELATIVE_PATH,
            repo_root,
            "metadata planner implementation",
        )
        _require_tracked_clean_path(
            repo_root / PLANNER_TEST_RELATIVE_PATH,
            repo_root,
            "metadata planner test",
        )
    design_receipt = contract["parents"]["d2_design"]
    t0_receipt = contract["parents"]["t0_result"]
    design_path = _resolve(
        repo_root,
        execution_contract_path.parent,
        str(design_receipt["path"]),
    )
    t0_result_path = _resolve(
        repo_root,
        execution_contract_path.parent,
        str(t0_receipt["path"]),
    )
    if _sha256(design_path) != str(design_receipt["sha256"]):
        raise ValueError("D2 execution contract design hash mismatch")
    if _sha256(t0_result_path) != str(t0_receipt["sha256"]):
        raise ValueError("D2 execution contract T0 hash mismatch")
    frozen = _validate_parent_chain(
        design_path,
        t0_result_path,
        expected_planner_sha256,
        repo_root,
    )
    bindings = frozen["bindings"]
    _contract_parent(
        contract,
        "d2_design",
        Path(str(bindings["d2_design"]["path"])),
        str(bindings["d2_design"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
    )
    _contract_parent(
        contract,
        "t0_result",
        Path(str(bindings["t0_result"]["path"])),
        str(bindings["t0_result"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
        T0_TERMINAL,
    )
    _contract_parent(
        contract,
        "g0_source_plan",
        Path(str(bindings["g0_source_plan"]["path"])),
        str(bindings["g0_source_plan"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
        SOURCE_PLAN_TERMINAL,
    )
    _contract_parent(
        contract,
        "f0_burn_ledger",
        Path(str(bindings["f0_burn_ledger"]["path"])),
        str(bindings["f0_burn_ledger"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
    )
    _contract_parent(
        contract,
        "r4_burn_ledger",
        Path(str(bindings["r4_burn_ledger"]["path"])),
        str(bindings["r4_burn_ledger"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
    )
    _contract_parent(
        contract,
        "r3_1_burn_ledger",
        Path(str(bindings["r3_1_burn_ledger"]["path"])),
        str(bindings["r3_1_burn_ledger"]["sha256"]),
        repo_root,
        execution_contract_path.parent,
    )
    _contract_parent(
        contract,
        "f0_1_consumed_official_test_result",
        Path(
            str(
                bindings[
                    "f0_1_consumed_official_test_result"
                ]["path"]
            )
        ),
        str(
            bindings["f0_1_consumed_official_test_result"]["sha256"]
        ),
        repo_root,
        execution_contract_path.parent,
        F0_1_RESULT_TERMINAL,
    )
    split = contract["official_train_split"]
    if (
        str(split.get("object_generation"))
        != frozen["split_generation"]
        or str(split.get("text_sha256")) != frozen["split_sha256"]
    ):
        raise ValueError("D2 execution contract train split mismatch")
    source_selection = contract["source_selection"]
    if (
        source_selection.get("official_split") != "train"
        or source_selection.get("order") != "ascending_session_id"
        or int(source_selection.get("parent_count", -1)) != 6
        or source_selection.get("source_replacement_after_scan")
        is not False
        or source_selection.get("insufficient_terminal") != INSUFFICIENT
    ):
        raise ValueError(
            "D2 execution contract source selection mismatch"
        )
    canonical = str(
        contract["canonical_artifacts"]["metadata_qualification"]
    )
    if Path(canonical).as_posix() != CANONICAL_RELATIVE_OUTPUT.as_posix():
        raise ValueError(
            "D2 execution contract canonical output mismatch"
        )
    authorization = contract["authorization"]
    required_false = (
        "new_media_acquisition_authorized",
        "camera_pose_content_read_authorized",
        "geometry_teacher_execution_authorized",
        "student_execution_authorized",
        "d2_mechanics_execution_authorized",
        "reserved_official_test_open_authorized",
        "research_mainline_changed",
        "default_app_changed",
        "android_changed",
        "production_authorized",
        "safety_claim_authorized",
    )
    if (
        authorization.get("metadata_scan_execution_authorized")
        is not True
        or any(authorization.get(key) is not False for key in required_false)
    ):
        raise ValueError(
            "D2 execution contract authorization firewall mismatch"
        )
    failure_policy = contract["failure_policy"]
    if (
        int(
            failure_policy.get(
                "internal_retries_per_metadata_request",
                -1,
            )
        )
        != 3
        or failure_policy.get("do_not_rerun_same_scan") is not True
        or failure_policy.get("do_not_replace_or_append_sources_after_scan")
        is not True
        or failure_policy.get("do_not_open_media_or_pose_content")
        is not True
        or failure_policy.get("execution_failure_terminal")
        != "D2_METADATA_QUALIFICATION_NOT_EVALUABLE_NO_RETRY"
    ):
        raise ValueError(
            "D2 execution contract failure policy mismatch"
        )
    frozen["execution_contract"] = contract
    frozen["internal_retries_per_metadata_request"] = 3
    frozen["bindings"] = {
        "execution_contract": {
            "path": str(execution_contract_path.resolve()),
            "sha256": _sha256(execution_contract_path),
        },
        **bindings,
        "metadata_planner_test": {
            "path": PLANNER_TEST_RELATIVE_PATH,
            "sha256": _sha256(
                repo_root / PLANNER_TEST_RELATIVE_PATH
            ),
        },
    }
    return frozen


def _receipt(
    item: dict[str, Any],
    expected_name: str,
    label: str,
) -> dict[str, Any]:
    receipt = object_inventory(item)
    if str(receipt.get("name")) != expected_name:
        raise CandidateIneligible(f"{label} object name mismatch")
    generation = str(receipt.get("generation") or "")
    md5 = str(receipt.get("md5_base64") or "")
    size = receipt.get("size")
    if not generation or not md5 or size is None or int(size) <= 0:
        raise CandidateIneligible(
            f"{label} requires generation, positive size, and MD5"
        )
    return receipt


def _validate_intrinsics(dimensions: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("fx", "fy", "cx", "cy"):
        value = float(dimensions[key])
        if not math.isfinite(value) or value <= 0:
            raise CandidateIneligible(
                f"camera intrinsic {key} must be finite and positive"
            )
        result[key] = value
    for key in ("image_width", "image_height"):
        value = int(dimensions[key])
        if value <= 0:
            raise CandidateIneligible(
                f"camera dimension {key} must be positive"
            )
        result[key] = value
    return result


def _timeline(source_fps: float) -> list[int]:
    if source_fps == 5.0:
        return list(range(13))
    if source_fps == 20.0:
        return list(range(0, 49, 4))
    raise CandidateIneligible("source fps must be exactly 5 or 20")


def _modality_receipt(
    objects: list[dict[str, Any]],
    suffix: str,
    label: str,
) -> dict[str, Any]:
    try:
        indexed = indexed_objects(objects, suffix)
    except ValueError as error:
        raise CandidateIneligible(str(error)) from error
    required = set(range(50))
    if not required.issubset(indexed):
        raise CandidateIneligible(
            f"{label} requires aligned source frames 0..49"
        )
    receipts: list[dict[str, Any]] = []
    for frame_index in range(50):
        item = indexed[frame_index]
        name = str(item.get("name", ""))
        receipt = _receipt(item, name, f"{label} frame {frame_index}")
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
        "required_frame_count": 50,
        "required_frame_indices": [0, 49],
        "required_frame_receipts_sha256": _canonical_json_sha256(
            receipts
        ),
        "required_frame_receipts": receipts,
    }


def _qualify_candidate_unchecked(
    session_id: str,
    rank: int,
    retries: int,
    api: MetadataApi,
) -> dict[str, Any]:
    prefix = f"{GCS_PREFIX}/sanpo-synthetic/{session_id}"
    description_name = f"{prefix}/description.json"
    pose_name = f"{prefix}/camera_chest/camera_poses.csv"
    description_object = api.get_object(description_name, retries)
    pose_object = api.get_object(pose_name, retries)
    description_receipt = _receipt(
        description_object,
        description_name,
        "description",
    )
    pose_receipt = _receipt(
        pose_object,
        pose_name,
        "camera pose",
    )
    description = api.fetch_json(
        media_url(description_name, description_receipt["generation"]),
        retries,
    )
    if description.get("session_type") != "synthetic":
        raise CandidateIneligible("description session_type is not synthetic")
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
        modalities[label] = _modality_receipt(
            objects,
            suffix,
            label,
        )
    return {
        "session_id": session_id,
        "official_split": "train",
        "role": "one_shot_thesis_development_mechanics_evaluation",
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


def _qualify_candidate(
    session_id: str,
    rank: int,
    retries: int,
    api: MetadataApi,
) -> dict[str, Any]:
    try:
        return _qualify_candidate_unchecked(
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


def _parse_split_ids(split_text: str) -> list[str]:
    values = [
        line.strip()
        for line in split_text.splitlines()
        if line.strip()
    ]
    if (
        not values
        or len(values) != len(set(values))
        or any(SESSION_ID_RE.fullmatch(value) is None for value in values)
    ):
        raise ValueError("Official train split contains invalid session IDs")
    return sorted(values)


def _scan_candidates(
    ordered_session_ids: list[str],
    excluded: set[str],
    required_count: int,
    qualifier: Callable[[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
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
            candidate = qualifier(session_id, len(selected) + 1)
        except CandidateIneligible as error:
            ledger.append(
                {
                    "session_id": session_id,
                    "metadata_eligible": False,
                    "reason": str(error),
                }
            )
            continue
        selected.append(candidate)
        ledger.append(candidate)
        if len(selected) == required_count:
            break
    return selected, ledger


def plan(
    execution_contract_path: Path,
    retries: int,
    api: MetadataApi = DEFAULT_API,
    frozen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frozen = frozen or _validate_frozen_inputs(
        execution_contract_path,
    )
    if retries != int(
        frozen["internal_retries_per_metadata_request"]
    ):
        raise ValueError(
            "Retries must equal the frozen contract value 3"
        )
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/train_session_ids.txt"
    )
    split_object = api.get_object(split_name, retries)
    split_receipt = _receipt(
        split_object,
        split_name,
        "official train split",
    )
    if str(split_receipt["generation"]) != frozen["split_generation"]:
        raise ValueError("Official train split generation drift")
    split_text = api.fetch_text(
        media_url(split_name, split_receipt["generation"]),
        retries,
    )
    if _sha256_text(split_text) != frozen["split_sha256"]:
        raise ValueError("Official train split SHA-256 drift")
    ordered_ids = _parse_split_ids(split_text)
    required_count = int(
        frozen["selection"]["requested_parent_count"]
    )
    selected, scan_ledger = _scan_candidates(
        ordered_ids,
        frozen["excluded"],
        required_count,
        lambda session_id, rank: _qualify_candidate(
            session_id,
            rank,
            retries,
            api,
        ),
    )
    terminal = (
        READY if len(selected) == required_count else INSUFFICIENT
    )
    actual_end_hash = _sha256(Path(__file__).resolve())
    if (
        actual_end_hash
        != frozen["bindings"]["metadata_planner"]["sha256"]
    ):
        raise ValueError("D2 metadata planner changed during execution")
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "evidence_role": (
            "METADATA_ONLY_SOURCE_QUALIFICATION_"
            "NOT_MEDIA_OR_EFFECT_EVIDENCE"
        ),
        "bindings": frozen["bindings"],
        "official_train_split": {
            "object_receipt": split_receipt,
            "text_sha256": frozen["split_sha256"],
            "session_count": len(ordered_ids),
            "selection_order": "ascending_session_id",
            "input_split_order_used_for_selection": False,
        },
        "exclusions": {
            "category_counts": {
                key: len(value)
                for key, value in frozen[
                    "exclusion_categories"
                ].items()
            },
            "excluded_parent_count": len(frozen["excluded"]),
            "category_session_ids": frozen["exclusion_categories"],
        },
        "requested_parent_count": required_count,
        "qualified_parent_count": len(selected),
        "qualified_parents": selected,
        "scan_ledger": scan_ledger,
        "firewall": {
            "rgb_bytes_read": False,
            "panoptic_mask_bytes_read": False,
            "metric_depth_bytes_read": False,
            "camera_pose_content_read": False,
            "geometry_teacher_outcome_read": False,
            "student_outcome_read": False,
            "reserved_official_test_opened": False,
        },
        "authorization": {
            "source_replacement_after_scan": False,
            "new_media_acquisition_authorized": False,
            "camera_pose_content_read_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "student_execution_authorized": False,
            "d2_mechanics_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }


def _require_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    expected = (repo_root / CANONICAL_RELATIVE_OUTPUT).resolve()
    if path.resolve() != expected:
        raise ValueError("D2 metadata qualification output is not canonical")
    return expected


def _write_json_exclusive(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _open_attempt(
    output: Path,
    frozen: dict[str, Any],
    retries: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=False)
    attempt_path = output.parent / "attempt.json"
    _write_json_exclusive(
        attempt_path,
        {
            "schema": (
                "blindassist_hftf_stage_c_d2_official_train_"
                "metadata_qualification_attempt"
            ),
            "status": "ATTEMPT_OPENED_BEFORE_FIRST_NETWORK_REQUEST",
            "workflow_profile": "THESIS_DEVELOPMENT",
            "bindings": frozen["bindings"],
            "internal_retries_per_metadata_request": retries,
            "rerun_authorized": False,
            "new_media_acquisition_authorized": False,
            "camera_pose_content_read_authorized": False,
            "d2_mechanics_execution_authorized": False,
        },
    )
    return attempt_path


def _execution_failure_report(
    frozen: dict[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "terminal": (
            "D2_METADATA_QUALIFICATION_NOT_EVALUABLE_NO_RETRY"
        ),
        "workflow_profile": "THESIS_DEVELOPMENT",
        "evidence_role": (
            "FAILED_METADATA_ONLY_SOURCE_QUALIFICATION_"
            "NOT_MEDIA_OR_EFFECT_EVIDENCE"
        ),
        "bindings": frozen["bindings"],
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "firewall": {
            "rgb_bytes_read": False,
            "panoptic_mask_bytes_read": False,
            "metric_depth_bytes_read": False,
            "camera_pose_content_read": False,
            "geometry_teacher_outcome_read": False,
            "student_outcome_read": False,
            "reserved_official_test_opened": False,
        },
        "authorization": {
            "rerun_same_scan_authorized": False,
            "source_replacement_after_scan": False,
            "new_media_acquisition_authorized": False,
            "camera_pose_content_read_authorized": False,
            "geometry_teacher_execution_authorized": False,
            "student_execution_authorized": False,
            "d2_mechanics_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: Path | None = None
    frozen: dict[str, Any] | None = None
    attempt_opened = False
    try:
        output = _require_output(args.output)
        if output.exists() or output.parent.exists():
            raise FileExistsError(
                "Refusing to overwrite D2 metadata qualification"
            )
        frozen = _validate_frozen_inputs(
            args.execution_contract.resolve(),
        )
        if args.retries != int(
            frozen["internal_retries_per_metadata_request"]
        ):
            raise ValueError(
                "Retries must equal the frozen contract value 3"
            )
        _open_attempt(output, frozen, args.retries)
        attempt_opened = True
        report = plan(
            args.execution_contract.resolve(),
            args.retries,
            frozen=frozen,
        )
        _write_json_exclusive(output, report)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "qualified_parent_count": report[
                        "qualified_parent_count"
                    ],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"] == READY else 2
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        KeyboardInterrupt,
    ) as error:
        terminal: str | None = None
        if (
            attempt_opened
            and output is not None
            and frozen is not None
            and not output.exists()
        ):
            failure = _execution_failure_report(frozen, error)
            _write_json_exclusive(output, failure)
            terminal = str(failure["terminal"])
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                    "terminal": terminal,
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
