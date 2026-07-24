#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import candidate_replay_r2 as parent
from exploratory_profiles_r2_l1 import atomic_write_json, load_json, sha256_file

MINIMUM_AVAILABLE_MEMORY_BYTES_A2 = 4 * 1024**3


def short_trace_paths(
    output_root: Path, candidate_id: str, descriptor: dict[str, Any]
) -> tuple[Path, Path]:
    identity = (
        f"{descriptor['source_id']}\0{descriptor['sequence_id']}".encode("utf-8")
    )
    ledger_key = hashlib.sha256(identity).hexdigest()[:24]
    root = output_root / "candidate-traces" / candidate_id / ledger_key
    return root / "attempts", root / "authoritative-receipt.json"


def activate_short_trace_paths() -> None:
    parent.trace_paths = short_trace_paths


def a2_implementation_paths(repo: Path) -> dict[str, Path]:
    module = repo / "scripts/research/ustrf_route_target_evidence_closure"
    return {
        "continuation_a2_core_sha256": module
        / "candidate_replay_r2_continuation_a2.py",
        "continuation_a2_runner_sha256": module
        / "run_candidate_replay_r2_continuation_a2.py",
        "continuation_a2_validator_sha256": module
        / "validate_candidate_replay_r2_continuation_a2.py",
        "continuation_a2_tests_sha256": module
        / "test_candidate_replay_r2_continuation_a2.py",
    }


def load_a2_context(
    repo: Path, config_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    list[tuple[dict[str, Any], list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    context = parent.load_replay_config(repo, config_path)
    config, base_config, bindings, groups, resets, mask = context
    amendment = config.get("resource_amendment_a2", {})
    if (
        amendment.get("authorized_by_user") is not True
        or amendment.get("previous_minimum_available_memory_bytes") != 6 * 1024**3
        or amendment.get("minimum_available_memory_bytes")
        != MINIMUM_AVAILABLE_MEMORY_BYTES_A2
        or amendment.get("candidate_logic_changed") is not False
        or amendment.get("input_or_threshold_changed") is not False
    ):
        raise parent.InputBlocked("continuation_a2_resource_amendment_invalid")
    parent_config = parent.verify_bound_file(
        repo,
        config["continuation_a2_bindings"]["parent_candidate_replay_config"],
        "continuation_a2_parent_config",
    )
    bindings["continuation_a2_parent_config_sha256"] = sha256_file(parent_config)
    for name, path in a2_implementation_paths(repo).items():
        if not path.is_file():
            raise parent.InputBlocked(f"{name}_missing")
        observed = sha256_file(path)
        expected = config["implementation_bindings"][name]
        if observed != expected:
            raise parent.InputBlocked(
                f"{name}_mismatch:expected={expected}:observed={observed}"
            )
        bindings[name] = observed
    base_config = copy.deepcopy(base_config)
    base_config["resource_guards"][
        "minimum_system_available_physical_memory_bytes"
    ] = MINIMUM_AVAILABLE_MEMORY_BYTES_A2
    return config, base_config, bindings, groups, resets, mask


def adopt_parent_traces(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    input_pairs: dict[tuple[str, str], tuple[Path, Path]],
    output_root: Path,
) -> int:
    parent_contract = config["parent_trace_adoption"]
    parent_config_path = (
        repo / parent_contract["parent_config_path"]
    ).resolve()
    (
        parent_config,
        _,
        parent_bindings,
        parent_groups,
        _,
        _,
    ) = parent.load_replay_config(repo, parent_config_path)
    if sha256_file(parent_config_path) != parent_contract["parent_config_sha256"]:
        raise parent.InputBlocked("parent_trace_config_sha256_mismatch")
    parent_output = (repo / parent_config["output_root"]).resolve()
    parent_groups_by_key = {
        (descriptor["source_id"], descriptor["sequence_id"]): (descriptor, rows)
        for descriptor, rows in parent_groups
    }
    default_trace_paths = parent.trace_paths
    adopted = 0
    try:
        for candidate_id in config["candidate_roster"]:
            for descriptor, rows in groups:
                key = (descriptor["source_id"], descriptor["sequence_id"])
                parent_descriptor, parent_rows = parent_groups_by_key[key]
                _, parent_authoritative = default_trace_paths(
                    parent_output, candidate_id, parent_descriptor
                )
                parent_receipt = parent.validate_authoritative_trace(
                    repo,
                    parent_config,
                    parent_bindings,
                    candidate_id,
                    parent_descriptor,
                    parent_rows,
                    input_pairs[key],
                    parent_authoritative,
                )
                if parent_receipt is None:
                    continue
                _, a2_authoritative = short_trace_paths(
                    output_root, candidate_id, descriptor
                )
                if a2_authoritative.exists():
                    continue
                a2_authoritative.parent.mkdir(parents=True, exist_ok=True)
                adopted_receipt = {
                    **parent_receipt,
                    "config_sha256": bindings["config_sha256"],
                    "continuation_a2_core_sha256": bindings[
                        "continuation_a2_core_sha256"
                    ],
                    "adopted_without_candidate_rerun": True,
                    "parent_authoritative_receipt_path": str(
                        parent_authoritative.relative_to(repo)
                    ).replace("\\", "/"),
                    "parent_authoritative_receipt_sha256": sha256_file(
                        parent_authoritative
                    ),
                }
                atomic_write_json(a2_authoritative, adopted_receipt)
                validated = parent.validate_authoritative_trace(
                    repo,
                    config,
                    bindings,
                    candidate_id,
                    descriptor,
                    rows,
                    input_pairs[key],
                    a2_authoritative,
                )
                if validated is None:
                    raise parent.ExecutionAborted(
                        "continuation_a2_parent_trace_adoption_invalid"
                    )
                adopted += 1
    finally:
        parent.trace_paths = default_trace_paths
    expected = parent_contract["expected_parent_authoritative_trace_count"]
    if adopted != expected:
        existing = collect_a2_receipts(
            repo, config, bindings, groups, input_pairs, output_root
        )
        adopted_existing = sum(
            1
            for row in existing
            if load_json(repo / row["authoritative_receipt_path"]).get(
                "adopted_without_candidate_rerun"
            )
            is True
        )
        if adopted_existing != expected:
            raise parent.InputBlocked(
                f"parent_trace_adoption_count_mismatch:expected={expected}:"
                f"observed={adopted_existing}"
            )
    return expected


def bind_a2_marker(
    repo: Path, marker_path: Path, bindings: dict[str, str]
) -> None:
    marker = load_json(marker_path)
    if marker.get("continuation_a2_core_sha256") == bindings[
        "continuation_a2_core_sha256"
    ]:
        return
    marker["continuation_a2_core_sha256"] = bindings[
        "continuation_a2_core_sha256"
    ]
    marker["adopted_without_candidate_rerun"] = False
    atomic_write_json(marker_path, marker)


def collect_a2_receipts(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    input_pairs: dict[tuple[str, str], tuple[Path, Path]],
    output_root: Path,
) -> list[dict[str, Any]]:
    activate_short_trace_paths()
    receipts = parent.collect_authoritative_receipts(
        repo, config, bindings, groups, input_pairs, output_root
    )
    for row in receipts:
        marker = load_json(repo / row["authoritative_receipt_path"])
        if marker.get("continuation_a2_core_sha256") != bindings[
            "continuation_a2_core_sha256"
        ]:
            raise parent.ExecutionAborted("continuation_a2_marker_binding_missing")
    return receipts

