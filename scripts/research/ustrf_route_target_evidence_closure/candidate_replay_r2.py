#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exploratory_profiles_r2_l1 import (
    ExecutionAborted,
    InputBlocked,
    assert_candidate_input_uncontaminated,
    atomic_write_json,
    compact_paths,
    identity,
    load_and_verify_config,
    load_json,
    load_route_map,
    replay_candidate_ledger,
    sha256_file,
    stable_slug,
    validate_compact_ledger,
    validate_mask_contract,
)

CONFIG_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_r2"
TRACE_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_trace_r2"
TRACE_RECEIPT_SCHEMA = (
    "blindassist_ustrf_route_target_l1_candidate_replay_trace_receipt_r2"
)
TERMINAL_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_terminal_r2"
PREFLIGHT_SCHEMA = "blindassist_ustrf_route_target_l1_candidate_replay_preflight_r2"
TERMINAL_STATES = {
    "CANDIDATE_REPLAY_COMPLETE",
    "FAIL_CLOSED_INPUT_BLOCKED",
    "FAIL_CLOSED_EXECUTION_ABORTED",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def verify_bound_file(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = (repo / binding["path"]).resolve()
    if not path.is_file():
        raise InputBlocked(f"{label}_missing:{binding['path']}")
    observed = sha256_file(path)
    if observed != binding["sha256"]:
        raise InputBlocked(
            f"{label}_sha256_mismatch:expected={binding['sha256']}:observed={observed}"
        )
    return path


def implementation_paths(repo: Path) -> dict[str, Path]:
    module = repo / "scripts/research/ustrf_route_target_evidence_closure"
    return {
        "legacy_replay_core_sha256": module / "exploratory_profiles_r2_l1.py",
        "replay_core_sha256": module / "candidate_replay_r2.py",
        "runner_sha256": module / "run_candidate_replay_r2.py",
        "validator_sha256": module / "validate_candidate_replay_r2.py",
        "mutation_tests_sha256": module / "test_candidate_replay_r2.py",
        "terminal_schema_sha256": (
            repo / "schemas/ustrf_route_target_l1_candidate_replay_r2.schema.json"
        ),
    }


def load_replay_config(
    repo: Path, config_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    list[tuple[dict[str, Any], list[dict[str, Any]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise InputBlocked("unexpected_candidate_replay_config_schema")
    if config.get("stage") != "R2-L1-CANDIDATE-REPLAY-R2":
        raise InputBlocked("unexpected_candidate_replay_stage")
    if set(config.get("terminal_states", [])) != TERMINAL_STATES:
        raise InputBlocked("candidate_replay_terminal_state_contract_drift")
    authority = config.get("authority", {})
    closed = (
        "metric_profile",
        "truth_join",
        "candidate_comparison",
        "winner",
        "ranking",
        "selection",
        "l2",
        "l3",
        "android_shadow",
        "h2",
        "training",
        "human_outcome",
        "independent_walking_safety",
        "production",
        "new_data",
        "device_materialization",
    )
    if any(authority.get(field) is not False for field in closed):
        raise InputBlocked("candidate_replay_authority_must_remain_closed")

    bindings: dict[str, str] = {"config_sha256": sha256_file(config_path)}
    parent = config["parent_bindings"]
    base_path = verify_bound_file(repo, parent["base_exploratory_config"], "base_config")
    base_config, base_bindings = load_and_verify_config(repo, base_path)
    bindings["base_exploratory_config_sha256"] = sha256_file(base_path)
    for key, value in base_bindings.items():
        if key == "config_sha256":
            continue
        bindings[f"base_{key}"] = value

    old_terminal_path = verify_bound_file(
        repo, parent["old_exploratory_terminal"], "old_exploratory_terminal"
    )
    old_terminal = load_json(old_terminal_path)
    if old_terminal.get("terminal_state") != "FAIL_CLOSED_EXECUTION_ABORTED":
        raise InputBlocked("old_exploratory_terminal_not_fail_closed")
    bindings["old_exploratory_terminal_sha256"] = sha256_file(old_terminal_path)

    old_guard_path = verify_bound_file(
        repo, parent["old_resource_guard"], "old_resource_guard"
    )
    old_guard = load_json(old_guard_path)
    if (
        old_guard.get("automatic_retry_allowed_after_receipt") is not False
        or old_guard.get("retry_limit_exhausted") is not True
        or len(old_guard.get("attempts", [])) != 3
    ):
        raise InputBlocked("old_resource_guard_not_exhausted_and_closed")
    bindings["old_resource_guard_sha256"] = sha256_file(old_guard_path)

    recovery_path = verify_bound_file(
        repo, parent["recovery_completion_terminal"], "recovery_completion_terminal"
    )
    recovery = load_json(recovery_path)
    coverage = recovery.get("final_coverage", {})
    if (
        recovery.get("status") != "CANONICAL_INPUT_41_OF_41_COMPLETE"
        or coverage.get("complete") is not True
        or coverage.get("verified_ledgers") != 41
        or coverage.get("verified_frames") != 62229
        or coverage.get("discontinuity_resets") != 15
        or recovery.get("c1_c2_c3_executed") is not False
        or recovery.get("candidate_trace_count") != 0
        or recovery.get("candidate_profile_count") != 0
    ):
        raise InputBlocked("recovery_completion_terminal_contract_drift")
    bindings["recovery_completion_terminal_sha256"] = sha256_file(recovery_path)

    for name, path in implementation_paths(repo).items():
        if not path.is_file():
            raise InputBlocked(f"candidate_replay_implementation_missing:{path}")
        observed = sha256_file(path)
        expected = config["implementation_bindings"][name]
        if observed != expected:
            raise InputBlocked(
                f"{name}_mismatch:expected={expected}:observed={observed}"
            )
        bindings[name] = observed

    mask = load_json(repo / base_config["parent_bindings"]["eligibility_mask"]["path"])
    groups, resets = validate_mask_contract(base_config, mask)
    expected = config["expected_scope"]
    if (
        len(groups) != expected["sequence_ledgers"]
        or sum(len(rows) for _, rows in groups) != expected["frames"]
        or len(resets) != expected["discontinuity_resets"]
        or len(config["candidate_roster"]) != expected["candidates"]
        or len(groups) * len(config["candidate_roster"])
        != expected["candidate_ledger_traces"]
    ):
        raise InputBlocked("candidate_replay_expected_scope_drift")
    if config["candidate_roster"] != base_config["candidate_roster"]:
        raise InputBlocked("candidate_roster_drift_from_frozen_base")
    return config, base_config, bindings, groups, resets, mask


def resolve_input_ledgers(
    repo: Path,
    config: dict[str, Any],
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[tuple[str, str], tuple[Path, Path]], list[dict[str, Any]]]:
    roots = [
        (item["id"], (repo / item["path"]).resolve(), item)
        for item in config["canonical_input_roots"]
    ]
    selected: dict[tuple[str, str], tuple[Path, Path]] = {}
    inventory: list[dict[str, Any]] = []
    counts = {root_id: 0 for root_id, _, _ in roots}
    for descriptor, rows in groups:
        key = (descriptor["source_id"], descriptor["sequence_id"])
        present: list[tuple[str, Path, Path, dict[str, Any]]] = []
        for root_id, root, root_contract in roots:
            ledger, successor = compact_paths(
                root.parent, descriptor["source_id"], descriptor["sequence_id"]
            )
            if ledger.exists() or successor.exists():
                present.append((root_id, ledger, successor, root_contract))
        if len(present) > 1:
            raise InputBlocked(f"duplicate_canonical_input_authority:{key[0]}/{key[1]}")
        if not present:
            raise InputBlocked(f"canonical_input_pair_missing:{key[0]}/{key[1]}")
        root_id, ledger, successor, root_contract = present[0]
        if not validate_compact_ledger(ledger, successor, descriptor, rows):
            raise InputBlocked(f"canonical_input_pair_invalid:{key[0]}/{key[1]}")
        allowed_prefixes = tuple(root_contract["allowed_source_prefixes"])
        if not descriptor["source_id"].startswith(allowed_prefixes):
            raise InputBlocked(
                f"canonical_input_source_root_mismatch:{root_id}:{descriptor['source_id']}"
            )
        selected[key] = (ledger, successor)
        counts[root_id] += 1
        inventory.append(
            {
                "root_id": root_id,
                "source_id": descriptor["source_id"],
                "sequence_id": descriptor["sequence_id"],
                "frame_count": len(rows),
                "frame_mask_sha256": descriptor["frame_mask_sha256"],
                "compact_ledger_path": str(ledger.relative_to(repo)).replace("\\", "/"),
                "compact_ledger_sha256": sha256_file(ledger),
                "successor_receipt_path": str(successor.relative_to(repo)).replace(
                    "\\", "/"
                ),
                "successor_receipt_sha256": sha256_file(successor),
            }
        )
    for root_id, _, contract in roots:
        if counts[root_id] != contract["expected_ledger_count"]:
            raise InputBlocked(
                f"canonical_input_root_count_mismatch:{root_id}:"
                f"expected={contract['expected_ledger_count']}:observed={counts[root_id]}"
            )
    return selected, inventory


def trace_paths(
    output_root: Path, candidate_id: str, descriptor: dict[str, Any]
) -> tuple[Path, Path]:
    slug = stable_slug(descriptor["source_id"], descriptor["sequence_id"])
    candidate_root = output_root / "candidate-traces" / candidate_id / slug
    return candidate_root / "attempts", candidate_root / "authoritative-receipt.json"


def validate_authoritative_trace(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
    candidate_id: str,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
    input_pair: tuple[Path, Path],
    authoritative_path: Path,
) -> dict[str, Any] | None:
    if not authoritative_path.is_file():
        return None
    try:
        receipt = load_json(authoritative_path)
        if receipt.get("schema") != TRACE_RECEIPT_SCHEMA:
            return None
        attempt_trace = repo / receipt["trace_path"]
        if not attempt_trace.is_file():
            return None
        trace = load_json(attempt_trace)
        if (
            receipt.get("status") != "FIRST_VALID_COMPLETE_TRACE"
            or receipt.get("candidate_id") != candidate_id
            or receipt.get("source_id") != descriptor["source_id"]
            or receipt.get("sequence_id") != descriptor["sequence_id"]
            or receipt.get("frame_mask_sha256") != descriptor["frame_mask_sha256"]
            or receipt.get("frame_count") != len(rows)
            or receipt.get("config_sha256") != bindings["config_sha256"]
            or receipt.get("candidate_implementation_sha256")
            != bindings["base_candidate_implementation_sha256"]
            or receipt.get("replay_core_sha256") != bindings["replay_core_sha256"]
            or receipt.get("trace_sha256") != sha256_file(attempt_trace)
            or receipt.get("compact_ledger_sha256") != sha256_file(input_pair[0])
            or receipt.get("successor_receipt_sha256") != sha256_file(input_pair[1])
            or trace.get("schema") != TRACE_SCHEMA
            or trace.get("candidate_id") != candidate_id
            or trace.get("frame_mask_sha256") != descriptor["frame_mask_sha256"]
            or trace.get("frame_count") != len(rows)
        ):
            return None
        frames = trace.get("frames")
        if not isinstance(frames, list) or [identity(row) for row in frames] != [
            identity(row) for row in rows
        ]:
            return None
        assert_candidate_input_uncontaminated(frames)
        return receipt
    except (OSError, KeyError, TypeError, ValueError):
        return None


def materialize_candidate_trace(
    repo: Path,
    config: dict[str, Any],
    base_config: dict[str, Any],
    bindings: dict[str, str],
    candidate_id: str,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
    input_pair: tuple[Path, Path],
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
    reset_next_identities: set[tuple[str, str, int, int]],
    tracker_config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    attempts_root, authoritative_path = trace_paths(
        output_root, candidate_id, descriptor
    )
    existing = validate_authoritative_trace(
        repo,
        config,
        bindings,
        candidate_id,
        descriptor,
        rows,
        input_pair,
        authoritative_path,
    )
    if existing is not None:
        return existing
    if authoritative_path.exists():
        raise ExecutionAborted(
            f"invalid_authoritative_trace_receipt:{candidate_id}:"
            f"{descriptor['source_id']}/{descriptor['sequence_id']}"
        )
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    maximum_attempts = 1 + int(config["execution_contract"]["maximum_retry_count"])
    if len(attempts) >= maximum_attempts:
        raise ExecutionAborted(
            f"candidate_trace_retry_limit_exhausted:{candidate_id}:"
            f"{descriptor['source_id']}/{descriptor['sequence_id']}"
        )
    attempt_number = len(attempts) + 1
    attempt_id = (
        f"{candidate_id}:{stable_slug(descriptor['source_id'], descriptor['sequence_id'])}:"
        f"{attempt_number:03d}:{uuid.uuid4().hex}"
    )
    attempt_root = attempts_root / f"attempt-{attempt_number:03d}-{uuid.uuid4().hex[:12]}"
    attempt_root.mkdir(parents=True)
    started = time.monotonic()
    try:
        ledger = load_json(input_pair[0])
        trace_frames, maximum_rss = replay_candidate_ledger(
            candidate_id,
            ledger,
            route_map,
            reset_next_identities,
            tracker_config,
            base_config,
        )
        for frame in trace_frames:
            frame.pop("candidate_consume_timestamp_ns", None)
        assert_candidate_input_uncontaminated(trace_frames)
        reset_count = sum(
            bool(frame["state_reset_before_frame"]) for frame in trace_frames
        ) - 1
        trace_payload = {
            "schema": TRACE_SCHEMA,
            "stage": config["stage"],
            "authority": "candidate_trace_only_no_profile_comparison_or_promotion_authority",
            "candidate_id": candidate_id,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(trace_frames),
            "discontinuity_reset_count": reset_count,
            "frames": trace_frames,
        }
        trace_path = attempt_root / "trace.json"
        atomic_write_json(trace_path, trace_payload)
        receipt = {
            "schema": TRACE_RECEIPT_SCHEMA,
            "stage": config["stage"],
            "status": "FIRST_VALID_COMPLETE_TRACE",
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "candidate_id": candidate_id,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(trace_frames),
            "first_frame_identity": list(identity(trace_frames[0])),
            "last_frame_identity": list(identity(trace_frames[-1])),
            "discontinuity_reset_count": reset_count,
            "trace_path": str(trace_path.relative_to(repo)).replace("\\", "/"),
            "trace_sha256": sha256_file(trace_path),
            "compact_ledger_sha256": sha256_file(input_pair[0]),
            "successor_receipt_sha256": sha256_file(input_pair[1]),
            "config_sha256": bindings["config_sha256"],
            "candidate_implementation_sha256": bindings[
                "base_candidate_implementation_sha256"
            ],
            "replay_core_sha256": bindings["replay_core_sha256"],
            "maximum_host_rss_bytes": maximum_rss,
            "wall_time_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_authority": False,
            "candidate_comparison_authority": False,
        }
        attempt_receipt = attempt_root / "trace-receipt.json"
        atomic_write_json(attempt_receipt, receipt)
        receipt["attempt_receipt_path"] = str(attempt_receipt.relative_to(repo)).replace(
            "\\", "/"
        )
        receipt["attempt_receipt_sha256"] = sha256_file(attempt_receipt)
        atomic_write_json(authoritative_path, receipt)
        validated = validate_authoritative_trace(
            repo,
            config,
            bindings,
            candidate_id,
            descriptor,
            rows,
            input_pair,
            authoritative_path,
        )
        if validated is None:
            raise ExecutionAborted("candidate_trace_postwrite_validation_failed")
        return validated
    except Exception as error:
        failure_path = attempt_root / "failure-receipt.json"
        if not failure_path.exists():
            atomic_write_json(
                failure_path,
                {
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "status": "INCOMPLETE_NO_PROFILE_AUTHORITY",
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                    "wall_time_seconds": time.monotonic() - started,
                },
            )
        raise


def collect_authoritative_receipts(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    input_pairs: dict[tuple[str, str], tuple[Path, Path]],
    output_root: Path,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for candidate_id in config["candidate_roster"]:
        for descriptor, rows in groups:
            _, path = trace_paths(output_root, candidate_id, descriptor)
            receipt = validate_authoritative_trace(
                repo,
                config,
                bindings,
                candidate_id,
                descriptor,
                rows,
                input_pairs[(descriptor["source_id"], descriptor["sequence_id"])],
                path,
            )
            if receipt is not None:
                receipts.append(
                    {
                        "candidate_id": candidate_id,
                        "source_id": descriptor["source_id"],
                        "sequence_id": descriptor["sequence_id"],
                        "authoritative_receipt_path": str(path.relative_to(repo)).replace(
                            "\\", "/"
                        ),
                        "authoritative_receipt_sha256": sha256_file(path),
                        "trace_path": receipt["trace_path"],
                        "trace_sha256": receipt["trace_sha256"],
                        "frame_count": receipt["frame_count"],
                        "discontinuity_reset_count": receipt[
                            "discontinuity_reset_count"
                        ],
                    }
                )
    return receipts


def base_terminal_receipt(
    state: str,
    config: dict[str, Any],
    bindings: dict[str, str],
    input_inventory: list[dict[str, Any]],
    trace_inventory: list[dict[str, Any]],
    blocker: str | None,
) -> dict[str, Any]:
    if state not in TERMINAL_STATES:
        raise ValueError(state)
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": config["stage"],
        "terminal_state": state,
        "authority": "candidate_replay_only_no_profile_comparison_or_promotion_authority",
        "bindings": bindings,
        "verified_input": {
            "sequence_ledgers": len(input_inventory),
            "frames": sum(row["frame_count"] for row in input_inventory),
            "discontinuity_resets": config["expected_scope"]["discontinuity_resets"],
            "inventory": input_inventory,
        },
        "candidate_execution": {
            "candidate_order": config["candidate_roster"],
            "expected_authoritative_trace_count": config["expected_scope"][
                "candidate_ledger_traces"
            ],
            "authoritative_trace_count": len(trace_inventory),
            "trace_inventory": trace_inventory,
            "partial_trace_profile_authority": False,
            "candidate_comparison_authority": False,
        },
        "profiles": {
            "generated": False,
            "count": 0,
            "authority": False,
        },
        "blocker": blocker,
        "claim_boundary": {
            "truth_joined": False,
            "metric_profile_generated": False,
            "candidate_comparison_allowed": False,
            "winner_or_ranking_allowed": False,
            "selection_allowed": False,
            "l2_or_l3_allowed": False,
            "android_shadow_allowed": False,
            "h2_allowed": False,
            "human_outcome_allowed": False,
            "independent_walking_safety_allowed": False,
            "production_allowed": False,
            "new_data_added": False,
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def preflight_receipt(
    config: dict[str, Any],
    bindings: dict[str, str],
    input_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PREFLIGHT_SCHEMA,
        "stage": config["stage"],
        "status": "INPUT_PREFLIGHT_PASS",
        "bindings": bindings,
        "verified_sequence_ledgers": len(input_inventory),
        "verified_frames": sum(row["frame_count"] for row in input_inventory),
        "verified_discontinuity_resets": config["expected_scope"][
            "discontinuity_resets"
        ],
        "candidate_outputs_executed": False,
        "profile_authority": False,
        "input_inventory_sha256": __import__("hashlib")
        .sha256(canonical_bytes(input_inventory))
        .hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
