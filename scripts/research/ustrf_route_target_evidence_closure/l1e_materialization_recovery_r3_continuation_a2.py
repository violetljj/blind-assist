"""Outcome-unseen A2 controller fix for remaining R3 materialization.

A1 correctly stopped on its terminal receipt.  Its second and third controller
invocations never reached the materializer because a Windows path exceeded the
temporary receipt path limit.  A2 uses short hash-addressed control paths and
counts only durable child receipts toward the cumulative three-attempt limit.
"""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_remaining as a1
import run_l1e_materialization_recovery_r3_canary as recovery


STAGE = "R2-L1E-RECOVERY-B1-CONTINUATION-A2"
SCHEMA = "blindassist_ustrf_l1e_materialization_recovery_r3_continuation_a2"
MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS = 3
EXPECTED_STARTING_LEDGERS = 12
EXPECTED_STARTING_FRAMES = 20_844
EXPECTED_REMAINING_SHARDS = 29
PARENT_MEMORY_GUARD_BYTES = 6 * 1024**3
A2_MEMORY_GUARD_BYTES = 4 * 1024**3
EXPECTED_PARENT_BINDINGS = {
    "first_shard_materializer_config",
    "a1_config",
    "a1_terminal",
    "a1_real_failure_receipt",
    "a1_controller_path_failure_002",
    "a1_controller_path_failure_003",
}
EXPECTED_IMPLEMENTATION_BINDINGS = {
    "a2_core",
    "a2_one_shard_runner",
    "a2_serial_orchestrator",
    "a2_contract_tests",
}


class ContinuationA2Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationA2Error(f"expected_json_object:{path}")
    return value


def verify_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    if set(config) != {
        "schema",
        "stage",
        "status",
        "frozen_on",
        "parent_bindings",
        "execution",
        "expected_coverage",
        "implementation_bindings",
        "authority",
    }:
        raise ContinuationA2Error("a2_config_key_roster_drift")
    if (
        config["schema"] != SCHEMA
        or config["stage"] != STAGE
        or config["status"] != "FROZEN_AFTER_A1_TERMINAL_BEFORE_A2_DEVICE_OUTPUT"
    ):
        raise ContinuationA2Error("a2_config_identity_drift")
    if set(config["parent_bindings"]) != EXPECTED_PARENT_BINDINGS:
        raise ContinuationA2Error("a2_parent_binding_roster_drift")
    if set(config["implementation_bindings"]) != EXPECTED_IMPLEMENTATION_BINDINGS:
        raise ContinuationA2Error("a2_implementation_binding_roster_drift")
    for label, binding in config["parent_bindings"].items():
        recovery.verify_binding(repo, binding, f"a2_parent_{label}")
    for label, binding in config["implementation_bindings"].items():
        recovery.verify_binding(repo, binding, f"a2_implementation_{label}")
    if config["execution"] != {
        "maximum_remaining_crowdbot_shards": EXPECTED_REMAINING_SHARDS,
        "one_host_process_per_shard": True,
        "child_processes_serial": True,
        "maximum_cumulative_child_attempts_per_ledger": (
            MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS
        ),
        "count_only_durable_child_receipts_as_attempts": True,
        "short_hash_addressed_control_paths": True,
        "parent_minimum_system_available_physical_memory_bytes": (
            PARENT_MEMORY_GUARD_BYTES
        ),
        "minimum_system_available_physical_memory_bytes": A2_MEMORY_GUARD_BYTES,
        "memory_guard_amendment_authority": "explicit_user_direction_2026-07-24",
        "stop_after_exhausted_ledger": True,
        "candidate_execution_forbidden": True,
    }:
        raise ContinuationA2Error("a2_execution_contract_drift")
    if config["expected_coverage"] != {
        "starting_ledgers": EXPECTED_STARTING_LEDGERS,
        "starting_frames": EXPECTED_STARTING_FRAMES,
        "final_ledgers": a1.EXPECTED_LEDGERS,
        "final_frames": a1.EXPECTED_FRAMES,
        "discontinuity_resets": a1.EXPECTED_RESETS,
    }:
        raise ContinuationA2Error("a2_coverage_contract_drift")
    expected_authority = {
        "candidate_execution": False,
        "candidate_trace": False,
        "candidate_profile": False,
        "selection": False,
        "ranking": False,
        "recommendation": False,
        "provisional_selection": False,
        "l2_execution": False,
        "l3_execution": False,
        "android_shadow": False,
        "h2": False,
        "human_outcome": False,
        "independent_walking_safety": False,
        "production": False,
    }
    if config["authority"] != expected_authority:
        raise ContinuationA2Error("a2_authority_drift")
    terminal = load_json(repo / config["parent_bindings"]["a1_terminal"]["path"])
    if terminal.get("status") != "FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED":
        raise ContinuationA2Error("a1_terminal_status_drift")
    if terminal.get("current_coverage", {}).get("verified_ledgers") != 12:
        raise ContinuationA2Error("a1_terminal_ledger_coverage_drift")
    if terminal.get("current_coverage", {}).get("verified_frames") != 20_844:
        raise ContinuationA2Error("a1_terminal_frame_coverage_drift")
    failure_002 = (
        repo / config["parent_bindings"]["a1_controller_path_failure_002"]["path"]
    ).read_text(encoding="utf-8")
    failure_003 = (
        repo / config["parent_bindings"]["a1_controller_path_failure_003"]["path"]
    ).read_text(encoding="utf-8")
    for label, text in (("002", failure_002), ("003", failure_003)):
        if "FileNotFoundError" not in text or "control-receipt.json.tmp-" not in text:
            raise ContinuationA2Error(f"a1_controller_path_failure_{label}_drift")
    return config


def materializer_config_path(repo: Path, config: dict[str, Any]) -> Path:
    return (
        repo / config["parent_bindings"]["first_shard_materializer_config"]["path"]
    ).resolve()


def amended_materializer_overlay(
    parent_overlay: dict[str, Any],
) -> dict[str, Any]:
    overlay = copy.deepcopy(parent_overlay)
    if (
        overlay["execution"]["minimum_system_available_physical_memory_bytes"]
        != PARENT_MEMORY_GUARD_BYTES
    ):
        raise ContinuationA2Error("parent_materializer_memory_guard_drift")
    overlay["execution"][
        "minimum_system_available_physical_memory_bytes"
    ] = A2_MEMORY_GUARD_BYTES
    return overlay


def amended_canary_config(parent_canary: dict[str, Any]) -> dict[str, Any]:
    canary = copy.deepcopy(parent_canary)
    if (
        canary["resource_guards"][
            "minimum_system_available_physical_memory_bytes"
        ]
        != PARENT_MEMORY_GUARD_BYTES
    ):
        raise ContinuationA2Error("parent_canary_memory_guard_drift")
    canary["resource_guards"][
        "minimum_system_available_physical_memory_bytes"
    ] = A2_MEMORY_GUARD_BYTES
    return canary


def coverage(repo: Path, materializer_config: dict[str, Any]):
    return a1.coverage(repo, materializer_config)


def output_root(repo: Path, materializer_config: dict[str, Any]) -> Path:
    return a1.output_root(repo, materializer_config)


def short_id(source_id: str, sequence_id: str) -> str:
    return hashlib.sha256(f"{source_id}\0{sequence_id}".encode()).hexdigest()[:16]


def _a1_durable_attempts(
    root: Path, source_id: str, sequence_id: str
) -> int:
    attempt_root = root / "continuation-attempts"
    if not attempt_root.exists():
        return 0
    count = 0
    for path in attempt_root.glob("*/attempt-*/control-receipt.json"):
        try:
            receipt = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        selected = receipt.get("selected", {})
        if (
            selected.get("source_id") == source_id
            and selected.get("sequence_id") == sequence_id
        ):
            count += 1
    return count


def _a2_attempt_root(root: Path, source_id: str, sequence_id: str) -> Path:
    return root / "c-a2" / short_id(source_id, sequence_id)


def cumulative_attempt_count(
    root: Path, source_id: str, sequence_id: str
) -> int:
    a1_count = _a1_durable_attempts(root, source_id, sequence_id)
    a2_root = _a2_attempt_root(root, source_id, sequence_id)
    a2_count = len(list(a2_root.glob("a*/r.json"))) if a2_root.exists() else 0
    return a1_count + a2_count


def create_control_attempt(
    root: Path, selected: dict[str, Any]
) -> tuple[int, Path, Path]:
    existing = cumulative_attempt_count(
        root, selected["source_id"], selected["sequence_id"]
    )
    if existing >= MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
        raise ContinuationA2Error("a2_cumulative_retry_limit_exhausted")
    cumulative_number = existing + 1
    directory = (
        _a2_attempt_root(root, selected["source_id"], selected["sequence_id"])
        / f"a{cumulative_number:03d}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return cumulative_number, directory, directory / "r.json"


def selected_crowdbot_input(
    repo: Path,
    materializer_config: dict[str, Any],
    expected: dict[str, Any],
):
    return a1.selected_crowdbot_input(repo, materializer_config, expected)


def exclusive_shard_lock(root: Path):
    return a1.exclusive_shard_lock(root)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    r1.atomic_write_json(path, payload)
