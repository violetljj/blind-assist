"""A3 Windows extended-path amendment for serial R3 materialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_continuation_a2 as a2
import run_l1e_materialization_recovery_r3_canary as recovery


STAGE = "R2-L1E-RECOVERY-B1-CONTINUATION-A3"
SCHEMA = "blindassist_ustrf_l1e_materialization_recovery_r3_continuation_a3"
EXPECTED_STARTING_LEDGERS = 13
EXPECTED_STARTING_FRAMES = 22_699
EXPECTED_REMAINING_SHARDS = 28
MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS = 3
A3_MEMORY_GUARD_BYTES = 4 * 1024**3
EXPECTED_PARENT_BINDINGS = {
    "first_shard_materializer_config",
    "a2_config",
    "a2_failure_receipt",
    "a2_failure_log",
    "a2_validated_compact_ledger",
    "a2_validated_successor",
}
EXPECTED_IMPLEMENTATION_BINDINGS = {
    "a3_core",
    "a3_one_shard_runner",
    "a3_serial_orchestrator",
    "a3_contract_tests",
}


class ContinuationA3Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationA3Error(f"expected_json_object:{path}")
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
        raise ContinuationA3Error("a3_config_key_roster_drift")
    if (
        config["schema"] != SCHEMA
        or config["stage"] != STAGE
        or config["status"] != "FROZEN_AFTER_A2_OUTPUT_BEFORE_A3_DEVICE_OUTPUT"
    ):
        raise ContinuationA3Error("a3_config_identity_drift")
    if set(config["parent_bindings"]) != EXPECTED_PARENT_BINDINGS:
        raise ContinuationA3Error("a3_parent_binding_roster_drift")
    if set(config["implementation_bindings"]) != EXPECTED_IMPLEMENTATION_BINDINGS:
        raise ContinuationA3Error("a3_implementation_binding_roster_drift")
    for label, binding in config["parent_bindings"].items():
        recovery.verify_binding(repo, binding, f"a3_parent_{label}")
    for label, binding in config["implementation_bindings"].items():
        recovery.verify_binding(repo, binding, f"a3_implementation_{label}")
    if config["execution"] != {
        "maximum_remaining_crowdbot_shards": EXPECTED_REMAINING_SHARDS,
        "one_host_process_per_shard": True,
        "child_processes_serial": True,
        "maximum_cumulative_child_attempts_per_ledger": (
            MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS
        ),
        "windows_extended_path_atomic_writes": True,
        "minimum_system_available_physical_memory_bytes": A3_MEMORY_GUARD_BYTES,
        "memory_guard_amendment_authority": "explicit_user_direction_2026-07-24",
        "stop_after_exhausted_ledger": True,
        "candidate_execution_forbidden": True,
    }:
        raise ContinuationA3Error("a3_execution_contract_drift")
    if config["expected_coverage"] != {
        "starting_ledgers": EXPECTED_STARTING_LEDGERS,
        "starting_frames": EXPECTED_STARTING_FRAMES,
        "final_ledgers": a2.a1.EXPECTED_LEDGERS,
        "final_frames": a2.a1.EXPECTED_FRAMES,
        "discontinuity_resets": a2.a1.EXPECTED_RESETS,
    }:
        raise ContinuationA3Error("a3_coverage_contract_drift")
    if config["authority"] != {
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
    }:
        raise ContinuationA3Error("a3_authority_drift")
    failure = load_json(repo / config["parent_bindings"]["a2_failure_receipt"]["path"])
    if (
        failure.get("status") != "FAIL_CLOSED_ONE_SHARD"
        or failure.get("coverage_after", {}).get("verified_ledgers") != 13
        or failure.get("coverage_after", {}).get("verified_frames") != 22_699
        or failure.get("error", {}).get("type") != "FileNotFoundError"
        or "host-materialization-receipt.json.tmp-" not in failure["error"]["message"]
    ):
        raise ContinuationA3Error("a2_host_receipt_path_failure_drift")
    return config


def materializer_config_path(repo: Path, config: dict[str, Any]) -> Path:
    return (
        repo / config["parent_bindings"]["first_shard_materializer_config"]["path"]
    ).resolve()


def coverage(repo: Path, materializer_config: dict[str, Any]):
    return a2.coverage(repo, materializer_config)


def output_root(repo: Path, materializer_config: dict[str, Any]) -> Path:
    return a2.output_root(repo, materializer_config)


def short_id(source_id: str, sequence_id: str) -> str:
    return a2.short_id(source_id, sequence_id)


def cumulative_attempt_count(
    root: Path, source_id: str, sequence_id: str
) -> int:
    prior = a2.cumulative_attempt_count(root, source_id, sequence_id)
    a3_root = root / "c-a3" / short_id(source_id, sequence_id)
    current = len(list(a3_root.glob("a*/r.json"))) if a3_root.exists() else 0
    return prior + current


def create_control_attempt(
    root: Path, selected: dict[str, Any]
) -> tuple[int, Path]:
    existing = cumulative_attempt_count(
        root, selected["source_id"], selected["sequence_id"]
    )
    if existing >= MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
        raise ContinuationA3Error("a3_cumulative_retry_limit_exhausted")
    number = existing + 1
    path = (
        root
        / "c-a3"
        / short_id(selected["source_id"], selected["sequence_id"])
        / f"a{number:03d}"
        / "r.json"
    )
    path.parent.mkdir(parents=True, exist_ok=False)
    return number, path


def extended_windows_path(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return resolved


def atomic_write_with_extended_path(
    original,
    path: Path,
    payload: Any,
) -> None:
    target = extended_windows_path(path) if len(str(path.resolve())) >= 220 else path
    original(target, payload)


def selected_crowdbot_input(repo: Path, materializer_config, expected):
    return a2.selected_crowdbot_input(repo, materializer_config, expected)


def exclusive_shard_lock(root: Path):
    return a2.exclusive_shard_lock(root)


def atomic_write(path: Path, payload: Any) -> None:
    r1.atomic_write_json(path, payload)

