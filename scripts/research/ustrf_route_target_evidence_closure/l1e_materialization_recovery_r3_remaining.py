"""Fail-closed helpers for serial R3 materialization continuation.

This module only reasons about canonical detector-input ledgers.  It does not
import or execute C1-C3 candidate code.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import exploratory_profiles_r2_l1 as r1
import run_l1e_materialization_recovery_r3_canary as recovery


STAGE = "R2-L1E-RECOVERY-B1-CONTINUATION-A1"
SCHEMA = "blindassist_ustrf_l1e_materialization_recovery_r3_continuation_a1"
EXPECTED_LEDGERS = 41
EXPECTED_FRAMES = 62_229
EXPECTED_RESETS = 15
EXPECTED_REMAINING_CROWDBOT_SHARDS = 38
MATERIALIZER_STAGE = recovery.STAGE
MATERIALIZER_NAMESPACE = recovery.ATTEMPT_NAMESPACE
EXPECTED_PARENT_BINDINGS = {
    "first_shard_materializer_config",
    "first_shard_host_receipt",
    "first_shard_compact_ledger",
    "first_shard_successor",
}
EXPECTED_IMPLEMENTATION_BINDINGS = {
    "continuation_core",
    "remaining_one_shard_runner",
    "serial_orchestrator",
    "contract_tests",
}


class ContinuationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"expected_json_object:{path}")
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
        raise ContinuationError("continuation_config_key_roster_drift")
    if (
        config["schema"] != SCHEMA
        or config["stage"] != STAGE
        or config["status"] != "FROZEN_BEFORE_REMAINING_R3_MATERIALIZATION"
    ):
        raise ContinuationError("continuation_config_identity_drift")
    if set(config["parent_bindings"]) != EXPECTED_PARENT_BINDINGS:
        raise ContinuationError("continuation_parent_binding_roster_drift")
    if set(config["implementation_bindings"]) != EXPECTED_IMPLEMENTATION_BINDINGS:
        raise ContinuationError("continuation_implementation_binding_roster_drift")
    for label, binding in config["parent_bindings"].items():
        recovery.verify_binding(repo, binding, f"continuation_parent_{label}")
    for label, binding in config["implementation_bindings"].items():
        recovery.verify_binding(repo, binding, f"continuation_implementation_{label}")
    if config["execution"] != {
        "maximum_total_remaining_crowdbot_shards": EXPECTED_REMAINING_CROWDBOT_SHARDS,
        "one_host_process_per_shard": True,
        "child_processes_serial": True,
        "initial_attempts_per_ledger": 1,
        "bounded_retries_per_ledger": 2,
        "stop_after_exhausted_ledger": True,
        "materializer_stage": MATERIALIZER_STAGE,
        "materializer_attempt_namespace": MATERIALIZER_NAMESPACE,
        "candidate_execution_forbidden": True,
    }:
        raise ContinuationError("continuation_execution_contract_drift")
    if config["expected_coverage"] != {
        "starting_ledgers": 3,
        "starting_frames": 6049,
        "final_ledgers": EXPECTED_LEDGERS,
        "final_frames": EXPECTED_FRAMES,
        "discontinuity_resets": EXPECTED_RESETS,
    }:
        raise ContinuationError("continuation_coverage_contract_drift")
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
        raise ContinuationError("continuation_authority_drift")
    return config


def materializer_config_path(repo: Path, config: dict[str, Any]) -> Path:
    binding = config["parent_bindings"]["first_shard_materializer_config"]
    return (repo / binding["path"]).resolve()


def base_prereg_path(repo: Path, materializer_config: dict[str, Any]) -> Path:
    canary_path = repo / materializer_config["canary_config"]["path"]
    canary = recovery.verify_config(repo, canary_path)
    return (repo / canary["parent_bindings"]["base_prereg"]["path"]).resolve()


def output_root(repo: Path, materializer_config: dict[str, Any]) -> Path:
    canary_path = repo / materializer_config["canary_config"]["path"]
    canary = recovery.verify_config(repo, canary_path)
    root = (repo / canary["output_root"]).resolve()
    try:
        root.relative_to((repo / "artifacts.local").resolve())
    except ValueError as error:
        raise ContinuationError("continuation_output_root_escapes_artifacts_local") from error
    return root


def coverage(
    repo: Path,
    materializer_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_path = base_prereg_path(repo, materializer_config)
    prereg, context = recovery.build_context(repo, prereg_path)
    parent_root = (repo / prereg["execution_recovery"]["output_root"]).resolve()
    current_root = output_root(repo, materializer_config)
    verified_ledgers = 0
    verified_frames = 0
    missing: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for descriptor, rows in context["groups"]:
        valid_roots = []
        for label, root in (("parent_r2", parent_root), ("recovery_r3", current_root)):
            ledger_path, successor_path = r1.compact_paths(
                root, descriptor["source_id"], descriptor["sequence_id"]
            )
            valid = r1.validate_compact_ledger(
                ledger_path, successor_path, descriptor, rows
            )
            if (ledger_path.exists() or successor_path.exists()) and not valid:
                raise ContinuationError(
                    "invalid_or_partial_canonical_pair:"
                    f"{label}:{descriptor['source_id']}:{descriptor['sequence_id']}"
                )
            if valid:
                valid_roots.append(label)
        row = {
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(rows),
        }
        if valid_roots:
            if len(valid_roots) != 1:
                raise ContinuationError(
                    "ledger_has_multiple_authoritative_roots:"
                    f"{descriptor['source_id']}:{descriptor['sequence_id']}"
                )
            verified_ledgers += 1
            verified_frames += len(rows)
            verified.append({**row, "root": valid_roots[0]})
        else:
            missing.append(row)
    reset_count = len(context["resets"])
    if len(context["groups"]) != EXPECTED_LEDGERS:
        raise ContinuationError("continuation_group_count_drift")
    if sum(len(rows) for _, rows in context["groups"]) != EXPECTED_FRAMES:
        raise ContinuationError("continuation_frame_count_drift")
    if reset_count != EXPECTED_RESETS:
        raise ContinuationError("continuation_reset_count_drift")
    if any(not row["source_id"].startswith("crowdbot_") for row in missing):
        raise ContinuationError("continuation_non_crowdbot_input_gap")
    expected_files: dict[Path, set[Path]] = {
        parent_root: set(),
        current_root: set(),
    }
    for descriptor, rows in context["groups"]:
        for root in expected_files:
            ledger_path, successor_path = r1.compact_paths(
                root, descriptor["source_id"], descriptor["sequence_id"]
            )
            if r1.validate_compact_ledger(
                ledger_path, successor_path, descriptor, rows
            ):
                expected_files[root].update({ledger_path.resolve(), successor_path.resolve()})
    for root, expected in expected_files.items():
        ledger_root = root / "detector-ledgers"
        actual = (
            {path.resolve() for path in ledger_root.glob("*.json")}
            if ledger_root.exists()
            else set()
        )
        unexpected = actual - expected
        if unexpected:
            raise ContinuationError(
                "unexpected_detector_ledger_files:"
                + ",".join(sorted(str(path) for path in unexpected))
            )
    summary = {
        "expected_ledgers": EXPECTED_LEDGERS,
        "verified_ledgers": verified_ledgers,
        "expected_frames": EXPECTED_FRAMES,
        "verified_frames": verified_frames,
        "discontinuity_resets": reset_count,
        "missing_ledgers": len(missing),
        "missing_frames": EXPECTED_FRAMES - verified_frames,
        "complete": (
            verified_ledgers == EXPECTED_LEDGERS
            and verified_frames == EXPECTED_FRAMES
            and reset_count == EXPECTED_RESETS
            and not missing
        ),
        "next_missing": missing[0] if missing else None,
    }
    return summary, {
        "prereg": prereg,
        "context": context,
        "parent_root": parent_root,
        "current_root": current_root,
        "verified": verified,
        "missing": missing,
    }


def selected_crowdbot_input(
    repo: Path,
    materializer_config: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    Path,
    list[dict[str, Any]],
]:
    summary, details = coverage(repo, materializer_config)
    if summary["next_missing"] != expected:
        raise ContinuationError("selected_shard_is_no_longer_next_missing")
    context = details["context"]
    for descriptor, rows in context["groups"]:
        if (
            descriptor["source_id"] == expected["source_id"]
            and descriptor["sequence_id"] == expected["sequence_id"]
        ):
            bundle, image_rows = r1.load_crowdbot_images(
                repo, context["base_config"], descriptor, rows
            )
            return descriptor, rows, context, bundle, image_rows
    raise ContinuationError("selected_shard_descriptor_missing")


def attempt_root(
    root: Path, source_id: str, sequence_id: str
) -> Path:
    return root / "continuation-attempts" / r1.stable_slug(source_id, sequence_id)


def count_control_attempts(root: Path, source_id: str, sequence_id: str) -> int:
    directory = attempt_root(root, source_id, sequence_id)
    return len(list(directory.glob("attempt-*"))) if directory.exists() else 0


def create_control_attempt(
    root: Path,
    selected: dict[str, Any],
    maximum_attempts: int,
) -> tuple[int, Path]:
    directory = attempt_root(
        root, selected["source_id"], selected["sequence_id"]
    )
    existing = count_control_attempts(
        root, selected["source_id"], selected["sequence_id"]
    )
    if existing >= maximum_attempts:
        raise ContinuationError("continuation_ledger_retry_limit_exhausted")
    attempt_number = existing + 1
    path = directory / f"attempt-{attempt_number:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return attempt_number, path


@contextlib.contextmanager
def exclusive_shard_lock(root: Path) -> Iterator[None]:
    lock_path = root / "continuation-one-shard.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if lock_path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ContinuationError(
                    "another_continuation_shard_process_is_active"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ContinuationError(
                    "another_continuation_shard_process_is_active"
                ) from error
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
