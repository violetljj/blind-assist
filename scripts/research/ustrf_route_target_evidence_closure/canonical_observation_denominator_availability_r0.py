#!/usr/bin/env python3
"""Denominator-only availability upper bound for USTRF G0-B."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "blindassist_ustrf_canonical_observation_denominator_availability_r0"
INVENTORY_SCHEMA = "blindassist_ustrf_canonical_observation_authority_inventory_r0"
LEDGER_SCHEMA = "blindassist_ustrf_canonical_observation_authority_frame_ledger_r0"
AVAILABILITY_SCHEMA = "blindassist_ustrf_canonical_observation_availability_upper_bound_r0"
TERMINAL_SCHEMA = "blindassist_ustrf_canonical_observation_authority_terminal_r0"
AUDIT_SCHEMA = "blindassist_ustrf_canonical_observation_authority_audit_r0"
STAGE = "CANONICAL-OBSERVATION-AUTHORITY-AND-REPAIRABILITY-AUDIT-R0-B"
RESEARCH_IMPLEMENTATIONS = {
    "availability_core": "scripts/research/ustrf_route_target_evidence_closure/canonical_observation_denominator_availability_r0.py",
    "availability_runner": "scripts/research/ustrf_route_target_evidence_closure/run_canonical_observation_denominator_availability_r0.py",
    "independent_validator": "scripts/research/ustrf_route_target_evidence_closure/validate_canonical_observation_authority_repairability_r0.py",
    "contract_tests": "scripts/research/ustrf_route_target_evidence_closure/test_canonical_observation_authority_repairability_r0.py",
}
LEGAL_TERMINALS = (
    "FAIL_CLOSED_AUDIT_INCOMPLETE",
    "SOURCE_AUTHORITY_ABSENT",
    "AVAILABILITY_UPPER_BOUND_INSUFFICIENT",
    "AUTHORITY_PRESENT_AND_REPAIRABLE",
)
REQUIRED_SCALE_FIELDS = (
    "source_geometry",
    "canonical_transform",
    "bbox_coordinate_frame",
    "severe_truncation",
    "rgb_continuity",
    "capture_time",
    "frame_membership",
)
ELIGIBLE_STATES = {"authoritative", "verifiable_transform"}


class AvailabilityAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AvailabilityAuditError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    require(load_json(path) == value, f"atomic_verify_failed:{path}")


def _verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label}_binding_keys_drift")
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    return path


def verify_inventory_first(
    repo: Path, inventory_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """This must run before the B config or any denominator projection is opened."""
    require(inventory_path.is_file(), "inventory_missing")
    actual_sha = sha256_file(inventory_path)
    require(actual_sha == expected_sha256, "inventory_sha256_drift_before_denominator_open")
    inventory = load_json(inventory_path)
    require(inventory.get("schema") == INVENTORY_SCHEMA, "inventory_schema_drift")
    require(inventory.get("status") == "AUTHORITY_INVENTORY_FROZEN", "inventory_not_frozen")
    return inventory, {
        "role": "authority_inventory_first_read_and_verify",
        "path": inventory_path.resolve().relative_to(repo.resolve()).as_posix(),
        "bytes": inventory_path.stat().st_size,
        "sha256": actual_sha,
    }


def load_and_verify_config(
    repo: Path, config_path: Path, *, inventory_path: Path, inventory_sha256: str
) -> dict[str, Any]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(config.get("status") == "frozen_after_a_inventory", "config_not_frozen")
    require(tuple(config["terminal_states"]) == LEGAL_TERMINALS, "terminal_states_drift")
    binding = config["authority_inventory"]
    require(
        (repo / binding["path"]).resolve() == inventory_path.resolve()
        and binding["sha256"] == inventory_sha256,
        "inventory_binding_drift",
    )
    for label, parent in config["opaque_parent_hash_bindings"].items():
        _verify_binding(repo, parent, f"opaque_parent_{label}")
    for label, implementation in config["implementation_bindings"].items():
        _verify_binding(repo, implementation, f"implementation_{label}")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(RESEARCH_IMPLEMENTATIONS), "research_implementation_digest_keys_drift")
    for label, relative_path in RESEARCH_IMPLEMENTATIONS.items():
        require(
            sha256_file(repo / relative_path) == digests[label],
            f"research_implementation_{label}_sha256_drift",
        )
    safe = config["safe_denominator_projection"]
    require(
        safe
        == {
            "projection_role": "aggregate_identity_and_membership_only_no_outcome_payload",
            "scale_discovery_supported_unique_events": 11,
            "mechanically_mapped_supported_cells": 33,
            "mechanical_cells_per_unique_event": 3,
            "independent_event_units": 11,
            "negative_interval_count": 836,
            "negative_eligible_pair_count": 3801,
            "negative_duration_ns": 297376110945,
            "negative_duration_minutes": 4.95626851575,
            "sequence_count": 41,
            "frame_membership_count": 62229,
            "candidate_projection_count": 123,
            "candidate_independent_sequence_count": 41,
            "first_opportunity_unit_role": "per_track_reset",
            "first_opportunity_inventory_status": "not_separately_materialized_parent_limitation",
        },
        "safe_denominator_projection_drift",
    )
    return config


def _verify_inventory_ledgers(
    repo: Path, inventory: dict[str, Any]
) -> tuple[int, int, dict[str, dict[str, int]], bool]:
    frames = 0
    sequences = 0
    state_counts: dict[str, dict[str, int]] = {}
    complete = True
    seen_keys: set[tuple[str, str]] = set()
    for row in inventory["ledger_inventory"]:
        path = repo / row["path"]
        require(path.is_file(), "inventory_frame_ledger_missing")
        require(sha256_file(path) == row["sha256"], "inventory_frame_ledger_sha_drift")
        ledger = load_json(path)
        require(ledger.get("schema") == LEDGER_SCHEMA, "inventory_frame_ledger_schema_drift")
        key = (str(ledger["source_id"]), str(ledger["sequence_id"]))
        require(key not in seen_keys, "inventory_sequence_duplicate")
        seen_keys.add(key)
        require(len(ledger["frames"]) == row["frame_count"], "inventory_ledger_frame_count_drift")
        for frame in ledger["frames"]:
            fields = frame.get("fields")
            if not isinstance(fields, dict) or set(fields) != set(
                (
                    "source_geometry",
                    "canonical_transform",
                    "bbox_coordinate_frame",
                    "severe_truncation",
                    "rgb_continuity",
                    "capture_time",
                    "frame_membership",
                    "background_feature_input",
                )
            ):
                complete = False
                continue
            for field_name, field in fields.items():
                counts = state_counts.setdefault(field_name, {})
                state = str(field.get("state"))
                counts[state] = counts.get(state, 0) + 1
        frames += len(ledger["frames"])
        sequences += 1
    return sequences, frames, state_counts, complete


def _terminal(
    *,
    audit_complete: bool,
    required_source_authority_absent: bool,
    availability_complete: bool,
) -> str:
    if not audit_complete:
        return LEGAL_TERMINALS[0]
    if required_source_authority_absent:
        return LEGAL_TERMINALS[1]
    if not availability_complete:
        return LEGAL_TERMINALS[2]
    return LEGAL_TERMINALS[3]


def run_availability(
    repo: Path,
    config_path: Path,
    inventory_path: Path,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    inventory, first_read = verify_inventory_first(
        repo, inventory_path, expected_inventory_sha256
    )
    config = load_and_verify_config(
        repo,
        config_path,
        inventory_path=inventory_path,
        inventory_sha256=expected_inventory_sha256,
    )
    sequences, frames, state_counts, ledger_complete = _verify_inventory_ledgers(
        repo, inventory
    )
    safe = config["safe_denominator_projection"]
    decoded = inventory["decoded_counters"]
    counters_clean = all(value == 0 for value in decoded.values())
    scope_complete = (
        sequences == safe["sequence_count"]
        and frames == safe["frame_membership_count"]
        and inventory["evidence_universe"]["sequence_count"] == sequences
        and inventory["evidence_universe"]["frame_count"] == frames
    )
    severe = state_counts.get("severe_truncation", {})
    source_authority_absent = severe.get("absent", 0) == frames
    all_required_frame_states_eligible = all(
        sum(
            count
            for state, count in state_counts.get(field, {}).items()
            if state in ELIGIBLE_STATES
        )
        == frames
        for field in REQUIRED_SCALE_FIELDS
    )
    # If severe-truncation authority is absent for every frozen frame/object
    # transport, no legal scale observation window exists. This dominance proof
    # needs no event/window payload and therefore preserves B isolation.
    if source_authority_absent:
        available_events = 0
        available_cells = 0
        available_negative_ns = 0
        available_negative_intervals = 0
        available_negative_pairs = 0
    else:
        # Current R0 is intentionally not a general signal-window runner.
        # Non-zero support would require a versioned, identifier-only safe pack.
        available_events = safe["scale_discovery_supported_unique_events"] if all_required_frame_states_eligible else 0
        available_cells = safe["mechanically_mapped_supported_cells"] if all_required_frame_states_eligible else 0
        available_negative_ns = safe["negative_duration_ns"] if all_required_frame_states_eligible else 0
        available_negative_intervals = safe["negative_interval_count"] if all_required_frame_states_eligible else 0
        available_negative_pairs = safe["negative_eligible_pair_count"] if all_required_frame_states_eligible else 0
    availability_complete = (
        available_events == safe["scale_discovery_supported_unique_events"]
        and available_cells == safe["mechanically_mapped_supported_cells"]
        and available_negative_ns == safe["negative_duration_ns"]
        and available_negative_intervals == safe["negative_interval_count"]
        and available_negative_pairs == safe["negative_eligible_pair_count"]
        and scope_complete
    )
    audit_complete = ledger_complete and counters_clean and scope_complete
    terminal = _terminal(
        audit_complete=audit_complete,
        required_source_authority_absent=source_authority_absent,
        availability_complete=availability_complete,
    )
    output_root = repo / config["outputs"]["root"]
    availability = {
        "schema": AVAILABILITY_SCHEMA,
        "stage": STAGE,
        "status": "DENOMINATOR_ONLY_AVAILABILITY_FROZEN",
        "process_id": os.getpid(),
        "inventory_sha256": expected_inventory_sha256,
        "config_sha256": sha256_file(config_path),
        "independence_note": "33 cells are mechanical mappings of 11 independent events",
        "safe_denominator_projection": safe,
        "availability_upper_bound": {
            "scale_discovery_supported_unique_events": {
                "available": available_events,
                "required": safe["scale_discovery_supported_unique_events"],
            },
            "mechanically_mapped_supported_cells": {
                "available": available_cells,
                "required": safe["mechanically_mapped_supported_cells"],
            },
            "negative_interval_count": {
                "available": available_negative_intervals,
                "required": safe["negative_interval_count"],
            },
            "negative_eligible_pair_count": {
                "available": available_negative_pairs,
                "required": safe["negative_eligible_pair_count"],
            },
            "negative_duration_ns": {
                "available": available_negative_ns,
                "required": safe["negative_duration_ns"],
            },
            "sequence_count": {"available": sequences, "required": safe["sequence_count"]},
            "frame_membership_count": {
                "available": frames,
                "required": safe["frame_membership_count"],
            },
        },
        "dominance_proof": {
            "applied": source_authority_absent,
            "reason": "all frozen frame/object transports lack authoritative severe-truncation state; unknown must abstain",
            "signal_windows_decoded": 0,
            "event_identifiers_decoded": 0,
            "cell_identifiers_decoded": 0,
            "negative_interval_identifiers_decoded": 0,
        },
        "audit_complete": audit_complete,
        "availability_complete": availability_complete,
        "field_state_counts": state_counts,
        "read_manifest": [
            first_read,
            {
                "role": "b_config_after_inventory_verification",
                "path": config_path.resolve().relative_to(repo.resolve()).as_posix(),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
        ],
        "decoded_counters": {
            "truth": 0,
            "oracle": 0,
            "signal": 0,
            "candidate": 0,
            "outcome": 0,
        },
    }
    availability_path = output_root / config["outputs"]["availability"]
    atomic_write_json(availability_path, availability)
    gap_matrix = {
        "required_field_families": {
            field: state_counts.get(field, {}) for field in REQUIRED_SCALE_FIELDS
        },
        "source_authority_absent": [
            "severe_truncation"
        ]
        if source_authority_absent
        else [],
        "repairable_but_unbound": [
            "source_geometry",
            "capture_time",
            "rgb_continuity",
            "frame_membership",
        ],
        "unknown_transform": ["canonical_transform"],
        "planning_only": {
            "background_feature_input": state_counts.get("background_feature_input", {}),
            "does_not_change_scale_terminal": True,
        },
    }
    terminal_receipt = {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": terminal,
        "terminal_priority_order": list(LEGAL_TERMINALS),
        "inventory_sha256": expected_inventory_sha256,
        "availability_sha256": sha256_file(availability_path),
        "gap_matrix": gap_matrix,
        "uncertainty_reduced": "separates repairable geometry/time/RGB bindings from absent severe-truncation source authority",
        "evidence_delta": "proved a frozen current-input authority and availability upper bound",
        "claim_boundary": {
            "g1_repair_authorized": terminal == "AUTHORITY_PRESENT_AND_REPAIRABLE",
            "signal_authorized": False,
            "android_authorized": False,
            "human_authorized": False,
            "production_authorized": False,
        },
    }
    terminal_path = output_root / config["outputs"]["terminal"]
    atomic_write_json(terminal_path, terminal_receipt)
    audit = {
        "schema": AUDIT_SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_RECOMPUTED",
        "process_id": os.getpid(),
        "inventory_process_id": inventory["process_id"],
        "process_isolation": os.getpid() != inventory["process_id"],
        "inventory_sha256": expected_inventory_sha256,
        "availability_sha256": sha256_file(availability_path),
        "terminal_sha256": sha256_file(terminal_path),
        "terminal_state": terminal,
    }
    require(audit["process_isolation"], "a_b_process_isolation_failed")
    audit_path = output_root / config["outputs"]["audit"]
    atomic_write_json(audit_path, audit)
    return {
        "status": terminal,
        "process_id": os.getpid(),
        "inventory_process_id": inventory["process_id"],
        "availability_path": availability_path.resolve().relative_to(repo.resolve()).as_posix(),
        "terminal_path": terminal_path.resolve().relative_to(repo.resolve()).as_posix(),
        "audit_path": audit_path.resolve().relative_to(repo.resolve()).as_posix(),
    }
