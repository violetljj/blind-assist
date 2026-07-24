from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from known_route_eligible_delivery_failure_attribution import (
    CANDIDATES,
    _clearance_events,
    _event_tracks,
    _trace_maps,
    build_blind_trace,
    canonical_bytes,
    load_and_verify_config as load_parent_config,
    load_and_verify_event_scope_blind_pack,
    load_json,
    require,
    sha256_file,
)
from metric_profiles_r2_l1 import load_truth_frame_index


CONFIG_SCHEMA = (
    "blindassist_ustrf_eligible_target_attribution_ordered_isolated_opening_r1"
)
TOKEN_LEDGER_SCHEMA = (
    "blindassist_ustrf_eligible_target_attribution_token_ledger_r1"
)
TOKEN_INVENTORY_SCHEMA = (
    "blindassist_ustrf_eligible_target_attribution_token_inventory_r1"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_eligible_target_attribution_ordered_isolated_opening_"
    "terminal_r1"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_eligible_target_attribution_ordered_isolated_opening_"
    "validation_r1"
)
TERMINAL_STATE = "ORACLE_MECHANISM_REPAIR_DIAGNOSTIC_COMPLETE"

OUTCOMES = (
    "parent_formed_delivery_cell_remains_token_qualified",
    "recovered_episode_opened_before_alertable_window",
    "recovered_episode_opened_inside_window_before_target_attribution",
    "recovered_pre_invalid_baseline_latch_guard_quarantine",
    "remaining_never_active_relation",
)

TOKEN_LEDGER_KEYS = {
    "schema",
    "stage",
    "authority",
    "candidate_id",
    "event_id",
    "source_id",
    "sequence_id",
    "frame_count",
    "frames",
}
TOKEN_FRAME_KEYS = {
    "frame_id",
    "source_capture_timestamp_ns",
    "reset_segment",
    "state_reset_before_frame",
    "route_known",
    "eligible_attributed_track_ids",
    "background_namespace_active",
    "background_namespace_opening_count",
}

IMPLEMENTATION_FILES = {
    "mechanism_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "eligible_target_attribution_ordered_isolated_opening.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_eligible_target_attribution_ordered_isolated_opening.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_eligible_target_attribution_ordered_isolated_opening.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_eligible_target_attribution_ordered_isolated_opening.py"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _verify_binding(repo: Path, row: dict[str, Any], label: str) -> Path:
    path = repo / str(row["path"])
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == row["sha256"], f"{label}_sha_drift")
    return path


def load_and_verify_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    config = load_json(config_path)
    require(
        set(config)
        == {
            "schema",
            "stage",
            "status",
            "frozen_on",
            "purpose",
            "authority",
            "non_goals",
            "candidate_roster",
            "phase_order",
            "mechanism",
            "parent_bindings",
            "implementation_bindings",
            "evaluation",
            "expected_scope",
            "outputs",
        },
        "config_top_level_keys_drift",
    )
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(
        config.get("stage")
        == "ELIGIBLE-TARGET-ATTRIBUTION-ORDERED-ISOLATED-OPENING-R1",
        "config_stage_drift",
    )
    require(
        config.get("status")
        == "preregistered_truth_assisted_oracle_mechanism_repair_diagnostic",
        "config_status_drift",
    )
    require(config.get("frozen_on") == "2026-07-24", "frozen_on_drift")
    require(
        config.get("purpose")
        == (
            "Test one isolated mechanism variable on the immutable 36 "
            "candidate-event cells: an event-scoped one-shot opener may start "
            "only after a frozen eligible-target-attribution token reaches the "
            "unchanged two-frame qualification gate. Baseline pre-window, "
            "pre-attribution, and pre-invalid episode state remains isolated "
            "and cannot consume the target delivery edge."
        ),
        "purpose_drift",
    )
    require(
        config.get("candidate_roster") == list(CANDIDATES),
        "candidate_roster_drift",
    )
    require(
        config.get("non_goals")
        == [
            (
                "modify or rerun the frozen detector, T0 association, route, "
                "C1-C3 candidate, route-invalid/reset guard, truth, or clearance"
            ),
            (
                "change min_alert_frames, min_clear_frames, IoU, route "
                "thresholds, metric thresholds, denominators, or event windows"
            ),
            (
                "use event identity, truth boxes, alertable boundaries, or "
                "oracle tokens as a deployable runtime input"
            ),
            (
                "diagnose or repair post-delivery relation clearance, repeats, "
                "false-alert rate, evidence age, or unknown-route lifecycle"
            ),
            (
                "score, rank, select, or authorize L2, L3, Android shadow, H2, "
                "human use, independent walking, or production"
            ),
        ],
        "non_goals_drift",
    )
    expected_authority = {
        "maximum": "TRUTH_ASSISTED_ORACLE_MECHANISM_DIAGNOSTIC_ONLY",
        "candidate_rerun": False,
        "detector_rerun": False,
        "route_rerun": False,
        "truth_used_only_to_build_frozen_attribution_tokens": True,
        "raw_truth_payload_available_to_opener": False,
        "truth_derived_event_scoped_tokens_available_to_opener": True,
        "truth_window_boundaries_encoded_by_ledger_extent": True,
        "causal_candidate_blind_token_producer_proven": False,
        "candidate_comparison": False,
        "winner_or_ranking": False,
        "selection": False,
        "threshold_or_denominator_change": False,
        "baseline_fsm_change": False,
        "route_invalid_guard_change": False,
        "clearance_change": False,
        "l2_or_l3": False,
        "android_shadow": False,
        "h2": False,
        "human_outcome": False,
        "independent_walking_safety": False,
        "production": False,
        "new_data": False,
        "training": False,
    }
    require(config.get("authority") == expected_authority, "authority_drift")
    expected_phase_order = [
        "verify_parent_attribution_config_terminal_validation_and_event_scope_inventory",
        (
            "verify_event_scope_pack_and_reconstruct_full_reset_scope_"
            "candidate_blind_facts"
        ),
        "join_frozen_truth_only_to_build_eligible_attribution_token_ledgers",
        "persist_and_reverify_all_36_token_ledgers_before_opener_execution",
        (
            "run_raw_truth_payload_blind_event_scoped_one_shot_opener_"
            "from_frozen_truth_derived_tokens"
        ),
        "compare_mechanism_outcomes_to_parent_mutually_exclusive_attribution_labels",
        "fail_closed_on_any_ordering_isolation_duplicate_or_accounting_gap",
    ]
    require(
        config.get("phase_order") == expected_phase_order,
        "phase_order_drift",
    )
    expected_mechanism = {
        "token_authority": (
            "truth_assisted_oracle_upper_bound_not_runtime_authority"
        ),
        "token_scope": "candidate_source_sequence_reset_event",
        "ledger_extent": "reset_scope_start_through_truth_terminal_clear",
        "token_condition": (
            "route_known_and_target_iou_attributed_active_relation"
        ),
        "opening_condition": (
            "unchanged_min_alert_consecutive_eligible_attribution_tokens"
        ),
        "c1_continuity": "same_attributed_track_id",
        "c2_c3_continuity": "same_frozen_event_identity",
        "pre_token_baseline_state": (
            "explicit_background_non_delivering_namespace"
        ),
        "pre_window_baseline_state": (
            "explicit_background_non_delivering_namespace"
        ),
        "pre_invalid_baseline_and_guard_state": (
            "explicit_background_non_delivering_namespace"
        ),
        "isolation_test": (
            "target_delivery_output_invariant_to_background_namespace_mutation"
        ),
        "delivery_cardinality": "at_most_one_per_candidate_event_cell",
        "delivery_frame": "qualification_frame_not_support_start_frame",
        "clearance_semantics": "out_of_scope_unchanged",
    }
    require(config.get("mechanism") == expected_mechanism, "mechanism_drift")
    evaluation = config.get("evaluation", {})
    require(
        evaluation
        == {
            "truth_attribution_minimum_iou": 0.3,
            "min_alert_frames": 2,
            "clearance_events_per_candidate": 12,
            "candidate_event_cells": 36,
            "parent_formed_delivery_cells": 6,
            "expected_token_qualified_cells": 33,
            "expected_recovered_cells": 27,
            "expected_remaining_no_support_cells": 3,
            "opening_before_qualification_max": 0,
            "background_namespace_invariance_failures_max": 0,
            "one_shot_cardinality_violations_max": 0,
            "duplicate_delivery_keys_max": 0,
            "accounting_gap_max": 0,
            "expected_outcome_counts": {
                "parent_formed_delivery_cell_remains_token_qualified": 6,
                "recovered_episode_opened_before_alertable_window": 9,
                (
                    "recovered_episode_opened_inside_window_before_"
                    "target_attribution"
                ): 13,
                (
                    "recovered_pre_invalid_baseline_latch_guard_quarantine"
                ): 5,
                "remaining_never_active_relation": 3,
            },
        },
        "evaluation_drift",
    )
    require(
        config.get("expected_scope")
        == {
            "candidate_count": 3,
            "clearance_events_per_candidate": 12,
            "candidate_event_cells": 36,
            "token_ledger_count": 36,
            "token_frame_count": 5043,
            "candidate_reruns": 0,
            "detector_reruns": 0,
            "route_reruns": 0,
            "threshold_changes": 0,
            "new_data": 0,
        },
        "expected_scope_drift",
    )
    require(
        config.get("outputs")
        == {
            "root": (
                "artifacts.local/evidence/"
                "ustrf-eligible-target-attribution-ordered-isolated-opening-r1"
            ),
            "token_inventory": "eligible-attribution-token-inventory-r1.json",
            "terminal_receipt": "terminal-receipt-r1.json",
            "validation_receipt": "validation-receipt-r1.json",
        },
        "outputs_drift",
    )
    require(
        set(config.get("parent_bindings", {}))
        == {
            "failure_attribution_config",
            "failure_attribution_terminal",
            "failure_attribution_validation",
            "event_scope_blind_inventory",
        },
        "parent_binding_keys_drift",
    )
    require(
        set(config.get("implementation_bindings", {}))
        == set(IMPLEMENTATION_FILES),
        "implementation_binding_keys_drift",
    )
    bindings: dict[str, str] = {}
    for label, row in config["parent_bindings"].items():
        path = _verify_binding(repo, row, f"parent_{label}")
        bindings[label] = sha256_file(path)
    for label, relative in IMPLEMENTATION_FILES.items():
        expected = config["implementation_bindings"].get(label)
        require(
            isinstance(expected, str)
            and len(expected) == 64
            and expected != "__TO_FILL__",
            f"{label}_not_frozen",
        )
        path = repo / relative
        require(path.is_file(), f"{label}_missing")
        require(sha256_file(path) == expected, f"{label}_sha_drift")
        bindings[label] = expected
    return config, bindings


def _parent_blind_context(
    repo: Path,
    config: dict[str, Any],
    *,
    include_full_traces: bool = True,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    parent_config_path = (
        repo / config["parent_bindings"]["failure_attribution_config"]["path"]
    )
    parent_config, parent_bindings = load_parent_config(
        repo, parent_config_path
    )
    terminal = load_json(
        repo
        / config["parent_bindings"]["failure_attribution_terminal"]["path"]
    )
    validation = load_json(
        repo
        / config["parent_bindings"]["failure_attribution_validation"]["path"]
    )
    require(
        terminal.get("terminal_state")
        == "FAILURE_ATTRIBUTION_COMPLETE",
        "parent_terminal_state_drift",
    )
    require(
        terminal.get("attribution_gate", {}).get("passed") is True,
        "parent_attribution_gate_not_passed",
    )
    require(
        validation.get("status") == "VALID",
        "parent_validation_not_valid",
    )
    _, event_scope_traces = load_and_verify_event_scope_blind_pack(
        repo, parent_config, parent_bindings
    )
    require(
        sha256_file(
            repo
            / config["parent_bindings"]["event_scope_blind_inventory"][
                "path"
            ]
        )
        == config["parent_bindings"]["event_scope_blind_inventory"]["sha256"],
        "event_scope_inventory_drift",
    )
    events = _clearance_events(repo, parent_config)
    if not include_full_traces:
        return terminal, events, {}, parent_config
    baseline, guarded, _ = _trace_maps(repo, parent_config)
    require(set(baseline) == set(guarded), "parent_full_trace_key_drift")
    full_traces = {
        key: build_blind_trace(key[0], baseline[key], guarded[key])
        for key in sorted(baseline)
    }
    for key, packed in event_scope_traces.items():
        require(key in full_traces, "event_scope_not_in_full_trace")
        by_frame = {
            int(frame["frame_id"]): frame
            for frame in full_traces[key]["frames"]
        }
        require(
            packed["frames"]
            == [by_frame[int(frame["frame_id"])] for frame in packed["frames"]],
            "event_scope_full_trace_recompute_drift",
        )
    return terminal, events, full_traces, parent_config


def _load_truth_index(
    repo: Path, parent_config: dict[str, Any]
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    metric_profile_config = load_json(
        repo / parent_config["parent_bindings"]["metric_profile_config"]["path"]
    )
    return load_truth_frame_index(metric_profile_config, repo)


def _event_reset_scope_frames(
    event: dict[str, Any], trace: dict[str, Any]
) -> list[dict[str, Any]]:
    start = int(event["anchors"]["alertable_start_frame"])
    clear = int(event["anchors"]["truth_terminal_clear_frame"])
    through_clear = [
        frame for frame in trace["frames"] if int(frame["frame_id"]) <= clear
    ]
    require(
        through_clear and int(through_clear[-1]["frame_id"]) == clear,
        "truth_clear_frame_missing",
    )
    reset_indexes = [
        index
        for index, frame in enumerate(through_clear)
        if bool(frame["state_reset_before_frame"])
        and int(frame["frame_id"]) <= start
    ]
    require(reset_indexes, "event_reset_scope_start_missing")
    frames = through_clear[reset_indexes[-1] :]
    require(
        frames
        and bool(frames[0]["state_reset_before_frame"])
        and int(frames[0]["frame_id"]) <= start,
        "event_reset_scope_start_drift",
    )
    require(
        [int(frame["frame_id"]) for frame in frames]
        == list(
            range(
                int(frames[0]["frame_id"]),
                clear + 1,
            )
        ),
        "event_reset_scope_frame_gap",
    )
    return frames


def build_token_ledger(
    *,
    candidate_id: str,
    event: dict[str, Any],
    trace: dict[str, Any],
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
) -> dict[str, Any]:
    alertable_start = int(event["anchors"]["alertable_start_frame"])
    truth_clear = int(event["anchors"]["truth_terminal_clear_frame"])
    frames = _event_reset_scope_frames(event, trace)
    token_frames = []
    for frame in frames:
        attributed_active = (
            _event_tracks(
                frame,
                event["event_id"],
                frame["active_relation_track_ids"],
                truth_index,
                minimum_iou,
            )
            if (
                frame["route_known"]
                and alertable_start <= int(frame["frame_id"]) <= truth_clear
            )
            else []
        )
        token_frames.append(
            {
                "frame_id": int(frame["frame_id"]),
                "source_capture_timestamp_ns": int(
                    frame["source_capture_timestamp_ns"]
                ),
                "reset_segment": int(frame["reset_segment"]),
                "state_reset_before_frame": bool(
                    frame["state_reset_before_frame"]
                ),
                "route_known": bool(frame["route_known"]),
                "eligible_attributed_track_ids": sorted(
                    int(value) for value in attributed_active
                ),
                "background_namespace_active": bool(
                    frame["baseline_candidate_active_after"]
                ),
                "background_namespace_opening_count": len(
                    frame["baseline_deliveries"]
                ),
            }
        )
    return {
        "schema": TOKEN_LEDGER_SCHEMA,
        "stage": (
            "ELIGIBLE-TARGET-ATTRIBUTION-ORDERED-ISOLATED-OPENING-R1"
        ),
        "authority": (
            "truth_assisted_oracle_attribution_token_not_runtime_authority"
        ),
        "candidate_id": candidate_id,
        "event_id": event["event_id"],
        "source_id": event["source_id"],
        "sequence_id": event["sequence_id"],
        "frame_count": len(token_frames),
        "frames": token_frames,
    }


def _ledger_relative_path(
    config: dict[str, Any],
    candidate_id: str,
    event_id: str,
) -> Path:
    digest = sha256_bytes(f"{candidate_id}::{event_id}".encode())[:16]
    return (
        Path(config["outputs"]["root"])
        / "eligible-attribution-token-ledgers"
        / candidate_id
        / f"{digest}.json"
    )


def build_and_write_token_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    _, events, traces, parent_config = _parent_blind_context(repo, config)
    truth_index = _load_truth_index(repo, parent_config)
    minimum_iou = float(
        config["evaluation"]["truth_attribution_minimum_iou"]
    )
    inventory = []
    total_frames = 0
    for candidate_id in CANDIDATES:
        for event in events:
            key = (
                candidate_id,
                event["source_id"],
                event["sequence_id"],
            )
            require(key in traces, "event_scope_trace_missing")
            ledger = build_token_ledger(
                candidate_id=candidate_id,
                event=event,
                trace=traces[key],
                truth_index=truth_index,
                minimum_iou=minimum_iou,
            )
            relative = _ledger_relative_path(
                config, candidate_id, event["event_id"]
            )
            path = repo / relative
            atomic_write_json(path, ledger)
            total_frames += ledger["frame_count"]
            inventory.append(
                {
                    "candidate_id": candidate_id,
                    "event_id": event["event_id"],
                    "source_id": event["source_id"],
                    "sequence_id": event["sequence_id"],
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "frame_count": ledger["frame_count"],
                }
            )
    receipt = {
        "schema": TOKEN_INVENTORY_SCHEMA,
        "stage": config["stage"],
        "status": "ELIGIBLE_ATTRIBUTION_TOKEN_INVENTORY_FROZEN",
        "authority": (
            "truth_assisted_oracle_tokens_only_no_runtime_authority"
        ),
        "bindings": bindings,
        "ledger_count": len(inventory),
        "frame_count": total_frames,
        "raw_truth_payload_available_to_opener": False,
        "truth_derived_event_scoped_tokens_available_to_opener": True,
        "truth_window_boundaries_encoded_by_ledger_extent": True,
        "inventory": inventory,
    }
    require(
        len(inventory) == config["expected_scope"]["token_ledger_count"],
        "token_ledger_count_drift",
    )
    require(
        total_frames == config["expected_scope"]["token_frame_count"],
        "token_frame_count_drift",
    )
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["token_inventory"]
    )
    atomic_write_json(path, receipt)
    return receipt


def assert_opener_input(ledger: dict[str, Any]) -> None:
    require(
        isinstance(ledger, dict) and set(ledger) == TOKEN_LEDGER_KEYS,
        "opener_ledger_keys_drift",
    )
    require(
        ledger.get("schema") == TOKEN_LEDGER_SCHEMA,
        "opener_ledger_schema_drift",
    )
    require(
        ledger.get("stage")
        == "ELIGIBLE-TARGET-ATTRIBUTION-ORDERED-ISOLATED-OPENING-R1",
        "opener_ledger_stage_drift",
    )
    require(
        ledger.get("authority")
        == "truth_assisted_oracle_attribution_token_not_runtime_authority",
        "opener_ledger_authority_drift",
    )
    require(
        ledger.get("candidate_id") in CANDIDATES,
        "opener_ledger_candidate_drift",
    )
    for field in ("event_id", "source_id", "sequence_id"):
        require(
            isinstance(ledger.get(field), str) and bool(ledger[field]),
            f"opener_ledger_{field}_drift",
        )
    frames = ledger.get("frames")
    require(
        isinstance(frames, list)
        and bool(frames)
        and ledger.get("frame_count") == len(frames),
        "opener_ledger_frame_count_drift",
    )
    previous_frame_id: int | None = None
    previous_timestamp: int | None = None
    previous_segment: int | None = None
    for index, frame in enumerate(frames):
        require(
            isinstance(frame, dict) and set(frame) == TOKEN_FRAME_KEYS,
            f"opener_frame_keys_drift:{index}",
        )
        frame_id = frame.get("frame_id")
        timestamp = frame.get("source_capture_timestamp_ns")
        segment = frame.get("reset_segment")
        require(
            isinstance(frame_id, int)
            and isinstance(timestamp, int)
            and isinstance(segment, int),
            f"opener_frame_identity_type_drift:{index}",
        )
        require(
            isinstance(frame.get("state_reset_before_frame"), bool)
            and isinstance(frame.get("route_known"), bool)
            and isinstance(frame.get("background_namespace_active"), bool),
            f"opener_frame_boolean_type_drift:{index}",
        )
        opening_count = frame.get("background_namespace_opening_count")
        require(
            isinstance(opening_count, int) and opening_count >= 0,
            f"opener_background_opening_count_drift:{index}",
        )
        attributed = frame.get("eligible_attributed_track_ids")
        require(
            isinstance(attributed, list)
            and attributed == sorted(set(attributed))
            and all(isinstance(value, int) for value in attributed),
            f"opener_attributed_track_ids_drift:{index}",
        )
        require(
            bool(frame["route_known"]) or not attributed,
            "eligible_token_on_unknown_route",
        )
        if index == 0:
            require(
                frame["state_reset_before_frame"] is True,
                "opener_first_frame_not_reset_scope_start",
            )
        else:
            require(
                frame_id == previous_frame_id + 1,
                f"opener_frame_id_gap:{index}",
            )
            require(
                timestamp > previous_timestamp,
                f"opener_timestamp_not_monotonic:{index}",
            )
            if frame["state_reset_before_frame"]:
                require(
                    segment == previous_segment + 1,
                    f"opener_reset_segment_transition_drift:{index}",
                )
            else:
                require(
                    segment == previous_segment,
                    f"opener_segment_changed_without_reset:{index}",
                )
        previous_frame_id = frame_id
        previous_timestamp = timestamp
        previous_segment = segment


def load_and_verify_token_inventory(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], dict[str, Any]],
]:
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["token_inventory"]
    )
    require(path.is_file(), "token_inventory_missing")
    receipt = load_json(path)
    require(
        set(receipt)
        == {
            "schema",
            "stage",
            "status",
            "authority",
            "bindings",
            "ledger_count",
            "frame_count",
            "raw_truth_payload_available_to_opener",
            "truth_derived_event_scoped_tokens_available_to_opener",
            "truth_window_boundaries_encoded_by_ledger_extent",
            "inventory",
        },
        "token_inventory_keys_drift",
    )
    require(
        receipt.get("schema") == TOKEN_INVENTORY_SCHEMA,
        "token_inventory_schema_drift",
    )
    require(receipt.get("bindings") == bindings, "token_binding_drift")
    require(
        receipt.get("raw_truth_payload_available_to_opener") is False
        and receipt.get(
            "truth_derived_event_scoped_tokens_available_to_opener"
        )
        is True
        and receipt.get("truth_window_boundaries_encoded_by_ledger_extent")
        is True,
        "opener_truth_information_contract_drift",
    )
    require(
        receipt.get("ledger_count")
        == config["expected_scope"]["token_ledger_count"],
        "token_ledger_count_drift",
    )
    ledgers = {}
    total_frames = 0
    for row in receipt["inventory"]:
        ledger = _load_verified_token_ledger(repo, row)
        require(
            ledger["frame_count"] == row["frame_count"],
            "token_ledger_frame_count_drift",
        )
        require(
            {
                "candidate_id": ledger["candidate_id"],
                "event_id": ledger["event_id"],
                "source_id": ledger["source_id"],
                "sequence_id": ledger["sequence_id"],
            }
            == {
                "candidate_id": row["candidate_id"],
                "event_id": row["event_id"],
                "source_id": row["source_id"],
                "sequence_id": row["sequence_id"],
            },
            "token_ledger_inventory_identity_drift",
        )
        key = (row["candidate_id"], row["event_id"])
        require(key not in ledgers, "duplicate_token_ledger")
        ledgers[key] = ledger
        total_frames += int(row["frame_count"])
    require(
        total_frames == receipt["frame_count"],
        "token_inventory_frame_count_drift",
    )
    return receipt, ledgers


def _load_verified_token_ledger(
    repo: Path, row: dict[str, Any]
) -> dict[str, Any]:
    require(
        set(row)
        == {
            "candidate_id",
            "event_id",
            "source_id",
            "sequence_id",
            "path",
            "sha256",
            "frame_count",
        },
        "token_inventory_row_keys_drift",
    )
    ledger_path = repo / row["path"]
    require(ledger_path.is_file(), "token_ledger_missing")
    require(
        sha256_file(ledger_path) == row["sha256"],
        "token_ledger_sha_drift",
    )
    ledger = load_json(ledger_path)
    assert_opener_input(ledger)
    return ledger


def apply_ordered_isolated_opener(
    ledger: dict[str, Any], min_alert: int
) -> dict[str, Any]:
    assert_opener_input(ledger)
    candidate_id = str(ledger["candidate_id"])
    require(candidate_id in CANDIDATES, "opener_candidate_drift")
    require(min_alert == 2, "opener_min_alert_drift")
    deliveries = []
    support_start: int | None = None
    qualification_frame: int | None = None
    qualification_segment: int | None = None
    if candidate_id == CANDIDATES[0]:
        runs: dict[int, int] = defaultdict(int)
        starts: dict[int, int] = {}
        for frame in ledger["frames"]:
            if frame["state_reset_before_frame"]:
                runs.clear()
                starts.clear()
            matched = set(int(v) for v in frame["eligible_attributed_track_ids"])
            for track_id in list(runs):
                if track_id not in matched:
                    runs[track_id] = 0
                    starts.pop(track_id, None)
            for track_id in sorted(matched):
                if runs[track_id] == 0:
                    starts[track_id] = int(frame["frame_id"])
                runs[track_id] += 1
                if runs[track_id] >= min_alert and not deliveries:
                    support_start = starts[track_id]
                    qualification_frame = int(frame["frame_id"])
                    qualification_segment = int(frame["reset_segment"])
                    deliveries.append(
                        {
                            "delivery_key": (
                                f"{candidate_id}::{ledger['source_id']}::"
                                f"{ledger['sequence_id']}::"
                                f"reset-{qualification_segment}::"
                                f"{ledger['event_id']}"
                            ),
                            "frame_id": qualification_frame,
                            "attributed_track_ids": [track_id],
                        }
                    )
    else:
        run = 0
        current_start: int | None = None
        for frame in ledger["frames"]:
            if frame["state_reset_before_frame"]:
                run = 0
                current_start = None
            matched = [
                int(v) for v in frame["eligible_attributed_track_ids"]
            ]
            if matched:
                if run == 0:
                    current_start = int(frame["frame_id"])
                run += 1
                if run >= min_alert and not deliveries:
                    support_start = current_start
                    qualification_frame = int(frame["frame_id"])
                    qualification_segment = int(frame["reset_segment"])
                    deliveries.append(
                        {
                            "delivery_key": (
                                f"{candidate_id}::{ledger['source_id']}::"
                                f"{ledger['sequence_id']}::"
                                f"reset-{qualification_segment}::"
                                f"{ledger['event_id']}"
                            ),
                            "frame_id": qualification_frame,
                            "attributed_track_ids": sorted(matched),
                        }
                    )
            else:
                run = 0
                current_start = None
    require(len(deliveries) <= 1, "duplicate_one_shot_delivery")
    if deliveries:
        require(
            deliveries[0]["frame_id"] == qualification_frame,
            "delivery_not_on_qualification_frame",
        )
        require(
            qualification_frame is not None
            and support_start is not None
            and qualification_frame >= support_start,
            "opening_before_eligible_support",
        )
    background_cutoff = (
        qualification_frame
        if qualification_frame is not None
        else int(ledger["frames"][-1]["frame_id"])
    )
    prequalification_frames = [
        frame
        for frame in ledger["frames"]
        if int(frame["frame_id"]) <= background_cutoff
    ]
    return {
        "candidate_id": candidate_id,
        "event_id": ledger["event_id"],
        "support_start_frame": support_start,
        "qualification_frame": qualification_frame,
        "deliveries": deliveries,
        "background_namespace_active_frame_count": sum(
            int(frame["background_namespace_active"])
            for frame in ledger["frames"]
        ),
        "background_namespace_opening_count": sum(
            int(frame["background_namespace_opening_count"])
            for frame in ledger["frames"]
        ),
        "background_namespace_prequalification_active_frame_count": sum(
            int(frame["background_namespace_active"])
            for frame in prequalification_frames
        ),
        "background_namespace_prequalification_opening_count": sum(
            int(frame["background_namespace_opening_count"])
            for frame in prequalification_frames
        ),
    }


def mutate_background_namespace(
    ledger: dict[str, Any],
) -> dict[str, Any]:
    assert_opener_input(ledger)
    return {
        **ledger,
        "frames": [
            {
                **frame,
                "background_namespace_active": (
                    not frame["background_namespace_active"]
                ),
                "background_namespace_opening_count": (
                    frame["background_namespace_opening_count"] + 17
                ),
            }
            for frame in ledger["frames"]
        ],
    }


def _target_opening_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "candidate_id",
            "event_id",
            "support_start_frame",
            "qualification_frame",
            "deliveries",
        )
    }


def _outcome_for_parent_label(
    parent_label: str, delivered: bool
) -> str:
    if parent_label == "formed_eligible_delivery":
        require(delivered, "baseline_delivery_not_preserved")
        return "parent_formed_delivery_cell_remains_token_qualified"
    if parent_label in {
        "episode_opened_before_alertable_window",
        "episode_opened_inside_window_before_target_attribution",
        "pre_invalid_baseline_latch_guard_quarantine",
    }:
        require(delivered, f"recoverable_parent_label_not_recovered:{parent_label}")
        return f"recovered_{parent_label}"
    require(
        parent_label == "never_active_relation" and not delivered,
        f"unexpected_parent_label_or_delivery:{parent_label}:{delivered}",
    )
    return "remaining_never_active_relation"


def build_terminal_from_frozen_tokens(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    parent_terminal, events, _, _ = _parent_blind_context(
        repo, config, include_full_traces=False
    )
    inventory, ledgers = load_and_verify_token_inventory(
        repo, config, bindings
    )
    parent_rows = {
        (row["candidate_id"], row["event_id"]): row
        for candidate in parent_terminal["candidate_results"]
        for row in candidate["events"]
    }
    require(len(parent_rows) == 36, "parent_cell_count_drift")
    results = []
    outcome_counts: Counter[str] = Counter()
    delivery_keys = set()
    opening_before_qualification = 0
    background_namespace_invariance_failures = 0
    background_namespace_mutation_checks = 0
    recoverable_cells_with_prequalification_background_activity = 0
    one_shot_cardinality_violations = 0
    token_qualified = 0
    for candidate_id in CANDIDATES:
        rows = []
        for event in events:
            key = (candidate_id, event["event_id"])
            require(key in ledgers, "token_ledger_cell_missing")
            opener = apply_ordered_isolated_opener(
                ledgers[key],
                int(config["evaluation"]["min_alert_frames"]),
            )
            mutated_opener = apply_ordered_isolated_opener(
                mutate_background_namespace(ledgers[key]),
                int(config["evaluation"]["min_alert_frames"]),
            )
            background_namespace_mutation_checks += 1
            background_namespace_invariance_failures += int(
                _target_opening_projection(opener)
                != _target_opening_projection(mutated_opener)
            )
            parent = parent_rows[key]
            delivered = bool(opener["deliveries"])
            one_shot_cardinality_violations += int(
                len(opener["deliveries"]) > 1
            )
            token_qualified += int(delivered)
            outcome = _outcome_for_parent_label(
                parent["failure_label"], delivered
            )
            if parent["failure_label"] in {
                "episode_opened_before_alertable_window",
                "episode_opened_inside_window_before_target_attribution",
                "pre_invalid_baseline_latch_guard_quarantine",
            }:
                has_background_activity = bool(
                    opener[
                        "background_namespace_prequalification_active_frame_count"
                    ]
                    or opener[
                        "background_namespace_prequalification_opening_count"
                    ]
                )
                require(
                    has_background_activity,
                    (
                        "recoverable_cell_without_prequalification_"
                        "background_namespace_activity"
                    ),
                )
                recoverable_cells_with_prequalification_background_activity += 1
            outcome_counts[outcome] += 1
            for delivery in opener["deliveries"]:
                delivery_key = delivery["delivery_key"]
                require(
                    delivery_key not in delivery_keys,
                    "duplicate_delivery_key",
                )
                delivery_keys.add(delivery_key)
                opening_before_qualification += int(
                    int(delivery["frame_id"])
                    < int(opener["qualification_frame"])
                )
            rows.append(
                {
                    **opener,
                    "background_namespace_mutation_invariant": (
                        _target_opening_projection(opener)
                        == _target_opening_projection(mutated_opener)
                    ),
                    "parent_failure_label": parent["failure_label"],
                    "mechanism_outcome": outcome,
                }
            )
        results.append(
            {
                "candidate_id": candidate_id,
                "cell_count": len(rows),
                "token_qualified_cells": sum(
                    bool(row["deliveries"]) for row in rows
                ),
                "events": rows,
            }
        )
    observed = {
        outcome: outcome_counts.get(outcome, 0) for outcome in OUTCOMES
    }
    require(
        observed == config["evaluation"]["expected_outcome_counts"],
        f"outcome_count_drift:{observed!r}",
    )
    recovered = sum(
        count
        for outcome, count in observed.items()
        if outcome.startswith("recovered_")
    )
    remaining = observed["remaining_never_active_relation"]
    accounting_gap = 36 - token_qualified - remaining
    gates = {
        "token_qualified_cells": token_qualified,
        "recovered_cells": recovered,
        "remaining_no_support_cells": remaining,
        "opening_before_qualification_count": opening_before_qualification,
        "background_namespace_mutation_checks": (
            background_namespace_mutation_checks
        ),
        "background_namespace_invariance_failure_count": (
            background_namespace_invariance_failures
        ),
        "recoverable_cells_with_prequalification_background_activity": (
            recoverable_cells_with_prequalification_background_activity
        ),
        "one_shot_cardinality_violation_count": (
            one_shot_cardinality_violations
        ),
        "duplicate_delivery_key_count": 0,
        "accounting_gap_count": accounting_gap,
        "ordered_opening_passed": (
            opening_before_qualification
            <= config["evaluation"]["opening_before_qualification_max"]
        ),
        "isolation_passed": (
            background_namespace_invariance_failures
            <= config["evaluation"][
                "background_namespace_invariance_failures_max"
            ]
            and background_namespace_mutation_checks == 36
            and recoverable_cells_with_prequalification_background_activity
            == recovered
        ),
        "one_shot_passed": (
            one_shot_cardinality_violations
            <= config["evaluation"]["one_shot_cardinality_violations_max"]
            and len(delivery_keys) == token_qualified
        ),
        "accounting_passed": (
            accounting_gap <= config["evaluation"]["accounting_gap_max"]
        ),
    }
    gates["passed"] = all(
        [
            gates["token_qualified_cells"]
            == config["evaluation"]["expected_token_qualified_cells"],
            gates["recovered_cells"]
            == config["evaluation"]["expected_recovered_cells"],
            gates["remaining_no_support_cells"]
            == config["evaluation"]["expected_remaining_no_support_cells"],
            gates["ordered_opening_passed"],
            gates["isolation_passed"],
            gates["one_shot_passed"],
            gates["accounting_passed"],
        ]
    )
    require(gates["passed"], "mechanism_gate_failed")
    token_inventory_path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["token_inventory"]
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": config["stage"],
        "terminal_state": TERMINAL_STATE,
        "authority": (
            "truth_assisted_oracle_mechanism_diagnostic_only_"
            "no_runtime_or_selection_authority"
        ),
        "bindings": bindings,
        "config_sha256": sha256_file(config_path),
        "phase_evidence": {
            "parent_event_scope_candidate_blind_inventory_verified": True,
            "truth_join_used_only_to_build_tokens": True,
            "token_inventory_frozen_before_opener_execution": True,
            "token_inventory_path": (
                f"{config['outputs']['root']}/"
                f"{config['outputs']['token_inventory']}"
            ),
            "token_inventory_sha256": sha256_file(token_inventory_path),
            "token_ledger_count": inventory["ledger_count"],
            "token_frame_count": inventory["frame_count"],
            "raw_truth_payload_available_to_opener": False,
            "truth_derived_event_scoped_tokens_available_to_opener": True,
            "truth_window_boundaries_encoded_by_ledger_extent": True,
        },
        "verified_scope": {
            **config["expected_scope"],
            "baseline_fsm_changes": 0,
            "route_invalid_guard_changes": 0,
            "clearance_changes": 0,
            "candidate_blind_runtime_token_producers_proven": 0,
        },
        "outcome_counts": observed,
        "candidate_results": results,
        "mechanism_gate": gates,
        "claim_boundary": {
            **config["authority"],
            "oracle_upper_bound_only": True,
            "deployable_repair_proven": False,
        },
    }


def run(repo: Path, config_path: Path) -> dict[str, Any]:
    build_and_write_token_inventory(repo, config_path)
    terminal = build_terminal_from_frozen_tokens(repo, config_path)
    config, _ = load_and_verify_config(repo, config_path)
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["terminal_receipt"]
    )
    atomic_write_json(path, terminal)
    return terminal


def validate_outputs(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    parent_terminal, events, traces, parent_config = (
        _parent_blind_context(repo, config)
    )
    truth_index = _load_truth_index(repo, parent_config)
    inventory, ledgers = load_and_verify_token_inventory(
        repo, config, bindings
    )
    minimum_iou = float(
        config["evaluation"]["truth_attribution_minimum_iou"]
    )
    recomputed = {}
    for candidate_id in CANDIDATES:
        for event in events:
            trace_key = (
                candidate_id,
                event["source_id"],
                event["sequence_id"],
            )
            key = (candidate_id, event["event_id"])
            recomputed[key] = build_token_ledger(
                candidate_id=candidate_id,
                event=event,
                trace=traces[trace_key],
                truth_index=truth_index,
                minimum_iou=minimum_iou,
            )
            require(
                recomputed[key] == ledgers[key],
                f"token_ledger_recompute_drift:{key!r}",
            )
    expected_terminal = build_terminal_from_frozen_tokens(
        repo, config_path
    )
    terminal_path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["terminal_receipt"]
    )
    require(terminal_path.is_file(), "terminal_receipt_missing")
    observed_terminal = load_json(terminal_path)
    require(
        observed_terminal == expected_terminal,
        "terminal_receipt_recompute_drift",
    )
    require(
        parent_terminal["attribution_gate"]["passed"] is True,
        "parent_attribution_gate_drift",
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "stage": config["stage"],
        "status": "VALID",
        "config_sha256": sha256_file(config_path),
        "terminal_sha256": sha256_file(terminal_path),
        "token_inventory_sha256": sha256_file(
            repo
            / config["outputs"]["root"]
            / config["outputs"]["token_inventory"]
        ),
        "checks": {
            "parent_attribution_receipts_verified": True,
            "token_ledgers_exactly_recomputed": len(recomputed),
            "token_inventory_ledgers_verified": inventory["ledger_count"],
            "raw_truth_payload_fields_rejected": True,
            "truth_derived_token_and_window_scope_declared": True,
            "background_namespace_mutation_invariance_verified": 36,
            "ordered_isolated_one_shot_gate_passed": True,
            "terminal_exactly_recomputed": True,
            "candidate_comparison_or_selection_authority": False,
            "runtime_or_production_authority": False,
        },
    }
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["validation_receipt"]
    )
    atomic_write_json(path, validation)
    return validation
