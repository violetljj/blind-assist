#!/usr/bin/env python3
"""Diagnose a route-invalid, reset-scoped lifecycle guard on frozen R2 traces.

The guard is deliberately post-candidate. It does not rerun or alter C1-C3, and
it never feeds truth into lifecycle state. Truth is joined only after the guarded
trace has been constructed, for the frozen 1,500 ms clearance diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from metric_profiles_r2_l1 import (  # noqa: E402
    _attributed_event_ids_for_track_ids,
    _delivery_groups,
    identity,
    load_truth_frame_index,
    percentile_higher,
)

CONFIG_SCHEMA = (
    "blindassist_ustrf_route_invalid_reset_lifecycle_diagnostic_r1"
)
TRACE_SCHEMA = (
    "blindassist_ustrf_route_invalid_reset_lifecycle_trace_r1"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_route_invalid_reset_lifecycle_terminal_r1"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_route_invalid_reset_lifecycle_validation_r1"
)
TERMINAL_STATE = "MECHANISM_DIAGNOSTIC_COMPLETE"
CANDIDATES = (
    "C1_CAUSAL_ROUTE_RELATION_FSM",
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
    "C3_DUAL_KEY_CLEARANCE_FSM",
)
IMPLEMENTATION_PATHS = {
    "diagnostic_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "route_invalid_reset_lifecycle_diagnostic.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_route_invalid_reset_lifecycle_diagnostic.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_route_invalid_reset_lifecycle_diagnostic.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_route_invalid_reset_lifecycle_diagnostic.py"
    ),
    "postoutput_truth_join_dependency_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "metric_profiles_r2_l1.py"
    ),
}
GUARD_FORBIDDEN_INPUT_KEYS = {
    "truth",
    "truth_status",
    "truth_clear",
    "truth_terminal_clear",
    "truth_terminal_clear_frame",
    "critical",
    "eligibility",
    "metric_eligibility",
    "metric_classification",
    "scoring_label",
    "event_id",
}


class DiagnosticContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticContractError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    data = canonical_bytes(value)
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(load_json(temporary) == value, f"atomic_write_verify_failed:{path}")
    os.replace(temporary, path)


def scoped_key(
    source_id: str,
    sequence_id: str,
    reset_segment: int,
    local_key: int,
    activation_ordinal: int,
) -> str:
    return (
        f"{source_id}::{sequence_id}::reset-{reset_segment}"
        f"::local-{local_key}::activation-{activation_ordinal}"
    )


def _closure_event(
    *,
    key: str,
    local_key: int,
    reason: str,
    key_reset_segment: int,
) -> dict[str, Any]:
    return {
        "kind": "closure",
        "scoped_episode_key": key,
        "local_delivery_key": local_key,
        "reason": reason,
        "key_reset_segment": key_reset_segment,
    }


def assert_guard_input_uncontaminated(
    value: Any, path: str = "$"
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(
                key not in GUARD_FORBIDDEN_INPUT_KEYS,
                f"guard_input_forbidden_field:{path}.{key}",
            )
            assert_guard_input_uncontaminated(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_guard_input_uncontaminated(
                child, f"{path}[{index}]"
            )


def apply_guard(
    frames: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply only route-invalid closure and reset-scoped keying."""

    active: dict[int, tuple[str, int]] = {}
    ordinals: dict[tuple[int, int], int] = defaultdict(int)
    output: list[dict[str, Any]] = []
    reset_segment = -1
    counters: dict[str, int] = defaultdict(int)
    previous_route_known: bool | None = None
    previous_baseline_active = False

    for frame_index, frame in enumerate(frames):
        assert_guard_input_uncontaminated(frame)
        events: list[dict[str, Any]] = []
        active_before_frame = bool(active)
        if (
            not bool(frame["route_known"])
            and previous_route_known is True
            and previous_baseline_active
            and bool(frame["candidate_active"])
        ):
            counters[
                "baseline_known_to_invalid_active_transitions"
            ] += 1
        if bool(frame["state_reset_before_frame"]):
            counters["reset_frames"] += 1
            for local_key, (key, key_segment) in sorted(active.items()):
                events.append(
                    _closure_event(
                        key=key,
                        local_key=local_key,
                        reason="reset_scope_end",
                        key_reset_segment=key_segment,
                    )
                )
                counters["reset_scope_closures"] += 1
            active.clear()
            reset_segment += 1
        require(reset_segment >= 0, "first_frame_must_start_reset_scope")

        if not bool(frame["route_known"]):
            require(
                not frame["deliveries"],
                "baseline_delivery_on_route_invalid_frame",
            )
            if active:
                counters["route_invalid_entries_with_active"] += 1
            for local_key, (key, key_segment) in sorted(active.items()):
                events.append(
                    _closure_event(
                        key=key,
                        local_key=local_key,
                        reason="route_invalid",
                        key_reset_segment=key_segment,
                    )
                )
                counters["route_invalid_closures"] += 1
            active.clear()
        else:
            for raw_local_key in frame["closures"]:
                local_key = int(raw_local_key)
                existing = active.pop(local_key, None)
                if existing is None:
                    counters["orphan_baseline_closures_after_guard"] += 1
                    continue
                key, key_segment = existing
                events.append(
                    _closure_event(
                        key=key,
                        local_key=local_key,
                        reason="baseline_known_route_closure",
                        key_reset_segment=key_segment,
                    )
                )
                counters["baseline_known_route_closures"] += 1

            for raw_local_key, track_ids in _delivery_groups(frame):
                local_key = int(raw_local_key)
                require(
                    local_key not in active,
                    "baseline_duplicate_delivery_while_guard_active",
                )
                ordinal_key = (reset_segment, local_key)
                ordinals[ordinal_key] += 1
                key = scoped_key(
                    str(frame["source_id"]),
                    str(frame["sequence_id"]),
                    reset_segment,
                    local_key,
                    ordinals[ordinal_key],
                )
                active[local_key] = (key, reset_segment)
                events.append(
                    {
                        "kind": "delivery",
                        "scoped_episode_key": key,
                        "local_delivery_key": local_key,
                        "delivery_track_ids": [
                            int(value) for value in track_ids
                        ],
                        "key_reset_segment": reset_segment,
                    }
                )
                counters["guarded_deliveries"] += 1

        if not bool(frame["route_known"]) and active:
            counters["route_invalid_active_after_frame"] += 1
        if any(segment != reset_segment for _, segment in active.values()):
            counters["active_key_cross_reset"] += 1
        if bool(frame["state_reset_before_frame"]) and active_before_frame:
            counters["reset_frames_with_prior_active"] += 1

        output.append(
            {
                "source_id": str(frame["source_id"]),
                "sequence_id": str(frame["sequence_id"]),
                "frame_id": int(frame["frame_id"]),
                "source_capture_timestamp_ns": int(
                    frame["source_capture_timestamp_ns"]
                ),
                "state_reset_before_frame": bool(
                    frame["state_reset_before_frame"]
                ),
                "reset_segment": reset_segment,
                "route_known": bool(frame["route_known"]),
                "baseline_candidate_active": bool(
                    frame["candidate_active"]
                ),
                "guarded_active": bool(active),
                "lifecycle_events": events,
            }
        )
        counters["frames"] += 1
        counters["baseline_route_invalid_active_frames"] += int(
            not bool(frame["route_known"])
            and (
                bool(frame["candidate_active"])
                or bool(frame["deliveries"])
            )
        )
        counters["guarded_route_invalid_active_frames"] += int(
            not bool(frame["route_known"]) and bool(active)
        )
        counters["baseline_deliveries"] += len(frame["deliveries"])
        counters["baseline_closures"] += len(frame["closures"])
        if frame_index == 0:
            require(
                bool(frame["state_reset_before_frame"]),
                "trace_first_frame_without_reset",
            )
        previous_route_known = bool(frame["route_known"])
        previous_baseline_active = bool(frame["candidate_active"])

    return output, dict(counters)


def _verify_file_binding(
    repo: Path, binding: dict[str, Any], label: str
) -> Path:
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing")
    require(
        sha256_file(path) == binding["sha256"],
        f"{label}_sha256_mismatch",
    )
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
            "single_variable_intervention",
            "terminalization_contract",
            "parent_bindings",
            "implementation_bindings",
            "expected_scope",
            "evaluation",
            "outputs",
        },
        "config_key_inventory_drift",
    )
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_mismatch")
    require(
        config.get("stage") == "ROUTE-INVALID-RESET-LIFECYCLE-DIAGNOSTIC-R1",
        "config_stage_mismatch",
    )
    require(
        config.get("status")
        == "preregistered_single_variable_frozen_trace_mechanism_diagnostic",
        "config_status_drift",
    )
    require(config.get("frozen_on") == "2026-07-24", "frozen_on_drift")
    require(
        config.get("non_goals")
        == [
            "modify the frozen C1-C3 candidate implementation or authoritative traces",
            "rerun YOLO, T0 association, route construction, or any candidate",
            "change min_alert_frames, min_clear_frames, route thresholds, metric thresholds, denominators, or truth",
            "repair or synthesize candidate_consume_timestamp_ns",
            "compute a total score, compare candidates, rank, select, promote, or authorize shadow/H2/production",
        ],
        "non_goals_drift",
    )
    require(
        tuple(config.get("candidate_roster", [])) == CANDIDATES,
        "candidate_roster_drift",
    )
    authority = config.get("authority", {})
    require(
        authority
        == {
            "maximum": "MECHANISM_DIAGNOSTIC_ONLY",
            "candidate_rerun": False,
            "detector_rerun": False,
            "candidate_comparison": False,
            "winner_or_ranking": False,
            "selection": False,
            "threshold_or_denominator_change": False,
            "consume_timestamp_repair": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
            "new_data": False,
            "training": False,
        },
        "authority_opened",
    )
    intervention = config.get("single_variable_intervention", {})
    require(
        intervention
        == {
            "baseline_candidate_outputs_immutable": True,
            "known_route_delivery_and_closure_inputs_immutable": True,
            "on_route_invalid": "close_all_active_same_frame_and_quarantine_until_new_delivery",
            "on_reset": "close_prior_scope_before_processing_new_scope",
            "episode_key_scope": "source_sequence_reset_local_key_activation_ordinal",
            "truth_available_to_guard": False,
        },
        "single_variable_intervention_drift",
    )
    require(
        config.get("terminalization_contract")
        == {
            "route_invalid_reason": "route_invalid",
            "reset_reason": "reset_scope_end",
            "terminalization_is_not_truth_clearance": True,
            "only_relation_based_known_route_closure_can_enter_clearance_numerator": True,
            "route_recovery_cannot_reactivate_a_pre_invalid_episode": True,
        },
        "terminalization_contract_drift",
    )
    require(
        config.get("evaluation")
        == {
            "route_invalid_active_frames_max": 0,
            "active_key_cross_reset_max": 0,
            "duplicate_scoped_episode_keys_max": 0,
            "truth_attribution_minimum_iou": 0.3,
            "clearance_horizon_ms": 1500,
            "clearance_required_rate": 0.9,
            "clearance_numerator_allowed_closure_reason": "baseline_known_route_closure",
            "route_invalid_or_reset_terminalization_clearance_credit": 0,
        },
        "evaluation_contract_drift",
    )
    require(
        config.get("expected_scope")
        == {
            "baseline_authoritative_traces": 123,
            "baseline_frames": 186687,
            "sequences_per_candidate": 41,
            "frames_per_candidate": 62229,
            "sequence_scope_starts_per_candidate": 41,
            "discontinuity_resets_per_candidate": 15,
            "reset_frames_per_candidate": 56,
            "clearance_denominator_per_candidate": 12,
            "candidate_reruns": 0,
            "detector_reruns": 0,
            "new_data": 0,
        },
        "expected_scope_drift",
    )
    require(
        config.get("outputs")
        == {
            "root": (
                "artifacts.local/evidence/"
                "ustrf-route-invalid-reset-lifecycle-diagnostic-r1"
            ),
            "terminal_receipt": "terminal-receipt-r1.json",
            "validation_receipt": "validation-receipt-r1.json",
        },
        "output_namespace_drift",
    )

    bindings: dict[str, str] = {
        "config_sha256": sha256_file(config_path)
    }
    require(
        set(config.get("parent_bindings", {}))
        == {
            "baseline_metric_profile_config",
            "baseline_metric_profile_terminal",
            "baseline_metric_profile_validation",
            "candidate_replay_a2_terminal",
            "eligibility_mask",
            "crowdbot_truth",
            "lilocbench_truth",
            "lilocbench_replacement_truth",
        },
        "parent_binding_inventory_drift",
    )
    for name, binding in config["parent_bindings"].items():
        path = _verify_file_binding(repo, binding, f"parent_{name}")
        bindings[f"{name}_sha256"] = sha256_file(path)
    implementation = config["implementation_bindings"]
    require(
        set(implementation) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_inventory_drift",
    )
    for name, relative_path in IMPLEMENTATION_PATHS.items():
        path = repo / relative_path
        require(path.is_file(), f"implementation_{name}_missing")
        actual = sha256_file(path)
        require(
            actual == implementation[name],
            f"implementation_{name}_sha256_mismatch",
        )
        bindings[name] = actual
    return config, bindings


def _mask_groups(
    mask: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mask["preoutput_frame_ledger"]:
        groups[(str(row["source_id"]), str(row["sequence_id"]))].append(
            row
        )
    require(len(groups) == 41, "mask_sequence_count_drift")
    return groups


def _load_baseline_traces(
    repo: Path,
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, Any]]],
    str,
    int,
]:
    terminal = load_json(
        repo / config["parent_bindings"]["candidate_replay_a2_terminal"]["path"]
    )
    require(
        terminal.get("terminal_state") == "CANDIDATE_REPLAY_COMPLETE",
        "parent_candidate_terminal_not_complete",
    )
    inventory = terminal["candidate_execution"]["trace_inventory"]
    require(len(inventory) == 123, "parent_trace_inventory_count_drift")
    grouped: dict[
        str, dict[tuple[str, str], dict[str, Any]]
    ] = defaultdict(dict)
    inventory_hash_rows = []
    consume_timestamp_frames = 0
    for row in inventory:
        candidate_id = str(row["candidate_id"])
        require(candidate_id in CANDIDATES, "unknown_parent_candidate")
        key = (str(row["source_id"]), str(row["sequence_id"]))
        require(key not in grouped[candidate_id], "duplicate_parent_trace")
        trace_path = repo / str(row["trace_path"])
        require(trace_path.is_file(), "parent_trace_missing")
        require(
            sha256_file(trace_path) == row["trace_sha256"],
            "parent_trace_sha256_mismatch",
        )
        trace = load_json(trace_path)
        require(trace.get("candidate_id") == candidate_id, "trace_candidate_drift")
        require(
            (trace.get("source_id"), trace.get("sequence_id")) == key,
            "trace_source_sequence_drift",
        )
        require(
            len(trace["frames"]) == int(row["frame_count"]),
            "trace_frame_count_drift",
        )
        consume_timestamp_frames += sum(
            isinstance(frame.get("candidate_consume_timestamp_ns"), int)
            for frame in trace["frames"]
        )
        trace["verified_source_file_sha256"] = row["trace_sha256"]
        grouped[candidate_id][key] = trace
        inventory_hash_rows.append(
            {
                "candidate_id": candidate_id,
                "source_id": key[0],
                "sequence_id": key[1],
                "trace_sha256": row["trace_sha256"],
            }
        )
    require(set(grouped) == set(CANDIDATES), "parent_candidate_set_drift")
    require(
        all(len(grouped[candidate]) == 41 for candidate in CANDIDATES),
        "parent_sequence_count_drift",
    )
    return (
        grouped,
        sha256_bytes(canonical_bytes(inventory_hash_rows)),
        consume_timestamp_frames,
    )


def _output_trace_path(
    config: dict[str, Any],
    candidate_id: str,
    source_id: str,
    sequence_id: str,
) -> str:
    digest = hashlib.sha256(
        f"{source_id}\0{sequence_id}".encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{config['outputs']['root']}/traces/{candidate_id}/"
        f"{digest}.json"
    )


def _clearance_diagnostic(
    *,
    candidate_id: str,
    guarded_traces: dict[tuple[str, str], dict[str, Any]],
    baseline_traces: dict[tuple[str, str], dict[str, Any]],
    mask: dict[str, Any],
    truth_index: dict[
        tuple[str, str, int, int], list[dict[str, Any]]
    ],
    minimum_iou: float,
    horizon_ms: float,
    required_rate: float,
) -> dict[str, Any]:
    deliveries: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    frame_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}

    for key in sorted(guarded_traces):
        guarded = guarded_traces[key]
        baseline_by_identity = {
            identity(frame): frame
            for frame in baseline_traces[key]["frames"]
        }
        for frame in guarded["frames"]:
            frame_lookup[
                (
                    frame["source_id"],
                    frame["sequence_id"],
                    int(frame["frame_id"]),
                )
            ] = frame
            baseline = baseline_by_identity[identity(frame)]
            for event in frame["lifecycle_events"]:
                common = {
                    "source_id": frame["source_id"],
                    "sequence_id": frame["sequence_id"],
                    "frame_id": int(frame["frame_id"]),
                    "source_capture_timestamp_ns": int(
                        frame["source_capture_timestamp_ns"]
                    ),
                    "scoped_episode_key": event["scoped_episode_key"],
                }
                if event["kind"] == "delivery":
                    event_ids = _attributed_event_ids_for_track_ids(
                        baseline,
                        truth_index.get(identity(baseline), []),
                        event["delivery_track_ids"],
                        minimum_iou,
                    )
                    deliveries.append(
                        {
                            **common,
                            "event_ids": event_ids,
                            "delivery_track_ids": event[
                                "delivery_track_ids"
                            ],
                        }
                    )
                else:
                    closures.append({**common, "reason": event["reason"]})

    clearance_events = [
        event
        for event in mask["events"]
        if event["metrics"]["clearance"]["classification"] == "eligible"
    ]
    require(len(clearance_events) == 12, "clearance_denominator_drift")
    event_rows = []
    success_delays = []
    success_count = 0
    for event in clearance_events:
        event_deliveries = [
            delivery
            for delivery in deliveries
            if event["event_id"] in delivery["event_ids"]
            and delivery["source_id"] == event["source_id"]
            and delivery["sequence_id"] == event["sequence_id"]
            and int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["truth_terminal_clear_frame"])
        ]
        if not event_deliveries:
            event_rows.append(
                {
                    "event_id": event["event_id"],
                    "source_id": event["source_id"],
                    "status": "no_guarded_delivery",
                    "delay_ms": None,
                    "closure_reason": None,
                }
            )
            continue
        first = event_deliveries[0]
        matching = [
            closure
            for closure in closures
            if closure["source_id"] == event["source_id"]
            and closure["sequence_id"] == event["sequence_id"]
            and closure["scoped_episode_key"]
            == first["scoped_episode_key"]
            and int(event["anchors"]["truth_terminal_clear_frame"])
            <= closure["frame_id"]
            <= int(event["anchors"]["end_frame"])
            and closure["reason"] == "baseline_known_route_closure"
        ]
        if not matching:
            event_rows.append(
                {
                    "event_id": event["event_id"],
                    "source_id": event["source_id"],
                    "status": "no_same_scope_post_truth_clear_closure",
                    "delay_ms": None,
                    "closure_reason": None,
                }
            )
            continue
        truth_clear_frame = frame_lookup[
            (
                event["source_id"],
                event["sequence_id"],
                int(event["anchors"]["truth_terminal_clear_frame"]),
            )
        ]
        delay_ms = max(
            0.0,
            (
                int(matching[0]["source_capture_timestamp_ns"])
                - int(truth_clear_frame["source_capture_timestamp_ns"])
            )
            / 1_000_000.0,
        )
        passed = delay_ms <= horizon_ms
        success_count += int(passed)
        if passed:
            success_delays.append(delay_ms)
        event_rows.append(
            {
                "event_id": event["event_id"],
                "source_id": event["source_id"],
                "status": (
                    "closed_within_horizon"
                    if passed
                    else "closure_after_horizon"
                ),
                "delay_ms": delay_ms,
                "closure_reason": matching[0]["reason"],
            }
        )

    return {
        "candidate_id": candidate_id,
        "truth_join_timing": "after_guarded_output_only",
        "numerator": success_count,
        "denominator": len(clearance_events),
        "value": success_count / len(clearance_events),
        "required_rate": required_rate,
        "horizon_ms": horizon_ms,
        "p95_delay_ms": percentile_higher(success_delays, 95),
        "gate_passed": (
            success_count / len(clearance_events) >= required_rate
        ),
        "events": event_rows,
    }


def build_expected(
    repo: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    config, bindings = load_and_verify_config(repo, config_path)
    mask = load_json(
        repo / config["parent_bindings"]["eligibility_mask"]["path"]
    )
    mask_groups = _mask_groups(mask)
    (
        baseline,
        inventory_sha256,
        consume_timestamp_frames,
    ) = _load_baseline_traces(repo, config)
    require(
        consume_timestamp_frames == 0,
        "parent_consume_timestamp_observation_must_remain_separate",
    )
    outputs: dict[
        str, dict[tuple[str, str], dict[str, Any]]
    ] = defaultdict(dict)
    candidate_aggregates: dict[str, dict[str, int]] = {}
    trace_inventory = []
    for candidate_id in CANDIDATES:
        aggregate: dict[str, int] = defaultdict(int)
        delivered_keys: set[str] = set()
        for key in sorted(baseline[candidate_id]):
            source_frames = baseline[candidate_id][key]["frames"]
            require(key in mask_groups, "trace_not_in_mask")
            require(
                [identity(frame) for frame in source_frames]
                == [identity(frame) for frame in mask_groups[key]],
                "trace_mask_identity_drift",
            )
            guarded_frames, counters = apply_guard(source_frames)
            for name, value in counters.items():
                aggregate[name] += int(value)
            for frame in guarded_frames:
                for event in frame["lifecycle_events"]:
                    if event["kind"] == "delivery":
                        require(
                            event["scoped_episode_key"] not in delivered_keys,
                            "duplicate_scoped_episode_key",
                        )
                        delivered_keys.add(event["scoped_episode_key"])
            trace = {
                "schema": TRACE_SCHEMA,
                "stage": config["stage"],
                "authority": "mechanism_diagnostic_only",
                "candidate_id": candidate_id,
                "source_id": key[0],
                "sequence_id": key[1],
                "baseline_trace_sha256": baseline[candidate_id][key][
                    "verified_source_file_sha256"
                ],
                "frame_count": len(guarded_frames),
                "frames": guarded_frames,
            }
            outputs[candidate_id][key] = trace
            output_path = _output_trace_path(
                config, candidate_id, key[0], key[1]
            )
            trace_inventory.append(
                {
                    "candidate_id": candidate_id,
                    "source_id": key[0],
                    "sequence_id": key[1],
                    "path": output_path,
                    "frame_count": len(guarded_frames),
                    "sha256": sha256_bytes(canonical_bytes(trace)),
                }
            )
        require(aggregate["frames"] == 62229, "candidate_frame_count_drift")
        require(aggregate["reset_frames"] == 56, "reset_frame_count_drift")
        candidate_aggregates[candidate_id] = aggregate

    require(len(trace_inventory) == 123, "output_trace_inventory_drift")
    profile_config = load_json(
        repo
        / config["parent_bindings"]["baseline_metric_profile_config"]["path"]
    )
    truth_index = load_truth_frame_index(profile_config, repo)

    candidate_results = []
    for candidate_id in CANDIDATES:
        aggregate = candidate_aggregates[candidate_id]
        clearance = _clearance_diagnostic(
            candidate_id=candidate_id,
            guarded_traces=outputs[candidate_id],
            baseline_traces=baseline[candidate_id],
            mask=mask,
            truth_index=truth_index,
            minimum_iou=float(
                config["evaluation"]["truth_attribution_minimum_iou"]
            ),
            horizon_ms=float(config["evaluation"]["clearance_horizon_ms"]),
            required_rate=float(
                config["evaluation"]["clearance_required_rate"]
            ),
        )
        route_gate = (
            aggregate["guarded_route_invalid_active_frames"] == 0
            and aggregate["route_invalid_active_after_frame"] == 0
        )
        reset_gate = aggregate["active_key_cross_reset"] == 0
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "verified_frames": aggregate["frames"],
                "discontinuity_resets": 15,
                "sequence_scope_starts": 41,
                "baseline_route_invalid_active_frames": aggregate[
                    "baseline_route_invalid_active_frames"
                ],
                "baseline_known_to_invalid_active_transitions": aggregate[
                    "baseline_known_to_invalid_active_transitions"
                ],
                "guarded_route_invalid_active_frames": aggregate[
                    "guarded_route_invalid_active_frames"
                ],
                "route_invalid_entries_with_active": aggregate[
                    "route_invalid_entries_with_active"
                ],
                "route_invalid_closures": aggregate[
                    "route_invalid_closures"
                ],
                "route_invalid_gate_passed": route_gate,
                "reset_frames_with_prior_active": aggregate[
                    "reset_frames_with_prior_active"
                ],
                "reset_scope_closures": aggregate[
                    "reset_scope_closures"
                ],
                "active_key_cross_reset": aggregate[
                    "active_key_cross_reset"
                ],
                "duplicate_scoped_episode_keys": 0,
                "reset_scope_gate_passed": reset_gate,
                "orphan_baseline_closures_after_guard": aggregate[
                    "orphan_baseline_closures_after_guard"
                ],
                "clearance": clearance,
                "mechanism_gate_passed": (
                    route_gate and reset_gate and clearance["gate_passed"]
                ),
            }
        )

    terminal = {
        "schema": TERMINAL_SCHEMA,
        "stage": config["stage"],
        "terminal_state": TERMINAL_STATE,
        "authority": "mechanism_diagnostic_only_no_candidate_comparison",
        "bindings": bindings,
        "verified_scope": {
            "baseline_trace_inventory_sha256": inventory_sha256,
            "baseline_authoritative_traces": 123,
            "baseline_frames": 186687,
            "candidate_reruns": 0,
            "detector_reruns": 0,
            "new_data": 0,
            "guarded_lifecycle_traces": 123,
            "guarded_lifecycle_frames": 186687,
            "truth_joined_after_guarded_output": True,
            "consume_timestamp_frames_read": consume_timestamp_frames,
            "consume_timestamp_frames_written": 0,
        },
        "candidate_results": candidate_results,
        "trace_inventory": trace_inventory,
        "overall_mechanism_gate_passed": all(
            row["mechanism_gate_passed"] for row in candidate_results
        ),
        "claim_boundary": {
            "candidate_comparison": False,
            "winner_or_ranking": False,
            "selection": False,
            "threshold_or_denominator_change": False,
            "consume_timestamp_repair": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
        },
    }
    return terminal, outputs


def write_expected(
    repo: Path,
    config_path: Path,
    terminal: dict[str, Any],
    outputs: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> Path:
    config = load_json(config_path)
    output_root = repo / config["outputs"]["root"]
    require(not output_root.exists(), "refusing_to_overwrite_output_namespace")
    for row in terminal["trace_inventory"]:
        trace = outputs[row["candidate_id"]][
            (row["source_id"], row["sequence_id"])
        ]
        atomic_write_json(repo / row["path"], trace)
    terminal_path = output_root / config["outputs"]["terminal_receipt"]
    atomic_write_json(terminal_path, terminal)
    return terminal_path


def validate_written(
    repo: Path,
    config_path: Path,
    terminal_path: Path,
) -> dict[str, Any]:
    expected, outputs = build_expected(repo, config_path)
    observed = load_json(terminal_path)
    require(
        observed == expected,
        "terminal_not_exact_independent_recomputation",
    )
    for row in expected["trace_inventory"]:
        path = repo / row["path"]
        require(path.is_file(), "guarded_trace_missing")
        require(sha256_file(path) == row["sha256"], "guarded_trace_sha_drift")
        require(
            load_json(path)
            == outputs[row["candidate_id"]][
                (row["source_id"], row["sequence_id"])
            ],
            "guarded_trace_content_drift",
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "VALID",
        "terminal_state": TERMINAL_STATE,
        "terminal_receipt_sha256": sha256_file(terminal_path),
        "baseline_traces_reverified": 123,
        "baseline_frames_reverified": 186687,
        "guarded_traces_recomputed": 123,
        "guarded_frames_recomputed": 186687,
        "overall_mechanism_gate_passed": expected[
            "overall_mechanism_gate_passed"
        ],
    }
