#!/usr/bin/env python3
"""Attribute known-route eligible-delivery formation failures on frozen traces.

The first phase builds and persists a candidate-blind formation ledger from the
already-authoritative A2 candidate traces and the route-invalid/reset guard
traces.  Truth and clearance eligibility are loaded only after every blind
ledger has been written and hash-inventoried.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from metric_profiles_r2_l1 import (  # noqa: E402
    _attributed_event_ids_for_track_ids,
    _delivery_groups,
    identity,
    load_truth_frame_index,
)

CONFIG_SCHEMA = (
    "blindassist_ustrf_known_route_eligible_delivery_failure_attribution_r1"
)
BLIND_TRACE_SCHEMA = (
    "blindassist_ustrf_candidate_blind_delivery_formation_trace_r1"
)
BLIND_INVENTORY_SCHEMA = (
    "blindassist_ustrf_candidate_blind_delivery_formation_inventory_r1"
)
EVENT_SCOPE_BLIND_INVENTORY_SCHEMA = (
    "blindassist_ustrf_candidate_blind_event_scope_formation_inventory_r1"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_known_route_eligible_delivery_failure_attribution_terminal_r1"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_known_route_eligible_delivery_failure_attribution_validation_r1"
)
TERMINAL_STATE = "FAILURE_ATTRIBUTION_COMPLETE"
CANDIDATES = (
    "C1_CAUSAL_ROUTE_RELATION_FSM",
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
    "C3_DUAL_KEY_CLEARANCE_FSM",
)
LABELS = (
    "formed_eligible_delivery",
    "no_observation",
    "never_active_relation",
    "active_streak_lt_min_alert",
    "episode_opened_before_alertable_window",
    "episode_opened_inside_window_before_target_attribution",
    "pre_invalid_baseline_latch_guard_quarantine",
    "reset_split",
    "unexplained_gap",
)
FORBIDDEN_BLIND_KEYS = {
    "truth",
    "truth_status",
    "truth_clear",
    "truth_terminal_clear",
    "truth_terminal_clear_frame",
    "event_id",
    "eligibility",
    "metric_eligibility",
    "metric_classification",
    "scoring_label",
    "failure_label",
}
IMPLEMENTATION_PATHS = {
    "attribution_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "known_route_eligible_delivery_failure_attribution.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_known_route_eligible_delivery_failure_attribution.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_known_route_eligible_delivery_failure_attribution.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_known_route_eligible_delivery_failure_attribution.py"
    ),
    "truth_join_dependency_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "metric_profiles_r2_l1.py"
    ),
}


class AttributionContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionContractError(message)


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
    require(
        load_json(temporary) == value,
        f"atomic_write_verification_failed:{path}",
    )
    os.replace(temporary, path)


def _verify_binding(
    repo: Path, binding: dict[str, Any], label: str
) -> Path:
    require(
        set(binding) == {"path", "sha256"},
        f"{label}_binding_inventory_drift",
    )
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing:{binding['path']}")
    require(
        sha256_file(path) == binding["sha256"],
        f"{label}_sha256_drift",
    )
    return path


def load_and_verify_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(
        config.get("stage")
        == "KNOWN-ROUTE-ELIGIBLE-DELIVERY-FAILURE-ATTRIBUTION-R1",
        "config_stage_drift",
    )
    require(
        config.get("status")
        == "preregistered_candidate_blind_failure_attribution",
        "config_status_drift",
    )
    require(
        tuple(config.get("candidate_roster", [])) == CANDIDATES,
        "candidate_roster_drift",
    )
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
            "failure_taxonomy",
            "parent_bindings",
            "implementation_bindings",
            "evaluation",
            "expected_scope",
            "outputs",
        },
        "config_key_inventory_drift",
    )
    expected_taxonomy = {
        "labels": list(LABELS),
        "priority": [
            "formed_eligible_delivery",
            "no_observation",
            "never_active_relation",
            "pre_invalid_baseline_latch_guard_quarantine",
            "reset_split",
            "episode_opened_before_alertable_window",
            "episode_opened_inside_window_before_target_attribution",
            "active_streak_lt_min_alert",
            "unexplained_gap",
        ],
        "exactly_one_label_per_candidate_event_cell": True,
        "unexplained_gap_is_fail_closed": True,
    }
    require(
        config.get("failure_taxonomy") == expected_taxonomy,
        "failure_taxonomy_drift",
    )
    expected_counts = {
        "formed_eligible_delivery": 6,
        "no_observation": 0,
        "never_active_relation": 3,
        "active_streak_lt_min_alert": 0,
        "episode_opened_before_alertable_window": 9,
        "episode_opened_inside_window_before_target_attribution": 13,
        "pre_invalid_baseline_latch_guard_quarantine": 5,
        "reset_split": 0,
        "unexplained_gap": 0,
    }
    require(
        config.get("evaluation")
        == {
            "truth_attribution_minimum_iou": 0.3,
            "min_alert_frames": 2,
            "clearance_events_per_candidate": 12,
            "candidate_event_cells": 36,
            "unexplained_gap_max": 0,
            "expected_aggregate_counts": expected_counts,
        },
        "evaluation_contract_drift",
    )
    require(
        config.get("expected_scope")
        == {
            "baseline_authoritative_traces": 123,
            "guarded_lifecycle_traces": 123,
            "candidate_blind_traces": 123,
            "candidate_blind_frames": 186687,
            "candidate_count": 3,
            "clearance_events_per_candidate": 12,
            "candidate_event_cells": 36,
            "candidate_reruns": 0,
            "detector_reruns": 0,
            "route_reruns": 0,
            "new_data": 0,
        },
        "expected_scope_drift",
    )
    require(
        config.get("outputs")
        == {
            "root": (
                "artifacts.local/evidence/"
                "ustrf-known-route-eligible-delivery-failure-attribution-r1"
            ),
            "blind_inventory": "candidate-blind-inventory-r1.json",
            "event_scope_blind_inventory": (
                "event-scope-candidate-blind-inventory-r1.json"
            ),
            "terminal_receipt": "terminal-receipt-r1.json",
            "validation_receipt": "validation-receipt-r1.json",
        },
        "output_contract_drift",
    )
    expected_authority = {
        "maximum": "MECHANISM_DIAGNOSTIC_ONLY",
        "candidate_rerun": False,
        "detector_rerun": False,
        "route_rerun": False,
        "truth_available_to_blind_ledger": False,
        "candidate_comparison": False,
        "winner_or_ranking": False,
        "selection": False,
        "threshold_or_denominator_change": False,
        "guard_change": False,
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
    require(config.get("authority") == expected_authority, "authority_opened")
    require(
        config.get("phase_order")
        == [
            "verify_parent_receipts_and_all_123_trace_hashes",
            "construct_and_persist_all_123_candidate_blind_formation_trace_receipts",
            "persist_and_reverify_blind_trace_inventory",
            "load_12_event_clearance_mask_and_construct_event_scope_blind_traces",
            "hash_event_scope_blind_traces_before_truth_decode",
            "load_frozen_truth_payloads",
            "assign_exactly_one_failure_label_per_candidate_event_cell",
        ],
        "phase_order_drift",
    )
    bindings: dict[str, str] = {
        "config_sha256": sha256_file(config_path)
    }
    for name, binding in config["parent_bindings"].items():
        path = _verify_binding(repo, binding, f"parent_{name}")
        bindings[f"{name}_sha256"] = sha256_file(path)
    require(
        set(config["implementation_bindings"]) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_inventory_drift",
    )
    for name, relative in IMPLEMENTATION_PATHS.items():
        path = repo / relative
        require(path.is_file(), f"implementation_missing:{relative}")
        digest = sha256_file(path)
        require(
            digest == config["implementation_bindings"][name],
            f"{name}_drift",
        )
        bindings[name] = digest
    return config, bindings


def assert_blind(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(
                key not in FORBIDDEN_BLIND_KEYS,
                f"blind_trace_forbidden_field:{path}.{key}",
            )
            assert_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_blind(child, f"{path}[{index}]")


def _trace_maps(
    repo: Path, config: dict[str, Any]
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    str,
]:
    a2_terminal = load_json(
        repo / config["parent_bindings"]["candidate_replay_a2_terminal"]["path"]
    )
    guarded_terminal = load_json(
        repo
        / config["parent_bindings"]["route_invalid_terminal"]["path"]
    )
    require(
        a2_terminal.get("terminal_state") == "CANDIDATE_REPLAY_COMPLETE",
        "a2_terminal_not_complete",
    )
    require(
        guarded_terminal.get("terminal_state")
        == "MECHANISM_DIAGNOSTIC_COMPLETE",
        "route_invalid_terminal_not_complete",
    )
    baseline: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in a2_terminal["candidate_execution"]["trace_inventory"]:
        path = repo / row["trace_path"]
        require(path.is_file(), "baseline_trace_missing")
        require(sha256_file(path) == row["trace_sha256"], "baseline_trace_drift")
        key = (row["candidate_id"], row["source_id"], row["sequence_id"])
        require(key not in baseline, "duplicate_baseline_trace")
        baseline[key] = load_json(path)
    guarded: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in guarded_terminal["trace_inventory"]:
        path = repo / row["path"]
        require(path.is_file(), "guarded_trace_missing")
        require(sha256_file(path) == row["sha256"], "guarded_trace_drift")
        key = (row["candidate_id"], row["source_id"], row["sequence_id"])
        require(key not in guarded, "duplicate_guarded_trace")
        guarded[key] = load_json(path)
    require(set(baseline) == set(guarded), "parent_trace_key_mismatch")
    require(len(baseline) == 123, "parent_trace_count_drift")
    baseline_inventory = {
        (row["candidate_id"], row["source_id"], row["sequence_id"]): row
        for row in a2_terminal["candidate_execution"]["trace_inventory"]
    }
    guarded_inventory = {
        (row["candidate_id"], row["source_id"], row["sequence_id"]): row
        for row in guarded_terminal["trace_inventory"]
    }
    inventory = []
    for key in sorted(baseline):
        inventory.append(
            {
                "candidate_id": key[0],
                "source_id": key[1],
                "sequence_id": key[2],
                "baseline_frame_count": len(baseline[key]["frames"]),
                "guarded_frame_count": len(guarded[key]["frames"]),
                "baseline_trace_sha256": baseline_inventory[key][
                    "trace_sha256"
                ],
                "guarded_trace_sha256": guarded_inventory[key]["sha256"],
            }
        )
    return baseline, guarded, sha256_bytes(canonical_bytes(inventory))


def _active_rows(active: dict[int, int]) -> list[dict[str, int]]:
    return [
        {"local_key": key, "opening_frame": active[key]}
        for key in sorted(active)
    ]


def build_blind_trace(
    candidate_id: str,
    baseline: dict[str, Any],
    guarded: dict[str, Any],
) -> dict[str, Any]:
    require(
        len(baseline["frames"]) == len(guarded["frames"]),
        "blind_parent_frame_count_mismatch",
    )
    baseline_active: dict[int, int] = {}
    guarded_active: dict[int, int] = {}
    guarded_last_terminal: dict[int, str] = {}
    frames: list[dict[str, Any]] = []
    for base, guard in zip(
        baseline["frames"], guarded["frames"], strict=True
    ):
        require(identity(base) == identity(guard), "blind_parent_identity_mismatch")
        if bool(base["state_reset_before_frame"]):
            baseline_active.clear()
            guarded_active.clear()
            guarded_last_terminal.clear()
        baseline_active_before = dict(baseline_active)
        for raw_key in base["closures"]:
            baseline_active.pop(int(raw_key), None)
        delivery_rows = []
        for raw_key, track_ids in _delivery_groups(base):
            key = int(raw_key)
            baseline_active[key] = int(base["frame_id"])
            delivery_rows.append(
                {
                    "local_key": key,
                    "delivery_track_ids": sorted(
                        int(value) for value in track_ids
                    ),
                }
            )
        guard_events = []
        for event in guard["lifecycle_events"]:
            local_key = int(event["local_delivery_key"])
            if event["kind"] == "delivery":
                guarded_active[local_key] = int(base["frame_id"])
                guarded_last_terminal.pop(local_key, None)
                guard_events.append(
                    {
                        "kind": "delivery",
                        "local_key": local_key,
                        "scoped_episode_key": event["scoped_episode_key"],
                        "delivery_track_ids": sorted(
                            int(value)
                            for value in event["delivery_track_ids"]
                        ),
                    }
                )
            else:
                guarded_active.pop(local_key, None)
                guarded_last_terminal[local_key] = str(event["reason"])
                guard_events.append(
                    {
                        "kind": "closure",
                        "local_key": local_key,
                        "scoped_episode_key": event["scoped_episode_key"],
                        "reason": str(event["reason"]),
                    }
                )
        quarantined = sorted(
            key
            for key in baseline_active
            if key not in guarded_active
            and guarded_last_terminal.get(key) == "route_invalid"
        )
        row = {
            "source_id": str(base["source_id"]),
            "sequence_id": str(base["sequence_id"]),
            "frame_id": int(base["frame_id"]),
            "source_capture_timestamp_ns": int(
                base["source_capture_timestamp_ns"]
            ),
            "state_reset_before_frame": bool(
                base["state_reset_before_frame"]
            ),
            "reset_segment": int(guard["reset_segment"]),
            "route_known": bool(base["route_known"]),
            "observed_tracks": [
                {
                    "track_id": int(track["track_id"]),
                    "box": [float(value) for value in track["box"]],
                }
                for track in base["observed_tracks"]
            ],
            "active_relation_track_ids": sorted(
                int(value) for value in base["active_relation_track_ids"]
            ),
            "baseline_candidate_active_before": bool(baseline_active_before),
            "baseline_candidate_active_after": bool(base["candidate_active"]),
            "baseline_active_keys_after": _active_rows(baseline_active),
            "baseline_deliveries": delivery_rows,
            "baseline_closures": sorted(
                int(value) for value in base["closures"]
            ),
            "guarded_active_keys_after": _active_rows(guarded_active),
            "guard_events": guard_events,
            "route_invalid_quarantined_local_keys_after": quarantined,
        }
        assert_blind(row)
        frames.append(row)
    trace = {
        "schema": BLIND_TRACE_SCHEMA,
        "stage": "KNOWN-ROUTE-ELIGIBLE-DELIVERY-FAILURE-ATTRIBUTION-R1",
        "authority": "candidate_blind_formation_facts_only",
        "candidate_id": candidate_id,
        "source_id": baseline["source_id"],
        "sequence_id": baseline["sequence_id"],
        "frame_count": len(frames),
        "frames": frames,
    }
    assert_blind(trace)
    return trace


def build_blind_trace_receipt(
    candidate_id: str,
    baseline: dict[str, Any],
    guarded: dict[str, Any],
) -> dict[str, Any]:
    """Hash all blind formation facts without duplicating full frame payloads."""

    require(
        len(baseline["frames"]) == len(guarded["frames"]),
        "blind_parent_frame_count_mismatch",
    )
    digest = hashlib.sha256()
    counters: Counter[str] = Counter()
    for base, guard in zip(
        baseline["frames"], guarded["frames"], strict=True
    ):
        require(identity(base) == identity(guard), "blind_parent_identity_mismatch")
        fact = {
            "identity": list(identity(base)),
            "state_reset_before_frame": bool(
                base["state_reset_before_frame"]
            ),
            "reset_segment": int(guard["reset_segment"]),
            "route_known": bool(base["route_known"]),
            "observed_track_ids": sorted(
                int(track["track_id"]) for track in base["observed_tracks"]
            ),
            "active_relation_track_ids": sorted(
                int(value) for value in base["active_relation_track_ids"]
            ),
            "baseline_candidate_active": bool(base["candidate_active"]),
            "baseline_deliveries": [
                {
                    "local_key": int(key),
                    "delivery_track_ids": sorted(
                        int(value) for value in track_ids
                    ),
                }
                for key, track_ids in _delivery_groups(base)
            ],
            "baseline_closures": sorted(
                int(value) for value in base["closures"]
            ),
            "guard_events": guard["lifecycle_events"],
        }
        assert_blind(fact)
        digest.update(canonical_bytes(fact))
        counters["frames"] += 1
        counters["resets"] += int(bool(base["state_reset_before_frame"]))
        counters["route_known_frames"] += int(bool(base["route_known"]))
        counters["baseline_deliveries"] += len(base["deliveries"])
        counters["baseline_closures"] += len(base["closures"])
        counters["guard_deliveries"] += sum(
            event["kind"] == "delivery"
            for event in guard["lifecycle_events"]
        )
        counters["guard_closures"] += sum(
            event["kind"] == "closure"
            for event in guard["lifecycle_events"]
        )
    receipt = {
        "schema": BLIND_TRACE_SCHEMA,
        "stage": "KNOWN-ROUTE-ELIGIBLE-DELIVERY-FAILURE-ATTRIBUTION-R1",
        "authority": "candidate_blind_formation_fact_stream_receipt_only",
        "candidate_id": candidate_id,
        "source_id": baseline["source_id"],
        "sequence_id": baseline["sequence_id"],
        "frame_count": len(baseline["frames"]),
        "formation_fact_stream_sha256": digest.hexdigest(),
        "fact_counts": dict(sorted(counters.items())),
        "truth_payloads_decoded": 0,
        "clearance_mask_payload_decoded": False,
    }
    assert_blind(receipt)
    return receipt


def build_and_write_blind_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    baseline, guarded, parent_inventory_sha256 = _trace_maps(repo, config)
    output_root = repo / config["outputs"]["root"]
    inventory = []
    frame_count = 0
    for index, key in enumerate(sorted(baseline)):
        trace = build_blind_trace_receipt(
            key[0], baseline[key], guarded[key]
        )
        relative = (
            Path(config["outputs"]["root"])
            / "blind-trace-receipts"
            / key[0]
            / f"{index:03d}-{sha256_bytes(('::'.join(key)).encode())[:16]}.json"
        )
        path = repo / relative
        atomic_write_json(path, trace)
        frame_count += int(trace["frame_count"])
        inventory.append(
            {
                "candidate_id": key[0],
                "source_id": key[1],
                "sequence_id": key[2],
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "frame_count": int(trace["frame_count"]),
            }
        )
    receipt = {
        "schema": BLIND_INVENTORY_SCHEMA,
        "stage": config["stage"],
        "status": "CANDIDATE_BLIND_FORMATION_INVENTORY_COMPLETE",
        "authority": "candidate_blind_formation_facts_only",
        "bindings": bindings,
        "parent_trace_inventory_sha256": parent_inventory_sha256,
        "trace_count": len(inventory),
        "frame_count": frame_count,
        "truth_payloads_decoded": 0,
        "clearance_mask_payload_decoded": False,
        "inventory": inventory,
    }
    require(len(inventory) == 123, "blind_trace_count_drift")
    require(frame_count == 186687, "blind_frame_count_drift")
    atomic_write_json(
        output_root / config["outputs"]["blind_inventory"], receipt
    )
    return receipt


def load_and_verify_blind_inventory(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["blind_inventory"]
    )
    require(path.is_file(), "blind_inventory_missing")
    receipt = load_json(path)
    require(receipt.get("schema") == BLIND_INVENTORY_SCHEMA, "blind_schema_drift")
    require(
        receipt.get("truth_payloads_decoded") == 0,
        "blind_truth_decode_drift",
    )
    require(
        receipt.get("clearance_mask_payload_decoded") is False,
        "blind_clearance_mask_decode_drift",
    )
    require(receipt.get("trace_count") == 123, "blind_trace_count_drift")
    require(receipt.get("frame_count") == 186687, "blind_frame_count_drift")
    traces = {}
    for row in receipt["inventory"]:
        trace_path = repo / row["path"]
        require(trace_path.is_file(), "blind_trace_missing")
        require(sha256_file(trace_path) == row["sha256"], "blind_trace_sha_drift")
        trace_receipt = load_json(trace_path)
        assert_blind(trace_receipt)
        require(
            trace_receipt.get("frame_count") == row["frame_count"],
            "blind_trace_frame_count_drift",
        )
        key = (row["candidate_id"], row["source_id"], row["sequence_id"])
        require(key not in traces, "duplicate_blind_trace")
        traces[key] = trace_receipt
    require(len(traces) == 123, "blind_trace_inventory_drift")
    return receipt, traces


def _clearance_events(
    repo: Path, config: dict[str, Any]
) -> list[dict[str, Any]]:
    mask = load_json(
        repo / config["parent_bindings"]["eligibility_mask"]["path"]
    )
    events = [
        event
        for event in mask["events"]
        if event["metrics"]["clearance"]["classification"] == "eligible"
    ]
    require(len(events) == 12, "clearance_event_count_drift")
    return events


def build_and_write_event_scope_blind_pack(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    blind_receipt, trace_receipts = load_and_verify_blind_inventory(
        repo, config
    )
    require(
        blind_receipt.get("bindings") == bindings,
        "blind_inventory_binding_drift",
    )
    baseline, guarded, parent_inventory_sha256 = _trace_maps(repo, config)
    require(
        parent_inventory_sha256
        == blind_receipt["parent_trace_inventory_sha256"],
        "blind_parent_inventory_drift",
    )
    require(set(trace_receipts) == set(baseline), "blind_parent_key_drift")
    for key in sorted(baseline):
        require(
            trace_receipts[key]
            == build_blind_trace_receipt(
                key[0], baseline[key], guarded[key]
            ),
            f"blind_trace_receipt_not_exact_recomputation:{key!r}",
        )
    events = _clearance_events(repo, config)
    windows: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for candidate_id in CANDIDATES:
        for event in events:
            key = (
                candidate_id,
                event["source_id"],
                event["sequence_id"],
            )
            windows[key].update(
                range(
                    int(event["anchors"]["alertable_start_frame"]),
                    int(event["anchors"]["truth_terminal_clear_frame"]) + 1,
                )
            )
    output_root = repo / config["outputs"]["root"]
    inventory = []
    packed_frames = 0
    for index, key in enumerate(sorted(windows)):
        full = build_blind_trace(key[0], baseline[key], guarded[key])
        selected = [
            frame
            for frame in full["frames"]
            if int(frame["frame_id"]) in windows[key]
        ]
        packed = {
            **{name: value for name, value in full.items() if name != "frames"},
            "authority": "candidate_blind_event_scope_formation_fact_pack",
            "frame_count": len(selected),
            "frames": selected,
        }
        assert_blind(packed)
        relative = (
            Path(config["outputs"]["root"])
            / "event-scope-blind-traces"
            / key[0]
            / f"{index:02d}-{sha256_bytes(('::'.join(key)).encode())[:16]}.json"
        )
        path = repo / relative
        atomic_write_json(path, packed)
        packed_frames += len(selected)
        inventory.append(
            {
                "candidate_id": key[0],
                "source_id": key[1],
                "sequence_id": key[2],
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "frame_count": len(selected),
            }
        )
    receipt = {
        "schema": EVENT_SCOPE_BLIND_INVENTORY_SCHEMA,
        "stage": config["stage"],
        "status": "EVENT_SCOPE_CANDIDATE_BLIND_PACK_COMPLETE",
        "authority": "candidate_blind_event_scope_formation_facts_only",
        "bindings": bindings,
        "blind_inventory_sha256": sha256_file(
            output_root / config["outputs"]["blind_inventory"]
        ),
        "eligibility_mask_sha256": config["parent_bindings"][
            "eligibility_mask"
        ]["sha256"],
        "trace_count": len(inventory),
        "packed_frame_count": packed_frames,
        "truth_payloads_decoded": 0,
        "inventory": inventory,
    }
    require(len(inventory) == 12, "event_scope_trace_count_drift")
    atomic_write_json(
        output_root / config["outputs"]["event_scope_blind_inventory"],
        receipt,
    )
    return receipt


def load_and_verify_event_scope_blind_pack(
    repo: Path, config: dict[str, Any], bindings: dict[str, str]
) -> tuple[
    dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]
]:
    output_root = repo / config["outputs"]["root"]
    path = output_root / config["outputs"]["event_scope_blind_inventory"]
    require(path.is_file(), "event_scope_blind_inventory_missing")
    receipt = load_json(path)
    require(
        receipt.get("schema") == EVENT_SCOPE_BLIND_INVENTORY_SCHEMA,
        "event_scope_blind_inventory_schema_drift",
    )
    require(
        receipt.get("bindings") == bindings,
        "event_scope_blind_binding_drift",
    )
    require(
        receipt.get("blind_inventory_sha256")
        == sha256_file(output_root / config["outputs"]["blind_inventory"]),
        "event_scope_parent_blind_inventory_drift",
    )
    require(
        receipt.get("truth_payloads_decoded") == 0,
        "event_scope_truth_decode_drift",
    )
    require(receipt.get("trace_count") == 12, "event_scope_count_drift")
    traces = {}
    for row in receipt["inventory"]:
        trace_path = repo / row["path"]
        require(trace_path.is_file(), "event_scope_blind_trace_missing")
        require(
            sha256_file(trace_path) == row["sha256"],
            "event_scope_blind_trace_sha_drift",
        )
        trace = load_json(trace_path)
        assert_blind(trace)
        require(
            trace["frame_count"] == row["frame_count"],
            "event_scope_blind_trace_frame_count_drift",
        )
        key = (row["candidate_id"], row["source_id"], row["sequence_id"])
        require(key not in traces, "duplicate_event_scope_blind_trace")
        traces[key] = trace
    return receipt, traces


def _event_tracks(
    frame: dict[str, Any],
    event_id: str,
    track_ids: list[int],
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
) -> list[int]:
    return [
        int(track_id)
        for track_id in track_ids
        if event_id
        in _attributed_event_ids_for_track_ids(
            frame,
            truth_index.get(identity(frame), []),
            [int(track_id)],
            minimum_iou,
        )
    ]


def _opening_map(frame: dict[str, Any]) -> dict[int, int]:
    return {
        int(row["local_key"]): int(row["opening_frame"])
        for row in frame["baseline_active_keys_after"]
    }


def _qualifying_support(
    candidate_id: str,
    frames: list[dict[str, Any]],
    event_id: str,
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
    min_alert: int,
) -> dict[str, int] | None:
    if candidate_id == CANDIDATES[0]:
        runs: dict[int, int] = defaultdict(int)
        starts: dict[int, int] = {}
        for frame in frames:
            matched = set(
                _event_tracks(
                    frame,
                    event_id,
                    frame["active_relation_track_ids"],
                    truth_index,
                    minimum_iou,
                )
                if frame["route_known"]
                else []
            )
            if frame["state_reset_before_frame"]:
                runs.clear()
                starts.clear()
            for track_id in list(runs):
                if track_id not in matched:
                    runs[track_id] = 0
                    starts.pop(track_id, None)
            for track_id in matched:
                if runs[track_id] == 0:
                    starts[track_id] = int(frame["frame_id"])
                runs[track_id] += 1
                if runs[track_id] >= min_alert:
                    return {
                        "qualification_frame": int(frame["frame_id"]),
                        "support_start_frame": starts[track_id],
                        "local_key": track_id,
                        "run_length": runs[track_id],
                    }
        return None
    run = 0
    start_frame: int | None = None
    for frame in frames:
        risk = bool(
            frame["route_known"]
            and frame["active_relation_track_ids"]
        )
        if frame["state_reset_before_frame"]:
            run = 0
            start_frame = None
        if risk:
            if run == 0:
                start_frame = int(frame["frame_id"])
            run += 1
            if run >= min_alert:
                active = _opening_map(frame)
                local_key = min(
                    active,
                    key=lambda key: (active[key], key),
                )
                return {
                    "qualification_frame": int(frame["frame_id"]),
                    "support_start_frame": int(start_frame),
                    "local_key": int(local_key),
                    "run_length": run,
                }
        else:
            run = 0
            start_frame = None
    return None


def _reset_split(
    candidate_id: str,
    frames: list[dict[str, Any]],
    event_id: str,
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
    min_alert: int,
) -> bool:
    if not any(frame["state_reset_before_frame"] for frame in frames):
        return False
    without_reset = [
        {**frame, "state_reset_before_frame": False} for frame in frames
    ]
    return (
        _qualifying_support(
            candidate_id,
            without_reset,
            event_id,
            truth_index,
            minimum_iou,
            min_alert,
        )
        is not None
    )


def _formed_delivery_frames(
    frames: list[dict[str, Any]],
    event_id: str,
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
) -> list[int]:
    result = []
    for frame in frames:
        for event in frame["guard_events"]:
            if event["kind"] != "delivery":
                continue
            if _event_tracks(
                frame,
                event_id,
                event["delivery_track_ids"],
                truth_index,
                minimum_iou,
            ):
                result.append(int(frame["frame_id"]))
    return result


def attribute_event(
    *,
    candidate_id: str,
    event: dict[str, Any],
    trace: dict[str, Any],
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    minimum_iou: float,
    min_alert: int,
) -> dict[str, Any]:
    start = int(event["anchors"]["alertable_start_frame"])
    clear = int(event["anchors"]["truth_terminal_clear_frame"])
    frames = [
        frame
        for frame in trace["frames"]
        if start <= int(frame["frame_id"]) <= clear
    ]
    require(
        [int(frame["frame_id"]) for frame in frames]
        == list(range(start, clear + 1)),
        "event_window_frame_gap",
    )
    deliveries = _formed_delivery_frames(
        frames, event["event_id"], truth_index, minimum_iou
    )
    observation_frames = [
        int(frame["frame_id"])
        for frame in frames
        if _event_tracks(
            frame,
            event["event_id"],
            [int(track["track_id"]) for track in frame["observed_tracks"]],
            truth_index,
            minimum_iou,
        )
    ]
    active_frames = [
        int(frame["frame_id"])
        for frame in frames
        if _event_tracks(
            frame,
            event["event_id"],
            frame["active_relation_track_ids"],
            truth_index,
            minimum_iou,
        )
    ]
    base = {
        "candidate_id": candidate_id,
        "event_id": event["event_id"],
        "source_id": event["source_id"],
        "sequence_id": event["sequence_id"],
        "alertable_start_frame": start,
        "truth_terminal_clear_frame": clear,
        "observation_frame_count": len(observation_frames),
        "active_relation_frame_count": len(active_frames),
        "first_observation_frame": (
            observation_frames[0] if observation_frames else None
        ),
        "first_active_relation_frame": (
            active_frames[0] if active_frames else None
        ),
        "eligible_delivery_frames": deliveries,
        "qualification": None,
        "episode_opening_frame": None,
        "failure_label": None,
    }
    if deliveries:
        return {**base, "failure_label": "formed_eligible_delivery"}
    if not observation_frames:
        return {**base, "failure_label": "no_observation"}
    if not active_frames:
        return {**base, "failure_label": "never_active_relation"}
    quarantine = False
    for frame in frames:
        matched = _event_tracks(
            frame,
            event["event_id"],
            frame["active_relation_track_ids"],
            truth_index,
            minimum_iou,
        )
        quarantined = set(
            int(value)
            for value in frame[
                "route_invalid_quarantined_local_keys_after"
            ]
        )
        if candidate_id == CANDIDATES[0]:
            quarantine = quarantine or bool(quarantined & set(matched))
        else:
            quarantine = quarantine or bool(quarantined and matched)
    qualification = _qualifying_support(
        candidate_id,
        frames,
        event["event_id"],
        truth_index,
        minimum_iou,
        min_alert,
    )
    if quarantine:
        return {
            **base,
            "qualification": qualification,
            "failure_label": "pre_invalid_baseline_latch_guard_quarantine",
        }
    if qualification is None:
        label = (
            "reset_split"
            if _reset_split(
                candidate_id,
                frames,
                event["event_id"],
                truth_index,
                minimum_iou,
                min_alert,
            )
            else "active_streak_lt_min_alert"
        )
        return {**base, "failure_label": label}
    if candidate_id == CANDIDATES[0]:
        opening_frame_state = next(
            frame
            for frame in frames
            if int(frame["frame_id"])
            == qualification["qualification_frame"]
        )
        opening = _opening_map(opening_frame_state).get(
            int(qualification["local_key"])
        )
        attribution_start = int(qualification["support_start_frame"])
    else:
        # C2/C3 use one route-occupancy episode.  The causal carry-in is
        # therefore the episode already open at the target's first active
        # support, even if that episode closes and a second one opens before
        # the target later reaches the two-frame qualification run.
        opening_frame_state = next(
            frame
            for frame in frames
            if int(frame["frame_id"]) == active_frames[0]
        )
        openings = _opening_map(opening_frame_state)
        opening = (
            min(openings.values()) if openings else None
        )
        attribution_start = active_frames[0]
    if opening is None:
        return {
            **base,
            "qualification": qualification,
            "failure_label": "unexplained_gap",
        }
    if opening < start:
        label = "episode_opened_before_alertable_window"
    elif opening < attribution_start:
        label = "episode_opened_inside_window_before_target_attribution"
    else:
        label = "unexplained_gap"
    return {
        **base,
        "qualification": qualification,
        "episode_opening_frame": opening,
        "failure_label": label,
    }


def build_terminal_from_persisted_blind(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    blind_receipt, _ = load_and_verify_blind_inventory(repo, config)
    require(blind_receipt.get("bindings") == bindings, "blind_binding_drift")
    event_scope_receipt, blind_trace_cache = (
        load_and_verify_event_scope_blind_pack(repo, config, bindings)
    )
    events = _clearance_events(repo, config)
    expected_event_scope_keys = {
        (
            candidate_id,
            event["source_id"],
            event["sequence_id"],
        )
        for candidate_id in CANDIDATES
        for event in events
    }
    require(
        set(blind_trace_cache) == expected_event_scope_keys,
        "event_scope_blind_trace_key_drift",
    )
    # Truth payloads are decoded only after the persisted fact pack that the
    # attribution below consumes has been loaded and hash-verified.
    truth_config = load_json(
        repo / config["parent_bindings"]["metric_profile_config"]["path"]
    )
    truth_index = load_truth_frame_index(truth_config, repo)
    minimum_iou = float(
        config["evaluation"]["truth_attribution_minimum_iou"]
    )
    min_alert = int(config["evaluation"]["min_alert_frames"])
    candidate_results = []
    aggregate: Counter[str] = Counter()
    event_rows = []
    for candidate_id in CANDIDATES:
        rows = []
        for event in events:
            key = (
                candidate_id,
                event["source_id"],
                event["sequence_id"],
            )
            require(
                key in blind_trace_cache,
                "event_scope_blind_trace_missing",
            )
            row = attribute_event(
                candidate_id=candidate_id,
                event=event,
                trace=blind_trace_cache[key],
                truth_index=truth_index,
                minimum_iou=minimum_iou,
                min_alert=min_alert,
            )
            rows.append(row)
            event_rows.append(row)
        counts = Counter(row["failure_label"] for row in rows)
        aggregate.update(counts)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "cell_count": len(rows),
                "label_counts": {
                    label: counts.get(label, 0) for label in LABELS
                },
                "events": rows,
            }
        )
    require(len(event_rows) == 36, "candidate_event_cell_count_drift")
    require(
        all(row["failure_label"] in LABELS for row in event_rows),
        "unclassified_candidate_event_cell",
    )
    expected = config["evaluation"]["expected_aggregate_counts"]
    observed = {label: aggregate.get(label, 0) for label in LABELS}
    require(
        observed == expected,
        f"aggregate_attribution_count_drift:{observed!r}",
    )
    unexplained = aggregate.get("unexplained_gap", 0)
    receipt = {
        "schema": TERMINAL_SCHEMA,
        "stage": config["stage"],
        "terminal_state": TERMINAL_STATE,
        "authority": "mechanism_diagnostic_only_no_candidate_comparison",
        "bindings": bindings,
        "blind_phase": {
            "inventory_path": (
                f"{config['outputs']['root']}/"
                f"{config['outputs']['blind_inventory']}"
            ),
            "inventory_sha256": sha256_file(
                repo
                / config["outputs"]["root"]
                / config["outputs"]["blind_inventory"]
            ),
            "trace_count": blind_receipt["trace_count"],
            "frame_count": blind_receipt["frame_count"],
            "truth_payloads_decoded_before_inventory_freeze": 0,
            "clearance_mask_payload_decoded_before_inventory_freeze": False,
            "event_scope_blind_trace_count": len(blind_trace_cache),
            "event_scope_blind_trace_inventory_sha256": (
                sha256_file(
                    repo
                    / config["outputs"]["root"]
                    / config["outputs"]["event_scope_blind_inventory"]
                )
            ),
            "event_scope_packed_frame_count": event_scope_receipt[
                "packed_frame_count"
            ],
            "truth_payloads_decoded_after_persisted_pack_verification": True,
        },
        "verified_scope": {
            "candidate_count": 3,
            "clearance_events_per_candidate": 12,
            "candidate_event_cells": 36,
            "formed_eligible_delivery_cells": aggregate.get(
                "formed_eligible_delivery", 0
            ),
            "no_eligible_delivery_cells": 36
            - aggregate.get("formed_eligible_delivery", 0),
            "candidate_reruns": 0,
            "detector_reruns": 0,
            "route_reruns": 0,
            "threshold_changes": 0,
            "guard_changes": 0,
            "new_data": 0,
        },
        "failure_taxonomy": {
            "labels": list(LABELS),
            "exactly_one_label_per_cell": True,
            "priority": config["failure_taxonomy"]["priority"],
        },
        "aggregate_label_counts": observed,
        "candidate_results": candidate_results,
        "attribution_gate": {
            "all_36_cells_classified": True,
            "no_delivery_cells_explained": unexplained == 0,
            "unexplained_gap_count": unexplained,
            "passed": unexplained == 0,
        },
        "claim_boundary": {
            "candidate_comparison": False,
            "winner_or_ranking": False,
            "selection": False,
            "threshold_or_denominator_change": False,
            "guard_or_clearance_change": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
        },
    }
    return receipt


def validate_outputs(
    repo: Path, config_path: Path, terminal_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    blind_receipt, trace_receipts = load_and_verify_blind_inventory(
        repo, config
    )
    require(
        blind_receipt.get("bindings") == bindings,
        "validation_blind_binding_drift",
    )
    baseline, guarded, parent_inventory_sha256 = _trace_maps(repo, config)
    require(
        parent_inventory_sha256
        == blind_receipt["parent_trace_inventory_sha256"],
        "validation_parent_inventory_drift",
    )
    for key in sorted(baseline):
        require(
            trace_receipts[key]
            == build_blind_trace_receipt(
                key[0], baseline[key], guarded[key]
            ),
            f"validation_blind_receipt_recompute_drift:{key!r}",
        )
    _, packed_traces = load_and_verify_event_scope_blind_pack(
        repo, config, bindings
    )
    events = _clearance_events(repo, config)
    expected_frames: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for candidate_id in CANDIDATES:
        for event in events:
            key = (
                candidate_id,
                event["source_id"],
                event["sequence_id"],
            )
            expected_frames[key].update(
                range(
                    int(event["anchors"]["alertable_start_frame"]),
                    int(event["anchors"]["truth_terminal_clear_frame"]) + 1,
                )
            )
    require(
        set(packed_traces) == set(expected_frames),
        "validation_event_scope_key_drift",
    )
    for key in sorted(expected_frames):
        full = build_blind_trace(key[0], baseline[key], guarded[key])
        expected_pack = {
            **{
                name: value
                for name, value in full.items()
                if name != "frames"
            },
            "authority": "candidate_blind_event_scope_formation_fact_pack",
            "frame_count": len(expected_frames[key]),
            "frames": [
                frame
                for frame in full["frames"]
                if int(frame["frame_id"]) in expected_frames[key]
            ],
        }
        require(
            packed_traces[key] == expected_pack,
            f"validation_event_scope_pack_recompute_drift:{key!r}",
        )
    observed = load_json(terminal_path)
    expected = build_terminal_from_persisted_blind(repo, config_path)
    require(
        observed == expected,
        "terminal_receipt_not_exact_canonical_recomputation",
    )
    require(
        observed["attribution_gate"]["passed"] is True,
        "attribution_gate_not_passed",
    )
    require(
        observed["aggregate_label_counts"]
        == config["evaluation"]["expected_aggregate_counts"],
        "expected_aggregate_counts_drift",
    )
    return {
        "schema": VALIDATION_SCHEMA,
        "stage": config["stage"],
        "status": "VALID",
        "terminal_state": TERMINAL_STATE,
        "blind_traces_reverified": 123,
        "blind_frames_reverified": 186687,
        "event_scope_blind_traces_recomputed": 12,
        "candidate_event_cells_recomputed": 36,
        "unexplained_gap_count": 0,
        "validator_mode": "separate_process_exact_recomputation_shared_audited_core",
        "terminal_receipt_sha256": sha256_file(terminal_path),
    }
