#!/usr/bin/env python3
"""Build and audit truth-blind causal per-track attribution tokens.

Phase 1 accepts only full-sequence detector/T0/route/reset facts.  It persists
and hash-inventories candidate-independent per-sequence ledgers before phase 2
loads any oracle token or negative-exposure evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

import known_route_eligible_delivery_failure_attribution as parent  # noqa: E402

CONFIG_SCHEMA = (
    "blindassist_ustrf_truth_blind_causal_per_track_attribution_token_"
    "producer_audit_r0"
)
LEDGER_SCHEMA = (
    "blindassist_ustrf_truth_blind_causal_per_track_token_ledger_r0"
)
INVENTORY_SCHEMA = (
    "blindassist_ustrf_truth_blind_causal_per_track_token_inventory_r0"
)
EXTRA_SCHEMA = "blindassist_ustrf_causal_extra_token_audit_ledger_r0"
REPEAT_SCHEMA = (
    "blindassist_ustrf_causal_repeat_activation_audit_ledger_r0"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_truth_blind_causal_per_track_token_audit_terminal_r0"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_truth_blind_causal_per_track_token_audit_validation_r0"
)
CANDIDATES = parent.CANDIDATES

FORBIDDEN_INPUT_KEYS = {
    "event_id",
    "event_ids",
    "truth",
    "truth_box",
    "truth_boxes",
    "truth_status",
    "alertable_window",
    "alertable_start",
    "alertable_start_frame",
    "clearance",
    "truth_clear",
    "truth_terminal_clear",
    "truth_terminal_clear_frame",
    "future",
    "future_frames",
    "oracle",
    "oracle_token",
    "oracle_tokens",
    "candidate_id",
}
ALLOWED_FRAME_KEYS = {
    "source_id",
    "sequence_id",
    "frame_id",
    "source_capture_timestamp_ns",
    "reset_segment",
    "state_reset_before_frame",
    "route_known",
    "observed_track_ids",
    "active_relation_track_ids",
}
IMPLEMENTATION_PATHS = {
    "producer_audit_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "causal_per_track_attribution_token_audit_r0.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_causal_per_track_attribution_token_audit_r0.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_causal_per_track_attribution_token_audit_r0.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_causal_per_track_attribution_token_audit_r0.py"
    ),
}


class CausalTokenContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalTokenContractError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


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
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def _assert_no_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(
                key not in FORBIDDEN_INPUT_KEYS,
                f"producer_forbidden_input:{path}.{key}",
            )
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def assert_producer_frame(frame: dict[str, Any]) -> None:
    _assert_no_forbidden_keys(frame)
    require(set(frame) == ALLOWED_FRAME_KEYS, "producer_frame_keys_drift")
    require(
        isinstance(frame["source_id"], str)
        and isinstance(frame["sequence_id"], str),
        "producer_frame_identity_invalid",
    )
    require(
        isinstance(frame["frame_id"], int)
        and isinstance(frame["source_capture_timestamp_ns"], int)
        and isinstance(frame["reset_segment"], int),
        "producer_frame_numeric_identity_invalid",
    )
    require(
        isinstance(frame["state_reset_before_frame"], bool)
        and isinstance(frame["route_known"], bool),
        "producer_frame_boolean_invalid",
    )
    observed = frame["observed_track_ids"]
    active = frame["active_relation_track_ids"]
    require(
        observed == sorted(set(observed))
        and active == sorted(set(active)),
        "producer_track_ids_not_unique_sorted",
    )
    require(
        all(isinstance(value, int) for value in observed + active),
        "producer_track_id_invalid",
    )
    require(set(active).issubset(observed), "active_relation_track_unobserved")
    require(
        frame["route_known"] or not active,
        "active_relation_on_unknown_route",
    )


def project_allowed_frame(frame: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "source_id": str(frame["source_id"]),
        "sequence_id": str(frame["sequence_id"]),
        "frame_id": int(frame["frame_id"]),
        "source_capture_timestamp_ns": int(
            frame["source_capture_timestamp_ns"]
        ),
        "reset_segment": int(frame["reset_segment"]),
        "state_reset_before_frame": bool(frame["state_reset_before_frame"]),
        "route_known": bool(frame["route_known"]),
        "observed_track_ids": sorted(
            int(track["track_id"]) for track in frame["observed_tracks"]
        ),
        "active_relation_track_ids": sorted(
            int(value) for value in frame["active_relation_track_ids"]
        ),
    }
    assert_producer_frame(projected)
    return projected


def collapse_candidate_projections(
    traces: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[
        tuple[str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(dict)
    for (candidate_id, source_id, sequence_id), trace in traces.items():
        require(candidate_id in CANDIDATES, "producer_candidate_unknown")
        require(
            candidate_id not in grouped[(source_id, sequence_id)],
            "producer_candidate_projection_duplicate",
        )
        grouped[(source_id, sequence_id)][candidate_id] = [
            project_allowed_frame(frame) for frame in trace["frames"]
        ]
    collapsed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in sorted(grouped):
        projections = grouped[key]
        require(
            set(projections) == set(CANDIDATES),
            "producer_candidate_projection_roster_drift",
        )
        reference = projections[CANDIDATES[0]]
        for candidate_id in CANDIDATES[1:]:
            require(
                projections[candidate_id] == reference,
                f"producer_candidate_projection_mismatch:{key!r}:"
                f"{candidate_id}",
            )
        collapsed[key] = reference
    return collapsed


def _token_id(
    source_id: str,
    sequence_id: str,
    reset_segment: int,
    track_id: int,
) -> str:
    return (
        f"{source_id}::{sequence_id}::reset-{reset_segment}::"
        f"track-{track_id}"
    )


def produce_sequence_ledger(
    source_id: str,
    sequence_id: str,
    frames: list[dict[str, Any]],
    *,
    min_consecutive_relation_frames: int,
) -> dict[str, Any]:
    require(min_consecutive_relation_frames >= 1, "producer_min_streak_invalid")
    require(bool(frames), "producer_sequence_empty")
    streaks: dict[int, int] = {}
    emitted: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    token_count = 0
    repeat_count = 0
    previous_frame_id: int | None = None
    previous_timestamp: int | None = None
    current_segment: int | None = None
    for index, frame in enumerate(frames):
        assert_producer_frame(frame)
        require(
            frame["source_id"] == source_id
            and frame["sequence_id"] == sequence_id,
            "producer_sequence_identity_drift",
        )
        if index == 0:
            require(
                frame["state_reset_before_frame"],
                "producer_first_frame_missing_reset",
            )
        if frame["state_reset_before_frame"]:
            streaks.clear()
            emitted.clear()
            current_segment = frame["reset_segment"]
        else:
            require(current_segment is not None, "producer_reset_state_missing")
            require(
                frame["reset_segment"] == current_segment,
                "producer_segment_changed_without_reset",
            )
            require(
                previous_frame_id is not None
                and frame["frame_id"] == previous_frame_id + 1,
                "producer_frame_gap_without_reset",
            )
        require(
            previous_timestamp is None
            or frame["source_capture_timestamp_ns"] > previous_timestamp,
            "producer_timestamp_not_strictly_increasing",
        )
        active = (
            set(frame["active_relation_track_ids"])
            if frame["route_known"]
            else set()
        )
        for track_id in list(streaks):
            if track_id not in active:
                del streaks[track_id]
        activations: list[dict[str, Any]] = []
        repeats: list[dict[str, Any]] = []
        for track_id in sorted(active):
            streaks[track_id] = streaks.get(track_id, 0) + 1
            if streaks[track_id] != min_consecutive_relation_frames:
                continue
            activation = {
                "token_id": _token_id(
                    source_id,
                    sequence_id,
                    int(frame["reset_segment"]),
                    track_id,
                ),
                "track_id": track_id,
                "reset_segment": int(frame["reset_segment"]),
                "qualification_frame_id": int(frame["frame_id"]),
                "qualification_timestamp_ns": int(
                    frame["source_capture_timestamp_ns"]
                ),
                "support_start_frame_id": int(frame["frame_id"])
                - min_consecutive_relation_frames
                + 1,
            }
            if track_id in emitted:
                repeat_count += 1
                repeats.append(
                    {
                        **activation,
                        "original_token_id": emitted[track_id]["token_id"],
                        "original_qualification_frame_id": emitted[track_id][
                            "qualification_frame_id"
                        ],
                    }
                )
            else:
                emitted[track_id] = activation
                token_count += 1
                activations.append(activation)
        rows.append(
            {
                **frame,
                "token_activations": activations,
                "repeat_activations_suppressed": repeats,
            }
        )
        previous_frame_id = int(frame["frame_id"])
        previous_timestamp = int(frame["source_capture_timestamp_ns"])
    return {
        "schema": LEDGER_SCHEMA,
        "stage": "TRUTH-BLIND-CAUSAL-PER-TRACK-TOKEN-PRODUCER-AUDIT-R0",
        "authority": "truth_blind_runtime_feasibility_audit_only",
        "source_id": source_id,
        "sequence_id": sequence_id,
        "frame_count": len(rows),
        "token_count": token_count,
        "repeat_activation_count": repeat_count,
        "min_consecutive_relation_frames": min_consecutive_relation_frames,
        "frames": rows,
    }


def _verify_binding(
    repo: Path, row: dict[str, Any], label: str
) -> tuple[Path, str]:
    require(set(row) == {"path", "sha256"}, f"{label}_binding_keys_drift")
    expected = str(row["sha256"])
    require(len(expected) == 64, f"{label}_sha_not_frozen")
    path = repo / row["path"]
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == expected, f"{label}_sha_drift")
    return path, expected


def load_and_verify_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(
        config.get("status") == "frozen_before_any_r0_output",
        "config_not_preregistered",
    )
    bindings: dict[str, str] = {}
    for label, row in config["parent_bindings"].items():
        _, bindings[label] = _verify_binding(repo, row, label)
    for label, relative in IMPLEMENTATION_PATHS.items():
        expected = config["implementation_bindings"].get(label)
        require(
            isinstance(expected, str) and len(expected) == 64,
            f"{label}_not_frozen",
        )
        path = repo / relative
        require(path.is_file(), f"{label}_missing")
        require(sha256_file(path) == expected, f"{label}_sha_drift")
        bindings[label] = expected
    require(
        config["producer"]["allowed_inputs"]
        == [
            "current_and_historical_detector_track_ids",
            "t0_track_identity",
            "current_and_historical_route_relation",
            "route_validity",
            "reset",
        ],
        "producer_allowed_inputs_drift",
    )
    require(
        config["producer"]["forbidden_inputs"]
        == [
            "event_id",
            "truth_box_or_truth_identity",
            "alertable_or_event_window",
            "future_frames",
            "clearance",
            "oracle_token",
            "candidate_identity",
        ],
        "producer_forbidden_inputs_drift",
    )
    require(
        config["claim_boundary"]["maximum"]
        == "TRUTH_BLIND_CAUSAL_TOKEN_FEASIBILITY_AND_EXTRA_TOKEN_AUDIT_ONLY",
        "claim_boundary_drift",
    )
    return config, bindings


def _blind_parent_traces(
    repo: Path, config: dict[str, Any]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    failure_config_path = (
        repo / config["parent_bindings"]["failure_attribution_config"]["path"]
    )
    failure_config, _ = parent.load_and_verify_config(
        repo, failure_config_path
    )
    baseline, guarded, _ = parent._trace_maps(repo, failure_config)
    require(set(baseline) == set(guarded), "producer_parent_trace_key_drift")
    traces = {
        key: parent.build_blind_trace(key[0], baseline[key], guarded[key])
        for key in sorted(baseline)
    }
    return traces


def _ledger_relative_path(
    config: dict[str, Any], source_id: str, sequence_id: str
) -> Path:
    digest = sha256_bytes(f"{source_id}::{sequence_id}".encode())[:16]
    return (
        Path(config["outputs"]["root"])
        / "causal-token-ledgers"
        / f"{source_id}__{digest}.json"
    )


def build_and_freeze_blind_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    traces = _blind_parent_traces(repo, config)
    collapsed = collapse_candidate_projections(traces)
    expected = config["expected_scope"]
    require(
        len(traces) == expected["candidate_projection_traces"],
        "producer_candidate_projection_count_drift",
    )
    require(
        len(collapsed) == expected["candidate_independent_sequences"],
        "producer_candidate_independent_sequence_count_drift",
    )
    inventory = []
    total_frames = 0
    total_tokens = 0
    total_repeats = 0
    for (source_id, sequence_id), frames in collapsed.items():
        ledger = produce_sequence_ledger(
            source_id,
            sequence_id,
            frames,
            min_consecutive_relation_frames=int(
                config["producer"]["min_consecutive_relation_frames"]
            ),
        )
        relative = _ledger_relative_path(config, source_id, sequence_id)
        path = repo / relative
        atomic_write_json(path, ledger)
        inventory.append(
            {
                "source_id": source_id,
                "sequence_id": sequence_id,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "frame_count": ledger["frame_count"],
                "token_count": ledger["token_count"],
                "repeat_activation_count": ledger[
                    "repeat_activation_count"
                ],
            }
        )
        total_frames += ledger["frame_count"]
        total_tokens += ledger["token_count"]
        total_repeats += ledger["repeat_activation_count"]
    require(
        len(inventory) == expected["sequence_ledgers"],
        "producer_sequence_ledger_count_drift",
    )
    require(total_frames == expected["frames"], "producer_frame_count_drift")
    receipt = {
        "schema": INVENTORY_SCHEMA,
        "stage": config["stage"],
        "status": "FULL_SEQUENCE_TRUTH_BLIND_TOKEN_INVENTORY_FROZEN",
        "authority": "truth_blind_runtime_feasibility_audit_only",
        "bindings": bindings,
        "candidate_projection_count_verified": len(traces),
        "candidate_independent_sequence_count": len(inventory),
        "frame_count": total_frames,
        "token_count": total_tokens,
        "repeat_activation_count": total_repeats,
        "truth_payloads_decoded": 0,
        "event_windows_decoded": 0,
        "oracle_tokens_decoded": 0,
        "inventory": inventory,
    }
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["token_inventory"]
    )
    atomic_write_json(path, receipt)
    return receipt


def load_and_verify_blind_inventory(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["token_inventory"]
    )
    require(path.is_file(), "producer_inventory_missing")
    inventory = load_json(path)
    require(
        inventory.get("schema") == INVENTORY_SCHEMA,
        "producer_inventory_schema_drift",
    )
    require(
        inventory.get("status")
        == "FULL_SEQUENCE_TRUTH_BLIND_TOKEN_INVENTORY_FROZEN",
        "producer_inventory_not_frozen",
    )
    require(
        inventory.get("bindings") == bindings,
        "producer_inventory_binding_drift",
    )
    require(
        inventory.get("truth_payloads_decoded") == 0
        and inventory.get("event_windows_decoded") == 0
        and inventory.get("oracle_tokens_decoded") == 0,
        "producer_inventory_truth_ordering_violation",
    )
    ledgers = {}
    token_ids: set[str] = set()
    frames = tokens = repeats = 0
    for row in inventory["inventory"]:
        ledger_path = repo / row["path"]
        require(ledger_path.is_file(), "producer_ledger_missing")
        require(
            sha256_file(ledger_path) == row["sha256"],
            "producer_ledger_sha_drift",
        )
        ledger = load_json(ledger_path)
        key = (ledger["source_id"], ledger["sequence_id"])
        require(key not in ledgers, "producer_ledger_duplicate")
        ledger_token_count = 0
        ledger_repeat_count = 0
        for frame in ledger["frames"]:
            for activation in frame["token_activations"]:
                require(
                    activation["token_id"] not in token_ids,
                    "producer_duplicate_token_id",
                )
                token_ids.add(activation["token_id"])
                ledger_token_count += 1
            ledger_repeat_count += len(
                frame["repeat_activations_suppressed"]
            )
        require(
            ledger_token_count == ledger["token_count"] == row["token_count"],
            "producer_ledger_token_count_drift",
        )
        require(
            ledger_repeat_count
            == ledger["repeat_activation_count"]
            == row["repeat_activation_count"],
            "producer_ledger_repeat_count_drift",
        )
        frames += ledger["frame_count"]
        tokens += ledger_token_count
        repeats += ledger_repeat_count
        ledgers[key] = ledger
    require(frames == inventory["frame_count"], "producer_inventory_frame_drift")
    require(tokens == inventory["token_count"], "producer_inventory_token_drift")
    require(
        repeats == inventory["repeat_activation_count"],
        "producer_inventory_repeat_drift",
    )
    return inventory, ledgers


def _producer_tokens(
    ledgers: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    tokens = {}
    for (source_id, sequence_id), ledger in ledgers.items():
        for frame in ledger["frames"]:
            for activation in frame["token_activations"]:
                token = {
                    **activation,
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                }
                require(token["token_id"] not in tokens, "duplicate_token_id")
                tokens[token["token_id"]] = token
    return tokens


def _oracle_qualifications(
    ledger: dict[str, Any]
) -> list[dict[str, int]]:
    qualifications = []
    prior_tracks: set[int] = set()
    prior_segment: int | None = None
    for frame in ledger["frames"]:
        current = set(frame["eligible_attributed_track_ids"])
        if prior_segment == frame["reset_segment"]:
            for track_id in sorted(current & prior_tracks):
                qualifications.append(
                    {
                        "track_id": track_id,
                        "reset_segment": int(frame["reset_segment"]),
                        "qualification_frame_id": int(frame["frame_id"]),
                    }
                )
        prior_tracks = current
        prior_segment = int(frame["reset_segment"])
    return qualifications


def _load_oracle_ledgers(
    repo: Path, config: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    import eligible_target_attribution_ordered_isolated_opening as oracle

    oracle_config_path = (
        repo / config["parent_bindings"]["oracle_config"]["path"]
    )
    oracle_config, oracle_bindings = oracle.load_and_verify_config(
        repo, oracle_config_path
    )
    _, ledgers = oracle.load_and_verify_token_inventory(
        repo, oracle_config, oracle_bindings
    )
    return ledgers


def _negative_interval_index(
    mask: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]], int
]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    total_duration_ns = 0
    for interval in mask["negative_exposure_intervals"]:
        require(
            int(interval["duration_ns"])
            == int(interval["end_ns"]) - int(interval["start_ns"]),
            "negative_interval_duration_drift",
        )
        total_duration_ns += int(interval["duration_ns"])
        index[(interval["source_id"], interval["sequence_id"])].append(interval)
    for rows in index.values():
        rows.sort(key=lambda row: (int(row["start_ns"]), int(row["end_ns"])))
    return index, total_duration_ns


def _audit_posthoc(
    repo: Path,
    config: dict[str, Any],
    inventory: dict[str, Any],
    ledgers: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    tokens = _producer_tokens(ledgers)
    token_by_track = {
        (
            row["source_id"],
            row["sequence_id"],
            row["reset_segment"],
            row["track_id"],
        ): row
        for row in tokens.values()
    }
    require(
        len(token_by_track) == len(tokens),
        "producer_track_scope_token_duplicate",
    )
    oracle_ledgers = _load_oracle_ledgers(repo, config)
    cell_rows = []
    matched_token_ids: set[str] = set()
    supported_event_ids: set[str] = set()
    unsupported_event_ids: set[str] = set()
    for (candidate_id, event_id), oracle_ledger in sorted(
        oracle_ledgers.items()
    ):
        qualifications = _oracle_qualifications(oracle_ledger)
        matches = []
        for qualification in qualifications:
            key = (
                oracle_ledger["source_id"],
                oracle_ledger["sequence_id"],
                qualification["reset_segment"],
                qualification["track_id"],
            )
            token = token_by_track.get(key)
            if (
                token is not None
                and token["qualification_frame_id"]
                <= qualification["qualification_frame_id"]
            ):
                matches.append(token)
        unique_matches = {
            row["token_id"]: row for row in matches
        }
        covered = bool(unique_matches)
        if covered:
            supported_event_ids.add(event_id)
            matched_token_ids.update(unique_matches)
        else:
            unsupported_event_ids.add(event_id)
        cell_rows.append(
            {
                "posthoc_oracle_cell_id": f"{candidate_id}::{event_id}",
                "candidate_id": candidate_id,
                "event_id": event_id,
                "source_id": oracle_ledger["source_id"],
                "sequence_id": oracle_ledger["sequence_id"],
                "oracle_qualification_count": len(qualifications),
                "covered": covered,
                "matched_producer_token_ids": sorted(unique_matches),
            }
        )
    expected = config["evaluation"]
    covered_cells = sum(row["covered"] for row in cell_rows)
    unsupported_cells = len(cell_rows) - covered_cells
    require(
        len(cell_rows) == expected["oracle_candidate_event_cells"],
        "oracle_cell_count_drift",
    )

    extra_rows = [
        row for token_id, row in sorted(tokens.items()) if token_id not in matched_token_ids
    ]
    mask = load_json(
        repo / config["parent_bindings"]["eligibility_mask"]["path"]
    )
    intervals, negative_duration_ns = _negative_interval_index(mask)
    negative_rows = []
    for token in tokens.values():
        matches = [
            interval["unit_id"]
            for interval in intervals.get(
                (token["source_id"], token["sequence_id"]), []
            )
            if int(interval["start_ns"])
            <= int(token["qualification_timestamp_ns"])
            <= int(interval["end_ns"])
        ]
        if matches:
            negative_rows.append(
                {
                    "token_id": token["token_id"],
                    "qualification_timestamp_ns": token[
                        "qualification_timestamp_ns"
                    ],
                    "negative_exposure_unit_ids": sorted(matches),
                }
            )
    negative_minutes = negative_duration_ns / 60_000_000_000
    negative_rate = (
        len(negative_rows) / negative_minutes if negative_minutes > 0 else None
    )
    repeat_rows = []
    unknown_route_activations = 0
    cross_reset_violations = 0
    for ledger in ledgers.values():
        current_segment = None
        for frame in ledger["frames"]:
            if frame["state_reset_before_frame"]:
                current_segment = frame["reset_segment"]
            for activation in frame["token_activations"]:
                unknown_route_activations += int(not frame["route_known"])
                cross_reset_violations += int(
                    activation["reset_segment"] != current_segment
                )
            repeat_rows.extend(
                {
                    **row,
                    "source_id": ledger["source_id"],
                    "sequence_id": ledger["sequence_id"],
                }
                for row in frame["repeat_activations_suppressed"]
            )

    output_root = repo / config["outputs"]["root"]
    extra_ledger = {
        "schema": EXTRA_SCHEMA,
        "stage": config["stage"],
        "producer_token_count": len(tokens),
        "matched_unique_producer_token_count": len(matched_token_ids),
        "extra_token_count": len(extra_rows),
        "negative_exposure_interval_count": len(
            mask["negative_exposure_intervals"]
        ),
        "negative_exposure_duration_ns": negative_duration_ns,
        "negative_exposure_minutes": negative_minutes,
        "negative_exposure_token_count": len(negative_rows),
        "negative_exposure_tokens_per_minute": negative_rate,
        "extra_tokens": extra_rows,
        "negative_exposure_tokens": sorted(
            negative_rows, key=lambda row: row["token_id"]
        ),
    }
    repeat_ledger = {
        "schema": REPEAT_SCHEMA,
        "stage": config["stage"],
        "repeat_activation_count": len(repeat_rows),
        "repeat_activations": sorted(
            repeat_rows,
            key=lambda row: (
                row["source_id"],
                row["sequence_id"],
                row["reset_segment"],
                row["track_id"],
                row["qualification_frame_id"],
            ),
        ),
    }
    extra_path = output_root / config["outputs"]["extra_token_ledger"]
    repeat_path = output_root / config["outputs"]["repeat_activation_ledger"]
    atomic_write_json(extra_path, extra_ledger)
    atomic_write_json(repeat_path, repeat_ledger)

    duplicate_token_ids = len(tokens) - len(set(tokens))
    coverage_passed = (
        covered_cells == expected["required_supported_cell_coverage"]
        and unsupported_cells == expected["required_no_active_relation_cells"]
    )
    integrity_passed = (
        duplicate_token_ids <= expected["duplicate_token_id_max"]
        and unknown_route_activations
        <= expected["unknown_route_token_activation_max"]
        and cross_reset_violations <= expected["cross_reset_token_max"]
    )
    credible_extra_bound = (
        negative_minutes
        >= float(expected["minimum_negative_exposure_minutes_for_bound"])
        and expected["extra_token_acceptance_threshold_per_minute"] is not None
    )
    if not coverage_passed or not integrity_passed:
        terminal_state = "REJECT"
    elif not credible_extra_bound:
        terminal_state = "HOLD_FOR_POLICY_GATE"
    else:
        terminal_state = "R0_FEASIBILITY_AND_POLICY_GATE_PASSED"
    require(
        terminal_state in config["terminal_states"],
        "terminal_state_not_preregistered",
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": config["stage"],
        "terminal_state": terminal_state,
        "authority": (
            "mechanism_feasibility_and_extra_token_audit_only_"
            "no_integration_or_selection_authority"
        ),
        "config_sha256": sha256_file(
            repo / "configs/ustrf_truth_blind_causal_per_track_"
            "attribution_token_producer_audit_r0.json"
        ),
        "phase_evidence": {
            "full_sequence_inventory_frozen_before_oracle_decode": True,
            "token_inventory_path": (
                Path(config["outputs"]["root"])
                / config["outputs"]["token_inventory"]
            ).as_posix(),
            "token_inventory_sha256": sha256_file(
                output_root / config["outputs"]["token_inventory"]
            ),
            "truth_payloads_decoded_before_freeze": 0,
            "event_windows_decoded_before_freeze": 0,
            "oracle_tokens_decoded_before_freeze": 0,
            "candidate_projection_count_verified": inventory[
                "candidate_projection_count_verified"
            ],
            "candidate_independent_sequence_ledgers": inventory[
                "candidate_independent_sequence_count"
            ],
            "full_sequence_frames": inventory["frame_count"],
        },
        "coverage_audit": {
            "oracle_candidate_event_cells": len(cell_rows),
            "oracle_supported_cells_covered": covered_cells,
            "oracle_no_active_relation_cells_fail_closed": unsupported_cells,
            "unique_supported_oracle_events": len(supported_event_ids),
            "unique_no_active_relation_oracle_events": len(
                unsupported_event_ids
            ),
            "unique_matched_producer_tokens": len(matched_token_ids),
            "candidate_cell_coverage_passed": coverage_passed,
            "cells": cell_rows,
        },
        "integrity_audit": {
            "producer_token_count": len(tokens),
            "duplicate_token_id_count": duplicate_token_ids,
            "unknown_route_token_activation_count": unknown_route_activations,
            "cross_reset_token_count": cross_reset_violations,
            "passed": integrity_passed,
        },
        "extra_token_audit": {
            "matched_unique_producer_token_count": len(matched_token_ids),
            "full_sequence_extra_token_count": len(extra_rows),
            "negative_exposure_interval_count": len(
                mask["negative_exposure_intervals"]
            ),
            "negative_exposure_duration_ns": negative_duration_ns,
            "negative_exposure_minutes": negative_minutes,
            "negative_exposure_token_count": len(negative_rows),
            "negative_exposure_tokens_per_minute": negative_rate,
            "repeat_activation_count": len(repeat_rows),
            "credible_extra_token_risk_bound": credible_extra_bound,
            "extra_token_acceptance_threshold_per_minute": expected[
                "extra_token_acceptance_threshold_per_minute"
            ],
            "extra_token_ledger_path": (
                Path(config["outputs"]["root"])
                / config["outputs"]["extra_token_ledger"]
            ).as_posix(),
            "extra_token_ledger_sha256": sha256_file(extra_path),
            "repeat_activation_ledger_path": (
                Path(config["outputs"]["root"])
                / config["outputs"]["repeat_activation_ledger"]
            ).as_posix(),
            "repeat_activation_ledger_sha256": sha256_file(repeat_path),
        },
        "claim_boundary": config["claim_boundary"],
    }


def build_terminal_from_frozen_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    inventory, ledgers = load_and_verify_blind_inventory(
        repo, config, bindings
    )
    terminal = _audit_posthoc(repo, config, inventory, ledgers)
    atomic_write_json(
        repo
        / config["outputs"]["root"]
        / config["outputs"]["terminal_receipt"],
        terminal,
    )
    return terminal


def run(repo: Path, config_path: Path) -> dict[str, Any]:
    build_and_freeze_blind_inventory(repo, config_path)
    return build_terminal_from_frozen_inventory(repo, config_path)


def validate_outputs(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    inventory, persisted_ledgers = load_and_verify_blind_inventory(
        repo, config, bindings
    )
    traces = _blind_parent_traces(repo, config)
    collapsed = collapse_candidate_projections(traces)
    recomputed_count = 0
    for key, frames in collapsed.items():
        expected = produce_sequence_ledger(
            key[0],
            key[1],
            frames,
            min_consecutive_relation_frames=int(
                config["producer"]["min_consecutive_relation_frames"]
            ),
        )
        require(
            expected == persisted_ledgers[key],
            f"producer_ledger_recompute_drift:{key!r}",
        )
        recomputed_count += 1
    expected_terminal = _audit_posthoc(
        repo, config, inventory, persisted_ledgers
    )
    terminal_path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["terminal_receipt"]
    )
    require(terminal_path.is_file(), "terminal_receipt_missing")
    require(
        load_json(terminal_path) == expected_terminal,
        "terminal_receipt_recompute_drift",
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "stage": config["stage"],
        "status": "VALID",
        "config_sha256": sha256_file(config_path),
        "terminal_sha256": sha256_file(terminal_path),
        "checks": {
            "candidate_projections_exactly_compared": len(traces),
            "candidate_independent_ledgers_exactly_recomputed": (
                recomputed_count
            ),
            "full_sequence_frames_recomputed": inventory["frame_count"],
            "truth_blind_inventory_reverified_before_posthoc_join": True,
            "forbidden_input_fields_rejected": True,
            "oracle_cell_coverage_recomputed": True,
            "extra_token_ledger_recomputed": True,
            "negative_exposure_rate_recomputed": True,
            "repeat_activation_ledger_recomputed": True,
        },
    }
    atomic_write_json(
        repo
        / config["outputs"]["root"]
        / config["outputs"]["validation_receipt"],
        validation,
    )
    return validation
