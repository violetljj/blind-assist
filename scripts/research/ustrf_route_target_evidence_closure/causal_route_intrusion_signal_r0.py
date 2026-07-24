#!/usr/bin/env python3
"""Candidate-independent causal route-relative intrusion separability probe R0."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import causal_per_track_attribution_token_audit_r0 as producer_parent
import causal_token_policy_risk_gate_r1 as policy
import current_input_policy_feasibility_bound_r0 as current_bound


CONFIG_SCHEMA = "blindassist_ustrf_causal_route_intrusion_signal_r0"
INVENTORY_SCHEMA = "blindassist_ustrf_causal_route_intrusion_signal_inventory_r0"
LEDGER_SCHEMA = "blindassist_ustrf_causal_route_intrusion_signal_ledger_r0"
TERMINAL_SCHEMA = "blindassist_ustrf_causal_route_intrusion_signal_terminal_r0"
VALIDATION_SCHEMA = "blindassist_ustrf_causal_route_intrusion_signal_validation_r0"
STAGE = "CANDIDATE-INDEPENDENT-CAUSAL-ROUTE-INTRUSION-SIGNAL-R0"
LEGAL_TERMINALS = (
    "SIGNAL_SEPARABILITY_PASS_RISK_SUPPORT_INSUFFICIENT",
    "SIGNAL_REJECT",
)


class SignalContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SignalContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


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
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    data = canonical_bytes(value)
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require(load_json(temporary) == value, f"atomic_verify_failed:{path}")
    os.replace(temporary, path)


def _verify(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label}_binding_drift")
    path = repo / binding["path"]
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == binding["sha256"], f"{label}_sha_drift")
    return path


def load_and_verify_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(
        config.get("status") == "frozen_before_any_signal_output",
        "config_not_frozen_before_output",
    )
    boundary = config["claim_boundary"]
    require(
        boundary["maximum"] == "NEW_SIGNAL_SEPARABILITY_PROBE_ONLY"
        and boundary["candidate_independent"] is True
        and boundary["signal_output_visible_at_freeze"] is False,
        "claim_boundary_drift",
    )
    require(
        all(
            value is False
            for key, value in boundary.items()
            if key
            not in {"maximum", "candidate_independent", "signal_output_visible_at_freeze"}
        ),
        "higher_authority_opened",
    )
    signal = config["signal"]
    require(
        signal["name"] == "causal_route_relative_intrusion_trend_2_of_3"
        and signal["history_length_frames"] == 5
        and signal["positive_component_count_required"] == 2
        and signal["component_thresholds_per_second"]
        == {
            "route_relative_radial_distance_slope": 0.0,
            "route_relative_lateral_distance_slope": 0.0,
            "log_bbox_height_slope": 0.0,
        }
        and signal["one_token_per_track_per_reset"] is True
        and signal["renewal"] is False,
        "signal_definition_drift",
    )
    gate = config["evaluation_gate"]
    require(
        gate["required_supported_unique_events"] == 11
        and gate["required_supported_candidate_cells"] == 33
        and gate["maximum_empirical_negative_token_count"] == 2
        and gate["point_rate_max_per_minute"] == 0.5
        and gate["poisson_ucb_max_per_minute"] == 0.5
        and gate["minimum_zero_event_exposure_minutes"] == 5.9915
        and gate["current_negative_exposure_minutes"] == 4.95626851575
        and gate["minimum_negative_sequences_per_source"] == 3,
        "evaluation_gate_drift",
    )
    for label, binding in config["parent_bindings"].items():
        _verify(repo, binding, label)
    current_terminal = load_json(
        repo / config["parent_bindings"]["current_family_terminal"]["path"]
    )
    require(
        current_terminal.get("terminal_state")
        == "CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE",
        "required_parent_stop_terminal_missing",
    )
    require(config["terminal_states"] == list(LEGAL_TERMINALS), "terminal_drift")
    return config


def _load_route_map(
    repo: Path, config: dict[str, Any]
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for binding in config["route_bindings"]:
        path = _verify(repo, binding, f"route_{len(result)}")
        payload = load_json(path)
        if payload.get("schema") == "blindassist_ustrf_route_role_review_bundle_r1":
            for source in payload["sources"]:
                for window in source["windows"]:
                    for row in window["frames"]:
                        key = (
                            source["source_id"],
                            source["source_id"],
                            int(row["frame_id"]),
                            int(row["source_capture_timestamp_ns"]),
                        )
                        value = {
                            "status": row["route_status"],
                            "uv": row["route_uv"],
                        }
                        require(
                            key not in result or result[key] == value,
                            "route_duplicate_drift",
                        )
                        result[key] = value
        else:
            for source in payload["sources"]:
                for sequence in source["sequences"]:
                    for row in sequence["route_predictions"]:
                        key = (
                            source["source_id"],
                            sequence["sequence_id"],
                            int(row["frame_id"]),
                            int(row["source_capture_timestamp_ns"]),
                        )
                        result[key] = {
                            "status": row["status"],
                            "uv": row.get("uv"),
                        }
    return result


def _median_pairwise_slope(history: list[dict[str, float]], field: str) -> float:
    slopes = []
    for left in range(len(history)):
        for right in range(left + 1, len(history)):
            elapsed = history[right]["timestamp_s"] - history[left]["timestamp_s"]
            require(elapsed > 0.0, "nonincreasing_signal_timestamp")
            slopes.append((history[right][field] - history[left][field]) / elapsed)
    require(bool(slopes), "empty_slope_set")
    return float(statistics.median(slopes))


def signal_components(history: list[dict[str, float]]) -> dict[str, Any]:
    require(len(history) == 5, "signal_history_not_five")
    radial = _median_pairwise_slope(history, "radial")
    lateral = _median_pairwise_slope(history, "lateral")
    scale = _median_pairwise_slope(history, "log_height")
    finite = all(math.isfinite(value) for value in (radial, lateral, scale))
    require(finite, "nonfinite_signal_slope")
    positives = {
        "route_relative_radial_converging": radial < 0.0,
        "route_relative_lateral_converging": lateral < 0.0,
        "bbox_scale_expanding": scale > 0.0,
    }
    return {
        "route_relative_radial_distance_slope_per_s": radial,
        "route_relative_lateral_distance_slope_per_s": lateral,
        "log_bbox_height_slope_per_s": scale,
        "positive_component_count": sum(positives.values()),
        "components": positives,
        "signal_positive": sum(positives.values()) >= 2,
    }


def build_signal_ledger(
    source_id: str,
    sequence_id: str,
    frames: list[dict[str, Any]],
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    diagonal = math.hypot(width, height)
    histories: dict[int, deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=5)
    )
    emitted: set[tuple[int, int]] = set()
    activation_rows: list[dict[str, Any]] = []
    active_episodes: dict[int, dict[str, Any]] = {}
    completed_episodes: list[dict[str, Any]] = []
    reset_segment: int | None = None
    projected_frames = []
    previous_frame_id: int | None = None
    previous_timestamp: int | None = None
    for frame in frames:
        frame_id = int(frame["frame_id"])
        timestamp = int(frame["source_capture_timestamp_ns"])
        if frame["state_reset_before_frame"]:
            for episode in active_episodes.values():
                episode["valid_through_frame"] = (
                    previous_frame_id if previous_frame_id is not None else frame_id
                )
                completed_episodes.append(episode)
            histories.clear()
            active_episodes.clear()
            reset_segment = int(frame["reset_segment"])
            previous_frame_id = None
            previous_timestamp = None
        require(reset_segment is not None, "reset_segment_missing")
        require(
            previous_timestamp is None or timestamp > previous_timestamp,
            "sequence_timestamp_not_increasing",
        )
        route = route_map.get((source_id, sequence_id, frame_id, timestamp))
        require(route is not None, "causal_route_frame_missing")
        route_known = route.get("status") == "known" and route.get("uv") is not None
        observed = {
            int(track["track_id"]): track["box"] for track in frame["observed_tracks"]
        }
        require(len(observed) == len(frame["observed_tracks"]), "duplicate_track")
        active = (
            set(int(value) for value in frame["active_relation_track_ids"])
            if route_known
            else set()
        )
        require(active <= set(observed), "active_track_not_observed")
        for track_id in list(histories):
            if track_id not in active:
                histories.pop(track_id, None)
        for track_id in list(active_episodes):
            if track_id not in active:
                episode = active_episodes.pop(track_id)
                episode["valid_through_frame"] = (
                    previous_frame_id if previous_frame_id is not None else frame_id
                )
                completed_episodes.append(episode)
        frame_signals = []
        if route_known:
            u, v = [float(value) for value in route["uv"]]
            require(math.isfinite(u) and math.isfinite(v), "nonfinite_route_uv")
            for track_id in sorted(active):
                x1, y1, x2, y2 = [float(value) for value in observed[track_id]]
                require(
                    all(math.isfinite(value) for value in (x1, y1, x2, y2))
                    and x2 > x1
                    and y2 > y1,
                    "invalid_track_box",
                )
                history = histories[track_id]
                if history and (
                    int(history[-1]["frame_id"]) + 1 != frame_id
                    or timestamp <= int(history[-1]["timestamp_ns"])
                ):
                    history.clear()
                center_x = (x1 + x2) / 2.0
                measurement = {
                    "frame_id": float(frame_id),
                    "timestamp_ns": float(timestamp),
                    "timestamp_s": timestamp / 1e9,
                    "radial": math.hypot(center_x - u, y2 - v) / diagonal,
                    "lateral": abs(center_x - u) / width,
                    "log_height": math.log((y2 - y1) / height),
                }
                history.append(measurement)
                if track_id not in active_episodes:
                    active_episodes[track_id] = {
                        "reset_segment": reset_segment,
                        "track_id": track_id,
                        "start_frame": frame_id,
                    }
                if len(history) == 5:
                    components = signal_components(list(history))
                    frame_signals.append(
                        {
                            "track_id": track_id,
                            "signal_positive": components["signal_positive"],
                            "positive_component_count": components[
                                "positive_component_count"
                            ],
                        }
                    )
                    scope = (reset_segment, track_id)
                    if components["signal_positive"] and scope not in emitted:
                        emitted.add(scope)
                        activation = {
                            "reset_segment": reset_segment,
                            "track_id": track_id,
                            "activation_frame": frame_id,
                            "activation_timestamp_ns": timestamp,
                            "component_slopes": {
                                key: components[key]
                                for key in (
                                    "route_relative_radial_distance_slope_per_s",
                                    "route_relative_lateral_distance_slope_per_s",
                                    "log_bbox_height_slope_per_s",
                                )
                            },
                            "positive_component_count": components[
                                "positive_component_count"
                            ],
                        }
                        activation_rows.append(activation)
                        active_episodes[track_id]["activation_index"] = (
                            len(activation_rows) - 1
                        )
        projected_frames.append(
            {
                "frame_id": frame_id,
                "source_capture_timestamp_ns": timestamp,
                "reset_segment": reset_segment,
                "route_known": route_known,
                "signal_states": frame_signals,
            }
        )
        previous_frame_id = frame_id
        previous_timestamp = timestamp
    for episode in active_episodes.values():
        episode["valid_through_frame"] = (
            previous_frame_id if previous_frame_id is not None else episode["start_frame"]
        )
        completed_episodes.append(episode)
    for episode in completed_episodes:
        if "activation_index" in episode:
            activation_rows[int(episode["activation_index"])]["valid_through_frame"] = int(
                episode["valid_through_frame"]
            )
    require(
        all("valid_through_frame" in row for row in activation_rows),
        "activation_validity_unclosed",
    )
    return {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "authority": "candidate_independent_signal_output_before_truth_join",
        "source_id": source_id,
        "sequence_id": sequence_id,
        "frame_count": len(frames),
        "frames": projected_frames,
        "activations": activation_rows,
        "activation_count": len(activation_rows),
        "truth_payloads_decoded": 0,
        "event_windows_decoded": 0,
        "oracle_tokens_decoded": 0,
        "negative_exposure_decoded": 0,
    }


def _producer_inputs(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[Any, Any]]:
    producer_config_path = repo / config["parent_bindings"]["producer_config"]["path"]
    producer_config, _ = producer_parent.load_and_verify_config(
        repo, producer_config_path
    )
    traces = producer_parent._blind_parent_traces(repo, producer_config)
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        dict
    )
    for (candidate_id, source_id, sequence_id), trace in traces.items():
        projected = []
        for frame in trace["frames"]:
            projected.append(
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
                    "reset_segment": int(frame["reset_segment"]),
                    "route_known": bool(frame["route_known"]),
                    "observed_tracks": [
                        {
                            "track_id": int(track["track_id"]),
                            "box": [float(value) for value in track["box"]],
                        }
                        for track in frame["observed_tracks"]
                    ],
                    "active_relation_track_ids": [
                        int(value)
                        for value in frame["active_relation_track_ids"]
                    ],
                }
            )
        grouped[(source_id, sequence_id)][candidate_id] = projected
    collapsed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_candidates = set(producer_parent.CANDIDATES)
    for key, projections in grouped.items():
        require(
            set(projections) == expected_candidates,
            "signal_candidate_projection_roster_drift",
        )
        reference = projections[producer_parent.CANDIDATES[0]]
        for candidate_id in producer_parent.CANDIDATES[1:]:
            require(
                projections[candidate_id] == reference,
                f"signal_bbox_projection_mismatch:{key!r}:{candidate_id}",
            )
        collapsed[key] = reference
    route_map = _load_route_map(repo, config)
    expected = config["expected_scope"]
    require(len(traces) == expected["candidate_projections"], "projection_count_drift")
    require(len(collapsed) == expected["sequences"], "sequence_count_drift")
    require(
        sum(len(rows) for rows in collapsed.values()) == expected["frames"],
        "frame_count_drift",
    )
    return collapsed, route_map


def run_producer(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    collapsed, route_map = _producer_inputs(repo, config)
    root = repo / config["outputs"]["root"]
    ledger_root = root / "signal-ledgers"
    inventory_rows = []
    for (source_id, sequence_id), frames in sorted(collapsed.items()):
        ledger = build_signal_ledger(source_id, sequence_id, frames, route_map)
        slug = sha256_bytes(f"{source_id}|{sequence_id}".encode())[:16]
        path = ledger_root / f"{slug}.json"
        atomic_write_json(path, ledger)
        inventory_rows.append(
            {
                "source_id": source_id,
                "sequence_id": sequence_id,
                "path": path.relative_to(repo).as_posix(),
                "sha256": sha256_file(path),
                "frame_count": ledger["frame_count"],
                "activation_count": ledger["activation_count"],
            }
        )
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "stage": STAGE,
        "status": "SIGNAL_INVENTORY_FROZEN_BEFORE_TRUTH_JOIN",
        "config_sha256": sha256_file(config_path),
        "sequence_count": len(inventory_rows),
        "frame_count": sum(row["frame_count"] for row in inventory_rows),
        "activation_count": sum(row["activation_count"] for row in inventory_rows),
        "candidate_projection_count_verified": config["expected_scope"][
            "candidate_projections"
        ],
        "truth_payloads_decoded": 0,
        "event_windows_decoded": 0,
        "oracle_tokens_decoded": 0,
        "negative_exposure_decoded": 0,
        "ledgers": inventory_rows,
    }
    require(
        inventory["sequence_count"] == config["expected_scope"]["sequences"]
        and inventory["frame_count"] == config["expected_scope"]["frames"],
        "inventory_scope_drift",
    )
    atomic_write_json(root / config["outputs"]["inventory"], inventory)
    return inventory


def _poisson_upper_mean(count: int, confidence: float) -> float:
    target = 1.0 - confidence

    def cdf(mean: float) -> float:
        term = math.exp(-mean)
        total = term
        for value in range(1, count + 1):
            term *= mean / value
            total += term
        return total

    low, high = 0.0, max(8.0, float(count + 8))
    while cdf(high) > target:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > target:
            low = middle
        else:
            high = middle
    return high


def run_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    root = repo / config["outputs"]["root"]
    inventory_path = root / config["outputs"]["inventory"]
    require(inventory_path.is_file(), "signal_inventory_missing")
    inventory = load_json(inventory_path)
    require(inventory.get("schema") == INVENTORY_SCHEMA, "inventory_schema_drift")
    require(
        inventory.get("config_sha256") == sha256_file(config_path),
        "inventory_config_drift",
    )
    activations: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in inventory["ledgers"]:
        path = repo / row["path"]
        require(path.is_file(), "signal_ledger_missing")
        require(sha256_file(path) == row["sha256"], "signal_ledger_sha_drift")
        ledger = load_json(path)
        require(ledger.get("schema") == LEDGER_SCHEMA, "signal_ledger_schema_drift")
        require(
            ledger["truth_payloads_decoded"] == 0
            and ledger["event_windows_decoded"] == 0
            and ledger["oracle_tokens_decoded"] == 0
            and ledger["negative_exposure_decoded"] == 0,
            "producer_information_leakage",
        )
        for activation in ledger["activations"]:
            key = (
                ledger["source_id"],
                ledger["sequence_id"],
                int(activation["reset_segment"]),
                int(activation["track_id"]),
            )
            require(key not in activations, "duplicate_track_reset_activation")
            activations[key] = activation
    current_config_path = (
        repo / config["parent_bindings"]["current_family_config"]["path"]
    )
    _, events, negative, duration_ns = current_bound._load_frozen_inputs(
        repo, load_json(current_config_path)
    )
    supported_events = 0
    event_rows = []
    for key, qualifications in sorted(events.items()):
        covered = False
        for qualification in qualifications:
            scope = (
                key[0],
                key[1],
                int(qualification["reset_segment"]),
                int(qualification["track_id"]),
            )
            activation = activations.get(scope)
            if (
                activation is not None
                and int(activation["activation_frame"])
                <= int(qualification["oracle_frame_id"])
                <= int(activation["valid_through_frame"])
            ):
                covered = True
                break
        if qualifications and covered:
            supported_events += 1
        event_rows.append(
            {
                "source_id": key[0],
                "sequence_id": key[1],
                "event_id": key[2],
                "oracle_supported": bool(qualifications),
                "signal_covered": covered,
            }
        )
    negative_count = 0
    negative_by_source: dict[str, int] = defaultdict(int)
    for scope, activation in activations.items():
        timestamp = int(activation["activation_timestamp_ns"])
        intervals = negative.get((scope[0], scope[1]), [])
        if any(
            int(row["start_ns"]) <= timestamp < int(row["end_ns"])
            for row in intervals
        ):
            negative_count += 1
            negative_by_source[scope[0]] += 1
    exposure_minutes = duration_ns / 60_000_000_000
    point_rate = negative_count / exposure_minutes
    ucb_rate = _poisson_upper_mean(negative_count, 0.95) / exposure_minutes
    gate = config["evaluation_gate"]
    coverage_pass = supported_events == gate["required_supported_unique_events"]
    point_risk_pass = negative_count <= gate["maximum_empirical_negative_token_count"]
    credible_support_pass = (
        exposure_minutes >= gate["minimum_zero_event_exposure_minutes"]
        and False
    )
    separability_pass = coverage_pass and point_risk_pass
    terminal_state = (
        "SIGNAL_SEPARABILITY_PASS_RISK_SUPPORT_INSUFFICIENT"
        if separability_pass and not credible_support_pass
        else "SIGNAL_REJECT"
    )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": terminal_state,
        "authority": "new_signal_separability_probe_only",
        "config_sha256": sha256_file(config_path),
        "inventory_sha256": sha256_file(inventory_path),
        "signal": {
            "name": config["signal"]["name"],
            "activation_count": len(activations),
            "supported_unique_event_coverage": supported_events,
            "required_supported_unique_event_coverage": gate[
                "required_supported_unique_events"
            ],
            "supported_candidate_cell_coverage": supported_events * 3,
            "required_supported_candidate_cell_coverage": gate[
                "required_supported_candidate_cells"
            ],
            "event_rows": event_rows,
        },
        "risk": {
            "negative_activation_count": negative_count,
            "maximum_empirical_negative_token_count": gate[
                "maximum_empirical_negative_token_count"
            ],
            "negative_exposure_minutes": exposure_minutes,
            "point_rate_per_minute": point_rate,
            "one_sided_95pct_poisson_ucb_per_minute": ucb_rate,
            "negative_activation_count_by_source": dict(sorted(negative_by_source.items())),
            "point_risk_pass": point_risk_pass,
            "credible_support_pass": credible_support_pass,
        },
        "comparison_to_current_family": {
            "current_family_max_supported_unique_events": 8,
            "current_family_max_supported_cells": 24,
            "current_family_risk_limited_supported_unique_events": 2,
            "current_family_risk_limited_supported_cells": 6,
            "new_signal_supported_unique_event_delta_vs_unconstrained_bound": (
                supported_events - 8
            ),
            "new_signal_supported_cell_delta_vs_unconstrained_bound": (
                supported_events * 3 - 24
            ),
        },
        "decision": {
            "coverage_pass": coverage_pass,
            "point_risk_pass": point_risk_pass,
            "credible_support_pass": credible_support_pass,
            "retain_signal_for_policy_work": separability_pass,
            "discard_signal_if_reject": terminal_state == "SIGNAL_REJECT",
        },
        "authority_flags": {
            "successor_policy": False,
            "threshold_tuning": False,
            "opener": False,
            "selection": False,
            "android_shadow": False,
            "h2": False,
            "human": False,
            "production": False,
        },
    }
    terminal_path = root / config["outputs"]["terminal"]
    atomic_write_json(terminal_path, terminal)
    return terminal


def validate_outputs(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    recomputed_inventory = run_producer(repo, config_path)
    recomputed_terminal = run_audit(repo, config_path)
    root = repo / config["outputs"]["root"]
    result = {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "decision": "VALID",
        "config_sha256": sha256_file(config_path),
        "inventory_sha256": sha256_file(root / config["outputs"]["inventory"]),
        "terminal_sha256": sha256_file(root / config["outputs"]["terminal"]),
        "sequence_count_recomputed": recomputed_inventory["sequence_count"],
        "frame_count_recomputed": recomputed_inventory["frame_count"],
        "activation_count_recomputed": recomputed_inventory["activation_count"],
        "terminal_state_recomputed": recomputed_terminal["terminal_state"],
        "candidate_execution_count": 0,
        "successor_policy_authorized": False,
    }
    atomic_write_json(root / config["outputs"]["validation"], result)
    return result
