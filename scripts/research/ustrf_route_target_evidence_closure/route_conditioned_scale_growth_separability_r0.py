#!/usr/bin/env python3
"""Fail-closed contract preflight for route-conditioned scale growth R0."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import causal_per_track_attribution_token_audit_r0 as producer_parent


CONFIG_SCHEMA = "blindassist_ustrf_route_conditioned_scale_growth_separability_r0"
CONTRACT_SCHEMA = (
    "blindassist_ustrf_route_conditioned_scale_growth_contract_violation_r0"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_route_conditioned_scale_growth_separability_terminal_r0"
)
AUDIT_SCHEMA = (
    "blindassist_ustrf_route_conditioned_scale_growth_separability_audit_r0"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_route_conditioned_scale_growth_separability_validation_r0"
)
STAGE = "ROUTE-CONDITIONED-SCALE-GROWTH-SEPARABILITY-R0"
BLOCKED = "FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED"
LEGAL_TERMINALS = (
    "PURE_SCALE_GROWTH_NOT_SUFFICIENT_FOR_STANDALONE_TOKEN_QUALIFICATION",
    "SCALE_GROWTH_DISCOVERY_CANDIDATE_FROZEN",
    BLOCKED,
)
IMPLEMENTATION_PATHS = {
    "core": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "route_conditioned_scale_growth_separability_r0.py"
    ),
    "runner": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_route_conditioned_scale_growth_separability_r0.py"
    ),
    "validator": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_route_conditioned_scale_growth_separability_r0.py"
    ),
    "tests": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_route_conditioned_scale_growth_separability_r0.py"
    ),
}


class ScaleGrowthContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScaleGrowthContractError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


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


def _verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label}_binding_keys_drift")
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing")
    require(
        sha256_file(path) == str(binding["sha256"]),
        f"{label}_sha256_drift",
    )
    return path


def normalized_scale(
    box: list[float], source_width: int, source_height: int
) -> float:
    require(source_width > 0 and source_height > 0, "source_size_invalid")
    require(len(box) == 4, "bbox_not_xyxy")
    x1, y1, x2, y2 = (float(value) for value in box)
    require(
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x2 > x1
        and y2 > y1,
        "bbox_invalid",
    )
    width_norm = (x2 - x1) / source_width
    height_norm = (y2 - y1) / source_height
    return 0.5 * math.log(width_norm * height_norm)


def bbox_touches_boundary(
    box: list[float], source_width: int, source_height: int
) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 <= 0.0 or y1 <= 0.0 or x2 >= source_width or y2 >= source_height


def theil_sen_slope(observations: list[dict[str, float]]) -> float:
    require(len(observations) >= 2, "theil_sen_needs_two_observations")
    slopes: list[float] = []
    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            elapsed = (
                float(observations[right]["timestamp_s"])
                - float(observations[left]["timestamp_s"])
            )
            require(elapsed > 0.0, "timestamp_not_strictly_increasing")
            slopes.append(
                (
                    float(observations[right]["scale"])
                    - float(observations[left]["scale"])
                )
                / elapsed
            )
    slope = float(statistics.median(slopes))
    require(math.isfinite(slope), "theil_sen_nonfinite")
    return slope


def eligible_window(
    observations: list[dict[str, float]],
    *,
    window_ms: int = 600,
    minimum_observations: int = 5,
    maximum_gap_ms: int = 150,
) -> tuple[list[dict[str, float]], str | None]:
    require(bool(observations), "observation_history_empty")
    current = float(observations[-1]["timestamp_s"])
    lower = current - window_ms / 1000.0
    window = [
        row for row in observations if float(row["timestamp_s"]) >= lower
    ]
    if len(window) < minimum_observations:
        return window, "minimum_observations_not_met"
    gaps = [
        (
            float(window[index]["timestamp_s"])
            - float(window[index - 1]["timestamp_s"])
        )
        * 1000.0
        for index in range(1, len(window))
    ]
    require(all(gap > 0.0 for gap in gaps), "timestamp_not_strictly_increasing")
    if max(gaps, default=0.0) > maximum_gap_ms + 1e-9:
        return window, "maximum_adjacent_gap_exceeded"
    return window, None


def complete_threshold_breakpoints(scores: list[float]) -> list[float]:
    require(bool(scores), "score_frontier_empty")
    require(all(math.isfinite(score) for score in scores), "score_nonfinite")
    return sorted(set(float(score) for score in scores), reverse=True)


def load_and_verify_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(
        config.get("status") == "frozen_before_any_signal_outcome",
        "config_not_frozen_before_signal_outcome",
    )
    require(
        tuple(config["terminal_states"]) == LEGAL_TERMINALS,
        "terminal_states_drift",
    )
    boundary = config["claim_boundary"]
    require(
        boundary["maximum"]
        == "SIGNAL_AVAILABILITY_AND_DISCOVERY_SEPARABILITY_ONLY"
        and boundary["candidate_independent"] is True
        and boundary["research_only"] is True
        and all(
            boundary[key] is False
            for key in (
                "online_policy",
                "causal_producer",
                "opener",
                "android",
                "shadow",
                "h2",
                "human",
                "production",
            )
        ),
        "claim_boundary_drift",
    )
    signal = config["signal_contract"]
    require(
        signal
        == {
            "formula": "S_t=0.5*log(w_norm*h_norm)",
            "window_ms": 600,
            "minimum_valid_observations": 5,
            "maximum_adjacent_gap_ms": 150,
            "timestamp_axis": "source_capture_timestamp_seconds",
            "slope_estimator": "theil_sen_median_all_pairwise_slopes",
            "interpolation": False,
            "touching_boundary_invalid": True,
            "severe_truncation_invalid": True,
            "diagnostic_only": "log_height_norm",
            "only_swept_variable": "looming_score_threshold",
            "threshold_comparison": "looming_score_greater_than_or_equal",
            "threshold_breakpoints": "all_unique_finite_observed_slopes_descending",
        },
        "signal_contract_drift",
    )
    delay = config["delay_contract"]
    require(
        delay["maximum_first_qualification_delay_ms"] == 5000
        and delay["reference"] == "frozen_event_window_start_timestamp"
        and delay["frozen_before_signal_outcome"] is True
        and delay["inherited_from_parent_evaluator"] is False,
        "delay_contract_drift",
    )
    _verify_binding(repo, delay["rationale_binding"], "delay_rationale")
    gate = config["discovery_gate"]
    require(
        gate["supported_unique_events"] == 11
        and gate["supported_candidate_cells"] == 33
        and gate["negative_first_opportunity_max"] == 2
        and gate["negative_exposure_duration_ns"] == 297376110945
        and gate["negative_exposure_minutes"] == 4.95626851575,
        "discovery_gate_drift",
    )
    for label, binding in config["parent_bindings"].items():
        _verify_binding(repo, binding, label)
    require(
        set(config["implementation_bindings"]) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_inventory_drift",
    )
    for label, relative in IMPLEMENTATION_PATHS.items():
        path = repo / relative
        require(path.is_file(), f"implementation_missing:{relative}")
        require(
            sha256_file(path) == config["implementation_bindings"][label],
            f"implementation_sha256_drift:{label}",
        )
    return config


def _candidate_blind_frames(
    repo: Path, config: dict[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    parent_config_path = (
        repo / config["parent_bindings"]["producer_parent_config"]["path"]
    )
    parent_config, _ = producer_parent.load_and_verify_config(
        repo, parent_config_path
    )
    traces = producer_parent._blind_parent_traces(repo, parent_config)
    grouped: dict[
        tuple[str, str], dict[str, list[dict[str, Any]]]
    ] = defaultdict(dict)
    for (candidate_id, source_id, sequence_id), trace in traces.items():
        projected = [
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
                "observed_tracks": frame["observed_tracks"],
                "active_relation_track_ids": frame[
                    "active_relation_track_ids"
                ],
                "source_size": frame.get("source_size"),
                "rotation_receipt": frame.get("rotation_receipt"),
            }
            for frame in trace["frames"]
        ]
        grouped[(source_id, sequence_id)][candidate_id] = projected
    collapsed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_candidates = set(producer_parent.CANDIDATES)
    for key, projections in grouped.items():
        require(
            set(projections) == expected_candidates,
            "candidate_projection_roster_drift",
        )
        reference = projections[producer_parent.CANDIDATES[0]]
        for candidate_id in producer_parent.CANDIDATES[1:]:
            require(
                projections[candidate_id] == reference,
                f"candidate_projection_drift:{key!r}:{candidate_id}",
            )
        collapsed[key] = reference
    expected = config["expected_scope"]
    require(len(traces) == 123, "candidate_projection_count_drift")
    require(len(collapsed) == expected["sequences"], "sequence_count_drift")
    require(
        sum(len(frames) for frames in collapsed.values())
        == expected["frames"],
        "frame_count_drift",
    )
    return collapsed


def build_contract_violation(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    sequences = _candidate_blind_frames(repo, config)
    missing_size = 0
    missing_rotation = 0
    missing_truncation = 0
    observed_track_count = 0
    for frames in sequences.values():
        for frame in frames:
            if frame["source_size"] is None:
                missing_size += 1
            if frame["rotation_receipt"] is None:
                missing_rotation += 1
            for track in frame["observed_tracks"]:
                observed_track_count += 1
                if "severe_truncation" not in track:
                    missing_truncation += 1
    gaps = []
    if missing_size:
        gaps.append(
            {
                "code": "CANONICAL_SOURCE_SIZE_NOT_BOUND_PER_FRAME",
                "affected_frame_count": missing_size,
                "required_resolution": (
                    "bind authoritative canonical source width and height to "
                    "each candidate-blind frame"
                ),
            }
        )
    if missing_rotation:
        gaps.append(
            {
                "code": "ROTATION_RECEIPT_NOT_BOUND_PER_FRAME",
                "affected_frame_count": missing_rotation,
                "required_resolution": (
                    "bind authoritative source-to-canonical orientation or "
                    "rotation receipt to each candidate-blind frame"
                ),
            }
        )
    if missing_truncation:
        gaps.append(
            {
                "code": "SEVERE_TRUNCATION_STATUS_NOT_BOUND_PER_TRACK",
                "affected_observed_track_count": missing_truncation,
                "required_resolution": (
                    "bind authoritative severe-truncation status; boundary "
                    "touch alone cannot prove the remaining boxes untruncated"
                ),
            }
        )
    require(bool(gaps), "expected_contract_gap_no_longer_present")
    return {
        "schema": CONTRACT_SCHEMA,
        "stage": STAGE,
        "status": "BLOCKED_BEFORE_SIGNAL_OUTCOME",
        "config_sha256": sha256_file(config_path),
        "process_id": os.getpid(),
        "candidate_blind_scope": {
            "candidate_projections_verified": 123,
            "sequence_count": len(sequences),
            "frame_count": sum(len(frames) for frames in sequences.values()),
            "observed_track_count": observed_track_count,
        },
        "violations": gaps,
        "signal_scores_computed": 0,
        "truth_payloads_decoded": 0,
        "event_windows_decoded": 0,
        "oracle_tokens_decoded": 0,
        "negative_exposure_decoded": 0,
        "candidate_cells_decoded": 0,
    }


def _terminal(
    config_path: Path, contract_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": BLOCKED,
        "authority": "signal_availability_and_discovery_separability_only",
        "config_sha256": sha256_file(config_path),
        "contract_violation_sha256": sha256_file(contract_path),
        "producer_preflight_process_id": contract["process_id"],
        "gap_codes": [row["code"] for row in contract["violations"]],
        "signal_outcome_generated": False,
        "inventory_generated": False,
        "frontier_generated": False,
        "candidate_frozen": False,
        "authority_flags": {
            "online_policy": False,
            "causal_producer": False,
            "opener": False,
            "android": False,
            "shadow": False,
            "h2": False,
            "human": False,
            "production": False,
        },
    }


def run_blocking_preflight(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    root = repo / config["outputs"]["root"]
    contract_path = root / config["outputs"]["contract_violation"]
    terminal_path = root / config["outputs"]["terminal"]
    for forbidden in (
        config["outputs"]["inventory"],
        config["outputs"]["frontier"],
    ):
        require(not (root / forbidden).exists(), f"forbidden_output_exists:{forbidden}")
    contract = build_contract_violation(repo, config_path)
    atomic_write_json(contract_path, contract)
    terminal = _terminal(config_path, contract_path, contract)
    atomic_write_json(terminal_path, terminal)
    return terminal


def _same_except_process_id(
    actual: dict[str, Any], expected: dict[str, Any]
) -> bool:
    actual_copy = dict(actual)
    expected_copy = dict(expected)
    actual_copy.pop("process_id", None)
    expected_copy.pop("process_id", None)
    return actual_copy == expected_copy


def run_blocked_audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    root = repo / config["outputs"]["root"]
    contract_path = root / config["outputs"]["contract_violation"]
    terminal_path = root / config["outputs"]["terminal"]
    require(contract_path.is_file(), "contract_violation_receipt_missing")
    require(terminal_path.is_file(), "terminal_receipt_missing")
    persisted_contract = load_json(contract_path)
    require(
        persisted_contract.get("schema") == CONTRACT_SCHEMA,
        "contract_violation_schema_drift",
    )
    require(
        int(persisted_contract["process_id"]) != os.getpid(),
        "producer_and_audit_process_not_isolated",
    )
    expected_contract = build_contract_violation(repo, config_path)
    require(
        _same_except_process_id(persisted_contract, expected_contract),
        "contract_violation_recompute_drift",
    )
    expected_terminal = _terminal(
        config_path, contract_path, persisted_contract
    )
    require(
        load_json(terminal_path) == expected_terminal,
        "terminal_receipt_recompute_drift",
    )
    for forbidden in (
        config["outputs"]["inventory"],
        config["outputs"]["frontier"],
    ):
        require(not (root / forbidden).exists(), f"forbidden_output_exists:{forbidden}")
    audit = {
        "schema": AUDIT_SCHEMA,
        "stage": STAGE,
        "decision": "BLOCKED_TERMINAL_REVERIFIED",
        "config_sha256": sha256_file(config_path),
        "contract_violation_sha256": sha256_file(contract_path),
        "terminal_sha256": sha256_file(terminal_path),
        "producer_process_id": int(persisted_contract["process_id"]),
        "audit_process_id": os.getpid(),
        "separate_process_verified": True,
        "signal_outcome_absent": True,
        "inventory_absent": True,
        "frontier_absent": True,
    }
    atomic_write_json(root / config["outputs"]["audit"], audit)
    return audit


def validate_blocked_outputs(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    root = repo / config["outputs"]["root"]
    contract_path = root / config["outputs"]["contract_violation"]
    terminal_path = root / config["outputs"]["terminal"]
    audit_path = root / config["outputs"]["audit"]
    for path, label in (
        (contract_path, "contract"),
        (terminal_path, "terminal"),
        (audit_path, "audit"),
    ):
        require(path.is_file(), f"{label}_receipt_missing")
    contract = load_json(contract_path)
    audit = load_json(audit_path)
    require(os.getpid() not in {
        int(contract["process_id"]),
        int(audit["audit_process_id"]),
    }, "validator_process_not_independent")
    expected_contract = build_contract_violation(repo, config_path)
    require(
        _same_except_process_id(contract, expected_contract),
        "validator_contract_recompute_drift",
    )
    require(
        load_json(terminal_path) == _terminal(
            config_path, contract_path, contract
        ),
        "validator_terminal_recompute_drift",
    )
    require(
        audit["producer_process_id"] == contract["process_id"]
        and audit["producer_process_id"] != audit["audit_process_id"]
        and audit["separate_process_verified"] is True,
        "validator_phase_isolation_drift",
    )
    for forbidden in (
        config["outputs"]["inventory"],
        config["outputs"]["frontier"],
    ):
        require(not (root / forbidden).exists(), f"forbidden_output_exists:{forbidden}")
    validation = {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "decision": "VALID",
        "terminal_state_recomputed": BLOCKED,
        "config_sha256": sha256_file(config_path),
        "contract_violation_sha256": sha256_file(contract_path),
        "terminal_sha256": sha256_file(terminal_path),
        "audit_sha256": sha256_file(audit_path),
        "validator_process_id": os.getpid(),
        "contract_gap_count_recomputed": len(contract["violations"]),
        "signal_scores_recomputed": 0,
        "inventory_absent": True,
        "frontier_absent": True,
        "candidate_authorized": False,
        "producer_authorized": False,
    }
    atomic_write_json(root / config["outputs"]["validation"], validation)
    return validation
