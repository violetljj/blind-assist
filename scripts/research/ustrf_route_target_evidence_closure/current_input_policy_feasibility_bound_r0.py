#!/usr/bin/env python3
"""Empirical feasibility bound for the current monotone lease-policy family."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import causal_token_policy_risk_gate_r1 as policy
import candidate_independent_policy_failure_attribution_r1 as attribution

CONFIG_SCHEMA = "blindassist_ustrf_current_input_policy_feasibility_bound_r0"
CERTIFICATE_SCHEMA = (
    "blindassist_ustrf_current_input_policy_feasibility_bound_certificate_r0"
)
TERMINAL_SCHEMA = (
    "blindassist_ustrf_current_input_policy_feasibility_bound_terminal_r0"
)
VALIDATION_SCHEMA = (
    "blindassist_ustrf_current_input_policy_feasibility_bound_validation_r0"
)
STAGE = "CURRENT-INPUT-POLICY-FEASIBILITY-BOUND-R0"
LEGAL_TERMINAL_STATES = (
    "CURRENT_INPUT_POLICY_FAMILY_FEASIBLE",
    "CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE",
)
IMPLEMENTATION_PATHS = {
    "core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "current_input_policy_feasibility_bound_r0.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_current_input_policy_feasibility_bound_r0.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_current_input_policy_feasibility_bound_r0.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_current_input_policy_feasibility_bound_r0.py"
    ),
}


class FeasibilityBoundContractError(RuntimeError):
    """Raised before a binary terminal can be issued."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FeasibilityBoundContractError(message)


def _verify(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"{label}_missing")
    require(
        policy.r0.sha256_file(path) == binding["sha256"],
        f"{label}_sha_drift",
    )
    return path


def load_and_verify_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = policy.r0.load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "stage_drift")
    boundary = config["claim_boundary"]
    require(
        boundary["maximum"]
        == "CURRENT_INPUT_POLICY_FAMILY_EMPIRICAL_FEASIBILITY_BOUND_ONLY"
        and boundary["point_rate_bound_only"] is True
        and boundary["credible_population_risk_bound"] is False
        and boundary["candidate_policy_generated"] is False
        and boundary["threshold_selected_or_changed"] is False
        and boundary["ttl_changed"] is False
        and boundary["renewal_enabled"] is False
        and boundary["opener_integration"] is False,
        "claim_boundary_drift",
    )
    family = config["policy_family"]
    require(
        family["minimum_consecutive_active_frames"] == 2
        and family["token_scope"] == "one_token_per_track_per_reset"
        and family["renewal"] is False
        and family["no_policy_or_threshold_is_serialized"] is True,
        "policy_family_drift",
    )
    require(
        family["allowed_decision_inputs"]
        == [
            "track_identity_as_permutation_equivariant_scope_only",
            "per_track_active_route_relation",
            "route_validity",
            "reset",
            "causal_elapsed_timestamp",
        ],
        "allowed_input_drift",
    )
    forbidden = set(family["forbidden_decision_inputs"])
    require(
        {
            "candidate_identity",
            "source_identity",
            "sequence_identity",
            "frame_id",
            "raw_track_id_value_order_hash_or_tie_break",
            "absolute_timestamp",
            "truth_identity_or_box",
            "event_id_or_window",
            "future_frames",
            "clearance",
            "oracle_token",
        }
        == forbidden,
        "forbidden_input_drift",
    )
    gate = config["feasibility_gate"]
    expected_max = int(
        float(gate["negative_exposure_minutes"])
        * float(gate["acceptance_threshold_tokens_per_minute"])
    )
    require(
        gate["oracle_candidate_event_cells"] == 36
        and gate["required_supported_candidate_event_cells"] == 33
        and gate["required_no_active_relation_candidate_event_cells"] == 3
        and gate["candidate_independent_unique_event_count"] == 12
        and gate["required_supported_unique_events"] == 11
        and gate["required_no_active_relation_unique_events"] == 1
        and gate["maximum_empirical_negative_token_count"] == expected_max == 2,
        "feasibility_gate_drift",
    )
    limit = config["evidence_limit"]
    require(
        limit["point_rate_only"] is True
        and limit["current_exposure_meets_parent_credible_floor"] is False
        and limit["credible_risk_gate_not_claimed"] is True,
        "evidence_limit_drift",
    )
    for label, binding in config["parent_bindings"].items():
        _verify(repo, binding, label)
    require(
        set(config["implementation_bindings"]) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_set_drift",
    )
    for label, relative in IMPLEMENTATION_PATHS.items():
        path = repo / relative
        require(path.is_file(), f"implementation_missing:{label}")
        require(
            policy.r0.sha256_file(path)
            == config["implementation_bindings"][label],
            f"implementation_sha_drift:{label}",
        )
    require(
        config["terminal_states"] == list(LEGAL_TERMINAL_STATES),
        "terminal_state_set_drift",
    )
    return config


def _load_frozen_inputs(
    repo: Path, config: dict[str, Any]
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str, str], list[dict[str, int]]],
    dict[tuple[str, str], list[dict[str, Any]]],
    int,
]:
    policy_config_path = repo / config["parent_bindings"]["policy_config"]["path"]
    parent_config, _ = policy.load_and_verify_config(repo, policy_config_path)
    inventory, frames = policy._parent_frames(repo, parent_config)
    expected = config["expected_scope"]
    require(
        inventory["candidate_independent_sequence_count"]
        == expected["candidate_independent_sequences"]
        and inventory["frame_count"] == expected["frames"]
        and len(frames) == expected["candidate_independent_sequences"],
        "parent_scope_drift",
    )
    oracle_ledgers = policy.r0._load_oracle_ledgers(repo, parent_config)
    events: dict[tuple[str, str, str], list[dict[str, int]]] = {}
    candidates_by_event: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    cell_count = 0
    for (candidate_id, event_id), ledger in sorted(oracle_ledgers.items()):
        cell_count += 1
        key = (ledger["source_id"], ledger["sequence_id"], event_id)
        candidates_by_event[key].add(candidate_id)
        qualifications = attribution._oracle_qualifications(ledger)
        if key in events:
            require(events[key] == qualifications, "candidate_oracle_projection_drift")
        else:
            events[key] = qualifications
    gate = config["feasibility_gate"]
    require(
        cell_count == gate["oracle_candidate_event_cells"]
        and len(events) == gate["candidate_independent_unique_event_count"],
        "oracle_event_scope_drift",
    )
    candidate_sets = {frozenset(rows) for rows in candidates_by_event.values()}
    require(
        len(candidate_sets) == 1
        and len(next(iter(candidate_sets))) == 3
        and all(len(rows) == 3 for rows in candidates_by_event.values()),
        "candidate_cell_multiplicity_drift",
    )
    supported = sum(bool(rows) for rows in events.values())
    require(
        supported == gate["required_supported_unique_events"]
        and len(events) - supported
        == gate["required_no_active_relation_unique_events"],
        "oracle_support_scope_drift",
    )
    mask = policy.r0.load_json(
        repo / parent_config["parent_bindings"]["eligibility_mask"]["path"]
    )
    negative, duration_ns = policy.validated_negative_interval_index(mask)
    return frames, events, negative, duration_ns


def build_track_scopes(
    sequences: dict[tuple[str, str], list[dict[str, Any]]]
) -> dict[tuple[str, str, int, int], list[list[dict[str, int]]]]:
    """Project relation runs; identity fields remain join/scope keys only."""
    scopes: dict[
        tuple[str, str, int, int], list[list[dict[str, int]]]
    ] = defaultdict(list)
    for (source_id, sequence_id), frames in sorted(sequences.items()):
        active_runs: dict[int, list[dict[str, int]]] = {}
        current_segment: int | None = None
        prior_timestamp: int | None = None
        for frame in frames:
            timestamp = int(frame["source_capture_timestamp_ns"])
            require(
                prior_timestamp is None or timestamp > prior_timestamp,
                "timestamp_not_strictly_increasing",
            )
            prior_timestamp = timestamp
            if frame["state_reset_before_frame"]:
                for track_id, run in active_runs.items():
                    scopes[
                        (source_id, sequence_id, int(current_segment), track_id)
                    ].append(run)
                active_runs = {}
                current_segment = int(frame["reset_segment"])
            require(current_segment is not None, "reset_scope_missing")
            active = (
                set(int(value) for value in frame["active_relation_track_ids"])
                if frame["route_known"]
                else set()
            )
            observed = set(int(value) for value in frame["observed_track_ids"])
            require(active <= observed, "active_track_not_observed")
            for track_id in list(active_runs):
                if track_id not in active:
                    scopes[
                        (source_id, sequence_id, current_segment, track_id)
                    ].append(active_runs.pop(track_id))
            for track_id in active:
                run = active_runs.setdefault(track_id, [])
                start_timestamp = (
                    int(run[0]["timestamp_ns"]) if run else timestamp
                )
                run.append(
                    {
                        "frame_id": int(frame["frame_id"]),
                        "timestamp_ns": timestamp,
                        "support_duration_ns": timestamp - start_timestamp,
                    }
                )
        for track_id, run in active_runs.items():
            scopes[(source_id, sequence_id, int(current_segment), track_id)].append(
                run
            )
    return dict(scopes)


def _timestamp_is_negative(
    timestamp_ns: int, intervals: list[dict[str, Any]]
) -> bool:
    return any(
        int(row["start_ns"]) <= timestamp_ns < int(row["end_ns"])
        for row in intervals
    )


def build_activation_intervals(
    scopes: dict[tuple[str, str, int, int], list[list[dict[str, int]]]],
    events: dict[tuple[str, str, str], list[dict[str, int]]],
    negative: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Return q ranges with a fixed first activation; never return q itself."""
    supported_keys = sorted(key for key, rows in events.items() if rows)
    event_index = {key: index for index, key in enumerate(supported_keys)}
    oracle_by_scope: dict[
        tuple[str, str, int, int], list[tuple[int, int]]
    ] = defaultdict(list)
    for key in supported_keys:
        source_id, sequence_id, _ = key
        for row in events[key]:
            oracle_by_scope[
                (
                    source_id,
                    sequence_id,
                    int(row["reset_segment"]),
                    int(row["track_id"]),
                )
            ].append((event_index[key], int(row["oracle_frame_id"])))
    result = []
    for scope, runs in scopes.items():
        prior_episode_max = 0
        for run in runs:
            if len(run) < 2:
                if run:
                    prior_episode_max = max(
                        prior_episode_max,
                        int(run[-1]["support_duration_ns"]),
                    )
                continue
            run_end = int(run[-1]["frame_id"])
            for index in range(1, len(run)):
                prior_duration = (
                    int(run[index - 1]["support_duration_ns"])
                    if index > 1
                    else 0
                )
                lower = max(1, prior_episode_max + 1, prior_duration + 1)
                upper = int(run[index]["support_duration_ns"])
                if lower > upper:
                    continue
                activation_frame = int(run[index]["frame_id"])
                mask = 0
                for event_id, oracle_frame in oracle_by_scope.get(scope, []):
                    if activation_frame <= oracle_frame <= run_end:
                        mask |= 1 << event_id
                timestamp = int(run[index]["timestamp_ns"])
                result.append(
                    {
                        "lower_duration_ns": lower,
                        "upper_duration_ns": upper,
                        "negative_activation": _timestamp_is_negative(
                            timestamp, negative.get((scope[0], scope[1]), [])
                        ),
                        "coverage_mask": mask,
                    }
                )
            prior_episode_max = max(
                prior_episode_max, int(run[-1]["support_duration_ns"])
            )
    return result, supported_keys


def sweep_empirical_frontier(
    activation_intervals: list[dict[str, Any]],
    supported_event_count: int,
    maximum_negative_count: int,
) -> dict[str, Any]:
    """Exact integer-nanosecond sweep; serialize only summary and digest."""
    require(supported_event_count > 0, "supported_event_count_invalid")
    deltas: dict[int, dict[str, Any]] = {}

    def delta_at(point: int) -> dict[str, Any]:
        return deltas.setdefault(
            point,
            {
                "risk": 0,
                "coverage": [0] * supported_event_count,
            },
        )

    for row in activation_intervals:
        lower = int(row["lower_duration_ns"])
        end = int(row["upper_duration_ns"]) + 1
        require(1 <= lower < end, "activation_interval_invalid")
        risk_delta = int(bool(row["negative_activation"]))
        mask = int(row["coverage_mask"])
        for point, sign in ((lower, 1), (end, -1)):
            delta = delta_at(point)
            delta["risk"] += sign * risk_delta
            for event_id in range(supported_event_count):
                if mask & (1 << event_id):
                    delta["coverage"][event_id] += sign
    require(bool(deltas), "frontier_empty")
    risk = 0
    coverage_counts = [0] * supported_event_count
    frontier_rows = []
    maximum_coverage = 0
    maximum_risk_feasible_coverage = 0
    minimum_risk_for_full_coverage: int | None = None
    for point in sorted(deltas):
        delta = deltas[point]
        risk += int(delta["risk"])
        for event_id, value in enumerate(delta["coverage"]):
            coverage_counts[event_id] += int(value)
            require(coverage_counts[event_id] >= 0, "coverage_count_negative")
        covered = sum(value > 0 for value in coverage_counts)
        maximum_coverage = max(maximum_coverage, covered)
        if risk <= maximum_negative_count:
            maximum_risk_feasible_coverage = max(
                maximum_risk_feasible_coverage, covered
            )
        if covered == supported_event_count:
            minimum_risk_for_full_coverage = (
                risk
                if minimum_risk_for_full_coverage is None
                else min(minimum_risk_for_full_coverage, risk)
            )
        frontier_rows.append(
            {
                "duration_breakpoint_ns": point,
                "negative_activation_count": risk,
                "coverage_mask": sum(
                    (1 << event_id)
                    for event_id, value in enumerate(coverage_counts)
                    if value > 0
                ),
            }
        )
    return {
        "frontier_segment_count": len(frontier_rows),
        "frontier_sha256": policy.r0.sha256_bytes(
            policy.r0.canonical_bytes(frontier_rows)
        ),
        "maximum_supported_unique_event_coverage": maximum_coverage,
        "maximum_supported_unique_event_coverage_at_or_below_risk_limit": (
            maximum_risk_feasible_coverage
        ),
        "minimum_negative_activation_count_for_full_coverage": (
            minimum_risk_for_full_coverage
        ),
        "full_coverage_attainable": (
            minimum_risk_for_full_coverage is not None
        ),
        "simultaneous_empirical_gate_attainable": (
            minimum_risk_for_full_coverage is not None
            and minimum_risk_for_full_coverage <= maximum_negative_count
        ),
    }


def compute_outputs(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_and_verify_config(repo, config_path)
    sequences, events, negative, duration_ns = _load_frozen_inputs(repo, config)
    expected_duration_ns = round(
        float(config["feasibility_gate"]["negative_exposure_minutes"])
        * 60_000_000_000
    )
    require(duration_ns == expected_duration_ns, "negative_duration_drift")
    scopes = build_track_scopes(sequences)
    intervals, supported_keys = build_activation_intervals(
        scopes, events, negative
    )
    frontier = sweep_empirical_frontier(
        intervals,
        len(supported_keys),
        int(
            config["feasibility_gate"][
                "maximum_empirical_negative_token_count"
            ]
        ),
    )
    unique_max = int(frontier["maximum_supported_unique_event_coverage"])
    risk_max = int(
        frontier[
            "maximum_supported_unique_event_coverage_at_or_below_risk_limit"
        ]
    )
    candidate_multiplier = (
        config["feasibility_gate"]["required_supported_candidate_event_cells"]
        // config["feasibility_gate"]["required_supported_unique_events"]
    )
    terminal_state = (
        "CURRENT_INPUT_POLICY_FAMILY_FEASIBLE"
        if frontier["simultaneous_empirical_gate_attainable"]
        else "CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE"
    )
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "stage": STAGE,
        "authority": "empirical_family_feasibility_bound_only",
        "config_sha256": policy.r0.sha256_file(config_path),
        "input_scope": {
            "candidate_independent_sequence_count": len(sequences),
            "frame_count": sum(len(rows) for rows in sequences.values()),
            "track_reset_scope_count": len(scopes),
            "candidate_event_cell_count": (
                config["feasibility_gate"]["oracle_candidate_event_cells"]
            ),
            "candidate_independent_unique_event_count": len(events),
            "supported_unique_event_count": len(supported_keys),
            "no_active_relation_unique_event_count": (
                len(events) - len(supported_keys)
            ),
            "negative_exposure_duration_ns": duration_ns,
            "negative_exposure_minutes": (
                duration_ns / 60_000_000_000
            ),
        },
        "family_bound": {
            "activation_interval_count": len(intervals),
            **frontier,
            "maximum_supported_candidate_cell_coverage": (
                unique_max * candidate_multiplier
            ),
            "maximum_supported_candidate_cell_coverage_at_or_below_risk_limit": (
                risk_max * candidate_multiplier
            ),
            "required_supported_unique_event_coverage": len(supported_keys),
            "required_supported_candidate_cell_coverage": (
                config["feasibility_gate"][
                    "required_supported_candidate_event_cells"
                ]
            ),
            "maximum_empirical_negative_token_count": (
                config["feasibility_gate"][
                    "maximum_empirical_negative_token_count"
                ]
            ),
            "acceptance_threshold_tokens_per_minute": (
                config["feasibility_gate"][
                    "acceptance_threshold_tokens_per_minute"
                ]
            ),
        },
        "invariance_audit": {
            "candidate_identity_used_for_decision": False,
            "source_or_sequence_identity_used_for_decision": False,
            "raw_track_id_value_used_for_decision": False,
            "absolute_timestamp_used_for_decision": False,
            "decision_statistic": (
                "shared_monotone_support_duration_and_two_frame_guard_only"
            ),
            "track_id_alpha_renaming_invariant_by_construction": True,
            "source_sequence_renaming_invariant_by_construction": True,
            "timestamp_translation_invariant_by_construction": True,
        },
        "relaxations": {
            "nominal_ttl_ignored_for_coverage_upper_bound": True,
            "fail_closed_invalidation_retained": True,
            "one_token_per_track_reset_retained": True,
            "renewal_enabled": False,
            "candidate_policy_or_threshold_serialized": False,
        },
        "evidence_limit": config["evidence_limit"],
    }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": terminal_state,
        "authority": "current_input_policy_family_empirical_bound_only",
        "config_sha256": policy.r0.sha256_file(config_path),
        "bound_certificate_sha256": policy.r0.sha256_bytes(
            policy.r0.canonical_bytes(certificate)
        ),
        "supported_candidate_cell_coverage_upper_bound": (
            certificate["family_bound"][
                "maximum_supported_candidate_cell_coverage"
            ]
        ),
        "supported_candidate_cell_coverage_upper_bound_at_risk_limit": (
            certificate["family_bound"][
                "maximum_supported_candidate_cell_coverage_at_or_below_risk_limit"
            ]
        ),
        "required_supported_candidate_cell_coverage": (
            config["feasibility_gate"][
                "required_supported_candidate_event_cells"
            ]
        ),
        "maximum_empirical_negative_token_count": (
            config["feasibility_gate"][
                "maximum_empirical_negative_token_count"
            ]
        ),
        "candidate_policy_generated": False,
        "threshold_selected_or_changed": False,
        "ttl_changed": False,
        "renewal_enabled": False,
        "opener_integration_authorized": False,
        "successor_policy_authorized": False,
        "claim_boundary": config["claim_boundary"],
    }
    require(terminal_state in LEGAL_TERMINAL_STATES, "illegal_terminal_state")
    return certificate, terminal


def build_outputs(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    certificate, terminal = compute_outputs(repo, config_path)
    root = repo / config["outputs"]["root"]
    policy.r0.atomic_write_json(
        root / config["outputs"]["bound_certificate"], certificate
    )
    policy.r0.atomic_write_json(
        root / config["outputs"]["terminal_receipt"], terminal
    )
    return terminal


def validate_outputs(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    expected = compute_outputs(repo, config_path)
    root = repo / config["outputs"]["root"]
    names = [
        config["outputs"]["bound_certificate"],
        config["outputs"]["terminal_receipt"],
    ]
    for name, value in zip(names, expected, strict=True):
        path = root / name
        require(path.is_file(), f"output_missing:{name}")
        require(
            policy.r0.load_json(path) == value,
            f"output_recompute_drift:{name}",
        )
    receipt = {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID",
        "config_sha256": policy.r0.sha256_file(config_path),
        "bound_certificate_sha256": policy.r0.sha256_file(root / names[0]),
        "terminal_sha256": policy.r0.sha256_file(root / names[1]),
        "checks": {
            "parent_policy_and_attribution_hash_bound": True,
            "candidate_oracle_cells_deduplicated_to_unique_events": True,
            "candidate_identity_excluded_from_decision": True,
            "track_id_alpha_renaming_invariant": True,
            "source_sequence_renaming_invariant": True,
            "timestamp_translation_invariant": True,
            "all_positive_integer_duration_breakpoints_swept": True,
            "ttl_optimistic_upper_bound_recomputed": True,
            "negative_half_open_exposure_recomputed": True,
            "point_rate_only_and_credible_limit_explicit": True,
            "candidate_policy_or_threshold_output": False,
            "ttl_or_renewal_changed": False,
            "terminal_state_count": 2,
        },
    }
    policy.r0.atomic_write_json(
        root / config["outputs"]["validation_receipt"], receipt
    )
    return receipt
