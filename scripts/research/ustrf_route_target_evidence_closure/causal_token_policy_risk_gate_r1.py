#!/usr/bin/env python3
"""Candidate-independent causal-token policy and risk gate R1.

The producer phase reads only the already-frozen truth-blind R0 sequence
ledgers, discards their R0 token decisions, and emits a new policy inventory.
Oracle/event and negative-exposure evidence are decoded only by a later audit
process after every policy ledger has been persisted and hash-inventoried.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import causal_per_track_attribution_token_audit_r0 as r0


CONFIG_SCHEMA = "blindassist_ustrf_causal_token_policy_risk_gate_r1"
LEDGER_SCHEMA = "blindassist_ustrf_candidate_independent_causal_token_policy_ledger_r1"
INVENTORY_SCHEMA = "blindassist_ustrf_candidate_independent_causal_token_policy_inventory_r1"
RISK_SCHEMA = "blindassist_ustrf_causal_token_policy_risk_ledger_r1"
REPEAT_SCHEMA = "blindassist_ustrf_causal_token_policy_repeat_ledger_r1"
TERMINAL_SCHEMA = "blindassist_ustrf_causal_token_policy_risk_terminal_r1"
VALIDATION_SCHEMA = "blindassist_ustrf_causal_token_policy_risk_validation_r1"
STAGE = "CANDIDATE-INDEPENDENT-CAUSAL-TOKEN-POLICY-RISK-GATE-R1"

IMPLEMENTATION_PATHS = {
    "policy_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "causal_token_policy_risk_gate_r1.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_causal_token_policy_risk_gate_r1.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_causal_token_policy_risk_gate_r1.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_causal_token_policy_risk_gate_r1.py"
    ),
}


class PolicyGateContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyGateContractError(message)


def _verify_file(repo: Path, binding: dict[str, Any], label: str) -> str:
    path = repo / binding["path"]
    require(path.is_file(), f"{label}_missing")
    actual = r0.sha256_file(path)
    require(actual == binding["sha256"], f"{label}_sha_drift")
    return actual


def load_and_verify_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    config = r0.load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(
        config.get("status") == "frozen_before_any_r1_output",
        "config_not_preoutput_frozen",
    )
    policy = config["policy"]
    require(
        policy["allowed_inputs"]
        == [
            "detector_track_identity",
            "per_track_route_relation",
            "route_validity",
            "reset",
            "causal_timestamp",
        ],
        "policy_allowed_inputs_drift",
    )
    require(
        policy["forbidden_inputs"]
        == [
            "candidate_identity",
            "truth_identity_or_box",
            "event_id_or_window",
            "future_frames",
            "clearance",
            "oracle_token",
        ],
        "policy_forbidden_inputs_drift",
    )
    require(
        int(policy["minimum_consecutive_active_frames"]) == 2,
        "policy_min_frames_drift",
    )
    require(
        int(policy["minimum_active_relation_duration_ns"]) == 500_000_000,
        "policy_min_duration_drift",
    )
    require(
        int(policy["maximum_token_ttl_ns"]) == 500_000_000,
        "policy_ttl_drift",
    )
    require(
        policy["invalidation"]
        == [
            "reset_before_frame",
            "route_unknown",
            "track_unobserved",
            "active_relation_gap",
            "ttl_elapsed",
            "sequence_end",
        ],
        "policy_invalidation_drift",
    )
    require(
        policy["requalification"]
        == "one_token_per_track_per_reset_suppress_and_ledger_fresh_support_run_requalification_only",
        "policy_requalification_drift",
    )
    require(
        policy["continuous_active_after_ttl"]
        == "no_renewal_attempt_support_run_remains_qualified_until_gap",
        "policy_continuous_active_semantics_drift",
    )
    require(
        config["phase_order"]
        == [
            "verify_parent_r0_truth_blind_inventory_and_all_ledger_hashes",
            "project_only_candidate_independent_causal_track_route_reset_timestamp_facts",
            "apply_frozen_policy_to_all_41_full_sequences",
            "persist_and_hash_inventory_all_policy_ledgers",
            "reverify_all_policy_ledgers_in_a_separate_process",
            "only_then_decode_and_join_frozen_oracle_and_complete_negative_exposure",
            "apply_predeclared_coverage_integrity_and_risk_terminal_rules",
        ],
        "phase_order_drift",
    )
    boundary = config["claim_boundary"]
    require(
        boundary["maximum"]
        == "CANDIDATE_INDEPENDENT_TOKEN_POLICY_AND_RISK_AUDIT_ONLY"
        and boundary["candidate_independent"] is True
        and boundary["truth_blind_policy_producer"] is True
        and boundary["full_sequence_before_truth_or_negative_join"] is True,
        "claim_boundary_positive_drift",
    )
    for name in (
        "opener_integration",
        "c1_c3_modified_or_rerun",
        "clearance_change",
        "candidate_comparison",
        "winner_or_ranking",
        "selection",
        "l2_or_l3",
        "android_shadow",
        "h2",
        "human_outcome",
        "independent_walking_safety",
        "production",
        "new_data",
        "training",
    ):
        require(boundary[name] is False, f"claim_boundary_opened:{name}")
    coverage = config["coverage_gate"]
    require(
        coverage
        == {
            "oracle_candidate_event_cells": 36,
            "required_supported_cell_coverage": 33,
            "required_no_active_relation_cells": 3,
            "coverage_requires_oracle_qualification_within_policy_token_validity": True,
            "posthoc_only": True,
        },
        "coverage_gate_drift",
    )
    risk = config["risk_gate"]
    require(
        float(risk["acceptance_threshold_tokens_per_minute"]) == 0.5,
        "risk_threshold_drift",
    )
    require(float(risk["confidence_level"]) == 0.95, "risk_confidence_drift")
    require(
        float(risk["minimum_zero_event_exposure_minutes"]) >= 5.9915,
        "risk_exposure_floor_weakened",
    )
    require(
        risk["inference"]
        == "poisson_working_model_plus_cluster_bound_required",
        "risk_inference_drift",
    )
    require(
        risk["cluster_unit"] == "source_sequence",
        "risk_cluster_unit_drift",
    )
    require(
        risk["risk_event"]
        == "candidate_independent_policy_token_activation_whose_qualification_timestamp_falls_in_any_frozen_half_open_negative_exposure_interval"
        and risk["denominator"]
        == "complete_unique_half_open_negative_exposure_interval_ns"
        and risk["poisson_upper_bound"] == "one_sided_exact_chi_square"
        and risk["cluster_stratification"] == "source_id"
        and risk["worst_source_multiple_comparison_correction"]
        == "bonferroni"
        and risk["cluster_bootstrap_statistic"]
        == "max_source_rate_per_iteration"
        and risk["worst_source_upper_bound_required"] is True
        and risk["zero_over_zero_pass_forbidden"] is True,
        "risk_contract_drift",
    )
    require(
        risk["observed_point_rate_above_threshold"]
        == "pooled_or_any_source_above_threshold_is_POLICY_RISK_REJECT",
        "risk_point_reject_semantics_drift",
    )
    require(
        int(risk["minimum_sequences_per_source"]) >= 3,
        "risk_cluster_floor_weakened",
    )
    require(
        int(risk["cluster_bootstrap_iterations"]) >= 10_000,
        "risk_cluster_iterations_weakened",
    )
    require(
        isinstance(risk["cluster_bootstrap_seed"], int),
        "risk_cluster_seed_invalid",
    )
    require(
        config["terminal_precedence"]
        == [
            "POLICY_INTEGRITY_REJECT",
            "POLICY_COVERAGE_REJECT",
            "POLICY_RISK_REJECT",
            "HOLD_FOR_CREDIBLE_RISK_BOUND",
            "POLICY_RISK_GATE_PASSED",
        ],
        "terminal_precedence_drift",
    )
    expected = config["expected_scope"]
    require(
        expected
        == {
            "candidate_independent_sequences": 41,
            "frames": 62_229,
            "candidate_or_fsm_executions": 0,
            "truth_decodes_before_inventory_freeze": 0,
            "event_window_decodes_before_inventory_freeze": 0,
            "oracle_token_decodes_before_inventory_freeze": 0,
        },
        "expected_scope_drift",
    )
    bindings: dict[str, str] = {}
    for name, binding in config["parent_bindings"].items():
        bindings[name] = _verify_file(repo, binding, f"parent_{name}")
    for name, rel_path in IMPLEMENTATION_PATHS.items():
        actual = r0.sha256_file(repo / rel_path)
        require(
            actual == config["implementation_bindings"][name],
            f"implementation_{name}_drift",
        )
        bindings[name] = actual
    bindings["config_sha256"] = r0.sha256_file(config_path)
    return config, bindings


def _parent_frames(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    parent_config_path = repo / config["parent_bindings"]["r0_config"]["path"]
    parent_config, parent_bindings = r0.load_and_verify_config(
        repo, parent_config_path
    )
    inventory, ledgers = r0.load_and_verify_blind_inventory(
        repo, parent_config, parent_bindings
    )
    projected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, ledger in ledgers.items():
        frames = []
        for row in ledger["frames"]:
            frame = {name: row[name] for name in r0.ALLOWED_FRAME_KEYS}
            r0.assert_producer_frame(frame)
            frames.append(frame)
        projected[key] = frames
    return inventory, projected


def _token_id(
    source_id: str, sequence_id: str, reset_segment: int, track_id: int
) -> str:
    return (
        f"{source_id}::{sequence_id}::reset-{reset_segment}::"
        f"track-{track_id}::policy-r1"
    )


def produce_policy_ledger(
    source_id: str,
    sequence_id: str,
    frames: list[dict[str, Any]],
    *,
    minimum_consecutive_active_frames: int,
    minimum_active_relation_duration_ns: int,
    maximum_token_ttl_ns: int,
) -> dict[str, Any]:
    require(bool(frames), "policy_sequence_empty")
    require(minimum_consecutive_active_frames >= 2, "policy_min_frames_invalid")
    require(minimum_active_relation_duration_ns > 0, "policy_duration_invalid")
    require(maximum_token_ttl_ns > 0, "policy_ttl_invalid")
    supports: dict[int, dict[str, int]] = {}
    emitted: dict[int, dict[str, Any]] = {}
    active_tokens: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    prior_frame_id: int | None = None
    prior_timestamp_ns: int | None = None
    current_segment: int | None = None
    token_count = repeat_count = expiration_count = 0

    def expire(
        track_id: int,
        *,
        reason: str,
        effective_timestamp_ns: int,
        last_valid_frame_id: int,
        sink: list[dict[str, Any]],
    ) -> None:
        nonlocal expiration_count
        token = active_tokens.pop(track_id)
        token["effective_valid_until_timestamp_ns"] = min(
            int(token["nominal_valid_until_timestamp_ns"]),
            int(effective_timestamp_ns),
        )
        token["last_valid_frame_id"] = int(last_valid_frame_id)
        token["invalidation_reason"] = reason
        sink.append(
            {
                "token_id": token["token_id"],
                "track_id": track_id,
                "reset_segment": token["reset_segment"],
                "invalidation_reason": reason,
                "effective_valid_until_timestamp_ns": token[
                    "effective_valid_until_timestamp_ns"
                ],
                "last_valid_frame_id": token["last_valid_frame_id"],
            }
        )
        expiration_count += 1

    for index, frame in enumerate(frames):
        r0.assert_producer_frame(frame)
        require(
            frame["source_id"] == source_id
            and frame["sequence_id"] == sequence_id,
            "policy_sequence_identity_drift",
        )
        timestamp_ns = int(frame["source_capture_timestamp_ns"])
        frame_id = int(frame["frame_id"])
        expirations: list[dict[str, Any]] = []
        if index == 0:
            require(
                frame["state_reset_before_frame"],
                "policy_first_frame_missing_reset",
            )
        if frame["state_reset_before_frame"]:
            for track_id in sorted(active_tokens):
                expire(
                    track_id,
                    reason="reset_before_frame",
                    effective_timestamp_ns=timestamp_ns,
                    last_valid_frame_id=(
                        prior_frame_id if prior_frame_id is not None else frame_id
                    ),
                    sink=expirations,
                )
            supports.clear()
            emitted.clear()
            current_segment = int(frame["reset_segment"])
        else:
            require(current_segment is not None, "policy_reset_state_missing")
            require(
                int(frame["reset_segment"]) == current_segment,
                "policy_segment_changed_without_reset",
            )
            require(
                prior_frame_id is not None and frame_id == prior_frame_id + 1,
                "policy_frame_gap_without_reset",
            )
        require(
            prior_timestamp_ns is None or timestamp_ns > prior_timestamp_ns,
            "policy_timestamp_not_strictly_increasing",
        )
        active = (
            set(frame["active_relation_track_ids"])
            if frame["route_known"]
            else set()
        )
        observed = set(frame["observed_track_ids"])
        for track_id in sorted(list(active_tokens)):
            token = active_tokens[track_id]
            reason = None
            if not frame["route_known"]:
                reason = "route_unknown"
            elif track_id not in observed:
                reason = "track_unobserved"
            elif track_id not in active:
                reason = "active_relation_gap"
            elif timestamp_ns >= int(token["nominal_valid_until_timestamp_ns"]):
                reason = "ttl_elapsed"
            if reason is not None:
                effective_ns = (
                    int(token["nominal_valid_until_timestamp_ns"])
                    if reason == "ttl_elapsed"
                    else timestamp_ns
                )
                expire(
                    track_id,
                    reason=reason,
                    effective_timestamp_ns=effective_ns,
                    last_valid_frame_id=(
                        prior_frame_id if prior_frame_id is not None else frame_id
                    ),
                    sink=expirations,
                )
        for track_id in list(supports):
            if track_id not in active:
                del supports[track_id]
        activations: list[dict[str, Any]] = []
        repeats: list[dict[str, Any]] = []
        for track_id in sorted(active):
            if track_id not in supports:
                supports[track_id] = {
                    "start_frame_id": frame_id,
                    "start_timestamp_ns": timestamp_ns,
                    "frame_count": 0,
                }
            support = supports[track_id]
            support["frame_count"] += 1
            duration_ns = timestamp_ns - int(support["start_timestamp_ns"])
            qualifies = (
                int(support["frame_count"])
                >= minimum_consecutive_active_frames
                and duration_ns >= minimum_active_relation_duration_ns
            )
            if not qualifies or support.get("qualification_recorded"):
                continue
            support["qualification_recorded"] = 1
            activation = {
                "token_id": _token_id(
                    source_id,
                    sequence_id,
                    int(frame["reset_segment"]),
                    track_id,
                ),
                "track_id": track_id,
                "reset_segment": int(frame["reset_segment"]),
                "qualification_frame_id": frame_id,
                "qualification_timestamp_ns": timestamp_ns,
                "support_start_frame_id": int(support["start_frame_id"]),
                "support_start_timestamp_ns": int(
                    support["start_timestamp_ns"]
                ),
                "support_frame_count": int(support["frame_count"]),
                "support_duration_ns": duration_ns,
                "nominal_valid_until_timestamp_ns": (
                    timestamp_ns + maximum_token_ttl_ns
                ),
                "effective_valid_until_timestamp_ns": None,
                "last_valid_frame_id": None,
                "invalidation_reason": None,
            }
            if track_id in emitted:
                repeat_count += 1
                attempt_id = (
                    f"{activation['token_id']}::requal-frame-{frame_id}"
                )
                repeats.append(
                    {
                        "requalification_attempt_id": attempt_id,
                        "original_token_id": emitted[track_id]["token_id"],
                        "original_qualification_frame_id": emitted[track_id][
                            "qualification_frame_id"
                        ],
                        "track_id": track_id,
                        "reset_segment": int(frame["reset_segment"]),
                        "qualification_frame_id": frame_id,
                        "qualification_timestamp_ns": timestamp_ns,
                        "support_start_frame_id": int(
                            support["start_frame_id"]
                        ),
                        "support_start_timestamp_ns": int(
                            support["start_timestamp_ns"]
                        ),
                        "support_frame_count": int(support["frame_count"]),
                        "support_duration_ns": duration_ns,
                        "suppressed": True,
                    }
                )
            else:
                emitted[track_id] = activation
                active_tokens[track_id] = activation
                activations.append(activation)
                token_count += 1
        rows.append(
            {
                **frame,
                "token_activations": activations,
                "token_invalidations": expirations,
                "requalifications_suppressed": repeats,
            }
        )
        prior_frame_id = frame_id
        prior_timestamp_ns = timestamp_ns
    final_expirations: list[dict[str, Any]] = []
    for track_id in sorted(list(active_tokens)):
        expire(
            track_id,
            reason="sequence_end",
            effective_timestamp_ns=int(prior_timestamp_ns),
            last_valid_frame_id=int(prior_frame_id),
            sink=final_expirations,
        )
    rows[-1]["token_invalidations"].extend(final_expirations)
    require(
        expiration_count == token_count,
        "policy_token_invalidation_cardinality_drift",
    )
    return {
        "schema": LEDGER_SCHEMA,
        "stage": STAGE,
        "authority": "candidate_independent_policy_risk_audit_only",
        "source_id": source_id,
        "sequence_id": sequence_id,
        "frame_count": len(rows),
        "token_count": token_count,
        "token_invalidation_count": expiration_count,
        "requalification_suppressed_count": repeat_count,
        "policy": {
            "minimum_consecutive_active_frames": (
                minimum_consecutive_active_frames
            ),
            "minimum_active_relation_duration_ns": (
                minimum_active_relation_duration_ns
            ),
            "maximum_token_ttl_ns": maximum_token_ttl_ns,
            "one_token_per_track_per_reset": True,
        },
        "frames": rows,
    }


def _ledger_path(root: str, source_id: str, sequence_id: str) -> Path:
    suffix = r0.sha256_bytes(
        r0.canonical_bytes([source_id, sequence_id])
    )[:16]
    return (
        Path(root)
        / "policy-ledgers"
        / f"{source_id}__{suffix}.json"
    )


def build_and_freeze_policy_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    parent_inventory, sequences = _parent_frames(repo, config)
    expected = config["expected_scope"]
    require(
        len(sequences) == expected["candidate_independent_sequences"],
        "policy_sequence_count_drift",
    )
    policy = config["policy"]
    inventory_rows = []
    total_frames = total_tokens = total_invalidations = total_repeats = 0
    for (source_id, sequence_id), frames in sorted(sequences.items()):
        ledger = produce_policy_ledger(
            source_id,
            sequence_id,
            frames,
            minimum_consecutive_active_frames=int(
                policy["minimum_consecutive_active_frames"]
            ),
            minimum_active_relation_duration_ns=int(
                policy["minimum_active_relation_duration_ns"]
            ),
            maximum_token_ttl_ns=int(policy["maximum_token_ttl_ns"]),
        )
        rel_path = _ledger_path(
            config["outputs"]["root"], source_id, sequence_id
        )
        r0.atomic_write_json(repo / rel_path, ledger)
        inventory_rows.append(
            {
                "source_id": source_id,
                "sequence_id": sequence_id,
                "path": rel_path.as_posix(),
                "sha256": r0.sha256_file(repo / rel_path),
                "frame_count": ledger["frame_count"],
                "token_count": ledger["token_count"],
                "token_invalidation_count": ledger[
                    "token_invalidation_count"
                ],
                "requalification_suppressed_count": ledger[
                    "requalification_suppressed_count"
                ],
            }
        )
        total_frames += ledger["frame_count"]
        total_tokens += ledger["token_count"]
        total_invalidations += ledger["token_invalidation_count"]
        total_repeats += ledger["requalification_suppressed_count"]
    require(total_frames == expected["frames"], "policy_frame_count_drift")
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "stage": STAGE,
        "status": "FULL_SEQUENCE_CANDIDATE_INDEPENDENT_POLICY_FROZEN",
        "authority": "policy_risk_audit_only",
        "bindings": bindings,
        "parent_r0_inventory_sha256": r0.sha256_file(
            repo / config["parent_bindings"]["r0_inventory"]["path"]
        ),
        "candidate_independent_sequence_count": len(inventory_rows),
        "frame_count": total_frames,
        "token_count": total_tokens,
        "token_invalidation_count": total_invalidations,
        "requalification_suppressed_count": total_repeats,
        "truth_payloads_decoded": 0,
        "event_windows_decoded": 0,
        "oracle_tokens_decoded": 0,
        "parent_candidate_projection_count_verified": parent_inventory[
            "candidate_projection_count_verified"
        ],
        "inventory": inventory_rows,
    }
    path = repo / config["outputs"]["root"] / config["outputs"]["inventory"]
    r0.atomic_write_json(path, inventory)
    return inventory


def load_and_verify_policy_inventory(
    repo: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    path = repo / config["outputs"]["root"] / config["outputs"]["inventory"]
    require(path.is_file(), "policy_inventory_missing")
    inventory = r0.load_json(path)
    require(inventory.get("schema") == INVENTORY_SCHEMA, "inventory_schema_drift")
    require(
        inventory.get("status")
        == "FULL_SEQUENCE_CANDIDATE_INDEPENDENT_POLICY_FROZEN",
        "inventory_not_frozen",
    )
    require(inventory.get("bindings") == bindings, "inventory_binding_drift")
    require(
        inventory.get("parent_r0_inventory_sha256")
        == config["parent_bindings"]["r0_inventory"]["sha256"],
        "inventory_parent_r0_binding_drift",
    )
    require(
        inventory.get("truth_payloads_decoded") == 0
        and inventory.get("event_windows_decoded") == 0
        and inventory.get("oracle_tokens_decoded") == 0,
        "inventory_truth_ordering_violation",
    )
    ledgers = {}
    token_ids: set[str] = set()
    token_records: dict[str, dict[str, Any]] = {}
    invalidation_ids: set[str] = set()
    requalification_attempt_ids: set[str] = set()
    totals = defaultdict(int)
    require(
        len(inventory["inventory"])
        == config["expected_scope"]["candidate_independent_sequences"],
        "policy_inventory_sequence_count_drift",
    )
    for row in inventory["inventory"]:
        ledger_path = repo / row["path"]
        require(ledger_path.is_file(), "policy_ledger_missing")
        require(
            r0.sha256_file(ledger_path) == row["sha256"],
            "policy_ledger_sha_drift",
        )
        ledger = r0.load_json(ledger_path)
        require(ledger.get("schema") == LEDGER_SCHEMA, "ledger_schema_drift")
        key = (ledger["source_id"], ledger["sequence_id"])
        require(key not in ledgers, "policy_ledger_duplicate")
        require(
            row["source_id"] == ledger["source_id"]
            and row["sequence_id"] == ledger["sequence_id"],
            "policy_inventory_ledger_identity_drift",
        )
        require(
            Path(row["path"])
            == _ledger_path(
                config["outputs"]["root"],
                ledger["source_id"],
                ledger["sequence_id"],
            ),
            "policy_ledger_path_drift",
        )
        require(
            ledger["frame_count"] == len(ledger["frames"]),
            "policy_ledger_frame_count_drift",
        )
        counts = defaultdict(int)
        for frame in ledger["frames"]:
            for token in frame["token_activations"]:
                require(
                    token["token_id"] not in token_ids,
                    "policy_duplicate_token_id",
                )
                token_ids.add(token["token_id"])
                token_records[token["token_id"]] = token
                require(
                    token["effective_valid_until_timestamp_ns"] is not None
                    and token["last_valid_frame_id"] is not None
                    and token["invalidation_reason"] is not None,
                    "policy_token_not_terminalized",
                )
                counts["token"] += 1
            for invalidation in frame["token_invalidations"]:
                token_id = invalidation["token_id"]
                require(
                    token_id not in invalidation_ids,
                    "policy_duplicate_token_invalidation",
                )
                invalidation_ids.add(token_id)
                require(
                    token_id in token_ids,
                    "policy_invalidation_without_activation",
                )
                activation = token_records[token_id]
                require(
                    invalidation["track_id"] == activation["track_id"]
                    and invalidation["reset_segment"]
                    == activation["reset_segment"]
                    and invalidation["invalidation_reason"]
                    == activation["invalidation_reason"]
                    and invalidation["effective_valid_until_timestamp_ns"]
                    == activation["effective_valid_until_timestamp_ns"]
                    and invalidation["last_valid_frame_id"]
                    == activation["last_valid_frame_id"],
                    "policy_invalidation_activation_linkage_drift",
                )
                counts["invalidation"] += 1
            for repeat in frame["requalifications_suppressed"]:
                attempt_id = repeat["requalification_attempt_id"]
                require(
                    attempt_id not in requalification_attempt_ids,
                    "policy_duplicate_requalification_attempt_id",
                )
                requalification_attempt_ids.add(attempt_id)
                require(
                    repeat["original_token_id"] in token_ids
                    and repeat["suppressed"] is True
                    and "token_id" not in repeat,
                    "policy_requalification_linkage_drift",
                )
                original = token_records[repeat["original_token_id"]]
                require(
                    repeat["track_id"] == original["track_id"]
                    and repeat["reset_segment"] == original["reset_segment"]
                    and repeat["original_qualification_frame_id"]
                    == original["qualification_frame_id"]
                    and attempt_id
                    == (
                        f"{repeat['original_token_id']}::requal-frame-"
                        f"{repeat['qualification_frame_id']}"
                    ),
                    "policy_requalification_original_linkage_drift",
                )
                counts["repeat"] += 1
        require(
            counts["token"] == ledger["token_count"] == row["token_count"],
            "policy_token_count_drift",
        )
        require(
            counts["invalidation"]
            == ledger["token_invalidation_count"]
            == row["token_invalidation_count"],
            "policy_invalidation_count_drift",
        )
        require(
            counts["repeat"]
            == ledger["requalification_suppressed_count"]
            == row["requalification_suppressed_count"],
            "policy_repeat_count_drift",
        )
        totals["frame"] += ledger["frame_count"]
        totals["token"] += counts["token"]
        totals["invalidation"] += counts["invalidation"]
        totals["repeat"] += counts["repeat"]
        ledgers[key] = ledger
    require(totals["frame"] == inventory["frame_count"], "inventory_frame_drift")
    require(
        totals["frame"] == config["expected_scope"]["frames"],
        "inventory_expected_frame_drift",
    )
    require(totals["token"] == inventory["token_count"], "inventory_token_drift")
    require(
        totals["invalidation"] == inventory["token_invalidation_count"],
        "inventory_invalidation_drift",
    )
    require(
        totals["repeat"] == inventory["requalification_suppressed_count"],
        "inventory_repeat_drift",
    )
    require(
        invalidation_ids == token_ids,
        "policy_activation_invalidation_set_drift",
    )
    return inventory, ledgers


def poisson_one_sided_upper_rate(
    count: int, exposure_minutes: float, confidence_level: float
) -> float | None:
    require(count >= 0, "poisson_count_invalid")
    require(0 < confidence_level < 1, "poisson_confidence_invalid")
    if exposure_minutes <= 0:
        return None
    from scipy.stats import chi2

    upper_count = 0.5 * float(
        chi2.ppf(confidence_level, 2 * (count + 1))
    )
    return upper_count / exposure_minutes


def clustered_rate_upper_bound(
    sessions: list[tuple[int, float]],
    confidence_level: float,
    *,
    iterations: int,
    seed: int,
) -> float | None:
    require(0 < confidence_level < 1, "cluster_confidence_invalid")
    require(iterations >= 1, "cluster_iterations_invalid")
    if not sessions or any(count < 0 or minutes <= 0 for count, minutes in sessions):
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(iterations):
        sampled = [sessions[rng.randrange(len(sessions))] for _ in sessions]
        count = sum(row[0] for row in sampled)
        minutes = sum(row[1] for row in sampled)
        rates.append(count / minutes)
    rates.sort()
    index = min(
        len(rates) - 1,
        max(0, math.ceil(confidence_level * len(rates)) - 1),
    )
    return rates[index]


def stratified_worst_source_cluster_upper_bound(
    source_sessions: dict[str, list[tuple[int, float]]],
    confidence_level: float,
    *,
    iterations: int,
    seed: int,
    minimum_sessions_per_source: int,
) -> float | None:
    require(0 < confidence_level < 1, "cluster_confidence_invalid")
    require(iterations >= 1, "cluster_iterations_invalid")
    if (
        not source_sessions
        or any(
            len(sessions) < minimum_sessions_per_source
            or any(count < 0 or minutes <= 0 for count, minutes in sessions)
            for sessions in source_sessions.values()
        )
    ):
        return None
    rng = random.Random(seed)
    worst_rates = []
    for _ in range(iterations):
        source_rates = []
        for source_id in sorted(source_sessions):
            sessions = source_sessions[source_id]
            sampled = [
                sessions[rng.randrange(len(sessions))] for _ in sessions
            ]
            source_rates.append(
                sum(row[0] for row in sampled)
                / sum(row[1] for row in sampled)
            )
        worst_rates.append(max(source_rates))
    worst_rates.sort()
    index = min(
        len(worst_rates) - 1,
        max(0, math.ceil(confidence_level * len(worst_rates)) - 1),
    )
    return worst_rates[index]


def validated_negative_interval_index(
    mask: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]], int
]:
    intervals, total_duration_ns = r0._negative_interval_index(mask)
    unit_ids: set[str] = set()
    recomputed_duration_ns = 0
    for key, rows in intervals.items():
        prior_end_ns: int | None = None
        for row in rows:
            unit_id = row["unit_id"]
            require(
                unit_id not in unit_ids,
                "negative_interval_unit_id_duplicate",
            )
            unit_ids.add(unit_id)
            start_ns = int(row["start_ns"])
            end_ns = int(row["end_ns"])
            duration_ns = int(row["duration_ns"])
            require(end_ns > start_ns, "negative_interval_not_positive")
            require(
                prior_end_ns is None or start_ns >= prior_end_ns,
                f"negative_interval_overlap:{key!r}",
            )
            prior_end_ns = end_ns
            recomputed_duration_ns += duration_ns
    require(
        recomputed_duration_ns == total_duration_ns,
        "negative_interval_total_duration_drift",
    )
    return intervals, total_duration_ns


def _tokens(
    ledgers: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for (source_id, sequence_id), ledger in ledgers.items():
        for frame in ledger["frames"]:
            for token in frame["token_activations"]:
                row = {
                    **token,
                    "source_id": source_id,
                    "sequence_id": sequence_id,
                }
                require(row["token_id"] not in result, "duplicate_token_id")
                result[row["token_id"]] = row
    return result


def _coverage_audit(
    repo: Path,
    config: dict[str, Any],
    ledgers: dict[tuple[str, str], dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    oracle_ledgers = r0._load_oracle_ledgers(repo, config)
    token_by_track = {
        (
            token["source_id"],
            token["sequence_id"],
            token["reset_segment"],
            token["track_id"],
        ): token
        for token in tokens.values()
    }
    cells = []
    matched: set[str] = set()
    for (candidate_id, event_id), oracle_ledger in sorted(
        oracle_ledgers.items()
    ):
        qualifications = r0._oracle_qualifications(oracle_ledger)
        matches = {}
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
                and int(token["qualification_frame_id"])
                <= int(qualification["qualification_frame_id"])
                <= int(token["last_valid_frame_id"])
            ):
                matches[token["token_id"]] = token
        matched.update(matches)
        cells.append(
            {
                "posthoc_oracle_cell_id": f"{candidate_id}::{event_id}",
                "candidate_id": candidate_id,
                "event_id": event_id,
                "source_id": oracle_ledger["source_id"],
                "sequence_id": oracle_ledger["sequence_id"],
                "oracle_qualification_count": len(qualifications),
                "covered_within_token_validity": bool(matches),
                "matched_policy_token_ids": sorted(matches),
            }
        )
    gate = config["coverage_gate"]
    supported_cells = [
        row for row in cells if row["oracle_qualification_count"] > 0
    ]
    no_active_cells = [
        row for row in cells if row["oracle_qualification_count"] == 0
    ]
    supported_covered = sum(
        row["covered_within_token_validity"] for row in supported_cells
    )
    no_active_closed = sum(
        not row["covered_within_token_validity"] for row in no_active_cells
    )
    passed = (
        len(cells) == gate["oracle_candidate_event_cells"]
        and len(supported_cells) == gate["required_supported_cell_coverage"]
        and supported_covered == gate["required_supported_cell_coverage"]
        and len(no_active_cells)
        == gate["required_no_active_relation_cells"]
        and no_active_closed == gate["required_no_active_relation_cells"]
    )
    return (
        {
            "cells": cells,
            "oracle_candidate_event_cells": len(cells),
            "oracle_supported_cell_count": len(supported_cells),
            "oracle_supported_cells_covered_within_validity": (
                supported_covered
            ),
            "oracle_supported_cell_miss_count": (
                len(supported_cells) - supported_covered
            ),
            "oracle_no_active_relation_cell_count": len(no_active_cells),
            "oracle_no_active_relation_cells_fail_closed": no_active_closed,
            "unique_matched_policy_tokens": len(matched),
            "passed": passed,
        },
        matched,
    )


def _risk_audit(
    repo: Path,
    config: dict[str, Any],
    ledgers: dict[tuple[str, str], dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    matched: set[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    mask = r0.load_json(
        repo / config["parent_bindings"]["eligibility_mask"]["path"]
    )
    intervals, total_duration_ns = validated_negative_interval_index(mask)
    negative_rows = []
    session_counts: dict[tuple[str, str], int] = defaultdict(int)
    for token in tokens.values():
        unit_ids = [
            interval["unit_id"]
            for interval in intervals.get(
                (token["source_id"], token["sequence_id"]), []
            )
            if int(interval["start_ns"])
            <= int(token["qualification_timestamp_ns"])
            < int(interval["end_ns"])
        ]
        if unit_ids:
            negative_rows.append(
                {
                    "token_id": token["token_id"],
                    "source_id": token["source_id"],
                    "sequence_id": token["sequence_id"],
                    "qualification_timestamp_ns": token[
                        "qualification_timestamp_ns"
                    ],
                    "negative_exposure_unit_ids": sorted(unit_ids),
                }
            )
            session_counts[(token["source_id"], token["sequence_id"])] += 1
    session_duration_ns: dict[tuple[str, str], int] = defaultdict(int)
    for key, rows in intervals.items():
        session_duration_ns[key] = sum(int(row["duration_ns"]) for row in rows)
    per_source = []
    source_sessions: dict[str, set[str]] = defaultdict(set)
    for source_id, sequence_id in session_duration_ns:
        source_sessions[source_id].add(sequence_id)
    confidence = float(config["risk_gate"]["confidence_level"])
    min_sessions = int(config["risk_gate"]["minimum_sequences_per_source"])
    cluster_supported = bool(source_sessions)
    worst_source_poisson_ucb = 0.0
    bootstrap_iterations = int(
        config["risk_gate"]["cluster_bootstrap_iterations"]
    )
    bootstrap_seed = int(config["risk_gate"]["cluster_bootstrap_seed"])
    simultaneous_source_confidence = (
        1.0 - (1.0 - confidence) / len(source_sessions)
        if source_sessions
        else confidence
    )
    source_session_pairs: dict[str, list[tuple[int, float]]] = {}
    for source_id in sorted(source_sessions):
        keys = [
            (source_id, sequence_id)
            for sequence_id in sorted(source_sessions[source_id])
        ]
        duration_ns = sum(session_duration_ns[key] for key in keys)
        minutes = duration_ns / 60_000_000_000
        count = sum(session_counts[key] for key in keys)
        ucb = poisson_one_sided_upper_rate(
            count, minutes, simultaneous_source_confidence
        )
        source_supported = len(keys) >= min_sessions
        cluster_supported &= source_supported
        source_session_pairs[source_id] = [
            (
                session_counts[key],
                session_duration_ns[key] / 60_000_000_000,
            )
            for key in keys
        ]
        cluster_ucb = (
            clustered_rate_upper_bound(
                source_session_pairs[source_id],
                confidence,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed
                + int(r0.sha256_bytes(source_id.encode("utf-8"))[:8], 16),
            )
            if source_supported
            else None
        )
        if ucb is not None:
            worst_source_poisson_ucb = max(
                worst_source_poisson_ucb, ucb
            )
        per_source.append(
            {
                "source_id": source_id,
                "sequence_count": len(keys),
                "negative_exposure_minutes": minutes,
                "negative_token_count": count,
                "point_rate_tokens_per_minute": (
                    count / minutes if minutes > 0 else None
                ),
                "poisson_upper_rate_tokens_per_minute": ucb,
                "poisson_simultaneous_confidence_level": (
                    simultaneous_source_confidence
                ),
                "cluster_bootstrap_upper_rate_tokens_per_minute": cluster_ucb,
                "minimum_sequence_support_met": source_supported,
            }
        )
    minutes = total_duration_ns / 60_000_000_000
    count = len(negative_rows)
    point_rate = count / minutes if minutes > 0 else None
    poisson_ucb = poisson_one_sided_upper_rate(count, minutes, confidence)
    worst_source_cluster_ucb = (
        stratified_worst_source_cluster_upper_bound(
            source_session_pairs,
            confidence,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            minimum_sessions_per_source=min_sessions,
        )
    )
    threshold = float(
        config["risk_gate"]["acceptance_threshold_tokens_per_minute"]
    )
    exposure_floor_met = (
        minutes
        >= float(config["risk_gate"]["minimum_zero_event_exposure_minutes"])
    )
    point_gate_passed = point_rate is not None and point_rate <= threshold
    point_rejecting_sources = [
        row["source_id"]
        for row in per_source
        if row["point_rate_tokens_per_minute"] is not None
        and row["point_rate_tokens_per_minute"] > threshold
    ]
    worst_source_point_gate_passed = not point_rejecting_sources
    point_rejection_triggered = (
        not point_gate_passed or not worst_source_point_gate_passed
    )
    poisson_gate_passed = (
        poisson_ucb is not None and poisson_ucb <= threshold
    )
    cluster_gate_passed = (
        cluster_supported
        and worst_source_cluster_ucb is not None
        and worst_source_cluster_ucb <= threshold
    )
    worst_source_poisson_gate_passed = (
        bool(per_source) and worst_source_poisson_ucb <= threshold
    )
    credible_bound = (
        exposure_floor_met
        and point_gate_passed
        and worst_source_point_gate_passed
        and poisson_gate_passed
        and worst_source_poisson_gate_passed
        and cluster_gate_passed
    )
    repeats = []
    invalidations = []
    unknown_activations = cross_reset = expired_unterminalized = 0
    for ledger in ledgers.values():
        current_segment = None
        for frame in ledger["frames"]:
            if frame["state_reset_before_frame"]:
                current_segment = frame["reset_segment"]
            for token in frame["token_activations"]:
                unknown_activations += int(not frame["route_known"])
                cross_reset += int(token["reset_segment"] != current_segment)
                expired_unterminalized += int(
                    token["effective_valid_until_timestamp_ns"] is None
                )
            repeats.extend(
                {
                    **row,
                    "source_id": ledger["source_id"],
                    "sequence_id": ledger["sequence_id"],
                }
                for row in frame["requalifications_suppressed"]
            )
            invalidations.extend(
                {
                    **row,
                    "source_id": ledger["source_id"],
                    "sequence_id": ledger["sequence_id"],
                }
                for row in frame["token_invalidations"]
            )
    integrity = {
        "producer_token_count": len(tokens),
        "duplicate_token_id_count": len(tokens) - len(set(tokens)),
        "unknown_route_token_activation_count": unknown_activations,
        "cross_reset_token_count": cross_reset,
        "unterminated_token_count": expired_unterminalized,
    }
    integrity["passed"] = all(
        integrity[name] == 0
        for name in (
            "duplicate_token_id_count",
            "unknown_route_token_activation_count",
            "cross_reset_token_count",
            "unterminated_token_count",
        )
    )
    risk = {
        "full_sequence_token_count": len(tokens),
        "matched_unique_policy_token_count": len(matched),
        "full_sequence_extra_token_count": len(tokens) - len(matched),
        "negative_exposure_interval_count": len(
            mask["negative_exposure_intervals"]
        ),
        "negative_exposure_duration_ns": total_duration_ns,
        "negative_exposure_minutes": minutes,
        "negative_exposure_token_count": count,
        "negative_exposure_tokens_per_minute": point_rate,
        "acceptance_threshold_tokens_per_minute": threshold,
        "confidence_level": confidence,
        "poisson_upper_rate_tokens_per_minute": poisson_ucb,
        "minimum_zero_event_exposure_minutes": config["risk_gate"][
            "minimum_zero_event_exposure_minutes"
        ],
        "exposure_floor_met": exposure_floor_met,
        "point_gate_passed": point_gate_passed,
        "point_rejecting_sources": point_rejecting_sources,
        "worst_source_point_gate_passed": worst_source_point_gate_passed,
        "point_rejection_triggered": point_rejection_triggered,
        "poisson_gate_passed": poisson_gate_passed,
        "cluster_bound_supported": cluster_supported,
        "worst_source_poisson_upper_rate_tokens_per_minute": (
            worst_source_poisson_ucb
        ),
        "worst_source_poisson_simultaneous_confidence_level": (
            simultaneous_source_confidence
        ),
        "worst_source_poisson_gate_passed": (
            worst_source_poisson_gate_passed
        ),
        "cluster_bootstrap_iterations": bootstrap_iterations,
        "cluster_bootstrap_seed": bootstrap_seed,
        "worst_source_cluster_bootstrap_upper_rate_tokens_per_minute": (
            worst_source_cluster_ucb
        ),
        "cluster_gate_passed": cluster_gate_passed,
        "credible_risk_bound_passed": credible_bound,
        "per_source": per_source,
    }
    risk_ledger = {
        "schema": RISK_SCHEMA,
        "stage": STAGE,
        "risk": dict(risk),
        "negative_exposure_tokens": sorted(
            negative_rows, key=lambda row: row["token_id"]
        ),
        "extra_tokens": sorted(
            (
                row
                for token_id, row in tokens.items()
                if token_id not in matched
            ),
            key=lambda row: row["token_id"],
        ),
        "token_invalidations": sorted(
            invalidations,
            key=lambda row: (
                row["source_id"],
                row["sequence_id"],
                row["reset_segment"],
                row["track_id"],
            ),
        ),
    }
    repeat_ledger = {
        "schema": REPEAT_SCHEMA,
        "stage": STAGE,
        "requalification_suppressed_count": len(repeats),
        "requalifications_suppressed": sorted(
            repeats,
            key=lambda row: (
                row["source_id"],
                row["sequence_id"],
                row["reset_segment"],
                row["track_id"],
                row["qualification_frame_id"],
            ),
        ),
    }
    risk_path = (
        Path(config["outputs"]["root"]) / config["outputs"]["risk_ledger"]
    )
    repeat_path = (
        Path(config["outputs"]["root"]) / config["outputs"]["repeat_ledger"]
    )
    risk["risk_ledger_path"] = risk_path.as_posix()
    risk["risk_ledger_sha256"] = r0.sha256_bytes(
        r0.canonical_bytes(risk_ledger)
    )
    risk["repeat_ledger_path"] = repeat_path.as_posix()
    risk["repeat_ledger_sha256"] = r0.sha256_bytes(
        r0.canonical_bytes(repeat_ledger)
    )
    risk["requalification_suppressed_count"] = len(repeats)
    risk["token_invalidation_count"] = len(invalidations)
    return risk, integrity, risk_ledger, repeat_ledger


def _compute_terminal_from_frozen_inventory(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, bindings = load_and_verify_config(repo, config_path)
    inventory, ledgers = load_and_verify_policy_inventory(
        repo, config, bindings
    )
    tokens = _tokens(ledgers)
    coverage, matched = _coverage_audit(repo, config, ledgers, tokens)
    risk, integrity, risk_ledger, repeat_ledger = _risk_audit(
        repo, config, ledgers, tokens, matched
    )
    triggered_rejections = []
    if not integrity["passed"]:
        triggered_rejections.append("POLICY_INTEGRITY_REJECT")
    if not coverage["passed"]:
        triggered_rejections.append("POLICY_COVERAGE_REJECT")
    if risk["point_rejection_triggered"]:
        triggered_rejections.append("POLICY_RISK_REJECT")
    if not integrity["passed"]:
        terminal_state = "POLICY_INTEGRITY_REJECT"
    elif not coverage["passed"]:
        terminal_state = "POLICY_COVERAGE_REJECT"
    elif risk["point_rejection_triggered"]:
        terminal_state = "POLICY_RISK_REJECT"
    elif not risk["credible_risk_bound_passed"]:
        terminal_state = "HOLD_FOR_CREDIBLE_RISK_BOUND"
    else:
        terminal_state = "POLICY_RISK_GATE_PASSED"
    require(
        terminal_state in config["terminal_states"],
        "terminal_state_not_preregistered",
    )
    root = repo / config["outputs"]["root"]
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": terminal_state,
        "terminal_precedence": config["terminal_precedence"],
        "triggered_rejections": triggered_rejections,
        "authority": (
            "candidate_independent_token_policy_and_risk_audit_only_"
            "no_opener_integration_or_higher_authority"
        ),
        "config_sha256": r0.sha256_file(config_path),
        "phase_evidence": {
            "policy_inventory_frozen_before_oracle_or_negative_join": True,
            "inventory_path": (
                Path(config["outputs"]["root"])
                / config["outputs"]["inventory"]
            ).as_posix(),
            "inventory_sha256": r0.sha256_file(
                root / config["outputs"]["inventory"]
            ),
            "truth_payloads_decoded_before_freeze": 0,
            "event_windows_decoded_before_freeze": 0,
            "oracle_tokens_decoded_before_freeze": 0,
            "candidate_independent_sequence_ledgers": inventory[
                "candidate_independent_sequence_count"
            ],
            "full_sequence_frames": inventory["frame_count"],
        },
        "policy": config["policy"],
        "coverage_audit": coverage,
        "integrity_audit": integrity,
        "risk_audit": risk,
        "claim_boundary": config["claim_boundary"],
    }
    return terminal, risk_ledger, repeat_ledger


def build_terminal_from_frozen_inventory(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    terminal, risk_ledger, repeat_ledger = (
        _compute_terminal_from_frozen_inventory(repo, config_path)
    )
    config = r0.load_json(config_path)
    root = repo / config["outputs"]["root"]
    r0.atomic_write_json(root / config["outputs"]["risk_ledger"], risk_ledger)
    r0.atomic_write_json(
        root / config["outputs"]["repeat_ledger"], repeat_ledger
    )
    r0.atomic_write_json(root / config["outputs"]["terminal_receipt"], terminal)
    return terminal


def validate_outputs(
    repo: Path, config_path: Path
) -> dict[str, Any]:
    config, bindings = load_and_verify_config(repo, config_path)
    inventory, persisted = load_and_verify_policy_inventory(
        repo, config, bindings
    )
    _, sequences = _parent_frames(repo, config)
    require(
        set(persisted) == set(sequences),
        "policy_ledger_keyset_drift",
    )
    policy = config["policy"]
    for key, frames in sequences.items():
        expected = produce_policy_ledger(
            key[0],
            key[1],
            frames,
            minimum_consecutive_active_frames=int(
                policy["minimum_consecutive_active_frames"]
            ),
            minimum_active_relation_duration_ns=int(
                policy["minimum_active_relation_duration_ns"]
            ),
            maximum_token_ttl_ns=int(policy["maximum_token_ttl_ns"]),
        )
        require(expected == persisted[key], f"policy_recompute_drift:{key!r}")
    terminal_path = (
        repo / config["outputs"]["root"] / config["outputs"]["terminal_receipt"]
    )
    require(terminal_path.is_file(), "terminal_receipt_missing")
    persisted_terminal = r0.load_json(terminal_path)
    risk_path = (
        repo / config["outputs"]["root"] / config["outputs"]["risk_ledger"]
    )
    repeat_path = (
        repo / config["outputs"]["root"] / config["outputs"]["repeat_ledger"]
    )
    require(risk_path.is_file(), "risk_ledger_missing")
    require(repeat_path.is_file(), "repeat_ledger_missing")
    persisted_risk = r0.load_json(risk_path)
    persisted_repeat = r0.load_json(repeat_path)
    expected_terminal, expected_risk, expected_repeat = (
        _compute_terminal_from_frozen_inventory(repo, config_path)
    )
    require(
        persisted_terminal == expected_terminal,
        "terminal_recompute_drift",
    )
    require(
        persisted_risk == expected_risk,
        "risk_ledger_recompute_drift",
    )
    require(
        persisted_repeat == expected_repeat,
        "repeat_ledger_recompute_drift",
    )
    validation = {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID",
        "config_sha256": r0.sha256_file(config_path),
        "terminal_sha256": r0.sha256_file(terminal_path),
        "checks": {
            "parent_r0_inventory_reverified": True,
            "candidate_independent_ledgers_recomputed": len(sequences),
            "full_sequence_frames_recomputed": inventory["frame_count"],
            "truth_blind_phase_order_reverified": True,
            "ttl_and_invalidation_recomputed": True,
            "requalification_suppression_recomputed": True,
            "oracle_validity_coverage_recomputed": True,
            "complete_negative_exposure_recomputed": True,
            "poisson_working_bound_recomputed": True,
            "source_sequence_cluster_support_recomputed": True,
            "terminal_recomputed": True,
        },
    }
    r0.atomic_write_json(
        repo
        / config["outputs"]["root"]
        / config["outputs"]["validation_receipt"],
        validation,
    )
    return validation
