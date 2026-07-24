#!/usr/bin/env python3
"""Post-hoc attribution for the frozen candidate-independent policy failure."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import causal_token_policy_risk_gate_r1 as parent

CONFIG_SCHEMA = (
    "blindassist_ustrf_candidate_independent_policy_failure_attribution_r1"
)
MISS_SCHEMA = "blindassist_ustrf_policy_supported_cell_miss_attribution_r1"
NEGATIVE_SCHEMA = "blindassist_ustrf_policy_negative_token_attribution_r1"
TERMINAL_SCHEMA = "blindassist_ustrf_policy_failure_attribution_terminal_r1"
VALIDATION_SCHEMA = "blindassist_ustrf_policy_failure_attribution_validation_r1"
STAGE = "CANDIDATE-INDEPENDENT-POLICY-FAILURE-ATTRIBUTION-R1"
IMPLEMENTATION_PATHS = {
    "core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "candidate_independent_policy_failure_attribution_r1.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_candidate_independent_policy_failure_attribution_r1.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_candidate_independent_policy_failure_attribution_r1.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_candidate_independent_policy_failure_attribution_r1.py"
    ),
}

INVALIDATION_CLASS = {
    "route_unknown": "ROUTE_UNKNOWN_BEFORE_ORACLE",
    "track_unobserved": "TRACK_UNOBSERVED_BEFORE_ORACLE",
    "active_relation_gap": "RELATION_GAP_BEFORE_ORACLE",
    "reset_before_frame": "RESET_BEFORE_ORACLE",
    "sequence_end": "SEQUENCE_END_BEFORE_ORACLE",
    "ttl_elapsed": "ORACLE_AFTER_TTL",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise parent.PolicyGateContractError(message)


def _verify(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding["path"]
    require(path.is_file(), f"{label}_missing")
    require(parent.r0.sha256_file(path) == binding["sha256"], f"{label}_sha_drift")
    return path


def load_and_verify_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = parent.r0.load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "attribution_config_schema_drift")
    require(config.get("stage") == STAGE, "attribution_stage_drift")
    boundary = config["claim_boundary"]
    require(
        boundary
        == {
            "maximum": "CANDIDATE_INDEPENDENT_POLICY_FAILURE_ATTRIBUTION_ONLY",
            "diagnostic_only": True,
            "new_policy": False,
            "threshold_change": False,
            "ttl_change": False,
            "opener_integration": False,
            "candidate_execution_or_comparison": False,
            "selection": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
        },
        "attribution_claim_boundary_drift",
    )
    miss = config["supported_miss_contract"]
    expected_classes = [
        "ROUTE_UNKNOWN_BEFORE_ORACLE",
        "TRACK_UNOBSERVED_BEFORE_ORACLE",
        "RELATION_GAP_BEFORE_ORACLE",
        "RESET_BEFORE_ORACLE",
        "SEQUENCE_END_BEFORE_ORACLE",
        "ORACLE_AFTER_TTL",
        "QUALIFICATION_INSUFFICIENT",
        "UNEXPLAINED",
    ]
    require(
        miss["opportunity_classes"] == expected_classes, "miss_classes_drift"
    )
    require(
        miss["parent_supported_cell_miss_count"] == 24
        and miss["parent_miss_oracle_qualification_opportunity_count"] == 96
        and miss["validity_check"]
        == (
            "qualification_timestamp_ns <= oracle_timestamp_ns < "
            "effective_valid_until_timestamp_ns"
        )
        and miss["one_class_per_oracle_qualification_opportunity"] is True
        and miss["cell_summary_retains_mixed_reason_sets"] is True
        and miss["unexplained_required_zero"] is True,
        "miss_closure_contract_drift",
    )
    negative = config["negative_token_contract"]
    require(
        negative
        == {
            "parent_negative_exposure_token_count": 34,
            "join_key": "token_id",
            "group_by": ["source_id", "sequence_id", "invalidation_reason"],
            "one_invalidation_per_negative_token": True,
            "unattributed_required_zero": True,
        },
        "negative_token_contract_drift",
    )
    for label, binding in config["parent_bindings"].items():
        _verify(repo, binding, label)
    implementation = config["implementation_bindings"]
    require(
        set(implementation) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_set_drift",
    )
    for label, relative_path in IMPLEMENTATION_PATHS.items():
        path = repo / relative_path
        require(path.is_file(), f"implementation_{label}_missing")
        require(
            parent.r0.sha256_file(path) == implementation[label],
            f"implementation_{label}_sha_drift",
        )
    return config


def _load_verified_parent(
    repo: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    parent_config_path = repo / config["parent_bindings"]["policy_config"]["path"]
    parent_config, parent_bindings = parent.load_and_verify_config(
        repo, parent_config_path
    )
    inventory, ledgers = parent.load_and_verify_policy_inventory(
        repo, parent_config, parent_bindings
    )
    terminal, expected_risk, _ = parent._compute_terminal_from_frozen_inventory(
        repo, parent_config_path
    )
    persisted_terminal = parent.r0.load_json(
        repo / config["parent_bindings"]["policy_terminal"]["path"]
    )
    persisted_risk = parent.r0.load_json(
        repo / config["parent_bindings"]["policy_risk_ledger"]["path"]
    )
    require(terminal == persisted_terminal, "parent_terminal_recompute_drift")
    require(expected_risk == persisted_risk, "parent_risk_recompute_drift")
    require(
        terminal["terminal_state"] == "POLICY_COVERAGE_REJECT",
        "parent_not_coverage_reject",
    )
    require(
        inventory["token_count"] == len(parent._tokens(ledgers)),
        "parent_token_inventory_drift",
    )
    return parent_config, ledgers, persisted_risk


def _oracle_qualifications(ledger: dict[str, Any]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    prior_tracks: set[int] = set()
    prior_segment: int | None = None
    for frame in ledger["frames"]:
        current = set(frame["eligible_attributed_track_ids"])
        if prior_segment == frame["reset_segment"]:
            for track_id in sorted(current & prior_tracks):
                result.append(
                    {
                        "track_id": int(track_id),
                        "reset_segment": int(frame["reset_segment"]),
                        "oracle_frame_id": int(frame["frame_id"]),
                        "oracle_timestamp_ns": int(
                            frame["source_capture_timestamp_ns"]
                        ),
                    }
                )
        prior_tracks = current
        prior_segment = int(frame["reset_segment"])
    return result


def classify_opportunity(evidence: str) -> str:
    require(evidence != "COVERED_WITHIN_VALIDITY", "covered_cell_given_to_miss")
    if evidence in {
        "NO_POLICY_TOKEN_FOR_TRACK_RESET",
        "ORACLE_BEFORE_POLICY_QUALIFICATION",
    }:
        return "QUALIFICATION_INSUFFICIENT"
    if evidence.startswith("ORACLE_AFTER_"):
        reason = evidence.removeprefix("ORACLE_AFTER_").lower()
        return INVALIDATION_CLASS.get(reason, "UNEXPLAINED")
    return "UNEXPLAINED"


def opportunity_evidence(
    token: dict[str, Any] | None, oracle_timestamp_ns: int
) -> tuple[str, str | None]:
    if token is None:
        return "NO_POLICY_TOKEN_FOR_TRACK_RESET", None
    token_id = token["token_id"]
    qualification_ns = int(token["qualification_timestamp_ns"])
    effective_until_ns = int(token["effective_valid_until_timestamp_ns"])
    require(effective_until_ns >= qualification_ns, "token_validity_interval_invalid")
    if oracle_timestamp_ns < qualification_ns:
        return "ORACLE_BEFORE_POLICY_QUALIFICATION", token_id
    if oracle_timestamp_ns < effective_until_ns:
        return "COVERED_WITHIN_VALIDITY", token_id
    return (
        f"ORACLE_AFTER_{str(token['invalidation_reason']).upper()}",
        token_id,
    )


def _support_snapshot(
    ledger: dict[str, Any],
    *,
    reset_segment: int,
    track_id: int,
    oracle_frame_id: int,
    oracle_timestamp_ns: int,
) -> dict[str, Any]:
    frames = ledger["frames"]
    index = next(
        (
            i
            for i, frame in enumerate(frames)
            if int(frame["frame_id"]) == oracle_frame_id
            and int(frame["reset_segment"]) == reset_segment
        ),
        None,
    )
    require(index is not None, "oracle_frame_missing_from_policy_ledger")
    frame = frames[index]
    require(
        int(frame["source_capture_timestamp_ns"]) == oracle_timestamp_ns,
        "oracle_policy_timestamp_identity_drift",
    )
    active = bool(
        frame["route_known"]
        and track_id in set(frame["active_relation_track_ids"])
    )
    if not active:
        return {
            "route_known": bool(frame["route_known"]),
            "track_observed": track_id in set(frame["observed_track_ids"]),
            "active_relation": False,
            "support_frame_count": 0,
            "support_duration_ns": 0,
            "support_start_frame_id": None,
            "support_start_timestamp_ns": None,
        }
    start = index
    while start > 0:
        prior = frames[start - 1]
        if (
            int(prior["reset_segment"]) != reset_segment
            or not prior["route_known"]
            or track_id not in set(prior["active_relation_track_ids"])
        ):
            break
        start -= 1
    first = frames[start]
    return {
        "route_known": True,
        "track_observed": track_id in set(frame["observed_track_ids"]),
        "active_relation": True,
        "support_frame_count": index - start + 1,
        "support_duration_ns": (
            oracle_timestamp_ns - int(first["source_capture_timestamp_ns"])
        ),
        "support_start_frame_id": int(first["frame_id"]),
        "support_start_timestamp_ns": int(
            first["source_capture_timestamp_ns"]
        ),
    }


def _supported_miss_attribution(
    repo: Path,
    config: dict[str, Any],
    parent_config: dict[str, Any],
    ledgers: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    tokens = parent._tokens(ledgers)
    token_by_track = {
        (
            row["source_id"],
            row["sequence_id"],
            int(row["reset_segment"]),
            int(row["track_id"]),
        ): row
        for row in tokens.values()
    }
    oracle_ledgers = parent.r0._load_oracle_ledgers(repo, parent_config)
    policy_ledgers = ledgers
    rows = []
    for (candidate_id, event_id), oracle_ledger in sorted(oracle_ledgers.items()):
        qualifications = _oracle_qualifications(oracle_ledger)
        if not qualifications:
            continue
        evidence_rows = []
        covered = False
        for qualification in qualifications:
            key = (
                oracle_ledger["source_id"],
                oracle_ledger["sequence_id"],
                qualification["reset_segment"],
                qualification["track_id"],
            )
            token = token_by_track.get(key)
            support = _support_snapshot(
                policy_ledgers[
                    (oracle_ledger["source_id"], oracle_ledger["sequence_id"])
                ],
                reset_segment=qualification["reset_segment"],
                track_id=qualification["track_id"],
                oracle_frame_id=qualification["oracle_frame_id"],
                oracle_timestamp_ns=qualification["oracle_timestamp_ns"],
            )
            evidence, token_id = opportunity_evidence(
                token, qualification["oracle_timestamp_ns"]
            )
            covered |= evidence == "COVERED_WITHIN_VALIDITY"
            evidence_rows.append(
                {
                    **qualification,
                    "policy_token_id": token_id,
                    "evidence_class": evidence,
                    "policy_qualification_frame_id": (
                        int(token["qualification_frame_id"]) if token else None
                    ),
                    "policy_last_valid_frame_id": (
                        int(token["last_valid_frame_id"]) if token else None
                    ),
                    "policy_invalidation_reason": (
                        token["invalidation_reason"] if token else None
                    ),
                    "later_policy_token_exists": bool(
                        token
                        and qualification["oracle_timestamp_ns"]
                        < int(token["qualification_timestamp_ns"])
                    ),
                    "support_at_oracle": support,
                }
            )
        if covered:
            continue
        evidence_counts = Counter(
            row["evidence_class"] for row in evidence_rows
        )
        opportunity_counts = Counter(
            classify_opportunity(row["evidence_class"])
            for row in evidence_rows
        )
        cause_set = sorted(opportunity_counts)
        rows.append(
            {
                "cell_id": f"{candidate_id}::{event_id}",
                "candidate_id": candidate_id,
                "event_id": event_id,
                "source_id": oracle_ledger["source_id"],
                "sequence_id": oracle_ledger["sequence_id"],
                "cause_set": cause_set,
                "mixed": len(cause_set) > 1,
                "evidence_class_counts": dict(sorted(evidence_counts.items())),
                "opportunity_class_counts": dict(
                    sorted(opportunity_counts.items())
                ),
                "oracle_qualifications": evidence_rows,
            }
        )
    opportunity_class_counts = Counter()
    for row in rows:
        opportunity_class_counts.update(row["opportunity_class_counts"])
    signature_counts = Counter(
        "+".join(row["cause_set"]) for row in rows
    )
    return {
        "schema": MISS_SCHEMA,
        "stage": STAGE,
        "authority": "diagnostic_only_no_policy_change",
        "supported_cell_miss_count": len(rows),
        "oracle_qualification_opportunity_count": sum(
            opportunity_class_counts.values()
        ),
        "opportunity_class_counts": dict(
            sorted(opportunity_class_counts.items())
        ),
        "cell_cause_signature_counts": dict(sorted(signature_counts.items())),
        "mixed_cell_count": sum(row["mixed"] for row in rows),
        "unexplained_count": opportunity_class_counts["UNEXPLAINED"],
        "rows": rows,
    }


def _negative_token_attribution(
    config: dict[str, Any], risk: dict[str, Any]
) -> dict[str, Any]:
    invalidations: dict[str, dict[str, Any]] = {}
    for row in risk["token_invalidations"]:
        token_id = row["token_id"]
        require(token_id not in invalidations, "duplicate_parent_invalidation")
        invalidations[token_id] = row
    rows = []
    groups = Counter()
    unattributed = 0
    for negative in risk["negative_exposure_tokens"]:
        invalidation = invalidations.get(negative["token_id"])
        if invalidation is None:
            unattributed += 1
            reason = None
        else:
            reason = invalidation["invalidation_reason"]
            groups[
                (
                    negative["source_id"],
                    negative["sequence_id"],
                    reason,
                )
            ] += 1
        rows.append(
            {
                **negative,
                "invalidation_reason": reason,
                "effective_valid_until_timestamp_ns": (
                    invalidation["effective_valid_until_timestamp_ns"]
                    if invalidation
                    else None
                ),
                "last_valid_frame_id": (
                    invalidation["last_valid_frame_id"] if invalidation else None
                ),
            }
        )
    group_rows = [
        {
            "source_id": key[0],
            "sequence_id": key[1],
            "invalidation_reason": key[2],
            "negative_token_count": count,
        }
        for key, count in sorted(groups.items())
    ]
    reason_counts = Counter(row["invalidation_reason"] for row in rows)
    return {
        "schema": NEGATIVE_SCHEMA,
        "stage": STAGE,
        "authority": "diagnostic_only_no_policy_change",
        "negative_exposure_token_count": len(rows),
        "unattributed_count": unattributed,
        "invalidation_reason_counts": dict(sorted(reason_counts.items())),
        "groups": group_rows,
        "rows": sorted(rows, key=lambda row: row["token_id"]),
    }


def compute_outputs(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_and_verify_config(repo, config_path)
    parent_config, ledgers, risk = _load_verified_parent(repo, config)
    misses = _supported_miss_attribution(
        repo, config, parent_config, ledgers
    )
    negatives = _negative_token_attribution(config, risk)
    miss_ok = (
        misses["supported_cell_miss_count"]
        == config["supported_miss_contract"]["parent_supported_cell_miss_count"]
        and misses["oracle_qualification_opportunity_count"]
        == config["supported_miss_contract"][
            "parent_miss_oracle_qualification_opportunity_count"
        ]
        and misses["unexplained_count"] == 0
        and len({row["cell_id"] for row in misses["rows"]})
        == misses["supported_cell_miss_count"]
    )
    negative_ok = (
        negatives["negative_exposure_token_count"]
        == config["negative_token_contract"]["parent_negative_exposure_token_count"]
        and negatives["unattributed_count"] == 0
        and len({row["token_id"] for row in negatives["rows"]})
        == negatives["negative_exposure_token_count"]
    )
    terminal_state = (
        "POLICY_FAILURE_ATTRIBUTION_CLOSED"
        if miss_ok and negative_ok
        else "ATTRIBUTION_INTEGRITY_REJECT"
    )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "stage": STAGE,
        "terminal_state": terminal_state,
        "authority": "candidate_independent_policy_failure_attribution_only",
        "config_sha256": parent.r0.sha256_file(config_path),
        "parent_terminal_state": "POLICY_COVERAGE_REJECT",
        "supported_miss_summary": {
            "count": misses["supported_cell_miss_count"],
            "oracle_qualification_opportunity_count": misses[
                "oracle_qualification_opportunity_count"
            ],
            "opportunity_class_counts": misses[
                "opportunity_class_counts"
            ],
            "cell_cause_signature_counts": misses[
                "cell_cause_signature_counts"
            ],
            "mixed_cell_count": misses["mixed_cell_count"],
            "unexplained_count": misses["unexplained_count"],
            "closed": miss_ok,
        },
        "negative_token_summary": {
            "count": negatives["negative_exposure_token_count"],
            "invalidation_reason_counts": negatives[
                "invalidation_reason_counts"
            ],
            "unattributed_count": negatives["unattributed_count"],
            "closed": negative_ok,
        },
        "claim_boundary": config["claim_boundary"],
        "policy_changed": False,
        "successor_policy_authorized": False,
        "opener_integration_authorized": False,
    }
    require(terminal_state in config["terminal_states"], "terminal_state_illegal")
    return misses, negatives, terminal


def build_outputs(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    misses, negatives, terminal = compute_outputs(repo, config_path)
    root = repo / config["outputs"]["root"]
    parent.r0.atomic_write_json(
        root / config["outputs"]["supported_miss_ledger"], misses
    )
    parent.r0.atomic_write_json(
        root / config["outputs"]["negative_token_ledger"], negatives
    )
    parent.r0.atomic_write_json(
        root / config["outputs"]["terminal_receipt"], terminal
    )
    return terminal


def validate_outputs(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    expected = compute_outputs(repo, config_path)
    root = repo / config["outputs"]["root"]
    names = [
        config["outputs"]["supported_miss_ledger"],
        config["outputs"]["negative_token_ledger"],
        config["outputs"]["terminal_receipt"],
    ]
    for name, value in zip(names, expected, strict=True):
        path = root / name
        require(path.is_file(), f"output_missing:{name}")
        require(parent.r0.load_json(path) == value, f"output_recompute_drift:{name}")
    receipt = {
        "schema": VALIDATION_SCHEMA,
        "stage": STAGE,
        "status": "VALID",
        "config_sha256": parent.r0.sha256_file(config_path),
        "supported_miss_ledger_sha256": parent.r0.sha256_file(root / names[0]),
        "negative_token_ledger_sha256": parent.r0.sha256_file(root / names[1]),
        "terminal_sha256": parent.r0.sha256_file(root / names[2]),
        "checks": {
            "parent_policy_outputs_hash_bound_and_recomputed": True,
            "all_supported_misses_uniquely_classified": True,
            "timestamp_half_open_validity_recomputed": True,
            "all_oracle_qualification_evidence_retained": True,
            "all_negative_tokens_joined_to_one_invalidation": True,
            "source_sequence_reason_groups_recomputed": True,
            "policy_or_threshold_modified": False,
        },
    }
    parent.r0.atomic_write_json(
        root / config["outputs"]["validation_receipt"], receipt
    )
    return receipt
