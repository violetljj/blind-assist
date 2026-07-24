#!/usr/bin/env python3
"""Build R2-L1 per-metric profiles from already-authoritative candidate traces.

This module never runs a candidate. It verifies the frozen trace inventory first,
then joins truth and produces per-candidate exploratory profiles without ranking.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CONFIG_SCHEMA = "blindassist_ustrf_route_target_r2_l1_metric_profile_r1"
PROFILE_SCHEMA = "blindassist_ustrf_route_target_r2_l1_candidate_metric_profile_r1"
TERMINAL_SCHEMA = "blindassist_ustrf_route_target_r2_l1_metric_profile_terminal_r1"
TERMINAL_STATE = "METRIC_PROFILES_COMPLETE"
IMPLEMENTATION_PATHS = {
    "terminal_schema_sha256": (
        "schemas/ustrf_route_target_r2_l1_metric_profile_r1.schema.json"
    ),
    "profile_core_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "metric_profiles_r2_l1.py"
    ),
    "runner_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "run_metric_profiles_r2_l1_from_traces.py"
    ),
    "validator_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "validate_metric_profiles_r2_l1.py"
    ),
    "tests_sha256": (
        "scripts/research/ustrf_route_target_evidence_closure/"
        "test_metric_profiles_r2_l1.py"
    ),
}
METRICS = (
    "critical_miss",
    "clearance",
    "unknown_or_stale_alert",
    "repeat",
    "evidence_age",
    "event_recall",
    "regeneration",
    "false_alerts_per_minute",
)


class ProfileContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileContractError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["source_id"]),
        str(row["sequence_id"]),
        int(row["frame_id"]),
        int(row["source_capture_timestamp_ns"]),
    )


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_truth_frame_index(
    config: dict[str, Any], repo: Path
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for binding in config["input_contract"]["truth_join_after_output_only"]:
        payload = load_json(repo / binding["path"])
        if (
            "sources" in payload
            and payload.get("schema") == "blindassist_ustrf_route_role_truth_r1"
        ):
            for source in payload["sources"]:
                for episode in source["person_episodes"]:
                    event_id = episode.get("legacy_event_id") or episode.get(
                        "risk_event_id"
                    )
                    for frame in episode["frames"]:
                        key = (
                            source["source_id"],
                            source["source_id"],
                            int(frame["frame_id"]),
                            int(frame["source_capture_timestamp_ns"]),
                        )
                        result[key].append(
                            {
                                "person_id": episode["person_id"],
                                "event_id": event_id,
                                "bbox_xyxy": frame["bbox_xyxy"],
                                "role": frame.get("role"),
                            }
                        )
        elif "frames" in payload:
            for frame in payload["frames"]:
                for person in frame.get("persons", []):
                    result[identity(frame)].append(
                        {
                            "person_id": person["person_id"],
                            "event_id": person.get("event_id"),
                            "bbox_xyxy": person["bbox_xyxy"],
                            "role": person.get("role"),
                        }
                    )
    return result


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        __import__("os").fsync(handle.fileno())
    temporary.replace(path)


def _verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) >= {"path", "sha256"}, f"{label}_binding_incomplete")
    path = repo / binding["path"]
    require(path.is_file(), f"{label}_missing:{binding['path']}")
    require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    return path


def load_and_verify_config(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    config = load_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_mismatch")
    require(
        config.get("status") == "frozen_trace_only_postoutput_truth_join",
        "config_status_mismatch",
    )
    authority = config["authority"]
    require(
        authority["maximum"] == "L1_EXPLORATORY_METRIC_PROFILE",
        "maximum_authority_drift",
    )
    require(
        all(
            authority[key] is False
            for key in (
                "candidate_execution",
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
            )
        ),
        "authority_opened",
    )
    bindings: dict[str, str] = {"config_sha256": sha256_file(config_path)}
    for name, binding in config["parent_bindings"].items():
        _verify_binding(repo, binding, name)
        bindings[f"{name}_sha256"] = binding["sha256"]
    for index, binding in enumerate(
        config["input_contract"]["truth_join_after_output_only"]
    ):
        _verify_binding(repo, binding, f"truth_join_{index}")
        bindings[f"truth_join_{index}_sha256"] = binding["sha256"]
    require(
        set(config["implementation_bindings"]) == set(IMPLEMENTATION_PATHS),
        "implementation_binding_inventory_drift",
    )
    for name, path_value in IMPLEMENTATION_PATHS.items():
        path = repo / path_value
        require(path.is_file(), f"implementation_missing:{path_value}")
        expected_sha256 = config["implementation_bindings"][name]
        require(
            sha256_file(path) == expected_sha256,
            f"implementation_sha256_drift:{name}",
        )
        bindings[name] = expected_sha256
    require(
        config["candidate_roster"]
        == [
            "C1_CAUSAL_ROUTE_RELATION_FSM",
            "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
            "C3_DUAL_KEY_CLEARANCE_FSM",
        ],
        "candidate_roster_drift",
    )
    return config, bindings


def _group_mask_frames(
    mask: dict[str, Any],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mask["preoutput_frame_ledger"]:
        rows_by_key[(row["source_id"], row["sequence_id"])].append(row)
    groups = []
    for descriptor in mask["preoutput_frame_masks"]:
        key = (descriptor["source_id"], descriptor["sequence_id"])
        rows = rows_by_key[key]
        require(len(rows) == descriptor["frame_count"], "mask_frame_count_drift")
        require(
            [row["frame_id"] for row in rows]
            == sorted(row["frame_id"] for row in rows),
            "mask_frame_order_drift",
        )
        groups.append((descriptor, rows))
    require(len(groups) == 41, "mask_sequence_count_drift")
    require(sum(len(rows) for _, rows in groups) == 62229, "mask_total_frames_drift")
    return groups


def _validate_parent_receipts(
    config: dict[str, Any], repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    a2_terminal = load_json(
        repo / config["parent_bindings"]["candidate_replay_a2_terminal"]["path"]
    )
    a2_validation = load_json(
        repo / config["parent_bindings"]["candidate_replay_a2_validation"]["path"]
    )
    a3_terminal = load_json(
        repo / config["parent_bindings"]["candidate_replay_a3_terminal"]["path"]
    )
    eligibility_validation = load_json(
        repo / config["parent_bindings"]["eligibility_validation"]["path"]
    )
    a3_validation = load_json(
        repo / config["parent_bindings"]["candidate_replay_a3_validation"]["path"]
    )
    a4_validation = load_json(
        repo / config["parent_bindings"]["candidate_replay_a4_memory_validation"][
            "path"
        ]
    )
    require(
        eligibility_validation["status"] == "VALID",
        "eligibility_validation_not_valid",
    )
    require(
        a2_terminal["terminal_state"] == "CANDIDATE_REPLAY_COMPLETE",
        "a2_terminal_not_complete",
    )
    require(a2_validation["status"] == "VALID", "a2_validation_not_valid")
    require(a3_validation["status"] == "VALID", "a3_validation_not_valid")
    require(a4_validation["status"] == "PASS", "a4_memory_validation_not_pass")
    require(
        a3_terminal["terminal_state"] == "CANDIDATE_REPLAY_COMPLETE",
        "a3_terminal_not_complete",
    )
    require(
        a3_terminal["profiles"] == {
            "authority": False,
            "count": 0,
            "generated": False,
        },
        "a3_already_contains_profile",
    )
    require(
        a3_terminal["claim_boundary"]["truth_joined"] is False,
        "a3_truth_join_authority_drift",
    )
    require(
        a3_terminal["parent_evidence"]["a2_terminal_sha256"]
        == config["parent_bindings"]["candidate_replay_a2_terminal"]["sha256"],
        "a3_a2_terminal_binding_drift",
    )
    require(
        a3_terminal["parent_evidence"]["a2_validation_sha256"]
        == config["parent_bindings"]["candidate_replay_a2_validation"]["sha256"],
        "a3_a2_validation_binding_drift",
    )
    return a2_terminal, a2_validation, a3_terminal


def _load_verified_traces(
    config: dict[str, Any],
    repo: Path,
    mask: dict[str, Any],
    a2_terminal: dict[str, Any],
) -> tuple[
    dict[str, dict[tuple[str, str], list[dict[str, Any]]]],
    dict[str, str],
]:
    groups = _group_mask_frames(mask)
    expected_keys = [
        (descriptor["source_id"], descriptor["sequence_id"])
        for descriptor, _ in groups
    ]
    descriptor_by_key = {
        (descriptor["source_id"], descriptor["sequence_id"]): descriptor
        for descriptor, _ in groups
    }
    inventory = a2_terminal["candidate_execution"]["trace_inventory"]
    require(len(inventory) == 123, "trace_inventory_count_drift")
    by_candidate: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        candidate: {} for candidate in config["candidate_roster"]
    }
    trace_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    reset_counts: dict[str, int] = defaultdict(int)
    for row in inventory:
        candidate = row["candidate_id"]
        require(candidate in by_candidate, "unexpected_candidate_trace")
        key = (row["source_id"], row["sequence_id"])
        require(key not in by_candidate[candidate], "duplicate_candidate_sequence_trace")
        require(key in descriptor_by_key, "trace_not_in_frozen_mask")
        require(
            row["frame_count"] == descriptor_by_key[key]["frame_count"],
            "trace_inventory_mask_frame_count_drift",
        )
        path = repo / row["trace_path"]
        require(path.is_file(), f"trace_missing:{row['trace_path']}")
        require(sha256_file(path) == row["trace_sha256"], "trace_sha256_drift")
        authoritative_receipt_path = repo / row["authoritative_receipt_path"]
        require(
            authoritative_receipt_path.is_file(),
            f"authoritative_receipt_missing:{row['authoritative_receipt_path']}",
        )
        require(
            sha256_file(authoritative_receipt_path)
            == row["authoritative_receipt_sha256"],
            "authoritative_receipt_sha256_drift",
        )
        trace = load_json(path)
        require(trace["candidate_id"] == candidate, "trace_candidate_drift")
        require(
            (trace["source_id"], trace["sequence_id"]) == key,
            "trace_sequence_identity_drift",
        )
        require(
            trace["frame_mask_sha256"]
            == descriptor_by_key[key]["frame_mask_sha256"],
            "trace_frame_mask_sha256_drift",
        )
        require(trace["frame_count"] == row["frame_count"], "trace_frame_count_drift")
        require(len(trace["frames"]) == row["frame_count"], "trace_rows_incomplete")
        trace_resets = sum(
            bool(frame["state_reset_before_frame"]) for frame in trace["frames"]
        ) - 1
        require(
            trace_resets == row["discontinuity_reset_count"],
            "trace_discontinuity_reset_count_drift",
        )
        reset_counts[candidate] += trace_resets
        by_candidate[candidate][key] = trace["frames"]
        trace_hashes[candidate].append(
            {
                "source_id": row["source_id"],
                "sequence_id": row["sequence_id"],
                "trace_sha256": row["trace_sha256"],
                "authoritative_receipt_sha256": row[
                    "authoritative_receipt_sha256"
                ],
            }
        )
    inventory_hashes: dict[str, str] = {}
    for candidate in config["candidate_roster"]:
        require(
            list(by_candidate[candidate]) == expected_keys,
            f"trace_sequence_order_or_membership_drift:{candidate}",
        )
        require(
            reset_counts[candidate] == 15,
            f"trace_discontinuity_total_drift:{candidate}",
        )
        for descriptor, mask_rows in groups:
            key = (descriptor["source_id"], descriptor["sequence_id"])
            trace_rows = by_candidate[candidate][key]
            require(
                len(trace_rows) == len(mask_rows),
                f"trace_mask_length_drift:{candidate}:{key}",
            )
            for trace_row, mask_row in zip(trace_rows, mask_rows, strict=True):
                require(
                    identity(trace_row)
                    == (
                        mask_row["source_id"],
                        mask_row["sequence_id"],
                        int(mask_row["frame_id"]),
                        int(mask_row["source_capture_timestamp_ns"]),
                    ),
                    f"trace_mask_identity_drift:{candidate}:{key}",
                )
                require(
                    bool(trace_row["route_known"])
                    == (mask_row["route_validity_state"] == "known"),
                    f"trace_route_state_drift:{candidate}:{key}",
                )
        inventory_hashes[candidate] = sha256_bytes(
            canonical_bytes(trace_hashes[candidate])
        )
    return by_candidate, inventory_hashes


def _attributed_event_ids_for_track_ids(
    frame: dict[str, Any],
    truth_people: list[dict[str, Any]],
    track_ids: list[int],
    minimum_iou: float,
) -> list[str]:
    tracks = [
        track
        for track in frame["observed_tracks"]
        if int(track["track_id"]) in {int(value) for value in track_ids}
    ]
    matched: set[str] = set()
    active_roles = {"approaching_route", "route_intersecting"}
    for track in tracks:
        candidates = sorted(
            (
                (box_iou(track["box"], person["bbox_xyxy"]), person)
                for person in truth_people
                if person.get("event_id") and person.get("role") in active_roles
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        if candidates and candidates[0][0] >= minimum_iou:
            matched.add(str(candidates[0][1]["event_id"]))
    return sorted(matched)


def _delivery_groups(frame: dict[str, Any]) -> list[tuple[Any, list[int]]]:
    keys = list(frame["deliveries"])
    track_ids = [int(value) for value in frame["delivery_track_ids"]]
    if not keys:
        return []
    if len(keys) == len(track_ids):
        return [(key, [track_id]) for key, track_id in zip(keys, track_ids, strict=True)]
    if len(keys) == 1 and track_ids:
        return [(keys[0], track_ids)]
    raise ProfileContractError("ambiguous_delivery_key_to_track_mapping")


def wilson_interval(numerator: int, denominator: int) -> dict[str, float] | None:
    if denominator <= 0:
        return None
    z = 1.959963984540054
    p = numerator / denominator
    z2 = z * z
    centre = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    radius = (
        z
        * math.sqrt(
            (p * (1 - p) / denominator) + z2 / (4 * denominator * denominator)
        )
        / (1 + z2 / denominator)
    )
    return {
        "method": "wilson_score_two_sided_95",
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def percentile_higher(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for row in rows:
        result[str(row[key])] += 1
    return dict(sorted(result.items()))


def _rate_source_results(
    rows: list[dict[str, Any]], positive_ids: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    result = []
    for source_id in sorted(grouped):
        source_rows = grouped[source_id]
        numerator = sum(row["event_id"] in positive_ids for row in source_rows)
        denominator = len(source_rows)
        result.append(
            {
                "source_id": source_id,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "ci_95": wilson_interval(numerator, denominator),
            }
        )
    return result


def _frame_source_results(
    rows: list[dict[str, Any]], positive_unit_ids: set[str]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    result = []
    for source_id in sorted(grouped):
        source_rows = grouped[source_id]
        numerator = sum(row["unit_id"] in positive_unit_ids for row in source_rows)
        denominator = len(source_rows)
        result.append(
            {
                "source_id": source_id,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "ci_95": wilson_interval(numerator, denominator),
            }
        )
    return result


def _contribution_summary(
    rows: list[dict[str, Any]], positive_ids: set[str], id_key: str
) -> dict[str, Any]:
    positive_rows = [row for row in rows if str(row[id_key]) in positive_ids]
    return {
        "denominator_by_source": _counts(rows, "source_id"),
        "denominator_by_provenance_family": _counts(
            rows, "provenance_family"
        ),
        "numerator_by_source": _counts(positive_rows, "source_id"),
        "numerator_by_provenance_family": _counts(
            positive_rows, "provenance_family"
        ),
    }


def _l1_rate_metric(
    *,
    numerator: int,
    denominator: int,
    point_pass: bool,
    hard_veto: bool,
    bound_sufficient: bool,
    source_results: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if hard_veto and numerator > 0:
        result_status = "observed_failure"
        gate_result = "hard_veto_observed"
    elif not point_pass:
        result_status = "observed_failure"
        gate_result = "point_fail"
    elif bound_sufficient:
        result_status = "point_and_bound_pass"
        gate_result = "pass"
    else:
        result_status = "estimate_only"
        gate_result = "point_pass_bound_insufficient"
    payload = {
        "support_status": "evaluable_l1",
        "result_status": result_status,
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
        "ci_95": wilson_interval(numerator, denominator),
        "bound_sufficient": bound_sufficient,
        "gate_result": gate_result,
        "source_results": source_results,
    }
    if extra:
        payload.update(extra)
    return payload


def _diagnostic_metric(
    numerator: int | None,
    denominator: int | float | None,
    value: int | float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "support_status": "diagnostic_only",
        "result_status": "not_tested",
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "ci_95": None,
        "bound_sufficient": False,
        "gate_result": "not_applicable",
        "source_results": [],
    }
    if extra:
        payload.update(extra)
    return payload


def validate_profile_contract(
    profile: dict[str, Any], config: dict[str, Any]
) -> None:
    require(profile["schema"] == PROFILE_SCHEMA, "profile_schema_mismatch")
    require(set(profile["metrics"]) == set(METRICS), "profile_metric_inventory_drift")
    metrics = profile["metrics"]
    require(metrics["critical_miss"]["denominator"] == 8, "critical_denominator_drift")
    require(metrics["clearance"]["denominator"] == 12, "clearance_denominator_drift")
    require(
        metrics["clearance"]["pre_clear_units_excluded"] == 6357,
        "pre_clear_entered_clearance",
    )
    require(
        metrics["unknown_or_stale_alert"]["denominator"] == 62229,
        "unknown_stale_denominator_drift",
    )
    require(
        metrics["repeat"]["denominator_source"]
        == "first_delivery_then_complete_observation",
        "repeat_truth_pool_substituted",
    )
    evidence = metrics["evidence_age"]
    require(evidence["required_timestamp_frame_count"] == 62229, "age_required_drift")
    if evidence["timestamp_frame_count"] != 62229:
        require(evidence["support_status"] == "not_evaluable", "age_missing_not_closed")
        require(
            all(
                evidence[key] is None
                for key in ("numerator", "denominator", "value", "ci_95")
            ),
            "age_partial_denominator_emitted",
        )
    for name in ("event_recall", "regeneration", "false_alerts_per_minute"):
        metric = metrics[name]
        require(metric["support_status"] == "diagnostic_only", f"{name}_l0_opened")
        require(metric["result_status"] == "not_tested", f"{name}_result_opened")
        require(metric["gate_result"] == "not_applicable", f"{name}_gate_opened")
    require(
        all(value is False for value in profile["claim_boundary"].values()),
        "profile_claim_boundary_opened",
    )
    require(
        config["authority"]["candidate_comparison"] is False,
        "comparison_authority_drift",
    )


def build_candidate_profile(
    candidate_id: str,
    traces: dict[tuple[str, str], list[dict[str, Any]]],
    inventory_sha256: str,
    mask: dict[str, Any],
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    all_frames: list[dict[str, Any]] = []
    deliveries: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    frame_by_sequence_id: dict[tuple[str, str, int], dict[str, Any]] = {}
    for descriptor, _ in _group_mask_frames(mask):
        key = (descriptor["source_id"], descriptor["sequence_id"])
        reset_segment = -1
        for frame in traces[key]:
            if frame["state_reset_before_frame"]:
                reset_segment += 1
            enriched = {**frame, "reset_segment": reset_segment}
            all_frames.append(enriched)
            frame_by_sequence_id[
                (frame["source_id"], frame["sequence_id"], int(frame["frame_id"]))
            ] = enriched
            for delivery_key, delivery_track_ids in _delivery_groups(frame):
                event_ids = _attributed_event_ids_for_track_ids(
                    frame,
                    truth_index.get(identity(frame), []),
                    delivery_track_ids,
                    float(
                        config["scoring_contract"][
                            "truth_attribution_minimum_iou"
                        ]
                    ),
                )
                deliveries.append(
                    {
                        "source_id": frame["source_id"],
                        "sequence_id": frame["sequence_id"],
                        "frame_id": int(frame["frame_id"]),
                        "source_capture_timestamp_ns": int(
                            frame["source_capture_timestamp_ns"]
                        ),
                        "delivery_key": delivery_key,
                        "delivery_track_ids": delivery_track_ids,
                        "reset_segment": reset_segment,
                        "event_ids": event_ids,
                    }
                )
            for closure_key in frame["closures"]:
                closures.append(
                    {
                        "source_id": frame["source_id"],
                        "sequence_id": frame["sequence_id"],
                        "frame_id": int(frame["frame_id"]),
                        "source_capture_timestamp_ns": int(
                            frame["source_capture_timestamp_ns"]
                        ),
                        "closure_key": closure_key,
                        "reset_segment": reset_segment,
                    }
                )
    require(len(all_frames) == 62229, "candidate_frame_count_drift")
    events = mask["events"]
    event_by_id = {event["event_id"]: event for event in events}
    event_deliveries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_active_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for delivery in deliveries:
        for event_id in delivery["event_ids"]:
            if event_id in event_by_id:
                event_deliveries[event_id].append(delivery)
    for frame in all_frames:
        active_event_ids = _attributed_event_ids_for_track_ids(
            frame,
            truth_index.get(identity(frame), []),
            [int(value) for value in frame["active_relation_track_ids"]],
            float(config["scoring_contract"]["truth_attribution_minimum_iou"]),
        )
        for event_id in active_event_ids:
            if event_id in event_by_id:
                event_active_frames[event_id].append(frame)

    critical_events = [
        event
        for event in events
        if event["metrics"]["critical_miss"]["classification"] == "eligible"
    ]
    critical_miss_ids: set[str] = set()
    for event in critical_events:
        details = event["metrics"]["critical_miss"]["details"]
        observed = [
            frame
            for frame in event_active_frames[event["event_id"]]
            if int(details["critical_interval_start_frame"])
            <= frame["frame_id"]
            <= int(details["critical_interval_end_frame"])
        ]
        if not observed:
            critical_miss_ids.add(event["event_id"])

    clearance_events = [
        event
        for event in events
        if event["metrics"]["clearance"]["classification"] == "eligible"
    ]
    clearance_success_ids: set[str] = set()
    clearance_delays_ms: list[float] = []
    for event in clearance_events:
        eligible_deliveries = [
            delivery
            for delivery in event_deliveries[event["event_id"]]
            if int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["truth_terminal_clear_frame"])
        ]
        if not eligible_deliveries:
            continue
        first = eligible_deliveries[0]
        matching = [
            closure
            for closure in closures
            if closure["source_id"] == event["source_id"]
            and closure["sequence_id"] == event["sequence_id"]
            and closure["closure_key"] == first["delivery_key"]
            and closure["reset_segment"] == first["reset_segment"]
            and int(event["anchors"]["truth_terminal_clear_frame"])
            <= closure["frame_id"]
            <= int(event["anchors"]["end_frame"])
        ]
        if not matching:
            continue
        clear_frame = frame_by_sequence_id[
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
                - int(clear_frame["source_capture_timestamp_ns"])
            )
            / 1_000_000.0,
        )
        if delay_ms <= float(config["metric_contract"]["clearance_horizon_ms"]):
            clearance_success_ids.add(event["event_id"])
            clearance_delays_ms.append(delay_ms)

    repeat_events = [
        event
        for event in events
        if event["metrics"]["repeat"]["classification"] == "eligible"
    ]
    repeat_denominator_events: list[dict[str, Any]] = []
    repeated_event_ids: set[str] = set()
    raw_repeat_delivery_count = 0
    for event in repeat_events:
        observed = [
            delivery
            for delivery in event_deliveries[event["event_id"]]
            if int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["truth_terminal_clear_frame"])
        ]
        if observed:
            repeat_denominator_events.append(event)
            if len(observed) > 1:
                repeated_event_ids.add(event["event_id"])
                raw_repeat_delivery_count += len(observed) - 1

    unknown_alert_rows = []
    for frame, mask_row in zip(
        all_frames, mask["preoutput_frame_ledger"], strict=True
    ):
        if mask_row["route_validity_state"] != "known" and (
            bool(frame["deliveries"]) or bool(frame["candidate_active"])
        ):
            unknown_alert_rows.append(mask_row)
    unknown_unit_ids = {row["unit_id"] for row in unknown_alert_rows}

    consume_ages = [
        int(frame["candidate_consume_timestamp_ns"])
        - int(frame["source_capture_timestamp_ns"])
        for frame in all_frames
        if isinstance(frame.get("candidate_consume_timestamp_ns"), int)
    ]
    require(all(age >= 0 for age in consume_ages), "negative_evidence_age")

    negative_intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for interval in mask["negative_exposure_intervals"]:
        negative_intervals[
            (interval["source_id"], interval["sequence_id"])
        ].append((int(interval["start_ns"]), int(interval["end_ns"])))
    false_alert_deliveries = [
        delivery
        for delivery in deliveries
        if not delivery["event_ids"]
        and any(
            start_ns <= delivery["source_capture_timestamp_ns"] < end_ns
            for start_ns, end_ns in negative_intervals[
                (delivery["source_id"], delivery["sequence_id"])
            ]
        )
    ]
    regeneration_count = sum(
        delivery["frame_id"]
        > int(event_by_id[event_id]["anchors"]["truth_terminal_clear_frame"])
        for delivery in deliveries
        for event_id in delivery["event_ids"]
        if event_id in event_by_id
        and event_by_id[event_id]["anchors"]["truth_terminal_clear_frame"] is not None
    )

    critical_bound = (
        len(critical_events)
        >= int(config["metric_contract"]["critical_bound_minimum_denominator"])
        and len(critical_miss_ids) == 0
    )
    clearance_rate = len(clearance_success_ids) / len(clearance_events)
    clearance_bound = (
        len(clearance_events)
        >= int(config["metric_contract"]["clearance_bound_minimum_denominator"])
        and clearance_rate >= float(config["metric_contract"]["clearance_rate_min"])
    )
    repeat_denominator = len(repeat_denominator_events)
    if repeat_denominator == 0:
        repeat_support = "not_evaluable"
        repeat_status = "not_evaluable"
        repeat_gate = "not_applicable"
        repeat_value = None
        repeat_ci = None
    else:
        repeat_support = (
            "evaluable_underpowered"
            if repeat_denominator
            < int(config["metric_contract"]["repeat_minimum_denominator"])
            else "evaluable_l1"
        )
        repeat_value = len(repeated_event_ids) / repeat_denominator
        repeat_ci = wilson_interval(len(repeated_event_ids), repeat_denominator)
        if repeated_event_ids:
            repeat_status = "observed_failure"
            repeat_gate = "point_fail"
        else:
            repeat_status = "estimate_only"
            repeat_gate = (
                "not_applicable_underpowered"
                if repeat_support == "evaluable_underpowered"
                else "point_pass"
            )
    if len(consume_ages) == 62229:
        age_p95 = percentile_higher(consume_ages, 95)
        age_metric = {
            "support_status": "evaluable_l1",
            "result_status": (
                "estimate_only"
                if age_p95
                <= int(config["metric_contract"]["evidence_age_p95_ns_max"])
                else "observed_failure"
            ),
            "numerator": len(consume_ages),
            "denominator": 62229,
            "value": age_p95,
            "ci_95": None,
            "bound_sufficient": False,
            "gate_result": (
                "point_pass"
                if age_p95
                <= int(config["metric_contract"]["evidence_age_p95_ns_max"])
                else "point_fail"
            ),
            "source_results": [],
            "timestamp_frame_count": len(consume_ages),
            "required_timestamp_frame_count": 62229,
            "value_unit": "p95_nanoseconds",
        }
    else:
        age_metric = {
            "support_status": "not_evaluable",
            "result_status": "not_evaluable",
            "numerator": None,
            "denominator": None,
            "value": None,
            "ci_95": None,
            "bound_sufficient": False,
            "gate_result": "not_applicable",
            "source_results": [],
            "timestamp_frame_count": len(consume_ages),
            "required_timestamp_frame_count": 62229,
            "value_unit": "p95_nanoseconds",
        }
    negative_exposure_ns = int(
        config["metric_contract"]["negative_exposure_ns"]
    )
    negative_exposure_minutes = negative_exposure_ns / 60_000_000_000.0
    metrics = {
        "critical_miss": _l1_rate_metric(
            numerator=len(critical_miss_ids),
            denominator=len(critical_events),
            point_pass=not critical_miss_ids,
            hard_veto=True,
            bound_sufficient=critical_bound,
            source_results=_rate_source_results(
                critical_events, critical_miss_ids
            ),
            extra={
                "value_unit": "miss_fraction",
                "contributions": _contribution_summary(
                    critical_events, critical_miss_ids, "event_id"
                ),
            },
        ),
        "clearance": _l1_rate_metric(
            numerator=len(clearance_success_ids),
            denominator=len(clearance_events),
            point_pass=clearance_rate
            >= float(config["metric_contract"]["clearance_rate_min"]),
            hard_veto=False,
            bound_sufficient=clearance_bound,
            source_results=_rate_source_results(
                clearance_events, clearance_success_ids
            ),
            extra={
                "value_unit": "success_fraction",
                "p95_delay_ms": percentile_higher(clearance_delays_ms, 95),
                "pre_clear_units_excluded": 6357,
                "contributions": _contribution_summary(
                    clearance_events, clearance_success_ids, "event_id"
                ),
            },
        ),
        "unknown_or_stale_alert": _l1_rate_metric(
            numerator=len(unknown_alert_rows),
            denominator=len(all_frames),
            point_pass=not unknown_alert_rows,
            hard_veto=True,
            bound_sufficient=False,
            source_results=_frame_source_results(
                mask["preoutput_frame_ledger"], unknown_unit_ids
            ),
            extra={
                "value_unit": "active_alert_frame_fraction",
                "unknown_or_stale_truth_frame_count": sum(
                    row["route_validity_state"] != "known"
                    for row in mask["preoutput_frame_ledger"]
                ),
                "contributions": _contribution_summary(
                    mask["preoutput_frame_ledger"],
                    unknown_unit_ids,
                    "unit_id",
                ),
            },
        ),
        "repeat": {
            "support_status": repeat_support,
            "result_status": repeat_status,
            "numerator": (
                len(repeated_event_ids) if repeat_denominator > 0 else None
            ),
            "denominator": repeat_denominator if repeat_denominator > 0 else None,
            "value": repeat_value,
            "ci_95": repeat_ci,
            "bound_sufficient": False,
            "gate_result": repeat_gate,
            "source_results": _rate_source_results(
                repeat_denominator_events, repeated_event_ids
            ),
            "denominator_source": "first_delivery_then_complete_observation",
            "truth_pool_size": len(repeat_events),
            "minimum_denominator": int(
                config["metric_contract"]["repeat_minimum_denominator"]
            ),
            "raw_repeat_delivery_count": raw_repeat_delivery_count,
            "value_unit": "event_repeat_fraction",
            "contributions": _contribution_summary(
                repeat_denominator_events, repeated_event_ids, "event_id"
            ),
        },
        "evidence_age": age_metric,
        "event_recall": _diagnostic_metric(
            len(event_deliveries),
            0,
            None,
            {
                "raw_attributed_event_count": len(event_deliveries),
                "value_unit": "not_authorized",
            },
        ),
        "regeneration": _diagnostic_metric(
            regeneration_count,
            0,
            None,
            {"raw_post_clear_delivery_count": regeneration_count},
        ),
        "false_alerts_per_minute": _diagnostic_metric(
            len(false_alert_deliveries),
            negative_exposure_minutes,
            len(false_alert_deliveries) / negative_exposure_minutes,
            {
                "negative_exposure_ns": negative_exposure_ns,
                "minimum_l1_exposure_ns": int(
                    config["metric_contract"]["minimum_l1_negative_exposure_ns"]
                ),
                "value_unit": "raw_diagnostic_deliveries_per_minute",
            },
        ),
    }
    profile = {
        "schema": PROFILE_SCHEMA,
        "stage": "R2-L1-METRIC-PROFILE-R1",
        "authority": "per_metric_l1_exploratory_only_no_candidate_comparison",
        "candidate_id": candidate_id,
        "bindings": {
            "eligibility_mask_sha256": config["parent_bindings"][
                "eligibility_mask"
            ]["sha256"],
            "candidate_replay_a2_terminal_sha256": config["parent_bindings"][
                "candidate_replay_a2_terminal"
            ]["sha256"],
            "candidate_replay_a3_terminal_sha256": config["parent_bindings"][
                "candidate_replay_a3_terminal"
            ]["sha256"],
            "trace_inventory_sha256": inventory_sha256,
        },
        "verified_scope": {
            "authoritative_trace_count": 41,
            "authoritative_trace_frames": 62229,
            "truth_join_timing": "after_authoritative_candidate_output_only",
        },
        "metrics": metrics,
        "claim_boundary": {
            "candidate_comparison": False,
            "winner_or_ranking": False,
            "selection": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
        },
    }
    validate_profile_contract(profile, config)
    return profile


def build_terminal_receipt(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config, bindings = load_and_verify_config(repo, config_path)
    a2_terminal, _, _ = _validate_parent_receipts(config, repo)
    mask = load_json(repo / config["parent_bindings"]["eligibility_mask"]["path"])
    denominator = load_json(
        repo / config["parent_bindings"]["denominator_receipt"]["path"]
    )
    require(
        denominator["candidate_outputs_executed"] is False,
        "denominator_receipt_candidate_blind_drift",
    )
    traces, inventory_hashes = _load_verified_traces(
        config, repo, mask, a2_terminal
    )
    truth_index = load_truth_frame_index(config, repo)
    profiles: dict[str, dict[str, Any]] = {}
    profile_inventory = []
    for candidate in config["candidate_roster"]:
        profile = build_candidate_profile(
            candidate,
            traces[candidate],
            inventory_hashes[candidate],
            mask,
            truth_index,
            config,
        )
        profiles[candidate] = profile
        profile_inventory.append(
            {
                "candidate_id": candidate,
                "profile_path": (
                    "artifacts.local/evidence/"
                    "ustrf-route-target-r2-l1-metric-profile-r1/profiles/"
                    f"{candidate}.json"
                ),
                "profile_sha256": sha256_bytes(canonical_bytes(profile)),
            }
        )
    receipt = {
        "schema": TERMINAL_SCHEMA,
        "stage": "R2-L1-METRIC-PROFILE-R1",
        "terminal_state": TERMINAL_STATE,
        "authority": "per_metric_l1_exploratory_only_no_comparison_selection_or_promotion",
        "bindings": bindings,
        "verified_scope": {
            "candidate_count": 3,
            "sequence_ledgers_per_candidate": 41,
            "authoritative_traces": 123,
            "authoritative_trace_frames": 186687,
            "eligibility_mask_frames": 62229,
            "truth_joined_after_output": True,
            "candidate_reruns": 0,
            "new_authoritative_traces": 0,
        },
        "profile_inventory": profile_inventory,
        "metric_authority": {
            "l1": [
                "critical_miss",
                "clearance",
                "unknown_or_stale_alert",
            ],
            "conditional_l1": ["repeat", "evidence_age"],
            "l0_diagnostic_only": [
                "event_recall",
                "regeneration",
                "false_alerts_per_minute",
            ],
        },
        "claim_boundary": {
            "candidate_comparison": False,
            "winner_or_ranking": False,
            "selection": False,
            "threshold_or_denominator_change": False,
            "l2_or_l3": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "independent_walking_safety": False,
            "production": False,
        },
    }
    return receipt, profiles


def validate_terminal_receipt(
    repo: Path, config_path: Path, receipt_path: Path
) -> dict[str, Any]:
    expected, profiles = build_terminal_receipt(repo, config_path)
    observed = load_json(receipt_path)
    require(observed == expected, "terminal_receipt_not_canonical_recomputation")
    for row in expected["profile_inventory"]:
        profile_path = repo / row["profile_path"]
        require(profile_path.is_file(), f"profile_missing:{row['candidate_id']}")
        require(
            load_json(profile_path) == profiles[row["candidate_id"]],
            f"profile_content_drift:{row['candidate_id']}",
        )
        require(
            sha256_file(profile_path) == row["profile_sha256"],
            f"profile_sha256_drift:{row['candidate_id']}",
        )
    return {
        "schema": "blindassist_ustrf_route_target_r2_l1_metric_profile_validation_r1",
        "status": "VALID",
        "terminal_state": TERMINAL_STATE,
        "candidate_profiles": 3,
        "authoritative_traces_reverified": 123,
        "authoritative_trace_frames_reverified": 186687,
        "terminal_receipt_sha256": sha256_file(receipt_path),
    }
