#!/usr/bin/env python3
"""Evaluate the materialized R1.2c v2 six-positive truth/geometry oracle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import load_json, sha256_file, write_json
    from .evaluate_r12c_truth_geometry_consistency import validate_protocol as validate_v1_protocol
    from .validate_r12c_seen_positive_prereg import validate as validate_replacement
except ImportError:
    from contract import load_json, sha256_file, write_json
    from evaluate_r12c_truth_geometry_consistency import validate_protocol as validate_v1_protocol
    from validate_r12c_seen_positive_prereg import validate as validate_replacement


PROTOCOL_SCHEMA = "blindassist_ustrf_crosscam_truth_geometry_preregistration_v2"
CONTINUOUS_SCHEMA = "blindassist_ustrf_crosscam_continuous_event_protocol_v2"
RESULT_SCHEMA = "blindassist_ustrf_crosscam_truth_geometry_r12c_result_v2"
EXPECTED_POSITIVES = 6
EXPECTED_EVENTS = 12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve(repo: Path, value: str) -> Path:
    path = (repo / value).resolve()
    require(path.is_relative_to(repo), f"referenced path escapes repository: {value}")
    return path


def _hash_bound(repo: Path, parent: dict[str, Any], prefix: str) -> Path:
    path = _resolve(repo, parent[f"{prefix}_path"])
    require(path.is_file(), f"missing hash-bound input: {path}")
    require(sha256_file(path) == parent[f"{prefix}_sha256"], f"{prefix} SHA-256 mismatch")
    return path


def _event_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    events = value.get("events")
    require(isinstance(events, list), "continuous event inventory missing")
    result = {str(row["event_id"]): row for row in events}
    require(len(result) == len(events), "continuous event inventory repeats event_id")
    return result


def validate_continuous_inventory(continuous: dict[str, Any], repo: Path) -> dict[str, Path]:
    require(continuous.get("schema") == CONTINUOUS_SCHEMA, "R1.2c v2 continuous schema mismatch")
    require(continuous.get("dataset_role") == "seen_diagnostic_not_held_out",
            "R1.2c v2 must remain seen diagnostic")
    parents = continuous["parents"]
    paths = {
        "r12a": _hash_bound(repo, parents, "continuous_r12a"),
        "replacement": _hash_bound(repo, parents, "seen_positive_prereg"),
    }
    old_events = _event_map(load_json(paths["r12a"]))
    new_events = _event_map(continuous)
    replacement = continuous["replacement"]
    removed = replacement["removed_event_id"]
    added = replacement["added_event_id"]
    require(removed == "japan_path_intrusion" and added == "bangkok_tactile_cone_intrusion",
            "R1.2c v2 must replace Japan with Bangkok")
    require(removed in old_events and removed not in new_events, "Japan was not removed from v2")
    require(added not in old_events and added in new_events, "Bangkok was not added to v2")
    require(len(old_events) == len(new_events) == EXPECTED_EVENTS, "v2 must preserve 12 events")
    for event_id in old_events.keys() - {removed}:
        require(new_events[event_id] == old_events[event_id], f"{event_id}: unrelated event changed in v2")
    positives = [row for row in new_events.values() if row["expected_class"] == "positive"]
    require(len(positives) == EXPECTED_POSITIVES, "v2 must contain six positives")
    bangkok = new_events[added]
    require(bangkok["clip_window_ms"] == [328000, 340000], "Bangkok clip window drifted")
    require(bangkok["alertable_start_ms"] == 333000, "Bangkok alertable start drifted")
    require(bangkok["latest_useful_alert_ms"] == 336000, "Bangkok alertable end drifted")
    require(bangkok["known_not_visible_from_ms"] == 339000, "Bangkok clearance time drifted")
    require(replacement["r13_slot_consumed"] is False
            and replacement["old_japan_result_remains_immutable"] is True,
            "replacement authority/history boundary drifted")
    authority = continuous["authority"]
    require(authority["truth_geometry_oracle_authorized"] is True, "v2 oracle is not authorized")
    for key in ("new_held_out_read", "london_768_candidate_execution_authorized",
                "full_continuous_replay_authorized", "device_soak_authorized",
                "r13_inventory_unlock_authorized", "training_authorized",
                "android_runtime_change_authorized", "production_model_replacement_authorized"):
        require(authority[key] is False, f"continuous v2 over-authorizes {key}")
    return paths


def _source_map(oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = oracle.get("sources")
    require(isinstance(sources, list), "legacy oracle sources missing")
    result = {str(row["event_id"]): row for row in sources}
    require(len(result) == len(sources), "legacy oracle repeats event_id")
    return result


def validate_protocol(protocol: dict[str, Any], repo: Path) -> dict[str, Any]:
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "R1.2c v2 protocol schema mismatch")
    require(protocol.get("dataset_role") == "seen_diagnostic_not_held_out",
            "R1.2c v2 must remain seen diagnostic")
    parents = protocol["parents"]
    paths = {
        "r12c_v1": _hash_bound(repo, parents, "r12c_v1"),
        "continuous_v2": _hash_bound(repo, parents, "continuous_v2"),
        "replacement": _hash_bound(repo, parents, "seen_positive_prereg"),
        "replacement_validation": _hash_bound(repo, parents, "seen_positive_validation"),
        "r13_v2": _hash_bound(repo, parents, "r13_prereg_v2"),
    }
    v1 = load_json(paths["r12c_v1"])
    validate_v1_protocol(v1, repo)
    continuous = load_json(paths["continuous_v2"])
    continuous_paths = validate_continuous_inventory(continuous, repo)
    require(continuous_paths["replacement"] == paths["replacement"],
            "continuous and oracle protocols bind different replacement contracts")

    replacement_result = validate_replacement(paths["replacement"])
    saved_validation = load_json(paths["replacement_validation"])
    for key in ("contract_sha256", "event_id", "replaces_event_id",
                "alertable_robust_inside_anchor_count", "eligible_positive_count_after_validation",
                "r13_slot_consumed"):
        require(saved_validation.get(key) == replacement_result.get(key),
                f"saved Bangkok validation differs on {key}")

    r13 = load_json(paths["r13_v2"])
    novelty = r13["novelty_and_access"]
    require(novelty["source_discovery_authorized"] is False
            and novelty["download_decode_or_detector_inference_authorized"] is False
            and novelty["result_access_authorized"] is False,
            "R1.3 inventory is not sealed")

    oracle = protocol["independent_truth_geometry_oracle"]
    paths["r11_oracle"] = _hash_bound(repo, oracle, "legacy_r11_oracle")
    paths["r12_oracle"] = _hash_bound(repo, oracle, "legacy_r12_oracle")
    positive_ids = oracle["positive_event_ids"]
    require(len(positive_ids) == len(set(positive_ids)) == oracle["required_positive_event_count"]
            == EXPECTED_POSITIVES, "R1.2c v2 requires six unique positives")
    new_events = _event_map(continuous)
    require(set(positive_ids) == {event_id for event_id, row in new_events.items()
                                  if row["expected_class"] == "positive"},
            "R1.2c v2 oracle inventory differs from continuous v2")
    require("japan_path_intrusion" not in positive_ids
            and oracle["replacement_event_id"] == "bangkok_tactile_cone_intrusion",
            "R1.2c v2 replacement inventory drifted")
    require(oracle["uncertainty_frame_ratios"] == [0.01, 0.02, 0.03],
            "oracle uncertainty ratios drifted")
    require(oracle["excluded_event_old_failure_must_remain_immutable"] is True,
            "Japan v1 failure must remain immutable")
    candidate = protocol["candidate_inheritance"]
    require(candidate["candidate_id"] == v1["london_single_variable_candidate"]["candidate_id"],
            "v2 candidate differs from frozen v1 candidate")
    require(candidate["candidate_count"] == 1
            and candidate["additional_resolution_candidates_allowed"] is False
            and candidate["threshold_class_prompt_bbox_polygon_or_tracker_change_allowed"] is False,
            "v2 candidate scope widened")
    policy = protocol["authorization_policy"]
    require(policy["oracle_pass_may_authorize_london_768_candidate_execution"] is True,
            "oracle cannot authorize the frozen candidate")
    for key in ("oracle_pass_may_authorize_full_continuous_replay",
                "oracle_pass_may_authorize_device_soak",
                "oracle_pass_may_authorize_r13_inventory_unlock"):
        require(policy[key] is False, f"oracle over-authorizes {key}")
    return {"paths": paths, "continuous": continuous, "replacement_result": replacement_result}


def evaluate(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite output: {output_path}")
    protocol_path = protocol_path.resolve()
    repo = protocol_path.parent.parent
    protocol = load_json(protocol_path)
    validated = validate_protocol(protocol, repo)
    paths = validated["paths"]
    continuous = validated["continuous"]
    event_specs = _event_map(continuous)
    positive_ids = protocol["independent_truth_geometry_oracle"]["positive_event_ids"]
    sources = {
        **_source_map(load_json(paths["r11_oracle"])),
        **_source_map(load_json(paths["r12_oracle"])),
    }

    replacement_contract = load_json(paths["replacement"])
    replacement_validation = validated["replacement_result"]
    replacement_frames = []
    for row in replacement_validation["anchor_results"]:
        if not row["scored_contact"]:
            continue
        replacement_frames.append({
            "frame_id": row["frame_id"],
            "timestamp_ms": row["timestamp_ms"],
            "visibility": "visible",
            "robust_relation": row["robust_relation"],
            "oracle_role": row["role"],
        })
    sources[replacement_contract["event"]["event_id"]] = {
        "event_id": replacement_contract["event"]["event_id"],
        "source_id": replacement_contract["source"]["source_id"],
        "target_instance_id": replacement_contract["event"]["target_instance_id"],
        "frames": replacement_frames,
    }

    events = []
    for event_id in positive_ids:
        spec = event_specs[event_id]
        source = sources.get(event_id)
        require(source is not None, f"{event_id}: independent oracle source missing")
        alertable_start = spec.get("alertable_start_ms")
        require(isinstance(alertable_start, int), f"{event_id}: alertable_start_ms missing")
        anchors = [
            {
                "frame_id": frame["frame_id"],
                "timestamp_ms": int(frame["timestamp_ms"]),
                "robust_relation": frame.get("robust_relation"),
                **({"oracle_role": frame["oracle_role"]} if "oracle_role" in frame else {}),
            }
            for frame in source["frames"]
            if frame.get("visibility") == "visible" and int(frame["timestamp_ms"]) >= alertable_start
        ]
        robust_inside = [row for row in anchors if row["robust_relation"] == "inside"]
        status = "consistent" if robust_inside else "truth_geometry_conflict"
        events.append({
            "event_id": event_id,
            "parent_round": spec["round"],
            "source_id": source.get("source_id"),
            "target_instance_id": source.get("target_instance_id"),
            "alertable_start_ms": alertable_start,
            "alertable_anchor_count": len(anchors),
            "alertable_robust_inside_count": len(robust_inside),
            "alertable_anchors": anchors,
            "status": status,
        })

    conflicts = [row["event_id"] for row in events if row["status"] == "truth_geometry_conflict"]
    all_consistent = not conflicts and len(events) == EXPECTED_POSITIVES
    result = {
        "schema": RESULT_SCHEMA,
        "contract_id": protocol["contract_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": protocol["dataset_role"],
        "protocol_sha256": sha256_file(protocol_path),
        "input_sha256": {
            "r12c_v1": sha256_file(paths["r12c_v1"]),
            "continuous_v2": sha256_file(paths["continuous_v2"]),
            "seen_positive_prereg": sha256_file(paths["replacement"]),
            "seen_positive_validation": sha256_file(paths["replacement_validation"]),
            "r11_oracle": sha256_file(paths["r11_oracle"]),
            "r12_oracle": sha256_file(paths["r12_oracle"]),
            "r13_prereg_v2": sha256_file(paths["r13_v2"]),
        },
        "replacement": {
            "removed_event_id": "japan_path_intrusion",
            "removed_event_disposition": "unknown_exclude_from_score_v1_history_immutable",
            "added_event_id": "bangkok_tactile_cone_intrusion",
            "r13_slot_consumed": False,
        },
        "positive_event_count": len(events),
        "consistent_positive_event_count": len(events) - len(conflicts),
        "eligible_positive_event_count": len(events) - len(conflicts),
        "truth_geometry_conflict_count": len(conflicts),
        "truth_geometry_conflict_event_ids": conflicts,
        "unresolved_truth_geometry_conflict_event_ids": conflicts,
        "all_positive_truth_geometry_consistent": all_consistent,
        "events": events,
        "authorization": {
            "london_768_candidate_execution_authorized": all_consistent,
            "full_continuous_replay_authorized": False,
            "device_soak_authorized": False,
            "r13_inventory_unlock_authorized": False,
        },
        "next_action": (
            "export_and_run_the_single_preregistered_london_fp16_768_gpu_candidate"
            if all_consistent else "resolve_truth_geometry_conflicts_without_moving_frozen_polygons"
        ),
        "authority": protocol["authority"],
    }
    write_json(output_path, result)
    Path(str(output_path) + ".sha256").write_text(sha256_file(output_path) + "\n", encoding="ascii")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate(args.protocol, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "consistent_positive_events": result["consistent_positive_event_count"],
        "truth_geometry_conflicts": result["truth_geometry_conflict_event_ids"],
        "london_768_authorized": result["authorization"]["london_768_candidate_execution_authorized"],
        "r13_slot_consumed": result["replacement"]["r13_slot_consumed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
