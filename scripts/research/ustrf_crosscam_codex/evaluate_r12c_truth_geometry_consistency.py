#!/usr/bin/env python3
"""Freeze the R1.2c positive-event truth/route consistency decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import load_json, sha256_file, write_json
except ImportError:
    from contract import load_json, sha256_file, write_json


PROTOCOL_SCHEMA = "blindassist_ustrf_crosscam_truth_geometry_preregistration_v1"
RESULT_SCHEMA = "blindassist_ustrf_crosscam_truth_geometry_r12c_result_v1"
ADJUDICATION_SCHEMA = "blindassist_ustrf_crosscam_model_adjudication_v1"
EXPECTED_POSITIVES = 6


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


def validate_protocol(protocol: dict[str, Any], repo: Path) -> dict[str, Path]:
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "R1.2c protocol schema mismatch")
    require(protocol.get("dataset_role") == "seen_diagnostic_not_held_out", "R1.2c must remain seen diagnostic")
    parents = protocol["parents"]
    paths = {
        "r12a": _hash_bound(repo, parents, "continuous_r12a"),
        "r12b": _hash_bound(repo, parents, "mobile_r12b"),
        "r12b_result": _hash_bound(repo, parents, "mobile_r12b_result"),
        "r13_v2": _hash_bound(repo, parents, "r13_prereg_v2"),
    }
    oracle = protocol["independent_truth_geometry_oracle"]
    paths["r11_oracle"] = _hash_bound(repo, oracle, "r11_oracle")
    paths["r12_oracle"] = _hash_bound(repo, oracle, "r12_oracle")
    require(oracle["detector_outputs_may_define_truth_or_geometry"] is False,
            "detector output cannot define truth or geometry")
    require(oracle["association_outputs_may_define_truth_or_geometry"] is False,
            "association output cannot define truth or geometry")
    require(oracle["required_positive_event_count"] == EXPECTED_POSITIVES,
            "R1.2c must preserve six positives")
    require(len(oracle["positive_event_ids"]) == EXPECTED_POSITIVES
            and len(set(oracle["positive_event_ids"])) == EXPECTED_POSITIVES,
            "R1.2c requires six unique positive events")
    require(oracle["uncertainty_frame_ratios"] == [0.01, 0.02, 0.03],
            "oracle uncertainty ratios drifted")
    require(oracle["no_alertable_robust_inside_status"] == "truth_geometry_conflict",
            "conflicts must fail closed")

    adjudication = protocol["conflict_adjudication"]
    require(adjudication["fail_closed"] is True, "conflict adjudication must fail closed")
    require(adjudication["manual_polygon_move_or_refit_to_rescue_old_result_allowed"] is False,
            "polygon rescue is forbidden")
    require(adjudication["old_result_may_be_reclassified_as_pass"] is False,
            "old failures must remain immutable")
    require(adjudication["candidate_execution_before_all_conflicts_resolved_allowed"] is False,
            "London candidate cannot run before consistency")

    candidate = protocol["london_single_variable_candidate"]
    require(candidate["candidate_id"] == "r12c_c1_sameweights_fp16_768_gpu_london_only",
            "unexpected R1.2c candidate")
    require(candidate["target_event_id"] == "london_center_marker_intrusion", "candidate must target London only")
    require(candidate["input_size"] == 768 and candidate["execution_backend"] == "gpu_delegate"
            and candidate["precision"] == "fp16", "candidate must be FP16-768 GPU")
    require(candidate["confidence_threshold"] == 0.05
            and candidate["target_anchor_iou_threshold"] == 0.30
            and candidate["nms_iou_threshold"] == 0.45, "candidate thresholds drifted")
    require(candidate["frozen_static_classes"] == ["traffic cone", "delineator", "bollard"],
            "candidate classes drifted")
    for key in ("class_or_prompt_change_allowed", "threshold_change_allowed",
                "bbox_contact_or_polygon_change_allowed", "training_or_distillation_authorized"):
        require(candidate[key] is False, f"candidate.{key} must remain false")

    sequence = protocol["execution_sequence"]
    require(sequence["candidate_count"] == 1 and sequence["additional_resolution_candidates_allowed"] is False,
            "R1.2c permits exactly one candidate")
    event_gate = protocol["full_continuous_event_gate"]
    require(event_gate["positive_event_recall_required"] == "6/6", "event gate must require 6/6 positives")
    for key in ("negative_false_alert_count_at_most", "delivered_repeated_alert_count_at_most",
                "cooccurrence_triggered_target_event_count_at_most", "identity_switch_count_at_most"):
        require(event_gate[key] == 0, f"event gate {key} must be zero")
    require(protocol["device_soak_gate"]["run_only_after_full_continuous_event_gate_passes"] is True,
            "soak must follow the event gate")
    require(protocol["device_soak_gate"]["duration_seconds"] == 600, "soak must remain 600 seconds")
    stops = protocol["hard_stops"]
    require(stops["if_768_misses_london"] == "stop_resolution_search_and_preregister_a_new_small_target_detector_hypothesis",
            "London miss stop rule drifted")
    require(stops["fp16_320_allowed"] is False and stops["int8_allowed"] is False
            and stops["tracker_optimization_allowed"] is False,
            "320, INT8 and tracker work must remain closed")
    require(protocol["authority"]["new_held_out_read"] is False, "R1.3 inventory opened early")
    return paths


def _source_map(oracle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = oracle.get("sources")
    require(isinstance(sources, list), "oracle sources missing")
    result = {str(row["event_id"]): row for row in sources}
    require(len(result) == len(sources), "oracle repeats event_id")
    return result


def _load_adjudication(path: Path, repo: Path) -> dict[str, Any]:
    receipt = load_json(path)
    require(receipt.get("schema") == ADJUDICATION_SCHEMA, "adjudication schema mismatch")
    require(receipt.get("event_id") == "japan_path_intrusion", "unexpected adjudication event")
    review = receipt["review_process"]
    require(review["project_human_role_is_performed_by_models"] is True,
            "project adjudication role must be explicit")
    require(review["reviewer_a_and_b_independent"] is True
            and review["reviewer_a_and_b_saw_each_others_outputs"] is False,
            "reviewers A/B were not independent")
    require(review["detector_tracker_or_association_outputs_read"] is False,
            "adjudication read prohibited detector/tracker evidence")
    for name, binding in receipt["hash_bound_inputs"].items():
        bound_path = _resolve(repo, binding["path"])
        require(bound_path.is_file(), f"adjudication input missing: {name}")
        require(sha256_file(bound_path) == binding["sha256"], f"adjudication input SHA mismatch: {name}")
    decision = receipt["adjudication"]
    require(decision["status"] == "complete", "adjudication is incomplete")
    require(decision["event_truth"] == "unknown_exclude_from_score", "unexpected Japan truth decision")
    require(decision["positive_gate_decision"] == "reject_as_strict_positive",
            "Japan must not retain positive eligibility")
    require(decision["strict_contact_alertable_interval_ms"] is None,
            "Japan strict positive interval must remain empty")
    require(decision["japan_may_fill_sixth_positive_slot"] is False,
            "Japan cannot fill the sixth positive slot")
    return receipt


def evaluate(protocol_path: Path, output_path: Path, adjudication_path: Path | None = None) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite output: {output_path}")
    protocol_path = protocol_path.resolve()
    repo = protocol_path.parent.parent
    protocol = load_json(protocol_path)
    paths = validate_protocol(protocol, repo)
    continuous = load_json(paths["r12a"])
    event_specs = {str(row["event_id"]): row for row in continuous["events"]}
    positive_ids = protocol["independent_truth_geometry_oracle"]["positive_event_ids"]
    require(set(positive_ids) == {event_id for event_id, row in event_specs.items()
                                  if row["expected_class"] == "positive"},
            "R1.2c positive inventory differs from R1.2a")
    oracle_sources = {
        **_source_map(load_json(paths["r11_oracle"])),
        **_source_map(load_json(paths["r12_oracle"])),
    }

    events = []
    for event_id in positive_ids:
        spec = event_specs[event_id]
        source = oracle_sources.get(event_id)
        require(source is not None, f"{event_id}: independent oracle source missing")
        alertable_start = spec.get("alertable_start_ms")
        require(isinstance(alertable_start, int), f"{event_id}: alertable_start_ms missing")
        anchors = [
            {
                "frame_id": frame["frame_id"],
                "timestamp_ms": int(frame["timestamp_ms"]),
                "robust_relation": frame.get("robust_relation"),
            }
            for frame in source["frames"]
            if frame.get("visibility") == "visible" and int(frame["timestamp_ms"]) >= alertable_start
        ]
        robust_inside = [row for row in anchors if row["robust_relation"] == "inside"]
        status = "consistent" if robust_inside else "truth_geometry_conflict"
        events.append({
            "event_id": event_id,
            "parent_round": spec["round"],
            "alertable_start_ms": alertable_start,
            "alertable_anchor_count": len(anchors),
            "alertable_robust_inside_count": len(robust_inside),
            "alertable_anchors": anchors,
            "status": status,
            "adjudication_required": status == "truth_geometry_conflict",
        })

    conflicts = [row["event_id"] for row in events if row["status"] == "truth_geometry_conflict"]
    adjudication = None
    if adjudication_path is not None:
        adjudication = _load_adjudication(adjudication_path.resolve(), repo)
        target = next(row for row in events if row["event_id"] == adjudication["event_id"])
        require(target["status"] == "truth_geometry_conflict", "adjudication does not resolve a frozen conflict")
        target["status"] = "adjudicated_exclude_from_score"
        target["adjudication_required"] = False
        target["adjudicated_event_truth"] = adjudication["adjudication"]["event_truth"]
        target["positive_gate_decision"] = adjudication["adjudication"]["positive_gate_decision"]
        target["strict_contact_alertable_interval_ms"] = None
    unresolved = [row["event_id"] for row in events if row["status"] == "truth_geometry_conflict"]
    consistent = [row for row in events if row["status"] == "consistent"]
    excluded = [row for row in events if row["status"] == "adjudicated_exclude_from_score"]
    all_consistent = not unresolved and not excluded and len(consistent) == EXPECTED_POSITIVES
    result = {
        "schema": RESULT_SCHEMA,
        "contract_id": protocol["contract_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": protocol["dataset_role"],
        "protocol_sha256": sha256_file(protocol_path),
        "input_sha256": {
            "continuous_r12a": sha256_file(paths["r12a"]),
            "r11_oracle": sha256_file(paths["r11_oracle"]),
            "r12_oracle": sha256_file(paths["r12_oracle"]),
        },
        "positive_event_count": len(events),
        "consistent_positive_event_count": len(consistent),
        "eligible_positive_event_count": len(consistent),
        "adjudicated_excluded_positive_event_count": len(excluded),
        "truth_geometry_conflict_count": len(conflicts),
        "truth_geometry_conflict_event_ids": conflicts,
        "unresolved_truth_geometry_conflict_event_ids": unresolved,
        "all_positive_truth_geometry_consistent": all_consistent,
        "model_adjudication_sha256": sha256_file(adjudication_path) if adjudication_path is not None else None,
        "events": events,
        "authorization": {
            "london_768_candidate_execution_authorized": all_consistent,
            "full_continuous_replay_authorized": False,
            "device_soak_authorized": False,
            "r13_inventory_unlock_authorized": False,
        },
        "next_action": (
            "export_and_run_the_single_preregistered_london_fp16_768_gpu_candidate"
            if all_consistent
            else (
                "preregister_one_non_r13_seen_positive_with_independent_event_truth_and_route_geometry_then_rerun_r12c"
                if excluded
                else "run_two_independent_model_reviews_and_third_model_adjudication_without_moving_the_old_polygon"
            )
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
    parser.add_argument("--adjudication", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate(args.protocol, args.output, args.adjudication)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "consistent_positive_events": result["consistent_positive_event_count"],
        "unresolved_truth_geometry_conflicts": result["unresolved_truth_geometry_conflict_event_ids"],
        "adjudicated_excluded_positive_events": result["adjudicated_excluded_positive_event_count"],
        "london_768_authorized": result["authorization"]["london_768_candidate_execution_authorized"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
