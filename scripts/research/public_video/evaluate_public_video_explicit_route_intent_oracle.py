#!/usr/bin/env python3
"""Evaluate an explicit-route interface upper bound on causal actionability events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_actionability_linear_probe as actionability
import run_public_video_dense_future_ego_trace_probe as future_trace


SCHEMA = "blindassist_public_video_explicit_route_intent_oracle_v1"


def first_consecutive_open(frames: list[dict[str, Any]], threshold: float) -> int | None:
    previous_time: int | None = None
    for frame in frames:
        timestamp = int(frame["timestamp_ms"])
        score = frame.get("trace_intrusion_score")
        active = score is not None and float(score) >= threshold
        if active and previous_time is not None and timestamp - previous_time == 1000:
            return timestamp
        previous_time = timestamp if active else None
    return None


def merge_feature_sources(feature_contract: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    sources: dict[str, dict[str, Any]] = {}
    bindings = []
    for key, binding in feature_contract["feature_reports"].items():
        path = Path(binding["path"])
        actual = common.sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        report = common.load_json(path)
        bindings.append({"feature_key": key, "path": path.as_posix(), "sha256": actual})
        for incoming in report["sources"]:
            source_id = incoming["source_id"]
            if source_id in sources:
                actionability.merge_source(sources[source_id], incoming)
            else:
                sources[source_id] = {**incoming, "samples": list(incoming["samples"])}
    verified_videos = []
    for source_id, source in sorted(sources.items()):
        video_path = Path(source["local_video_path"])
        actual_video_sha256 = common.sha256_file(video_path)
        if actual_video_sha256 != source["video_sha256"]:
            raise ValueError(f"local video hash mismatch: {source_id}")
        verified_videos.append({
            "source_id": source_id, "path": video_path.as_posix(), "sha256": actual_video_sha256
        })
    return sources, bindings, verified_videos


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("refusing to overwrite explicit-route oracle output")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    manifest_path = Path(bound["actionability_manifest_path"])
    feature_contract_path = Path(bound["r790_feature_contract_path"])
    teacher_contract_path = Path(bound["r755_teacher_contract_path"])
    for path, key in [(manifest_path, "actionability_manifest_sha256"),
                      (feature_contract_path, "r790_feature_contract_sha256"),
                      (teacher_contract_path, "r755_teacher_contract_sha256")]:
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    manifest = common.load_json(manifest_path)
    if manifest.get("deterministic_actionability_probe_ready") is not True:
        raise ValueError("actionability manifest is not probe-ready")
    feature_contract = common.load_json(feature_contract_path)
    teacher_contract = common.load_json(teacher_contract_path)
    sources, verified_bindings, verified_videos = merge_feature_sources(feature_contract)
    policy = dict(teacher_contract["teacher"])
    oracle = contract["oracle_route_intent"]
    for key in ("future_horizons_ms", "future_anchor_xy_norm", "valid_trace_bounds_xy_norm",
                "obstacle_expansion_object_heights"):
        if policy[key] != oracle[key]:
            raise ValueError(f"teacher policy drift: {key}")
    synthetic_features = {"sources": [sources[key] for key in sorted(sources)]}
    threshold = float(oracle["frame_intersection_threshold"])
    events = []
    for item in manifest["items"]:
        source_id = item["parent_source_id"]
        if source_id not in sources:
            raise ValueError(f"missing source features: {source_id}")
        diagnostic = future_trace.evaluate_event(
            synthetic_features, source_id, tuple(map(int, item["window_ms"])), policy
        )
        open_timestamp = first_consecutive_open(diagnostic["frames"], threshold)
        predicted = open_timestamp is not None
        reference = bool(item["intervention_required"])
        first_reference_open = next((int(row["timestamp_ms"]) for row in item["transitions"]
                                     if row["state"] == "intervention_needed"), None)
        events.append({
            "item_id": item["item_id"], "parent_source_id": source_id,
            "reference_actionability_class": item["actionability_class"],
            "reference_intervention_required": reference,
            "explicit_route_intervention": predicted,
            "first_explicit_route_open_timestamp_ms": open_timestamp,
            "first_reference_open_timestamp_ms": first_reference_open,
            "lead_vs_reference_ms": (first_reference_open - open_timestamp)
                if first_reference_open is not None and open_timestamp is not None else None,
            "valid_frame_fraction": diagnostic["valid_frame_fraction"],
            "mean_route_intersection": diagnostic["mean_trace_intrusion_score"],
            "frames": diagnostic["frames"],
            "agreement": predicted == reference,
            "disagreement_interpretation": None if predicted == reference else
                "explicit_plan_changes_information_state; do not rewrite current/past-only reference",
        })
    positives = [row for row in events if row["reference_intervention_required"]]
    negatives = [row for row in events if not row["reference_intervention_required"]]
    positive_recall = sum(row["explicit_route_intervention"] for row in positives) / len(positives)
    negative_recall = sum(not row["explicit_route_intervention"] for row in negatives) / len(negatives)
    balanced = (positive_recall + negative_recall) / 2.0
    source_count = len({row["parent_source_id"] for row in events})
    minimum_valid = min(row["valid_frame_fraction"] for row in events)
    gate = contract["gate"]
    checks = {
        "minimum_event_count": len(events) >= int(gate["minimum_event_count"]),
        "minimum_source_count": source_count >= int(gate["minimum_source_count"]),
        "minimum_intervention_event_recall": positive_recall >= float(gate["minimum_intervention_event_recall"]),
        "minimum_context_event_recall": negative_recall >= float(gate["minimum_context_event_recall"]),
        "minimum_balanced_accuracy": balanced >= float(gate["minimum_balanced_accuracy"]),
        "maximum_context_false_upgrade_rate": (1.0 - negative_recall) <= float(gate["maximum_context_false_upgrade_rate"]),
        "minimum_valid_frame_fraction": minimum_valid >= float(gate["minimum_valid_frame_fraction"]),
        "all_source_and_video_hash_bindings_valid": True,
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "actionability_manifest_sha256": common.sha256_file(manifest_path),
                   "r790_feature_contract_sha256": common.sha256_file(feature_contract_path),
                   "r755_teacher_contract_sha256": common.sha256_file(teacher_contract_path),
                   "verified_feature_reports": verified_bindings,
                   "verified_local_videos": verified_videos},
        "oracle_route_intent": oracle,
        "summary": {"event_count": len(events), "source_count": source_count,
                    "intervention_event_count": len(positives), "context_event_count": len(negatives),
                    "intervention_event_recall": positive_recall, "context_event_recall": negative_recall,
                    "balanced_accuracy": balanced, "context_false_upgrade_rate": 1.0 - negative_recall,
                    "minimum_valid_frame_fraction": minimum_valid},
        "events": events, "checks": checks, "explicit_route_intent_interface_supported": bool(all(checks.values())),
        "evidence_limit": contract["comparison"]["interpretation"],
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"summary": result["summary"],
                      "explicit_route_intent_interface_supported": result["explicit_route_intent_interface_supported"]}))
