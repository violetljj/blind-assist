#!/usr/bin/env python3
"""Test whether LEFT/STRAIGHT/RIGHT templates preserve the exact-route oracle gain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2

import explicit_route_intent_fusion as fusion
import evaluate_public_video_explicit_route_intent_oracle as exact
import run_public_silver_frozen_feature_probe as common
import run_public_video_future_ego_trace_probe as point_hit


SCHEMA = "blindassist_public_video_explicit_choice_route_template_v1"


def direction_from_mean_x(mean_x: float, left_below: float, right_above: float) -> str:
    if mean_x < left_below:
        return "LEFT"
    if mean_x > right_above:
        return "RIGHT"
    return "STRAIGHT"


def template_points(spec: dict[str, Any], direction: str) -> list[tuple[float, float]]:
    xs = spec[f"{direction}_x_norm"]
    ys = spec["y_norm"]
    if len(xs) != 3 or len(ys) != 3:
        raise ValueError("route template must contain three points")
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def source_dimensions(source: dict[str, Any]) -> tuple[int, int]:
    width = int(source.get("video_width") or 0)
    height = int(source.get("video_height") or 0)
    if width > 0 and height > 0:
        return width, height
    capture = cv2.VideoCapture(str(source["local_video_path"]))
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"unable to read video dimensions: {source['source_id']}")
    return width, height


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite explicit-choice output")
    contract = common.load_json(contract_path)
    bound = contract["bound_inputs"]
    exact_path = Path(bound["r797a_report_path"])
    feature_contract_path = Path(bound["r790_feature_contract_path"])
    if common.sha256_file(exact_path) != bound["r797a_report_sha256"]:
        raise ValueError("r797a report hash mismatch")
    if common.sha256_file(feature_contract_path) != bound["r790_feature_contract_sha256"]:
        raise ValueError("feature contract hash mismatch")
    exact_report = common.load_json(exact_path)
    if exact_report.get("explicit_route_intent_interface_supported") is not True:
        raise ValueError("exact route-intent oracle did not pass")
    sources, verified_features, verified_videos = exact.merge_feature_sources(common.load_json(feature_contract_path))
    proxy = contract["choice_proxy"]
    template_spec = contract["fixed_templates"]
    threshold = float(contract["lifecycle"]["frame_intersection_fraction_threshold"])
    events = []
    for event in exact_report["events"]:
        source = sources[event["parent_source_id"]]
        exact_x = [float(anchor["point_xy_norm"][0]) for frame in event["frames"] for anchor in frame["anchors"]]
        if not exact_x:
            raise ValueError(f"event has no exact waypoints: {event['item_id']}")
        mean_x = float(np.mean(exact_x))
        direction = direction_from_mean_x(mean_x, float(proxy["left_if_mean_x_below"]),
                                          float(proxy["right_if_mean_x_above"]))
        points = template_points(template_spec, direction)
        width, height = source_dimensions(source)
        samples_by_time = {int(row["timestamp_ms"]): row for row in source["samples"]}
        risk_samples = []
        frame_rows = []
        for frame in event["frames"]:
            timestamp = int(frame["timestamp_ms"])
            detections = samples_by_time[timestamp].get("detections", [])
            hits = [point_hit.point_hits_expanded_detection(
                point, detections, width, height,
                float(template_spec["obstacle_expansion_object_heights"])
            ) for point in points]
            score = sum(hits) / len(hits)
            risk_samples.append(fusion.RouteRiskSample(timestamp, True, score))
            frame_rows.append({"timestamp_ms": timestamp, "intersection_fraction": score, "hits": hits})
        transitions = fusion.decode_route_risk_lifecycle(
            risk_samples, threshold=threshold,
            open_consecutive=int(contract["lifecycle"]["open_consecutive_one_second_samples"]),
            clear_consecutive=int(contract["lifecycle"]["clear_consecutive_one_second_samples"]),
        )
        open_timestamp = fusion.first_intervention_timestamp(transitions)
        predicted = open_timestamp is not None
        reference = bool(event["reference_intervention_required"])
        events.append({
            "item_id": event["item_id"], "parent_source_id": event["parent_source_id"],
            "reference_intervention_required": reference, "proxy_mean_exact_x": mean_x,
            "explicit_choice": direction, "template_points_xy_norm": points,
            "template_intervention": predicted, "first_template_open_timestamp_ms": open_timestamp,
            "frames": frame_rows, "agreement": predicted == reference,
        })
    positives = [row for row in events if row["reference_intervention_required"]]
    negatives = [row for row in events if not row["reference_intervention_required"]]
    positive_recall = sum(row["template_intervention"] for row in positives) / len(positives)
    negative_recall = sum(not row["template_intervention"] for row in negatives) / len(negatives)
    balanced = (positive_recall + negative_recall) / 2.0
    gate = contract["gate"]
    checks = {
        "minimum_event_count": len(events) >= int(gate["minimum_event_count"]),
        "minimum_source_count": len({row["parent_source_id"] for row in events}) >= int(gate["minimum_source_count"]),
        "minimum_intervention_event_recall": positive_recall >= float(gate["minimum_intervention_event_recall"]),
        "minimum_context_event_recall": negative_recall >= float(gate["minimum_context_event_recall"]),
        "minimum_balanced_accuracy": balanced >= float(gate["minimum_balanced_accuracy"]),
        "all_feature_and_video_hashes_valid": True,
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   "r797a_report_sha256": common.sha256_file(exact_path),
                   "verified_feature_reports": verified_features, "verified_local_videos": verified_videos},
        "summary": {"event_count": len(events), "source_count": len({row["parent_source_id"] for row in events}),
                    "intervention_event_recall": positive_recall, "context_event_recall": negative_recall,
                    "balanced_accuracy": balanced,
                    "direction_counts": {key: sum(row["explicit_choice"] == key for row in events)
                                         for key in ("LEFT", "STRAIGHT", "RIGHT")}},
        "events": events, "checks": checks,
        "three_state_choice_provider_supported": bool(all(checks.values())),
        "authorization": contract["authorization"],
        "evidence_limit": "Direction is proxied from the exact offline oracle; a pass still requires real user-choice sources."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output_path) + ".sha256").write_text(common.sha256_file(output_path) + "\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output)
    print(json.dumps({"summary": result["summary"],
                      "three_state_choice_provider_supported": result["three_state_choice_provider_supported"]}))


if __name__ == "__main__":
    main()
