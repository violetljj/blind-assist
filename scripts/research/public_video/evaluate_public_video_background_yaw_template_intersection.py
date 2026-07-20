#!/usr/bin/env python3
"""Intersect r803 background-yaw choices with the frozen r799 route templates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import explicit_route_intent_fusion as fusion
import evaluate_public_video_explicit_choice_route_template as choice
import evaluate_public_video_explicit_route_intent_oracle as exact
import run_public_silver_frozen_feature_probe as common
import run_public_video_future_ego_trace_probe as point_hit


SCHEMA = "blindassist_public_video_background_yaw_template_intersection_v1"


def score_candidate(candidate: dict[str, Any], source: dict[str, Any], template_spec: dict[str, Any],
                    lifecycle_spec: dict[str, Any]) -> dict[str, Any]:
    direction = candidate["direction"]
    if direction not in lifecycle_spec["directions_allowed"]:
        raise ValueError(f"unsupported direction: {direction}")
    points = choice.template_points(template_spec, direction)
    width, height = choice.source_dimensions(source)
    samples = {int(row["timestamp_ms"]): row for row in source["samples"]}
    risk_samples = []
    frames = []
    for timestamp in map(int, candidate["timestamps_ms"]):
        detections = samples[timestamp].get("detections", [])
        hits = [point_hit.point_hits_expanded_detection(
            point, detections, width, height,
            float(template_spec["obstacle_expansion_object_heights"])
        ) for point in points]
        fraction = sum(hits) / len(hits)
        risk_samples.append(fusion.RouteRiskSample(timestamp, True, fraction))
        frames.append({"timestamp_ms": timestamp, "intersection_fraction": fraction, "hits": hits,
                       "detection_count": len(detections)})
    transitions = fusion.decode_route_risk_lifecycle(
        risk_samples,
        threshold=float(lifecycle_spec["frame_intersection_fraction_threshold"]),
        open_consecutive=int(lifecycle_spec["open_consecutive_one_second_samples"]),
        clear_consecutive=int(lifecycle_spec["clear_consecutive_one_second_samples"]),
        expected_step_ms=int(lifecycle_spec["expected_step_ms"]),
    )
    first_open = fusion.first_intervention_timestamp(transitions)
    return {**candidate, "template_points_xy_norm": points, "frames": frames,
            "template_intervention": first_open is not None,
            "first_template_open_timestamp_ms": first_open, "transitions": transitions}


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite background-yaw template report")
    contract = common.load_json(contract_path)
    bound = contract["bound_inputs"]
    paths = {key[:-5]: Path(value) for key, value in bound.items() if key.endswith("_path")}
    for stem, path in paths.items():
        if common.sha256_file(path) != bound[f"{stem}_sha256"]:
            raise ValueError(f"bound input hash mismatch: {stem}")
    r803 = common.load_json(paths["r803_report"])
    if r803.get("review_queue_ready") is not True:
        raise ValueError("r803 review queue did not pass")
    sources, feature_hashes, video_hashes = exact.merge_feature_sources(common.load_json(paths["source_feature_contract"]))
    template_spec = common.load_json(paths["r799_template_contract"])["fixed_templates"]
    scored = [score_candidate(candidate, sources[candidate["parent_source_id"]], template_spec, contract["scoring"])
              for candidate in r803["candidates"]]
    active = [row for row in scored if row["template_intervention"]]
    by_direction = {name: [row for row in active if row["direction"] == name] for name in ("LEFT", "RIGHT")}
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   **{f"{stem}_sha256": common.sha256_file(path) for stem, path in paths.items()},
                   "verified_feature_reports": feature_hashes, "verified_local_videos": video_hashes},
        "summary": {"candidate_count": len(scored), "template_intervention_count": len(active),
                    "left_template_intervention_count": len(by_direction["LEFT"]),
                    "right_template_intervention_count": len(by_direction["RIGHT"]),
                    "left_source_count": len({row["parent_source_id"] for row in by_direction["LEFT"]}),
                    "right_source_count": len({row["parent_source_id"] for row in by_direction["RIGHT"]})},
        "candidates": scored,
        "review_candidate_ids": [row["candidate_id"] for row in active],
        "review_queue_ready": bool(active),
        "authorization": contract["authorization"],
        "evidence_limit": "Template intersection is deterministic detector geometry on a retrospective turn candidate. It remains a model-review proposal, not event truth or coverage credit."
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
    report = run(args.contract, args.output)
    print(json.dumps({"summary": report["summary"], "review_queue_ready": report["review_queue_ready"]}))


if __name__ == "__main__":
    main()
