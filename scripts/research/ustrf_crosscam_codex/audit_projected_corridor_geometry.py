#!/usr/bin/env python3
"""Audit fixed-width bbox gating against a projected polygon footpoint gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import BUNDLE_SCHEMA, CONTRACT_ID, load_json, require_false_flags, sha256_file, write_json
    from .projected_corridor_geometry import classify_bottom_center, robust_relation
except ImportError:  # Direct execution through scripts/run_research_tool.py.
    from contract import BUNDLE_SCHEMA, CONTRACT_ID, load_json, require_false_flags, sha256_file, write_json
    from projected_corridor_geometry import classify_bottom_center, robust_relation


SCHEMA = "blindassist_ustrf_crosscam_projected_corridor_audit_v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    bundle = load_json(args.bundle_manifest)
    config = load_json(args.config)
    projection = load_json(args.projection_receipt)
    android = load_json(args.android_output)
    if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("contract_id") != CONTRACT_ID:
        raise ValueError("bundle schema/contract mismatch")
    require_false_flags(bundle, "bundle")
    if config.get("schema") != "blindassist_ustrf_crosscam_projected_corridor_config_v1" or config.get("contract_id") != CONTRACT_ID:
        raise ValueError("projected corridor config schema/contract mismatch")
    require_false_flags(config, "config")
    if projection.get("schema") != "blindassist_ustrf_crosscam_route_projection_receipt_v1" or projection.get("contract_id") != CONTRACT_ID:
        raise ValueError("route projection receipt schema/contract mismatch")
    require_false_flags(projection, "route projection receipt")
    if projection.get("authority") != "manual_current_frame_proxy_only" or projection.get("dynamic_projection_present") is not False:
        raise ValueError("R1 audit only admits disclosed static manual proxy projection")
    profiles = config.get("projection_uncertainty_frame_ratios")
    if not isinstance(profiles, list) or len(profiles) < 3 or profiles != sorted(profiles):
        raise ValueError("config needs sorted narrow/nominal/wide projection uncertainty profiles")
    polygon = projection["route_polygon_xy_norm"]
    if polygon != bundle["assumed_geometry"]["route_polygon_xy_norm"]:
        raise ValueError("projection receipt polygon differs from bundle")
    decoded = {row["frame_id"]: row for row in android["android_backend_receipt"]["decoded_frames"]}
    rows: list[dict[str, Any]] = []
    for frame in android["route_conditioning_receipt"]["frames"]:
        frame_id = frame["frame_id"]
        width, height = int(decoded[frame_id]["width"]), int(decoded[frame_id]["height"])
        for detection in frame["detections"]:
            classifications = [
                classify_bottom_center(
                    detection["source_box_xyxy_px"], frame_width=width, frame_height=height,
                    polygon_xy_norm=polygon, uncertainty_frame_ratio=float(ratio),
                )
                for ratio in profiles
            ]
            rows.append({
                "frame_id": frame_id,
                "video_pts_ms": frame["frame_timestamp_ms"],
                "detection_index": detection["detection_index"],
                "label_diagnostic_only": detection["label"],
                "confidence": detection["confidence"],
                "fixed_polyline_kept": detection["kept"],
                "fixed_polyline_minimum_distance_px": detection["minimum_route_distance_px"],
                "fixed_polyline_half_width_px": detection["corridor_half_width_px"],
                "footpoint_xy_px": list(classifications[0].footpoint_xy_px),
                "nominal_polygon_inside": classifications[0].nominal_inside,
                "polygon_boundary_distance_px": classifications[0].boundary_distance_px,
                "sensitivity": [
                    {
                        "uncertainty_frame_ratio": ratio,
                        "uncertainty_px": classification.uncertainty_px,
                        "relation": classification.relation,
                    }
                    for ratio, classification in zip(profiles, classifications)
                ],
                "robust_relation": robust_relation([classification.relation for classification in classifications]),
            })
    current_kept = [row for row in rows if row["fixed_polyline_kept"]]
    robust_inside = [row for row in rows if row["robust_relation"] == "inside"]
    uncertain = [row for row in rows if row["robust_relation"] == "uncertain_boundary"]
    profile_summary = []
    for profile_index, ratio in enumerate(profiles):
        relations = [row["sensitivity"][profile_index]["relation"] for row in rows]
        profile_summary.append({
            "uncertainty_frame_ratio": ratio,
            "inside_count": relations.count("inside"),
            "uncertain_boundary_count": relations.count("uncertain_boundary"),
            "outside_count": relations.count("outside"),
            "current_fixed_polyline_kept_inside_count": sum(
                row["fixed_polyline_kept"] and relation == "inside"
                for row, relation in zip(rows, relations)
            ),
            "current_fixed_polyline_kept_uncertain_count": sum(
                row["fixed_polyline_kept"] and relation == "uncertain_boundary"
                for row, relation in zip(rows, relations)
            ),
            "current_fixed_polyline_kept_outside_count": sum(
                row["fixed_polyline_kept"] and relation == "outside"
                for row, relation in zip(rows, relations)
            ),
        })
    report = {
        "schema": SCHEMA,
        "contract_id": CONTRACT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256_file(args.bundle_manifest),
        "android_output_sha256": sha256_file(args.android_output),
        "projection_receipt_sha256": sha256_file(args.projection_receipt),
        "config_sha256": sha256_file(args.config),
        "route_polygon_xy_norm": polygon,
        "policy": {
            "object_geometry": "bbox_bottom_center_ground_contact_proxy_v1",
            "route_geometry": "explicit_convex_polygon_current_camera_frame_v1",
            "classification": "inside_outside_uncertain_by_boundary_distance_v1",
            "robust_rule": "all_sensitivity_profiles_must_agree_v1",
            "labels_used_for_gate": False,
        },
        "summary": {
            "detection_count": len(rows),
            "current_fixed_polyline_kept_count": len(current_kept),
            "projected_polygon_robust_inside_count": len(robust_inside),
            "projected_polygon_uncertain_count": len(uncertain),
            "projected_polygon_robust_outside_count": len(rows) - len(robust_inside) - len(uncertain),
            "current_kept_reclassified_uncertain_count": sum(
                row["fixed_polyline_kept"] and row["robust_relation"] == "uncertain_boundary" for row in rows
            ),
            "current_kept_reclassified_outside_count": sum(
                row["fixed_polyline_kept"] and row["robust_relation"] == "outside" for row in rows
            ),
        },
        "profile_summary": profile_summary,
        "detections": rows,
        "decision": "GEOMETRY_AUDIT_ONLY_NOT_CANDIDATE_RANKING",
        "human_event_truth_present": False,
        "metric_geometry_present": False,
        "training_authorized": False,
        "u0_authority_granted": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    write_json(args.output, report)
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--android-output", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
