#!/usr/bin/env python3
"""Validate the non-R1.3 seen-positive replacement preregistration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .contract import load_json, sha256_file, write_json
    from .projected_corridor_geometry import classify_contact_point, robust_relation
except ImportError:
    from contract import load_json, sha256_file, write_json
    from projected_corridor_geometry import classify_contact_point, robust_relation


SCHEMA = "blindassist_ustrf_crosscam_seen_positive_preregistration_v1"
RESULT_SCHEMA = "blindassist_ustrf_crosscam_seen_positive_preregistration_result_v1"
UNCERTAINTY_RATIOS = [0.01, 0.02, 0.03]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_bound(repo: Path, value: str, sha256: str) -> Path:
    path = (repo / value).resolve()
    require(path.is_relative_to(repo), f"referenced path escapes repository: {value}")
    require(path.is_file(), f"missing hash-bound input: {value}")
    require(sha256_file(path) == sha256, f"SHA-256 mismatch: {value}")
    return path


def validate(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    repo = contract_path.parent.parent
    contract = load_json(contract_path)
    require(contract.get("schema") == SCHEMA, "seen-positive schema mismatch")
    require(contract.get("dataset_role") == "seen_diagnostic_not_held_out", "replacement must be seen diagnostic")

    parents = contract["parents"]
    r12c_v1 = resolve_bound(repo, parents["r12c_v1_path"], parents["r12c_v1_sha256"])
    r13_v2 = resolve_bound(repo, parents["r13_v2_path"], parents["r13_v2_sha256"])
    r13 = load_json(r13_v2)
    novelty = r13["novelty_and_access"]
    require(novelty["source_discovery_authorized"] is False, "R1.3 discovery is not sealed")
    require(novelty["download_decode_or_detector_inference_authorized"] is False, "R1.3 data access is not sealed")
    require(novelty["result_access_authorized"] is False, "R1.3 results are not sealed")

    source = contract["source"]
    resolve_bound(repo, source["local_video_path"], source["video_sha256"])
    resolve_bound(repo, source["source_registry_path"], source["source_registry_sha256"])
    for evidence in source["seen_evidence"]:
        resolve_bound(repo, evidence["path"], evidence["sha256"])
    require(source["first_opened_at_utc"] < contract["frozen_at_utc"], "replacement was not seen before freeze")

    guard = contract["non_r13_guard"]
    for key in ("r13_source_discovery_performed", "r13_download_decode_or_detector_inference_performed",
                "r13_result_access_performed", "r13_slot_consumed", "new_held_out_claim_authorized"):
        require(guard[key] is False, f"non-R1.3 guard violated: {key}")
    require(guard["replacement_source_was_already_seen_before_r12c"] is True, "replacement is not declared seen")

    review = contract["independent_model_review"]
    require(review["reviewer_a_and_b_independent"] is True, "reviews were not independent")
    require(review["reviewer_a_and_b_saw_each_others_outputs"] is False, "reviewers saw each other's result")
    require(review["detector_tracker_or_association_outputs_read"] is False, "review consumed detector evidence")
    require(review["r13_inventory_read"] is False, "review consumed R1.3")
    require(review["agreement"] is True and review["third_adjudicator_required"] is False,
            "replacement truth has unresolved model disagreement")
    require(review["reviewer_a_decision"] == review["reviewer_b_decision"] == contract["event"]["event_truth"],
            "review decisions do not match frozen event truth")

    geometry = contract["geometry_contract"]
    require(geometry["uncertainty_frame_ratios"] == UNCERTAINTY_RATIOS, "uncertainty ratios drifted")
    require(geometry["route_polygon_may_be_moved_after_detector_result"] is False, "polygon rescue must stay forbidden")
    require(geometry["detector_outputs_may_define_target_or_route"] is False, "detector cannot define geometry")
    width, height = int(geometry["frame_width"]), int(geometry["frame_height"])
    anchor_results = []
    for anchor in geometry["anchors"]:
        resolve_bound(repo, anchor["image_path"], anchor["image_sha256"])
        contact = anchor.get("contact_xy_px")
        point = contact if contact is not None else anchor["visible_boundary_proxy_xy_px"]
        profiles = [
            classify_contact_point(point, frame_width=width, frame_height=height,
                                   polygon_xy_norm=anchor["route_polygon_xy_norm"], uncertainty_frame_ratio=ratio)
            for ratio in UNCERTAINTY_RATIOS
        ]
        robust = robust_relation([profile.relation for profile in profiles])
        expected = anchor.get("expected_robust_relation", anchor.get("expected_visible_proxy_relation"))
        require(robust == expected, f"{anchor['frame_id']}: expected {expected}, got {robust}")
        anchor_results.append({
            "frame_id": anchor["frame_id"],
            "timestamp_ms": anchor["timestamp_ms"],
            "role": anchor["role"],
            "scored_contact": contact is not None,
            "robust_relation": robust,
            "boundary_distance_px": profiles[0].boundary_distance_px,
            "profile_relations": [profile.relation for profile in profiles],
        })

    alertable_inside = [row for row in anchor_results
                        if row["role"] == "alertable_positive" and row["robust_relation"] == "inside"]
    clearance = [row for row in anchor_results
                 if row["role"] == "clearance_visible_truncation_not_contact_scored"]
    gate = contract["replacement_gate"]
    require(len(alertable_inside) >= gate["minimum_alertable_robust_inside_anchor_count"],
            "replacement lacks robust-inside alertable anchors")
    require(len(clearance) == 1 and clearance[0]["robust_relation"] == "outside",
            "replacement clearance proxy is not robust outside")
    require(gate["existing_eligible_positive_count"] + gate["replacement_positive_count"]
            == gate["eligible_positive_count_after_validation"] == 6, "replacement does not close the sixth slot")

    authority = contract["authority"]
    for key in ("new_held_out_read", "london_768_candidate_execution_authorized",
                "full_continuous_replay_authorized", "device_soak_authorized",
                "r13_inventory_unlock_authorized", "training_authorized",
                "app_default_backend_change_authorized", "production_model_replacement_authorized"):
        require(authority[key] is False, f"preregistration over-authorizes {key}")

    return {
        "schema": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(contract_path),
        "r12c_v1_sha256": sha256_file(r12c_v1),
        "r13_v2_sha256": sha256_file(r13_v2),
        "source_id": source["source_id"],
        "event_id": contract["event"]["event_id"],
        "replaces_event_id": contract["event"]["replaces_event_id"],
        "alertable_robust_inside_anchor_count": len(alertable_inside),
        "anchor_results": anchor_results,
        "candidate_qualified_as_non_r13_seen_positive": True,
        "eligible_positive_count_after_validation": 6,
        "r13_slot_consumed": False,
        "next_action": "materialize_r12c_v2_with_bangkok_replacing_japan_then_rerun_full_six_positive_oracle_before_768",
        "authorization": authority,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate(args.contract)
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"refusing to overwrite output: {args.output}")
            write_json(args.output, result)
            Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "event_id": result["event_id"],
        "alertable_robust_inside_anchors": result["alertable_robust_inside_anchor_count"],
        "eligible_positive_count": result["eligible_positive_count_after_validation"],
        "r13_slot_consumed": result["r13_slot_consumed"],
        "london_768_authorized": result["authorization"]["london_768_candidate_execution_authorized"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
