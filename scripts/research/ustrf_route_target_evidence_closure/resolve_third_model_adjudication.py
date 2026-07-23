from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg
from fuse_seen_person_proposals import greedy_matches, run_bytetrack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite adjudication result: {args.output}")

    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    fusion_config = config["seen_truth_proposal_protocol"]["fusion"]
    adjudicator = fusion_config["third_model_adjudicator"]
    criteria = adjudicator["resolution_contract"]
    expected_output = repo / adjudicator["planned_resolution_path"]
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("adjudication resolution path differs from preregistration")

    fusion_path = repo / fusion_config["materialized_output"]["path"]
    bundle_path = repo / adjudicator["planned_bundle_path"]
    proposal_path = repo / adjudicator["output_path"]
    fusion = load_json(fusion_path)
    bundle = load_json(bundle_path)
    proposal_payload = load_json(proposal_path)
    proposals = {frame["image_sha256"]: frame["person_proposals"] for frame in proposal_payload["frames"]}
    tracked, third_tracks = run_bytetrack(
        bundle["frames"],
        proposals,
        model_name="third_model",
        tracker_config=fusion_config["proposal_identity_tracker"],
    )

    match_iou = float(criteria["node_match_iou_min"])
    nodes_by_frame: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for tracklet in fusion["tracklets"]:
        if tracklet["status"] == "third_model_adjudication_required":
            for member in tracklet["members"]:
                nodes_by_frame[member["image_sha256"]].append((tracklet["proposal_track_id"], member))

    node_matches: dict[tuple[str, str, str], tuple[str, float]] = {}
    for image_sha256, tagged_nodes in nodes_by_frame.items():
        node_boxes = [{"bbox_xyxy": node["bbox_xyxy"]} for _, node in tagged_nodes]
        third_boxes = tracked.get(image_sha256, [])
        for node_index, third_index, overlap in greedy_matches(node_boxes, third_boxes, match_iou):
            tracklet_id, node = tagged_nodes[node_index]
            key = (tracklet_id, node["frame_id"], node["frame_node_id"])
            node_matches[key] = (third_boxes[third_index]["proposal_track_id"], overlap)

    third_to_tracklets: dict[str, set[str]] = defaultdict(set)
    for (tracklet_id, _, _), (third_track_id, _) in node_matches.items():
        third_to_tracklets[third_track_id].add(tracklet_id)
    cross_tracklet_third_ids = {track_id for track_id, owners in third_to_tracklets.items() if len(owners) > 1}

    resolved = []
    for tracklet in fusion["tracklets"]:
        if len(tracklet["person_identity_hints"]) == 1:
            seed_members = [member for member in tracklet["members"] if member["evidence"] == "frozen_seed_truth"]
            if seed_members:
                resolved.append({
                    **tracklet,
                    "members": seed_members,
                    "frame_count": len(seed_members),
                    "first_frame": min(member["frame_id"] for member in seed_members),
                    "last_frame": max(member["frame_id"] for member in seed_members),
                    "adjudication_decision": "accepted_frozen_seed_precedence",
                    "quarantine_reasons": [],
                    "quarantined_extension_reasons": ["non_seed_extensions_excluded_by_seed_precedence"] if len(seed_members) != len(tracklet["members"]) else [],
                })
                continue
        if tracklet["status"] == "consensus_tracklet":
            resolved.append({
                **tracklet,
                "adjudication_decision": "accepted_dual_model_consensus",
                "quarantine_reasons": [],
            })
            continue
        reasons = []
        if tracklet["identity_conflict"]:
            reasons.append("frozen_seed_identity_conflict")
        matched_lineages: dict[str, dict[str, set[str]]] = {
            "pass_a": defaultdict(set),
            "pass_b": defaultdict(set),
        }
        matched_third_ids = set()
        annotated_members = []
        for member in tracklet["members"]:
            key = (tracklet["proposal_track_id"], member["frame_id"], member["frame_node_id"])
            match = node_matches.get(key)
            needs_match = member["evidence"] in ("pass_a_only", "pass_b_only") or member["association_ambiguous"]
            if needs_match and match is None:
                reasons.append("required_node_without_third_model_match")
            third_track_id = match[0] if match else None
            matched_third_ids.update([third_track_id] if third_track_id else [])
            if match:
                for model_key, field in (("pass_a", "pass_a_track_id"), ("pass_b", "pass_b_track_id")):
                    if member.get(field):
                        matched_lineages[model_key][third_track_id].add(member[field])
            annotated_members.append({
                **member,
                "third_model_track_id": third_track_id,
                "third_model_iou": round(match[1], 6) if match else None,
            })
        if any(track_id in cross_tracklet_third_ids for track_id in matched_third_ids):
            reasons.append("third_model_identity_crosses_fusion_tracklets")
        for model_key in ("pass_a", "pass_b"):
            if any(len(lineages) > 1 for lineages in matched_lineages[model_key].values()):
                reasons.append(f"third_model_maps_multiple_{model_key}_lineages")
        reasons = sorted(set(reasons))
        if reasons and len(tracklet["person_identity_hints"]) == 1:
            seed_members = [member for member in annotated_members if member["evidence"] == "frozen_seed_truth"]
            if seed_members:
                resolved.append({
                    **tracklet,
                    "members": seed_members,
                    "frame_count": len(seed_members),
                    "first_frame": min(member["frame_id"] for member in seed_members),
                    "last_frame": max(member["frame_id"] for member in seed_members),
                    "adjudication_decision": "accepted_frozen_seed_precedence",
                    "quarantine_reasons": [],
                    "quarantined_extension_reasons": reasons,
                })
                continue
        resolved.append({
            **tracklet,
            "members": annotated_members,
            "adjudication_decision": "quarantined_unresolved" if reasons else "accepted_third_model_adjudicated",
            "quarantine_reasons": reasons,
        })

    payload = {
        "schema": "blindassist_ustrf_seen_person_identity_adjudication_r1",
        "authority": "model_proxy_identity_truth_with_unresolved_episodes_quarantined_not_human_truth_or_candidate_credit",
        "config_sha256": sha256_file(args.config),
        "fusion_sha256": sha256_file(fusion_path),
        "adjudication_bundle_sha256": sha256_file(bundle_path),
        "third_model_proposals_sha256": sha256_file(proposal_path),
        "candidate_alerts_exposed": False,
        "baseline_app_detector_outputs_exposed": False,
        "node_match_iou_min": match_iou,
        "third_model_track_count": len(third_tracks),
        "tracklet_count": len(resolved),
        "accepted_tracklet_count": sum(row["adjudication_decision"].startswith("accepted_") for row in resolved),
        "quarantined_tracklet_count": sum(row["adjudication_decision"] == "quarantined_unresolved" for row in resolved),
        "seed_precedence_tracklet_count": sum(row["adjudication_decision"] == "accepted_frozen_seed_precedence" for row in resolved),
        "tracklets": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "third_model_track_count", "tracklet_count", "accepted_tracklet_count", "quarantined_tracklet_count"
    )} | {"sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
