"""Build Brain-ready Silver-B Development episodes from reviewed frame truth.

This adapter keeps the frozen P0 evaluator and the P0-S0 materializer intact.
Grounding DINO proposals are candidate inputs only.  UNIQUE or SET_VALUED
semantics must arrive in a separate, explicitly reviewed annotation document;
the adapter never infers referent truth from proposal scores or map geometry.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_grounding import p0_evaluator
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


POLICY_ID = "P0-SILVER-B-BRAIN-COHORT-ADAPTER-V1"
RESOLUTIONS = {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}
DISTRACTOR_ROLES = {
    "OTHER_BUILDING_ENTRANCE",
    "NON_ENTRANCE_DOOR",
    "SAME_BUILDING_NON_TARGET_ENTRANCE",
}


class BrainCohortError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BrainCohortError(message)


def _region(frame_id: str, bbox: Sequence[float], width: int, height: int) -> dict[str, Any]:
    _require(len(bbox) == 4 and width > 0 and height > 0, "invalid proposal bbox or image dimensions")
    values = [float(bbox[0]) / width, float(bbox[1]) / height, float(bbox[2]) / width, float(bbox[3]) / height]
    _require(0.0 <= values[0] < values[2] <= 1.0 and 0.0 <= values[1] < values[3] <= 1.0, "proposal bbox outside image")
    return {
        "frame_id": frame_id,
        "coordinate_space": "NORMALIZED_XYXY",
        "x_min": round(values[0], 12),
        "y_min": round(values[1], 12),
        "x_max": round(values[2], 12),
        "y_max": round(values[3], 12),
    }


def _proposal_candidates(
    frame_id: str,
    image: Mapping[str, Any],
    receipt_image: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(receipt_image.get("image_sha256") == image.get("image_sha256"), "proposal/image hash mismatch")
    result = []
    for rank, proposal in enumerate(receipt_image.get("proposals", []), start=1):
        result.append({
            "candidate_id": f"gdino-{frame_id}-{rank:03d}",
            "region": _region(frame_id, proposal["bbox_xyxy"], int(image["width"]), int(image["height"])),
            "category_label": str(proposal["label"]),
            "proposal_score": float(proposal["score"]),
            "provider_rank": rank,
            "provider_id": "grounding-dino-tiny",
            "score_semantics": "MODEL_PROPOSAL_RANKING_SCORE_NOT_TRUTH",
        })
    return result


def _annotation_index(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _require(document.get("schema_version") == 1, "annotation schema_version must be 1")
    _require(document.get("annotation_authority") == "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B", "unsupported annotation authority")
    annotations = document.get("episodes")
    _require(isinstance(annotations, list), "annotation episodes missing")
    result: dict[str, Mapping[str, Any]] = {}
    for annotation in annotations:
        _require(isinstance(annotation, Mapping) and annotation.get("episode_id"), "annotation episode identity missing")
        _require(annotation.get("source_episode_id"), "annotation source_episode_id missing")
        episode_id = str(annotation["episode_id"])
        _require(episode_id not in result, "duplicate annotation episode_id")
        result[episode_id] = annotation
    return result


def _target_annotations(
    annotation: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets = []
    regions = []
    for item in annotation.get("valid_targets", []):
        candidate_id = item.get("candidate_id")
        if candidate_id is not None:
            _require(str(candidate_id) in candidates, "valid target audit reference names unknown proposal candidate")
        values = item.get("region_normalized_xyxy")
        _require(isinstance(values, list) and len(values) == 4, "valid target requires independent normalized region")
        normalized = [float(value) for value in values]
        _require(
            0.0 <= normalized[0] < normalized[2] <= 1.0
            and 0.0 <= normalized[1] < normalized[3] <= 1.0,
            "valid target normalized region is invalid",
        )
        frame_id = str(next(iter(candidates.values()))["region"]["frame_id"])
        region = {
            "frame_id": frame_id,
            "coordinate_space": "NORMALIZED_XYXY",
            "x_min": round(normalized[0], 12),
            "y_min": round(normalized[1], 12),
            "x_max": round(normalized[2], 12),
            "y_max": round(normalized[3], 12),
        }
        targets.append({
            "target_instance_id": str(item["target_instance_id"]),
            "target_name": str(annotation["target_name"]),
            "relation": "entrance_of",
            "regions": [region],
        })
        regions.append(region)
    return targets, regions


def _distractors(annotation: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in annotation.get("distractors", []):
        candidate_id = str(item.get("candidate_id"))
        role = str(item.get("semantic_role"))
        _require(candidate_id in candidates, "distractor references unknown proposal candidate")
        _require(role in DISTRACTOR_ROLES, "unknown distractor semantic role")
        result.append({
            "distractor_instance_id": str(item["distractor_instance_id"]),
            "semantic_role": role,
            "region": dict(candidates[candidate_id]["region"]),
        })
    return result


def _evaluator_episode(
    source_episode: Mapping[str, Any],
    annotation: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolution = str(annotation.get("resolution"))
    _require(resolution in RESOLUTIONS, "unknown referent resolution")
    by_id = {str(item["candidate_id"]): item for item in candidates}
    targets, target_regions = _target_annotations(annotation, by_id)
    if resolution == "UNIQUE":
        _require(len(targets) == 1, "UNIQUE requires exactly one reviewed target")
    elif resolution == "SET_VALUED":
        _require(len(targets) >= 2, "SET_VALUED requires at least two reviewed targets")
    else:
        _require(not targets, "AMBIGUOUS cannot carry reviewed targets")

    frame_id = str(source_episode["frame_id"])
    timestamp = int(source_episode["captured_at"])
    scene = annotation.get("scene_condition")
    _require(isinstance(scene, Mapping), "scene_condition missing")
    visible = resolution != "AMBIGUOUS"
    episode = {
        "schema_version": 1,
        "episode_id": str(annotation["episode_id"]),
        "goal_spec": {
            "goal_type": "NAMED_BUILDING_ENTRANCE",
            "target_name": str(annotation["target_name"]),
            "requested_relation": "entrance_of",
        },
        "observation_window": {
            "frame_ids": [frame_id],
            "start_timestamp_ms": timestamp,
            "end_timestamp_ms": timestamp,
        },
        "observation_valid": True,
        "goal_reference_resolution": resolution,
        "target_visible": visible,
        "valid_target_instances": targets,
        "acceptable_spatial_regions": target_regions,
        "distractor_instances": _distractors(annotation, by_id),
        "target_min_side_px": float(annotation["target_min_side_px"]) if visible else None,
        "visibility_fraction": float(annotation["visibility_fraction"]) if visible else None,
        "text_support": str(annotation["text_support"]) if visible else "NOT_APPLICABLE",
        "scene_condition": dict(scene),
        "grounding_expectation": "MUST_GROUND" if visible else "MUST_BE_AMBIGUOUS",
    }
    try:
        p0_evaluator.validate_episode(episode)
    except p0_evaluator.EpisodeContractError as error:
        raise BrainCohortError(f"frozen evaluator rejected {episode['episode_id']}: {error}") from error
    return episode


def build_brain_cohort(
    silver_b_report: Mapping[str, Any],
    metadata_document: Mapping[str, Any],
    proposal_receipt: Mapping[str, Any],
    annotation_document: Mapping[str, Any],
) -> dict[str, Any]:
    _require(silver_b_report.get("development_quality_class", "SILVER_B_MAP_GEOMETRY") == "SILVER_B_MAP_GEOMETRY", "unexpected Silver-B class")
    metadata = {str(item["id"]): item for item in metadata_document.get("images", [])}
    receipt = {str(item["image_id"]): item for item in proposal_receipt.get("images", [])}
    annotations = _annotation_index(annotation_document)
    source_episodes = {str(item["episode_id"]): item for item in silver_b_report.get("episodes", [])}
    output_episodes = []
    resolution_counts = {value: 0 for value in sorted(RESOLUTIONS)}
    source_frame_ids: set[str] = set()
    for episode_id, annotation in annotations.items():
        source_episode_id = str(annotation["source_episode_id"])
        _require(source_episode_id in source_episodes, "annotation references non-Silver-B source episode")
        source_episode = source_episodes[source_episode_id]
        frame_id = str(source_episode["frame_id"])
        source_frame_ids.add(frame_id)
        _require(frame_id in metadata and frame_id in receipt, "reviewed frame missing metadata or proposal receipt")
        candidates = _proposal_candidates(frame_id, metadata[frame_id], receipt[frame_id])
        _require(bool(candidates), "reviewed frame has no proposal candidates")
        evaluator_episode = _evaluator_episode(source_episode, annotation, candidates)
        resolution = evaluator_episode["goal_reference_resolution"]
        resolution_counts[resolution] += 1
        output_episodes.append({
            "episode_id": episode_id,
            "source_episode_id": source_episode_id,
            "image_path": source_episode["image_path"],
            "image_sha256": source_episode["image_sha256"],
            "goal_text": str(annotation["goal_text"]),
            "candidates": candidates,
            "evaluator_episode": evaluator_episode,
            "annotation_provenance": {
                "authority": annotation_document["annotation_authority"],
                "reviewer_id": str(annotation["reviewer_id"]),
                "reviewed_at": str(annotation["reviewed_at"]),
                "evidence_note": str(annotation["evidence_note"]),
            },
            "development_quality_class": "SILVER_B_MAP_GEOMETRY",
        })
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "source_silver_b_report_sha256": silver_b_report.get("report_sha256"),
        "proposal_receipt_sha256": proposal_receipt.get("receipt_sha256"),
        "annotation_document_sha256": materializer.content_sha256(annotation_document),
        "episode_count": len(output_episodes),
        "unique_source_frame_count": len(source_frame_ids),
        "resolution_counts": resolution_counts,
        "episodes": output_episodes,
        "allowed_analysis": "CONDITIONED_CANDIDATE_SELECTION_AND_WEAK_GROUNDING_MECHANICS",
        "claim_ceiling": "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def aggregate_brain_cohorts(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    source_frames: set[str] = set()
    counts = {value: 0 for value in sorted(RESOLUTIONS)}
    part_receipts = []
    for part in parts:
        _require(part.get("policy_id") == POLICY_ID, "unexpected cohort part policy")
        _require(
            part.get("claim_ceiling") == "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
            "cohort part exceeds Silver-B claim ceiling",
        )
        for episode in part.get("episodes", []):
            episode_id = str(episode["episode_id"])
            _require(episode_id not in seen_episode_ids, "duplicate aggregate episode_id")
            seen_episode_ids.add(episode_id)
            source_frames.add(str(episode["evaluator_episode"]["observation_window"]["frame_ids"][0]))
            resolution = str(episode["evaluator_episode"]["goal_reference_resolution"])
            counts[resolution] += 1
            episodes.append(dict(episode))
        part_receipts.append({
            "report_sha256": str(part["report_sha256"]),
            "episode_count": int(part["episode_count"]),
            "unique_source_frame_count": int(part["unique_source_frame_count"]),
        })
    report = {
        "schema_version": 1,
        "policy_id": f"{POLICY_ID}-AGGREGATE-V1",
        "parts": part_receipts,
        "episode_count": len(episodes),
        "unique_source_frame_count": len(source_frames),
        "resolution_counts": counts,
        "episodes": episodes,
        "allowed_analysis": "CONDITIONED_CANDIDATE_SELECTION_AND_WEAK_GROUNDING_MECHANICS",
        "claim_ceiling": "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-b", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--proposal-receipt", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--part", action="append", default=[], type=Path, help="Aggregate an already built cohort part; repeat as needed.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    direct_paths = (args.silver_b, args.metadata, args.proposal_receipt, args.annotations)
    _require(bool(args.part) != any(direct_paths), "use either --part or all four direct cohort inputs")
    if args.part:
        inputs = [json.loads(path.read_text(encoding="utf-8")) for path in args.part]
        first = aggregate_brain_cohorts(inputs)
        second = aggregate_brain_cohorts(inputs)
    else:
        _require(all(direct_paths), "all four direct cohort inputs are required")
        inputs = [json.loads(path.read_text(encoding="utf-8")) for path in direct_paths]
        first = build_brain_cohort(*inputs)
        second = build_brain_cohort(*inputs)
    _require(materializer.canonical_bytes(first) == materializer.canonical_bytes(second), "deterministic replay mismatch")
    materializer.write_json(args.output, first)
    print(json.dumps({key: first[key] for key in ("episode_count", "resolution_counts", "claim_ceiling", "report_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
