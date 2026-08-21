"""Export map+geometry candidates for bounded P0 Development use.

The exporter does not modify the frozen P0-S0 materializer or P0-S1 identity
gate.  A parent A or B admission may be reused at the lower Silver-B authority,
but never gains exact-entrance truth authority here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


POLICY_ID = "P0-SILVER-B-DEVELOPMENT-ADDENDUM-V1"
ELIGIBLE_PARENT_CLASSES = {"SILVER_A_PRIMARY", "SILVER_B_MAP_GEOMETRY"}
ALLOWED_USES = [
    "P0_PIPELINE_DEVELOPMENT_AND_DEBUGGING",
    "MAP_GEOMETRY_CONDITIONED_CANDIDATE_YIELD",
    "WEAK_CANDIDATE_RANKING_PROTOTYPING",
    "FAILURE_ANALYSIS_AND_ABSTENTION_DESIGN",
]
FORBIDDEN_CLAIMS = [
    "EXACT_SAME_PHYSICAL_ENTRANCE_IDENTITY",
    "GROUNDING_DINO_RECALL_OR_PRECISION",
    "EXACT_BRAIN_SELECTION_ACCURACY",
    "END_TO_END_GROUNDING_ACCURACY",
    "SILVER_A_OR_HUMAN_TRUTH_EQUIVALENCE",
]


class SilverBDevelopmentError(ValueError):
    pass


def _metadata_by_id(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    images = document.get("images")
    if not isinstance(images, list):
        raise SilverBDevelopmentError("metadata images missing")
    result = {}
    for image in images:
        if not isinstance(image, Mapping) or not image.get("id"):
            raise SilverBDevelopmentError("metadata image identity missing")
        result[str(image["id"])] = image
    return result


def _normalized_bbox(candidate: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[float]:
    bbox = candidate.get("bbox_xyxy")
    width, height = metadata.get("width"), metadata.get("height")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise SilverBDevelopmentError("bbox or image dimensions missing")
    normalized = [float(bbox[0]) / width, float(bbox[1]) / height, float(bbox[2]) / width, float(bbox[3]) / height]
    if not (0.0 <= normalized[0] < normalized[2] <= 1.0 and 0.0 <= normalized[1] < normalized[3] <= 1.0):
        raise SilverBDevelopmentError("normalized bbox outside image")
    return [round(value, 12) for value in normalized]


def export_development_cohort(
    bundle: Mapping[str, Any],
    metadata_document: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    *,
    data_role: str,
) -> dict[str, Any]:
    if data_role not in {"CONSUMED_DEVELOPMENT", "DEVELOPMENT"}:
        raise SilverBDevelopmentError("data_role must be DEVELOPMENT or CONSUMED_DEVELOPMENT")
    if parent_result.get("input_sha256") != materializer.content_sha256(bundle):
        raise SilverBDevelopmentError("parent materialization input hash mismatch")
    metadata = _metadata_by_id(metadata_document)
    records = {str(item.get("record_id")): item for item in bundle.get("records", []) if isinstance(item, Mapping)}
    episodes = []
    parent_class_counts: dict[str, int] = {}
    for result in parent_result.get("results", []):
        parent_class = str(result.get("quality_class"))
        parent_class_counts[parent_class] = parent_class_counts.get(parent_class, 0) + 1
        if parent_class not in ELIGIBLE_PARENT_CLASSES:
            continue
        record_id = str(result.get("record_id"))
        record = records.get(record_id)
        if record is None:
            raise SilverBDevelopmentError("parent result record missing from bundle")
        replayed = materializer.admit_record(record)
        if replayed.get("quality_class") != parent_class:
            raise SilverBDevelopmentError("parent admission does not replay")
        positive_ids = set(result.get("positive_candidate_ids", []))
        candidates_by_frame: dict[str, list[Mapping[str, Any]]] = {}
        for candidate in record.get("candidates", []):
            if candidate.get("candidate_id") in positive_ids:
                candidates_by_frame.setdefault(str(candidate["frame_id"]), []).append(candidate)
        for frame_id, candidates in sorted(candidates_by_frame.items()):
            image = metadata.get(frame_id)
            if image is None:
                raise SilverBDevelopmentError("admitted frame metadata missing")
            weak_candidates = []
            for candidate in sorted(candidates, key=lambda value: (int(value["proposal_rank"]), str(value["candidate_id"]))):
                weak_candidates.append({
                    "candidate_id": candidate["candidate_id"],
                    "bbox_normalized_xyxy": _normalized_bbox(candidate, image),
                    "proposal_label": candidate["proposal_label"],
                    "proposal_score": candidate["proposal_score"],
                    "proposal_score_semantics": candidate["proposal_score_semantics"],
                    "building_id": candidate["building_id"],
                    "anchor_id": candidate["anchor_id"],
                    "predicted_entrance_geo": candidate["predicted_entrance_geo"],
                    "candidate_anchor_distance_m": candidate["candidate_anchor_distance_m"],
                    "generator_provenance": candidate["generator_provenance"],
                    "label_authority": "TARGET_BUILDING_WALL_ENTRANCE_CANDIDATE_ONLY",
                    "same_physical_entrance_identity": "NOT_ESTABLISHED",
                })
            episodes.append({
                "episode_id": f"silver-b-dev--{record_id}--{frame_id}",
                "record_id": record_id,
                "frame_id": frame_id,
                "image_path": image["path"],
                "image_sha256": image["image_sha256"],
                "captured_at": image["captured_at"],
                "sequence_id": image["sequence_id"],
                "target_building_id": record["crosswalk"]["building_id"],
                "target_anchor_id": record["anchor"]["anchor_id"],
                "goal_reference_truth": {
                    "resolution": "AMBIGUOUS",
                    "valid_target_instance_ids": [],
                    "reason": "SILVER_B_DOES_NOT_ESTABLISH_EXACT_PHYSICAL_ENTRANCE_REFERENTS",
                },
                "weak_positive_candidates": weak_candidates,
                "parent_quality_class": parent_class,
                "development_quality_class": "SILVER_B_MAP_GEOMETRY",
                "data_role": data_role,
            })
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "data_role": data_role,
        "parent_protocol_id": parent_result.get("protocol_id"),
        "parent_materialization_report_sha256": parent_result.get("report_sha256"),
        "parent_class_counts": parent_class_counts,
        "episode_count": len(episodes),
        "candidate_count": sum(len(item["weak_positive_candidates"]) for item in episodes),
        "episodes": episodes,
        "allowed_uses": ALLOWED_USES,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "claim_ceiling": "P0_DEVELOPMENT_MAP_GEOMETRY_CANDIDATES_ONLY_NO_EXACT_ENTRANCE_TRUTH",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--materialization-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-role", choices=("DEVELOPMENT", "CONSUMED_DEVELOPMENT"), required=True)
    args = parser.parse_args(argv)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    parent = json.loads(args.materialization_result.read_text(encoding="utf-8"))
    first = export_development_cohort(bundle, metadata, parent, data_role=args.data_role)
    second = export_development_cohort(bundle, metadata, parent, data_role=args.data_role)
    if materializer.canonical_bytes(first) != materializer.canonical_bytes(second):
        raise SilverBDevelopmentError("deterministic replay mismatch")
    materializer.write_json(args.output, first)
    print(json.dumps({
        "episode_count": first["episode_count"],
        "candidate_count": first["candidate_count"],
        "claim_ceiling": first["claim_ceiling"],
        "report_sha256": first["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
