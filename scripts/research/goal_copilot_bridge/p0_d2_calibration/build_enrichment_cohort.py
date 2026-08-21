"""Build a score-blind P0-D2 enrichment cohort from compact manual decisions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer, silver_b_brain_cohort


class EnrichmentError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnrichmentError(message)


def _place_names(source_report: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, list[str]] = {}
    for row in source_report.get("place_building_crosswalk_candidates", []):
        if row.get("status") == "CANDIDATE_ONLY" and len(row.get("building_ids", [])) == 1:
            values.setdefault(str(row["building_ids"][0]), []).append(str(row["place_name"]).strip())
    return {building_id: names[0] for building_id, names in values.items() if len(names) == 1}


def build(manifest: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    source_report = json.loads((root / str(manifest["source_report"])).read_text(encoding="utf-8"))
    names = _place_names(source_report)
    excluded_names = {str(value) for value in manifest.get("excluded_target_names", [])}
    excluded_frames = {str(value) for value in manifest.get("excluded_frame_ids", [])}
    resolved = {str(item["source_episode_id"]): item for item in manifest.get("resolved_decisions", [])}
    _require(len(resolved) == len(manifest.get("resolved_decisions", [])), "duplicate resolved source episode")
    parts = []
    excluded_source_episode_ids = []
    included_frames: set[str] = set()
    included_names: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in manifest.get("sources", []):
        source_id = str(source["source_id"])
        silver = json.loads((root / str(source["silver_b"])).read_text(encoding="utf-8"))
        metadata = json.loads((root / str(source["metadata"])).read_text(encoding="utf-8"))
        receipt = json.loads((root / str(source["proposal_receipt"])).read_text(encoding="utf-8"))
        metadata_by_frame = {str(item["id"]): item for item in metadata["images"]}
        annotations = []
        for source_episode in silver.get("episodes", []):
            source_episode_id = str(source_episode["episode_id"])
            target_name = names.get(str(source_episode["target_building_id"]))
            _require(target_name is not None, "source episode lacks unique target name")
            if target_name in excluded_names:
                _require(source_episode_id not in resolved, "excluded target cannot have a resolved decision")
                excluded_source_episode_ids.append(source_episode_id)
                continue
            frame_id = str(source_episode["frame_id"])
            if frame_id in excluded_frames:
                _require(source_episode_id not in resolved, "excluded frame cannot have a resolved decision")
                excluded_source_episode_ids.append(source_episode_id)
                continue
            decision = resolved.get(source_episode_id)
            resolution = "UNIQUE" if decision else "AMBIGUOUS"
            targets = [dict(item) for item in decision.get("valid_targets", [])] if decision else []
            if resolution == "UNIQUE":
                _require(len(targets) == 1, "resolved enrichment decision must carry one target")
                metadata_image = metadata_by_frame[frame_id]
                region = [float(value) for value in targets[0]["region_normalized_xyxy"]]
                min_side = min((region[2] - region[0]) * int(metadata_image["width"]), (region[3] - region[1]) * int(metadata_image["height"]))
            else:
                min_side = None
            annotations.append({
                "episode_id": f"p0-d2-enrichment--{frame_id}--{target_name.lower().replace(' ', '-')}",
                "source_episode_id": source_episode_id,
                "reviewer_id": str(manifest["reviewer_id"]),
                "reviewed_at": str(manifest["reviewed_at"]),
                "evidence_note": str(decision["evidence_note"]) if decision else "Score-blind full-frame review did not establish a unique or set-valued physical entrance referent.",
                "target_name": target_name,
                "goal_text": f"Find the entrance to {target_name}.",
                "resolution": resolution,
                "valid_targets": targets,
                "distractors": [],
                "target_min_side_px": round(min_side, 3) if min_side is not None else None,
                "visibility_fraction": float(decision.get("visibility_fraction", 1.0)) if decision else None,
                "text_support": str(decision.get("text_support", "READABLE")) if decision else "NOT_APPLICABLE",
                "scene_condition": {
                    "target_size": "MEDIUM" if decision else "ABSENT",
                    "visibility": "PARTIAL" if decision and float(decision.get("visibility_fraction", 1.0)) < 1.0 else ("FULL" if decision else "ABSENT"),
                    "entrance_count": "SINGLE",
                    "same_class_distractor": True,
                    "illumination": "DAYLIGHT",
                    "view_angle": "OBLIQUE",
                },
            })
            included_frames.add(frame_id)
            included_names.add(target_name)
        annotation_document = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "source_id": source_id,
            "episodes": annotations,
        }
        materializer.write_json(output_dir / f"{source_id}-annotations.json", annotation_document)
        part = silver_b_brain_cohort.build_brain_cohort(silver, metadata, receipt, annotation_document)
        materializer.write_json(output_dir / f"{source_id}-brain-cohort.json", part)
        parts.append(part)
    _require(set(resolved) <= {str(item["source_episode_id"]) for part in parts for item in part["episodes"]}, "resolved decision did not enter cohort")
    aggregate = silver_b_brain_cohort.aggregate_brain_cohorts(parts)
    old_names: set[str] = set()
    old_frames: set[str] = set()
    for value in manifest.get("overlap_reference_cohorts", []):
        cohort = json.loads((root / str(value)).read_text(encoding="utf-8"))
        for episode in cohort.get("episodes", []):
            old_names.add(str(episode["evaluator_episode"]["goal_spec"]["target_name"]))
            old_frames.update(str(frame) for frame in episode["evaluator_episode"]["observation_window"]["frame_ids"])
    name_overlap = sorted(included_names & old_names)
    frame_overlap = sorted(included_frames & old_frames)
    _require(not name_overlap and not frame_overlap, "enrichment cohort is not parent/frame disjoint after exclusions")
    materializer.write_json(output_dir / "brain-cohort.json", aggregate)
    audit = {
        "schema_version": 1,
        "status": "P0_D2_RESOLVABLE_ENRICHMENT_REVIEW_COMPLETE",
        "cohort_report_sha256": aggregate["report_sha256"],
        "excluded_target_names": sorted(excluded_names),
        "excluded_frame_ids": sorted(excluded_frames),
        "excluded_source_episode_ids": sorted(excluded_source_episode_ids),
        "target_name_overlap": name_overlap,
        "frame_overlap": frame_overlap,
        "venue_parent_count": len(included_names),
        "resolution_counts": aggregate["resolution_counts"],
        "claim_ceiling": "SCORE_BLIND_SILVER_B_DEVELOPMENT_ENRICHMENT_ONLY_NO_CALIBRATOR_OR_MODEL_PERFORMANCE_CLAIM",
    }
    audit["report_sha256"] = materializer.content_sha256(audit)
    materializer.write_json(output_dir / "enrichment-audit.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build(manifest, Path.cwd(), args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
