"""Materialize a reviewed multi-run Silver-B Brain cohort.

The input manifest must enumerate every score-blind reviewed source frame.  It
may use an explicit AMBIGUOUS list for reviewed unresolved frames and explicit
overrides for UNIQUE/SET_VALUED or derived referring-expression episodes.
Nothing in this utility infers referent truth from proposal scores.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import silver_b_brain_cohort


class ReviewedCohortError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewedCohortError(message)


def _load(root: Path, value: str) -> dict[str, Any]:
    path = (root / value).resolve()
    _require(path.is_file(), f"missing manifest input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _annotation(
    decision: Mapping[str, Any],
    source_episode: Mapping[str, Any],
    review_episode: Mapping[str, Any],
    metadata_image: Mapping[str, Any],
    *,
    reviewer_id: str,
    reviewed_at: str,
) -> dict[str, Any]:
    resolution = str(decision["resolution"])
    targets = [dict(item) for item in decision.get("valid_targets", [])]
    visible = resolution != "AMBIGUOUS"
    if visible:
        _require(targets, "resolved decision requires valid_targets")
        width, height = int(metadata_image["width"]), int(metadata_image["height"])
        min_side = min(
            min((float(item["region_normalized_xyxy"][2]) - float(item["region_normalized_xyxy"][0])) * width,
                (float(item["region_normalized_xyxy"][3]) - float(item["region_normalized_xyxy"][1])) * height)
            for item in targets
        )
    else:
        _require(not targets, "AMBIGUOUS decision cannot carry valid_targets")
        min_side = None
    scene = decision.get("scene_condition") or {
        "target_size": "MEDIUM" if visible else "ABSENT",
        "visibility": "FULL" if visible else "ABSENT",
        "entrance_count": "MULTIPLE" if resolution == "SET_VALUED" else "SINGLE",
        "same_class_distractor": True,
        "illumination": "DAYLIGHT",
        "view_angle": "OBLIQUE",
    }
    target_name = str(review_episode["target_name"])
    return {
        "episode_id": str(decision["episode_id"]),
        "source_episode_id": str(source_episode["episode_id"]),
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at,
        "evidence_note": str(decision["evidence_note"]),
        "target_name": target_name,
        "goal_text": str(decision.get("goal_text") or f"Find the entrance to {target_name}."),
        "resolution": resolution,
        "valid_targets": targets,
        "distractors": [dict(item) for item in decision.get("distractors", [])],
        "target_min_side_px": round(min_side, 3) if min_side is not None else None,
        "visibility_fraction": float(decision.get("visibility_fraction", 1.0)) if visible else None,
        "text_support": str(decision.get("text_support", "NONE")) if visible else "NOT_APPLICABLE",
        "scene_condition": dict(scene),
    }


def build_from_manifest(manifest: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    _require(manifest.get("schema_version") == 1, "manifest schema_version must be 1")
    reviewer_id = str(manifest["reviewer_id"])
    reviewed_at = str(manifest["reviewed_at"])
    source_state: dict[str, dict[str, Any]] = {}
    frame_owner: dict[str, str] = {}
    all_frames: set[str] = set()

    for source in manifest.get("sources", []):
        source_id = str(source["source_id"])
        _require(source_id not in source_state, "duplicate source_id")
        silver = _load(root, str(source["silver_b"]))
        metadata = _load(root, str(source["metadata"]))
        receipt = _load(root, str(source["proposal_receipt"]))
        review = _load(root, str(source["review_index"]))
        metadata_by_frame = {str(item["id"]): item for item in metadata["images"]}
        reviews_by_frame = {str(item["frame_id"]): item for item in review["episodes"]}
        episodes_by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for episode in silver["episodes"]:
            episodes_by_frame[str(episode["frame_id"])].append(episode)
        for frame_id in episodes_by_frame:
            _require(frame_id in reviews_by_frame and frame_id in metadata_by_frame, "source frame lacks review or metadata")
            _require(frame_id not in frame_owner, "frame appears in multiple source groups")
            frame_owner[frame_id] = source_id
        all_frames.update(episodes_by_frame)
        source_state[source_id] = {
            "silver": silver,
            "metadata": metadata,
            "receipt": receipt,
            "reviews": reviews_by_frame,
            "metadata_by_frame": metadata_by_frame,
            "episodes_by_frame": episodes_by_frame,
        }

    replacements: dict[str, Mapping[str, Any]] = {}
    derived: list[Mapping[str, Any]] = []
    episode_ids: set[str] = set()
    for decision in manifest.get("resolved_or_derived_decisions", []):
        frame_id = str(decision["frame_id"])
        _require(frame_id in all_frames, "decision references unknown reviewed frame")
        episode_id = str(decision["episode_id"])
        _require(episode_id not in episode_ids, "duplicate decision episode_id")
        episode_ids.add(episode_id)
        if decision.get("replace_default", True):
            _require(frame_id not in replacements, "multiple replacement decisions for one frame")
            replacements[frame_id] = decision
        else:
            derived.append(decision)

    ambiguous_frames = {str(value) for value in manifest.get("reviewed_ambiguous_frame_ids", [])}
    _require(len(ambiguous_frames) == len(manifest.get("reviewed_ambiguous_frame_ids", [])), "duplicate ambiguous frame_id")
    _require(ambiguous_frames == all_frames - set(replacements), "ambiguous list must exactly cover every non-replaced reviewed frame")

    decisions_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_decisions: list[Mapping[str, Any]] = []
    for frame_id in sorted(all_frames):
        if frame_id in replacements:
            base_decisions.append(replacements[frame_id])
        else:
            base_decisions.append({
                "frame_id": frame_id,
                "episode_id": f"brain-dev--{frame_id}--ambiguous",
                "resolution": "AMBIGUOUS",
                "evidence_note": "Score-blind full-frame review did not establish one or more valid physical entrance referents at Silver-B authority.",
            })
    for decision in [*base_decisions, *derived]:
        frame_id = str(decision["frame_id"])
        source_id = frame_owner[frame_id]
        state = source_state[source_id]
        candidates = sorted(state["episodes_by_frame"][frame_id], key=lambda item: str(item["episode_id"]))
        requested_source = decision.get("source_episode_id")
        if requested_source:
            candidates = [item for item in candidates if str(item["episode_id"]) == str(requested_source)]
            _require(len(candidates) == 1, "requested source_episode_id does not match reviewed frame")
        source_episode = candidates[0]
        decisions_by_source[source_id].append(_annotation(
            decision,
            source_episode,
            state["reviews"][frame_id],
            state["metadata_by_frame"][frame_id],
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for source_id in sorted(source_state):
        state = source_state[source_id]
        annotations = {
            "schema_version": 1,
            "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
            "source_id": source_id,
            "episodes": decisions_by_source[source_id],
        }
        annotation_path = output_dir / f"{source_id}-annotations.json"
        materializer.write_json(annotation_path, annotations)
        part = silver_b_brain_cohort.build_brain_cohort(
            state["silver"], state["metadata"], state["receipt"], annotations,
        )
        materializer.write_json(output_dir / f"{source_id}-brain-cohort.json", part)
        parts.append(part)
    aggregate = silver_b_brain_cohort.aggregate_brain_cohorts(parts)
    aggregate["review_manifest_sha256"] = materializer.content_sha256(manifest)
    aggregate["report_sha256"] = materializer.content_sha256({key: value for key, value in aggregate.items() if key != "report_sha256"})
    materializer.write_json(output_dir / "brain-cohort.json", aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    first = build_from_manifest(manifest, Path.cwd(), args.output_dir)
    second = build_from_manifest(manifest, Path.cwd(), args.output_dir)
    _require(materializer.canonical_bytes(first) == materializer.canonical_bytes(second), "deterministic replay mismatch")
    print(json.dumps({key: first[key] for key in ("episode_count", "unique_source_frame_count", "resolution_counts", "claim_ceiling", "report_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
