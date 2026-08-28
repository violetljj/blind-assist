from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from scenefun3d_functional_handoff_ceiling import _load_json
from scenefun3d_functional_set_integrity import (
    _build_proposals,
    _sha256,
    _source_paths,
)


def _download_once(url: str, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _rows(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = [
            (str(row["visit_id"]), str(row["video_id"]))
            for row in csv.DictReader(stream)
        ]
    return sorted(set(values))


def _folds(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            str(row["video_id"]): str(row["fold"])
            for row in csv.DictReader(stream)
        }


def _download_scene(
    data_root: Path,
    visit_id: str,
    video_id: str,
    fold: str,
    scenefun3d_base_url: str,
    arkitscenes_base_url: str,
) -> dict[str, Path]:
    paths = _source_paths(data_root, visit_id, video_id)
    scene_base = f"{scenefun3d_base_url.rstrip('/')}/train/{visit_id}"
    video_base = f"{scene_base}/{video_id}"
    arkit_base = f"{arkitscenes_base_url.rstrip('/')}/raw/{fold}/{video_id}"
    assets = {
        paths["descriptions"]: f"{scene_base}/{visit_id}_descriptions.json",
        paths["annotations"]: f"{scene_base}/{visit_id}_annotations.json",
        paths["laser_scan"]: f"{scene_base}/{visit_id}_laser_scan.ply",
        paths["transform"]: f"{video_base}/{video_id}_refined_transform.npy",
        paths["object_boxes"]: f"{arkit_base}/{video_id}_3dod_annotation.json",
    }
    for output, url in assets.items():
        _download_once(url, output)
    return paths


def _label_to_family(protocol: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family, labels in protocol["action_families"].items():
        for label in labels:
            if label in mapping:
                raise ValueError(f"Action label belongs to two families: {label}")
            mapping[label] = family
    return mapping


def _conflict_summary(
    paths: dict[str, Path], protocol: dict[str, Any]
) -> dict[str, Any]:
    proposals, unmatched = _build_proposals(paths)
    labels = {
        row["annot_id"]: row["label"]
        for row in _load_json(paths["annotations"])["annotations"]
        if row["label"] != "exclude"
    }
    family_for_label = _label_to_family(protocol)
    by_parent: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ignored_labels: set[str] = set()
    for candidate_id, proposal in proposals.items():
        label = labels[candidate_id]
        family = family_for_label.get(label)
        if family is None:
            ignored_labels.add(label)
            continue
        by_parent[proposal.parent.binding_id][family].append(candidate_id)

    cross_family: list[dict[str, Any]] = []
    redundant: list[dict[str, Any]] = []
    for parent_id in sorted(by_parent):
        family_counts = {
            family: len(candidate_ids)
            for family, candidate_ids in sorted(by_parent[parent_id].items())
        }
        if len(family_counts) < 2:
            continue
        row = {
            "parent_binding_id": parent_id,
            "family_counts": family_counts,
        }
        cross_family.append(row)
        if max(family_counts.values()) >= 2:
            redundant.append(row)
    return {
        "functional_annotations_parent_bound": len(proposals),
        "functional_annotations_parent_unmatched": unmatched,
        "actionable_parent_count": len(by_parent),
        "cross_family_parent_count": len(cross_family),
        "redundancy_eligible_conflict_parent_count": len(redundant),
        "conflict_parents": cross_family,
        "ignored_function_labels": sorted(ignored_labels),
    }


def admit_sources(
    protocol: dict[str, Any],
    cohort_csv: Path,
    metadata_csv: Path,
    data_root: Path,
) -> dict[str, Any]:
    expected_cohort_hash = protocol["source"]["cohort_csv_sha256"]
    expected_metadata_hash = protocol["source"]["arkitscenes_metadata_sha256"]
    cohort_hash = _sha256(cohort_csv)
    metadata_hash = _sha256(metadata_csv)
    if cohort_hash != expected_cohort_hash:
        raise ValueError("Cohort CSV hash mismatch")
    if metadata_hash != expected_metadata_hash:
        raise ValueError("ARKitScenes metadata hash mismatch")

    selection = protocol["selection"]
    consumed = set(protocol["consumed_visit_ids"])
    folds = _folds(metadata_csv)
    maximum_candidates = int(selection["maximum_candidate_scenes"])
    minimum_descriptions = int(selection["minimum_descriptions_per_scene"])
    minimum_conflicts = int(selection["minimum_cross_family_parents"])
    minimum_redundant = int(
        selection["minimum_redundancy_eligible_conflict_parents"]
    )
    required_scenes = int(selection["selected_scene_count"])
    scanned: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    failures = 0
    candidate_count = 0

    for visit_id, video_id in _rows(cohort_csv):
        if visit_id in consumed:
            continue
        if candidate_count >= maximum_candidates:
            break
        candidate_count += 1
        try:
            paths = _source_paths(data_root, visit_id, video_id)
            scene_base = (
                f"{protocol['source']['scenefun3d_base_url'].rstrip('/')}/train/{visit_id}"
            )
            _download_once(
                f"{scene_base}/{visit_id}_descriptions.json",
                paths["descriptions"],
            )
            descriptions_payload = _load_json(paths["descriptions"])
            if str(descriptions_payload.get("visit_id")) != visit_id:
                raise ValueError("visit_id mismatch in descriptions")
            description_count = len(descriptions_payload.get("descriptions", []))
            if description_count < minimum_descriptions:
                scanned.append(
                    {
                        "visit_id": visit_id,
                        "video_id": video_id,
                        "descriptions": description_count,
                        "eligible": False,
                        "reason": "DESCRIPTION_PREFILTER_NOT_MET",
                    }
                )
                continue
            fold = folds[video_id]
            paths = _download_scene(
                data_root,
                visit_id,
                video_id,
                fold,
                protocol["source"]["scenefun3d_base_url"],
                protocol["source"]["arkitscenes_base_url"],
            )
            summary = _conflict_summary(paths, protocol)
            eligible = (
                description_count >= minimum_descriptions
                and summary["cross_family_parent_count"] >= minimum_conflicts
                and summary["redundancy_eligible_conflict_parent_count"]
                >= minimum_redundant
            )
            row = {
                "visit_id": visit_id,
                "video_id": video_id,
                "descriptions": description_count,
                "eligible": eligible,
                **summary,
            }
            scanned.append(row)
            if eligible:
                row["source_sha256"] = {
                    name: _sha256(path) for name, path in paths.items()
                }
                selected.append(row)
                if len(selected) == required_scenes:
                    break
        except Exception as error:  # availability is not algorithm evidence
            failures += 1
            scanned.append(
                {
                    "visit_id": visit_id,
                    "video_id": video_id,
                    "eligible": None,
                    "reason": "SOURCE_UNAVAILABLE_OR_INVALID",
                    "error_type": type(error).__name__,
                }
            )

    labels = protocol["decision_labels"]
    if len(selected) == required_scenes:
        decision = labels["pass"]
    elif failures:
        decision = labels["incomplete"]
    else:
        decision = labels["insufficient"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort_csv_sha256": cohort_hash,
        "arkitscenes_metadata_sha256": metadata_hash,
        "selection": selection,
        "action_families": protocol["action_families"],
        "denominators": {
            "candidate_scenes_scanned": len(scanned),
            "source_failures": failures,
            "eligible_scenes": len(selected),
            "required_scenes": required_scenes,
        },
        "selected": selected,
        "scanned": scanned,
        "authority_boundary": (
            "Admission uses proposal labels, proposal-to-parent geometry, parent OBBs, "
            "and description count. It never reads description annot_id values, task target "
            "membership, selector output, or evaluator scores."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    result = admit_sources(
        protocol,
        args.cohort_csv.resolve(),
        args.metadata_csv.resolve(),
        args.data_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "selected": [
                    {
                        "visit_id": row["visit_id"],
                        "video_id": row["video_id"],
                        "descriptions": row["descriptions"],
                        "cross_family_parents": row["cross_family_parent_count"],
                        "redundancy_eligible_conflict_parents": row[
                            "redundancy_eligible_conflict_parent_count"
                        ],
                    }
                    for row in result["selected"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
