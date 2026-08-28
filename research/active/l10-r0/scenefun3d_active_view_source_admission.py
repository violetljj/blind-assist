from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scenefun3d_active_view_ordinal import (
    load_camera_trajectory,
    parse_directional_ordinal,
    select_active_ordinal_view,
)
from scenefun3d_backend_proposals import MeasuredProposalBuilder
from scenefun3d_conflict_source_admission import _download_once, _download_scene, _folds, _rows
from scenefun3d_functional_handoff_ceiling import _load_json
from scenefun3d_functional_set_integrity import _sha256, _source_paths
from scenefun3d_ordinal_axis_integrity import fit_unoriented_ordinal_axis


def _directional_tasks(descriptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in descriptions:
        directional = parse_directional_ordinal(row["description"])
        text = row["description"].casefold()
        if directional is None or "unplug" in text or ("plug" not in text and "connect" not in text):
            continue
        rows.append(
            {
                "desc_id": row["desc_id"],
                "description": row["description"],
                "ordinal": directional.ordinal,
                "direction": directional.direction,
            }
        )
    return rows


def admit_sources(
    protocol: dict[str, Any],
    cohort_csv: Path,
    metadata_csv: Path,
    data_root: Path,
    backend_receipt: Path,
) -> dict[str, Any]:
    if _sha256(cohort_csv) != protocol["source"]["cohort_csv_sha256"]:
        raise ValueError("COHORT_CSV_HASH_MISMATCH")
    if _sha256(metadata_csv) != protocol["source"]["arkitscenes_metadata_sha256"]:
        raise ValueError("METADATA_CSV_HASH_MISMATCH")
    selection = protocol["selection"]
    algorithm = protocol["frozen_algorithm"]
    folds = _folds(metadata_csv)
    consumed = set(protocol["consumed_or_target_exposed_visit_ids"])
    scanned: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    failures = 0
    candidates_seen = 0
    builder: MeasuredProposalBuilder | None = None
    backend_record: dict[str, Any] | None = None
    for visit_id, video_id in _rows(cohort_csv):
        if visit_id in consumed:
            continue
        if candidates_seen >= int(selection["maximum_candidate_scenes"]):
            break
        candidates_seen += 1
        try:
            paths = _source_paths(data_root, visit_id, video_id)
            scene_base = f"{protocol['source']['scenefun3d_base_url'].rstrip('/')}/train/{visit_id}"
            _download_once(f"{scene_base}/{visit_id}_descriptions.json", paths["descriptions"])
            descriptions = _load_json(paths["descriptions"])["descriptions"]
            tasks = _directional_tasks(descriptions)
            if len(tasks) < int(selection["minimum_horizontal_directional_ordinal_connect_tasks"]):
                scanned.append({"visit_id": visit_id, "video_id": video_id, "directional_ordinal_tasks": len(tasks), "eligible": False, "reason": "DIRECTIONAL_TEXT_PREFILTER_NOT_MET"})
                continue
            fold = folds[video_id]
            paths = _download_scene(
                data_root, visit_id, video_id, fold,
                protocol["source"]["scenefun3d_base_url"], protocol["source"]["arkitscenes_base_url"],
            )
            pose_name = str(protocol["source"].get("camera_pose_asset", "hires_poses.traj"))
            pose_path = paths["transform"].parent / pose_name
            pose_url = f"{protocol['source']['arkitscenes_base_url'].rstrip('/')}/raw/{fold}/{video_id}/{pose_name}"
            _download_once(pose_url, pose_path)
            poses = load_camera_trajectory(pose_path)
            if builder is None:
                builder = MeasuredProposalBuilder(paths, backend_receipt)
                backend_record = builder.select()
            proposals, unmatched = builder.build(paths)
            labels = {
                row["annot_id"]: row["label"]
                for row in _load_json(paths["annotations"])["annotations"]
                if row["label"] != "exclude"
            }
            by_parent: dict[str, dict[str, Any]] = defaultdict(dict)
            for candidate_id, proposal in proposals.items():
                by_parent[proposal.parent.binding_id][candidate_id] = proposal
            active_parents = []
            for parent_id, parent_candidates in sorted(by_parent.items()):
                same_action = {
                    candidate_id: proposal
                    for candidate_id, proposal in parent_candidates.items()
                    if labels[candidate_id] == algorithm["candidate_action_label"]
                }
                if not same_action:
                    continue
                parent = next(iter(parent_candidates.values())).parent
                axis = fit_unoriented_ordinal_axis(parent, same_action, algorithm)
                if axis is None:
                    continue
                if not any(task["ordinal"] <= len(same_action) for task in tasks):
                    continue
                view = select_active_ordinal_view(same_action, poses, algorithm)
                if view is None:
                    continue
                active_parents.append(
                    {
                        "parent_binding_id": parent_id,
                        "parent_label": parent.label,
                        "parent_center": parent.center.tolist(),
                        "parent_lengths": parent.lengths.tolist(),
                        "parent_axes": parent.axes.tolist(),
                        "axis_diagnostics": {
                            "pitch": axis.pitch,
                            "maximum_pitch_relative_deviation": axis.maximum_pitch_relative_deviation,
                            "maximum_orthogonal_residual_pitch_ratio": axis.maximum_orthogonal_residual_pitch_ratio,
                        },
                        "active_view": {
                            "timestamp": view.timestamp,
                            "camera_to_world": view.camera_to_world.tolist(),
                            "ordered_candidate_ids_left_to_right": list(view.ordered_candidate_ids_left_to_right),
                            "normalized_image_xy": view.normalized_image_xy,
                            "horizontal_span": view.horizontal_span,
                            "vertical_span": view.vertical_span,
                            "horizontal_to_vertical_ratio": view.horizontal_to_vertical_ratio,
                            "median_depth_m": view.median_depth_m,
                        },
                        "candidates": {
                            candidate_id: {
                                "label": labels[candidate_id],
                                "center": proposal.center.tolist(),
                                "parent_coverage": proposal.parent_coverage,
                            }
                            for candidate_id, proposal in sorted(parent_candidates.items())
                        },
                    }
                )
            eligible = bool(active_parents)
            row = {
                "visit_id": visit_id,
                "video_id": video_id,
                "directional_ordinal_tasks": len(tasks),
                "directional_task_text": tasks,
                "trajectory_pose_count": len(poses),
                "functional_annotations_parent_bound": len(proposals),
                "functional_annotations_parent_unmatched": unmatched,
                "active_view_parent_count": len(active_parents),
                "active_view_parents": active_parents,
                "eligible": eligible,
            }
            scanned.append(row)
            if eligible:
                row["source_sha256"] = {
                    **{name: _sha256(path) for name, path in paths.items()},
                    "camera_poses": _sha256(pose_path),
                }
                selected.append(row)
                if len(selected) == int(selection["selected_scene_count"]):
                    break
        except Exception as error:
            failures += 1
            scanned.append({"visit_id": visit_id, "video_id": video_id, "eligible": None, "reason": "SOURCE_UNAVAILABLE_OR_INVALID", "error_type": type(error).__name__, "error": str(error)})
    labels = protocol["decision_labels"]
    decision = labels["pass"] if len(selected) == int(selection["selected_scene_count"]) else (labels["incomplete"] if failures else labels["insufficient"])
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": protocol["protocol_sha256"],
        "execution_backend": backend_record,
        "denominators": {"candidate_scenes_scanned": len(scanned), "source_failures": failures, "eligible_scenes": len(selected), "required_scenes": int(selection["selected_scene_count"])},
        "selected": selected,
        "scanned": scanned,
        "authority_boundary": "Admission reads public directional task text, privileged functional labels, proposal geometry, parent OBBs, and real camera poses. It never reads description annot_id, target membership, selector output, or evaluator scores.",
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    result = admit_sources(protocol, args.cohort_csv.resolve(), args.metadata_csv.resolve(), args.data_root.resolve(), args.backend_receipt.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "denominators": result["denominators"], "backend": None if result["execution_backend"] is None else {key: result["execution_backend"][key] for key in ("selected_backend", "selected_device_type", "selection_reason")}, "selected": [{"visit_id": row["visit_id"], "directional_tasks": row["directional_ordinal_tasks"], "active_view_parents": row["active_view_parent_count"], "poses": row["trajectory_pose_count"]} for row in result["selected"]]}, indent=2))


if __name__ == "__main__":
    main()
