from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scenefun3d_active_view_ordinal import (
    load_camera_trajectory,
    parse_directional_ordinal,
    select_active_ordinal_view,
    select_consensus_active_ordinal_view,
    select_depth_visible_consensus_active_ordinal_view,
)
from scenefun3d_active_view_source_admission import _directional_tasks
from scenefun3d_backend_proposals import MeasuredFunctionalCenterBuilder
from scenefun3d_conflict_source_admission import _download_once, _download_scene, _folds, _rows
from scenefun3d_contextual_lattice_ordinal import connected_components, fit_metric_lattice
from scenefun3d_functional_handoff_ceiling import _load_json, _load_parent_boxes
from scenefun3d_functional_set_integrity import _sha256, _source_paths


def _distance_to_box(point: np.ndarray, box: Any) -> float:
    local = (point - box.center) @ box.axes.T
    outside = np.maximum(np.abs(local) - box.lengths / 2.0, 0.0)
    return float(np.linalg.norm(outside))


def _extract_archive_once(archive_path: Path, output_directory: Path, suffix: str) -> None:
    complete = output_directory / ".complete"
    if complete.is_file():
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [row for row in archive.infolist() if not row.is_dir() and row.filename.casefold().endswith(suffix)]
        if not members:
            raise ValueError(f"ARCHIVE_HAS_NO_{suffix.upper()}_FILES")
        for member in members:
            destination = output_directory / Path(member.filename).name
            if destination.is_file():
                continue
            temporary = destination.with_suffix(destination.suffix + ".extracting")
            with archive.open(member) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(destination)
    complete.write_text(f"{len(members)}\n", encoding="utf-8")


def _directional_ordinal_inventories(
    tasks: list[dict[str, Any]], component_size: int
) -> dict[str, list[int]]:
    inventories: dict[str, list[int]] = {}
    for direction in ("FROM_LEFT", "FROM_RIGHT"):
        values = sorted({int(row["ordinal"]) for row in tasks if row["direction"] == direction})
        if len(values) == component_size and values == list(range(values[0], values[-1] + 1)):
            inventories[direction] = values
    return inventories


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
    builder: MeasuredFunctionalCenterBuilder | None = None
    backend_record: dict[str, Any] | None = None
    for visit_id, video_id in _rows(cohort_csv):
        start_after_visit_id = selection.get("start_after_visit_id")
        if start_after_visit_id is not None and visit_id <= str(start_after_visit_id):
            continue
        if visit_id in consumed:
            continue
        if candidates_seen >= int(selection["maximum_candidate_scenes"]):
            break
        candidates_seen += 1
        try:
            paths = _source_paths(data_root, visit_id, video_id)
            scene_base = f"{protocol['source']['scenefun3d_base_url'].rstrip('/')}/train/{visit_id}"
            _download_once(f"{scene_base}/{visit_id}_descriptions.json", paths["descriptions"])
            tasks = _directional_tasks(_load_json(paths["descriptions"])["descriptions"])
            if algorithm.get("carrier_mode", "CONTEXT_ANCHOR_OBB") != "SELF_CARRIER_LOCAL_ACTION_LATTICE":
                tasks = [row for row in tasks if "door" in row["description"].casefold()]
            if len(tasks) < int(selection["minimum_horizontal_directional_ordinal_connect_tasks"]):
                scanned.append({"visit_id": visit_id, "video_id": video_id, "directional_context_tasks": len(tasks), "eligible": False, "reason": "DIRECTIONAL_CONTEXT_TEXT_PREFILTER_NOT_MET"})
                continue
            fold = folds[video_id]
            paths = _download_scene(data_root, visit_id, video_id, fold, protocol["source"]["scenefun3d_base_url"], protocol["source"]["arkitscenes_base_url"])
            pose_name = protocol["source"]["camera_pose_asset"]
            pose_path = paths["transform"].parent / pose_name
            _download_once(f"{protocol['source']['arkitscenes_base_url'].rstrip('/')}/raw/{fold}/{video_id}/{pose_name}", pose_path)
            poses = load_camera_trajectory(pose_path)
            if builder is None:
                builder = MeasuredFunctionalCenterBuilder(paths, backend_receipt)
                backend_record = builder.select()
            centers = builder.build(paths)
            action_centers = {
                candidate_id: row.center
                for candidate_id, row in centers.items()
                if row.label == algorithm["candidate_action_label"]
            }
            carrier_mode = algorithm.get("carrier_mode", "CONTEXT_ANCHOR_OBB")
            anchors = [box for box in _load_parent_boxes(paths["object_boxes"]) if "door" in box.label.casefold()]
            contextual_lattices = []
            view_diagnostics: list[dict[str, Any]] = []
            carrier_rows = (
                [(None, action_centers)]
                if carrier_mode == "SELF_CARRIER_LOCAL_ACTION_LATTICE"
                else [
                    (
                        anchor,
                        {
                            candidate_id: center
                            for candidate_id, center in action_centers.items()
                            if _distance_to_box(center, anchor)
                            <= float(algorithm["maximum_candidate_to_anchor_obb_distance_m"])
                        },
                    )
                    for anchor in sorted(anchors, key=lambda row: row.binding_id)
                ]
            )
            for anchor, near in carrier_rows:
                for component in connected_components(near, float(algorithm["local_lattice_link_radius_m"])):
                    component_centers = {candidate_id: near[candidate_id] for candidate_id in component}
                    lattice = fit_metric_lattice(component_centers, algorithm)
                    if lattice is None:
                        continue
                    depth_visible_mode = algorithm.get("view_selection_mode") == "DEPTH_VISIBLE_TEMPORAL_ORDER_CONSENSUS"
                    ordinal_inventories = _directional_ordinal_inventories(tasks, len(component))
                    if depth_visible_mode:
                        if not ordinal_inventories:
                            continue
                    elif not any(task["ordinal"] <= len(component) for task in tasks):
                        continue
                    component_rows = {candidate_id: centers[candidate_id] for candidate_id in component}
                    if depth_visible_mode:
                        video_directory = paths["transform"].parent
                        archive_hashes: dict[str, str] = {}
                        extracted: dict[str, Path] = {}
                        for source_key, suffix in (("depth_asset", ".png"), ("intrinsics_asset", ".pincam")):
                            asset_name = protocol["source"][source_key]
                            archive_path = video_directory / asset_name
                            asset_url = f"{protocol['source']['arkitscenes_base_url'].rstrip('/')}/raw/{fold}/{video_id}/{asset_name}"
                            _download_once(asset_url, archive_path)
                            archive_hashes[source_key] = _sha256(archive_path)
                            output_directory = video_directory / Path(asset_name).stem
                            _extract_archive_once(archive_path, output_directory, suffix)
                            extracted[source_key] = output_directory
                        component_diagnostics: dict[str, Any] = {
                            "component_candidate_ids": list(component),
                            "directional_ordinal_inventories": ordinal_inventories,
                        }
                        view = select_depth_visible_consensus_active_ordinal_view(
                            component_rows,
                            poses,
                            extracted["depth_asset"],
                            extracted["intrinsics_asset"],
                            algorithm,
                            component_diagnostics,
                        )
                        view_diagnostics.append(component_diagnostics)
                    elif algorithm.get("view_selection_mode") == "TEMPORAL_ORDER_CONSENSUS":
                        view = select_consensus_active_ordinal_view(
                            component_rows, poses, algorithm
                        )
                    else:
                        view = select_active_ordinal_view(component_rows, poses, algorithm)
                    if view is None:
                        continue
                    anchor_id = "SELF_CARRIER" if anchor is None else anchor.binding_id
                    cluster_id = f"{anchor_id}:{','.join(component)}"
                    contextual_lattices.append(
                        {
                            "cluster_id": cluster_id,
                            "context_anchor_id": anchor_id,
                            "context_anchor_label": None if anchor is None else anchor.label,
                            "context_anchor_center": None if anchor is None else anchor.center.tolist(),
                            "context_anchor_lengths": None if anchor is None else anchor.lengths.tolist(),
                            "directional_ordinal_inventories": ordinal_inventories,
                            "lattice": {
                                "candidate_ids_up_to_reversal": list(lattice.ordered_candidate_ids_up_to_reversal),
                                "pitch_m": lattice.pitch_m,
                                "maximum_pitch_relative_deviation": lattice.maximum_pitch_relative_deviation,
                                "maximum_orthogonal_residual_pitch_ratio": lattice.maximum_orthogonal_residual_pitch_ratio,
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
                                "consensus_window_pose_count": view.consensus_window_pose_count,
                                "consensus_support_count": view.consensus_support_count,
                                "consensus_support_fraction": view.consensus_support_fraction,
                                "depth_timestamp": view.depth_timestamp,
                                "intrinsic_timestamp": view.intrinsic_timestamp,
                                "frame_alignment_seconds": view.frame_alignment_seconds,
                                "visibility_depth_residual_m": view.visibility_depth_residual_m,
                            },
                            "candidates": {
                                candidate_id: {"label": centers[candidate_id].label, "center": centers[candidate_id].center.tolist()}
                                for candidate_id in component
                            },
                        }
                    )
            eligible = bool(contextual_lattices)
            row = {
                "visit_id": visit_id,
                "video_id": video_id,
                "directional_context_tasks": len(tasks),
                "directional_task_text": tasks,
                "trajectory_pose_count": len(poses),
                "functional_center_count": len(centers),
                "action_center_count": len(action_centers),
                "context_anchor_count": len(anchors),
                "contextual_lattice_count": len(contextual_lattices),
                "contextual_lattices": contextual_lattices,
                "view_diagnostics": view_diagnostics,
                "eligible": eligible,
            }
            scanned.append(row)
            if eligible:
                row["source_sha256"] = {**{name: _sha256(path) for name, path in paths.items()}, "camera_poses": _sha256(pose_path)}
                if algorithm.get("view_selection_mode") == "DEPTH_VISIBLE_TEMPORAL_ORDER_CONSENSUS":
                    row["source_sha256"].update(archive_hashes)
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
        "authority_boundary": "Admission reads public directional/context text and ordinal inventory, all privileged functional centers, privileged 3D contextual anchors, real camera poses, and, when frozen by protocol, frame-aligned depth/intrinsics. It never reads description annot_id, target membership, selector output, or evaluator scores.",
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
    print(json.dumps({"decision": result["decision"], "denominators": result["denominators"], "backend": None if result["execution_backend"] is None else {key: result["execution_backend"][key] for key in ("selected_backend", "selected_device_type", "selection_reason")}, "selected": [{"visit_id": row["visit_id"], "tasks": row["directional_context_tasks"], "lattices": row["contextual_lattice_count"], "poses": row["trajectory_pose_count"]} for row in result["selected"]]}, indent=2))


if __name__ == "__main__":
    main()
