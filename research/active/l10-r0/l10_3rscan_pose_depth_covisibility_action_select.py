#!/usr/bin/env python3
"""Select a baseline-rich action view while preserving RGB-D covisibility."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_query_mask_3d_track as track  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_temporal_scale_vacancy_confirmation_source as old  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-pose-depth-covisibility-action-select-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-pose-depth-covisibility-action-select-result-v1"


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _target_image(row: dict[str, Any], cohort: dict[str, Any]) -> dict[str, Any]:
    candidate = cohort["candidate"]
    return {
        "episode_id": "FDV_covisibility_action",
        "role": "query",
        "scan_id": str(candidate["rescan_id"]),
        "target_instance_id": int(candidate["target_instance_id"]),
        "target_label": "door",
        "frame": int(row["frame"]),
        "color_size": row["color_size"],
        "bbox_xyxy": row["bbox_xyxy"],
        "zip_member": row["zip_member"],
    }


def run(protocol_path: Path, output_path: Path, cohort_output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    inputs: dict[str, dict[str, Any]] = {}
    for name in ("source_result", "base_source_protocol", "cohort", "failed_confirmation"):
        row = protocol[name]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{name.upper()}_HASH")
        inputs[name] = pixel.load_json(path)
    source = inputs["source_result"]
    base = inputs["base_source_protocol"]
    cohort = inputs["cohort"]
    pixel.require(int(source["selected_queue_index"]) == int(protocol["queue_index"]), "SOURCE_QUEUE_INDEX")
    pixel.require(int(cohort["source_queue_index"]) == int(protocol["queue_index"]), "COHORT_QUEUE_INDEX")
    pixel.require(inputs["failed_confirmation"]["conclusion"] == protocol["failed_confirmation"]["required_conclusion"], "FAILED_CONFIRMATION_CONCLUSION")

    scan_id = str(cohort["candidate"]["rescan_id"])
    artifact_root = Path(cohort["artifact_root"])
    zip_row = cohort["source_manifest"][f"{scan_id}/sequence.zip"]
    zip_path = artifact_root / zip_row["path"]
    pixel.require(zip_path.stat().st_size == int(zip_row["bytes"]), "ZIP_BYTES")
    pixel.require(pixel.sha256(zip_path) == zip_row["sha256"], "ZIP_HASH")
    current_key = str(protocol["selection"]["current_query_key"])
    current_frame = int(cohort["images"][current_key]["frame"])
    excluded = {int(frame) for frame in protocol["selection"]["excluded_frames"]}
    minimum_translation = float(protocol["selection"]["minimum_translation_metres"])
    maximum_translation = float(protocol["selection"]["maximum_translation_metres"])
    maximum_rotation = float(protocol["selection"]["maximum_rotation_degrees"])
    tolerance = float(protocol["selection"]["depth_consistency_metres"])

    with zipfile.ZipFile(zip_path) as archive:
        info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        frames = sorted(
            int(match.group(1))
            for name in archive.namelist()
            if (match := re.fullmatch(r"frame-(\d{6})\.pose\.txt", name))
        )
        current_pose = pixel.read_pose(archive, current_frame)
        current_depth = pixel.decode_depth(archive, current_frame)
        full_mask = np.ones((int(info["color_height"]), int(info["color_width"])), dtype=np.bool_)
        current_points = track._lift(full_mask, current_depth, current_pose, info)
        receipts = []
        depth_frames_opened = {current_frame}
        for frame in frames:
            if frame in excluded:
                continue
            pose = pixel.read_pose(archive, frame)
            if not np.isfinite(pose).all():
                continue
            translation = float(np.linalg.norm(current_pose[:3, 3] - pose[:3, 3]))
            rotation = _angle_degrees(current_pose, pose)
            if not (minimum_translation <= translation <= maximum_translation and rotation <= maximum_rotation):
                continue
            depth = pixel.decode_depth(archive, frame)
            depth_frames_opened.add(frame)
            candidate_points = track._lift(full_mask, depth, pose, info)
            _, current_to_candidate_visible, current_to_candidate_inside = track._coverage(
                current_points, full_mask, depth, pose, info, tolerance
            )
            _, candidate_to_current_visible, candidate_to_current_inside = track._coverage(
                candidate_points, full_mask, current_depth, current_pose, info, tolerance
            )
            current_fraction = current_to_candidate_visible / max(len(current_points), 1)
            candidate_fraction = candidate_to_current_visible / max(len(candidate_points), 1)
            mutual_covisibility = math.sqrt(current_fraction * candidate_fraction)
            receipts.append(
                {
                    "frame": frame,
                    "translation_metres": translation,
                    "rotation_degrees": rotation,
                    "current_to_candidate_inside_points": current_to_candidate_inside,
                    "current_to_candidate_visible_points": current_to_candidate_visible,
                    "candidate_to_current_inside_points": candidate_to_current_inside,
                    "candidate_to_current_visible_points": candidate_to_current_visible,
                    "mutual_covisibility": mutual_covisibility,
                    "selection_score": translation * mutual_covisibility,
                }
            )
    pixel.require(bool(receipts), "NO_ACTION_CANDIDATES")
    ranked = sorted(
        receipts,
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["mutual_covisibility"]),
            -float(row["translation_metres"]),
            int(row["frame"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected = ranked[0]

    data_root = artifact_root / "datasets/3rscan"
    _, target_candidates, target_opened = old.views._candidates(
        data_root,
        scan_id,
        int(cohort["candidate"]["target_instance_id"]),
        base["candidate_view_rules"],
    )
    target_rows = {int(row["frame"]): row for row, _, _ in target_candidates}
    evaluable = int(selected["frame"]) in target_rows
    selected_target = target_rows.get(int(selected["frame"]))
    development_cohort = deepcopy(cohort)
    development_cohort.update(
        {
            "authority": "CONSUMED_QUEUE_ROW_3_PRE_RGB_POSE_DEPTH_COVISIBILITY_ACTION_DEVELOPMENT_COHORT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": pixel.sha256(protocol_path),
        }
    )
    if evaluable:
        action_key = str(protocol["selection"]["action_query_key"])
        development_cohort["images"][action_key] = _target_image(selected_target, cohort)
        development_cohort["panel"]["action_query_key"] = action_key
        development_cohort["panel"]["ordered_query_keys"] = [
            str(protocol["selection"]["initial_query_key"]),
            current_key,
            action_key,
        ]
        development_cohort["panel"]["fixed_action"] = "POSE_DEPTH_COVISIBILITY_BASELINE"
        development_cohort["panel"]["selection"] = (
            "Truth-free rank 1 over all pose/depth frames satisfying frozen motion bounds; "
            "target geometry is attached only after action selection for evaluation."
        )
        pixel.atomic_write_json(cohort_output_path, development_cohort)

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_QUEUE_ROW_3_POST_FAILURE_PRE_RGB_POSE_DEPTH_ACTION_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "queue_index": int(protocol["queue_index"]),
        "current_frame": current_frame,
        "pose_frame_count": len(frames),
        "admissible_action_count": len(ranked),
        "ranked_actions": ranked,
        "selected_action": selected,
        "selected_target_evaluation_only": selected_target,
        "selected_action_target_evaluable": evaluable,
        "development_cohort": (
            {"path": cohort_output_path.name, "sha256": pixel.sha256(cohort_output_path)}
            if evaluable
            else None
        ),
        "runtime": {
            "pose_members_opened": len(frames),
            "depth_members_opened_for_selection": len(depth_frames_opened),
            "rgb_members_opened": 0,
            "model_calls": 0,
            "target_geometry_used_for_selection": False,
            "target_geometry_used_after_selection_for_evaluation": True,
            "target_screen_opened": target_opened,
        },
        "literature_motivation": protocol["literature_motivation"],
        "conclusion": (
            "L10_3RSCAN_POSE_DEPTH_COVISIBILITY_ACTION_DEVELOPMENT_SOURCE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_POSE_DEPTH_COVISIBILITY_ACTION_DEVELOPMENT_SOURCE_NOT_EVALUABLE"
        ),
        "next_action": protocol["next_action"] if evaluable else protocol["fallback_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve(), args.cohort_output.resolve())


if __name__ == "__main__":
    main()
