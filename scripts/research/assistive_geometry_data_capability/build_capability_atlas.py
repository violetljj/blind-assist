#!/usr/bin/env python3
"""Build the AG-DCA TRAIN-only capability atlas and evaluate hypothesis contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import unproject_depth
from scripts.research.assistive_geometry_cbf.audit_grid_support import (
    evaluate_frame_metrics as evaluate_cbf_frame_metrics,
    frame_grid_metrics as cbf_frame_grid_metrics,
    ground_axes,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = Path(
    "docs/research/assistive-geometry-data-capability/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_PROTOCOL_2026-08-10.json"
)
REQUIREMENTS_RELATIVE = Path(
    "docs/research/assistive-geometry-data-capability/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_HYPOTHESIS_REQUIREMENTS_2026-08-10.json"
)
TARGET_FIELDS = (
    "depth_m_source",
    "depth_valid_source",
    "ground_probability_source",
    "ground_label_valid_source",
    "intrinsics_source",
    "up_camera",
    "camera_height_m",
    "ground_plane_valid",
    "clearance_m",
    "clearance_valid",
    "occupancy",
    "occupancy_valid",
)
EXPECTED_TARGET_KEYS = (
    "band_confidence_valid",
    "camera_height_m",
    "clearance_m",
    "clearance_valid",
    "depth_m_source",
    "depth_valid_source",
    "ground_label_valid_source",
    "ground_plane_valid",
    "ground_probability_source",
    "intrinsics_source",
    "intrinsics_tensor",
    "occupancy",
    "occupancy_valid",
    "orientation_index",
    "target_hw",
    "up_camera",
)
EXPECTED_DTYPES = {
    "band_confidence_valid": "bool",
    "camera_height_m": "float32",
    "clearance_m": "float32",
    "clearance_valid": "bool",
    "depth_m_source": "float32",
    "depth_valid_source": "bool",
    "ground_label_valid_source": "bool",
    "ground_plane_valid": "bool",
    "ground_probability_source": "float32",
    "intrinsics_source": "float32",
    "intrinsics_tensor": "float32",
    "occupancy": "float32",
    "occupancy_valid": "bool",
    "orientation_index": "int8",
    "target_hw": "int32",
    "up_camera": "float32",
}
CAPABILITY_NAMES = (
    "finite_clearance_event",
    "right_censor",
    "ground_plane_valid",
    "forward_ground_0_2m",
    "forward_ground_0_5m",
    "lateral_observation_pm_0_5m",
    "lateral_observation_pm_1_0m",
    "lateral_observation_pm_2_0m",
    "full_2_5d_grid",
    "occupancy_1_0m",
    "occupancy_1_5m",
    "occupancy_2_0m",
    "consecutive_temporal_pair",
    "explicit_timestamp_materialized",
    "valid_camera_geometry",
    "pose_transform_materialized",
    "truth_clear",
    "truth_occupied",
    "oracle_depth_factor",
    "oracle_ground_factor",
    "oracle_support_factor",
    "oracle_obstacle_factor",
    "fci_factor_truth_bundle",
    "fci_truth_clear_bundle",
    "fci_truth_occupied_bundle",
    "r2_depth_uncertainty_truth_materialized",
    "r2_support_uncertainty_truth_materialized",
    "r2_obstacle_boundary_truth_materialized",
    "r2_complete_factor_schema_truth",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"output collision: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def flatten_manifest(manifest: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    parents: list[str] = []
    frames: list[dict[str, Any]] = []
    for video in manifest.get("videos", []):
        parent = str(video["video_id"])
        require(str(video.get("role")) == "TRAIN", f"non-TRAIN parent: {parent}")
        parents.append(parent)
        frames.extend({**frame, "video_id": parent} for frame in video.get("frames", []))
    return parents, frames


def validate_manifest_contract(manifest: dict[str, Any], protocol: dict[str, Any]) -> None:
    expected = protocol["manifest_contract"]
    require(manifest.get("schema") == expected["schema"], "manifest schema drift")
    require(int(manifest.get("video_count", -1)) == int(expected["video_count"]), "video count drift")
    require(int(manifest.get("frame_count", -1)) == int(expected["frame_count"]), "frame count drift")
    require(
        int(manifest.get("portrait_frame_count", -1)) == int(expected["portrait_frame_count"]),
        "portrait frame count drift",
    )
    require(
        int(manifest.get("landscape_frame_count", -1)) == int(expected["landscape_frame_count"]),
        "landscape frame count drift",
    )
    for key in ("source_manifest_sha256", "producer_sha256"):
        require(str(manifest.get(key, "")).upper() == str(expected[key]).upper(), f"manifest {key} drift")
    for key in ("development_or_confirmation_content_opened", "model_outputs_read"):
        require(bool(manifest.get(key)) is bool(expected[key]), f"manifest {key} drift")
    visits: list[str] = []
    videos: list[str] = []
    stems: set[str] = set()
    target_paths: set[str] = set()
    source_paths: set[str] = set()
    for video in manifest.get("videos", []):
        parent = str(video["video_id"])
        visit = str(video["visit_id"])
        visits.append(visit)
        videos.append(parent)
        require(str(video.get("role")) == "TRAIN", f"non-TRAIN parent: {parent}")
        require(int(video.get("frame_count", -1)) == int(expected["frames_per_parent"]), f"frame count drift: {parent}")
        previous_timestamp: float | None = None
        for expected_index, frame in enumerate(video.get("frames", [])):
            require(int(frame["frame_index"]) == expected_index, f"frame index drift: {parent}")
            stem = str(frame["frame_stem"])
            timestamp = _timestamp(stem)
            require(previous_timestamp is None or timestamp > previous_timestamp, f"timestamp order drift: {parent}")
            previous_timestamp = timestamp
            target_path = str(frame["target"]["path"])
            source_path = str(frame["rgb_source"]["path"])
            require(stem not in stems, f"duplicate frame stem: {stem}")
            require(target_path not in target_paths, f"duplicate target path: {target_path}")
            require(source_path not in source_paths, f"duplicate source path: {source_path}")
            require("\\raw\\Training\\" in source_path.replace("/", "\\"), f"source escaped Training: {stem}")
            stems.add(stem)
            target_paths.add(target_path)
            source_paths.add(source_path)
    require(len(set(videos)) == len(videos) == int(expected["video_count"]), "video identity collision")
    require(len(set(visits)) == len(visits) == int(expected["video_count"]), "visit identity collision")
    require(len(set(zip(visits, videos))) == int(expected["video_count"]), "visit/video mapping collision")
    require(len(stems) == int(expected["frame_count"]), "unique frame count drift")
    protected_parents = set(str(value) for value in protocol["protected_parent_ids"])
    protected_visits = set(str(value) for value in protocol["protected_visit_ids"])
    require(not (set(videos) & protected_parents), "TRAIN parent intersects protected roster")
    require(not (set(visits) & protected_visits), "TRAIN visit intersects protected roster")


def _timestamp(frame_stem: str) -> float:
    try:
        return float(frame_stem.split("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"frame stem lacks a numeric timestamp: {frame_stem}") from error


def temporal_pair_flags(frames: list[dict[str, Any]], maximum_gap_seconds: float) -> dict[str, bool]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_parent[str(frame["video_id"])].append(frame)
    flags: dict[str, bool] = {}
    for values in by_parent.values():
        previous: dict[str, Any] | None = None
        for frame in values:
            frame_id = f"{frame['video_id']}/{frame['frame_stem']}"
            if previous is None:
                flags[frame_id] = False
            else:
                delta = _timestamp(str(frame["frame_stem"])) - _timestamp(str(previous["frame_stem"]))
                flags[frame_id] = (
                    int(frame["frame_index"]) == int(previous["frame_index"]) + 1
                    and 0.0 < delta <= maximum_gap_seconds
                )
            previous = frame
    return flags


def _grid_counts(
    forward: np.ndarray,
    lateral: np.ndarray,
    mask: np.ndarray,
    *,
    forward_range: tuple[float, float],
    lateral_range: tuple[float, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    forward_min, forward_max = forward_range
    lateral_min, lateral_max = lateral_range
    rows, columns = grid_shape
    inside = (
        mask
        & (forward >= forward_min)
        & (forward < forward_max)
        & (lateral >= lateral_min)
        & (lateral < lateral_max)
    )
    grid = np.zeros((rows, columns), dtype=bool)
    if np.any(inside):
        row = np.floor((forward[inside] - forward_min) / (forward_max - forward_min) * rows).astype(np.int64)
        column = np.floor((lateral[inside] - lateral_min) / (lateral_max - lateral_min) * columns).astype(np.int64)
        grid[np.clip(row, 0, rows - 1), np.clip(column, 0, columns - 1)] = True
    return grid


def _finite_camera_contract(target: dict[str, np.ndarray]) -> bool:
    depth = np.asarray(target["depth_m_source"])
    matrix = np.asarray(target["intrinsics_source"], dtype=np.float64)
    up = np.asarray(target["up_camera"], dtype=np.float64)
    height = float(np.asarray(target["camera_height_m"]).item())
    return bool(
        depth.ndim == 2
        and matrix.shape == (3, 3)
        and np.all(np.isfinite(matrix))
        and float(matrix[0, 0]) > 0.0
        and float(matrix[1, 1]) > 0.0
        and up.shape == (3,)
        and np.all(np.isfinite(up))
        and float(np.linalg.norm(up)) > 1e-9
        and np.isfinite(height)
        and bool(np.asarray(target["ground_plane_valid"]).item())
    )


def derive_capabilities(
    target: dict[str, np.ndarray],
    policy: dict[str, Any],
    *,
    consecutive_temporal_pair: bool,
) -> dict[str, bool]:
    depth = np.asarray(target["depth_m_source"], dtype=np.float64)
    depth_valid = np.asarray(target["depth_valid_source"], dtype=bool)
    ground_probability = np.asarray(target["ground_probability_source"], dtype=np.float64)
    ground_label_valid = np.asarray(target["ground_label_valid_source"], dtype=bool)
    clearance = np.asarray(target["clearance_m"], dtype=np.float64)
    clearance_valid = np.asarray(target["clearance_valid"], dtype=bool)
    occupancy = np.asarray(target["occupancy"], dtype=np.float64)
    occupancy_valid = np.asarray(target["occupancy_valid"], dtype=bool)
    require(depth_valid.shape == depth.shape, "depth validity shape drift")
    require(ground_probability.shape == depth.shape, "ground probability shape drift")
    require(ground_label_valid.shape == depth.shape, "ground label validity shape drift")
    require(clearance.shape == (3,) and clearance_valid.shape == (3,), "clearance shape drift")
    require(occupancy.shape == (3, 3) and occupancy_valid.shape == (3, 3), "occupancy shape drift")

    event = clearance_valid & np.isfinite(clearance) & (clearance <= 2.0)
    fully_clear_band = occupancy_valid.all(axis=-1) & (~(occupancy >= 0.5)).all(axis=-1)
    right_censor = (clearance_valid & np.isfinite(clearance) & (clearance > 2.0)) | (
        ~clearance_valid & fully_clear_band
    )
    truth_clear = bool(np.any(fully_clear_band))
    truth_occupied = bool(np.any(occupancy_valid & (occupancy >= 0.5)))
    camera_valid = _finite_camera_contract(target)

    values = {name: False for name in CAPABILITY_NAMES}
    values.update(
        {
            "finite_clearance_event": bool(np.any(event)),
            "right_censor": bool(np.any(right_censor)),
            "ground_plane_valid": bool(np.asarray(target["ground_plane_valid"]).item()),
            "occupancy_1_0m": bool(np.all(occupancy_valid[:, 0])),
            "occupancy_1_5m": bool(np.all(occupancy_valid[:, 1])),
            "occupancy_2_0m": bool(np.all(occupancy_valid[:, 2])),
            "consecutive_temporal_pair": consecutive_temporal_pair,
            "explicit_timestamp_materialized": False,
            "valid_camera_geometry": camera_valid,
            "pose_transform_materialized": False,
            "truth_clear": truth_clear,
            "truth_occupied": truth_occupied,
        }
    )

    valid_depth = (
        depth_valid
        & np.isfinite(depth)
        & (depth >= float(policy["depth_min_m"]))
        & (depth <= float(policy["depth_max_m"]))
    )
    metric_depth = int(np.count_nonzero(valid_depth)) >= int(policy["minimum_valid_depth_points"])
    values["oracle_depth_factor"] = metric_depth
    if not camera_valid or not metric_depth:
        return values

    valid_geometry = valid_depth & ground_label_valid
    intrinsics = np.asarray(target["intrinsics_source"], dtype=np.float64)
    points, pixels = unproject_depth(depth, intrinsics, valid_geometry, int(policy["point_stride"]))
    heading, lateral_axis = ground_axes(np.asarray(target["up_camera"], dtype=np.float64))
    forward = points @ heading
    lateral = points @ lateral_axis
    pixel_ground = ground_probability[pixels[:, 1], pixels[:, 0]] >= float(
        policy["ground_probability_threshold"]
    )
    observed = np.ones(len(points), dtype=bool)

    def ground_forward(maximum: float) -> bool:
        grid = _grid_counts(
            forward,
            lateral,
            pixel_ground,
            forward_range=(0.2, maximum),
            lateral_range=(-2.0, 2.0),
            grid_shape=(16, 31),
        )
        return bool(
            np.count_nonzero(grid) >= int(policy["minimum_ground_cells"])
            and min(np.count_nonzero(part) for part in np.array_split(grid, 4, axis=0))
            >= int(policy["minimum_ground_cells_per_forward_quartile"])
        )

    def lateral_observation(half_width: float) -> bool:
        grid = _grid_counts(
            forward,
            lateral,
            observed,
            forward_range=(0.2, 2.0),
            lateral_range=(-half_width, half_width),
            grid_shape=(16, 15),
        )
        return bool(
            np.count_nonzero(grid) >= int(policy["minimum_observed_cells"])
            and min(np.count_nonzero(part) for part in np.array_split(grid, 3, axis=1))
            >= int(policy["minimum_observed_cells_per_lateral_third"])
        )

    forward_2 = ground_forward(2.0)
    forward_5 = ground_forward(5.0)
    lateral_05 = lateral_observation(0.5)
    lateral_10 = lateral_observation(1.0)
    lateral_20 = lateral_observation(2.0)
    cbf_metrics = cbf_frame_grid_metrics(target, policy["cbf_grid_policy"])
    cbf_evaluable, _ = evaluate_cbf_frame_metrics(cbf_metrics, policy["cbf_grid_policy"])
    obstacle_factor = cbf_metrics["observed_cell_count"] >= int(policy["minimum_observed_cells"])
    ground_factor = cbf_metrics["ground_cell_count"] >= int(policy["minimum_ground_cells"])
    factor_bundle = metric_depth and ground_factor and forward_2 and obstacle_factor
    values.update(
        {
            "forward_ground_0_2m": forward_2,
            "forward_ground_0_5m": forward_5,
            "lateral_observation_pm_0_5m": lateral_05,
            "lateral_observation_pm_1_0m": lateral_10,
            "lateral_observation_pm_2_0m": lateral_20,
            "full_2_5d_grid": cbf_evaluable,
            "oracle_ground_factor": ground_factor,
            "oracle_support_factor": forward_2,
            "oracle_obstacle_factor": obstacle_factor,
            "fci_factor_truth_bundle": factor_bundle,
            "fci_truth_clear_bundle": factor_bundle and truth_clear,
            "fci_truth_occupied_bundle": factor_bundle and truth_occupied,
        }
    )
    return values


def validate_target_contract(loaded: Any, frame: dict[str, Any], frame_id: str) -> None:
    require(set(loaded.files) == set(EXPECTED_TARGET_KEYS), f"target keyset drift: {frame_id}")
    for field, expected_dtype in EXPECTED_DTYPES.items():
        require(str(np.asarray(loaded[field]).dtype) == expected_dtype, f"target dtype drift: {frame_id}/{field}")
    source_hw = tuple(int(value) for value in frame["source_hw"])
    target_hw = tuple(int(value) for value in frame["target_hw"])
    for field in (
        "depth_m_source",
        "depth_valid_source",
        "ground_probability_source",
        "ground_label_valid_source",
    ):
        require(np.asarray(loaded[field]).shape == source_hw, f"source shape drift: {frame_id}/{field}")
    require(np.asarray(loaded["intrinsics_source"]).shape == (3, 3), f"K source shape drift: {frame_id}")
    require(np.asarray(loaded["intrinsics_tensor"]).shape == (3, 3), f"K tensor shape drift: {frame_id}")
    require(np.asarray(loaded["up_camera"]).shape == (3,), f"up shape drift: {frame_id}")
    require(np.asarray(loaded["clearance_m"]).shape == (3,), f"clearance shape drift: {frame_id}")
    require(np.asarray(loaded["clearance_valid"]).shape == (3,), f"clearance valid shape drift: {frame_id}")
    require(np.asarray(loaded["occupancy"]).shape == (3, 3), f"occupancy shape drift: {frame_id}")
    require(np.asarray(loaded["occupancy_valid"]).shape == (3, 3), f"occupancy valid shape drift: {frame_id}")
    require(np.asarray(loaded["band_confidence_valid"]).shape == (3,), f"confidence shape drift: {frame_id}")
    require(tuple(int(value) for value in np.asarray(loaded["target_hw"]).tolist()) == target_hw, f"target HW drift: {frame_id}")
    require(int(np.asarray(loaded["orientation_index"]).item()) == int(frame["orientation_index"]), f"orientation drift: {frame_id}")
    require(
        bool(np.asarray(loaded["ground_plane_valid"]).item()) is bool(frame["ground_plane_valid"]),
        f"ground validity receipt drift: {frame_id}",
    )
    require(np.all(np.isfinite(np.asarray(loaded["ground_probability_source"]))), f"ground probability nonfinite: {frame_id}")
    require(np.all(np.isfinite(np.asarray(loaded["intrinsics_source"]))), f"K source nonfinite: {frame_id}")
    require(np.all(np.isfinite(np.asarray(loaded["up_camera"]))), f"up nonfinite: {frame_id}")
    require(np.all(np.isfinite(np.asarray(loaded["clearance_m"]))), f"clearance nonfinite: {frame_id}")
    require(np.all(np.isfinite(np.asarray(loaded["occupancy"]))), f"occupancy nonfinite: {frame_id}")


def aggregate_capabilities(
    frame_rows: list[dict[str, Any]],
    parent_order: list[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for capability in CAPABILITY_NAMES:
        parent_counts = {parent: 0 for parent in parent_order}
        orientation_counts = {"portrait": 0, "landscape": 0}
        for row in frame_rows:
            if row["capabilities"][capability]:
                parent_counts[row["video_id"]] += 1
                orientation_counts[row["orientation_family"]] += 1
        result[capability] = {
            "frame_count": int(sum(parent_counts.values())),
            "parent_frame_counts": parent_counts,
            "parents_with_any_support": [parent for parent in parent_order if parent_counts[parent] > 0],
            "orientation_frame_counts": orientation_counts,
        }
    return result


def evaluate_hypothesis(
    atlas: dict[str, Any],
    contract: dict[str, Any],
    authority_facts: dict[str, bool],
) -> dict[str, Any]:
    capability_results: dict[str, Any] = {}
    eligible_sets: list[set[str]] = []
    data_pass = True
    for capability, gate in contract.get("capabilities", {}).items():
        require(capability in atlas["capabilities"], f"unknown capability: {capability}")
        value = atlas["capabilities"][capability]
        minimum_per_parent = int(gate.get("minimum_frames_per_parent", 1))
        eligible = [
            parent
            for parent in atlas["parent_order"]
            if int(value["parent_frame_counts"][parent]) >= minimum_per_parent
        ]
        orientation_minimum = int(gate.get("minimum_frames_per_orientation", 0))
        passed = (
            int(value["frame_count"]) >= int(gate.get("minimum_total_frames", 0))
            and len(eligible) >= int(gate.get("minimum_parents", 0))
            and all(
                int(value["orientation_frame_counts"].get(name, 0)) >= orientation_minimum
                for name in ("portrait", "landscape")
            )
        )
        data_pass = data_pass and passed
        eligible_sets.append(set(eligible))
        capability_results[capability] = {
            "passed": passed,
            "observed_total_frames": value["frame_count"],
            "eligible_parents": eligible,
            "observed_orientation_frames": value["orientation_frame_counts"],
            "gate": gate,
        }
    joint = set(atlas["parent_order"]) if not eligible_sets else set.intersection(*eligible_sets)
    joint_gate = contract.get("joint_parent_gate", {})
    joint_required = int(joint_gate.get("minimum_joint_parents", 0))
    fit_required = int(joint_gate.get("minimum_fit_parents", 0))
    eval_required = int(joint_gate.get("minimum_eval_parents", 0))
    joint_pass = len(joint) >= max(joint_required, fit_required + eval_required)
    data_pass = data_pass and joint_pass

    authority_results: dict[str, Any] = {}
    authority_pass = True
    for fact, expected in contract.get("authority_requirements", {}).items():
        require(fact in authority_facts, f"unknown authority fact: {fact}")
        actual = bool(authority_facts[fact])
        passed = actual is bool(expected)
        authority_pass = authority_pass and passed
        authority_results[fact] = {"expected": bool(expected), "actual": actual, "passed": passed}
    if data_pass and authority_pass:
        terminal = "SUPPORTED_FOR_PROTOCOL_LOCK"
    elif not data_pass and not authority_pass:
        terminal = "NOT_SUPPORTED_DATA_AND_AUTHORITY"
    elif not data_pass:
        terminal = "NOT_SUPPORTED_DATA"
    else:
        terminal = "NOT_SUPPORTED_AUTHORITY"
    return {
        "terminal": terminal,
        "data_pass": data_pass,
        "authority_pass": authority_pass,
        "capability_results": capability_results,
        "joint_eligible_parents": [parent for parent in atlas["parent_order"] if parent in joint],
        "joint_parent_gate": {**joint_gate, "passed": joint_pass},
        "authority_results": authority_results,
        "execution_authorized": False,
        "claim_ceiling": "Data/authority admission decision only; never algorithm execution or scientific promotion authority.",
    }


def evaluate_requirements(
    atlas: dict[str, Any],
    requirements: dict[str, Any],
    authority_facts: dict[str, bool],
) -> dict[str, Any]:
    require(
        requirements.get("schema") == "blindassist.assistive_geometry_dca.hypothesis_requirements.v1",
        "requirements schema drift",
    )
    return {
        hypothesis: evaluate_hypothesis(atlas, contract, authority_facts)
        for hypothesis, contract in requirements.get("hypotheses", {}).items()
    }


def _validate_target_path(path: Path) -> None:
    normalized = str(path.resolve()).replace("/", "\\").lower()
    require(
        "assistive-geometry-b1-train-targets-20260809-r0\\targets\\" in normalized,
        f"target escaped frozen TRAIN root: {path}",
    )
    require("development" not in normalized and "confirmation" not in normalized, "protected target path")


def validate_protocol(protocol: dict[str, Any], requirements: dict[str, Any]) -> None:
    require(
        protocol.get("schema") == "blindassist.assistive_geometry_dca.r0_protocol.v1",
        "protocol schema drift",
    )
    require(protocol.get("status") == "TRAIN_CAPABILITY_ATLAS_LOCKED_NOT_RUN", "status drift")
    require(tuple(protocol.get("target_field_access", [])) == TARGET_FIELDS, "target field access drift")
    require(set(protocol.get("target_keyset", [])) == set(EXPECTED_TARGET_KEYS), "target keyset drift")
    for key in ("rgb_access", "model_access", "feature_access", "development_access", "confirmation_access"):
        require(protocol["authority"].get(key) is False, f"protected authority leaked: {key}")
    require(
        set(protocol.get("authority_facts", {}))
        == {
            "r2_f0_reducer_tracked_and_frozen",
            "oracle_factor_injection_interface_tracked_and_frozen",
            "fresh_selection_eligible_paired_outcome_available",
            "b1_consumed_development_allowed_for_r2_selection",
        },
        "authority fact set drift",
    )
    source = protocol["input"]
    require(source.get("data_role") == "TRAIN", "input role drift")
    source_path = REPO_ROOT / source["path"]
    require(source_path.is_file(), "input manifest missing")
    require(sha256_file(source_path) == source["sha256"], "input manifest SHA drift")
    role_protocol = protocol["role_protocol"]
    role_path = REPO_ROOT / role_protocol["path"]
    require(role_path.is_file(), "role protocol missing")
    require(sha256_file(role_path) == role_protocol["sha256"], "role protocol SHA drift")
    requirement_path = REPO_ROOT / protocol["requirements"]["path"]
    require(requirement_path.resolve() == (REPO_ROOT / REQUIREMENTS_RELATIVE).resolve(), "requirements path drift")
    require(sha256_file(requirement_path) == protocol["requirements"]["sha256"], "requirements SHA drift")
    require(requirements.get("schema") == "blindassist.assistive_geometry_dca.hypothesis_requirements.v1", "requirements schema drift")
    f0_evidence = protocol["authority_evidence"]["r2_f0"]
    for item in ("result", "reducer"):
        evidence_path = REPO_ROOT / f0_evidence[item]["path"]
        require(evidence_path.is_file(), f"R2 F0 authority evidence missing: {item}")
        require(sha256_file(evidence_path) == f0_evidence[item]["sha256"], f"R2 F0 authority SHA drift: {item}")
    f0_result = load_json(REPO_ROOT / f0_evidence["result"]["path"])
    require(f0_result.get("passed") is True, "R2 F0 authority evidence is not PASS")
    require(
        f0_result.get("terminal")
        == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS",
        "R2 F0 authority terminal drift",
    )
    require(protocol["authority_facts"]["r2_f0_reducer_tracked_and_frozen"] is True, "R2 F0 fact drift")
    implementation = protocol["implementation"]
    expected_paths = {
        "scripts/research/assistive_geometry_data_capability/build_capability_atlas.py",
        "scripts/research/assistive_geometry_data_capability/check_hypothesis_requirements.py",
        "scripts/research/assistive_geometry_data_capability/test_build_capability_atlas.py",
        "scripts/research/assistive_geometry/arkitscenes_truth_reader.py",
        "scripts/research/assistive_geometry_cbf/audit_grid_support.py",
    }
    require(set(implementation) == expected_paths, "implementation path set drift")
    for logical, expected_sha in implementation.items():
        require(sha256_file(REPO_ROOT / logical) == expected_sha, f"implementation SHA drift: {logical}")


def execute(protocol_path: Path, atlas_output: Path, decisions_output: Path) -> int:
    canonical = (REPO_ROOT / PROTOCOL_RELATIVE).resolve()
    require(protocol_path.resolve() == canonical, "custom protocol is not authorized")
    protocol = load_json(canonical)
    requirements = load_json(REPO_ROOT / protocol["requirements"]["path"])
    validate_protocol(protocol, requirements)
    require(atlas_output.resolve() == (REPO_ROOT / protocol["outputs"]["atlas"]).resolve(), "atlas output drift")
    require(decisions_output.resolve() == (REPO_ROOT / protocol["outputs"]["decisions"]).resolve(), "decision output drift")
    require(not atlas_output.exists() and not decisions_output.exists(), "output collision")

    manifest = load_json(REPO_ROOT / protocol["input"]["path"])
    validate_manifest_contract(manifest, protocol)
    parent_order, frames = flatten_manifest(manifest)
    require(parent_order == protocol["parent_order"], "parent order drift")
    require(len(frames) == int(protocol["expected_frame_count"]), "frame count drift")
    pair_flags = temporal_pair_flags(frames, float(protocol["capability_policy"]["maximum_temporal_gap_seconds"]))
    started = time.perf_counter()
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        receipt = frame["target"]
        path = Path(str(receipt["path"]))
        frame_id = f"{frame['video_id']}/{frame['frame_stem']}"
        _validate_target_path(path)
        require(path.is_file(), f"missing target: {frame_id}")
        require(path.stat().st_size == int(receipt["bytes"]), f"target bytes drift: {frame_id}")
        require(sha256_file(path) == str(receipt["sha256"]).upper(), f"target SHA drift: {frame_id}")
        with np.load(path, allow_pickle=False) as loaded:
            validate_target_contract(loaded, frame, frame_id)
            require(all(field in loaded.files for field in TARGET_FIELDS), f"target field missing: {frame_id}")
            target = {field: np.asarray(loaded[field]) for field in TARGET_FIELDS}
        orientation = str(frame["orientation_family"])
        require(orientation in ("portrait", "landscape"), f"orientation drift: {frame_id}")
        frame_rows.append(
            {
                "video_id": str(frame["video_id"]),
                "orientation_family": orientation,
                "capabilities": derive_capabilities(
                    target,
                    protocol["capability_policy"],
                    consecutive_temporal_pair=pair_flags[frame_id],
                ),
            }
        )
    atlas = {
        "schema": "blindassist.assistive_geometry_dca.r0_atlas.v1",
        "protocol_sha256": sha256_file(canonical),
        "input_manifest_sha256": protocol["input"]["sha256"],
        "frame_count": len(frame_rows),
        "parent_order": parent_order,
        "capabilities": aggregate_capabilities(frame_rows, parent_order),
        "authority_facts": protocol["authority_facts"],
        "elapsed_seconds": time.perf_counter() - started,
        "claim_ceiling": "TRAIN-only truth/source capability inventory; no model, causal attribution, selection, promotion, product or safety authority.",
    }
    decisions = {
        "schema": "blindassist.assistive_geometry_dca.r0_hypothesis_decisions.v1",
        "protocol_sha256": atlas["protocol_sha256"],
        "requirements_sha256": protocol["requirements"]["sha256"],
        "atlas_sha256_after_write": None,
        "decisions": evaluate_requirements(atlas, requirements, protocol["authority_facts"]),
        "execution_authorized": False,
    }
    _write_json_exclusive(atlas_output, atlas)
    decisions["atlas_sha256_after_write"] = sha256_file(atlas_output)
    _write_json_exclusive(decisions_output, decisions)
    print(
        json.dumps(
            {
                "frame_count": atlas["frame_count"],
                "elapsed_seconds": atlas["elapsed_seconds"],
                "decisions": {
                    name: value["terminal"] for name, value in decisions["decisions"].items()
                },
                "atlas_output": str(atlas_output),
                "decisions_output": str(decisions_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument(
        "--atlas-output",
        type=Path,
        default=REPO_ROOT / "artifacts.local/evidence/assistive-geometry-data-capability/r0/atlas.json",
    )
    parser.add_argument(
        "--decisions-output",
        type=Path,
        default=REPO_ROOT / "artifacts.local/evidence/assistive-geometry-data-capability/r0/hypothesis-decisions.json",
    )
    args = parser.parse_args()
    return execute(args.protocol.resolve(), args.atlas_output.resolve(), args.decisions_output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
