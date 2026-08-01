#!/usr/bin/env python3
"""Run the frozen HFTF H1 geometry-teacher mechanism canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from verify_sanpo_pose_geometry_authority import (
    SCHEMA as AUTHORITY_SCHEMA,
    _load_json,
    _load_jsonl,
    _quaternion_matrix_xyzw,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
)


PROTOCOL_RESULT_CONTRACTS = {
    "blindassist_hftf_h1_geometry_teacher_canary_protocol_r0": {
        "result_schema": (
            "blindassist_hftf_h1_geometry_teacher_canary_result_r0"
        ),
        "claim_ceiling": "SYNTHETIC_GEOMETRY_PROXY_MECHANICS_ONLY",
    },
    "blindassist_hftf_h1_forward_sector_geometry_teacher_canary_protocol_r1": {
        "result_schema": (
            "blindassist_hftf_h1_forward_sector_geometry_teacher_canary_result_r1"
        ),
        "claim_ceiling": (
            "SYNTHETIC_FORWARD_SECTOR_GEOMETRY_PROXY_MECHANICS_ONLY"
        ),
    },
}
EXPECTED_TRANSFORM = (
    "p_world = R_xyzw @ p_opencv_camera + camera_translation_m"
)
ADMITTED_AUTHORITY_TERMINALS = {
    "HFTF_H0_1_SANPO_PROXY_FRAME_ADMITTED",
    "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(binding: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(binding["position_m"], dtype=np.float64)
    rotation = _quaternion_matrix_xyzw(binding["quaternion_xyzw"])
    return translation, rotation


def _select_horizon_indices(
    timestamps_ms: list[int],
    source_frame_indices: list[int],
    target_delta_ms: int,
    tolerance_ms: int,
) -> list[int | None]:
    selected: list[int | None] = []
    values = np.asarray(timestamps_ms, dtype=np.int64)
    for index, (timestamp, source_frame) in enumerate(
        zip(timestamps_ms, source_frame_indices)
    ):
        target = timestamp + target_delta_ms
        candidates = [
            candidate
            for candidate, candidate_source_frame in enumerate(
                source_frame_indices
            )
            if candidate_source_frame > source_frame
        ]
        if not candidates:
            selected.append(None)
            continue
        candidate = min(
            candidates,
            key=lambda candidate_index: (
                abs(int(values[candidate_index]) - target),
                source_frame_indices[candidate_index],
            ),
        )
        error = abs(int(values[candidate]) - target)
        selected.append(
            candidate if error <= tolerance_ms else None
        )
    return selected


def _required_denominators(
    usable_anchor_count: int,
    theta_count: int,
    distance_count: int,
    height_count: int,
) -> dict[str, int]:
    return {
        "known_per_horizon": (
            usable_anchor_count
            * theta_count
            * distance_count
            * height_count
        ),
        "height_disagreement": (
            usable_anchor_count * theta_count * distance_count
        ),
        "future_union": (
            usable_anchor_count
            * theta_count
            * distance_count
            * height_count
        ),
    }


def _coverage_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _theta_edges(field_contract: dict[str, Any]) -> np.ndarray:
    theta_count = int(field_contract["theta_bin_count"])
    theta_range = np.asarray(
        field_contract["theta_range_degrees"], dtype=np.float64
    )
    if (
        theta_count <= 0
        or theta_range.shape != (2,)
        or not np.all(np.isfinite(theta_range))
        or not theta_range[0] < theta_range[1]
        or theta_range[0] < -180.0
        or theta_range[1] > 180.0
    ):
        raise ValueError("Invalid frozen theta field contract")
    return np.linspace(
        math.radians(float(theta_range[0])),
        math.radians(float(theta_range[1])),
        theta_count + 1,
    )


def _anchor_basis(
    binding: dict[str, Any],
    plane: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_position, rotation = _pose(binding)
    origin = np.asarray(
        plane["camera_ground_projection_m"], dtype=np.float64
    )
    up = np.asarray(plane["normal_toward_camera"], dtype=np.float64)
    up /= np.linalg.norm(up)
    forward = rotation @ np.asarray([0.0, 0.0, 1.0])
    forward = forward - float(forward @ up) * up
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-8:
        raise ValueError("Camera forward is degenerate on local ground plane")
    forward /= forward_norm
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    if float((camera_position - origin) @ up) <= 0:
        raise ValueError("Local ground normal does not point toward camera")
    return origin, forward, right, up


def _obstacle_points_world(
    replay_root: Path,
    row: dict[str, Any],
    binding: dict[str, Any],
    camera: dict[str, Any],
    *,
    stride: int,
    offset: int,
    excluded_classes: set[int],
    dynamic_classes: set[int],
    depth_override: np.ndarray | None = None,
    semantic_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(row["width"]), int(row["height"])
    depth = (
        depth_override
        if depth_override is not None
        else _read_depth(
            _resolve_inside(replay_root, str(row["source_depth_path"])),
            width,
            height,
        )
    )
    semantic = (
        semantic_override
        if semantic_override is not None
        else _read_semantic_class(
            _resolve_inside(replay_root, str(row["source_mask_path"])),
            width,
            height,
        )
    )
    y_grid = np.arange(offset, height, stride)
    x_grid = np.arange(offset, width, stride)
    u, v = np.meshgrid(x_grid, y_grid)
    z = depth[v, u]
    classes = semantic[v, u]
    valid = (
        np.isfinite(z)
        & (z > 0.0)
        & ~np.isin(classes, list(excluded_classes))
    )
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    classes = classes[valid]
    points_camera = np.stack(
        (
            (u - float(camera["cx"])) * z / float(camera["fx"]),
            (v - float(camera["cy"])) * z / float(camera["fy"]),
            z,
        ),
        axis=0,
    )
    translation, rotation = _pose(binding)
    points_world = rotation @ points_camera + translation[:, None]
    dynamic = np.isin(classes, list(dynamic_classes))
    return points_world, dynamic


def _bin_obstacle_support(
    points_world: np.ndarray,
    dynamic: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    height_bands: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    theta_count = len(theta_edges) - 1
    if theta_count <= 0 or not np.all(np.diff(theta_edges) > 0.0):
        raise ValueError("Theta edges must be finite and increasing")
    origin, forward, right, up = basis
    relative = points_world - origin[:, None]
    height = up @ relative
    forward_coordinate = forward @ relative
    right_coordinate = right @ relative
    radial = np.hypot(forward_coordinate, right_coordinate)
    theta = np.arctan2(right_coordinate, forward_coordinate)
    full_circle = math.isclose(
        float(theta_edges[-1] - theta_edges[0]),
        2.0 * math.pi,
        abs_tol=1e-12,
        rel_tol=0.0,
    )
    if full_circle:
        theta_for_bin = (
            (theta - theta_edges[0]) % (2.0 * math.pi)
        ) + theta_edges[0]
    else:
        theta_for_bin = theta
    theta_index = np.searchsorted(
        theta_edges, theta_for_bin, side="right"
    ) - 1
    if not full_circle:
        theta_index = np.where(
            np.isclose(
                theta_for_bin,
                theta_edges[-1],
                atol=1e-12,
                rtol=0.0,
            ),
            theta_count - 1,
            theta_index,
        )
    theta_valid = (
        (theta_index >= 0)
        & (theta_index < theta_count)
        & np.isfinite(theta_for_bin)
    )
    distance_index = np.searchsorted(
        distance_edges, radial, side="right"
    ) - 1
    distance_count = len(distance_edges) - 1
    distance_index = np.where(
        np.isclose(radial, distance_edges[-1], atol=1e-12, rtol=0.0),
        distance_count - 1,
        distance_index,
    )
    counts = np.zeros(
        (theta_count, distance_count, len(height_bands)),
        dtype=np.int64,
    )
    dynamic_counts = np.zeros_like(counts)
    for height_index, (lower, upper) in enumerate(height_bands):
        upper_comparison = height <= upper if height_index == len(
            height_bands
        ) - 1 else height < upper
        valid = (
            theta_valid
            &
            (distance_index >= 0)
            & (distance_index < distance_count)
            & (height >= lower)
            & upper_comparison
        )
        np.add.at(
            counts,
            (
                theta_index[valid],
                distance_index[valid],
                np.full(int(valid.sum()), height_index),
            ),
            1,
        )
        dynamic_valid = valid & dynamic
        np.add.at(
            dynamic_counts,
            (
                theta_index[dynamic_valid],
                distance_index[dynamic_valid],
                np.full(int(dynamic_valid.sum()), height_index),
            ),
            1,
        )
    return counts, dynamic_counts


def _cell_probes_world(
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    height_bands: list[tuple[float, float]],
) -> np.ndarray:
    origin, forward, right, up = basis
    theta_count = len(theta_edges) - 1
    probes: list[np.ndarray] = []
    for theta_index in range(theta_count):
        theta_lower = theta_edges[theta_index]
        theta_upper = theta_edges[theta_index + 1]
        theta_center = (theta_lower + theta_upper) / 2.0
        for distance_index in range(len(distance_edges) - 1):
            distance_lower = float(distance_edges[distance_index])
            distance_upper = float(distance_edges[distance_index + 1])
            distance_center = (distance_lower + distance_upper) / 2.0
            for height_lower, height_upper in height_bands:
                height_center = (height_lower + height_upper) / 2.0
                polar_points = [
                    (theta_center, distance_center, height_center)
                ]
                polar_points.extend(
                    (
                        theta_value,
                        distance_value,
                        height_value,
                    )
                    for theta_value in (theta_lower, theta_upper)
                    for distance_value in (distance_lower, distance_upper)
                    for height_value in (height_lower, height_upper)
                )
                world = np.stack(
                    [
                        origin
                        + forward
                        * (
                            distance_value
                            * math.cos(theta_value)
                        )
                        + right
                        * (
                            distance_value
                            * math.sin(theta_value)
                        )
                        + up * height_value
                        for (
                            theta_value,
                            distance_value,
                            height_value,
                        ) in polar_points
                    ],
                    axis=1,
                )
                probes.append(world)
    return np.stack(probes, axis=0)


def _known_field(
    probes_world: np.ndarray,
    replay_root: Path,
    observation_row: dict[str, Any],
    observation_binding: dict[str, Any],
    camera: dict[str, Any],
    theta_count: int,
    distance_count: int,
    height_count: int,
    tolerance_m: float,
    minimum_passing: int,
    depth_override: np.ndarray | None = None,
    semantic_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(observation_row["width"]), int(
        observation_row["height"]
    )
    depth = (
        depth_override
        if depth_override is not None
        else _read_depth(
            _resolve_inside(
                replay_root, str(observation_row["source_depth_path"])
            ),
            width,
            height,
        )
    )
    translation, rotation = _pose(observation_binding)
    flat = probes_world.transpose(1, 0, 2).reshape(3, -1)
    camera_points = rotation.T @ (flat - translation[:, None])
    z = camera_points[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(
            float(camera["fx"]) * camera_points[0] / z
            + float(camera["cx"])
            + 0.5
        ).astype(np.int64)
        v = np.floor(
            float(camera["fy"]) * camera_points[1] / z
            + float(camera["cy"])
            + 0.5
        ).astype(np.int64)
    inside = (
        np.isfinite(z)
        & (z > 0.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    observed = np.zeros(z.shape, dtype=np.float64)
    observed[inside] = depth[v[inside], u[inside]]
    semantic = (
        semantic_override
        if semantic_override is not None
        else _read_semantic_class(
            _resolve_inside(
                replay_root, str(observation_row["source_mask_path"])
            ),
            width,
            height,
        )
    )
    observed_semantic = np.zeros(z.shape, dtype=np.int64)
    observed_semantic[inside] = semantic[v[inside], u[inside]]
    passing = (
        inside
        & np.isfinite(observed)
        & (observed > 0.0)
        & (observed + tolerance_m >= z)
        & (observed_semantic != 0)
    )
    cell_count = theta_count * distance_count * height_count
    passing = passing.reshape(cell_count, 9)
    passing_count = passing.sum(axis=1)
    known_score = passing_count.astype(np.float64) / 9.0
    known = passing_count >= minimum_passing
    shape = (theta_count, distance_count, height_count)
    return known.reshape(shape), known_score.reshape(shape)


def _field(
    replay_root: Path,
    row: dict[str, Any],
    binding: dict[str, Any],
    camera: dict[str, Any],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    probes_world: np.ndarray,
    protocol: dict[str, Any],
    observation_cache: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    teacher = protocol["teacher"]
    field_contract = protocol["field"]
    theta_count = int(field_contract["theta_bin_count"])
    theta_edges = _theta_edges(field_contract)
    distance_edges = np.asarray(
        field_contract["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(item) for item in field_contract["height_bands_m"][name])
        for name in ("foot", "body", "head")
    ]
    cache_key = str(row["id"])
    if cache_key not in observation_cache:
        width, height = int(row["width"]), int(row["height"])
        depth = _read_depth(
            _resolve_inside(
                replay_root, str(row["source_depth_path"])
            ),
            width,
            height,
        )
        semantic = _read_semantic_class(
            _resolve_inside(
                replay_root, str(row["source_mask_path"])
            ),
            width,
            height,
        )
        points, dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            camera,
            stride=int(teacher["point_sample_stride_xy"]),
            offset=int(teacher["point_sample_offset_xy"]),
            excluded_classes=set(
                teacher["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                teacher["dynamic_semantic_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
        observation_cache[cache_key] = {
            "depth": depth,
            "semantic": semantic,
            "points_world": points,
            "dynamic": dynamic,
        }
    cached = observation_cache[cache_key]
    points = cached["points_world"]
    dynamic = cached["dynamic"]
    counts, dynamic_counts = _bin_obstacle_support(
        points,
        dynamic,
        basis,
        theta_edges,
        distance_edges,
        height_bands,
    )
    known, known_score = _known_field(
        probes_world,
        replay_root,
        row,
        binding,
        camera,
        theta_count,
        len(distance_edges) - 1,
        len(height_bands),
        float(teacher["depth_front_tolerance_m"]),
        int(teacher["minimum_passing_known_probes"]),
        depth_override=cached["depth"],
        semantic_override=cached["semantic"],
    )
    saturation = float(teacher["risk_support_saturation_point_count"])
    risk = np.minimum(1.0, counts.astype(np.float64) / saturation)
    single_risk = np.max(risk, axis=2)
    return {
        "known": known,
        "known_score": known_score,
        "risk": risk,
        "single_risk": single_risk,
        "counts": counts,
        "dynamic_counts": dynamic_counts,
    }


def _validate_authority(
    replay_root: Path,
    rows: list[dict[str, Any]],
    authority_path: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = _load_json(authority_path)
    session_ids = sorted({str(row.get("session_id")) for row in rows})
    consumed_file_hashes_match = all(
        _sha256(
            _resolve_inside(
                replay_root, str(row["source_depth_path"])
            )
        )
        == row.get("source_depth_sha256")
        and _sha256(
            _resolve_inside(
                replay_root, str(row["source_mask_path"])
            )
        )
        == row.get("source_mask_sha256")
        for row in rows
    )
    checks = {
        "schema": authority.get("schema") == AUTHORITY_SCHEMA,
        "terminal": authority.get("terminal")
        in ADMITTED_AUTHORITY_TERMINALS,
        "single_matching_session": (
            authority.get("source_session_ids") == session_ids
            and len(session_ids) == 1
            and session_ids[0] == expected["source_session_id"]
        ),
        "authority_report_hash": _sha256(authority_path)
        == expected["authority_report_sha256"],
        "manifest_frame_count": authority.get("manifest_frame_count")
        == len(rows),
        "manifest_hash": authority.get("input_hashes", {}).get(
            "manifest_sha256"
        )
        == _sha256(replay_root / "manifest.replay.jsonl"),
        "frozen_manifest_hash": _sha256(
            replay_root / "manifest.replay.jsonl"
        )
        == expected["manifest_sha256"],
        "dataset_spec_hash": authority.get("input_hashes", {}).get(
            "dataset_spec_sha256"
        )
        == _sha256(replay_root / "dataset_spec.json"),
        "frozen_dataset_spec_hash": _sha256(
            replay_root / "dataset_spec.json"
        )
        == expected["dataset_spec_sha256"],
        "pose_hash": authority.get("input_hashes", {}).get(
            "camera_poses_sha256"
        )
        == _sha256(replay_root / "source_metadata/camera_poses.csv"),
        "frozen_camera_poses_hash": _sha256(
            replay_root / "source_metadata/camera_poses.csv"
        )
        == expected["camera_poses_sha256"],
        "consumed_depth_and_mask_hashes": consumed_file_hashes_match,
        "transform": authority.get("transform_direction_canary", {}).get(
            "admitted_semantics"
        )
        == EXPECTED_TRANSFORM,
        "vertical_axis": authority.get(
            "ground_and_body_proxy_canary", {}
        ).get("vertical_axis")
        == "+Z",
        "proxy_frame": authority.get(
            "ground_and_body_proxy_canary", {}
        ).get("standard_body_proxy_frame_admitted_for_h1")
        is True,
        "physical_calibration_not_claimed": authority.get(
            "ground_and_body_proxy_canary", {}
        ).get("physical_camera_to_body_calibration_admitted")
        is False,
    }
    return authority, {"checks": checks, "ok": all(checks.values())}


def _session_result(
    replay_root: Path,
    authority_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    spec = _load_json(replay_root / "dataset_spec.json")
    session_id = str(rows[0]["session_id"])
    expected_by_id = {
        item["source_session_id"]: item
        for item in protocol["required_sessions"]
    }
    if session_id not in expected_by_id:
        raise ValueError(f"Unfrozen H1 source session: {session_id}")
    authority, authority_validation = _validate_authority(
        replay_root,
        rows,
        authority_path,
        expected_by_id[session_id],
    )
    result: dict[str, Any] = {
        "source_session_id": session_id,
        "replay_root": str(replay_root),
        "authority_report_path": str(authority_path.resolve()),
        "authority_report_sha256": _sha256(authority_path),
        "manifest_sha256": _sha256(
            replay_root / "manifest.replay.jsonl"
        ),
        "authority_validation": authority_validation,
        "ok": False,
    }
    if not authority_validation["ok"]:
        return result

    bindings = authority["source_pose_authority"]["bindings"]
    binding_by_id = {
        binding["manifest_id"]: binding for binding in bindings
    }
    plane_by_id = {
        frame["manifest_id"]: frame["local_ground_plane"]
        for frame in authority["ground_and_body_proxy_canary"]["per_frame"]
        if frame.get("local_ground_plane") is not None
    }
    if any(row["id"] not in binding_by_id for row in rows) or any(
        row["id"] not in plane_by_id for row in rows
    ):
        result["error"] = "binding_or_ground_plane_missing"
        return result

    source_fps = float(
        authority["source_pose_authority"]["source_fps"]
    )
    source_frame_indices = [
        int(row["source_frame_index"]) for row in rows
    ]
    timestamps = [
        int(round(source_frame * 1000.0 / source_fps))
        for source_frame in source_frame_indices
    ]
    horizon_contract = protocol["field"]["horizons_ms"]
    tolerance_ms = int(protocol["field"]["horizon_tolerance_ms"])
    near_indices = _select_horizon_indices(
        timestamps,
        source_frame_indices,
        int(horizon_contract["near"]),
        tolerance_ms,
    )
    far_indices = _select_horizon_indices(
        timestamps,
        source_frame_indices,
        int(horizon_contract["far"]),
        tolerance_ms,
    )
    usable = [
        index
        for index, (near, far) in enumerate(
            zip(near_indices, far_indices)
        )
        if near is not None and far is not None
    ]

    theta_edges = _theta_edges(protocol["field"])
    theta_count = len(theta_edges) - 1
    distance_edges = np.asarray(
        protocol["field"]["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(
            float(item)
            for item in protocol["field"]["height_bands_m"][name]
        )
        for name in ("foot", "body", "head")
    ]
    cell_count = theta_count * (len(distance_edges) - 1) * len(
        height_bands
    )
    fields: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    denominators = _required_denominators(
        len(usable),
        theta_count,
        len(distance_edges) - 1,
        len(height_bands),
    )
    current_required = denominators["known_per_horizon"]
    horizon_known = {"current": 0, "near": 0, "far": 0}
    consistency_error = 0.0
    dynamic_support = 0
    atlas: list[dict[str, Any]] = []
    unknown_cell_atlas: list[dict[str, Any]] = []
    dynamic_cell_atlas: list[dict[str, Any]] = []
    observation_cache: dict[str, dict[str, Any]] = {}

    for anchor_index, anchor_row in enumerate(rows):
        if anchor_index not in usable:
            atlas.append(
                {
                    "anchor_manifest_id": anchor_row["id"],
                    "near_manifest_id": (
                        rows[near_indices[anchor_index]]["id"]
                        if near_indices[anchor_index] is not None
                        else None
                    ),
                    "far_manifest_id": (
                        rows[far_indices[anchor_index]]["id"]
                        if far_indices[anchor_index] is not None
                        else None
                    ),
                    "excluded_from_usable_anchor_set": True,
                    "current_unknown_fraction": 1.0,
                    "dynamic_support_points": 0,
                }
            )
            continue
        anchor_binding = binding_by_id[anchor_row["id"]]
        basis = _anchor_basis(
            anchor_binding, plane_by_id[anchor_row["id"]]
        )
        probes = _cell_probes_world(
            basis,
            theta_edges,
            distance_edges,
            height_bands,
        )
        targets = {
            "current": anchor_index,
            "near": near_indices[anchor_index],
            "far": far_indices[anchor_index],
        }
        anchor_fields: dict[str, dict[str, np.ndarray]] = {}
        for horizon_name, target_index in targets.items():
            if target_index is None:
                continue
            target_row = rows[target_index]
            value = _field(
                replay_root,
                target_row,
                binding_by_id[target_row["id"]],
                spec["camera"],
                basis,
                probes,
                protocol,
                observation_cache,
            )
            fields[(anchor_index, target_index)] = value
            anchor_fields[horizon_name] = value
            horizon_known[horizon_name] += int(value["known"].sum())
            dynamic_support += int(value["dynamic_counts"].sum())
            consistency_error = max(
                consistency_error,
                float(
                    np.max(
                        np.abs(
                            value["single_risk"]
                            - np.max(value["risk"], axis=2)
                        )
                    )
                ),
            )
        current_value = anchor_fields["current"]
        atlas.append(
            {
                "anchor_manifest_id": anchor_row["id"],
                "near_manifest_id": (
                    rows[near_indices[anchor_index]]["id"]
                    if near_indices[anchor_index] is not None
                    else None
                ),
                "far_manifest_id": (
                    rows[far_indices[anchor_index]]["id"]
                    if far_indices[anchor_index] is not None
                    else None
                ),
                "current_unknown_fraction": float(
                    1.0 - current_value["known"].mean()
                ),
                "dynamic_support_points": int(
                    sum(
                        int(value["dynamic_counts"].sum())
                        for value in anchor_fields.values()
                    )
                ),
                "excluded_from_usable_anchor_set": False,
            }
        )
        unknown_indices = np.argwhere(~current_value["known"])
        for theta_index, distance_index, height_index in unknown_indices[:3]:
            unknown_cell_atlas.append(
                {
                    "anchor_manifest_id": anchor_row["id"],
                    "theta_index": int(theta_index),
                    "distance_index": int(distance_index),
                    "height_band": ("foot", "body", "head")[
                        int(height_index)
                    ],
                    "known_score": float(
                        current_value["known_score"][
                            theta_index, distance_index, height_index
                        ]
                    ),
                }
            )
        dynamic_counts = current_value["dynamic_counts"]
        nonzero_dynamic = np.argwhere(dynamic_counts > 0)
        ranked_dynamic = sorted(
            nonzero_dynamic.tolist(),
            key=lambda index: (
                -int(dynamic_counts[tuple(index)]),
                index,
            ),
        )
        for theta_index, distance_index, height_index in ranked_dynamic[:3]:
            dynamic_cell_atlas.append(
                {
                    "anchor_manifest_id": anchor_row["id"],
                    "theta_index": int(theta_index),
                    "distance_index": int(distance_index),
                    "height_band": ("foot", "body", "head")[
                        int(height_index)
                    ],
                    "dynamic_support_point_count": int(
                        dynamic_counts[
                            theta_index, distance_index, height_index
                        ]
                    ),
                }
            )

    height_numerator = 0
    height_denominator = 0
    future_numerator = 0
    future_denominator = 0
    change_atlas: list[dict[str, Any]] = []
    delta_height = float(protocol["gates"]["height_disagreement_delta"])
    delta_future = float(protocol["gates"]["future_change_delta"])
    for anchor_index in usable:
        current = fields[(anchor_index, anchor_index)]
        all_height_known = np.all(current["known"], axis=2)
        height_delta = np.max(current["risk"], axis=2) - np.min(
            current["risk"], axis=2
        )
        height_denominator += (
            theta_count * (len(distance_edges) - 1)
        )
        height_numerator += int(
            (all_height_known & (height_delta >= delta_height)).sum()
        )
        near_index = near_indices[anchor_index]
        far_index = far_indices[anchor_index]
        future_values = [
            fields[(anchor_index, index)]
            for index in (near_index, far_index)
            if index is not None
        ]
        if not future_values:
            continue
        jointly_known = np.zeros(current["known"].shape, dtype=bool)
        changed = np.zeros_like(jointly_known)
        maximum_delta = np.zeros(current["known"].shape, dtype=np.float64)
        for future in future_values:
            joint = current["known"] & future["known"]
            delta = np.abs(future["risk"] - current["risk"])
            jointly_known |= joint
            changed |= joint & (delta >= delta_future)
            maximum_delta = np.maximum(
                maximum_delta, np.where(joint, delta, 0.0)
            )
        future_denominator += cell_count
        future_numerator += int(changed.sum())
        if jointly_known.any():
            masked_delta = np.where(
                jointly_known, maximum_delta, -1.0
            )
            flat_index = int(np.argmax(masked_delta))
            theta_index, distance_index, height_index = np.unravel_index(
                flat_index, maximum_delta.shape
            )
            change_atlas.append(
                {
                    "anchor_manifest_id": rows[anchor_index]["id"],
                    "theta_index": int(theta_index),
                    "distance_index": int(distance_index),
                    "height_band": ("foot", "body", "head")[
                        height_index
                    ],
                    "maximum_jointly_known_future_delta": float(
                        maximum_delta[
                            theta_index, distance_index, height_index
                        ]
                    ),
                }
            )

    gates = protocol["gates"]
    coverage = {
        name: _coverage_fraction(horizon_known[name], current_required)
        for name in ("current", "near", "far")
    }
    height_fraction = (
        height_numerator / height_denominator
        if height_denominator
        else 0.0
    )
    future_fraction = (
        future_numerator / future_denominator
        if future_denominator
        else 0.0
    )
    checks = {
        "minimum_usable_anchors": len(usable)
        >= int(gates["minimum_usable_anchors_per_session"]),
        "single_multi_consistency": consistency_error
        <= float(gates["maximum_single_multi_consistency_error"]),
        "current_known_coverage": coverage["current"]
        >= float(gates["minimum_current_known_coverage_per_session"]),
        "near_known_coverage": coverage["near"]
        >= float(gates["minimum_near_known_coverage_per_session"]),
        "far_known_coverage": coverage["far"]
        >= float(gates["minimum_far_known_coverage_per_session"]),
        "multi_height_nonredundancy": height_fraction
        >= float(gates["minimum_height_disagreement_fraction_per_session"]),
        "future_nonredundancy": future_fraction
        >= float(gates["minimum_future_union_change_fraction_per_session"]),
    }
    result.update(
        {
            "frame_count": len(rows),
            "usable_anchor_count": len(usable),
            "missing_near_anchor_count": sum(
                index is None for index in near_indices
            ),
            "missing_far_anchor_count": sum(
                index is None for index in far_indices
            ),
            "required_cells_per_horizon": current_required,
            "frozen_denominators": denominators,
            "known_cells": horizon_known,
            "unknown_cells": {
                name: current_required - horizon_known[name]
                for name in ("current", "near", "far")
            },
            "known_coverage": coverage,
            "single_multi_max_consistency_error": consistency_error,
            "height_disagreement": {
                "numerator": height_numerator,
                "denominator": height_denominator,
                "fraction": height_fraction,
            },
            "future_union_change": {
                "numerator": future_numerator,
                "denominator": future_denominator,
                "fraction": future_fraction,
            },
            "dynamic_support_point_count": dynamic_support,
            "occlusion_unknown_atlas": atlas,
            "unknown_cell_atlas": unknown_cell_atlas[:30],
            "dynamic_cell_atlas": sorted(
                dynamic_cell_atlas,
                key=lambda item: (
                    -item["dynamic_support_point_count"],
                    item["anchor_manifest_id"],
                    item["theta_index"],
                    item["distance_index"],
                    item["height_band"],
                ),
            )[:30],
            "largest_future_change_atlas": sorted(
                change_atlas,
                key=lambda item: item[
                    "maximum_jointly_known_future_delta"
                ],
                reverse=True,
            )[:10],
            "checks": checks,
            "mechanics_valid": all(
                checks[name]
                for name in (
                    "minimum_usable_anchors",
                    "single_multi_consistency",
                    "current_known_coverage",
                    "near_known_coverage",
                    "far_known_coverage",
                )
            ),
            "multi_height_supported": checks[
                "multi_height_nonredundancy"
            ],
            "future_supported": checks["future_nonredundancy"],
            "ok": all(checks.values()),
        }
    )
    return result


def _decide_terminal(sessions: list[dict[str, Any]]) -> str:
    if not sessions or any(
        not session.get("authority_validation", {}).get("ok")
        or not session.get("mechanics_valid")
        for session in sessions
    ):
        return "H1_GEOMETRY_TEACHER_NOT_EVALUABLE"
    if any(not session.get("multi_height_supported") for session in sessions):
        return "H1_MULTI_HEIGHT_PROXY_NOT_SUPPORTED_STOP"
    if any(not session.get("future_supported") for session in sessions):
        return "H1_FUTURE_PROXY_NOT_SUPPORTED_STOP"
    return "GEOMETRY_PROXY_MECHANISM_SUPPORTED"


def run(
    protocol_path: Path,
    session_inputs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    protocol_contract = PROTOCOL_RESULT_CONTRACTS.get(
        str(protocol.get("schema"))
    )
    if (
        protocol_contract is None
        or protocol.get("status") != "FROZEN_RESULT_NOT_RUN"
    ):
        raise ValueError("H1 protocol is not a supported frozen contract")
    _theta_edges(protocol["field"])
    required_count = int(protocol["required_session_count"])
    if len(session_inputs) != required_count:
        raise ValueError(
            f"Expected exactly {required_count} session inputs"
        )
    sessions: list[dict[str, Any]] = []
    for replay, authority in session_inputs:
        try:
            sessions.append(
                _session_result(replay, authority, protocol)
            )
        except (OSError, ValueError, KeyError) as error:
            session_id = None
            try:
                rows = _load_jsonl(
                    replay.resolve() / "manifest.replay.jsonl"
                )
                if rows:
                    session_id = rows[0].get("session_id")
            except (OSError, ValueError, KeyError):
                pass
            sessions.append(
                {
                    "source_session_id": session_id,
                    "replay_root": str(replay.resolve()),
                    "authority_report_path": str(authority.resolve()),
                    "authority_validation": {
                        "ok": False,
                        "error": str(error),
                    },
                    "mechanics_valid": False,
                    "multi_height_supported": False,
                    "future_supported": False,
                    "ok": False,
                }
            )
    ids = [
        str(session["source_session_id"])
        for session in sessions
        if session.get("source_session_id")
    ]
    independent = len(ids) == len(set(ids))
    expected_ids = {
        item["source_session_id"]
        for item in protocol["required_sessions"]
    }
    exact_session_set = set(ids) == expected_ids and len(ids) == len(
        sessions
    )
    terminal = (
        _decide_terminal(sessions)
        if independent and exact_session_set
        else "H1_GEOMETRY_TEACHER_NOT_EVALUABLE"
    )
    return {
        "schema": protocol_contract["result_schema"],
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": protocol_contract["claim_ceiling"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "implementation_dependency_hashes": {
            "pose_geometry_authority_module_sha256": _sha256(
                Path(__file__).resolve().parent
                / "verify_sanpo_pose_geometry_authority.py"
            )
        },
        "source_session_count": len(sessions),
        "unique_source_session_count": len(set(ids)),
        "parent_units_are_independent": independent,
        "exact_frozen_session_set": exact_session_set,
        "sessions": sessions,
        "worst_session": {
            "current_known_coverage": min(
                (
                    session.get("known_coverage", {}).get("current", 0.0)
                    for session in sessions
                ),
                default=0.0,
            ),
            "near_known_coverage": min(
                (
                    session.get("known_coverage", {}).get("near", 0.0)
                    for session in sessions
                ),
                default=0.0,
            ),
            "far_known_coverage": min(
                (
                    session.get("known_coverage", {}).get("far", 0.0)
                    for session in sessions
                ),
                default=0.0,
            ),
            "height_disagreement_fraction": min(
                (
                    session.get("height_disagreement", {}).get(
                        "fraction", 0.0
                    )
                    for session in sessions
                ),
                default=0.0,
            ),
            "future_union_change_fraction": min(
                (
                    session.get("future_union_change", {}).get(
                        "fraction", 0.0
                    )
                    for session in sessions
                ),
                default=0.0,
            ),
        },
        "allowed_next_step": (
            "FREEZE_H2_CAUSAL_STUDENT_PROTOCOL"
            if terminal == "GEOMETRY_PROXY_MECHANISM_SUPPORTED"
            else "STOP_OR_REFORMULATE_FAILED_H1_EVIDENCE_VERSION"
        ),
        "h2_automatically_authorized": False,
        "mainline_changed": False,
        "default_app_changed": False,
        "prohibited_inferences": protocol["prohibited_claims"],
    }


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--session",
        action="append",
        nargs=2,
        metavar=("REPLAY_ROOT", "AUTHORITY_REPORT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(
            args.protocol.resolve(),
            [
                (Path(replay).resolve(), Path(authority).resolve())
                for replay, authority in args.session
            ],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "allowed_next_step": report["allowed_next_step"],
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
