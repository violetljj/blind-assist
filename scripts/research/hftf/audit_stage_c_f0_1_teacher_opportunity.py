#!/usr/bin/env python3
"""Audit frozen F0.1 body/head teacher opportunity without a corpus."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_stage_c_f0_1_sanpo_authority import (
    READY as AUTHORITY_COHORT_READY,
    SCHEMA as AUTHORITY_COHORT_SCHEMA,
)
from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (
    _advected_basis,
    _anchor_basis,
    _causal_tangent_velocity,
    _obstacle_points_world,
    _pose,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
    _sha256,
    _theta_edges,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_c_f0_1_teacher_opportunity_audit"
READY = "F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS"
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE"
)
EXECUTION_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_teacher_opportunity_execution_contract_f0_1"
)
EXECUTION_SHA256 = (
    "29e449f729942bfea8919d93bb404360829c42681cdc7d8f8fc86559c99b79b6"
)
F0_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_body_head_temporal_student_canary_f0"
)
F0_SHA256 = (
    "0ba70780352534a79c420f46821b139aceafb67d10f769ced8fd57b8bfeb986d"
)
F0_1_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_cross_split_body_head_temporal_student_canary_f0_1"
)
F0_1_SHA256 = (
    "9bfcd253c70320c398923eec18561443b90f969417631a42748584d684b1b21e"
)
MECHANICS_SCHEMA = (
    "blindassist_hftf_stage_b_swept_envelope_label_mechanics_canary_d0"
)
MECHANICS_SHA256 = (
    "a69d25d77f1e2b72f407980f005c758b965517fd032562a009f91746ea1e0e6a"
)
SOURCE_LOCK_SCHEMA = "blindassist_hftf_stage_c_f0_1_sanpo_source_lock"
SOURCE_LOCK_SHA256 = (
    "f7353779315757b8b4ca5ba13b3544c4348c25f2ac4daa4befe47ad80fc79f62"
)
ACQUISITION_SHA256 = (
    "c4ca200a486b61b8f16ffb530c40224008f40c2fa3c83ead1077f7dfd1a1681a"
)
AUTHORITY_COHORT_SHA256 = (
    "d2ce7d4eaf73cb5e8a85b4cbbff46aa4882be90d5651b8c359f1c72b8bd803db"
)
HEIGHTS = ("body", "head")
HORIZONS = ("current", "future")
VIEWS = ("candidate", "reference")


def _root_name(source: dict[str, Any]) -> str:
    return (
        f"hftf-f0-1-{source['role']}-{source['official_split']}-"
        f"{str(source['session_id'])[:8]}-25frames-20260801"
    )


def _timeline_contract(target_fps: float) -> dict[str, Any]:
    if target_fps not in (5.0, 10.0):
        raise ValueError("F0.1 target timeline fps must be 5 or 10")
    history_offsets = [
        -int(round(seconds * target_fps))
        for seconds in (0.8, 0.6, 0.4, 0.2, 0.0)
    ]
    future_offset = int(round(0.4 * target_fps))
    velocity_history_offset = -future_offset
    first = -history_offsets[0]
    last = 24 - future_offset
    return {
        "target_fps": target_fps,
        "history_offsets": history_offsets,
        "velocity_history_offset": velocity_history_offset,
        "future_offset": future_offset,
        "usable_anchor_indices": list(range(first, last + 1)),
    }


def _pixel_lattices_disjoint(width: int, height: int) -> bool:
    candidate_x = np.arange(4, width, 8)
    candidate_y = np.arange(4, height, 8)
    reference_x = np.arange(2, width, 4)
    reference_y = np.arange(2, height, 4)
    return (
        np.intersect1d(candidate_x, reference_x).size == 0
        and np.intersect1d(candidate_y, reference_y).size == 0
    )


def _causal_future_basis(
    history_binding: dict[str, Any],
    anchor_binding: dict[str, Any],
    anchor_plane: dict[str, Any],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]:
    current = _anchor_basis(anchor_binding, anchor_plane)
    velocity = _causal_tangent_velocity(
        history_binding, anchor_binding, 0.4, current[3]
    )
    future = _advected_basis(current, velocity, 400)
    return current, future, velocity


def _probe_passes(
    probes_world: np.ndarray,
    observation_row: dict[str, Any],
    observation_binding: dict[str, Any],
    camera: dict[str, Any],
    depth: np.ndarray,
    semantic: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    width, height = int(observation_row["width"]), int(
        observation_row["height"]
    )
    if depth.shape != (height, width) or semantic.shape != (height, width):
        raise ValueError("Observation arrays disagree with manifest dimensions")
    translation, rotation = _pose(observation_binding)
    cell_count = probes_world.shape[0]
    if probes_world.shape[1:] != (3, 9):
        raise ValueError("Exactly nine 3D probes are required per cell")
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
    observed_depth = np.zeros(z.shape, dtype=np.float64)
    observed_semantic = np.zeros(z.shape, dtype=np.int64)
    observed_depth[inside] = depth[v[inside], u[inside]]
    observed_semantic[inside] = semantic[v[inside], u[inside]]
    passing = (
        inside
        & np.isfinite(observed_depth)
        & (observed_depth > 0.0)
        & (observed_depth + tolerance_m >= z)
        & (observed_semantic != 0)
    )
    return passing.reshape(cell_count, 9)


def _union_support(
    anchor_counts: np.ndarray,
    future_counts: np.ndarray,
    anchor_probe_passes: np.ndarray,
    future_probe_passes: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if anchor_counts.shape != shape or future_counts.shape != shape:
        raise ValueError("Support tensors disagree with frozen field shape")
    cell_count = math.prod(shape)
    if (
        anchor_probe_passes.shape != (cell_count, 9)
        or future_probe_passes.shape != (cell_count, 9)
    ):
        raise ValueError("Probe tensors disagree with frozen field shape")
    counts = np.maximum(anchor_counts, future_counts)
    known = (
        np.logical_or(anchor_probe_passes, future_probe_passes).sum(axis=1)
        >= 5
    ).reshape(shape)
    return counts, known


def _target_counts(
    known: np.ndarray,
    support: np.ndarray,
    *,
    safe_assignment: np.ndarray | None = None,
) -> dict[str, int]:
    if known.shape != support.shape:
        raise ValueError("Known and support tensors must have equal shapes")
    positive = known & (support >= 2)
    negative = known & (support < 2)
    safe = negative if safe_assignment is None else safe_assignment
    if safe.shape != known.shape:
        raise ValueError("Safe assignment tensor must match known")
    return {
        "known": int(known.sum()),
        "positive_known": int(positive.sum()),
        "negative_known": int(negative.sum()),
        "unknown": int((~known).sum()),
        "unknown_to_safe_violations": int(((~known) & safe).sum()),
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _next_step_authorization(passed: bool) -> dict[str, bool]:
    return {
        "train_candidate_corpus_materialization_authorized": passed,
        "dev_reference_target_materialization_authorized": passed,
        "heldout_training_corpus_materialization_authorized": False,
        "heldout_reference_target_materialization_authorized_before_frozen_checkpoint": False,
        "student_training_authorized_before_corpus_validation": False,
    }


def _validate_inputs(
    execution_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    acquisition_path: Path,
    authority_cohort_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    expected = (
        (execution_path, EXECUTION_SHA256),
        (f0_path, F0_SHA256),
        (f0_1_path, F0_1_SHA256),
        (mechanics_path, MECHANICS_SHA256),
        (source_lock_path, SOURCE_LOCK_SHA256),
        (acquisition_path, ACQUISITION_SHA256),
        (authority_cohort_path, AUTHORITY_COHORT_SHA256),
    )
    failures = [
        str(path)
        for path, digest in expected
        if _sha256(path) != digest
    ]
    if failures:
        raise ValueError(
            f"Frozen F0.1 input hash mismatch: {','.join(failures)}"
        )
    execution = _load_json(execution_path)
    f0 = _load_json(f0_path)
    f0_1 = _load_json(f0_1_path)
    mechanics = _load_json(mechanics_path)
    source_lock = _load_json(source_lock_path)
    acquisition = _load_json(acquisition_path)
    authority_cohort = _load_json(authority_cohort_path)
    if (
        execution.get("schema") != EXECUTION_SCHEMA
        or execution.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_TEACHER_GEOMETRY_OUTCOME"
        or f0.get("schema") != F0_SCHEMA
        or f0.get("status") != "FROZEN_BEFORE_F0_SOURCE_OUTCOME"
        or f0_1.get("schema") != F0_1_SCHEMA
        or f0_1.get("status") != "FROZEN_BEFORE_F0_1_SOURCE_OUTCOME"
        or mechanics.get("schema") != MECHANICS_SCHEMA
        or source_lock.get("schema") != SOURCE_LOCK_SCHEMA
        or authority_cohort.get("schema") != AUTHORITY_COHORT_SCHEMA
        or authority_cohort.get("terminal") != AUTHORITY_COHORT_READY
        or authority_cohort.get("all_sources_authority_ready") is not True
        or authority_cohort.get("allowed_next_step")
        != "F0_1_TEACHER_OPPORTUNITY_AUDIT"
        or acquisition.get("all_sources_ok") is not True
    ):
        raise ValueError("Frozen F0.1 input contract mismatch")
    if (
        f0_1.get(
            "all_field_causal_teacher_unknown_student_training_and_effect_contracts_inherited_exactly_from_f0"
        )
        is not True
        or source_lock.get("teacher_label_or_corpus_authorized") is not False
        or source_lock.get("student_training_authorized") is not False
        or authority_cohort.get("authorization", {}).get(
            "teacher_geometry_opportunity_audit_authorized"
        )
        is not True
        or authority_cohort.get("authorization", {}).get(
            "teacher_label_or_corpus_materialization_authorized"
        )
        is not False
    ):
        raise ValueError("F0.1 authorization boundary mismatch")
    return f0, mechanics, source_lock, authority_cohort


def _source_metrics(
    source: dict[str, Any],
    cohort_source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    f0: dict[str, Any],
    mechanics: dict[str, Any],
) -> dict[str, Any]:
    root = (datasets_root / _root_name(source)).resolve()
    authority_path = (
        authority_root / str(source["session_id"])[:8] / "authority.json"
    ).resolve()
    rows = _load_jsonl(root / "manifest.replay.jsonl")
    spec = _load_json(root / "dataset_spec.json")
    authority = _load_json(authority_path)
    session_id = str(source["session_id"])
    if (
        len(rows) != 25
        or {str(row.get("session_id")) for row in rows} != {session_id}
        or _sha256(root / "manifest.replay.jsonl")
        != cohort_source["manifest_sha256"]
        or _sha256(root / "dataset_spec.json")
        != cohort_source["dataset_spec_sha256"]
        or _sha256(root / "source_metadata/camera_poses.csv")
        != cohort_source["camera_poses_sha256"]
        or _sha256(authority_path)
        != cohort_source["authority_report_sha256"]
    ):
        raise ValueError(f"{session_id}: cohort input binding mismatch")
    bindings = authority["source_pose_authority"]["bindings"]
    binding_by_id = {
        str(item["manifest_id"]): item for item in bindings
    }
    plane_by_id = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    if set(binding_by_id) != {str(row["id"]) for row in rows} or set(
        plane_by_id
    ) != {str(row["id"]) for row in rows}:
        raise ValueError(f"{session_id}: incomplete authority geometry")
    timeline = _timeline_contract(float(source["target_fps"]))
    source_frames = [int(row["source_frame_index"]) for row in rows]
    timestamps = [int(row["source_timestamp_ms"]) for row in rows]
    if (
        source_frames != source["selected_source_frames"]
        or timestamps
        != [
            round(frame * 1000 / float(source["source_fps"]))
            for frame in source_frames
        ]
    ):
        raise ValueError(f"{session_id}: physical timeline mismatch")
    camera = spec["camera"]
    width, height = int(camera["image_width"]), int(camera["image_height"])
    if not _pixel_lattices_disjoint(width, height):
        raise ValueError(f"{session_id}: teacher pixel lattices overlap")
    field = f0["field_contract"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(field["distance_edges_m"], dtype=np.float64)
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][name])
        for name in HEIGHTS
    ]
    widths = np.asarray(
        [
            mechanics["standard_synthetic_envelope"][
                "effective_lateral_half_width_m"
            ][name]
            for name in HEIGHTS
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    field_shape = (6, 6, 2)
    required_indices = sorted(
        {
            index
            for anchor in timeline["usable_anchor_indices"]
            for index in (anchor, anchor + timeline["future_offset"])
        }
    )

    @functools.lru_cache(maxsize=8)
    def observation(index: int) -> tuple[np.ndarray, np.ndarray]:
        row = rows[index]
        depth = _read_depth(
            _resolve_inside(root, str(row["source_depth_path"])),
            int(row["width"]),
            int(row["height"]),
        )
        semantic = _read_semantic_class(
            _resolve_inside(root, str(row["source_mask_path"])),
            int(row["width"]),
            int(row["height"]),
        )
        return depth, semantic

    points: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        view: {} for view in VIEWS
    }
    view_sampling = {
        "candidate": (8, 4),
        "reference": (4, 2),
    }
    for index in required_indices:
        depth, semantic = observation(index)
        row = rows[index]
        binding = binding_by_id[str(row["id"])]
        for view, (stride, offset) in view_sampling.items():
            points[view][index] = _obstacle_points_world(
                root,
                row,
                binding,
                camera,
                stride=stride,
                offset=offset,
                excluded_classes=set(
                    obstacle["excluded_semantic_class_ids"]
                ),
                dynamic_classes=set(
                    obstacle["dynamic_provenance_class_ids"]
                ),
                depth_override=depth,
                semantic_override=semantic,
            )
    observation.cache_clear()
    accumulators = {
        view: {
            horizon: {
                height_name: {
                    "known": 0,
                    "positive_known": 0,
                    "negative_known": 0,
                    "unknown": 0,
                    "unknown_to_safe_violations": 0,
                }
                for height_name in HEIGHTS
            }
            for horizon in HORIZONS
        }
        for view in VIEWS
    }
    tangent_speeds: list[float] = []
    predicted_to_observed_errors: list[float] = []
    for anchor_index in timeline["usable_anchor_indices"]:
        history_index = anchor_index + timeline["velocity_history_offset"]
        future_index = anchor_index + timeline["future_offset"]
        anchor_row = rows[anchor_index]
        history_row = rows[history_index]
        future_row = rows[future_index]
        anchor_binding = binding_by_id[str(anchor_row["id"])]
        future_binding = binding_by_id[str(future_row["id"])]
        current_basis, future_basis, velocity = _causal_future_basis(
            binding_by_id[str(history_row["id"])],
            anchor_binding,
            plane_by_id[str(anchor_row["id"])],
        )
        tangent_speeds.append(float(np.linalg.norm(velocity)))
        future_ground = np.asarray(
            plane_by_id[str(future_row["id"])]["camera_ground_projection_m"],
            dtype=np.float64,
        )
        predicted_to_observed_errors.append(
            float(np.linalg.norm(future_basis[0] - future_ground))
        )
        current_probes = _swept_prism_probes_world(
            current_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        future_probes = _swept_prism_probes_world(
            future_basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        anchor_depth, anchor_semantic = observation(anchor_index)
        future_depth, future_semantic = observation(future_index)
        current_passing = _probe_passes(
            current_probes,
            anchor_row,
            anchor_binding,
            camera,
            anchor_depth,
            anchor_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        future_anchor_passing = _probe_passes(
            future_probes,
            anchor_row,
            anchor_binding,
            camera,
            anchor_depth,
            anchor_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        future_observation_passing = _probe_passes(
            future_probes,
            future_row,
            future_binding,
            camera,
            future_depth,
            future_semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        current_known = (current_passing.sum(axis=1) >= 5).reshape(
            field_shape
        )
        for view in VIEWS:
            anchor_points, anchor_dynamic = points[view][anchor_index]
            future_points, future_dynamic = points[view][future_index]
            current_support, _ = _swept_prism_counts(
                anchor_points,
                anchor_dynamic,
                current_basis,
                theta_edges,
                distance_edges,
                height_bands,
                widths,
            )
            anchor_future_support, _ = _swept_prism_counts(
                anchor_points,
                anchor_dynamic,
                future_basis,
                theta_edges,
                distance_edges,
                height_bands,
                widths,
            )
            future_observation_support, _ = _swept_prism_counts(
                future_points,
                future_dynamic,
                future_basis,
                theta_edges,
                distance_edges,
                height_bands,
                widths,
            )
            future_support, future_known = _union_support(
                anchor_future_support,
                future_observation_support,
                future_anchor_passing,
                future_observation_passing,
                field_shape,
            )
            for horizon, known, support in (
                ("current", current_known, current_support),
                ("future", future_known, future_support),
            ):
                for height_index, height_name in enumerate(HEIGHTS):
                    values = _target_counts(
                        known[:, :, height_index],
                        support[:, :, height_index],
                    )
                    for key, value in values.items():
                        accumulators[view][horizon][height_name][key] += value
    denominator = len(timeline["usable_anchor_indices"]) * 36
    views: dict[str, Any] = {}
    for view in VIEWS:
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            heights: dict[str, Any] = {}
            for height_name in HEIGHTS:
                counts = accumulators[view][horizon][height_name]
                heights[height_name] = {
                    **counts,
                    "denominator": denominator,
                    "known_coverage": counts["known"] / denominator,
                }
            horizons[horizon] = heights
        views[view] = {"horizons": horizons}
    role_view = "candidate" if source["role"] == "train" else "reference"
    gate_metrics = views[role_view]["horizons"]
    checks = {
        "known_coverage_each_height_each_horizon": all(
            gate_metrics[horizon][height_name]["known_coverage"] >= 0.1
            for horizon in HORIZONS
            for height_name in HEIGHTS
        ),
        "future_positive_known_each_height": all(
            gate_metrics["future"][height_name]["positive_known"] >= 5
            for height_name in HEIGHTS
        ),
        "future_negative_known_each_height": all(
            gate_metrics["future"][height_name]["negative_known"] >= 20
            for height_name in HEIGHTS
        ),
        "zero_unknown_to_safe_violations": all(
            gate_metrics[horizon][height_name][
                "unknown_to_safe_violations"
            ]
            == 0
            for horizon in HORIZONS
            for height_name in HEIGHTS
        ),
    }
    return {
        "role": source["role"],
        "official_split": source["official_split"],
        "session_id": session_id,
        "source_fps": source["source_fps"],
        "target_fps": source["target_fps"],
        "manifest_sha256": cohort_source["manifest_sha256"],
        "dataset_spec_sha256": cohort_source["dataset_spec_sha256"],
        "camera_poses_sha256": cohort_source["camera_poses_sha256"],
        "authority_report_sha256": cohort_source["authority_report_sha256"],
        "timeline": timeline,
        "usable_anchor_count": len(timeline["usable_anchor_indices"]),
        "frozen_denominator_per_height_per_horizon": denominator,
        "teacher_views": views,
        "role_gate_view": role_view,
        "checks": checks,
        "passed": all(checks.values()),
        "causal_diagnostics": {
            "history_only_tangent_speed_mps": {
                "minimum": min(tangent_speeds),
                "median": float(np.median(tangent_speeds)),
                "maximum": max(tangent_speeds),
            },
            "predicted_to_observed_future_ground_origin_error_m": {
                "minimum": min(predicted_to_observed_errors),
                "median": float(np.median(predicted_to_observed_errors)),
                "maximum": max(predicted_to_observed_errors),
            },
            "diagnostic_only_not_a_gate": True,
        },
    }


def _structural_canaries() -> dict[str, bool]:
    plane = {
        "camera_ground_projection_m": [0.0, 0.0, 0.0],
        "normal_toward_camera": [0.0, 0.0, 1.0],
    }
    history = {
        "position_m": [-0.4, 0.0, 1.3],
        "quaternion_xyzw": [
            0.0,
            math.sqrt(0.5),
            0.0,
            math.sqrt(0.5),
        ],
    }
    anchor = {
        "position_m": [0.0, 0.0, 1.3],
        "quaternion_xyzw": [
            0.0,
            math.sqrt(0.5),
            0.0,
            math.sqrt(0.5),
        ],
    }
    current, future, _ = _causal_future_basis(history, anchor, plane)
    perturbed_future_pose = {
        "position_m": [100.0, -50.0, 20.0],
        "quaternion_xyzw": [0.5, 0.5, 0.5, 0.5],
    }
    future_pose_finite = all(
        math.isfinite(float(value))
        for value in perturbed_future_pose["position_m"]
        + perturbed_future_pose["quaternion_xyzw"]
    )
    counts_a = np.asarray([[[1], [2]]])
    counts_b = np.asarray([[[1], [0]]])
    passes_a = np.zeros((2, 9), dtype=bool)
    passes_b = np.zeros((2, 9), dtype=bool)
    passes_a[0, :3] = True
    passes_b[0, 3:5] = True
    union_counts, union_known = _union_support(
        counts_a, counts_b, passes_a, passes_b, (1, 2, 1)
    )
    return {
        "five_fps_exact_anchor_count": len(
            _timeline_contract(5.0)["usable_anchor_indices"]
        )
        == 19,
        "ten_fps_exact_anchor_count": len(
            _timeline_contract(10.0)["usable_anchor_indices"]
        )
        == 13,
        "candidate_reference_pixel_lattices_disjoint": (
            _pixel_lattices_disjoint(2208, 1242)
        ),
        "future_pose_is_not_an_input_to_causal_basis": (
            future_pose_finite
            and np.allclose(current[1], future[1])
            and np.allclose(current[2], future[2])
            and np.allclose(current[3], future[3])
            and np.allclose(future[0], [0.4, 0.0, 0.0])
        ),
        "future_observation_union_uses_max_not_addition": (
            np.array_equal(union_counts, np.asarray([[[1], [2]]]))
        ),
        "future_known_unions_probes_before_five_of_nine": bool(
            union_known[0, 0, 0]
        ),
        "unknown_never_defaults_to_safe": (
            _target_counts(
                np.asarray([False, True]),
                np.asarray([0, 0]),
            )["unknown_to_safe_violations"]
            == 0
        ),
    }


def _payload(
    execution_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    acquisition_path: Path,
    authority_cohort_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> dict[str, Any]:
    f0, mechanics, source_lock, authority_cohort = _validate_inputs(
        execution_path,
        f0_path,
        f0_1_path,
        mechanics_path,
        source_lock_path,
        acquisition_path,
        authority_cohort_path,
    )
    sources = source_lock["sources"]
    cohort_sources = authority_cohort["sources"]
    if [item["session_id"] for item in sources] != [
        item["session_id"] for item in cohort_sources
    ]:
        raise ValueError("F0.1 source order differs from authority cohort")
    canaries = _structural_canaries()
    source_results = [
        _source_metrics(
            source,
            cohort_source,
            datasets_root,
            authority_root,
            f0,
            mechanics,
        )
        for source, cohort_source in zip(
            sources, cohort_sources, strict=True
        )
    ]
    role_positive_source_counts = {
        role: {
            height_name: sum(
                result["role"] == role
                and result["teacher_views"][result["role_gate_view"]][
                    "horizons"
                ]["future"][height_name]["positive_known"]
                >= 1
                for result in source_results
            )
            for height_name in HEIGHTS
        }
        for role in ("train", "dev", "heldout")
    }
    role_gate = all(
        role_positive_source_counts[role][height_name] >= 2
        for role in role_positive_source_counts
        for height_name in HEIGHTS
    )
    source_gate = all(result["passed"] for result in source_results)
    pre_determinism_pass = (
        all(canaries.values()) and source_gate and role_gate
    )
    return {
        "schema": SCHEMA,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "SANPO_SYNTHETIC_BODY_HEAD_TEACHER_OPPORTUNITY_ONLY",
        "claim_ceiling": "SYNTHETIC_BODY_HEAD_GEOMETRY_PROXY_ONLY",
        "execution_contract_path": str(execution_path.resolve()),
        "execution_contract_sha256": _sha256(execution_path),
        "f0_protocol_path": str(f0_path.resolve()),
        "f0_protocol_sha256": _sha256(f0_path),
        "f0_1_protocol_path": str(f0_1_path.resolve()),
        "f0_1_protocol_sha256": _sha256(f0_1_path),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": _sha256(source_lock_path),
        "acquisition_audit_path": str(acquisition_path.resolve()),
        "acquisition_audit_sha256": _sha256(acquisition_path),
        "authority_cohort_path": str(authority_cohort_path.resolve()),
        "authority_cohort_sha256": _sha256(authority_cohort_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "source_count": len(source_results),
        "role_counts": {
            role: sum(result["role"] == role for result in source_results)
            for role in ("train", "dev", "heldout")
        },
        "structural_canaries": canaries,
        "source_results": source_results,
        "role_positive_future_source_counts_by_height": (
            role_positive_source_counts
        ),
        "pre_determinism_gate": {
            "all_structural_canaries_pass": all(canaries.values()),
            "all_source_gates_pass": source_gate,
            "every_role_has_two_positive_future_sources_each_height": role_gate,
            "passed": pre_determinism_pass,
        },
        "firewall": {
            "teacher_summary_counts_only": True,
            "teacher_cell_corpus_materialized": False,
            "rgb_student_input_read": False,
            "student_training_or_output_computed": False,
            "heldout_student_output_read": False,
            "heldout_used_for_threshold_checkpoint_augmentation_or_source_selection": False,
            "future_pose_used_to_select_origin_direction_anchor_or_sample": False,
            "foot_ground_label_computed": False,
            "unknown_defaults_to_safe": False,
            "unknown_safe_check_is_constructive_mask_invariant": True,
            "research_mainline_changed": False,
            "default_app_changed": False,
        },
        "prohibited_inferences": [
            "teacher proxy is human collision or safety truth",
            "teacher opportunity proves student effect",
            "body and head success completes HFTF",
            "research mainline promotion",
            "Android or production authorization",
        ],
    }


def run(
    execution_path: Path,
    f0_path: Path,
    f0_1_path: Path,
    mechanics_path: Path,
    source_lock_path: Path,
    acquisition_path: Path,
    authority_cohort_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> dict[str, Any]:
    args = (
        execution_path,
        f0_path,
        f0_1_path,
        mechanics_path,
        source_lock_path,
        acquisition_path,
        authority_cohort_path,
        datasets_root,
        authority_root,
    )
    first = _payload(*args)
    second = _payload(*args)
    deterministic = _canonical_bytes(first) == _canonical_bytes(second)
    passed = first["pre_determinism_gate"]["passed"] and deterministic
    first["determinism_check"] = {
        "canonical_payload_byte_exact": deterministic,
        "canonical_serialization": (
            "utf8_json_sort_keys_true_separators_comma_colon_allow_nan_false"
        ),
    }
    first["terminal"] = READY if passed else NOT_EVALUABLE
    first["next_step_authorization"] = _next_step_authorization(passed)
    return first


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--f0-protocol", type=Path, required=True)
    parser.add_argument("--f0-1-protocol", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--acquisition-audit", type=Path, required=True)
    parser.add_argument("--authority-cohort", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(
            args.execution_contract.resolve(),
            args.f0_protocol.resolve(),
            args.f0_1_protocol.resolve(),
            args.mechanics_protocol.resolve(),
            args.source_lock.resolve(),
            args.acquisition_audit.resolve(),
            args.authority_cohort.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "source_count": report["source_count"],
                    "pre_determinism_gate": report[
                        "pre_determinism_gate"
                    ],
                    "determinism_check": report["determinism_check"],
                    "role_positive_future_source_counts_by_height": report[
                        "role_positive_future_source_counts_by_height"
                    ],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"] == READY else 3
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
