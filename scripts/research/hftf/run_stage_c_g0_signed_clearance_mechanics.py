#!/usr/bin/env python3
"""Audit signed-clearance teacher mechanics on consumed F0.1 sources."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from audit_stage_c_f0_1_teacher_opportunity import (
    _pixel_lattices_disjoint,
    _probe_passes,
    _root_name,
)
from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from plan_stage_c_f0_sanpo_inventory import _sha256
from run_geometry_teacher_canary import (
    _anchor_basis,
    _obstacle_points_world,
    _read_depth,
    _read_semantic_class,
    _resolve_inside,
    _theta_edges,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_F0_1_STOP_BEFORE_G0_CLEARANCE_OR_SOURCE_SCAN_OUTCOME"
)
SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_mechanics_result"
)
SUPPORTED = (
    "G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY"
)
NOT_SUPPORTED = "G0_SIGNED_CLEARANCE_MECHANICS_NOT_SUPPORTED_STOP"
NOT_EVALUABLE = "G0_SIGNED_CLEARANCE_MECHANICS_NOT_EVALUABLE"
HEIGHTS = ("body", "head")


def _resolve_parent(
    protocol_path: Path, receipt: dict[str, Any]
) -> Path:
    raw = Path(str(receipt["path"]))
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (protocol_path.parents[3] / raw).resolve()
    return (protocol_path.parent / raw).resolve()


def _parent(
    protocol_path: Path,
    protocol: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = protocol["parents"][key]
    path = _resolve_parent(protocol_path, receipt)
    if _sha256(path) != str(receipt["sha256"]):
        raise ValueError(f"G0 mechanics parent hash mismatch: {key}")
    return path, _load_json(path)


def _trajectory_directions(
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    return np.cos(centers), np.sin(centers)


def _mask_unknown_targets(
    clipped_target: np.ndarray,
    known: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if clipped_target.shape != known.shape or known.dtype != np.bool_:
        raise ValueError("Known-mask and clearance-target shape mismatch")
    nullable_target = np.full(clipped_target.shape, None, dtype=object)
    nullable_safe = np.full(clipped_target.shape, None, dtype=object)
    nullable_target[known] = clipped_target[known]
    nullable_safe[known] = clipped_target[known] >= 0.0
    return nullable_target, nullable_safe


def _box_support_equivalent_clearance(
    along: np.ndarray,
    cross: np.ndarray,
    height: np.ndarray,
    inside: np.ndarray,
    *,
    distance_lower: float,
    distance_upper: float,
    height_lower: float,
    height_upper: float,
    half_width: float,
) -> np.ndarray:
    center = np.asarray(
        [
            (distance_lower + distance_upper) / 2.0,
            0.0,
            (height_lower + height_upper) / 2.0,
        ],
        dtype=np.float64,
    )
    half_extent = np.asarray(
        [
            (distance_upper - distance_lower) / 2.0,
            half_width,
            (height_upper - height_lower) / 2.0,
        ],
        dtype=np.float64,
    )
    coordinates = np.stack((along, cross, height), axis=1)
    excess = np.abs(coordinates - center[None, :]) - half_extent[None, :]
    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
    inside_distance = np.minimum(np.max(excess, axis=1), 0.0)
    closed_box_sdf = outside + inside_distance
    magnitude = np.abs(closed_box_sdf)
    negative_tie = np.nextafter(
        np.float64(0.0), np.float64(-np.inf)
    )
    positive_tie = np.nextafter(
        np.float64(0.0), np.float64(np.inf)
    )
    proxy = np.where(inside, -magnitude, magnitude)
    proxy = np.where(inside & (magnitude == 0.0), negative_tie, proxy)
    proxy = np.where((~inside) & (magnitude == 0.0), positive_tie, proxy)
    return proxy


def _signed_clearance_field(
    points_world: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    theta_edges: np.ndarray,
    distance_edges: np.ndarray,
    height_bands: list[tuple[float, float]],
    lateral_half_widths: np.ndarray,
    *,
    order_statistic: int,
    final_edge_atol_m: float,
    final_edge_rtol: float,
    clip_min_m: float,
    clip_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        points_world.ndim != 2
        or points_world.shape[0] != 3
        or not np.isfinite(points_world).all()
    ):
        raise ValueError("Signed-clearance points must be finite 3xN")
    if order_statistic <= 0:
        raise ValueError("Signed-clearance order statistic must be positive")
    if final_edge_atol_m != 1e-12 or final_edge_rtol != 0.0:
        raise ValueError("Signed-clearance final-edge tolerance drifted")
    if not clip_min_m < 0.0 < clip_max_m:
        raise ValueError("Signed-clearance clip interval is invalid")
    if lateral_half_widths.shape != (len(height_bands),):
        raise ValueError("Signed-clearance height width count mismatch")
    origin, forward, right, up = basis
    relative = points_world - origin[:, None]
    forward_coordinate = forward @ relative
    right_coordinate = right @ relative
    height_coordinate = up @ relative
    cosine, sine = _trajectory_directions(theta_edges)
    shape = (
        len(theta_edges) - 1,
        len(distance_edges) - 1,
        len(height_bands),
    )
    raw = np.full(shape, np.inf, dtype=np.float64)
    inside_counts = np.zeros(shape, dtype=np.int64)
    for theta_index in range(shape[0]):
        along = (
            forward_coordinate * cosine[theta_index]
            + right_coordinate * sine[theta_index]
        )
        cross = (
            -forward_coordinate * sine[theta_index]
            + right_coordinate * cosine[theta_index]
        )
        distance_index_for_point = np.searchsorted(
            distance_edges, along, side="right"
        ) - 1
        distance_index_for_point = np.where(
            np.isclose(
                along,
                distance_edges[-1],
                atol=final_edge_atol_m,
                rtol=final_edge_rtol,
            ),
            shape[1] - 1,
            distance_index_for_point,
        )
        for distance_index in range(shape[1]):
            lower = float(distance_edges[distance_index])
            upper = float(distance_edges[distance_index + 1])
            for height_index, (height_lower, height_upper) in enumerate(
                height_bands
            ):
                upper_ok = (
                    height_coordinate <= height_upper
                    if height_index == shape[2] - 1
                    else height_coordinate < height_upper
                )
                inside = (
                    (distance_index_for_point == distance_index)
                    & (
                        np.abs(cross)
                        <= float(lateral_half_widths[height_index])
                    )
                    & (height_coordinate >= height_lower)
                    & upper_ok
                )
                signed = _box_support_equivalent_clearance(
                    along,
                    cross,
                    height_coordinate,
                    inside,
                    distance_lower=lower,
                    distance_upper=upper,
                    height_lower=float(height_lower),
                    height_upper=float(height_upper),
                    half_width=float(lateral_half_widths[height_index]),
                )
                inside_counts[
                    theta_index, distance_index, height_index
                ] = int(inside.sum())
                if len(signed) >= order_statistic:
                    raw[theta_index, distance_index, height_index] = float(
                        np.partition(signed, order_statistic - 1)[
                            order_statistic - 1
                        ]
                    )
    clipped = np.clip(raw, clip_min_m, clip_max_m)
    return raw, clipped, inside_counts


def _structural_canaries(protocol: dict[str, Any]) -> dict[str, bool]:
    field = protocol["field_contract"]
    clearance = protocol["signed_clearance_contract"]
    theta_edges = np.radians(np.asarray([-15.0, 15.0]))
    distance_edges = np.asarray([0.0, 2.0])
    bands = [(0.35, 1.35), (1.35, 2.05)]
    basis = (
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    )
    kwargs = {
        "order_statistic": int(clearance["order_statistic"]),
        "final_edge_atol_m": float(
            clearance["final_distance_edge_isclose"]["atol_m"]
        ),
        "final_edge_rtol": float(
            clearance["final_distance_edge_isclose"]["rtol"]
        ),
        "clip_min_m": float(clearance["raw_clearance_clip_m"][0]),
        "clip_max_m": float(clearance["raw_clearance_clip_m"][1]),
    }

    def result(points: list[list[float]], widths: list[float]) -> np.ndarray:
        _, clipped, _ = _signed_clearance_field(
            np.asarray(points, dtype=np.float64).T,
            basis,
            theta_edges,
            distance_edges,
            bands,
            np.asarray(widths, dtype=np.float64),
            **kwargs,
        )
        return clipped

    two_body = result(
        [[1.0, 0.0, 0.8], [1.2, 0.1, 0.9]], [0.4, 0.28]
    )
    one_body = result([[1.0, 0.0, 0.8]], [0.4, 0.28])
    lateral = [[1.0, 0.35, 0.8], [1.2, 0.36, 0.9]]
    narrow = result(lateral, [0.2, 0.2])
    wide = result(lateral, [0.4, 0.4])
    far = result(
        [[1.0, 0.7, 0.8], [1.2, 0.75, 0.9]], [0.4, 0.28]
    )
    near = result(
        [[1.0, 0.45, 0.8], [1.2, 0.46, 0.9]], [0.4, 0.28]
    )
    body_only = result(
        [[1.0, 0.0, 0.8], [1.2, 0.1, 0.9]], [0.4, 0.28]
    )
    head_only = result(
        [[1.0, 0.0, 1.7], [1.2, 0.1, 1.8]], [0.4, 0.28]
    )
    known = np.asarray([False, True])
    derived: list[bool | None] = [
        bool(value < 0.0) if is_known else None
        for value, is_known in zip(
            [float(two_body[0, 0, 0]), float(two_body[0, 0, 0])],
            known,
            strict=True,
        )
    ]
    return {
        "second_order_requires_two_inside_points": bool(
            two_body[0, 0, 0] < 0.0 and one_body[0, 0, 0] > 0.0
        ),
        "widening_envelope_never_increases_clearance": bool(
            wide[0, 0, 0] <= narrow[0, 0, 0]
        ),
        "moving_obstacle_toward_prism_never_increases_clearance": bool(
            near[0, 0, 0] <= far[0, 0, 0]
        ),
        "height_specific_obstacle_changes_only_matching_layer": bool(
            body_only[0, 0, 0] < 0.0
            and body_only[0, 0, 1] > 0.0
            and head_only[0, 0, 0] > 0.0
            and head_only[0, 0, 1] < 0.0
        ),
        "unknown_never_derives_safe": derived == [None, True],
        "protocol_widths_match_structural_widths": (
            field["effective_lateral_half_width_m"]
            == {"body": 0.4, "head": 0.28}
        ),
    }


def _source_result(
    source: dict[str, Any],
    cohort_source: dict[str, Any],
    datasets_root: Path,
    authority_root: Path,
    protocol: dict[str, Any],
    mechanics: dict[str, Any],
) -> dict[str, Any]:
    root = (datasets_root / _root_name(source)).resolve()
    session_id = str(source["session_id"])
    authority_path = (
        authority_root / session_id[:8] / "authority.json"
    ).resolve()
    rows = _load_jsonl(root / "manifest.replay.jsonl")
    spec = _load_json(root / "dataset_spec.json")
    authority = _load_json(authority_path)
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
        raise ValueError(f"{session_id}: G0 cohort binding mismatch")
    bindings = {
        str(item["manifest_id"]): item
        for item in authority["source_pose_authority"]["bindings"]
    }
    planes = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
    }
    if (
        set(bindings) != {str(row["id"]) for row in rows}
        or set(planes) != {str(row["id"]) for row in rows}
    ):
        raise ValueError(f"{session_id}: G0 authority coverage mismatch")
    camera = spec["camera"]
    if not _pixel_lattices_disjoint(
        int(camera["image_width"]), int(camera["image_height"])
    ):
        raise ValueError(f"{session_id}: G0 teacher lattices overlap")
    field = protocol["field_contract"]
    clearance_contract = protocol["signed_clearance_contract"]
    theta_edges = _theta_edges(
        {
            "theta_bin_count": field["theta_bin_count"],
            "theta_range_degrees": field["theta_range_degrees"],
        }
    )
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][height])
        for height in HEIGHTS
    ]
    widths = np.asarray(
        [
            field["effective_lateral_half_width_m"][height]
            for height in HEIGHTS
        ],
        dtype=np.float64,
    )
    obstacle_contract = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    view = field["teacher_view_for_every_role"]
    aggregates = {
        height: {
            "known": 0,
            "positive_known": 0,
            "negative_known": 0,
            "unknown": 0,
            "unknown_nonnull_target_violations": 0,
            "unknown_to_safe_violations": 0,
            "binary_equivalence_violations": 0,
            "known_nonfinite_clipped_target": 0,
            "known_preclip_infinite": 0,
            "known_exact_tie_values": 0,
            "known_near_boundary": 0,
            "risk_clip_min": 0,
            "risk_not_clip_min": 0,
            "safe_clip_max": 0,
            "safe_not_clip_max": 0,
            "known_preclip_finite_values": [],
            "known_clipped_values": [],
        }
        for height in HEIGHTS
    }
    for row in rows:
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
        binding = bindings[str(row["id"])]
        basis = _anchor_basis(binding, planes[str(row["id"])])
        points, dynamic = _obstacle_points_world(
            root,
            row,
            binding,
            camera,
            stride=int(view["point_sample_stride_xy"]),
            offset=int(view["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle_contract["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle_contract["dynamic_provenance_class_ids"]
            ),
            depth_override=depth,
            semantic_override=semantic,
        )
        probes = _swept_prism_probes_world(
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        passing = _probe_passes(
            probes,
            row,
            binding,
            camera,
            depth,
            semantic,
            float(known_contract["depth_front_tolerance_m"]),
        )
        known = (passing.sum(axis=1) >= 5).reshape((6, 6, 2))
        support, _ = _swept_prism_counts(
            points,
            dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        raw, clipped, inside_counts = _signed_clearance_field(
            points,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
            order_statistic=int(clearance_contract["order_statistic"]),
            final_edge_atol_m=float(
                clearance_contract["final_distance_edge_isclose"][
                    "atol_m"
                ]
            ),
            final_edge_rtol=float(
                clearance_contract["final_distance_edge_isclose"]["rtol"]
            ),
            clip_min_m=float(
                clearance_contract["raw_clearance_clip_m"][0]
            ),
            clip_max_m=float(
                clearance_contract["raw_clearance_clip_m"][1]
            ),
        )
        if not np.array_equal(support, inside_counts):
            raise ValueError(f"{session_id}: support/inside count mismatch")
        derived = clipped < 0.0
        target = support >= 2
        for height_index, height in enumerate(HEIGHTS):
            mask = known[:, :, height_index]
            values = clipped[:, :, height_index][mask]
            raw_values = raw[:, :, height_index][mask]
            metrics = aggregates[height]
            metrics["known"] += int(mask.sum())
            metrics["positive_known"] += int(
                (target[:, :, height_index] & mask).sum()
            )
            metrics["negative_known"] += int(
                ((~target[:, :, height_index]) & mask).sum()
            )
            metrics["unknown"] += int((~mask).sum())
            nullable_target, nullable_safe = _mask_unknown_targets(
                clipped[:, :, height_index], mask
            )
            metrics["unknown_nonnull_target_violations"] += sum(
                value is not None for value in nullable_target[~mask]
            )
            metrics["unknown_to_safe_violations"] += int(
                sum(value is True for value in nullable_safe[~mask])
            )
            metrics["binary_equivalence_violations"] += int(
                ((derived[:, :, height_index] != target[:, :, height_index])
                 & mask).sum()
            )
            metrics["known_nonfinite_clipped_target"] += int(
                (~np.isfinite(values)).sum()
            )
            metrics["known_preclip_infinite"] += int(
                (~np.isfinite(raw_values)).sum()
            )
            tie = np.nextafter(
                np.float64(0.0), np.float64(np.inf)
            )
            metrics["known_exact_tie_values"] += int(
                (np.abs(raw_values) == tie).sum()
            )
            metrics["known_near_boundary"] += int(
                (np.abs(values) <= 0.2).sum()
            )
            risk_mask = target[:, :, height_index] & mask
            safe_mask = (~target[:, :, height_index]) & mask
            clip_min = float(clearance_contract["raw_clearance_clip_m"][0])
            clip_max = float(clearance_contract["raw_clearance_clip_m"][1])
            metrics["risk_clip_min"] += int(
                ((clipped[:, :, height_index] == clip_min) & risk_mask).sum()
            )
            metrics["risk_not_clip_min"] += int(
                ((clipped[:, :, height_index] > clip_min) & risk_mask).sum()
            )
            metrics["safe_clip_max"] += int(
                ((clipped[:, :, height_index] == clip_max) & safe_mask).sum()
            )
            metrics["safe_not_clip_max"] += int(
                ((clipped[:, :, height_index] < clip_max) & safe_mask).sum()
            )
            metrics["known_preclip_finite_values"].extend(
                float(value) for value in raw_values[np.isfinite(raw_values)]
            )
            metrics["known_clipped_values"].extend(
                float(value) for value in values
            )
    source_metrics: dict[str, Any] = {}
    gates = protocol["g0_d0_consumed_mechanics_canary"]["data_gates"]
    for height, values in aggregates.items():
        known_count = int(values["known"])
        distinct_millimeter_bins = len(
            {
                int(np.rint(float(value) * 1000.0))
                for value in values["known_clipped_values"]
            }
        )
        risk_count = int(values["positive_known"])
        safe_count = int(values["negative_known"])
        source_metrics[height] = {
            key: int(values[key])
            for key in (
                "known",
                "positive_known",
                "negative_known",
                "unknown",
                "unknown_nonnull_target_violations",
                "unknown_to_safe_violations",
                "binary_equivalence_violations",
                "known_nonfinite_clipped_target",
                "known_preclip_infinite",
                "known_exact_tie_values",
                "known_near_boundary",
                "risk_clip_min",
                "risk_not_clip_min",
                "safe_clip_max",
                "safe_not_clip_max",
            )
        }
        source_metrics[height].update(
            {
                "distinct_clipped_target_millimeter_bins": (
                    distinct_millimeter_bins
                ),
                "risk_clip_min_fraction": (
                    values["risk_clip_min"] / risk_count
                    if risk_count
                    else 1.0
                ),
                "safe_clip_max_fraction": (
                    values["safe_clip_max"] / safe_count
                    if safe_count
                    else 1.0
                ),
                "preclip_finite_minimum_m": min(
                    values["known_preclip_finite_values"],
                    default=None,
                ),
                "preclip_finite_median_m": (
                    float(
                        np.median(values["known_preclip_finite_values"])
                    )
                    if values["known_preclip_finite_values"]
                    else None
                ),
                "preclip_finite_maximum_m": max(
                    values["known_preclip_finite_values"],
                    default=None,
                ),
                "passed": (
                    values["positive_known"] > 0
                    and values["negative_known"] > 0
                    and values["binary_equivalence_violations"] == 0
                    and values["known_nonfinite_clipped_target"] == 0
                    and values["unknown_nonnull_target_violations"] == 0
                    and values["unknown_to_safe_violations"] == 0
                    and distinct_millimeter_bins
                    >= int(
                        gates[
                            "each_source_height_distinct_clipped_"
                            "millimeter_bins_minimum"
                        ]
                    )
                    and values["known_near_boundary"]
                    >= int(
                        gates[
                            "each_source_height_known_near_boundary_"
                            "count_minimum"
                        ]
                    )
                    and values["risk_not_clip_min"] > 0
                    and values["safe_not_clip_max"] > 0
                ),
            }
        )
    return {
        "session_id": session_id,
        "prior_role": source["role"],
        "fresh_evidence_credit": False,
        "frame_count": len(rows),
        "teacher_view": "reference",
        "height_metrics": source_metrics,
        "passed": all(item["passed"] for item in source_metrics.values()),
    }


def run(
    protocol_path: Path,
    datasets_root: Path,
    authority_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("G0 mechanics protocol identity mismatch")
    implementation = protocol.get("implementations", {}).get(
        "signed_clearance_mechanics_runner", {}
    )
    if (
        Path(str(implementation.get("path", ""))).as_posix()
        != "scripts/research/hftf/"
        "run_stage_c_g0_signed_clearance_mechanics.py"
        or implementation.get("sha256")
        != _sha256(Path(__file__).resolve())
        or implementation.get("execution_authorized") is not True
    ):
        raise ValueError("G0 mechanics implementation receipt mismatch")
    repository_root = Path(__file__).resolve().parents[3]
    if (
        datasets_root.resolve()
        != (repository_root / "artifacts.local/evidence/datasets").resolve()
        or authority_root.resolve()
        != (
            repository_root
            / "artifacts.local/evidence/hftf/"
            "stage-c-f0-1-sanpo-authority-20260801"
        ).resolve()
    ):
        raise ValueError("G0 mechanics noncanonical input root")
    parents = {
        key: _parent(protocol_path, protocol, key)[1]
        for key in protocol["parents"]
    }
    result = parents["f0_1_heldout_effect_result"]
    source_lock = parents["f0_1_source_lock"]
    acquisition = parents["f0_1_acquisition_audit"]
    cohort = parents["f0_1_authority_cohort"]
    opportunity = parents["f0_1_teacher_opportunity"]
    mechanics = parents["swept_envelope_mechanics"]
    if (
        result.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_"
        "STUDENT_SIGNAL_NOT_SUPPORTED_STOP"
        or source_lock.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"
        or cohort.get("terminal")
        != "F0_1_SANPO_SOURCE_AUTHORITY_COHORT_READY"
        or acquisition.get("terminal")
        != "F0_1_SANPO_ACQUISITION_AND_TRANSPORT_READY"
        or opportunity.get("terminal")
        != "F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS"
        or len(source_lock.get("sources", [])) != 12
        or len(acquisition.get("sources", [])) != 12
        or len(cohort.get("sources", [])) != 12
        or len(opportunity.get("source_results", [])) != 12
    ):
        raise ValueError("G0 mechanics parent terminal mismatch")
    source_ids = [
        str(source["session_id"]) for source in source_lock["sources"]
    ]
    if (
        source_ids
        != [
            str(source["session_id"])
            for source in acquisition["sources"]
        ]
        or source_ids
        != [str(source["session_id"]) for source in cohort["sources"]]
        or source_ids
        != [
            str(source["session_id"])
            for source in opportunity["source_results"]
        ]
    ):
        raise ValueError("G0 mechanics outcome-open source order mismatch")
    structural = _structural_canaries(protocol)
    cohort_by_id = {
        str(item["session_id"]): item for item in cohort["sources"]
    }
    sources = [
        _source_result(
            source,
            cohort_by_id[str(source["session_id"])],
            datasets_root,
            authority_root,
            protocol,
            mechanics,
        )
        for source in source_lock["sources"]
    ]
    checks = {
        "all_structural_canaries_pass": all(structural.values()),
        "all_12_sources_decode_and_bind": len(sources) == 12,
        "all_sources_all_heights_pass": all(
            source["passed"] for source in sources
        ),
        "zero_binary_equivalence_violations": all(
            metrics["binary_equivalence_violations"] == 0
            for source in sources
            for metrics in source["height_metrics"].values()
        ),
        "zero_known_nonfinite_clipped_target": all(
            metrics["known_nonfinite_clipped_target"] == 0
            for source in sources
            for metrics in source["height_metrics"].values()
        ),
        "zero_unknown_to_safe_violations": all(
            metrics["unknown_to_safe_violations"] == 0
            for source in sources
            for metrics in source["height_metrics"].values()
        ),
        "all_unknown_targets_remain_null": all(
            metrics["unknown_nonnull_target_violations"] == 0
            for source in sources
            for metrics in source["height_metrics"].values()
        ),
    }
    terminal = SUPPORTED if all(checks.values()) else NOT_SUPPORTED
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "CONSUMED_SOURCE_MECHANICS_ONLY",
        "claim_ceiling": "SYNTHETIC_SIGNED_CLEARANCE_PROXY_MECHANICS",
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "structural_canaries": structural,
        "source_count": len(sources),
        "frame_count": sum(source["frame_count"] for source in sources),
        "sources": sources,
        "checks": checks,
        "fresh_evidence_credit": False,
        "student_output_computed": False,
        "authorization": {
            "g0_source_plan_may_be_used": True,
            "fresh_evaluation_acquisition_contract_may_be_frozen": (
                terminal == SUPPORTED
            ),
            "fresh_evaluation_acquisition_executed": False,
            "student_training_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def _require_output(path: Path) -> Path:
    expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-signed-clearance-mechanics-20260801/result.json"
    ).resolve()
    if path.resolve() != expected:
        raise ValueError("G0 mechanics output path is not canonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_output(args.output)
        if output.exists():
            raise FileExistsError("Refusing to overwrite G0 mechanics result")
        report = run(
            args.protocol.resolve(),
            args.datasets_root.resolve(),
            args.authority_root.resolve(),
        )
        output.parent.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(
            tempfile.mkdtemp(
                prefix=f"{output.parent.name}.partial-",
                dir=output.parent.parent,
            )
        )
        with (partial / "result.json").open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if output.parent.exists():
            raise FileExistsError("G0 mechanics output root appeared")
        partial.replace(output.parent)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "source_count": report["source_count"],
                    "frame_count": report["frame_count"],
                    "result_sha256": _sha256(output),
                }
            )
        )
        return 0 if report["terminal"] == SUPPORTED else 2
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
