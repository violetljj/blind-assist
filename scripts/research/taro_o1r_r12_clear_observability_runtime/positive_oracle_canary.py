#!/usr/bin/env python3
"""Pose-first TARO positive-occupancy observability canary on consumed Bonn RGB-D.

The source cohort and reference frames are selected from timestamps and camera poses
before any depth payload is opened.  Every non-static arm receives exactly one extra
frame.  Outputs are OCCUPIED or UNKNOWN only; CLEAR is label-side evidence and
UNKNOWN is never used as a negative.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary


SCHEMA = "blindassist.taro.task_directed_observability_positive_oracle_canary.v2"
MODE = "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT"
NATIVE_SIZE_WH = (640, 480)
LOW_SIZE_WH = (256, 192)
BONN_INTRINSICS = np.asarray(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DEPTH_SCALE = 5000.0
MAX_RGB_DEPTH_DELTA_S = 0.02
MAX_POSE_BRACKET_S = 0.08
PASSIVE_MIN_GAP_S = 0.15
PASSIVE_MAX_GAP_S = 1.0
MICRO_TRANSLATION_RANGE_M = (0.04, 0.08)
MICRO_TARGET_TRANSLATION_M = 0.06
MICRO_MAX_ROTATION_DEG = 5.0
MIN_REFERENCE_SEPARATION_S = 0.5
MAX_REFERENCES_PER_PARENT = 12
MIN_CAPABILITY_PARENTS = 4
MIN_SELECTED_REFERENCES = 48
MIN_EVALUATED_REFERENCES = 48
MIN_RECOVERY_OPPORTUNITY_PARENTS = 4
MIN_CLEAR_DENOMINATOR_PARENTS = 4
LOCAL_STABILITY_RANGE_M = 0.08
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
BONN_T_ROS = np.asarray(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
BONN_T_MARKER = np.asarray(
    [
        [1.0157, 0.1828, -0.2389, 0.0113],
        [0.0009, -0.8431, -0.6413, -0.0098],
        [-0.3009, 0.6147, -0.8085, 0.0111],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
BONN_CALIBRATION_URL = "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/"
ARM_NAMES = ("static_r7", "passive", "fixed_micro", "generic_max_parallax", "task_directed_oracle")


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _parse_index(path: Path, columns: int) -> list[list[str]]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    require(rows and all(len(row) == columns for row in rows), f"invalid index: {path}")
    times = [float(row[0]) for row in rows]
    require(all(right > left for left, right in zip(times, times[1:])), f"non-monotonic index: {path}")
    return rows


def _normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    require(quaternion.shape == (4,) and norm > 0.0 and np.all(np.isfinite(quaternion)), "invalid quaternion")
    return quaternion / norm


def _slerp(left: Sequence[float], right: Sequence[float], alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + alpha * (q1 - q0))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return _normalize_quaternion(
        math.sin((1.0 - alpha) * angle) / sine * q0
        + math.sin(alpha * angle) / sine * q1
    )


def _quaternion_matrix_xyzw(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(translation: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_matrix_xyzw(quaternion_xyzw)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _interpolate_pose(poses: Sequence[Sequence[str]], pose_times: Sequence[float], timestamp: float) -> np.ndarray | None:
    index = bisect.bisect_left(pose_times, timestamp)
    if index == 0 or index == len(poses):
        return None
    left, right = poses[index - 1], poses[index]
    left_time, right_time = float(left[0]), float(right[0])
    if timestamp - left_time > MAX_POSE_BRACKET_S or right_time - timestamp > MAX_POSE_BRACKET_S:
        return None
    alpha = (timestamp - left_time) / (right_time - left_time)
    translation = (1.0 - alpha) * np.asarray(left[1:4], dtype=np.float64) + alpha * np.asarray(right[1:4], dtype=np.float64)
    quaternion = _slerp([float(item) for item in left[4:8]], [float(item) for item in right[4:8]], alpha)
    raw_pose = _pose_matrix(translation, quaternion)
    # The official Bonn evaluation page documents both a ROS-frame bug transform
    # and the RGB-D-sensor-to-marker calibration.  Raw groundtruth quaternions are
    # therefore not optical-camera poses until this exact chain is applied.
    return np.ascontiguousarray(BONN_T_ROS @ raw_pose @ BONN_T_ROS @ BONN_T_MARKER, dtype=np.float64)


def _associate_unique_nearest(
    primary: Sequence[Sequence[str]],
    secondary: Sequence[Sequence[str]],
    maximum_delta_s: float,
) -> list[tuple[Sequence[str], Sequence[str]]]:
    secondary_times = [float(row[0]) for row in secondary]
    used: set[int] = set()
    output: list[tuple[Sequence[str], Sequence[str]]] = []
    for row in primary:
        timestamp = float(row[0])
        index = bisect.bisect_left(secondary_times, timestamp)
        candidates = [item for item in (index - 1, index) if 0 <= item < len(secondary)]
        available = [item for item in candidates if item not in used]
        if not available:
            continue
        selected = min(available, key=lambda item: (abs(secondary_times[item] - timestamp), item))
        if abs(secondary_times[selected] - timestamp) <= maximum_delta_s:
            used.add(selected)
            output.append((row, secondary[selected]))
    return output


@dataclass(frozen=True)
class Frame:
    parent_id: str
    timestamp_s: float
    rgb_path: Path
    depth_path: Path
    camera_to_world: np.ndarray

    @property
    def frame_id(self) -> str:
        return f"{self.parent_id}:{self.timestamp_s:.5f}"


@dataclass(frozen=True)
class Pair:
    reference: Frame
    neighbor: Frame
    gap_s: float
    translation_m: float
    rotation_deg: float


@dataclass(frozen=True)
class ReferenceSupport:
    reference: Frame
    candidates: tuple[Pair, ...]
    micro_candidates: tuple[Pair, ...]


def _pair(reference: Frame, neighbor: Frame) -> Pair:
    relative = np.linalg.inv(reference.camera_to_world) @ neighbor.camera_to_world
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = min(1.0, max(-1.0, (float(np.trace(relative[:3, :3])) - 1.0) / 2.0))
    rotation = math.degrees(math.acos(cosine))
    return Pair(reference, neighbor, reference.timestamp_s - neighbor.timestamp_s, translation, rotation)


def load_parent_frames(sequence_root: Path) -> tuple[list[Frame], dict[str, Any]]:
    rgb_path = sequence_root / "rgb.txt"
    depth_path = sequence_root / "depth.txt"
    pose_path = sequence_root / "groundtruth.txt"
    require(all(path.is_file() for path in (rgb_path, depth_path, pose_path)), f"Bonn sequence incomplete: {sequence_root}")
    rgb_rows = _parse_index(rgb_path, 2)
    depth_rows = _parse_index(depth_path, 2)
    pose_rows = _parse_index(pose_path, 8)
    pose_times = [float(row[0]) for row in pose_rows]
    frames: list[Frame] = []
    for rgb, depth in _associate_unique_nearest(rgb_rows, depth_rows, MAX_RGB_DEPTH_DELTA_S):
        timestamp = float(rgb[0])
        pose = _interpolate_pose(pose_rows, pose_times, timestamp)
        if pose is None:
            continue
        rgb_file = sequence_root / rgb[1]
        depth_file = sequence_root / depth[1]
        if rgb_file.is_file() and depth_file.is_file():
            frames.append(Frame(sequence_root.name, timestamp, rgb_file, depth_file, pose))
    return frames, {
        "parent_id": sequence_root.name,
        "rgb_index_sha256": sha256_file(rgb_path),
        "depth_index_sha256": sha256_file(depth_path),
        "groundtruth_sha256": sha256_file(pose_path),
        "rgb_index_count": len(rgb_rows),
        "depth_index_count": len(depth_rows),
        "pose_index_count": len(pose_rows),
        "associated_pose_valid_frame_count": len(frames),
    }


def build_reference_support(frames: Sequence[Frame]) -> list[ReferenceSupport]:
    output: list[ReferenceSupport] = []
    start = 0
    for index, reference in enumerate(frames):
        while start < index and reference.timestamp_s - frames[start].timestamp_s > PASSIVE_MAX_GAP_S:
            start += 1
        candidates = tuple(
            _pair(reference, neighbor)
            for neighbor in frames[start:index]
            if PASSIVE_MIN_GAP_S <= reference.timestamp_s - neighbor.timestamp_s <= PASSIVE_MAX_GAP_S
        )
        micro = tuple(
            pair
            for pair in candidates
            if MICRO_TRANSLATION_RANGE_M[0] <= pair.translation_m <= MICRO_TRANSLATION_RANGE_M[1]
            and pair.rotation_deg <= MICRO_MAX_ROTATION_DEG
        )
        if candidates and micro:
            output.append(ReferenceSupport(reference, candidates, micro))
    return output


def select_references(rows: Sequence[ReferenceSupport], limit: int = MAX_REFERENCES_PER_PARENT) -> list[ReferenceSupport]:
    thinned: list[ReferenceSupport] = []
    for row in rows:
        if thinned and row.reference.timestamp_s - thinned[-1].reference.timestamp_s < MIN_REFERENCE_SEPARATION_S:
            continue
        thinned.append(row)
    if len(thinned) <= limit:
        return thinned
    # Spread references over the complete pose-eligible sequence.  This remains
    # outcome-blind and avoids turning the sequence prefix into an accidental
    # scene-content selector.
    indices = np.linspace(0, len(thinned) - 1, num=limit, dtype=np.int64)
    return [thinned[int(index)] for index in indices]


def audit_capability(dataset_root: Path, limit: int = MAX_REFERENCES_PER_PARENT) -> tuple[dict[str, Any], list[ReferenceSupport]]:
    sequence_roots = sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "rgb.txt").is_file() and (path / "depth.txt").is_file() and (path / "groundtruth.txt").is_file()
    )
    require(sequence_roots, f"no Bonn sequences found: {dataset_root}")
    parent_rows: list[dict[str, Any]] = []
    selected: list[ReferenceSupport] = []
    for root in sequence_roots:
        frames, receipt = load_parent_frames(root)
        supports = build_reference_support(frames)
        parent_selected = select_references(supports, limit)
        selected.extend(parent_selected)
        parent_rows.append(
            {
                **receipt,
                "legal_reference_count": len(supports),
                "legal_pair_count": sum(len(row.candidates) for row in supports),
                "micro_pair_count": sum(len(row.micro_candidates) for row in supports),
                "selected_reference_count": len(parent_selected),
            }
        )
    eligible_parents = sum(row["selected_reference_count"] > 0 for row in parent_rows)
    identity_rows = [
        {
            "reference_frame_id": row.reference.frame_id,
            "candidate_frame_ids": [pair.neighbor.frame_id for pair in row.candidates],
            "micro_candidate_frame_ids": [pair.neighbor.frame_id for pair in row.micro_candidates],
        }
        for row in selected
    ]
    passed = eligible_parents >= MIN_CAPABILITY_PARENTS and len(selected) >= MIN_SELECTED_REFERENCES
    return {
        "source_family": "BONN_RGBD_DYNAMIC",
        "analysis_role": "PROJECT_CONSUMED_DEVELOPMENT",
        "selection_inputs": ["rgb.txt", "depth.txt timestamps and paths", "groundtruth.txt pose"],
        "pose_coordinate_chain": {
            "formula": "T_world_camera = T_ROS * T_groundtruth * T_ROS * T_marker",
            "official_calibration_url": BONN_CALIBRATION_URL,
            "t_ros": BONN_T_ROS.tolist(),
            "t_marker": BONN_T_MARKER.tolist(),
        },
        "selection_reads_task_outcome": False,
        "image_payload_reads_during_selection": 0,
        "parent_count": len(parent_rows),
        "eligible_parent_count": eligible_parents,
        "selected_reference_count": len(selected),
        "minimum_capability_parent_count": MIN_CAPABILITY_PARENTS,
        "minimum_selected_reference_count": MIN_SELECTED_REFERENCES,
        "passive_window_s": [PASSIVE_MIN_GAP_S, PASSIVE_MAX_GAP_S],
        "micro_translation_range_m": list(MICRO_TRANSLATION_RANGE_M),
        "micro_max_rotation_deg": MICRO_MAX_ROTATION_DEG,
        "maximum_references_per_parent": limit,
        "parents": parent_rows,
        "selection_identity_sha256": hashlib.sha256(canonical_json_bytes(identity_rows)).hexdigest().upper(),
        "decision": "POSE_PAIR_CAPABILITY_PASS" if passed else "NOT_EVALUABLE_DATA_OBSERVABILITY",
    }, selected


def _scaled_intrinsics(matrix: np.ndarray, source_wh: tuple[int, int], target_wh: tuple[int, int]) -> np.ndarray:
    scale_x = target_wh[0] / source_wh[0]
    scale_y = target_wh[1] / source_wh[1]
    output = np.asarray(matrix, dtype=np.float64).copy()
    output[0, 0] *= scale_x
    output[1, 1] *= scale_y
    output[0, 2] = (output[0, 2] + 0.5) * scale_x - 0.5
    output[1, 2] = (output[1, 2] + 0.5) * scale_y - 0.5
    return output


LOW_INTRINSICS = _scaled_intrinsics(BONN_INTRINSICS, NATIVE_SIZE_WH, LOW_SIZE_WH)


@lru_cache(maxsize=96)
def _load_depth(path_text: str) -> np.ndarray:
    raw = cv2.imread(path_text, cv2.IMREAD_UNCHANGED)
    require(raw is not None and raw.shape == (NATIVE_SIZE_WH[1], NATIVE_SIZE_WH[0]) and raw.dtype == np.uint16, f"invalid Bonn depth: {path_text}")
    return np.ascontiguousarray(raw.astype(np.float64) / DEPTH_SCALE, dtype=np.float64)


def _low_observation(depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    low = cv2.resize(depth_m, LOW_SIZE_WH, interpolation=cv2.INTER_NEAREST)
    valid = (low >= adapter.DEPTH_RANGE_M[0]) & (low <= adapter.DEPTH_RANGE_M[1])
    maximum = ndimage.maximum_filter(np.where(valid, low, -np.inf), size=3, mode="constant", cval=-np.inf)
    minimum = ndimage.minimum_filter(np.where(valid, low, np.inf), size=3, mode="constant", cval=np.inf)
    stable = valid & np.isfinite(maximum) & np.isfinite(minimum) & ((maximum - minimum) <= LOCAL_STABILITY_RANGE_M)
    rows, columns = np.indices(low.shape, dtype=np.float64)
    points = np.stack(
        (
            (columns - LOW_INTRINSICS[0, 2]) * low / LOW_INTRINSICS[0, 0],
            (rows - LOW_INTRINSICS[1, 2]) * low / LOW_INTRINSICS[1, 1],
            low,
        ),
        axis=-1,
    )
    return np.ascontiguousarray(low), np.ascontiguousarray(points), np.ascontiguousarray(stable), float(np.mean(stable))


def _gravity_up_camera(frame: Frame) -> np.ndarray:
    return adapter._normalize_vector(frame.camera_to_world[:3, :3].T @ WORLD_UP, "BONN_GRAVITY_INVALID")


def _queries(reference: Frame, low_depth: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    plane = prospective._fit_depth_plane(low_depth, LOW_INTRINSICS, _gravity_up_camera(reference))
    if not plane["evaluable"]:
        return None
    receipts = prospective._build_queries(
        reference.frame_id,
        hashlib.sha256(reference.frame_id.encode("utf-8")).hexdigest().upper(),
        round(reference.timestamp_s * 1_000_000_000),
        plane,
    )
    return receipts, plane


def _states(points: np.ndarray, valid: np.ndarray, queries: Sequence[Mapping[str, Any]]) -> tuple[bool, ...]:
    return tuple(bool(r7_canary._occupied_grid(points, valid, query)[0][0][2]) for query in queries)


def _transform_points(points: np.ndarray, reference: Frame, neighbor: Frame) -> np.ndarray:
    relative = np.linalg.inv(reference.camera_to_world) @ neighbor.camera_to_world
    flat = points.reshape(-1, 3)
    transformed = flat @ relative[:3, :3].T + relative[:3, 3]
    return np.ascontiguousarray(transformed.reshape(points.shape), dtype=np.float64)


def _labels(reference: Frame, depth_m: np.ndarray, queries: Sequence[Mapping[str, Any]]) -> tuple[str, ...] | None:
    plane = prospective._fit_depth_plane(depth_m, BONN_INTRINSICS, _gravity_up_camera(reference))
    if not plane["evaluable"]:
        return None
    geometry = prospective._build_geometry(depth_m, adapter.canonical_sha256(depth_m), BONN_INTRINSICS)
    return tuple(str(r7_canary._truth_query_label(geometry, plane, BONN_INTRINSICS, query)["state"]) for query in queries)


def _pair_states(
    row: ReferenceSupport,
    queries: Sequence[Mapping[str, Any]],
    static_states: Sequence[bool],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for pair in row.candidates:
        depth = _load_depth(str(pair.neighbor.depth_path))
        _low, points, valid, coverage = _low_observation(depth)
        transformed = _transform_points(points, row.reference, pair.neighbor)
        observed = _states(transformed, valid, queries)
        output.append(
            {
                "pair": pair,
                "coverage": coverage,
                "states": tuple(bool(left or right) for left, right in zip(static_states, observed, strict=True)),
            }
        )
    return output


def _arm_contribution(static: Sequence[bool], states: Sequence[bool], labels: Sequence[str]) -> tuple[int, int, int]:
    recovered = sum((not before) and after and label == "OCCUPIED_OBSERVED" for before, after, label in zip(static, states, labels, strict=True))
    false_occupied = sum(after and label == "CLEAR_OBSERVED" for after, label in zip(states, labels, strict=True))
    occupied = sum(after and label == "OCCUPIED_OBSERVED" for after, label in zip(states, labels, strict=True))
    return recovered, false_occupied, occupied


def select_arm_rows(
    row: ReferenceSupport,
    pair_rows: Sequence[Mapping[str, Any]],
    static: Sequence[bool],
    labels: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    by_id = {item["pair"].neighbor.frame_id: item for item in pair_rows}
    passive = max(pair_rows, key=lambda item: (float(item["coverage"]), -item["pair"].gap_s, item["pair"].neighbor.frame_id))
    micro_pair = min(row.micro_candidates, key=lambda item: (abs(item.translation_m - MICRO_TARGET_TRANSLATION_M), item.rotation_deg, item.gap_s, item.neighbor.frame_id))
    generic = max(pair_rows, key=lambda item: (item["pair"].translation_m, item["pair"].rotation_deg, -item["pair"].gap_s, item["pair"].neighbor.frame_id))
    def task_score(item: Mapping[str, Any]) -> tuple[int, int, int, float, str]:
        recovered, false_occupied, occupied = _arm_contribution(static, item["states"], labels)
        return (
            recovered,
            -false_occupied,
            occupied,
            -item["pair"].translation_m,
            item["pair"].neighbor.frame_id,
        )

    task = max(pair_rows, key=task_score)
    return {
        "passive": passive,
        "fixed_micro": by_id[micro_pair.neighbor.frame_id],
        "generic_max_parallax": generic,
        "task_directed_oracle": task,
    }


def _empty_counts() -> dict[str, int | float]:
    return {
        "query_count": 0,
        "truth_occupied": 0,
        "truth_clear": 0,
        "truth_unknown": 0,
        "predicted_occupied": 0,
        "true_occupied": 0,
        "false_occupied": 0,
        "recovered_occupied": 0,
        "static_unknown_occupied_opportunity": 0,
        "known_retention_failures": 0,
        "extra_frame_count": 0,
        "gap_s_sum": 0.0,
        "translation_m_sum": 0.0,
        "rotation_deg_sum": 0.0,
    }


def _accumulate(
    counts: dict[str, int | float],
    states: Sequence[bool],
    static: Sequence[bool],
    labels: Sequence[str],
    pair: Pair | None,
) -> None:
    counts["query_count"] += len(labels)
    for state, before, label in zip(states, static, labels, strict=True):
        counts["truth_occupied"] += label == "OCCUPIED_OBSERVED"
        counts["truth_clear"] += label == "CLEAR_OBSERVED"
        counts["truth_unknown"] += label == "UNKNOWN"
        counts["predicted_occupied"] += state
        counts["true_occupied"] += state and label == "OCCUPIED_OBSERVED"
        counts["false_occupied"] += state and label == "CLEAR_OBSERVED"
        counts["static_unknown_occupied_opportunity"] += (not before) and label == "OCCUPIED_OBSERVED"
        counts["recovered_occupied"] += (not before) and state and label == "OCCUPIED_OBSERVED"
        counts["known_retention_failures"] += before and not state
    if pair is not None:
        counts["extra_frame_count"] += 1
        counts["gap_s_sum"] += pair.gap_s
        counts["translation_m_sum"] += pair.translation_m
        counts["rotation_deg_sum"] += pair.rotation_deg


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else round(float(numerator) / float(denominator), 12)


def _finalize(counts: Mapping[str, int | float], parent_counts: Mapping[str, Mapping[str, int | float]]) -> dict[str, Any]:
    parent_recovery = [
        float(row["recovered_occupied"]) / float(row["static_unknown_occupied_opportunity"])
        for row in parent_counts.values()
        if row["static_unknown_occupied_opportunity"]
    ]
    parent_false = [
        float(row["false_occupied"]) / float(row["truth_clear"])
        for row in parent_counts.values()
        if row["truth_clear"]
    ]
    extra = int(counts["extra_frame_count"])
    return {
        **{key: (round(value, 12) if isinstance(value, float) else int(value)) for key, value in counts.items()},
        "occupied_recall": _ratio(counts["true_occupied"], counts["truth_occupied"]),
        "recovery_rate": _ratio(counts["recovered_occupied"], counts["static_unknown_occupied_opportunity"]),
        "false_occupied_rate": _ratio(counts["false_occupied"], counts["truth_clear"]),
        "unknown_output_count": int(counts["query_count"] - counts["predicted_occupied"]),
        "parent_macro_recovery_rate": None if not parent_recovery else round(float(np.mean(parent_recovery)), 12),
        "parent_macro_false_occupied_rate": None if not parent_false else round(float(np.mean(parent_false)), 12),
        "mean_extra_frame_gap_s": _ratio(counts["gap_s_sum"], extra),
        "mean_translation_m": _ratio(counts["translation_m_sum"], extra),
        "mean_rotation_deg": _ratio(counts["rotation_deg_sum"], extra),
        "parent_count_with_recovery_opportunity": len(parent_recovery),
        "parent_count_with_clear_denominator": len(parent_false),
    }


def evaluate(dataset_root: Path, limit: int = MAX_REFERENCES_PER_PARENT) -> dict[str, Any]:
    capability, selected = audit_capability(dataset_root, limit)
    require(capability["decision"] == "POSE_PAIR_CAPABILITY_PASS", capability["decision"])
    totals = {name: _empty_counts() for name in ARM_NAMES}
    per_parent: dict[str, dict[str, dict[str, int | float]]] = defaultdict(lambda: {name: _empty_counts() for name in ARM_NAMES})
    evaluated = 0
    geometry_abstentions = 0
    reference_receipts: list[dict[str, Any]] = []
    for row in selected:
        reference_depth = _load_depth(str(row.reference.depth_path))
        low_depth, points, valid, coverage = _low_observation(reference_depth)
        query_result = _queries(row.reference, low_depth)
        if query_result is None:
            geometry_abstentions += 1
            continue
        queries, _source_plane = query_result
        labels = _labels(row.reference, reference_depth, queries)
        if labels is None:
            geometry_abstentions += 1
            continue
        static = _states(points, valid, queries)
        pair_rows = _pair_states(row, queries, static)
        arms = select_arm_rows(row, pair_rows, static, labels)
        _accumulate(totals["static_r7"], static, static, labels, None)
        _accumulate(per_parent[row.reference.parent_id]["static_r7"], static, static, labels, None)
        selected_ids: dict[str, str] = {}
        for name, item in arms.items():
            _accumulate(totals[name], item["states"], static, labels, item["pair"])
            _accumulate(per_parent[row.reference.parent_id][name], item["states"], static, labels, item["pair"])
            selected_ids[name] = item["pair"].neighbor.frame_id
        reference_receipts.append(
            {
                "reference_frame_id": row.reference.frame_id,
                "source_valid_fraction": round(coverage, 12),
                "candidate_count": len(pair_rows),
                "selected_extra_frame_ids": selected_ids,
                "label_counts": {state: int(sum(item == state for item in labels)) for state in ("OCCUPIED_OBSERVED", "CLEAR_OBSERVED", "UNKNOWN")},
            }
        )
        evaluated += 1
    require(evaluated > 0, "NOT_EVALUABLE_GEOMETRY")
    metrics = {
        name: _finalize(totals[name], {parent: rows[name] for parent, rows in per_parent.items()})
        for name in ARM_NAMES
    }
    passive = metrics["passive"]["parent_macro_recovery_rate"] or 0.0
    micro = metrics["fixed_micro"]["parent_macro_recovery_rate"] or 0.0
    generic = metrics["generic_max_parallax"]["parent_macro_recovery_rate"] or 0.0
    task = metrics["task_directed_oracle"]["parent_macro_recovery_rate"] or 0.0
    passive_false = metrics["passive"]["parent_macro_false_occupied_rate"] or 0.0
    micro_false = metrics["fixed_micro"]["parent_macro_false_occupied_rate"] or 0.0
    task_false = metrics["task_directed_oracle"]["parent_macro_false_occupied_rate"] or 0.0
    generic_false = metrics["generic_max_parallax"]["parent_macro_false_occupied_rate"] or 0.0
    evaluability = {
        "minimum_evaluated_references": evaluated >= MIN_EVALUATED_REFERENCES,
        "minimum_recovery_opportunity_parents": metrics["static_r7"]["parent_count_with_recovery_opportunity"] >= MIN_RECOVERY_OPPORTUNITY_PARENTS,
        "minimum_clear_denominator_parents": metrics["static_r7"]["parent_count_with_clear_denominator"] >= MIN_CLEAR_DENOMINATOR_PARENTS,
    }
    terminal: str | None = None
    if not all(evaluability.values()):
        decisions: dict[str, bool | None] = {
            "passive_improves_static": None,
            "micro_improves_passive": None,
            "task_oracle_beats_passive_and_generic": None,
        }
        terminal = "NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR"
    else:
        decisions = {
            "passive_improves_static": passive > 0.0 and passive_false <= 0.05,
            "micro_improves_passive": micro > passive and micro_false <= passive_false + 0.01,
            "task_oracle_beats_passive_and_generic": task > max(passive, generic) and task_false <= max(passive_false, generic_false) + 0.01,
        }
    if terminal == "NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR":
        pass
    elif not decisions["passive_improves_static"]:
        terminal = "STOP_TEMPORAL_NOT_BETTER_THAN_STATIC"
    elif not decisions["micro_improves_passive"]:
        terminal = "STOP_ACTIVE_MICRO_NOT_BETTER_THAN_PASSIVE"
    elif not decisions["task_oracle_beats_passive_and_generic"]:
        terminal = "STOP_TASK_DIRECTION_NOT_BETTER_THAN_GENERIC"
    else:
        terminal = "TASK_DIRECTED_ORACLE_CANARY_PASS_LEARNED_SCORER_JUSTIFIED"
    require(terminal is not None, "terminal missing")
    result = {
        "schema": SCHEMA,
        "mode": MODE,
        "question": "Can one extra pose-valid observation recover truth-consistent positive occupancy better when selected for the task than by passive, fixed-micro, or generic-parallax rules?",
        "source_capability": capability,
        "source_observation": {
            "static_semantics": "R7 positive OCCUPIED-or-UNKNOWN reducer on 256x192 metric depth",
            "confidence_proxy": f"3x3 local metric-depth range <= {LOCAL_STABILITY_RANGE_M:.2f} m",
            "truth_role": "same-frame native Bonn registered depth, source-derived Development label",
            "truth_minimum_obstacle_pixels": r7_canary.MINIMUM_TRUTH_OBSTACLE_PIXELS,
            "clear_output_allowed": False,
            "unknown_is_negative": False,
        },
        "arm_budget": {
            "static_r7_extra_frames": 0,
            "all_other_arms_extra_frames_per_reference": 1,
            "task_oracle_uses_label_only_for_oracle_selection": True,
            "passive_selection": "maximum stable-depth coverage, then most recent",
            "fixed_micro_selection": "closest to 0.06 m inside 0.04-0.08 m and <=5 deg",
            "generic_selection": "maximum pose translation inside the same passive window",
        },
        "evaluated_reference_count": evaluated,
        "geometry_abstention_count": geometry_abstentions,
        "metrics": metrics,
        "evaluability": {
            "minimum_evaluated_reference_count": MIN_EVALUATED_REFERENCES,
            "minimum_recovery_opportunity_parent_count": MIN_RECOVERY_OPPORTUNITY_PARENTS,
            "minimum_clear_denominator_parent_count": MIN_CLEAR_DENOMINATOR_PARENTS,
            "checks": evaluability,
        },
        "decisions": decisions,
        "terminal": terminal,
        "reference_receipt_sha256": hashlib.sha256(canonical_json_bytes(reference_receipts)).hexdigest().upper(),
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "depth_payload_role": "SOURCE_OBSERVATION_AND_SOURCE_DERIVED_DEVELOPMENT_LABEL",
            "model_runs": 0,
            "training_steps": 0,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "Consumed Bonn RGB-D source-derived Development oracle evidence only; not fresh Confirmation, CLEAR classification, learned scoring, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capability-only", action="store_true")
    parser.add_argument("--max-references-per-parent", type=int, default=MAX_REFERENCES_PER_PARENT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.max_references_per_parent > 0, "max references must be positive")
    if args.capability_only:
        result, _selected = audit_capability(args.dataset_root.resolve(), args.max_references_per_parent)
    else:
        result = evaluate(args.dataset_root.resolve(), args.max_references_per_parent)
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if result.get("decision", "PASS") != "NOT_EVALUABLE_DATA_OBSERVABILITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
