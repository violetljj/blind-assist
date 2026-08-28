from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from scenefun3d_functional_handoff_ceiling import FunctionalProposal, ParentBox, _load_json
from scenefun3d_functional_set_integrity import _score, _sha256
from scenefun3d_ordinal_axis_integrity import _parent_from_row, _proposal, _sc31_selection, parse_ordinal


@dataclass(frozen=True)
class DirectionalOrdinal:
    ordinal: int
    direction: str


@dataclass(frozen=True)
class ActiveOrdinalView:
    timestamp: str
    camera_to_world: np.ndarray
    ordered_candidate_ids_left_to_right: tuple[str, ...]
    normalized_image_xy: dict[str, tuple[float, float]]
    horizontal_span: float
    vertical_span: float
    horizontal_to_vertical_ratio: float
    median_depth_m: float
    consensus_window_pose_count: int = 0
    consensus_support_count: int = 0
    consensus_support_fraction: float = 0.0
    depth_timestamp: str | None = None
    intrinsic_timestamp: str | None = None
    frame_alignment_seconds: float | None = None
    visibility_depth_residual_m: dict[str, float] | None = None


def parse_directional_ordinal(text: str) -> DirectionalOrdinal | None:
    ordinal = parse_ordinal(text)
    if ordinal is None:
        return None
    normalized = " ".join(text.casefold().split())
    match = re.search(r"\bfrom\s+(?:the\s+)?(left|right)\b", normalized)
    if match is None:
        return None
    return DirectionalOrdinal(ordinal, f"FROM_{match.group(1).upper()}")


def load_camera_trajectory(path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            tokens = line.split()
            if len(tokens) != 7:
                raise ValueError("INVALID_CAMERA_TRAJECTORY_ROW")
            timestamp = tokens[0]
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = Rotation.from_rotvec(
                np.asarray(tokens[1:4], dtype=np.float64)
            ).as_matrix()
            world_to_camera[:3, 3] = np.asarray(tokens[4:7], dtype=np.float64)
            poses[timestamp] = np.linalg.inv(world_to_camera)
    if not poses:
        raise ValueError("EMPTY_CAMERA_TRAJECTORY")
    return poses


def select_active_ordinal_view(
    candidates: dict[str, FunctionalProposal],
    poses: dict[str, np.ndarray],
    algorithm: dict[str, Any],
) -> ActiveOrdinalView | None:
    candidate_ids = sorted(candidates)
    points = np.asarray([candidates[candidate_id].center for candidate_id in candidate_ids])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    minimum_depth = float(algorithm["minimum_camera_depth_m"])
    maximum_depth = float(algorithm["maximum_camera_depth_m"])
    maximum_coordinate = float(algorithm["maximum_absolute_normalized_image_coordinate"])
    minimum_span = float(algorithm["minimum_horizontal_normalized_span"])
    minimum_ratio = float(algorithm["minimum_horizontal_to_vertical_span_ratio"])
    best: tuple[tuple[float, float], ActiveOrdinalView] | None = None
    for timestamp in sorted(poses, key=float):
        camera_to_world = poses[timestamp]
        world_to_camera = np.linalg.inv(camera_to_world)
        camera = (homogeneous @ world_to_camera.T)[:, :3]
        depth = camera[:, 2]
        if np.any(depth < minimum_depth) or np.any(depth > maximum_depth):
            continue
        normalized_xy = camera[:, :2] / depth[:, None]
        if np.any(np.abs(normalized_xy) > maximum_coordinate):
            continue
        horizontal_span = float(np.ptp(normalized_xy[:, 0]))
        vertical_span = float(np.ptp(normalized_xy[:, 1]))
        ratio = horizontal_span / max(vertical_span, 1e-9)
        if horizontal_span < minimum_span or ratio < minimum_ratio:
            continue
        order = np.argsort(normalized_xy[:, 0], kind="stable")
        view = ActiveOrdinalView(
            timestamp=timestamp,
            camera_to_world=camera_to_world,
            ordered_candidate_ids_left_to_right=tuple(
                candidate_ids[int(index)] for index in order
            ),
            normalized_image_xy={
                candidate_ids[index]: tuple(float(value) for value in normalized_xy[index])
                for index in range(len(candidate_ids))
            },
            horizontal_span=horizontal_span,
            vertical_span=vertical_span,
            horizontal_to_vertical_ratio=ratio,
            median_depth_m=float(np.median(depth)),
        )
        score = (horizontal_span, -view.median_depth_m)
        if best is None or score > best[0]:
            best = (score, view)
    return None if best is None else best[1]


def select_consensus_active_ordinal_view(
    candidates: dict[str, FunctionalProposal],
    poses: dict[str, np.ndarray],
    algorithm: dict[str, Any],
) -> ActiveOrdinalView | None:
    candidate_ids = sorted(candidates)
    points = np.asarray([candidates[candidate_id].center for candidate_id in candidate_ids])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    valid: list[tuple[float, ActiveOrdinalView]] = []
    for timestamp in sorted(poses, key=float):
        camera_to_world = poses[timestamp]
        camera = (homogeneous @ np.linalg.inv(camera_to_world).T)[:, :3]
        depth = camera[:, 2]
        if np.any(depth < float(algorithm["minimum_camera_depth_m"])) or np.any(
            depth > float(algorithm["maximum_camera_depth_m"])
        ):
            continue
        xy = camera[:, :2] / depth[:, None]
        if np.any(
            np.abs(xy)
            > float(algorithm["maximum_absolute_normalized_image_coordinate"])
        ):
            continue
        horizontal_span = float(np.ptp(xy[:, 0]))
        if horizontal_span < float(algorithm["minimum_horizontal_normalized_span"]):
            continue
        vertical_span = float(np.ptp(xy[:, 1]))
        order = np.argsort(xy[:, 0], kind="stable")
        valid.append(
            (
                float(timestamp),
                ActiveOrdinalView(
                    timestamp=timestamp,
                    camera_to_world=camera_to_world,
                    ordered_candidate_ids_left_to_right=tuple(
                        candidate_ids[int(index)] for index in order
                    ),
                    normalized_image_xy={
                        candidate_ids[index]: tuple(float(value) for value in xy[index])
                        for index in range(len(candidate_ids))
                    },
                    horizontal_span=horizontal_span,
                    vertical_span=vertical_span,
                    horizontal_to_vertical_ratio=horizontal_span / max(vertical_span, 1e-9),
                    median_depth_m=float(np.median(depth)),
                ),
            )
        )
    window_seconds = float(algorithm["consensus_window_seconds"])
    minimum_support = int(algorithm["minimum_consensus_support_poses"])
    minimum_fraction = float(algorithm["minimum_consensus_support_fraction"])
    best: tuple[tuple[float, float, int, float], ActiveOrdinalView] | None = None
    for timestamp, view in valid:
        window = [
            neighbor
            for neighbor_time, neighbor in valid
            if abs(neighbor_time - timestamp) <= window_seconds
        ]
        support = sum(
            neighbor.ordered_candidate_ids_left_to_right
            == view.ordered_candidate_ids_left_to_right
            for neighbor in window
        )
        fraction = support / len(window) if window else 0.0
        if support < minimum_support or fraction < minimum_fraction:
            continue
        candidate = replace(
            view,
            consensus_window_pose_count=len(window),
            consensus_support_count=support,
            consensus_support_fraction=fraction,
        )
        score = (
            candidate.horizontal_span,
            fraction,
            support,
            -candidate.median_depth_m,
        )
        if best is None or score > best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def _timestamped_files(directory: Path, suffix: str) -> list[tuple[float, str, Path]]:
    rows = []
    for path in directory.rglob(f"*{suffix}"):
        timestamp = path.stem.split("_")[-1]
        try:
            rows.append((float(timestamp), timestamp, path))
        except ValueError:
            continue
    return sorted(rows)


def _nearest_timestamped_file(
    rows: list[tuple[float, str, Path]], timestamp: float
) -> tuple[float, str, Path] | None:
    if not rows:
        return None
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    insertion = int(np.searchsorted(times, timestamp))
    choices = [index for index in (insertion - 1, insertion) if 0 <= index < len(rows)]
    if not choices:
        return None
    selected = min(choices, key=lambda index: abs(rows[index][0] - timestamp))
    return rows[selected]


def _read_intrinsic(path: Path) -> tuple[int, int, np.ndarray]:
    width, height, fx, fy, cx, cy = [
        float(value) for value in path.read_text(encoding="utf-8").split()
    ]
    return int(width), int(height), np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def select_depth_visible_consensus_active_ordinal_view(
    candidates: dict[str, FunctionalProposal],
    poses: dict[str, np.ndarray],
    depth_directory: Path,
    intrinsic_directory: Path,
    algorithm: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> ActiveOrdinalView | None:
    candidate_ids = sorted(candidates)
    points = np.asarray([candidates[candidate_id].center for candidate_id in candidate_ids])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    depth_files = _timestamped_files(depth_directory, ".png")
    intrinsic_files = _timestamped_files(intrinsic_directory, ".pincam")
    alignment_limit = float(algorithm["maximum_frame_alignment_seconds"])
    minimum_depth = float(algorithm["minimum_camera_depth_m"])
    maximum_depth = float(algorithm["maximum_camera_depth_m"])
    minimum_span = float(algorithm["minimum_horizontal_normalized_span"])
    maximum_coordinate = float(algorithm["maximum_absolute_normalized_image_coordinate"])
    radius = int(algorithm["depth_window_radius_pixels"])
    minimum_valid = int(algorithm["minimum_valid_depth_pixels_per_candidate"])
    absolute_tolerance = float(algorithm["depth_absolute_tolerance_m"])
    relative_tolerance = float(algorithm["depth_relative_tolerance"])
    valid: list[tuple[float, ActiveOrdinalView]] = []
    stage_counts = {
        "poses_total": len(poses),
        "frames_aligned": 0,
        "geometry_valid": 0,
        "pixels_in_frame": 0,
        "depth_decoded": 0,
        "all_candidates_depth_visible": 0,
        "horizontal_span_valid": 0,
        "temporal_consensus_valid": 0,
    }
    minimum_maximum_residual = float("inf")
    for pose_timestamp in sorted(poses, key=float):
        pose_time = float(pose_timestamp)
        depth_row = _nearest_timestamped_file(depth_files, pose_time)
        intrinsic_row = _nearest_timestamped_file(intrinsic_files, pose_time)
        if depth_row is None or intrinsic_row is None:
            continue
        alignment = max(abs(depth_row[0] - pose_time), abs(intrinsic_row[0] - pose_time))
        if alignment > alignment_limit or abs(depth_row[0] - intrinsic_row[0]) > alignment_limit:
            continue
        stage_counts["frames_aligned"] += 1
        camera_to_world = poses[pose_timestamp]
        camera = (homogeneous @ np.linalg.inv(camera_to_world).T)[:, :3]
        predicted_depth = camera[:, 2]
        if np.any(predicted_depth < minimum_depth) or np.any(predicted_depth > maximum_depth):
            continue
        normalized_xy = camera[:, :2] / predicted_depth[:, None]
        if np.any(np.abs(normalized_xy) > maximum_coordinate):
            continue
        stage_counts["geometry_valid"] += 1
        width, height, intrinsic = _read_intrinsic(intrinsic_row[2])
        projected = (intrinsic @ camera.T).T
        pixels = projected[:, :2] / projected[:, 2, None]
        if np.any(pixels[:, 0] < 0) or np.any(pixels[:, 0] >= width) or np.any(
            pixels[:, 1] < 0
        ) or np.any(pixels[:, 1] >= height):
            continue
        stage_counts["pixels_in_frame"] += 1
        depth_mm = cv2.imread(str(depth_row[2]), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth_mm = np.squeeze(depth_mm)
        if depth_mm.shape != (height, width):
            continue
        stage_counts["depth_decoded"] += 1
        residuals: dict[str, float] = {}
        all_visible = True
        for index, candidate_id in enumerate(candidate_ids):
            u, v = np.rint(pixels[index]).astype(np.int64)
            x1, x2 = max(0, u - radius), min(width, u + radius + 1)
            y1, y2 = max(0, v - radius), min(height, v + radius + 1)
            observed = depth_mm[y1:y2, x1:x2].astype(np.float64).ravel() / 1000.0
            observed = observed[observed > 0.0]
            if len(observed) < minimum_valid:
                all_visible = False
                break
            residual = float(abs(np.median(observed) - predicted_depth[index]))
            tolerance = absolute_tolerance + relative_tolerance * float(predicted_depth[index])
            if residual > tolerance:
                all_visible = False
            residuals[candidate_id] = residual
        if len(residuals) == len(candidate_ids):
            minimum_maximum_residual = min(
                minimum_maximum_residual, max(residuals.values())
            )
        if not all_visible:
            continue
        stage_counts["all_candidates_depth_visible"] += 1
        horizontal_span = float(np.ptp(normalized_xy[:, 0]))
        if horizontal_span < minimum_span:
            continue
        stage_counts["horizontal_span_valid"] += 1
        vertical_span = float(np.ptp(normalized_xy[:, 1]))
        order = np.argsort(normalized_xy[:, 0], kind="stable")
        valid.append(
            (
                pose_time,
                ActiveOrdinalView(
                    timestamp=pose_timestamp,
                    camera_to_world=camera_to_world,
                    ordered_candidate_ids_left_to_right=tuple(
                        candidate_ids[int(index)] for index in order
                    ),
                    normalized_image_xy={
                        candidate_ids[index]: tuple(float(value) for value in normalized_xy[index])
                        for index in range(len(candidate_ids))
                    },
                    horizontal_span=horizontal_span,
                    vertical_span=vertical_span,
                    horizontal_to_vertical_ratio=horizontal_span / max(vertical_span, 1e-9),
                    median_depth_m=float(np.median(predicted_depth)),
                    depth_timestamp=depth_row[1],
                    intrinsic_timestamp=intrinsic_row[1],
                    frame_alignment_seconds=alignment,
                    visibility_depth_residual_m=residuals,
                ),
            )
        )
    window_seconds = float(algorithm["consensus_window_seconds"])
    minimum_support = int(algorithm["minimum_consensus_support_poses"])
    minimum_fraction = float(algorithm["minimum_consensus_support_fraction"])
    best: tuple[tuple[float, float, int, float], ActiveOrdinalView] | None = None
    for timestamp, view in valid:
        window = [
            neighbor
            for neighbor_time, neighbor in valid
            if abs(neighbor_time - timestamp) <= window_seconds
        ]
        support = sum(
            neighbor.ordered_candidate_ids_left_to_right
            == view.ordered_candidate_ids_left_to_right
            for neighbor in window
        )
        fraction = support / len(window) if window else 0.0
        if support < minimum_support or fraction < minimum_fraction:
            continue
        stage_counts["temporal_consensus_valid"] += 1
        candidate = replace(
            view,
            consensus_window_pose_count=len(window),
            consensus_support_count=support,
            consensus_support_fraction=fraction,
        )
        score = (candidate.horizontal_span, fraction, support, -candidate.median_depth_m)
        if best is None or score > best[0]:
            best = (score, candidate)
    if diagnostics is not None:
        diagnostics.update(stage_counts)
        diagnostics["minimum_maximum_candidate_depth_residual_m"] = (
            None if not np.isfinite(minimum_maximum_residual) else minimum_maximum_residual
        )
    return None if best is None else best[1]


def build_provider(
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    source_result: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    algorithm = protocol["frozen_algorithm"]
    source_by_visit = {row["visit_id"]: row for row in source_result["selected"]}
    scenes: list[dict[str, Any]] = []
    for source in cohort["cohort"]:
        visit_id = source["visit_id"]
        descriptions = _load_json(
            data_root / visit_id / f"{visit_id}_descriptions.json"
        )["descriptions"]
        source_row = source_by_visit[visit_id]
        parent_rows = {
            row["parent_binding_id"]: row
            for row in source_row["active_view_parents"]
        }
        candidate_to_parent = {
            candidate_id: parent_id
            for parent_id, row in parent_rows.items()
            for candidate_id in row["candidates"]
        }
        tasks: list[dict[str, Any]] = []
        not_evaluable: list[dict[str, Any]] = []
        for description in descriptions:
            directional = parse_directional_ordinal(description["description"])
            if directional is None:
                continue
            target_parent_ids = {
                candidate_to_parent[target_id]
                for target_id in description["annot_id"]
                if target_id in candidate_to_parent
            }
            if len(target_parent_ids) != 1:
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_ACTIVE_VIEW_PARENT_BINDING"}
                )
                continue
            parent_row = parent_rows[next(iter(target_parent_ids))]
            parent = _parent_from_row(parent_row)
            labels = {
                candidate_id: row["label"]
                for candidate_id, row in parent_row["candidates"].items()
            }
            candidates = {
                candidate_id: _proposal(candidate_id, row, parent)
                for candidate_id, row in parent_row["candidates"].items()
            }
            baseline, semantic_admitted, requested_labels = _sc31_selection(
                description["description"], parent, candidates, labels, algorithm
            )
            compatible = {
                candidate_id: proposal
                for candidate_id, proposal in candidates.items()
                if labels[candidate_id] in requested_labels
            }
            ordered = list(parent_row["active_view"]["ordered_candidate_ids_left_to_right"])
            if set(ordered) != set(compatible):
                matches: list[str] = []
            else:
                index = (
                    directional.ordinal - 1
                    if directional.direction == "FROM_LEFT"
                    else len(ordered) - directional.ordinal
                )
                matches = [ordered[index]] if 0 <= index < len(ordered) else []
            admitted = semantic_admitted and len(matches) == 1
            successor = tuple(matches) if admitted else baseline
            tasks.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "parent_binding_id": parent.binding_id,
                    "requested_ordinal": directional.ordinal,
                    "direction": directional.direction,
                    "active_view_timestamp": parent_row["active_view"]["timestamp"],
                    "ordered_candidate_ids_left_to_right": ordered,
                    "active_view_admitted": admitted,
                    "baseline_selected_candidate_ids": list(baseline),
                    "successor_selected_candidate_ids": list(successor),
                }
            )
        scenes.append(
            {
                "visit_id": visit_id,
                "video_id": source["video_id"],
                "tasks": tasks,
                "not_evaluable": not_evaluable,
            }
        )
    return {
        "schema_version": 1,
        "provider": "L10-SC36-ACTIVE-VIEW-DIRECTIONAL-ORDINAL-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort_protocol_sha256": cohort["protocol_sha256"],
        "source_admission_result_sha256": source_result["result_sha256"],
        "truth_isolation": "Source admission seals public directional text, provider-public candidate geometry, and a target-independent real camera pose. Target IDs are opened only here to recover the authorized exact parent and never choose the pose or candidate order.",
        "scenes": scenes,
    }


def evaluate_provider(
    protocol: dict[str, Any],
    provider: dict[str, Any],
    provider_hash: str,
    data_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in provider["scenes"]:
        descriptions = {
            row["desc_id"]: row
            for row in _load_json(
                data_root / scene["visit_id"] / f"{scene['visit_id']}_descriptions.json"
            )["descriptions"]
        }
        for task in scene["tasks"]:
            target = set(descriptions[task["desc_id"]]["annot_id"])
            rows.append(
                {
                    "visit_id": scene["visit_id"],
                    "desc_id": task["desc_id"],
                    "description": task["description"],
                    "active_view_admitted": task["active_view_admitted"],
                    "baseline": _score(task["baseline_selected_candidate_ids"], target),
                    "successor": _score(task["successor_selected_candidate_ids"], target),
                }
            )
    admissions = sum(row["active_view_admitted"] for row in rows)
    regressions = sum(
        row["successor"]["target_set_recall"] < row["baseline"]["target_set_recall"]
        or row["successor"]["wrong_part_count"] > row["baseline"]["wrong_part_count"]
        for row in rows
    )
    baseline_legal = sum(row["baseline"]["legal_commit"] for row in rows)
    successor_legal = sum(row["successor"]["legal_commit"] for row in rows)
    baseline_wrong = sum(row["baseline"]["wrong_part_count"] for row in rows)
    successor_wrong = sum(row["successor"]["wrong_part_count"] for row in rows)
    baseline_recall = float(np.mean([row["baseline"]["target_set_recall"] for row in rows])) if rows else 0.0
    successor_recall = float(np.mean([row["successor"]["target_set_recall"] for row in rows])) if rows else 0.0
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluable_tasks"]):
        decision = protocol["decision_labels"]["insufficient_tasks"]
    elif admissions < int(gate["minimum_active_view_admissions"]):
        decision = protocol["decision_labels"]["insufficient_admissions"]
    elif (
        successor_legal - baseline_legal >= int(gate["minimum_legal_commit_gain"])
        and successor_recall >= baseline_recall
        and successor_wrong <= baseline_wrong
        and regressions <= int(gate["maximum_taskwise_regressions"])
    ):
        decision = protocol["decision_labels"]["pass"]
    else:
        decision = protocol["decision_labels"]["fail"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": protocol["protocol_sha256"],
        "provider_sha256": provider_hash,
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "scenes": len(provider["scenes"]),
            "tasks_evaluable": len(rows),
            "tasks_not_evaluable": sum(len(scene["not_evaluable"]) for scene in provider["scenes"]),
            "active_view_admissions": admissions,
            "taskwise_regressions": regressions,
        },
        "baseline": {"legal_commit_count": baseline_legal, "mean_target_set_recall": baseline_recall, "wrong_part_count": baseline_wrong},
        "successor": {"legal_commit_count": successor_legal, "mean_target_set_recall": successor_recall, "wrong_part_count": successor_wrong},
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    cohort = _load_json(args.cohort)
    cohort["protocol_sha256"] = _sha256(args.cohort)
    source_result = _load_json(args.source_result)
    source_result["result_sha256"] = _sha256(args.source_result)
    if cohort["source_admission"]["result_sha256"] != source_result["result_sha256"]:
        raise ValueError("SOURCE_ADMISSION_RESULT_HASH_MISMATCH")
    if protocol["source"]["cohort_protocol_sha256"] != cohort["protocol_sha256"]:
        raise ValueError("COHORT_PROTOCOL_HASH_MISMATCH")
    provider = build_provider(protocol, cohort, source_result, args.data_root.resolve())
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_hash = _sha256(args.provider_output)
    result = evaluate_provider(protocol, provider, provider_hash, args.data_root.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("decision", "denominators", "baseline", "successor")}, indent=2))


if __name__ == "__main__":
    main()
