from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from functional_part_binding import FunctionalBindingDecision, FunctionalBindingState
from scenefun3d_factorized_endpoint_observer import (
    _observational_ready,
    _read_intrinsic,
)
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _transform_points,
)
from scenefun3d_functional_set_integrity import _sha256, apply_integrity


def _verify_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected={expected}, actual={actual}")


def _integrity_tasks(
    scene_dir: Path,
    video_id: str,
    sc11_provider: dict[str, Any],
    sc11_result: dict[str, Any],
    sc21_protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    parents = {
        parent.binding_id: parent
        for parent in _load_parent_boxes(
            scene_dir / video_id / f"{video_id}_3dod_annotation.json"
        )
    }
    candidates: dict[str, FunctionalProposal] = {}
    for parent_id, rows in sc11_provider["arms"]["multiview"].items():
        for row in rows:
            center = np.asarray(row["center_xyz_m"], dtype=np.float64)
            candidates[row["candidate_id"]] = FunctionalProposal(
                row["candidate_id"],
                center[None, :],
                center,
                parents[parent_id],
                1.0,
            )
    link_radius = float(
        sc21_protocol["frozen_algorithm"]["normalized_component_link_radius"]
    )
    minimum_component_size = int(
        sc21_protocol["frozen_algorithm"]["minimum_dominant_component_size"]
    )
    tasks: dict[str, dict[str, Any]] = {}
    for row in sc11_result["arms"]["multiview"]["task"]["tasks"]:
        baseline = FunctionalBindingDecision(
            FunctionalBindingState(row["state"]),
            tuple(row["selected_candidate_ids"]),
            row["action"],
            "SC11_FROZEN_TASK_RELATIONAL_SELECTION",
            row["relation"],
        )
        parent = parents[row["parent_binding_id"]]
        parent_candidates = {
            candidate_id: candidate
            for candidate_id, candidate in candidates.items()
            if candidate.parent.binding_id == parent.binding_id
        }
        successor = apply_integrity(
            baseline,
            parent,
            parent_candidates,
            link_radius=link_radius,
            minimum_component_size=minimum_component_size,
        )
        tasks[row["desc_id"]] = {
            "desc_id": row["desc_id"],
            "description": row["description"],
            "selected_candidate_ids": list(successor.selected_candidate_ids),
            "quarantined_candidate_ids": list(successor.quarantined_candidate_ids),
            "predicted_points_xyz_m": [
                candidates[candidate_id].center.tolist()
                for candidate_id in successor.selected_candidate_ids
            ],
        }
    return tasks


def build_provider(
    protocol_path: Path,
    scene_dir: Path,
    video_id: str,
    sc20_provider_path: Path,
    sc20_result_path: Path,
    sc11_provider_path: Path,
    sc11_result_path: Path,
    sc21_protocol_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    source = protocol["source"]
    for path, key in (
        (sc20_provider_path, "sc20_provider_sha256"),
        (sc20_result_path, "sc20_result_sha256"),
        (sc11_provider_path, "sc11_provider_sha256"),
        (sc11_result_path, "sc11_result_sha256"),
        (sc21_protocol_path, "sc21_protocol_sha256"),
    ):
        _verify_hash(path, source[key])
    sc20_provider = _load_json(sc20_provider_path)
    sc11_provider = _load_json(sc11_provider_path)
    sc11_result = _load_json(sc11_result_path)
    sc21_protocol = _load_json(sc21_protocol_path)
    tasks = _integrity_tasks(
        scene_dir, video_id, sc11_provider, sc11_result, sc21_protocol
    )

    frames: list[dict[str, Any]] = []
    for frame in sc20_provider["frames"]:
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
        intrinsic_path = (
            scene_dir
            / video_id
            / "lowres_wide_intrinsics"
            / Path(frame["frame_name"]).with_suffix(".pincam")
        )
        width, height, intrinsic = _read_intrinsic(intrinsic_path)
        depth_mm = cv2.imread(
            str(scene_dir / video_id / "lowres_depth" / frame["depth_frame_name"]),
            cv2.IMREAD_UNCHANGED,
        )
        if depth_mm is None:
            raise ValueError(f"Missing depth for {frame['frame_name']}")
        depth_mm = np.squeeze(depth_mm)
        if depth_mm.shape != (height, width):
            raise ValueError(f"Depth shape mismatch for {frame['frame_name']}")
        baseline_by_task = {row["desc_id"]: row for row in frame["tasks"]}
        task_rows: list[dict[str, Any]] = []
        for desc_id, task in tasks.items():
            points = np.asarray(task["predicted_points_xyz_m"], dtype=np.float64)
            ready, factors = _observational_ready(
                points, camera_to_world, intrinsic, depth_mm
            )
            task_rows.append(
                {
                    "desc_id": desc_id,
                    "sc20_factorized_ready": bool(
                        baseline_by_task[desc_id]["factorized_observational_ready"]
                    ),
                    "integrity_factorized_ready": ready,
                    "integrity_factors": factors,
                }
            )
        frames.append(
            {
                "frame_name": frame["frame_name"],
                "depth_frame_name": frame["depth_frame_name"],
                "camera_to_world": frame["camera_to_world"],
                "tasks": task_rows,
            }
        )
    return {
        "schema_version": 1,
        "provider": "L10-SC22-INTEGRITY-FILTERED-FACTORIZED-ENDPOINT-PROVIDER",
        "protocol_sha256": _sha256(protocol_path),
        "source": source,
        "truth_isolation": (
            "The composition consumes only SC20 frame geometry and SC11 selected candidate geometry, "
            "then applies the already-frozen SC21 integrity rule and unchanged SC20 endpoint factors. "
            "SceneFun3D target annotations are not loaded until after this provider is sealed."
        ),
        "tasks": list(tasks.values()),
        "frames": frames,
    }


def evaluate_provider(
    protocol_path: Path,
    scene_dir: Path,
    video_id: str,
    provider: dict[str, Any],
    provider_sha256: str,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    visit_id = scene_dir.name
    annotations_path = scene_dir / f"{visit_id}_annotations.json"
    descriptions_path = scene_dir / f"{visit_id}_descriptions.json"
    laser_path = scene_dir / f"{visit_id}_laser_scan.ply"
    transform_path = scene_dir / video_id / f"{video_id}_transform.npy"
    annotations = {
        row["annot_id"]: row for row in _load_json(annotations_path)["annotations"]
    }
    descriptions = {
        row["desc_id"]: row for row in _load_json(descriptions_path)["descriptions"]
    }
    xyz = _load_ply_xyz(laser_path)
    transform = np.load(transform_path)
    truth_points: dict[str, np.ndarray] = {}
    for task in provider["tasks"]:
        parts = [
            _transform_points(
                xyz[
                    np.asarray(
                        annotations[target_id]["indices"], dtype=np.int64
                    )
                ],
                transform,
            )
            for target_id in descriptions[task["desc_id"]]["annot_id"]
        ]
        merged = np.concatenate(parts, axis=0)
        if len(merged) > 256:
            merged = merged[
                np.linspace(0, len(merged) - 1, 256, dtype=np.int64)
            ]
        truth_points[task["desc_id"]] = merged

    names = ("sc20_factorized", "integrity_factorized")
    counts = {
        name: {"ready": 0, "true_ready": 0, "false_ready": 0}
        for name in names
    }
    per_task = {
        task["desc_id"]: {
            "description": task["description"],
            "truth_ready": 0,
            "sc20_factorized_true_ready": 0,
            "integrity_factorized_true_ready": 0,
        }
        for task in provider["tasks"]
    }
    truth_ready_total = 0
    for frame in provider["frames"]:
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
        intrinsic_path = (
            scene_dir
            / video_id
            / "lowres_wide_intrinsics"
            / Path(frame["frame_name"]).with_suffix(".pincam")
        )
        width, height, intrinsic = _read_intrinsic(intrinsic_path)
        depth_mm = cv2.imread(
            str(scene_dir / video_id / "lowres_depth" / frame["depth_frame_name"]),
            cv2.IMREAD_UNCHANGED,
        )
        if depth_mm is None:
            raise ValueError(f"Missing evaluator depth for {frame['frame_name']}")
        depth_mm = np.squeeze(depth_mm)
        if depth_mm.shape != (height, width):
            raise ValueError(f"Evaluator depth shape mismatch for {frame['frame_name']}")
        for row in frame["tasks"]:
            desc_id = row["desc_id"]
            truth_ready, _ = _observational_ready(
                truth_points[desc_id], camera_to_world, intrinsic, depth_mm
            )
            truth_ready_total += int(truth_ready)
            per_task[desc_id]["truth_ready"] += int(truth_ready)
            for name in names:
                ready = bool(row[f"{name}_ready"])
                counts[name]["ready"] += int(ready)
                counts[name]["true_ready"] += int(ready and truth_ready)
                counts[name]["false_ready"] += int(ready and not truth_ready)
                per_task[desc_id][f"{name}_true_ready"] += int(
                    ready and truth_ready
                )

    def metrics(name: str) -> dict[str, Any]:
        arm = counts[name]
        precision = arm["true_ready"] / arm["ready"] if arm["ready"] else 0.0
        recall = arm["true_ready"] / truth_ready_total if truth_ready_total else 0.0
        return {
            "ready_frames": arm["ready"],
            "true_ready_frames": arm["true_ready"],
            "false_ready_frames": arm["false_ready"],
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tasks_with_true_ready": sum(
                row[f"{name}_true_ready"] > 0 for row in per_task.values()
            ),
        }

    baseline = metrics("sc20_factorized")
    successor = metrics("integrity_factorized")
    gate = protocol["frozen_gate"]
    passed = (
        successor["tasks_with_true_ready"]
        >= int(gate["minimum_tasks_with_true_ready"])
        and successor["false_ready_frames"]
        <= int(gate["maximum_false_ready_frames"])
        and successor["precision"] >= float(gate["minimum_precision"])
        and successor["recall"] >= float(gate["minimum_recall"])
        and successor["f1"] >= float(gate["minimum_f1"])
    )
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": (
            "SC22_INTEGRITY_ENDPOINT_COMPOSITION_MECHANICS_SIGNAL"
            if passed
            else "SC22_INTEGRITY_ENDPOINT_COMPOSITION_GATE_NOT_MET"
        ),
        "claim_layer": "CONSUMED_REAL_RGBD_TRAJECTORY_COMPOSITION_MECHANICS",
        "protocol_sha256": _sha256(protocol_path),
        "provider_sha256": provider_sha256,
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "tasks": len(provider["tasks"]),
            "frames": len(provider["frames"]),
            "task_frames": len(provider["tasks"]) * len(provider["frames"]),
            "truth_ready_frames": truth_ready_total,
        },
        "baseline_sc20": baseline,
        "successor_integrity_endpoint": successor,
        "per_task": [
            {"desc_id": desc_id, **row} for desc_id, row in per_task.items()
        ],
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sc20-provider", type=Path, required=True)
    parser.add_argument("--sc20-result", type=Path, required=True)
    parser.add_argument("--sc11-provider", type=Path, required=True)
    parser.add_argument("--sc11-result", type=Path, required=True)
    parser.add_argument("--sc21-protocol", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provider = build_provider(
        args.protocol.resolve(),
        args.scene_dir.resolve(),
        args.video_id,
        args.sc20_provider.resolve(),
        args.sc20_result.resolve(),
        args.sc11_provider.resolve(),
        args.sc11_result.resolve(),
        args.sc21_protocol.resolve(),
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)
    result = evaluate_provider(
        args.protocol.resolve(),
        args.scene_dir.resolve(),
        args.video_id,
        provider,
        provider_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline": result["baseline_sc20"],
                "successor": result["successor_integrity_endpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
