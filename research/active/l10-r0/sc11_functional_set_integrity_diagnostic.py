from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from functional_part_binding import FunctionalBindingDecision, FunctionalBindingState
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _transform_points,
)
from scenefun3d_functional_set_integrity import _sha256, apply_integrity


FUNCTIONAL_PART_MATCH_TOLERANCE_M = 0.12


def _score(
    selected: tuple[str, ...],
    candidates: dict[str, FunctionalProposal],
    target_points: dict[str, np.ndarray],
) -> dict[str, Any]:
    matched_targets: set[str] = set()
    wrong = 0
    distances: list[float] = []
    for candidate_id in selected:
        center = candidates[candidate_id].center
        nearest_target, distance = min(
            [
                (
                    target_id,
                    float(np.linalg.norm(points - center, axis=1).min()),
                )
                for target_id, points in target_points.items()
            ],
            key=lambda item: item[1],
        )
        distances.append(distance)
        if distance <= FUNCTIONAL_PART_MATCH_TOLERANCE_M:
            matched_targets.add(nearest_target)
        else:
            wrong += 1
    return {
        "legal_commit": bool(selected) and wrong == 0,
        "target_set_recall": len(matched_targets) / len(target_points),
        "wrong_part_count": wrong,
        "selected_nearest_target_m": [round(distance, 6) for distance in distances],
    }


def run_diagnostic(
    protocol_path: Path,
    scene_dir: Path,
    video_id: str,
    sc11_provider_path: Path,
    sc11_result_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    link_radius = float(protocol["frozen_algorithm"]["normalized_component_link_radius"])
    minimum_component_size = int(
        protocol["frozen_algorithm"]["minimum_dominant_component_size"]
    )
    provider = _load_json(sc11_provider_path)
    result = _load_json(sc11_result_path)
    annotations_path = scene_dir / f"{scene_dir.name}_annotations.json"
    laser_path = scene_dir / f"{scene_dir.name}_laser_scan.ply"
    transform_path = scene_dir / video_id / f"{video_id}_transform.npy"
    annotations = {
        row["annot_id"]: row for row in _load_json(annotations_path)["annotations"]
    }
    xyz = _load_ply_xyz(laser_path)
    transform = np.load(transform_path)
    parents = {
        parent.binding_id: parent
        for parent in _load_parent_boxes(
            scene_dir / video_id / f"{video_id}_3dod_annotation.json"
        )
    }
    candidates: dict[str, FunctionalProposal] = {}
    for parent_id, rows in provider["arms"]["multiview"].items():
        parent = parents[parent_id]
        for row in rows:
            center = np.asarray(row["center_xyz_m"], dtype=np.float64)
            candidates[row["candidate_id"]] = FunctionalProposal(
                candidate_id=row["candidate_id"],
                points=center[None, :],
                center=center,
                parent=parent,
                parent_coverage=1.0,
            )

    rows: list[dict[str, Any]] = []
    for task in result["arms"]["multiview"]["task"]["tasks"]:
        baseline_selected = tuple(task["selected_candidate_ids"])
        baseline = FunctionalBindingDecision(
            FunctionalBindingState(task["state"]),
            baseline_selected,
            task["action"],
            "SC11_FROZEN_TASK_RELATIONAL_SELECTION",
            task["relation"],
        )
        parent = parents[task["parent_binding_id"]]
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
        target_points = {
            target_id: _transform_points(
                xyz[
                    np.asarray(
                        annotations[target_id]["indices"], dtype=np.int64
                    )
                ],
                transform,
            )
            for target_id in task["evaluator_target_ids"]
        }
        rows.append(
            {
                "desc_id": task["desc_id"],
                "description": task["description"],
                "baseline_selected_candidate_ids": list(baseline_selected),
                "successor_selected_candidate_ids": list(
                    successor.selected_candidate_ids
                ),
                "quarantined_candidate_ids": list(
                    successor.quarantined_candidate_ids
                ),
                "component_sizes": list(successor.component_sizes),
                "integrity_opportunity": successor.integrity_opportunity,
                "baseline": _score(
                    baseline_selected, candidates, target_points
                ),
                "successor": _score(
                    successor.selected_candidate_ids,
                    candidates,
                    target_points,
                ),
            }
        )

    def aggregate(arm: str) -> dict[str, Any]:
        legal = sum(row[arm]["legal_commit"] for row in rows)
        wrong = sum(row[arm]["wrong_part_count"] for row in rows)
        recall = float(
            np.mean([row[arm]["target_set_recall"] for row in rows])
        ) if rows else 0.0
        return {
            "legal_commit_count": legal,
            "legal_commit_rate": legal / len(rows) if rows else 0.0,
            "mean_target_set_recall": recall,
            "wrong_part_count": wrong,
        }

    baseline_metrics = aggregate("baseline")
    successor_metrics = aggregate("successor")
    effect = (
        successor_metrics["legal_commit_count"] > baseline_metrics["legal_commit_count"]
        and successor_metrics["wrong_part_count"] < baseline_metrics["wrong_part_count"]
        and successor_metrics["mean_target_set_recall"]
        >= baseline_metrics["mean_target_set_recall"]
    )
    return {
        "schema_version": 1,
        "experiment": "L10-SC21-SC11-CONSUMED-FUNCTIONAL-SET-INTEGRITY-DIAGNOSTIC",
        "decision": (
            "SC21_SC11_CONSUMED_DIAGNOSTIC_MECHANISM_EFFECT"
            if effect
            else "SC21_SC11_CONSUMED_DIAGNOSTIC_NO_MECHANISM_EFFECT"
        ),
        "protocol_sha256": _sha256(protocol_path),
        "source": {
            "visit_id": scene_dir.name,
            "video_id": video_id,
            "sc11_provider_sha256": _sha256(sc11_provider_path),
            "sc11_result_sha256": _sha256(sc11_result_path),
            "annotations_sha256": _sha256(annotations_path),
            "laser_scan_sha256": _sha256(laser_path),
            "transform_sha256": _sha256(transform_path),
        },
        "denominators": {
            "tasks": len(rows),
            "integrity_opportunities": sum(
                row["integrity_opportunity"] for row in rows
            ),
        },
        "baseline": baseline_metrics,
        "successor": successor_metrics,
        "rows": rows,
        "claim_boundary": (
            "Read-only mechanism attribution on the already-consumed SC11 420683 output. "
            "The component radius and dominance rule were frozen in SC21 v0 before this diagnostic. "
            "This can localize a candidate-integrity mechanism but cannot confirm transfer, RGB proposal "
            "generation, endpoint reachability, arrival, product benefit, user benefit, or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sc11-provider", type=Path, required=True)
    parser.add_argument("--sc11-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(
        args.protocol.resolve(),
        args.scene_dir.resolve(),
        args.video_id,
        args.sc11_provider.resolve(),
        args.sc11_result.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline": result["baseline"],
                "successor": result["successor"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
