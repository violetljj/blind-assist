from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from functional_part_binding import (
    FunctionalBindingState,
    FunctionalPartCandidate,
    TaskRelationalFunctionalSelector,
)


PARENT_CONTAINMENT_MARGIN_M = 0.03
PARENT_CONTAINMENT_MIN_FRACTION = 0.80
FUNCTIONAL_CONTACT_TOLERANCE_M = 0.02


@dataclass(frozen=True)
class ParentBox:
    binding_id: str
    label: str
    center: np.ndarray
    lengths: np.ndarray
    axes: np.ndarray


@dataclass(frozen=True)
class FunctionalProposal:
    candidate_id: str
    points: np.ndarray
    center: np.ndarray
    parent: ParentBox
    parent_coverage: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_ply_xyz(path: Path) -> np.ndarray:
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    scalar_types = {
        "float": "<f4",
        "float32": "<f4",
        "uchar": "u1",
        "uint8": "u1",
    }
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Expected a PLY file")
        in_vertex = False
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header ended before end_header")
            line = raw.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element "):
                in_vertex = False
                continue
            if in_vertex and line.startswith("property "):
                _, type_name, name = line.split()
                if type_name not in scalar_types:
                    raise ValueError(f"Unsupported PLY scalar type: {type_name}")
                properties.append((name, scalar_types[type_name]))
                continue
            if line == "end_header":
                offset = stream.tell()
                break
    if vertex_count is None or not {"x", "y", "z"}.issubset(name for name, _ in properties):
        raise ValueError("PLY vertex count or XYZ fields are missing")
    vertices = np.memmap(
        path,
        dtype=np.dtype(properties),
        mode="r",
        offset=offset,
        shape=(vertex_count,),
    )
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"]))


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :3]


def _load_parent_boxes(path: Path) -> list[ParentBox]:
    boxes: list[ParentBox] = []
    for row in _load_json(path)["data"]:
        obb = row["segments"]["obbAligned"]
        boxes.append(
            ParentBox(
                binding_id=row["uid"],
                label=row["label"],
                center=np.asarray(obb["centroid"], dtype=np.float64),
                lengths=np.asarray(obb["axesLengths"], dtype=np.float64),
                axes=np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3),
            )
        )
    return boxes


def _containment_fraction(points: np.ndarray, box: ParentBox) -> float:
    local = (points - box.center) @ box.axes.T
    return float(
        np.mean(
            np.all(
                np.abs(local) <= box.lengths / 2.0 + PARENT_CONTAINMENT_MARGIN_M,
                axis=1,
            )
        )
    )


def _match_parent(points: np.ndarray, boxes: list[ParentBox]) -> tuple[ParentBox, float] | None:
    ranked = sorted(
        (
            (
                _containment_fraction(points, box),
                -float(np.linalg.norm(points.mean(axis=0) - box.center)),
                box,
            )
            for box in boxes
        ),
        key=lambda row: (row[0], row[1], row[2].binding_id),
        reverse=True,
    )
    if not ranked or ranked[0][0] < PARENT_CONTAINMENT_MIN_FRACTION:
        return None
    return ranked[0][2], ranked[0][0]


def _nearest_distance(point: np.ndarray, proposals: list[FunctionalProposal]) -> float:
    return min(
        float(np.linalg.norm(proposal.points - point, axis=1).min())
        for proposal in proposals
    )


def _baseline_selection(
    parent: ParentBox, proposals: list[FunctionalProposal]
) -> tuple[str, ...]:
    selected = min(
        proposals,
        key=lambda proposal: (
            _nearest_distance(parent.center, [proposal]),
            proposal.candidate_id,
        ),
    )
    return (selected.candidate_id,)


def run_ceiling(scene_dir: Path, video_id: str) -> dict[str, Any]:
    visit_id = scene_dir.name
    source_paths = {
        "annotations": scene_dir / f"{visit_id}_annotations.json",
        "descriptions": scene_dir / f"{visit_id}_descriptions.json",
        "laser_scan": scene_dir / f"{visit_id}_laser_scan.ply",
        "transform": scene_dir / video_id / f"{video_id}_transform.npy",
        "object_boxes": scene_dir / video_id / f"{video_id}_3dod_annotation.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SceneFun3D inputs: {missing}")

    annotations = _load_json(source_paths["annotations"])["annotations"]
    descriptions = _load_json(source_paths["descriptions"])["descriptions"]
    xyz = _load_ply_xyz(source_paths["laser_scan"])
    transform = np.load(source_paths["transform"])
    parent_boxes = _load_parent_boxes(source_paths["object_boxes"])

    proposals: dict[str, FunctionalProposal] = {}
    unmatched_annotations: list[str] = []
    for annotation in annotations:
        if annotation["label"] == "exclude":
            continue
        points = _transform_points(
            xyz[np.asarray(annotation["indices"], dtype=np.int64)], transform
        )
        parent_match = _match_parent(points, parent_boxes)
        if parent_match is None:
            unmatched_annotations.append(annotation["annot_id"])
            continue
        parent, coverage = parent_match
        proposals[annotation["annot_id"]] = FunctionalProposal(
            candidate_id=annotation["annot_id"],
            points=points,
            center=points.mean(axis=0),
            parent=parent,
            parent_coverage=coverage,
        )

    selector = TaskRelationalFunctionalSelector()
    task_rows: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    identity_parent_violations = 0
    for description in descriptions:
        target_ids = tuple(sorted(description["annot_id"]))
        target_proposals = [proposals[target_id] for target_id in target_ids if target_id in proposals]
        if len(target_proposals) != len(target_ids):
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "reason": "NOT_EVALUABLE_PARENT_BINDING",
                    "target_count": len(target_ids),
                    "parent_bound_target_count": len(target_proposals),
                }
            )
            continue
        parent_ids = {proposal.parent.binding_id for proposal in target_proposals}
        if len(parent_ids) != 1:
            not_evaluable.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK",
                    "parent_count": len(parent_ids),
                }
            )
            continue

        parent = target_proposals[0].parent
        parent_candidates = [
            proposal
            for proposal in proposals.values()
            if proposal.parent.binding_id == parent.binding_id
        ]
        baseline_ids = _baseline_selection(parent, parent_candidates)
        decision = selector.select(
            description["description"],
            parent.binding_id,
            [
                FunctionalPartCandidate(
                    proposal.candidate_id,
                    proposal.parent.binding_id,
                    tuple(float(value) for value in proposal.center),
                )
                for proposal in parent_candidates
            ],
        )
        successor_ids = decision.selected_candidate_ids
        target_set = set(target_ids)
        baseline_set = set(baseline_ids)
        successor_set = set(successor_ids)
        if any(
            proposals[candidate_id].parent.binding_id != parent.binding_id
            for candidate_id in successor_ids
        ):
            identity_parent_violations += 1

        parent_distance = _nearest_distance(parent.center, target_proposals)
        task_rows.append(
            {
                "desc_id": description["desc_id"],
                "description": description["description"],
                "parent_binding_id": parent.binding_id,
                "parent_label": parent.label,
                "candidate_count": len(parent_candidates),
                "evaluator_target_ids": list(target_ids),
                "parent_center": [round(float(value), 6) for value in parent.center],
                "parent_center_nearest_target_m": round(parent_distance, 6),
                "parent_center_false_handoff": parent_distance > FUNCTIONAL_CONTACT_TOLERANCE_M,
                "baseline": {
                    "selected_candidate_ids": list(baseline_ids),
                    "legal_commit": bool(baseline_set) and baseline_set.issubset(target_set),
                    "target_set_recall": len(baseline_set & target_set) / len(target_set),
                    "wrong_part_count": len(baseline_set - target_set),
                },
                "task_relational": {
                    "state": decision.state.value,
                    "relation": decision.relation,
                    "selected_candidate_ids": list(successor_ids),
                    "legal_commit": bool(successor_set) and successor_set.issubset(target_set),
                    "target_set_recall": len(successor_set & target_set) / len(target_set),
                    "wrong_part_count": len(successor_set - target_set),
                    "authority": decision.authority,
                    "action": decision.action,
                },
            }
        )

    evaluable_count = len(task_rows)
    if evaluable_count:
        baseline_legal = sum(row["baseline"]["legal_commit"] for row in task_rows)
        successor_legal = sum(row["task_relational"]["legal_commit"] for row in task_rows)
        baseline_wrong = sum(row["baseline"]["wrong_part_count"] for row in task_rows)
        successor_wrong = sum(row["task_relational"]["wrong_part_count"] for row in task_rows)
        baseline_recall = float(
            np.mean([row["baseline"]["target_set_recall"] for row in task_rows])
        )
        successor_recall = float(
            np.mean([row["task_relational"]["target_set_recall"] for row in task_rows])
        )
        parent_false_handoffs = sum(row["parent_center_false_handoff"] for row in task_rows)
    else:
        baseline_legal = successor_legal = baseline_wrong = successor_wrong = 0
        baseline_recall = successor_recall = 0.0
        parent_false_handoffs = 0

    baseline_rate = baseline_legal / evaluable_count if evaluable_count else 0.0
    successor_rate = successor_legal / evaluable_count if evaluable_count else 0.0
    if evaluable_count < 5:
        decision = "SC8_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_TASKS"
    elif (
        successor_rate - baseline_rate >= 0.25
        and successor_wrong == 0
        and successor_recall >= 0.80
        and identity_parent_violations == 0
    ):
        decision = "SC8_TASK_RELATIONAL_FUNCTIONAL_BINDING_DEVELOPMENT_SIGNAL"
    else:
        decision = "SC8_TASK_RELATIONAL_FUNCTIONAL_BINDING_GATE_NOT_MET"

    return {
        "schema_version": 1,
        "experiment": "L10-SC8-SCENEFUN3D-PROPOSAL-CONDITIONAL-FUNCTIONAL-BINDING",
        "decision": decision,
        "claim_layer": "PROPOSAL_CONDITIONAL_FUNCTIONAL_PART_SELECTION_DEVELOPMENT",
        "source": {
            "dataset": "SceneFun3D v1 train",
            "visit_id": visit_id,
            "video_id": video_id,
            "official_urls": {
                "dataset": "https://scenefun3d.github.io/documentation/",
                "toolkit": "https://github.com/SceneFun3D/scenefun3d",
            },
            "sha256": {name: _sha256(path) for name, path in source_paths.items()},
        },
        "frozen_contract": {
            "parent_containment_margin_m": PARENT_CONTAINMENT_MARGIN_M,
            "parent_containment_min_fraction": PARENT_CONTAINMENT_MIN_FRACTION,
            "functional_contact_tolerance_m": FUNCTIONAL_CONTACT_TOLERANCE_M,
            "algorithm_inputs": [
                "public task description",
                "opaque exact-parent binding",
                "unlabeled functional-part proposal centroids",
            ],
            "forbidden_algorithm_inputs": [
                "SceneFun3D affordance label",
                "description annot_id target mapping",
                "evaluator target identity",
            ],
        },
        "denominators": {
            "descriptions_total": len(descriptions),
            "tasks_evaluable": evaluable_count,
            "tasks_not_evaluable": len(not_evaluable),
            "functional_annotations_parent_bound": len(proposals),
            "functional_annotations_parent_unmatched": len(unmatched_annotations),
        },
        "metrics": {
            "parent_center_false_handoff_count": parent_false_handoffs,
            "baseline_legal_commit_count": baseline_legal,
            "baseline_legal_commit_rate": baseline_rate,
            "baseline_mean_target_set_recall": baseline_recall,
            "baseline_wrong_part_count": baseline_wrong,
            "task_relational_legal_commit_count": successor_legal,
            "task_relational_legal_commit_rate": successor_rate,
            "task_relational_mean_target_set_recall": successor_recall,
            "task_relational_wrong_part_count": successor_wrong,
            "identity_parent_violations": identity_parent_violations,
        },
        "tasks": task_rows,
        "not_evaluable": not_evaluable,
        "claim_boundary": (
            "This is one SceneFun3D Development scene with evaluator-provided functional-part "
            "proposals and ARKit parent boxes. It tests task-conditioned part selection only; "
            "it does not establish proposal generation, RGB transfer, reachability, orientation, "
            "arrival, completion, product benefit, user benefit, or safety."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_ceiling(args.scene_dir.resolve(), args.video_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], **result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
