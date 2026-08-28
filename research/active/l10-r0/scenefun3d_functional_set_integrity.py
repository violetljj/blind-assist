from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from functional_part_binding import (
    FunctionalBindingDecision,
    FunctionalBindingState,
    FunctionalPartCandidate,
    TaskRelationalFunctionalSelector,
)
from scenefun3d_functional_handoff_ceiling import (
    FunctionalProposal,
    ParentBox,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
    _match_parent,
    _transform_points,
)


PARENT_CONTAINMENT_INPUT_LABEL = "AUTHORIZED_EXACT_PARENT_BINDING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntegrityDecision:
    state: FunctionalBindingState
    selected_candidate_ids: tuple[str, ...]
    quarantined_candidate_ids: tuple[str, ...]
    action: str
    integrity_opportunity: bool
    component_sizes: tuple[int, ...]


def _normalized_parent_coordinate(point: np.ndarray, parent: ParentBox) -> np.ndarray:
    local = (point - parent.center) @ parent.axes.T
    return local / np.maximum(parent.lengths, 1e-6)


def _connected_components(
    selected: list[FunctionalProposal], parent: ParentBox, link_radius: float
) -> list[list[FunctionalProposal]]:
    if not selected:
        return []
    coordinates = np.asarray(
        [_normalized_parent_coordinate(proposal.center, parent) for proposal in selected]
    )
    unseen = set(range(len(selected)))
    components: list[list[FunctionalProposal]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        indices = [seed]
        while stack:
            current = stack.pop()
            neighbors = sorted(
                index
                for index in unseen
                if float(np.linalg.norm(coordinates[current] - coordinates[index]))
                <= link_radius
            )
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
                indices.append(neighbor)
        components.append(
            sorted((selected[index] for index in indices), key=lambda row: row.candidate_id)
        )
    return sorted(
        components,
        key=lambda rows: (-len(rows), tuple(row.candidate_id for row in rows)),
    )


def apply_integrity(
    baseline: FunctionalBindingDecision,
    parent: ParentBox,
    proposals: dict[str, FunctionalProposal],
    *,
    link_radius: float,
    minimum_component_size: int,
) -> IntegrityDecision:
    selected = [proposals[candidate_id] for candidate_id in baseline.selected_candidate_ids]
    components = _connected_components(selected, parent, link_radius)
    sizes = tuple(sorted((len(component) for component in components), reverse=True))
    if len(selected) <= 1 or len(components) <= 1:
        return IntegrityDecision(
            baseline.state,
            baseline.selected_candidate_ids,
            (),
            baseline.action,
            False,
            sizes,
        )

    largest_size = len(components[0])
    tied = sum(len(component) == largest_size for component in components) > 1
    if largest_size < minimum_component_size or tied:
        return IntegrityDecision(
            FunctionalBindingState.SET_VALUED,
            baseline.selected_candidate_ids,
            (),
            "REQUEST_INTEGRITY_VIEW",
            False,
            sizes,
        )

    retained = tuple(sorted(row.candidate_id for row in components[0]))
    retained_set = set(retained)
    quarantined = tuple(
        sorted(candidate_id for candidate_id in baseline.selected_candidate_ids if candidate_id not in retained_set)
    )
    return IntegrityDecision(
        FunctionalBindingState.UNIQUE if len(retained) == 1 else FunctionalBindingState.SET_VALUED,
        retained,
        quarantined,
        "PASS_COHERENT_FUNCTIONAL_SET_TO_GEOMETRY",
        bool(quarantined),
        sizes,
    )


def _source_paths(data_root: Path, visit_id: str, video_id: str) -> dict[str, Path]:
    scene_dir = data_root / visit_id
    return {
        "annotations": scene_dir / f"{visit_id}_annotations.json",
        "descriptions": scene_dir / f"{visit_id}_descriptions.json",
        "laser_scan": scene_dir / f"{visit_id}_laser_scan.ply",
        "transform": scene_dir / video_id / f"{video_id}_transform.npy",
        "object_boxes": scene_dir / video_id / f"{video_id}_3dod_annotation.json",
    }


def _verify_source(paths: dict[str, Path], expected: dict[str, str]) -> None:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SceneFun3D inputs: {missing}")
    actual = {name: _sha256(path) for name, path in paths.items()}
    if actual != expected:
        raise ValueError(f"Source hash mismatch: expected={expected}, actual={actual}")


def _build_proposals(paths: dict[str, Path]) -> tuple[dict[str, FunctionalProposal], int]:
    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = np.load(paths["transform"])
    parents = _load_parent_boxes(paths["object_boxes"])
    proposals: dict[str, FunctionalProposal] = {}
    unmatched = 0
    for annotation in _load_json(paths["annotations"])["annotations"]:
        if annotation["label"] == "exclude":
            continue
        points = _transform_points(
            xyz[np.asarray(annotation["indices"], dtype=np.int64)], transform
        )
        parent_match = _match_parent(points, parents)
        if parent_match is None:
            unmatched += 1
            continue
        parent, coverage = parent_match
        proposals[annotation["annot_id"]] = FunctionalProposal(
            candidate_id=annotation["annot_id"],
            points=points,
            center=points.mean(axis=0),
            parent=parent,
            parent_coverage=coverage,
        )
    return proposals, unmatched


def build_provider(data_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    selector = TaskRelationalFunctionalSelector()
    link_radius = float(protocol["frozen_algorithm"]["normalized_component_link_radius"])
    minimum_component_size = int(protocol["frozen_algorithm"]["minimum_dominant_component_size"])
    scenes: list[dict[str, Any]] = []
    for cohort in protocol["cohort"]:
        visit_id = cohort["visit_id"]
        video_id = cohort["video_id"]
        paths = _source_paths(data_root, visit_id, video_id)
        _verify_source(paths, cohort["sha256"])
        proposals, unmatched = _build_proposals(paths)
        tasks: list[dict[str, Any]] = []
        not_evaluable: list[dict[str, Any]] = []
        for description in _load_json(paths["descriptions"])["descriptions"]:
            target_proposals = [
                proposals[target_id]
                for target_id in description["annot_id"]
                if target_id in proposals
            ]
            if len(target_proposals) != len(description["annot_id"]):
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_PARENT_BINDING"}
                )
                continue
            parent_ids = {proposal.parent.binding_id for proposal in target_proposals}
            if len(parent_ids) != 1:
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_MULTI_PARENT_TASK"}
                )
                continue
            parent = target_proposals[0].parent
            parent_candidates = {
                candidate_id: proposal
                for candidate_id, proposal in proposals.items()
                if proposal.parent.binding_id == parent.binding_id
            }
            baseline = selector.select(
                description["description"],
                parent.binding_id,
                [
                    FunctionalPartCandidate(
                        proposal.candidate_id,
                        parent.binding_id,
                        tuple(float(value) for value in proposal.center),
                    )
                    for proposal in parent_candidates.values()
                ],
            )
            successor = apply_integrity(
                baseline,
                parent,
                parent_candidates,
                link_radius=link_radius,
                minimum_component_size=minimum_component_size,
            )
            tasks.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "parent_binding_id": parent.binding_id,
                    "parent_binding_authority": PARENT_CONTAINMENT_INPUT_LABEL,
                    "candidate_ids": sorted(parent_candidates),
                    "baseline": {
                        "state": baseline.state.value,
                        "relation": baseline.relation,
                        "selected_candidate_ids": list(baseline.selected_candidate_ids),
                        "action": baseline.action,
                    },
                    "successor": {
                        "state": successor.state.value,
                        "selected_candidate_ids": list(successor.selected_candidate_ids),
                        "quarantined_candidate_ids": list(successor.quarantined_candidate_ids),
                        "action": successor.action,
                        "integrity_opportunity": successor.integrity_opportunity,
                        "component_sizes": list(successor.component_sizes),
                    },
                }
            )
        scenes.append(
            {
                "visit_id": visit_id,
                "video_id": video_id,
                "source_sha256": cohort["sha256"],
                "functional_annotations_parent_bound": len(proposals),
                "functional_annotations_parent_unmatched": unmatched,
                "tasks": tasks,
                "not_evaluable": not_evaluable,
            }
        )
    return {
        "schema_version": 1,
        "provider": "L10-SC21-TASK-FUNCTIONAL-SET-INTEGRITY-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
        "truth_isolation": (
            "Target IDs are used only to derive the already-authorized exact parent binding. "
            "The baseline and integrity successor receive task text, opaque parent binding, parent geometry, "
            "and unlabeled proposal centers; evaluator target membership is not passed to either selector."
        ),
        "frozen_algorithm": protocol["frozen_algorithm"],
        "scenes": scenes,
    }


def _score(selected: list[str], target_set: set[str]) -> dict[str, Any]:
    selected_set = set(selected)
    return {
        "legal_commit": bool(selected_set) and selected_set.issubset(target_set),
        "target_set_recall": len(selected_set & target_set) / len(target_set),
        "wrong_part_count": len(selected_set - target_set),
    }


def evaluate_provider(
    data_root: Path,
    protocol: dict[str, Any],
    provider: dict[str, Any],
    provider_sha256: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cross_parent_violations = 0
    for scene in provider["scenes"]:
        visit_id = scene["visit_id"]
        video_id = scene["video_id"]
        paths = _source_paths(data_root, visit_id, video_id)
        descriptions = {
            row["desc_id"]: row
            for row in _load_json(paths["descriptions"])["descriptions"]
        }
        for task in scene["tasks"]:
            target_set = set(descriptions[task["desc_id"]]["annot_id"])
            candidate_set = set(task["candidate_ids"])
            baseline_selected = task["baseline"]["selected_candidate_ids"]
            successor_selected = task["successor"]["selected_candidate_ids"]
            if not set(successor_selected).issubset(candidate_set):
                cross_parent_violations += 1
            rows.append(
                {
                    "visit_id": visit_id,
                    "desc_id": task["desc_id"],
                    "description": task["description"],
                    "baseline": _score(baseline_selected, target_set),
                    "successor": _score(successor_selected, target_set),
                    "integrity_opportunity": task["successor"]["integrity_opportunity"],
                    "quarantined_count": len(task["successor"]["quarantined_candidate_ids"]),
                    "component_sizes": task["successor"]["component_sizes"],
                }
            )

    def aggregate(arm: str) -> dict[str, Any]:
        legal = sum(row[arm]["legal_commit"] for row in rows)
        wrong = sum(row[arm]["wrong_part_count"] for row in rows)
        mean_recall = float(np.mean([row[arm]["target_set_recall"] for row in rows])) if rows else 0.0
        return {
            "legal_commit_count": legal,
            "legal_commit_rate": legal / len(rows) if rows else 0.0,
            "mean_target_set_recall": mean_recall,
            "wrong_part_count": wrong,
        }

    baseline = aggregate("baseline")
    successor = aggregate("successor")
    opportunities = sum(row["integrity_opportunity"] for row in rows)
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluable_tasks"]):
        decision = "SC21_NOT_EVALUABLE_INSUFFICIENT_PARENT_BOUND_TASKS"
    elif opportunities < int(gate["minimum_integrity_opportunities"]):
        decision = "SC21_NOT_EVALUABLE_INSUFFICIENT_INTEGRITY_OPPORTUNITIES"
    elif (
        baseline["wrong_part_count"] - successor["wrong_part_count"]
        >= int(gate["minimum_wrong_part_reduction"])
        and successor["legal_commit_count"] >= baseline["legal_commit_count"]
        and successor["mean_target_set_recall"] >= baseline["mean_target_set_recall"]
        and cross_parent_violations <= int(gate["maximum_cross_parent_violations"])
    ):
        decision = "SC21_FUNCTIONAL_SET_INTEGRITY_DEVELOPMENT_SIGNAL"
    else:
        decision = "SC21_FUNCTIONAL_SET_INTEGRITY_GATE_NOT_MET"
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "claim_layer": "PRIVILEGED_PROPOSAL_CONDITIONAL_FUNCTIONAL_SET_INTEGRITY_DEVELOPMENT",
        "protocol_sha256": protocol["protocol_sha256"],
        "provider_sha256": provider_sha256,
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "scenes": len(provider["scenes"]),
            "tasks_evaluable": len(rows),
            "tasks_not_evaluable": sum(len(scene["not_evaluable"]) for scene in provider["scenes"]),
            "integrity_opportunities": opportunities,
        },
        "baseline": baseline,
        "successor": successor,
        "wrong_part_reduction": baseline["wrong_part_count"] - successor["wrong_part_count"],
        "cross_parent_violations": cross_parent_violations,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _load_json(args.protocol)
    protocol["protocol_sha256"] = _sha256(args.protocol)
    provider = build_provider(args.data_root.resolve(), protocol)
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)
    result = evaluate_provider(
        args.data_root.resolve(), protocol, provider, provider_sha256
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
                "wrong_part_reduction": result["wrong_part_reduction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
