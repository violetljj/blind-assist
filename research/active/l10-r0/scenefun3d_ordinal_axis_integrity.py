from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from functional_part_binding import FunctionalBindingState, FunctionalPartCandidate, TaskRelationalFunctionalSelector
from scenefun3d_functional_handoff_ceiling import FunctionalProposal, ParentBox, _load_json
from scenefun3d_functional_set_integrity import _score, _sha256, apply_integrity
from scenefun3d_semantic_topology_integrity import infer_task_action_witness


ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
}


@dataclass(frozen=True)
class OrdinalAxis:
    slot_by_candidate: dict[str, int]
    pitch: float
    maximum_pitch_relative_deviation: float
    maximum_orthogonal_residual_pitch_ratio: float
    boundary_gap_pitch_ratio: float
    opposite_boundary_gap_pitch_ratio: float
    inferred_hidden_slots: int
    anchored_endpoint: str


@dataclass(frozen=True)
class UnorientedOrdinalAxis:
    ordered_candidate_ids: tuple[str, ...]
    pitch: float
    maximum_pitch_relative_deviation: float
    maximum_orthogonal_residual_pitch_ratio: float


def parse_ordinal(text: str) -> int | None:
    normalized = " ".join(text.casefold().split())
    matches = [
        value
        for token, value in ORDINALS.items()
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized)
    ]
    return matches[0] if len(set(matches)) == 1 else None


def has_explicit_axis_direction(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return bool(re.search(r"\bfrom (?:the )?(?:left|right|top|bottom)\b", normalized))


def _ray_to_parent_boundary(point: np.ndarray, direction: np.ndarray) -> float:
    distances: list[float] = []
    for coordinate, component in zip(point, direction):
        if component > 1e-9:
            distances.append(float((0.5 - coordinate) / component))
        elif component < -1e-9:
            distances.append(float((-0.5 - coordinate) / component))
    positive = [value for value in distances if value >= 0.0]
    return min(positive) if positive else float("inf")


def fit_ordinal_axis(
    parent: ParentBox,
    candidates: dict[str, FunctionalProposal],
    algorithm: dict[str, Any],
) -> OrdinalAxis | None:
    minimum = int(algorithm["minimum_same_action_candidates"])
    if len(candidates) < minimum:
        return None
    candidate_ids = sorted(candidates)
    coordinates = np.asarray(
        [
            ((candidates[candidate_id].center - parent.center) @ parent.axes.T)
            / np.maximum(parent.lengths, 1e-9)
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    centered = coordinates - coordinates.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    projections = centered @ axis
    order = np.argsort(projections, kind="stable")
    ordered_projections = projections[order]
    gaps = np.diff(ordered_projections)
    pitch = float(np.median(gaps)) if len(gaps) else 0.0
    if pitch <= float(algorithm["minimum_normalized_pitch"]):
        return None
    maximum_pitch_deviation = float(np.max(np.abs(gaps - pitch)) / pitch)
    reconstructed = np.outer(projections, axis)
    maximum_residual_ratio = float(
        np.max(np.linalg.norm(centered - reconstructed, axis=1)) / pitch
    )
    if maximum_pitch_deviation > float(algorithm["maximum_pitch_relative_deviation"]):
        return None
    if maximum_residual_ratio > float(
        algorithm["maximum_orthogonal_residual_pitch_ratio"]
    ):
        return None

    low_index = int(order[0])
    high_index = int(order[-1])
    low_gap = _ray_to_parent_boundary(coordinates[low_index], -axis)
    high_gap = _ray_to_parent_boundary(coordinates[high_index], axis)
    near_gap, far_gap = (low_gap, high_gap) if low_gap <= high_gap else (high_gap, low_gap)
    if not np.isfinite(near_gap) or far_gap <= 0.0:
        return None
    if near_gap / far_gap > float(algorithm["maximum_boundary_gap_ratio"]):
        return None
    hidden_float = near_gap / pitch - 0.5
    hidden = max(0, int(round(hidden_float)))
    if hidden > int(algorithm["maximum_inferred_hidden_slots"]):
        return None
    if abs(hidden_float - hidden) > float(
        algorithm["maximum_hidden_slot_rounding_residual"]
    ):
        return None

    anchored_order = list(order) if low_gap <= high_gap else list(reversed(order))
    slots = {
        candidate_ids[int(index)]: hidden + rank + 1
        for rank, index in enumerate(anchored_order)
    }
    return OrdinalAxis(
        slot_by_candidate=slots,
        pitch=pitch,
        maximum_pitch_relative_deviation=maximum_pitch_deviation,
        maximum_orthogonal_residual_pitch_ratio=maximum_residual_ratio,
        boundary_gap_pitch_ratio=near_gap / pitch,
        opposite_boundary_gap_pitch_ratio=far_gap / pitch,
        inferred_hidden_slots=hidden,
        anchored_endpoint="LOW_PCA_ENDPOINT" if low_gap <= high_gap else "HIGH_PCA_ENDPOINT",
    )


def fit_unoriented_ordinal_axis(
    parent: ParentBox,
    candidates: dict[str, FunctionalProposal],
    algorithm: dict[str, Any],
) -> UnorientedOrdinalAxis | None:
    minimum = int(algorithm["minimum_same_action_candidates"])
    if len(candidates) < minimum:
        return None
    candidate_ids = sorted(candidates)
    coordinates = np.asarray(
        [
            ((candidates[candidate_id].center - parent.center) @ parent.axes.T)
            / np.maximum(parent.lengths, 1e-9)
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    centered = coordinates - coordinates.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    projections = centered @ axis
    order = np.argsort(projections, kind="stable")
    gaps = np.diff(projections[order])
    pitch = float(np.median(gaps)) if len(gaps) else 0.0
    if pitch <= float(algorithm["minimum_normalized_pitch"]):
        return None
    maximum_pitch_deviation = float(np.max(np.abs(gaps - pitch)) / pitch)
    reconstructed = np.outer(projections, axis)
    maximum_residual_ratio = float(
        np.max(np.linalg.norm(centered - reconstructed, axis=1)) / pitch
    )
    if maximum_pitch_deviation > float(algorithm["maximum_pitch_relative_deviation"]):
        return None
    if maximum_residual_ratio > float(
        algorithm["maximum_orthogonal_residual_pitch_ratio"]
    ):
        return None
    return UnorientedOrdinalAxis(
        ordered_candidate_ids=tuple(candidate_ids[int(index)] for index in order),
        pitch=pitch,
        maximum_pitch_relative_deviation=maximum_pitch_deviation,
        maximum_orthogonal_residual_pitch_ratio=maximum_residual_ratio,
    )


def _parent_from_row(row: dict[str, Any]) -> ParentBox:
    return ParentBox(
        binding_id=row["parent_binding_id"],
        label=row["parent_label"],
        center=np.asarray(row["parent_center"], dtype=np.float64),
        lengths=np.asarray(row["parent_lengths"], dtype=np.float64),
        axes=np.asarray(row["parent_axes"], dtype=np.float64),
    )


def _proposal(candidate_id: str, row: dict[str, Any], parent: ParentBox) -> FunctionalProposal:
    center = np.asarray(row["center"], dtype=np.float64)
    return FunctionalProposal(candidate_id, np.empty((0, 3)), center, parent, float(row["parent_coverage"]))


def _sc31_selection(
    description: str,
    parent: ParentBox,
    candidates: dict[str, FunctionalProposal],
    labels: dict[str, str],
    algorithm: dict[str, Any],
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    selector = TaskRelationalFunctionalSelector()
    requested, witness = infer_task_action_witness(
        description, "EXACT_THEN_REDUNDANCY_GATED_ACTION_FAMILY"
    )
    compatible = {
        candidate_id: proposal
        for candidate_id, proposal in candidates.items()
        if labels[candidate_id] in requested
    }
    redundancy = witness != "FAMILY" or len(compatible) >= 2
    admitted = bool(requested and compatible and redundancy)
    semantic = compatible if admitted else candidates
    relational = selector.select(
        description,
        parent.binding_id,
        [
            FunctionalPartCandidate(
                candidate_id,
                parent.binding_id,
                tuple(float(value) for value in proposal.center),
            )
            for candidate_id, proposal in semantic.items()
        ],
    )
    integrity = apply_integrity(
        relational,
        parent,
        semantic,
        link_radius=float(algorithm["normalized_component_link_radius"]),
        minimum_component_size=int(algorithm["minimum_dominant_component_size"]),
    )
    return integrity.selected_candidate_ids, admitted, requested


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
        description_path = data_root / visit_id / f"{visit_id}_descriptions.json"
        descriptions = _load_json(description_path)["descriptions"]
        source_row = source_by_visit[visit_id]
        parent_rows = {
            row["parent_binding_id"]: row for row in source_row["ordinal_lattice_parents"]
        }
        candidate_to_parent = {
            candidate_id: parent_id
            for parent_id, row in parent_rows.items()
            for candidate_id in row["candidates"]
        }
        tasks: list[dict[str, Any]] = []
        not_evaluable: list[dict[str, Any]] = []
        for description in descriptions:
            requested_ordinal = parse_ordinal(description["description"])
            if requested_ordinal is None or has_explicit_axis_direction(description["description"]):
                continue
            target_parent_ids = {
                candidate_to_parent[target_id]
                for target_id in description["annot_id"]
                if target_id in candidate_to_parent
            }
            if len(target_parent_ids) != 1:
                not_evaluable.append({"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_ORDINAL_PARENT_BINDING"})
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
            ordinal_mode = algorithm.get("ordinal_mode", "BOUNDARY_ANCHORED_ABSOLUTE")
            if ordinal_mode == "ORIENTATION_QUOTIENT_COMPLETE_INVENTORY":
                ordered = list(parent_row["axis_diagnostics"]["ordered_candidate_ids"])
                if len(ordered) == len(compatible) and set(ordered) == set(compatible):
                    forward_index = requested_ordinal - 1
                    reverse_index = len(ordered) - requested_ordinal
                    if 0 <= forward_index < len(ordered):
                        ordinal_matches = sorted(
                            {ordered[forward_index], ordered[reverse_index]}
                        )
                    else:
                        ordinal_matches = []
                else:
                    ordinal_matches = []
                axis_slots = {
                    "polarity_hypotheses": ordinal_matches,
                    "ordered_candidate_ids_up_to_reversal": ordered,
                }
            else:
                axis = fit_ordinal_axis(parent, compatible, algorithm)
                ordinal_matches = [] if axis is None else [
                    candidate_id
                    for candidate_id, slot in axis.slot_by_candidate.items()
                    if slot == requested_ordinal
                ]
                axis_slots = {} if axis is None else axis.slot_by_candidate
            ordinal_admitted = semantic_admitted and len(ordinal_matches) == 1
            successor = tuple(ordinal_matches) if ordinal_admitted else baseline
            tasks.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "parent_binding_id": parent.binding_id,
                    "requested_ordinal": requested_ordinal,
                    "semantic_admitted": semantic_admitted,
                    "ordinal_admitted": ordinal_admitted,
                    "ordinal_slots": axis_slots,
                    "baseline_selected_candidate_ids": list(baseline),
                    "successor_selected_candidate_ids": list(successor),
                }
            )
        scenes.append({"visit_id": visit_id, "video_id": source["video_id"], "tasks": tasks, "not_evaluable": not_evaluable})
    return {
        "schema_version": 1,
        "provider": "L10-SC33-BOUNDARY-ANCHORED-ORDINAL-AXIS-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort_protocol_sha256": cohort["protocol_sha256"],
        "source_admission_result_sha256": source_result["result_sha256"],
        "truth_isolation": "Source admission sealed task text plus provider-public proposal geometry without reading annot_id. Target IDs are opened only here to recover the already-authorized exact parent and are not passed into ordinal fitting.",
        "scenes": scenes,
    }


def evaluate_provider(protocol: dict[str, Any], provider: dict[str, Any], provider_hash: str, data_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in provider["scenes"]:
        descriptions = {
            row["desc_id"]: row
            for row in _load_json(data_root / scene["visit_id"] / f"{scene['visit_id']}_descriptions.json")["descriptions"]
        }
        for task in scene["tasks"]:
            target = set(descriptions[task["desc_id"]]["annot_id"])
            rows.append(
                {
                    "visit_id": scene["visit_id"],
                    "desc_id": task["desc_id"],
                    "description": task["description"],
                    "requested_ordinal": task["requested_ordinal"],
                    "ordinal_admitted": task["ordinal_admitted"],
                    "baseline": _score(task["baseline_selected_candidate_ids"], target),
                    "successor": _score(task["successor_selected_candidate_ids"], target),
                }
            )

    def aggregate(arm: str) -> dict[str, Any]:
        count = len(rows)
        return {
            "legal_commit_count": sum(row[arm]["legal_commit"] for row in rows),
            "legal_commit_rate": sum(row[arm]["legal_commit"] for row in rows) / count if count else 0.0,
            "mean_target_set_recall": float(np.mean([row[arm]["target_set_recall"] for row in rows])) if rows else 0.0,
            "wrong_part_count": sum(row[arm]["wrong_part_count"] for row in rows),
        }
    baseline = aggregate("baseline")
    successor = aggregate("successor")
    admissions = sum(row["ordinal_admitted"] for row in rows)
    regressions = sum(
        row["successor"]["target_set_recall"] < row["baseline"]["target_set_recall"]
        or row["successor"]["wrong_part_count"] > row["baseline"]["wrong_part_count"]
        for row in rows
    )
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluable_tasks"]):
        decision = protocol["decision_labels"]["insufficient_tasks"]
    elif admissions < int(gate["minimum_ordinal_admissions"]):
        decision = protocol["decision_labels"]["insufficient_admissions"]
    elif (
        successor["legal_commit_count"] - baseline["legal_commit_count"] >= int(gate["minimum_legal_commit_gain"])
        and baseline["wrong_part_count"] - successor["wrong_part_count"] >= int(gate["minimum_wrong_part_reduction"])
        and successor["mean_target_set_recall"] >= baseline["mean_target_set_recall"]
        and regressions == 0
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
            "ordinal_admissions": admissions,
            "taskwise_regressions": regressions,
        },
        "baseline": baseline,
        "successor": successor,
        "legal_commit_gain": successor["legal_commit_count"] - baseline["legal_commit_count"],
        "wrong_part_reduction": baseline["wrong_part_count"] - successor["wrong_part_count"],
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
    print(json.dumps({key: result[key] for key in ("decision", "denominators", "baseline", "successor", "legal_commit_gain", "wrong_part_reduction")}, indent=2))


if __name__ == "__main__":
    main()
