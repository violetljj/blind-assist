from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenefun3d_active_view_ordinal import parse_directional_ordinal
from scenefun3d_functional_handoff_ceiling import _load_json
from scenefun3d_functional_set_integrity import _score, _sha256


@dataclass(frozen=True)
class MetricLattice:
    ordered_candidate_ids_up_to_reversal: tuple[str, ...]
    pitch_m: float
    maximum_pitch_relative_deviation: float
    maximum_orthogonal_residual_pitch_ratio: float


def fit_metric_lattice(
    centers: dict[str, np.ndarray], algorithm: dict[str, Any]
) -> MetricLattice | None:
    if len(centers) < int(algorithm["minimum_same_action_candidates"]):
        return None
    candidate_ids = sorted(centers)
    coordinates = np.asarray([centers[candidate_id] for candidate_id in candidate_ids])
    centered = coordinates - coordinates.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    projections = centered @ axis
    order = np.argsort(projections, kind="stable")
    gaps = np.diff(projections[order])
    pitch = float(np.median(gaps)) if len(gaps) else 0.0
    if pitch < float(algorithm["minimum_metric_pitch_m"]) or pitch > float(
        algorithm["maximum_metric_pitch_m"]
    ):
        return None
    pitch_deviation = float(np.max(np.abs(gaps - pitch)) / pitch)
    reconstructed = np.outer(projections, axis)
    residual_ratio = float(
        np.max(np.linalg.norm(centered - reconstructed, axis=1)) / pitch
    )
    if pitch_deviation > float(algorithm["maximum_pitch_relative_deviation"]):
        return None
    if residual_ratio > float(algorithm["maximum_orthogonal_residual_pitch_ratio"]):
        return None
    return MetricLattice(
        ordered_candidate_ids_up_to_reversal=tuple(
            candidate_ids[int(index)] for index in order
        ),
        pitch_m=pitch,
        maximum_pitch_relative_deviation=pitch_deviation,
        maximum_orthogonal_residual_pitch_ratio=residual_ratio,
    )


def connected_components(
    centers: dict[str, np.ndarray], link_radius: float
) -> list[tuple[str, ...]]:
    unseen = set(centers)
    components: list[tuple[str, ...]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            neighbors = sorted(
                candidate_id
                for candidate_id in unseen
                if float(np.linalg.norm(centers[current] - centers[candidate_id]))
                <= link_radius
            )
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(tuple(sorted(component)))
    return sorted(components)


def build_provider(
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    source_result: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    source_by_visit = {row["visit_id"]: row for row in source_result["selected"]}
    scenes: list[dict[str, Any]] = []
    for source in cohort["cohort"]:
        visit_id = source["visit_id"]
        descriptions = _load_json(
            data_root / visit_id / f"{visit_id}_descriptions.json"
        )["descriptions"]
        clusters = {
            row["cluster_id"]: row
            for row in source_by_visit[visit_id]["contextual_lattices"]
        }
        candidate_to_clusters: dict[str, set[str]] = {}
        for cluster_id, cluster in clusters.items():
            for candidate_id in cluster["candidates"]:
                candidate_to_clusters.setdefault(candidate_id, set()).add(cluster_id)
        tasks: list[dict[str, Any]] = []
        not_evaluable: list[dict[str, Any]] = []
        for description in descriptions:
            directional = parse_directional_ordinal(description["description"])
            carrier_mode = protocol["frozen_algorithm"].get(
                "carrier_mode", "CONTEXT_ANCHOR_OBB"
            )
            if directional is None or (
                carrier_mode != "SELF_CARRIER_LOCAL_ACTION_LATTICE"
                and "door" not in description["description"].casefold()
            ):
                continue
            target_clusters = set().union(
                *(candidate_to_clusters.get(target_id, set()) for target_id in description["annot_id"])
            )
            if len(target_clusters) != 1:
                not_evaluable.append(
                    {"desc_id": description["desc_id"], "reason": "NOT_EVALUABLE_CONTEXTUAL_LATTICE_BINDING"}
                )
                continue
            cluster = clusters[next(iter(target_clusters))]
            ordered = list(cluster["active_view"]["ordered_candidate_ids_left_to_right"])
            inventories = cluster.get("directional_ordinal_inventories", {})
            inventory = inventories.get(directional.direction)
            if inventory is None:
                ordinal_position = directional.ordinal - 1
            else:
                ordinal_position = (
                    inventory.index(directional.ordinal)
                    if directional.ordinal in inventory
                    else -1
                )
            index = ordinal_position if directional.direction == "FROM_LEFT" else len(ordered) - 1 - ordinal_position
            successor = [ordered[index]] if 0 <= index < len(ordered) else list(cluster["candidates"])
            tasks.append(
                {
                    "desc_id": description["desc_id"],
                    "description": description["description"],
                    "cluster_id": cluster["cluster_id"],
                    "context_anchor_id": cluster["context_anchor_id"],
                    "requested_ordinal": directional.ordinal,
                    "direction": directional.direction,
                    "directional_ordinal_inventory": inventory,
                    "hidden_prefix_slots": None if inventory is None else inventory[0] - 1,
                    "active_view_timestamp": cluster["active_view"]["timestamp"],
                    "ordered_candidate_ids_left_to_right": ordered,
                    "baseline_selected_candidate_ids": sorted(cluster["candidates"]),
                    "successor_selected_candidate_ids": successor,
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
        "provider": "L10-SC36-CONTEXTUAL-LATTICE-ACTIVE-VIEW-ORDINAL-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
        "cohort_protocol_sha256": cohort["protocol_sha256"],
        "source_admission_result_sha256": source_result["result_sha256"],
        "truth_isolation": "Source admission seals public directional/context text and ordinal inventory, all privileged functional centers, an optional contextual 3D anchor, and a target-independent real camera observation with any protocol-required visibility authority. Target IDs are opened only here to determine whether the sealed contextual lattice contains the target.",
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
                    "baseline": _score(task["baseline_selected_candidate_ids"], target),
                    "successor": _score(task["successor_selected_candidate_ids"], target),
                }
            )
    baseline_legal = sum(row["baseline"]["legal_commit"] for row in rows)
    successor_legal = sum(row["successor"]["legal_commit"] for row in rows)
    baseline_wrong = sum(row["baseline"]["wrong_part_count"] for row in rows)
    successor_wrong = sum(row["successor"]["wrong_part_count"] for row in rows)
    baseline_recall = float(np.mean([row["baseline"]["target_set_recall"] for row in rows])) if rows else 0.0
    successor_recall = float(np.mean([row["successor"]["target_set_recall"] for row in rows])) if rows else 0.0
    regressions = sum(
        row["successor"]["target_set_recall"] < row["baseline"]["target_set_recall"]
        or row["successor"]["wrong_part_count"] > row["baseline"]["wrong_part_count"]
        for row in rows
    )
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluable_tasks"]):
        decision = protocol["decision_labels"]["insufficient_tasks"]
    elif (
        successor_legal - baseline_legal >= int(gate["minimum_legal_commit_gain"])
        and baseline_wrong - successor_wrong >= int(gate["minimum_wrong_part_reduction"])
        and successor_recall >= baseline_recall
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
