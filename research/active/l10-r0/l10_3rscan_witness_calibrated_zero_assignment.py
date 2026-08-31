#!/usr/bin/env python3
"""Confirm witness-calibrated zero assignment on a fresh 3RScan family."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as base  # noqa: E402
import l10_3rscan_open_roster_zero_assignment as open_zero  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402
import l10_3rscan_registered_surface_zero_assignment as surface  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-witness-calibrated-zero-assignment-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-witness-calibrated-zero-assignment-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-witness-calibrated-zero-assignment-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = base.load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    return protocol


def validate_dependencies(protocol: dict[str, Any], artifact_root: Path) -> Path:
    data_root = artifact_root / protocol["source"]["dataset_relative_path"]
    require(data_root.is_dir(), "3RSCAN_DATA_ROOT_MISSING")
    base.verify_path(
        HERE / protocol["source"]["candidate_protocol_path"],
        protocol["source"]["candidate_protocol_sha256"],
        "CANDIDATE_PROTOCOL",
    )
    predecessor = protocol["predecessor"]
    for key in ("protocol", "cohort", "result", "implementation"):
        base.verify_path(
            HERE / predecessor[f"{key}_path"],
            predecessor[f"{key}_sha256"],
            "REGISTERED_SURFACE_PREDECESSOR",
        )
    predecessor_result = base.load_json(HERE / predecessor["result_path"])
    require(
        predecessor_result.get("conclusion") == predecessor["required_conclusion"],
        "REGISTERED_SURFACE_PREDECESSOR_CONCLUSION",
    )
    return data_root


def consumed_physical_targets(protocol: dict[str, Any]) -> tuple[set[tuple[str, int]], list[dict[str, Any]]]:
    consumed: set[tuple[str, int]] = set()
    receipts: list[dict[str, Any]] = []
    for record in protocol["source"]["consumed_target_cohorts"]:
        path = HERE / record["path"]
        base.verify_path(path, record["sha256"], "CONSUMED_COHORT")
        cohort = base.load_json(path)
        before = len(consumed)
        for episode in cohort.get("episodes", []):
            if "reference_scan_id" in episode and "target_instance_id" in episode:
                consumed.add(
                    (str(episode["reference_scan_id"]), int(episode["target_instance_id"]))
                )
        receipts.append(
            {
                "path": path.name,
                "sha256": record["sha256"],
                "new_physical_targets": len(consumed) - before,
            }
        )
    return consumed, receipts


def group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = (str(row["reference_scan_id"]), str(row["rescan_id"]))
        grouped.setdefault(key, []).append(row)
    return list(grouped.values())


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    data_root = validate_dependencies(protocol, artifact_root)
    consumed, exclusion_receipts = consumed_physical_targets(protocol)
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    target_count = int(protocol["frozen_cohort"]["physical_targets"])
    selected: list[dict[str, Any]] | None = None
    selected_reference_points: dict[int, np.ndarray] = {}
    selected_query_points: dict[int, np.ndarray] = {}
    families_considered = 0
    for family in group_rows(rows):
        reference_scan = str(family[0]["reference_scan_id"])
        rescan = str(family[0]["rescan_id"])
        candidates = []
        used: set[int] = set()
        for row in family:
            target_id = int(row["target_instance_id"])
            if (reference_scan, target_id) in consumed or target_id in used:
                continue
            candidates.append(row)
            used.add(target_id)
        if len(candidates) < target_count:
            continue
        families_considered += 1
        target_ids = {int(row["target_instance_id"]) for row in candidates}
        reference_points = extent.ply_instance_points(
            data_root / reference_scan / "labels.instances.annotated.v2.ply", target_ids
        )
        query_points = extent.ply_instance_points(
            data_root / rescan / "labels.instances.annotated.v2.ply", target_ids
        )
        admissible = [
            row
            for row in candidates
            if len(reference_points.get(int(row["target_instance_id"]), [])) >= 4
            and len(query_points.get(int(row["target_instance_id"]), [])) >= 4
        ]
        if len(admissible) < target_count:
            continue
        selected = admissible[:target_count]
        selected_reference_points = reference_points
        selected_query_points = query_points
        break
    require(selected is not None, "FRESH_WITNESS_FAMILY_SOURCE_NOT_EVALUABLE")
    reference_scan = str(selected[0]["reference_scan_id"])
    rescan = str(selected[0]["rescan_id"])
    matrices = [extent.provider_matrix(row["transform"]) for row in selected]
    require(all(np.allclose(matrices[0], matrix, atol=1e-8) for matrix in matrices[1:]), "FAMILY_TRANSFORM_CONFLICT")
    episodes = []
    for index, row in enumerate(selected, 1):
        target_id = int(row["target_instance_id"])
        episodes.append(
            {
                "episode_id": f"WZ{index:02d}",
                **row,
                "reference_target_vertices": len(selected_reference_points[target_id]),
                "query_target_vertices": len(selected_query_points[target_id]),
                "role": "REGISTRATION_WITNESS" if index <= 2 else "EVALUATED_TARGET",
            }
        )
    source_manifest = {}
    for scan_id in (reference_scan, rescan):
        for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply"):
            source_manifest[f"{scan_id}/{name}"] = base.source_record(
                data_root / scan_id / name, artifact_root
            )
    for name in ("3RScan.json", "objects.json"):
        source_manifest[name] = base.source_record(data_root / name, artifact_root)
    episode_ids = [episode["episode_id"] for episode in episodes]
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_REPLAY_PHYSICAL_TARGET_DISJOINT_WITNESS_CALIBRATED_SURFACE_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_physical_targets": len(consumed),
            "families_considered_with_four_candidates": families_considered,
            "reference_scan_id": reference_scan,
            "rescan_id": rescan,
            "physical_target_ids": [int(row["target_instance_id"]) for row in selected],
            "witness_episode_ids": episode_ids[:2],
            "evaluated_episode_ids": episode_ids[2:],
            "rgb_members_opened": 0,
            "depth_members_opened": 0,
            "surface_distances_opened": 0,
            "exclusion_receipts": exclusion_receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": episodes,
        "provider_transform": selected[0]["transform"],
        "scenarios": open_zero.scenario_records(episode_ids),
        "counts": {
            "physical_targets": len(episodes),
            "registration_witnesses": 2,
            "evaluated_targets": 2,
            "scenarios": 4,
            "truth_matches_across_scenarios": 12,
            "truth_unmatched_nodes_across_scenarios": 4,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(cohort_path, cohort)


def witness_zero_assignment(matrix: np.ndarray, witness_ceiling: float) -> list[tuple[int, int]]:
    matches = open_zero.reciprocal_zero_assignment(matrix)
    return [
        (row, column)
        for row, column in matches
        if -float(matrix[row, column]) <= witness_ceiling
    ]


def replay(
    protocol_path: Path,
    cohort_path: Path,
    artifact_root: Path,
    result_path: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    cohort = base.load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == base.sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    require(cohort["entrypoint_sha256"] == base.sha256(Path(__file__).resolve()), "COHORT_ENTRYPOINT_SHA256")
    data_root = validate_dependencies(protocol, artifact_root)
    centroid_score, surface_score, diagnostics = surface.score_matrices(cohort, data_root)
    witness_count = int(cohort["counts"]["registration_witnesses"])
    witness_distances = [-float(surface_score[index, index]) for index in range(witness_count)]
    witness_ceiling = max(witness_distances)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    index = {target: position for position, target in enumerate(target_ids)}
    scenario_results = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        require(
            all(witness in references and witness in queries for witness in cohort["selection"]["witness_episode_ids"]),
            f"SCENARIO_WITNESS_MISSING:{scenario['id']}",
        )
        rows = [index[target] for target in references]
        columns = [index[target] for target in queries]
        scores = surface_score[np.ix_(rows, columns)]
        scenario_results.append(
            {
                **scenario,
                "surface_score_matrix": scores.round(6).tolist(),
                "methods": {
                    "complete_surface_hungarian": open_zero.evaluate_matches(
                        references, queries, open_zero.complete_assignment(scores)
                    ),
                    "rank_only_surface_zero": open_zero.evaluate_matches(
                        references, queries, open_zero.reciprocal_zero_assignment(scores)
                    ),
                    "witness_calibrated_surface_zero": open_zero.evaluate_matches(
                        references, queries, witness_zero_assignment(scores, witness_ceiling)
                    ),
                },
            }
        )
    methods = list(scenario_results[0]["methods"])
    aggregates = {name: open_zero.aggregate(scenario_results, name) for name in methods}
    upgraded = aggregates["witness_calibrated_surface_zero"]
    swap = next(row for row in scenario_results if row["id"] == "balanced-swap")
    swap_upgraded = swap["methods"]["witness_calibrated_surface_zero"]
    gate_met = (
        upgraded["true_positive"] == int(cohort["counts"]["truth_matches_across_scenarios"])
        and upgraded["false_positive"] == 0
        and upgraded["false_negative"] == 0
        and upgraded["zero_assignment_exact_scenarios"] == len(scenario_results)
        and swap_upgraded["true_positive"] == 2
        and swap_upgraded["false_positive"] == 0
        and swap_upgraded["zero_assignment_exact"]
    )
    baseline = aggregates["complete_surface_hungarian"]
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_PHYSICAL_TARGET_DISJOINT_WITNESS_CALIBRATED_SURFACE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_WITNESS_CALIBRATED_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_WITNESS_CALIBRATED_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "witness_calibration": {
            "episode_ids": cohort["selection"]["witness_episode_ids"],
            "surface_distances_metres": [round(value, 6) for value in witness_distances],
            "maximum_witness_surface_distance_metres": round(witness_ceiling, 6),
            "coverage_claim": "NONE_ENGINEERING_CALIBRATION_ONLY",
        },
        "metrics": {
            "aggregate": aggregates,
            "scenarios": scenario_results,
            "full_surface_score_matrix": surface_score.round(6).tolist(),
            "full_centroid_score_matrix": centroid_score.round(6).tolist(),
            "pair_diagnostics": diagnostics,
        },
        "gain_over_complete_assignment": {
            "false_positive_reduction": baseline["false_positive"] - upgraded["false_positive"],
            "f1_delta": round(upgraded["f1"] - baseline["f1"], 6),
            "balanced_swap_false_positive_reduction": (
                swap["methods"]["complete_surface_hungarian"]["false_positive"]
                - swap_upgraded["false_positive"]
            ),
        },
        "rgb_members_opened": 0,
        "depth_members_opened": 0,
        "decision_gate": protocol["decision_gate"],
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--artifact-root", type=Path, required=True)
    freeze_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--artifact-root", type=Path, required=True)
    replay_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.artifact_root, args.cohort)
    else:
        replay(args.protocol, args.cohort, args.artifact_root, args.result)


if __name__ == "__main__":
    main()
