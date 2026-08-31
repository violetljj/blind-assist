#!/usr/bin/env python3
"""Match partial same-scene door rosters with registered 3D surface evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as base  # noqa: E402
import l10_3rscan_open_roster_zero_assignment as open_zero  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-registered-surface-zero-assignment-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-registered-surface-zero-assignment-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-registered-surface-zero-assignment-result-v1"


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
            "OPEN_ROSTER_PREDECESSOR",
        )
    predecessor_result = base.load_json(HERE / predecessor["result_path"])
    require(
        predecessor_result.get("conclusion") == predecessor["required_conclusion"],
        "OPEN_ROSTER_PREDECESSOR_CONCLUSION",
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
    selected_family: list[dict[str, Any]] | None = None
    selected_reference_points: dict[int, np.ndarray] = {}
    selected_query_points: dict[int, np.ndarray] = {}
    families_considered = 0

    for family in group_rows(rows):
        reference_scan = str(family[0]["reference_scan_id"])
        rescan = str(family[0]["rescan_id"])
        candidates = []
        used_ids: set[int] = set()
        for row in family:
            target_id = int(row["target_instance_id"])
            physical = (reference_scan, target_id)
            if physical in consumed or target_id in used_ids:
                continue
            candidates.append(row)
            used_ids.add(target_id)
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
        selected_family = admissible[:target_count]
        selected_reference_points = reference_points
        selected_query_points = query_points
        break

    require(selected_family is not None, "COMMON_SCAN_FAMILY_SOURCE_NOT_EVALUABLE")
    reference_scan = str(selected_family[0]["reference_scan_id"])
    rescan = str(selected_family[0]["rescan_id"])
    matrices = [extent.provider_matrix(row["transform"]) for row in selected_family]
    require(all(np.allclose(matrices[0], matrix, atol=1e-8) for matrix in matrices[1:]), "FAMILY_TRANSFORM_CONFLICT")
    episodes: list[dict[str, Any]] = []
    for index, row in enumerate(selected_family, 1):
        target_id = int(row["target_instance_id"])
        episodes.append(
            {
                "episode_id": f"SZ{index:02d}",
                **row,
                "reference_target_vertices": len(selected_reference_points[target_id]),
                "query_target_vertices": len(selected_query_points[target_id]),
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
        "authority": "FROZEN_PRE_REPLAY_PHYSICAL_TARGET_DISJOINT_COMMON_FAMILY_SURFACE_COHORT",
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
            "physical_target_ids": [int(row["target_instance_id"]) for row in selected_family],
            "rgb_members_opened": 0,
            "depth_members_opened": 0,
            "exclusion_receipts": exclusion_receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": episodes,
        "provider_transform": selected_family[0]["transform"],
        "scenarios": open_zero.scenario_records(episode_ids),
        "counts": {
            "physical_targets": len(episodes),
            "scenarios": 4,
            "truth_matches_across_scenarios": 12,
            "truth_unmatched_nodes_across_scenarios": 4,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(cohort_path, cohort)


def symmetric_surface_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_to_second = cKDTree(second).query(first, k=1, workers=1)[0]
    second_to_first = cKDTree(first).query(second, k=1, workers=1)[0]
    return float(max(np.median(first_to_second), np.median(second_to_first)))


def score_matrices(
    cohort: dict[str, Any], data_root: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    episodes = cohort["episodes"]
    target_ids = [int(episode["target_instance_id"]) for episode in episodes]
    reference_scan = cohort["selection"]["reference_scan_id"]
    rescan = cohort["selection"]["rescan_id"]
    reference_points = extent.ply_instance_points(
        data_root / reference_scan / "labels.instances.annotated.v2.ply", set(target_ids)
    )
    query_raw = extent.ply_instance_points(
        data_root / rescan / "labels.instances.annotated.v2.ply", set(target_ids)
    )
    matrix = extent.provider_matrix(cohort["provider_transform"])
    query_points = {
        target_id: extent.transform_points(points, matrix) for target_id, points in query_raw.items()
    }
    centroids_reference = {key: np.mean(value, axis=0) for key, value in reference_points.items()}
    centroids_query = {key: np.mean(value, axis=0) for key, value in query_points.items()}
    centroid_score = np.zeros((len(target_ids), len(target_ids)), dtype=np.float64)
    surface_score = np.zeros_like(centroid_score)
    diagnostics = {}
    for i, reference_id in enumerate(target_ids):
        for j, query_id in enumerate(target_ids):
            centroid_distance = float(
                np.linalg.norm(centroids_reference[reference_id] - centroids_query[query_id])
            )
            surface_distance = symmetric_surface_distance(
                reference_points[reference_id], query_points[query_id]
            )
            centroid_score[i, j] = -centroid_distance
            surface_score[i, j] = -surface_distance
            diagnostics[f"{episodes[i]['episode_id']}->{episodes[j]['episode_id']}"] = {
                "reference_target_instance_id": reference_id,
                "query_target_instance_id": query_id,
                "centroid_distance_metres": round(centroid_distance, 6),
                "symmetric_median_surface_distance_metres": round(surface_distance, 6),
            }
    return centroid_score, surface_score, diagnostics


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
    centroid_score, surface_score, diagnostics = score_matrices(cohort, data_root)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    index = {target: position for position, target in enumerate(target_ids)}
    scenario_results = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [index[target] for target in references]
        columns = [index[target] for target in queries]
        centroid = centroid_score[np.ix_(rows, columns)]
        surface = surface_score[np.ix_(rows, columns)]
        scenario_results.append(
            {
                **scenario,
                "centroid_score_matrix": centroid.round(6).tolist(),
                "surface_score_matrix": surface.round(6).tolist(),
                "methods": {
                    "complete_surface_hungarian": open_zero.evaluate_matches(
                        references, queries, open_zero.complete_assignment(surface)
                    ),
                    "centroid_reciprocal_zero": open_zero.evaluate_matches(
                        references, queries, open_zero.reciprocal_zero_assignment(centroid)
                    ),
                    "surface_reciprocal_zero": open_zero.evaluate_matches(
                        references, queries, open_zero.reciprocal_zero_assignment(surface)
                    ),
                },
            }
        )
    methods = list(scenario_results[0]["methods"])
    aggregates = {name: open_zero.aggregate(scenario_results, name) for name in methods}
    upgraded = aggregates["surface_reciprocal_zero"]
    swap = next(row for row in scenario_results if row["id"] == "balanced-swap")
    swap_upgraded = swap["methods"]["surface_reciprocal_zero"]
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
        "authority": "CONSUMED_PHYSICAL_TARGET_DISJOINT_COMMON_FAMILY_REGISTERED_SURFACE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_REGISTERED_SURFACE_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_REGISTERED_SURFACE_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "aggregate": aggregates,
            "scenarios": scenario_results,
            "full_centroid_score_matrix": centroid_score.round(6).tolist(),
            "full_surface_score_matrix": surface_score.round(6).tolist(),
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
