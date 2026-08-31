#!/usr/bin/env python3
"""Test registered planar-extent support for partial-roster zero assignment."""

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
import l10_3rscan_scan_family_disjoint_witness_incremental as predecessor  # noqa: E402
import l10_3rscan_witness_calibrated_zero_assignment as witness_parent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-registered-extent-support-zero-assignment-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-registered-extent-support-zero-assignment-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-registered-extent-support-zero-assignment-result-v1"


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
    prior = protocol["predecessor"]
    for key in ("protocol", "cohort", "result", "implementation"):
        base.verify_path(
            HERE / prior[f"{key}_path"],
            prior[f"{key}_sha256"],
            "RAW_WITNESS_PREDECESSOR",
        )
    result = base.load_json(HERE / prior["result_path"])
    require(
        result.get("conclusion") == prior["required_conclusion"],
        "RAW_WITNESS_PREDECESSOR_CONCLUSION",
    )
    return data_root


def scenarios(episode_ids: list[str]) -> list[dict[str, Any]]:
    rows = [
        {
            "id": "closed-six",
            "reference_targets": episode_ids,
            "query_targets": episode_ids,
        },
        {
            "id": "query-extra",
            "reference_targets": episode_ids[:-1],
            "query_targets": episode_ids,
        },
        {
            "id": "reference-extra",
            "reference_targets": episode_ids,
            "query_targets": episode_ids[:-1],
        },
    ]
    for missing_reference in episode_ids:
        for missing_query in episode_ids:
            if missing_reference == missing_query:
                continue
            rows.append(
                {
                    "id": f"balanced-swap-{missing_reference}-{missing_query}",
                    "reference_targets": [
                        target for target in episode_ids if target != missing_reference
                    ],
                    "query_targets": [
                        target for target in episode_ids if target != missing_query
                    ],
                }
            )
    return rows


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    data_root = validate_dependencies(protocol, artifact_root)
    consumed_targets, receipts = witness_parent.consumed_physical_targets(protocol)
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    target_count = int(protocol["frozen_cohort"]["physical_targets"])
    selected: list[dict[str, Any]] | None = None
    selected_reference_points: dict[int, np.ndarray] = {}
    selected_query_points: dict[int, np.ndarray] = {}
    groups_considered = 0
    for family in witness_parent.group_rows(rows):
        reference_scan = str(family[0]["reference_scan_id"])
        rescan = str(family[0]["rescan_id"])
        by_label: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        seen: set[int] = set()
        for row in family:
            target_id = int(row["target_instance_id"])
            if (reference_scan, target_id) in consumed_targets or target_id in seen:
                continue
            by_label.setdefault(predecessor.normalized_label(row), []).append(row)
            seen.add(target_id)
        for label_rows in by_label.values():
            if len(label_rows) < target_count:
                continue
            groups_considered += 1
            target_ids = {int(row["target_instance_id"]) for row in label_rows}
            reference_points = extent.ply_instance_points(
                data_root / reference_scan / "labels.instances.annotated.v2.ply", target_ids
            )
            query_points = extent.ply_instance_points(
                data_root / rescan / "labels.instances.annotated.v2.ply", target_ids
            )
            admissible = [
                row
                for row in label_rows
                if len(reference_points.get(int(row["target_instance_id"]), [])) >= 4
                and len(query_points.get(int(row["target_instance_id"]), [])) >= 4
            ]
            if len(admissible) < target_count:
                continue
            selected = admissible[:target_count]
            selected_reference_points = reference_points
            selected_query_points = query_points
            break
        if selected is not None:
            break
    require(selected is not None, "FRESH_SAME_CLASS_EXTENT_SUPPORT_SOURCE_NOT_EVALUABLE")
    reference_scan = str(selected[0]["reference_scan_id"])
    rescan = str(selected[0]["rescan_id"])
    matrices = [extent.provider_matrix(row["transform"]) for row in selected]
    require(all(np.allclose(matrices[0], matrix, atol=1e-8) for matrix in matrices[1:]), "FAMILY_TRANSFORM_CONFLICT")
    require(len({predecessor.normalized_label(row) for row in selected}) == 1, "SAME_CLASS_CONTRACT")
    episodes = []
    for index, row in enumerate(selected, 1):
        target_id = int(row["target_instance_id"])
        episodes.append(
            {
                "episode_id": f"ES{index:02d}",
                **row,
                "reference_target_vertices": len(selected_reference_points[target_id]),
                "query_target_vertices": len(selected_query_points[target_id]),
                "role": "EVALUATED_TARGET",
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
    scenario_rows = scenarios(episode_ids)
    truth_matches = sum(
        len(set(row["reference_targets"]) & set(row["query_targets"])) for row in scenario_rows
    )
    truth_unmatched = sum(
        len(set(row["reference_targets"]) ^ set(row["query_targets"])) for row in scenario_rows
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_REPLAY_PHYSICAL_TARGET_DISJOINT_SAME_CLASS_EXTENT_SUPPORT_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_physical_targets": len(consumed_targets),
            "same_class_groups_considered": groups_considered,
            "reference_scan_id": reference_scan,
            "rescan_id": rescan,
            "normalized_target_class": predecessor.normalized_label(selected[0]),
            "physical_target_ids": [int(row["target_instance_id"]) for row in selected],
            "rgb_members_opened": 0,
            "depth_members_opened": 0,
            "surface_distances_opened": 0,
            "extent_overlaps_opened": 0,
            "exclusion_receipts": receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": episodes,
        "provider_transform": selected[0]["transform"],
        "scenarios": scenario_rows,
        "counts": {
            "physical_targets": len(episodes),
            "scenarios": len(scenario_rows),
            "balanced_swap_scenarios": len(scenario_rows) - 3,
            "truth_matches_across_scenarios": truth_matches,
            "truth_unmatched_nodes_across_scenarios": truth_unmatched,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(cohort_path, cohort)


def registered_extent_iou_matrix(
    cohort: dict[str, Any], data_root: Path
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target_ids = {int(episode["target_instance_id"]) for episode in cohort["episodes"]}
    reference_scan = str(cohort["selection"]["reference_scan_id"])
    rescan = str(cohort["selection"]["rescan_id"])
    reference_points = extent.ply_instance_points(
        data_root / reference_scan / "labels.instances.annotated.v2.ply", target_ids
    )
    query_points = extent.ply_instance_points(
        data_root / rescan / "labels.instances.annotated.v2.ply", target_ids
    )
    transform = extent.provider_matrix(cohort["provider_transform"])
    transformed_queries = {
        target_id: extent.transform_points(points, transform)
        for target_id, points in query_points.items()
    }
    size = len(cohort["episodes"])
    matrix = np.zeros((size, size), dtype=np.float64)
    diagnostics = []
    for row, reference_episode in enumerate(cohort["episodes"]):
        reference_id = int(reference_episode["target_instance_id"])
        frame = extent.portal_frame(reference_points[reference_id])
        reference_hull = extent.convex_hull(
            extent.project_uv(reference_points[reference_id], *frame)
        )
        for column, query_episode in enumerate(cohort["episodes"]):
            query_id = int(query_episode["target_instance_id"])
            query_hull = extent.convex_hull(
                extent.project_uv(transformed_queries[query_id], *frame)
            )
            value = extent.polygon_iou(reference_hull, query_hull)
            matrix[row, column] = value
            diagnostics.append(
                {
                    "reference": reference_episode["episode_id"],
                    "query": query_episode["episode_id"],
                    "registered_planar_extent_iou": round(value, 6),
                    "positive_support": bool(value > 0.0),
                }
            )
    return matrix, diagnostics


def support_zero_assignment(
    surface_scores: np.ndarray, extent_support: np.ndarray
) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row, column in open_zero.reciprocal_zero_assignment(surface_scores)
        if float(extent_support[row, column]) > 0.0
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
    centroid_score, surface_score, surface_diagnostics = surface.score_matrices(cohort, data_root)
    extent_support, support_diagnostics = registered_extent_iou_matrix(cohort, data_root)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    target_index = {target: position for position, target in enumerate(target_ids)}
    scenario_results = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[target] for target in references]
        columns = [target_index[target] for target in queries]
        scores = surface_score[np.ix_(rows, columns)]
        support = extent_support[np.ix_(rows, columns)]
        scenario_results.append(
            {
                **scenario,
                "surface_score_matrix": scores.round(6).tolist(),
                "registered_extent_iou_matrix": support.round(6).tolist(),
                "methods": {
                    "complete_surface_hungarian": open_zero.evaluate_matches(
                        references, queries, open_zero.complete_assignment(scores)
                    ),
                    "rank_only_surface_zero": open_zero.evaluate_matches(
                        references, queries, open_zero.reciprocal_zero_assignment(scores)
                    ),
                    "extent_support_surface_zero": open_zero.evaluate_matches(
                        references, queries, support_zero_assignment(scores, support)
                    ),
                },
            }
        )
    methods = list(scenario_results[0]["methods"])
    aggregates = {name: open_zero.aggregate(scenario_results, name) for name in methods}
    rank_only = aggregates["rank_only_surface_zero"]
    upgraded = aggregates["extent_support_surface_zero"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    gate_met = (
        rank_only["false_positive"] >= 1
        and upgraded["true_positive"] == expected_true
        and upgraded["false_positive"] == 0
        and upgraded["false_negative"] == 0
        and upgraded["zero_assignment_exact_scenarios"] == len(scenario_results)
        and upgraded["true_positive"] >= rank_only["true_positive"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_PHYSICAL_TARGET_DISJOINT_SAME_CLASS_EXTENT_SUPPORT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_REGISTERED_EXTENT_SUPPORT_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_REGISTERED_EXTENT_SUPPORT_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "aggregate": aggregates,
            "scenarios": scenario_results,
            "full_surface_score_matrix": surface_score.round(6).tolist(),
            "full_centroid_score_matrix": centroid_score.round(6).tolist(),
            "full_registered_extent_iou_matrix": extent_support.round(6).tolist(),
            "surface_pair_diagnostics": surface_diagnostics,
            "extent_support_diagnostics": support_diagnostics,
        },
        "incremental_gain_over_rank_only": {
            "true_positive_delta": upgraded["true_positive"] - rank_only["true_positive"],
            "false_positive_reduction": rank_only["false_positive"] - upgraded["false_positive"],
            "false_negative_reduction": rank_only["false_negative"] - upgraded["false_negative"],
            "f1_delta": round(upgraded["f1"] - rank_only["f1"], 6),
            "exact_zero_assignment_scenario_gain": (
                upgraded["zero_assignment_exact_scenarios"]
                - rank_only["zero_assignment_exact_scenarios"]
            ),
        },
        "rgb_members_opened": 0,
        "depth_members_opened": 0,
        "support_rule": "strictly positive registered planar convex-hull intersection; no IoU threshold",
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
