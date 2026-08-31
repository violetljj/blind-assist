#!/usr/bin/env python3
"""Confirm registered extent support on a disjoint 3RScan family."""

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
import l10_3rscan_registered_extent_support_zero_assignment as support  # noqa: E402
import l10_3rscan_registered_surface_zero_assignment as surface  # noqa: E402
import l10_3rscan_scan_family_disjoint_witness_incremental as family_parent  # noqa: E402
import l10_3rscan_witness_calibrated_zero_assignment as witness_parent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-extent-support-scan-family-confirmation-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-extent-support-scan-family-confirmation-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-extent-support-scan-family-confirmation-result-v1"


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
            "EXTENT_SUPPORT_PREDECESSOR",
        )
    result = base.load_json(HERE / prior["result_path"])
    require(
        result.get("conclusion") == prior["required_conclusion"],
        "EXTENT_SUPPORT_PREDECESSOR_CONCLUSION",
    )
    return data_root


def consumed_state(protocol: dict[str, Any]) -> tuple[set[tuple[str, int]], set[str], list[dict[str, Any]]]:
    targets, receipts = witness_parent.consumed_physical_targets(protocol)
    reference_scans: set[str] = set()
    for record in protocol["source"]["consumed_target_cohorts"]:
        cohort = base.load_json(HERE / record["path"])
        for episode in cohort.get("episodes", []):
            if "reference_scan_id" in episode:
                reference_scans.add(str(episode["reference_scan_id"]))
    return targets, reference_scans, receipts


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    data_root = validate_dependencies(protocol, artifact_root)
    consumed_targets, consumed_reference_scans, receipts = consumed_state(protocol)
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    target_count = int(protocol["frozen_cohort"]["physical_targets"])
    selected: list[dict[str, Any]] | None = None
    selected_reference_points: dict[int, np.ndarray] = {}
    selected_query_points: dict[int, np.ndarray] = {}
    groups_considered = 0
    for family in witness_parent.group_rows(rows):
        reference_scan = str(family[0]["reference_scan_id"])
        if reference_scan in consumed_reference_scans:
            continue
        rescan = str(family[0]["rescan_id"])
        by_label: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        seen: set[int] = set()
        for row in family:
            target_id = int(row["target_instance_id"])
            if (reference_scan, target_id) in consumed_targets or target_id in seen:
                continue
            by_label.setdefault(family_parent.normalized_label(row), []).append(row)
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
    require(selected is not None, "SCAN_FAMILY_DISJOINT_EXTENT_SUPPORT_SOURCE_NOT_EVALUABLE")
    reference_scan = str(selected[0]["reference_scan_id"])
    rescan = str(selected[0]["rescan_id"])
    matrices = [extent.provider_matrix(row["transform"]) for row in selected]
    require(all(np.allclose(matrices[0], matrix, atol=1e-8) for matrix in matrices[1:]), "FAMILY_TRANSFORM_CONFLICT")
    require(len({family_parent.normalized_label(row) for row in selected}) == 1, "SAME_CLASS_CONTRACT")
    episodes = []
    for index, row in enumerate(selected, 1):
        target_id = int(row["target_instance_id"])
        episodes.append(
            {
                "episode_id": f"FC{index:02d}",
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
    scenario_rows = support.scenarios(episode_ids)
    truth_matches = sum(
        len(set(row["reference_targets"]) & set(row["query_targets"])) for row in scenario_rows
    )
    truth_unmatched = sum(
        len(set(row["reference_targets"]) ^ set(row["query_targets"])) for row in scenario_rows
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_REPLAY_SCAN_FAMILY_DISJOINT_EXTENT_SUPPORT_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows_with_local_geometry": len(rows),
            "consumed_physical_targets": len(consumed_targets),
            "consumed_reference_scan_families": sorted(consumed_reference_scans),
            "same_class_groups_considered": groups_considered,
            "reference_scan_id": reference_scan,
            "rescan_id": rescan,
            "normalized_target_class": family_parent.normalized_label(selected[0]),
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
    extent_support, support_diagnostics = support.registered_extent_iou_matrix(cohort, data_root)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    target_index = {target: position for position, target in enumerate(target_ids)}
    scenario_results = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[target] for target in references]
        columns = [target_index[target] for target in queries]
        scores = surface_score[np.ix_(rows, columns)]
        support_matrix = extent_support[np.ix_(rows, columns)]
        scenario_results.append(
            {
                **scenario,
                "surface_score_matrix": scores.round(6).tolist(),
                "registered_extent_iou_matrix": support_matrix.round(6).tolist(),
                "methods": {
                    "complete_surface_hungarian": open_zero.evaluate_matches(
                        references, queries, open_zero.complete_assignment(scores)
                    ),
                    "rank_only_surface_zero": open_zero.evaluate_matches(
                        references, queries, open_zero.reciprocal_zero_assignment(scores)
                    ),
                    "extent_support_surface_zero": open_zero.evaluate_matches(
                        references,
                        queries,
                        support.support_zero_assignment(scores, support_matrix),
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
        upgraded["true_positive"] == expected_true
        and upgraded["false_positive"] == 0
        and upgraded["false_negative"] == 0
        and upgraded["zero_assignment_exact_scenarios"] == len(scenario_results)
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SCAN_FAMILY_DISJOINT_EXTENT_SUPPORT_CONFIRMATION_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_EXTENT_SUPPORT_SCAN_FAMILY_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_EXTENT_SUPPORT_SCAN_FAMILY_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
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
            "opportunity_observed": bool(rank_only["false_positive"] > 0),
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
        "support_rule": "unchanged strictly positive registered planar convex-hull intersection",
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
