#!/usr/bin/env python3
"""Test witness-calibrated zero assignment on a disjoint 3RScan family."""

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
import l10_3rscan_witness_calibrated_zero_assignment as parent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-scan-family-disjoint-witness-incremental-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-scan-family-disjoint-witness-incremental-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-scan-family-disjoint-witness-incremental-result-v1"


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
            "WITNESS_PREDECESSOR",
        )
    predecessor_result = base.load_json(HERE / predecessor["result_path"])
    require(
        predecessor_result.get("conclusion") == predecessor["required_conclusion"],
        "WITNESS_PREDECESSOR_CONCLUSION",
    )
    return data_root


def consumed_state(protocol: dict[str, Any]) -> tuple[set[tuple[str, int]], set[str], list[dict[str, Any]]]:
    targets, receipts = parent.consumed_physical_targets(protocol)
    reference_scans: set[str] = set()
    for record in protocol["source"]["consumed_target_cohorts"]:
        cohort = base.load_json(HERE / record["path"])
        for episode in cohort.get("episodes", []):
            if "reference_scan_id" in episode:
                reference_scans.add(str(episode["reference_scan_id"]))
    return targets, reference_scans, receipts


def normalized_label(row: dict[str, Any]) -> str:
    return str(row["target_label"]).lower().replace(" ", "")


def scenarios(episode_ids: list[str], witness_count: int) -> list[dict[str, Any]]:
    witnesses = episode_ids[:witness_count]
    evaluated = episode_ids[witness_count:]
    records = [
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
    for missing_reference in evaluated:
        for missing_query in evaluated:
            if missing_reference == missing_query:
                continue
            records.append(
                {
                    "id": f"balanced-swap-{missing_reference}-{missing_query}",
                    "reference_targets": witnesses
                    + [target for target in evaluated if target != missing_reference],
                    "query_targets": witnesses
                    + [target for target in evaluated if target != missing_query],
                }
            )
    return records


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    data_root = validate_dependencies(protocol, artifact_root)
    consumed_targets, consumed_reference_scans, receipts = consumed_state(protocol)
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    target_count = int(protocol["frozen_cohort"]["physical_targets"])
    required_label_count = int(protocol["frozen_cohort"]["same_class_targets"])
    selected: list[dict[str, Any]] | None = None
    selected_reference_points: dict[int, np.ndarray] = {}
    selected_query_points: dict[int, np.ndarray] = {}
    families_considered = 0
    for family in parent.group_rows(rows):
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
            by_label.setdefault(normalized_label(row), []).append(row)
            seen.add(target_id)
        for label_rows in by_label.values():
            if len(label_rows) < required_label_count:
                continue
            families_considered += 1
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
    require(selected is not None, "SCAN_FAMILY_DISJOINT_SAME_CLASS_SOURCE_NOT_EVALUABLE")
    reference_scan = str(selected[0]["reference_scan_id"])
    rescan = str(selected[0]["rescan_id"])
    matrices = [extent.provider_matrix(row["transform"]) for row in selected]
    require(all(np.allclose(matrices[0], matrix, atol=1e-8) for matrix in matrices[1:]), "FAMILY_TRANSFORM_CONFLICT")
    require(len({normalized_label(row) for row in selected}) == 1, "SAME_CLASS_CONTRACT")
    witness_count = int(protocol["frozen_cohort"]["registration_witnesses"])
    episodes = []
    for index, row in enumerate(selected, 1):
        target_id = int(row["target_instance_id"])
        episodes.append(
            {
                "episode_id": f"FI{index:02d}",
                **row,
                "reference_target_vertices": len(selected_reference_points[target_id]),
                "query_target_vertices": len(selected_query_points[target_id]),
                "role": "REGISTRATION_WITNESS" if index <= witness_count else "EVALUATED_TARGET",
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
    scenario_rows = scenarios(episode_ids, witness_count)
    truth_matches = sum(
        len(set(row["reference_targets"]) & set(row["query_targets"])) for row in scenario_rows
    )
    truth_unmatched = sum(
        len(set(row["reference_targets"]) ^ set(row["query_targets"])) for row in scenario_rows
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_REPLAY_SCAN_FAMILY_DISJOINT_SAME_CLASS_INCREMENTAL_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_physical_targets": len(consumed_targets),
            "consumed_reference_scan_families": sorted(consumed_reference_scans),
            "families_considered_with_six_same_class_targets": families_considered,
            "reference_scan_id": reference_scan,
            "rescan_id": rescan,
            "normalized_target_class": normalized_label(selected[0]),
            "physical_target_ids": [int(row["target_instance_id"]) for row in selected],
            "witness_episode_ids": episode_ids[:witness_count],
            "evaluated_episode_ids": episode_ids[witness_count:],
            "rgb_members_opened": 0,
            "depth_members_opened": 0,
            "surface_distances_opened": 0,
            "exclusion_receipts": receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": episodes,
        "provider_transform": selected[0]["transform"],
        "scenarios": scenario_rows,
        "counts": {
            "physical_targets": len(episodes),
            "registration_witnesses": witness_count,
            "evaluated_targets": len(episodes) - witness_count,
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
    centroid_score, surface_score, diagnostics = surface.score_matrices(cohort, data_root)
    witness_count = int(cohort["counts"]["registration_witnesses"])
    witness_distances = [-float(surface_score[index, index]) for index in range(witness_count)]
    witness_ceiling = max(witness_distances)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    target_index = {target: position for position, target in enumerate(target_ids)}
    scenario_results = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        require(
            all(
                witness in references and witness in queries
                for witness in cohort["selection"]["witness_episode_ids"]
            ),
            f"SCENARIO_WITNESS_MISSING:{scenario['id']}",
        )
        rows = [target_index[target] for target in references]
        columns = [target_index[target] for target in queries]
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
                        references,
                        queries,
                        parent.witness_zero_assignment(scores, witness_ceiling),
                    ),
                },
            }
        )
    methods = list(scenario_results[0]["methods"])
    aggregates = {name: open_zero.aggregate(scenario_results, name) for name in methods}
    rank_only = aggregates["rank_only_surface_zero"]
    upgraded = aggregates["witness_calibrated_surface_zero"]
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
        "authority": "CONSUMED_SCAN_FAMILY_DISJOINT_SAME_CLASS_WITNESS_INCREMENTAL_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_SCAN_FAMILY_DISJOINT_WITNESS_INCREMENTAL_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_SCAN_FAMILY_DISJOINT_WITNESS_INCREMENTAL_DEVELOPMENT_GATE_NOT_MET"
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
