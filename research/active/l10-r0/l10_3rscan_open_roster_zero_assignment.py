#!/usr/bin/env python3
"""Evaluate zero-assignment identity matching on partial 3RScan rosters."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_V1 = "blindassist-l10-3rscan-open-roster-zero-assignment-protocol-v1"
PROTOCOL_V2 = "blindassist-l10-3rscan-open-roster-zero-assignment-protocol-v2"
COHORT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-open-roster-zero-assignment-cohort-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-open-roster-zero-assignment-cohort-v2",
}
RESULT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-open-roster-zero-assignment-result-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-open-roster-zero-assignment-result-v2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = base.load_json(path)
    schema = raw.get("schema")
    require(schema in COHORT_SCHEMAS, "PROTOCOL_SCHEMA_MISMATCH")
    if schema == PROTOCOL_V1:
        return raw, raw
    base_path = HERE / raw["base_protocol_path"]
    base.verify_path(base_path, raw["base_protocol_sha256"], "BASE_OPEN_ROSTER_PROTOCOL")
    source = base.load_json(base_path)
    require(source.get("schema") == PROTOCOL_V1, "BASE_OPEN_ROSTER_PROTOCOL_SCHEMA")
    resolved = base.merge_dict(source, raw["overrides"])
    resolved["schema"] = schema
    return raw, resolved


def validate_dependencies(protocol: dict[str, Any], artifact_root: Path) -> tuple[Path, Path]:
    data_root = artifact_root / protocol["source"]["dataset_relative_path"]
    model_root = artifact_root / protocol["frozen_model"]["model_relative_path"]
    require(data_root.is_dir(), "3RSCAN_DATA_ROOT_MISSING")
    require(model_root.is_dir(), "MODEL_ROOT_MISSING")
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
            "CLOSED_ROSTER_PREDECESSOR",
        )
    predecessor_result = base.load_json(HERE / predecessor["result_path"])
    require(
        predecessor_result.get("conclusion") == predecessor["required_conclusion"],
        "CLOSED_ROSTER_PREDECESSOR_CONCLUSION",
    )
    base.verify_path(
        model_root / "model.safetensors",
        protocol["frozen_model"]["weights_sha256"],
        "MODEL",
    )
    return data_root, model_root


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


def select_views(
    data_root: Path,
    scan_id: str,
    target_id: int,
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, int]]:
    """Select a high-visibility view and the farthest other eligible camera view."""
    scan_root = data_root / scan_id
    points = extent.ply_instance_points(
        scan_root / "labels.instances.annotated.v2.ply", {target_id}
    )[target_id]
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    candidates: list[dict[str, Any]] = []
    with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
        info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        for frame in pixel.pose_frames(archive):
            try:
                pose = pixel.read_pose(archive, frame)
            except ValueError:
                continue
            opened["pose_members"] += 1
            if not np.isfinite(pose).all():
                continue
            try:
                depth = pixel.decode_depth(archive, frame)
                opened["depth_members"] += 1
                stats = pixel.frame_visibility(
                    points,
                    pose,
                    info,
                    depth,
                    float(rules["depth_consistency_metres"]),
                )
            except ValueError:
                continue
            if not pixel.eligible(stats, rules):
                continue
            quality = (
                float(stats["depth_visible_vertices"]),
                float(stats["depth_visible_ratio"]),
                float(stats["projected_area_pixels"]),
                float(stats["image_margin_pixels"]),
                -float(frame),
            )
            candidates.append(
                {
                    "frame": int(frame),
                    "color_size": [info["color_width"], info["color_height"]],
                    "camera_center": pose[:3, 3].astype(float).tolist(),
                    "quality_key": list(quality),
                    **stats,
                }
            )
    if len(candidates) < 2:
        return None, opened
    first = max(candidates, key=lambda row: tuple(row["quality_key"]))
    first_center = np.asarray(first["camera_center"], dtype=np.float64)
    remaining = [row for row in candidates if row["frame"] != first["frame"]]
    second = max(
        remaining,
        key=lambda row: (
            float(np.linalg.norm(np.asarray(row["camera_center"]) - first_center)),
            *tuple(row["quality_key"]),
        ),
    )
    separation = float(
        np.linalg.norm(np.asarray(second["camera_center"], dtype=np.float64) - first_center)
    )
    output = []
    for index, row in enumerate((first, second), 1):
        clean = dict(row)
        clean.pop("quality_key")
        clean["view_index"] = index
        clean["selected_pair_translation_separation_metres"] = separation
        output.append(clean)
    return output, opened


def scenario_records(target_ids: list[str]) -> list[dict[str, Any]]:
    require(len(target_ids) == 4, "FOUR_TARGETS_REQUIRED")
    first, second, third, fourth = target_ids
    return [
        {
            "id": "closed-four",
            "reference_targets": [first, second, third, fourth],
            "query_targets": [first, second, third, fourth],
        },
        {
            "id": "query-extra",
            "reference_targets": [first, second, third],
            "query_targets": [first, second, third, fourth],
        },
        {
            "id": "reference-extra",
            "reference_targets": [first, second, third, fourth],
            "query_targets": [first, second, third],
        },
        {
            "id": "balanced-swap",
            "reference_targets": [first, second, third],
            "query_targets": [first, second, fourth],
        },
    ]


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    raw_protocol, protocol = load_protocol(protocol_path)
    data_root, _ = validate_dependencies(protocol, artifact_root)
    consumed, exclusion_receipts = consumed_physical_targets(protocol)
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    rules = protocol["frozen_cohort"]["frame_rules"]
    target_count = int(protocol["frozen_cohort"]["physical_targets"])
    maximum_per_reference = int(protocol["frozen_cohort"]["maximum_targets_per_reference_scan"])
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    cache: dict[tuple[str, int], tuple[list[dict[str, Any]] | None, dict[str, int]]] = {}
    selected: list[dict[str, Any]] = []
    used_physical: set[tuple[str, int]] = set()
    reference_counts: dict[str, int] = {}
    considered = 0

    def views(scan_id: str, target_id: int) -> list[dict[str, Any]] | None:
        key = (scan_id, target_id)
        if key not in cache:
            cache[key] = select_views(data_root, scan_id, target_id, rules)
            for name, count in cache[key][1].items():
                opened[name] += count
        return cache[key][0]

    for row in rows:
        reference_scan = str(row["reference_scan_id"])
        rescan = str(row["rescan_id"])
        target_id = int(row["target_instance_id"])
        physical = (reference_scan, target_id)
        if (
            physical in consumed
            or physical in used_physical
            or reference_counts.get(reference_scan, 0) >= maximum_per_reference
        ):
            continue
        if not all((data_root / scan_id / "sequence.zip").is_file() for scan_id in (reference_scan, rescan)):
            continue
        considered += 1
        reference_views = views(reference_scan, target_id)
        query_views = views(rescan, target_id)
        if reference_views is None or query_views is None:
            continue
        episode_id = f"OZ{len(selected) + 1:02d}"
        selected.append(
            {
                "episode_id": episode_id,
                **row,
                "reference_views": reference_views,
                "query_views": query_views,
            }
        )
        used_physical.add(physical)
        reference_counts[reference_scan] = reference_counts.get(reference_scan, 0) + 1
        if len(selected) == target_count:
            break

    require(opened["rgb_members"] == 0, "RGB_OPENED_BEFORE_FREEZE")
    require(len(selected) == target_count, f"OPEN_ROSTER_COHORT_NOT_EVALUABLE:{len(selected)}")
    require(
        len(reference_counts) >= int(protocol["frozen_cohort"]["minimum_reference_scans"]),
        f"REFERENCE_SCAN_DIVERSITY_NOT_EVALUABLE:{len(reference_counts)}",
    )
    source_manifest: dict[str, dict[str, Any]] = {}
    images: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for scan_id in (episode["reference_scan_id"], episode["rescan_id"]):
            for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip"):
                source_manifest[f"{scan_id}/{name}"] = base.source_record(
                    data_root / scan_id / name, artifact_root
                )
        for role, scan_key in (("reference", "reference_scan_id"), ("query", "rescan_id")):
            for view in episode[f"{role}_views"]:
                key = f"{episode['episode_id']}_{role}_v{view['view_index']}"
                images[key] = {
                    "episode_id": episode["episode_id"],
                    "role": role,
                    "view_index": view["view_index"],
                    "scan_id": episode[scan_key],
                    "target_instance_id": episode["target_instance_id"],
                    "target_label": episode["target_label"],
                    "frame": view["frame"],
                    "color_size": view["color_size"],
                    "bbox_xyxy": view["bbox_xyxy"],
                    "image_margin_pixels": view["image_margin_pixels"],
                    "inside_vertex_fraction": view["inside_vertex_fraction"],
                    "depth_visible_ratio": view["depth_visible_ratio"],
                    "zip_member": f"frame-{int(view['frame']):06d}.color.jpg",
                }
    for name in ("3RScan.json", "objects.json"):
        source_manifest[name] = base.source_record(data_root / name, artifact_root)
    target_ids = [episode["episode_id"] for episode in selected]
    scenarios = scenario_records(target_ids)
    cohort = {
        "schema": COHORT_SCHEMAS[raw_protocol["schema"]],
        "authority": "FROZEN_PRE_RGB_PHYSICAL_TARGET_DISJOINT_PARTIAL_ROSTER_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_physical_targets": len(consumed),
            "candidate_rows_considered": considered,
            "unique_physical_target_rule": True,
            "maximum_targets_per_reference_scan": maximum_per_reference,
            "reference_scan_target_counts": dict(sorted(reference_counts.items())),
            "views_per_target_side": 2,
            "opened_members": opened,
            "frame_rules": rules,
            "exclusion_receipts": exclusion_receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": selected,
        "images": images,
        "scenarios": scenarios,
        "counts": {
            "physical_targets": len(selected),
            "rgb_members_after_replay": len(images),
            "scenarios": len(scenarios),
            "truth_matches_across_scenarios": 12,
            "truth_unmatched_nodes_across_scenarios": 4,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(cohort_path, cohort)


def identity_score_matrices(
    embeddings: dict[str, np.ndarray], target_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, list[list[float]]]]:
    one_view = np.zeros((len(target_ids), len(target_ids)), dtype=np.float64)
    multiview = np.zeros_like(one_view)
    view_matrices: dict[str, list[list[float]]] = {}
    for i, reference in enumerate(target_ids):
        for j, query in enumerate(target_ids):
            scores = np.zeros((2, 2), dtype=np.float64)
            for r in range(2):
                for q in range(2):
                    _, score = base.pair_scores(
                        embeddings[f"{reference}_reference_v{r + 1}"],
                        embeddings[f"{query}_query_v{q + 1}"],
                    )
                    scores[r, q] = score
            one_view[i, j] = scores[0, 0]
            reference_coverage = float(np.min(np.max(scores, axis=1)))
            query_coverage = float(np.min(np.max(scores, axis=0)))
            multiview[i, j] = min(reference_coverage, query_coverage)
            view_matrices[f"{reference}->{query}"] = scores.round(6).tolist()
    return one_view, multiview, view_matrices


def complete_assignment(matrix: np.ndarray) -> list[tuple[int, int]]:
    rows, columns = linear_sum_assignment(-matrix)
    return [(int(row), int(column)) for row, column in zip(rows, columns, strict=True)]


def reciprocal_zero_assignment(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return only strict mutual maxima; all other rows and columns are zero-assigned."""
    matches: list[tuple[int, int]] = []
    for row in range(matrix.shape[0]):
        column = int(np.argmax(matrix[row]))
        row_values = np.delete(matrix[row], column)
        column_values = np.delete(matrix[:, column], row)
        row_strict = not len(row_values) or matrix[row, column] > float(np.max(row_values))
        column_strict = not len(column_values) or matrix[row, column] > float(np.max(column_values))
        if row_strict and column_strict and row == int(np.argmax(matrix[:, column])):
            matches.append((row, column))
    return matches


def evaluate_matches(
    reference_targets: list[str],
    query_targets: list[str],
    matches: list[tuple[int, int]],
) -> dict[str, Any]:
    predicted = {
        (reference_targets[row], query_targets[column]) for row, column in matches
    }
    truth_ids = set(reference_targets) & set(query_targets)
    truth = {(target, target) for target in truth_ids}
    true_positive = len(predicted & truth)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    predicted_matched_references = {left for left, _ in predicted}
    predicted_matched_queries = {right for _, right in predicted}
    predicted_unmatched_references = set(reference_targets) - predicted_matched_references
    predicted_unmatched_queries = set(query_targets) - predicted_matched_queries
    truth_unmatched_references = set(reference_targets) - truth_ids
    truth_unmatched_queries = set(query_targets) - truth_ids
    precision = true_positive / len(predicted) if predicted else float(not truth)
    recall = true_positive / len(truth) if truth else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "matches": [
            {"reference": reference_targets[row], "query": query_targets[column]}
            for row, column in matches
        ],
        "predicted_unmatched_references": sorted(predicted_unmatched_references),
        "predicted_unmatched_queries": sorted(predicted_unmatched_queries),
        "truth_unmatched_references": sorted(truth_unmatched_references),
        "truth_unmatched_queries": sorted(truth_unmatched_queries),
        "zero_assignment_exact": (
            predicted_unmatched_references == truth_unmatched_references
            and predicted_unmatched_queries == truth_unmatched_queries
        ),
    }


def aggregate(scenarios: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [scenario["methods"][method] for scenario in scenarios]
    true_positive = sum(row["true_positive"] for row in rows)
    false_positive = sum(row["false_positive"] for row in rows)
    false_negative = sum(row["false_negative"] for row in rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "zero_assignment_exact_scenarios": sum(row["zero_assignment_exact"] for row in rows),
        "scenarios": len(rows),
    }


def replay(
    protocol_path: Path,
    cohort_path: Path,
    artifact_root: Path,
    crop_dir: Path,
    result_path: Path,
) -> None:
    raw_protocol, protocol = load_protocol(protocol_path)
    cohort = base.load_json(cohort_path)
    require(
        cohort.get("schema") == COHORT_SCHEMAS[raw_protocol["schema"]],
        "COHORT_SCHEMA_MISMATCH",
    )
    require(cohort["protocol_sha256"] == base.sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    require(cohort["entrypoint_sha256"] == base.sha256(Path(__file__).resolve()), "COHORT_ENTRYPOINT_SHA256")
    validate_dependencies(protocol, artifact_root)
    embeddings, rgb_receipts = base.encode(protocol, cohort, artifact_root, crop_dir)
    target_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    one_view, multiview, view_matrices = identity_score_matrices(embeddings, target_ids)
    index = {target: position for position, target in enumerate(target_ids)}
    scenario_results: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [index[target] for target in references]
        columns = [index[target] for target in queries]
        one = one_view[np.ix_(rows, columns)]
        multi = multiview[np.ix_(rows, columns)]
        scenario_results.append(
            {
                **scenario,
                "one_view_score_matrix": one.round(6).tolist(),
                "multiview_bottleneck_score_matrix": multi.round(6).tolist(),
                "methods": {
                    "complete_one_view_hungarian": evaluate_matches(
                        references, queries, complete_assignment(one)
                    ),
                    "one_view_reciprocal_zero": evaluate_matches(
                        references, queries, reciprocal_zero_assignment(one)
                    ),
                    "multiview_reciprocal_zero": evaluate_matches(
                        references, queries, reciprocal_zero_assignment(multi)
                    ),
                },
            }
        )
    method_names = list(scenario_results[0]["methods"])
    aggregates = {name: aggregate(scenario_results, name) for name in method_names}
    upgraded = aggregates["multiview_reciprocal_zero"]
    balanced_swap = next(row for row in scenario_results if row["id"] == "balanced-swap")
    swap_upgraded = balanced_swap["methods"]["multiview_reciprocal_zero"]
    gate_met = (
        upgraded["true_positive"] == int(cohort["counts"]["truth_matches_across_scenarios"])
        and upgraded["false_positive"] == 0
        and upgraded["false_negative"] == 0
        and upgraded["zero_assignment_exact_scenarios"] == len(scenario_results)
        and swap_upgraded["true_positive"] == 2
        and swap_upgraded["false_positive"] == 0
        and swap_upgraded["zero_assignment_exact"]
    )
    result = {
        "schema": RESULT_SCHEMAS[raw_protocol["schema"]],
        "authority": "CONSUMED_PHYSICAL_TARGET_DISJOINT_PARTIAL_ROSTER_ZERO_ASSIGNMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": base.sha256(Path(__file__).resolve()),
        },
        "conclusion": (
            "L10_3RSCAN_OPEN_ROSTER_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_OPEN_ROSTER_ZERO_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "aggregate": aggregates,
            "scenarios": scenario_results,
            "full_one_view_score_matrix": one_view.round(6).tolist(),
            "full_multiview_bottleneck_score_matrix": multiview.round(6).tolist(),
            "cross_view_pair_score_matrices": view_matrices,
        },
        "gain_over_complete_assignment": {
            "false_positive_reduction": (
                aggregates["complete_one_view_hungarian"]["false_positive"]
                - upgraded["false_positive"]
            ),
            "f1_delta": round(
                upgraded["f1"] - aggregates["complete_one_view_hungarian"]["f1"], 6
            ),
            "balanced_swap_false_positive_reduction": (
                balanced_swap["methods"]["complete_one_view_hungarian"]["false_positive"]
                - swap_upgraded["false_positive"]
            ),
        },
        "rgb_members_opened": len(rgb_receipts),
        "rgb_receipts": rgb_receipts,
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
    replay_parser.add_argument("--crop-dir", type=Path, required=True)
    replay_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.artifact_root, args.cohort)
    else:
        replay(args.protocol, args.cohort, args.artifact_root, args.crop_dir, args.result)


if __name__ == "__main__":
    main()
