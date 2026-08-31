#!/usr/bin/env python3
"""Confirm parameter-free closed-roster assignment on new 3RScan targets."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_V1 = "blindassist-l10-3rscan-roster-assignment-confirmation-protocol-v1"
PROTOCOL_V2 = "blindassist-l10-3rscan-roster-assignment-confirmation-protocol-v2"
COHORT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-roster-assignment-confirmation-cohort-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-roster-assignment-confirmation-cohort-v2",
}
RESULT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-roster-assignment-confirmation-result-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-roster-assignment-confirmation-result-v2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = base.load_json(path)
    schema = raw.get("schema")
    require(schema in COHORT_SCHEMAS, "PROTOCOL_SCHEMA_MISMATCH")
    if schema == PROTOCOL_V1:
        return raw, raw
    base_path = HERE / raw["base_protocol_path"]
    base.verify_path(base_path, raw["base_protocol_sha256"], "BASE_ASSIGNMENT_PROTOCOL")
    source = base.load_json(base_path)
    require(source.get("schema") == PROTOCOL_V1, "BASE_ASSIGNMENT_PROTOCOL_SCHEMA")
    resolved = base.merge_dict(source, raw["overrides"])
    resolved["schema"] = schema
    return raw, resolved


def validate_dependencies(protocol: dict[str, Any], artifact_root: Path) -> tuple[Path, Path]:
    data_root, model_root = base.verify_protocol_dependencies(protocol, artifact_root)
    predecessor = protocol["predecessor_pairwise"]
    for key in ("protocol", "cohort", "result", "implementation"):
        base.verify_path(
            HERE / predecessor[f"{key}_path"],
            predecessor[f"{key}_sha256"],
            "PAIRWISE_PREDECESSOR",
        )
    result = base.load_json(HERE / predecessor["result_path"])
    require(result.get("conclusion") == predecessor["required_conclusion"], "PAIRWISE_CONCLUSION")
    return data_root, model_root


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    raw_protocol, protocol = resolve_protocol(protocol_path)
    data_root, _ = validate_dependencies(protocol, artifact_root)
    excluded, exclusion_receipts = base.consumed_triples(protocol)
    excluded_physical = {(reference, target) for reference, _, target in excluded}
    exclude_prior_physical = bool(protocol["source"].get("exclude_prior_physical_targets", False))
    candidate_protocol = base.load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    rules = protocol["frozen_cohort"]["frame_rules"]
    maximum_per_reference = int(protocol["frozen_cohort"]["maximum_targets_per_reference_scan"])
    cache: dict[tuple[str, int], tuple[dict[str, Any] | None, dict[str, int]]] = {}
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    selected: list[dict[str, Any]] = []
    used_physical: set[tuple[str, int]] = set()
    reference_counts: dict[str, int] = {}
    considered = 0

    def frame(scan_id: str, target_id: int) -> dict[str, Any] | None:
        key = (scan_id, target_id)
        if key not in cache:
            cache[key] = pixel.select_frame(data_root, scan_id, target_id, rules)
            for name, count in cache[key][1].items():
                opened[name] += count
        return cache[key][0]

    for row in rows:
        triple = (
            str(row["reference_scan_id"]),
            str(row["rescan_id"]),
            int(row["target_instance_id"]),
        )
        physical = (triple[0], triple[2])
        if (
            triple in excluded
            or (exclude_prior_physical and physical in excluded_physical)
            or physical in used_physical
            or reference_counts.get(triple[0], 0) >= maximum_per_reference
        ):
            continue
        if not all((data_root / scan / "sequence.zip").is_file() for scan in triple[:2]):
            continue
        considered += 1
        reference = frame(triple[0], triple[2])
        query = frame(triple[1], triple[2])
        if reference is None or query is None:
            continue
        selected.append(
            {
                "episode_id": f"RA{len(selected) + 1:02d}",
                **row,
                "reference": reference,
                "query": query,
            }
        )
        used_physical.add(physical)
        reference_counts[triple[0]] = reference_counts.get(triple[0], 0) + 1
        if len(selected) == int(protocol["frozen_cohort"]["physical_targets"]):
            break

    require(opened["rgb_members"] == 0, "RGB_OPENED_BEFORE_FREEZE")
    require(
        len(selected) == int(protocol["frozen_cohort"]["physical_targets"]),
        f"ROSTER_CONFIRMATION_COHORT_NOT_EVALUABLE:{len(selected)}",
    )
    source_manifest: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for scan_id in (episode["reference_scan_id"], episode["rescan_id"]):
            for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip"):
                path = data_root / scan_id / name
                source_manifest[f"{scan_id}/{name}"] = base.source_record(path, artifact_root)
    for name in ("3RScan.json", "objects.json"):
        source_manifest[name] = base.source_record(data_root / name, artifact_root)

    images: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for role, scan_key in (("reference", "reference_scan_id"), ("query", "rescan_id")):
            selected_frame = episode[role]
            key = f"{episode['episode_id']}_{role}"
            images[key] = {
                "episode_id": episode["episode_id"],
                "role": role,
                "scan_id": episode[scan_key],
                "target_instance_id": episode["target_instance_id"],
                "target_label": episode["target_label"],
                "frame": selected_frame["frame"],
                "color_size": selected_frame["color_size"],
                "bbox_xyxy": selected_frame["bbox_xyxy"],
                "image_margin_pixels": selected_frame["image_margin_pixels"],
                "inside_vertex_fraction": selected_frame["inside_vertex_fraction"],
                "depth_visible_ratio": selected_frame["depth_visible_ratio"],
                "zip_member": f"frame-{int(selected_frame['frame']):06d}.color.jpg",
            }
    episode_ids = [episode["episode_id"] for episode in selected]
    anchors = []
    for anchor_role, candidate_role in (("reference", "query"), ("query", "reference")):
        for episode_id in episode_ids:
            anchors.append(
                {
                    "anchor": f"{episode_id}_{anchor_role}",
                    "positive": f"{episode_id}_{candidate_role}",
                    "candidates": [f"{candidate}_{candidate_role}" for candidate in episode_ids],
                }
            )
    cohort = {
        "schema": COHORT_SCHEMAS[raw_protocol["schema"]],
        "authority": "FROZEN_PRE_RGB_TARGET_TRIPLE_DISJOINT_CLOSED_ROSTER_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "entrypoint_sha256": base.sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_target_triples": len(excluded),
            "consumed_physical_targets": len(excluded_physical),
            "exclude_prior_physical_targets": exclude_prior_physical,
            "candidate_rows_considered": considered,
            "unique_physical_target_rule": True,
            "maximum_targets_per_reference_scan": maximum_per_reference,
            "reference_scan_target_counts": dict(sorted(reference_counts.items())),
            "opened_members": opened,
            "frame_rules": rules,
            "exclusion_receipts": exclusion_receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": selected,
        "images": images,
        "anchors": anchors,
        "counts": {
            "physical_targets": 3,
            "directed_independent_anchors": 6,
            "positive_pairs": 6,
            "negative_pairs": 12,
            "complete_assignments": 6,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    base.write_json(cohort_path, cohort)


def assignment_metrics(matrix: np.ndarray, episode_ids: list[str]) -> dict[str, Any]:
    rows, columns = linear_sum_assignment(-matrix)
    require(np.array_equal(rows, np.arange(len(episode_ids))), "ASSIGNMENT_ROWS")
    totals = []
    for permutation in itertools.permutations(range(len(episode_ids))):
        total = float(sum(matrix[index, candidate] for index, candidate in enumerate(permutation)))
        totals.append((total, permutation))
    totals.sort(key=lambda item: (-item[0], item[1]))
    selected_total = float(matrix[rows, columns].sum())
    require(abs(selected_total - totals[0][0]) < 1e-6, "ASSIGNMENT_TOTAL")
    correct = int(np.count_nonzero(columns == np.arange(len(episode_ids))))
    return {
        "assigned_correct": correct,
        "assigned_accuracy": round(correct / len(episode_ids), 6),
        "bidirectional_identity_equivalent_correct": correct * 2,
        "selected_total_score": round(selected_total, 6),
        "second_best_total_score": round(totals[1][0], 6),
        "assignment_margin": round(selected_total - totals[1][0], 6),
        "assignment": [
            {"reference": episode_ids[index], "query": episode_ids[int(columns[index])]}
            for index in range(len(episode_ids))
        ],
    }


def replay(
    protocol_path: Path,
    cohort_path: Path,
    artifact_root: Path,
    crop_dir: Path,
    result_path: Path,
) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    raw_protocol, protocol = resolve_protocol(protocol_path)
    cohort = base.load_json(cohort_path)
    require(
        cohort.get("schema") == COHORT_SCHEMAS[raw_protocol["schema"]],
        "COHORT_SCHEMA_MISMATCH",
    )
    require(cohort["protocol_sha256"] == base.sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    require(cohort["entrypoint_sha256"] == base.sha256(Path(__file__).resolve()), "COHORT_ENTRYPOINT_SHA256")
    validate_dependencies(protocol, artifact_root)
    embeddings, rgb_receipts = base.encode(protocol, cohort, artifact_root, crop_dir)
    episode_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    baseline_matrix = np.zeros((len(episode_ids), len(episode_ids)), dtype=np.float64)
    upgraded_matrix = np.zeros_like(baseline_matrix)
    rows: list[dict[str, Any]] = []
    for i, reference in enumerate(episode_ids):
        for j, query in enumerate(episode_ids):
            baseline_score, upgraded_score = base.pair_scores(
                embeddings[f"{reference}_reference"], embeddings[f"{query}_query"]
            )
            baseline_matrix[i, j] = baseline_score
            upgraded_matrix[i, j] = upgraded_score
    for group in cohort["anchors"]:
        for candidate in group["candidates"]:
            anchor_episode, anchor_role = group["anchor"].rsplit("_", 1)
            candidate_episode, _ = candidate.rsplit("_", 1)
            i = episode_ids.index(anchor_episode)
            j = episode_ids.index(candidate_episode)
            if anchor_role == "query":
                i, j = j, i
            rows.append(
                {
                    "anchor": group["anchor"],
                    "candidate": candidate,
                    "label": int(candidate == group["positive"]),
                    "baseline": float(baseline_matrix[i, j]),
                    "upgraded": float(upgraded_matrix[i, j]),
                }
            )
    labels = np.asarray([row["label"] for row in rows])
    baseline_scores = np.asarray([row["baseline"] for row in rows])
    upgraded_scores = np.asarray([row["upgraded"] for row in rows])
    baseline_assignment = assignment_metrics(baseline_matrix, episode_ids)
    upgraded_assignment = assignment_metrics(upgraded_matrix, episode_ids)
    upgraded_independent = base.retrieval(rows, "upgraded")
    gate_met = (
        upgraded_assignment["assigned_correct"] == len(episode_ids)
        and upgraded_assignment["assignment_margin"] > 0.0
    )
    result = {
        "schema": RESULT_SCHEMAS[raw_protocol["schema"]],
        "authority": "CONSUMED_TARGET_TRIPLE_DISJOINT_CLOSED_ROSTER_ASSIGNMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": base.sha256(cohort_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__).resolve())},
        "conclusion": (
            "L10_3RSCAN_ROSTER_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_ROSTER_ASSIGNMENT_TARGET_DISJOINT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "baseline": {
                "pair_auroc": round(float(roc_auc_score(labels, baseline_scores)), 6),
                "pair_average_precision": round(float(average_precision_score(labels, baseline_scores)), 6),
                "independent_retrieval": base.retrieval(rows, "baseline"),
                "roster_assignment": baseline_assignment,
                "score_matrix": baseline_matrix.round(6).tolist(),
            },
            "upgraded": {
                "pair_auroc": round(float(roc_auc_score(labels, upgraded_scores)), 6),
                "pair_average_precision": round(float(average_precision_score(labels, upgraded_scores)), 6),
                "independent_retrieval": upgraded_independent,
                "roster_assignment": upgraded_assignment,
                "score_matrix": upgraded_matrix.round(6).tolist(),
            },
        },
        "assignment_gain": {
            "directed_correct_equivalent": (
                upgraded_assignment["bidirectional_identity_equivalent_correct"]
                - upgraded_independent["top1_correct"]
            ),
            "directed_accuracy_equivalent": round(
                upgraded_assignment["assigned_accuracy"]
                - upgraded_independent["top1_accuracy"],
                6,
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
