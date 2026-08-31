#!/usr/bin/env python3
"""Gate a new 3RScan family on joint surface and viewpoint opportunity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as joint  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-transport-opportunity-source-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-transport-opportunity-source-result-v1"


def _unit_view(pose: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    direction = centroid - pose[:3, 3]
    norm = float(np.linalg.norm(direction))
    pixel.require(norm > 0.0, "ZERO_VIEW_DIRECTION")
    return direction / norm


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for row in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"DEPENDENCY_HASH:{row['path']}")
    candidate_path = HERE / protocol["candidate"]["path"]
    pixel.require(pixel.sha256(candidate_path) == protocol["candidate"]["sha256"], "CANDIDATE_HASH")
    candidate = pixel.load_json(candidate_path)["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    data_root = artifact_root / "datasets/3rscan"
    target_id = int(candidate["target_instance_id"])
    ref_points, ref_candidates, ref_opened = joint._candidates(
        data_root, str(candidate["reference_scan_id"]), target_id, protocol["candidate_view_rules"]
    )
    query_points, query_candidates, query_opened = joint._candidates(
        data_root, str(candidate["rescan_id"]), target_id, protocol["candidate_view_rules"]
    )
    matrix = extent.provider_matrix(candidate["transform"])
    query_in_reference = extent.transform_points(query_points, matrix)
    distance = np.linalg.norm(ref_points[:, None, :] - query_in_reference[None, :, :], axis=2)
    ref_to_query = np.argmin(distance, axis=1)
    query_to_ref = np.argmin(distance, axis=0)
    ref_indices = np.arange(len(ref_points), dtype=np.int64)
    mutual = query_to_ref[ref_to_query] == ref_indices
    mutual &= distance[ref_indices, ref_to_query] <= float(protocol["surface_correspondence"]["maximum_distance_metres"])
    matched_ref = ref_indices[mutual]
    matched_query = ref_to_query[mutual]
    pixel.require(len(matched_ref) > 0, "NO_MUTUAL_TARGET_SURFACE_CORRESPONDENCES")
    centroid = np.mean(ref_points[matched_ref], axis=0)
    pairs = []
    for ref_row, ref_visible, ref_pose in ref_candidates:
        ref_direction = _unit_view(ref_pose, centroid)
        for query_row, query_visible, query_pose in query_candidates:
            common = int(np.count_nonzero(ref_visible[matched_ref] & query_visible[matched_query]))
            fraction = float(common / len(matched_ref))
            cosine = float(np.dot(ref_direction, _unit_view(matrix @ query_pose, centroid)))
            score = fraction * max(0.0, cosine)
            pairs.append(
                {
                    "reference": ref_row,
                    "query": query_row,
                    "mutual_surface_vertices": int(len(matched_ref)),
                    "joint_visible_surface_vertices": common,
                    "joint_visible_surface_fraction": fraction,
                    "view_direction_cosine": cosine,
                    "transport_opportunity_score": score,
                }
            )
    selected = max(
        pairs,
        key=lambda row: (
            float(row["transport_opportunity_score"]),
            -int(row["reference"]["frame"]),
            -int(row["query"]["frame"]),
        ),
        default=None,
    )
    evaluable = bool(
        selected is not None
        and float(selected["joint_visible_surface_fraction"]) >= float(protocol["decision_gate"]["minimum_joint_visible_surface_fraction"])
        and float(selected["view_direction_cosine"]) >= float(protocol["decision_gate"]["minimum_view_direction_cosine"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_NEW_FAMILY_PRE_RGB_PRE_MODEL_TRANSPORT_OPPORTUNITY_SOURCE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": (
            "L10_3RSCAN_TRANSPORT_OPPORTUNITY_SOURCE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_TRANSPORT_OPPORTUNITY_SOURCE_NOT_EVALUABLE"
        ),
        "source_evaluable": evaluable,
        "reference_candidate_views": len(ref_candidates),
        "query_candidate_views": len(query_candidates),
        "joint_candidate_pairs": len(pairs),
        "mutual_target_surface_vertices": int(len(matched_ref)),
        "selected": selected,
        "decision_gate": protocol["decision_gate"],
        "opened": {"reference": ref_opened, "query": query_opened},
        "rgb_members_opened": 0,
        "model_calls": 0,
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
