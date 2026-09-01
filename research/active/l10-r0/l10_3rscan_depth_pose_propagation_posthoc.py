#!/usr/bin/env python3
"""Generate one missing proposal by depth-and-pose propagation across adjacent frames."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_temporal_scale_vacancy_confirmation as confirm  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-depth-pose-propagation-posthoc-protocol-v1"


def _matrix(values: str) -> np.ndarray:
    matrix = np.asarray([float(value) for value in values.split()], dtype=np.float64)
    return matrix.reshape(4, 4)


def _sequence_info(payload: bytes) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key.strip()] = value.strip()
    return rows


def _reference_areas(
    cohort: dict[str, Any], query_names: list[str]
) -> list[float]:
    prefixes = {name.split("_")[0] for name in query_names}
    areas = [
        confirm.consensus.base._area_fraction(
            row["bbox_xyxy"], *map(int, row["color_size"])
        )
        for key, row in cohort["images"].items()
        if key.endswith("_reference") and key.split("_")[0] in prefixes
    ]
    confirm.consensus.base.pixel.require(len(areas) == 3, "REFERENCE_SCALE_COUNT")
    return areas


def _preserved_sets(
    candidates_result: dict[str, Any],
    cohort: dict[str, Any],
    query_names: list[str],
    penalty: float,
) -> tuple[dict[str, list[dict[str, Any]]], str, list[str], list[float]]:
    reference_areas = _reference_areas(cohort, query_names)
    fused = {
        query: confirm.consensus.base._best(
            candidates_result["query_receipts"][query]["ranked_candidates"],
            lambda row: float(row["layer18_nids_fused_score"]),
        )
        for query in query_names
    }
    reference_order = sorted(
        {
            reference
            for query in query_names
            for reference in candidates_result["query_receipts"][query][
                "ranked_candidates"
            ][0]["per_reference_scores"]
        }
    )
    votes = [fused[query]["winning_target_reference"] for query in query_names]
    dominant_reference = sorted(
        set(votes),
        key=lambda reference: (-votes.count(reference), reference_order.index(reference)),
    )[0]
    preserved: dict[str, list[dict[str, Any]]] = {}
    for query in query_names:
        candidates = candidates_result["query_receipts"][query]["ranked_candidates"]
        width, height = map(int, cohort["images"][query]["color_size"])
        scale = confirm.consensus.base._best(
            candidates,
            lambda row: float(row["layer18_local_appearance_score"])
            - penalty
            * confirm._scale_distance(
                row["box_xyxy"], width, height, reference_areas
            ),
        )
        contributors = [
            ("semantic_local_fusion", fused[query]),
            ("reference_scale_mixture", scale),
        ]
        constrained = [
            row
            for row in candidates
            if row["winning_target_reference"] == dominant_reference
        ]
        if constrained:
            contributors.append(
                (
                    "cross_view_reference_consensus",
                    confirm.consensus.base._best(
                        constrained,
                        lambda row: float(row["layer18_nids_fused_score"]),
                    ),
                )
            )
        hypotheses: list[dict[str, Any]] = []
        for mechanism, candidate in contributors:
            key = confirm._candidate_key(candidate)
            existing = next(
                (row for row in hypotheses if row["candidate_key"] == key), None
            )
            if existing is None:
                hypotheses.append(
                    {
                        "candidate_key": key,
                        "mechanisms": [mechanism],
                        "candidate": candidate,
                    }
                )
            else:
                existing["mechanisms"].append(mechanism)
        preserved[query] = hypotheses
    return preserved, dominant_reference, votes, reference_areas


def _anchor(
    preserved: dict[str, list[dict[str, Any]]], query_names: list[str]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
    for query_index, query in enumerate(query_names):
        for hypothesis in preserved[query]:
            candidate = hypothesis["candidate"]
            ranked.append(
                (
                    (
                        float(len(hypothesis["mechanisms"])),
                        float(candidate["layer18_nids_fused_score"]),
                        float(-query_index),
                    ),
                    query,
                    hypothesis,
                )
            )
    key, query, hypothesis = max(ranked, key=lambda row: row[0])
    return query, hypothesis, {
        "mechanism_agreement_count": int(key[0]),
        "layer18_nids_fused_score": key[1],
        "query_tie_index": int(-key[2]),
    }


def _propagate_box(
    sequence_zip: Path,
    anchor_frame: int,
    target_frame: int,
    anchor_box: list[float],
) -> tuple[list[float], dict[str, Any]]:
    with ZipFile(sequence_zip) as archive:
        info_payload = archive.read("_info.txt")
        info = _sequence_info(info_payload)
        color_width = int(info["m_colorWidth"])
        color_height = int(info["m_colorHeight"])
        depth_shift = float(info["m_depthShift"])
        color_intrinsic = _matrix(info["m_calibrationColorIntrinsic"])[0:3, 0:3]
        depth_intrinsic = _matrix(info["m_calibrationDepthIntrinsic"])[0:3, 0:3]
        color_extrinsic = _matrix(info["m_calibrationColorExtrinsic"])
        depth_extrinsic = _matrix(info["m_calibrationDepthExtrinsic"])
        confirm.consensus.base.pixel.require(
            np.allclose(color_extrinsic, np.eye(4))
            and np.allclose(depth_extrinsic, np.eye(4)),
            "NON_IDENTITY_SENSOR_EXTRINSIC",
        )
        depth_name = f"frame-{anchor_frame:06d}.depth.pgm"
        anchor_pose_name = f"frame-{anchor_frame:06d}.pose.txt"
        target_pose_name = f"frame-{target_frame:06d}.pose.txt"
        depth_payload = archive.read(depth_name)
        anchor_pose_payload = archive.read(anchor_pose_name)
        target_pose_payload = archive.read(target_pose_name)
        depth = (
            np.asarray(Image.open(BytesIO(depth_payload))).astype(np.float64)
            / depth_shift
        )
        anchor_pose = np.loadtxt(BytesIO(anchor_pose_payload), dtype=np.float64)
        target_pose = np.loadtxt(BytesIO(target_pose_payload), dtype=np.float64)

    rows, columns = np.indices(depth.shape)
    valid = depth > 0.0
    depth_points = np.stack(
        [
            (columns - depth_intrinsic[0, 2]) * depth / depth_intrinsic[0, 0],
            (rows - depth_intrinsic[1, 2]) * depth / depth_intrinsic[1, 1],
            depth,
        ],
        axis=-1,
    )
    projected_color = depth_points @ color_intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_color = projected_color[..., :2] / projected_color[..., 2:3]
    selected = (
        valid
        & (projected_color[..., 0] >= float(anchor_box[0]))
        & (projected_color[..., 0] <= float(anchor_box[2]))
        & (projected_color[..., 1] >= float(anchor_box[1]))
        & (projected_color[..., 1] <= float(anchor_box[3]))
    )
    selected_depth = depth[selected]
    confirm.consensus.base.pixel.require(selected_depth.size > 0, "NO_ANCHOR_DEPTH")
    median_depth = float(np.median(selected_depth))
    corners = np.asarray(
        [
            [anchor_box[0], anchor_box[1], 1.0],
            [anchor_box[2], anchor_box[1], 1.0],
            [anchor_box[2], anchor_box[3], 1.0],
            [anchor_box[0], anchor_box[3], 1.0],
        ],
        dtype=np.float64,
    )
    anchor_points = (
        np.linalg.inv(color_intrinsic) @ corners.T
    ).T * median_depth
    target_from_anchor = np.linalg.inv(target_pose) @ anchor_pose
    target_points = (
        target_from_anchor
        @ np.concatenate(
            [anchor_points, np.ones((anchor_points.shape[0], 1))], axis=1
        ).T
    ).T[:, :3]
    confirm.consensus.base.pixel.require(
        bool(np.all(target_points[:, 2] > 0.0)), "PROPAGATED_BEHIND_CAMERA"
    )
    target_pixels = (color_intrinsic @ target_points.T).T
    target_pixels = target_pixels[:, :2] / target_pixels[:, 2:3]
    box = [
        float(np.clip(target_pixels[:, 0].min(), 0.0, color_width)),
        float(np.clip(target_pixels[:, 1].min(), 0.0, color_height)),
        float(np.clip(target_pixels[:, 0].max(), 0.0, color_width)),
        float(np.clip(target_pixels[:, 1].max(), 0.0, color_height)),
    ]
    confirm.consensus.base.pixel.require(
        box[2] > box[0] and box[3] > box[1], "EMPTY_PROPAGATED_BOX"
    )
    return box, {
        "anchor_frame": anchor_frame,
        "target_frame": target_frame,
        "valid_anchor_depth_pixels": int(selected_depth.size),
        "median_anchor_depth_m": median_depth,
        "projected_corner_pixels": target_pixels.tolist(),
        "color_intrinsic": color_intrinsic.tolist(),
        "depth_intrinsic": depth_intrinsic.tolist(),
        "info_sha256": hashlib.sha256(info_payload).hexdigest(),
        "depth_member": depth_name,
        "depth_sha256": hashlib.sha256(depth_payload).hexdigest(),
        "anchor_pose_member": anchor_pose_name,
        "anchor_pose_sha256": hashlib.sha256(anchor_pose_payload).hexdigest(),
        "target_pose_member": target_pose_name,
        "target_pose_sha256": hashlib.sha256(target_pose_payload).hexdigest(),
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = confirm.consensus.base.pixel.load_json(protocol_path)
    confirm.consensus.base.pixel.require(
        protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA"
    )
    confirm.consensus.base.pixel.require(
        confirm.consensus.base.pixel.sha256(Path(__file__))
        == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        confirm.consensus.base.pixel.require(
            confirm.consensus.base.pixel.sha256(HERE / dependency["path"])
            == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    for key in ("predecessor", "cohort", "candidate_result"):
        row = protocol[key]
        confirm.consensus.base.pixel.require(
            confirm.consensus.base.pixel.sha256(HERE / row["path"])
            == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = confirm.consensus.base.pixel.load_json(
        HERE / protocol["predecessor"]["path"]
    )
    confirm.consensus.base.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    cohort = confirm.consensus.base.pixel.load_json(HERE / protocol["cohort"]["path"])
    candidates_result = confirm.consensus.base.pixel.load_json(
        HERE / protocol["candidate_result"]["path"]
    )
    sequence_zip = Path(protocol["geometry"]["sequence_zip"])
    confirm.consensus.base.pixel.require(
        confirm.consensus.base.pixel.sha256(sequence_zip)
        == protocol["geometry"]["sequence_sha256"],
        "SEQUENCE_HASH",
    )
    query_names = [str(value) for value in protocol["geometry"]["query_images"]]
    penalty = float(protocol["hypothesis_generation"]["scale_log_area_penalty"])
    preserved, dominant_reference, votes, reference_areas = _preserved_sets(
        candidates_result, cohort, query_names, penalty
    )
    anchor_query, anchor_hypothesis, anchor_receipt = _anchor(preserved, query_names)
    target_query = next(query for query in query_names if query != anchor_query)
    anchor_box = anchor_hypothesis["candidate"]["box_xyxy"]
    propagated_box, geometry_receipt = _propagate_box(
        sequence_zip,
        int(cohort["images"][anchor_query]["frame"]),
        int(cohort["images"][target_query]["frame"]),
        anchor_box,
    )

    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    maximum_size = int(protocol["decision_gate"]["maximum_set_size"])
    query_receipts: dict[str, Any] = {}
    preserved_covered = 0
    final_covered = 0
    best_ious: list[float] = []
    propagated_hits = 0
    incremental_hits = 0
    for query in query_names:
        hypotheses = deepcopy(preserved[query])
        truth = cohort["images"][query]["bbox_xyxy"]
        preserved_best = max(
            confirm._iou(row["candidate"]["box_xyxy"], truth) for row in hypotheses
        )
        preserved_covered += int(preserved_best >= minimum_iou)
        propagated_added = False
        if query == target_query and len(hypotheses) < maximum_size:
            hypotheses.append(
                {
                    "candidate_key": (
                        "depth_pose_propagation",
                        anchor_query,
                        tuple(propagated_box),
                    ),
                    "mechanisms": ["depth_pose_rectangle_propagation"],
                    "candidate": {
                        "box_xyxy": propagated_box,
                        "anchor_query": anchor_query,
                        "anchor_candidate_key": list(
                            confirm._candidate_key(anchor_hypothesis["candidate"])
                        ),
                        "anchor_mechanisms": anchor_hypothesis["mechanisms"],
                    },
                }
            )
            propagated_added = True
        confirm.consensus.base.pixel.require(
            len(hypotheses) <= maximum_size, f"SET_SIZE:{query}"
        )
        evaluated: list[dict[str, Any]] = []
        propagated_iou = None
        for hypothesis in hypotheses:
            candidate = deepcopy(hypothesis["candidate"])
            iou = confirm._iou(candidate["box_xyxy"], truth)
            candidate["target_metrics_evaluation_only"] = {"iou": iou}
            evaluated.append(
                {"mechanisms": hypothesis["mechanisms"], "candidate": candidate}
            )
            if "depth_pose_rectangle_propagation" in hypothesis["mechanisms"]:
                propagated_iou = iou
        propagated_hits += int(
            propagated_iou is not None and propagated_iou >= minimum_iou
        )
        incremental_hits += int(
            propagated_iou is not None
            and preserved_best < minimum_iou
            and propagated_iou >= minimum_iou
        )
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in evaluated
        )
        covered = best_iou >= minimum_iou
        final_covered += int(covered)
        best_ious.append(best_iou)
        query_receipts[query] = {
            "query_truth_used_for_hypothesis_generation": False,
            "is_anchor_query": query == anchor_query,
            "propagated_candidate_added": propagated_added,
            "preserved_best_iou_evaluation_only": preserved_best,
            "propagated_iou_evaluation_only": propagated_iou,
            "hypothesis_count": len(evaluated),
            "hypotheses": evaluated,
            "best_hypothesis_iou_evaluation_only": best_iou,
            "target_covered_at_iou_gate": covered,
        }
    gate_met = final_covered == int(
        protocol["decision_gate"]["required_covered_queries"]
    )
    result = {
        "schema": "blindassist-l10-3rscan-depth-pose-propagation-posthoc-result-v1",
        "authority": "CONSUMED_TENTH_FAMILY_DEPTH_POSE_PROPAGATION_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": confirm.consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": confirm.consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_DEPTH_POSE_PROPAGATION_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_DEPTH_POSE_PROPAGATION_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "anchor_selection": {
            "query": anchor_query,
            "candidate_key": list(confirm._candidate_key(anchor_hypothesis["candidate"])),
            "mechanisms": anchor_hypothesis["mechanisms"],
            **anchor_receipt,
        },
        "target_query": target_query,
        "dominant_reference": dominant_reference,
        "semantic_local_fusion_reference_votes": votes,
        "reference_area_fractions": sorted(reference_areas),
        "geometry_receipt": geometry_receipt,
        "metrics": {
            "query_count": len(query_names),
            "preserved_covered_queries": preserved_covered,
            "final_covered_queries": final_covered,
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "propagated_iou_gate_queries": propagated_hits,
            "propagated_incremental_iou_gate_queries": incremental_hits,
        },
        "query_receipts": query_receipts,
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    confirm.consensus.base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
