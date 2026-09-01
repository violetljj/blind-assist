#!/usr/bin/env python3
"""Rank adjacent-view proposal masks by symmetric depth/pose 3D persistence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-query-mask-3d-track-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-query-mask-3d-track-result-v1"


def _lift(
    mask: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    info: dict[str, Any],
) -> np.ndarray:
    rows, columns = np.indices(depth.shape)
    metres = depth.astype(np.float64) / 1000.0
    valid = metres > 0.0
    intrinsic = info["depth_intrinsic"]
    camera = np.stack(
        (
            (columns - intrinsic[0, 2]) * metres / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * metres / intrinsic[1, 1],
            metres,
        ),
        axis=-1,
    )
    projected = camera @ info["color_intrinsic"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[..., :2] / projected[..., 2:3]
    finite = np.isfinite(pixels).all(axis=-1)
    xs = np.zeros(depth.shape, dtype=np.int64)
    ys = np.zeros(depth.shape, dtype=np.int64)
    xs[finite] = np.rint(pixels[..., 0][finite]).astype(np.int64)
    ys[finite] = np.rint(pixels[..., 1][finite]).astype(np.int64)
    inside = (
        valid
        & finite
        & (xs >= 0)
        & (xs < int(info["color_width"]))
        & (ys >= 0)
        & (ys < int(info["color_height"]))
    )
    selected = np.zeros_like(inside)
    selected[inside] = mask[ys[inside], xs[inside]]
    points = camera[selected]
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    scan = np.column_stack((points, np.ones(len(points), dtype=np.float64))) @ pose.T
    return scan[:, :3]


def _coverage(
    points_scan: np.ndarray,
    target_mask: np.ndarray,
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    info: dict[str, Any],
    tolerance_metres: float,
) -> tuple[float, int, int]:
    if not len(points_scan):
        return 0.0, 0, 0
    camera, color_pixels, color_inside = pixel.project_points(
        points_scan,
        target_pose,
        info["color_intrinsic"],
        int(info["color_width"]),
        int(info["color_height"]),
    )
    _, depth_pixels, depth_inside = pixel.project_points(
        points_scan,
        target_pose,
        info["depth_intrinsic"],
        int(info["depth_width"]),
        int(info["depth_height"]),
    )
    indices = np.flatnonzero(color_inside & depth_inside)
    if not len(indices):
        return 0.0, 0, 0
    dx = np.rint(depth_pixels[indices, 0]).astype(np.int64).clip(0, int(info["depth_width"]) - 1)
    dy = np.rint(depth_pixels[indices, 1]).astype(np.int64).clip(0, int(info["depth_height"]) - 1)
    observed = target_depth[dy, dx].astype(np.float64) / 1000.0
    visible = (observed > 0.0) & (np.abs(observed - camera[indices, 2]) <= tolerance_metres)
    visible_indices = indices[visible]
    if not len(visible_indices):
        return 0.0, 0, len(indices)
    cx = np.rint(color_pixels[visible_indices, 0]).astype(np.int64).clip(0, int(info["color_width"]) - 1)
    cy = np.rint(color_pixels[visible_indices, 1]).astype(np.int64).clip(0, int(info["color_height"]) - 1)
    matched = int(np.count_nonzero(target_mask[cy, cx]))
    return float(matched / len(visible_indices)), int(len(visible_indices)), int(len(indices))


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    inputs = {}
    for key in ("source_protocol", "cohort", "intermediate"):
        row = protocol[key]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
        inputs[key] = pixel.load_json(path)
        pixel.require(inputs[key]["schema"] == row["required_schema"], f"{key.upper()}_SCHEMA")
    source_protocol = inputs["source_protocol"]
    cohort = inputs["cohort"]
    intermediate = inputs["intermediate"]
    for dependency in source_protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"SOURCE_DEPENDENCY_HASH:{dependency['path']}")
    masker = source_protocol["masker"]
    pixel.require(pixel.sha256(ROOT / masker["model_path"]) == masker["model_sha256"], "MASKER_HASH")

    images, image_rows = nids.ffa._load_images(source_protocol, cohort)
    query_keys = list(protocol["evaluation"]["query_keys"])
    candidates = {key: list(intermediate["query_receipts"][key]["ranked_candidates"]) for key in query_keys}

    import torch
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / masker["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    masks = {}
    sam_receipts = {}
    for key in query_keys:
        generated, receipt = nids.sam_base._sam_masks(
            processor,
            model,
            images[key],
            [row["box_xyxy"] for row in candidates[key]],
            images[key].size,
            torch,
            np,
        )
        masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in generated]
        sam_receipts[key] = receipt
        for row, mask in zip(candidates[key], masks[key], strict=True):
            actual = hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
            pixel.require(actual == row["mask_sha256"], f"MASK_REPLAY_MISMATCH:{key}:{row['postprocess_index']}")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    scan_ids = {image_rows[key]["scan_id"] for key in query_keys}
    pixel.require(len(scan_ids) == 1, "QUERY_SCAN_MISMATCH")
    scan_id = next(iter(scan_ids))
    zip_receipt = cohort["source_manifest"][f"{scan_id}/sequence.zip"]
    zip_path = ROOT / source_protocol["source"]["artifact_root"] / zip_receipt["path"]
    with zipfile.ZipFile(zip_path) as archive:
        info_payload = archive.read("_info.txt")
        info = pixel.parse_info(info_payload.decode("utf-8"))
        poses = {key: pixel.read_pose(archive, int(image_rows[key]["frame"])) for key in query_keys}
        depths = {key: pixel.decode_depth(archive, int(image_rows[key]["frame"])) for key in query_keys}
        geometry_receipts = {
            "info_sha256": hashlib.sha256(info_payload).hexdigest(),
            "frames": {
                key: {
                    "frame": int(image_rows[key]["frame"]),
                    "pose_sha256": hashlib.sha256(archive.read(f"frame-{int(image_rows[key]['frame']):06d}.pose.txt")).hexdigest(),
                    "depth_sha256": hashlib.sha256(archive.read(f"frame-{int(image_rows[key]['frame']):06d}.depth.pgm")).hexdigest(),
                }
                for key in query_keys
            },
        }
    lifted = {
        key: [_lift(mask, depths[key], poses[key], info) for mask in masks[key]]
        for key in query_keys
    }
    first, second = query_keys
    pair_rows = []
    matrix = np.zeros((len(candidates[first]), len(candidates[second])), dtype=np.float64)
    tolerance = float(protocol["geometry"]["depth_consistency_metres"])
    minimum_visible = int(protocol["geometry"]["minimum_visible_projected_points"])
    for left_index, left_points in enumerate(lifted[first]):
        for right_index, right_points in enumerate(lifted[second]):
            left_to_right, left_visible, left_inside = _coverage(
                left_points, masks[second][right_index], depths[second], poses[second], info, tolerance
            )
            right_to_left, right_visible, right_inside = _coverage(
                right_points, masks[first][left_index], depths[first], poses[first], info, tolerance
            )
            score = (
                math.sqrt(left_to_right * right_to_left)
                if left_visible >= minimum_visible and right_visible >= minimum_visible
                else 0.0
            )
            matrix[left_index, right_index] = score
            pair_rows.append(
                {
                    "first_candidate_index": left_index,
                    "second_candidate_index": right_index,
                    "first_to_second_coverage": left_to_right,
                    "second_to_first_coverage": right_to_left,
                    "first_visible_projected_points": left_visible,
                    "second_visible_projected_points": right_visible,
                    "first_inside_projected_points": left_inside,
                    "second_inside_projected_points": right_inside,
                    "symmetric_track_score": score,
                }
            )
    best_second = np.argmax(matrix, axis=1)
    best_first = np.argmax(matrix, axis=0)
    candidate_scores = {
        first: [],
        second: [],
    }
    for index in range(len(candidates[first])):
        partner = int(best_second[index])
        candidate_scores[first].append(
            {
                "candidate_index": index,
                "partner_index": partner,
                "mutual_best": int(best_first[partner]) == index,
                "track_score": float(matrix[index, partner]),
                "lifted_depth_points": int(len(lifted[first][index])),
                "depth_supported": bool(len(lifted[first][index])),
            }
        )
    for index in range(len(candidates[second])):
        partner = int(best_first[index])
        candidate_scores[second].append(
            {
                "candidate_index": index,
                "partner_index": partner,
                "mutual_best": int(best_second[partner]) == index,
                "track_score": float(matrix[partner, index]),
                "lifted_depth_points": int(len(lifted[second][index])),
                "depth_supported": bool(len(lifted[second][index])),
            }
        )

    episodes = []
    for key in query_keys:
        truth_ious = np.asarray([float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates[key]])
        correct_index = int(np.argmax(truth_ious))
        order = sorted(
            range(len(candidates[key])),
            key=lambda index: (
                -int(candidate_scores[key][index]["depth_supported"]),
                -int(candidate_scores[key][index]["mutual_best"]),
                -candidate_scores[key][index]["track_score"],
                -float(candidates[key][index]["objectness_score"]),
                index,
            ),
        )
        track_rank = order.index(correct_index) + 1
        original_rank = correct_index + 1
        episodes.append(
            {
                "query_key": key,
                "candidate_count": len(order),
                "reachable_correct_candidate_index": correct_index,
                "reachable_correct_iou_evaluation_only": float(truth_ious[correct_index]),
                "original_fused_rank": original_rank,
                "track_rank": track_rank,
                "rank_improvement": original_rank - track_rank,
                "track_recall_at_3": track_rank <= 3,
                "track_top1_iou_evaluation_only": float(truth_ious[order[0]]),
                "ranked_candidates": [
                    {
                        **candidate_scores[key][index],
                        "truth_iou_evaluation_only": float(truth_ious[index]),
                    }
                    for index in order
                ],
            }
        )
    gate_met = (
        all(row["track_recall_at_3"] for row in episodes)
        and sum(row["rank_improvement"] > 0 for row in episodes) >= int(protocol["gate"]["minimum_improved_queries"])
        and min(row["rank_improvement"] for row in episodes) >= 0
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_ARV_ADJACENT_QUERY_MASK_3D_TRACK_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "source_protocol": protocol["source_protocol"],
        "cohort": protocol["cohort"],
        "intermediate": protocol["intermediate"],
        "geometry": protocol["geometry"],
        "sam_receipts": sam_receipts,
        "geometry_receipts": geometry_receipts,
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
        "episodes": episodes,
        "metrics": {
            "query_count": len(episodes),
            "original_recall_at_3": sum(row["original_fused_rank"] <= 3 for row in episodes),
            "track_recall_at_3": sum(row["track_recall_at_3"] for row in episodes),
            "improved_queries": sum(row["rank_improvement"] > 0 for row in episodes),
            "regressed_queries": sum(row["rank_improvement"] < 0 for row in episodes),
            "mean_rank_improvement": float(np.mean([row["rank_improvement"] for row in episodes])),
            "minimum_track_top1_iou": min(row["track_top1_iou_evaluation_only"] for row in episodes),
            "mean_track_top1_iou": float(np.mean([row["track_top1_iou_evaluation_only"] for row in episodes])),
            "depth_unsupported_candidates": sum(
                not row["depth_supported"]
                for rows in candidate_scores.values()
                for row in rows
            ),
        },
        "literature_motivation": protocol["literature_motivation"],
        "gate": {**protocol["gate"], "met": gate_met},
        "runtime": {
            "rgb_members_opened_for_mask_replay": len(query_keys),
            "sam_mask_calls": len(query_keys),
            "pose_members_opened": len(query_keys),
            "depth_members_opened": len(query_keys),
            "appearance_model_calls": 0,
            "model_training_steps": 0,
        },
        "conclusion": (
            "L10_3RSCAN_QUERY_MASK_3D_TRACK_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_QUERY_MASK_3D_TRACK_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": protocol["next_action"] if gate_met else protocol["fallback_action"],
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
