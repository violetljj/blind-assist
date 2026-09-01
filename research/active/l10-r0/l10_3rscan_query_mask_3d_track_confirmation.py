#!/usr/bin/env python3
"""Confirm bounded 3D-track candidate recovery on a fresh 3RScan family."""

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
import l10_3rscan_query_mask_3d_track as track  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-query-mask-3d-track-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-query-mask-3d-track-confirmation-result-v1"


def _edge(
    left_key: str,
    right_key: str,
    candidates: dict[str, list[dict[str, Any]]],
    masks: dict[str, list[np.ndarray]],
    lifted: dict[str, list[np.ndarray]],
    depths: dict[str, np.ndarray],
    poses: dict[str, np.ndarray],
    info: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, list[dict[str, Any]]]]:
    matrix = np.zeros((len(candidates[left_key]), len(candidates[right_key])), dtype=np.float64)
    tolerance = float(protocol["geometry"]["depth_consistency_metres"])
    minimum_visible = int(protocol["geometry"]["minimum_visible_projected_points"])
    for left_index, left_points in enumerate(lifted[left_key]):
        for right_index, right_points in enumerate(lifted[right_key]):
            left_to_right, left_visible, _ = track._coverage(
                left_points,
                masks[right_key][right_index],
                depths[right_key],
                poses[right_key],
                info,
                tolerance,
            )
            right_to_left, right_visible, _ = track._coverage(
                right_points,
                masks[left_key][left_index],
                depths[left_key],
                poses[left_key],
                info,
                tolerance,
            )
            if left_visible >= minimum_visible and right_visible >= minimum_visible:
                matrix[left_index, right_index] = math.sqrt(left_to_right * right_to_left)
    best_right = np.argmax(matrix, axis=1)
    best_left = np.argmax(matrix, axis=0)
    receipts = {left_key: [], right_key: []}
    for index in range(len(candidates[left_key])):
        partner = int(best_right[index])
        receipts[left_key].append(
            {
                "edge": f"{left_key}->{right_key}",
                "partner_index": partner,
                "mutual_best": int(best_left[partner]) == index,
                "track_score": float(matrix[index, partner]),
            }
        )
    for index in range(len(candidates[right_key])):
        partner = int(best_left[index])
        receipts[right_key].append(
            {
                "edge": f"{right_key}->{left_key}",
                "partner_index": partner,
                "mutual_best": int(best_right[partner]) == index,
                "track_score": float(matrix[partner, index]),
            }
        )
    return matrix, receipts


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    gate_freeze = None
    if "gate_freeze" in protocol:
        gate_freeze_row = protocol["gate_freeze"]
        gate_freeze_path = HERE / gate_freeze_row["path"]
        pixel.require(pixel.sha256(gate_freeze_path) == gate_freeze_row["sha256"], "GATE_FREEZE_HASH")
        gate_freeze = pixel.load_json(gate_freeze_path)
        pixel.require(gate_freeze["schema"] == gate_freeze_row["required_schema"], "GATE_FREEZE_SCHEMA")
    cohort_row = protocol["cohort"]
    cohort_path = HERE / cohort_row["path"]
    pixel.require(pixel.sha256(cohort_path) == cohort_row["sha256"], "COHORT_HASH")
    cohort = pixel.load_json(cohort_path)
    pixel.require(cohort["schema"] == cohort_row["required_schema"], "COHORT_SCHEMA")
    if "selector" in protocol:
        selector_row = protocol["selector"]
        selector_path = HERE / selector_row["path"]
        pixel.require(pixel.sha256(selector_path) == selector_row["sha256"], "SELECTOR_HASH")
        selector = pixel.load_json(selector_path)
        pixel.require(selector["conclusion"] == selector_row["required_conclusion"], "SELECTOR_CONCLUSION")
        pixel.require(int(selector["selected_action"]["frame"]) == int(selector_row["selected_frame"]), "SELECTOR_FRAME")
        action_key = str(cohort["panel"]["action_query_key"])
        pixel.require(int(cohort["images"][action_key]["frame"]) == int(selector_row["selected_frame"]), "SELECTOR_COHORT_FRAME")
    if gate_freeze is not None:
        pixel.require(
            int(cohort["source_queue_index"]) == int(gate_freeze["queue"]["selected_queue_index"]),
            "GATE_FREEZE_QUEUE_INDEX",
        )
    predecessor_row = protocol["predecessor"]
    predecessor_path = HERE / predecessor_row["path"]
    pixel.require(pixel.sha256(predecessor_path) == predecessor_row["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == predecessor_row["required_conclusion"], "PREDECESSOR_CONCLUSION")
    for section in ("proposal", "masker"):
        row = protocol[section]
        pixel.require(pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"], f"MODEL_HASH:{section}")

    images, image_rows = nids.ffa._load_images(protocol, cohort)
    proposals, proposal_runtime = nids.tiled._tiled_proposals(protocol, images)

    import torch
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    masks = {}
    sam_receipts = {}
    for key in protocol["evaluation"]["query_images"]:
        generated, receipt = nids.sam_base._sam_masks(
            processor,
            model,
            images[key],
            [row["box_xyxy"] for row in proposals[key]],
            images[key].size,
            torch,
            np,
        )
        masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in generated]
        sam_receipts[key] = receipt
        for row, mask in zip(proposals[key], masks[key], strict=True):
            row["mask_sha256"] = hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    query_keys = list(protocol["evaluation"]["query_images"])
    scan_ids = {image_rows[key]["scan_id"] for key in query_keys}
    pixel.require(len(scan_ids) == 1, "QUERY_SCAN_MISMATCH")
    scan_id = next(iter(scan_ids))
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    zip_receipt = cohort["source_manifest"][f"{scan_id}/sequence.zip"]
    zip_path = artifact_root / zip_receipt["path"]
    pixel.require(zip_path.stat().st_size == int(zip_receipt["bytes"]), "ZIP_BYTES")
    pixel.require(pixel.sha256(zip_path) == zip_receipt["sha256"], "ZIP_HASH")
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
                    "rgb_sha256": image_rows[key]["image_sha256"],
                    "pose_sha256": hashlib.sha256(archive.read(f"frame-{int(image_rows[key]['frame']):06d}.pose.txt")).hexdigest(),
                    "depth_sha256": hashlib.sha256(archive.read(f"frame-{int(image_rows[key]['frame']):06d}.depth.pgm")).hexdigest(),
                }
                for key in query_keys
            },
        }
    lifted = {
        key: [track._lift(mask, depths[key], poses[key], info) for mask in masks[key]]
        for key in query_keys
    }
    per_candidate_edges: dict[str, list[list[dict[str, Any]]]] = {
        key: [[] for _ in proposals[key]] for key in query_keys
    }
    edge_receipts = []
    for left_key, right_key in zip(query_keys[:-1], query_keys[1:], strict=True):
        matrix, receipts = _edge(
            left_key, right_key, proposals, masks, lifted, depths, poses, info, protocol
        )
        for index, receipt in enumerate(receipts[left_key]):
            per_candidate_edges[left_key][index].append(receipt)
        for index, receipt in enumerate(receipts[right_key]):
            per_candidate_edges[right_key][index].append(receipt)
        edge_receipts.append(
            {
                "left_query": left_key,
                "right_query": right_key,
                "pair_count": int(matrix.size),
                "score_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                "maximum_track_score": float(np.max(matrix)),
                "positive_track_pairs": int(np.count_nonzero(matrix > 0.0)),
            }
        )

    ranked_without_truth = {}
    for key in query_keys:
        rows = []
        for index, proposal in enumerate(proposals[key]):
            edge_rows = per_candidate_edges[key][index]
            rows.append(
                {
                    "candidate_index": index,
                    "proposal": proposal,
                    "depth_supported": bool(len(lifted[key][index])),
                    "mutual_best_edge_count": sum(row["mutual_best"] for row in edge_rows),
                    "track_score": max((row["track_score"] for row in edge_rows), default=0.0),
                    "edges": edge_rows,
                    "lifted_depth_points": int(len(lifted[key][index])),
                }
            )
        ranked_without_truth[key] = sorted(
            rows,
            key=lambda row: (
                -int(row["depth_supported"]),
                -int(row["mutual_best_edge_count"] > 0),
                -row["track_score"],
                -float(row["proposal"]["objectness_score"]),
                row["candidate_index"],
            ),
        )

    gate_mode = protocol["gate"].get("mode", "exact_best_proposal")
    if gate_mode == "target_region_coverage":
        threshold = float(protocol["gate"]["target_region_iou_threshold"])
        pixel.require(gate_freeze is not None, "REGION_COVERAGE_GATE_NOT_FROZEN")
        frozen_gate = gate_freeze["gate"]
        pixel.require(threshold == float(frozen_gate["target_region_iou_threshold"]), "FROZEN_IOU_THRESHOLD")
        pixel.require(int(protocol["gate"]["candidate_budget"]) == int(gate_freeze["mechanism"]["candidate_budget"]), "FROZEN_CANDIDATE_BUDGET")
        pixel.require(int(protocol["gate"]["required_opportunity_frames"]) == int(frozen_gate["required_opportunity_frames"]), "FROZEN_OPPORTUNITY_FRAMES")
        pixel.require(int(protocol["gate"]["required_track_top3_coverage_frames"]) == int(frozen_gate["required_track_top3_coverage_frames"]), "FROZEN_COVERAGE_FRAMES")
    else:
        pixel.require(gate_mode == "exact_best_proposal", "UNKNOWN_GATE_MODE")
        threshold = float(protocol["gate"]["minimum_opportunity_iou"])
    episodes = []
    for key in query_keys:
        truth = image_rows[key]["bbox_xyxy"]
        ordered = ranked_without_truth[key]
        for row in ordered:
            row["target_metrics_evaluation_only"] = nids.base._bbox_metrics(
                row["proposal"]["box_xyxy"], truth
            )
        best = max(ordered, key=lambda row: float(row["target_metrics_evaluation_only"]["iou"]))
        correct_rank = ordered.index(best) + 1
        objectness_order = sorted(
            ordered,
            key=lambda row: (-float(row["proposal"]["objectness_score"]), row["candidate_index"]),
        )
        objectness_rank = objectness_order.index(best) + 1
        best_iou = float(best["target_metrics_evaluation_only"]["iou"])
        candidate_budget = int(protocol["gate"].get("candidate_budget", 3))
        track_topk_best_iou = max(
            float(row["target_metrics_evaluation_only"]["iou"])
            for row in ordered[:candidate_budget]
        )
        objectness_topk_best_iou = max(
            float(row["target_metrics_evaluation_only"]["iou"])
            for row in objectness_order[:candidate_budget]
        )
        episodes.append(
            {
                "query_key": key,
                "frame": int(image_rows[key]["frame"]),
                "candidate_count": len(ordered),
                "proposal_opportunity": best_iou >= threshold,
                "best_reachable_iou_evaluation_only": best_iou,
                "objectness_rank": objectness_rank,
                "track_rank": correct_rank,
                "rank_improvement_over_objectness": objectness_rank - correct_rank,
                "track_recall_at_3": correct_rank <= 3,
                "track_top1_iou_evaluation_only": float(ordered[0]["target_metrics_evaluation_only"]["iou"]),
                "track_top3_best_iou_evaluation_only": track_topk_best_iou,
                "objectness_top3_best_iou_evaluation_only": objectness_topk_best_iou,
                "track_top3_region_coverage": track_topk_best_iou >= threshold,
                "objectness_top3_region_coverage": objectness_topk_best_iou >= threshold,
                "ranked_candidates": ordered,
            }
        )
    if gate_mode == "target_region_coverage":
        gate_met = (
            sum(row["proposal_opportunity"] for row in episodes) == int(protocol["gate"]["required_opportunity_frames"])
            and sum(row["track_top3_region_coverage"] for row in episodes)
            == int(protocol["gate"]["required_track_top3_coverage_frames"])
        )
    else:
        gate_met = (
            sum(row["proposal_opportunity"] for row in episodes) == int(protocol["gate"]["required_opportunity_frames"])
            and sum(row["track_recall_at_3"] for row in episodes) == int(protocol["gate"]["required_track_recall_at_3_frames"])
        )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_PRE_FROZEN_FAMILY_QUERY_MASK_3D_TRACK_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "cohort": cohort_row,
        "predecessor": predecessor_row,
        "proposal": protocol["proposal"],
        "masker": protocol["masker"],
        "geometry": protocol["geometry"],
        "proposal_runtime": proposal_runtime,
        "sam_receipts": sam_receipts,
        "geometry_receipts": geometry_receipts,
        "edge_receipts": edge_receipts,
        "episodes": episodes,
        "metrics": {
            "query_count": len(episodes),
            "proposal_opportunity_frames": sum(row["proposal_opportunity"] for row in episodes),
            "objectness_recall_at_3_frames": sum(row["objectness_rank"] <= 3 for row in episodes),
            "track_recall_at_3_frames": sum(row["track_recall_at_3"] for row in episodes),
            "objectness_top3_region_coverage_frames": sum(row["objectness_top3_region_coverage"] for row in episodes),
            "track_top3_region_coverage_frames": sum(row["track_top3_region_coverage"] for row in episodes),
            "improved_over_objectness_frames": sum(row["rank_improvement_over_objectness"] > 0 for row in episodes),
            "regressed_from_objectness_frames": sum(row["rank_improvement_over_objectness"] < 0 for row in episodes),
            "minimum_best_reachable_iou": min(row["best_reachable_iou_evaluation_only"] for row in episodes),
            "mean_best_reachable_iou": float(np.mean([row["best_reachable_iou_evaluation_only"] for row in episodes])),
            "minimum_track_top1_iou": min(row["track_top1_iou_evaluation_only"] for row in episodes),
            "mean_track_top1_iou": float(np.mean([row["track_top1_iou_evaluation_only"] for row in episodes])),
            "minimum_track_top3_best_iou": min(row["track_top3_best_iou_evaluation_only"] for row in episodes),
            "mean_track_top3_best_iou": float(np.mean([row["track_top3_best_iou_evaluation_only"] for row in episodes])),
            "depth_unsupported_candidates": sum(
                not row["depth_supported"] for rows in ranked_without_truth.values() for row in rows
            ),
        },
        "literature_motivation": protocol["literature_motivation"],
        "gate": {**protocol["gate"], "met": gate_met},
        "runtime": {
            "rgb_members_opened": len(query_keys),
            "grounding_dino_calls": proposal_runtime["grounding_dino_calls"],
            "sam_mask_calls": len(query_keys),
            "appearance_model_calls": 0,
            "model_training_steps": 0,
        },
        "conclusion": protocol["conclusions"]["met" if gate_met else "not_met"] if "conclusions" in protocol else (
            "L10_3RSCAN_QUERY_MASK_3D_TRACK_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_QUERY_MASK_3D_TRACK_FRESH_CONFIRMATION_GATE_NOT_MET"
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
