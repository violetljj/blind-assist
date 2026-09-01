#!/usr/bin/env python3
"""Select an action view by projected hypothesis-set disagreement."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_hypothesis_3d_prompt_transport as transport  # noqa: E402
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402
import l10_3rscan_query_mask_3d_track as track  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-hypothesis-disagreement-action-select-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-hypothesis-disagreement-action-select-result-v1"


def _load_hashed(protocol: dict[str, Any], key: str) -> dict[str, Any]:
    row = protocol[key]
    path = HERE / row["path"]
    pixel.require(pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
    value = pixel.load_json(path)
    if "required_conclusion" in row:
        pixel.require(value["conclusion"] == row["required_conclusion"], f"{key.upper()}_CONCLUSION")
    return value


def _image(archive: zipfile.ZipFile, frame: int) -> tuple[Image.Image, str]:
    payload = archive.read(f"frame-{frame:06d}.color.jpg")
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
    return image, hashlib.sha256(payload).hexdigest()


def _bbox_iou(first: list[float], second: list[float]) -> float:
    iw = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    ih = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = iw * ih
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def run(protocol_path: Path, output_path: Path, cohort_output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    cohort = _load_hashed(protocol, "cohort")
    initial = _load_hashed(protocol, "initial_result")
    motion = _load_hashed(protocol, "motion_candidates")
    for key in ("proposal", "masker"):
        pixel.require(pixel.sha256(Path(protocol[key]["model_path"])) == protocol[key]["model_sha256"], f"MODEL_HASH:{key}")

    current_key = str(protocol["selection"]["current_query_key"])
    action_key = str(protocol["selection"]["action_query_key"])
    current_frame = int(cohort["images"][current_key]["frame"])
    current_episode = next(row for row in initial["episodes"] if int(row["frame"]) == current_frame)
    budget = int(protocol["selection"]["candidate_budget"])
    hypotheses = current_episode["ranked_candidates"][:budget]
    pixel.require(len(hypotheses) == budget, "HYPOTHESIS_BUDGET")
    scan_id = str(cohort["candidate"]["rescan_id"])
    source_row = cohort["source_manifest"][f"{scan_id}/sequence.zip"]
    archive_path = Path(cohort["artifact_root"]) / source_row["path"]
    pixel.require(archive_path.stat().st_size == int(source_row["bytes"]), "ZIP_BYTES")
    pixel.require(pixel.sha256(archive_path) == source_row["sha256"], "ZIP_HASH")
    action_rows = list(motion["ranked_actions"])
    action_frames = [int(row["frame"]) for row in action_rows]

    images: dict[str, Image.Image] = {}
    image_hashes: dict[int, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        info_payload = archive.read("_info.txt")
        info = pixel.parse_info(info_payload.decode("utf-8"))
        current_image, current_hash = _image(archive, current_frame)
        current_depth = pixel.decode_depth(archive, current_frame)
        current_pose = pixel.read_pose(archive, current_frame)
        for frame in action_frames:
            images[str(frame)], image_hashes[frame] = _image(archive, frame)

    import torch
    from transformers import Sam2Model, Sam2Processor

    masker_root = Path(protocol["masker"]["model_root"])
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32).eval().to("cuda:0")
    generated, sam_receipt = nids.sam_base._sam_masks(
        processor,
        model,
        current_image,
        [row["proposal"]["box_xyxy"] for row in hypotheses],
        current_image.size,
        torch,
        np,
    )
    masks = []
    points = []
    for rank, (hypothesis, mask) in enumerate(zip(hypotheses, generated, strict=True), start=1):
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        actual = hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
        pixel.require(actual == hypothesis["proposal"]["mask_sha256"], f"MASK_REPLAY:{rank}")
        masks.append(mask)
        points.append(track._lift(mask, current_depth, current_pose, info))
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    proposal_protocol = deepcopy(protocol)
    proposal_protocol["evaluation"] = {"query_images": list(images)}
    proposals, proposal_runtime = nids.tiled._tiled_proposals(proposal_protocol, images)
    support_threshold = float(protocol["selection"]["support_iou_threshold"])
    minimum_visible = int(protocol["selection"]["minimum_visible_projected_points"])
    tolerance = float(protocol["selection"]["depth_consistency_metres"])
    ranked = []
    with zipfile.ZipFile(archive_path) as archive:
        for motion_row in action_rows:
            frame = int(motion_row["frame"])
            depth = pixel.decode_depth(archive, frame)
            pose = pixel.read_pose(archive, frame)
            supports = []
            projected_prompts = []
            for hypothesis_points in points:
                prompt, geometry = transport._visible_prompt(
                    hypothesis_points, depth, pose, info, tolerance, minimum_visible
                )
                projected_prompts.append({"bbox_xyxy": prompt, **geometry})
                support = 0.0 if prompt is None else max(
                    (_bbox_iou(prompt, proposal["box_xyxy"]) for proposal in proposals[str(frame)]),
                    default=0.0,
                )
                supports.append(float(support))
            mean_support = float(np.mean(supports))
            support_range = float(max(supports) - min(supports))
            information_gain = mean_support * support_range
            ranked.append(
                {
                    **motion_row,
                    "proposal_count": len(proposals[str(frame)]),
                    "hypothesis_support_iou": supports,
                    "supported_hypotheses": int(sum(value >= support_threshold for value in supports)),
                    "mean_hypothesis_support": mean_support,
                    "hypothesis_support_range": support_range,
                    "information_gain_score": information_gain,
                    "projected_hypotheses": projected_prompts,
                }
            )
    ranked.sort(
        key=lambda row: (
            -float(row["information_gain_score"]),
            -float(row["mean_hypothesis_support"]),
            -float(row["mutual_covisibility"]),
            int(row["frame"]),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["information_gain_rank"] = index
    selected = ranked[0]
    selected_frame = int(selected["frame"])

    target_id = int(cohort["candidate"]["target_instance_id"])
    target_points = extent.ply_instance_points(
        Path(cohort["artifact_root"]) / "datasets/3rscan" / scan_id / "labels.instances.annotated.v2.ply",
        {target_id},
    )[target_id]
    with zipfile.ZipFile(archive_path) as archive:
        selected_pose = pixel.read_pose(archive, selected_frame)
        selected_depth = pixel.decode_depth(archive, selected_frame)
        pose_hash = hashlib.sha256(archive.read(f"frame-{selected_frame:06d}.pose.txt")).hexdigest()
        depth_hash = hashlib.sha256(archive.read(f"frame-{selected_frame:06d}.depth.pgm")).hexdigest()
    _, target_stats = pixel.projected_hull(
        target_points,
        selected_pose,
        info["color_intrinsic"],
        int(info["color_width"]),
        int(info["color_height"]),
    )
    truth_bbox = [float(value) for value in target_stats["bbox_xyxy"]]
    best_target_iou = max(
        (_bbox_iou(truth_bbox, proposal["box_xyxy"]) for proposal in proposals[str(selected_frame)]),
        default=0.0,
    )
    selected_visibility = pixel.frame_visibility(
        target_points,
        selected_pose,
        info,
        selected_depth,
        tolerance,
    )

    action_image_row = {
        "episode_id": "FDV_hypothesis_disagreement_action",
        "role": "query",
        "scan_id": scan_id,
        "target_instance_id": target_id,
        "target_label": str(cohort["candidate"]["target_label"]),
        "frame": selected_frame,
        "color_size": [int(info["color_width"]), int(info["color_height"])],
        "bbox_xyxy": truth_bbox,
        "zip_member": f"frame-{selected_frame:06d}.color.jpg",
    }
    development_cohort = deepcopy(cohort)
    development_cohort["authority"] = "CONSUMED_QUEUE_ROW_3_HYPOTHESIS_DISAGREEMENT_ACTION_DEVELOPMENT_COHORT"
    development_cohort["protocol_path"] = protocol_path.name
    development_cohort["protocol_sha256"] = pixel.sha256(protocol_path)
    development_cohort["implementation"] = {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))}
    development_cohort["panel"]["fixed_action"] = "HYPOTHESIS_DISAGREEMENT_INFORMATION_GAIN"
    development_cohort["panel"]["selection"] = "Truth-free projected Top-3 hypothesis support disagreement over the pose/depth-admissible action roster."
    development_cohort["images"][action_key] = action_image_row
    development_cohort["rgb_members_opened_during_freeze"] = len(action_frames) + 1
    development_cohort["model_calls_during_freeze"] = int(proposal_runtime["grounding_dino_calls"]) + 1
    pixel.atomic_write_json(cohort_output_path, development_cohort)

    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_QUEUE_ROW_3_TRUTH_FREE_HYPOTHESIS_DISAGREEMENT_ACTION_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "current_frame": current_frame,
        "candidate_budget": budget,
        "admissible_action_count": len(ranked),
        "ranked_actions": ranked,
        "selected_action": selected,
        "selected_target_evaluation_only": {
            "target_bbox_xyxy": truth_bbox,
            "best_proposal_iou": best_target_iou,
            "proposal_opportunity_at_0_8": best_target_iou >= 0.8,
            "visibility": selected_visibility,
        },
        "selected_action_target_evaluable": True,
        "development_cohort": {
            "path": cohort_output_path.name,
            "sha256": pixel.sha256(cohort_output_path),
        },
        "runtime": {
            "current_rgb_sha256": current_hash,
            "action_rgb_members_opened": len(action_frames),
            "sam_mask_calls": 1,
            "sam_masks_generated": budget,
            "sam_receipt": sam_receipt,
            "grounding_dino_calls": proposal_runtime["grounding_dino_calls"],
            "selected_rgb_sha256": image_hashes[selected_frame],
            "selected_pose_sha256": pose_hash,
            "selected_depth_sha256": depth_hash,
            "info_sha256": hashlib.sha256(info_payload).hexdigest(),
        },
        "conclusion": "L10_3RSCAN_HYPOTHESIS_DISAGREEMENT_ACTION_DEVELOPMENT_SOURCE_EVALUABLE",
        "next_action": protocol["next_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve(), args.cohort_output.resolve())


if __name__ == "__main__":
    main()
