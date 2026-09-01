#!/usr/bin/env python3
"""Transport a bounded current hypothesis set into an action view as SAM prompts."""

from __future__ import annotations

import argparse
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
import l10_3rscan_contrastive_sam_refinement_posthoc as refine  # noqa: E402
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402
import l10_3rscan_query_mask_3d_track as track  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-hypothesis-3d-prompt-transport-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-hypothesis-3d-prompt-transport-result-v1"


def _load_image(cohort: dict[str, Any], key: str) -> tuple[Image.Image, str]:
    row = cohort["images"][key]
    archive_row = cohort["source_manifest"][f"{row['scan_id']}/sequence.zip"]
    archive_path = Path(cohort["artifact_root"]) / archive_row["path"]
    pixel.require(archive_path.stat().st_size == int(archive_row["bytes"]), f"ZIP_BYTES:{key}")
    pixel.require(pixel.sha256(archive_path) == archive_row["sha256"], f"ZIP_HASH:{key}")
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(row["zip_member"])
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
    return image, hashlib.sha256(payload).hexdigest()


def _visible_prompt(
    points_scan: np.ndarray,
    target_depth: np.ndarray,
    target_pose: np.ndarray,
    info: dict[str, Any],
    tolerance: float,
    minimum_visible: int,
) -> tuple[list[float] | None, dict[str, int]]:
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
        return None, {"projected_inside_points": 0, "depth_visible_points": 0}
    dx = np.rint(depth_pixels[indices, 0]).astype(np.int64).clip(0, int(info["depth_width"]) - 1)
    dy = np.rint(depth_pixels[indices, 1]).astype(np.int64).clip(0, int(info["depth_height"]) - 1)
    observed = target_depth[dy, dx].astype(np.float64) / 1000.0
    visible = (observed > 0.0) & (np.abs(observed - camera[indices, 2]) <= tolerance)
    visible_indices = indices[visible]
    if len(visible_indices) < minimum_visible:
        return None, {
            "projected_inside_points": int(len(indices)),
            "depth_visible_points": int(len(visible_indices)),
        }
    points = color_pixels[visible_indices]
    prompt = [
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0]) + 1.0),
        float(np.max(points[:, 1]) + 1.0),
    ]
    return prompt, {
        "projected_inside_points": int(len(indices)),
        "depth_visible_points": int(len(visible_indices)),
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    cohort_row = protocol["cohort"]
    cohort_path = HERE / cohort_row["path"]
    pixel.require(pixel.sha256(cohort_path) == cohort_row["sha256"], "COHORT_HASH")
    cohort = pixel.load_json(cohort_path)
    predecessor_row = protocol["predecessor"]
    predecessor_path = HERE / predecessor_row["path"]
    pixel.require(pixel.sha256(predecessor_path) == predecessor_row["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == predecessor_row["required_conclusion"], "PREDECESSOR_CONCLUSION")
    masker = protocol["masker"]
    pixel.require(pixel.sha256(Path(masker["model_path"])) == masker["model_sha256"], "MASKER_HASH")

    current_key = str(protocol["evaluation"]["current_query_key"])
    action_key = str(protocol["evaluation"]["action_query_key"])
    current_frame = int(cohort["images"][current_key]["frame"])
    action_frame = int(cohort["images"][action_key]["frame"])
    current_episode = next(row for row in predecessor["episodes"] if int(row["frame"]) == current_frame)
    action_episode = next(row for row in predecessor["episodes"] if int(row["frame"]) == action_frame)
    budget = int(protocol["transport"]["candidate_budget"])
    hypotheses = current_episode["ranked_candidates"][:budget]
    pixel.require(len(hypotheses) == budget, "HYPOTHESIS_BUDGET")

    current_image, current_image_hash = _load_image(cohort, current_key)
    action_image, action_image_hash = _load_image(cohort, action_key)
    scan_id = str(cohort["candidate"]["rescan_id"])
    zip_path = Path(cohort["artifact_root"]) / cohort["source_manifest"][f"{scan_id}/sequence.zip"]["path"]
    with zipfile.ZipFile(zip_path) as archive:
        info_payload = archive.read("_info.txt")
        info = pixel.parse_info(info_payload.decode("utf-8"))
        current_pose = pixel.read_pose(archive, current_frame)
        action_pose = pixel.read_pose(archive, action_frame)
        current_depth = pixel.decode_depth(archive, current_frame)
        action_depth = pixel.decode_depth(archive, action_frame)

    import torch
    from transformers import Sam2Model, Sam2Processor

    model_root = Path(masker["model_root"])
    processor = Sam2Processor.from_pretrained(model_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        model_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    current_masks, current_sam_receipt = nids.sam_base._sam_masks(
        processor,
        model,
        current_image,
        [row["proposal"]["box_xyxy"] for row in hypotheses],
        current_image.size,
        torch,
        np,
    )
    prompts: list[list[float]] = []
    transport_receipts = []
    for rank, (hypothesis, mask) in enumerate(zip(hypotheses, current_masks, strict=True), start=1):
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        mask_hash = hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
        pixel.require(mask_hash == hypothesis["proposal"]["mask_sha256"], f"CURRENT_MASK_REPLAY:{rank}")
        points = track._lift(mask, current_depth, current_pose, info)
        prompt, geometry = _visible_prompt(
            points,
            action_depth,
            action_pose,
            info,
            float(protocol["transport"]["depth_consistency_metres"]),
            int(protocol["transport"]["minimum_visible_points"]),
        )
        pixel.require(prompt is not None, f"TRANSPORT_NOT_VISIBLE:{rank}")
        prompts.append(prompt)
        transport_receipts.append(
            {
                "hypothesis_rank": rank,
                "source_candidate_index": int(hypothesis["candidate_index"]),
                "source_box_xyxy": hypothesis["proposal"]["box_xyxy"],
                "source_mask_sha256": mask_hash,
                "lifted_points": int(len(points)),
                "transported_prompt_xyxy": prompt,
                **geometry,
            }
        )
    action_masks, action_sam_receipt = nids.sam_base._sam_masks(
        processor, model, action_image, prompts, action_image.size, torch, np
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    candidates = []
    for receipt, mask in zip(transport_receipts, action_masks, strict=True):
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        candidates.append(
            {
                **receipt,
                "action_mask_bbox_xyxy": refine._tight_bbox(mask),
                "action_mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
                "action_mask_pixels": int(mask.sum()),
            }
        )

    truth = cohort["images"][action_key]["bbox_xyxy"]
    for candidate in candidates:
        candidate["prompt_target_metrics_evaluation_only"] = nids.base._bbox_metrics(
            candidate["transported_prompt_xyxy"], truth
        )
        candidate["action_mask_bbox_target_metrics_evaluation_only"] = nids.base._bbox_metrics(
            candidate["action_mask_bbox_xyxy"], truth
        )
    threshold = float(protocol["gate"]["minimum_iou"])
    transported_best = max(
        float(row["action_mask_bbox_target_metrics_evaluation_only"]["iou"])
        for row in candidates
    )
    baseline_best = float(action_episode["track_top3_best_iou_evaluation_only"])
    gate_met = transported_best >= threshold
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_QUEUE_ROW_3_HYPOTHESIS_3D_PROMPT_TRANSPORT_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "cohort": cohort_row,
        "predecessor": predecessor_row,
        "current_frame": current_frame,
        "action_frame": action_frame,
        "candidate_budget": budget,
        "candidates": candidates,
        "metrics": {
            "detector_track_top3_best_iou": baseline_best,
            "transport_sam_top3_best_iou": transported_best,
            "absolute_iou_gain": transported_best - baseline_best,
            "transported_candidates_at_or_above_gate": sum(
                float(row["action_mask_bbox_target_metrics_evaluation_only"]["iou"])
                >= threshold
                for row in candidates
            ),
        },
        "receipts": {
            "current_image_sha256": current_image_hash,
            "action_image_sha256": action_image_hash,
            "info_sha256": hashlib.sha256(info_payload).hexdigest(),
            "current_sam": current_sam_receipt,
            "action_sam": action_sam_receipt,
        },
        "gate": {**protocol["gate"], "met": gate_met},
        "runtime": {
            "rgb_members_opened": 2,
            "sam_mask_calls": 2,
            "sam_masks_generated": budget * 2,
            "grounding_dino_calls": 0,
            "appearance_model_calls": 0,
            "model_training_steps": 0,
        },
        "literature_motivation": protocol["literature_motivation"],
        "conclusion": (
            "L10_3RSCAN_HYPOTHESIS_3D_PROMPT_TRANSPORT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_HYPOTHESIS_3D_PROMPT_TRANSPORT_DEVELOPMENT_GATE_NOT_MET"
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
