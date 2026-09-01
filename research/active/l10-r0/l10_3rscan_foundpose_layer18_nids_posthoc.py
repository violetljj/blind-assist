#!/usr/bin/env python3
"""Use FoundPose's DINOv2 layer-18 patches in the frozen NIDS D15 chain."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-nids-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-nids-posthoc-result-v1"


def _representation(
    model: Any,
    image: Any,
    mask: np.ndarray,
    box: list[float],
    protocol: dict[str, Any],
) -> tuple[np.ndarray, torch.Tensor]:
    left, top, right, bottom = nids.ffa._crop_bounds(image, box)
    crop = image.crop((left, top, right, bottom))
    size = int(protocol["descriptor"]["input_size"])
    image_tensor = nids.feature._tensor(crop, size).to("cuda:0")
    mask_crop = torch.from_numpy(
        np.ascontiguousarray(mask[top:bottom, left:right], dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0).to("cuda:0")
    resized_mask = F.interpolate(
        mask_crop, size=(size, size), mode="bicubic", align_corners=False
    ).clamp_(0.0, 1.0)
    with torch.inference_mode():
        tokens = model.prepare_tokens_with_masks(image_tensor)
        appearance_tokens = None
        appearance_block = int(protocol["appearance"]["zero_based_block_index"])
        for index, block in enumerate(model.blocks):
            tokens = block(tokens)
            if index == appearance_block:
                appearance_tokens = model.norm(tokens)[:, 1:].float()
        nids.pixel.require(appearance_tokens is not None, "APPEARANCE_LAYER")
        final_patches = model.norm(tokens)[:, 1:].float()
        side = int(round(math.sqrt(int(final_patches.shape[1]))))
        nids.pixel.require(side * side == int(final_patches.shape[1]), "PATCH_GRID")
        weights = F.interpolate(
            resized_mask, size=(side, side), mode="bilinear", align_corners=False
        ).flatten(1)
        denominator = weights.sum(dim=1, keepdim=True)
        nids.pixel.require(float(denominator.item()) > 0.0, "EMPTY_PATCH_MASK")
        descriptor = (final_patches * weights.unsqueeze(-1)).sum(dim=1) / denominator
        descriptor = F.normalize(descriptor[0], dim=0)
        selector = weights[0] > float(protocol["appearance"]["patch_mask_threshold"])
        nids.pixel.require(bool(selector.any().item()), "EMPTY_BINARY_PATCH_MASK")
        foreground_patches = F.normalize(appearance_tokens[0, selector], dim=-1)
    return descriptor.cpu().numpy(), foreground_patches.cpu()


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = nids.pixel.load_json(protocol_path)
    nids.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    nids.pixel.require(
        nids.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        nids.pixel.require(
            nids.pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    cohort_path = HERE / protocol["source"]["cohort_path"]
    predecessor_path = HERE / protocol["predecessor"]["path"]
    nids.pixel.require(
        nids.pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"],
        "COHORT_HASH",
    )
    nids.pixel.require(
        nids.pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"],
        "PREDECESSOR_HASH",
    )
    predecessor = nids.pixel.load_json(predecessor_path)
    nids.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    for section in ("proposal", "descriptor", "masker"):
        row = protocol[section]
        nids.pixel.require(
            nids.pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"],
            f"MODEL_HASH:{section}",
        )

    cohort = nids.pixel.load_json(cohort_path)
    images, image_rows = nids.ffa._load_images(protocol, cohort)
    proposals, proposal_runtime = nids.tiled._tiled_proposals(protocol, images)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    memory_masks: dict[str, np.ndarray] = {}
    memory_mask_receipts: dict[str, Any] = {}
    for key in protocol["memory"]["target_images"]:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            images[key],
            [image_rows[key]["bbox_xyxy"]],
            images[key].size,
            torch,
            np,
        )
        memory_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        memory_mask_receipts[key] = receipt
    query_masks: dict[str, list[np.ndarray]] = {}
    query_mask_receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            images[key],
            [row["box_xyxy"] for row in proposals[key]],
            images[key].size,
            torch,
            np,
        )
        query_masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in masks]
        query_mask_receipts[key] = receipt
    sam_model_type = type(sam_model).__name__
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    from romatch.models.transformer import vit_large

    weights = torch.load(
        ROOT / protocol["descriptor"]["model_path"], map_location="cpu", weights_only=True
    )
    descriptor_model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    descriptor_model.load_state_dict(weights)
    descriptor_model = descriptor_model.to("cuda:0")
    target_memory = {
        key: _representation(
            descriptor_model,
            images[key],
            memory_masks[key],
            image_rows[key]["bbox_xyxy"],
            protocol,
        )
        for key in protocol["memory"]["target_images"]
    }

    minimum = float(protocol["decision_gate"]["minimum_iou"])
    opportunities = 0
    successes = 0
    refined_ious: list[float] = []
    receipts: dict[str, Any] = {}
    representation_calls = len(target_memory)
    empty_mask_candidates = 0
    for key in protocol["evaluation"]["query_images"]:
        truth = image_rows[key]["bbox_xyxy"]
        candidates: list[dict[str, Any]] = []
        for source, mask in zip(proposals[key], query_masks[key]):
            try:
                descriptor, query_patches = _representation(
                    descriptor_model, images[key], mask, source["box_xyxy"], protocol
                )
            except ValueError as error:
                if str(error) not in {"EMPTY_PATCH_MASK", "EMPTY_BINARY_PATCH_MASK"}:
                    raise
                empty_mask_candidates += 1
                continue
            representation_calls += 1
            per_reference = {
                name: float(np.dot(reference[0], descriptor))
                for name, reference in target_memory.items()
            }
            winning_reference = sorted(
                per_reference, key=lambda name: (-per_reference[name], name)
            )[0]
            instance_score = per_reference[winning_reference]
            local_score = nids._appearance_score(
                query_patches, target_memory[winning_reference][1]
            )
            fused_score = float((instance_score + local_score) / 2.0)
            evaluation = nids.base._bbox_metrics(source["box_xyxy"], truth)
            candidates.append(
                {
                    **source,
                    "target_memory_score": instance_score,
                    "per_reference_scores": per_reference,
                    "winning_target_reference": winning_reference,
                    "layer18_local_appearance_score": local_score,
                    "layer18_nids_fused_score": fused_score,
                    "target_metrics_evaluation_only": evaluation,
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                    "layer18_patch_sha256": hashlib.sha256(
                        query_patches.numpy().tobytes()
                    ).hexdigest(),
                    "mask_sha256": hashlib.sha256(
                        mask.astype(np.uint8).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["layer18_nids_fused_score"],
                -row["target_memory_score"],
                -row["objectness_score"],
                *row["box_xyxy"],
                row["postprocess_index"],
            )
        )
        nids.pixel.require(bool(candidates), f"NO_NONEMPTY_MASK_CANDIDATE:{key}")
        best_reachable = max(
            float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates
        )
        opportunity = best_reachable >= minimum
        opportunities += int(opportunity)
        selected = candidates[0]
        selected_index = next(
            index
            for index, source in enumerate(proposals[key])
            if int(source["postprocess_index"]) == int(selected["postprocess_index"])
            and source["box_xyxy"] == selected["box_xyxy"]
        )
        refined_box = nids.refine._tight_bbox(query_masks[key][selected_index])
        refined = nids.base._bbox_metrics(refined_box, truth)
        refined_iou = float(refined["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= minimum)
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "target_opportunity_at_iou_gate": opportunity,
            "best_reachable_target_iou_evaluation_only": best_reachable,
            "selected": selected,
            "top1_layer18_nids_fused_margin": float(
                selected["layer18_nids_fused_score"]
                - candidates[1]["layer18_nids_fused_score"]
            ),
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": refined,
            "ranked_candidates": candidates,
            "query_sam_receipt": query_mask_receipts[key],
        }

    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = (
        opportunities == int(protocol["decision_gate"]["required_opportunity_queries"])
        and successes == int(protocol["decision_gate"]["required_refined_queries"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_FOUNDPOSE_LAYER18_NIDS_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": nids.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_FOUNDPOSE_LAYER18_NIDS_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_FOUNDPOSE_LAYER18_NIDS_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(receipts),
            "opportunity_queries": opportunities,
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "query_receipts": receipts,
        "memory_receipts": {
            "target": {
                key: {
                    "final_ffa_sha256": hashlib.sha256(value[0].tobytes()).hexdigest(),
                    "layer18_patch_sha256": hashlib.sha256(
                        value[1].numpy().tobytes()
                    ).hexdigest(),
                }
                for key, value in target_memory.items()
            },
            "sam": memory_mask_receipts,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "proposal": proposal_runtime,
            "sam_model_type": sam_model_type,
            "grounding_dino_calls": proposal_runtime["grounding_dino_calls"],
            "sam2_image_calls": len(memory_masks) + len(query_masks),
            "dinov2_representation_calls": representation_calls,
            "empty_mask_candidates_skipped": empty_mask_candidates,
        },
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    nids.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
