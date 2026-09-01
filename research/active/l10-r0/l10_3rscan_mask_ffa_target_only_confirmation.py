#!/usr/bin/env python3
"""Confirm target-only mask-foreground FFA localization on a frozen new family."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_contrastive_sam_refinement_posthoc as refine  # noqa: E402
import l10_3rscan_mask_ffa_memory_posthoc as ffa  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-mask-ffa-target-only-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-mask-ffa-target-only-confirmation-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    predecessor_path = HERE / protocol["predecessor"]["path"]
    pixel.require(pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    pixel.require(pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    for section in ("proposal", "descriptor", "masker"):
        row = protocol[section]
        pixel.require(pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"], f"MODEL_HASH:{section}")

    cohort = pixel.load_json(cohort_path)
    images, image_rows = ffa._load_images(protocol, cohort)
    proposals, proposal_runtime = base._objectness_proposals(protocol, images)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")

    memory_masks: dict[str, np.ndarray] = {}
    memory_mask_receipts: dict[str, Any] = {}
    for key in protocol["memory"]["target_images"]:
        masks, receipt = sam_base._sam_masks(
            sam_processor, sam_model, images[key], [image_rows[key]["bbox_xyxy"]], images[key].size, torch, np
        )
        pixel.require(len(masks) == 1, f"MEMORY_MASK_COUNT:{key}")
        memory_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        memory_mask_receipts[key] = receipt

    query_masks: dict[str, list[np.ndarray]] = {}
    query_mask_receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        boxes = [row["box_xyxy"] for row in proposals[key]]
        masks, receipt = sam_base._sam_masks(
            sam_processor, sam_model, images[key], boxes, images[key].size, torch, np
        )
        pixel.require(len(masks) == len(boxes), f"QUERY_MASK_COUNT:{key}")
        query_masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in masks]
        query_mask_receipts[key] = receipt
    sam_model_type = type(sam_model).__name__
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    from romatch.models.transformer import vit_large

    descriptor_path = ROOT / protocol["descriptor"]["model_path"]
    weights = torch.load(descriptor_path, map_location="cpu", weights_only=True)
    descriptor_model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]), patch_size=14,
        init_values=1.0, ffn_layer="mlp", block_chunks=0,
    ).eval()
    descriptor_model.load_state_dict(weights)
    descriptor_model = descriptor_model.to("cuda:0")
    target_memory = {
        key: ffa._ffa_descriptor(
            descriptor_model, images[key], memory_masks[key], image_rows[key]["bbox_xyxy"], protocol
        )
        for key in protocol["memory"]["target_images"]
    }

    opportunities = 0
    successes = 0
    refined_ious: list[float] = []
    query_receipts: dict[str, Any] = {}
    descriptor_calls = len(target_memory)
    for key in protocol["evaluation"]["query_images"]:
        target_box = image_rows[key]["bbox_xyxy"]
        candidates: list[dict[str, Any]] = []
        for source, mask in zip(proposals[key], query_masks[key]):
            descriptor = ffa._ffa_descriptor(descriptor_model, images[key], mask, source["box_xyxy"], protocol)
            descriptor_calls += 1
            scores = {name: float(np.dot(value, descriptor)) for name, value in target_memory.items()}
            winner = max(scores, key=scores.get)
            candidates.append(
                {
                    **source,
                    "target_memory_score": scores[winner],
                    "winning_target_reference": winner,
                    "target_metrics_evaluation_only": base._bbox_metrics(source["box_xyxy"], target_box),
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["target_memory_score"], -row["objectness_score"],
                *row["box_xyxy"], row["postprocess_index"],
            )
        )
        pixel.require(bool(candidates), f"NO_OBJECTNESS_PROPOSAL:{key}")
        best_reachable_iou = max(float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates)
        opportunity = best_reachable_iou >= float(protocol["decision_gate"]["minimum_iou"])
        opportunities += int(opportunity)
        selected = candidates[0]
        selected_index = next(
            index for index, source in enumerate(proposals[key])
            if int(source["postprocess_index"]) == int(selected["postprocess_index"])
            and source["box_xyxy"] == selected["box_xyxy"]
        )
        refined_box = refine._tight_bbox(query_masks[key][selected_index])
        evaluation = base._bbox_metrics(refined_box, target_box)
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key] = {
            "query_truth_used_for_selection": False,
            "selection_score": "MAX_TARGET_MEMORY_FFA_COSINE",
            "target_bbox_xyxy_evaluation_only": target_box,
            "proposal_count": len(candidates),
            "target_opportunity_at_iou_gate": opportunity,
            "best_reachable_target_iou_evaluation_only": best_reachable_iou,
            "selected": selected,
            "top1_target_memory_margin": float(selected["target_memory_score"] - candidates[1]["target_memory_score"]) if len(candidates) > 1 else None,
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "ranked_candidates": candidates,
            "query_sam_receipt": query_mask_receipts[key],
        }

    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate = protocol["decision_gate"]
    gate_met = opportunities == int(gate["required_opportunity_queries"]) and successes == int(gate["required_refined_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_PRE_MODEL_NEW_FAMILY_SAME_PROVIDER_TARGET_ONLY_FFA_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_MASK_FFA_TARGET_ONLY_NEW_FAMILY_CONFIRMATION_GATE_MET"
            if gate_met else "L10_3RSCAN_MASK_FFA_TARGET_ONLY_NEW_FAMILY_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "opportunity_queries": opportunities,
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "query_receipts": query_receipts,
        "memory_receipts": {
            "target": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in target_memory.items()},
            "sam": memory_mask_receipts,
            "sibling_appearance_used": False,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "proposal": proposal_runtime,
            "sam_model_type": sam_model_type,
            "grounding_dino_calls": len(protocol["evaluation"]["query_images"]),
            "sam2_image_calls": len(memory_masks) + len(query_masks),
            "dinov2_ffa_calls": descriptor_calls,
        },
        "literature_motivation": protocol["literature_motivation"],
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
