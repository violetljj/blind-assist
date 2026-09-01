#!/usr/bin/env python3
"""Replace max-over-memory FFA with a top-two consensus on consumed D15."""

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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-robust-ffa-consensus-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-robust-ffa-consensus-posthoc-result-v1"


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
    for section in ("descriptor", "masker"):
        row = protocol[section]
        pixel.require(pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"], f"MODEL_HASH:{section}")

    cohort = pixel.load_json(cohort_path)
    images, image_rows = ffa._load_images(protocol, cohort)
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
        source_candidates = predecessor["query_receipts"][key]["ranked_candidates"]
        masks, receipt = sam_base._sam_masks(
            sam_processor, sam_model, images[key], [row["box_xyxy"] for row in source_candidates],
            images[key].size, torch, np,
        )
        pixel.require(len(masks) == len(source_candidates), f"QUERY_MASK_COUNT:{key}")
        query_masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in masks]
        query_mask_receipts[key] = receipt
    sam_model_type = type(sam_model).__name__
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    from romatch.models.transformer import vit_large

    weights = torch.load(ROOT / protocol["descriptor"]["model_path"], map_location="cpu", weights_only=True)
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

    successes = 0
    refined_ious: list[float] = []
    query_receipts: dict[str, Any] = {}
    descriptor_calls = len(target_memory)
    for key in protocol["evaluation"]["query_images"]:
        source_candidates = predecessor["query_receipts"][key]["ranked_candidates"]
        candidates: list[dict[str, Any]] = []
        for source, mask in zip(source_candidates, query_masks[key]):
            descriptor = ffa._ffa_descriptor(descriptor_model, images[key], mask, source["box_xyxy"], protocol)
            descriptor_calls += 1
            per_reference = {
                name: float(np.dot(value, descriptor)) for name, value in target_memory.items()
            }
            ordered = sorted(per_reference.items(), key=lambda row: (-row[1], row[0]))
            support = ordered[: int(protocol["memory"]["consensus_reference_count"])]
            consensus = float(np.mean([row[1] for row in support]))
            candidates.append(
                {
                    **source,
                    "target_memory_score": consensus,
                    "per_reference_scores": per_reference,
                    "consensus_references": [row[0] for row in support],
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
        selected = candidates[0]
        selected_index = next(
            index for index, source in enumerate(source_candidates)
            if int(source["postprocess_index"]) == int(selected["postprocess_index"])
            and source["box_xyxy"] == selected["box_xyxy"]
        )
        refined_box = refine._tight_bbox(query_masks[key][selected_index])
        evaluation = base._bbox_metrics(refined_box, image_rows[key]["bbox_xyxy"])
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key] = {
            "query_truth_used_for_selection": False,
            "selection_score": "MEAN_OF_TOP_TWO_TARGET_MEMORY_FFA_COSINES",
            "predecessor_target_opportunity_at_iou_gate": predecessor["query_receipts"][key]["target_opportunity_at_iou_gate"],
            "selected": selected,
            "top1_consensus_margin": float(selected["target_memory_score"] - candidates[1]["target_memory_score"]) if len(candidates) > 1 else None,
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "ranked_candidates": candidates,
            "query_sam_receipt": query_mask_receipts[key],
        }

    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()
    required = int(protocol["decision_gate"]["required_refined_queries"])
    gate_met = successes >= required
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_ROBUST_TOP_TWO_FFA_CONSENSUS_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_ROBUST_FFA_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_ROBUST_FFA_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "predecessor_opportunity_queries": predecessor["metrics"]["opportunity_queries"],
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "query_receipts": query_receipts,
        "memory_receipts": {
            "target": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in target_memory.items()},
            "sam": memory_mask_receipts,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "sam_model_type": sam_model_type,
            "grounding_dino_calls": 0,
            "sam2_image_calls": len(memory_masks) + len(query_masks),
            "dinov2_ffa_calls": descriptor_calls,
        },
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
