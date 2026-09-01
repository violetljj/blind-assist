#!/usr/bin/env python3
"""Confirm full-frame proposal, contrastive memory and SAM2 on a new family."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_contrastive_sam_refinement_posthoc as refine  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-objectness-pipeline-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-objectness-pipeline-confirmation-result-v1"


def _load_images(
    protocol: dict[str, Any], cohort: dict[str, Any]
) -> tuple[dict[str, Image.Image], dict[str, dict[str, Any]]]:
    wanted = set(
        protocol["memory"]["target_images"]
        + protocol["memory"]["sibling_images"]
        + protocol["evaluation"]["query_images"]
    )
    rows = {key: value for key, value in cohort["images"].items() if key in wanted}
    pixel.require(set(rows) == wanted, "IMAGE_KEYS")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    for key, row in rows.items():
        archive_path = artifact_root / cohort["source_manifest"][f"{row['scan_id']}/sequence.zip"]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        rows[key] = {**row, "image_sha256": hashlib.sha256(payload).hexdigest()}
    return images, rows


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
    images, image_rows = _load_images(protocol, cohort)
    proposals, proposal_runtime = base._objectness_proposals(protocol, images)

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
        key: base._descriptor(descriptor_model, images[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["target_images"]
    }
    sibling_memory = {
        key: base._descriptor(descriptor_model, images[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["sibling_images"]
    }

    query_receipts: dict[str, Any] = {}
    opportunities = 0
    selected_boxes: dict[str, list[float]] = {}
    for key in protocol["evaluation"]["query_images"]:
        target_box = image_rows[key]["bbox_xyxy"]
        candidates: list[dict[str, Any]] = []
        for source in proposals[key]:
            descriptor = base._descriptor(descriptor_model, images[key], source["box_xyxy"], protocol)
            target_scores = {name: float(np.dot(value, descriptor)) for name, value in target_memory.items()}
            sibling_scores = {name: float(np.dot(value, descriptor)) for name, value in sibling_memory.items()}
            target_winner = max(target_scores, key=target_scores.get)
            sibling_winner = max(sibling_scores, key=sibling_scores.get)
            candidates.append(
                {
                    **source,
                    "target_memory_score": target_scores[target_winner],
                    "sibling_memory_score": sibling_scores[sibling_winner],
                    "contrastive_score": target_scores[target_winner] - sibling_scores[sibling_winner],
                    "winning_target_reference": target_winner,
                    "winning_sibling_reference": sibling_winner,
                    "target_metrics_evaluation_only": base._bbox_metrics(source["box_xyxy"], target_box),
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["contrastive_score"], -row["target_memory_score"],
                -row["objectness_score"], *row["box_xyxy"], row["postprocess_index"],
            )
        )
        pixel.require(bool(candidates), f"NO_OBJECTNESS_PROPOSAL:{key}")
        best_reachable_iou = max(float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates)
        opportunity = best_reachable_iou >= float(protocol["decision_gate"]["minimum_iou"])
        opportunities += int(opportunity)
        selected = candidates[0]
        selected_boxes[key] = selected["box_xyxy"]
        query_receipts[key] = {
            "query_truth_used_for_selection": False,
            "target_bbox_xyxy_evaluation_only": target_box,
            "proposal_count": len(candidates),
            "target_opportunity_at_iou_gate": opportunity,
            "best_reachable_target_iou_evaluation_only": best_reachable_iou,
            "selected_before_sam": selected,
            "top1_contrastive_margin": float(selected["contrastive_score"] - candidates[1]["contrastive_score"]) if len(candidates) > 1 else None,
            "ranked_candidates": candidates,
        }
    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    successes = 0
    refined_ious: list[float] = []
    for key in protocol["evaluation"]["query_images"]:
        masks, sam_receipt = sam_base._sam_masks(
            sam_processor, sam_model, images[key], [selected_boxes[key]], images[key].size, torch, np
        )
        pixel.require(len(masks) == 1, f"MASK_COUNT:{key}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = refine._tight_bbox(mask)
        evaluation = base._bbox_metrics(refined_box, image_rows[key]["bbox_xyxy"])
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key]["sam_refinement"] = {
            "selection_changed": False,
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "mask_pixels": int(mask.sum()),
            "sam_receipt": sam_receipt,
        }
    runtime = {
        "device": torch.cuda.get_device_name(0),
        "proposal": proposal_runtime,
        "sam_model_type": type(sam_model).__name__,
        "grounding_dino_calls": len(protocol["evaluation"]["query_images"]),
        "sam2_calls": len(protocol["evaluation"]["query_images"]),
    }
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    gate = protocol["decision_gate"]
    gate_met = opportunities == int(gate["required_opportunity_queries"]) and successes == int(gate["required_refined_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_PRE_MODEL_NEW_FAMILY_SAME_PROVIDER_OBJECTNESS_PIPELINE_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_OBJECTNESS_PIPELINE_NEW_FAMILY_CONFIRMATION_GATE_MET"
            if gate_met else "L10_3RSCAN_OBJECTNESS_PIPELINE_NEW_FAMILY_CONFIRMATION_GATE_NOT_MET"
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
            "sibling": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in sibling_memory.items()},
        },
        "runtime": runtime,
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
