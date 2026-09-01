#!/usr/bin/env python3
"""Replace global CLS crops with SAM-mask foreground DINOv2 patch averages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_contrastive_sam_refinement_posthoc as refine  # noqa: E402
import l10_3rscan_object_context_memory_posthoc as feature  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-mask-ffa-memory-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-mask-ffa-memory-posthoc-result-v1"


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


def _crop_bounds(image: Image.Image, box: list[float]) -> tuple[int, int, int, int]:
    left = max(0, int(math.floor(float(box[0]))))
    top = max(0, int(math.floor(float(box[1]))))
    right = min(image.width, int(math.ceil(float(box[2]))))
    bottom = min(image.height, int(math.ceil(float(box[3]))))
    pixel.require(right > left and bottom > top, "EMPTY_CROP")
    return left, top, right, bottom


def _ffa_descriptor(
    model: Any, image: Image.Image, mask: np.ndarray, box: list[float], protocol: dict[str, Any]
) -> np.ndarray:
    left, top, right, bottom = _crop_bounds(image, box)
    crop = image.crop((left, top, right, bottom))
    size = int(protocol["descriptor"]["input_size"])
    image_tensor = feature._tensor(crop, size).to("cuda:0")
    mask_crop = torch.from_numpy(
        np.ascontiguousarray(mask[top:bottom, left:right], dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0).to("cuda:0")
    resized_mask = F.interpolate(mask_crop, size=(size, size), mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
    with torch.inference_mode():
        output = model.forward_features(image_tensor)
        patches = output["x_norm_patchtokens"].float()
        side = int(round(math.sqrt(int(patches.shape[1]))))
        pixel.require(side * side == int(patches.shape[1]), "PATCH_GRID")
        weights = F.interpolate(resized_mask, size=(side, side), mode="bilinear", align_corners=False).flatten(1)
        denominator = weights.sum(dim=1, keepdim=True)
        pixel.require(float(denominator.item()) > 0.0, "EMPTY_PATCH_MASK")
        descriptor = (patches * weights.unsqueeze(-1)).sum(dim=1) / denominator
        descriptor = F.normalize(descriptor[0], dim=0)
    return descriptor.cpu().numpy()


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
    images, image_rows = _load_images(protocol, cohort)
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")

    memory_masks: dict[str, np.ndarray] = {}
    memory_mask_receipts: dict[str, Any] = {}
    for key in protocol["memory"]["target_images"] + protocol["memory"]["sibling_images"]:
        box = image_rows[key]["bbox_xyxy"]
        masks, receipt = sam_base._sam_masks(sam_processor, sam_model, images[key], [box], images[key].size, torch, np)
        pixel.require(len(masks) == 1, f"MEMORY_MASK_COUNT:{key}")
        memory_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        memory_mask_receipts[key] = receipt

    query_masks: dict[str, list[np.ndarray]] = {}
    query_mask_receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        candidates = predecessor["query_receipts"][key]["ranked_candidates"]
        boxes = [row["box_xyxy"] for row in candidates]
        masks, receipt = sam_base._sam_masks(sam_processor, sam_model, images[key], boxes, images[key].size, torch, np)
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
        key: _ffa_descriptor(descriptor_model, images[key], memory_masks[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["target_images"]
    }
    sibling_memory = {
        key: _ffa_descriptor(descriptor_model, images[key], memory_masks[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["sibling_images"]
    }

    successes = 0
    selected_ious: list[float] = []
    query_receipts: dict[str, Any] = {}
    descriptor_calls = len(target_memory) + len(sibling_memory)
    for key in protocol["evaluation"]["query_images"]:
        source_candidates = predecessor["query_receipts"][key]["ranked_candidates"]
        candidates: list[dict[str, Any]] = []
        for source, mask in zip(source_candidates, query_masks[key]):
            descriptor = _ffa_descriptor(descriptor_model, images[key], mask, source["box_xyxy"], protocol)
            descriptor_calls += 1
            target_scores = {name: float(np.dot(value, descriptor)) for name, value in target_memory.items()}
            sibling_scores = {name: float(np.dot(value, descriptor)) for name, value in sibling_memory.items()}
            target_winner = max(target_scores, key=target_scores.get)
            sibling_winner = max(sibling_scores, key=sibling_scores.get)
            candidates.append(
                {
                    "postprocess_index": source["postprocess_index"],
                    "objectness_score": source["objectness_score"],
                    "box_xyxy": source["box_xyxy"],
                    "target_memory_score": target_scores[target_winner],
                    "sibling_memory_score": sibling_scores[sibling_winner],
                    "contrastive_score": target_scores[target_winner] - sibling_scores[sibling_winner],
                    "winning_target_reference": target_winner,
                    "winning_sibling_reference": sibling_winner,
                    "target_metrics_evaluation_only": source["target_metrics_evaluation_only"],
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["contrastive_score"], -row["target_memory_score"],
                -row["objectness_score"], *row["box_xyxy"], row["postprocess_index"],
            )
        )
        selected = candidates[0]
        selected_index = next(
            index for index, source in enumerate(source_candidates)
            if int(source["postprocess_index"]) == int(selected["postprocess_index"])
            and source["box_xyxy"] == selected["box_xyxy"]
        )
        selected_mask = query_masks[key][selected_index]
        refined_box = refine._tight_bbox(selected_mask)
        evaluation = base._bbox_metrics(refined_box, image_rows[key]["bbox_xyxy"])
        selected_iou = float(evaluation["iou"])
        selected_ious.append(selected_iou)
        successes += int(selected_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key] = {
            "query_truth_used_for_selection": False,
            "selected": selected,
            "top1_contrastive_margin": float(selected["contrastive_score"] - candidates[1]["contrastive_score"]) if len(candidates) > 1 else None,
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "ranked_candidates": candidates,
            "query_sam_receipt": query_mask_receipts[key],
        }

    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D13_MASK_FOREGROUND_FFA_MEMORY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_MASK_FFA_MEMORY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_MASK_FFA_MEMORY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(selected_ious),
            "mean_refined_iou": float(np.mean(selected_ious)),
        },
        "query_receipts": query_receipts,
        "memory_receipts": {
            "target": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in target_memory.items()},
            "sibling": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in sibling_memory.items()},
            "sam": memory_mask_receipts,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "sam_model_type": sam_model_type,
            "sam2_image_calls": len(memory_masks) + len(query_masks),
            "dinov2_ffa_calls": descriptor_calls,
            "grounding_dino_calls": 0,
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
