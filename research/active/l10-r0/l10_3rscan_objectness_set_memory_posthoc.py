#!/usr/bin/env python3
"""Rank class-agnostic GroundingDINO proposals with frozen set memory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_object_context_memory_posthoc as feature  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as ground  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-objectness-set-memory-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-objectness-set-memory-posthoc-result-v1"


def _load_images(
    protocol: dict[str, Any], cohort: dict[str, Any]
) -> tuple[dict[str, Image.Image], dict[str, dict[str, Any]]]:
    wanted = set(protocol["memory"]["reference_images"] + protocol["evaluation"]["query_images"])
    rows = {key: value for key, value in cohort["images"].items() if key in wanted}
    pixel.require(set(rows) == wanted, "IMAGE_KEYS")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    for key, row in rows.items():
        manifest_key = f"{row['scan_id']}/sequence.zip"
        archive_path = artifact_root / cohort["source_manifest"][manifest_key]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        pixel.require(list(images[key].size) == row["color_size"], f"IMAGE_SIZE:{key}")
        rows[key] = {**row, "image_sha256": hashlib.sha256(payload).hexdigest()}
    return images, rows


def _objectness_proposals(
    protocol: dict[str, Any], images: dict[str, Image.Image]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proposal = protocol["proposal"]
    model_root = (ROOT / proposal["model_root"]).resolve()
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    results: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    for key in protocol["evaluation"]["query_images"]:
        image = images[key]
        width, height = image.size
        started = time.perf_counter()
        inputs = processor(images=image, text=[proposal["prompt"]], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(proposal["box_threshold"]),
            text_threshold=float(proposal["text_threshold"]),
            target_sizes=[(height, width)],
        )[0]
        scores = processed["scores"].detach().cpu().tolist()
        boxes = processed["boxes"].detach().cpu().tolist()
        candidates: list[dict[str, Any]] = []
        for index, (score, raw_box) in enumerate(zip(scores, boxes)):
            box = ground._clamp_box(raw_box, image.size)
            if box is None or not math.isfinite(float(score)):
                continue
            candidates.append(
                {
                    "postprocess_index": index,
                    "objectness_score": float(score),
                    "box_xyxy": box,
                }
            )
        candidates.sort(key=lambda row: (-row["objectness_score"], *row["box_xyxy"], row["postprocess_index"]))
        results[key] = candidates[: int(proposal["maximum_proposals_per_image"])]
        timings[key] = time.perf_counter() - started
    runtime = {
        "model_type": type(model).__name__,
        "device": torch.cuda.get_device_name(0),
        "seconds_by_image": timings,
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results, runtime


def _descriptor(model: Any, image: Image.Image, box: list[float], protocol: dict[str, Any]) -> np.ndarray:
    tensors = [
        feature._tensor(
            feature._crop(image, box, float(scale)), int(protocol["descriptor"]["input_size"])
        )
        for scale in protocol["descriptor"]["crop_scales"]
    ]
    with torch.inference_mode():
        tokens = model.forward_features(torch.cat(tensors, dim=0).to("cuda:0"))["x_norm_clstoken"].float()
        tokens = torch.nn.functional.normalize(tokens, dim=1)
        value = torch.nn.functional.normalize(tokens.flatten(), dim=0)
    return value.cpu().numpy()


def _bbox_metrics(proposal: list[float], target: list[float]) -> dict[str, float]:
    left, top = max(proposal[0], target[0]), max(proposal[1], target[1])
    right, bottom = min(proposal[2], target[2]), min(proposal[3], target[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    proposal_area = (proposal[2] - proposal[0]) * (proposal[3] - proposal[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    union = proposal_area + target_area - intersection
    return {
        "iou": intersection / union if union > 0.0 else 0.0,
        "target_recall": intersection / target_area if target_area > 0.0 else 0.0,
        "proposal_precision": intersection / proposal_area if proposal_area > 0.0 else 0.0,
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    pixel.require(pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    predecessor_path = HERE / protocol["predecessor"]["path"]
    pixel.require(pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    grounder_path = ROOT / protocol["proposal"]["model_path"]
    descriptor_path = ROOT / protocol["descriptor"]["model_path"]
    pixel.require(pixel.sha256(grounder_path) == protocol["proposal"]["model_sha256"], "GROUNDER_HASH")
    pixel.require(pixel.sha256(descriptor_path) == protocol["descriptor"]["model_sha256"], "DESCRIPTOR_HASH")

    cohort = pixel.load_json(cohort_path)
    images, image_rows = _load_images(protocol, cohort)
    proposals, proposal_runtime = _objectness_proposals(protocol, images)

    from romatch.models.transformer import vit_large

    weights = torch.load(descriptor_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]), patch_size=14,
        init_values=1.0, ffn_layer="mlp", block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to("cuda:0")

    references: dict[str, np.ndarray] = {}
    for key in protocol["memory"]["reference_images"]:
        references[key] = _descriptor(model, images[key], image_rows[key]["bbox_xyxy"], protocol)

    query_receipts: dict[str, Any] = {}
    successful_queries = 0
    opportunity_queries = 0
    selected_ious: list[float] = []
    for key in protocol["evaluation"]["query_images"]:
        target_box = image_rows[key]["bbox_xyxy"]
        candidate_rows: list[dict[str, Any]] = []
        for candidate in proposals[key]:
            descriptor = _descriptor(model, images[key], candidate["box_xyxy"], protocol)
            pair_scores = {name: float(np.dot(value, descriptor)) for name, value in references.items()}
            winner = max(pair_scores, key=pair_scores.get)
            candidate_rows.append(
                {
                    **candidate,
                    "memory_score": pair_scores[winner],
                    "winning_reference": winner,
                    "pair_scores": pair_scores,
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                    "target_metrics_evaluation_only": _bbox_metrics(candidate["box_xyxy"], target_box),
                }
            )
        candidate_rows.sort(
            key=lambda row: (-row["memory_score"], -row["objectness_score"], *row["box_xyxy"], row["postprocess_index"])
        )
        best_reachable_iou = max((row["target_metrics_evaluation_only"]["iou"] for row in candidate_rows), default=0.0)
        opportunity = best_reachable_iou >= float(protocol["decision_gate"]["minimum_iou"])
        opportunity_queries += int(opportunity)
        selected = candidate_rows[0] if candidate_rows else None
        selected_iou = selected["target_metrics_evaluation_only"]["iou"] if selected else 0.0
        selected_ious.append(selected_iou)
        successful_queries += int(selected_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key] = {
            "selection_authority": "MAX_FROZEN_SET_MEMORY_COSINE_WITH_OBJECTNESS_ONLY_TIEBREAK",
            "query_truth_used_for_selection": False,
            "target_bbox_xyxy_evaluation_only": target_box,
            "proposal_count": len(candidate_rows),
            "target_opportunity_at_iou_gate": opportunity,
            "best_reachable_target_iou_evaluation_only": best_reachable_iou,
            "selected": selected,
            "ranked_candidates": candidate_rows,
            "image_sha256": image_rows[key]["image_sha256"],
        }

    del model, weights
    torch.cuda.empty_cache()
    gate = protocol["decision_gate"]
    gate_met = (
        opportunity_queries == int(gate["required_opportunity_queries"])
        and successful_queries == int(gate["required_top1_queries"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_C16_CLASS_AGNOSTIC_PROPOSAL_SET_MEMORY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_OBJECTNESS_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_OBJECTNESS_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "opportunity_queries": opportunity_queries,
            "top1_target_queries": successful_queries,
            "minimum_selected_iou": min(selected_ious),
            "mean_selected_iou": float(np.mean(selected_ious)),
        },
        "proposal_runtime": proposal_runtime,
        "query_receipts": query_receipts,
        "memory_receipts": {
            key: {
                "descriptor_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "bbox_xyxy": image_rows[key]["bbox_xyxy"],
                "image_sha256": image_rows[key]["image_sha256"],
            }
            for key, value in references.items()
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
