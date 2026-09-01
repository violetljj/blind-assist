#!/usr/bin/env python3
"""Disambiguate class-agnostic proposals with target-minus-sibling memory."""

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
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-contrastive-set-memory-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-contrastive-set-memory-posthoc-result-v1"


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
    descriptor_path = ROOT / protocol["descriptor"]["model_path"]
    pixel.require(pixel.sha256(descriptor_path) == protocol["descriptor"]["model_sha256"], "DESCRIPTOR_HASH")

    cohort = pixel.load_json(cohort_path)
    images, image_rows = _load_images(protocol, cohort)
    from romatch.models.transformer import vit_large

    weights = torch.load(descriptor_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]), patch_size=14,
        init_values=1.0, ffn_layer="mlp", block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to("cuda:0")

    target_memory = {
        key: base._descriptor(model, images[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["target_images"]
    }
    sibling_memory = {
        key: base._descriptor(model, images[key], image_rows[key]["bbox_xyxy"], protocol)
        for key in protocol["memory"]["sibling_images"]
    }

    query_receipts: dict[str, Any] = {}
    successes = 0
    margins: list[float] = []
    selected_ious: list[float] = []
    for key in protocol["evaluation"]["query_images"]:
        candidates: list[dict[str, Any]] = []
        for source in predecessor["query_receipts"][key]["ranked_candidates"]:
            descriptor = base._descriptor(model, images[key], source["box_xyxy"], protocol)
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
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["contrastive_score"], -row["target_memory_score"],
                -row["objectness_score"], *row["box_xyxy"], row["postprocess_index"],
            )
        )
        selected = candidates[0]
        selected_iou = float(selected["target_metrics_evaluation_only"]["iou"])
        selected_ious.append(selected_iou)
        successes += int(selected_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        margin = float(selected["contrastive_score"] - candidates[1]["contrastive_score"]) if len(candidates) > 1 else float("inf")
        margins.append(margin)
        query_receipts[key] = {
            "selection_authority": "MAX_TARGET_MINUS_SIBLING_COSINE_WITHOUT_QUERY_TRUTH",
            "query_truth_used_for_selection": False,
            "selected": selected,
            "top1_contrastive_margin": margin,
            "ranked_candidates": candidates,
            "predecessor_proposal_count": len(candidates),
        }

    del model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_top1_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_C16_TARGET_MINUS_SIBLING_MEMORY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CONTRASTIVE_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_CONTRASTIVE_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "top1_target_queries": successes,
            "minimum_selected_iou": min(selected_ious),
            "mean_selected_iou": float(np.mean(selected_ious)),
            "minimum_top1_contrastive_margin": min(margins),
        },
        "query_receipts": query_receipts,
        "memory_receipts": {
            "target": {
                key: {"descriptor_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "bbox_xyxy": image_rows[key]["bbox_xyxy"]}
                for key, value in target_memory.items()
            },
            "sibling": {
                key: {"descriptor_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "bbox_xyxy": image_rows[key]["bbox_xyxy"]}
                for key, value in sibling_memory.items()
            },
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
