#!/usr/bin/env python3
"""Score every proposal against every reference patch set independently."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_foundpose_layer18_nids_posthoc as foundpose  # noqa: E402
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-reference-decoupled-patch-memory-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-reference-decoupled-patch-memory-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = nids.pixel.load_json(protocol_path)
    nids.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    nids.pixel.require(nids.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        nids.pixel.require(nids.pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    inputs = {}
    for key in ("source_protocol", "cohort", "intermediate"):
        row = protocol[key]
        path = HERE / row["path"]
        nids.pixel.require(nids.pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
        inputs[key] = nids.pixel.load_json(path)
        nids.pixel.require(inputs[key]["schema"] == row["required_schema"], f"{key.upper()}_SCHEMA")
    source_protocol = inputs["source_protocol"]
    cohort = inputs["cohort"]
    intermediate = inputs["intermediate"]
    for dependency in source_protocol["dependencies"]:
        nids.pixel.require(
            nids.pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"SOURCE_DEPENDENCY_HASH:{dependency['path']}",
        )
    for section in ("descriptor", "masker"):
        row = source_protocol[section]
        nids.pixel.require(nids.pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"], f"MODEL_HASH:{section}")

    images, image_rows = nids.ffa._load_images(source_protocol, cohort)
    reference_keys = list(protocol["evaluation"]["reference_keys"])
    query_keys = list(protocol["evaluation"]["query_keys"])
    query_candidates = {
        key: list(intermediate["query_receipts"][key]["ranked_candidates"])
        for key in query_keys
    }

    import torch
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / source_protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    reference_masks = {}
    sam_receipts = {"references": {}, "queries": {}}
    for key in reference_keys:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            images[key],
            [image_rows[key]["bbox_xyxy"]],
            images[key].size,
            torch,
            np,
        )
        reference_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        sam_receipts["references"][key] = receipt
    query_masks = {}
    for key in query_keys:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            images[key],
            [row["box_xyxy"] for row in query_candidates[key]],
            images[key].size,
            torch,
            np,
        )
        query_masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in masks]
        sam_receipts["queries"][key] = receipt
        for row, mask in zip(query_candidates[key], query_masks[key], strict=True):
            actual = hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest()
            nids.pixel.require(actual == row["mask_sha256"], f"MASK_REPLAY_MISMATCH:{key}:{row['postprocess_index']}")
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    from romatch.models.transformer import vit_large

    weights = torch.load(ROOT / source_protocol["descriptor"]["model_path"], map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(source_protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to("cuda:0")
    references = {
        key: foundpose._representation(
            model,
            images[key],
            reference_masks[key],
            image_rows[key]["bbox_xyxy"],
            source_protocol,
        )[1]
        for key in reference_keys
    }
    reference_receipts = {
        key: {
            "patch_count": int(value.shape[0]),
            "patch_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
        for key, value in references.items()
    }

    episodes = []
    for key in query_keys:
        scored = []
        for source, mask in zip(query_candidates[key], query_masks[key], strict=True):
            _, query_patches = foundpose._representation(
                model, images[key], mask, source["box_xyxy"], source_protocol
            )
            per_reference = {
                reference_key: nids._appearance_score(query_patches, reference_patches)
                for reference_key, reference_patches in references.items()
            }
            winning_reference = sorted(per_reference, key=lambda name: (-per_reference[name], name))[0]
            scored.append(
                {
                    "candidate": source,
                    "per_reference_patch_scores": per_reference,
                    "winning_patch_reference": winning_reference,
                    "reference_decoupled_patch_score": float(per_reference[winning_reference]),
                    "query_patch_count": int(query_patches.shape[0]),
                    "query_patch_sha256": hashlib.sha256(query_patches.numpy().tobytes()).hexdigest(),
                }
            )
        decoupled_order = sorted(
            range(len(scored)),
            key=lambda index: (
                -scored[index]["reference_decoupled_patch_score"],
                -float(scored[index]["candidate"]["objectness_score"]),
                index,
            ),
        )
        truth_ious = np.asarray(
            [float(row["candidate"]["target_metrics_evaluation_only"]["iou"]) for row in scored]
        )
        correct_index = int(np.argmax(truth_ious))
        conditional_order = sorted(
            range(len(scored)),
            key=lambda index: (-float(scored[index]["candidate"]["layer18_local_appearance_score"]), index),
        )
        conditional_rank = conditional_order.index(correct_index) + 1
        decoupled_rank = decoupled_order.index(correct_index) + 1
        episodes.append(
            {
                "query_key": key,
                "candidate_count": len(scored),
                "reachable_correct_candidate_index": correct_index,
                "reachable_correct_iou_evaluation_only": float(truth_ious[correct_index]),
                "conditional_single_reference_patch_rank": conditional_rank,
                "reference_decoupled_patch_rank": decoupled_rank,
                "rank_improvement": conditional_rank - decoupled_rank,
                "reference_decoupled_recall_at_3": decoupled_rank <= 3,
                "reference_decoupled_top1_iou_evaluation_only": float(truth_ious[decoupled_order[0]]),
                "ranked_candidates": [scored[index] for index in decoupled_order],
            }
        )
    del model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = (
        all(row["reference_decoupled_recall_at_3"] for row in episodes)
        and sum(row["rank_improvement"] > 0 for row in episodes) >= int(protocol["gate"]["minimum_improved_queries"])
        and min(row["rank_improvement"] for row in episodes) >= 0
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_ARV_REFERENCE_DECOUPLED_PATCH_MEMORY_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": nids.pixel.sha256(Path(__file__))},
        "source_protocol": protocol["source_protocol"],
        "cohort": protocol["cohort"],
        "intermediate": protocol["intermediate"],
        "mechanism": protocol["mechanism"],
        "reference_receipts": reference_receipts,
        "sam_receipts": sam_receipts,
        "episodes": episodes,
        "metrics": {
            "query_count": len(episodes),
            "conditional_recall_at_3": sum(row["conditional_single_reference_patch_rank"] <= 3 for row in episodes),
            "reference_decoupled_recall_at_3": sum(row["reference_decoupled_recall_at_3"] for row in episodes),
            "improved_queries": sum(row["rank_improvement"] > 0 for row in episodes),
            "regressed_queries": sum(row["rank_improvement"] < 0 for row in episodes),
            "mean_rank_improvement": float(np.mean([row["rank_improvement"] for row in episodes])),
            "minimum_top1_iou": min(row["reference_decoupled_top1_iou_evaluation_only"] for row in episodes),
            "mean_top1_iou": float(np.mean([row["reference_decoupled_top1_iou_evaluation_only"] for row in episodes])),
        },
        "gate": {**protocol["gate"], "met": gate_met},
        "runtime": {
            "rgb_members_opened": len(images),
            "sam_mask_calls": len(reference_keys) + len(query_keys),
            "representation_calls": len(reference_keys) + sum(len(rows) for rows in query_candidates.values()),
            "model_training_steps": 0,
        },
        "conclusion": (
            "L10_3RSCAN_REFERENCE_DECOUPLED_PATCH_MEMORY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_REFERENCE_DECOUPLED_PATCH_MEMORY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": protocol["next_action"] if gate_met else protocol["fallback_action"],
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
