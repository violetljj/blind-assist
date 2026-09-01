#!/usr/bin/env python3
"""Ablate final-layer FFA fusion from the consumed FoundPose layer-18 roster."""

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
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-local-only-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-local-only-posthoc-result-v1"


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
    predecessor_path = HERE / protocol["predecessor"]["path"]
    cohort_path = HERE / protocol["source"]["cohort_path"]
    nids.pixel.require(
        nids.pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"],
        "PREDECESSOR_HASH",
    )
    nids.pixel.require(
        nids.pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"],
        "COHORT_HASH",
    )
    predecessor = nids.pixel.load_json(predecessor_path)
    nids.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    nids.pixel.require(
        nids.pixel.sha256(ROOT / protocol["masker"]["model_path"])
        == protocol["masker"]["model_sha256"],
        "MASKER_HASH",
    )

    cohort = nids.pixel.load_json(cohort_path)
    images, image_rows = nids.ffa._load_images(protocol, cohort)
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    receipts: dict[str, Any] = {}
    refined_ious: list[float] = []
    successes = 0
    minimum = float(protocol["decision_gate"]["minimum_iou"])
    for key in protocol["evaluation"]["query_images"]:
        candidates = sorted(
            predecessor["query_receipts"][key]["ranked_candidates"],
            key=lambda row: (
                -row["layer18_local_appearance_score"],
                -row["target_memory_score"],
                -row["objectness_score"],
                *row["box_xyxy"],
                row["postprocess_index"],
            ),
        )
        selected = candidates[0]
        masks, sam_receipt = nids.sam_base._sam_masks(
            processor,
            model,
            images[key],
            [selected["box_xyxy"]],
            images[key].size,
            torch,
            np,
        )
        nids.pixel.require(len(masks) == 1, f"MASK_COUNT:{key}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = nids.refine._tight_bbox(mask)
        evaluation = nids.base._bbox_metrics(refined_box, image_rows[key]["bbox_xyxy"])
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= minimum)
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "selection_score": "LAYER18_QUERY_DIRECTED_LOCAL_APPEARANCE_ONLY",
            "selected": selected,
            "top1_local_appearance_margin": float(
                selected["layer18_local_appearance_score"]
                - candidates[1]["layer18_local_appearance_score"]
            ),
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(
                mask.astype(np.uint8).tobytes(order="C")
            ).hexdigest(),
            "sam_receipt": sam_receipt,
        }
    runtime = {
        "device": torch.cuda.get_device_name(0),
        "sam_model_type": type(model).__name__,
        "sam2_calls": len(receipts),
        "grounding_dino_calls": 0,
        "dinov2_calls": 0,
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_refined_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_FOUNDPOSE_LAYER18_LOCAL_ONLY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": nids.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_FOUNDPOSE_LAYER18_LOCAL_ONLY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_FOUNDPOSE_LAYER18_LOCAL_ONLY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(receipts),
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "query_receipts": receipts,
        "runtime": runtime,
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
