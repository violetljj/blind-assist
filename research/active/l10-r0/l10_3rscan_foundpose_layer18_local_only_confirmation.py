#!/usr/bin/env python3
"""Confirm frozen layer-18 local-only ranking on one fresh 3RScan family."""

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
import l10_3rscan_foundpose_layer18_nids_posthoc as intermediate  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-local-only-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-foundpose-layer18-local-only-confirmation-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = intermediate.nids.pixel.load_json(protocol_path)
    intermediate.nids.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    intermediate.nids.pixel.require(
        intermediate.nids.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for key in ("intermediate_protocol", "development_predecessor"):
        row = protocol[key]
        intermediate.nids.pixel.require(
            intermediate.nids.pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = intermediate.nids.pixel.load_json(
        HERE / protocol["development_predecessor"]["path"]
    )
    intermediate.nids.pixel.require(
        predecessor["conclusion"] == protocol["development_predecessor"]["required_conclusion"],
        "DEVELOPMENT_PREDECESSOR_CONCLUSION",
    )

    base_protocol_path = HERE / protocol["intermediate_protocol"]["path"]
    intermediate_path = output_path.with_name(output_path.stem + ".intermediate.tmp.json")
    intermediate_result: dict[str, Any]
    try:
        intermediate.run(base_protocol_path, intermediate_path)
        intermediate_result = intermediate.nids.pixel.load_json(intermediate_path)
    finally:
        intermediate_path.unlink(missing_ok=True)

    base_protocol = intermediate.nids.pixel.load_json(base_protocol_path)
    cohort_path = HERE / base_protocol["source"]["cohort_path"]
    cohort = intermediate.nids.pixel.load_json(cohort_path)
    images, image_rows = intermediate.nids.ffa._load_images(base_protocol, cohort)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / base_protocol["masker"]["model_root"]).resolve()
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
            intermediate_result["query_receipts"][key]["ranked_candidates"],
            key=lambda row: (
                -row["layer18_local_appearance_score"],
                -row["target_memory_score"],
                -row["objectness_score"],
                *row["box_xyxy"],
                row["postprocess_index"],
            ),
        )
        selected = candidates[0]
        masks, sam_receipt = intermediate.nids.sam_base._sam_masks(
            processor,
            model,
            images[key],
            [selected["box_xyxy"]],
            images[key].size,
            torch,
            np,
        )
        intermediate.nids.pixel.require(len(masks) == 1, f"MASK_COUNT:{key}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = intermediate.nids.refine._tight_bbox(mask)
        evaluation = intermediate.nids.base._bbox_metrics(
            refined_box, image_rows[key]["bbox_xyxy"]
        )
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
            "ranked_candidates": candidates,
        }
    runtime = {
        "device": torch.cuda.get_device_name(0),
        "intermediate": intermediate_result["runtime"],
        "confirmation_sam2_calls": len(receipts),
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_refined_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_PRE_RGB_PRE_MODEL_SCAN_FAMILY_LAYER18_LOCAL_ONLY_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": intermediate.nids.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": intermediate.nids.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_FOUNDPOSE_LAYER18_LOCAL_ONLY_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_FOUNDPOSE_LAYER18_LOCAL_ONLY_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(receipts),
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "query_receipts": receipts,
        "intermediate_receipt": {
            "protocol_path": base_protocol_path.name,
            "protocol_sha256": intermediate.nids.pixel.sha256(base_protocol_path),
            "conclusion": intermediate_result["conclusion"],
            "metrics": intermediate_result["metrics"],
            "memory_receipts": intermediate_result["memory_receipts"],
        },
        "runtime": runtime,
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    intermediate.nids.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
