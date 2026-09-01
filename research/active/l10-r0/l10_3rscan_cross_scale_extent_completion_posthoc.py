#!/usr/bin/env python3
"""Complete a tile-localized part with its best containing full-frame proposal."""

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
import l10_3rscan_foundpose_layer18_local_only_confirmation as confirmation  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cross-scale-extent-completion-posthoc-protocol-v1"


def _select(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    anchor = candidates[0]
    if anchor["tile_id"] == "full":
        return anchor, False
    left, top, right, bottom = (float(value) for value in anchor["box_xyxy"])
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    completions = [
        row
        for row in candidates
        if row["tile_id"] == "full"
        and row["winning_target_reference"] == anchor["winning_target_reference"]
        and float(row["box_xyxy"][0]) <= center_x <= float(row["box_xyxy"][2])
        and float(row["box_xyxy"][1]) <= center_y <= float(row["box_xyxy"][3])
    ]
    if not completions:
        return anchor, False
    completions.sort(
        key=lambda row: (
            -row["layer18_local_appearance_score"],
            -row["target_memory_score"],
            -row["objectness_score"],
            *row["box_xyxy"],
            row["postprocess_index"],
        )
    )
    return completions[0], True


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = confirmation.intermediate.nids.pixel.load_json(protocol_path)
    confirmation.intermediate.nids.pixel.require(
        protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA"
    )
    confirmation.intermediate.nids.pixel.require(
        confirmation.intermediate.nids.pixel.sha256(Path(__file__))
        == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for key in ("predecessor", "confirmation_protocol"):
        row = protocol[key]
        confirmation.intermediate.nids.pixel.require(
            confirmation.intermediate.nids.pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = confirmation.intermediate.nids.pixel.load_json(
        HERE / protocol["predecessor"]["path"]
    )
    confirmation.intermediate.nids.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    confirmation_protocol = confirmation.intermediate.nids.pixel.load_json(
        HERE / protocol["confirmation_protocol"]["path"]
    )
    base_protocol = confirmation.intermediate.nids.pixel.load_json(
        HERE / confirmation_protocol["intermediate_protocol"]["path"]
    )
    cohort = confirmation.intermediate.nids.pixel.load_json(
        HERE / base_protocol["source"]["cohort_path"]
    )
    images, image_rows = confirmation.intermediate.nids.ffa._load_images(base_protocol, cohort)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / base_protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    receipts: dict[str, Any] = {}
    ious: list[float] = []
    successes = 0
    minimum = float(protocol["decision_gate"]["minimum_iou"])
    for key in protocol["evaluation"]["query_images"]:
        candidates = predecessor["query_receipts"][key]["ranked_candidates"]
        selected, completed = _select(candidates)
        masks, sam_receipt = confirmation.intermediate.nids.sam_base._sam_masks(
            processor, model, images[key], [selected["box_xyxy"]], images[key].size, torch, np
        )
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = confirmation.intermediate.nids.refine._tight_bbox(mask)
        evaluation = confirmation.intermediate.nids.base._bbox_metrics(
            refined_box, image_rows[key]["bbox_xyxy"]
        )
        iou = float(evaluation["iou"])
        ious.append(iou)
        successes += int(iou >= minimum)
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "part_anchor": candidates[0],
            "extent_completion_applied": completed,
            "selected": selected,
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "sam_receipt": sam_receipt,
        }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_refined_queries"])
    result = {
        "schema": "blindassist-l10-3rscan-cross-scale-extent-completion-posthoc-result-v1",
        "authority": "CONSUMED_FRESH_FAMILY_CROSS_SCALE_EXTENT_COMPLETION_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": confirmation.intermediate.nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": confirmation.intermediate.nids.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(receipts),
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(ious),
            "mean_refined_iou": float(np.mean(ious)),
            "extent_completions": sum(int(row["extent_completion_applied"]) for row in receipts.values()),
        },
        "query_receipts": receipts,
        "runtime": {"device": torch.cuda.get_device_name(0), "sam2_calls": len(receipts)},
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    confirmation.intermediate.nids.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
