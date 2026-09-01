#!/usr/bin/env python3
"""Confirm frozen cross-scale target-extent completion on one fresh 3RScan family."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
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
import l10_3rscan_cross_scale_extent_completion_preservation_audit as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cross-scale-extent-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cross-scale-extent-confirmation-result-v1"


def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            -row["layer18_local_appearance_score"],
            -row["target_memory_score"],
            -row["objectness_score"],
            *row["box_xyxy"],
            row["postprocess_index"],
        ),
    )


def _select(
    candidates: list[dict[str, Any]], minimum_containment: float, minimum_retention: float
) -> tuple[dict[str, Any], dict[str, Any], bool, float]:
    ranked = _rank(candidates)
    anchor = ranked[0]
    if anchor["tile_id"] == "full":
        return anchor, anchor, False, 1.0
    eligible = [
        (row, extent._containment(anchor["box_xyxy"], row["box_xyxy"]))
        for row in ranked
        if row["tile_id"] == "full"
        and row["winning_target_reference"] == anchor["winning_target_reference"]
        and float(row["layer18_local_appearance_score"])
        >= minimum_retention * float(anchor["layer18_local_appearance_score"])
    ]
    eligible = [row for row in eligible if row[1] >= minimum_containment]
    if not eligible:
        return anchor, anchor, False, 0.0
    return anchor, eligible[0][0], True, float(eligible[0][1])


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
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            intermediate.run(base_protocol_path, intermediate_path)
        intermediate_result = intermediate.nids.pixel.load_json(intermediate_path)
    finally:
        intermediate_path.unlink(missing_ok=True)

    base_protocol = intermediate.nids.pixel.load_json(base_protocol_path)
    cohort = intermediate.nids.pixel.load_json(HERE / base_protocol["source"]["cohort_path"])
    images, image_rows = intermediate.nids.ffa._load_images(base_protocol, cohort)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / base_protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    minimum_containment = float(protocol["selection"]["minimum_anchor_containment"])
    minimum_retention = float(protocol["selection"]["minimum_local_score_retention"])
    successes = 0
    completions = 0
    refined_ious: list[float] = []
    receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        candidates = intermediate_result["query_receipts"][key]["ranked_candidates"]
        anchor, selected, applied, containment = _select(
            candidates, minimum_containment, minimum_retention
        )
        masks, sam_receipt = intermediate.nids.sam_base._sam_masks(
            processor, model, images[key], [selected["box_xyxy"]], images[key].size,
            torch, np,
        )
        intermediate.nids.pixel.require(len(masks) == 1, f"MASK_COUNT:{key}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = intermediate.nids.refine._tight_bbox(mask)
        evaluation = intermediate.nids.base._bbox_metrics(
            refined_box, image_rows[key]["bbox_xyxy"]
        )
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= minimum_iou)
        completions += int(applied)
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "anchor": anchor,
            "selected": selected,
            "extent_completion_applied": applied,
            "anchor_containment": containment,
            "local_score_retention": float(selected["layer18_local_appearance_score"])
            / float(anchor["layer18_local_appearance_score"]),
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "sam_receipt": sam_receipt,
            "ranked_candidates": _rank(candidates),
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
        "authority": "FRESH_PRE_RGB_PRE_MODEL_SCAN_FAMILY_CROSS_SCALE_EXTENT_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": intermediate.nids.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": intermediate.nids.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_CROSS_SCALE_EXTENT_FRESH_CONFIRMATION_GATE_MET"
            if gate_met else
            "L10_3RSCAN_CROSS_SCALE_EXTENT_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "selection": protocol["selection"],
        "metrics": {
            "query_count": len(receipts),
            "refined_iou_gate_queries": successes,
            "extent_completion_queries": completions,
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
