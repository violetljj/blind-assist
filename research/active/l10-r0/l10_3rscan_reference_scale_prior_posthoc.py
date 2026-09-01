#!/usr/bin/env python3
"""Audit a reference-derived scale prior across three consumed 3RScan families."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_foundpose_layer18_nids_posthoc as foundpose  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-reference-scale-prior-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-reference-scale-prior-posthoc-result-v1"


def _area_fraction(box: list[float], size: tuple[int, int]) -> float:
    width, height = size
    area = max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
    return area / float(width * height)


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = foundpose.nids.pixel.load_json(protocol_path)
    foundpose.nids.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    foundpose.nids.pixel.require(
        foundpose.nids.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for family, spec in protocol["families"].items():
        result_path = HERE / spec["candidate_result"]["path"]
        base_path = HERE / spec["intermediate_protocol"]["path"]
        foundpose.nids.pixel.require(
            foundpose.nids.pixel.sha256(result_path) == spec["candidate_result"]["sha256"],
            f"RESULT_HASH:{family}",
        )
        foundpose.nids.pixel.require(
            foundpose.nids.pixel.sha256(base_path) == spec["intermediate_protocol"]["sha256"],
            f"PROTOCOL_HASH:{family}",
        )
        loaded[family] = (
            foundpose.nids.pixel.load_json(result_path),
            foundpose.nids.pixel.load_json(base_path),
            base_path,
        )

    first_protocol = next(iter(loaded.values()))[1]
    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / first_protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    penalty = float(protocol["selection"]["log_area_penalty"])
    minimum = float(protocol["decision_gate"]["minimum_iou"])
    all_ious: list[float] = []
    total_successes = 0
    family_receipts: dict[str, Any] = {}
    for family, (candidates_result, base_protocol, base_path) in loaded.items():
        cohort = foundpose.nids.pixel.load_json(HERE / base_protocol["source"]["cohort_path"])
        images, image_rows = foundpose.nids.ffa._load_images(base_protocol, cohort)
        reference_areas = sorted(
            _area_fraction(image_rows[key]["bbox_xyxy"], images[key].size)
            for key in base_protocol["memory"]["target_images"]
        )
        reference_scale = reference_areas[len(reference_areas) // 2]
        receipts: dict[str, Any] = {}
        successes = 0
        ious: list[float] = []
        for key in base_protocol["evaluation"]["query_images"]:
            candidates = candidates_result["query_receipts"][key]["ranked_candidates"]
            ranked = sorted(
                candidates,
                key=lambda row: (
                    -(
                        float(row["layer18_local_appearance_score"])
                        - penalty
                        * abs(math.log(_area_fraction(row["box_xyxy"], images[key].size) / reference_scale))
                    ),
                    -row["layer18_local_appearance_score"],
                    -row["target_memory_score"],
                    -row["objectness_score"],
                    *row["box_xyxy"],
                    row["postprocess_index"],
                ),
            )
            selected = ranked[0]
            masks, sam_receipt = foundpose.nids.sam_base._sam_masks(
                processor, model, images[key], [selected["box_xyxy"]], images[key].size,
                torch, np,
            )
            mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
            refined_box = foundpose.nids.refine._tight_bbox(mask)
            evaluation = foundpose.nids.base._bbox_metrics(
                refined_box, image_rows[key]["bbox_xyxy"]
            )
            iou = float(evaluation["iou"])
            ious.append(iou)
            all_ious.append(iou)
            successes += int(iou >= minimum)
            total_successes += int(iou >= minimum)
            receipts[key] = {
                "query_truth_used_for_selection": False,
                "reference_median_area_fraction": reference_scale,
                "selected_area_fraction": _area_fraction(selected["box_xyxy"], images[key].size),
                "scale_adjusted_score": float(selected["layer18_local_appearance_score"])
                - penalty * abs(math.log(_area_fraction(selected["box_xyxy"], images[key].size) / reference_scale)),
                "selected": selected,
                "refined_mask_bbox_xyxy": refined_box,
                "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
                "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
                "sam_receipt": sam_receipt,
            }
        family_receipts[family] = {
            "intermediate_protocol_path": base_path.name,
            "reference_area_fractions": reference_areas,
            "metrics": {
                "query_count": len(receipts),
                "refined_iou_gate_queries": successes,
                "minimum_refined_iou": min(ious),
                "mean_refined_iou": float(np.mean(ious)),
            },
            "query_receipts": receipts,
        }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    required = int(protocol["decision_gate"]["required_refined_queries"])
    gate_met = total_successes == required
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_THREE_FAMILY_REFERENCE_SCALE_PRIOR_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": foundpose.nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": foundpose.nids.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_REFERENCE_SCALE_PRIOR_THREE_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met else
            "L10_3RSCAN_REFERENCE_SCALE_PRIOR_THREE_FAMILY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "selection": protocol["selection"],
        "metrics": {
            "family_count": len(family_receipts),
            "query_count": len(all_ious),
            "refined_iou_gate_queries": total_successes,
            "minimum_refined_iou": min(all_ious),
            "mean_refined_iou": float(np.mean(all_ious)),
        },
        "families": family_receipts,
        "runtime": {"device": torch.cuda.get_device_name(0), "sam2_calls": len(all_ious)},
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    foundpose.nids.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
