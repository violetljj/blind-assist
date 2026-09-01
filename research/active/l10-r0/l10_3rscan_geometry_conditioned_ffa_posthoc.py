#!/usr/bin/env python3
"""Condition robust FFA consensus on frozen reference aspect geometry."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_contrastive_sam_refinement_posthoc as refine  # noqa: E402
import l10_3rscan_mask_ffa_memory_posthoc as ffa  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-geometry-conditioned-ffa-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-geometry-conditioned-ffa-posthoc-result-v1"


def _aspect(box: list[float]) -> float:
    return (float(box[2]) - float(box[0])) / (float(box[3]) - float(box[1]))


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
    masker = protocol["masker"]
    pixel.require(pixel.sha256(ROOT / masker["model_path"]) == masker["model_sha256"], "MASKER_HASH")
    cohort = pixel.load_json(cohort_path)
    images, image_rows = ffa._load_images(protocol, cohort)
    reference_aspects = [_aspect(image_rows[key]["bbox_xyxy"]) for key in protocol["memory"]["target_images"]]
    reference_aspect = float(statistics.median(reference_aspects))

    selected: dict[str, dict[str, Any]] = {}
    ranked: dict[str, list[dict[str, Any]]] = {}
    for key in protocol["evaluation"]["query_images"]:
        rows: list[dict[str, Any]] = []
        for source in predecessor["query_receipts"][key]["ranked_candidates"]:
            candidate_aspect = _aspect(source["box_xyxy"])
            shape_similarity = math.exp(-abs(math.log(candidate_aspect / reference_aspect)))
            rows.append(
                {
                    **source,
                    "candidate_aspect_ratio": candidate_aspect,
                    "reference_median_aspect_ratio": reference_aspect,
                    "shape_similarity": shape_similarity,
                    "geometry_conditioned_score": float(source["target_memory_score"]) * shape_similarity,
                }
            )
        rows.sort(
            key=lambda row: (
                -row["geometry_conditioned_score"], -row["target_memory_score"],
                -row["objectness_score"], *row["box_xyxy"], row["postprocess_index"],
            )
        )
        selected[key] = rows[0]
        ranked[key] = rows

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / masker["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    successes = 0
    refined_ious: list[float] = []
    receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        masks, sam_receipt = sam_base._sam_masks(
            processor, model, images[key], [selected[key]["box_xyxy"]], images[key].size, torch, np
        )
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = refine._tight_bbox(mask)
        evaluation = base._bbox_metrics(refined_box, image_rows[key]["bbox_xyxy"])
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "selected": selected[key],
            "top1_geometry_conditioned_margin": float(
                selected[key]["geometry_conditioned_score"] - ranked[key][1]["geometry_conditioned_score"]
            ),
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "sam_receipt": sam_receipt,
        }
    model_type = type(model).__name__
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes >= int(protocol["decision_gate"]["required_refined_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_GEOMETRY_CONDITIONED_ROBUST_FFA_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_GEOMETRY_CONDITIONED_FFA_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_GEOMETRY_CONDITIONED_FFA_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(protocol["evaluation"]["query_images"]),
            "predecessor_opportunity_queries": predecessor["metrics"]["predecessor_opportunity_queries"],
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "reference_geometry": {
            "aspect_ratios": reference_aspects,
            "median_aspect_ratio": reference_aspect,
        },
        "query_receipts": receipts,
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "sam_model_type": model_type,
            "grounding_dino_calls": 0,
            "dinov2_calls": 0,
            "sam2_calls": len(receipts),
        },
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
