#!/usr/bin/env python3
"""Refine contrastive-memory selections with frozen native SAM2 masks."""

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
import l10_3rscan_objectness_set_memory_posthoc as metrics  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-contrastive-sam-refinement-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-contrastive-sam-refinement-posthoc-result-v1"


def _tight_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    pixel.require(len(xs) > 0, "EMPTY_MASK")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    predecessor_path = HERE / protocol["predecessor"]["path"]
    cohort_path = HERE / protocol["source"]["cohort_path"]
    pixel.require(pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    pixel.require(pixel.sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    predecessor = pixel.load_json(predecessor_path)
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    model_path = ROOT / protocol["masker"]["model_path"]
    pixel.require(pixel.sha256(model_path) == protocol["masker"]["model_sha256"], "MASKER_HASH")

    cohort = pixel.load_json(cohort_path)
    query_keys = protocol["evaluation"]["query_images"]
    rows = {key: value for key, value in cohort["images"].items() if key in query_keys}
    pixel.require(set(rows) == set(query_keys), "QUERY_IMAGE_KEYS")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    image_hashes: dict[str, str] = {}
    for key, row in rows.items():
        archive_path = artifact_root / cohort["source_manifest"][f"{row['scan_id']}/sequence.zip"]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        image_hashes[key] = hashlib.sha256(payload).hexdigest()

    from transformers import Sam2Model, Sam2Processor

    model_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    processor = Sam2Processor.from_pretrained(model_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        model_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")

    query_receipts: dict[str, Any] = {}
    successes = 0
    refined_ious: list[float] = []
    for key in query_keys:
        selected = predecessor["query_receipts"][key]["selected"]
        input_box = selected["box_xyxy"]
        masks, receipt = sam_base._sam_masks(processor, model, images[key], [input_box], images[key].size, torch, np)
        pixel.require(len(masks) == 1, f"MASK_COUNT:{key}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        refined_box = _tight_bbox(mask)
        evaluation = metrics._bbox_metrics(refined_box, rows[key]["bbox_xyxy"])
        refined_iou = float(evaluation["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= float(protocol["decision_gate"]["minimum_iou"]))
        query_receipts[key] = {
            "selection_changed": False,
            "input_box_xyxy": input_box,
            "input_box_target_metrics_evaluation_only": selected["target_metrics_evaluation_only"],
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": evaluation,
            "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(),
            "mask_pixels": int(mask.sum()),
            "sam_receipt": receipt,
            "image_sha256": image_hashes[key],
        }

    runtime = {"model_type": type(model).__name__, "device": torch.cuda.get_device_name(0), "sam2_calls": len(query_keys)}
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = successes == int(protocol["decision_gate"]["required_queries"])
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_C16_CONTRASTIVE_SELECTION_NATIVE_SAM2_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CONTRASTIVE_SAM_REFINEMENT_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_CONTRASTIVE_SAM_REFINEMENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(query_keys),
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "runtime": runtime,
        "query_receipts": query_receipts,
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
