#!/usr/bin/env python3
"""Add fixed overlapping tile proposals to the frozen target-only FFA chain."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_mask_ffa_target_only_confirmation as confirmation  # noqa: E402
import l10_3rscan_objectness_set_memory_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-tiled-mask-ffa-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-tiled-mask-ffa-posthoc-result-v1"


def _tiled_proposals(
    protocol: dict[str, Any], images: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proposal = protocol["proposal"]
    tiling = protocol["tiling"]
    model_root = (ROOT / proposal["model_root"]).resolve()
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    results: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    calls = 0
    for key in protocol["evaluation"]["query_images"]:
        image = images[key]
        width, height = image.size
        tile_width = int(round(width * float(tiling["tile_width_fraction"])))
        tile_height = int(round(height * float(tiling["tile_height_fraction"])))
        windows = [
            ("full", 0, 0, width, height, int(proposal["maximum_proposals_per_image"])),
            ("top_left", 0, 0, tile_width, tile_height, int(tiling["maximum_proposals_per_tile"])),
            ("top_right", width - tile_width, 0, width, tile_height, int(tiling["maximum_proposals_per_tile"])),
            ("bottom_left", 0, height - tile_height, tile_width, height, int(tiling["maximum_proposals_per_tile"])),
            ("bottom_right", width - tile_width, height - tile_height, width, height, int(tiling["maximum_proposals_per_tile"])),
        ]
        started = time.perf_counter()
        combined: list[dict[str, Any]] = []
        for tile_id, left, top, right, bottom, cap in windows:
            crop = image.crop((left, top, right, bottom))
            crop_width, crop_height = crop.size
            inputs = processor(images=crop, text=[proposal["prompt"]], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                outputs = model(**inputs)
            calls += 1
            processed = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=float(proposal["box_threshold"]),
                text_threshold=float(proposal["text_threshold"]),
                target_sizes=[(crop_height, crop_width)],
            )[0]
            tile_rows: list[dict[str, Any]] = []
            for raw_index, (score, raw_box) in enumerate(
                zip(processed["scores"].detach().cpu().tolist(), processed["boxes"].detach().cpu().tolist())
            ):
                box = base.ground._clamp_box(raw_box, crop.size)
                if box is None or not math.isfinite(float(score)):
                    continue
                mapped = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
                tile_rows.append(
                    {
                        "objectness_score": float(score),
                        "box_xyxy": mapped,
                        "tile_id": tile_id,
                        "tile_xyxy": [left, top, right, bottom],
                        "tile_postprocess_index": raw_index,
                    }
                )
            tile_rows.sort(
                key=lambda row: (
                    -row["objectness_score"], *row["box_xyxy"], row["tile_postprocess_index"]
                )
            )
            combined.extend(tile_rows[:cap])
        for index, row in enumerate(combined):
            row["postprocess_index"] = index
        results[key] = combined
        timings[key] = time.perf_counter() - started
    runtime = {
        "model_type": type(model).__name__,
        "device": torch.cuda.get_device_name(0),
        "seconds_by_image": timings,
        "grounding_dino_calls": calls,
        "window_count_per_image": 5,
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results, runtime


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    predecessor = protocol["predecessor"]
    pixel.require(
        pixel.sha256(HERE / predecessor["path"]) == predecessor["sha256"], "PREDECESSOR_HASH"
    )
    previous = pixel.load_json(HERE / predecessor["path"])
    pixel.require(previous["conclusion"] == predecessor["required_conclusion"], "PREDECESSOR_CONCLUSION")
    original_proposals = base._objectness_proposals
    original_file = confirmation.__file__
    original_protocol_schema = confirmation.PROTOCOL_SCHEMA
    original_result_schema = confirmation.RESULT_SCHEMA
    try:
        base._objectness_proposals = _tiled_proposals
        confirmation.__file__ = __file__
        confirmation.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
        confirmation.RESULT_SCHEMA = RESULT_SCHEMA
        confirmation.run(protocol_path, output_path)
    finally:
        base._objectness_proposals = original_proposals
        confirmation.__file__ = original_file
        confirmation.PROTOCOL_SCHEMA = original_protocol_schema
        confirmation.RESULT_SCHEMA = original_result_schema
    result = pixel.load_json(output_path)
    result["authority"] = "CONSUMED_D03_FIXED_OVERLAPPING_TILE_PROPOSAL_POSTHOC_DEVELOPMENT"
    result["conclusion"] = (
        "L10_3RSCAN_TILED_MASK_FFA_POSTHOC_DEVELOPMENT_GATE_MET"
        if result["gate_met"] else "L10_3RSCAN_TILED_MASK_FFA_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
    )
    result["tiling"] = protocol["tiling"]
    result["claim_boundary"] = protocol["claim_boundary"]
    pixel.atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
