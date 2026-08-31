#!/usr/bin/env python3
"""Native reference-SAM mask successor for 3RScan cycle prompts."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-sam-reference-mask-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-sam-reference-mask-posthoc-result-v1"


@contextmanager
def successor_surface(reference_masks: dict[tuple[Any, ...], np.ndarray]):
    saved = {
        "PROTOCOL_SCHEMA": base.PROTOCOL_SCHEMA,
        "RESULT_SCHEMA": base.RESULT_SCHEMA,
        "__file__": base.__file__,
        "rectangle_mask": base.rectangle_mask,
    }

    def supplied_reference_mask(size: tuple[int, int], bbox: list[float]) -> np.ndarray:
        key = (int(size[0]), int(size[1]), *(float(value) for value in bbox))
        base.require(key in reference_masks, "REFERENCE_SAM_MASK_NOT_FROZEN")
        return np.ascontiguousarray(reference_masks[key], dtype=np.bool_)

    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.RESULT_SCHEMA = RESULT_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    base.rectangle_mask = supplied_reference_mask
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def make_reference_masks(
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    images: dict[str, Any],
    inputs: dict[str, Any],
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[str, Any], str]:
    from transformers import Sam2Model, Sam2Processor

    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    frozen: dict[tuple[Any, ...], np.ndarray] = {}
    receipts: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        key = f"{episode_id}:reference"
        image = images[key]
        bbox = inputs[key]["target_bbox_xyxy_evaluation_only"]
        masks, masker = base.sam_base._sam_masks(
            processor, model, image, [bbox], image.size, torch, np
        )
        base.require(len(masks) == 1, f"REFERENCE_SAM_MASK_COUNT:{episode_id}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        base.require(int(mask.sum()) > 0, f"EMPTY_REFERENCE_SAM_MASK:{episode_id}")
        lookup_key = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        frozen[lookup_key] = mask
        mask_box = base.mask_bbox(mask)
        iou, recall, precision = base.bbox_iou(mask_box, bbox)
        receipts[episode_id] = {
            "selection_authority": "ONE_NATIVE_SAM2_MASK_FROM_PRIVILEGED_INITIAL_REFERENCE_BBOX",
            "mask_sha256": base.mask_sha256(mask),
            "mask_pixels": int(mask.sum()),
            "mask_bbox_xyxy": mask_box,
            "binding_bbox_iou": iou,
            "binding_bbox_recall": recall,
            "reference_mask_bbox_precision": precision,
            "masker": masker,
        }
    device = torch.cuda.get_device_name(0)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return frozen, receipts, device


def replay(protocol_path: Path, output_path: Path) -> None:
    with successor_surface({}):
        protocol = base.load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    reference_masks, reference_receipts, masker_device = make_reference_masks(
        protocol, cohort, images, inputs
    )
    with successor_surface(reference_masks):
        base.replay(protocol_path, output_path)
    result = base.load_json(output_path)
    result["authority"] = "CONSUMED_POSTHOC_NATIVE_REFERENCE_MASK_CONDITIONED_MULTI_DOOR_PROPOSAL_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_3RSCAN_ROMA_CYCLE_PROMPT_SAM_REFERENCE_MASK_POSTHOC_DEVELOPMENT_GATE_MET"
        if result["gate_met"]
        else "L10_3RSCAN_ROMA_CYCLE_PROMPT_SAM_REFERENCE_MASK_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
    )
    result["reference_proposal_receipts"] = reference_receipts
    result["runtime"]["sam2_calls"] = int(result["runtime"]["sam2_calls"]) + len(reference_receipts)
    result["runtime"]["reference_masker_device"] = masker_device
    base.roma_base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
