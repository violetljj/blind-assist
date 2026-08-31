#!/usr/bin/env python3
"""Dual-surface reference support and extent for 3RScan cycle prompts."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-dual-surface-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-dual-surface-posthoc-result-v1"


def dual_surface_cycle_affine_prompt(
    extent_masks: dict[str, np.ndarray],
    warp: torch.Tensor,
    certainty: torch.Tensor,
    reference_mask: np.ndarray,
    query_size: tuple[int, int],
    matcher: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    height, double_width = certainty.shape
    width = double_width // 2
    base.require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    support_mask = base.context_base.resize_mask(reference_mask, width, certainty.device)
    support_hash = base.mask_sha256(reference_mask)
    base.require(support_hash in extent_masks, "REFERENCE_EXTENT_NOT_FROZEN")
    extent_mask = base.context_base.resize_mask(extent_masks[support_hash], width, certainty.device)
    forward = warp[:, :width]
    backward = warp[:, width:]
    source_coords = forward[..., :2]
    target_coords = forward[..., 2:]
    sampled_backward_coords = F.grid_sample(
        backward[..., :2].permute(2, 0, 1)[None],
        target_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    sampled_backward_certainty = F.grid_sample(
        certainty[:, width:][None, None],
        target_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    threshold = float(matcher["official_certainty_threshold"])
    in_bounds = torch.all(torch.abs(target_coords) <= 1.0, dim=-1)
    high = support_mask & (certainty[:, :width] >= threshold)
    cycle = (
        high
        & in_bounds
        & (sampled_backward_certainty >= threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    support_count = int(support_mask.sum().item())
    high_count = int(high.sum().item())
    cycle_count = int(cycle.sum().item())
    base.require(support_count > 0 and cycle_count > 0, "NO_REFERENCE_CYCLES")
    component, component_receipt = base.largest_cycle_component(cycle)
    component = component.to(device=warp.device)
    selected_source = source_coords[component].detach().cpu().numpy().astype(np.float64)
    selected_target = target_coords[component].detach().cpu().numpy().astype(np.float64)
    design = np.concatenate(
        [selected_source, np.ones((len(selected_source), 1), dtype=np.float64)], axis=1
    )
    base.require(len(selected_source) >= 3 and np.linalg.matrix_rank(design) == 3, "AFFINE_SUPPORT_RANK")
    coefficients, _, _, _ = np.linalg.lstsq(design, selected_target, rcond=None)
    extent_domain = source_coords[extent_mask].detach().cpu().numpy().astype(np.float64)
    sx0, sy0 = np.min(extent_domain, axis=0)
    sx1, sy1 = np.max(extent_domain, axis=0)
    corners = np.asarray(
        [[sx0, sy0, 1.0], [sx1, sy0, 1.0], [sx1, sy1, 1.0], [sx0, sy1, 1.0]]
    )
    projected = corners @ coefficients
    query_width, query_height = query_size
    xs = np.clip((projected[:, 0] + 1.0) * query_width / 2.0, 0.0, float(query_width))
    ys = np.clip((projected[:, 1] + 1.0) * query_height / 2.0, 0.0, float(query_height))
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    base.require(box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0, "DEGENERATE_PROMPT_BOX")
    predicted = design @ coefficients
    residual = np.linalg.norm(predicted - selected_target, axis=1)
    return box, {
        "selection_authority": "REFERENCE_SAM_SUPPORT_TO_LARGEST_CYCLE_COMPONENT_WITH_BOUND_BBOX_EXTENT",
        "support_mask_sha256": support_hash,
        "support_mask_pixels_at_match_resolution": support_count,
        "extent_mask_pixels_at_match_resolution": int(extent_mask.sum().item()),
        "high_certainty_pixels": high_count,
        "all_cycle_pixels": cycle_count,
        "all_cycle_fraction": cycle_count / support_count,
        "all_cycle_purity": cycle_count / high_count if high_count else 0.0,
        "component": component_receipt,
        "selected_component_fraction_of_support": len(selected_source) / support_count,
        "selected_component_fraction_of_cycles": len(selected_source) / cycle_count,
        "affine_rank": int(np.linalg.matrix_rank(design)),
        "affine_mean_residual_normalized": float(np.mean(residual)),
        "affine_max_residual_normalized": float(np.max(residual)),
        "projected_bound_extent_corners_normalized": projected.tolist(),
        "prompt_box_xyxy": box,
    }


@contextmanager
def dual_surface(reference_masks: dict[tuple[Any, ...], np.ndarray], extent_masks: dict[str, np.ndarray]):
    saved = {
        "PROTOCOL_SCHEMA": base.PROTOCOL_SCHEMA,
        "RESULT_SCHEMA": base.RESULT_SCHEMA,
        "__file__": base.__file__,
        "rectangle_mask": base.rectangle_mask,
        "cycle_affine_prompt": base.cycle_affine_prompt,
    }

    def supplied_reference_mask(size: tuple[int, int], bbox: list[float]) -> np.ndarray:
        key = (int(size[0]), int(size[1]), *(float(value) for value in bbox))
        base.require(key in reference_masks, "REFERENCE_SAM_MASK_NOT_FROZEN")
        return np.ascontiguousarray(reference_masks[key], dtype=np.bool_)

    def supplied_cycle_prompt(
        warp: torch.Tensor,
        certainty: torch.Tensor,
        reference_mask: np.ndarray,
        query_size: tuple[int, int],
        matcher: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        return dual_surface_cycle_affine_prompt(
            extent_masks, warp, certainty, reference_mask, query_size, matcher
        )

    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.RESULT_SCHEMA = RESULT_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    base.rectangle_mask = supplied_reference_mask
    base.cycle_affine_prompt = supplied_cycle_prompt
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def replay(protocol_path: Path, output_path: Path) -> None:
    with dual_surface({}, {}):
        protocol = base.load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    reference_masks, reference_receipts, masker_device = reference_base.make_reference_masks(
        protocol, cohort, images, inputs
    )
    extent_masks: dict[str, np.ndarray] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        key = f"{episode_id}:reference"
        image = images[key]
        bbox = inputs[key]["target_bbox_xyxy_evaluation_only"]
        lookup_key = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        support_mask = reference_masks[lookup_key]
        extent_masks[base.mask_sha256(support_mask)] = base.rectangle_mask(image.size, bbox)
    with dual_surface(reference_masks, extent_masks):
        base.replay(protocol_path, output_path)
    result = base.load_json(output_path)
    result["authority"] = "CONSUMED_POSTHOC_DUAL_SURFACE_REFERENCE_CONDITIONED_MULTI_DOOR_PROPOSAL_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_POSTHOC_DEVELOPMENT_GATE_MET"
        if result["gate_met"]
        else "L10_3RSCAN_ROMA_CYCLE_PROMPT_DUAL_SURFACE_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
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
