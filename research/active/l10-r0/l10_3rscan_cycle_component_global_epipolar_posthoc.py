#!/usr/bin/env python3
"""Gate local target cycles with target-excluded global epipolar geometry."""

from __future__ import annotations

import argparse
import gc
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_cycle_component_sibling_door_posthoc as sibling  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-global-epipolar-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-global-epipolar-posthoc-result-v1"


@contextmanager
def protocol_surface():
    base = open_set.base
    saved_schema = base.PROTOCOL_SCHEMA
    saved_file = base.__file__
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    try:
        yield
    finally:
        base.PROTOCOL_SCHEMA = saved_schema
        base.__file__ = saved_file


def verify_sibling_absence(protocol_path: Path) -> dict[str, Any]:
    saved_schema = sibling.PROTOCOL_SCHEMA
    sibling.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    try:
        return sibling.verify_sibling_absence(protocol_path)
    finally:
        sibling.PROTOCOL_SCHEMA = saved_schema


def pixel_coords(normalized: np.ndarray, size: int) -> np.ndarray:
    return (normalized + 1.0) * float(size) / 2.0


def cycle_domains(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    reference_mask_native: np.ndarray,
    matcher: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base = open_set.base
    height, double_width = certainty.shape
    width = double_width // 2
    base.require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    reference_mask = base.context_base.resize_mask(reference_mask_native, width, certainty.device)
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
    valid_cycle = (
        (certainty[:, :width] >= threshold)
        & torch.all(torch.abs(target_coords) <= 1.0, dim=-1)
        & (sampled_backward_certainty >= threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    target_cycle = valid_cycle & reference_mask
    background_cycle = valid_cycle & ~reference_mask
    target_count = int(target_cycle.sum().item())
    background_count = int(background_cycle.sum().item())
    base.require(background_count >= 8, "INSUFFICIENT_BACKGROUND_CYCLES")
    if target_count == 0:
        target_source = np.empty((0, 2), dtype=np.float64)
        target_query = np.empty((0, 2), dtype=np.float64)
        component_receipt = {"component_count": 0, "selected_label": None, "selected_pixels": 0}
    else:
        component, component_receipt = base.largest_cycle_component(target_cycle)
        target_source = source_coords[component].detach().cpu().numpy().astype(np.float64)
        target_query = target_coords[component].detach().cpu().numpy().astype(np.float64)
    background_source = source_coords[background_cycle].detach().cpu().numpy().astype(np.float64)
    background_query = target_coords[background_cycle].detach().cpu().numpy().astype(np.float64)
    return target_source, target_query, background_source, background_query, {
        "target_cycle_pixels": target_count,
        "target_component": component_receipt,
        "background_cycle_pixels": background_count,
    }


def epipolar_receipt(
    target_source: np.ndarray,
    target_query: np.ndarray,
    background_source: np.ndarray,
    background_query: np.ndarray,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    gate = protocol["epipolar_gate"]
    size = int(protocol["matcher"]["upsample_resolution"])
    source_pixels = pixel_coords(background_source, size)
    query_pixels = pixel_coords(background_query, size)
    maximum = int(gate["maximum_background_fit_points"])
    if len(source_pixels) > maximum:
        indices = np.linspace(0, len(source_pixels) - 1, maximum, dtype=np.int64)
        source_pixels = source_pixels[indices]
        query_pixels = query_pixels[indices]
    cv2.setRNGSeed(int(gate["opencv_rng_seed"]))
    fundamental, inliers = cv2.findFundamentalMat(
        source_pixels,
        query_pixels,
        cv2.FM_RANSAC,
        float(gate["ransac_reprojection_threshold_pixels"]),
        float(gate["ransac_confidence"]),
        int(gate["ransac_max_iterations"]),
    )
    open_set.base.require(fundamental is not None and fundamental.shape == (3, 3), "FUNDAMENTAL_MATRIX")
    background_inlier_fraction = float(np.asarray(inliers).reshape(-1).mean())
    if len(target_source) == 0:
        return {
            "background_fit_points": int(len(source_pixels)),
            "background_ransac_inlier_fraction": background_inlier_fraction,
            "target_component_points": 0,
            "target_epipolar_inlier_fraction": 0.0,
            "target_median_sampson_error_pixels_squared": None,
            "global_epipolar_support": False,
        }
    target_source_pixels = pixel_coords(target_source, size)
    target_query_pixels = pixel_coords(target_query, size)
    ones = np.ones((len(target_source_pixels), 1), dtype=np.float64)
    first = np.concatenate((target_source_pixels, ones), axis=1)
    second = np.concatenate((target_query_pixels, ones), axis=1)
    f_first = first @ fundamental.T
    ft_second = second @ fundamental
    residual = np.sum(second * f_first, axis=1)
    denominator = f_first[:, 0] ** 2 + f_first[:, 1] ** 2 + ft_second[:, 0] ** 2 + ft_second[:, 1] ** 2
    sampson = residual ** 2 / np.maximum(denominator, 1e-12)
    threshold_squared = float(gate["target_sampson_threshold_pixels"]) ** 2
    inlier_fraction = float(np.mean(sampson <= threshold_squared))
    supported = inlier_fraction >= float(gate["minimum_target_epipolar_inlier_fraction"])
    return {
        "background_fit_points": int(len(source_pixels)),
        "background_ransac_inlier_fraction": background_inlier_fraction,
        "target_component_points": int(len(target_source_pixels)),
        "target_epipolar_inlier_fraction": inlier_fraction,
        "target_median_sampson_error_pixels_squared": float(np.median(sampson)),
        "global_epipolar_support": supported,
    }


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch

    base = open_set.base
    absence_receipts = verify_sibling_absence(protocol_path)
    with protocol_surface():
        protocol = base.load_protocol(protocol_path)
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    predecessor = base.load_json(predecessor_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    episodes = {str(row["episode_id"]): row for row in cohort["episodes"]}
    reference_masks, reference_receipts, device_name = reference_base.make_reference_masks(
        protocol, cohort, images, inputs
    )
    reference_mask_by_episode: dict[str, np.ndarray] = {}
    for episode_id in episodes:
        image = images[f"{episode_id}:reference"]
        bbox = inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
        lookup = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        reference_mask_by_episode[episode_id] = reference_masks[lookup]

    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(
        model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True
    )
    matcher_model = romatch.roma_indoor(
        device="cuda",
        weights=weights,
        dinov2_weights=dinov2_weights,
        coarse_res=int(protocol["matcher"]["coarse_resolution"]),
        upsample_res=int(protocol["matcher"]["upsample_resolution"]),
        symmetric=True,
        use_custom_corr=False,
        upsample_preds=True,
    )
    receipts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        with torch.inference_mode():
            warp_batch, certainty_batch = matcher_model.match(
                images[f"{reference_id}:reference"], images[f"{query_id}:query"]
            )
        domains = cycle_domains(
            warp_batch[0].detach().cpu(),
            certainty_batch[0].detach().cpu(),
            reference_mask_by_episode[reference_id],
            protocol["matcher"],
        )
        target_source, target_query, background_source, background_query, cycle_receipt = domains
        receipt = epipolar_receipt(
            target_source, target_query, background_source, background_query, protocol
        )
        receipts[pair_id] = {**cycle_receipt, **receipt}
        predecessor_commit = bool(predecessor["decisions"][pair_id]["commit"])
        commit = predecessor_commit and bool(receipt["global_epipolar_support"])
        decisions[pair_id] = {
            **pair,
            "predecessor_bilateral_mask_paired_commit": predecessor_commit,
            "global_epipolar_support": bool(receipt["global_epipolar_support"]),
            "commit": commit,
            "correct": commit if pair["label"] == "target_present" else not commit,
        }
    del matcher_model, weights, dinov2_weights
    gc.collect()
    torch.cuda.empty_cache()

    positive_rows = [row for row in decisions.values() if row["label"] == "target_present"]
    negative_rows = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positive_rows)
    negative_false_commits = sum(bool(row["commit"]) for row in negative_rows)
    decision_gate = protocol["decision_gate"]
    gate_met = (
        len(positive_rows) == int(decision_gate["required_positive_pairs"])
        and len(negative_rows) == int(decision_gate["required_target_absent_pairs"])
        and positive_commits >= int(decision_gate["minimum_positive_commits"])
        and negative_false_commits <= int(decision_gate["maximum_target_absent_false_commits"])
    )
    base.roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_TARGET_EXCLUDED_GLOBAL_EPIPOLAR_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
            "conclusion": (
                "L10_3RSCAN_GLOBAL_EPIPOLAR_CYCLE_COMPONENT_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_3RSCAN_GLOBAL_EPIPOLAR_CYCLE_COMPONENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positive_rows),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negative_rows),
                "target_absent_false_commits": negative_false_commits,
            },
            "decisions": decisions,
            "global_epipolar_receipts": receipts,
            "reference_support_receipts": reference_receipts,
            "sibling_absence_receipts": absence_receipts,
            "runtime": {
                "device": device_name,
                "roma_calls": len(decisions),
                "sam2_calls": len(reference_receipts),
                "grounding_dino_calls": 0,
            },
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
