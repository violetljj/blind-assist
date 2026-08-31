#!/usr/bin/env python3
"""Replace affine extent extrapolation with one robust planar homography."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from copy import deepcopy
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
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-projective-extent-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-projective-extent-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def projective_prompt(
    extent_masks: dict[str, np.ndarray],
    warp: torch.Tensor,
    certainty: torch.Tensor,
    reference_mask: np.ndarray,
    query_size: tuple[int, int],
    matcher: dict[str, Any],
    reprojection_threshold_pixels: float,
) -> tuple[list[float], dict[str, Any]]:
    base = open_set.base
    height, double_width = certainty.shape
    width = double_width // 2
    require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    support_mask = base.context_base.resize_mask(reference_mask, width, certainty.device)
    support_hash = base.mask_sha256(reference_mask)
    require(support_hash in extent_masks, "REFERENCE_EXTENT_NOT_FROZEN")
    extent_mask = base.context_base.resize_mask(extent_masks[support_hash], width, certainty.device)
    forward = warp[:, :width]
    backward = warp[:, width:]
    source_coords = forward[..., :2]
    target_coords = forward[..., 2:]
    sampled_backward_coords = F.grid_sample(
        backward[..., :2].permute(2, 0, 1)[None], target_coords[None],
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )[0].permute(1, 2, 0)
    sampled_backward_certainty = F.grid_sample(
        certainty[:, width:][None, None], target_coords[None],
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )[0, 0]
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    certainty_threshold = float(matcher["official_certainty_threshold"])
    high = support_mask & (certainty[:, :width] >= certainty_threshold)
    cycle = (
        high
        & torch.all(torch.abs(target_coords) <= 1.0, dim=-1)
        & (sampled_backward_certainty >= certainty_threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    support_count = int(support_mask.sum().item())
    high_count = int(high.sum().item())
    cycle_count = int(cycle.sum().item())
    require(support_count > 0 and cycle_count > 0, "NO_REFERENCE_CYCLES")
    component, component_receipt = base.largest_cycle_component(cycle)
    component = component.to(device=warp.device)
    selected_source = source_coords[component].detach().cpu().numpy().astype(np.float64)
    selected_target = target_coords[component].detach().cpu().numpy().astype(np.float64)
    require(len(selected_source) >= 4, "PROJECTIVE_SUPPORT_COUNT")
    source_pixels = (selected_source + 1.0) * width / 2.0
    target_pixels = (selected_target + 1.0) * height / 2.0
    cv2.setRNGSeed(0)
    homography, inlier_mask = cv2.findHomography(
        source_pixels,
        target_pixels,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=reprojection_threshold_pixels,
        maxIters=10000,
        confidence=0.999,
    )
    require(homography is not None and np.isfinite(homography).all(), "PROJECTIVE_FIT")
    require(np.linalg.matrix_rank(homography) == 3, "PROJECTIVE_RANK")
    inliers = inlier_mask.reshape(-1).astype(bool)
    require(int(inliers.sum()) >= 4, "PROJECTIVE_INLIERS")
    predicted = cv2.perspectiveTransform(source_pixels[None].astype(np.float64), homography)[0]
    residual = np.linalg.norm(predicted - target_pixels, axis=1)

    extent_domain = source_coords[extent_mask].detach().cpu().numpy().astype(np.float64)
    sx0, sy0 = np.min(extent_domain, axis=0)
    sx1, sy1 = np.max(extent_domain, axis=0)
    corners_normalized = np.asarray([[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]])
    corners_pixels = (corners_normalized + 1.0) * width / 2.0
    projected_pixels = cv2.perspectiveTransform(corners_pixels[None].astype(np.float64), homography)[0]
    query_width, query_height = query_size
    xs = np.clip(projected_pixels[:, 0] * query_width / width, 0.0, float(query_width))
    ys = np.clip(projected_pixels[:, 1] * query_height / height, 0.0, float(query_height))
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    require(box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0, "DEGENERATE_PROJECTIVE_BOX")
    return box, {
        "selection_authority": "REFERENCE_SAM_SUPPORT_LARGEST_CYCLE_COMPONENT_USAC_MAGSAC_HOMOGRAPHY",
        "support_mask_sha256": support_hash,
        "support_mask_pixels_at_match_resolution": support_count,
        "extent_mask_pixels_at_match_resolution": int(extent_mask.sum().item()),
        "high_certainty_pixels": high_count,
        "all_cycle_pixels": cycle_count,
        "all_cycle_fraction": cycle_count / support_count,
        "all_cycle_purity": cycle_count / high_count if high_count else 0.0,
        "component": component_receipt,
        "selected_component_fraction_of_cycles": len(selected_source) / cycle_count,
        "homography": homography.tolist(),
        "homography_rank": int(np.linalg.matrix_rank(homography)),
        "magsac_reprojection_threshold_pixels": reprojection_threshold_pixels,
        "homography_inliers": int(inliers.sum()),
        "homography_inlier_fraction": float(inliers.mean()),
        "homography_inlier_mean_residual_pixels": float(np.mean(residual[inliers])),
        "homography_inlier_max_residual_pixels": float(np.max(residual[inliers])),
        "projected_bound_extent_corners_match_pixels": projected_pixels.tolist(),
        "prompt_box_xyxy": box,
    }


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch

    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    predecessor = load_json(predecessor_path)
    require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    inner_path = HERE / protocol["local_carrier_protocol_path"]
    require(sha256(inner_path) == protocol["local_carrier_protocol_sha256"], "LOCAL_PROTOCOL_HASH")
    with open_set.protocol_surface():
        inner = open_set.base.load_protocol(inner_path)
    cohort = open_set.base.load_json(HERE / inner["source"]["cohort_path"])
    images, inputs = open_set.base.load_images(inner, cohort)
    target_id = str(protocol["evaluation"]["positive_pair_id"])
    positive = next(row for row in inner["evaluation"]["pairs"] if str(row["id"]) == target_id)
    episode_id = str(positive["reference_episode"])
    reduced_cohort = deepcopy(cohort)
    reduced_cohort["episodes"] = [row for row in cohort["episodes"] if str(row["episode_id"]) == episode_id]
    reduced_images = {key: value for key, value in images.items() if key.startswith(f"{episode_id}:")}
    reduced_inputs = {key: value for key, value in inputs.items() if key.startswith(f"{episode_id}:")}
    reference_masks, reference_receipts, device_name = reference_base.make_reference_masks(
        inner, reduced_cohort, reduced_images, reduced_inputs
    )
    reference_image = images[f"{episode_id}:reference"]
    query_image = images[f"{episode_id}:query"]
    bbox = inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
    lookup = (int(reference_image.size[0]), int(reference_image.size[1]), *(float(v) for v in bbox))
    reference_mask = reference_masks[lookup]
    extent_masks = {open_set.base.mask_sha256(reference_mask): open_set.base.rectangle_mask(reference_image.size, bbox)}

    model_root = ROOT / inner["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    backbone = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher_model = romatch.roma_indoor(
        device="cuda", weights=weights, dinov2_weights=backbone,
        coarse_res=int(inner["matcher"]["coarse_resolution"]),
        upsample_res=int(inner["matcher"]["upsample_resolution"]),
        symmetric=True, use_custom_corr=False, upsample_preds=True,
    )
    with torch.inference_mode():
        warp_batch, certainty_batch = matcher_model.match(reference_image, query_image)
    box, receipt = projective_prompt(
        extent_masks,
        warp_batch[0].detach().cpu(),
        certainty_batch[0].detach().cpu(),
        reference_mask,
        query_image.size,
        inner["matcher"],
        float(protocol["projective_model"]["reprojection_threshold_pixels"]),
    )
    del matcher_model, weights, backbone
    gc.collect()
    torch.cuda.empty_cache()
    target_box = inputs[f"{episode_id}:query"]["target_bbox_xyxy_evaluation_only"]
    iou, recall, precision = open_set.base.bbox_iou(box, target_box)
    receipt.update({
        "target_bbox_iou_evaluation_only": iou,
        "target_bbox_recall_evaluation_only": recall,
        "prompt_bbox_precision_evaluation_only": precision,
    })
    old_prompt = predecessor["local_carrier"]["prompt_receipts"][target_id]
    cycle_ok = bool(
        float(receipt["all_cycle_fraction"]) >= float(protocol["decision_gate"]["minimum_reference_cycle_fraction"])
        and float(receipt["selected_component_fraction_of_cycles"])
        >= float(protocol["decision_gate"]["minimum_dominant_component_cycle_fraction"])
    )
    extent_ok = iou >= float(protocol["decision_gate"]["minimum_positive_extent_target_bbox_iou"])
    direct_ok = float(receipt["all_cycle_fraction"]) >= float(protocol["decision_gate"]["minimum_direct_local_cycle_fraction"])
    projective_support = bool(
        float(receipt["homography_inlier_fraction"])
        >= float(protocol["decision_gate"]["minimum_projective_inlier_fraction"])
    )
    positive_commit = bool(cycle_ok and extent_ok and direct_ok and projective_support)
    negative_decisions = {
        pair_id: row for pair_id, row in predecessor["decisions"].items() if row["label"] == "target_absent"
    }
    inherited_false_commits = sum(bool(row["commit"]) for row in negative_decisions.values())
    gate_met = bool(
        positive_commit
        and inherited_false_commits <= int(protocol["decision_gate"]["maximum_target_absent_false_commits"])
    )
    write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SC34_PROJECTIVE_EXTENT_POSTHOC_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "predecessor": {"path": predecessor_path.name, "sha256": sha256(predecessor_path)},
        "conclusion": (
            "L10_3RSCAN_PROJECTIVE_EXTENT_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_PROJECTIVE_EXTENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "positive_pairs": 1,
            "positive_commits": int(positive_commit),
            "target_absent_pairs": len(negative_decisions),
            "target_absent_false_commits": inherited_false_commits,
            "affine_positive_extent_iou": float(old_prompt["target_bbox_iou_evaluation_only"]),
            "projective_positive_extent_iou": iou,
            "extent_iou_gain": iou - float(old_prompt["target_bbox_iou_evaluation_only"]),
        },
        "decision": {
            "id": target_id,
            "label": "target_present",
            "cycle_support": cycle_ok,
            "direct_exit_support": direct_ok,
            "projective_inlier_support": projective_support,
            "projective_extent_gate": extent_ok,
            "commit": positive_commit,
        },
        "affine_receipt": old_prompt,
        "projective_receipt": receipt,
        "inherited_negative_decisions": negative_decisions,
        "reference_support_receipt": reference_receipts[episode_id],
        "runtime": {"device": device_name, "roma_calls": 1, "sam2_calls": 1, "grounding_dino_calls": 0},
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
