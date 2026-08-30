#!/usr/bin/env python3
"""Spatially coherent bounded reference-transfer endpoint masks on 3RScan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_bounded_reference_mask as bounded  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel_field  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-spatial-reference-mask-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-spatial-reference-mask-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-spatial-reference-mask-result-v1"


def select_spatial_component(
    active: np.ndarray,
    margin: np.ndarray,
    mutual_positive: np.ndarray,
    minimum_patches: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    require_shape = active.shape == margin.shape == mutual_positive.shape
    bounded.require(require_shape and active.ndim == 2, "SPATIAL_FIELD_SHAPE")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8
    )
    candidates: list[tuple[int, float, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_patches:
            continue
        component = labels == label
        candidates.append(
            (
                int(np.count_nonzero(mutual_positive & component)),
                float(np.sum(margin[component])),
                area,
                -label,
            )
        )
    if not candidates:
        return None, {
            "active_patches": int(np.count_nonzero(active)),
            "eligible_components": 0,
            "failure": "NO_SPATIAL_COMPONENT",
        }
    seed_hits, margin_mass, area, negative_label = max(candidates)
    selected_label = -negative_label
    return np.ascontiguousarray(labels == selected_label), {
        "active_patches": int(np.count_nonzero(active)),
        "eligible_components": len(candidates),
        "selected_component_label": selected_label,
        "selected_component_patches": area,
        "selected_mutual_positive_seed_hits": seed_hits,
        "selected_positive_margin_mass": margin_mass,
        "failure": None,
    }


def empty_prediction(
    arm_name: str,
    fraction: float,
    query_shape: tuple[int, int],
    full_polygon: np.ndarray,
    supervision_polygon: np.ndarray,
    reference_positive: np.ndarray,
    reference_full: np.ndarray,
    transfer: dict[str, Any],
    failure: str,
) -> tuple[np.ndarray, list[list[float]], np.ndarray, dict[str, Any]]:
    support = np.zeros(query_shape, dtype=bool)
    return (
        np.empty((0, 2), dtype=np.float32),
        [],
        support,
        {
            "arm": arm_name,
            "retained_width_fraction": fraction,
            "reference_full_polygon_pixels": full_polygon.tolist(),
            "reference_supervision_polygon_pixels": supervision_polygon.tolist(),
            "reference_positive_patches": int(np.count_nonzero(reference_positive)),
            "reference_full_patches": int(np.count_nonzero(reference_full)),
            "transfer": transfer,
            "masker": {
                "failure": failure,
                "selected_support_pixels": 0,
                "endpoint_pixels": [],
                "endpoints_are_supported_query_pixels": False,
                "homography_used": False,
                "coordinate_regression_used": False,
                "image_space_extrapolation_used": False,
            },
            "support_mask_uint8_sha256": hashlib.sha256(
                support.astype(np.uint8).tobytes(order="C")
            ).hexdigest(),
        },
    )


def spatial_arm_prediction(
    arm_name: str,
    fraction: float,
    reference_points: np.ndarray,
    reference_pose: np.ndarray,
    reference_info: dict[str, Any],
    reference_features: torch.Tensor,
    query_features: torch.Tensor,
    query_rgb: np.ndarray,
    grid: tuple[int, int],
    transfer_config: dict[str, Any],
    sam_config: dict[str, Any],
    sam_processor: Any,
    sam_model: Any,
) -> tuple[np.ndarray, list[list[float]], np.ndarray, dict[str, Any]]:
    full_polygon, _ = pixel_field.projected_hull(
        reference_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    portal_frame = extent.portal_frame(reference_points)
    portal_uv = extent.project_uv(reference_points, *portal_frame)
    supervised_points = (
        reference_points
        if fraction >= 1.0
        else extent.partial_fragment(reference_points, portal_uv, fraction)
    )
    supervision_polygon, _ = pixel_field.projected_hull(
        supervised_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    reference_centres = pixel_field.grid_centres(
        reference_info["color_width"], reference_info["color_height"], grid
    )
    query_centres = pixel_field.grid_centres(query_rgb.shape[1], query_rgb.shape[0], grid)
    reference_positive = pixel_field.points_in_polygon(reference_centres, supervision_polygon)
    reference_full = pixel_field.points_in_polygon(reference_centres, full_polygon)
    bounded.require(np.count_nonzero(reference_positive) >= 2, "REFERENCE_POSITIVE_SUPPORT")
    bounded.require(np.count_nonzero(~reference_full) >= 2, "REFERENCE_BACKGROUND_SUPPORT")

    device = reference_features.device
    positive_tensor = torch.from_numpy(reference_positive).to(device=device)
    background_tensor = torch.from_numpy(~reference_full).to(device=device)
    with torch.inference_mode():
        similarities = query_features @ reference_features.T
        foreground_scores = torch.max(similarities[:, positive_tensor], dim=1).values
        background_scores = torch.max(similarities[:, background_tensor], dim=1).values
        margin_tensor = foreground_scores - background_scores
        query_best_reference = torch.argmax(similarities, dim=1)
        reference_best_query = torch.argmax(similarities, dim=0)
        query_indices = torch.arange(len(query_features), device=device)
        mutual = reference_best_query[query_best_reference] == query_indices
        mutual_positive_tensor = mutual & positive_tensor[query_best_reference]
    foreground = foreground_scores.detach().cpu().numpy()
    background = background_scores.detach().cpu().numpy()
    margin = margin_tensor.detach().cpu().numpy()
    mutual_positive = mutual_positive_tensor.detach().cpu().numpy()
    active = margin >= 0.0
    grid_height, grid_width = grid
    component, component_debug = select_spatial_component(
        active.reshape(grid),
        margin.reshape(grid),
        mutual_positive.reshape(grid),
        int(transfer_config["minimum_component_patches"]),
    )
    transfer: dict[str, Any] = {
        "foreground_score_minimum": float(np.min(foreground)),
        "foreground_score_maximum": float(np.max(foreground)),
        "background_score_minimum": float(np.min(background)),
        "background_score_maximum": float(np.max(background)),
        "margin_minimum": float(np.min(margin)),
        "margin_maximum": float(np.max(margin)),
        "mutual_positive_matches": int(np.count_nonzero(mutual_positive)),
        "component": component_debug,
    }
    if component is None:
        return empty_prediction(
            arm_name,
            fraction,
            query_rgb.shape[:2],
            full_polygon,
            supervision_polygon,
            reference_positive,
            reference_full,
            transfer,
            "NO_SPATIAL_COMPONENT",
        )

    component_indices = np.flatnonzero(component.reshape(-1))
    separation = int(transfer_config["minimum_grid_cell_separation"])
    positive_indices = bounded.spread_select(
        component_indices,
        margin,
        grid,
        int(transfer_config["positive_prompt_limit"]),
        separation,
    )
    ys, xs = np.nonzero(component)
    expansion = int(transfer_config["roi_expansion_grid_cells"])
    cell_x0 = max(0, int(np.min(xs)) - expansion)
    cell_y0 = max(0, int(np.min(ys)) - expansion)
    cell_x1 = min(grid_width, int(np.max(xs)) + expansion + 1)
    cell_y1 = min(grid_height, int(np.max(ys)) + expansion + 1)
    roi_grid = np.zeros(grid, dtype=bool)
    roi_grid[cell_y0:cell_y1, cell_x0:cell_x1] = True
    negative_candidates = np.flatnonzero(roi_grid.reshape(-1) & (~active))
    negative_indices = bounded.spread_select(
        negative_candidates,
        -margin,
        grid,
        int(transfer_config["negative_prompt_limit"]),
        separation,
    )
    points = [query_centres[index].astype(float).tolist() for index in positive_indices]
    points.extend(query_centres[index].astype(float).tolist() for index in negative_indices)
    labels = [1] * len(positive_indices) + [0] * len(negative_indices)
    transfer.update(
        {
            "positive_prompt_indices": positive_indices,
            "negative_prompt_indices": negative_indices,
            "positive_prompt_points": points[: len(positive_indices)],
            "negative_prompt_points": points[len(positive_indices) :],
            "positive_prompt_margins": [float(margin[index]) for index in positive_indices],
            "negative_prompt_background_margins": [float(-margin[index]) for index in negative_indices],
            "minimum_positive_prompts_met": len(positive_indices)
            >= int(transfer_config["minimum_positive_prompts"]),
            "roi_grid_xyxy_exclusive": [cell_x0, cell_y0, cell_x1, cell_y1],
        }
    )
    if not transfer["minimum_positive_prompts_met"]:
        return empty_prediction(
            arm_name,
            fraction,
            query_rgb.shape[:2],
            full_polygon,
            supervision_polygon,
            reference_positive,
            reference_full,
            transfer,
            "INSUFFICIENT_SPATIAL_POSITIVE_PROMPTS",
        )

    _, _, raw_support, raw_masker = bounded.sam_supported_extent(
        sam_processor,
        sam_model,
        query_rgb,
        points,
        labels,
        sam_config,
    )
    height, width = query_rgb.shape[:2]
    pixel_x0 = int(np.floor(cell_x0 * width / grid_width))
    pixel_y0 = int(np.floor(cell_y0 * height / grid_height))
    pixel_x1 = int(np.ceil(cell_x1 * width / grid_width))
    pixel_y1 = int(np.ceil(cell_y1 * height / grid_height))
    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = True
    bounded_support = raw_support & roi_mask
    try:
        selected_support, component_masker = bounded.select_prompt_component(
            bounded_support, points[: len(positive_indices)]
        )
        polygon, endpoints, endpoint_debug = bounded.support_extent(selected_support)
    except ValueError:
        return empty_prediction(
            arm_name,
            fraction,
            query_rgb.shape[:2],
            full_polygon,
            supervision_polygon,
            reference_positive,
            reference_full,
            transfer,
            "SAM_SUPPORT_EMPTY_INSIDE_SPATIAL_ROI",
        )
    masker = {
        **raw_masker,
        "pre_roi_selected_support_pixels": int(np.count_nonzero(raw_support)),
        "roi_pixels_xyxy_exclusive": [pixel_x0, pixel_y0, pixel_x1, pixel_y1],
        "roi_pixels": int(np.count_nonzero(roi_mask)),
        "post_roi_support_pixels": int(np.count_nonzero(bounded_support)),
        **component_masker,
        **endpoint_debug,
        "failure": None,
    }
    return polygon, endpoints, selected_support, {
        "arm": arm_name,
        "retained_width_fraction": fraction,
        "reference_full_polygon_pixels": full_polygon.tolist(),
        "reference_supervision_polygon_pixels": supervision_polygon.tolist(),
        "reference_positive_patches": int(np.count_nonzero(reference_positive)),
        "reference_full_patches": int(np.count_nonzero(reference_full)),
        "transfer": transfer,
        "masker": masker,
        "support_mask_uint8_sha256": hashlib.sha256(
            selected_support.astype(np.uint8).tobytes(order="C")
        ).hexdigest(),
    }


def configure_bounded_module() -> None:
    bounded.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    bounded.COHORT_SCHEMA = COHORT_SCHEMA
    bounded.RESULT_SCHEMA = RESULT_SCHEMA
    bounded.__file__ = str(Path(__file__).resolve())
    bounded.arm_prediction = spatial_arm_prediction


def freeze(protocol: Path, artifact_root: Path, output: Path) -> dict[str, Any]:
    configure_bounded_module()
    cohort = bounded.freeze(protocol, artifact_root, output)
    cohort["authority"] = "FROZEN_PRE_RGB_SPATIAL_REFERENCE_MASK_CANARY"
    cohort["episodes"][0]["episode_id"] = "SRM01"
    bounded.atomic_write_json(output, cohort)
    return cohort


def replay(
    protocol: Path,
    cohort: Path,
    artifact_root: Path,
    output: Path,
    preview_dir: Path | None,
) -> dict[str, Any]:
    configure_bounded_module()
    result = bounded.replay(protocol, cohort, artifact_root, output, preview_dir)
    passed = all(bool(value) for value in result["decision_conditions"].values())
    result["experiment"] = "L10 3RScan Spatially Coherent Bounded Reference Mask"
    result["status"] = (
        "L10_3RSCAN_SPATIAL_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_MET"
        if passed
        else "L10_3RSCAN_SPATIAL_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_NOT_MET"
    )
    result["execution_note"] = {
        "replay_attempts": 1,
        "protocol_or_decoder_tuning_after_rgb": False,
    }
    bounded.atomic_write_json(output, result)
    return result


def self_test() -> dict[str, Any]:
    active = np.zeros((5, 7), dtype=bool)
    active[1:3, 1:3] = True
    active[3:5, 5:7] = True
    margin = np.zeros((5, 7), dtype=np.float32)
    margin[active] = 0.2
    seeds = np.zeros((5, 7), dtype=bool)
    seeds[3, 5] = True
    component, debug = select_spatial_component(active, margin, seeds, 3)
    bounded.require(component is not None and bool(component[3, 5]), "SELF_COMPONENT")
    return {"status": "SELF_TEST_PASS", "component": debug}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_spatial_reference_mask_protocol_v1.json",
    )
    freeze_parser.add_argument("--artifact-root", type=Path, required=True)
    freeze_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_spatial_reference_mask_cohort_v1.json",
    )
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_spatial_reference_mask_protocol_v1.json",
    )
    replay_parser.add_argument(
        "--cohort",
        type=Path,
        default=HERE / "l10_3rscan_spatial_reference_mask_cohort_v1.json",
    )
    replay_parser.add_argument("--artifact-root", type=Path, required=True)
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_spatial_reference_mask_result_v1.json",
    )
    replay_parser.add_argument("--preview-dir", type=Path)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.protocol.resolve(), args.artifact_root.resolve(), args.output.resolve())
    elif args.command == "replay":
        result = replay(
            args.protocol.resolve(),
            args.cohort.resolve(),
            args.artifact_root.resolve(),
            args.output.resolve(),
            args.preview_dir.resolve() if args.preview_dir else None,
        )
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
