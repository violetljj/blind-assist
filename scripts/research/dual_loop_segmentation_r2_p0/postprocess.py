"""Frozen component filter primitive shared by R2-P0 rehearsal and refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..dual_loop_segmentation_candidate_utility.component_metrics import connected_components


class PostprocessContractError(ValueError):
    """Raised when a candidate postprocess identity is incomplete."""


def load_postprocess(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostprocessContractError("postprocess config must be a JSON object")
    required = {
        "candidate_id",
        "minimum_component_area_pixels",
        "minimum_component_confidence_median",
        "minimum_component_margin_median",
        "minimum_component_bottom_fraction",
        "connectivity",
        "remove_yolo_overlap_before_components",
        "hazard_classes",
        "unknown_nonwalkable_excluded",
    }
    if required - value.keys():
        raise PostprocessContractError(
            f"postprocess config missing fields: {sorted(required - value.keys())}"
        )
    if int(value["minimum_component_area_pixels"]) <= 0:
        raise PostprocessContractError("minimum component area must be positive")
    for key in (
        "minimum_component_confidence_median",
        "minimum_component_margin_median",
        "minimum_component_bottom_fraction",
    ):
        if not 0.0 <= float(value[key]) <= 1.0:
            raise PostprocessContractError(f"{key} must be within [0,1]")
    if int(value["connectivity"]) != 8:
        raise PostprocessContractError("R2-P0 requires 8-connectivity")
    if value["remove_yolo_overlap_before_components"] is not True:
        raise PostprocessContractError("R2-P0 requires YOLO overlap removal before filtering")
    if value["hazard_classes"] != ["boundary_step_curb", "obstacle"]:
        raise PostprocessContractError("R2-P0 hazard classes are frozen")
    if value["unknown_nonwalkable_excluded"] is not True:
        raise PostprocessContractError("unknown_nonwalkable must remain excluded")
    return value


def filter_candidate_by_class(
    *,
    ids: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    detector_mask: np.ndarray,
    class_to_id: dict[str, int],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not (
        ids.shape == confidence.shape == margin.shape == detector_mask.shape == (256, 256)
    ):
        raise PostprocessContractError("postprocess tensors must all be 256x256")
    output: dict[str, np.ndarray] = {}
    for class_name in config["hazard_classes"]:
        raw = (ids == int(class_to_id[class_name])) & ~detector_mask
        kept = np.zeros_like(raw, dtype=bool)
        for component in connected_components(raw, connectivity=8):
            component_confidence = float(np.median(confidence[component.mask]))
            component_margin = float(np.median(margin[component.mask]))
            bottom_fraction = float(component.bbox[3] / 256.0)
            if (
                component.area >= int(config["minimum_component_area_pixels"])
                and component_confidence
                >= float(config["minimum_component_confidence_median"])
                and component_margin >= float(config["minimum_component_margin_median"])
                and bottom_fraction >= float(config["minimum_component_bottom_fraction"])
            ):
                kept |= component.mask
        output[class_name] = kept
    return output
