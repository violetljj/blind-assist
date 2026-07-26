from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RotationCompensationResult:
    image: np.ndarray
    valid_mask: np.ndarray
    overlap_fraction: float


def compensate_current_to_previous(
    current_image: np.ndarray,
    current_valid_mask: np.ndarray,
    previous_valid_mask: np.ndarray,
    rotation_homography_previous_to_current: np.ndarray,
) -> RotationCompensationResult:
    if current_image.ndim != 2:
        raise ValueError("RCLE_ROTATION_WARP_REQUIRES_GRAYSCALE")
    height, width = current_image.shape
    inverse = np.linalg.inv(rotation_homography_previous_to_current)
    compensated = cv2.warpPerspective(
        current_image,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        current_valid_mask,
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    previous = previous_valid_mask > 0
    common = previous & (valid > 0)
    denominator = int(previous.sum())
    overlap = float(common.sum() / denominator) if denominator else 0.0
    return RotationCompensationResult(
        image=np.ascontiguousarray(compensated),
        valid_mask=np.ascontiguousarray(valid),
        overlap_fraction=overlap,
    )
