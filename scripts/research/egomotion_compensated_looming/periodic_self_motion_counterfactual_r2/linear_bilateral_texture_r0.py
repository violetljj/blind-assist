"""Frozen QMS-R0 linear-RGB bilateral texture operator.

The operator is response-blind and accepts only an RGB image.  It is a
nonlinear, edge-preserving spatial filter; it is not a linear PSF operator and
therefore must never be represented as ``PSF_NONE``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


OPERATOR_ID = "QMS_R0_LINEAR_BILATERAL_TEXTURE"
OPERATOR_CLASS = "NONLINEAR_EDGE_PRESERVING_SPATIAL_FILTER"
COLOR_SPACE = "LINEAR_RGB_FLOAT32"
DIAMETER = 7
SIGMA_COLOR = 0.08
SIGMA_SPACE = 3.0
BORDER_TYPE = cv2.BORDER_REFLECT_101
BORDER_NAME = "BORDER_REFLECT_101"
PSF_NONE = False


def frozen_operator_identity() -> dict[str, Any]:
    """Return a new copy of the complete frozen operator identity."""

    return {
        "operator_id": "QMS_R0_LINEAR_BILATERAL_TEXTURE",
        "operator_class": "NONLINEAR_EDGE_PRESERVING_SPATIAL_FILTER",
        "color_space": "LINEAR_RGB_FLOAT32",
        "diameter": 7,
        "sigma_color": 0.08,
        "sigma_space": 3.0,
        "border_type": int(cv2.BORDER_REFLECT_101),
        "border_name": "BORDER_REFLECT_101",
        "psf_none": False,
    }


def assert_frozen_operator_identity() -> None:
    """Fail closed if any runtime parameter or semantic label has drifted."""

    actual = {
        "operator_id": OPERATOR_ID,
        "operator_class": OPERATOR_CLASS,
        "color_space": COLOR_SPACE,
        "diameter": DIAMETER,
        "sigma_color": SIGMA_COLOR,
        "sigma_space": SIGMA_SPACE,
        "border_type": int(BORDER_TYPE),
        "border_name": BORDER_NAME,
        "psf_none": PSF_NONE,
    }
    if actual != frozen_operator_identity():
        raise RuntimeError("QMS_R0_OPERATOR_IDENTITY_DRIFT")


def validate_operator_identity(candidate: dict[str, Any]) -> None:
    """Reject a receipt or caller identity that differs from QMS-R0."""

    if candidate != frozen_operator_identity():
        raise ValueError("QMS_R0_OPERATOR_IDENTITY_INVALID")


def srgb_u8_to_linear_rgb_float32(rgb: np.ndarray) -> np.ndarray:
    """Decode an HxWx3 sRGB uint8 image into linear RGB float32."""

    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.shape[0] == 0
        or rgb.shape[1] == 0
    ):
        raise ValueError("RGB_MUST_BE_NONEMPTY_HXWX3_UINT8")
    channel = rgb.astype(np.float32) / np.float32(255.0)
    linear = np.where(
        channel <= np.float32(0.04045),
        channel / np.float32(12.92),
        np.power(
            (channel + np.float32(0.055)) / np.float32(1.055),
            np.float32(2.4),
        ),
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(linear)):
        raise ValueError("NONFINITE_LINEAR_RGB")
    return np.ascontiguousarray(linear)


def linear_rgb_float32_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    """Encode an HxWx3 linear RGB float32 image into rounded sRGB uint8."""

    if (
        not isinstance(linear, np.ndarray)
        or linear.dtype != np.float32
        or linear.ndim != 3
        or linear.shape[2] != 3
        or linear.shape[0] == 0
        or linear.shape[1] == 0
    ):
        raise ValueError("LINEAR_RGB_MUST_BE_NONEMPTY_HXWX3_FLOAT32")
    if not np.all(np.isfinite(linear)):
        raise ValueError("NONFINITE_LINEAR_RGB")
    clipped = np.clip(linear, np.float32(0.0), np.float32(1.0))
    srgb = np.where(
        clipped <= np.float32(0.0031308),
        np.float32(12.92) * clipped,
        np.float32(1.055)
        * np.power(clipped, np.float32(1.0 / 2.4))
        - np.float32(0.055),
    )
    return np.rint(
        np.clip(srgb, np.float32(0.0), np.float32(1.0))
        * np.float32(255.0)
    ).astype(np.uint8)


def apply_linear_bilateral_texture(rgb: np.ndarray) -> np.ndarray:
    """Apply the frozen QMS-R0 operator without mutating the input image."""

    assert_frozen_operator_identity()
    linear = srgb_u8_to_linear_rgb_float32(rgb)
    filtered = cv2.bilateralFilter(
        linear,
        d=DIAMETER,
        sigmaColor=SIGMA_COLOR,
        sigmaSpace=SIGMA_SPACE,
        borderType=BORDER_TYPE,
    )
    if filtered.dtype != np.float32 or filtered.shape != linear.shape:
        raise RuntimeError("QMS_R0_BILATERAL_OUTPUT_INVALID")
    return linear_rgb_float32_to_srgb_u8(filtered)
