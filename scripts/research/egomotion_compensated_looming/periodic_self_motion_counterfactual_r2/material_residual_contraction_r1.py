"""QMS-R1 paired material-residual contraction renderer.

Clean and low-texture RGB are derived from one P1 geometry raycast.  This
module is generator-side and response-blind; it does not import any P4 or RCLE
estimator implementation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from . import generator_geometry as p1


OPERATOR_ID = "QMS_R1_MATERIAL_RESIDUAL_CONTRACTION"
CLEAN_MODULATION_LOW = 0.65
CLEAN_MODULATION_RANGE = 0.35
MATERIAL_MEAN_MODULATION = 0.825
ALPHA = 0.15
PSF_NONE = True


def frozen_operator_identity() -> dict[str, Any]:
    """Return a fresh copy of the frozen QMS-R1 identity."""

    return {
        "operator_id": "QMS_R1_MATERIAL_RESIDUAL_CONTRACTION",
        "clean_modulation_formula": "0.65+0.35*checker",
        "material_mean_modulation": 0.825,
        "low_modulation_formula": "0.825+0.15*(clean-0.825)",
        "alpha": 0.15,
        "pairing": "ONE_RAYCAST_SHARED_GEOMETRY",
        "domain": "PREQUANTIZATION_LINEAR_RGB",
        "psf_none": True,
    }


def assert_frozen_operator_identity() -> None:
    """Fail closed if a runtime parameter or semantic label has drifted."""

    actual = {
        "operator_id": OPERATOR_ID,
        "clean_modulation_formula": (
            f"{CLEAN_MODULATION_LOW:g}+"
            f"{CLEAN_MODULATION_RANGE:g}*checker"
        ),
        "material_mean_modulation": MATERIAL_MEAN_MODULATION,
        "low_modulation_formula": (
            f"{MATERIAL_MEAN_MODULATION:g}+{ALPHA:g}*"
            f"(clean-{MATERIAL_MEAN_MODULATION:g})"
        ),
        "alpha": ALPHA,
        "pairing": "ONE_RAYCAST_SHARED_GEOMETRY",
        "domain": "PREQUANTIZATION_LINEAR_RGB",
        "psf_none": PSF_NONE,
    }
    if actual != frozen_operator_identity():
        raise RuntimeError("QMS_R1_OPERATOR_IDENTITY_DRIFT")


def validate_operator_identity(candidate: dict[str, Any]) -> None:
    """Reject any external identity that differs from the frozen operator."""

    if candidate != frozen_operator_identity():
        raise ValueError("QMS_R1_OPERATOR_IDENTITY_INVALID")


def _sha256_array(array: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(linear)):
        raise ValueError("NONFINITE_LINEAR_RGB")
    clipped = np.clip(linear.astype(np.float64, copy=False), 0.0, 1.0)
    srgb = np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_pair(
    scene: dict[str, Any],
    rotation_world_from_camera: np.ndarray,
    translation_world_from_camera: np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """Render frozen clean/low RGB from one shared P1 raycast."""

    assert_frozen_operator_identity()
    strength = float(alpha)
    if not math.isfinite(strength) or strength != ALPHA:
        raise ValueError("QMS_R1_ALPHA_MUST_EQUAL_FROZEN_0_15")
    rotation = np.asarray(rotation_world_from_camera, dtype=np.float64)
    translation = np.asarray(
        translation_world_from_camera,
        dtype=np.float64,
    )
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("ROTATION_MUST_BE_FINITE_3X3")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("TRANSLATION_MUST_BE_FINITE_LENGTH_3")

    u, v = np.meshgrid(
        np.arange(p1.WIDTH, dtype=np.float64),
        np.arange(p1.HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    depth, object_id, world = p1._raycast(
        scene,
        rotation,
        translation,
        uv,
    )
    valid = np.isfinite(depth)
    clean_linear = np.zeros((len(uv), 3), dtype=np.float64)
    low_linear = np.zeros((len(uv), 3), dtype=np.float64)
    material_mean_linear = np.zeros((len(uv), 3), dtype=np.float64)
    by_id = {
        int(obj["object_id"]): obj for obj in scene["world"]["objects"]
    }
    for identifier in np.unique(object_id[valid]):
        selected = object_id == identifier
        obj = by_id.get(int(identifier))
        if obj is None:
            raise ValueError("RAYCAST_OBJECT_ID_MISSING_FROM_SCENE")
        base = np.asarray(obj["linear_rgb"], dtype=np.float64)
        if (
            base.shape != (3,)
            or not np.all(np.isfinite(base))
            or np.any(base < 0.0)
            or np.any(base > 1.0)
        ):
            raise ValueError("MATERIAL_LINEAR_RGB_INVALID")
        texture = obj["texture"]
        frequency = float(texture["cycles_per_m"])
        phase = float(texture["phase"])
        if not math.isfinite(frequency) or not math.isfinite(phase):
            raise ValueError("MATERIAL_TEXTURE_INVALID")
        checker = (
            np.floor((world[selected, 0] * frequency + phase) % 2.0)
            + np.floor((world[selected, 1] * frequency + phase) % 2.0)
        ) % 2.0
        clean_modulation = (
            CLEAN_MODULATION_LOW + CLEAN_MODULATION_RANGE * checker
        )
        material_mean_linear[selected] = (
            base[None, :] * MATERIAL_MEAN_MODULATION
        )
        clean_linear[selected] = np.clip(
            base[None, :] * clean_modulation[:, None],
            0.0,
            1.0,
        )
        low_linear[selected] = np.clip(
            material_mean_linear[selected]
            + strength
            * (
                clean_linear[selected]
                - material_mean_linear[selected]
            ),
            0.0,
            1.0,
        )

    expected_low = material_mean_linear + strength * (
        clean_linear - material_mean_linear
    )
    residual_error = float(
        np.max(np.abs(low_linear[valid] - expected_low[valid]))
    ) if np.any(valid) else 0.0
    if residual_error != 0.0:
        raise RuntimeError("QMS_R1_PREQUANTIZATION_RESIDUAL_DRIFT")

    shape = (p1.HEIGHT, p1.WIDTH)
    clean_image = clean_linear.reshape(*shape, 3)
    low_image = low_linear.reshape(*shape, 3)
    mean_image = material_mean_linear.reshape(*shape, 3)
    valid_image = valid.reshape(shape)
    object_image = object_id.reshape(shape)
    scene_hash = scene.get("scene_geometry_sha256")
    return {
        "rgb_pair": {
            "clean": _linear_to_srgb_u8(clean_image),
            "low": _linear_to_srgb_u8(low_image),
        },
        "valid_mask": valid_image,
        "object_id": object_image,
        "geometry_identity": {
            "scene_geometry_sha256": scene_hash,
            "valid_mask_sha256": _sha256_array(valid_image, "|u1"),
            "object_id_sha256": _sha256_array(object_image, "<i4"),
        },
        "prequantization_identity": {
            **frozen_operator_identity(),
            "clean_linear_rgb_sha256": _sha256_array(
                clean_image,
                "<f8",
            ),
            "low_linear_rgb_sha256": _sha256_array(low_image, "<f8"),
            "material_mean_linear_rgb_sha256": _sha256_array(
                mean_image,
                "<f8",
            ),
            "residual_relation_max_abs_error": residual_error,
        },
    }
