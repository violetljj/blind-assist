#!/usr/bin/env python3
"""Frozen crop, rotation, and symmetry mechanics for GRAIL-R1C-P."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from grail_visual_orientation_r1cv import bbox_center, group_members, _rank_bin


def masked_group_crop(image: np.ndarray, candidates: list[dict[str, Any]], indices: list[int],
                      mask_root: Path, padding_fraction: float = 0.10) -> Image.Image:
    """Return the frozen union-mask crop with white background."""
    if not indices:
        raise ValueError("owner group is empty")
    union = np.zeros(image.shape[:2], dtype=bool)
    for index in indices:
        mask = np.asarray(Image.open(mask_root / candidates[index]["mask_image"]).convert("L")) > 0
        if mask.shape != union.shape:
            raise ValueError("proposal mask/image shape mismatch")
        union |= mask
    ys, xs = np.nonzero(union)
    if not len(xs):
        raise ValueError("owner group mask is empty")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad_x = int(round((x1 - x0) * padding_fraction))
    pad_y = int(round((y1 - y0) * padding_fraction))
    x0, x1 = max(0, x0 - pad_x), min(image.shape[1], x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(image.shape[0], y1 + pad_y)
    crop = image[y0:y1, x0:x1].copy()
    crop[~union[y0:y1, x0:x1]] = 255
    return Image.fromarray(crop)


def _axis_rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    value /= np.linalg.norm(value)
    x, y, z = value
    theta = math.radians(degrees)
    c, s, d = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return np.asarray([
        [c + x*x*d, x*y*d - z*s, x*z*d + y*s],
        [y*x*d + z*s, c + y*y*d, y*z*d - x*s],
        [z*x*d - y*s, z*y*d + x*s, c + z*z*d],
    ])


def orientation_matrix(azimuth: float, elevation: float, roll: float) -> np.ndarray:
    """Exact numpy equivalent of OA-V2 azi_ele_rot_to_Obj_Rmatrix_batch."""
    return _axis_rotation((1, 0, 0), roll) @ _axis_rotation((0, 1, 0), elevation) @ \
        _axis_rotation((0, 0, 1), -azimuth)


def projected_basis(matrix: np.ndarray, minimum_norm: float = 1e-6) -> dict[str, Any]:
    """Project OA-V2 canonical right(Y) and up(Z) into image coordinates."""
    # OA-V2 represents front in camera coordinates as (depth, image-right,
    # image-up); this is also the convention inverted by
    # Cam_Rmatrix_to_azi_ele_rot_batch in the official code.
    right = np.asarray([matrix[1, 1], -matrix[2, 1]], dtype=np.float64)
    up = np.asarray([matrix[1, 2], -matrix[2, 2]], dtype=np.float64)
    right_norm, up_norm = float(np.linalg.norm(right)), float(np.linalg.norm(up))
    if right_norm < minimum_norm or up_norm < minimum_norm:
        return {"evaluable": False, "right": None, "down": None}
    return {"evaluable": True, "right": right / right_norm, "down": -up / up_norm}


def reference_mode_matrices(azimuth: int, elevation: int, roll: int, alpha: int) -> list[np.ndarray]:
    if alpha not in (1, 2, 4):
        return []
    return [orientation_matrix((azimuth + mode * 360.0 / alpha) % 360.0, elevation, roll)
            for mode in range(alpha)]


def paired_mode_bases(ref_azimuth: int, ref_elevation: int, ref_roll: int, alpha: int,
                      rel_azimuth: int, rel_elevation: int, rel_roll: int) -> list[dict[str, Any]]:
    relative = orientation_matrix(rel_azimuth, rel_elevation, rel_roll)
    result = []
    for reference in reference_mode_matrices(ref_azimuth, ref_elevation, ref_roll, alpha):
        result.append({"reference": projected_basis(reference), "query": projected_basis(relative @ reference)})
    return result


def ordinals_from_basis(candidates: list[dict[str, Any]], groups: list[int],
                        bases: dict[tuple[int, str], dict[str, Any]]) -> list[tuple[str, str]]:
    centers = [bbox_center(candidate) for candidate in candidates]
    result = [("NOT_EVALUABLE", "NOT_EVALUABLE") for _ in candidates]
    for key, indices in group_members(candidates, groups).items():
        basis = bases[key]
        if not basis["evaluable"]:
            for index in indices:
                result[index] = ("SINGLE" if len(indices) < 2 else "NOT_EVALUABLE", "NOT_EVALUABLE")
            continue
        horizontal_values = [float(np.dot(centers[index], basis["right"])) for index in indices]
        vertical_values = [float(np.dot(centers[index], basis["down"])) for index in indices]
        for index in indices:
            result[index] = (
                _rank_bin(float(np.dot(centers[index], basis["right"])), horizontal_values,
                          ("LEFT", "CENTER", "RIGHT")),
                _rank_bin(float(np.dot(centers[index], basis["down"])), vertical_values,
                          ("TOP", "MIDDLE", "BOTTOM")),
            )
    return result


def consensus_index(indices: list[int | None]) -> int | None:
    """Retain a selection only when every symmetry mode agrees."""
    return indices[0] if indices and indices[0] is not None and all(value == indices[0] for value in indices) else None
