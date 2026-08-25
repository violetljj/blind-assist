#!/usr/bin/env python3
"""Deterministic RGB/proposal owner-orientation estimator for GRAIL-R1C-V."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from grail_grouping_r1a import _rank_bin


SIGN_THRESHOLD = 0.05
SUPPORT_PADDING_FRACTION = 0.10


def bbox_center(candidate: dict[str, Any]) -> np.ndarray:
    x0, y0, x1, y1 = candidate["bbox"]
    return np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)


def group_members(candidates: list[dict[str, Any]], groups: list[int]) -> dict[tuple[int, str], list[int]]:
    members: dict[tuple[int, str], list[int]] = {}
    for index, (candidate, group) in enumerate(zip(candidates, groups)):
        members.setdefault((group, candidate["object_type"]), []).append(index)
    return members


def support_box(candidates: list[dict[str, Any]], indices: list[int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x0 = min(float(candidates[index]["bbox"][0]) for index in indices)
    y0 = min(float(candidates[index]["bbox"][1]) for index in indices)
    x1 = max(float(candidates[index]["bbox"][2]) for index in indices)
    y1 = max(float(candidates[index]["bbox"][3]) for index in indices)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    pad_x, pad_y = width * SUPPORT_PADDING_FRACTION, height * SUPPORT_PADDING_FRACTION
    image_height, image_width = image_shape[:2]
    return (
        max(0, int(math.floor(x0 - pad_x))),
        max(0, int(math.floor(y0 - pad_y))),
        min(image_width, int(math.ceil(x1 + pad_x))),
        min(image_height, int(math.ceil(y1 + pad_y))),
    )


def canonicalize_undirected_axis(axis: np.ndarray) -> np.ndarray:
    normalized = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(normalized))
    if norm < 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    normalized = normalized / norm
    if normalized[0] < 0.0 or (abs(normalized[0]) < 1e-12 and normalized[1] < 0.0):
        normalized = -normalized
    return normalized


def estimate_undirected_axis(centers: list[np.ndarray]) -> tuple[np.ndarray, str]:
    if len(centers) < 2:
        return np.asarray([1.0, 0.0], dtype=np.float64), "HORIZONTAL_FALLBACK"
    matrix = np.stack(centers)
    centered = matrix - matrix.mean(axis=0)
    if float(np.linalg.norm(centered)) < 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64), "HORIZONTAL_FALLBACK"
    covariance = centered.T @ centered / len(centers)
    _, eigenvectors = np.linalg.eigh(covariance)
    options = [canonicalize_undirected_axis(eigenvectors[:, index]) for index in range(2)]
    axis = max(options, key=lambda value: (abs(float(value[0])), -abs(float(value[1]))))
    return axis, "CENTER_PCA_HORIZONTAL_COMPONENT"


def estimate_sign(image: np.ndarray, box: tuple[int, int, int, int], axis: np.ndarray) -> dict[str, Any]:
    axis = canonicalize_undirected_axis(axis)
    x0, y0, x1, y1 = box
    crop = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if crop.size == 0:
        return {"evaluable": False, "moment": 0.0, "directed_axis": None}
    luminance = (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]) / 255.0
    gradient_x = np.zeros_like(luminance)
    gradient_y = np.zeros_like(luminance)
    if luminance.shape[1] > 2:
        gradient_x[:, 1:-1] = (luminance[:, 2:] - luminance[:, :-2]) / 2.0
    if luminance.shape[0] > 2:
        gradient_y[1:-1, :] = (luminance[2:, :] - luminance[:-2, :]) / 2.0
    weights = np.hypot(gradient_x, gradient_y)
    total_weight = float(weights.sum())
    if total_weight < 1e-12:
        return {"evaluable": False, "moment": 0.0, "directed_axis": None}
    yy, xx = np.indices(luminance.shape, dtype=np.float64)
    xx += x0 + 0.5
    yy += y0 + 0.5
    center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)
    projection = (xx - center[0]) * axis[0] + (yy - center[1]) * axis[1]
    corners = np.asarray([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=np.float64)
    scale = max(float(np.max(np.abs((corners - center) @ axis))), 1.0)
    moment = float(np.sum(weights * projection / scale) / total_weight)
    if abs(moment) < SIGN_THRESHOLD:
        return {"evaluable": False, "moment": moment, "directed_axis": None}
    directed = axis if moment > 0.0 else -axis
    return {"evaluable": True, "moment": moment, "directed_axis": directed}


def predict_visual_frames(image: np.ndarray, candidates: list[dict[str, Any]], groups: list[int]) -> dict[tuple[int, str], dict[str, Any]]:
    """Prediction-only function. It has no native/evaluator input by construction."""
    frames: dict[tuple[int, str], dict[str, Any]] = {}
    for key, indices in group_members(candidates, groups).items():
        centers = [bbox_center(candidates[index]) for index in indices]
        axis, source = estimate_undirected_axis(centers)
        box = support_box(candidates, indices, image.shape)
        sign = estimate_sign(image, box, axis)
        projected = [float(np.dot(center, axis)) for center in centers]
        frames[key] = {
            "indices": indices,
            "support_box": box,
            "undirected_axis": axis,
            "axis_source": source,
            "sign_evaluable": sign["evaluable"],
            "sign_moment": sign["moment"],
            "directed_axis": sign["directed_axis"],
            "horizontal_single": len(projected) < 2 or max(projected) - min(projected) < 1e-6,
        }
    return frames


def oracle_directed_frames(candidates: list[dict[str, Any]], groups: list[int],
                           native_coordinates: dict[str, dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    """Evaluator-only projection fit from native local-right to image centers."""
    frames: dict[tuple[int, str], dict[str, Any]] = {}
    for key, indices in group_members(candidates, groups).items():
        centers, rights = [], []
        for index in indices:
            coordinate = native_coordinates[candidates[index]["object_id"]]
            if coordinate.get("evaluable"):
                centers.append(bbox_center(candidates[index]))
                rights.append(float(coordinate["local_right"]))
        if len(centers) < 2 or max(rights, default=0.0) - min(rights, default=0.0) < 1e-6:
            frames[key] = {"evaluable": False, "directed_axis": None, "horizontal_single": True}
            continue
        centered_right = np.asarray(rights) - np.mean(rights)
        centered_pixels = np.stack(centers) - np.mean(np.stack(centers), axis=0)
        direction = np.sum(centered_pixels * centered_right[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            frames[key] = {"evaluable": False, "directed_axis": None, "horizontal_single": False}
        else:
            frames[key] = {"evaluable": True, "directed_axis": direction / norm, "horizontal_single": False}
    return frames


def arm_frames(image: np.ndarray, predicted: dict[tuple[int, str], dict[str, Any]],
               oracle: dict[tuple[int, str], dict[str, Any]], arm: str) -> dict[tuple[int, str], dict[str, Any]]:
    output: dict[tuple[int, str], dict[str, Any]] = {}
    for key, frame in predicted.items():
        oracle_frame = oracle[key]
        if arm == "R1C_V_FINAL":
            output[key] = {"directed_axis": frame["directed_axis"], "evaluable": frame["sign_evaluable"],
                           "horizontal_single": frame["horizontal_single"]}
        elif arm == "AXIS_ONLY_DIAGNOSTIC":
            if not oracle_frame["evaluable"]:
                output[key] = {"directed_axis": None, "evaluable": False,
                               "horizontal_single": oracle_frame["horizontal_single"]}
            else:
                estimated = frame["undirected_axis"]
                oracle_axis = oracle_frame["directed_axis"]
                output[key] = {
                    "directed_axis": estimated if float(np.dot(estimated, oracle_axis)) >= 0.0 else -estimated,
                    "evaluable": True,
                    "horizontal_single": False,
                }
        elif arm == "SIGN_ONLY_DIAGNOSTIC":
            if not oracle_frame["evaluable"]:
                output[key] = {"directed_axis": None, "evaluable": False,
                               "horizontal_single": oracle_frame["horizontal_single"]}
            else:
                box = frame["support_box"]
                sign = estimate_sign(image, box, oracle_frame["directed_axis"])
                output[key] = {"directed_axis": sign["directed_axis"], "evaluable": sign["evaluable"],
                               "horizontal_single": False}
        else:
            raise ValueError(f"unknown R1C-V arm: {arm}")
    return output


def ordinals_from_frames(candidates: list[dict[str, Any]], groups: list[int],
                         frames: dict[tuple[int, str], dict[str, Any]]) -> list[tuple[str, str]]:
    centers = [bbox_center(candidate) for candidate in candidates]
    result: list[tuple[str, str]] = [("NOT_EVALUABLE", "NOT_EVALUABLE") for _ in candidates]
    for key, indices in group_members(candidates, groups).items():
        frame = frames[key]
        if not frame["evaluable"]:
            horizontal = "SINGLE" if frame.get("horizontal_single", len(indices) < 2) else "NOT_EVALUABLE"
            for index in indices:
                vertical_values = [float(centers[item][1]) for item in indices]
                result[index] = (
                    horizontal,
                    _rank_bin(float(centers[index][1]), vertical_values, ("TOP", "MIDDLE", "BOTTOM")),
                )
            continue
        axis = frame["directed_axis"]
        horizontal_values = [float(np.dot(centers[item], axis)) for item in indices]
        vertical_values = [float(centers[item][1]) for item in indices]
        for index in indices:
            result[index] = (
                _rank_bin(float(np.dot(centers[index], axis)), horizontal_values, ("LEFT", "CENTER", "RIGHT")),
                _rank_bin(float(centers[index][1]), vertical_values, ("TOP", "MIDDLE", "BOTTOM")),
            )
    return result


def undirected_angle_degrees(estimated: np.ndarray, oracle: np.ndarray) -> float:
    cosine = min(1.0, max(0.0, abs(float(np.dot(estimated, oracle)))))
    return math.degrees(math.acos(cosine))
