#!/usr/bin/env python3
"""Training-free RGB+bbox grouping helpers for the GRAIL-R1A probe."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _bbox_geometry(bbox: list[int]) -> tuple[float, float, float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    return x0, y0, x1, y1, width, height


def pair_affinity(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float | bool]:
    """Combine expanded-bbox contact with frozen expanded-context DINO similarity."""
    ax0, ay0, ax1, ay1, aw, ah = _bbox_geometry(first["bbox"])
    bx0, by0, bx1, by1, bw, bh = _bbox_geometry(second["bbox"])
    gap_x = max(0.0, ax0 - bx1, bx0 - ax1)
    gap_y = max(0.0, ay0 - by1, by0 - ay1)
    normalized_gap = math.hypot(gap_x / max((aw + bw) / 2.0, 1.0), gap_y / max((ah + bh) / 2.0, 1.0))
    expanded_contact = not (
        ax1 + aw / 2.0 < bx0 - bw / 2.0
        or bx1 + bw / 2.0 < ax0 - aw / 2.0
        or ay1 + ah / 2.0 < by0 - bh / 2.0
        or by1 + bh / 2.0 < ay0 - ah / 2.0
    )
    context_cosine = float(np.dot(
        np.asarray(first["embedding"], dtype=np.float32),
        np.asarray(second["embedding"], dtype=np.float32),
    ))
    linked = bool(
        first["object_type"] == second["object_type"]
        and expanded_contact
        and context_cosine >= 0.5
    )
    return {
        "normalized_gap": normalized_gap,
        "context_cosine": context_cosine,
        "expanded_contact": expanded_contact,
        "linked": linked,
    }


def predict_groups(candidates: list[dict[str, Any]]) -> list[int]:
    """Connected components from same-type local contact; never reads object/root IDs."""
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            if pair_affinity(candidates[first], candidates[second])["linked"]:
                union(first, second)
    roots = [find(index) for index in range(len(candidates))]
    canonical = {root: group for group, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots]


def _rank_bin(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) < 2 or max(values) - min(values) < 1e-6:
        return "SINGLE"
    fraction = (value - min(values)) / (max(values) - min(values))
    return labels[0] if fraction <= 1 / 3 else labels[1] if fraction <= 2 / 3 else labels[2]


def predicted_ordinals(candidates: list[dict[str, Any]], groups: list[int]) -> list[tuple[str, str]]:
    """Deterministically rank bbox centroids within a predicted same-type root group."""
    by_group: dict[tuple[int, str], list[int]] = {}
    for index, (candidate, group) in enumerate(zip(candidates, groups)):
        by_group.setdefault((group, candidate["object_type"]), []).append(index)
    result: list[tuple[str, str]] = []
    for index, (candidate, group) in enumerate(zip(candidates, groups)):
        members = by_group[(group, candidate["object_type"])]
        centers = [
            ((candidates[item]["bbox"][0] + candidates[item]["bbox"][2]) / 2.0,
             (candidates[item]["bbox"][1] + candidates[item]["bbox"][3]) / 2.0)
            for item in members
        ]
        bbox = candidate["bbox"]
        center_x, center_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        result.append((
            _rank_bin(center_x, [value[0] for value in centers], ("LEFT", "CENTER", "RIGHT")),
            _rank_bin(center_y, [value[1] for value in centers], ("TOP", "MIDDLE", "BOTTOM")),
        ))
    return result


def aligned_context_score(candidate_tokens: np.ndarray, reference_tokens: np.ndarray) -> float:
    candidate = np.asarray(candidate_tokens, dtype=np.float32)
    reference = np.asarray(reference_tokens, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError(f"token shape mismatch: {candidate.shape} != {reference.shape}")
    return float(np.mean(np.sum(candidate * reference, axis=1)))


def shifted_context_score(candidate_tokens: np.ndarray, reference_tokens: np.ndarray, radius: int = 2) -> float:
    candidate = np.asarray(candidate_tokens, dtype=np.float32)
    reference = np.asarray(reference_tokens, dtype=np.float32)
    side = int(round(math.sqrt(candidate.shape[0])))
    if candidate.shape != reference.shape or side * side != candidate.shape[0]:
        raise ValueError("shifted context score requires equal square token grids")
    candidate_grid = candidate.reshape(side, side, -1)
    reference_grid = reference.reshape(side, side, -1)
    best = -1.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ref_y = slice(max(0, -dy), min(side, side - dy))
            ref_x = slice(max(0, -dx), min(side, side - dx))
            can_y = slice(max(0, dy), min(side, side + dy))
            can_x = slice(max(0, dx), min(side, side + dx))
            score = float(np.mean(np.sum(reference_grid[ref_y, ref_x] * candidate_grid[can_y, can_x], axis=-1)))
            best = max(best, score)
    return best


def select_carrier(scores: list[float], candidate_indices: list[int], spatial_keys: list[Any]) -> int | None:
    if not candidate_indices:
        return None
    return max(candidate_indices, key=lambda index: (scores[index], spatial_keys[index]))


def select_by_predicted_ordinal(
    target_type: str,
    target_ordinal: tuple[str, str] | None,
    candidates: list[dict[str, Any]],
    ordinals: list[tuple[str, str]],
    appearance_scores: list[float],
    spatial_keys: list[Any],
) -> tuple[int | None, float, str]:
    if target_ordinal is None:
        return None, 0.0, "NO_TARGET_TYPE_CARRIER"
    exact = [
        index for index, candidate in enumerate(candidates)
        if candidate["object_type"] == target_type and ordinals[index] == target_ordinal
    ]
    if not exact:
        return None, 0.0, "NO_PREDICTED_ORDINAL_MATCH"
    selected = max(exact, key=lambda index: (appearance_scores[index], spatial_keys[index]))
    if len(exact) == 1:
        return selected, 1.0, "UNIQUE_PREDICTED_ORDINAL_MATCH"
    return selected, float(appearance_scores[selected]), "ORDINAL_COLLISION_APPEARANCE_TIEBREAK"
