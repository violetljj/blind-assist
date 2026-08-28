"""Spatially transport public-reference identity evidence to entrance proposals.

The field is intentionally non-OCR. Reciprocal DINO patch matches that agree
with one affine layout vote for target-specific image locations. Entrance
proposals may bind only when their attachment region intersects that support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence

import cv2
import numpy as np


class SupportBindingState(str, Enum):
    SEARCH = "SEARCH"
    SET_VALUED = "SET_VALUED"
    COMMIT = "COMMIT"


@dataclass(frozen=True)
class SupportFieldConfig:
    minimum_mutual_matches: int = 4
    ransac_reprojection_threshold: float = 0.12
    minimum_entrance_score: float = 0.10
    minimum_support_score: float = 0.10
    commit_edge_margin: float = 0.08
    set_edge_margin: float = 0.03
    maximum_set_size: int = 3


@dataclass(frozen=True)
class SupportProposal:
    proposal_id: str
    entrance_score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class SupportEdge:
    proposal_id: str
    entrance_score: float
    support_score: float
    support_mass_fraction: float
    support_peak: float
    support_patch_count: int
    edge_score: float
    attachment_box_xyxy: tuple[float, float, float, float]
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class SupportBindingDecision:
    state: SupportBindingState
    selected_proposal_id: str | None
    candidate_set: tuple[str, ...]
    reason: str
    field_active_patches: int
    edges: tuple[SupportEdge, ...]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _patch_coordinates(grid: int) -> np.ndarray:
    denominator = max(grid - 1, 1)
    return np.asarray(
        [[(index % grid) / denominator, (index // grid) / denominator] for index in range(grid * grid)],
        dtype=np.float32,
    )


def _reference_support(
    query_patches: np.ndarray,
    reference_patches: np.ndarray,
    grid: int,
    config: SupportFieldConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    query = _normalize(query_patches)
    reference = _normalize(reference_patches)
    similarity = query @ reference.T
    query_to_reference = similarity.argmax(axis=1)
    reference_to_query = similarity.argmax(axis=0)
    mutual = [
        (query_index, int(reference_index))
        for query_index, reference_index in enumerate(query_to_reference)
        if reference_to_query[int(reference_index)] == query_index
    ]
    support = np.zeros(grid * grid, dtype=np.float32)
    if len(mutual) < config.minimum_mutual_matches:
        return support, {"mutual_matches": len(mutual), "inliers": 0, "mean_similarity": 0.0}
    coordinates = _patch_coordinates(grid)
    source = np.asarray([coordinates[reference_index] for _, reference_index in mutual], dtype=np.float32)
    destination = np.asarray([coordinates[query_index] for query_index, _ in mutual], dtype=np.float32)
    _, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if inlier_mask is None:
        return support, {"mutual_matches": len(mutual), "inliers": 0, "mean_similarity": 0.0}
    inliers = [pair for pair, keep in zip(mutual, inlier_mask.reshape(-1), strict=True) if bool(keep)]
    for query_index, reference_index in inliers:
        support[query_index] = max(support[query_index], float(similarity[query_index, reference_index]))
    mean_similarity = float(np.mean([similarity[q, r] for q, r in inliers])) if inliers else 0.0
    return support, {
        "mutual_matches": len(mutual),
        "inliers": len(inliers),
        "mean_similarity": mean_similarity,
    }


def build_target_support_field(
    goal_target_id: str,
    query_patches: np.ndarray,
    reference_patches: Mapping[str, Sequence[np.ndarray]],
    grid: int,
    config: SupportFieldConfig = SupportFieldConfig(),
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a discriminative goal-support field on the query patch grid."""
    if goal_target_id not in reference_patches:
        raise ValueError("GOAL_TARGET_NOT_IN_REFERENCE_ROSTER")
    if len(reference_patches) < 2:
        raise ValueError("TARGET_ROSTER_REQUIRES_DISTRACTOR")
    if query_patches.shape[0] != grid * grid:
        raise ValueError("QUERY_PATCH_GRID_MISMATCH")
    per_target: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    for target_id, references in reference_patches.items():
        target_support = np.zeros(grid * grid, dtype=np.float32)
        target_diagnostics = []
        for reference in references:
            support, detail = _reference_support(query_patches, reference, grid, config)
            target_support = np.maximum(target_support, support)
            target_diagnostics.append(detail)
        per_target[target_id] = target_support
        diagnostics[target_id] = target_diagnostics
    distractor = np.max(
        np.stack([value for target_id, value in per_target.items() if target_id != goal_target_id]),
        axis=0,
    )
    goal = per_target[goal_target_id]
    field = np.maximum(0.0, goal - distractor) * np.maximum(0.0, goal)
    maximum = float(field.max())
    if maximum > 0.0:
        field = field / maximum
    return field.reshape(grid, grid), {
        "per_target": diagnostics,
        "goal_raw_active_patches": int(np.count_nonzero(goal > 0.0)),
        "discriminative_active_patches": int(np.count_nonzero(field > 0.0)),
        "discriminative_mass": float(field.sum()),
    }


def _attachment_box(
    box_xyxy: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    width, height = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - 0.25 * width),
        max(0.0, y1 - height),
        min(float(image_width), x2 + 0.25 * width),
        min(float(image_height), y2),
    )


def _harmonic(left: float, right: float) -> float:
    left, right = max(0.0, min(1.0, left)), max(0.0, min(1.0, right))
    return 0.0 if left <= 0.0 or right <= 0.0 else 2.0 * left * right / (left + right)


def bind_support_field_to_entrances(
    field: np.ndarray,
    proposals: Sequence[SupportProposal],
    image_width: int,
    image_height: int,
    config: SupportFieldConfig = SupportFieldConfig(),
) -> SupportBindingDecision:
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("SUPPORT_FIELD_MUST_BE_SQUARE")
    grid = field.shape[0]
    field = np.maximum(0.0, np.asarray(field, dtype=np.float32))
    total_mass = float(field.sum())
    active_patches = int(np.count_nonzero(field > 0.0))
    centers = np.asarray(
        [
            [((x + 0.5) / grid) * image_width, ((y + 0.5) / grid) * image_height]
            for y in range(grid)
            for x in range(grid)
        ],
        dtype=np.float32,
    )
    flat = field.reshape(-1)
    edges = []
    for proposal in proposals:
        attachment = _attachment_box(proposal.box_xyxy, image_width, image_height)
        inside = (
            (centers[:, 0] >= attachment[0])
            & (centers[:, 0] <= attachment[2])
            & (centers[:, 1] >= attachment[1])
            & (centers[:, 1] <= attachment[3])
        )
        local = flat[inside]
        local_mass = float(local.sum())
        mass_fraction = local_mass / total_mass if total_mass > 0.0 else 0.0
        peak = float(local.max()) if local.size else 0.0
        support_score = 0.60 * mass_fraction + 0.40 * peak
        edges.append(
            SupportEdge(
                proposal.proposal_id,
                float(proposal.entrance_score),
                support_score,
                mass_fraction,
                peak,
                int(np.count_nonzero(local > 0.0)),
                _harmonic(float(proposal.entrance_score), support_score),
                attachment,
                proposal.box_xyxy,
            )
        )
    ordered = tuple(sorted(edges, key=lambda edge: (-edge.edge_score, edge.proposal_id)))
    eligible = [
        edge
        for edge in ordered
        if edge.entrance_score >= config.minimum_entrance_score
        and edge.support_score >= config.minimum_support_score
        and edge.support_patch_count > 0
    ]
    if not eligible:
        return SupportBindingDecision(
            SupportBindingState.SEARCH,
            None,
            (),
            "NO_ENTRANCE_ATTACHED_TO_TARGET_SUPPORT",
            active_patches,
            ordered,
        )
    second = eligible[1].edge_score if len(eligible) > 1 else 0.0
    if eligible[0].edge_score - second >= config.commit_edge_margin:
        return SupportBindingDecision(
            SupportBindingState.COMMIT,
            eligible[0].proposal_id,
            (eligible[0].proposal_id,),
            "UNIQUE_TARGET_SUPPORT_ATTACHED_ENTRANCE",
            active_patches,
            ordered,
        )
    candidate_set = tuple(
        edge.proposal_id
        for edge in eligible
        if eligible[0].edge_score - edge.edge_score <= config.set_edge_margin
    )[: config.maximum_set_size]
    return SupportBindingDecision(
        SupportBindingState.SET_VALUED,
        None,
        candidate_set,
        "MULTIPLE_TARGET_SUPPORT_ATTACHED_ENTRANCES",
        active_patches,
        ordered,
    )


def self_test() -> dict[str, object]:
    field = np.zeros((3, 3), dtype=np.float32)
    field[1, 1] = 1.0
    correct = SupportProposal("target-door", 0.70, (35.0, 50.0, 65.0, 95.0))
    distractor = SupportProposal("generic-door", 0.95, (70.0, 50.0, 98.0, 95.0))
    decision = bind_support_field_to_entrances(field, [distractor, correct], 100, 100)
    if decision.state != SupportBindingState.COMMIT or decision.selected_proposal_id != "target-door":
        raise AssertionError("TARGET_SUPPORT_DID_NOT_REJECT_GENERIC_HIGH_SCORE_ENTRANCE")
    empty = bind_support_field_to_entrances(np.zeros((3, 3)), [distractor], 100, 100)
    if empty.state != SupportBindingState.SEARCH:
        raise AssertionError("EMPTY_TARGET_SUPPORT_DID_NOT_FAIL_CLOSED")
    twin = SupportProposal("target-door-right", 0.69, (38.0, 50.0, 68.0, 95.0))
    ambiguous = bind_support_field_to_entrances(field, [correct, twin], 100, 100)
    if ambiguous.state != SupportBindingState.SET_VALUED or len(ambiguous.candidate_set) != 2:
        raise AssertionError("SAME_SUPPORT_TWINS_WERE_NOT_PRESERVED")
    return {
        "schema": "l10-target-support-field-self-test-v1",
        "status": "PASS",
        "reject_generic": asdict(decision),
        "empty_support": asdict(empty),
        "same_support_twins": asdict(ambiguous),
    }
