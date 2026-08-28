"""Bind entrance proposals below native-grid target-identity support rays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from named_poi_target_support_field import SupportProposal


class RayBindingState(str, Enum):
    SEARCH = "SEARCH"
    SET_VALUED = "SET_VALUED"
    COMMIT = "COMMIT"


@dataclass(frozen=True)
class SupportRayConfig:
    minimum_entrance_score: float = 0.10
    minimum_ray_score: float = 0.10
    commit_edge_margin: float = 0.08
    set_edge_margin: float = 0.03
    maximum_set_size: int = 3


@dataclass(frozen=True)
class RayEdge:
    proposal_id: str
    entrance_score: float
    ray_score: float
    ray_mass_fraction: float
    ray_peak: float
    ray_patch_count: int
    edge_score: float
    ray_column_xyxy: tuple[float, float, float, float]
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RayBindingDecision:
    state: RayBindingState
    selected_proposal_id: str | None
    candidate_set: tuple[str, ...]
    reason: str
    field_active_patches: int
    edges: tuple[RayEdge, ...]


def _harmonic(left: float, right: float) -> float:
    left, right = max(0.0, min(1.0, left)), max(0.0, min(1.0, right))
    return 0.0 if left <= 0.0 or right <= 0.0 else 2.0 * left * right / (left + right)


def bind_support_rays_to_entrances(
    field: np.ndarray,
    proposals: Sequence[SupportProposal],
    image_width: int,
    image_height: int,
    config: SupportRayConfig = SupportRayConfig(),
) -> RayBindingDecision:
    """Attach a door to identity support in the same image column above it."""
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("SUPPORT_FIELD_MUST_BE_SQUARE")
    grid = int(field.shape[0])
    field = np.maximum(0.0, np.asarray(field, dtype=np.float32))
    flat = field.reshape(-1)
    total_mass = float(flat.sum())
    active = int(np.count_nonzero(flat > 0.0))
    centers = np.asarray(
        [
            [((x + 0.5) / grid) * image_width, ((y + 0.5) / grid) * image_height]
            for y in range(grid)
            for x in range(grid)
        ],
        dtype=np.float32,
    )
    edges = []
    for proposal in proposals:
        x1, y1, x2, y2 = proposal.box_xyxy
        width = x2 - x1
        column = (
            max(0.0, x1 - 0.25 * width),
            0.0,
            min(float(image_width), x2 + 0.25 * width),
            min(float(image_height), y2),
        )
        eligible = (
            (centers[:, 0] >= column[0])
            & (centers[:, 0] <= column[2])
            & (centers[:, 1] <= column[3])
        )
        vertical_gap = np.maximum(0.0, y1 - centers[:, 1]) / max(float(image_height), 1.0)
        proximity = np.maximum(0.0, 1.0 - vertical_gap)
        contributions = flat * proximity * eligible.astype(np.float32)
        mass = float(contributions.sum())
        mass_fraction = mass / total_mass if total_mass > 0.0 else 0.0
        peak = float(contributions.max()) if contributions.size else 0.0
        ray_score = 0.60 * mass_fraction + 0.40 * peak
        edges.append(
            RayEdge(
                proposal.proposal_id,
                float(proposal.entrance_score),
                ray_score,
                mass_fraction,
                peak,
                int(np.count_nonzero(contributions > 0.0)),
                _harmonic(float(proposal.entrance_score), ray_score),
                column,
                proposal.box_xyxy,
            )
        )
    ordered = tuple(sorted(edges, key=lambda edge: (-edge.edge_score, edge.proposal_id)))
    eligible_edges = [
        edge
        for edge in ordered
        if edge.entrance_score >= config.minimum_entrance_score
        and edge.ray_score >= config.minimum_ray_score
        and edge.ray_patch_count > 0
    ]
    if not eligible_edges:
        return RayBindingDecision(
            RayBindingState.SEARCH,
            None,
            (),
            "NO_ENTRANCE_BELOW_NATIVE_TARGET_SUPPORT",
            active,
            ordered,
        )
    second = eligible_edges[1].edge_score if len(eligible_edges) > 1 else 0.0
    if eligible_edges[0].edge_score - second >= config.commit_edge_margin:
        return RayBindingDecision(
            RayBindingState.COMMIT,
            eligible_edges[0].proposal_id,
            (eligible_edges[0].proposal_id,),
            "UNIQUE_ENTRANCE_BELOW_NATIVE_TARGET_SUPPORT",
            active,
            ordered,
        )
    candidate_set = tuple(
        edge.proposal_id
        for edge in eligible_edges
        if eligible_edges[0].edge_score - edge.edge_score <= config.set_edge_margin
    )[: config.maximum_set_size]
    return RayBindingDecision(
        RayBindingState.SET_VALUED,
        None,
        candidate_set,
        "MULTIPLE_ENTRANCES_BELOW_NATIVE_TARGET_SUPPORT",
        active,
        ordered,
    )


def self_test() -> dict[str, object]:
    field = np.zeros((8, 8), dtype=np.float32)
    field[1:4, 2:4] = 1.0
    correct = SupportProposal("target-door", 0.70, (22.0, 60.0, 48.0, 95.0))
    distractor = SupportProposal("generic-door", 0.95, (68.0, 60.0, 98.0, 95.0))
    decision = bind_support_rays_to_entrances(field, [distractor, correct], 100, 100)
    if decision.state != RayBindingState.COMMIT or decision.selected_proposal_id != "target-door":
        raise AssertionError("NATIVE_SUPPORT_RAY_DID_NOT_REJECT_GENERIC_ENTRANCE")
    empty = bind_support_rays_to_entrances(np.zeros((8, 8)), [distractor], 100, 100)
    if empty.state != RayBindingState.SEARCH:
        raise AssertionError("EMPTY_NATIVE_SUPPORT_DID_NOT_FAIL_CLOSED")
    twin = SupportProposal("target-door-right", 0.69, (24.0, 60.0, 50.0, 95.0))
    ambiguous = bind_support_rays_to_entrances(field, [correct, twin], 100, 100)
    if ambiguous.state != RayBindingState.SET_VALUED or len(ambiguous.candidate_set) != 2:
        raise AssertionError("SAME_SUPPORT_RAY_TWINS_WERE_NOT_PRESERVED")
    return {
        "schema": "l10-native-target-support-ray-self-test-v1",
        "status": "PASS",
        "reject_generic": asdict(decision),
        "empty_support": asdict(empty),
        "same_support_twins": asdict(ambiguous),
    }
