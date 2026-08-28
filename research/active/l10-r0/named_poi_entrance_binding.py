"""Candidate-conditioned target-to-entrance binding for the L10 named-POI route.

The reducer deliberately does not accept a scene-level identity boolean plus a
generic entrance boolean.  Each entrance proposal must carry its own context
feature and win the public target roster before it can be committed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class BindingState(str, Enum):
    SEARCH = "SEARCH"
    SET_VALUED = "SET_VALUED"
    COMMIT = "COMMIT"


@dataclass(frozen=True)
class PublicTargetReference:
    target_id: str
    clip_vectors: np.ndarray
    dino_vectors: np.ndarray


@dataclass(frozen=True)
class EntranceProposalContext:
    proposal_id: str
    entrance_score: float
    box_xyxy: tuple[float, float, float, float]
    clip_vector: np.ndarray
    dino_vector: np.ndarray


@dataclass(frozen=True)
class BindingConfig:
    minimum_entrance_score: float = 0.10
    minimum_identity_margin_ratio: float = 0.05
    commit_edge_margin: float = 0.08
    set_edge_margin: float = 0.03
    maximum_set_size: int = 3


@dataclass(frozen=True)
class CandidateEdge:
    proposal_id: str
    target_id: str
    target_rank: int
    entrance_score: float
    target_score: float
    identity_margin: float
    identity_margin_ratio: float
    edge_score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class BindingDecision:
    state: BindingState
    goal_target_id: str
    selected_proposal_id: str | None
    candidate_set: tuple[str, ...]
    reason: str
    edges: tuple[CandidateEdge, ...]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def expanded_context_box(
    box_xyxy: Sequence[float], image_width: int, image_height: int, scale: float = 1.8
) -> tuple[int, int, int, int]:
    """Expand a proposal around its center while remaining inside the image."""
    if image_width < 1 or image_height < 1 or scale < 1.0:
        raise ValueError("INVALID_CONTEXT_BOUNDS")
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if not (x2 > x1 and y2 > y1):
        raise ValueError("INVALID_PROPOSAL_BOX")
    center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    width, height = scale * (x2 - x1), scale * (y2 - y1)
    return (
        max(0, int(np.floor(center_x - 0.5 * width))),
        max(0, int(np.floor(center_y - 0.5 * height))),
        min(image_width, int(np.ceil(center_x + 0.5 * width))),
        min(image_height, int(np.ceil(center_y + 0.5 * height))),
    )


def _reference_score(
    proposal: EntranceProposalContext, reference: PublicTargetReference
) -> float:
    clip = _normalize(proposal.clip_vector.reshape(1, -1))[0]
    dino = _normalize(proposal.dino_vector.reshape(1, -1))[0]
    clip_refs = _normalize(reference.clip_vectors)
    dino_refs = _normalize(reference.dino_vectors)
    return 0.55 * float(np.max(clip_refs @ clip)) + 0.45 * float(np.max(dino_refs @ dino))


def _harmonic_edge(entrance_score: float, identity_margin_ratio: float) -> float:
    entrance = max(0.0, min(1.0, float(entrance_score)))
    identity = max(0.0, min(1.0, float(identity_margin_ratio)))
    if entrance <= 0.0 or identity <= 0.0:
        return 0.0
    return 2.0 * entrance * identity / (entrance + identity)


def bind_target_to_entrance(
    goal_target_id: str,
    proposals: Sequence[EntranceProposalContext],
    references: Sequence[PublicTargetReference],
    config: BindingConfig = BindingConfig(),
) -> BindingDecision:
    """Bind entrance candidates to one public target or fail closed.

    A proposal can support the goal only when its own expanded context ranks the
    goal first.  Global scene identity is intentionally absent from the API.
    """
    if len({reference.target_id for reference in references}) != len(references):
        raise ValueError("DUPLICATE_TARGET_REFERENCE")
    target_ids = [reference.target_id for reference in references]
    if goal_target_id not in target_ids:
        raise ValueError("GOAL_TARGET_NOT_IN_PUBLIC_ROSTER")
    if len(references) < 2:
        raise ValueError("TARGET_ROSTER_REQUIRES_DISTRACTOR")
    goal_index = target_ids.index(goal_target_id)
    edges = []
    for proposal in proposals:
        if not 0.0 <= proposal.entrance_score <= 1.0:
            raise ValueError(f"INVALID_ENTRANCE_SCORE:{proposal.proposal_id}")
        scores = np.asarray(
            [_reference_score(proposal, reference) for reference in references], dtype=np.float32
        )
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == goal_index)[0]) + 1
        other_scores = np.delete(scores, goal_index)
        margin = float(scores[goal_index] - np.max(other_scores))
        spread = float(np.max(scores) - np.min(scores))
        margin_ratio = margin / max(spread, 1e-6)
        edge_score = _harmonic_edge(proposal.entrance_score, margin_ratio)
        edges.append(
            CandidateEdge(
                proposal_id=proposal.proposal_id,
                target_id=goal_target_id,
                target_rank=rank,
                entrance_score=float(proposal.entrance_score),
                target_score=float(scores[goal_index]),
                identity_margin=margin,
                identity_margin_ratio=margin_ratio,
                edge_score=edge_score,
                box_xyxy=proposal.box_xyxy,
            )
        )
    eligible = sorted(
        (
            edge
            for edge in edges
            if edge.target_rank == 1
            and edge.entrance_score >= config.minimum_entrance_score
            and edge.identity_margin_ratio >= config.minimum_identity_margin_ratio
        ),
        key=lambda edge: (-edge.edge_score, edge.proposal_id),
    )
    all_edges = tuple(sorted(edges, key=lambda edge: (-edge.edge_score, edge.proposal_id)))
    if not eligible:
        return BindingDecision(
            BindingState.SEARCH,
            goal_target_id,
            None,
            (),
            "NO_PROPOSAL_WITH_TARGET_LOCAL_IDENTITY",
            all_edges,
        )
    second_score = eligible[1].edge_score if len(eligible) > 1 else 0.0
    if eligible[0].edge_score - second_score >= config.commit_edge_margin:
        return BindingDecision(
            BindingState.COMMIT,
            goal_target_id,
            eligible[0].proposal_id,
            (eligible[0].proposal_id,),
            "UNIQUE_TARGET_LOCAL_ENTRANCE_EDGE",
            all_edges,
        )
    candidate_set = tuple(
        edge.proposal_id
        for edge in eligible
        if eligible[0].edge_score - edge.edge_score <= config.set_edge_margin
    )[: config.maximum_set_size]
    return BindingDecision(
        BindingState.SET_VALUED,
        goal_target_id,
        None,
        candidate_set,
        "MULTIPLE_TARGET_LOCAL_ENTRANCE_EDGES",
        all_edges,
    )


def _self_test() -> dict[str, object]:
    target_a = PublicTargetReference(
        "target-a",
        np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
    )
    target_b = PublicTargetReference(
        "target-b",
        np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        np.asarray([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32),
    )
    correct = EntranceProposalContext(
        "door-a", 0.72, (10.0, 20.0, 40.0, 90.0), np.asarray([0.95, 0.05]), np.asarray([0.9, 0.1])
    )
    generic_distractor = EntranceProposalContext(
        "door-b", 0.93, (60.0, 15.0, 95.0, 90.0), np.asarray([0.05, 0.95]), np.asarray([0.1, 0.9])
    )
    decision = bind_target_to_entrance("target-a", [generic_distractor, correct], [target_a, target_b])
    if decision.state != BindingState.COMMIT or decision.selected_proposal_id != "door-a":
        raise AssertionError("TARGET_LOCAL_BINDING_DID_NOT_REJECT_GENERIC_HIGH_SCORE_DOOR")
    absent = bind_target_to_entrance("target-a", [generic_distractor], [target_a, target_b])
    if absent.state != BindingState.SEARCH:
        raise AssertionError("WRONG_TARGET_CONTEXT_DID_NOT_FAIL_CLOSED")
    twin_left = EntranceProposalContext(
        "door-a-left", 0.70, (5.0, 20.0, 35.0, 90.0), np.asarray([0.95, 0.05]), np.asarray([0.9, 0.1])
    )
    twin_right = EntranceProposalContext(
        "door-a-right", 0.69, (45.0, 20.0, 75.0, 90.0), np.asarray([0.94, 0.06]), np.asarray([0.89, 0.11])
    )
    ambiguous = bind_target_to_entrance("target-a", [twin_left, twin_right], [target_a, target_b])
    if ambiguous.state != BindingState.SET_VALUED or len(ambiguous.candidate_set) != 2:
        raise AssertionError("AMBIGUOUS_TARGET_ENTRANCES_WERE_NOT_PRESERVED")
    return {
        "schema": "l10-target-local-entrance-binding-self-test-v1",
        "status": "PASS",
        "generic_high_score_distractor": asdict(decision),
        "wrong_target_only": asdict(absent),
        "same_target_twins": asdict(ambiguous),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; model adapters call bind_target_to_entrance directly")
    import json

    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
