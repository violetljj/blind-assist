from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


ROW_ALIGNMENT_TOLERANCE_M = 0.08


class FunctionalBindingState(str, Enum):
    NOT_EVALUABLE = "NOT_EVALUABLE"
    AMBIGUOUS = "AMBIGUOUS"
    SET_VALUED = "SET_VALUED"
    UNIQUE = "UNIQUE"


@dataclass(frozen=True)
class FunctionalPartCandidate:
    """A provider proposal bound to an already-authorized parent instance."""

    candidate_id: str
    parent_binding_id: str
    center_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class FunctionalBindingDecision:
    state: FunctionalBindingState
    selected_candidate_ids: tuple[str, ...]
    action: str
    authority: str | None
    relation: str | None = None


def _row_clusters(
    candidates: Iterable[FunctionalPartCandidate],
) -> list[list[FunctionalPartCandidate]]:
    """Group roughly level parts, ordered from highest to lowest."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate.center_xyz_m[2], candidate.candidate_id),
    )
    rows: list[list[FunctionalPartCandidate]] = []
    for candidate in ordered:
        if not rows:
            rows.append([candidate])
            continue
        row_height = sum(item.center_xyz_m[2] for item in rows[-1]) / len(rows[-1])
        if abs(candidate.center_xyz_m[2] - row_height) <= ROW_ALIGNMENT_TOLERANCE_M:
            rows[-1].append(candidate)
        else:
            rows.append([candidate])
    return rows


def _relation(task_description: str) -> tuple[str | None, int | None]:
    words = set(task_description.casefold().replace("-", " ").split())
    if "top" in words or "upper" in words:
        return "TOP", 0
    if "second" in words:
        return "SECOND_FROM_TOP", 1
    if "bottom" in words or "lowest" in words:
        return "BOTTOM", -1
    if "left" in words:
        return "LEFT", None
    if "right" in words:
        return "RIGHT", None
    if "under" in words or "below" in words:
        return "UNDER", None
    if "above" in words or "over" in words:
        return "ABOVE", None
    return None, None


class TaskRelationalFunctionalSelector:
    """Select functional-part proposals without changing exact-instance authority.

    Candidate IDs are opaque. The controller uses only the public task text,
    parent binding, and proposal geometry. It never consumes affordance labels or
    evaluator target IDs.
    """

    def select(
        self,
        task_description: str,
        parent_binding_id: str,
        candidates: Iterable[FunctionalPartCandidate],
    ) -> FunctionalBindingDecision:
        eligible = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.parent_binding_id == parent_binding_id
            ),
            key=lambda candidate: candidate.candidate_id,
        )
        if not eligible:
            return FunctionalBindingDecision(
                FunctionalBindingState.NOT_EVALUABLE,
                (),
                "REQUEST_FUNCTIONAL_PROPOSALS",
                None,
            )

        relation, row_index = _relation(task_description)
        rows = _row_clusters(eligible)
        selected: list[FunctionalPartCandidate]
        if row_index is not None:
            try:
                selected = rows[row_index]
            except IndexError:
                return FunctionalBindingDecision(
                    FunctionalBindingState.NOT_EVALUABLE,
                    (),
                    "REQUEST_RELATIONAL_VIEW",
                    None,
                    relation,
                )
        elif relation == "LEFT" and len(eligible) > 1:
            min_x = min(candidate.center_xyz_m[0] for candidate in eligible)
            selected = [
                candidate
                for candidate in eligible
                if candidate.center_xyz_m[0] <= min_x + ROW_ALIGNMENT_TOLERANCE_M
            ]
        elif relation == "RIGHT" and len(eligible) > 1:
            max_x = max(candidate.center_xyz_m[0] for candidate in eligible)
            selected = [
                candidate
                for candidate in eligible
                if candidate.center_xyz_m[0] >= max_x - ROW_ALIGNMENT_TOLERANCE_M
            ]
        elif len(rows) == 1:
            selected = rows[0]
        elif len(eligible) == 1:
            selected = eligible
        else:
            return FunctionalBindingDecision(
                FunctionalBindingState.AMBIGUOUS,
                tuple(candidate.candidate_id for candidate in eligible),
                "REQUEST_TASK_RELATION_OR_VIEW",
                "INSTANCE_BOUND_GEOMETRY",
                relation,
            )

        selected_ids = tuple(sorted(candidate.candidate_id for candidate in selected))
        state = (
            FunctionalBindingState.UNIQUE
            if len(selected_ids) == 1
            else FunctionalBindingState.SET_VALUED
        )
        return FunctionalBindingDecision(
            state,
            selected_ids,
            "PASS_FUNCTIONAL_SET_TO_GEOMETRY",
            "PUBLIC_TASK_RELATION+INSTANCE_BOUND_GEOMETRY",
            relation,
        )
