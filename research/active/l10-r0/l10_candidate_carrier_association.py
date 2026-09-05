"""Associate public text observations with physical candidate carriers.

All boxes use normalized coordinates in the declared source/frame/ROI. Unique
containment supplies local spatial support only. Initial identity additionally
needs an upstream identity-sign role with provenance, or an explicit supported
IDENTITY_TEXT_OF relation. The geometric route also requires a declared complete
candidate roster. Neither route establishes entrance ownership or endpoint extent.

Legacy rows without spatial provenance remain unresolved. No evaluator field is
read, and failure to associate a witness never means the target is absent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class AssociationState(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    LOCAL_SUPPORT = "LOCAL_SUPPORT"
    IDENTITY_SUPPORT = "IDENTITY_SUPPORT"
    CONTRADICTED = "CONTRADICTED"


NON_IDENTITY_ROLES = frozenset({"ADVERTISEMENT", "DIRECTION_SIGN", "SIBLING_SIGN"})


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, (list, tuple)) else []


def _identifier(value: Any) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return x1, y1, x2, y2


@dataclass(frozen=True)
class Region:
    source_id: str
    frame_id: str
    roi_id: str
    box: tuple[float, float, float, float] | None

    @classmethod
    def parse(cls, row: Mapping[str, Any], context: Mapping[str, Any], *, allow_unlocalized: bool = False) -> Region | None:
        if any(
            _identifier(context.get(key)) is not None and key in row
            and _identifier(row[key]) != _identifier(context[key])
            for key in ("source_id", "frame_id")
        ):
            return None
        ids = [_identifier(row.get(key, context.get(key))) for key in ("source_id", "frame_id", "roi_id")]
        box = _box(row.get("bbox_xyxy_norm"))
        return cls(*ids, box) if all(ids) and (box is not None or allow_unlocalized) else None

    def shares_coordinates(self, other: Region) -> bool:
        return (self.source_id, self.frame_id, self.roi_id) == (other.source_id, other.frame_id, other.roi_id)

    def contains(self, other: Region) -> bool:
        a, b = self.box, other.box
        return bool(a and b and self.shares_coordinates(other) and a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3])


@dataclass(frozen=True)
class AssociatedCandidate:
    candidate_id: str
    evidence: Mapping[str, Any]
    region: Region | None
    state: AssociationState
    local_text_ids: tuple[str, ...]
    identity_text_ids: tuple[str, ...]
    identity_texts: tuple[str, ...]
    reasons: tuple[str, ...]
    complete_identity_names: tuple[str, ...] = ()

    @property
    def track_key(self) -> tuple[str, str] | None:
        track_id = _identifier(self.evidence.get("track_id"))
        return (self.region.source_id, track_id) if self.region and track_id else None


@dataclass(frozen=True)
class AssociationResult:
    candidates: tuple[AssociatedCandidate, ...]
    unresolved_text_ids: tuple[str, ...]
    claim_scope: str = "CANDIDATE_CARRIER_ASSOCIATION_ONLY_NO_OWNERSHIP_OR_ABSENCE"


def associate_frame(frame: Mapping[str, Any]) -> AssociationResult:
    """Resolve candidates jointly, so one text cannot be borrowed by a sibling.

    Candidate rows live in ``candidate_evidence`` (or ``appearance_evidence``).
    Text rows require ``text_id``. Explicit ``carrier_relations`` contain
    candidate_id, text_id, source_id, frame_id, roi_id, evidence_id, status and
    predicate. Supported LOCAL_OBSERVATION_OF preserves a producer's local crop
    witness, even when a sign is outside a door's endpoint box; it does not create
    identity. A typed IDENTITY_TEXT_OF edge is a producer claim, never invented
    from geometry here. Cross-ROI producer edges must name text_roi_id and
    candidate_roi_id; they may retain a local witness without an exported crop
    transform. Such an unlocalized text never supplies geometric containment.
    Duplicate IDs and competing witnesses stay unresolved.
    """
    raw_candidates = _rows(frame.get("candidate_evidence", frame.get("appearance_evidence", ())))
    texts = _rows(frame.get("text_evidence", ()))
    relations = _rows(frame.get("carrier_relations", ()))
    candidate_ids = [_identifier(row.get("candidate_id")) for row in raw_candidates]
    text_ids = [_identifier(row.get("text_id")) for row in texts]
    regions = [Region.parse(row, frame) for row in raw_candidates]
    local: list[dict[str, str]] = [{} for _ in raw_candidates]
    identity: list[dict[str, str]] = [{} for _ in raw_candidates]
    complete_names: list[dict[str, str]] = [{} for _ in raw_candidates]
    reasons: list[set[str]] = [set() for _ in raw_candidates]
    contradicted: set[int] = set()
    unresolved: list[str] = []
    for index, candidate_id in enumerate(candidate_ids):
        if candidate_id is None or candidate_ids.count(candidate_id) != 1 or regions[index] is None:
            reasons[index].add("MISSING_OR_AMBIGUOUS_CARRIER_PROVENANCE")
    valid = [not value for value in reasons]

    for text_index, text in enumerate(texts):
        text_id = text_ids[text_index]
        region = Region.parse(text, frame, allow_unlocalized=True)
        if text_id is None or text_ids.count(text_id) != 1 or region is None:
            unresolved.append(text_id or f"unidentified-text-{text_index}")
            continue
        geometric = [i for i, value in enumerate(regions) if valid[i] and value.contains(region)]
        explicit: dict[int, str] = {}
        rejected: set[int] = set()
        unresolved_relation: set[int] = set()
        for relation in relations:
            if relation.get("text_id") != text_id or not _identifier(relation.get("evidence_id")):
                continue
            if tuple(_identifier(relation.get(k)) for k in ("source_id", "frame_id")) != (region.source_id, region.frame_id):
                continue
            if _identifier(relation.get("text_roi_id", relation.get("roi_id"))) != region.roi_id:
                continue
            matches = [i for i, value in enumerate(candidate_ids) if value == relation.get("candidate_id")]
            if len(matches) != 1:
                continue
            i = matches[0]
            if not valid[i] or (regions[i].source_id, regions[i].frame_id) != (region.source_id, region.frame_id):
                continue
            if _identifier(relation.get("candidate_roi_id", relation.get("roi_id"))) != regions[i].roi_id:
                continue
            if relation.get("status") == "CONTRADICTED":
                rejected.add(i)
            elif relation.get("status") == "SUPPORTED" and relation.get("predicate") in (
                "IDENTITY_TEXT_OF", "LOCAL_OBSERVATION_OF", "LOCAL_TOPOLOGY_SUPPORTS"
            ):
                predicate = str(relation["predicate"])
                if explicit.get(i) != "IDENTITY_TEXT_OF":
                    explicit[i] = predicate
            elif relation.get("status") == "UNKNOWN" or relation.get("predicate") in (
                "ADVERTISEMENT_ON", "DIRECTION_TO", "SIBLING_OF"
            ):
                unresolved_relation.add(i)
        for i in rejected:
            reasons[i].add("EXPLICIT_CARRIER_RELATION_CONTRADICTION")
            contradicted.add(i)
            explicit.pop(i, None)
        linked = list(explicit) if explicit else [i for i in geometric if i not in rejected]
        if len(linked) != 1:
            unresolved.append(text_id)
            for i in linked:
                reasons[i].add("AMBIGUOUS_TEXT_CARRIER")
            continue
        i = linked[0]
        local[i][text_id] = str(text.get("text", ""))
        role = str(text.get("role", "UNKNOWN"))
        identity_role = role == "IDENTITY_SIGN" and _identifier(text.get("role_evidence_id")) is not None
        if i in unresolved_relation:
            reasons[i].add("EXPLICIT_CARRIER_IDENTITY_RELATION_UNRESOLVED")
        elif role in NON_IDENTITY_ROLES:
            reasons[i].add("TEXT_ROLE_DOES_NOT_IDENTIFY_CARRIER")
        elif explicit.get(i) == "IDENTITY_TEXT_OF" or (
            not explicit and identity_role and frame.get("candidate_roster_complete") is True
        ):
            identity[i][text_id] = str(text.get("text", ""))
            if text.get("name_completeness") == "COMPLETE":
                complete_names[i][text_id] = str(text.get("text", ""))
        else:
            reasons[i].add("LOCAL_SUPPORT_WITHOUT_IDENTITY_RELATION")
            if identity_role and frame.get("candidate_roster_complete") is not True:
                reasons[i].add("CANDIDATE_ROSTER_COMPLETENESS_UNKNOWN")

    result = []
    for i, raw in enumerate(raw_candidates):
        state = AssociationState.UNRESOLVED
        if identity[i]:
            state = AssociationState.IDENTITY_SUPPORT
        elif local[i]:
            state = AssociationState.LOCAL_SUPPORT
        # Contradiction blocks this candidate's current identity. It is not an
        # assertion that the target is absent anywhere in the scene.
        if i in contradicted:
            state = AssociationState.CONTRADICTED
            identity[i].clear()
            complete_names[i].clear()
        result.append(AssociatedCandidate(
            candidate_ids[i] or f"unidentified-candidate-{i}", raw, regions[i], state,
            tuple(local[i]), tuple(identity[i]), tuple(identity[i].values()), tuple(sorted(reasons[i])),
            tuple(complete_names[i].values()),
        ))
    return AssociationResult(tuple(result), tuple(unresolved))


class BoundedCarrierContinuity:
    """A current track may carry a prior binding for a fixed number of new views.

    Appearance cannot seed the lease. Repeated/missing observation IDs cannot
    refresh it; a new source/track or explicit contradiction cannot inherit it.
    The producer must supply a current ``track_evidence_id`` to continue a track.
    """
    def __init__(self, max_track_only_frames: int = 4):
        if max_track_only_frames < 0:
            raise ValueError("max_track_only_frames must be nonnegative")
        self.limit = max_track_only_frames
        self.reset()

    def reset(self) -> None:
        self.track_key: tuple[str, str] | None = None
        self.seen_frames: set[str] = set()
        self.age = 0

    def confirm(self, candidate: AssociatedCandidate) -> None:
        self.reset()
        if candidate.state == AssociationState.IDENTITY_SUPPORT and candidate.region:
            self.track_key = candidate.track_key
            self.seen_frames.add(candidate.region.frame_id)

    def advance(self, candidate: AssociatedCandidate | None) -> bool:
        self.age += 1
        if self.age > self.limit:
            self.reset()
            return False
        if candidate is None or candidate.region is None or candidate.track_key != self.track_key or self.track_key is None:
            return False
        if candidate.state == AssociationState.CONTRADICTED or candidate.evidence.get("track_status") == "CONTRADICTED":
            self.reset()
            return False
        frame_key = candidate.region.frame_id
        if frame_key in self.seen_frames or not _identifier(candidate.evidence.get("track_evidence_id")):
            return False
        self.seen_frames.add(frame_key)
        return True
