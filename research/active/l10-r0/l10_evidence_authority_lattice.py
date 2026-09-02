from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Authority(str, Enum):
    UNKNOWN = "UNKNOWN"
    DIRECTIONAL = "DIRECTIONAL"
    BINDING = "BINDING"
    TERMINAL = "TERMINAL"


class Action(str, Enum):
    SEARCH = "SEARCH"
    ORIENT_LEFT = "ORIENT_LEFT"
    ORIENT_RIGHT = "ORIENT_RIGHT"
    APPROACH = "APPROACH"
    TRACK = "TRACK"
    COMMIT = "COMMIT"
    HANDOFF = "HANDOFF"


@dataclass(frozen=True)
class Evidence:
    """Truth-free runtime evidence presented to the authority reducer.

    The `*_supported` fields are claims made by upstream evidence producers.
    Evaluator-private facade, entrance, target-absence, and handoff truth must
    never be copied into this object.
    """

    directional_supported: bool = False
    bearing: float | None = None
    approach_expected_to_improve_readability: bool = False

    referent_id: str | None = None
    facade_id: str | None = None
    same_referent_continuity_supported: bool = False
    sibling_exclusion_supported: bool = False
    sign_facade_association_supported: bool = False

    entrance_id: str | None = None
    entrance_ownership_supported: bool = False
    endpoint_visible: bool = False
    terminal_pose_supported: bool = False
    terminal_commit_supported: bool = False
    handoff_supported: bool = False

    contradiction: bool = False
    observed_distance_m: float | None = None

    def has_directional_authority(self) -> bool:
        actionable_bearing = self.bearing is not None and self.bearing != 0.0
        return self.directional_supported and (
            actionable_bearing or self.approach_expected_to_improve_readability
        )

    def has_binding_authority(self) -> bool:
        return (
            self.referent_id is not None
            and self.facade_id is not None
            and self.same_referent_continuity_supported
            and self.sibling_exclusion_supported
            and self.sign_facade_association_supported
        )

    def has_terminal_authority(self) -> bool:
        return (
            self.has_binding_authority()
            and self.entrance_id is not None
            and self.entrance_ownership_supported
            and self.endpoint_visible
            and self.terminal_pose_supported
            and (self.terminal_commit_supported or self.handoff_supported)
        )


@dataclass(frozen=True)
class Decision:
    authority: Authority
    action: Action
    referent_id: str | None
    facade_id: str | None
    entrance_id: str | None
    reason: str


class EvidenceAuthorityLattice:
    """Reversible L10 authority reducer with no distance transition rule."""

    def __init__(self) -> None:
        self.authority = Authority.UNKNOWN
        self.referent_id: str | None = None
        self.facade_id: str | None = None
        self.entrance_id: str | None = None

    def reset(self) -> Decision:
        self.authority = Authority.UNKNOWN
        self.referent_id = None
        self.facade_id = None
        self.entrance_id = None
        return self._decision(Action.SEARCH, "RESET")

    def step(self, evidence: Evidence) -> Decision:
        if evidence.contradiction:
            self.authority = Authority.UNKNOWN
            self.referent_id = None
            self.facade_id = None
            self.entrance_id = None
            return self._decision(Action.SEARCH, "CONTRADICTION_REVOKED_AUTHORITY")

        binding = evidence.has_binding_authority()
        if binding and self.referent_id is not None:
            binding = (
                evidence.referent_id == self.referent_id
                and evidence.facade_id == self.facade_id
            )
            if not binding:
                self.authority = Authority.UNKNOWN
                self.referent_id = None
                self.facade_id = None
                self.entrance_id = None
                return self._decision(Action.SEARCH, "REFERENT_OR_FACADE_CONFLICT")

        if binding:
            self.referent_id = evidence.referent_id
            self.facade_id = evidence.facade_id
            if evidence.has_terminal_authority():
                self.authority = Authority.TERMINAL
                self.entrance_id = evidence.entrance_id
                if evidence.handoff_supported:
                    return self._decision(Action.HANDOFF, "ENDPOINT_AND_HANDOFF_SUPPORTED")
                return self._decision(Action.COMMIT, "ENDPOINT_COMMIT_SUPPORTED")
            self.authority = Authority.BINDING
            self.entrance_id = None
            return self._decision(Action.TRACK, "EXACT_REFERENT_CONTINUITY_SUPPORTED")

        if evidence.has_directional_authority():
            self.authority = Authority.DIRECTIONAL
            self.referent_id = None
            self.facade_id = None
            self.entrance_id = None
            if evidence.bearing is not None and evidence.bearing < 0.0:
                return self._decision(Action.ORIENT_LEFT, "DIRECTIONAL_BEARING")
            if evidence.bearing is not None and evidence.bearing > 0.0:
                return self._decision(Action.ORIENT_RIGHT, "DIRECTIONAL_BEARING")
            return self._decision(Action.APPROACH, "EXPECTED_READABILITY_GAIN")

        self.authority = Authority.UNKNOWN
        self.referent_id = None
        self.facade_id = None
        self.entrance_id = None
        return self._decision(Action.SEARCH, "INSUFFICIENT_CURRENT_EVIDENCE")

    def _decision(self, action: Action, reason: str) -> Decision:
        return Decision(
            authority=self.authority,
            action=action,
            referent_id=self.referent_id,
            facade_id=self.facade_id,
            entrance_id=self.entrance_id,
            reason=reason,
        )
