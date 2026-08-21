"""Deliberately simple score-threshold baseline for the P1-R0 contract.

The baseline consumes only algorithmic identity evidence. It has no evaluator
truth, image access, semantic-grounding authority, learned model, or Sky path.
Its eager highest-score selection and reacquisition are intentionally ordinary
so the safety-oriented evaluator can expose identity-switch headroom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL_ID = "BA-P1-TARGET-PERSISTENCE-R0-V1"
PUBLIC_INPUT_KEYS = {"schema_version", "protocol_id", "episode_id", "handoff", "frames"}
MIN_SUPPORT = 0.55
MAX_CONTRADICTION = 0.45
MIN_STABILITY = 0.45
AMBIGUITY_GAP = 0.08
MAX_OSCILLATION = 0.60
TEMP_UNOBSERVABLE_FRAMES = 2


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    return float(candidate["identity_support"]) - float(candidate["identity_contradiction"])


def _ordered_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(candidates, key=lambda item: (-_candidate_score(item), str(item["candidate_id"])))


def extract_public_input(evaluator_episode: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only system-visible fields out of an evaluator episode."""
    return {key: evaluator_episode[key] for key in PUBLIC_INPUT_KEYS}


def run_baseline(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Return one deterministic, protocol-shaped output for an episode."""
    if set(episode) != PUBLIC_INPUT_KEYS:
        raise ValueError("baseline input must be the public P1 surface without evaluator truth")
    if episode["schema_version"] != 1 or episode["protocol_id"] != PROTOCOL_ID:
        raise ValueError("baseline input protocol drift")
    handoff = episode["handoff"]
    if handoff.get("status") not in {"REFERENT_ESTABLISHED", "NO_REFERENT"}:
        raise ValueError("baseline handoff status drift")
    if handoff["status"] == "REFERENT_ESTABLISHED" and not isinstance(handoff.get("referent_id"), str):
        raise ValueError("established baseline handoff requires referent_id")
    if handoff["status"] == "NO_REFERENT" and handoff.get("referent_id") is not None:
        raise ValueError("NO_REFERENT baseline handoff requires null referent_id")
    referent_id = handoff["referent_id"] if handoff["status"] == "REFERENT_ESTABLISHED" else None
    state = "TRACKING" if referent_id is not None else "UNBOUND"
    frames_since_confirmed: int | None = 0 if referent_id is not None else None
    frame_outputs: list[dict[str, Any]] = []

    for frame in episode["frames"]:
        if referent_id is None:
            frame_outputs.append({
                "frame_index": frame["frame_index"],
                "state": "UNBOUND",
                "current_candidate_id": None,
                "identity_score": None,
                "stability_score": None,
                "oscillation_score": None,
                "frames_since_confirmed": None,
                "event": "NONE",
            })
            continue

        candidates = _ordered_candidates(frame["candidates"])
        previous_state = state
        current_candidate_id = None
        identity_score = None
        stability_score = None
        oscillation_score = None
        event = "NONE"

        if not candidates:
            assert frames_since_confirmed is not None
            frames_since_confirmed += 1
            if frames_since_confirmed <= TEMP_UNOBSERVABLE_FRAMES:
                state = "TEMP_UNOBSERVABLE"
                if previous_state != "TEMP_UNOBSERVABLE":
                    event = "TARGET_TEMP_UNOBSERVABLE"
            else:
                state = "LOST"
                if previous_state != "LOST":
                    event = "LOSS_DETECTED"
        else:
            best = candidates[0]
            identity_score = float(best["identity_support"])
            stability_score = float(best["stability"])
            oscillation_score = float(best["oscillation"])
            score_gap = (
                _candidate_score(best) - _candidate_score(candidates[1])
                if len(candidates) > 1
                else float("inf")
            )
            ambiguous = score_gap <= AMBIGUITY_GAP or oscillation_score >= MAX_OSCILLATION
            eligible = (
                identity_score >= MIN_SUPPORT
                and float(best["identity_contradiction"]) <= MAX_CONTRADICTION
                and stability_score >= MIN_STABILITY
            )
            if eligible and not ambiguous:
                current_candidate_id = str(best["candidate_id"])
                state = "TRACKING"
                frames_since_confirmed = 0
                if previous_state == "LOST":
                    event = "REACQUIRED"
            else:
                assert frames_since_confirmed is not None
                frames_since_confirmed += 1
                state = "UNCERTAIN"

        frame_outputs.append({
            "frame_index": frame["frame_index"],
            "state": state,
            "current_candidate_id": current_candidate_id,
            "identity_score": identity_score,
            "stability_score": stability_score,
            "oscillation_score": oscillation_score,
            "frames_since_confirmed": frames_since_confirmed,
            "event": event,
        })

    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "episode_id": episode["episode_id"],
        "referent_id": referent_id,
        "score_semantics": "ALGORITHMIC_EVIDENCE_NOT_CALIBRATED_PROBABILITY",
        "frames": frame_outputs,
    }
