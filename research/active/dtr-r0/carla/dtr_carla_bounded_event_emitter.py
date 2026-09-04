"""Frozen simple temporal emitter for DTR evidence-model comparisons.

This layer has no collision geometry, tracker, learned parameters, or access to
evaluator truth.  It converts an evidence arm's framewise route-risk state into
an alert segment using one bounded 0.60 s hold, qualified only by unchanged
route mode and issued-plan identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "DTR_BOUNDED_EVENT_EMITTER_0P60S"
ARM_X94_HYSTERESIS = "X94_EVIDENCE_PLUS_SIMPLE_HYSTERESIS_0P60S"
HOLD_SECONDS = 0.60
EPSILON = 1e-9


def fixed_constants() -> dict[str, Any]:
    return {
        "hold_seconds": HOLD_SECONDS,
        "refresh": "CURRENT_EVIDENCE_ROUTE_RISK_TRUE",
        "release": [
            "HOLD_HORIZON_EXPIRED",
            "ROUTE_MODE_CHANGED",
            "PLAN_RECEIPT_IDENTITY_CHANGED",
        ],
        "learned_parameters": 0,
        "collision_geometry": "NONE_CONSUMES_UPSTREAM_EVIDENCE_ONLY",
    }


@dataclass
class BoundedEventEmitter:
    last_evidence_time_s: float | None = None
    last_entry_s: float | None = None
    route_identity: tuple[str, str] | None = None

    def update(
        self,
        *,
        time_s: float,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        now_s = float(time_s)
        identity = (
            str(evidence.get("route_mode") or ""),
            str(evidence.get("plan_receipt_sha256") or ""),
        )
        observed = bool(evidence.get("route_risk"))
        if observed:
            self.last_evidence_time_s = now_s
            raw_entry = evidence.get("minimum_entry_s")
            self.last_entry_s = None if raw_entry is None else float(raw_entry)
            self.route_identity = identity
            state = "ACTIVE_EVIDENCE"
            risk = True
            entry_s = self.last_entry_s
        elif (
            self.last_evidence_time_s is not None
            and now_s - self.last_evidence_time_s <= HOLD_SECONDS + EPSILON
            and identity == self.route_identity
            and not bool(evidence.get("route_mode_changed"))
        ):
            state = "ACTIVE_BOUNDED_HOLD"
            risk = True
            elapsed = now_s - self.last_evidence_time_s
            entry_s = (
                None
                if self.last_entry_s is None
                else max(0.0, self.last_entry_s - elapsed)
            )
        else:
            state = "CLEAR"
            risk = False
            entry_s = None
            self.last_evidence_time_s = None
            self.last_entry_s = None
            self.route_identity = None
        return {
            "route_mode": evidence.get("route_mode"),
            "authority": evidence.get("authority"),
            "plan_receipt_sha256": evidence.get("plan_receipt_sha256"),
            "route_mode_changed": bool(evidence.get("route_mode_changed")),
            "route_risk": risk,
            "minimum_entry_s": entry_s,
            "emission_state": state,
            "raw_evidence_route_risk": observed,
        }


def apply_episode(
    episode: Mapping[str, Any],
    *,
    evidence_arm: str,
    output_arm: str = ARM_X94_HYSTERESIS,
) -> dict[str, Any]:
    emitter = BoundedEventEmitter()
    frames: list[dict[str, Any]] = []
    state_counts = {
        "ACTIVE_EVIDENCE": 0,
        "ACTIVE_BOUNDED_HOLD": 0,
        "CLEAR": 0,
    }
    for source in episode["frames"]:
        emitted = emitter.update(
            time_s=float(source["time_s"]),
            evidence=source["arms"][evidence_arm],
        )
        state_counts[emitted["emission_state"]] += 1
        frames.append(
            {
                "sample_index": int(source["sample_index"]),
                "time_s": float(source["time_s"]),
                "world_frame": source.get("world_frame"),
                "arms": {output_arm: emitted},
            }
        )
    return {
        "episode_id": episode.get("episode_id"),
        "frames": frames,
        "diagnostics": {
            "frame_count": len(frames),
            "state_counts": state_counts,
        },
        "arms": {
            output_arm: {
                "route_risk_frames": sum(
                    bool(frame["arms"][output_arm]["route_risk"])
                    for frame in frames
                )
            }
        },
    }


def prediction_envelope(
    source: Mapping[str, Any],
    *,
    evidence_arm: str,
    output_arm: str = ARM_X94_HYSTERESIS,
) -> dict[str, Any]:
    episodes = {
        episode_id: apply_episode(
            episode,
            evidence_arm=evidence_arm,
            output_arm=output_arm,
        )
        for episode_id, episode in source["episodes"].items()
    }
    return {
        "schema": "blindassist-dtr-bounded-event-emitter-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "experiment_id": EXPERIMENT_ID,
        "truth_blind": True,
        "source_evidence_arm": evidence_arm,
        "arms": [output_arm],
        "episodes": episodes,
        "fixed_constants": fixed_constants(),
        "claim_boundary": {
            "collision_evidence_changed": False,
            "temporal_emission_only": True,
            "evaluator_opened": False,
        },
    }


__all__ = [
    "ARM_X94_HYSTERESIS",
    "BoundedEventEmitter",
    "apply_episode",
    "fixed_constants",
    "prediction_envelope",
]
