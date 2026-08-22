"""Lightweight append-only JSONL event log for exploratory episodes."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
from typing import Any

from .contract import CurrentFrameObservation, GuidanceDecision


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


class EpisodeEventLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(
        self,
        *,
        episode_id: str,
        event_id: str,
        observation: CurrentFrameObservation,
        decision: GuidanceDecision,
        spoken_command: str | None = None,
        haptic_command: str | None = None,
        user_denied: bool = False,
    ) -> dict[str, Any]:
        if not episode_id or not event_id:
            raise ValueError("episode_id and event_id are required")
        payload = {
            "schema_version": "blindassist_selective_guidance_episode_event_v0",
            "episode_id": episode_id,
            "event_id": event_id,
            "goal_contract": dict(observation.goal_contract),
            "frame_id": observation.frame_id,
            "observed_at_ms": observation.observed_at_ms,
            "decision_at_ms": observation.decision_at_ms,
            "visible_candidate_set": list(observation.visible_candidate_ids),
            "selected_referent": observation.selected_referent,
            "candidate_cardinality": observation.cardinality.value,
            "decision_state": decision.status.value,
            "decision_tokens": list(decision.tokens),
            "spoken_command": spoken_command,
            "haptic_command": haptic_command,
            "range_bucket": decision.range_bucket.value,
            "range_uncertainty": observation.range_uncertainty,
            "abstention": decision.status.value == "ABSTAIN",
            "contested": decision.status.value == "CONTESTED",
            "lost": decision.status.value == "LOST",
            "stale": decision.status.value == "STALE",
            "handoff_event": decision.status.value == "HANDOFF_READY",
            "user_confirmation": decision.status.value == "COMPLETED_BY_USER",
            "user_denial": user_denied,
            "latency_ms": observation.latency_ms,
            "provider_receipts": _jsonable([asdict(item) for item in observation.provider_receipts]),
            "reason": decision.reason,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        return payload

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != "blindassist_selective_guidance_episode_event_v0":
                    raise ValueError(f"unexpected event schema at line {line_number}")
                rows.append(row)
        return rows
