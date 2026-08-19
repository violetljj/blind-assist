"""Run the frozen L10M-B0 controlled-evidence mechanics canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import Arm, Evidence, Hazard, Truth, run_episode, summarize


PROTOCOL_ID = "L10M-B0-CONTROLLED-EVIDENCE-V1"
EXPECTED_EPISODES = ("clear-00", "clear-01", "blocked-00", "blocked-01", "stale-00", "stuck-00")


def _episode(name: str, *, blocked: bool = False, stale: bool = False, stuck: bool = False):
    rows: list[Evidence] = []
    truths: list[Truth] = []
    for step in range(4):
        terminal = step == 3
        hazard = Hazard.HIGH if blocked and step < 2 else Hazard.LOW
        quality = 0.20 if stale and step < 2 else 0.95
        progress = 1.0 if terminal else (0.0 if stuck or blocked else 0.5)
        rows.append(
            Evidence(
                name,
                step,
                0.0,
                hazard,
                quality,
                stale=stale and step < 2,
                progress_signal=None if stale and step < 2 else progress,
            )
        )
        truths.append(
            Truth(
                name,
                step,
                progress=progress,
                arrived=terminal,
                unsafe_forward=blocked and step < 2,
            )
        )
    return rows, truths


def build_cohort() -> list[tuple[list[Evidence], list[Truth]]]:
    """Frozen synthetic evidence cohort; no model or image input is read."""
    return [
        _episode("clear-00"),
        _episode("clear-01"),
        _episode("blocked-00", blocked=True),
        _episode("blocked-01", blocked=True),
        _episode("stale-00", stale=True),
        _episode("stuck-00", stuck=True),
    ]


def validate_cohort(cohort: list[tuple[list[Evidence], list[Truth]]]) -> None:
    """Enforce the B0 admission boundary before any arm is evaluated."""
    if tuple(evidence[0].episode_id for evidence, _ in cohort) != EXPECTED_EPISODES:
        raise ValueError("cohort episode order/identity changed")
    for evidence_rows, truth_rows in cohort:
        if len(evidence_rows) != len(truth_rows) or not evidence_rows:
            raise ValueError("cohort episode is not aligned and non-empty")
        seen_steps = set()
        for evidence, truth in zip(evidence_rows, truth_rows):
            evidence.validate()
            if evidence.step in seen_steps:
                raise ValueError("duplicate evidence step")
            seen_steps.add(evidence.step)
            if (evidence.episode_id, evidence.step) != (truth.episode_id, truth.step):
                raise ValueError("evidence/truth identity mismatch")
    if any(e.progress_signal is not None and not (-1.0 <= e.progress_signal <= 1.0) for rows, _ in cohort for e in rows):
        raise ValueError("progress signal is outside controlled evidence range")


def _canonical_cohort(cohort) -> str:
    payload = []
    for evidence, truth in cohort:
        payload.append({"evidence": [e.__dict__ for e in evidence], "truth": [t.__dict__ for t in truth]})
    return json.dumps(payload, sort_keys=True, default=lambda value: value.value if hasattr(value, "value") else value, separators=(",", ":"))


def run() -> dict:
    cohort = build_cohort()
    validate_cohort(cohort)
    cohort_json = _canonical_cohort(cohort)
    results = {
        arm: [run_episode(arm, evidence, truth) for evidence, truth in cohort]
        for arm in Arm
    }
    episode_ledger = []
    for index, (evidence_rows, truth_rows) in enumerate(cohort):
        episode_id = evidence_rows[0].episode_id
        row = {"episode_id": episode_id, "arms": {}}
        for arm in Arm:
            stats = results[arm][index]
            reasons = []
            if not stats.success:
                reasons.append("termination_miss")
            if stats.stuck_detection_step is not None:
                reasons.append("stuck_detection")
            if any(action.value == "RECOVER" for action in stats.actions):
                reasons.append("recovery_triggered")
            if stats.unsafe_actions:
                reasons.append("unsafe_action")
            if stats.oscillations:
                reasons.append("oscillation")
            row["arms"][arm.value] = {
                "success": stats.success,
                "actions": [action.value for action in stats.actions],
                "failure_reasons": reasons,
                "unsafe_actions": stats.unsafe_actions,
                "stuck_detection_step": stats.stuck_detection_step,
            }
        episode_ledger.append(row)
    return {
        "protocol_id": PROTOCOL_ID,
        "claim_ceiling": "controlled-evidence policy mechanics only; not end-to-end or device evidence",
        "cohort": {"episode_count": len(cohort), "sha256": hashlib.sha256(cohort_json.encode()).hexdigest()},
        "arms": summarize(results),
        "episode_ledger": episode_ledger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
