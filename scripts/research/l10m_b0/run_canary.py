"""Run the frozen L10M-B0 controlled-evidence mechanics canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluation import Arm, Evidence, Hazard, Truth, run_episode, summarize


PROTOCOL_ID = "L10M-B0-CONTROLLED-EVIDENCE-V1"


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


def _canonical_cohort(cohort) -> str:
    payload = []
    for evidence, truth in cohort:
        payload.append({"evidence": [e.__dict__ for e in evidence], "truth": [t.__dict__ for t in truth]})
    return json.dumps(payload, sort_keys=True, default=lambda value: value.value if hasattr(value, "value") else value, separators=(",", ":"))


def run() -> dict:
    cohort = build_cohort()
    cohort_json = _canonical_cohort(cohort)
    results = {
        arm: [run_episode(arm, evidence, truth) for evidence, truth in cohort]
        for arm in Arm
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "claim_ceiling": "controlled-evidence policy mechanics only; not end-to-end or device evidence",
        "cohort": {"episode_count": len(cohort), "sha256": hashlib.sha256(cohort_json.encode()).hexdigest()},
        "arms": summarize(results),
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
