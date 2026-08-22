"""Create a fail-closed truth-unknown private evaluator manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .truth_contract import empty_teacher_outputs


def make_annotation(public: dict) -> dict:
    if public.get("schema_version") != "blindassist_real_episode_public_manifest_v0":
        raise ValueError("public episode manifest schema mismatch")
    episodes = []
    for episode in public["episodes"]:
        episodes.append({
            "episode_id": episode["episode_id"],
            "completed_by_user": False,
            "completion_time_ms": None,
            "instruction_count": 0,
            "correction_count": 0,
            "user_denial_count": 0,
            "handoff_additional_actions": None,
            "post_visibility_reacquisition_failure": False,
            "observations": [
                {
                    "observation_id": row["observation_id"],
                    "truth_authority_tier": "UNKNOWN",
                    "teacher_outputs": empty_teacher_outputs(),
                    "teacher_agreement": None,
                    "functional_authority": "NOT_ESTABLISHED",
                    "functional_authority_sources": [],
                    "target_visibility": "UNKNOWN",
                    "legal_candidate_ids": [],
                    "legal_regions_normalized_xyxy": [],
                    "allowed_decision_states": [],
                    "range_truth": "RANGE_UNKNOWN",
                    "user_confirmed": False,
                    "user_denied": False,
                    "handoff_truth": False,
                    "notes": "",
                }
                for row in episode["observations"]
            ],
        })
    return {
        "schema_version": "blindassist_real_episode_annotation_v1",
        "private_evaluator_only": True,
        "truth_frozen": False,
        "public_manifest_schema": public["schema_version"],
        "episodes": episodes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("annotation output already exists")
    payload = make_annotation(json.loads(args.public_manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
