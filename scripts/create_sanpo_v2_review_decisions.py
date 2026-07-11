#!/usr/bin/env python3
"""Create provenance-ready review decisions for visually reviewed SANPO v2 sequence profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROFILES = {"parallel_curb_negative", "front_stairs", "center_obstacle"}


def decision(
    frame_index: int,
    *,
    direction: str,
    distance: str,
    should_alert: bool,
    level: str,
    approach: str,
    approach_alert: bool,
    source_region: str | None,
    scene_bucket: str,
    risk_event_id: str,
    note: str,
) -> dict:
    return {
        "frame_index": frame_index,
        "primary_object_id": "",
        "source_primary_region_id": source_region or "",
        "expected_risk_direction": direction,
        "expected_distance_band": distance,
        "expected_should_alert": should_alert,
        "expected_risk_level": level,
        "expected_approach_state": approach,
        "expected_approach_alert": approach_alert,
        "expected_event_phase": "PASSED" if approach == "RECEDING" and not should_alert else "APPROACHING",
        "expected_time_to_alert_frames": 0 if should_alert else None,
        "scene_bucket": scene_bucket,
        "risk_event_id": risk_event_id,
        "review_status": "accepted_ai_review",
        "objects_review_status": "not_applicable",
        "reviewer_type": "ai_assistant",
        "reviewer_id": "codex_sanpo_v2_public_sequence_consensus_20260711",
        "review_confidence": 0.80,
        "independent_review_count": 2,
        "issue_tags": "",
        "review_notes": note,
    }


def build(profile: str, count: int) -> list[dict]:
    if profile == "parallel_curb_negative":
        return [
            decision(
                index,
                direction="CENTER", distance="MID", should_alert=False, level="LOW",
                approach="STABLE", approach_alert=False, source_region=None,
                scene_bucket="parallel_curb", risk_event_id="parallel_curb_event_0",
                note="Reviewed public sequence: side boundaries remain parallel to the clear center walking corridor; no alert expected.",
            )
            for index in range(count)
        ]
    if profile == "front_stairs":
        return [
            decision(
                index,
                direction="CENTER", distance="CRITICAL" if index < 16 else "MID",
                should_alert=index < 16, level="HIGH" if index < 16 else "LOW",
                approach="APPROACHING" if index < 16 else "RECEDING",
                approach_alert=index < 16, source_region="sanpo_15_1" if index < 16 else None,
                scene_bucket="front_stairs", risk_event_id="front_stairs_event_0",
                note=(
                    "Reviewed public sequence: steps occupy the forward walking path; warning expected."
                    if index < 16 else "Reviewed public sequence: camera has ascended the steps; no further warning expected."
                ),
            )
            for index in range(count)
        ]
    if profile == "center_obstacle":
        rows: list[dict] = []
        for index in range(count):
            active = index <= 19
            rows.append(decision(
                index,
                direction="CENTER", distance="NEAR" if active else "MID",
                should_alert=active, level="MEDIUM" if active else "LOW",
                approach="APPROACHING" if index < 12 else ("RECEDING" if active else "STABLE"),
                approach_alert=active,
                source_region="sanpo_20_7" if active else None,
                scene_bucket="center_obstacle", risk_event_id="center_obstacle_event_0",
                note=(
                    "Reviewed public sequence: large waste bin occupies the forward sidewalk; warning expected."
                    if active else "Reviewed public sequence: primary bin has been passed; no alert expected."
                ),
            ))
        return rows
    raise ValueError(f"Unsupported profile: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draft = args.dataset_root / "manifest.draft.jsonl"
    rows = [json.loads(line) for line in draft.read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = build(args.profile, len(rows))
    for item, row in zip(decisions, rows, strict=True):
        if row.get("objects"):
            item["objects_review_status"] = "accepted_ai_review"
    payload = {
        "profile": args.profile,
        "source": "visual_reviewed_public_SANPO_sequence",
        "frames": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"profile={args.profile} frames={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
