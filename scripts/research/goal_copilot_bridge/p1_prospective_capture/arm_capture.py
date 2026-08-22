#!/usr/bin/env python3
"""Freeze an outcome-blind physical-capture plan from a C0 Goal Contract receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256
from scripts.research.goal_copilot_bridge.p1_prospective_capture.materialize_capture import (
    C0_SCHEMA,
    CAPTURE_INSTRUCTION,
    FRAME_OFFSETS_FROM_END_SECONDS,
    MINIMUM_EPISODES,
    SOURCE_ROLE,
    ProspectiveCaptureError,
    _timestamp,
    _verify_hashed_body,
)


CAPTURE_PLAN_SCHEMA = "blindassist_p1_pa3_prospective_first_person_capture_plan_v1"
SAFE_EPISODE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")


def arm_capture(c0: Mapping[str, Any], armed_at_utc: str) -> dict[str, Any]:
    if c0.get("schema_version") != C0_SCHEMA:
        raise ProspectiveCaptureError("C0 receipt schema mismatch")
    c0_body_sha = _verify_hashed_body(c0, "receipt_body_sha256", "C0 receipt")
    if c0.get("private_truth_access") is not False or c0.get("pa3_inference_authorized") is not False:
        raise ProspectiveCaptureError("C0 authority drift")
    try:
        armed = datetime.fromisoformat(armed_at_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProspectiveCaptureError("armed_at_utc must be ISO-8601") from error
    if armed.tzinfo is None or armed.utcoffset() is None:
        raise ProspectiveCaptureError("armed_at_utc must include timezone")
    episodes = c0.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < MINIMUM_EPISODES:
        raise ProspectiveCaptureError(f"C0 must freeze at least {MINIMUM_EPISODES} episodes")
    plan_rows = []
    seen = set()
    for row in episodes:
        if not isinstance(row, Mapping) or not isinstance(row.get("goal_provenance"), Mapping):
            raise ProspectiveCaptureError("C0 episode is malformed")
        episode_id = str(row.get("episode_id") or "")
        if not SAFE_EPISODE_ID.fullmatch(episode_id):
            raise ProspectiveCaptureError(f"unsafe episode_id: {episode_id}")
        if episode_id in seen:
            raise ProspectiveCaptureError(f"duplicate episode_id: {episode_id}")
        seen.add(episode_id)
        goal_at = _timestamp(row["goal_provenance"].get("goal_recorded_at_utc"), f"{episode_id} goal_recorded_at_utc")
        if not goal_at < armed:
            raise ProspectiveCaptureError(f"{episode_id} goal must precede capture arming")
        plan_rows.append({
            "episode_id": episode_id,
            "goal_text_original": str(row.get("goal_text_original") or "").strip(),
            "media_relative_path": f"{episode_id}.mp4",
            "camera_view": "FIRST_PERSON_FORWARD",
            "continuous_capture": True,
        })
        if not plan_rows[-1]["goal_text_original"]:
            raise ProspectiveCaptureError(f"{episode_id} goal text is missing")
    plan = {
        "schema_version": CAPTURE_PLAN_SCHEMA,
        "goal_receipt_body_sha256": c0_body_sha,
        "armed_at_utc": armed.isoformat(),
        "source_role": SOURCE_ROLE,
        "capture_instruction_id": CAPTURE_INSTRUCTION,
        "frame_selection_rule": "FIXED_OFFSETS_FROM_VIDEO_END_NO_PIXEL_OR_TRUTH_SELECTION",
        "frame_offsets_from_end_seconds": list(FRAME_OFFSETS_FROM_END_SECONDS),
        "truth_state_at_arming": "NOT_CREATED",
        "provider_state_at_arming": "NOT_STARTED",
        "capture_state_at_arming": "NOT_STARTED",
        "episode_count": len(plan_rows),
        "episodes": plan_rows,
        "claim_ceiling": "CAPTURE_ARMING_AND_PROVENANCE_MECHANICS_ONLY_NO_VISIBILITY_PROPOSAL_IDENTITY_PRODUCT_OR_SAFETY_CLAIM",
    }
    plan["capture_plan_body_sha256"] = content_sha256(plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ProspectiveCaptureError("capture plan already exists; plans are immutable")
    plan = arm_capture(json.loads(args.c0_receipt.read_text(encoding="utf-8")), datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"capture_plan": str(args.output), "capture_plan_body_sha256": plan["capture_plan_body_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
