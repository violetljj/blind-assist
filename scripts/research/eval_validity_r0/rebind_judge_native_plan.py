"""Rebind an output-blind native RGB plan to an immutable cohort receipt.

This is a narrow calibration-only operation.  It does not fetch pixels or
change frame receipts.  It only accepts a target cohort when every event,
session, camera/lens tuple and contiguous source-frame window still matches
the existing plan; the output records the new cohort hash explicitly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .common import PROTOCOL_ID, read_json, sha256_json
from .freeze_screening_cohort import SCHEMA as SCREENING_COHORT_SCHEMA
from .materialize_screening_inputs import PLAN_SCHEMA


class PlanRebindError(ValueError):
    """Raised when an existing plan cannot be safely rebound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanRebindError(message)


def rebind(*, plan: dict[str, Any], cohort: dict[str, Any], output: Path, source_plan_path: Path, target_cohort_path: Path) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite rebound plan: {output}")
    _require(plan.get("schema_version") == PLAN_SCHEMA and plan.get("protocol_id") == PROTOCOL_ID, "source plan schema/protocol mismatch")
    _require(plan.get("status") == "CONTINUOUS_NATIVE_ASSET_PLAN_FROZEN", "source plan is not frozen")
    _require(plan.get("candidate_outputs_opened") is False, "source plan is output-contaminated")
    _require(cohort.get("schema_version") == SCREENING_COHORT_SCHEMA and cohort.get("protocol_id") == PROTOCOL_ID, "target cohort schema/protocol mismatch")
    _require(cohort.get("status") == "OUTPUT_BLIND_SCREENING_COHORT_CONTINUOUS_WINDOWS_FROZEN", "target cohort is not the frozen continuous-window cohort")
    plan_items = {item.get("screening_event_id"): item for item in plan.get("items", []) if isinstance(item, dict)}
    cohort_items = {item.get("screening_event_id"): item for item in cohort.get("items", []) if isinstance(item, dict)}
    _require(len(plan_items) == 48 and len(cohort_items) == 48, "plan/cohort must each contain 48 events")
    _require(set(plan_items) == set(cohort_items), "target cohort event set differs from native plan")
    for event_id, cohort_item in cohort_items.items():
        plan_item = plan_items[event_id]
        _require(plan_item.get("source_session_id") == cohort_item.get("source_session_id"), f"{event_id}: source session changed")
        _require(plan_item.get("camera") == cohort_item.get("camera") and plan_item.get("lens") == cohort_item.get("lens"), f"{event_id}: camera/lens changed")
        window = cohort_item.get("source_window")
        frames = plan_item.get("frames")
        _require(isinstance(window, dict) and isinstance(frames, list) and frames, f"{event_id}: source window or plan frames missing")
        _require(len(frames) == window.get("frame_count") == 60, f"{event_id}: frame count changed")
        _require(frames[0].get("source_frame_index") == window.get("start_frame"), f"{event_id}: source window start changed")
        _require([frame.get("source_frame_index") for frame in frames] == list(range(window["start_frame"], window["start_frame"] + window["frame_count"])), f"{event_id}: native frame sequence changed")

    result = copy.deepcopy(plan)
    target_hash = sha256_json(cohort)
    result["screening_cohort_sha256"] = target_hash
    result["input_sha256"] = {"screening_cohort": target_hash}
    result["rebind_receipt"] = {
        "operation": "CALIBRATION_ONLY_NATIVE_PLAN_REBIND",
        "source_plan_sha256": sha256_json(plan),
        "source_plan_path": str(source_plan_path),
        "target_cohort_sha256": target_hash,
        "target_cohort_path": str(target_cohort_path),
        "pixel_payload_changed": False,
        "frame_receipts_changed": False,
        "candidate_outputs_opened": False,
        "reviewer_packet_materialized": False,
        "formal_denominator_inclusion": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "CALIBRATION_NATIVE_PLAN_REBOUND",
        "source_plan_sha256": sha256_json(plan),
        "target_cohort_sha256": target_hash,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--target-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rebind(
        plan=read_json(args.source_plan),
        cohort=read_json(args.target_cohort),
        output=args.output,
        source_plan_path=args.source_plan,
        target_cohort_path=args.target_cohort,
    )
    print(f"status={result['status']} target_cohort_sha256={result['target_cohort_sha256']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
