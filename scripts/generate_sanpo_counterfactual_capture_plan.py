#!/usr/bin/env python3
"""Generate a non-evidentiary capture plan for counterfactual episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class PlanError(ValueError):
    pass


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} must be a non-empty string")
    return value


def build_capture_plan(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "blindassist_sanpo_counterfactual_episode_collection_v1":
        raise PlanError("unexpected collection config schema")
    design = config.get("design")
    duration = config.get("episode_duration_policy")
    sessions = config.get("sessions")
    scenes = config.get("scenes")
    matrix = config.get("matrix_contract")
    if not all(isinstance(item, dict) for item in (design, duration, matrix)) or not isinstance(sessions, list) or not isinstance(scenes, list):
        raise PlanError("config is missing capture-plan fields")
    pair_count = design.get("matched_pairs_per_session_scene")
    minimum, maximum = duration.get("minimum_duration_ms"), duration.get("maximum_duration_ms")
    context_fields = matrix.get("matched_pair_members_must_share_capture_context")
    if not isinstance(pair_count, int) or pair_count <= 0:
        raise PlanError("matched pair count must be positive")
    if not isinstance(minimum, int) or not isinstance(maximum, int) or minimum <= 0 or maximum < minimum:
        raise PlanError("episode duration policy is invalid")
    if not isinstance(context_fields, list) or not context_fields or not all(isinstance(value, str) and value for value in context_fields):
        raise PlanError("capture context fields are invalid")
    session_ids = [_text(item.get("session_id") if isinstance(item, dict) else None, "sessions.session_id") for item in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise PlanError("session IDs must be unique")

    slots: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            raise PlanError("scene must be an object")
        scene_id = _text(scene.get("scene_id"), "scenes.scene_id")
        positive = _text(scene.get("positive_contract"), f"{scene_id}.positive_contract")
        negative = _text(scene.get("matched_negative_contract"), f"{scene_id}.matched_negative_contract")
        for session_id in session_ids:
            for pair_index in range(1, pair_count + 1):
                pair_id = f"{session_id}__{scene_id}__pair_{pair_index:02d}"
                records = (
                    ("positive", positive, {"primary_hazard_type": scene_id, "corridor_relation": "enters_or_blocks", "lifecycle": "approach_alertable_clear"}, {"approach": ["first_visible_ms", "alertable_start_ms"], "alertable": ["alertable_start_ms", "passed_or_cleared_ms"], "post_event": ["passed_or_cleared_ms", "duration_ms"]}),
                    ("matched_negative", negative, {"primary_hazard_type": scene_id, "corridor_relation": "outside_or_nonblocking", "lifecycle": "no_alert"}, {"non_alert": [0, "duration_ms"]}),
                )
                for role, contract, profile, lifecycle in records:
                    slots.append({
                        "slot_id": f"{pair_id}__{role}",
                        "status": "not_captured",
                        "session_id": session_id,
                        "scene_id": scene_id,
                        "matched_pair_id": pair_id,
                        "pair_role": role,
                        "capture_contract": contract,
                        "duration_ms_required_range": [minimum, maximum],
                        "must_share_with_pair": context_fields,
                        "risk_profile_template": profile,
                        "lifecycle_intervals_template": lifecycle,
                        "human_event_adjudication_required": True,
                    })
    expected = len(session_ids) * len(scenes) * pair_count * 2
    if len(slots) != expected:
        raise PlanError("capture-plan slot count does not match config")
    return {
        "format": "blindassist_sanpo_counterfactual_capture_plan_v1",
        "status": "collection_plan_only",
        "episode_slot_count": len(slots),
        "matched_pair_slot_count": len(slots) // 2,
        "training_eligible": False,
        "production_model_replacement_authorized": False,
        "important_limit": "Slots are empty instructions, not captured evidence, labels, receipts, or training data.",
        "slots": slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = build_capture_plan(json.loads(args.config.read_text(encoding="utf-8")))
        if args.output.exists():
            raise PlanError("refusing to overwrite existing capture plan")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, PlanError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "output": str(args.output.resolve()), "episode_slot_count": plan["episode_slot_count"], "training_eligible": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
