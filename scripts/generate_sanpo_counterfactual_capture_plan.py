#!/usr/bin/env python3
"""Generate a non-evidentiary autonomous-acquisition plan for counterfactual episodes."""

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


def build_capture_plan(config: dict[str, Any], *, pilot: bool = False) -> dict[str, Any]:
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

    pilot_policy = None
    if pilot:
        pilot_policy = design.get("pilot_before_full_matrix")
        if not isinstance(pilot_policy, dict) or pilot_policy.get("authority") != "collection-pipeline-audit-only":
            raise PlanError("pilot mode requires collection-pipeline-audit-only policy")
        pilot_session_count = pilot_policy.get("session_count")
        pilot_pair_count = pilot_policy.get("matched_pairs_per_scene")
        if not isinstance(pilot_session_count, int) or not 1 <= pilot_session_count <= len(session_ids):
            raise PlanError("pilot session count is invalid")
        if not isinstance(pilot_pair_count, int) or pilot_pair_count <= 0:
            raise PlanError("pilot matched-pair count is invalid")
        _text(pilot_policy.get("contract_id"), "pilot contract_id")
        _text(pilot_policy.get("origin_scope"), "pilot origin_scope")
        session_ids = session_ids[:pilot_session_count]
        pair_count = pilot_pair_count

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
                        "status": "awaiting_autonomous_acquisition",
                        "origin_scope": pilot_policy.get("origin_scope") if pilot else config.get("source_receipt_schema", {}).get("required_origin_scope"),
                        "session_id": session_id,
                        "scene_id": scene_id,
                        "matched_pair_id": pair_id,
                        "pair_role": role,
                        "capture_contract": contract,
                        "duration_ms_required_range": [minimum, maximum],
                        "must_share_with_pair": context_fields,
                        "risk_profile_template": profile,
                        "lifecycle_intervals_template": lifecycle,
                        "ai_event_adjudication_required": True,
                        "human_operator_required": False,
                        "human_fallback_forbidden": True,
                        "autonomous_acquisition_priority": [
                            "licensed_public_source_agent",
                            "automated_device_capture",
                            "simulation_or_synthetic_generation",
                            "model_generation_with_provenance",
                        ],
                        "evidence_requirements": {
                            "source_receipt": "agent-produced hash-bound license/consent reference, automated privacy audit, raw input and inventory",
                            "capture_clock_receipt": "nanosecond monotonic camera timestamps bound to the frame ledger",
                            "capture_frame_ledger": "ordered frame IDs, capture timestamps, video PTS and payload SHA256 bound to video/clock/route",
                            "explicit_route": "runtime-eligible current-camera route samples bound to the same frame ledger; no future-video oracle",
                            "annotation": "isolated GPT and Codex reviews plus hash-bound consensus or a fresh third-model adjudication",
                        },
                    })
    expected = len(session_ids) * len(scenes) * pair_count * 2
    if len(slots) != expected:
        raise PlanError("capture-plan slot count does not match config")
    if pilot and pilot_policy.get("episode_count") != len(slots):
        raise PlanError("pilot episode count does not match generated slots")
    return {
        "format": "blindassist_sanpo_counterfactual_capture_plan_v1",
        "contract_id": pilot_policy.get("contract_id") if pilot else config.get("contract_id"),
        "source_truth_contract_id": config.get("contract_id") if pilot else None,
        "collection_scope": "pipeline_audit_pilot" if pilot else config.get("collection_scope"),
        "status": "pilot_autonomous_acquisition_plan_only" if pilot else "autonomous_acquisition_plan_only",
        "pilot": pilot,
        "authority": pilot_policy.get("authority") if pilot else "full-matrix-collection-plan-only",
        "episode_slot_count": len(slots),
        "matched_pair_slot_count": len(slots) // 2,
        "route_conditioned_truth_eligible": False,
        "u0_evaluation_eligible": False,
        "s0_probe_eligible": False,
        "training_eligible": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
        "important_limit": "Slots are autonomous-agent instructions, not acquired evidence, labels, receipts, or training data.",
        "slots": slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_capture_plan(json.loads(args.config.read_text(encoding="utf-8")), pilot=args.pilot)
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
