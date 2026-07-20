#!/usr/bin/env python3
"""Compose DINOv2 general-static and chromatic-marker evidence offline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_multiexpert_risk_profile_prototype_v1"


def route_profile(dino_result: dict[str, Any], chromatic_result: dict[str, Any]) -> dict[str, Any]:
    dino_eval = dino_result.get("evaluation", {})
    dino_passed = dino_result.get("prospective_pair_gate", {}).get("passed") is True
    chromatic_acceptance = chromatic_result.get("acceptance", {})
    chromatic_passed = chromatic_acceptance.get("passed") is True
    chromatic_terminal_clear = (
        chromatic_result.get("lifecycle", {}).get("terminal_state") == "clear"
        and chromatic_result.get("lifecycle", {}).get("open_event") is None
    )
    selected: list[str] = []
    if dino_passed:
        selected.append("general_static_dinov2")
    if chromatic_passed:
        selected.append("chromatic_construction_marker")
    event_present = bool(selected)
    event_closed = bool(
        event_present
        and (not dino_passed or dino_eval.get("close_ordered") is True)
        and (not chromatic_passed or chromatic_terminal_clear)
    )
    return {
        "channels": {
            "general_static_dinov2": {
                "positive": dino_passed,
                "open_projection": dino_eval.get("open_projection"),
                "close_projection": dino_eval.get("close_projection"),
                "close_confirmed": dino_eval.get("close_ordered") is True,
            },
            "chromatic_construction_marker": {
                "positive": chromatic_passed,
                "risk_active_fraction": chromatic_result.get("diagnostics", {}).get("risk_present_window", {}).get("active_fraction"),
                "stable_clear_active_fraction": chromatic_result.get("diagnostics", {}).get("stable_clear_window", {}).get("active_fraction"),
                "terminal_clear": chromatic_terminal_clear,
            },
        },
        "selected_positive_channels": selected,
        "event_present": event_present,
        "event_closed": event_closed,
        "terminal_state": "clear" if event_closed else ("present" if event_present else "uncertain"),
        "fusion_rule": "independent positive-evidence OR; every channel that opened must independently confirm close; absence alone never clears",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dinov2_result, args.chromatic_result, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    dino_result = lifecycle.verify_json_sidecar(args.dinov2_result.resolve())
    chromatic_result = lifecycle.verify_json_sidecar(args.chromatic_result.resolve())
    if dino_result.get("source", {}).get("source_id") != chromatic_result.get("source_id"):
        raise ValueError("expert results refer to different sources")
    if dino_result.get("inputs", {}).get("video_sha256") != chromatic_result.get("video_sha256"):
        raise ValueError("expert results refer to different video bytes")
    profile = route_profile(dino_result, chromatic_result)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "posthoc_multiexpert_composition_after_japan_r722",
        "source_id": chromatic_result["source_id"],
        "video_sha256": chromatic_result["video_sha256"],
        "inputs": {"dinov2_result_sha256": common.sha256_file(args.dinov2_result), "chromatic_result_sha256": common.sha256_file(args.chromatic_result)},
        "risk_profile": profile,
        "retrospective_mechanism_coverage_passed": profile["event_present"] and profile["event_closed"],
        "pixel_segmentation_role": "auxiliary_only",
        "lifecycle_head": {"states": ["clear", "present", "uncertain"], "absence_is_clear": False, "channel_specific_close_required": True},
        "prospective_acceptance": False,
        "prospective_contract_freeze_authorized": profile["event_present"] and profile["event_closed"],
        "evidence_limit": "The fusion rule was assembled after opening the Japan source. It may freeze a new prospective contract but cannot retroactively turn Japan into a fusion pass or authorize deployment.",
        "training_authorized": False, "calibration_authorized": False, "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2-result", type=Path, required=True)
    parser.add_argument("--chromatic-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "channels": payload["risk_profile"]["selected_positive_channels"], "event_closed": payload["risk_profile"]["event_closed"], "contract_freeze_authorized": payload["prospective_contract_freeze_authorized"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
