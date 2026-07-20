"""Fuse relative score change with independent semantic-exit evidence.

This retrospective prototype keeps absolute scene classification out of the
lifecycle decision. A sufficiently large signed relative change can open or
close an event. A weak decrease can close only when the already-frozen causal
semantic-exit router independently identifies the same previous/current
episode boundary. Missing baselines, zero change and evidence conflicts remain
uncertain.

The 5% strong-change boundary comes from the published r7.14 post-result stress
grid. It is therefore diagnostic and cannot be called prospective calibration.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_pair_relative_lifecycle_probe as relative
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_dual_evidence_lifecycle_fusion_v1"
RETROSPECTIVE_CLOSE_SCHEMA = "blindassist_public_silver_retrospective_close_stress_v1"
EXIT_ROUTER_SCHEMA = "blindassist_public_silver_risk_profile_exit_router_v1"
STRONG_MARGIN = 0.05


def decide_transition(
    *,
    previous_state: str,
    normalized_signed_change: float,
    semantic_exit: bool,
    trusted_reference: bool,
    strong_margin: float = STRONG_MARGIN,
) -> dict[str, Any]:
    if previous_state not in {"clear", "risk"}:
        raise ValueError("previous state must be clear or risk")
    if not 0 < strong_margin < 1:
        raise ValueError("strong margin must be in (0,1)")
    if not trusted_reference:
        return {
            "predicted_transition": "uncertain",
            "next_state": "uncertain",
            "reason": "trusted_reference_missing",
        }

    if previous_state == "clear":
        if normalized_signed_change >= strong_margin and not semantic_exit:
            return {
                "predicted_transition": "open_event",
                "next_state": "risk",
                "reason": "strong_relative_increase",
            }
        return {
            "predicted_transition": "uncertain",
            "next_state": "uncertain",
            "reason": (
                "semantic_exit_conflicts_with_open_evidence"
                if semantic_exit and normalized_signed_change > 0
                else "insufficient_open_evidence"
            ),
        }

    if normalized_signed_change <= -strong_margin:
        return {
            "predicted_transition": "close_event",
            "next_state": "clear",
            "reason": (
                "strong_relative_decrease_with_semantic_support"
                if semantic_exit else "strong_relative_decrease"
            ),
        }
    if -strong_margin < normalized_signed_change < 0 and semantic_exit:
        return {
            "predicted_transition": "close_event",
            "next_state": "clear",
            "reason": "weak_relative_decrease_corroborated_by_semantic_exit",
        }
    return {
        "predicted_transition": "uncertain",
        "next_state": "uncertain",
        "reason": (
            "semantic_exit_conflicts_with_non_decreasing_risk"
            if semantic_exit and normalized_signed_change >= 0
            else "weak_decrease_without_independent_exit"
            if normalized_signed_change < 0
            else "no_close_evidence"
        ),
    }


def _validate_exit_router(report: dict[str, Any], *, expected_gap_ms: int) -> set[tuple[str, str, str]]:
    if report.get("schema") != EXIT_ROUTER_SCHEMA:
        raise ValueError("exit-router schema mismatch")
    contract = report.get("router_contract")
    if not isinstance(contract, dict) or contract.get("max_gap_ms") != expected_gap_ms:
        raise ValueError("exit-router gap contract mismatch")
    if contract.get("causal") is not True or contract.get("held_out_label_consumed_by_router") is not False:
        raise ValueError("exit router is not causal and label-independent")
    candidates: set[tuple[str, str, str]] = set()
    for row in report.get("exit_candidates", []):
        key = (row.get("source_id"), row.get("previous_episode_id"), row.get("episode_id"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("exit-router candidate key is incomplete")
        candidates.add(key)  # type: ignore[arg-type]
    return candidates


def evaluate(
    *,
    relative_report: dict[str, Any],
    retrospective_close_report: dict[str, Any],
    exit_router_report: dict[str, Any],
    gap_negative_control_report: dict[str, Any],
) -> dict[str, Any]:
    if relative_report.get("schema") != relative.SCHEMA:
        raise ValueError("pair-relative report schema mismatch")
    if not (relative_report.get("acceptance") or {}).get("passed"):
        raise ValueError("pair-relative direction gate is not passed")
    if retrospective_close_report.get("schema") != RETROSPECTIVE_CLOSE_SCHEMA:
        raise ValueError("retrospective close report schema mismatch")
    if not (retrospective_close_report.get("acceptance") or {}).get("passed"):
        raise ValueError("retrospective close stress gate is not passed")
    package_root = relative_report.get("package_root")
    if not isinstance(package_root, str):
        raise ValueError("pair-relative package root is missing")
    for label, report in (
        ("exit router", exit_router_report),
        ("gap negative control", gap_negative_control_report),
    ):
        if report.get("package_root") != package_root:
            raise ValueError(f"{label} package root mismatch")

    semantic_exits = _validate_exit_router(exit_router_report, expected_gap_ms=5000)
    negative_control_exits = _validate_exit_router(
        gap_negative_control_report, expected_gap_ms=1000
    )
    if not semantic_exits or negative_control_exits:
        raise ValueError("semantic exit or 1-second gap negative-control contract mismatch")

    transitions: list[dict[str, Any]] = []
    weak_close_row: dict[str, Any] | None = None
    for row in relative_report.get("transitions", []):
        scale = max(abs(float(row["earlier_score"])), abs(float(row["later_score"])), 1e-12)
        normalized_signed_change = float(row["signed_score_delta"]) / scale
        key = (row["source_id"], row["earlier_episode_id"], row["later_episode_id"])
        semantic_exit = key in semantic_exits
        decision = decide_transition(
            previous_state=str(row["earlier_state"]),
            normalized_signed_change=normalized_signed_change,
            semantic_exit=semantic_exit,
            trusted_reference=True,
        )
        item = {
            "case_id": row["counterfactual_pair_id"],
            "source_id": row["source_id"],
            "mechanism": row["mechanism"],
            "previous_state": row["earlier_state"],
            "expected_transition": row["expected_transition"],
            "normalized_signed_change": normalized_signed_change,
            "semantic_exit_evidence": semantic_exit,
            **decision,
        }
        item["correct"] = item["predicted_transition"] == item["expected_transition"]
        transitions.append(item)
        if decision["reason"] == "weak_relative_decrease_corroborated_by_semantic_exit":
            if weak_close_row is not None:
                raise ValueError("prototype expects exactly one weak corroborated close case")
            weak_close_row = item

    retrospective_transition = retrospective_close_report.get("transition")
    if not isinstance(retrospective_transition, dict):
        raise ValueError("retrospective close transition is missing")
    scale = max(
        abs(float(retrospective_transition["risk_score"])),
        abs(float(retrospective_transition["clear_score"])),
        1e-12,
    )
    retrospective_signed_change = float(retrospective_transition["signed_score_delta"]) / scale
    retrospective_decision = decide_transition(
        previous_state="risk",
        normalized_signed_change=retrospective_signed_change,
        semantic_exit=False,
        trusted_reference=True,
    )
    retrospective_item = {
        "case_id": "sk1-retrospective-dynamic-close-r715",
        "source_id": retrospective_close_report.get("source_id"),
        "mechanism": retrospective_close_report.get("mechanism"),
        "previous_state": "risk",
        "expected_transition": "close_event",
        "normalized_signed_change": retrospective_signed_change,
        "semantic_exit_evidence": False,
        **retrospective_decision,
    }
    retrospective_item["correct"] = retrospective_item["predicted_transition"] == "close_event"
    transitions.append(retrospective_item)
    if weak_close_row is None:
        raise ValueError("no weak semantic-corroborated close case was exercised")

    weak_change = float(weak_close_row["normalized_signed_change"])
    controls = [
        {
            "control_id": "remove_5s_semantic_exit_via_1s_gap_contract",
            **decide_transition(
                previous_state="risk",
                normalized_signed_change=weak_change,
                semantic_exit=False,
                trusted_reference=True,
            ),
            "expected_transition": "uncertain",
        },
        {
            "control_id": "semantic_exit_without_score_decrease",
            **decide_transition(
                previous_state="risk",
                normalized_signed_change=0.0,
                semantic_exit=True,
                trusted_reference=True,
            ),
            "expected_transition": "uncertain",
        },
        {
            "control_id": "semantic_exit_conflicts_with_rising_risk",
            **decide_transition(
                previous_state="risk",
                normalized_signed_change=0.20,
                semantic_exit=True,
                trusted_reference=True,
            ),
            "expected_transition": "uncertain",
        },
        {
            "control_id": "strong_decrease_without_trusted_reference",
            **decide_transition(
                previous_state="risk",
                normalized_signed_change=-0.80,
                semantic_exit=False,
                trusted_reference=False,
            ),
            "expected_transition": "uncertain",
        },
    ]
    for control in controls:
        control["passed"] = control["predicted_transition"] == control["expected_transition"]

    checks = {
        "all_retrospective_transitions_correct": all(row["correct"] for row in transitions),
        "both_mechanisms_represented": {
            row["mechanism"] for row in transitions
        } == {"dynamic_agent_approach", "static_corridor_narrowing"},
        "strong_relative_close_present": any(
            row["reason"] == "strong_relative_decrease" for row in transitions
        ),
        "weak_corroborated_close_present": any(
            row["reason"] == "weak_relative_decrease_corroborated_by_semantic_exit"
            for row in transitions
        ),
        "all_fail_closed_controls_passed": all(row["passed"] for row in controls),
    }
    return {
        "contract": {
            "strong_normalized_change_margin": STRONG_MARGIN,
            "strong_margin_provenance": "selected from the published r7.14 post-result stress grid; diagnostic only, not prospective calibration",
            "open_rule": "trusted clear reference plus relative increase >= strong margin and no conflicting semantic exit",
            "strong_close_rule": "trusted risk reference plus relative decrease >= strong margin",
            "weak_close_rule": "trusted risk reference plus 0 < relative decrease < strong margin plus exact causal semantic-exit boundary",
            "conflict_or_missing_reference": "uncertain",
            "absolute_scene_threshold_used": False,
            "learned_parameters": 0,
        },
        "transitions": transitions,
        "negative_controls": controls,
        "metrics": {
            "transition_count": len(transitions),
            "correct_transition_count": sum(bool(row["correct"]) for row in transitions),
            "transition_accuracy": sum(bool(row["correct"]) for row in transitions) / len(transitions),
            "close_event_count": sum(row["expected_transition"] == "close_event" for row in transitions),
            "strong_close_count": sum(
                str(row["reason"]).startswith("strong_relative_decrease")
                for row in transitions
            ),
            "weak_corroborated_close_count": sum(
                row["reason"] == "weak_relative_decrease_corroborated_by_semantic_exit"
                for row in transitions
            ),
        },
        "retrospective_acceptance": {**checks, "passed": all(checks.values())},
        "prospective_acceptance": {
            "passed": False,
            "reason": "fusion margin and weak-evidence rule were assembled after r7.14/r7.15 and have no new independent source challenge",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.relative_report, args.retrospective_close_report,
        args.exit_router_report, args.gap_negative_control_report, args.output,
    ):
        mil.reject_independent_direction(path)
    inputs = {
        "relative_report": args.relative_report.resolve(),
        "retrospective_close_report": args.retrospective_close_report.resolve(),
        "exit_router_report": args.exit_router_report.resolve(),
        "gap_negative_control_report": args.gap_negative_control_report.resolve(),
    }
    payloads = {key: lifecycle.verify_json_sidecar(path) for key, path in inputs.items()}
    result = evaluate(
        relative_report=payloads["relative_report"],
        retrospective_close_report=payloads["retrospective_close_report"],
        exit_router_report=payloads["exit_router_report"],
        gap_negative_control_report=payloads["gap_negative_control_report"],
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            key: {"path": str(path), "sha256": common.sha256_file(path)}
            for key, path in inputs.items()
        },
        **result,
        "isolation_contract": {
            "public_video_mainline_only": True,
            "independent_model_direction_data_used": False,
            "independent_model_direction_code_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
        "evidence_limit": "Post-hoc fusion of existing GPT/VLM provisional pairs, one retrospective close stress case, and an existing semantic router; not human truth, prospective validation, calibration, blind evaluation, or production evidence.",
        "training_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_integration_authorized": False,
        "production_authorized": False,
    }
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output or sidecar: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-report", type=Path, required=True)
    parser.add_argument("--retrospective-close-report", type=Path, required=True)
    parser.add_argument("--exit-router-report", type=Path, required=True)
    parser.add_argument("--gap-negative-control-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
