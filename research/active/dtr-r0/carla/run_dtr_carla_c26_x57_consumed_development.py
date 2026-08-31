"""Replay X57 composition once on consumed C26 Development evidence."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORK = Path(r"E:\linnan\linnan\artifacts.local\work\x31-growth-diagnostic-20260831")
for value in (HERE, WORK):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import dtr_carla_x57_retained_core_metric_handback as x57  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X54 = x57.x54.ARM_X54
ARM_X57 = x57.ARM_X57
EXPECTED_PROTOCOL_SHA256 = (
    "37335C08F91A6A3CC360094BBC672E7350484618D4E91C0E40E052F830D005FA"
)
EXPECTED_C26_DECISION = "DTR_CARLA_C26_X56_MECHANISM_NOT_EXERCISED"


def prediction_envelope(episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-c26-x57-consumed-development-predictions-v1",
        "status": "SEALED_BEFORE_CONSUMED_SCORE",
        "experiment_id": x57.EXPERIMENT_ID,
        "truth_blind_component_inputs": True,
        "arms": [ARM_X57],
        "episodes": episodes,
        "fixed_constants": x57.fixed_constants(),
        "claim_boundary": {
            "consumed_c26_posthoc_development": True,
            "c26_x56_confirmation_result_unchanged": True,
            "fresh_confirmation": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    aliased = copy.deepcopy(predictions)
    for episode in aliased["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X57]
    return aliased


def effect(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float | int]:
    return {
        "tp_delta": int(challenger["tp"] - baseline["tp"]),
        "fp_delta": int(challenger["fp"] - baseline["fp"]),
        "f1_delta": float(challenger["f1"] - baseline["f1"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--c26-confirmation-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    confirmation_dir = args.c26_confirmation_dir.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    runner.base.require(
        runner.base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c26_protocol_hash_drift",
    )
    protocol = runner.base.read_json(protocol_path)
    c26_summary_path = confirmation_dir / "summary.json"
    c26_summary = runner.base.read_json(c26_summary_path)
    runner.base.require(
        c26_summary.get("decision") == EXPECTED_C26_DECISION,
        "c26_terminal_result_drift",
    )

    x24_predictions_path = run_root / "predictions-x24.json"
    x54_predictions_path = confirmation_dir / "predictions-x54.json"
    x24_predictions = runner.base.read_json(x24_predictions_path)
    x54_predictions = runner.base.read_json(x54_predictions_path)
    runner.base.require(
        c26_summary["source"]["x24_predictions_sha256"]
        == runner.base.sha256_file(x24_predictions_path)
        and c26_summary["source"]["x54_predictions_sha256"]
        == runner.base.sha256_file(x54_predictions_path),
        "c26_sealed_prediction_drift",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / "predictions-x57.json"
    summary_path = output_dir / "summary.json"

    episodes = {
        episode_id: x57.apply_retained_core_metric_handback_episode(
            x54_predictions["episodes"][episode_id],
            x24_predictions["episodes"][episode_id],
        )
        for episode_id in runner.EPISODES
    }
    predictions = prediction_envelope(episodes)
    runner.base.write_json_exclusive(predictions_path, predictions)

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {
        ARM_X24: x24_predictions,
        ARM_X54: x54_predictions,
        ARM_X57: predictions,
    }
    predictions_full = {
        arm: {
            episode_id: runner.shared.c7.arm_frames_full(
                envelope, episode_id, arm
            )
            for episode_id in runner.EPISODES
        }
        for arm, envelope in envelopes.items()
    }
    for arm, values in predictions_full.items():
        for episode_id in runner.EPISODES:
            runner.base.align(
                evaluator_full[episode_id], values[episode_id], f"{arm}:{episode_id}"
            )
    evaluator = {
        episode_id: runner.shared.c7.prefix(rows, runner.SCORE_END[episode_id])
        for episode_id, rows in evaluator_full.items()
    }
    scored = {
        arm: {
            episode_id: runner.shared.c7.prefix(rows, runner.SCORE_END[episode_id])
            for episode_id, rows in values.items()
        }
        for arm, values in predictions_full.items()
    }
    aggregate = {
        arm: runner.base.confusion(evaluator, values)
        for arm, values in scored.items()
    }
    contacts = {
        episode_id: {
            arm: runner.base.contact_metrics(
                evaluator[episode_id], values[episode_id]
            )
            for arm, values in scored.items()
        }
        for episode_id in runner.CONTACT
    }
    safe = {
        episode_id: {
            arm: runner.base.false_segments(
                values[episode_id], runner.SAFE_START[episode_id]
            )
            for arm, values in scored.items()
        }
        for episode_id in runner.SAFE
    }
    selected = runner.shared.validate_occlusion_reports(
        protocol,
        runner.base.read_json(
            source_root / "evaluator" / "physical_occlusion_report.json"
        ),
        evaluator_full,
    )
    scoring_envelope = alias_for_transport_score(predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(
        scoring_envelope, selected
    )
    invariants = runner.shared.authority_invariants(
        scoring_envelope, runner.SCORE_END
    )

    diagnostics = {
        episode_id: episodes[episode_id]["diagnostics"]
        for episode_id in runner.EPISODES
    }
    handback_frames = sum(
        int(value.get("x57_zero_eligible_metric_handback_frames", 0))
        for value in diagnostics.values()
    )
    x24_metrics = aggregate[ARM_X24]
    x54_metrics = aggregate[ARM_X54]
    x57_metrics = aggregate[ARM_X57]
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X57]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    contact_recall = {
        episode_id: float(
            contacts[episode_id][ARM_X57]["future_positive_recall"]
        )
        for episode_id in runner.CONTACT
    }
    gates = protocol["c26_x56_preregistration"]["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    reference_checks = {
        "precision_at_least_0_85": float(x57_metrics["precision"]) + epsilon
        >= float(gates["minimum_precision"]),
        "recall_at_least_0_70": float(x57_metrics["recall"]) + epsilon
        >= float(gates["minimum_recall"]),
        "f1_at_least_0_78": float(x57_metrics["f1"]) + epsilon
        >= float(gates["minimum_f1"]),
        "each_contact_recall_at_least_0_55": all(
            value + epsilon >= float(gates["minimum_each_contact_recall"])
            for value in contact_recall.values()
        ),
        "each_safe_episode_has_at_most_4_segments": all(
            value <= int(gates["maximum_safe_false_alert_segments_per_episode"])
            for value in safe_counts.values()
        ),
        "total_safe_segments_at_most_10": sum(safe_counts.values())
        <= int(gates["maximum_total_safe_false_alert_segments"]),
        "x57_handback_frames_at_least_1": handback_frames >= 1,
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    reference_target_met = all(reference_checks.values())
    decision = (
        "DTR_CARLA_C26_X57_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_MET"
        if reference_target_met
        else "DTR_CARLA_C26_X57_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_NOT_MET"
    )
    summary = {
        "schema": "blindassist-dtr-carla-c26-x57-consumed-development-v1",
        "status": "COMPLETE",
        "decision": decision,
        "reference_target_met": reference_target_met,
        "reference_checks": reference_checks,
        "reference_thresholds": gates,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "authority_invariants": invariants,
        "x57_handback_frames": handback_frames,
        "x57_effect_vs_x54": effect(x57_metrics, x54_metrics),
        "x57_effect_vs_x24": effect(x57_metrics, x24_metrics),
        "elapsed_seconds": time.perf_counter() - started,
        "source": {
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "c26_summary_sha256": runner.base.sha256_file(c26_summary_path),
            "x24_predictions_sha256": runner.base.sha256_file(
                x24_predictions_path
            ),
            "x54_predictions_sha256": runner.base.sha256_file(
                x54_predictions_path
            ),
            "x57_predictor_sha256": runner.base.sha256_file(
                Path(x57.__file__).resolve()
            ),
            "x57_predictions_sha256": runner.base.sha256_file(predictions_path),
            "development_runner_sha256": runner.base.sha256_file(
                Path(__file__).resolve()
            ),
        },
        "claim_boundary": {
            "consumed_c26_posthoc_development": True,
            "c26_x56_frozen_decision_unchanged": True,
            "reference_thresholds_not_preregistered_for_x57": True,
            "fresh_confirmation": False,
            "synthetic_development_only": True,
            "real_world_confirmation": False,
            "deployment_or_safety_authority": False,
        },
    }
    runner.base.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "reference_target_met": reference_target_met,
                "aggregate": aggregate,
                "contact_recall": contact_recall,
                "safe_segments": safe_counts,
                "x57_handback_frames": handback_frames,
                "effect_vs_x54": summary["x57_effect_vs_x54"],
                "reference_checks": reference_checks,
                "summary_sha256": runner.base.sha256_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
