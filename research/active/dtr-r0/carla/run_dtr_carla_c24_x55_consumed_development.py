"""Run X56 once on consumed C24 and score Development."""

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

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x32_observation_conditioned_core_predictor as x32  # noqa: E402
import dtr_carla_x56_zero_eligible_fusion_metric_handback as x56  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X54 = x56.x55.x54.ARM_X54
ARM_X56 = x56.ARM_X56
EXPECTED_PROTOCOL_SHA256 = (
    "7767473E7EF9EEE7445E915EC2EF095F496BF1E1CB3524D957FC131555CD260B"
)


def prediction_envelope(episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-c24-x56-consumed-development-predictions-v1",
        "status": "SEALED_BEFORE_CONSUMED_SCORE",
        "experiment_id": x56.EXPERIMENT_ID,
        "truth_blind_prediction_inputs": True,
        "arms": [ARM_X56],
        "episodes": episodes,
        "fixed_constants": x56.fixed_constants(),
        "claim_boundary": {
            "consumed_c24_posthoc_development": True,
            "c24_x54_confirmation_result_unchanged": True,
            "x55_and_x56_posthoc_development": True,
            "fresh_confirmation": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    aliased = copy.deepcopy(predictions)
    for episode in aliased["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X56]
    return aliased


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--x54-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    x54_dir = args.x54_dir.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    runner.base.require(
        runner.base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c24_protocol_hash_drift",
    )
    protocol = runner.base.read_json(protocol_path)
    _, contract, candidate_values = x24.require_freeze(run_root)
    x54_predictions_path = x54_dir / "predictions-x54.json"
    x54_summary_path = x54_dir / "summary.json"
    x54_predictions = runner.base.read_json(x54_predictions_path)
    x54_summary = runner.base.read_json(x54_summary_path)
    runner.base.require(
        x54_summary.get("decision")
        == "DTR_CARLA_C24_X54_MECHANISM_NOT_EXERCISED",
        "c24_x54_terminal_result_drift",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / "predictions-x56.json"
    summary_path = output_dir / "summary.json"

    # Preserve the exact observation-conditioned core used by sealed X54.
    x56.x55.x45.x44.x43.x42.x32 = x32
    x56.x55.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    episodes: dict[str, Any] = {}
    cursor = 0
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = x56.predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
        print(f"predicted_once {episode.episode_id}", flush=True)
    runner.base.require(cursor == len(candidate_values), "c24_x56_candidate_cursor")
    predictions = prediction_envelope(episodes)
    runner.base.write_json_exclusive(predictions_path, predictions)

    # C24 truth is already consumed, but X56 predictions are still sealed first.
    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    envelopes = {
        ARM_X24: x24_predictions,
        ARM_X54: x54_predictions,
        ARM_X56: predictions,
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
        episode_id: {
            key: value
            for key, value in episodes[episode_id]["diagnostics"].items()
            if key.startswith(("x52_", "x53_", "x54_", "x55_", "x56_"))
        }
        for episode_id in runner.EPISODES
    }
    mechanism_counts = {
        "x52_provisional_parent_changes": sum(
            int(value.get("x52_provisional_parent_changes", 0))
            for value in diagnostics.values()
        ),
        "x53_anchor_redundancy_suppressions": sum(
            int(value.get("x53_anchor_redundancy_suppressions", 0))
            for value in diagnostics.values()
        ),
        "x54_dropout_continuations": sum(
            int(value.get("x54_metric_bootstrap_dropout_continuation_frames", 0))
            for value in diagnostics.values()
        ),
        "x55_parent_sibling_consensus_frames": sum(
            int(value.get("x55_parent_sibling_consensus_frames", 0))
            for value in diagnostics.values()
        ),
        "x56_zero_eligible_metric_handback_frames": sum(
            int(value.get("x56_zero_eligible_metric_handback_frames", 0))
            for value in diagnostics.values()
        ),
    }
    x24_metrics = aggregate[ARM_X24]
    x54_metrics = aggregate[ARM_X54]
    x56_metrics = aggregate[ARM_X56]
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X56]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    contact_recall = {
        episode_id: float(
            contacts[episode_id][ARM_X56]["future_positive_recall"]
        )
        for episode_id in runner.CONTACT
    }
    continuous_count = sum(
        int(value["continuous_route_risk"]) for value in continuity.values()
    )
    ancestry_count = sum(
        int(value["parent_ancestry_status"] == "PRESERVED")
        for value in continuity.values()
    )
    gates = protocol["c24_x54_preregistration"]["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    reference_checks = {
        "precision_at_least_0_80": float(x56_metrics["precision"]) + epsilon
        >= float(gates["minimum_precision"]),
        "recall_at_least_0_70": float(x56_metrics["recall"]) + epsilon
        >= float(gates["minimum_recall"]),
        "f1_at_least_0_76": float(x56_metrics["f1"]) + epsilon
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
        "x53_suppressions_at_least_1": mechanism_counts[
            "x53_anchor_redundancy_suppressions"
        ]
        >= 1,
        "x54_continuations_at_least_1": mechanism_counts[
            "x54_dropout_continuations"
        ]
        >= 1,
        "x55_consensus_frames_at_least_1": mechanism_counts[
            "x55_parent_sibling_consensus_frames"
        ]
        >= 1,
        "x56_handback_frames_at_least_1": mechanism_counts[
            "x56_zero_eligible_metric_handback_frames"
        ]
        >= 1,
        "continuous_contact_episodes_at_least_3": continuous_count >= 3,
        "parent_ancestry_episodes_at_least_3": ancestry_count >= 3,
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    reference_target_met = all(reference_checks.values())
    decision = (
        "DTR_CARLA_C24_X56_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_MET"
        if reference_target_met
        else "DTR_CARLA_C24_X56_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_NOT_MET"
    )
    summary = {
        "schema": "blindassist-dtr-carla-c24-x56-consumed-development-v1",
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
        "mechanism_counts": mechanism_counts,
        "mechanism_diagnostics_by_episode": diagnostics,
        "x56_effect_vs_x54": {
            "tp_delta": int(x56_metrics["tp"] - x54_metrics["tp"]),
            "fp_delta": int(x56_metrics["fp"] - x54_metrics["fp"]),
            "f1_delta": float(x56_metrics["f1"] - x54_metrics["f1"]),
        },
        "x56_effect_vs_x24": {
            "tp_delta": int(x56_metrics["tp"] - x24_metrics["tp"]),
            "fp_delta": int(x56_metrics["fp"] - x24_metrics["fp"]),
            "f1_delta": float(x56_metrics["f1"] - x24_metrics["f1"]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source": {
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "x54_predictions_sha256": runner.base.sha256_file(
                x54_predictions_path
            ),
            "x54_summary_sha256": runner.base.sha256_file(x54_summary_path),
            "x56_predictor_sha256": runner.base.sha256_file(
                Path(x56.__file__).resolve()
            ),
            "development_runner_sha256": runner.base.sha256_file(
                Path(__file__).resolve()
            ),
            "x56_predictions_sha256": runner.base.sha256_file(predictions_path),
        },
        "claim_boundary": {
            "consumed_c24_posthoc_development": True,
            "c24_x54_frozen_decision_unchanged": True,
            "reference_thresholds_not_preregistered_for_x55_or_x56": True,
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
                "mechanism_counts": mechanism_counts,
                "effect_vs_x54": summary["x56_effect_vs_x54"],
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
