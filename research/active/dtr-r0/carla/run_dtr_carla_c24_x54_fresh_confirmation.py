"""Run frozen X54 once on fresh C24, then open and score the evaluator once."""

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
import dtr_carla_x54_metric_bootstrap_dropout_continuation as x54  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X54 = x54.ARM_X54
EXPECTED_PROTOCOL_SHA256 = (
    "7767473E7EF9EEE7445E915EC2EF095F496BF1E1CB3524D957FC131555CD260B"
)
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"


def prediction_envelope(episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-c24-x54-fresh-predictions-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SINGLE_SCORE",
        "experiment_id": x54.EXPERIMENT_ID,
        "truth_blind_prediction_inputs": True,
        "arms": [ARM_X54],
        "episodes": episodes,
        "fixed_constants": x54.fixed_constants(),
        "claim_boundary": {
            "fresh_c24_source_corrected_generalization": True,
            "single_x54_scored_invocation": True,
            "threshold_or_scenario_tuning_after_capture": False,
            "evaluator_opened_during_prediction": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def require_frozen_inputs(
    protocol: dict[str, Any], protocol_path: Path, run_root: Path
) -> tuple[Any, Any, list[dict[str, Any]]]:
    runner.base.require(
        runner.base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c24_protocol_hash_drift",
    )
    prereg = protocol["c24_x54_preregistration"]
    runner.base.require(
        prereg["frozen_component_sha256"][Path(x54.__file__).name]
        == runner.base.sha256_file(Path(x54.__file__).resolve()),
        "c24_x54_predictor_hash_drift",
    )
    runner.base.require(
        prereg["single_x54_scored_invocation"] is True
        and prereg["no_post_capture_algorithm_threshold_or_scenario_tuning"] is True,
        "c24_preregistration_contract",
    )
    freeze, contract, candidate_values = x24.require_freeze(run_root)
    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    runner.base.require(
        x24_predictions.get("status") == "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "x24_predictions_not_sealed",
    )
    runner.base.require(
        x24_predictions["source"]["freeze_sha256"]
        == runner.base.sha256_file(run_root / "freeze-x24.json"),
        "x24_prediction_freeze_drift",
    )
    return freeze, contract, candidate_values


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    aliased = copy.deepcopy(predictions)
    for episode in aliased["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X54]
    return aliased


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = runner.base.read_json(protocol_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / "predictions-x54.json"
    summary_path = output_dir / "summary.json"

    freeze, contract, candidate_values = require_frozen_inputs(
        protocol, protocol_path, run_root
    )

    # Preserve the exact observation-conditioned core used by the frozen chain.
    x54.x53.x52.x45.x44.x43.x42.x32 = x32
    x54.x53.x52.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    started = time.perf_counter()
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        values = candidate_values[cursor : cursor + count]
        episodes[episode.episode_id] = x54.predict_episode(
            episode, values, contract.calibration
        )
        cursor += count
        print(f"predicted_once {episode.episode_id}", flush=True)
    runner.base.require(cursor == len(candidate_values), "c24_candidate_cursor")

    predictions = prediction_envelope(episodes)
    runner.base.write_json_exclusive(predictions_path, predictions)

    # Evaluator-bearing material is opened only after frozen X54 predictions exist.
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS,
        "c24_source_incomplete",
    )
    runner.base.require(
        bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "c24_source_gate_failed",
    )
    runner.base.require(
        source_result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "c24_source_protocol_drift",
    )
    runner.base.require(
        int(source_result.get("episode_count", 0)) == len(runner.EPISODES)
        and int(source_result.get("layout_count", 0)) == 4,
        "c24_source_cohort_count",
    )
    runner.base.require(
        runner.base.sha256_file(source_root / "model" / "manifest.json")
        == freeze["model_manifest"]["sha256"],
        "c24_source_model_manifest_drift",
    )

    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {ARM_X24: x24_predictions, ARM_X54: predictions}
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
            if key.startswith(("x52_", "x53_", "x54_"))
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
    }
    x24_metrics = aggregate[ARM_X24]
    x54_metrics = aggregate[ARM_X54]
    effect = {
        "tp_delta_vs_x24": int(x54_metrics["tp"] - x24_metrics["tp"]),
        "fp_delta_vs_x24": int(x54_metrics["fp"] - x24_metrics["fp"]),
        "f1_delta_vs_x24": float(x54_metrics["f1"] - x24_metrics["f1"]),
    }
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X54]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    contact_recall = {
        episode_id: float(
            contacts[episode_id][ARM_X54]["future_positive_recall"]
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
    prereg = protocol["c24_x54_preregistration"]
    gates = prereg["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    gate_checks = {
        "x54_precision_at_least_0_80": float(x54_metrics["precision"])
        + epsilon
        >= float(gates["minimum_precision"]),
        "x54_recall_at_least_0_70": float(x54_metrics["recall"])
        + epsilon
        >= float(gates["minimum_recall"]),
        "x54_f1_at_least_0_76": float(x54_metrics["f1"]) + epsilon
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
        "x52_provisional_parent_changes_at_least_1": mechanism_counts[
            "x52_provisional_parent_changes"
        ]
        >= int(gates["minimum_x52_provisional_parent_changes"]),
        "x53_anchor_redundancy_suppressions_at_least_1": mechanism_counts[
            "x53_anchor_redundancy_suppressions"
        ]
        >= int(gates["minimum_x53_anchor_redundancy_suppressions"]),
        "x54_dropout_continuations_at_least_1": mechanism_counts[
            "x54_dropout_continuations"
        ]
        >= int(gates["minimum_x54_dropout_continuations"]),
        "continuous_contact_episodes_at_least_3": continuous_count
        >= int(gates["required_continuous_contact_episodes"]),
        "parent_ancestry_episodes_at_least_3": ancestry_count
        >= int(gates["required_parent_ancestry_episodes"]),
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    mechanism_exercised = all(
        (
            gate_checks["x52_provisional_parent_changes_at_least_1"],
            gate_checks["x53_anchor_redundancy_suppressions_at_least_1"],
            gate_checks["x54_dropout_continuations_at_least_1"],
        )
    )
    gate_met = mechanism_exercised and all(gate_checks.values())
    if not mechanism_exercised:
        decision = "DTR_CARLA_C24_X54_MECHANISM_NOT_EXERCISED"
    elif gate_met:
        decision = "DTR_CARLA_C24_X54_GENERALIZATION_GATE_MET"
    else:
        decision = "DTR_CARLA_C24_X54_GENERALIZATION_GATE_NOT_MET"

    stretch = prereg["stretch_target"]
    summary = {
        "schema": "blindassist-dtr-carla-c24-x54-fresh-confirmation-v1",
        "status": "COMPLETE",
        "decision": decision,
        "gate_met": gate_met,
        "mechanism_exercised": mechanism_exercised,
        "elapsed_seconds": time.perf_counter() - started,
        "gate_checks": gate_checks,
        "thresholds": gates,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "authority_invariants": invariants,
        "mechanism_counts": mechanism_counts,
        "mechanism_diagnostics_by_episode": diagnostics,
        "x54_effect_vs_x24": effect,
        "stretch_target": {
            **stretch,
            "met": float(x54_metrics["precision"]) + epsilon
            >= float(stretch["precision"])
            and float(x54_metrics["recall"]) + epsilon
            >= float(stretch["recall"])
            and float(x54_metrics["f1"]) + epsilon >= float(stretch["f1"]),
        },
        "source": {
            "source_result_sha256": runner.base.sha256_file(
                source_root / "result.json"
            ),
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "model_manifest_sha256": runner.base.sha256_file(
                source_root / "model" / "manifest.json"
            ),
            "x24_freeze_sha256": runner.base.sha256_file(
                run_root / "freeze-x24.json"
            ),
            "x24_predictions_sha256": runner.base.sha256_file(
                run_root / "predictions-x24.json"
            ),
            "x54_predictions_sha256": runner.base.sha256_file(predictions_path),
            "x54_predictor_sha256": runner.base.sha256_file(
                Path(x54.__file__).resolve()
            ),
            "confirmation_runner_sha256": runner.base.sha256_file(
                Path(__file__).resolve()
            ),
        },
        "claim_boundary": {
            "fresh_scripted_carla_generalization_confirmation": True,
            "single_x54_scored_invocation": True,
            "same_map_route_layout_and_detector": True,
            "fresh_seed_and_four_weather_domains": True,
            "three_unseen_target_motion_profiles": True,
            "truth_blind_prediction_inputs": True,
            "evaluator_opened_only_after_predictions_were_sealed": True,
            "synthetic_development_only": True,
            "real_world_confirmation": False,
            "product_default_authority": False,
            "deployment_or_safety_authority": False,
        },
    }
    runner.base.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "gate_met": gate_met,
                "aggregate": aggregate,
                "contact_recall": contact_recall,
                "safe_segments": safe_counts,
                "continuity": {
                    key: value["continuous_route_risk"]
                    for key, value in continuity.items()
                },
                "ancestry": {
                    key: value["parent_ancestry_status"]
                    for key, value in continuity.items()
                },
                "effect": effect,
                "mechanism_counts": mechanism_counts,
                "authority_invariants": invariants,
                "gate_checks": gate_checks,
                "stretch_target_met": summary["stretch_target"]["met"],
                "summary_sha256": runner.base.sha256_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
