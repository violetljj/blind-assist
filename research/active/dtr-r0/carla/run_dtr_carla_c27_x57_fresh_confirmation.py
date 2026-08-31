"""Run frozen X54 and X57 on fresh C27, then open and score truth once."""

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
import dtr_carla_x57_retained_core_metric_handback as x57  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X54 = x54.ARM_X54
ARM_X57 = x57.ARM_X57
EXPECTED_PROTOCOL_SHA256 = (
    "263BAED96E23CD009979412A71A4396728A898A740094C3431E9942135CACDEB"
)
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"


def prediction_envelope(
    *, arm: str, experiment_id: str, constants: dict[str, Any], episodes: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": f"blindassist-dtr-carla-c27-{arm.lower()}-fresh-predictions-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SINGLE_SCORE",
        "experiment_id": experiment_id,
        "truth_blind_prediction_inputs": True,
        "arms": [arm],
        "episodes": episodes,
        "fixed_constants": constants,
        "claim_boundary": {
            "fresh_c27_daylight_weather_transfer": True,
            "single_x57_scored_invocation": True,
            "threshold_or_scenario_tuning_after_capture": False,
            "evaluator_opened_during_prediction": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def require_frozen_inputs(
    protocol: dict[str, Any], protocol_path: Path, run_root: Path
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    runner.base.require(
        runner.base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c27_protocol_hash_drift",
    )
    prereg = protocol["c27_x57_preregistration"]
    runner.base.require(
        prereg["single_x57_scored_invocation"] is True
        and prereg["no_post_capture_algorithm_threshold_or_scenario_tuning"] is True
        and prereg["baselines"] == ["X24", "X54", "X57"],
        "c27_preregistration_contract",
    )
    for file_name, expected_sha256 in prereg["frozen_component_sha256"].items():
        component_path = HERE / file_name
        runner.base.require(
            component_path.is_file()
            and runner.base.sha256_file(component_path) == expected_sha256,
            f"c27_frozen_component_hash_drift:{file_name}",
        )

    freeze, contract, candidate_values = x24.require_freeze(run_root)
    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    runner.base.require(
        x24_predictions.get("status") == "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "c27_x24_predictions_not_sealed",
    )
    runner.base.require(
        x24_predictions["source"]["freeze_sha256"]
        == runner.base.sha256_file(run_root / "freeze-x24.json"),
        "c27_x24_prediction_freeze_drift",
    )
    return freeze, contract, candidate_values, x24_predictions


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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = runner.base.read_json(protocol_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    x54_predictions_path = output_dir / "predictions-x54.json"
    x57_predictions_path = output_dir / "predictions-x57.json"
    summary_path = output_dir / "summary.json"

    freeze, contract, candidate_values, x24_predictions = require_frozen_inputs(
        protocol, protocol_path, run_root
    )

    x54.x53.x52.x45.x44.x43.x42.x32 = x32
    x54.x53.x52.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    x54_episodes: dict[str, Any] = {}
    x57_episodes: dict[str, Any] = {}
    cursor = 0
    for episode in contract.episodes:
        count = len(episode.observations)
        core = x54.predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        x54_episodes[episode.episode_id] = core
        x57_episodes[episode.episode_id] = (
            x57.apply_retained_core_metric_handback_episode(
                core, x24_predictions["episodes"][episode.episode_id]
            )
        )
        cursor += count
        print(f"predicted_truth_blind {episode.episode_id} X54 X57", flush=True)
    runner.base.require(cursor == len(candidate_values), "c27_candidate_cursor")

    x54_predictions = prediction_envelope(
        arm=ARM_X54,
        experiment_id=x54.EXPERIMENT_ID,
        constants=x54.fixed_constants(),
        episodes=x54_episodes,
    )
    x57_predictions = prediction_envelope(
        arm=ARM_X57,
        experiment_id=x57.EXPERIMENT_ID,
        constants=x57.fixed_constants(),
        episodes=x57_episodes,
    )
    runner.base.write_json_exclusive(x54_predictions_path, x54_predictions)
    runner.base.write_json_exclusive(x57_predictions_path, x57_predictions)
    print("sealed_truth_blind X54 X57", flush=True)

    # Evaluator-bearing material is opened only after both predictions are sealed.
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS,
        "c27_source_incomplete",
    )
    runner.base.require(
        bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "c27_source_gate_failed",
    )
    runner.base.require(
        source_result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "c27_source_protocol_drift",
    )
    runner.base.require(
        int(source_result.get("episode_count", 0)) == len(runner.EPISODES)
        and int(source_result.get("layout_count", 0)) == 4,
        "c27_source_cohort_count",
    )
    runner.base.require(
        runner.base.sha256_file(source_root / "model" / "manifest.json")
        == freeze["model_manifest"]["sha256"],
        "c27_source_model_manifest_drift",
    )

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {
        ARM_X24: x24_predictions,
        ARM_X54: x54_predictions,
        ARM_X57: x57_predictions,
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
    scoring_envelope = alias_for_transport_score(x57_predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(
        scoring_envelope, selected
    )
    invariants = runner.shared.authority_invariants(
        scoring_envelope, runner.SCORE_END
    )

    diagnostics = {
        episode_id: {
            key: value
            for key, value in x57_episodes[episode_id]["diagnostics"].items()
            if key.startswith(("x52_", "x53_", "x54_", "x57_"))
        }
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

    prereg = protocol["c27_x57_preregistration"]
    gates = prereg["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    gate_checks = {
        "x57_precision_at_least_0_85": float(x57_metrics["precision"]) + epsilon
        >= float(gates["minimum_precision"]),
        "x57_recall_at_least_0_70": float(x57_metrics["recall"]) + epsilon
        >= float(gates["minimum_recall"]),
        "x57_f1_at_least_0_78": float(x57_metrics["f1"]) + epsilon
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
        "x57_handback_frames_at_least_1": handback_frames
        >= int(gates["minimum_x57_zero_eligible_metric_handback_frames"]),
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    mechanism_exercised = gate_checks["x57_handback_frames_at_least_1"]
    gate_met = mechanism_exercised and all(gate_checks.values())
    if not mechanism_exercised:
        decision = "DTR_CARLA_C27_X57_MECHANISM_NOT_EXERCISED"
    elif gate_met:
        decision = "DTR_CARLA_C27_X57_GENERALIZATION_GATE_MET"
    else:
        decision = "DTR_CARLA_C27_X57_GENERALIZATION_GATE_NOT_MET"

    stretch = prereg["stretch_target"]
    summary = {
        "schema": "blindassist-dtr-carla-c27-x57-fresh-confirmation-v1",
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
        "x57_handback_frames": handback_frames,
        "mechanism_diagnostics_by_episode": diagnostics,
        "x57_effect_vs_x54": effect(x57_metrics, x54_metrics),
        "x57_effect_vs_x24": effect(x57_metrics, x24_metrics),
        "stretch_target": {
            **stretch,
            "met": float(x57_metrics["precision"]) + epsilon
            >= float(stretch["precision"])
            and float(x57_metrics["recall"]) + epsilon
            >= float(stretch["recall"])
            and float(x57_metrics["f1"]) + epsilon >= float(stretch["f1"]),
        },
        "inheritance_roles": prereg["inheritance_roles"],
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
            "x54_predictions_sha256": runner.base.sha256_file(
                x54_predictions_path
            ),
            "x57_predictions_sha256": runner.base.sha256_file(
                x57_predictions_path
            ),
            "x54_predictor_sha256": runner.base.sha256_file(
                Path(x54.__file__).resolve()
            ),
            "x57_predictor_sha256": runner.base.sha256_file(
                Path(x57.__file__).resolve()
            ),
            "confirmation_runner_sha256": runner.base.sha256_file(
                Path(__file__).resolve()
            ),
        },
        "claim_boundary": {
            "fresh_scripted_carla_generalization_confirmation": True,
            "single_x57_scored_invocation": True,
            "same_map_route_layout_detector_and_motion_profiles": True,
            "fresh_seed_and_four_weather_lighting_domains": True,
            "truth_blind_prediction_inputs": True,
            "evaluator_opened_only_after_x54_and_x57_predictions_were_sealed": True,
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
                "effect_vs_x54": summary["x57_effect_vs_x54"],
                "effect_vs_x24": summary["x57_effect_vs_x24"],
                "x57_handback_frames": handback_frames,
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
