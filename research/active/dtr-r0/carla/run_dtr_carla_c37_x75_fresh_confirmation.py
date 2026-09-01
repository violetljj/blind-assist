"""Run frozen X74/X75 once on fresh C37, then open and score truth."""

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

import dtr_carla_x75_collision_credentialed_object_permanence as x75  # noqa: E402
import run_dtr_carla_c36_x74_fresh_confirmation as base36  # noqa: E402


x24 = base36.x24
x25 = base36.x25
x32 = base36.x32
x54 = base36.x54
x65 = base36.x65
x67 = base36.x67
x68 = base36.x68
x69 = base36.x69
x70 = base36.x70
x71 = base36.x71
x72 = base36.x72
x73 = base36.x73
x74 = base36.x74
runner = base36.runner
ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X74 = x74.ARM_X74
ARM_X75 = x75.ARM_X75
EXPECTED_PROTOCOL_SHA256 = (
    "81AE71541B2DB5C6CD7DD29A97A360D0B4D9F02E9E19E5F759D428D8EDF9BF80"
)
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"


def prediction_envelope(
    arm: str, experiment_id: str, constants: dict[str, Any], episodes: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": f"blindassist-dtr-carla-c37-{arm.lower()}-fresh-predictions-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SINGLE_SCORE",
        "experiment_id": experiment_id,
        "truth_blind_prediction_inputs": True,
        "arms": [arm],
        "episodes": episodes,
        "fixed_constants": constants,
        "claim_boundary": {
            "fresh_c37_transfer": True,
            "single_x75_scored_invocation": True,
            "threshold_or_scenario_tuning_after_capture": False,
            "evaluator_opened_during_prediction": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(predictions)
    for episode in value["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X75]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = runner.base.read_json(protocol_path)
    runner.base.require(
        runner.base.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256,
        "c37_protocol_hash_drift",
    )
    prereg = protocol["c37_x75_preregistration"]
    runner.base.require(
        prereg["single_x75_scored_invocation"] is True
        and prereg["no_post_capture_algorithm_threshold_or_scenario_tuning"] is True
        and prereg["baselines"] == ["X24", "X74", "X75"],
        "c37_preregistration_contract",
    )
    for file_name, expected_sha256 in prereg["frozen_component_sha256"].items():
        component_path = HERE / file_name
        runner.base.require(
            component_path.is_file()
            and runner.base.sha256_file(component_path) == expected_sha256,
            f"c37_frozen_component_hash_drift:{file_name}",
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    rigid_path = output_dir / "predictions-x25-rigid.json"
    x74_path = output_dir / "predictions-x74.json"
    x75_path = output_dir / "predictions-x75.json"
    summary_path = output_dir / "summary.json"

    freeze, contract, candidate_values = x24.require_freeze(run_root)
    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    runner.base.require(
        x24_predictions.get("status") == "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "c37_x24_predictions_not_sealed",
    )
    runner.base.require(
        x24_predictions["source"]["freeze_sha256"]
        == runner.base.sha256_file(run_root / "freeze-x24.json"),
        "c37_x24_prediction_freeze_drift",
    )

    x54.x53.x52.x45.x44.x43.x42.x32 = x32
    x54.x53.x52.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    rigid_episodes: dict[str, Any] = {}
    x74_episodes: dict[str, Any] = {}
    x75_episodes: dict[str, Any] = {}
    cursor = 0
    for episode in contract.episodes:
        count = len(episode.observations)
        candidates = candidate_values[cursor : cursor + count]
        metric = x24_predictions["episodes"][episode.episode_id]
        rigid = x25.predict_episode(episode, candidates, contract.calibration)
        core54 = x54.predict_episode(episode, candidates, contract.calibration)
        core65 = x65.apply_ancestry_handback_episode(core54, metric)
        core67 = x67.apply_measurement_horizon_receding_release_episode(core65)
        core68 = x68.apply_object_local_lateral_dequantization_episode(core67, metric, episode)
        core69 = x69.apply_mature_cross_route_rigid_contradiction_episode(core68, rigid)
        core70 = x70.apply_triple_credential_surface_dropout_handback_episode(core69, rigid, metric)
        core71 = x71.apply_entry_cotransport_occupancy_birth_episode(core70, rigid, metric)
        core72 = x72.apply_credentialed_surface_boundary_completion_episode(core71, rigid)
        core73 = x73.apply_credentialed_parent_hull_reconstruction_episode(core72, rigid, episode)
        core74 = x74.apply_metric_handback_class_contradiction_episode(core73, rigid)
        core75 = x75.apply_collision_credentialed_object_permanence_episode(
            core74, rigid, metric
        )
        rigid_episodes[episode.episode_id] = rigid
        x74_episodes[episode.episode_id] = core74
        x75_episodes[episode.episode_id] = core75
        cursor += count
        print(f"predicted_truth_blind {episode.episode_id} X74 X75", flush=True)
    runner.base.require(cursor == len(candidate_values), "c37_candidate_cursor")

    rigid_predictions = base36.base35.base70.rigid_envelope("C37", rigid_episodes)
    x74_predictions = prediction_envelope(
        ARM_X74, x74.EXPERIMENT_ID, x74.fixed_constants(), x74_episodes
    )
    x75_predictions = prediction_envelope(
        ARM_X75, x75.EXPERIMENT_ID, x75.fixed_constants(), x75_episodes
    )
    runner.base.write_json_exclusive(rigid_path, rigid_predictions)
    runner.base.write_json_exclusive(x74_path, x74_predictions)
    runner.base.write_json_exclusive(x75_path, x75_predictions)
    print("sealed_truth_blind X25 X74 X75", flush=True)

    expected_source_hash = str(args.expected_source_result_sha256).upper()
    runner.base.require(
        runner.base.sha256_file(source_root / "result.json") == expected_source_hash,
        "c37_source_result_hash_drift",
    )
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS
        and bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "c37_source_gate_failed",
    )
    runner.base.require(
        source_result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "c37_source_protocol_drift",
    )
    runner.base.require(
        int(source_result.get("episode_count", 0)) == len(runner.EPISODES)
        and int(source_result.get("layout_count", 0)) == 4,
        "c37_source_cohort_count",
    )
    runner.base.require(
        runner.base.sha256_file(source_root / "model" / "manifest.json")
        == freeze["model_manifest"]["sha256"],
        "c37_source_model_manifest_drift",
    )

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {
        ARM_X24: x24_predictions,
        ARM_X74: x74_predictions,
        ARM_X75: x75_predictions,
    }
    predictions_full = {
        arm: {
            episode_id: runner.shared.c7.arm_frames_full(value, episode_id, arm)
            for episode_id in runner.EPISODES
        }
        for arm, value in envelopes.items()
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
        arm: runner.base.confusion(evaluator, values) for arm, values in scored.items()
    }
    contacts = {
        episode_id: {
            arm: runner.base.contact_metrics(evaluator[episode_id], values[episode_id])
            for arm, values in scored.items()
        }
        for episode_id in runner.CONTACT
    }
    safe = {
        episode_id: {
            arm: runner.base.false_segments(values[episode_id], runner.SAFE_START[episode_id])
            for arm, values in scored.items()
        }
        for episode_id in runner.SAFE
    }
    selected = runner.shared.validate_occlusion_reports(
        protocol,
        runner.base.read_json(source_root / "evaluator" / "physical_occlusion_report.json"),
        evaluator_full,
    )
    compatible = alias_for_transport_score(x75_predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(
        compatible, selected
    )
    invariants = runner.shared.authority_invariants(compatible, runner.SCORE_END)

    x74_metrics = aggregate[ARM_X74]
    x75_metrics = aggregate[ARM_X75]
    effect_vs_x74 = base36.base35.base70.effect(x75_metrics, x74_metrics)
    contact_recall = {
        episode_id: float(contacts[episode_id][ARM_X75]["future_positive_recall"])
        for episode_id in runner.CONTACT
    }
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X75]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    mechanism_counts = {
        "collision_credential_births": sum(
            int(value["diagnostics"].get("x75_collision_credential_births", 0))
            for value in x75_episodes.values()
        ),
        "existence_only_permanence_release_frames": sum(
            int(value["diagnostics"].get("x75_existence_only_permanence_release_frames", 0))
            for value in x75_episodes.values()
        ),
        "existence_only_permanence_released_tracks": sum(
            int(value["diagnostics"].get("x75_existence_only_permanence_released_tracks", 0))
            for value in x75_episodes.values()
        ),
    }
    gates = prereg["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    gate_checks = {
        "x75_precision_at_least_0_85": float(x75_metrics["precision"]) + epsilon
        >= float(gates["minimum_precision"]),
        "x75_recall_at_least_0_70": float(x75_metrics["recall"]) + epsilon
        >= float(gates["minimum_recall"]),
        "x75_f1_at_least_0_78": float(x75_metrics["f1"]) + epsilon
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
        "x75_permanence_release_at_least_1": (
            mechanism_counts["existence_only_permanence_release_frames"]
            >= int(gates["minimum_x75_permanence_release_frames"])
        ),
        "x75_tp_delta_vs_x74_at_least_0": effect_vs_x74["tp_delta"]
        >= int(gates["minimum_x75_tp_delta_vs_x74"]),
        "x75_fp_delta_vs_x74_at_most_minus_1": effect_vs_x74["fp_delta"]
        <= int(gates["maximum_x75_fp_delta_vs_x74"]),
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    mechanism_exercised = gate_checks["x75_permanence_release_at_least_1"]
    gate_met = mechanism_exercised and all(gate_checks.values())
    if not mechanism_exercised:
        decision = "DTR_CARLA_C37_X75_MECHANISM_NOT_EXERCISED"
    elif gate_met:
        decision = "DTR_CARLA_C37_X75_GENERALIZATION_GATE_MET"
    else:
        decision = "DTR_CARLA_C37_X75_GENERALIZATION_GATE_NOT_MET"

    stretch = prereg["stretch_target"]
    summary = {
        "schema": "blindassist-dtr-carla-c37-x75-fresh-confirmation-v1",
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
        "x75_effect_vs_x74": effect_vs_x74,
        "stretch_target": {
            **stretch,
            "met": float(x75_metrics["precision"]) + epsilon >= float(stretch["precision"])
            and float(x75_metrics["recall"]) + epsilon >= float(stretch["recall"])
            and float(x75_metrics["f1"]) + epsilon >= float(stretch["f1"]),
        },
        "source": {
            "source_result_sha256": expected_source_hash,
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "model_manifest_sha256": runner.base.sha256_file(source_root / "model" / "manifest.json"),
            "x24_freeze_sha256": runner.base.sha256_file(run_root / "freeze-x24.json"),
            "x24_predictions_sha256": runner.base.sha256_file(run_root / "predictions-x24.json"),
            "x25_rigid_predictions_sha256": runner.base.sha256_file(rigid_path),
            "x74_predictions_sha256": runner.base.sha256_file(x74_path),
            "x75_predictions_sha256": runner.base.sha256_file(x75_path),
            "x75_predictor_sha256": runner.base.sha256_file(Path(x75.__file__).resolve()),
            "confirmation_runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_generalization_confirmation": True,
            "single_x75_scored_invocation": True,
            "same_map_route_layout_detector_and_motion_profiles": True,
            "fresh_seed_weather_and_pixels": True,
            "truth_blind_prediction_inputs": True,
            "evaluator_opened_only_after_x24_x25_x74_x75_predictions_were_sealed": True,
            "synthetic_development_only": True,
            "real_world_confirmation": False,
            "product_default_authority": False,
            "deployment_or_safety_authority": False,
        },
    }
    runner.base.write_json_exclusive(summary_path, summary)
    print(json.dumps({
        "decision": decision,
        "gate_met": gate_met,
        "aggregate": aggregate,
        "contact_recall": contact_recall,
        "safe_segments": safe_counts,
        "effect_vs_x74": effect_vs_x74,
        "mechanism_counts": mechanism_counts,
        "authority_invariants": invariants,
        "gate_checks": gate_checks,
        "stretch_target_met": summary["stretch_target"]["met"],
        "summary_sha256": runner.base.sha256_file(summary_path),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
