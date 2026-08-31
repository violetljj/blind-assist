"""Run frozen X72/X73 once on fresh C35, then open and score truth."""

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
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x32_observation_conditioned_core_predictor as x32  # noqa: E402
import dtr_carla_x54_metric_bootstrap_dropout_continuation as x54  # noqa: E402
import dtr_carla_x65_ancestry_synchronized_conflict_handback as x65  # noqa: E402
import dtr_carla_x67_measurement_horizon_receding_release as x67  # noqa: E402
import dtr_carla_x68_object_local_lateral_dequantization as x68  # noqa: E402
import dtr_carla_x69_mature_cross_route_rigid_contradiction as x69  # noqa: E402
import dtr_carla_x70_triple_credential_surface_dropout_handback as x70  # noqa: E402
import dtr_carla_x71_entry_cotransport_occupancy_birth as x71  # noqa: E402
import dtr_carla_x72_credentialed_surface_boundary_completion as x72  # noqa: E402
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73  # noqa: E402
import run_dtr_carla_x70_consumed_development as base70  # noqa: E402


runner = base70.runner
ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X72 = x72.ARM_X72
ARM_X73 = x73.ARM_X73
EXPECTED_PROTOCOL_SHA256 = (
    "53E52FC4318E0ECD4F60870E3999B878DF18CF6C84C7890D8434C043B6718A7E"
)
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"


def prediction_envelope(
    arm: str, experiment_id: str, constants: dict[str, Any], episodes: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": f"blindassist-dtr-carla-c35-{arm.lower()}-fresh-predictions-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SINGLE_SCORE",
        "experiment_id": experiment_id,
        "truth_blind_prediction_inputs": True,
        "arms": [arm],
        "episodes": episodes,
        "fixed_constants": constants,
        "claim_boundary": {
            "fresh_c35_transfer": True,
            "single_x73_scored_invocation": True,
            "threshold_or_scenario_tuning_after_capture": False,
            "evaluator_opened_during_prediction": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(predictions)
    for episode in value["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X73]
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
        "c35_protocol_hash_drift",
    )
    prereg = protocol["c35_x73_preregistration"]
    runner.base.require(
        prereg["single_x73_scored_invocation"] is True
        and prereg["no_post_capture_algorithm_threshold_or_scenario_tuning"] is True
        and prereg["baselines"] == ["X24", "X72", "X73"],
        "c35_preregistration_contract",
    )
    for file_name, expected_sha256 in prereg["frozen_component_sha256"].items():
        component_path = HERE / file_name
        runner.base.require(
            component_path.is_file()
            and runner.base.sha256_file(component_path) == expected_sha256,
            f"c35_frozen_component_hash_drift:{file_name}",
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    rigid_path = output_dir / "predictions-x25-rigid.json"
    x72_path = output_dir / "predictions-x72.json"
    x73_path = output_dir / "predictions-x73.json"
    summary_path = output_dir / "summary.json"

    freeze, contract, candidate_values = x24.require_freeze(run_root)
    x24_predictions = runner.base.read_json(run_root / "predictions-x24.json")
    runner.base.require(
        x24_predictions.get("status") == "SEALED_TRUTH_BLIND_PENDING_SCORE",
        "c35_x24_predictions_not_sealed",
    )
    runner.base.require(
        x24_predictions["source"]["freeze_sha256"]
        == runner.base.sha256_file(run_root / "freeze-x24.json"),
        "c35_x24_prediction_freeze_drift",
    )

    x54.x53.x52.x45.x44.x43.x42.x32 = x32
    x54.x53.x52.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    rigid_episodes: dict[str, Any] = {}
    x72_episodes: dict[str, Any] = {}
    x73_episodes: dict[str, Any] = {}
    cursor = 0
    for episode in contract.episodes:
        count = len(episode.observations)
        candidates = candidate_values[cursor : cursor + count]
        metric = x24_predictions["episodes"][episode.episode_id]
        rigid = x25.predict_episode(episode, candidates, contract.calibration)
        core54 = x54.predict_episode(episode, candidates, contract.calibration)
        core65 = x65.apply_ancestry_handback_episode(core54, metric)
        core67 = x67.apply_measurement_horizon_receding_release_episode(core65)
        core68 = x68.apply_object_local_lateral_dequantization_episode(
            core67, metric, episode
        )
        core69 = x69.apply_mature_cross_route_rigid_contradiction_episode(
            core68, rigid
        )
        core70 = x70.apply_triple_credential_surface_dropout_handback_episode(
            core69, rigid, metric
        )
        core71 = x71.apply_entry_cotransport_occupancy_birth_episode(
            core70, rigid, metric
        )
        core72 = x72.apply_credentialed_surface_boundary_completion_episode(
            core71, rigid
        )
        core73 = x73.apply_credentialed_parent_hull_reconstruction_episode(
            core72, rigid, episode
        )
        rigid_episodes[episode.episode_id] = rigid
        x72_episodes[episode.episode_id] = core72
        x73_episodes[episode.episode_id] = core73
        cursor += count
        print(f"predicted_truth_blind {episode.episode_id} X72 X73", flush=True)
    runner.base.require(cursor == len(candidate_values), "c35_candidate_cursor")

    rigid_envelope = base70.rigid_envelope("C35", rigid_episodes)
    x72_predictions = prediction_envelope(
        ARM_X72, x72.EXPERIMENT_ID, x72.fixed_constants(), x72_episodes
    )
    x73_predictions = prediction_envelope(
        ARM_X73, x73.EXPERIMENT_ID, x73.fixed_constants(), x73_episodes
    )
    runner.base.write_json_exclusive(rigid_path, rigid_envelope)
    runner.base.write_json_exclusive(x72_path, x72_predictions)
    runner.base.write_json_exclusive(x73_path, x73_predictions)
    print("sealed_truth_blind X25 X72 X73", flush=True)

    # Evaluator-bearing material is opened only after every prediction is sealed.
    expected_source_hash = str(args.expected_source_result_sha256).upper()
    runner.base.require(
        runner.base.sha256_file(source_root / "result.json") == expected_source_hash,
        "c35_source_result_hash_drift",
    )
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS
        and bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "c35_source_gate_failed",
    )
    runner.base.require(
        source_result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "c35_source_protocol_drift",
    )
    runner.base.require(
        int(source_result.get("episode_count", 0)) == len(runner.EPISODES)
        and int(source_result.get("layout_count", 0)) == 4,
        "c35_source_cohort_count",
    )
    runner.base.require(
        runner.base.sha256_file(source_root / "model" / "manifest.json")
        == freeze["model_manifest"]["sha256"],
        "c35_source_model_manifest_drift",
    )

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {
        ARM_X24: x24_predictions,
        ARM_X72: x72_predictions,
        ARM_X73: x73_predictions,
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
        arm: runner.base.confusion(evaluator, values)
        for arm, values in scored.items()
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
    compatible = alias_for_transport_score(x73_predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(
        compatible, selected
    )
    invariants = runner.shared.authority_invariants(compatible, runner.SCORE_END)

    x72_metrics = aggregate[ARM_X72]
    x73_metrics = aggregate[ARM_X73]
    effect_vs_x72 = base70.effect(x73_metrics, x72_metrics)
    contact_recall = {
        episode_id: float(contacts[episode_id][ARM_X73]["future_positive_recall"])
        for episode_id in runner.CONTACT
    }
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X73]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    mechanism_counts = {
        "parent_hull_reconstruction_frames": sum(
            int(value["diagnostics"].get("x73_parent_hull_reconstruction_frames", 0))
            for value in x73_episodes.values()
        ),
        "parent_hull_reconstruction_tracks": sum(
            int(value["diagnostics"].get("x73_parent_hull_reconstruction_tracks", 0))
            for value in x73_episodes.values()
        ),
        "rigid_center_containment_rejections": sum(
            int(value["diagnostics"].get("x73_rigid_center_containment_rejections", 0))
            for value in x73_episodes.values()
        ),
        "parent_hull_route_absent_rejections": sum(
            int(value["diagnostics"].get("x73_parent_hull_route_absent_rejections", 0))
            for value in x73_episodes.values()
        ),
    }
    gates = prereg["primary_transfer_gate"]
    epsilon = runner.shared.EPSILON
    gate_checks = {
        "x73_precision_at_least_0_85": float(x73_metrics["precision"]) + epsilon
        >= float(gates["minimum_precision"]),
        "x73_recall_at_least_0_70": float(x73_metrics["recall"]) + epsilon
        >= float(gates["minimum_recall"]),
        "x73_f1_at_least_0_78": float(x73_metrics["f1"]) + epsilon
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
        "x73_parent_hull_reconstruction_at_least_1": (
            mechanism_counts["parent_hull_reconstruction_frames"]
            >= int(gates["minimum_x73_parent_hull_reconstruction_frames"])
        ),
        "x73_tp_delta_vs_x72_at_least_1": effect_vs_x72["tp_delta"]
        >= int(gates["minimum_x73_tp_delta_vs_x72"]),
        "x73_fp_delta_vs_x72_at_most_0": effect_vs_x72["fp_delta"]
        <= int(gates["maximum_x73_fp_delta_vs_x72"]),
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    mechanism_exercised = gate_checks[
        "x73_parent_hull_reconstruction_at_least_1"
    ]
    gate_met = mechanism_exercised and all(gate_checks.values())
    if not mechanism_exercised:
        decision = "DTR_CARLA_C35_X73_MECHANISM_NOT_EXERCISED"
    elif gate_met:
        decision = "DTR_CARLA_C35_X73_GENERALIZATION_GATE_MET"
    else:
        decision = "DTR_CARLA_C35_X73_GENERALIZATION_GATE_NOT_MET"

    stretch = prereg["stretch_target"]
    summary = {
        "schema": "blindassist-dtr-carla-c35-x73-fresh-confirmation-v1",
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
        "x73_effect_vs_x72": effect_vs_x72,
        "stretch_target": {
            **stretch,
            "met": float(x73_metrics["precision"]) + epsilon
            >= float(stretch["precision"])
            and float(x73_metrics["recall"]) + epsilon
            >= float(stretch["recall"])
            and float(x73_metrics["f1"]) + epsilon >= float(stretch["f1"]),
        },
        "source": {
            "source_result_sha256": expected_source_hash,
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "model_manifest_sha256": runner.base.sha256_file(
                source_root / "model" / "manifest.json"
            ),
            "x24_freeze_sha256": runner.base.sha256_file(run_root / "freeze-x24.json"),
            "x24_predictions_sha256": runner.base.sha256_file(
                run_root / "predictions-x24.json"
            ),
            "x25_rigid_predictions_sha256": runner.base.sha256_file(rigid_path),
            "x72_predictions_sha256": runner.base.sha256_file(x72_path),
            "x73_predictions_sha256": runner.base.sha256_file(x73_path),
            "x73_predictor_sha256": runner.base.sha256_file(Path(x73.__file__).resolve()),
            "confirmation_runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_generalization_confirmation": True,
            "single_x73_scored_invocation": True,
            "same_map_route_layout_detector_and_motion_profiles": True,
            "fresh_seed_weather_and_pixels": True,
            "truth_blind_prediction_inputs": True,
            "evaluator_opened_only_after_x24_x25_x72_x73_predictions_were_sealed": True,
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
                "effect_vs_x72": effect_vs_x72,
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
