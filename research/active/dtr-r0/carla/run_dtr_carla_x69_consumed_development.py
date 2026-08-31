"""Replay X69 on one consumed CARLA cohort for Development evidence."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORK = Path(r"E:\linnan\linnan\artifacts.local\work\x31-growth-diagnostic-20260831")
for value in (HERE, WORK):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x68_object_local_lateral_dequantization as x68  # noqa: E402
import dtr_carla_x69_mature_cross_route_rigid_contradiction as x69  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X31 = runner.ARM_X31
ARM_X68 = x68.ARM_X68
ARM_X69 = x69.ARM_X69


def envelope(cohort_id: str, episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-x69-consumed-development-predictions-v1",
        "status": "SEALED_CONSUMED_DEVELOPMENT_PENDING_SCORE",
        "experiment_id": x69.EXPERIMENT_ID,
        "cohort_id": cohort_id,
        "arms": [ARM_X69],
        "episodes": episodes,
        "fixed_constants": x69.fixed_constants(),
        "claim_boundary": {
            "consumed_source": True,
            "posthoc_development_only": True,
            "fresh_confirmation": False,
            "deployment_or_safety_authority": False,
        },
    }


def rigid_envelope(cohort_id: str, episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-x69-x25-rigid-predictions-v1",
        "status": "SEALED_CONSUMED_DEVELOPMENT_AUXILIARY",
        "experiment_id": x25.EXPERIMENT_ID,
        "cohort_id": cohort_id,
        "arms": [x25.ARM_X25],
        "episodes": episodes,
        "fixed_constants": x25.fixed_constants(),
        "claim_boundary": {
            "auxiliary_geometry_for_x69": True,
            "consumed_source": True,
            "fresh_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(predictions)
    for episode in value["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X69]
    return value


def effect(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "tp_delta": int(challenger["tp"] - baseline["tp"]),
        "fp_delta": int(challenger["fp"] - baseline["fp"]),
        "f1_delta": float(challenger["f1"] - baseline["f1"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--x68-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    x68_path = args.x68_predictions.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    rigid_path = output_dir / "predictions-x25-rigid.json"
    predictions_path = output_dir / "predictions-x69.json"
    summary_path = output_dir / "summary.json"

    protocol = runner.base.read_json(protocol_path)
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("protocol_sha256") == runner.base.sha256_file(protocol_path),
        "x69_consumed_source_protocol",
    )
    _freeze, contract, candidate_values = x24.require_freeze(run_root)
    x68_predictions = runner.base.read_json(x68_path)
    runner.base.require(
        x68_predictions.get("status") == "SEALED_CONSUMED_DEVELOPMENT_PENDING_SCORE",
        "x69_x68_prediction_contract",
    )

    cursor = 0
    rigid_episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        rigid_episodes[episode.episode_id] = x25.predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    runner.base.require(cursor == len(candidate_values), "x69_candidate_cursor")
    rigid_predictions = rigid_envelope(args.cohort_id, rigid_episodes)
    runner.base.write_json_exclusive(rigid_path, rigid_predictions)

    episodes = {
        episode.episode_id: x69.apply_mature_cross_route_rigid_contradiction_episode(
            x68_predictions["episodes"][episode.episode_id],
            rigid_episodes[episode.episode_id],
        )
        for episode in contract.episodes
    }
    runner.base.require(set(episodes) == set(runner.EPISODES), "x69_episode_set")
    predictions = envelope(args.cohort_id, episodes)
    runner.base.write_json_exclusive(predictions_path, predictions)

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {ARM_X68: x68_predictions, ARM_X69: predictions}
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
    scoring_envelope = alias_for_transport_score(predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(
        scoring_envelope, selected
    )
    invariants = runner.shared.authority_invariants(
        scoring_envelope, runner.SCORE_END
    )

    x68_metrics = aggregate[ARM_X68]
    x69_metrics = aggregate[ARM_X69]
    delta = effect(x69_metrics, x68_metrics)
    contact_recall = {
        episode_id: float(contacts[episode_id][ARM_X69]["future_positive_recall"])
        for episode_id in runner.CONTACT
    }
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X69]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    mechanism_counts = {
        "mature_cross_route_rigid_contradiction_release_frames": sum(
            int(value["diagnostics"].get(
                "x69_mature_cross_route_rigid_contradiction_release_frames", 0
            ))
            for value in episodes.values()
        ),
        "mature_cross_route_rigid_contradicted_tracks": sum(
            int(value["diagnostics"].get(
                "x69_mature_cross_route_rigid_contradicted_tracks", 0
            ))
            for value in episodes.values()
        ),
    }
    reference_checks = {
        "precision_at_least_0_85": float(x69_metrics["precision"]) >= 0.85,
        "recall_at_least_0_70": float(x69_metrics["recall"]) >= 0.70,
        "f1_at_least_0_78": float(x69_metrics["f1"]) >= 0.78,
        "each_contact_recall_at_least_0_55": all(
            value >= 0.55 for value in contact_recall.values()
        ),
        "each_safe_episode_at_most_4_segments": all(
            value <= 4 for value in safe_counts.values()
        ),
        "total_safe_segments_at_most_10": sum(safe_counts.values()) <= 10,
        "required_authority_invariants_zero": all(
            int(invariants[key]) == 0
            for key in (
                "confirmed_missing_track_references",
                "confirmed_non_rigid_risk_track_references",
                "confirmed_parent_identity_mismatches",
                "route_risk_without_confirmed_eligible_track_frames",
                "route_risk_without_confirmed_rigid_dynamic_frames",
            )
        ),
        "mature_cross_route_rigid_contradiction_exercised": mechanism_counts[
            "mature_cross_route_rigid_contradiction_release_frames"
        ]
        > 0,
    }
    if not reference_checks["mature_cross_route_rigid_contradiction_exercised"]:
        decision = "DTR_CARLA_X69_CONSUMED_DEVELOPMENT_MECHANISM_NOT_EXERCISED"
    elif delta["tp_delta"] == 0 and delta["fp_delta"] < 0:
        decision = "DTR_CARLA_X69_CONSUMED_DEVELOPMENT_EFFECT_POSITIVE"
    elif delta["tp_delta"] == 0 and delta["fp_delta"] == 0:
        decision = "DTR_CARLA_X69_CONSUMED_DEVELOPMENT_EFFECT_NEUTRAL"
    else:
        decision = "DTR_CARLA_X69_CONSUMED_DEVELOPMENT_EFFECT_REGRESSION"

    summary = {
        "schema": "blindassist-dtr-carla-x69-consumed-development-v1",
        "status": "COMPLETE",
        "cohort_id": args.cohort_id,
        "decision": decision,
        "reference_target_met": all(reference_checks.values()),
        "aggregate": aggregate,
        "contact_recall": contact_recall,
        "safe_segments": safe_counts,
        "authority_invariants": invariants,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "mechanism_counts": mechanism_counts,
        "reference_checks": reference_checks,
        "effect_vs_x68": delta,
        "source": {
            "source_result_sha256": runner.base.sha256_file(source_root / "result.json"),
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "x68_predictions_sha256": runner.base.sha256_file(x68_path),
            "x25_rigid_predictions_sha256": runner.base.sha256_file(rigid_path),
            "x69_predictions_sha256": runner.base.sha256_file(predictions_path),
            "x25_predictor_sha256": runner.base.sha256_file(Path(x25.__file__).resolve()),
            "x69_predictor_sha256": runner.base.sha256_file(Path(x69.__file__).resolve()),
            "runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "consumed_posthoc_synthetic_development": True,
            "fresh_confirmation": False,
            "real_world_or_safety_authority": False,
        },
    }
    runner.base.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "cohort_id": args.cohort_id,
                "decision": decision,
                "aggregate": aggregate,
                "contact_recall": contact_recall,
                "safe_segments": safe_counts,
                "mechanism_counts": mechanism_counts,
                "effect_vs_x68": delta,
                "authority_invariants": invariants,
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
