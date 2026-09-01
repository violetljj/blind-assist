"""Replay X82 against X81 on one consumed CARLA cohort."""

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

import dtr_carla_x81_zero_shift_cross_route_shape_release as x81  # noqa: E402
import dtr_carla_x82_held_proxy_consensus_release as x82  # noqa: E402
import run_dtr_carla_x81_consumed_development as base81  # noqa: E402


runner = base81.runner
ARM_X31 = runner.ARM_X31
ARM_X81 = x81.ARM_X81
ARM_X82 = x82.ARM_X82


def prediction_envelope(cohort_id: str, episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-x82-consumed-development-predictions-v1",
        "status": "SEALED_CONSUMED_DEVELOPMENT_PENDING_SCORE",
        "experiment_id": x82.EXPERIMENT_ID,
        "cohort_id": cohort_id,
        "arms": [ARM_X82],
        "episodes": episodes,
        "fixed_constants": x82.fixed_constants(),
        "claim_boundary": {
            "consumed_source": True,
            "posthoc_development_only": True,
            "fresh_confirmation": False,
            "deployment_or_safety_authority": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(predictions)
    for episode in value["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X82]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--x81-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    x81_path = args.x81_predictions.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / "predictions-x82.json"
    summary_path = output_dir / "summary.json"

    protocol = runner.base.read_json(protocol_path)
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("protocol_sha256") == runner.base.sha256_file(protocol_path),
        "x82_consumed_source_protocol",
    )
    x81_predictions = runner.base.read_json(x81_path)
    runner.base.require(
        str(x81_predictions.get("status", "")).startswith("SEALED_")
        and set(x81_predictions.get("arms", [])) == {ARM_X81},
        "x82_x81_prediction_contract",
    )
    runner.base.require(
        set(x81_predictions["episodes"]) == set(runner.EPISODES),
        "x82_episode_set",
    )

    episodes = {
        episode_id: x82.apply_held_proxy_consensus_release_episode(
            x81_predictions["episodes"][episode_id]
        )
        for episode_id in runner.EPISODES
    }
    predictions = prediction_envelope(args.cohort_id, episodes)
    runner.base.write_json_exclusive(predictions_path, predictions)

    evaluator_full = {
        episode_id: runner.base.read_jsonl(
            source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
        )
        for episode_id in runner.EPISODES
    }
    envelopes = {ARM_X81: x81_predictions, ARM_X82: predictions}
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
    compatible = alias_for_transport_score(predictions)
    continuity, ambiguity = runner.shared.contact_transport_continuity(compatible, selected)
    invariants = runner.shared.authority_invariants(compatible, runner.SCORE_END)

    x81_metrics = aggregate[ARM_X81]
    x82_metrics = aggregate[ARM_X82]
    delta = base81.effect(x82_metrics, x81_metrics)
    contact_recall = {
        episode_id: float(contacts[episode_id][ARM_X82]["future_positive_recall"])
        for episode_id in runner.CONTACT
    }
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X82]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    mechanism_counts = {
        "release_frames": sum(
            int(value["diagnostics"].get("x82_held_proxy_consensus_release_frames", 0))
            for value in episodes.values()
        ),
        "released_tracks": sum(
            int(value["diagnostics"].get("x82_held_proxy_consensus_released_tracks", 0))
            for value in episodes.values()
        ),
    }
    reference_checks = {
        "precision_at_least_0_85": float(x82_metrics["precision"]) >= 0.85,
        "recall_at_least_0_70": float(x82_metrics["recall"]) >= 0.70,
        "f1_at_least_0_78": float(x82_metrics["f1"]) >= 0.78,
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
    }
    exercised = mechanism_counts["release_frames"] > 0
    if exercised and delta["tp_delta"] == 0 and delta["fp_delta"] < 0:
        decision = "DTR_CARLA_X82_CONSUMED_DEVELOPMENT_EFFECT_POSITIVE"
    elif not exercised and delta["tp_delta"] == 0 and delta["fp_delta"] == 0:
        decision = "DTR_CARLA_X82_CONSUMED_DEVELOPMENT_MECHANISM_NOT_EXERCISED"
    elif delta["tp_delta"] == 0 and delta["fp_delta"] == 0:
        decision = "DTR_CARLA_X82_CONSUMED_DEVELOPMENT_EFFECT_NEUTRAL"
    else:
        decision = "DTR_CARLA_X82_CONSUMED_DEVELOPMENT_EFFECT_REGRESSION"

    summary = {
        "schema": "blindassist-dtr-carla-x82-consumed-development-v1",
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
        "effect_vs_x81": delta,
        "source": {
            "source_result_sha256": runner.base.sha256_file(source_root / "result.json"),
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "x81_predictions_sha256": runner.base.sha256_file(x81_path),
            "x82_predictions_sha256": runner.base.sha256_file(predictions_path),
            "x82_predictor_sha256": runner.base.sha256_file(Path(x82.__file__).resolve()),
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
                "mechanism_counts": mechanism_counts,
                "effect_vs_x81": delta,
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
