"""Replay frozen X59 modality-evidence composition on one consumed CARLA cohort."""

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

import dtr_carla_x59_modality_evidence_reliability_router as x59  # noqa: E402
import run_x33_dormant_transport_variant as runner  # noqa: E402


ARM_X24 = runner.ARM_X24
ARM_X31 = runner.ARM_X31
ARM_X54 = x59.x54.ARM_X54
ARM_X59 = x59.ARM_X59


def prediction_envelope(cohort_id: str, episodes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "blindassist-dtr-carla-x59-consumed-development-predictions-v1",
        "status": "SEALED_BEFORE_CONSUMED_SCORE",
        "cohort_id": cohort_id,
        "experiment_id": x59.EXPERIMENT_ID,
        "truth_blind_component_inputs": True,
        "arms": [ARM_X59],
        "episodes": episodes,
        "fixed_constants": x59.fixed_constants(),
        "claim_boundary": {
            "consumed_posthoc_development": True,
            "parent_confirmation_result_unchanged": True,
            "fresh_confirmation": False,
            "deployment_or_safety_confirmation": False,
        },
    }


def alias_for_transport_score(predictions: dict[str, Any]) -> dict[str, Any]:
    aliased = copy.deepcopy(predictions)
    for episode in aliased["episodes"].values():
        for frame in episode["frames"]:
            frame["arms"][ARM_X31] = frame["arms"][ARM_X59]
    return aliased


def effect(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float | int]:
    return {
        "tp_delta": int(challenger["tp"] - baseline["tp"]),
        "fp_delta": int(challenger["fp"] - baseline["fp"]),
        "f1_delta": float(challenger["f1"] - baseline["f1"]),
    }


def gates_from(protocol: dict[str, Any]) -> dict[str, Any]:
    for key in ("c27_x57_preregistration", "c26_x56_preregistration"):
        if key in protocol:
            return protocol[key]["primary_transfer_gate"]
    raise RuntimeError("unsupported consumed protocol")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--x54-predictions", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    x54_predictions_path = args.x54_predictions.resolve(strict=True)
    parent_summary_path = args.parent_summary.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    protocol = runner.base.read_json(protocol_path)
    source_result = runner.base.read_json(source_root / "result.json")
    parent_summary = runner.base.read_json(parent_summary_path)
    runner.base.require(
        source_result.get("status") == "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
        and source_result.get("protocol_sha256")
        == runner.base.sha256_file(protocol_path),
        "consumed_source_identity_drift",
    )
    runner.base.require(
        parent_summary.get("status") == "COMPLETE"
        and parent_summary["source"]["x54_predictions_sha256"]
        == runner.base.sha256_file(x54_predictions_path),
        "consumed_parent_identity_drift",
    )

    x24_predictions_path = run_root / "predictions-x24.json"
    x24_predictions = runner.base.read_json(x24_predictions_path)
    x54_predictions = runner.base.read_json(x54_predictions_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / "predictions-x59.json"
    summary_path = output_dir / "summary.json"

    episodes = {
        episode_id: x59.apply_modality_evidence_reliability_router_episode(
            x54_predictions["episodes"][episode_id],
            x24_predictions["episodes"][episode_id],
        )
        for episode_id in runner.EPISODES
    }
    predictions = prediction_envelope(protocol["cohort_id"], episodes)
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
        ARM_X59: predictions,
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

    measured_handbacks = sum(
        int(value["diagnostics"].get("x59_current_measured_closing_handback_frames", 0))
        for value in episodes.values()
    )
    receding_releases = sum(
        int(value["diagnostics"].get("x59_evidence_supported_receding_release_frames", 0))
        for value in episodes.values()
    )
    x24_metrics = aggregate[ARM_X24]
    x54_metrics = aggregate[ARM_X54]
    x59_metrics = aggregate[ARM_X59]
    safe_counts = {
        episode_id: int(safe[episode_id][ARM_X59]["false_alert_segment_count"])
        for episode_id in runner.SAFE
    }
    contact_recall = {
        episode_id: float(
            contacts[episode_id][ARM_X59]["future_positive_recall"]
        )
        for episode_id in runner.CONTACT
    }
    gates = gates_from(protocol)
    epsilon = runner.shared.EPSILON
    reference_checks = {
        "precision_at_least_0_85": float(x59_metrics["precision"]) + epsilon
        >= 0.85,
        "recall_at_least_0_70": float(x59_metrics["recall"]) + epsilon >= 0.70,
        "f1_at_least_0_78": float(x59_metrics["f1"]) + epsilon >= 0.78,
        "each_contact_recall_at_least_0_55": all(
            value + epsilon >= 0.55 for value in contact_recall.values()
        ),
        "each_safe_episode_has_at_most_4_segments": all(
            value <= int(gates["maximum_safe_false_alert_segments_per_episode"])
            for value in safe_counts.values()
        ),
        "total_safe_segments_at_most_10": sum(safe_counts.values())
        <= int(gates["maximum_total_safe_false_alert_segments"]),
        "measured_handback_exercised": measured_handbacks >= 1,
        "required_authority_invariants_are_zero": all(
            int(invariants[key]) == 0
            for key in gates["required_zero_authority_invariants"]
        ),
    }
    reference_target_met = all(reference_checks.values())
    decision = (
        "DTR_CARLA_X59_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_MET"
        if reference_target_met
        else "DTR_CARLA_X59_CONSUMED_DEVELOPMENT_REFERENCE_TARGET_NOT_MET"
    )
    summary = {
        "schema": "blindassist-dtr-carla-x59-consumed-development-v1",
        "status": "COMPLETE",
        "cohort_id": protocol["cohort_id"],
        "decision": decision,
        "reference_target_met": reference_target_met,
        "reference_checks": reference_checks,
        "aggregate": aggregate,
        "contacts": contacts,
        "safe": safe,
        "transport_continuity": continuity,
        "transport_ambiguity": ambiguity,
        "authority_invariants": invariants,
        "mechanism_counts": {
            "x59_current_measured_closing_handback_frames": measured_handbacks,
            "x59_evidence_supported_receding_release_frames": receding_releases,
        },
        "x59_effect_vs_x54": effect(x59_metrics, x54_metrics),
        "x59_effect_vs_x24": effect(x59_metrics, x24_metrics),
        "elapsed_seconds": time.perf_counter() - started,
        "source": {
            "protocol_sha256": runner.base.sha256_file(protocol_path),
            "parent_summary_sha256": runner.base.sha256_file(parent_summary_path),
            "x24_predictions_sha256": runner.base.sha256_file(x24_predictions_path),
            "x54_predictions_sha256": runner.base.sha256_file(x54_predictions_path),
            "x59_predictor_sha256": runner.base.sha256_file(Path(x59.__file__).resolve()),
            "x59_predictions_sha256": runner.base.sha256_file(predictions_path),
            "development_runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "consumed_posthoc_development": True,
            "parent_frozen_decision_unchanged": True,
            "reference_thresholds_not_preregistered_for_x59": True,
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
                "cohort_id": protocol["cohort_id"],
                "decision": decision,
                "reference_target_met": reference_target_met,
                "aggregate": aggregate,
                "contact_recall": contact_recall,
                "safe_segments": safe_counts,
                "mechanism_counts": summary["mechanism_counts"],
                "effect_vs_x54": summary["x59_effect_vs_x54"],
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

