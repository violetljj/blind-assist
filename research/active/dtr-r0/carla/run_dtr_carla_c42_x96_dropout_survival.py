"""Run one frozen C42-or-successor X96 dropout survival stress."""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
WORK = Path(r"E:\linnan\linnan\artifacts.local\work\x31-growth-diagnostic-20260831")
for value in (HERE, WORK):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import dtr_carla_x83_rigid_risk_reference_projection as x83  # noqa: E402
import dtr_carla_x84_branch_overloaded_closing_continuation_release as x84  # noqa: E402
import dtr_carla_x85_dequantization_completion_precedence_release as x85  # noqa: E402
import dtr_carla_x86_receding_handback_horizon_release as x86  # noqa: E402
import dtr_carla_x87_solo_completion_horizon_release as x87  # noqa: E402
import dtr_carla_x88_motion_epoch_contradiction_release as x88  # noqa: E402
import dtr_carla_x89_branch_overloaded_receding_release as x89  # noqa: E402
import dtr_carla_x90_collision_credentialed_lateral_dominant_release as x90  # noqa: E402
import dtr_carla_x91_held_risk_birth_horizon_release as x91  # noqa: E402
import dtr_carla_x92_held_risk_birth_horizon_latch as x92  # noqa: E402
import dtr_carla_x93_conflicted_nonclosing_future_release as x93  # noqa: E402
import dtr_carla_x94_one_frame_full_dropout_continuity as x94  # noqa: E402
import dtr_carla_x96_credentialed_bounded_dropout_survival as x96  # noqa: E402
import run_dtr_carla_c41_x82_fresh_confirmation as base41  # noqa: E402
import run_dtr_carla_x95_consumed_cross_validation as metrics95  # noqa: E402


x24 = base41.x24
x25 = base41.x25
x32 = base41.x32
x54 = base41.x54
x65 = base41.x65
x67 = base41.x67
x68 = base41.x68
x69 = base41.x69
x70 = base41.x70
x71 = base41.x71
x72 = base41.x72
x73 = base41.x73
x74 = base41.x74
x75 = base41.x75
x76 = base41.x76
x77 = base41.x77
x78 = base41.x78
x79 = base41.x79
x80 = base41.x80
x81 = base41.x81
x82 = base41.x82
runner = base41.runner

PROTOCOL_SHA256 = "48BF56D34E0B433BB2FD82DB6DA748C2C4E565EE3BF94F320120793893B825D1"
SOURCE_COMPLETE_STATUS = "DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_COMPLETE"
ARM_X94 = x94.ARM_X94
ARM_FORWARD_FILL = "BASELINE_RECURSIVE_FULL_DROPOUT_FORWARD_FILL"
ARM_HYSTERESIS = "BASELINE_ZERO_POINT_SIX_SECOND_HYSTERESIS"
ARM_X96 = x96.ARM_X96
ARMS = (ARM_X94, ARM_FORWARD_FILL, ARM_HYSTERESIS, ARM_X96)


def _prediction_row(
    frame: Mapping[str, Any], arm_name: str
) -> dict[str, Any]:
    arm = frame["arms"][arm_name]
    return {
        "sample_index": int(frame["sample_index"]),
        "time_s": float(frame["time_s"]),
        "route_risk": bool(arm.get("route_risk")),
        "minimum_entry_s": arm.get("minimum_entry_s") if arm.get("route_risk") else None,
    }


def _arm_rows(episode: Mapping[str, Any], arm_name: str) -> list[dict[str, Any]]:
    return [_prediction_row(frame, arm_name) for frame in episode["frames"]]


def _prefix(rows: Sequence[Mapping[str, Any]], end_s: float = 6.0) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(row))
        for row in rows
        if float(row["time_s"]) <= end_s + x24.EPSILON
    ]


def _candidate_dropout(
    values: Sequence[Mapping[str, Any]], start: int, length: int
) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(values))
    for sample_index in range(start, start + length):
        x24.require(0 <= sample_index < len(output), "c42_dropout_sample_range")
        output[sample_index]["candidates"] = []
        output[sample_index]["candidate_count"] = 0
        output[sample_index]["candidate_counts_by_class"] = {}
    return output


def _core_x93(
    episode: Any,
    candidates: list[dict[str, Any]],
    calibration: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metric = x24.predict_episode(episode, candidates, calibration)
    rigid = x25.predict_episode(episode, candidates, calibration)
    core54 = x54.predict_episode(episode, candidates, calibration)
    core65 = x65.apply_ancestry_handback_episode(core54, metric)
    core67 = x67.apply_measurement_horizon_receding_release_episode(core65)
    core68 = x68.apply_object_local_lateral_dequantization_episode(core67, metric, episode)
    core69 = x69.apply_mature_cross_route_rigid_contradiction_episode(core68, rigid)
    core70 = x70.apply_triple_credential_surface_dropout_handback_episode(core69, rigid, metric)
    core71 = x71.apply_entry_cotransport_occupancy_birth_episode(core70, rigid, metric)
    core72 = x72.apply_credentialed_surface_boundary_completion_episode(core71, rigid)
    core73 = x73.apply_credentialed_parent_hull_reconstruction_episode(core72, rigid, episode)
    core74 = x74.apply_metric_handback_class_contradiction_episode(core73, rigid)
    core75 = x75.apply_collision_credentialed_object_permanence_episode(core74, rigid, metric)
    core76 = x76.apply_zero_shift_parent_hull_motion_rejection_episode(core75)
    core77 = x77.apply_receding_metric_temporal_handoff_rejection_episode(core76)
    core78 = x78.apply_nonclosing_zero_shift_permanence_release_episode(core77)
    core79 = x79.apply_collision_credentialed_lateral_only_release_episode(core78)
    core80 = x80.apply_cross_route_footprint_credential_release_episode(core79)
    core81 = x81.apply_zero_shift_cross_route_shape_release_episode(core80)
    core82 = x82.apply_held_proxy_consensus_release_episode(core81)
    core83 = x83.apply_rigid_risk_reference_projection_episode(core82)
    core84 = x84.apply_branch_overloaded_closing_continuation_release_episode(core83)
    core85 = x85.apply_dequantization_completion_precedence_release_episode(core84)
    core86 = x86.apply_receding_handback_horizon_release_episode(core85)
    core87 = x87.apply_solo_completion_horizon_release_episode(core86)
    core88 = x88.apply_motion_epoch_contradiction_release_episode(core87)
    core89 = x89.apply_branch_overloaded_receding_release_episode(core88)
    core90 = x90.apply_collision_credentialed_lateral_dominant_release_episode(core89)
    core91 = x91.apply_held_risk_birth_horizon_release_episode(core90)
    core92 = x92.apply_held_risk_birth_horizon_latch_episode(core91)
    core93 = x93.apply_conflicted_nonclosing_future_release_episode(core92)
    return core93, metric, rigid


def _inject_plan_conflict(core: dict[str, Any], start: int, length: int) -> None:
    for sample_index in range(start, start + length):
        frame = core["frames"][sample_index]
        arm = frame["arms"][x93.ARM_X93]
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": [],
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": [],
                "confirmed_risk_parent_track_ids": [],
                "route_mode_changed": True,
                "controlled_plan_conflict": True,
            }
        )


def _baseline_rows(
    x94_episode: Mapping[str, Any], arm: str
) -> list[dict[str, Any]]:
    frames = x94_episode["frames"]
    if arm == ARM_FORWARD_FILL:
        return metrics95.baseline_forward_fill(frames)
    if arm == ARM_HYSTERESIS:
        return metrics95.baseline_hysteresis(frames)
    raise ValueError(arm)


def _case_id(episode_id: str, placement: str, length: int) -> str:
    # The suffix after ':' remains the canonical episode id because the shared
    # event scorer deliberately derives CONTACT/SAFE authority from it.
    return f"{placement}__L{length}:{episode_id}"


def _window_rows(rows: Sequence[Mapping[str, Any]], start: int, length: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if start <= int(row["sample_index"]) < start + length]


def _truth_positive(row: Mapping[str, Any]) -> bool:
    return bool(row["truth"]["future_contact_within_horizon"])


def _partition_evaluability(
    truth_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    cases: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    checks: dict[str, list[bool]] = {
        str(case["placement"]): [] for case in cases.values()
    }
    details: dict[str, dict[str, bool]] = {}
    for case_id, case in cases.items():
        truth = truth_by_case[case_id]
        start = int(case["start_sample_index"])
        length = int(case["length_frames"])
        placement = str(case["placement"])
        window = _window_rows(truth, start, length)
        labels = [_truth_positive(row) for row in window]
        if placement == "A_ACTIVE_MIDDLE":
            value = bool(labels) and all(labels)
        elif placement == "B_PRE_ONSET":
            value = not any(_truth_positive(row) for row in truth if int(row["sample_index"]) < start)
        elif placement == "C_RELEASE_BOUNDARY":
            transition_indices = [
                int(truth[index]["sample_index"])
                for index in range(1, len(truth))
                if _truth_positive(truth[index - 1]) and not _truth_positive(truth[index])
            ]
            value = any(abs(index - start) <= 1 for index in transition_indices)
        else:
            prior = [
                row
                for row in predictions[ARM_X94][case_id]
                if int(row["sample_index"]) == start - 1
            ]
            value = bool(prior) and bool(prior[0]["route_risk"])
        checks[placement].append(value)
        details[case_id] = {"evaluable": value}
    return {
        "by_partition": {name: all(values) for name, values in checks.items()},
        "by_case": details,
        "all_partitions_evaluable": all(all(values) for values in checks.values()),
    }


def _stress_metrics(
    arm: str,
    truth_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    arm_predictions = predictions[arm]
    aggregate = metrics95.aggregate_metrics(truth_by_case, arm_predictions)
    positive_recovered = {2: [0, 0], 3: [0, 0], 6: [0, 0]}
    negative_persistence = [0, 0]
    false_births = 0
    conflict_carries = 0
    interval_intersection = 0
    interval_union = 0
    interval_truth = 0
    release_overshoots: list[float] = []
    six_frame_tp = 0
    six_frame_fp = 0
    for case_id, case in cases.items():
        truth = truth_by_case[case_id]
        predicted = arm_predictions[case_id]
        start = int(case["start_sample_index"])
        length = int(case["length_frames"])
        placement = str(case["placement"])
        truth_window = _window_rows(truth, start, length)
        pred_window = _window_rows(predicted, start, length)
        labels = [_truth_positive(row) for row in truth_window]
        values = [bool(row["route_risk"]) for row in pred_window]
        interval_intersection += sum(label and value for label, value in zip(labels, values))
        interval_union += sum(label or value for label, value in zip(labels, values))
        interval_truth += sum(labels)
        if placement == "A_ACTIVE_MIDDLE":
            positive_recovered[length][0] += sum(label and value for label, value in zip(labels, values))
            positive_recovered[length][1] += sum(labels)
            if length == 6:
                six_frame_tp += sum(label and value for label, value in zip(labels, values))
                six_frame_fp += sum((not label) and value for label, value in zip(labels, values))
        elif placement == "B_PRE_ONSET":
            false_births += sum(values)
        elif placement == "C_RELEASE_BOUNDARY":
            negative_persistence[0] += sum((not label) and value for label, value in zip(labels, values))
            negative_persistence[1] += sum(not label for label in labels)
            transition = next(
                (
                    index
                    for index in range(1, len(truth))
                    if _truth_positive(truth[index - 1]) and not _truth_positive(truth[index])
                ),
                None,
            )
            if transition is not None:
                clear = next(
                    (
                        float(row["time_s"])
                        for row in predicted[transition:]
                        if not bool(row["route_risk"])
                    ),
                    None,
                )
                if clear is not None:
                    release_overshoots.append(max(0.0, clear - float(truth[transition]["time_s"])))
        else:
            conflict_carries += sum(values)
    recovery = {
        str(length): recovered / total if total else None
        for length, (recovered, total) in positive_recovered.items()
    }
    return {
        **aggregate,
        "positive_dropout_recovery_by_length": recovery,
        "negative_dropout_persistence": (
            negative_persistence[0] / negative_persistence[1]
            if negative_persistence[1]
            else None
        ),
        "false_birth_frames": false_births,
        "plan_conflict_carry_frames": conflict_carries,
        "dropout_interval_iou": interval_intersection / interval_union if interval_union else None,
        "dropout_interval_coverage": interval_intersection / interval_truth if interval_truth else None,
        "six_frame_tp": six_frame_tp,
        "six_frame_fp": six_frame_fp,
        "six_frame_tp_minus_fp": six_frame_tp - six_frame_fp,
        "median_release_overshoot_s": (
            statistics.median(release_overshoots) if release_overshoots else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-protocol-sha256", default=PROTOCOL_SHA256
    )
    parser.add_argument("--expected-source-result-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    run_root = args.run_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    expected_protocol_sha256 = str(args.expected_protocol_sha256).upper()
    runner.base.require(
        runner.base.sha256_file(protocol_path) == expected_protocol_sha256,
        "dropout_stress_protocol_hash_drift",
    )
    protocol = runner.base.read_json(protocol_path)
    match = re.search(r"DTR_CARLA_(C\d+)_", str(protocol.get("cohort_id", "")))
    runner.base.require(match is not None, "dropout_stress_cohort_id")
    cohort_tag = str(match.group(1))
    cohort_lower = cohort_tag.lower()
    prereg = protocol[f"{cohort_lower}_x96_preregistration"]
    for file_name, expected_sha256 in prereg["frozen_component_sha256"].items():
        path = HERE / file_name
        runner.base.require(
            path.is_file() and runner.base.sha256_file(path) == expected_sha256,
            f"c42_frozen_component_hash_drift:{file_name}",
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    predictions_path = output_dir / f"predictions-{cohort_lower}-x96-dropout-stress.json"
    summary_path = output_dir / "summary.json"
    freeze, contract, candidate_values = x24.require_freeze(run_root)
    x54.x53.x52.x45.x44.x43.x42.x32 = x32
    x54.x53.x52.x45.x44.x43.x42.x41.x40.x39.x38.x37.x35.x34.x33.x32 = x32

    episode_candidates: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for episode in contract.episodes:
        count = len(episode.observations)
        episode_candidates[episode.episode_id] = candidate_values[cursor : cursor + count]
        cursor += count
    runner.base.require(cursor == len(candidate_values), "c42_candidate_cursor")

    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    cases: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    episode_lookup = {episode.episode_id: episode for episode in contract.episodes}
    for episode_id in prereg["contact_episodes"]:
        episode = episode_lookup[episode_id]
        for placement, placement_spec in prereg["placements"].items():
            start = int(placement_spec["start_sample_index"])
            for length in map(int, prereg["dropout_lengths_frames"]):
                case_id = _case_id(episode_id, placement, length)
                candidates = _candidate_dropout(episode_candidates[episode_id], start, length)
                core93, metric, _rigid = _core_x93(episode, candidates, contract.calibration)
                if placement == "D_PLAN_CONFLICT":
                    _inject_plan_conflict(core93, start, length)
                out94 = x94.apply_one_frame_full_dropout_continuity_episode(core93)
                out96 = x96.apply_credentialed_bounded_dropout_survival_episode(core93)
                predictions[ARM_X94][case_id] = _arm_rows(out94, ARM_X94)
                predictions[ARM_FORWARD_FILL][case_id] = _baseline_rows(out94, ARM_FORWARD_FILL)
                predictions[ARM_HYSTERESIS][case_id] = _baseline_rows(out94, ARM_HYSTERESIS)
                predictions[ARM_X96][case_id] = _arm_rows(out96, ARM_X96)
                cases[case_id] = {
                    "episode_id": episode_id,
                    "placement": placement,
                    "start_sample_index": start,
                    "length_frames": length,
                    "dropout_sample_indices": list(range(start, start + length)),
                }
                diagnostics[case_id] = {
                    "metric_zero_candidate_frames": sum(
                        int(frame.get("raw_candidates", 0)) == 0
                        and int(frame.get("metric_footprint_measurements", 0)) == 0
                        for frame in metric["frames"][start : start + length]
                    ),
                    "x94_continuity_frames": int(
                        out94["diagnostics"].get("x94_one_frame_full_dropout_continuity_frames", 0)
                    ),
                    "x96_survival_frames": int(
                        out96["diagnostics"].get("x96_bounded_dropout_survival_frames", 0)
                    ),
                    "x96_conflict_rejections": int(
                        out96["diagnostics"].get("x96_conflict_rejections", 0)
                    ),
                }
                print(f"predicted_truth_blind {case_id}", flush=True)

    sealed = {
        "schema": f"blindassist-dtr-carla-{cohort_lower}-x96-dropout-stress-predictions-v1",
        "status": "SEALED_TRUTH_BLIND_PENDING_SINGLE_SCORE",
        "experiment_id": x96.EXPERIMENT_ID,
        "protocol_sha256": expected_protocol_sha256,
        "arms": list(ARMS),
        "cases": cases,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "fixed_constants": {
            "x94": x94.fixed_constants(),
            "x96": x96.fixed_constants(),
            "hysteresis_seconds": x24.HOLD_WINDOW_S,
        },
        "source": {
            "x24_freeze_sha256": runner.base.sha256_file(run_root / "freeze-x24.json"),
            "candidate_aggregate_sha256": freeze["candidates"]["aggregate_sha256"],
        },
        "claim_boundary": {
            "fresh_source_pixels": True,
            "controlled_candidate_dropout": True,
            "dropout_prevalence_claim": False,
            "truth_opened_during_prediction": False,
            "synthetic_development_only": True,
        },
    }
    runner.base.write_json_exclusive(predictions_path, sealed)
    print(f"sealed_truth_blind all {cohort_tag} cases and four arms", flush=True)

    expected_source_hash = str(args.expected_source_result_sha256).upper()
    runner.base.require(
        runner.base.sha256_file(source_root / "result.json") == expected_source_hash,
        "c42_source_result_hash_drift",
    )
    source_result = runner.base.read_json(source_root / "result.json")
    runner.base.require(
        source_result.get("status") == SOURCE_COMPLETE_STATUS
        and bool(source_result.get("checks"))
        and all(bool(value) for value in source_result["checks"].values()),
        "c42_source_gate_failed",
    )
    runner.base.require(
        source_result.get("protocol_sha256") == expected_protocol_sha256,
        "c42_source_protocol_drift",
    )
    runner.base.require(
        runner.base.sha256_file(source_root / "model" / "manifest.json")
        == freeze["model_manifest"]["sha256"],
        "c42_source_model_manifest_drift",
    )

    truth_by_episode = {
        episode_id: _prefix(
            runner.base.read_jsonl(
                source_root / "evaluator" / "episodes" / episode_id / "frames.jsonl"
            )
        )
        for episode_id in prereg["contact_episodes"]
    }
    truth_by_case = {
        case_id: truth_by_episode[str(case["episode_id"])] for case_id, case in cases.items()
    }
    scored_predictions = {
        arm: {case_id: _prefix(rows) for case_id, rows in values.items()}
        for arm, values in predictions.items()
    }
    for arm, values in scored_predictions.items():
        for case_id, rows in values.items():
            metrics95._align(truth_by_case[case_id], rows, f"{cohort_lower}:{arm}:{case_id}")

    evaluability = _partition_evaluability(
        truth_by_case, cases, scored_predictions
    )
    aggregate = {
        arm: _stress_metrics(arm, truth_by_case, scored_predictions, cases)
        for arm in ARMS
    }
    gates = prereg["primary_gate"]
    baseline = aggregate[ARM_X94]
    candidate = aggregate[ARM_X96]
    recovery_2 = candidate["positive_dropout_recovery_by_length"]["2"]
    recovery_3 = candidate["positive_dropout_recovery_by_length"]["3"]
    negative_persistence = candidate["negative_dropout_persistence"]
    release_overshoot = candidate["median_release_overshoot_s"]
    gate_checks = {
        "event_recall_not_below_x94": candidate["event_recall"] + x24.EPSILON >= baseline["event_recall"],
        "frame_f1_not_below_x94": candidate["f1"] + x24.EPSILON >= baseline["f1"],
        "two_frame_positive_recovery_at_least_0_80": recovery_2 is not None and recovery_2 + x24.EPSILON >= float(gates["minimum_2_3_frame_positive_dropout_recovery"]),
        "three_frame_positive_recovery_at_least_0_80": recovery_3 is not None and recovery_3 + x24.EPSILON >= float(gates["minimum_2_3_frame_positive_dropout_recovery"]),
        "negative_dropout_persistence_at_most_0_10": negative_persistence is not None and negative_persistence <= float(gates["maximum_negative_dropout_persistence"]) + x24.EPSILON,
        "zero_false_births": int(candidate["false_birth_frames"]) <= int(gates["maximum_false_births"]),
        "zero_plan_conflict_carries": int(candidate["plan_conflict_carry_frames"]) <= int(gates["maximum_cross_plan_or_conflict_carries"]),
        "six_frame_tp_minus_fp_at_least_1": int(candidate["six_frame_tp_minus_fp"]) >= int(gates["minimum_6_frame_tp_minus_fp"]),
        "median_release_overshoot_at_most_0_20_s": release_overshoot is not None and release_overshoot <= float(gates["maximum_median_release_overshoot_seconds"]) + x24.EPSILON,
    }
    mechanism_exercised = sum(
        int(value["x96_survival_frames"]) for value in diagnostics.values()
    ) > 0
    if not evaluability["all_partitions_evaluable"]:
        decision = f"DTR_CARLA_{cohort_tag}_X96_PARTITION_NOT_EVALUABLE"
        gate_met = False
    elif not mechanism_exercised:
        decision = f"DTR_CARLA_{cohort_tag}_X96_MECHANISM_NOT_EXERCISED"
        gate_met = False
    elif all(gate_checks.values()):
        decision = f"DTR_CARLA_{cohort_tag}_X96_GENERALIZATION_GATE_MET"
        gate_met = True
    else:
        decision = f"DTR_CARLA_{cohort_tag}_X96_GENERALIZATION_GATE_NOT_MET"
        gate_met = False

    summary = {
        "schema": f"blindassist-dtr-carla-{cohort_lower}-x96-dropout-survival-stress-v1",
        "status": "COMPLETE",
        "decision": decision,
        "gate_met": gate_met,
        "mechanism_exercised": mechanism_exercised,
        "elapsed_seconds": time.perf_counter() - started,
        "partition_evaluability": evaluability,
        "gate_checks": gate_checks,
        "thresholds": gates,
        "aggregate": aggregate,
        "mechanism_counts": {
            "x94_continuity_frames": sum(int(value["x94_continuity_frames"]) for value in diagnostics.values()),
            "x96_survival_frames": sum(int(value["x96_survival_frames"]) for value in diagnostics.values()),
            "x96_conflict_rejections": sum(int(value["x96_conflict_rejections"]) for value in diagnostics.values()),
            "dropout_frames_with_zero_detector_and_metric": sum(int(value["metric_zero_candidate_frames"]) for value in diagnostics.values()),
        },
        "source": {
            "source_result_sha256": expected_source_hash,
            "protocol_sha256": expected_protocol_sha256,
            "model_manifest_sha256": runner.base.sha256_file(source_root / "model" / "manifest.json"),
            "x24_freeze_sha256": runner.base.sha256_file(run_root / "freeze-x24.json"),
            "predictions_sha256": runner.base.sha256_file(predictions_path),
            "x94_predictor_sha256": runner.base.sha256_file(Path(x94.__file__).resolve()),
            "x96_predictor_sha256": runner.base.sha256_file(Path(x96.__file__).resolve()),
            "runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "fresh_scripted_carla_source": True,
            "new_source_seed_weather_plan_receipts_and_pixels": True,
            "trajectory_authority": prereg.get("trajectory_authority", "UNSPECIFIED"),
            "controlled_candidate_dropout": True,
            "controlled_plan_conflict": True,
            "natural_dropout_prevalence_evidence": False,
            "truth_opened_only_after_all_predictions_were_sealed": True,
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
                "gate_met": gate_met,
                "partition_evaluability": evaluability["by_partition"],
                "gate_checks": gate_checks,
                "aggregate": aggregate,
                "mechanism_counts": summary["mechanism_counts"],
                "summary_sha256": runner.base.sha256_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
