#!/usr/bin/env python3
"""Run P1-A3 temporal loss-declaration and conservative recovery discovery."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, deque
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_scripts_namespace = types.ModuleType("scripts")
_scripts_namespace.__path__ = [str(_REPO_ROOT / "scripts")]
sys.modules["scripts"] = _scripts_namespace

import run_p1_a2_dense_identity as a2
import run_p1_consumed_adt_baseline as r0
from materialize_p1_temporal_cohort import sha256


BUNDLE_SCHEMA = "blindassist_p1_a3_temporal_candidate_bundle_v1"
RESULT_SCHEMA = "blindassist_p1_a3_temporal_loss_sweep_v1"
A2_TRACE_SHA256 = "CF24FD7749E835C3F2C7B203361114793BEEC4774F6D1DF9E89644A922ECE471"
RAW_FEATURES = a2.POLICY_FEATURES
A2_CHATTER_REFERENCE = {
    "tracking_boundary_transitions": 151,
    "reacquisition_chatter_within_30_frames": 27,
}
GATES = {
    "correct_assertions_min": 79,
    "wrong_assertions_max": 488,
    "max_wrong_lock_duration_ms_max": 3399,
    "false_loss_frames_max": 152,
    "false_reacquisitions_max": 5,
    "tracking_boundary_transitions_max": 75,
    "reacquisition_chatter_within_30_frames_max": 5,
    "long_target_absence_loss_declarations_min": 3,
    "all_reacquisition_opportunity_loss_declarations_min": 3,
}


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()


def _spec_id(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{spec['operator']}-{hashlib.sha256(payload).hexdigest()[:12]}"


def operator_family() -> list[dict[str, Any]]:
    specs = []
    for low, high, exit_run, loss_run in product((0.30, 0.40), (0.60, 0.70), (2, 3), (5, 8)):
        specs.append({
            "operator": "CONSECUTIVE_HYSTERESIS",
            "low_threshold": low,
            "high_threshold": high,
            "exit_run": exit_run,
            "loss_run": loss_run,
            "recovery_run": 2,
            "reacquisition_confirm_frames": 5,
        })
    for window, low, high in product((5, 9), (0.30, 0.40), (0.60, 0.70)):
        specs.append({
            "operator": "SLIDING_WINDOW_VOTE",
            "window": window,
            "low_threshold": low,
            "high_threshold": high,
            "exit_low_fraction": 0.60,
            "loss_low_fraction": 0.80,
            "recovery_high_fraction": 0.60,
            "reacquisition_confirm_frames": 5,
        })
    for alpha, low, high, loss_hold in product((0.70, 0.85), (0.30, 0.40), (0.60, 0.70), (4, 7)):
        specs.append({
            "operator": "LEAKY_EVIDENCE_ACCUMULATOR",
            "alpha": alpha,
            "low_threshold": low,
            "high_threshold": high,
            "loss_hold": loss_hold,
            "reacquisition_confirm_frames": 5,
        })
    if len(specs) != 40 or Counter(spec["operator"] for spec in specs) != {
        "CONSECUTIVE_HYSTERESIS": 16,
        "SLIDING_WINDOW_VOTE": 8,
        "LEAKY_EVIDENCE_ACCUMULATOR": 16,
    }:
        raise ValueError("frozen A3 operator family drift")
    return specs


def _ranked_evidence(trace: dict[str, Any]) -> dict[str, dict[str, float]]:
    distributions = {feature: [] for feature in RAW_FEATURES}
    for episode in trace["episodes"]:
        for evidence in episode["identity_by_candidate"].values():
            for feature in RAW_FEATURES:
                distributions[feature].append(float(evidence[feature]))
    for values in distributions.values():
        values.sort()
    scores = {}
    for episode in trace["episodes"]:
        episode_scores = {}
        for candidate_id, evidence in episode["identity_by_candidate"].items():
            ranks = [
                bisect_right(distributions[feature], float(evidence[feature])) / len(distributions[feature])
                for feature in RAW_FEATURES
            ]
            ranks.sort()
            episode_scores[candidate_id] = (ranks[1] + ranks[2]) / 2.0
        scores[episode["episode_id"]] = episode_scores
    return scores


class TemporalSignals:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.low_run = 0
        self.high_run = 0
        self.window: deque[float] = deque(maxlen=int(spec.get("window", 1)))
        self.belief = 1.0
        self.low_hold = 0

    def update(self, score: float) -> tuple[bool, bool, bool]:
        spec = self.spec
        operator = spec["operator"]
        if operator == "CONSECUTIVE_HYSTERESIS":
            self.low_run = self.low_run + 1 if score < spec["low_threshold"] else 0
            self.high_run = self.high_run + 1 if score >= spec["high_threshold"] else 0
            return (
                self.low_run >= spec["exit_run"],
                self.low_run >= spec["loss_run"],
                self.high_run >= spec["recovery_run"],
            )
        if operator == "SLIDING_WINDOW_VOTE":
            self.window.append(score)
            if len(self.window) < spec["window"]:
                return False, False, False
            low_fraction = sum(value < spec["low_threshold"] for value in self.window) / len(self.window)
            high_fraction = sum(value >= spec["high_threshold"] for value in self.window) / len(self.window)
            return (
                low_fraction >= spec["exit_low_fraction"],
                low_fraction >= spec["loss_low_fraction"],
                high_fraction >= spec["recovery_high_fraction"],
            )
        if operator == "LEAKY_EVIDENCE_ACCUMULATOR":
            self.belief = spec["alpha"] * self.belief + (1.0 - spec["alpha"]) * score
            self.low_hold = self.low_hold + 1 if self.belief < spec["low_threshold"] else 0
            return (
                self.belief < spec["low_threshold"],
                self.low_hold >= spec["loss_hold"],
                self.belief >= spec["high_threshold"],
            )
        raise ValueError(f"unknown temporal operator: {operator}")


def run_temporal_episode(episode: dict[str, Any], scores: dict[str, float], spec: dict[str, Any]) -> dict[str, Any]:
    signal = TemporalSignals(spec)
    internal_state = "TRACKING"
    uncertain_age = 0
    pending_count = 0
    frames_since_confirmed = 0
    frames = []

    for frame in episode["p1_input"]["frames"]:
        candidates = frame["candidates"]
        candidate = candidates[0] if candidates else None
        candidate_id = None if candidate is None else candidate["candidate_id"]
        score = 1.0 if frame["frame_index"] == 0 else scores.get(candidate_id, 0.0)
        low_signal, loss_signal, high_signal = signal.update(score)
        previous_state = internal_state
        event = "NONE"

        if internal_state == "TRACKING":
            if low_signal:
                internal_state = "UNCERTAIN"
                uncertain_age = 1
        elif internal_state == "UNCERTAIN":
            uncertain_age += 1
            if high_signal and candidate is not None:
                internal_state = "TRACKING"
                uncertain_age = 0
            elif loss_signal and uncertain_age >= 2:
                internal_state = "LOST"
                event = "LOSS_DETECTED"
                pending_count = 0
        elif internal_state == "LOST":
            if high_signal and candidate is not None:
                internal_state = "REACQ_PENDING"
                pending_count = 1
        elif internal_state == "REACQ_PENDING":
            if low_signal or candidate is None:
                internal_state = "LOST"
                event = "LOSS_DETECTED"
                pending_count = 0
            elif high_signal:
                pending_count += 1
                if pending_count >= spec["reacquisition_confirm_frames"]:
                    internal_state = "TRACKING"
                    event = "REACQUIRED"
                    uncertain_age = 0
                    pending_count = 0

        asserting = internal_state == "TRACKING" and candidate is not None
        if asserting:
            frames_since_confirmed = 0
            output_state = "TRACKING"
        else:
            frames_since_confirmed += 1
            output_state = "LOST" if internal_state == "LOST" else "UNCERTAIN"
        if previous_state == "TRACKING" and candidate is None and internal_state == "TRACKING":
            output_state = "UNCERTAIN"

        frames.append({
            "frame_index": frame["frame_index"],
            "state": output_state,
            "current_candidate_id": candidate_id if asserting else None,
            "identity_score": None if candidate is None else float(candidate["identity_support"]),
            "stability_score": None if candidate is None else float(candidate["stability"]),
            "oscillation_score": None if candidate is None else float(candidate["oscillation"]),
            "frames_since_confirmed": frames_since_confirmed,
            "event": event,
        })

    return {
        "schema_version": 1,
        "protocol_id": episode["p1_output"]["protocol_id"],
        "episode_id": episode["episode_id"],
        "referent_id": episode["p1_output"]["referent_id"],
        "score_semantics": "ALGORITHMIC_EVIDENCE_NOT_CALIBRATED_PROBABILITY",
        "frames": frames,
    }


def _chatter_metrics(outputs: list[dict[str, Any]]) -> dict[str, int]:
    transitions = 0
    reacquisition_events = 0
    chatter = 0
    for output in outputs:
        frames = output["frames"]
        for left, right in zip(frames, frames[1:]):
            if (left["state"] == "TRACKING") != (right["state"] == "TRACKING"):
                transitions += 1
        for index, frame in enumerate(frames):
            if frame["event"] != "REACQUIRED":
                continue
            reacquisition_events += 1
            end = min(len(frames), index + 31)
            chatter += int(any(row["state"] != "TRACKING" for row in frames[index + 1:end]))
    return {
        "tracking_boundary_transitions": transitions,
        "reacquisition_events": reacquisition_events,
        "reacquisition_chatter_within_30_frames": chatter,
    }


def prepare_candidates(trace_path: Path, sealed_prediction_path: Path, bundle_path: Path) -> dict[str, Any]:
    if sha256(trace_path).upper() != A2_TRACE_SHA256:
        raise ValueError("frozen A2 dense identity trace hash drift")
    trace = r0.read_json(trace_path)
    sealed = r0.read_json(sealed_prediction_path)
    if trace.get("schema_version") != a2.TRACE_SCHEMA:
        raise ValueError("A2 trace schema drift")
    if trace["sealed_prediction_sha256"] != sha256(sealed_prediction_path):
        raise ValueError("A2 trace / sealed prediction binding drift")
    if trace.get("post_initialization_gt_reads") != 0 or trace.get("online_target_memory_updates") != 0:
        raise ValueError("A2 truth firewall or fixed-memory invariant drift")
    if trace.get("candidate_generator_changed") is not False or trace.get("global_search") is not False:
        raise ValueError("candidate-generator invariant drift")

    ranked = _ranked_evidence(trace)
    candidates = []
    for spec in operator_family():
        outputs = [run_temporal_episode(episode, ranked[episode["episode_id"]], spec) for episode in trace["episodes"]]
        candidates.append({
            "candidate_id": _spec_id(spec),
            "operator_spec": spec,
            "chatter": _chatter_metrics(outputs),
            "episodes": [
                {"episode_id": episode["episode_id"], "p1_output": output}
                for episode, output in zip(trace["episodes"], outputs)
            ],
        })
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "claim_role": "RGB_ONLY_TEMPORAL_CANDIDATES_BEFORE_PRIVATE_TRUTH",
        "a2_trace_sha256": sha256(trace_path),
        "sealed_prediction_sha256": sha256(sealed_prediction_path),
        "runner_sha256": _source_sha256(),
        "raw_representation": {
            "model": trace["model"],
            "features": list(RAW_FEATURES),
            "normalization": "per-feature empirical percentile over consumed RGB candidates",
            "aggregation": "median_of_four_percentile_ranks",
            "a2_winner_thresholds_inherited": False,
        },
        "invariants": {
            "candidate_generator_changed": False,
            "online_target_memory_updates": 0,
            "post_initialization_gt_reads": 0,
            "global_search": False,
            "new_data": False,
        },
        "operator_count": len(candidates),
        "operators": dict(Counter(spec["operator"] for spec in operator_family())),
        "a2_chatter_reference": A2_CHATTER_REFERENCE,
        "candidates": candidates,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    r0.write_json(bundle_path, bundle)
    return bundle


def _prediction_for_candidate(sealed: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    outputs = {episode["episode_id"]: episode["p1_output"] for episode in candidate["episodes"]}
    return {
        **sealed,
        "episodes": [
            {**episode, "p1_output": outputs[episode["episode_id"]]}
            for episode in sealed["episodes"]
        ],
    }


def _score_row(candidate: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    aggregate = evaluation["evaluation"]["aggregate"]
    opportunity_episodes = [
        episode for episode in evaluation["evaluation"]["episodes"]
        if episode["metrics"]["reacquisition_opportunity"]
    ]
    loss_declared = [episode for episode in opportunity_episodes if episode["metrics"]["loss_detection_latency_frames"] is not None]
    long_loss_declared = [
        episode for episode in loss_declared if episode["scenario_class"] == "LONG_TARGET_ABSENCE"
    ]
    values = {
        "correct_assertions": int(aggregate["correct_identity_coverage"]["numerator"]),
        "wrong_assertions": int(aggregate["wrong_instance_asserted_frames"]),
        "max_wrong_lock_duration_ms": int(aggregate["wrong_lock_persistence_max_duration_ms"]),
        "false_loss_frames": int(aggregate["false_loss_frames"]),
        "false_reacquisitions": int(aggregate["false_reacquisitions"]),
        "state_expectation_violations": int(aggregate["state_expectation_violations"]),
        "event_expectation_violations": int(aggregate["event_expectation_violations"]),
        "long_target_absence_loss_declarations": len(long_loss_declared),
        "all_reacquisition_opportunity_loss_declarations": len(loss_declared),
        **candidate["chatter"],
    }
    passes = {
        "correct": values["correct_assertions"] >= GATES["correct_assertions_min"],
        "wrong": values["wrong_assertions"] <= GATES["wrong_assertions_max"],
        "wrong_lock": values["max_wrong_lock_duration_ms"] <= GATES["max_wrong_lock_duration_ms_max"],
        "false_loss": values["false_loss_frames"] <= GATES["false_loss_frames_max"],
        "false_reacquisition": values["false_reacquisitions"] <= GATES["false_reacquisitions_max"],
        "transitions": values["tracking_boundary_transitions"] <= GATES["tracking_boundary_transitions_max"],
        "reacquisition_chatter": values["reacquisition_chatter_within_30_frames"] <= GATES["reacquisition_chatter_within_30_frames_max"],
        "long_loss_declaration": values["long_target_absence_loss_declarations"] >= GATES["long_target_absence_loss_declarations_min"],
        "all_loss_declaration": values["all_reacquisition_opportunity_loss_declarations"] >= GATES["all_reacquisition_opportunity_loss_declarations_min"],
        "contract": values["state_expectation_violations"] == 0 and values["event_expectation_violations"] == 0,
    }
    usability_keys = ("false_loss", "false_reacquisition", "transitions", "reacquisition_chatter")
    declaration_keys = ("long_loss_declaration", "all_loss_declaration", "contract")
    return {
        "candidate_id": candidate["candidate_id"],
        "operator_spec": candidate["operator_spec"],
        **values,
        "gate_passes": passes,
        "gate_pass_count": sum(passes.values()),
        "usability_pass": all(passes[key] for key in usability_keys),
        "loss_declaration_pass": all(passes[key] for key in declaration_keys),
        "admission_pass": all(passes.values()),
    }


def _success_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["false_reacquisitions"],
        row["false_loss_frames"],
        row["wrong_assertions"],
        row["max_wrong_lock_duration_ms"],
        -row["correct_assertions"],
        row["tracking_boundary_transitions"],
        row["reacquisition_chatter_within_30_frames"],
        row["candidate_id"],
    )


def choose_terminal(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    admitted = [row for row in rows if row["admission_pass"]]
    if admitted:
        return "TEMPORAL_LOSS_STATE_SIGNAL_ESTABLISHED", sorted(admitted, key=_success_rank)[0]
    delayed = [
        row for row in rows
        if row["gate_passes"]["correct"] and row["usability_pass"] and row["loss_declaration_pass"]
        and (not row["gate_passes"]["wrong"] or not row["gate_passes"]["wrong_lock"])
    ]
    if delayed:
        return "TEMPORAL_SMOOTHING_ONLY_DELAYS_FAILURE", sorted(delayed, key=_success_rank)[0]
    return "TEMPORAL_POLICY_INSUFFICIENT", sorted(
        rows,
        key=lambda row: (-row["gate_pass_count"], *_success_rank(row)),
    )[0]


def evaluate_candidates(
    bundle_path: Path,
    private_path: Path,
    sealed_prediction_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = r0.read_json(bundle_path)
    sealed = r0.read_json(sealed_prediction_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA or bundle.get("operator_count") != 40:
        raise ValueError("A3 candidate bundle drift")
    if bundle["runner_sha256"] != _source_sha256():
        raise ValueError("A3 runner changed after public candidate preparation")
    if bundle["sealed_prediction_sha256"] != sha256(sealed_prediction_path):
        raise ValueError("A3 bundle / sealed prediction binding drift")
    if bundle["raw_representation"].get("a2_winner_thresholds_inherited") is not False:
        raise ValueError("A2 frame threshold inheritance is forbidden")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    evaluations = {}
    with tempfile.TemporaryDirectory(prefix="p1_a3_eval_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        for index, candidate in enumerate(bundle["candidates"]):
            prediction = _prediction_for_candidate(sealed, candidate)
            prediction_path = temporary_dir / f"candidate_{index:03d}_prediction.json"
            evaluation_path = temporary_dir / f"candidate_{index:03d}_evaluation.json"
            r0.write_json(prediction_path, prediction)
            evaluation = r0.evaluate_predictions(private_path, prediction_path, evaluation_path)
            row = _score_row(candidate, evaluation)
            rows.append(row)
            evaluations[candidate["candidate_id"]] = evaluation

    terminal, winner = choose_terminal(rows)
    winner_candidate = next(candidate for candidate in bundle["candidates"] if candidate["candidate_id"] == winner["candidate_id"])
    winner_prediction = _prediction_for_candidate(sealed, winner_candidate)
    winner_evaluation = evaluations[winner["candidate_id"]]
    r0.write_json(output_dir / "winner_prediction.json", winner_prediction)
    r0.write_json(output_dir / "winner_evaluation.json", winner_evaluation)
    result = {
        "schema_version": RESULT_SCHEMA,
        "claim_role": "CONSUMED_ADT_TEMPORAL_POLICY_DEVELOPMENT_ONLY_NO_POLICY_ADMISSION_NO_SCIENTIFIC_VERDICT",
        "terminal": terminal,
        "policy_admission": "NO_POLICY_ADMISSION",
        "a2_trace_sha256": bundle["a2_trace_sha256"],
        "candidate_bundle_sha256": sha256(bundle_path),
        "raw_representation": bundle["raw_representation"],
        "invariants": bundle["invariants"],
        "search": {
            "operator_count": len(rows),
            "operators": bundle["operators"],
            "second_round_search": False,
            "learned_temporal_model": False,
        },
        "gates": GATES,
        "a2_references": {
            "correct_assertions": 80,
            "wrong_assertions": 445,
            "max_wrong_lock_duration_ms": 2700,
            "false_loss_frames": 304,
            "false_reacquisitions": 29,
            **A2_CHATTER_REFERENCE,
        },
        "winner": winner,
        "winner_frozen_evaluator": winner_evaluation,
        "admission_candidate_count": sum(row["admission_pass"] for row in rows),
        "all_candidates": rows,
    }
    r0.write_json(output_dir / "sweep_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--a2-trace", type=Path, required=True)
    prepare.add_argument("--sealed-prediction", type=Path, required=True)
    prepare.add_argument("--bundle", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--private-input", type=Path, required=True)
    evaluate.add_argument("--sealed-prediction", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        bundle = prepare_candidates(args.a2_trace, args.sealed_prediction, args.bundle)
        print(json.dumps({
            "operator_count": bundle["operator_count"],
            "operators": bundle["operators"],
            "a2_winner_thresholds_inherited": bundle["raw_representation"]["a2_winner_thresholds_inherited"],
            "post_initialization_gt_reads": bundle["invariants"]["post_initialization_gt_reads"],
        }, sort_keys=True))
    else:
        result = evaluate_candidates(args.bundle, args.private_input, args.sealed_prediction, args.output_dir)
        print(json.dumps({"terminal": result["terminal"], "winner": result["winner"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
