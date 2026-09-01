"""Run leave-one-cohort-out X95 Development on eleven consumed CARLA cohorts."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
WORK = Path(r"E:\linnan\linnan\artifacts.local\work\x31-growth-diagnostic-20260831")
for value in (HERE, WORK):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x94_one_frame_full_dropout_continuity as x94  # noqa: E402
import dtr_carla_x95_credentialed_hazard_state_model as x95  # noqa: E402
import run_dtr_carla_x94_consumed_development as base94  # noqa: E402


runner = base94.runner
ARM_X94 = x94.ARM_X94
ARM_HYSTERESIS = "BASELINE_X94_ZERO_POINT_SIX_SECOND_HYSTERESIS"
ARM_FORWARD_FILL = "BASELINE_X94_PLAIN_FULL_DROPOUT_FORWARD_FILL"
ARM_EMISSION = "X95_LOGISTIC_EMISSION_ONLY"
ARM_X95 = x95.ARM_X95
ARMS = (ARM_X94, ARM_HYSTERESIS, ARM_FORWARD_FILL, ARM_EMISSION, ARM_X95)

COHORT_EXPERIMENTS = {
    "C26": "dtr-carla-c26-x56-source-corrected",
    "C27": "dtr-carla-c27-x57-daylight-transfer",
    "C28": "dtr-carla-c28-x59-mixed-lighting",
    "C32": "dtr-carla-c32-x64-l03-restored",
    "C34": "dtr-carla-c34-x65-fresh-source",
    "C35": "dtr-carla-c35-x73-fresh-confirmation",
    "C36": "dtr-carla-c36-x74-fresh-confirmation",
    "C37": "dtr-carla-c37-x75-fresh-confirmation",
    "C39": "dtr-carla-c39-x79-fresh-confirmation",
    "C40": "dtr-carla-c40-x81-fresh-confirmation",
    "C41": "dtr-carla-c41-x82-fresh-confirmation",
}


def _one(paths: Sequence[Path], contract: str) -> Path:
    x24.require(len(paths) == 1, f"{contract}:{len(paths)}")
    return paths[0].resolve(strict=True)


def discover_inputs(
    x94_root: Path, experiments_root: Path
) -> dict[str, dict[str, Path]]:
    inputs: dict[str, dict[str, Path]] = {}
    for cohort_id, experiment_name in COHORT_EXPERIMENTS.items():
        prediction = _one(
            list(x94_root.glob(f"{cohort_id.lower()}-x94-*/predictions-x94.json")),
            f"x95_{cohort_id}_x94_prediction",
        )
        experiment = (experiments_root / experiment_name).resolve(strict=True)
        sources = [
            path.parent
            for path in experiment.rglob("result.json")
            if (path.parent / "evaluator" / "episodes").is_dir()
        ]
        source = _one(sources, f"x95_{cohort_id}_source")
        inputs[cohort_id] = {"prediction": prediction, "source": source}
    return inputs


def _prefix(rows: Sequence[Mapping[str, Any]], end_s: float) -> list[dict[str, Any]]:
    return [copy.deepcopy(dict(row)) for row in rows if float(row["time_s"]) <= end_s + x24.EPSILON]


def _align(
    truth: Sequence[Mapping[str, Any]], frames: Sequence[Mapping[str, Any]], contract: str
) -> None:
    x24.require(len(truth) == len(frames), f"{contract}_length")
    for expected, observed in zip(truth, frames):
        x24.require(
            int(expected["sample_index"]) == int(observed["sample_index"])
            and abs(float(expected["time_s"]) - float(observed["time_s"])) <= x24.EPSILON,
            f"{contract}_frame",
        )


def load_cohort(cohort_id: str, paths: Mapping[str, Path]) -> dict[str, Any]:
    prediction = runner.base.read_json(paths["prediction"])
    x24.require(
        str(prediction.get("status", "")).startswith("SEALED_")
        and set(prediction.get("arms", [])) == {ARM_X94},
        f"x95_{cohort_id}_sealed_x94",
    )
    episodes: dict[str, Any] = {}
    for episode_id in runner.EPISODES:
        frames = _prefix(
            prediction["episodes"][episode_id]["frames"],
            runner.SCORE_END[episode_id],
        )
        truth = _prefix(
            runner.base.read_jsonl(
                paths["source"] / "evaluator" / "episodes" / episode_id / "frames.jsonl"
            ),
            runner.SCORE_END[episode_id],
        )
        _align(truth, frames, f"x95_{cohort_id}_{episode_id}")
        episodes[episode_id] = {"frames": frames, "truth": truth}
    return {"prediction": prediction, "episodes": episodes}


def training_rows(
    cohorts: Mapping[str, Mapping[str, Any]], held_out: str
) -> tuple[list[np.ndarray], list[bool]]:
    vectors: list[np.ndarray] = []
    labels: list[bool] = []
    for cohort_id, cohort in cohorts.items():
        if cohort_id == held_out:
            continue
        for episode in cohort["episodes"].values():
            credentialed_parent_ids: set[str] = set()
            for frame, truth in zip(episode["frames"], episode["truth"]):
                arm = frame["arms"][ARM_X94]
                credentialed_parent_ids.update(
                    str(value)
                    for value in arm.get("x75_collision_credential_birth_parent_ids", [])
                )
                vectors.append(x95.observation(frame, credentialed_parent_ids)["vector"])
                labels.append(bool(truth["truth"]["future_contact_within_horizon"]))
    return vectors, labels


def _prediction_row(
    frame: Mapping[str, Any], risk: bool, minimum_entry_s: float | None
) -> dict[str, Any]:
    return {
        "sample_index": int(frame["sample_index"]),
        "time_s": float(frame["time_s"]),
        "route_risk": bool(risk),
        "minimum_entry_s": minimum_entry_s if risk else None,
    }


def baseline_x94(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _prediction_row(
            frame,
            bool(frame["arms"][ARM_X94].get("route_risk")),
            frame["arms"][ARM_X94].get("minimum_entry_s"),
        )
        for frame in frames
    ]


def baseline_hysteresis(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    last_risk_time_s: float | None = None
    last_entry: float | None = None
    last_receipt: str | None = None
    for frame in frames:
        arm = frame["arms"][ARM_X94]
        now_s = float(frame["time_s"])
        receipt = str(arm.get("plan_receipt_sha256") or "")
        release = x95._active_release(arm)
        observed = bool(arm.get("route_risk"))
        if observed:
            risk = True
            last_risk_time_s = now_s
            last_entry = arm.get("minimum_entry_s")
            last_receipt = receipt
        elif (
            last_risk_time_s is not None
            and now_s - last_risk_time_s <= x24.HOLD_WINDOW_S + x24.EPSILON
            and receipt == last_receipt
            and not release
            and not bool(arm.get("route_mode_changed"))
        ):
            risk = True
            last_entry = None if last_entry is None else max(0.0, float(last_entry) - (now_s - last_risk_time_s))
        else:
            risk = False
            last_risk_time_s = None
            last_entry = None
            last_receipt = None
        output.append(_prediction_row(frame, risk, last_entry))
    return output


def baseline_forward_fill(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_risk = False
    previous_entry: float | None = None
    for frame in frames:
        arm = frame["arms"][ARM_X94]
        observed = bool(arm.get("route_risk"))
        full_dropout = (
            int(frame.get("raw_candidates", 0)) == 0
            and int(frame.get("metric_footprint_measurements", 0)) == 0
        )
        risk = observed or (previous_risk and full_dropout)
        entry = arm.get("minimum_entry_s") if observed else previous_entry
        output.append(_prediction_row(frame, risk, entry))
        previous_risk = risk
        previous_entry = entry if risk else None
    return output


def emission_only(
    frames: Sequence[Mapping[str, Any]], model: x95.LogisticEmission
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    credentialed_parent_ids: set[str] = set()
    for frame in frames:
        arm = frame["arms"][ARM_X94]
        credentialed_parent_ids.update(
            str(value)
            for value in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        probability = model.probability(x95.observation(frame, credentialed_parent_ids)["vector"])
        risk = probability >= float(x95.fixed_constants()["emission_threshold"])
        row = _prediction_row(frame, risk, arm.get("minimum_entry_s"))
        row["emission_probability"] = probability
        output.append(row)
    return output


def _false_runs(values: Sequence[bool]) -> int:
    runs = 0
    inside = False
    for value in values:
        if not value and not inside:
            runs += 1
            inside = True
        elif value:
            inside = False
    return runs


def _clear_latencies(
    truth: Sequence[Mapping[str, Any]], prediction: Sequence[Mapping[str, Any]]
) -> tuple[list[float], int]:
    labels = [bool(row["truth"]["future_contact_within_horizon"]) for row in truth]
    values: list[float] = []
    censored = 0
    for index in range(1, len(labels)):
        if labels[index - 1] and not labels[index]:
            end_time = float(truth[index]["time_s"])
            clear = next(
                (
                    float(row["time_s"])
                    for row in prediction[index:]
                    if not bool(row["route_risk"])
                ),
                None,
            )
            if clear is None:
                censored += 1
            else:
                values.append(max(0.0, clear - end_time))
    return values, censored


def aggregate_metrics(
    truth: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    confusion = runner.base.confusion(truth, predictions)
    hits = 0
    false_segments = 0
    safe_seconds = 0.0
    leads: list[float] = []
    clear_latencies: list[float] = []
    clear_censored = 0
    fragment_runs = 0
    for key, truth_rows in truth.items():
        episode_id = key.split(":", 1)[1]
        predicted = predictions[key]
        if episode_id in runner.CONTACT:
            metric = runner.base.contact_metrics(truth_rows, predicted)
            hits += int(metric["event_detected_before_contact"])
            if metric["first_alert_lead_seconds"] is not None:
                leads.append(float(metric["first_alert_lead_seconds"]))
            positive_values = [
                bool(frame["route_risk"])
                for target, frame in zip(truth_rows, predicted)
                if bool(target["truth"]["future_contact_within_horizon"])
            ]
            fragment_runs += max(0, _false_runs(positive_values) - int(not positive_values[0]))
        else:
            segment = runner.base.false_segments(predicted, runner.SAFE_START[episode_id])
            false_segments += int(segment["false_alert_segment_count"])
            safe_seconds += max(0.0, float(predicted[-1]["time_s"]) - runner.SAFE_START[episode_id])
        values, censored = _clear_latencies(truth_rows, predicted)
        clear_latencies.extend(values)
        clear_censored += censored
    contact_events = sum(key.split(":", 1)[1] in runner.CONTACT for key in truth)
    event_precision = hits / (hits + false_segments) if hits + false_segments else 0.0
    event_recall = hits / contact_events if contact_events else 0.0
    event_f1 = (
        2.0 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    return {
        **confusion,
        "event_hits": hits,
        "event_total": contact_events,
        "false_alert_segments": false_segments,
        "false_alert_segments_per_minute": false_segments / max(safe_seconds / 60.0, x24.EPSILON),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "median_lead_s": None if not leads else statistics.median(leads),
        "p10_lead_s": None if not leads else float(np.quantile(np.asarray(leads), 0.10)),
        "median_clear_latency_s": None if not clear_latencies else statistics.median(clear_latencies),
        "clear_latency_censored_events": clear_censored,
        "fragment_false_runs": fragment_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x94-root", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    x94_root = args.x94_root.resolve(strict=True)
    experiments_root = args.experiments_root.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)
    inputs = discover_inputs(x94_root, experiments_root)
    cohorts = {cohort_id: load_cohort(cohort_id, paths) for cohort_id, paths in inputs.items()}

    truth_all: dict[str, list[dict[str, Any]]] = {}
    prediction_all = {arm: {} for arm in ARMS}
    fold_models: dict[str, Any] = {}
    fold_diagnostics: dict[str, Any] = {}
    for held_out in COHORT_EXPERIMENTS:
        vectors, labels = training_rows(cohorts, held_out)
        model = x95.fit_logistic(vectors, labels)
        fold_models[held_out] = {
            **model.to_json(),
            "training_rows": len(labels),
            "training_positive_rate": sum(labels) / len(labels),
        }
        state_counts = {name: 0 for name in x95.fixed_constants()["states"]}
        transitions: dict[str, int] = {}
        for episode_id, episode in cohorts[held_out]["episodes"].items():
            key = f"{held_out}:{episode_id}"
            frames = episode["frames"]
            truth_all[key] = episode["truth"]
            prediction_all[ARM_X94][key] = baseline_x94(frames)
            prediction_all[ARM_HYSTERESIS][key] = baseline_hysteresis(frames)
            prediction_all[ARM_FORWARD_FILL][key] = baseline_forward_fill(frames)
            prediction_all[ARM_EMISSION][key] = emission_only(frames, model)
            decoded, diagnostics = x95.decode_episode(frames, model)
            prediction_all[ARM_X95][key] = decoded
            for name, count in diagnostics["state_counts"].items():
                state_counts[name] += int(count)
            for name, count in diagnostics["transition_counts"].items():
                transitions[name] = transitions.get(name, 0) + int(count)
        fold_diagnostics[held_out] = {
            "state_counts": state_counts,
            "transition_counts": transitions,
        }

    aggregate = {
        arm: aggregate_metrics(truth_all, values)
        for arm, values in prediction_all.items()
    }
    baseline = aggregate[ARM_X94]
    candidate = aggregate[ARM_X95]
    effect = {
        "tp_delta": int(candidate["tp"]) - int(baseline["tp"]),
        "fp_delta": int(candidate["fp"]) - int(baseline["fp"]),
        "fn_delta": int(candidate["fn"]) - int(baseline["fn"]),
        "f1_delta": float(candidate["f1"]) - float(baseline["f1"]),
        "event_f1_delta": float(candidate["event_f1"]) - float(baseline["event_f1"]),
        "false_segments_delta": int(candidate["false_alert_segments"])
        - int(baseline["false_alert_segments"]),
    }
    checks = {
        "event_f1_gain_at_least_0_03": effect["event_f1_delta"] >= 0.03,
        "frame_f1_non_decreasing": float(candidate["f1"]) >= float(baseline["f1"]),
        "precision_within_0_005_of_x94": float(candidate["precision"])
        >= float(baseline["precision"]) - 0.005,
        "event_recall_non_decreasing": float(candidate["event_recall"])
        >= float(baseline["event_recall"]),
        "all_five_states_exercised": all(
            sum(fold["state_counts"][state] for fold in fold_diagnostics.values()) > 0
            for state in x95.fixed_constants()["states"]
        ),
    }
    decision = (
        "DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_STRUCTURAL_EFFECT_POSITIVE"
        if all(checks.values())
        else "DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_GATE_NOT_MET"
    )

    models_path = output_dir / "fold-models.json"
    predictions_path = output_dir / "predictions.json"
    summary_path = output_dir / "summary.json"
    runner.base.write_json_exclusive(models_path, fold_models)
    runner.base.write_json_exclusive(
        predictions_path,
        {
            "schema": "blindassist-dtr-carla-x95-consumed-cross-validation-predictions-v1",
            "arms": list(ARMS),
            "predictions": prediction_all,
        },
    )
    summary = {
        "schema": "blindassist-dtr-carla-x95-consumed-cross-validation-v1",
        "status": "COMPLETE",
        "decision": decision,
        "cohorts": list(COHORT_EXPERIMENTS),
        "validation": "LEAVE_ONE_CONSUMED_COHORT_OUT",
        "fixed_constants": x95.fixed_constants(),
        "aggregate": aggregate,
        "effect_vs_x94": effect,
        "checks": checks,
        "fold_diagnostics": fold_diagnostics,
        "source": {
            "x95_predictor_sha256": runner.base.sha256_file(Path(x95.__file__).resolve()),
            "runner_sha256": runner.base.sha256_file(Path(__file__).resolve()),
            "fold_models_sha256": runner.base.sha256_file(models_path),
            "predictions_sha256": runner.base.sha256_file(predictions_path),
            "inputs": {
                cohort_id: {
                    "x94_predictions_sha256": runner.base.sha256_file(paths["prediction"]),
                    "source_result_sha256": runner.base.sha256_file(paths["source"] / "result.json"),
                }
                for cohort_id, paths in inputs.items()
            },
        },
        "claim_boundary": {
            "consumed_posthoc_synthetic_development": True,
            "held_out_folds_are_not_fresh_confirmation": True,
            "event_layer_only": True,
            "bottom_up_geometry_and_credentials_inherited_from_x94": True,
            "real_world_or_safety_authority": False,
        },
    }
    runner.base.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "decision": decision,
                "aggregate": aggregate,
                "effect_vs_x94": effect,
                "checks": checks,
                "summary_sha256": runner.base.sha256_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
