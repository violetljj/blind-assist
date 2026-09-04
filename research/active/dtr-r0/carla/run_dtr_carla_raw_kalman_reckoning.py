"""Seal and score raw-input Kalman baselines on the 11 consumed CARLA cohorts.

All new predictions are written before this process opens evaluator rows.  The
result is retrospective Development evidence only; it cannot confirm or
promote an arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_raw_kalman_baseline as baseline  # noqa: E402
import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x94_one_frame_full_dropout_continuity as x94  # noqa: E402
import run_dtr_carla_x95_consumed_cross_validation as event_eval  # noqa: E402


SCHEMA = "blindassist-dtr-carla-raw-kalman-reckoning-v1"
ARM_X24 = x24.ARM_X24
ARM_X94 = x94.ARM_X94
ARMS = (*baseline.ARMS, ARM_X24, ARM_X94)

COHORTS = {
    "C26": {
        "experiment": "dtr-carla-c26-x56-source-corrected",
        "x24_run": "dtr-carla-x56-c26-confirmation/c26-x56-20260831-204200",
    },
    "C27": {
        "experiment": "dtr-carla-c27-x57-daylight-transfer",
        "x24_run": "dtr-carla-x57-c27-confirmation/c27-x57-20260831-220500",
    },
    "C28": {
        "experiment": "dtr-carla-c28-x59-mixed-lighting",
        "x24_run": "dtr-carla-x59-c28-confirmation/c28-x59-20260831-225500",
    },
    "C32": {
        "experiment": "dtr-carla-c32-x64-l03-restored",
        "x24_run": "dtr-carla-x64-c32-confirmation/c32-x64-20260901-024500",
    },
    "C34": {
        "experiment": "dtr-carla-c34-x65-fresh-source",
        "x24_run": "dtr-carla-x65-c34-confirmation/c34-x65-20260901-044500",
    },
    "C35": {
        "experiment": "dtr-carla-c35-x73-fresh-confirmation",
        "x24_run": "dtr-carla-x73-c35-confirmation/c35-x73-20260901-073300",
    },
    "C36": {
        "experiment": "dtr-carla-c36-x74-fresh-confirmation",
        "x24_run": "dtr-carla-x74-c36-confirmation/c36-x74-20260901-082100",
    },
    "C37": {
        "experiment": "dtr-carla-c37-x75-fresh-confirmation",
        "x24_run": "dtr-carla-x75-c37-confirmation/c37-x75-20260901-091600",
    },
    "C39": {
        "experiment": "dtr-carla-c39-x79-fresh-confirmation",
        "x24_run": "dtr-carla-x79-c39-confirmation/c39-x79-20260901-143109",
    },
    "C40": {
        "experiment": "dtr-carla-c40-x81-fresh-confirmation",
        "x24_run": "dtr-carla-x81-c40-confirmation/c40-x81-20260901-162200",
    },
    "C41": {
        "experiment": "dtr-carla-c41-x82-fresh-confirmation",
        "x24_run": "dtr-carla-x82-c41-confirmation/c41-x82-20260901-174000",
    },
}


def _one(values: Sequence[Path], label: str) -> Path:
    x24.require(len(values) == 1, f"{label}:{len(values)}")
    return values[0].resolve(strict=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    x24.require(isinstance(value, dict), f"json_object:{path}")
    return value


def discover_inputs(
    *,
    evidence_root: Path,
    experiments_root: Path,
    x94_root: Path,
    cohort_ids: Sequence[str],
) -> dict[str, dict[str, Path]]:
    output: dict[str, dict[str, Path]] = {}
    for cohort_id in cohort_ids:
        spec = COHORTS[cohort_id]
        x24_run = (evidence_root / spec["x24_run"]).resolve(strict=True)
        x24_prediction = x24.run_paths(x24_run)["predictions"].resolve(strict=True)
        x94_prediction = _one(
            list(x94_root.glob(f"{cohort_id.lower()}-x94-*/predictions-x94.json")),
            f"{cohort_id}_x94_prediction",
        )
        experiment = (experiments_root / spec["experiment"]).resolve(strict=True)
        source = _one(
            [
                result.parent
                for result in experiment.rglob("result.json")
                if (result.parent / "evaluator" / "episodes").is_dir()
            ],
            f"{cohort_id}_source",
        )
        output[cohort_id] = {
            "x24_run": x24_run,
            "x24_prediction": x24_prediction,
            "x94_prediction": x94_prediction,
            "source": source,
        }
    return output


def seal_raw_prediction(cohort_id: str, inputs: Mapping[str, Path], output_path: Path) -> dict[str, Any]:
    x24.require(not output_path.exists(), f"raw_kalman_prediction_exists:{output_path}")
    frozen, contract, candidate_values = x24.require_freeze(inputs["x24_run"])
    cursor = 0
    episodes: dict[str, Any] = {}
    for episode in contract.episodes:
        count = len(episode.observations)
        episodes[episode.episode_id] = baseline.predict_episode(
            episode,
            candidate_values[cursor : cursor + count],
            contract.calibration,
        )
        cursor += count
    x24.require(cursor == len(candidate_values), f"{cohort_id}_candidate_cursor")
    value = {
        "schema": baseline.PREDICTION_SCHEMA,
        "status": "SEALED_TRUTH_BLIND_PENDING_CONSUMED_SCORE",
        "experiment_id": baseline.EXPERIMENT_ID,
        "cohort_id": cohort_id,
        "truth_blind": True,
        "arms": list(baseline.ARMS),
        "episodes": episodes,
        "fixed_constants": baseline.fixed_constants(),
        "source": {
            "x24_freeze": str(x24.run_paths(inputs["x24_run"])["freeze"]),
            "x24_freeze_sha256": x24.sha256_file(x24.run_paths(inputs["x24_run"])["freeze"]),
            "model_manifest_sha256": contract.manifest_sha256,
            "candidate_aggregate_sha256": frozen["candidates"]["aggregate_sha256"],
        },
        "claim_boundary": {
            "evaluator_opened_during_prediction": False,
            "consumed_development_only": True,
            "fresh_confirmation": False,
            "x24_x73_x94_tracks_consumed": False,
        },
    }
    x24.write_json_exclusive(output_path, value)
    return {
        "path": str(output_path.resolve(strict=True)),
        "sha256": x24.sha256_file(output_path),
        "episodes": len(episodes),
        "frames": sum(len(row["frames"]) for row in episodes.values()),
    }


def _arm_rows(
    envelope: Mapping[str, Any], episode_id: str, arm: str
) -> list[dict[str, Any]]:
    frames = event_eval._prefix(
        envelope["episodes"][episode_id]["frames"],
        event_eval.runner.SCORE_END[episode_id],
    )
    return [
        {
            "sample_index": int(frame["sample_index"]),
            "time_s": float(frame["time_s"]),
            "route_risk": bool(frame["arms"][arm]["route_risk"]),
            "minimum_entry_s": frame["arms"][arm].get("minimum_entry_s"),
        }
        for frame in frames
    ]


def score(
    inputs: Mapping[str, Mapping[str, Path]], prediction_paths: Mapping[str, Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_all: dict[str, list[dict[str, Any]]] = {}
    predictions_all: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
    per_cohort: dict[str, Any] = {}

    for cohort_id, paths in inputs.items():
        raw = _read_json(prediction_paths[cohort_id])
        x24_envelope = _read_json(paths["x24_prediction"])
        x94_envelope = _read_json(paths["x94_prediction"])
        x24.require(raw.get("status") == "SEALED_TRUTH_BLIND_PENDING_CONSUMED_SCORE", f"{cohort_id}_raw_seal")
        x24.require(set(raw.get("arms", [])) == set(baseline.ARMS), f"{cohort_id}_raw_arms")
        x24.require(ARM_X24 in x24_envelope.get("arms", []), f"{cohort_id}_x24_arm")
        x24.require(set(x94_envelope.get("arms", [])) == {ARM_X94}, f"{cohort_id}_x94_arm")
        cohort_truth: dict[str, list[dict[str, Any]]] = {}
        cohort_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
            arm: {} for arm in ARMS
        }
        for episode_id in event_eval.runner.EPISODES:
            key = f"{cohort_id}:{episode_id}"
            truth = event_eval._prefix(
                event_eval.runner.base.read_jsonl(
                    paths["source"] / "evaluator" / "episodes" / episode_id / "frames.jsonl"
                ),
                event_eval.runner.SCORE_END[episode_id],
            )
            raw_reference = _arm_rows(raw, episode_id, baseline.ARM_ROUTE)
            event_eval._align(truth, raw_reference, f"{cohort_id}_{episode_id}_raw")
            truth_all[key] = truth
            cohort_truth[key] = truth
            for arm in baseline.ARMS:
                rows = _arm_rows(raw, episode_id, arm)
                predictions_all[arm][key] = rows
                cohort_predictions[arm][key] = rows
            for arm, envelope in ((ARM_X24, x24_envelope), (ARM_X94, x94_envelope)):
                rows = _arm_rows(envelope, episode_id, arm)
                event_eval._align(truth, rows, f"{cohort_id}_{episode_id}_{arm}")
                predictions_all[arm][key] = rows
                cohort_predictions[arm][key] = rows
        per_cohort[cohort_id] = {
            arm: event_eval.aggregate_metrics(cohort_truth, cohort_predictions[arm]) for arm in ARMS
        }

    aggregate = {
        arm: event_eval.aggregate_metrics(truth_all, predictions_all[arm]) for arm in ARMS
    }
    return aggregate, per_cohort


def _comparison(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    simple = metrics[baseline.ARM_HYSTERESIS]
    complete = metrics[ARM_X94]
    event_delta = float(complete["event_f1"]) - float(simple["event_f1"])
    if event_delta > 0.01:
        direction = "X94_AHEAD_POINT_ESTIMATE_UNCERTAINTY_PENDING"
    elif event_delta < -0.01:
        direction = "RAW_KALMAN_HYSTERESIS_AHEAD_POINT_ESTIMATE"
    else:
        direction = "PRACTICAL_TIE_POINT_ESTIMATE"
    return {
        "strong_simple_arm": baseline.ARM_HYSTERESIS,
        "x94_event_f1_delta": event_delta,
        "x94_false_segment_delta": int(complete["false_alert_segments"]) - int(simple["false_alert_segments"]),
        "x94_fragmentation_delta": int(complete["fragment_false_runs"]) - int(simple["fragment_false_runs"]),
        "x94_frame_f1_delta": float(complete["f1"]) - float(simple["f1"]),
        "direction": direction,
        "paired_cluster_bootstrap": "PENDING_SHARED_ONE_TO_ONE_EVENT_MATCHER",
        "paper_identity_decision": "NOT_YET_ADMISSIBLE",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--x94-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        action="append",
        choices=tuple(COHORTS),
        help="Run only the named cohort; repeat for more. Default: all eleven.",
    )
    args = parser.parse_args()

    evidence_root = args.evidence_root.resolve(strict=True)
    experiments_root = args.experiments_root.resolve(strict=True)
    x94_root = args.x94_root.resolve(strict=True)
    cohort_ids = tuple(args.cohort or COHORTS)
    inputs = discover_inputs(
        evidence_root=evidence_root,
        experiments_root=experiments_root,
        x94_root=x94_root,
        cohort_ids=cohort_ids,
    )
    # Resolve every durable input before creating a formal output directory.
    # A missing retained raw payload is a preflight failure, not a partial run.
    for cohort_id, paths in inputs.items():
        frozen = _read_json(x24.run_paths(paths["x24_run"])["freeze"])
        x24.require(Path(frozen["model_root"]).is_dir(), f"{cohort_id}_raw_model_missing")
        x24.require(Path(frozen["candidates"]["path"]).is_dir(), f"{cohort_id}_candidates_missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve(strict=True)

    prediction_receipts: dict[str, Any] = {}
    prediction_paths: dict[str, Path] = {}
    for cohort_id, paths in inputs.items():
        output_path = output_dir / f"predictions-{cohort_id.lower()}.json"
        prediction_receipts[cohort_id] = seal_raw_prediction(cohort_id, paths, output_path)
        prediction_paths[cohort_id] = output_path

    manifest = {
        "schema": SCHEMA,
        "status": "ALL_RAW_PREDICTIONS_SEALED_BEFORE_EVALUATOR_OPEN",
        "cohorts": list(cohort_ids),
        "new_arms": list(baseline.ARMS),
        "comparison_arms": [ARM_X24, ARM_X94],
        "fixed_constants": baseline.fixed_constants(),
        "algorithm": {
            "path": str(Path(baseline.__file__).resolve()),
            "sha256": x24.sha256_file(Path(baseline.__file__).resolve()),
        },
        "predictions": prediction_receipts,
        "existing_sealed_inputs": {
            cohort_id: {
                "x24_prediction_sha256": x24.sha256_file(paths["x24_prediction"]),
                "x94_prediction_sha256": x24.sha256_file(paths["x94_prediction"]),
                "source_root": str(paths["source"]),
            }
            for cohort_id, paths in inputs.items()
        },
        "claim_boundary": {
            "consumed_development_only": True,
            "fair_raw_measurement_input_for_new_arms": True,
            "fresh_confirmation": False,
            "full_eleven_cohort_reckoning": len(cohort_ids) == len(COHORTS),
            "no_x97": True,
        },
    }
    manifest_path = output_dir / "prediction-manifest.json"
    x24.write_json_exclusive(manifest_path, manifest)

    metrics, per_cohort = score(inputs, prediction_paths)
    summary = {
        "schema": SCHEMA,
        "status": "PRELIMINARY_CONSUMED_RAW_INPUT_RECKONING_POINT_ESTIMATES",
        "prediction_manifest": {
            "path": str(manifest_path),
            "sha256": x24.sha256_file(manifest_path),
        },
        "metrics": metrics,
        "per_cohort": per_cohort,
        "comparison": _comparison(metrics),
        "not_evaluable": {
            "CTRV": "NO_FROZEN_CAUSAL_TARGET_YAW_RATE_IN_SHARED_RAW_MEASUREMENT_CONTRACT",
            "tiny_learned_predictor": "PENDING_TRAINING_GROUP_FREEZE",
            "X73": "PENDING_RAW_REPRODUCTION_OR_VERIFIED_SEALED_EXPORT",
            "X95_fresh": "FORBIDDEN_ON_CONSUMED_COHORTS_FOR_PROMOTION",
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    summary_path = output_dir / "summary.json"
    x24.write_json_exclusive(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path.resolve(strict=True)),
                "summary_sha256": x24.sha256_file(summary_path),
                "status": summary["status"],
                "comparison": summary["comparison"],
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
