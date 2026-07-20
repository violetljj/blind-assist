#!/usr/bin/env python3
"""Evaluate source-isolated mechanism-routed alert experts.

This is a diagnostic upper-bound and routing test over the six qualified
matched pairs only.  Each held-out source is excluded from the mechanism
router and both alert experts.  The router sees frozen observable features and
mechanism supervision, never the alert label of a held-out episode.  Oracle
mechanism results are reported separately and cannot authorize deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_multichannel_risk_profile_probe as profile
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_segformer_free_space_probe as clearance


SCHEMA = "blindassist_public_silver_mechanism_routed_expert_probe_v1"
MECHANISMS = (temporal.DYNAMIC, temporal.STATIC)


def qualified_rows(
    episodes: Sequence[dict[str, Any]], mechanism_report: Path
) -> tuple[list[dict[str, Any]], np.ndarray]:
    qualified = temporal.load_qualified_pair_contract(mechanism_report)
    pair_to_mechanism = {
        pair_id: mechanism
        for mechanism, pair_ids in qualified.items()
        for pair_id in pair_ids
    }
    rows: list[dict[str, Any]] = []
    mechanisms: list[int] = []
    for episode in episodes:
        mechanism = pair_to_mechanism.get(episode.get("counterfactual_pair_id"))
        if mechanism is None:
            continue
        rows.append(episode)
        mechanisms.append(MECHANISMS.index(mechanism))
    if len(rows) != 12 or set(mechanisms) != {0, 1}:
        raise ValueError("mechanism expert probe requires exactly six qualified two-episode pairs")
    return rows, np.asarray(mechanisms, dtype=np.int64)


def predict(fitted: dict[str, Any], features: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=np.float64) @ fitted["kernel"] + fitted["bias"]
    return np.argmax(logits, axis=1).astype(np.int64)


def nested_source_evaluation(
    features: np.ndarray,
    alert_labels: np.ndarray,
    mechanism_labels: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    alerts = np.asarray(alert_labels, dtype=np.int64)
    mechanisms = np.asarray(mechanism_labels, dtype=np.int64)
    if x.ndim != 2 or not (len(x) == len(alerts) == len(mechanisms) == len(episode_ids) == len(source_ids)):
        raise ValueError("nested evaluation inputs must be aligned")
    source_array = np.asarray(source_ids, dtype=object)
    routed_alert = np.full(len(alerts), -1, dtype=np.int64)
    oracle_alert = np.full(len(alerts), -1, dtype=np.int64)
    routed_mechanism = np.full(len(alerts), -1, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for held_out_source in dict.fromkeys(source_ids):
        holdout = source_array == held_out_source
        train = ~holdout
        if set(mechanisms[train].tolist()) != {0, 1}:
            raise ValueError(f"router training fold lacks a mechanism: {held_out_source}")
        router = common.fit_episode_ridge(x[train], mechanisms[train], ridge=ridge, class_balanced=True)
        routed_mechanism[holdout] = predict(router, x[holdout])
        experts: dict[int, dict[str, Any]] = {}
        for mechanism_id in (0, 1):
            expert_train = train & (mechanisms == mechanism_id)
            if set(alerts[expert_train].tolist()) != {0, 1}:
                raise ValueError(f"expert fold lacks an alert class: {held_out_source}: {MECHANISMS[mechanism_id]}")
            experts[mechanism_id] = common.fit_episode_ridge(
                x[expert_train], alerts[expert_train], ridge=ridge, class_balanced=True
            )
        held_indices = np.flatnonzero(holdout)
        for index in held_indices:
            oracle_mechanism = int(mechanisms[index])
            predicted_mechanism = int(routed_mechanism[index])
            oracle_alert[index] = int(predict(experts[oracle_mechanism], x[index:index + 1])[0])
            routed_alert[index] = int(predict(experts[predicted_mechanism], x[index:index + 1])[0])
        folds.append({
            "held_out_source_id": held_out_source,
            "held_out_episode_ids": [episode_ids[index] for index in held_indices],
            "expected_mechanism": mechanisms[holdout].tolist(),
            "routed_mechanism": routed_mechanism[holdout].tolist(),
            "expected_alert": alerts[holdout].tolist(),
            "oracle_expert_alert": oracle_alert[holdout].tolist(),
            "routed_expert_alert": routed_alert[holdout].tolist(),
            "router_coefficient_sha256": router["coefficient_sha256"],
            "expert_coefficient_sha256": {
                MECHANISMS[index]: experts[index]["coefficient_sha256"] for index in (0, 1)
            },
        })
    if np.any(routed_alert < 0) or np.any(oracle_alert < 0) or np.any(routed_mechanism < 0):
        raise RuntimeError("nested evaluation left an episode without a prediction")
    per_mechanism: dict[str, Any] = {}
    for mechanism_id, mechanism_name in enumerate(MECHANISMS):
        selected = mechanisms == mechanism_id
        per_mechanism[mechanism_name] = {
            "oracle_expert_metrics": common.binary_metrics(alerts[selected], oracle_alert[selected]),
            "routed_expert_metrics": common.binary_metrics(alerts[selected], routed_alert[selected]),
            "episode_count": int(selected.sum()),
        }
    return {
        "router_metrics": common.binary_metrics(mechanisms, routed_mechanism),
        "oracle_expert_metrics": common.binary_metrics(alerts, oracle_alert),
        "routed_expert_metrics": common.binary_metrics(alerts, routed_alert),
        "routed_mechanisms": routed_mechanism.tolist(),
        "oracle_expert_predictions": oracle_alert.tolist(),
        "routed_expert_predictions": routed_alert.tolist(),
        "per_mechanism": per_mechanism,
        "folds": folds,
    }


def evaluate_feature_set(
    features: np.ndarray,
    alerts: np.ndarray,
    mechanisms: np.ndarray,
    episode_ids: Sequence[str],
    source_ids: Sequence[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    unified = common.leave_one_source_group_out(
        features, alerts, episode_ids, source_ids, ridge=ridge, class_balanced=True
    )
    first = nested_source_evaluation(
        features, alerts, mechanisms, episode_ids, source_ids, ridge=ridge
    )
    second = nested_source_evaluation(
        features, alerts, mechanisms, episode_ids, source_ids, ridge=ridge
    )
    return {
        "feature_matrix_sha256": hashlib.sha256(np.asarray(features, dtype="<f8").tobytes(order="C")).hexdigest(),
        "unified_alert_head": unified,
        "mechanism_routed": first,
        "repeat_exact": first == second,
    }


def gate(result: dict[str, Any], *, minimum_balanced_accuracy: float, minimum_class_recall: float) -> dict[str, Any]:
    unified = result["unified_alert_head"]["metrics"]
    routed = result["mechanism_routed"]["routed_expert_metrics"]
    oracle = result["mechanism_routed"]["oracle_expert_metrics"]
    router = result["mechanism_routed"]["router_metrics"]
    experts = result["mechanism_routed"]["per_mechanism"]
    expert_passes = []
    for mechanism in MECHANISMS:
        metrics = experts[mechanism]["oracle_expert_metrics"]
        expert_passes.append(
            metrics["balanced_accuracy"] >= minimum_balanced_accuracy
            and metrics["candidate_no_alert_recall"] >= minimum_class_recall
            and metrics["candidate_alert_recall"] >= minimum_class_recall
        )
    oracle_upper_bound = bool(result["repeat_exact"] and all(expert_passes) and oracle["balanced_accuracy"] > unified["balanced_accuracy"])
    deployable_route = bool(
        oracle_upper_bound
        and router["balanced_accuracy"] >= minimum_balanced_accuracy
        and routed["balanced_accuracy"] >= minimum_balanced_accuracy
        and routed["balanced_accuracy"] > unified["balanced_accuracy"]
        and routed["candidate_no_alert_recall"] >= minimum_class_recall
        and routed["candidate_alert_recall"] >= minimum_class_recall
    )
    return {
        "oracle_upper_bound_passed": oracle_upper_bound,
        "observable_routed_gate_passed": deployable_route,
        "per_mechanism_oracle_passes": dict(zip(MECHANISMS, expert_passes)),
        "thresholds": {
            "balanced_accuracy_gte": minimum_balanced_accuracy,
            "each_alert_class_recall_gte": minimum_class_recall,
            "router_balanced_accuracy_gte": minimum_balanced_accuracy,
            "strictly_improves_same_subset_unified_head": True,
            "repeat_exact": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.detector_weights, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    all_episodes, excluded = common.load_episode_specs(args.package_root.resolve())
    episodes, mechanisms = qualified_rows(all_episodes, args.mechanism_report.resolve())
    alerts = np.asarray([row["label"] for row in episodes], dtype=np.int64)
    episode_ids = [row["episode_id"] for row in episodes]
    source_ids = [row["source_id"] for row in episodes]

    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.cache_dir.resolve())
    from ultralytics import YOLO
    detector = YOLO(str(args.detector_weights.resolve()))
    trajectory_features, detection_summaries = trajectory.extract(
        detector, episodes, image_size=args.image_size, confidence=args.confidence
    )
    teacher = clearance.FrozenTeacher(args.model_dir.resolve())
    profile_features, profile_summaries = profile.extract_profile(
        episodes, teacher, motion_size=args.motion_size, batch_size=args.batch_size
    )
    feature_sets = {
        "trajectory_only": trajectory_features,
        "risk_profile_only": profile_features,
        "trajectory_plus_risk_profile": np.concatenate([trajectory_features, profile_features], axis=1),
    }
    evaluations = {
        name: evaluate_feature_set(values, alerts, mechanisms, episode_ids, source_ids, ridge=args.ridge)
        for name, values in feature_sets.items()
    }
    gates = {
        name: gate(value, minimum_balanced_accuracy=args.minimum_balanced_accuracy, minimum_class_recall=args.minimum_class_recall)
        for name, value in evaluations.items()
    }
    passing = [name for name, value in gates.items() if value["observable_routed_gate_passed"]]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_mechanism_conditioning_diagnosis",
        "qualified_episode_count": len(episodes),
        "excluded_nonqualified_or_abstain_count": len(all_episodes) - len(episodes) + len(excluded),
        "qualified_pair_count": len({row["counterfactual_pair_id"] for row in episodes}),
        "mechanism_counts": {MECHANISMS[index]: int(np.sum(mechanisms == index)) for index in (0, 1)},
        "feature_contract": {
            "trajectory_dimension": int(trajectory_features.shape[1]),
            "risk_profile_dimension": int(profile_features.shape[1]),
            "threshold_or_feature_search": False,
            "router_supervision": "qualified mechanism label only; held-out source excluded",
            "expert_supervision": "candidate alert label only within one qualified mechanism; held-out source excluded",
            "oracle_mechanism_available_at_runtime": False,
        },
        "evaluation_contract": {
            "split": "nested_leave_one_source_group_out",
            "group_key": "source_id",
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "ridge": args.ridge,
            "class_balanced": True,
        },
        "evaluations": evaluations,
        "gates": gates,
        "passing_observable_feature_sets": passing,
        "episode_contract": [
            {
                "episode_id": row["episode_id"],
                "source_id": row["source_id"],
                "counterfactual_pair_id": row["counterfactual_pair_id"],
                "mechanism": MECHANISMS[int(mechanisms[index])],
                "alert_label": int(alerts[index]),
            }
            for index, row in enumerate(episodes)
        ],
        "episode_detection_summaries": detection_summaries,
        "episode_profile_summaries": profile_summaries,
        "lifecycle_head_authorized": bool(passing),
        "evidence_limit": "Retrospective six-pair GPT/VLM-silver mechanism diagnosis. Oracle routing is an upper bound only. No calibration, blind evaluation, human accuracy, Android change or production promotion is authorized.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("../artifacts.local/cache/ultralytics-mechanism-router"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(320,), default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--motion-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-class-recall", type=float, default=0.50)
    args = parser.parse_args()
    if args.ridge <= 0 or not 0 < args.confidence < 1 or args.motion_size < 32 or args.batch_size < 1:
        parser.error("invalid probe settings")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "passing_observable_feature_sets": report["passing_observable_feature_sets"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
