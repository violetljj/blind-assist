#!/usr/bin/env python3
"""Test whether the D6 veto score adds held-out real-domain signal."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


BASELINE_FEATURES = (
    "comparator_mean",
    "comparator_p95",
    "comparator_max",
    "known_mean",
    "known_p95",
    "log1p_eligible_cell_count",
    "near_fraction",
    "body_fraction",
    "direction_2_fraction",
    "distance_mean_normalized",
)
CANDIDATE_FEATURES = BASELINE_FEATURES + (
    "candidate_mean",
    "candidate_p95",
    "candidate_max",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold_assignments(
    rows: list[dict[str, Any]],
    fold_count: int = 5,
) -> dict[str, int]:
    phases_by_session: dict[str, set[str]] = {}
    for row in rows:
        phases_by_session.setdefault(
            str(row["source_session_id"]),
            set(),
        ).add(str(row["phase"]))
    positive = []
    negative = []
    for session, phases in phases_by_session.items():
        if "positive_alertable" in phases:
            positive.append(session)
        elif phases == {"negative_event"}:
            negative.append(session)
        else:
            raise ValueError(
                f"Unexpected session phase composition: "
                f"{session} {sorted(phases)}"
            )

    def ordered(values: list[str]) -> list[str]:
        return sorted(
            values,
            key=lambda value: (
                hashlib.sha256(value.encode("utf-8")).hexdigest(),
                value,
            ),
        )

    assignments = {}
    for sessions in (ordered(positive), ordered(negative)):
        for index, session in enumerate(sessions):
            assignments[session] = index % fold_count
    if len(assignments) != len(phases_by_session):
        raise ValueError("Fold assignment lost a source session")
    return assignments


def ranking_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    return {
        "unit_count": int(target.size),
        "positive_count": int(target.sum()),
        "negative_count": int(target.size - target.sum()),
        "auroc": float(roc_auc_score(target, probability)),
        "average_precision": float(
            average_precision_score(target, probability)
        ),
        "brier": float(brier_score_loss(target, probability)),
    }


def fit_oof(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    assignments: dict[str, int],
    fold_count: int = 5,
) -> dict[str, Any]:
    matrix = np.asarray(
        [[float(row[name]) for name in features] for row in rows],
        dtype=np.float64,
    )
    target = np.asarray(
        [float(row["false_alert_target"]) for row in rows],
        dtype=np.float64,
    )
    folds = np.asarray(
        [assignments[str(row["source_session_id"])] for row in rows],
        dtype=np.int64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Calibration feature matrix is non-finite")
    probability = np.full(target.shape, np.nan, dtype=np.float64)
    fold_rows = []
    for fold in range(fold_count):
        train = folds != fold
        test = folds == fold
        if set(np.unique(target[train])) != {0.0, 1.0}:
            raise ValueError(f"Training fold {fold} lacks a class")
        if set(np.unique(target[test])) != {0.0, 1.0}:
            raise ValueError(f"Test fold {fold} lacks a class")
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_matrix = scaler.fit_transform(matrix[train])
        test_matrix = scaler.transform(matrix[test])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=0,
        )
        model.fit(train_matrix, target[train])
        probability[test] = model.predict_proba(test_matrix)[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "train_source_session_count": len(
                    {
                        str(rows[index]["source_session_id"])
                        for index in np.flatnonzero(train)
                    }
                ),
                "test_source_session_count": len(
                    {
                        str(rows[index]["source_session_id"])
                        for index in np.flatnonzero(test)
                    }
                ),
                "test": ranking_metrics(
                    target[test],
                    probability[test],
                ),
                "standardized_coefficients": {
                    name: float(value)
                    for name, value in zip(
                        features,
                        model.coef_[0],
                    )
                },
            }
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    return {
        "features": list(features),
        "metrics": ranking_metrics(target, probability),
        "probability": probability,
        "folds": folds,
        "fold_rows": fold_rows,
    }


def paired_direction(
    rows: list[dict[str, Any]],
    probability: np.ndarray,
) -> dict[str, Any]:
    by_session: dict[str, dict[str, float]] = {}
    for row, score in zip(rows, probability):
        by_session.setdefault(
            str(row["source_session_id"]),
            {},
        )[str(row["phase"])] = float(score)
    pairs = []
    for session, phases in sorted(by_session.items()):
        if not {
            "positive_alertable",
            "positive_passed",
        }.issubset(phases):
            continue
        delta = (
            phases["positive_passed"]
            - phases["positive_alertable"]
        )
        pairs.append(
            {
                "source_session_id": session,
                "passed_minus_alertable": delta,
            }
        )
    values = np.asarray(
        [row["passed_minus_alertable"] for row in pairs],
        dtype=np.float64,
    )
    return {
        "pair_count": len(pairs),
        "passed_score_higher_count": int((values > 0.0).sum()),
        "passed_score_equal_count": int((values == 0.0).sum()),
        "passed_score_lower_count": int((values < 0.0).sum()),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "minimum_delta": float(np.min(values)),
        "maximum_delta": float(np.max(values)),
        "pairs": pairs,
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_count": int((array > 0.0).sum()),
        "zero_count": int((array == 0.0).sum()),
        "negative_count": int((array < 0.0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite real calibration report")
    report = json.loads(
        args.ranking_report.read_text(encoding="utf-8")
    )
    if (
        report.get("schema")
        != "blindassist_hftf_stage_c_d6_sanpo_real_veto_ranking_v1"
        or int(report.get("model_count", 0)) != 9
    ):
        raise ValueError("Unexpected SANPO real veto ranking report")

    units = []
    auroc_deltas = []
    ap_deltas = []
    paired_direction_deltas = []
    reference_assignments = None
    for source_unit in report["units"]:
        rows = source_unit["phase_units"]
        assignments = stable_fold_assignments(rows)
        if reference_assignments is None:
            reference_assignments = assignments
        elif assignments != reference_assignments:
            raise ValueError("Model units disagree on source folds")
        baseline = fit_oof(rows, BASELINE_FEATURES, assignments)
        candidate = fit_oof(rows, CANDIDATE_FEATURES, assignments)
        baseline_pairs = paired_direction(
            rows,
            baseline["probability"],
        )
        candidate_pairs = paired_direction(
            rows,
            candidate["probability"],
        )
        auroc_delta = (
            candidate["metrics"]["auroc"]
            - baseline["metrics"]["auroc"]
        )
        ap_delta = (
            candidate["metrics"]["average_precision"]
            - baseline["metrics"]["average_precision"]
        )
        paired_delta = (
            candidate_pairs["mean_delta"]
            - baseline_pairs["mean_delta"]
        )
        auroc_deltas.append(float(auroc_delta))
        ap_deltas.append(float(ap_delta))
        paired_direction_deltas.append(float(paired_delta))
        units.append(
            {
                "seed": int(source_unit["seed"]),
                "fold": int(source_unit["fold"]),
                "baseline_only": {
                    key: value
                    for key, value in baseline.items()
                    if key not in {"probability", "folds"}
                },
                "candidate_aware": {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"probability", "folds"}
                },
                "baseline_only_positive_pairs": baseline_pairs,
                "candidate_aware_positive_pairs": candidate_pairs,
                "candidate_auroc_delta": float(auroc_delta),
                "candidate_average_precision_delta": float(
                    ap_delta
                ),
                "candidate_paired_mean_delta_increment": float(
                    paired_delta
                ),
            }
        )
        print(
            json.dumps(
                {
                    "seed": source_unit["seed"],
                    "fold": source_unit["fold"],
                    "candidate_auroc_delta": auroc_delta,
                    "candidate_ap_delta": ap_delta,
                    "candidate_pair_increment": paired_delta,
                }
            ),
            flush=True,
        )

    output = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_real_veto_"
            "calibration_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SANPO_REAL_VETO_CALIBRATION_ABLATION_COMPLETE",
        "policy": {
            "consumed_development": True,
            "source_session_held_out": True,
            "threshold_search": False,
            "feature_search": False,
            "model_search": False,
            "system_output_connected": False,
            "promotion_evidence": False,
        },
        "design": {
            "fold_count": 5,
            "split": (
                "hash-ordered round-robin within positive/negative "
                "source-session strata"
            ),
            "model": (
                "StandardScaler + L2 LogisticRegression("
                "C=1, liblinear, class_weight=balanced)"
            ),
            "baseline_features": list(BASELINE_FEATURES),
            "candidate_added_features": [
                name
                for name in CANDIDATE_FEATURES
                if name not in BASELINE_FEATURES
            ],
            "primary": "OOF event-phase AUROC C - B",
            "secondary": (
                "OOF average precision and positive-session "
                "passed-minus-alertable direction"
            ),
        },
        "ranking_report_path": str(
            args.ranking_report.resolve()
        ),
        "ranking_report_sha256": sha256(args.ranking_report),
        "source_fold_assignments": reference_assignments,
        "model_count": len(units),
        "summary": {
            "candidate_auroc_delta": summarize(auroc_deltas),
            "candidate_average_precision_delta": summarize(
                ap_deltas
            ),
            "candidate_paired_mean_delta_increment": summarize(
                paired_direction_deltas
            ),
        },
        "units": units,
        "evidence_limit": (
            "Consumed human-reviewed SANPO Development calibration "
            "ablation only. It tests incremental real-domain information "
            "in the current candidate score; it does not authorize a "
            "threshold, event utility, App behavior, promotion, or "
            "safety claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    Path(str(args.output) + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "summary": output["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
