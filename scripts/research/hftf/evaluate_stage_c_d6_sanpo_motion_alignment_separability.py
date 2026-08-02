#!/usr/bin/env python3
"""Compare raw and globally aligned RGB-pair phase separability on SANPO."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluate_stage_c_d6_sanpo_real_veto_ranking import (
    DEFAULT_MANIFEST,
    build_event_windows,
)
from run_stage_c_d6_sanpo_real_phase_early_pair_canary import (
    HELDOUT_FOLD,
    phase_group_weights,
    scored,
    stable_fold_assignments,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    fit_logistic,
    weighted_standardize,
)
from train_stage_c_d5_tartanground_development_student import (
    binary_metrics,
    sha256,
)


MIN_ALIGNMENT_COVERAGE = 0.90
MIN_HELDOUT_PHASE_COVERAGE = 0.80
MIN_AFFINE_INLIER_FRACTION = 0.40
L2_STRENGTH = 0.1
GRID_ROWS = 3
GRID_COLUMNS = 6
FEATURE_FAMILIES = ("mean", "p90", "fraction_gt_0_10")


class ImageCache:
    def __init__(self) -> None:
        self.values: dict[str, np.ndarray] = {}

    def grayscale(self, path: str) -> np.ndarray:
        value = self.values.get(path)
        if value is None:
            source = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if source is None:
                raise ValueError(f"Unreadable RGB frame: {path}")
            value = cv2.resize(
                source,
                (224, 128),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            value /= 255.0
            self.values[path] = value
        return value


def align_previous(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    previous_u8 = np.round(previous * 255.0).astype(np.uint8)
    current_u8 = np.round(current * 255.0).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(
        previous_u8,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return None, {"reason": "insufficient_corners"}
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous_u8,
        current_u8,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if moved is None or status is None or error is None:
        return None, {"reason": "lk_failed"}
    keep = (
        status.reshape(-1).astype(bool)
        & np.isfinite(moved.reshape(-1, 2)).all(axis=1)
        & (error.reshape(-1) < 30.0)
    )
    source = points.reshape(-1, 2)[keep]
    target = moved.reshape(-1, 2)[keep]
    if len(source) < 12:
        return None, {
            "reason": "insufficient_tracks",
            "track_count": int(len(source)),
        }
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None:
        return None, {
            "reason": "affine_failed",
            "track_count": int(len(source)),
        }
    inlier_count = int(inliers.sum())
    inlier_fraction = inlier_count / len(source)
    if (
        inlier_count < 8
        or inlier_fraction < MIN_AFFINE_INLIER_FRACTION
    ):
        return None, {
            "reason": "weak_affine_consensus",
            "track_count": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_fraction": inlier_fraction,
        }
    height, width = current.shape
    aligned = cv2.warpAffine(
        previous,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid = cv2.warpAffine(
        np.ones_like(previous, dtype=np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    if float(valid.mean()) < 0.75:
        return None, {
            "reason": "insufficient_warp_overlap",
            "track_count": int(len(source)),
            "inlier_count": inlier_count,
            "inlier_fraction": inlier_fraction,
            "valid_fraction": float(valid.mean()),
        }
    return np.stack((aligned, valid)), {
        "reason": "ok",
        "track_count": int(len(source)),
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
        "valid_fraction": float(valid.mean()),
        "translation_x": float(matrix[0, 2]),
        "translation_y": float(matrix[1, 2]),
        "rotation_scale_a": float(matrix[0, 0]),
        "rotation_scale_b": float(matrix[1, 0]),
    }


def residual_features(
    residual: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    rows = []
    height, width = residual.shape
    for row_index in range(GRID_ROWS):
        y0 = row_index * height // GRID_ROWS
        y1 = (row_index + 1) * height // GRID_ROWS
        for column_index in range(GRID_COLUMNS):
            x0 = column_index * width // GRID_COLUMNS
            x1 = (column_index + 1) * width // GRID_COLUMNS
            values = residual[y0:y1, x0:x1][
                valid[y0:y1, x0:x1] > 0.5
            ]
            if values.size == 0:
                raise ValueError("Residual grid cell has no valid pixels")
            rows.extend(
                (
                    float(np.mean(values)),
                    float(np.quantile(values, 0.90)),
                    float(np.mean(values > 0.10)),
                )
            )
    return np.asarray(rows, dtype=np.float64)


def extract_features(
    windows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache = ImageCache()
    raw_rows = []
    aligned_rows = []
    valid_rows = []
    diagnostics = []
    for index, window in enumerate(windows):
        previous = cache.grayscale(window["history_rgb_paths"][-2])
        current = cache.grayscale(window["history_rgb_paths"][-1])
        raw = np.abs(current - previous)
        raw_rows.append(
            residual_features(raw, np.ones_like(raw, dtype=np.float32))
        )
        aligned, diagnostic = align_previous(previous, current)
        diagnostic = {"window_index": index, **diagnostic}
        diagnostics.append(diagnostic)
        if aligned is None:
            valid_rows.append(False)
            aligned_rows.append(np.zeros_like(raw_rows[-1]))
            continue
        aligned_image, valid = aligned
        aligned_rows.append(
            residual_features(
                np.abs(current - aligned_image),
                valid,
            )
        )
        valid_rows.append(True)
    return (
        np.stack(raw_rows),
        np.stack(aligned_rows),
        np.asarray(valid_rows, dtype=bool),
        diagnostics,
    )


def fit_projection(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    mean, scale = weighted_standardize(matrix, weights)
    standardized = (matrix - mean) / scale
    coefficient, intercept, loss = fit_logistic(
        standardized,
        labels,
        weights,
        l2_strength=L2_STRENGTH,
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "weighted_regularized_train_loss": loss,
    }


def predict(matrix: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    standardized = (matrix - model["mean"]) / model["scale"]
    logits = (
        standardized @ model["coefficient"] + model["intercept"]
    )
    return 1.0 / (1.0 + np.exp(-logits))


def event_phase_rows(
    windows: list[dict[str, Any]],
    raw_probability: np.ndarray,
    aligned_probability: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for window, raw, aligned in zip(
        windows,
        raw_probability,
        aligned_probability,
        strict=True,
    ):
        key = (str(window["source_session_id"]), str(window["phase"]))
        group = groups.setdefault(
            key,
            {
                "target": float(window["false_alert_target"]),
                "raw": [],
                "aligned": [],
            },
        )
        group["raw"].append(float(raw))
        group["aligned"].append(float(aligned))
    return [
        {
            "source_session_id": key[0],
            "phase": key[1],
            "false_alert_target": value["target"],
            "window_count": len(value["raw"]),
            "raw_p95": float(np.quantile(value["raw"], 0.95)),
            "aligned_p95": float(
                np.quantile(value["aligned"], 0.95)
            ),
        }
        for key, value in sorted(groups.items())
    ]


def compare_event_phases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["phase"] in {"negative_event", "positive_alertable"}
    ]
    target = np.asarray(
        [row["false_alert_target"] for row in selected],
        dtype=np.float32,
    )
    mask = np.ones_like(target, dtype=bool)
    raw = binary_metrics(
        np.asarray([row["raw_p95"] for row in selected]),
        target,
        mask,
    )
    aligned = binary_metrics(
        np.asarray([row["aligned_p95"] for row in selected]),
        target,
        mask,
    )
    return {
        "unit_count": len(selected),
        "raw_pair": raw,
        "motion_aligned": aligned,
        "auroc_delta": (
            float(aligned["auroc"] - raw["auroc"])
            if aligned["auroc"] is not None and raw["auroc"] is not None
            else None
        ),
        "average_precision_delta": (
            float(
                aligned["average_precision"]
                - raw["average_precision"]
            )
            if aligned["average_precision"] is not None
            and raw["average_precision"] is not None
            else None
        ),
    }


def positive_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_session.setdefault(row["source_session_id"], {})[
            row["phase"]
        ] = row
    pairs = []
    for session, phases in sorted(by_session.items()):
        if not {
            "positive_alertable",
            "positive_passed",
        }.issubset(phases):
            continue
        pairs.append(
            {
                "source_session_id": session,
                "raw_delta": (
                    phases["positive_passed"]["raw_p95"]
                    - phases["positive_alertable"]["raw_p95"]
                ),
                "aligned_delta": (
                    phases["positive_passed"]["aligned_p95"]
                    - phases["positive_alertable"]["aligned_p95"]
                ),
            }
        )
    return {
        "pair_count": len(pairs),
        "raw_positive_count": sum(
            row["raw_delta"] > 0.0 for row in pairs
        ),
        "aligned_positive_count": sum(
            row["aligned_delta"] > 0.0 for row in pairs
        ),
        "raw_mean_delta": float(
            np.mean([row["raw_delta"] for row in pairs])
        ),
        "aligned_mean_delta": float(
            np.mean([row["aligned_delta"] for row in pairs])
        ),
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--heldout-fold",
        type=int,
        choices=range(5),
        default=HELDOUT_FOLD,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite motion-alignment audit")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in manifest["events"])
        != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    windows = scored(build_event_windows(args.manifest, manifest))
    assignments = stable_fold_assignments(windows)
    raw, aligned, valid, diagnostics = extract_features(windows)
    coverage = float(valid.mean())
    failure_counts = Counter(
        row["reason"] for row in diagnostics if row["reason"] != "ok"
    )
    train = np.asarray(
        [
            assignments[str(row["source_session_id"])]
            != args.heldout_fold
            for row in windows
        ]
    )
    heldout = ~train
    heldout_phase_coverage = {}
    for session, phase in sorted(
        {
            (str(row["source_session_id"]), str(row["phase"]))
            for row, selected in zip(windows, heldout, strict=True)
            if selected
        }
    ):
        indices = np.asarray(
            [
                selected
                and str(row["source_session_id"]) == session
                and str(row["phase"]) == phase
                for row, selected in zip(
                    windows,
                    heldout,
                    strict=True,
                )
            ]
        )
        heldout_phase_coverage[f"{session}/{phase}"] = float(
            valid[indices].mean()
        )
    evaluable = (
        coverage >= MIN_ALIGNMENT_COVERAGE
        and float(valid[heldout].mean()) >= MIN_ALIGNMENT_COVERAGE
        and all(
            value >= MIN_HELDOUT_PHASE_COVERAGE
            for value in heldout_phase_coverage.values()
        )
    )
    if not evaluable:
        terminal = "D6_MOTION_ALIGNMENT_SEPARABILITY_NOT_EVALUABLE"
        report = {
            "schema": (
                "blindassist_hftf_stage_c_d6_sanpo_motion_alignment_"
                "separability_v0"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": terminal,
            "alignment": {
                "coverage": coverage,
                "minimum_coverage": MIN_ALIGNMENT_COVERAGE,
                "heldout_coverage": float(valid[heldout].mean()),
                "minimum_heldout_phase_coverage": (
                    MIN_HELDOUT_PHASE_COVERAGE
                ),
                "heldout_phase_coverage": heldout_phase_coverage,
                "failure_counts": dict(sorted(failure_counts.items())),
                "diagnostics": diagnostics,
            },
            "evidence_limit": (
                "Alignment coverage failed before any supervised "
                "projection; this is not an algorithmic negative."
            ),
        }
    else:
        train_indices = np.flatnonzero(train & valid)
        heldout_indices = np.flatnonzero(heldout & valid)
        train_windows = [windows[index] for index in train_indices]
        heldout_windows = [windows[index] for index in heldout_indices]
        labels = np.asarray(
            [
                float(windows[index]["false_alert_target"])
                for index in train_indices
            ],
            dtype=np.float64,
        )
        weights = phase_group_weights(train_windows).astype(np.float64)
        raw_model = fit_projection(raw[train_indices], labels, weights)
        aligned_model = fit_projection(
            aligned[train_indices],
            labels,
            weights,
        )
        phase_rows = event_phase_rows(
            heldout_windows,
            predict(raw[heldout_indices], raw_model),
            predict(aligned[heldout_indices], aligned_model),
        )
        comparison = compare_event_phases(phase_rows)
        pairs = positive_pairs(phase_rows)
        supported = (
            comparison["auroc_delta"] is not None
            and comparison["average_precision_delta"] is not None
            and comparison["auroc_delta"] > 0.0
            and comparison["average_precision_delta"] > 0.0
        )
        terminal = (
            "D6_MOTION_ALIGNED_PAIR_SEPARABILITY_SUPPORTED_TO_TRAIN"
            if supported
            else "D6_MOTION_ALIGNED_PAIR_SEPARABILITY_NOT_SUPPORTED"
        )
        report = {
            "schema": (
                "blindassist_hftf_stage_c_d6_sanpo_motion_alignment_"
                "separability_v0"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "MOTION_ALIGNMENT_SEPARABILITY_AUDIT_COMPLETE",
            "decision": {
                "terminal": terminal,
                "supported_to_train": supported,
                "criterion": (
                    "held-out event-phase p95 AUROC delta > 0 AND "
                    "average-precision delta > 0"
                ),
            },
            "policy": {
                "data_role": "consumed_development",
                "source_session_heldout": True,
                "heldout_used_for_fit_or_standardization": False,
                "threshold_search": False,
                "feature_search": False,
                "training_authorized_by_positive_only": True,
                "app_or_safety_claim": False,
            },
            "design": {
                "raw_pair": "absolute grayscale t-1 to t residual",
                "motion_aligned": (
                    "absolute residual after sparse-LK RANSAC "
                    "partial-affine t-1 to t registration, minimum "
                    f"inlier fraction={MIN_AFFINE_INLIER_FRACTION}"
                ),
                "shared_features": (
                    "3x6 grid mean, p90, and fraction > 0.10"
                ),
                "projection": (
                    "weighted standardization + L2 logistic "
                    f"regression, strength={L2_STRENGTH}"
                ),
                "heldout_fold": args.heldout_fold,
            },
            "input": {
                "manifest_path": str(args.manifest.resolve()),
                "manifest_sha256": sha256(args.manifest),
                "window_count": len(windows),
            },
            "split": {
                "source_fold_assignments": assignments,
                "train_window_count": len(train_indices),
                "heldout_window_count": len(heldout_indices),
            },
            "alignment": {
                "coverage": coverage,
                "minimum_coverage": MIN_ALIGNMENT_COVERAGE,
                "heldout_coverage": float(valid[heldout].mean()),
                "minimum_heldout_phase_coverage": (
                    MIN_HELDOUT_PHASE_COVERAGE
                ),
                "heldout_phase_coverage": heldout_phase_coverage,
                "failure_counts": dict(sorted(failure_counts.items())),
                "diagnostics": diagnostics,
            },
            "projection": {
                "feature_count": int(raw.shape[1]),
                "raw_train_loss": raw_model[
                    "weighted_regularized_train_loss"
                ],
                "aligned_train_loss": aligned_model[
                    "weighted_regularized_train_loss"
                ],
            },
            "heldout_event_phase_p95": comparison,
            "heldout_positive_pairs": pairs,
            "phase_units": phase_rows,
            "evidence_limit": (
                "Single heldout-fold representation audit on consumed "
                "SANPO Development. Support only permits one fixed "
                "motion-aligned field-residual canary."
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
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
    print(
        json.dumps(
            {
                "ok": True,
                "terminal": terminal,
                "alignment_coverage": coverage,
                "heldout": report.get("heldout_event_phase_p95"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
