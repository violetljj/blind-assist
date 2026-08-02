#!/usr/bin/env python3
"""Evaluate pretrained RAFT motion representations on SANPO phases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow import raft_small

from evaluate_stage_c_d6_sanpo_motion_alignment_separability import (
    fit_projection,
    predict,
)
from evaluate_stage_c_d6_sanpo_real_veto_ranking import (
    DEFAULT_MANIFEST,
    build_event_windows,
)
from run_stage_c_d6_sanpo_real_phase_early_pair_canary import (
    phase_group_weights,
    scored,
    stable_fold_assignments,
)
from train_stage_c_d5_tartanground_development_student import (
    binary_metrics,
    sha256,
)


DEFAULT_RAFT_WEIGHTS = Path(
    "artifacts.local/models/hftf/torch/optical-flow/"
    "raft_small_C_T_V2-01064c6d.pth"
)
EXPECTED_RAFT_SHA256 = (
    "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
)
FOLD_COUNT = 5
MIN_ALIGNMENT_COVERAGE = 0.99
MIN_PHASE_COVERAGE = 0.90
GRID_ROWS = 3
GRID_COLUMNS = 6


class RaftPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, windows: list[dict[str, Any]]) -> None:
        self.windows = windows
        self.cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def image(self, path: str) -> torch.Tensor:
        value = self.cache.get(path)
        if value is None:
            source = cv2.imread(path, cv2.IMREAD_COLOR)
            if source is None:
                raise ValueError(f"Unreadable RGB frame: {path}")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = cv2.resize(
                source,
                (224, 128),
                interpolation=cv2.INTER_AREA,
            )
            value = torch.from_numpy(
                np.ascontiguousarray(source.transpose(2, 0, 1))
            )
            self.cache[path] = value
        return value.float().div(127.5).sub(1.0)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        paths = self.windows[index]["history_rgb_paths"]
        return self.image(paths[-2]), self.image(paths[-1]), index


def grid_features(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = []
    height, width = values.shape
    for row in range(GRID_ROWS):
        y0 = row * height // GRID_ROWS
        y1 = (row + 1) * height // GRID_ROWS
        for column in range(GRID_COLUMNS):
            x0 = column * width // GRID_COLUMNS
            x1 = (column + 1) * width // GRID_COLUMNS
            cell = values[y0:y1, x0:x1]
            output.extend(
                (
                    float(np.mean(cell)),
                    float(np.quantile(cell, 0.90)),
                    float(np.mean(cell > threshold)),
                )
            )
    return np.asarray(output, dtype=np.float64)


def remove_global_affine(
    flow: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    _, height, width = flow.shape
    y, x = np.mgrid[4:height:8, 4:width:8]
    source = np.stack((x.reshape(-1), y.reshape(-1)), axis=1).astype(
        np.float32
    )
    sampled = flow[:, y, x].transpose(1, 2, 0).reshape(-1, 2)
    target = source + sampled
    finite = np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite].astype(np.float32)
    if len(source) < 32:
        return None, {"reason": "insufficient_finite_flow"}

    def median_translation(reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        translation = np.median(
            target - source,
            axis=0,
        )
        residual = flow.astype(np.float64) - translation[
            :, None, None
        ]
        return residual, {
            "reason": "median_translation_fallback",
            "fallback_from": reason,
            "translation_x": float(translation[0]),
            "translation_y": float(translation[1]),
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
        return median_translation("affine_failed")
    inlier_count = int(inliers.sum())
    inlier_fraction = inlier_count / len(source)
    if inlier_count < 24 or inlier_fraction < 0.40:
        residual, diagnostic = median_translation(
            "weak_affine_consensus"
        )
        diagnostic["inlier_count"] = inlier_count
        diagnostic["inlier_fraction"] = inlier_fraction
        return residual, diagnostic
    full_y, full_x = np.mgrid[0:height, 0:width]
    predicted_x = (
        matrix[0, 0] * full_x
        + matrix[0, 1] * full_y
        + matrix[0, 2]
    )
    predicted_y = (
        matrix[1, 0] * full_x
        + matrix[1, 1] * full_y
        + matrix[1, 2]
    )
    global_flow = np.stack(
        (predicted_x - full_x, predicted_y - full_y)
    )
    residual = flow.astype(np.float64) - global_flow
    return residual, {
        "reason": "ok",
        "inlier_count": inlier_count,
        "inlier_fraction": inlier_fraction,
    }


def extract_features(
    windows: list[dict[str, Any]],
    weights_path: Path,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RAFT inference")
    model = raft_small(weights=None, progress=False)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = RaftPairDataset(windows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    rows = {
        "raw_pixel_pair": [None] * len(dataset),
        "raft_flow": [None] * len(dataset),
        "raft_residual_flow": [None] * len(dataset),
    }
    valid = np.zeros(len(dataset), dtype=bool)
    diagnostics: list[dict[str, Any] | None] = [None] * len(dataset)
    with torch.no_grad():
        for previous, current, indices in loader:
            previous = previous.to(device, non_blocking=True)
            current = current.to(device, non_blocking=True)
            flow_batch = model(previous, current)[-1].cpu().numpy()
            raw_batch = (
                (current - previous).abs().mean(dim=1).cpu().numpy()
                / 2.0
            )
            for local_index, window_index in enumerate(indices.tolist()):
                flow = flow_batch[local_index]
                residual, diagnostic = remove_global_affine(flow)
                diagnostics[window_index] = {
                    "window_index": window_index,
                    **diagnostic,
                }
                rows["raw_pixel_pair"][window_index] = grid_features(
                    raw_batch[local_index],
                    0.10,
                )
                normalized_flow = np.sqrt(
                    np.square(flow[0] / flow.shape[2])
                    + np.square(flow[1] / flow.shape[1])
                )
                rows["raft_flow"][window_index] = grid_features(
                    normalized_flow,
                    0.01,
                )
                if residual is None:
                    rows["raft_residual_flow"][window_index] = np.zeros(
                        54,
                        dtype=np.float64,
                    )
                    continue
                normalized_residual = np.sqrt(
                    np.square(residual[0] / flow.shape[2])
                    + np.square(residual[1] / flow.shape[1])
                )
                rows["raft_residual_flow"][window_index] = grid_features(
                    normalized_residual,
                    0.01,
                )
                valid[window_index] = True
    if any(row is None for values in rows.values() for row in values):
        raise RuntimeError("Feature extraction coverage is incomplete")
    if any(row is None for row in diagnostics):
        raise RuntimeError("Diagnostic coverage is incomplete")
    return (
        {
            name: np.stack(values)
            for name, values in rows.items()
        },
        valid,
        [row for row in diagnostics if row is not None],
    )


def phase_rows(
    windows: list[dict[str, Any]],
    probability: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for index, window in enumerate(windows):
        key = (str(window["source_session_id"]), str(window["phase"]))
        group = groups.setdefault(
            key,
            {
                "target": float(window["false_alert_target"]),
                "scores": {name: [] for name in probability},
            },
        )
        for name, values in probability.items():
            group["scores"][name].append(float(values[index]))
    return [
        {
            "source_session_id": key[0],
            "phase": key[1],
            "false_alert_target": value["target"],
            "window_count": len(next(iter(value["scores"].values()))),
            **{
                f"{name}_p95": float(np.quantile(scores, 0.95))
                for name, scores in value["scores"].items()
            },
        }
        for key, value in sorted(groups.items())
    ]


def metrics(
    rows: list[dict[str, Any]],
    names: list[str],
) -> dict[str, Any]:
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
    output = {
        name: binary_metrics(
            np.asarray([row[f"{name}_p95"] for row in selected]),
            target,
            mask,
        )
        for name in names
    }
    reference = output["raw_pixel_pair"]
    for name in names:
        if name == "raw_pixel_pair":
            continue
        output[f"{name}_vs_raw"] = {
            "auroc_delta": float(
                output[name]["auroc"] - reference["auroc"]
            ),
            "average_precision_delta": float(
                output[name]["average_precision"]
                - reference["average_precision"]
            ),
        }
    return {"unit_count": len(selected), **output}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=DEFAULT_RAFT_WEIGHTS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite RAFT representation report")
    if sha256(args.raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("Unexpected RAFT-small weights SHA-256")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    windows = scored(build_event_windows(args.manifest, manifest))
    assignments = stable_fold_assignments(windows)
    features, valid, diagnostics = extract_features(
        windows,
        args.raft_weights,
        args.batch_size,
    )
    coverage = float(valid.mean())
    failure_counts = Counter(
        row["reason"] for row in diagnostics if row["reason"] != "ok"
    )
    names = list(features)
    folds = []
    for fold in range(FOLD_COUNT):
        heldout = np.asarray(
            [
                assignments[str(row["source_session_id"])] == fold
                for row in windows
            ]
        )
        train = ~heldout
        phase_coverage = {}
        for session, phase in sorted(
            {
                (str(row["source_session_id"]), str(row["phase"]))
                for row, selected in zip(windows, heldout, strict=True)
                if selected
            }
        ):
            group = np.asarray(
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
            phase_coverage[f"{session}/{phase}"] = float(
                valid[group].mean()
            )
        evaluable = (
            coverage >= MIN_ALIGNMENT_COVERAGE
            and float(valid[heldout].mean()) >= MIN_ALIGNMENT_COVERAGE
            and all(
                value >= MIN_PHASE_COVERAGE
                for value in phase_coverage.values()
            )
        )
        if not evaluable:
            folds.append(
                {
                    "fold": fold,
                    "status": "NOT_EVALUABLE_ALIGNMENT_COVERAGE",
                    "heldout_coverage": float(valid[heldout].mean()),
                    "phase_coverage": phase_coverage,
                }
            )
            continue
        train_indices = np.flatnonzero(train & valid)
        heldout_indices = np.flatnonzero(heldout & valid)
        train_windows = [windows[index] for index in train_indices]
        heldout_windows = [windows[index] for index in heldout_indices]
        labels = np.asarray(
            [
                float(windows[index]["false_alert_target"])
                for index in train_indices
            ]
        )
        weights = phase_group_weights(train_windows).astype(np.float64)
        models = {
            name: fit_projection(
                matrix[train_indices],
                labels,
                weights,
            )
            for name, matrix in features.items()
        }
        probabilities = {
            name: predict(features[name][heldout_indices], model)
            for name, model in models.items()
        }
        units = phase_rows(heldout_windows, probabilities)
        fold_metrics = metrics(units, names)
        residual_delta = fold_metrics[
            "raft_residual_flow_vs_raw"
        ]
        supported = (
            residual_delta["auroc_delta"] > 0.0
            and residual_delta["average_precision_delta"] > 0.0
        )
        folds.append(
            {
                "fold": fold,
                "status": (
                    "SUPPORTED"
                    if supported
                    else "NOT_SUPPORTED"
                ),
                "heldout_coverage": float(valid[heldout].mean()),
                "phase_coverage": phase_coverage,
                "metrics": fold_metrics,
                "phase_units": units,
            }
        )
    evaluable_folds = [row for row in folds if "metrics" in row]
    supported_count = sum(
        row["status"] == "SUPPORTED" for row in evaluable_folds
    )
    stable = (
        len(evaluable_folds) >= 4
        and supported_count >= 4
        and all(row["status"] == "SUPPORTED" for row in evaluable_folds)
    )
    if len(evaluable_folds) < 4:
        terminal = "D6_RAFT_RESIDUAL_FLOW_SEPARABILITY_NOT_EVALUABLE"
    elif stable:
        terminal = (
            "D6_RAFT_RESIDUAL_FLOW_SEPARABILITY_"
            "SUPPORTED_TO_FIELD_CANARY"
        )
    else:
        terminal = "D6_RAFT_RESIDUAL_FLOW_SEPARABILITY_NOT_STABLE"
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_raft_motion_"
            "representation_v0"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RAFT_MOTION_REPRESENTATION_EVALUATION_COMPLETE",
        "decision": {
            "terminal": terminal,
            "supported_to_field_canary": stable,
            "criterion": (
                "at least four evaluable folds, all evaluable folds "
                "with residual-flow vs raw-pixel AUROC and AP > 0"
            ),
            "evaluable_fold_count": len(evaluable_folds),
            "supported_fold_count": supported_count,
        },
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "same_fold_and_projection_family": True,
            "feature_or_threshold_search": False,
            "heldout_used_for_fit_or_standardization": False,
            "app_or_safety_claim": False,
        },
        "design": {
            "representations": {
                "raw_pixel_pair": (
                    "mean absolute RGB t-1 to t residual"
                ),
                "raft_flow": (
                    "pretrained RAFT-small dense flow magnitude"
                ),
                "raft_residual_flow": (
                    "RAFT flow minus RANSAC partial-affine global flow"
                ),
            },
            "shared_summary": (
                "3x6 grid mean, p90, and fraction above fixed "
                "representation-scale threshold"
            ),
            "projection": (
                "train-only weighted standardization + L2 logistic"
            ),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "raft_weights_path": str(args.raft_weights.resolve()),
            "raft_weights_sha256": sha256(args.raft_weights),
            "window_count": len(windows),
        },
        "alignment": {
            "coverage": coverage,
            "minimum_coverage": MIN_ALIGNMENT_COVERAGE,
            "minimum_phase_coverage": MIN_PHASE_COVERAGE,
            "failure_counts": dict(sorted(failure_counts.items())),
            "diagnostics": diagnostics,
        },
        "folds": folds,
        "evidence_limit": (
            "Five-fold representation audit on consumed SANPO "
            "Development. A positive terminal only permits one fixed "
            "field-residual canary."
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
                "decision": report["decision"],
                "alignment_coverage": coverage,
                "folds": [
                    {
                        "fold": row["fold"],
                        "status": row["status"],
                        "residual_vs_raw": row.get(
                            "metrics",
                            {},
                        ).get("raft_residual_flow_vs_raw"),
                    }
                    for row in folds
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
