#!/usr/bin/env python3
"""Build a deterministic per-frame metric-scale uncertainty residual bank."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability import extract_features  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)
from train_ag_r2_metric_scale_uncertainty import (  # noqa: E402
    DEFAULT_CORPUS_RESULT,
    DEFAULT_METRIC_STUDENT_RESULT,
    EXPECTED_CORPUS_RESULT_SHA256,
    EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
    EXPECTED_METRIC_STUDENT_RESULT_SHA256,
    SIGMA_CAP,
    SIGMA_FLOOR,
    build_examples,
    load_metric_student,
    pooled_uncertainty_feature,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-r2-metric-scale-residual-bank-r2"
)
PCA_COMPONENTS = 32
K_CANDIDATES = (3, 5, 9, 15)
DISTANCE_ALPHA_CANDIDATES = (0.0, 0.5, 1.0, 2.0, 4.0)
ONE_SIGMA_TARGET_COVERAGE = 0.6826894921370859
ONE_SIGMA_COVERAGE_TOLERANCE = 0.04


def fit_embedding(features: np.ndarray) -> dict[str, np.ndarray]:
    require(features.ndim == 2 and features.shape[1] == 390, "bank feature shape drift")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std >= 1.0e-6, std, 1.0)
    standardized = np.clip((features - mean) / std, -8.0, 8.0)
    _, _, components = np.linalg.svd(standardized, full_matrices=False)
    components = components[:PCA_COMPONENTS]
    projected = standardized @ components.T
    component_std = projected.std(axis=0)
    component_std = np.where(component_std >= 1.0e-6, component_std, 1.0)
    embedded = projected / component_std
    require(np.isfinite(embedded).all(), "bank embedding non-finite")
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "pca_components": components.astype(np.float32),
        "pca_component_std": component_std.astype(np.float32),
        "bank_embedding": embedded.astype(np.float32),
    }


def transform_embedding(feature: np.ndarray, bank: dict[str, np.ndarray]) -> np.ndarray:
    standardized = np.clip(
        (np.asarray(feature, dtype=np.float64) - bank["feature_mean"])
        / bank["feature_std"],
        -8.0,
        8.0,
    )
    projected = standardized @ bank["pca_components"].T
    return np.asarray(projected / bank["pca_component_std"], dtype=np.float64)


def raw_sigma_from_bank(
    query_embedding: np.ndarray,
    bank_embedding: np.ndarray,
    residuals: np.ndarray,
    *,
    k: int,
    distance_alpha: float,
    distance_reference: float,
    global_rms: float,
) -> tuple[float, dict[str, float]]:
    require(len(bank_embedding) >= k, "insufficient residual bank neighbors")
    distances = np.linalg.norm(
        bank_embedding.astype(np.float64) - query_embedding[None], axis=1
    ) / math.sqrt(float(query_embedding.size))
    order = np.argsort(distances, kind="stable")[:k]
    selected_distance = distances[order]
    weights = 1.0 / np.maximum(selected_distance, 1.0e-3)
    local_rms = float(
        np.sqrt(
            np.sum(weights * np.square(residuals[order])) / np.sum(weights)
        )
    )
    nearest = float(selected_distance[0])
    distance_ratio = nearest / max(float(distance_reference), 1.0e-6)
    inflation = float(distance_alpha) * max(0.0, distance_ratio - 1.0) * global_rms
    return local_rms + inflation, {
        "nearest_distance": nearest,
        "distance_ratio": distance_ratio,
        "neighbor_residual_rms": local_rms,
        "distance_inflation": inflation,
    }


def leave_parent_out_rows(
    examples: list[dict[str, Any]],
    embedded: np.ndarray,
    residuals: np.ndarray,
    k: int,
    distance_alpha: float,
    multiplier: float,
    distance_reference: float,
    global_rms: float,
) -> list[dict[str, Any]]:
    parents = np.asarray([str(row["parent_id"]) for row in examples])
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        reference = parents != str(example["parent_id"])
        raw_sigma, diagnostics = raw_sigma_from_bank(
            embedded[index],
            embedded[reference],
            residuals[reference],
            k=k,
            distance_alpha=distance_alpha,
            distance_reference=distance_reference,
            global_rms=global_rms,
        )
        sigma = float(np.clip(raw_sigma * multiplier, SIGMA_FLOOR, SIGMA_CAP))
        residual = float(residuals[index])
        rows.append(
            {
                "sample_id": example["sample_id"],
                "parent_id": example["parent_id"],
                "role": example["role"],
                "signed_scale_residual": residual,
                "absolute_scale_residual": abs(residual),
                "predicted_sigma": sigma,
                "covered_at_one_sigma": abs(residual) <= sigma,
                "gaussian_nll": 0.5 * (residual / sigma) ** 2 + math.log(sigma),
                **diagnostics,
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_id"])].append(row)
    parent_metrics = {
        parent: {
            "frame_count": len(parent_rows),
            "gaussian_nll": float(
                np.mean([row["gaussian_nll"] for row in parent_rows])
            ),
            "one_sigma_coverage": float(
                np.mean([row["covered_at_one_sigma"] for row in parent_rows])
            ),
            "mean_predicted_sigma": float(
                np.mean([row["predicted_sigma"] for row in parent_rows])
            ),
            "mean_absolute_scale_residual": float(
                np.mean([row["absolute_scale_residual"] for row in parent_rows])
            ),
            "mean_nearest_distance": float(
                np.mean([row["nearest_distance"] for row in parent_rows])
            ),
            "mean_distance_ratio": float(
                np.mean([row["distance_ratio"] for row in parent_rows])
            ),
        }
        for parent, parent_rows in sorted(by_parent.items())
    }
    metric_names = tuple(next(iter(parent_metrics.values())))
    parent_macro = {
        name: float(np.mean([values[name] for values in parent_metrics.values()]))
        for name in metric_names
        if name != "frame_count"
    }
    return {
        "frame_count": len(rows),
        "parent_count": len(parent_metrics),
        "parent_macro_metrics": parent_macro,
        "parent_metrics": parent_metrics,
        "frames": rows,
    }


def parent_balanced_quantile(
    values: np.ndarray,
    parents: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    parents = np.asarray(parents)
    weights = np.zeros(len(values), dtype=np.float64)
    for parent in sorted(set(parents.tolist())):
        selected = parents == parent
        weights[selected] = 1.0 / float(np.sum(selected))
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    centers = (np.cumsum(weights) - 0.5 * weights) / float(np.sum(weights))
    return float(
        np.interp(
            probability,
            centers,
            values,
            left=values[0],
            right=values[-1],
        )
    )
def predict_scale_sigma(
    pooled_feature: torch.Tensor | np.ndarray,
    bank: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    feature = (
        pooled_feature.detach().float().cpu().numpy()
        if isinstance(pooled_feature, torch.Tensor)
        else np.asarray(pooled_feature, dtype=np.float32)
    )
    query = transform_embedding(feature.reshape(-1), bank)
    raw_sigma, diagnostics = raw_sigma_from_bank(
        query,
        np.asarray(bank["bank_embedding"], dtype=np.float64),
        np.asarray(bank["signed_scale_residuals"], dtype=np.float64),
        k=int(bank["selected_k"]),
        distance_alpha=float(bank["selected_distance_alpha"]),
        distance_reference=float(bank["distance_reference"]),
        global_rms=float(bank["global_residual_rms"]),
    )
    sigma = float(
        np.clip(
            raw_sigma * float(bank["selected_multiplier"]),
            SIGMA_FLOOR,
            SIGMA_CAP,
        )
    )
    return sigma, diagnostics


def load_residual_bank(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: np.asarray(payload[key]) for key in payload.files}
    scalar_names = (
        "selected_k",
        "selected_distance_alpha",
        "selected_multiplier",
        "distance_reference",
        "global_residual_rms",
    )
    for name in scalar_names:
        result[name] = np.asarray(result[name]).item()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(
        torch.cuda.is_available() and str(args.device).startswith("cuda"),
        "CUDA required",
    )
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(
        sha256_file(args.corpus_result) == EXPECTED_CORPUS_RESULT_SHA256,
        "multi-source corpus drift",
    )
    require(
        sha256_file(args.metric_student_result)
        == EXPECTED_METRIC_STUDENT_RESULT_SHA256,
        "multi-source metric student result drift",
    )
    require(
        sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256,
        "DepthART drift",
    )
    corpus = json.loads(args.corpus_result.read_text(encoding="utf-8"))
    metric_result = json.loads(args.metric_student_result.read_text(encoding="utf-8"))
    require(corpus["passed"] and metric_result["passed"], "prerequisite failed")
    rows = sorted(
        [
            row
            for row in corpus["frames"]
            if int(row["metric_depth_valid_pixels"]) > 0
        ],
        key=lambda row: str(row["sample_id"]),
    )
    require(len(rows) == 179, "residual bank roster drift")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    metric_model, metric_checkpoint = load_metric_student(metric_result, device)
    examples = build_examples(samples, metric_model, device)
    del metric_model
    torch.cuda.empty_cache()
    features = np.stack(
        [row["feature"].float().numpy() for row in examples]
    ).astype(np.float64)
    residuals = np.asarray(
        [row["signed_scale_residual"] for row in examples], dtype=np.float64
    )
    embedding = fit_embedding(features)
    embedded = np.asarray(embedding["bank_embedding"], dtype=np.float64)
    parents = np.asarray([str(row["parent_id"]) for row in examples])
    nearest_other_parent: list[float] = []
    for index, parent in enumerate(parents):
        reference = parents != parent
        distances = np.linalg.norm(
            embedded[reference] - embedded[index][None], axis=1
        ) / math.sqrt(float(PCA_COMPONENTS))
        nearest_other_parent.append(float(np.min(distances)))
    distance_reference = float(np.median(nearest_other_parent))
    global_rms = float(np.sqrt(np.mean(np.square(residuals))))
    require(
        math.isfinite(distance_reference)
        and distance_reference > 0.0
        and math.isfinite(global_rms)
        and global_rms > 0.0,
        "residual bank normalization invalid",
    )

    candidates: list[dict[str, Any]] = []
    for k in K_CANDIDATES:
        for alpha in DISTANCE_ALPHA_CANDIDATES:
            unit_rows = leave_parent_out_rows(
                examples,
                embedded,
                residuals,
                k,
                alpha,
                1.0,
                distance_reference,
                global_rms,
            )
            ratios = np.asarray(
                [
                    row["absolute_scale_residual"]
                    / max(row["predicted_sigma"], SIGMA_FLOOR)
                    for row in unit_rows
                ],
                dtype=np.float64,
            )
            multiplier = float(
                np.clip(
                    parent_balanced_quantile(
                        ratios,
                        np.asarray([row["parent_id"] for row in unit_rows]),
                        ONE_SIGMA_TARGET_COVERAGE,
                    ),
                    0.25,
                    4.0,
                )
            )
            loo_rows = leave_parent_out_rows(
                examples,
                embedded,
                residuals,
                k,
                alpha,
                multiplier,
                distance_reference,
                global_rms,
            )
            summary = summarize_rows(loo_rows)
            candidates.append(
                {
                    "k": k,
                    "distance_alpha": alpha,
                    "multiplier": multiplier,
                    "coverage_error": abs(
                        summary["parent_macro_metrics"]["one_sigma_coverage"]
                        - ONE_SIGMA_TARGET_COVERAGE
                    ),
                    "parent_macro_metrics": summary["parent_macro_metrics"],
                }
            )
    eligible = [
        row
        for row in candidates
        if row["coverage_error"] <= ONE_SIGMA_COVERAGE_TOLERANCE
    ]
    require(bool(eligible), "no statistically calibrated residual-bank candidate")
    selected = min(
        eligible,
        key=lambda row: (
            row["parent_macro_metrics"]["gaussian_nll"],
            row["coverage_error"],
            row["k"],
            row["distance_alpha"],
        ),
    )
    selected_rows = leave_parent_out_rows(
        examples,
        embedded,
        residuals,
        int(selected["k"]),
        float(selected["distance_alpha"]),
        float(selected["multiplier"]),
        distance_reference,
        global_rms,
    )
    evaluation = summarize_rows(selected_rows)

    bank_path = args.output_dir / "metric-scale-residual-bank.npz"
    payload = {
        "schema": np.asarray("blindassist_ag_r2_metric_scale_residual_bank_v1"),
        **embedding,
        "signed_scale_residuals": residuals.astype(np.float32),
        "sample_ids": np.asarray([row["sample_id"] for row in examples]),
        "parent_ids": np.asarray([row["parent_id"] for row in examples]),
        "roles": np.asarray([row["role"] for row in examples]),
        "selected_k": np.asarray(int(selected["k"]), dtype=np.int64),
        "selected_distance_alpha": np.asarray(
            float(selected["distance_alpha"]), dtype=np.float64
        ),
        "selected_multiplier": np.asarray(
            float(selected["multiplier"]), dtype=np.float64
        ),
        "distance_reference": np.asarray(distance_reference, dtype=np.float64),
        "global_residual_rms": np.asarray(global_rms, dtype=np.float64),
        "sigma_floor": np.asarray(SIGMA_FLOOR, dtype=np.float64),
        "sigma_cap": np.asarray(SIGMA_CAP, dtype=np.float64),
    }
    np.savez_compressed(bank_path, **payload)
    loaded = load_residual_bank(bank_path)
    require(
        set(loaded) == set(payload)
        and np.array_equal(loaded["bank_embedding"], payload["bank_embedding"]),
        "residual bank roundtrip drift",
    )
    parent_metrics = evaluation["parent_metrics"]
    all_sigmas = [row["predicted_sigma"] for row in evaluation["frames"]]
    gates = {
        "SCALEBANK_C01_EXACT_CONSUMED_CORPUS_STUDENT_AND_DEPTHART": True,
        "SCALEBANK_C02_179_FRAME_15_PARENT_BANK": bool(
            len(examples) == 179 and len(set(parents.tolist())) == 15
        ),
        "SCALEBANK_C03_LEAVE_PARENT_OUT_SELECTION": True,
        "SCALEBANK_C04_ONE_SIGMA_COVERAGE_CALIBRATED": bool(
            selected["coverage_error"] <= ONE_SIGMA_COVERAGE_TOLERANCE
        ),
        "SCALEBANK_C05_FINITE_POSITIVE_BOUNDED_SIGMA": bool(
            all(
                math.isfinite(value) and SIGMA_FLOOR <= value <= SIGMA_CAP
                for value in all_sigmas
            )
        ),
        "SCALEBANK_C06_ICL_WIDER_THAN_REAL_TUM": bool(
            parent_metrics["icl_living_room_kt1"]["mean_predicted_sigma"]
            > parent_metrics["rgbd_dataset_freiburg3_sitting_static"][
                "mean_predicted_sigma"
            ]
        ),
        "SCALEBANK_C07_BANK_ROUNDTRIP_AND_HASH": bool(bank_path.is_file()),
        "SCALEBANK_C08_FACTOR_ONLY_NO_TASK_OR_REDUCER_OUTPUT": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_r2_metric_scale_residual_bank_result_v1",
        "status": "AG_R2_METRIC_SCALE_RESIDUAL_BANK_PASS_READY_FOR_V2_SEAM"
        if passed
        else "AG_R2_METRIC_SCALE_RESIDUAL_BANK_FAIL",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": {
            "path": str(args.corpus_result.resolve()),
            "sha256": EXPECTED_CORPUS_RESULT_SHA256,
            "all_roles_consumed_for_bank_calibration": True,
        },
        "metric_depth_student": {
            "result": str(args.metric_student_result.resolve()),
            "result_sha256": EXPECTED_METRIC_STUDENT_RESULT_SHA256,
            "checkpoint": str(metric_checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_METRIC_STUDENT_CHECKPOINT_SHA256,
        },
        "feature_receipt": feature_receipt,
        "bank": {
            "path": str(bank_path.resolve()),
            "sha256": sha256_file(bank_path),
            "bytes": bank_path.stat().st_size,
            "frame_count": len(examples),
            "parent_count": len(set(parents.tolist())),
            "embedding_dimension": PCA_COMPONENTS,
            "distance_reference": distance_reference,
            "global_residual_rms": global_rms,
            "sigma_floor": SIGMA_FLOOR,
            "sigma_cap": SIGMA_CAP,
        },
        "candidate_grid": {
            "k": list(K_CANDIDATES),
            "distance_alpha": list(DISTANCE_ALPHA_CANDIDATES),
            "multiplier": "PARENT_BALANCED_CONFORMAL_ONE_SIGMA_QUANTILE",
            "target_coverage": ONE_SIGMA_TARGET_COVERAGE,
            "coverage_tolerance": ONE_SIGMA_COVERAGE_TOLERANCE,
        },
        "candidates": candidates,
        "selected": selected,
        "leave_parent_out_evaluation": evaluation,
        "gates": gates,
        "decision": {
            "per_frame_metric_scale_uncertainty_available": passed,
            "method": "PCA_WHITENED_KNN_RESIDUAL_RMS_WITH_OOD_DISTANCE_INFLATION",
            "task_or_reducer_output_used": False,
            "next_action": "Serialize this per-frame scale sigma separately from local shape sigma through FactorTensorAdapter v2 and run one consumed seam.",
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-result", type=Path, default=DEFAULT_CORPUS_RESULT)
    parser.add_argument(
        "--metric-student-result",
        type=Path,
        default=DEFAULT_METRIC_STUDENT_RESULT,
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument(
        "--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "corpus_result",
        "metric_student_result",
        "depthart_source",
        "depthart_checkpoint",
        "depthart_extension",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "bank": result["bank"],
                "selected": result["selected"],
                "parent_macro": result["leave_parent_out_evaluation"][
                    "parent_macro_metrics"
                ],
                "icl": result["leave_parent_out_evaluation"]["parent_metrics"][
                    "icl_living_room_kt1"
                ],
                "tum_real": result["leave_parent_out_evaluation"][
                    "parent_metrics"
                ]["rgbd_dataset_freiburg3_sitting_static"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
