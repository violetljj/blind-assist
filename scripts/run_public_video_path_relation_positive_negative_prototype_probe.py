#!/usr/bin/env python3
"""Test whether frozen DINO path-relation deltas are linearly separable.

Unlike the positive-only r7.35 direction, this probe builds unit positive and
negative prototypes inside each held-source fold.  Synthetic descendants share
the Jakarta parent source and are excluded whenever that source is held out.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_path_relation_positive_negative_prototype_probe_v2"
JAKARTA_SOURCE = "youtube_cc_jakarta_car_free_reopening_2026"


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError("prototype vector must be finite and non-degenerate")
    return value / norm


def prototype_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets) or set(targets.tolist()) != {0, 1}:
        raise ValueError("prototype fitting requires aligned features and both classes")
    normalized = np.stack([unit(row) for row in values])
    positive = unit(normalized[targets == 1].mean(axis=0))
    negative = unit(normalized[targets == 0].mean(axis=0))
    return positive - negative


def project_out_nuisance(delta: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    value = np.asarray(delta, dtype=np.float64)
    drift = np.asarray(nuisance, dtype=np.float64)
    if value.shape != drift.shape or value.ndim != 1:
        raise ValueError("delta and nuisance vectors must be aligned")
    norm = float(np.linalg.norm(drift))
    if norm <= 1e-12:
        return value.copy()
    direction = drift / norm
    return value - float(value @ direction) * direction


def leave_one_source_out(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[str],
    source_ids: Sequence[str],
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    sources = np.asarray(source_ids, dtype=object)
    predictions = np.full(len(targets), -1, dtype=np.int64)
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    folds = []
    directions: dict[str, np.ndarray] = {}
    for source_id in dict.fromkeys(source_ids):
        holdout = sources == source_id
        train = ~holdout
        direction = prototype_direction(values[train], targets[train])
        fold_scores = np.stack([unit(row) for row in values[holdout]]) @ direction
        fold_predictions = (fold_scores > 0.0).astype(np.int64)
        predictions[holdout] = fold_predictions
        scores[holdout] = fold_scores
        directions[source_id] = direction
        folds.append({
            "held_out_source_id": source_id,
            "held_out_sample_ids": [sample_ids[index] for index in np.flatnonzero(holdout)],
            "expected": targets[holdout].tolist(),
            "scores": fold_scores.tolist(),
            "predicted": fold_predictions.tolist(),
        })
    if np.any(predictions < 0) or not np.isfinite(scores).all():
        raise RuntimeError("prototype evaluation left an invalid prediction")
    return {
        "predictions": predictions.tolist(),
        "scores": scores.tolist(),
        "folds": folds,
        "metrics": common.binary_metrics(targets, predictions),
        "directions": directions,
    }


def mean_window(
    teacher: dino.FrozenDinoV2,
    video: Path,
    window: tuple[int, int],
    *,
    interval_ms: int,
    batch_size: int,
) -> np.ndarray:
    frames = background.decode_video_window(video, window[0], window[1], interval_ms=interval_ms)
    return teacher.extract(frames, batch_size=batch_size).mean(axis=0)


def video_delta(
    teacher: dino.FrozenDinoV2,
    video: Path,
    clear_window: tuple[int, int],
    marker_window: tuple[int, int],
    *,
    interval_ms: int,
    batch_size: int,
    nuisance_mode: str,
) -> tuple[np.ndarray, float]:
    clear_frames = background.decode_video_window(video, clear_window[0], clear_window[1], interval_ms=interval_ms)
    marker_frames = background.decode_video_window(video, marker_window[0], marker_window[1], interval_ms=interval_ms)
    clear_vectors = teacher.extract(clear_frames, batch_size=batch_size)
    marker_vectors = teacher.extract(marker_frames, batch_size=batch_size)
    delta = marker_vectors.mean(axis=0) - clear_vectors.mean(axis=0)
    nuisance_norm = 0.0
    if nuisance_mode == "project_out_clear_drift":
        if len(clear_vectors) < 4:
            raise ValueError("clear-drift projection requires at least four clear samples")
        midpoint = len(clear_vectors) // 2
        nuisance = clear_vectors[midpoint:].mean(axis=0) - clear_vectors[:midpoint].mean(axis=0)
        nuisance_norm = float(np.linalg.norm(nuisance))
        delta = project_out_nuisance(delta, nuisance)
    elif nuisance_mode != "none":
        raise ValueError(f"unsupported nuisance mode: {nuisance_mode}")
    return delta, nuisance_norm


def source_by_id(report: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = [row for row in report["sources"] if row["source_id"] == source_id]
    if len(rows) != 1:
        raise ValueError(f"expected one source in feature report: {source_id}")
    return rows[0]


def real_sample(
    teacher: dino.FrozenDinoV2,
    report: dict[str, Any],
    *,
    sample_id: str,
    source_id: str,
    label: int,
    clear_window: tuple[int, int],
    marker_window: tuple[int, int],
    interval_ms: int,
    batch_size: int,
    nuisance_mode: str,
) -> dict[str, Any]:
    source = source_by_id(report, source_id)
    video = Path(source["local_video_path"])
    delta, nuisance_norm = video_delta(
        teacher, video, clear_window, marker_window,
        interval_ms=interval_ms, batch_size=batch_size, nuisance_mode=nuisance_mode,
    )
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "label": label,
        "video_sha256": source["video_sha256"],
        "clear_window_ms": list(clear_window),
        "marker_window_ms": list(marker_window),
        "sample_interval_ms": interval_ms,
        "clear_drift_nuisance_norm": nuisance_norm,
        "delta": delta,
    }


def synthetic_samples(
    teacher: dino.FrozenDinoV2,
    dataset_root: Path,
    manifest: Sequence[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], {})[
            row["attributes"]["risk_state"]
        ] = row
    samples = []
    mirrors = []
    for pair_id, members in sorted(grouped.items()):
        if set(members) != {"clear", "risk"}:
            raise ValueError(f"synthetic pair must contain clear and risk states: {pair_id}")
        parent_sources = {members[state]["source"]["parent_source_id"] for state in ("clear", "risk")}
        if len(parent_sources) != 1:
            raise ValueError(f"synthetic pair states disagree on parent source: {pair_id}")
        parent_source_id = next(iter(parent_sources))
        images = [cv2.imread(str(dataset_root / members[state]["image_path"]), cv2.IMREAD_COLOR) for state in ("clear", "risk")]
        if any(image is None for image in images):
            raise ValueError(f"cannot decode synthetic pair: {pair_id}")
        vectors = teacher.extract(images, batch_size=batch_size)
        mirror_vectors = teacher.extract([cv2.flip(image, 1) for image in images], batch_size=batch_size)
        samples.append({
            "sample_id": f"synthetic_{pair_id}",
            "source_id": parent_source_id,
            "label": 1,
            "train_only": True,
            "delta": vectors[1] - vectors[0],
        })
        mirrors.append({
            "sample_id": f"synthetic_{pair_id}_mirror",
            "source_id": parent_source_id,
            "delta": mirror_vectors[1] - mirror_vectors[0],
        })
    return samples, mirrors


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        args.dataset_root, args.generation_report, args.model_dir,
        *args.additional_dataset_root, *args.additional_generation_report,
        args.japan_features, args.edmonton_features, args.jakarta_features,
        args.cape_town_features, args.bramwell_features, args.dallas_features,
        *args.review, args.output,
    ]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    if len(args.additional_dataset_root) != len(args.additional_generation_report):
        raise ValueError("additional dataset roots and generation reports must have equal counts")
    dataset_inputs = [(args.dataset_root, args.generation_report), *zip(args.additional_dataset_root, args.additional_generation_report)]
    verified_datasets = []
    for dataset_root, generation_report in dataset_inputs:
        generation = lifecycle.verify_json_sidecar(generation_report)
        manifest_path = dataset_root / "manifest.jsonl"
        if common.sha256_file(manifest_path) != generation["manifest"]["sha256"]:
            raise ValueError("manifest differs from generation report")
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        verified_datasets.append((dataset_root, generation_report, manifest_path, manifest))
    reports = {
        "japan": lifecycle.verify_json_sidecar(args.japan_features),
        "edmonton": lifecycle.verify_json_sidecar(args.edmonton_features),
        "jakarta": lifecycle.verify_json_sidecar(args.jakarta_features),
        "cape": lifecycle.verify_json_sidecar(args.cape_town_features),
        "bramwell": lifecycle.verify_json_sidecar(args.bramwell_features),
        "dallas": lifecycle.verify_json_sidecar(args.dallas_features),
    }
    review_hashes = [common.sha256_file(path) for path in args.review]
    for path in args.review:
        lifecycle.verify_json_sidecar(path)
    teacher = dino.FrozenDinoV2(args.model_dir, feature_mode=args.feature_mode)
    samples = []
    mirrors = []
    for dataset_root, _, _, manifest in verified_datasets:
        dataset_samples, dataset_mirrors = synthetic_samples(teacher, dataset_root, manifest, batch_size=args.batch_size)
        samples.extend(dataset_samples)
        mirrors.extend(dataset_mirrors)
    samples.extend([
        real_sample(teacher, reports["japan"], sample_id="japan_path_intrusion", source_id="wikimedia_commons_japan_rural_riverside_walk_2025", label=1, clear_window=(17000, 22000), marker_window=(10000, 14000), interval_ms=1000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["edmonton"], sample_id="edmonton_left_corridor_intrusion", source_id="youtube_cc_edmonton_city_construction_chaos_pov_2025", label=1, clear_window=(782000, 810000), marker_window=(671000, 735000), interval_ms=4000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["jakarta"], sample_id="jakarta_dense_boundary", source_id=JAKARTA_SOURCE, label=0, clear_window=(0, 15000), marker_window=(35000, 49000), interval_ms=2000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["cape"], sample_id="cape_town_wide_forecourt", source_id="youtube_cc_cape_town_waterfront_construction_walk_2026", label=0, clear_window=(115000, 125000), marker_window=(158000, 176000), interval_ms=2000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["bramwell"], sample_id="bramwell_grassy_shoulder_cone", source_id="wikimedia_commons_bramwell_west_virginia_walk_2019", label=0, clear_window=(370000, 375000), marker_window=(379000, 382000), interval_ms=1000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["dallas"], sample_id="dallas_grass_detour_panel", source_id="youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025", label=0, clear_window=(225000, 234000), marker_window=(236000, 240000), interval_ms=1000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
        real_sample(teacher, reports["dallas"], sample_id="dallas_road_edge_cone", source_id="youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025", label=0, clear_window=(263000, 267000), marker_window=(268000, 270000), interval_ms=1000, batch_size=args.batch_size, nuisance_mode=args.nuisance_mode),
    ])
    features = np.stack([row["delta"] for row in samples])
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    sample_ids = [row["sample_id"] for row in samples]
    source_ids = [row["source_id"] for row in samples]
    evaluation = leave_one_source_out(features, labels, sample_ids, source_ids)
    mirror_scores = [float(unit(row["delta"]) @ evaluation["directions"][row["source_id"]]) for row in mirrors]
    mirror_rows = [
        {
            "sample_id": row["sample_id"],
            "parent_source_id": row["source_id"],
            "held_parent_source_score": score,
            "predicted_positive": score > 0.0,
        }
        for row, score in zip(mirrors, mirror_scores)
    ]
    perfect = evaluation["metrics"]["balanced_accuracy"] == 1.0
    gate = bool(perfect and all(row["predicted_positive"] for row in mirror_rows))
    serializable_samples = []
    for row, score, prediction in zip(samples, evaluation["scores"], evaluation["predictions"]):
        serializable_samples.append({
            key: value for key, value in row.items() if key != "delta"
        } | {
            "delta_norm": float(np.linalg.norm(row["delta"])),
            "held_source_score": score,
            "predicted_label": prediction,
            "correct": prediction == row["label"],
        })
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_source_isolated_positive_negative_prototype_diagnostic",
        "inputs": {
            "generation_report_sha256": common.sha256_file(args.generation_report),
            "manifest_sha256": common.sha256_file(args.dataset_root / "manifest.jsonl"),
            "synthetic_datasets": [{
                "dataset_root": str(dataset_root.resolve()),
                "generation_report_sha256": common.sha256_file(generation_report),
                "manifest_sha256": common.sha256_file(manifest_path),
            } for dataset_root, generation_report, manifest_path, _ in verified_datasets],
            "model_weights_sha256": common.sha256_file(args.model_dir / "model.safetensors"),
            "feature_report_sha256": {key: common.sha256_file(path) for key, path in {
                "japan": args.japan_features, "edmonton": args.edmonton_features,
                "jakarta": args.jakarta_features, "cape_town": args.cape_town_features,
                "bramwell": args.bramwell_features, "dallas": args.dallas_features,
            }.items()},
            "review_sha256": review_hashes,
        },
        "feature_contract": {
            "model": "facebook/dinov2-small",
            "feature_mode": args.feature_mode,
            "regional_vector": ["cls", "global_patch_mean", "lower_half_mean", "lower_center_half_mean", "lower_center_minus_lower_peripheral"] if args.feature_mode == "regional_mean" else None,
            "spatial_grid": {"output_side": 4, "flattened_patch_token_grid": True, "cls_included": False} if args.feature_mode == "spatial_grid_4x4" else None,
            "pair_feature": "unit-normalized risk-or-marker minus clear regional vector",
            "fold_classifier": "unit positive prototype minus unit negative prototype; zero threshold",
            "nuisance_mode": args.nuisance_mode,
            "split": "leave one source out; every synthetic descendant inherits its own public-video parent source",
            "trainable_parameters": 0,
            "threshold_fitted": False,
            "saved_weights": False,
        },
        "samples": serializable_samples,
        "source_isolated_evaluation": {
            "folds": evaluation["folds"],
            "metrics": evaluation["metrics"],
        },
        "horizontal_mirror_parent_holdout": mirror_rows,
        "diagnostic_gate": {
            "passed": gate,
            "requirements": ["balanced accuracy equals one under source holdout", "all mirrored synthetic pairs are positive under their own parent-source holdout direction"],
        },
        "authorizations": {
            "five_prototype_bootstrap_short_runs": gate,
            "future_prospective_contract_freeze": False,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "Retrospective GPT/VLM silver with train-only synthetic descendants. A pass diagnoses head separability and only authorizes five short stability runs, not deployment or event truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--additional-dataset-root", type=Path, action="append", default=[])
    parser.add_argument("--additional-generation-report", type=Path, action="append", default=[])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--japan-features", type=Path, required=True)
    parser.add_argument("--edmonton-features", type=Path, required=True)
    parser.add_argument("--jakarta-features", type=Path, required=True)
    parser.add_argument("--cape-town-features", type=Path, required=True)
    parser.add_argument("--bramwell-features", type=Path, required=True)
    parser.add_argument("--dallas-features", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-mode", choices=("regional_mean", "spatial_grid_4x4"), default="regional_mean")
    parser.add_argument("--nuisance-mode", choices=("none", "project_out_clear_drift"), default="none")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "gate_passed": value["diagnostic_gate"]["passed"],
        "metrics": value["source_isolated_evaluation"]["metrics"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
