#!/usr/bin/env python3
"""Probe explicit obstacle-to-ego-route geometry with frozen teachers.

Unlike residual-width probes, this contract first restores walkable support
inside marker masks so an obstacle cannot move the route estimate away from
itself.  It then measures the q10 distance from that restored route to the
expanded marker mask.  There are no learned parameters or fitted thresholds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_free_space_topology_probe as topology
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_explicit_ego_route_relation_probe_v1"
HORIZON_RATIO = 0.48
LOWER_LIMIT_RATIO = 0.94
SAFETY_MARGIN_OBJECT_HEIGHTS = 0.5


def restore_walkable_support(walkable: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    values = np.asarray(walkable, dtype=np.float32)
    blocked = np.asarray(obstacle, dtype=bool)
    if values.ndim != 2 or values.shape != blocked.shape or min(values.shape) < 8:
        raise ValueError("walkable and obstacle maps must be matching 2D arrays")
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("walkable probability must be finite and within zero to one")
    if not blocked.any():
        return values.copy()
    # Fixed-radius Telea restoration is not a route label. It only prevents the
    # visible marker pixels from suppressing the surface that must be estimated
    # before marker-to-route distance is measured.
    restored = cv2.inpaint(values, blocked.astype(np.uint8), 3.0, cv2.INPAINT_TELEA)
    return np.clip(restored, 0.0, 1.0)


def explicit_route_relation(walkable: np.ndarray, obstacle: np.ndarray) -> dict[str, Any]:
    values = np.asarray(walkable, dtype=np.float32)
    blocked = np.asarray(obstacle, dtype=bool)
    restored = restore_walkable_support(values, blocked)
    height, width = restored.shape
    centers, _ = topology.trace_adaptive_path(restored, horizon_ratio=0.30)
    first_row = height - len(centers)
    rows = np.arange(first_row, height)
    selected = (rows >= int(round(height * HORIZON_RATIO))) & (rows <= int(round(height * LOWER_LIMIT_RATIO)))
    rows = rows[selected]
    centers = centers[selected]
    if not len(rows):
        raise ValueError("restored ego route has no rows in the scoring band")
    if blocked.any():
        distance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5)
        clearances = np.minimum(distance[rows, centers] / width, 1.0)
    else:
        clearances = np.ones(len(rows), dtype=np.float32)
    clearance_q10 = float(np.quantile(clearances, 0.10))
    intrusion_score = 1.0 - clearance_q10
    normalized_centers = centers / max(width - 1, 1)
    return {
        "route_clearance_q10": clearance_q10,
        "route_intrusion_score": intrusion_score,
        "route_center_mean": float(normalized_centers.mean()),
        "route_center_range": float(np.ptp(normalized_centers)),
        "obstacle_pixel_fraction": float(blocked.mean()),
        "restored_pixel_fraction": float(blocked.mean()),
    }


def source_by_id(report: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = [row for row in report["sources"] if row["source_id"] == source_id]
    if len(rows) != 1:
        raise ValueError(f"expected one source in feature report: {source_id}")
    return rows[0]


def window_score(
    teacher: route_width.FrozenWalkableTeacher,
    source: dict[str, Any],
    window: tuple[int, int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    samples = [row for row in source["samples"] if window[0] <= int(row["timestamp_ms"]) < window[1]]
    if not samples:
        raise ValueError(f"feature window is empty: {window}")
    timestamps = [int(row["timestamp_ms"]) for row in samples]
    frames = route_width.decode_at(Path(source["local_video_path"]), timestamps)
    maps = teacher.probability_maps(frames, batch_size=batch_size)
    rows = []
    for walkable, sample in zip(maps, samples):
        obstacle = route_width.obstacle_mask_from_detections(
            sample.get("detections", []),
            walkable.shape,
            safety_margin_object_heights=SAFETY_MARGIN_OBJECT_HEIGHTS,
        )
        rows.append({"timestamp_ms": int(sample["timestamp_ms"]), **explicit_route_relation(walkable, obstacle)})
    scores = [float(row["route_intrusion_score"]) for row in rows]
    return {
        "window_ms": list(window),
        "frame_count": len(rows),
        "median_route_intrusion_score": float(median(scores)),
        "maximum_route_intrusion_score": float(max(scores)),
        "frames": rows,
    }


def real_pressure(
    teacher: route_width.FrozenWalkableTeacher,
    report: dict[str, Any],
    *,
    sample_id: str,
    source_id: str,
    label: int,
    clear_window: tuple[int, int],
    marker_window: tuple[int, int],
    batch_size: int,
) -> dict[str, Any]:
    source = source_by_id(report, source_id)
    clear = window_score(teacher, source, clear_window, batch_size=batch_size)
    marker = window_score(teacher, source, marker_window, batch_size=batch_size)
    delta = float(marker["median_route_intrusion_score"] - clear["median_route_intrusion_score"])
    prediction = int(delta > 0.0)
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "label": label,
        "video_sha256": source["video_sha256"],
        "clear": clear,
        "marker": marker,
        "marker_minus_clear_intrusion": delta,
        "predicted_label": prediction,
        "correct": prediction == label,
    }


def load_synthetic_dataset(
    teacher: route_width.FrozenWalkableTeacher,
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
            raise ValueError(f"synthetic pair must contain clear and risk: {pair_id}")
        sources = {members[state]["source"]["parent_source_id"] for state in ("clear", "risk")}
        if len(sources) != 1:
            raise ValueError(f"synthetic pair states disagree on parent source: {pair_id}")
        images = [cv2.imread(str(dataset_root / members[state]["image_path"]), cv2.IMREAD_COLOR) for state in ("clear", "risk")]
        masks = [cv2.imread(str(dataset_root / "masks" / f"{pair_id}_{state}_mask.png"), cv2.IMREAD_UNCHANGED) for state in ("clear", "risk")]
        if any(value is None for value in images + masks):
            raise ValueError(f"cannot decode synthetic pair: {pair_id}")
        maps = teacher.probability_maps(images, batch_size=batch_size)
        values = [
            explicit_route_relation(
                walkable,
                route_width.obstacle_mask_from_intervention(
                    mask, walkable.shape, safety_margin_object_heights=SAFETY_MARGIN_OBJECT_HEIGHTS
                ),
            )
            for walkable, mask in zip(maps, masks)
        ]
        delta = float(values[1]["route_intrusion_score"] - values[0]["route_intrusion_score"])
        samples.append({
            "sample_id": f"synthetic_{pair_id}",
            "source_id": next(iter(sources)),
            "label": 1,
            "clear": values[0],
            "risk": values[1],
            "risk_minus_clear_intrusion": delta,
            "predicted_label": int(delta > 0.0),
            "correct": delta > 0.0,
            "train_only": True,
        })
        mirror_images = [cv2.flip(image, 1) for image in images]
        mirror_masks = [cv2.flip(mask, 1) for mask in masks]
        mirror_maps = teacher.probability_maps(mirror_images, batch_size=batch_size)
        mirror_values = [
            explicit_route_relation(
                walkable,
                route_width.obstacle_mask_from_intervention(
                    mask, walkable.shape, safety_margin_object_heights=SAFETY_MARGIN_OBJECT_HEIGHTS
                ),
            )
            for walkable, mask in zip(mirror_maps, mirror_masks)
        ]
        mirror_delta = float(mirror_values[1]["route_intrusion_score"] - mirror_values[0]["route_intrusion_score"])
        mirrors.append({
            "sample_id": f"synthetic_{pair_id}_mirror",
            "source_id": next(iter(sources)),
            "risk_minus_clear_intrusion": mirror_delta,
            "predicted_positive": mirror_delta > 0.0,
        })
    return samples, mirrors


def binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["predicted_label"] for row in rows], dtype=np.int64)
    return common.binary_metrics(labels, predictions)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        args.dataset_root,
        args.generation_report,
        *args.additional_dataset_root,
        *args.additional_generation_report,
        args.model_dir,
        args.japan_features,
        args.edmonton_features,
        args.jakarta_features,
        args.cape_town_features,
        args.bramwell_features,
        args.dallas_features,
        *args.review,
        args.output,
    ]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    if len(args.additional_dataset_root) != len(args.additional_generation_report):
        raise ValueError("additional dataset roots and reports must have equal counts")
    dataset_inputs = [(args.dataset_root, args.generation_report), *zip(args.additional_dataset_root, args.additional_generation_report)]
    datasets = []
    for root, report_path in dataset_inputs:
        generation = lifecycle.verify_json_sidecar(report_path)
        manifest_path = root / "manifest.jsonl"
        if common.sha256_file(manifest_path) != generation["manifest"]["sha256"]:
            raise ValueError("manifest differs from generation report")
        manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        datasets.append((root, report_path, manifest_path, manifest))
    reports = {
        "japan": lifecycle.verify_json_sidecar(args.japan_features),
        "edmonton": lifecycle.verify_json_sidecar(args.edmonton_features),
        "jakarta": lifecycle.verify_json_sidecar(args.jakarta_features),
        "cape": lifecycle.verify_json_sidecar(args.cape_town_features),
        "bramwell": lifecycle.verify_json_sidecar(args.bramwell_features),
        "dallas": lifecycle.verify_json_sidecar(args.dallas_features),
    }
    for review in args.review:
        lifecycle.verify_json_sidecar(review)
    teacher = route_width.FrozenWalkableTeacher(args.model_dir)
    synthetic = []
    mirrors = []
    for root, _, _, manifest in datasets:
        rows, mirror_rows = load_synthetic_dataset(teacher, root, manifest, batch_size=args.batch_size)
        synthetic.extend(rows)
        mirrors.extend(mirror_rows)
    real = [
        real_pressure(teacher, reports["japan"], sample_id="japan_path_intrusion", source_id="wikimedia_commons_japan_rural_riverside_walk_2025", label=1, clear_window=(17000, 22000), marker_window=(10000, 14000), batch_size=args.batch_size),
        real_pressure(teacher, reports["edmonton"], sample_id="edmonton_left_corridor_intrusion", source_id="youtube_cc_edmonton_city_construction_chaos_pov_2025", label=1, clear_window=(782000, 810000), marker_window=(671000, 735000), batch_size=args.batch_size),
        real_pressure(teacher, reports["jakarta"], sample_id="jakarta_dense_boundary", source_id="youtube_cc_jakarta_car_free_reopening_2026", label=0, clear_window=(0, 15000), marker_window=(35000, 49000), batch_size=args.batch_size),
        real_pressure(teacher, reports["cape"], sample_id="cape_town_wide_forecourt", source_id="youtube_cc_cape_town_waterfront_construction_walk_2026", label=0, clear_window=(115000, 125000), marker_window=(158000, 176000), batch_size=args.batch_size),
        real_pressure(teacher, reports["bramwell"], sample_id="bramwell_grassy_shoulder_cone", source_id="wikimedia_commons_bramwell_west_virginia_walk_2019", label=0, clear_window=(370000, 375000), marker_window=(379000, 382000), batch_size=args.batch_size),
        real_pressure(teacher, reports["dallas"], sample_id="dallas_grass_detour_panel", source_id="youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025", label=0, clear_window=(225000, 234000), marker_window=(236000, 240000), batch_size=args.batch_size),
        real_pressure(teacher, reports["dallas"], sample_id="dallas_road_edge_cone", source_id="youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025", label=0, clear_window=(263000, 267000), marker_window=(268000, 270000), batch_size=args.batch_size),
    ]
    real_metrics = binary_metrics(real)
    gate = bool(
        real_metrics["balanced_accuracy"] == 1.0
        and all(row["correct"] for row in synthetic)
        and all(row["predicted_positive"] for row in mirrors)
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_zero_parameter_explicit_obstacle_to_restored_ego_route_diagnostic",
        "inputs": {
            "datasets": [{
                "generation_report_sha256": common.sha256_file(report_path),
                "manifest_sha256": common.sha256_file(manifest_path),
            } for _, report_path, manifest_path, _ in datasets],
            "segformer_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
            "feature_report_sha256": {key: common.sha256_file(path) for key, path in {
                "japan": args.japan_features,
                "edmonton": args.edmonton_features,
                "jakarta": args.jakarta_features,
                "cape": args.cape_town_features,
                "bramwell": args.bramwell_features,
                "dallas": args.dallas_features,
            }.items()},
            "review_sha256": [common.sha256_file(path) for path in args.review],
        },
        "feature_contract": {
            "walkable_teacher": "nvidia/segformer-b2-finetuned-ade-512-512",
            "walkable_labels": list(route_width.WALKABLE_LABELS),
            "map_size": route_width.MAP_SIZE,
            "marker_safety_margin_object_heights_per_side": SAFETY_MARGIN_OBJECT_HEIGHTS,
            "route_reconstruction": "fixed-radius Telea restoration only inside expanded marker mask, then frozen adaptive walkable path",
            "relation": "one minus q10 normalized obstacle distance along restored ego route",
            "prediction": "marker-minus-clear relation delta greater than zero",
            "threshold_fitted": False,
            "trainable_parameters": 0,
            "saved_weights": False,
        },
        "synthetic_pairs": synthetic,
        "horizontal_mirror_pressure": mirrors,
        "real_video_pressure": real,
        "real_metrics": real_metrics,
        "diagnostic_gate": {
            "passed": gate,
            "requirements": [
                "real positive and negative balanced accuracy equals one under fixed zero delta",
                "all train-only equal-count pairs order risk above clear",
                "all mirrored train-only pairs retain ordering",
            ],
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
        "evidence_limit": "Retrospective GPT/VLM silver and train-only synthetic pressure. A pass only authorizes five fixed-seed diagnostic short runs, not event truth, Android, or deployment.",
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "gate_passed": value["diagnostic_gate"]["passed"],
        "real_metrics": value["real_metrics"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
