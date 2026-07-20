#!/usr/bin/env python3
"""Probe equal-count path-relation counterfactuals with frozen DINOv2-S."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_synthetic_pair_delta_probe as pair_probe


SCHEMA = "blindassist_public_video_path_relation_dinov2_probe_v1"


def leave_one_pair_out(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for held in deltas:
        training = [row["delta"] for row in deltas if row["pair_id"] != held["pair_id"]]
        direction = pair_probe.prototype_direction(training)
        projection = float(held["delta"] @ direction)
        rows.append({"held_out_pair_id": held["pair_id"], "projection": projection, "ordered": projection > 0.0})
    return rows


def mean_window(teacher: dino.FrozenDinoV2, video: Path, window: tuple[int, int], interval_ms: int, batch_size: int) -> np.ndarray:
    frames = background.decode_video_window(video, window[0], window[1], interval_ms=interval_ms)
    return teacher.extract(frames, batch_size=batch_size).mean(axis=0)


def pressure(
    teacher: dino.FrozenDinoV2,
    direction: np.ndarray,
    video: Path,
    clear_window: tuple[int, int],
    risk_window: tuple[int, int],
    interval_ms: int,
    batch_size: int,
) -> dict[str, Any]:
    clear = mean_window(teacher, video, clear_window, interval_ms, batch_size)
    risk = mean_window(teacher, video, risk_window, interval_ms, batch_size)
    projection = float((risk - clear) @ direction)
    return {
        "video_sha256": common.sha256_file(video),
        "clear_window_ms": list(clear_window),
        "risk_window_ms": list(risk_window),
        "sample_interval_ms": interval_ms,
        "projection": projection,
        "ordered_as_risk": projection > 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dataset_root, args.generation_report, args.model_dir, args.japan_video, args.edmonton_video, args.jakarta_video, args.cape_town_video, args.output):
        if "secondary-corridor-causal" in str(path.resolve()).replace("\\", "/").lower():
            raise ValueError("independent direction path is forbidden")
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    generation = lifecycle.verify_json_sidecar(args.generation_report)
    if not all(generation["summary"].values()):
        raise ValueError("equal-count generation audit did not pass")
    manifest_path = args.dataset_root / "manifest.jsonl"
    if common.sha256_file(manifest_path) != generation["manifest"]["sha256"]:
        raise ValueError("manifest hash differs from generation report")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], {})[row["attributes"]["risk_state"]] = row
    teacher = dino.FrozenDinoV2(args.model_dir)
    pairs = []
    mirrored = []
    for pair_id, members in sorted(grouped.items()):
        images = [cv2.imread(str(args.dataset_root / members[state]["image_path"]), cv2.IMREAD_COLOR) for state in ("clear", "risk")]
        if any(image is None for image in images):
            raise ValueError(f"cannot decode pair: {pair_id}")
        vectors = teacher.extract(images, batch_size=args.batch_size)
        pairs.append({"pair_id": pair_id, "delta": vectors[1] - vectors[0]})
        mirror_vectors = teacher.extract([cv2.flip(image, 1) for image in images], batch_size=args.batch_size)
        mirrored.append({"pair_id": pair_id, "delta": mirror_vectors[1] - mirror_vectors[0]})
    folds = leave_one_pair_out(pairs)
    direction = pair_probe.prototype_direction([row["delta"] for row in pairs])
    mirror_rows = [{"pair_id": row["pair_id"], "projection": float(row["delta"] @ direction), "ordered": bool(row["delta"] @ direction > 0.0)} for row in mirrored]
    cosine = np.stack([row["delta"] / max(np.linalg.norm(row["delta"]), 1e-12) for row in pairs])
    pairwise = cosine @ cosine.T
    real = {
        "japan_positive": pressure(teacher, direction, args.japan_video, (17000, 22000), (10000, 14000), 1000, args.batch_size),
        "edmonton_left_corridor_positive": pressure(teacher, direction, args.edmonton_video, (782000, 810000), (671000, 735000), 4000, args.batch_size),
        "jakarta_dense_boundary_negative": pressure(teacher, direction, args.jakarta_video, (0, 15000), (35000, 49000), 2000, args.batch_size),
        "cape_town_wide_forecourt_negative": pressure(teacher, direction, args.cape_town_video, (115000, 125000), (158000, 176000), 2000, args.batch_size),
    }
    gate = bool(
        all(row["ordered"] for row in folds)
        and all(row["ordered"] for row in mirror_rows)
        and real["japan_positive"]["ordered_as_risk"]
        and real["edmonton_left_corridor_positive"]["ordered_as_risk"]
        and not real["jakarta_dense_boundary_negative"]["ordered_as_risk"]
        and not real["cape_town_wide_forecourt_negative"]["ordered_as_risk"]
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_equal_count_path_relation_representation_diagnostic",
        "inputs": {
            "generation_report_sha256": common.sha256_file(args.generation_report),
            "manifest_sha256": common.sha256_file(manifest_path),
            "model_weights_sha256": common.sha256_file(args.model_dir / "model.safetensors"),
        },
        "feature_contract": {
            "model": "facebook/dinov2-small",
            "layer": "last_hidden_state only",
            "regions": ["cls", "global_patch_mean", "lower_half_mean", "lower_center_half_mean", "lower_center_minus_lower_peripheral"],
            "input": "processor-default fixed 224 crop",
            "pair_direction": "mean of unit risk-minus-clear pair deltas",
            "threshold_fitted": False,
            "trainable_parameters": 0,
            "saved_weights": False,
        },
        "synthetic_leave_one_pair_out": folds,
        "horizontal_mirror_pressure": mirror_rows,
        "pairwise_delta_cosine": pairwise.tolist(),
        "real_video_pressure": real,
        "diagnostic_gate": {
            "passed": gate,
            "requirements": ["all synthetic leave-one-pair-out ordered", "all mirror pairs ordered by original direction", "Japan and left-corridor Edmonton positives ordered", "Jakarta and Cape Town reviewed negatives not ordered as risk"],
        },
        "authorizations": {
            "future_prospective_contract_freeze": gate,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "Synthetic pairs are train-only and share one parent source. Real pressures are retrospective GPT/VLM silver episodes, not human truth or blind evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--japan-video", type=Path, required=True)
    parser.add_argument("--edmonton-video", type=Path, required=True)
    parser.add_argument("--jakarta-video", type=Path, required=True)
    parser.add_argument("--cape-town-video", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, "gate_passed": result["diagnostic_gate"]["passed"], "real_video_pressure": {key: value["projection"] for key, value in result["real_video_pressure"].items()}, "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
