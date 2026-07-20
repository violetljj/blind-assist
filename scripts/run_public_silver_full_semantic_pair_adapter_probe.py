#!/usr/bin/env python3
"""Probe a full-ADE semantic pair adapter with synthetic counterfactuals.

The frozen SegFormer stays unchanged. Each frame becomes a 450-D vector: all
ADE class probabilities averaged over fixed lower/core regions plus core minus
peripheral context. Unit risk-minus-clear pair deltas form a source-isolated
prototype. Synthetic descendants of every held-out parent source are excluded.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_segformer_free_space_probe as clearance
import run_public_silver_synthetic_pair_delta_probe as pair_probe


SCHEMA = "blindassist_public_silver_full_semantic_pair_adapter_probe_v1"


def semantic_vector(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("semantic vector requires finite CxHxW probabilities")
    masks = clearance.clearance_masks(values.shape[1], values.shape[2])
    lower = values[:, masks["lower"]].mean(axis=1)
    core = values[:, masks["core"]].mean(axis=1)
    peripheral = values[:, masks["peripheral"]].mean(axis=1)
    result = np.concatenate([lower, core, core - peripheral])
    if not np.isfinite(result).all():
        raise ValueError("semantic vector is not finite")
    return result


def extract_vectors(
    teacher: clearance.FrozenTeacher,
    images: Sequence[np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in batch]
        inputs = teacher.processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            probabilities = torch.softmax(teacher.model(**inputs).logits, dim=1).cpu().numpy()
        vectors.extend(semantic_vector(item) for item in probabilities)
    return np.stack(vectors)


def evaluate_folds(
    real_rows: Sequence[dict[str, Any]],
    synthetic_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held in real_rows:
        source = held["parent_source_id"]
        training = [row for row in list(real_rows) + list(synthetic_rows) if row["parent_source_id"] != source]
        direction = pair_probe.prototype_direction([row["delta"] for row in training])
        projection = float(held["delta"] @ direction)
        folds.append({
            "held_out_pair_id": held["pair_id"],
            "held_out_parent_source_id": source,
            "training_real_pair_ids": [row["pair_id"] for row in training if row["kind"] == "real"],
            "training_synthetic_pair_ids": [row["pair_id"] for row in training if row["kind"] == "synthetic"],
            "held_out_source_descendants_excluded": all(row["parent_source_id"] != source for row in training),
            "projection": projection,
            "ordered": projection > 0.0,
        })
    return folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.dataset_root, args.synthetic_audit, args.model_dir, args.rice_video, args.rice_review, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    synthetic_audit = lifecycle.verify_json_sidecar(args.synthetic_audit.resolve())
    if synthetic_audit.get("parent_source_isolated_representation_short_run_authorized") is not True:
        raise ValueError("synthetic audit did not authorize the source-isolated short run")
    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    static_ids = set(qualified[temporal.STATIC])
    teacher = clearance.FrozenTeacher(args.model_dir.resolve())

    real_rows: list[dict[str, Any]] = []
    for pair_id in sorted(static_ids):
        members = [row for row in episodes if row.get("counterfactual_pair_id") == pair_id]
        clear = next(row for row in members if int(row["label"]) == 0)
        risk = next(row for row in members if int(row["label"]) == 1)
        clear_vector = extract_vectors(teacher, background._episode_images(clear), batch_size=args.batch_size).mean(axis=0)
        risk_vector = extract_vectors(teacher, background._episode_images(risk), batch_size=args.batch_size).mean(axis=0)
        real_rows.append({
            "kind": "real", "pair_id": pair_id, "parent_source_id": clear["source_id"],
            "clear_episode_id": clear["episode_id"], "risk_episode_id": risk["episode_id"],
            "delta": risk_vector - clear_vector,
        })

    dataset_root = args.dataset_root.resolve()
    manifest = [json.loads(line) for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], []).append(row)
    synthetic_rows: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        clear = next(row for row in members if row["attributes"]["risk_state"] == "clear")
        risk = next(row for row in members if row["attributes"]["risk_state"] == "risk")
        images = [cv2.imread(str(dataset_root / row["image_path"]), cv2.IMREAD_COLOR) for row in (clear, risk)]
        if any(image is None for image in images):
            raise ValueError(f"cannot decode synthetic pair: {pair_id}")
        vectors = extract_vectors(teacher, images, batch_size=args.batch_size)
        synthetic_rows.append({
            "kind": "synthetic", "pair_id": pair_id,
            "parent_source_id": risk["source"]["parent_source_id"],
            "delta": vectors[1] - vectors[0],
        })

    folds = evaluate_folds(real_rows, synthetic_rows)
    ordering_rate = sum(row["ordered"] for row in folds) / len(folds)
    final_direction = pair_probe.prototype_direction([row["delta"] for row in real_rows + synthetic_rows])
    review = common.load_json(args.rice_review).get("review") or {}
    rice_vectors: dict[str, np.ndarray] = {}
    for name, field in {
        "pre_clear": "pre_risk_clear_window_ms",
        "risk": "risk_present_window_ms",
        "post_clear": "stable_post_clear_window_ms",
    }.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000)
        rice_vectors[name] = extract_vectors(teacher, images, batch_size=args.batch_size).mean(axis=0)
    open_projection = float((rice_vectors["risk"] - rice_vectors["pre_clear"]) @ final_direction)
    close_projection = float((rice_vectors["risk"] - rice_vectors["post_clear"]) @ final_direction)
    gate = bool(ordering_rate == 1.0 and open_projection > 0.0 and close_projection > 0.0 and all(row["held_out_source_descendants_excluded"] for row in folds))
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_full_semantic_adapter_after_r717_failure",
        "inputs": {
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "synthetic_audit_sha256": common.sha256_file(args.synthetic_audit),
            "manifest_sha256": common.sha256_file(dataset_root / "manifest.jsonl"),
            "model_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
        },
        "adapter_contract": {
            "feature_dimension": int(len(final_direction)),
            "regions": ["lower", "core", "core_minus_peripheral"],
            "semantic_classes": "all frozen ADE20K classes in model order",
            "pair_prototype": "unit-normalized risk-minus-clear deltas averaged and renormalized",
            "threshold_fitted": False,
            "backbone_trainable_parameters": 0,
            "saved_weights": False,
        },
        "source_isolated_real_pair_folds": folds,
        "real_pair_ordering_rate": ordering_rate,
        "rice_external_pressure": {
            "open_projection": open_projection,
            "close_projection": close_projection,
            "open_ordered": open_projection > 0.0,
            "close_ordered": close_projection > 0.0,
            "used_for_training": False,
        },
        "semantic_adapter_gate": {"passed": gate, "requirements": {"all_real_pairs_ordered": True, "rice_open_and_close_ordered": True, "parent_descendants_excluded": True}},
        "evidence_limit": "Retrospective six-synthetic-pair semantic adapter. Complementary success on Rice cannot rescue r7.17 or authorize post-hoc fusion; a new frozen prospective source is required.",
        "fusion_with_registered_residual_authorized": False,
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--synthetic-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({
        "ok": True,
        "real_pair_ordering_rate": payload["real_pair_ordering_rate"],
        "rice_open_ordered": payload["rice_external_pressure"]["open_ordered"],
        "rice_close_ordered": payload["rice_external_pressure"]["close_ordered"],
        "gate_passed": payload["semantic_adapter_gate"]["passed"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
