#!/usr/bin/env python3
"""Run one frozen DINOv2-S regional pair-direction probe.

This is a fixed alternative teacher test after ADE semantic and nonlinear
adapter failures. It uses the last DINOv2 layer only, fixed 224 input, and
fixed global/lower/core/context regions. No layer, region, threshold or head
search is performed and no weights are saved.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_synthetic_pair_delta_probe as pair_probe


SCHEMA = "blindassist_public_silver_dinov2_regional_pair_probe_v1"


def regional_vector(last_hidden_state: np.ndarray) -> np.ndarray:
    values = np.asarray(last_hidden_state, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("DINOv2 tokens must be finite TxD")
    cls, patches = values[0], values[1:]
    side = int(round(math.sqrt(len(patches))))
    if side * side != len(patches):
        raise ValueError("DINOv2 patch tokens are not a square grid")
    grid = patches.reshape(side, side, -1)
    lower = grid[side // 2 :, :, :]
    left, right = side // 4, side - side // 4
    core = grid[side // 2 :, left:right, :]
    peripheral = np.concatenate([grid[side // 2 :, :left, :].reshape(-1, grid.shape[-1]), grid[side // 2 :, right:, :].reshape(-1, grid.shape[-1])], axis=0)
    vector = np.concatenate([cls, grid.mean(axis=(0, 1)), lower.mean(axis=(0, 1)), core.mean(axis=(0, 1)), core.mean(axis=(0, 1)) - peripheral.mean(axis=0)])
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("DINOv2 regional vector is degenerate")
    return vector / norm


def spatial_grid_vector(last_hidden_state: np.ndarray, *, output_side: int = 4) -> np.ndarray:
    """Keep coarse patch position instead of averaging translation evidence away."""
    values = np.asarray(last_hidden_state, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("DINOv2 tokens must be finite TxD")
    patches = values[1:]
    side = int(round(math.sqrt(len(patches))))
    if side * side != len(patches) or side % output_side != 0:
        raise ValueError("DINOv2 patch grid must be square and divisible by output_side")
    grid = patches.reshape(side, side, -1)
    block = side // output_side
    pooled = grid.reshape(output_side, block, output_side, block, grid.shape[-1]).mean(axis=(1, 3))
    vector = pooled.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("DINOv2 spatial-grid vector is degenerate")
    return vector / norm


class FrozenDinoV2:
    def __init__(self, model_dir: Path, *, feature_mode: str = "regional_mean") -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
        if feature_mode not in {"regional_mean", "spatial_grid_4x4"}:
            raise ValueError(f"unsupported DINOv2 feature mode: {feature_mode}")
        self.feature_mode = feature_mode

    def extract(self, images: Sequence[np.ndarray], *, batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in batch]
            inputs = self.processor(images=rgb, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state.cpu().numpy()
            vectorizer = regional_vector if self.feature_mode == "regional_mean" else spatial_grid_vector
            vectors.extend(vectorizer(row) for row in hidden)
        return np.stack(vectors)


def evaluate_folds(real_pairs: Sequence[dict[str, Any]], auxiliary_pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for held in real_pairs:
        source = held["parent_source_id"]
        training = [row for row in list(real_pairs) + list(auxiliary_pairs) if row["parent_source_id"] != source]
        direction = pair_probe.prototype_direction([row["delta"] for row in training])
        projection = float(held["delta"] @ direction)
        folds.append({
            "held_out_pair_id": held["pair_id"],
            "held_out_parent_source_id": source,
            "training_real_pair_ids": [row["pair_id"] for row in training if row["kind"] == "real"],
            "training_forward_synthetic_pair_ids": [row["pair_id"] for row in training if row["kind"] == "synthetic"],
            "training_inverse_pair_ids": [row["pair_id"] for row in training if row["kind"] == "inverse"],
            "held_out_source_descendants_excluded": all(row["parent_source_id"] != source for row in training),
            "projection": projection,
            "ordered": projection > 0.0,
        })
    return folds


def load_manifest_pairs(teacher: FrozenDinoV2, dataset_root: Path, *, kind: str, batch_size: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], []).append(row)
    result: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        clear = next(row for row in members if row["attributes"]["risk_state"] == "clear")
        risk = next(row for row in members if row["attributes"]["risk_state"] == "risk")
        images = [cv2.imread(str(dataset_root / row["image_path"]), cv2.IMREAD_COLOR) for row in (clear, risk)]
        if any(image is None for image in images):
            raise ValueError(f"cannot decode {kind} pair: {pair_id}")
        vectors = teacher.extract(images, batch_size=batch_size)
        result.append({"kind": kind, "pair_id": pair_id, "parent_source_id": risk["source"]["parent_source_id"], "delta": vectors[1] - vectors[0]})
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.dataset_root, args.synthetic_audit, args.inverse_dataset_root, args.inverse_audit, args.model_dir, args.rice_video, args.rice_review, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    forward_audit = lifecycle.verify_json_sidecar(args.synthetic_audit.resolve())
    inverse_audit = lifecycle.verify_json_sidecar(args.inverse_audit.resolve())
    if forward_audit.get("parent_source_isolated_representation_short_run_authorized") is not True or inverse_audit.get("parent_source_isolated_representation_short_run_authorized") is not True:
        raise ValueError("an auxiliary dataset audit did not authorize the probe")
    teacher = FrozenDinoV2(args.model_dir.resolve())
    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    real_pairs: list[dict[str, Any]] = []
    for pair_id in sorted(set(qualified[temporal.STATIC])):
        members = [row for row in episodes if row.get("counterfactual_pair_id") == pair_id]
        clear = next(row for row in members if int(row["label"]) == 0)
        risk = next(row for row in members if int(row["label"]) == 1)
        clear_vector = teacher.extract(background._episode_images(clear), batch_size=args.batch_size).mean(axis=0)
        risk_vector = teacher.extract(background._episode_images(risk), batch_size=args.batch_size).mean(axis=0)
        real_pairs.append({"kind": "real", "pair_id": pair_id, "parent_source_id": clear["source_id"], "delta": risk_vector - clear_vector})
    forward_pairs = load_manifest_pairs(teacher, args.dataset_root.resolve(), kind="synthetic", batch_size=args.batch_size)
    inverse_pairs = load_manifest_pairs(teacher, args.inverse_dataset_root.resolve(), kind="inverse", batch_size=args.batch_size)
    auxiliary_pairs = forward_pairs + inverse_pairs
    folds = evaluate_folds(real_pairs, auxiliary_pairs)
    ordering_rate = sum(row["ordered"] for row in folds) / len(folds)
    final_direction = pair_probe.prototype_direction([row["delta"] for row in real_pairs + auxiliary_pairs])

    review = common.load_json(args.rice_review).get("review") or {}
    rice: dict[str, np.ndarray] = {}
    for name, field in {"pre_clear": "pre_risk_clear_window_ms", "risk": "risk_present_window_ms", "post_clear": "stable_post_clear_window_ms"}.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000)
        rice[name] = teacher.extract(images, batch_size=args.batch_size).mean(axis=0)
    open_projection = float((rice["risk"] - rice["pre_clear"]) @ final_direction)
    close_projection = float((rice["risk"] - rice["post_clear"]) @ final_direction)
    gate = bool(ordering_rate == 1.0 and open_projection > 0.0 and close_projection > 0.0 and all(row["held_out_source_descendants_excluded"] for row in folds))
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "single_fixed_frozen_dinov2_teacher_probe",
        "inputs": {
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "forward_audit_sha256": common.sha256_file(args.synthetic_audit),
            "inverse_audit_sha256": common.sha256_file(args.inverse_audit),
            "forward_manifest_sha256": common.sha256_file(args.dataset_root / "manifest.jsonl"),
            "inverse_manifest_sha256": common.sha256_file(args.inverse_dataset_root / "manifest.jsonl"),
            "model_weights_sha256": common.sha256_file(args.model_dir / "model.safetensors"),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
        },
        "feature_contract": {
            "model": "facebook/dinov2-small", "input": "processor-default fixed 224 crop", "layer": "last_hidden_state only",
            "regions": ["cls", "global_patch_mean", "lower_half_mean", "lower_center_half_mean", "lower_center_minus_lower_peripheral"],
            "frame_vector_l2_normalized": True, "episode_pool": "frame_mean", "feature_dimension": int(len(final_direction)),
            "pair_direction": "unit pair deltas averaged then unit normalized", "layer_search": False, "region_search": False,
            "threshold_fitted": False, "trainable_parameters": 0, "saved_weights": False,
        },
        "training_pair_counts": {"real": len(real_pairs), "forward_synthetic": len(forward_pairs), "inverse_real_risk": len(inverse_pairs)},
        "source_isolated_real_pair_folds": folds,
        "real_pair_ordering_rate": ordering_rate,
        "rice_external_pressure": {"open_projection": open_projection, "close_projection": close_projection, "open_ordered": open_projection > 0.0, "close_ordered": close_projection > 0.0, "used_for_training": False},
        "fixed_teacher_gate": {"passed": gate, "requirements": {"all_real_pairs_ordered": True, "rice_open_and_close_ordered": True, "parent_descendants_excluded": True}},
        "five_seed_bootstrap_authorized": False,
        "evidence_limit": "One retrospective frozen alternative-teacher probe. A pass would only authorize a separately frozen representation-training contract and new prospective source, not deployment.",
        "calibration_authorized": False, "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
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
    parser.add_argument("--inverse-dataset-root", type=Path, required=True)
    parser.add_argument("--inverse-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "real_pair_ordering_rate": payload["real_pair_ordering_rate"], "rice_open_ordered": payload["rice_external_pressure"]["open_ordered"], "rice_close_ordered": payload["rice_external_pressure"]["close_ordered"], "gate_passed": payload["fixed_teacher_gate"]["passed"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
