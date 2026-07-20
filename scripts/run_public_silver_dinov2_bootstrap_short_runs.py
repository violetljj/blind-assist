#!/usr/bin/env python3
"""Run five source-bootstrap pair-ranking heads on frozen DINOv2 features."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_dinov2_regional_pair_probe as dino
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_dinov2_bootstrap_short_runs_v1"
SEEDS = (2026071901, 2026071902, 2026071903, 2026071904, 2026071905)
STEPS = 80
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.01
MARGIN = 1.0


def unit_rows(values: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.stack([np.asarray(row, dtype=np.float64) for row in values])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(matrix).all():
        raise ValueError("pair deltas must be finite and non-degenerate")
    return matrix / norms


def fit_pair_head(
    pairs: Sequence[dict[str, Any]],
    *,
    seed: int,
    steps: int = STEPS,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pair head needs training pairs")
    sources = sorted({row["parent_source_id"] for row in pairs})
    rng = np.random.default_rng(seed)
    sampled_sources = rng.choice(sources, size=len(sources), replace=True).tolist()
    sampled = [row for source in sampled_sources for row in pairs if row["parent_source_id"] == source]
    all_unit = unit_rows([row["delta"] for row in pairs])
    sampled_unit = torch.from_numpy(unit_rows([row["delta"] for row in sampled]).astype(np.float32))
    initial = all_unit.mean(axis=0)
    initial /= np.linalg.norm(initial)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    weight = torch.nn.Parameter(torch.from_numpy(initial.astype(np.float32)))
    optimizer = torch.optim.AdamW([weight], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        projection = sampled_unit @ weight
        loss = torch.nn.functional.softplus(MARGIN - projection).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    direction = weight.detach().cpu().numpy().astype(np.float64)
    direction /= np.linalg.norm(direction)
    digest = hashlib.sha256(np.asarray(direction, dtype="<f8").tobytes()).hexdigest()
    return {
        "direction": direction,
        "sampled_source_ids": sampled_sources,
        "sampled_unique_source_count": len(set(sampled_sources)),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "coefficient_sha256": digest,
    }


def evaluate_seed(
    real_pairs: Sequence[dict[str, Any]],
    auxiliary_pairs: Sequence[dict[str, Any]],
    rice_open_delta: np.ndarray,
    rice_close_delta: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold_index, held in enumerate(real_pairs):
        source = held["parent_source_id"]
        training = [row for row in list(real_pairs) + list(auxiliary_pairs) if row["parent_source_id"] != source]
        fitted = fit_pair_head(training, seed=seed + fold_index * 1009)
        projection = float(np.asarray(held["delta"]) @ fitted["direction"])
        folds.append({
            "held_out_pair_id": held["pair_id"],
            "held_out_parent_source_id": source,
            "held_out_source_descendants_excluded": all(row["parent_source_id"] != source for row in training),
            "projection": projection,
            "ordered": projection > 0.0,
            "bootstrap_sampled_source_ids": fitted["sampled_source_ids"],
            "bootstrap_unique_source_count": fitted["sampled_unique_source_count"],
            "coefficient_sha256": fitted["coefficient_sha256"],
            "loss_first_last": [fitted["initial_loss"], fitted["final_loss"]],
        })
    final = fit_pair_head(list(real_pairs) + list(auxiliary_pairs), seed=seed + 7919)
    open_projection = float(np.asarray(rice_open_delta) @ final["direction"])
    close_projection = float(np.asarray(rice_close_delta) @ final["direction"])
    passed = bool(all(row["ordered"] and row["held_out_source_descendants_excluded"] for row in folds) and open_projection > 0.0 and close_projection > 0.0)
    return {
        "seed": seed,
        "source_isolated_real_pair_folds": folds,
        "real_pair_ordering_rate": sum(row["ordered"] for row in folds) / len(folds),
        "rice_external_pressure": {"open_projection": open_projection, "close_projection": close_projection, "open_ordered": open_projection > 0.0, "close_ordered": close_projection > 0.0, "used_for_training": False},
        "run_gate_passed": passed,
        "final_coefficient_sha256": final["coefficient_sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.package_root, args.mechanism_report, args.dataset_root, args.synthetic_audit, args.inverse_dataset_root, args.inverse_audit, args.model_dir, args.rice_video, args.rice_review, args.feature_gate_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    feature_gate = lifecycle.verify_json_sidecar(args.feature_gate_report.resolve())
    if feature_gate.get("fixed_teacher_gate", {}).get("passed") is not True:
        raise ValueError("frozen DINOv2 feature gate did not pass")
    for audit_path in (args.synthetic_audit, args.inverse_audit):
        if lifecycle.verify_json_sidecar(audit_path.resolve()).get("parent_source_isolated_representation_short_run_authorized") is not True:
            raise ValueError("auxiliary audit did not authorize short runs")

    teacher = dino.FrozenDinoV2(args.model_dir.resolve())
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
    forward_pairs = dino.load_manifest_pairs(teacher, args.dataset_root.resolve(), kind="synthetic", batch_size=args.batch_size)
    inverse_pairs = dino.load_manifest_pairs(teacher, args.inverse_dataset_root.resolve(), kind="inverse", batch_size=args.batch_size)
    auxiliary_pairs = forward_pairs + inverse_pairs

    review = common.load_json(args.rice_review).get("review") or {}
    rice: dict[str, np.ndarray] = {}
    for name, field in {"pre_clear": "pre_risk_clear_window_ms", "risk": "risk_present_window_ms", "post_clear": "stable_post_clear_window_ms"}.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000)
        rice[name] = teacher.extract(images, batch_size=args.batch_size).mean(axis=0)
    rice_open_delta = rice["risk"] - rice["pre_clear"]
    rice_close_delta = rice["risk"] - rice["post_clear"]
    runs = [evaluate_seed(real_pairs, auxiliary_pairs, rice_open_delta, rice_close_delta, seed=seed) for seed in SEEDS]
    passing = sum(row["run_gate_passed"] for row in runs)
    stable = passing >= 4 and min(row["real_pair_ordering_rate"] for row in runs) == 1.0
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "five_fixed_source_bootstrap_pair_ranking_short_runs",
        "inputs": {
            "feature_gate_report_sha256": common.sha256_file(args.feature_gate_report),
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "forward_audit_sha256": common.sha256_file(args.synthetic_audit),
            "inverse_audit_sha256": common.sha256_file(args.inverse_audit),
            "model_weights_sha256": common.sha256_file(args.model_dir / "model.safetensors"),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
        },
        "training_pair_counts": {"real": len(real_pairs), "forward_synthetic": len(forward_pairs), "inverse_real_risk": len(inverse_pairs)},
        "head_contract": {
            "initialization": "unit mean of all fold-training pair deltas", "bootstrap": "parent sources sampled with replacement",
            "loss": "softplus(pair_margin - unit_delta_dot_direction)", "steps": STEPS, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "pair_margin": MARGIN, "seeds": list(SEEDS), "saved_weights": False,
            "backbone_trainable_parameters": 0,
        },
        "runs": runs,
        "summary": {"run_count": len(runs), "runs_passing_complete_gate": passing, "real_pair_ordering_rates": [row["real_pair_ordering_rate"] for row in runs], "rice_open_passes": [row["rice_external_pressure"]["open_ordered"] for row in runs], "rice_close_passes": [row["rice_external_pressure"]["close_ordered"] for row in runs]},
        "bootstrap_stability_gate": {"passed": stable, "requirements": {"at_least_4_of_5_complete_runs": True, "every_run_orders_all_real_pairs": True}},
        "prospective_contract_authorized": stable,
        "evidence_limit": "Five retrospective source-bootstrap short runs on GPT/VLM provisional train-only pairs and the already-seen Rice source. Stability can authorize freezing a new prospective contract only; it is not prospective, blind, calibration or deployment evidence.",
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
    parser.add_argument("--feature-gate-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "runs_passing": payload["summary"]["runs_passing_complete_gate"], "stable": payload["bootstrap_stability_gate"]["passed"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
