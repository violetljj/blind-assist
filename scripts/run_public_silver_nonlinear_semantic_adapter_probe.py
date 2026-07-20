#!/usr/bin/env python3
"""Run one deterministic source-isolated nonlinear semantic-adapter short run.

The ADE20K SegFormer teacher remains frozen. A small positive-evidence bank is
trained on fixed 450-D regional semantic vectors with pair-ranking loss. The
readout is the unweighted mean of softplus evidence units, so this experiment
changes the representation rather than fitting another free endpoint head.
Synthetic descendants of every held-out real parent source are excluded.
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

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_background_normalized_static_probe as background
import run_public_silver_frozen_feature_probe as common
import run_public_silver_full_semantic_pair_adapter_probe as semantic
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_segformer_free_space_probe as clearance


SCHEMA = "blindassist_public_silver_nonlinear_semantic_adapter_probe_v1"
SEED = 0
EVIDENCE_UNITS = 4
STEPS = 300
LEARNING_RATE = 0.02
WEIGHT_DECAY = 0.01
MARGIN = 1.0


class PositiveEvidenceBank(torch.nn.Module):
    def __init__(self, feature_dim: int, units: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(feature_dim, units)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.projection(values)).mean(dim=-1)


def fit_adapter(
    pairs: Sequence[dict[str, Any]],
    *,
    seed: int = SEED,
    steps: int = STEPS,
) -> tuple[PositiveEvidenceBank, np.ndarray, np.ndarray, list[float]]:
    if not pairs:
        raise ValueError("adapter needs at least one pair")
    clear = np.stack([np.asarray(row["clear"], dtype=np.float32) for row in pairs])
    risk = np.stack([np.asarray(row["risk"], dtype=np.float32) for row in pairs])
    if clear.shape != risk.shape or clear.ndim != 2 or not np.isfinite(clear).all() or not np.isfinite(risk).all():
        raise ValueError("adapter pairs must be finite and shape-compatible")
    endpoints = np.concatenate([clear, risk], axis=0)
    mean = endpoints.mean(axis=0)
    scale = endpoints.std(axis=0)
    scale[scale < 1e-4] = 1.0
    clear_z = torch.from_numpy((clear - mean) / scale)
    risk_z = torch.from_numpy((risk - mean) / scale)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = PositiveEvidenceBank(clear.shape[1], EVIDENCE_UNITS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = model(risk_z) - model(clear_z)
        loss = torch.nn.functional.softplus(MARGIN - delta).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, mean, scale, losses


def score(model: PositiveEvidenceBank, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    vector = torch.from_numpy(((np.asarray(values, dtype=np.float32) - mean) / scale).reshape(1, -1))
    with torch.inference_mode():
        return float(model(vector)[0])


def evaluate_real_folds(
    real_pairs: Sequence[dict[str, Any]],
    auxiliary_pairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held in real_pairs:
        held_source = held["parent_source_id"]
        training = [row for row in list(real_pairs) + list(auxiliary_pairs) if row["parent_source_id"] != held_source]
        model, mean, scale, losses = fit_adapter(training)
        projection = score(model, held["risk"], mean, scale) - score(model, held["clear"], mean, scale)
        rows.append({
            "held_out_pair_id": held["pair_id"],
            "held_out_parent_source_id": held_source,
            "training_real_pair_ids": [row["pair_id"] for row in training if row["kind"] == "real"],
            "training_synthetic_pair_ids": [row["pair_id"] for row in training if row["kind"] == "synthetic"],
            "training_inverse_pair_ids": [row["pair_id"] for row in training if row["kind"] == "inverse"],
            "held_out_source_descendants_excluded": all(row["parent_source_id"] != held_source for row in training),
            "projection": projection,
            "ordered": projection > 0.0,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        })
    return rows


def load_manifest_pairs(
    teacher: clearance.FrozenTeacher,
    dataset_root: Path,
    *,
    kind: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    manifest = [json.loads(line) for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], []).append(row)
    pairs: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        clear_row = next(row for row in members if row["attributes"]["risk_state"] == "clear")
        risk_row = next(row for row in members if row["attributes"]["risk_state"] == "risk")
        images = [cv2.imread(str(dataset_root / row["image_path"]), cv2.IMREAD_COLOR) for row in (clear_row, risk_row)]
        if any(image is None for image in images):
            raise ValueError(f"cannot decode {kind} pair: {pair_id}")
        vectors = semantic.extract_vectors(teacher, images, batch_size=batch_size)
        pairs.append({"kind": kind, "pair_id": pair_id, "parent_source_id": risk_row["source"]["parent_source_id"], "clear": vectors[0], "risk": vectors[1]})
    return pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (args.inverse_dataset_root is None) != (args.inverse_audit is None):
        raise ValueError("inverse dataset root and audit must be supplied together")
    paths = [args.package_root, args.mechanism_report, args.dataset_root, args.synthetic_audit, args.model_dir, args.rice_video, args.rice_review, args.output]
    if args.inverse_dataset_root is not None:
        paths.extend([args.inverse_dataset_root, args.inverse_audit])
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    audit = lifecycle.verify_json_sidecar(args.synthetic_audit.resolve())
    if audit.get("parent_source_isolated_representation_short_run_authorized") is not True:
        raise ValueError("synthetic audit did not authorize the short run")
    inverse_audit: dict[str, Any] | None = None
    if args.inverse_audit is not None:
        inverse_audit = lifecycle.verify_json_sidecar(args.inverse_audit.resolve())
        if inverse_audit.get("parent_source_isolated_representation_short_run_authorized") is not True:
            raise ValueError("inverse audit did not authorize the short run")

    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    teacher = clearance.FrozenTeacher(args.model_dir.resolve())
    real_pairs: list[dict[str, Any]] = []
    for pair_id in sorted(set(qualified[temporal.STATIC])):
        members = [row for row in episodes if row.get("counterfactual_pair_id") == pair_id]
        clear = next(row for row in members if int(row["label"]) == 0)
        risk = next(row for row in members if int(row["label"]) == 1)
        clear_vector = semantic.extract_vectors(teacher, background._episode_images(clear), batch_size=args.batch_size).mean(axis=0)
        risk_vector = semantic.extract_vectors(teacher, background._episode_images(risk), batch_size=args.batch_size).mean(axis=0)
        real_pairs.append({"kind": "real", "pair_id": pair_id, "parent_source_id": clear["source_id"], "clear": clear_vector, "risk": risk_vector})

    dataset_root = args.dataset_root.resolve()
    synthetic_pairs = load_manifest_pairs(teacher, dataset_root, kind="synthetic", batch_size=args.batch_size)
    inverse_pairs: list[dict[str, Any]] = []
    if args.inverse_dataset_root is not None:
        inverse_pairs = load_manifest_pairs(teacher, args.inverse_dataset_root.resolve(), kind="inverse", batch_size=args.batch_size)

    auxiliary_pairs = synthetic_pairs + inverse_pairs
    folds = evaluate_real_folds(real_pairs, auxiliary_pairs)
    ordering_rate = sum(row["ordered"] for row in folds) / len(folds)
    final_model, final_mean, final_scale, final_losses = fit_adapter(real_pairs + auxiliary_pairs)
    review = common.load_json(args.rice_review).get("review") or {}
    rice_vectors: dict[str, np.ndarray] = {}
    for name, field in {"pre_clear": "pre_risk_clear_window_ms", "risk": "risk_present_window_ms", "post_clear": "stable_post_clear_window_ms"}.items():
        window = review.get(field)
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = background.decode_video_window(args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000)
        rice_vectors[name] = semantic.extract_vectors(teacher, images, batch_size=args.batch_size).mean(axis=0)
    rice_risk = score(final_model, rice_vectors["risk"], final_mean, final_scale)
    open_projection = rice_risk - score(final_model, rice_vectors["pre_clear"], final_mean, final_scale)
    close_projection = rice_risk - score(final_model, rice_vectors["post_clear"], final_mean, final_scale)
    gate = bool(ordering_rate == 1.0 and open_projection > 0.0 and close_projection > 0.0 and all(row["held_out_source_descendants_excluded"] for row in folds))
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "single_deterministic_representation_adapter_short_run",
        "inputs": {
            "mechanism_report_sha256": common.sha256_file(args.mechanism_report),
            "synthetic_audit_sha256": common.sha256_file(args.synthetic_audit),
            "manifest_sha256": common.sha256_file(dataset_root / "manifest.jsonl"),
            "inverse_audit_sha256": common.sha256_file(args.inverse_audit) if args.inverse_audit is not None else None,
            "inverse_manifest_sha256": common.sha256_file(args.inverse_dataset_root / "manifest.jsonl") if args.inverse_dataset_root is not None else None,
            "model_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
            "rice_video_sha256": common.sha256_file(args.rice_video),
            "rice_review_sha256": common.sha256_file(args.rice_review),
        },
        "adapter_contract": {
            "seed": SEED, "feature_dimension": int(final_mean.size), "positive_evidence_units": EVIDENCE_UNITS,
            "steps": STEPS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "pair_margin": MARGIN,
            "readout": "fixed unweighted mean of softplus evidence units", "hyperparameter_sweep": False,
            "teacher_trainable_parameters": 0, "saved_weights": False,
        },
        "source_isolated_real_pair_folds": folds,
        "training_pair_counts": {"real": len(real_pairs), "forward_synthetic": len(synthetic_pairs), "inverse_real_risk": len(inverse_pairs)},
        "real_pair_ordering_rate": ordering_rate,
        "full_training_loss": {"initial": final_losses[0], "final": final_losses[-1]},
        "rice_external_pressure": {"open_projection": open_projection, "close_projection": close_projection, "open_ordered": open_projection > 0.0, "close_ordered": close_projection > 0.0, "used_for_training": False},
        "single_run_gate": {"passed": gate, "requirements": {"all_real_pairs_ordered": True, "rice_open_and_close_ordered": True, "parent_descendants_excluded": True}},
        "five_seed_bootstrap_authorized": gate,
        "evidence_limit": "One retrospective deterministic adapter run. Even a pass only authorizes five-seed stability testing, not calibration, blind evaluation, Android or production.",
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
    parser.add_argument("--inverse-dataset-root", type=Path)
    parser.add_argument("--inverse-audit", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "real_pair_ordering_rate": payload["real_pair_ordering_rate"], "rice_open_ordered": payload["rice_external_pressure"]["open_ordered"], "rice_close_ordered": payload["rice_external_pressure"]["close_ordered"], "gate_passed": payload["single_run_gate"]["passed"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
