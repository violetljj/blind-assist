#!/usr/bin/env python3
"""Verify source-density-invariant angular-boundary gradient mass before training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from download_b0_arkitscenes_assets import require, sha256_file
from train_ag_st_masked_student import (
    compute_angular_boundary_only_losses,
    tier_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANGULAR_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-continuous-boundary-factors-angular-r0/result.json"
)
DEFAULT_BONN_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-bonn-fit-angular-factor-labels-r0/result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-angular-loss-mass-normalization-canary-r0/result.json"
)
EXPECTED_COUNTS = {"arkitscenes": 48, "tum_rgbd": 21, "bonn_rgbd_fit": 24}


def gradient_summary(
    target_hw: np.ndarray,
    valid_hw: np.ndarray,
    tier_hw: np.ndarray,
    *,
    loss_profile: str,
) -> dict[str, float | bool]:
    target = torch.from_numpy(np.asarray(target_hw, dtype=np.float32))[None, None]
    valid = torch.from_numpy(np.asarray(valid_hw, dtype=np.bool_))[None, None]
    tier = torch.from_numpy(np.asarray(tier_hw, dtype=np.uint8))[None, None]
    require(target.shape == valid.shape == tier.shape, "gradient canary shape drift")
    logits = torch.zeros_like(target, requires_grad=True)
    targets = {
        "boundary_angular_soft": target,
        "boundary_valid": valid,
        "boundary_tier": tier,
    }
    loss = compute_angular_boundary_only_losses(
        {"boundary_logits": logits},
        targets,
        loss_profile=loss_profile,
    )["total"]
    total_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    unknown_max = float(total_gradient[~valid].abs().max()) if bool((~valid).any()) else 0.0
    row: dict[str, float | bool] = {
        "loss": float(loss.detach()),
        "total_gradient_l1": float(total_gradient.abs().sum()),
        "unknown_gradient_max_abs": unknown_max,
        "finite": bool(torch.isfinite(loss) and torch.isfinite(total_gradient).all()),
    }
    if loss_profile != "target_mass_normalized_bce":
        return row

    weights = tier_weights(tier) * valid
    positive_mass = torch.sum(weights * target)
    negative_mass = torch.sum(weights * (1.0 - target))
    has_positive = bool(positive_mass > 0.0)
    has_negative = bool(negative_mass > 0.0)
    require(has_positive or has_negative, "gradient canary supervision mass empty")
    active_classes = float(int(has_positive) + int(has_negative))
    positive = torch.sum(weights * target * F.softplus(-logits))
    negative = torch.sum(weights * (1.0 - target) * F.softplus(logits))
    positive_term = (
        positive / positive_mass.clamp_min(1e-12) / active_classes
        if has_positive
        else positive * 0.0
    )
    negative_term = (
        negative / negative_mass.clamp_min(1e-12) / active_classes
        if has_negative
        else negative * 0.0
    )
    positive_gradient = torch.autograd.grad(positive_term, logits, retain_graph=True)[0]
    negative_gradient = torch.autograd.grad(negative_term, logits)[0]
    positive_gradient_mass = float(positive_gradient.abs().sum())
    negative_gradient_mass = float(negative_gradient.abs().sum())
    component_total = positive_gradient_mass + negative_gradient_mass
    reconstruction_error = float(
        (total_gradient - positive_gradient - negative_gradient).abs().max()
    )
    row.update(
        {
            "positive_target_mass": float(positive_mass),
            "negative_target_mass": float(negative_mass),
            "has_positive_target_mass": has_positive,
            "has_negative_target_mass": has_negative,
            "positive_component_gradient_mass": positive_gradient_mass,
            "negative_component_gradient_mass": negative_gradient_mass,
            "component_gradient_mass_total": component_total,
            "dual_class_component_balance_ratio": (
                min(positive_gradient_mass, negative_gradient_mass)
                / max(positive_gradient_mass, negative_gradient_mass, 1e-12)
                if has_positive and has_negative
                else 1.0
            ),
            "gradient_reconstruction_max_abs_error": reconstruction_error,
        }
    )
    return row


def _source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["normalized"] for row in rows]
    legacy = [row["legacy"] for row in rows]
    dual = [row for row in values if row["has_positive_target_mass"] and row["has_negative_target_mass"]]
    return {
        "frame_count": len(rows),
        "dual_class_frame_count": len(dual),
        "single_class_frame_count": len(rows) - len(dual),
        "target_mass_sum": float(sum(row["positive_target_mass"] for row in values)),
        "normalized_total_gradient_l1_mean": float(
            np.mean([row["total_gradient_l1"] for row in values])
        ),
        "normalized_component_gradient_mass_total_min": float(
            min(row["component_gradient_mass_total"] for row in values)
        ),
        "normalized_component_gradient_mass_total_max": float(
            max(row["component_gradient_mass_total"] for row in values)
        ),
        "normalized_dual_class_balance_ratio_min": float(
            min((row["dual_class_component_balance_ratio"] for row in dual), default=1.0)
        ),
        "legacy_total_gradient_l1_mean": float(
            np.mean([row["total_gradient_l1"] for row in legacy])
        ),
    }


def run(angular_result: Path, bonn_result: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"angular loss canary output exists: {output}")
    inputs = []
    frames: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result_path in (angular_result, bonn_result):
        require(result_path.is_file(), f"angular factor result missing: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        inputs.append(
            {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
                "schema": result.get("schema"),
                "status": result.get("status"),
            }
        )
        for descriptor in result["frames"]:
            source = str(descriptor["source"])
            if source not in EXPECTED_COUNTS:
                continue
            factor_path = Path(descriptor["output"])
            require(factor_path.is_file(), f"angular factor payload missing: {factor_path}")
            with np.load(factor_path, allow_pickle=False) as payload:
                target = payload["boundary_angular_soft_probability_hw"]
                valid = payload["boundary_truth_valid_hw"]
                tier = payload["boundary_quality_tier_hw"]
            row = {
                "source": source,
                "parent_id": str(descriptor["parent_id"]),
                "frame_id": str(descriptor["frame_id"]),
                "factor_path": str(factor_path),
                "factor_sha256": descriptor["output_sha256"],
                "shape_hw": list(target.shape),
                "normalized": gradient_summary(
                    target,
                    valid,
                    tier,
                    loss_profile="target_mass_normalized_bce",
                ),
                "legacy": gradient_summary(
                    target,
                    valid,
                    tier,
                    loss_profile="legacy_heat_dice",
                ),
            }
            frames.append(row)
            by_source[source].append(row)

    source_summary = {
        source: _source_summary(by_source[source]) for source in sorted(by_source)
    }
    normalized = [row["normalized"] for row in frames]
    dual = [row for row in normalized if row["has_positive_target_mass"] and row["has_negative_target_mass"]]
    gates = {
        "source_frame_counts_exact": {
            source: len(by_source[source]) for source in sorted(by_source)
        }
        == EXPECTED_COUNTS,
        "all_93_frames_consumed": len(frames) == 93,
        "all_gradients_finite": all(row["finite"] for row in normalized),
        "unknown_gradient_exactly_zero": max(
            row["unknown_gradient_max_abs"] for row in normalized
        )
        == 0.0,
        "normalized_components_reconstruct_loss_gradient": max(
            row["gradient_reconstruction_max_abs_error"] for row in normalized
        )
        <= 1e-8,
        "dual_class_components_balanced": min(
            row["dual_class_component_balance_ratio"] for row in dual
        )
        >= 0.99999,
        "active_component_gradient_mass_is_half": max(
            abs(row["component_gradient_mass_total"] - 0.5) for row in normalized
        )
        <= 1e-6,
        "single_class_fallback_bounded": sum(
            not row["has_positive_target_mass"] or not row["has_negative_target_mass"]
            for row in normalized
        )
        == 2,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_target_mass_normalized_angular_loss_canary_v1",
        "status": (
            "TARGET_MASS_NORMALIZED_ANGULAR_LOSS_CANARY_PASS"
            if passed
            else "TARGET_MASS_NORMALIZED_ANGULAR_LOSS_CANARY_FAIL"
        ),
        "question": "Does target-mass-normalized angular BCE remove source boundary-density scaling while preserving UNKNOWN masking before CUDA training?",
        "training_performed": False,
        "contract": {
            "dual_class": "positive and negative soft-target BCE components each receive one half of objective mass",
            "single_class": "the one active class receives the full normalized objective without inventing the absent class",
            "dice": "disabled for this profile so positive-only overlap pressure cannot reintroduce density scaling",
            "unknown": "boundary_truth_valid=false or tier weight zero contributes exactly zero gradient",
        },
        "inputs": inputs,
        "frame_count": len(frames),
        "source_summary": source_summary,
        "gates": gates,
        "frames": frames,
        "decision": {
            "authorize_cuda_probe": passed,
            "next_execution": (
                "Run the frozen R20 trisource probe with angular_loss_profile=target_mass_normalized_bce."
                if passed
                else "Do not train; repair the failed gradient invariant first."
            ),
        },
        "claim_boundary": "Loss-mechanics and existing-label coverage evidence only; no learned quality, external generalization, task utility, safety, deployment, or product claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angular-result", type=Path, default=DEFAULT_ANGULAR_RESULT)
    parser.add_argument("--bonn-result", type=Path, default=DEFAULT_BONN_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.angular_result.resolve(),
        args.bonn_result.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_summary": result["source_summary"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
