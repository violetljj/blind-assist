#!/usr/bin/env python3
"""Train a frozen-expert selector that falls back to metric DepthART by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    build_students,
    checkpoint_architecture,
    extract_rgb_only_feature_with_intrinsics,
)
from train_ag_st_bonn_anchored_student import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    DEFAULT_COHORT_MANIFEST,
    DEFAULT_LABEL_DIRS,
    DEFAULT_STAGE0A_RESULTS,
    _unknown_factor_targets,
    extract_bonn_anchor_frames,
)
from ag_st_tum_rgbd import (
    DEFAULT_TUM_COHORT_MANIFEST,
    load_tum_role_payloads,
)
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    CachedFrame,
    FrameDescriptor,
    aggregate_label_digest,
    build_frame_descriptor_batches,
    extract_depthart_features,
    masked_weighted_mean,
    move_targets,
    save_checkpoint_exclusive,
    tier_weights,
    write_json_exclusive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTOR_SCHEMA = "blindassist_ag_st_no_regret_depth_selector_checkpoint_v1"
SELECTOR_SPLIT_TOKEN = "AG_ST_NO_REGRET_SELECTOR_R0"
SELECTOR_INITIAL_PROBABILITY = 0.10
DEFAULT_EXPERT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-bonn-anchored-identity-gated-student-pilot10-r0/masked-factor-head.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-two-domain-r0"
)
DEFAULT_THRESHOLD_CANDIDATES = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
    0.995,
    0.999,
)


class NoRegretDepthSelector(nn.Module):
    """Predict whether a frozen correction expert beats the frozen metric base."""

    def __init__(
        self,
        feature_channels: int = 48,
        hidden: int = 32,
        *,
        global_context_profile: str = "none",
    ) -> None:
        super().__init__()
        require(hidden % 8 == 0, "selector hidden channels must be divisible by 8")
        require(
            global_context_profile in {"none", "mean_std"},
            "selector global context profile invalid",
        )
        # Guidance: log(base), signed/absolute expert correction, expert gate.
        self.feature_channels = int(feature_channels)
        self.hidden = int(hidden)
        self.global_context_profile = global_context_profile
        self.trunk = nn.Sequential(
            nn.Conv2d(feature_channels + 4, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.selector_logits = nn.Conv2d(hidden, 1, 1)
        self.global_context = (
            nn.Sequential(
                nn.Linear(2 * (feature_channels + 4), hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            if global_context_profile == "mean_std"
            else None
        )
        nn.init.zeros_(self.selector_logits.weight)
        nn.init.constant_(
            self.selector_logits.bias,
            math.log(
                SELECTOR_INITIAL_PROBABILITY
                / (1.0 - SELECTOR_INITIAL_PROBABILITY)
            ),
        )

    def forward(
        self,
        feature: torch.Tensor,
        base_depth_m: torch.Tensor,
        expert_depth_m: torch.Tensor,
        expert_gate: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        feature_hw = feature.shape[-2:]
        base = base_depth_m.clamp(0.05, 20.0)
        expert = expert_depth_m.clamp(0.05, 20.0)
        signed_correction = expert.log() - base.log()

        def downsample(value: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                value,
                feature_hw,
                mode="bilinear",
                align_corners=False,
            )

        guidance = torch.cat(
            [
                downsample(base.log()),
                downsample(signed_correction),
                downsample(signed_correction.abs()),
                downsample(expert_gate),
            ],
            dim=1,
        )
        combined = torch.cat([feature, guidance], dim=1)
        latent = self.trunk(combined)
        if self.global_context is not None:
            mean = combined.mean(dim=(-2, -1))
            std = combined.var(dim=(-2, -1), unbiased=False).clamp_min(1e-8).sqrt()
            context = self.global_context(torch.cat([mean, std], dim=1))
            latent = latent + context[:, :, None, None]
        logits = F.interpolate(
            self.selector_logits(latent),
            output_hw,
            mode="bilinear",
            align_corners=False,
        )
        return {
            "selector_logits": logits,
            "selector_probability": torch.sigmoid(logits),
        }


def extract_tum_anchor_frames(
    cohort_manifest: Path,
    depthart_source: Path,
    depthart_checkpoint: Path,
    device: torch.device,
    seed: int,
    *,
    cohort_role: str = "fit",
) -> tuple[list[CachedFrame], dict[str, Any]]:
    """Materialize TUM RGB/K features and source-native depth-only targets."""

    from train_ag_st_masked_student import load_depthart_backbone

    payloads, provenance = load_tum_role_payloads(cohort_manifest, cohort_role)
    extractor, scan = load_depthart_backbone(
        depthart_source,
        depthart_checkpoint,
        device,
        seed,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    started = time.perf_counter()
    frames: list[CachedFrame] = []
    frame_receipts: list[dict[str, Any]] = []
    for payload in payloads:
        rgb = payload.load_rgb()
        feature, base_depth = extract_rgb_only_feature_with_intrinsics(
            extractor,
            rgb,
            payload.intrinsics,
            "shared",
            device,
            amp_dtype,
        )
        depth_m, depth_valid = payload.load_depth()
        descriptor = FrameDescriptor(
            parent_id=payload.parent_id,
            frame_index=payload.rgb.row_index,
            frame_stem=f"{payload.parent_id}:{payload.rgb.row_index}",
            output_hw=depth_m.shape,
            label_path=payload.source_path,
            video={"source": "TUM_RGBD"},
        )
        frames.append(
            CachedFrame(
                descriptor=descriptor,
                feature=feature[0].to(dtype=torch.float16, device="cpu"),
                base_depth_m=base_depth[0].float().cpu(),
                targets=_unknown_factor_targets(depth_m, depth_valid),
            )
        )
        frame_receipts.append(
            {
                "parent_id": payload.parent_id,
                "rgb_row_index_zero_based": payload.rgb.row_index,
                "depth_row_index_zero_based": payload.depth.row_index,
                "rgb_relative_path": payload.rgb.relative_path,
                "depth_relative_path": payload.depth.relative_path,
                "rgb_depth_delta_seconds": payload.association_delta_seconds,
                "depth_valid_fraction": float(depth_valid.mean()),
                "intrinsics_fx_fy_cx_cy": [
                    float(payload.intrinsics[0, 0]),
                    float(payload.intrinsics[1, 1]),
                    float(payload.intrinsics[0, 2]),
                    float(payload.intrinsics[1, 2]),
                ],
            }
        )
    del extractor
    torch.cuda.empty_cache()
    return frames, {
        **provenance,
        "feature_profile": "shared",
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "elapsed_seconds": time.perf_counter() - started,
        "scan_backend": scan,
        "frame_receipts": frame_receipts,
    }


@dataclass(frozen=True)
class SelectorObservation:
    parent_id: str
    domain: str
    truth_depth_m: np.ndarray
    valid: np.ndarray
    base_depth_m: np.ndarray
    expert_depth_m: np.ndarray
    selector_probability: np.ndarray


def split_parent_roles(
    parent_ids: Iterable[str],
    *,
    calibration_count: int,
    domain: str,
    token: str = SELECTOR_SPLIT_TOKEN,
) -> tuple[list[str], list[str]]:
    unique = sorted(set(str(value) for value in parent_ids))
    require(
        0 < calibration_count < len(unique),
        "selector calibration parent count invalid",
    )
    ranked = sorted(
        unique,
        key=lambda parent: hashlib.sha256(
            f"{token}|{domain}|{parent}".encode("utf-8")
        ).hexdigest(),
    )
    calibration = sorted(ranked[:calibration_count])
    fit = sorted(ranked[calibration_count:])
    require(not (set(fit) & set(calibration)), "selector split overlap")
    return fit, calibration


def calibration_parent_count(parent_count: int, *, minimum: int = 1) -> int:
    """Reserve one quarter of a source domain without collapsing small cohorts."""

    require(parent_count >= 4, "selector domain needs at least four parents")
    require(minimum > 0, "selector calibration minimum invalid")
    count = max(minimum, parent_count // 4)
    require(count < parent_count, "selector calibration count consumes domain")
    return count


def compute_no_regret_selector_loss(
    outputs: dict[str, torch.Tensor],
    base_depth_m: torch.Tensor,
    expert_depth_m: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    decision_margin_m: float = 0.02,
    harmful_example_multiplier: float = 4.0,
) -> dict[str, torch.Tensor]:
    require(decision_margin_m > 0.0, "selector decision margin invalid")
    truth = targets["metric_depth_m"].clamp(0.05, 20.0)
    base = base_depth_m.clamp(0.05, 20.0)
    expert = expert_depth_m.clamp(0.05, 20.0)
    base_error = (base - truth).abs()
    expert_error = (expert - truth).abs()
    advantage = base_error - expert_error
    valid = (
        targets["metric_valid"].bool()
        & torch.isfinite(truth)
        & torch.isfinite(base)
        & torch.isfinite(expert)
        & (advantage.abs() >= decision_margin_m)
    )
    provenance_weight = tier_weights(targets["metric_tier"])
    target = (advantage > decision_margin_m).to(dtype=base.dtype)
    consequence = (advantage.abs() / 0.10).clamp(0.25, 4.0)
    harm_weight = torch.where(
        target >= 0.5,
        torch.ones_like(target),
        torch.full_like(target, harmful_example_multiplier),
    )
    weights = provenance_weight * consequence * harm_weight
    logits = outputs["selector_logits"]
    probability = outputs["selector_probability"]
    bce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = masked_weighted_mean(bce_raw, valid, weights)
    soft_regret = masked_weighted_mean(
        probability * (expert_error - base_error),
        valid,
        provenance_weight,
    )
    selected_probability = masked_weighted_mean(
        probability,
        valid,
        provenance_weight,
    )
    # BCE learns the ordering; regret and sparse fallback resolve close scores.
    total = bce + 2.0 * soft_regret + 0.01 * selected_probability
    return {
        "total": total,
        "raw/bce": bce,
        "raw/soft_regret_m": soft_regret,
        "raw/mean_selection_probability": selected_probability,
        "raw/decisive_fraction": valid.float().mean(),
        "raw/beneficial_fraction_of_decisive": masked_weighted_mean(
            target,
            valid,
            provenance_weight,
        ),
    }


def _expert_outputs(
    expert: nn.Module,
    feature: torch.Tensor,
    base_depth_m: torch.Tensor,
    output_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = expert(feature, base_depth_m, output_hw)
    require("depth_identity_gate" in outputs, "expert lacks identity gate")
    return outputs


def train_selector(
    selector: NoRegretDepthSelector,
    expert: nn.Module,
    frames: list[CachedFrame],
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    require(frames and epochs > 0 and learning_rate > 0.0, "selector training invalid")
    selector.train()
    expert.eval()
    expert.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        losses_by_name: dict[str, list[float]] = defaultdict(list)
        gradients: list[float] = []
        for index in rng.permutation(len(frames)):
            frame = frames[int(index)]
            flip = bool(rng.random() < 0.5)
            feature = frame.feature[None].float().to(device)
            base_depth = frame.base_depth_m[None].to(device)
            targets = move_targets(frame.targets, device, flip=flip)
            if flip:
                feature = torch.flip(feature, dims=(-1,))
                base_depth = torch.flip(base_depth, dims=(-1,))
            expert_outputs = _expert_outputs(
                expert,
                feature,
                base_depth,
                frame.descriptor.output_hw,
            )
            outputs = selector(
                feature,
                base_depth,
                expert_outputs["depth_m"],
                expert_outputs["depth_identity_gate"],
                frame.descriptor.output_hw,
            )
            losses = compute_no_regret_selector_loss(
                outputs,
                base_depth,
                expert_outputs["depth_m"],
                targets,
            )
            require(bool(torch.isfinite(losses["total"]).item()), "non-finite selector loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0))
            require(math.isfinite(gradient), "non-finite selector gradient")
            optimizer.step()
            gradients.append(gradient)
            for name, value in losses.items():
                losses_by_name[name].append(float(value.detach()))
        scheduler.step()
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 5 == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "mean_gradient_norm": float(np.mean(gradients)),
                    **{
                        f"mean_{name.replace('/', '_')}": float(np.mean(values))
                        for name, values in losses_by_name.items()
                    },
                }
            )
    return history, {
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": epochs * len(frames),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def train_selector_group_dro(
    selector: NoRegretDepthSelector,
    expert: nn.Module,
    frames_by_domain: dict[str, list[CachedFrame]],
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    smooth_max_temperature: float = 0.25,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Optimize a smooth worst-domain objective without exposing domain IDs to the model."""

    require(
        len(frames_by_domain) >= 2
        and all(frames for frames in frames_by_domain.values()),
        "group-DRO domain roster invalid",
    )
    require(
        epochs > 0 and learning_rate > 0.0 and smooth_max_temperature > 0.0,
        "group-DRO configuration invalid",
    )
    domains = sorted(frames_by_domain)
    steps_per_epoch = max(len(frames) for frames in frames_by_domain.values())
    selector.train()
    expert.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        orders = {
            domain: rng.permutation(len(frames_by_domain[domain])) for domain in domains
        }
        aggregate_values: list[float] = []
        gradient_values: list[float] = []
        domain_values: dict[str, list[float]] = defaultdict(list)
        for step in range(steps_per_epoch):
            domain_losses: list[torch.Tensor] = []
            for domain in domains:
                rows = frames_by_domain[domain]
                if step > 0 and step % len(rows) == 0:
                    orders[domain] = rng.permutation(len(rows))
                frame = rows[int(orders[domain][step % len(rows)])]
                flip = bool(rng.random() < 0.5)
                feature = frame.feature[None].float().to(device)
                base_depth = frame.base_depth_m[None].to(device)
                targets = move_targets(frame.targets, device, flip=flip)
                if flip:
                    feature = torch.flip(feature, dims=(-1,))
                    base_depth = torch.flip(base_depth, dims=(-1,))
                expert_outputs = _expert_outputs(
                    expert,
                    feature,
                    base_depth,
                    frame.descriptor.output_hw,
                )
                outputs = selector(
                    feature,
                    base_depth,
                    expert_outputs["depth_m"],
                    expert_outputs["depth_identity_gate"],
                    frame.descriptor.output_hw,
                )
                losses = compute_no_regret_selector_loss(
                    outputs,
                    base_depth,
                    expert_outputs["depth_m"],
                    targets,
                )
                require(
                    bool(torch.isfinite(losses["total"]).item()),
                    f"non-finite group-DRO selector loss: {domain}",
                )
                domain_losses.append(losses["total"])
                domain_values[domain].append(float(losses["total"].detach()))
            stacked = torch.stack(domain_losses)
            smooth_worst = smooth_max_temperature * (
                torch.logsumexp(stacked / smooth_max_temperature, dim=0)
                - math.log(len(domains))
            )
            aggregate = 0.25 * stacked.mean() + 0.75 * smooth_worst
            optimizer.zero_grad(set_to_none=True)
            aggregate.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0))
            require(math.isfinite(gradient), "non-finite group-DRO selector gradient")
            optimizer.step()
            aggregate_values.append(float(aggregate.detach()))
            gradient_values.append(gradient)
        scheduler.step()
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 5 == 0:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "mean_group_dro_total": float(np.mean(aggregate_values)),
                    "mean_gradient_norm": float(np.mean(gradient_values)),
                    **{
                        f"mean_domain_{domain.lower()}_loss": float(np.mean(values))
                        for domain, values in domain_values.items()
                    },
                }
            )
    return history, {
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": epochs * steps_per_epoch,
        "frame_visits": epochs * steps_per_epoch * len(domains),
        "steps_per_epoch": steps_per_epoch,
        "domains_per_step": domains,
        "smooth_max_temperature": smooth_max_temperature,
        "mean_loss_weight": 0.25,
        "smooth_worst_domain_loss_weight": 0.75,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def collect_selector_observations(
    selector: NoRegretDepthSelector,
    expert: nn.Module,
    frames: list[CachedFrame],
    parent_domains: dict[str, str],
    device: torch.device,
) -> list[SelectorObservation]:
    selector.eval()
    expert.eval()
    observations: list[SelectorObservation] = []
    with torch.inference_mode():
        for frame in frames:
            parent = frame.descriptor.parent_id
            feature = frame.feature[None].float().to(device)
            base_depth = frame.base_depth_m[None].to(device)
            expert_outputs = expert(
                feature,
                base_depth,
                frame.descriptor.output_hw,
            )
            outputs = selector(
                feature,
                base_depth,
                expert_outputs["depth_m"],
                expert_outputs["depth_identity_gate"],
                frame.descriptor.output_hw,
            )
            observations.append(
                SelectorObservation(
                    parent_id=parent,
                    domain=parent_domains[parent],
                    truth_depth_m=frame.targets["metric_depth_m"][0, 0].numpy(),
                    valid=frame.targets["metric_valid"][0, 0].numpy().astype(bool),
                    base_depth_m=base_depth[0, 0].float().cpu().numpy(),
                    expert_depth_m=expert_outputs["depth_m"][0, 0].float().cpu().numpy(),
                    selector_probability=outputs["selector_probability"][0, 0]
                    .float()
                    .cpu()
                    .numpy(),
                )
            )
    return observations


def _metric_sums(
    truth: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    error = np.abs(predicted[valid] - truth[valid]).astype(np.float64)
    return {
        "abs_sum": float(error.sum()),
        "bad_sum": float((error > 0.10).sum()),
        "count": float(error.size),
    }


def summarize_selector_observations(
    observations: list[SelectorObservation],
    threshold: float,
) -> dict[str, Any]:
    require(observations, "selector observations empty")
    accumulators: dict[str, dict[str, Any]] = {}
    for row in observations:
        values = accumulators.setdefault(
            row.parent_id,
            {
                "domain": row.domain,
                "base": defaultdict(float),
                "expert": defaultdict(float),
                "selected": defaultdict(float),
                "oracle": defaultdict(float),
                "selected_count": 0.0,
                "oracle_selected_count": 0.0,
                "selected_beneficial_count": 0.0,
                "selected_regret_sum": 0.0,
            },
        )
        require(values["domain"] == row.domain, "parent domain drift")
        selected_mask = row.selector_probability >= threshold
        selected_depth = np.where(selected_mask, row.expert_depth_m, row.base_depth_m)
        base_error = np.abs(row.base_depth_m - row.truth_depth_m)
        expert_error = np.abs(row.expert_depth_m - row.truth_depth_m)
        oracle_mask = row.valid & (expert_error < base_error)
        oracle_depth = np.where(oracle_mask, row.expert_depth_m, row.base_depth_m)
        for name, predicted in (
            ("base", row.base_depth_m),
            ("expert", row.expert_depth_m),
            ("selected", selected_depth),
            ("oracle", oracle_depth),
        ):
            sums = _metric_sums(row.truth_depth_m, predicted, row.valid)
            for key, value in sums.items():
                values[name][key] += value
        active = selected_mask & row.valid
        values["selected_count"] += float(active.sum())
        values["oracle_selected_count"] += float(oracle_mask.sum())
        values["selected_beneficial_count"] += float(
            (active & (expert_error < base_error)).sum()
        )
        values["selected_regret_sum"] += float(
            (expert_error[active] - base_error[active]).sum(dtype=np.float64)
        )

    parent_rows: list[dict[str, Any]] = []
    for parent, values in sorted(accumulators.items()):
        count = values["base"]["count"]
        require(count > 0, "selector parent denominator empty")
        selected_count = values["selected_count"]

        def finalize(name: str) -> dict[str, float]:
            metric = values[name]
            return {
                "mae_m": metric["abs_sum"] / metric["count"],
                "bad_gt_0_10_m_fraction": metric["bad_sum"] / metric["count"],
            }

        parent_rows.append(
            {
                "parent_id": parent,
                "domain": values["domain"],
                "base": finalize("base"),
                "expert": finalize("expert"),
                "selected": finalize("selected"),
                "oracle": finalize("oracle"),
                "selected_coverage_fraction": selected_count / count,
                "oracle_coverage_fraction": values["oracle_selected_count"] / count,
                "selected_beneficial_fraction": (
                    values["selected_beneficial_count"] / selected_count
                    if selected_count > 0
                    else None
                ),
                "selected_mean_regret_m": (
                    values["selected_regret_sum"] / selected_count
                    if selected_count > 0
                    else 0.0
                ),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        require(rows, "selector aggregation rows empty")
        coverage = float(np.mean([row["selected_coverage_fraction"] for row in rows]))
        result: dict[str, Any] = {
            name: {
                "mae_m": float(np.mean([row[name]["mae_m"] for row in rows])),
                "bad_gt_0_10_m_fraction": float(
                    np.mean([row[name]["bad_gt_0_10_m_fraction"] for row in rows])
                ),
            }
            for name in ("base", "expert", "selected", "oracle")
        }
        result["selected_coverage_fraction"] = coverage
        result["oracle_coverage_fraction"] = float(
            np.mean([row["oracle_coverage_fraction"] for row in rows])
        )
        result["selected_mae_delta_vs_base_m"] = (
            result["selected"]["mae_m"] - result["base"]["mae_m"]
        )
        result["selected_bad_delta_vs_base"] = (
            result["selected"]["bad_gt_0_10_m_fraction"]
            - result["base"]["bad_gt_0_10_m_fraction"]
        )
        result["oracle_mae_delta_vs_base_m"] = (
            result["oracle"]["mae_m"] - result["base"]["mae_m"]
        )
        result["oracle_bad_delta_vs_base"] = (
            result["oracle"]["bad_gt_0_10_m_fraction"]
            - result["base"]["bad_gt_0_10_m_fraction"]
        )
        result["improved_parent_count"] = sum(
            row["selected"]["mae_m"] < row["base"]["mae_m"] for row in rows
        )
        return result

    domains = sorted(set(row["domain"] for row in parent_rows))
    return {
        "threshold": float(threshold),
        "parent_macro": aggregate(parent_rows),
        "by_domain": {
            domain: aggregate([row for row in parent_rows if row["domain"] == domain])
            for domain in domains
        },
        "per_parent": parent_rows,
    }


def calibrate_selector_threshold(
    observations: list[SelectorObservation],
    *,
    candidates: Iterable[float] = DEFAULT_THRESHOLD_CANDIDATES,
    minimum_coverage: float = 0.01,
    minimum_nonzero_parent_fraction: float = 0.5,
    metric_tolerance: float = 1e-9,
) -> dict[str, Any]:
    require(0.0 <= minimum_coverage < 1.0, "selector minimum coverage invalid")
    require(
        0.0 <= minimum_nonzero_parent_fraction <= 1.0,
        "selector minimum nonzero-parent fraction invalid",
    )
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in candidates)):
        require(0.0 < threshold < 1.0, "selector threshold candidate invalid")
        summary = summarize_selector_observations(observations, threshold)
        overall = summary["parent_macro"]
        domains_no_regret = all(
            values["selected_mae_delta_vs_base_m"] <= metric_tolerance
            and values["selected_bad_delta_vs_base"] <= metric_tolerance
            for values in summary["by_domain"].values()
        )
        parent_rows = summary["per_parent"]
        harmful_parent_count = sum(
            row["selected"]["mae_m"] - row["base"]["mae_m"]
            > metric_tolerance
            or row["selected"]["bad_gt_0_10_m_fraction"]
            - row["base"]["bad_gt_0_10_m_fraction"]
            > metric_tolerance
            for row in parent_rows
        )
        nonzero_parent_count = sum(
            row["selected_coverage_fraction"] > 0.0 for row in parent_rows
        )
        nonzero_parent_fraction = nonzero_parent_count / len(parent_rows)
        admissible = (
            overall["selected_coverage_fraction"] >= minimum_coverage
            and overall["selected_mae_delta_vs_base_m"] <= metric_tolerance
            and overall["selected_bad_delta_vs_base"] <= metric_tolerance
            and domains_no_regret
            and harmful_parent_count == 0
            and nonzero_parent_fraction >= minimum_nonzero_parent_fraction
        )
        compact = {
            "threshold": threshold,
            "admissible": admissible,
            "parent_macro": overall,
            "by_domain": summary["by_domain"],
            "harmful_parent_count": harmful_parent_count,
            "nonzero_parent_count": nonzero_parent_count,
            "nonzero_parent_fraction": nonzero_parent_fraction,
        }
        rows.append(compact)
        if admissible:
            eligible.append(compact)
    if eligible:
        chosen = min(
            eligible,
            key=lambda row: (
                row["parent_macro"]["selected_mae_delta_vs_base_m"],
                row["parent_macro"]["selected_bad_delta_vs_base"],
                -row["parent_macro"]["selected_coverage_fraction"],
                row["threshold"],
            ),
        )
        threshold = float(chosen["threshold"])
        decision = "NONTRIVIAL_NO_REGRET_THRESHOLD_FROZEN"
    else:
        threshold = 1.001
        decision = "NO_ADMISSIBLE_THRESHOLD_BASE_FALLBACK_FROZEN"
    selected_summary = summarize_selector_observations(observations, threshold)
    return {
        "decision": decision,
        "threshold": threshold,
        "minimum_coverage": minimum_coverage,
        "minimum_nonzero_parent_fraction": minimum_nonzero_parent_fraction,
        "metric_tolerance": metric_tolerance,
        "candidate_count": len(rows),
        "admissible_candidate_count": len(eligible),
        "candidates": rows,
        "selected_summary": selected_summary,
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "selector output collision")
    output_dir.mkdir(parents=True, exist_ok=False)
    expert_checkpoint_path = args.expert_checkpoint.resolve()
    require(expert_checkpoint_path.is_file(), "selector expert checkpoint missing")
    expert_payload = torch.load(
        expert_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(isinstance(expert_payload, dict), "selector expert payload invalid")
    expert_architecture = checkpoint_architecture(expert_payload)
    require(
        expert_architecture["depth_gate_profile"] == "identity_sigmoid",
        "selector expert is not identity-gated",
    )

    stage0a_results = [value.resolve() for value in args.stage0a_result]
    label_dirs = [value.resolve() for value in args.label_dir]
    require(len(stage0a_results) == len(label_dirs) == 3, "expected three ARKit batches")
    descriptors, source_batches = build_frame_descriptor_batches(
        stage0a_results,
        label_dirs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "selector training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    arkit_frames, arkit_extraction = extract_depthart_features(
        descriptors,
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        args.seed,
        feature_profile=expert_architecture["feature_profile"],
    )
    bonn_frames, bonn_extraction = extract_bonn_anchor_frames(
        args.cohort_manifest.resolve(),
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        args.seed,
        cohort_role="fit",
    )
    tum_frames: list[CachedFrame] = []
    tum_extractions: list[dict[str, Any]] = []
    for manifest in args.tum_cohort_manifest or []:
        manifest_frames, manifest_extraction = extract_tum_anchor_frames(
            manifest.resolve(),
            args.depthart_source.resolve(),
            args.depthart_checkpoint.resolve(),
            device,
            args.seed,
            cohort_role="fit",
        )
        tum_frames.extend(manifest_frames)
        tum_extractions.append(manifest_extraction)
    arkit_parents = sorted({row.descriptor.parent_id for row in arkit_frames})
    bonn_parents = sorted({row.descriptor.parent_id for row in bonn_frames})
    tum_parents = sorted({row.descriptor.parent_id for row in tum_frames})
    require(len(arkit_parents) == 40 and len(bonn_parents) == 8, "selector source roster drift")
    require(
        not tum_frames or len(tum_parents) >= 4,
        "TUM selector source roster too small",
    )
    require(
        len(tum_frames) == 3 * len(tum_parents),
        "TUM selector duplicate parent or frame-count drift",
    )
    arkit_fit, arkit_calibration = split_parent_roles(
        arkit_parents,
        calibration_count=8,
        domain="ARKITSCENES",
    )
    bonn_fit, bonn_calibration = split_parent_roles(
        bonn_parents,
        calibration_count=2,
        domain="BONN_RGBD_DYNAMIC",
    )
    tum_fit: list[str] = []
    tum_calibration: list[str] = []
    if tum_parents:
        tum_fit, tum_calibration = split_parent_roles(
            tum_parents,
            calibration_count=calibration_parent_count(len(tum_parents)),
            domain="TUM_RGBD",
        )
    fit_by_domain = {
        "ARKITSCENES": set(arkit_fit),
        "BONN_RGBD_DYNAMIC": set(bonn_fit),
        **({"TUM_RGBD": set(tum_fit)} if tum_fit else {}),
    }
    calibration_by_domain = {
        "ARKITSCENES": set(arkit_calibration),
        "BONN_RGBD_DYNAMIC": set(bonn_calibration),
        **({"TUM_RGBD": set(tum_calibration)} if tum_calibration else {}),
    }
    parent_domains = {
        **{parent: "ARKITSCENES" for parent in arkit_parents},
        **{parent: "BONN_RGBD_DYNAMIC" for parent in bonn_parents},
        **{parent: "TUM_RGBD" for parent in tum_parents},
    }
    all_frames = [*arkit_frames, *bonn_frames, *tum_frames]
    fit_frames = [
        row
        for row in all_frames
        if row.descriptor.parent_id in fit_by_domain[parent_domains[row.descriptor.parent_id]]
    ]
    calibration_frames = [
        row
        for row in all_frames
        if row.descriptor.parent_id
        in calibration_by_domain[parent_domains[row.descriptor.parent_id]]
    ]
    fit_frames_by_domain = {
        domain: [
            row
            for row in fit_frames
            if parent_domains[row.descriptor.parent_id] == domain
        ]
        for domain in sorted(fit_by_domain)
    }
    target_domain_frame_visits = max(len(rows) for rows in fit_frames_by_domain.values())
    repeat_factors = {
        domain: int(math.ceil(target_domain_frame_visits / len(rows)))
        for domain, rows in fit_frames_by_domain.items()
    }
    optimization_frames = [
        row
        for domain, rows in fit_frames_by_domain.items()
        for row in rows * repeat_factors[domain]
    ]

    _, expert = build_students(expert_payload, expert_architecture, device)
    expert.eval().requires_grad_(False)
    torch.manual_seed(args.seed)
    selector = NoRegretDepthSelector(
        feature_channels=expert_architecture["channels"],
        hidden=args.hidden_channels,
        global_context_profile=args.global_context_profile,
    ).to(device)
    if args.training_profile == "group_dro":
        history, training = train_selector_group_dro(
            selector,
            expert,
            fit_frames_by_domain,
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        history, training = train_selector(
            selector,
            expert,
            optimization_frames,
            device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    calibration_observations = collect_selector_observations(
        selector,
        expert,
        calibration_frames,
        parent_domains,
        device,
    )
    threshold_calibration = calibrate_selector_threshold(
        calibration_observations,
        minimum_coverage=args.minimum_calibration_coverage,
    )
    threshold = float(threshold_calibration["threshold"])
    fit_observations = collect_selector_observations(
        selector,
        expert,
        fit_frames,
        parent_domains,
        device,
    )
    fit_metrics = summarize_selector_observations(fit_observations, threshold)

    split = {
        "method": "SHA256_PARENT_DISJOINT_WITHIN_EACH_DOMAIN",
        "token": SELECTOR_SPLIT_TOKEN,
        "selector_fit_parents": sorted(arkit_fit + bonn_fit + tum_fit),
        "selector_calibration_parents": sorted(
            arkit_calibration + bonn_calibration + tum_calibration
        ),
        "arkit_fit_parents": arkit_fit,
        "arkit_calibration_parents": arkit_calibration,
        "bonn_fit_parents": bonn_fit,
        "bonn_calibration_parents": bonn_calibration,
        "tum_fit_parents": tum_fit,
        "tum_calibration_parents": tum_calibration,
        "external_bonn_evaluation_parents_read": False,
        "external_tum_evaluation_parents_read": False,
    }
    architecture = {
        "feature_channels": expert_architecture["channels"],
        "hidden_channels": args.hidden_channels,
        "global_context_profile": args.global_context_profile,
        "guidance_channels": [
            "log_base_depth",
            "signed_expert_log_correction",
            "absolute_expert_log_correction",
            "expert_identity_gate",
        ],
        "output": "probability_expert_beats_base",
        "hard_fallback": "base_depth_where_probability_below_frozen_threshold",
    }
    checkpoint = save_checkpoint_exclusive(
        output_dir / "no-regret-selector.pt",
        {
            "schema": SELECTOR_SCHEMA,
            "state_dict": {
                key: value.detach().cpu() for key, value in selector.state_dict().items()
            },
            "architecture": architecture,
            "expert": {
                "checkpoint_path": str(expert_checkpoint_path),
                "checkpoint_sha256": sha256_file(expert_checkpoint_path),
                "architecture": expert_architecture,
            },
            "split": split,
            "threshold": threshold,
            "threshold_calibration_decision": threshold_calibration["decision"],
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "training_profile": args.training_profile,
        },
    )
    result = {
        "schema": "blindassist_ag_st_no_regret_selector_fit_result_v1",
        "status": "NO_REGRET_SELECTOR_FIT_COMPLETE_EXTERNAL_EVALUATION_UNREAD",
        "mode": "WILD_LAB_DEVELOPMENT",
        "question": (
            "Can a separately trained selector predict whether the frozen correction "
            "expert beats frozen metric DepthART and otherwise fall back to the base?"
        ),
        "expert": {
            "checkpoint_path": str(expert_checkpoint_path),
            "checkpoint_sha256": sha256_file(expert_checkpoint_path),
            "frozen": True,
        },
        "inputs": {
            "arkit_source_batches": source_batches,
            "arkit_factor_label_payloads": aggregate_label_digest(
                row.label_path for row in descriptors
            ),
            "bonn_depth_anchors": bonn_extraction,
            "tum_depth_anchors": (
                tum_extractions[0] if len(tum_extractions) == 1 else tum_extractions
            )
            if tum_extractions
            else None,
            "arkit_feature_extraction": arkit_extraction,
        },
        "split": split,
        "architecture": architecture,
        "training": {
            **training,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "training_profile": args.training_profile,
            "unique_fit_frame_count": len(fit_frames),
            "calibration_frame_count": len(calibration_frames),
            "optimizer_frame_visits_per_epoch": len(optimization_frames),
            "domain_repeat_factors": repeat_factors,
            "history": history,
        },
        "threshold_calibration": threshold_calibration,
        "fit_metrics_at_frozen_threshold": fit_metrics,
        "checkpoint": checkpoint,
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "total_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "complete_truth_required": False,
            "expert_checkpoint_modified": False,
            "selector_calibration_parent_disjoint_from_selector_fit": True,
            "external_bonn_evaluation_read": False,
            "external_tum_evaluation_read": False,
            "cross_dataset_gain_claim_authorized": False,
            "task_deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(output_dir / "result.json"),
                "checkpoint": checkpoint,
                "threshold": threshold,
                "threshold_decision": threshold_calibration["decision"],
                "calibration_parent_macro": threshold_calibration["selected_summary"][
                    "parent_macro"
                ],
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage0a-result",
        type=Path,
        action="append",
        default=list(DEFAULT_STAGE0A_RESULTS),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        action="append",
        default=list(DEFAULT_LABEL_DIRS),
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument(
        "--tum-cohort-manifest",
        type=Path,
        action="append",
        default=None,
        help="Repeat to combine disjoint TUM fit manifests before the frozen split.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--expert-checkpoint", type=Path, default=DEFAULT_EXPERT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument(
        "--global-context-profile",
        choices=("none", "mean_std"),
        default="none",
    )
    parser.add_argument(
        "--training-profile",
        choices=("iid_balanced", "group_dro"),
        default="iid_balanced",
    )
    parser.add_argument("--minimum-calibration-coverage", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2761)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
