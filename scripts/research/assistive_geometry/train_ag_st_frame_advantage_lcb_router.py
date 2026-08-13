#!/usr/bin/env python3
"""Train a conservative frame gate over the frozen AG pixel selector.

The pixel selector and correction expert remain frozen.  A small ensemble
predicts lower quantiles of the *observable candidate's* frame-level MAE and
bad-rate advantages.  The gate may only veto the candidate and fall back to
metric DepthART; it cannot open pixels the frozen selector rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_no_regret_selector_bonn import load_selector
from run_ag_factorwise_no_regret_oracle_parent_gate_canary import evaluate_lane_gate
from train_ag_st_bonn_anchored_student import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    DEFAULT_COHORT_MANIFEST,
    DEFAULT_LABEL_DIRS,
    DEFAULT_STAGE0A_RESULTS,
)
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    CachedFrame,
    build_frame_descriptor_batches,
    extract_depthart_features,
    save_checkpoint_exclusive,
    write_json_exclusive,
)
from train_ag_st_no_regret_selector import (
    SelectorObservation,
    calibration_parent_count,
    extract_bonn_anchor_frames,
    extract_tum_anchor_frames,
    split_parent_roles,
    summarize_selector_observations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIXEL_SELECTOR_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-three-domain-global-group-dro-r0/no-regret-selector.pt"
)
DEFAULT_TUM_MANIFESTS = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_RGBD_THIRD_DOMAIN_COHORT_R0_2026-08-10.json",
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_THIRD_TEACHER_COHORT_R2_2026-08-10.json",
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-frame-advantage-lcb-router-tum8-r0"
)
FRAME_LCB_SCHEMA = "blindassist_ag_frame_advantage_lcb_router_checkpoint_v1"
FRAME_LCB_SPLIT_TOKEN = "AG_FRAME_ADVANTAGE_LCB_ROUTER_TUM8_R0"
MAE_SCALE_M = 0.10
BAD_RATE_SCALE = 0.05
LOWER_QUANTILE = 0.10
DEFAULT_SCORE_THRESHOLDS = (
    0.0,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
)
DEFAULT_KNN_CANDIDATES = (3, 5, 8, 12, 16, 24)
KNN_LOWER_QUANTILE = 0.20


@dataclass(frozen=True)
class FrameAdvantageObservation:
    parent_id: str
    domain: str
    observable: np.ndarray
    selector: SelectorObservation
    mae_advantage_m: float
    bad_rate_advantage: float


class FrameAdvantageQuantileRouter(nn.Module):
    def __init__(self, input_features: int, hidden: int = 64) -> None:
        super().__init__()
        require(input_features > 0 and hidden > 0, "frame router architecture invalid")
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float = LOWER_QUANTILE,
) -> torch.Tensor:
    require(0.0 < quantile < 0.5, "frame LCB quantile invalid")
    residual = target - prediction
    return torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()


def _map_statistics(value: torch.Tensor) -> torch.Tensor:
    flat = value.float().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90), device=flat.device),
    )
    return torch.cat((flat.mean()[None], flat.std(unbiased=False)[None], quantiles))


def observable_vector(
    frame: CachedFrame,
    expert_depth: torch.Tensor,
    expert_gate: torch.Tensor,
    selector_probability: torch.Tensor,
) -> np.ndarray:
    feature = frame.feature.float()
    base = frame.base_depth_m.float().clamp(0.05, 20.0)
    expert = expert_depth.float().cpu().clamp(0.05, 20.0)
    gate = expert_gate.float().cpu().clamp(0.0, 1.0)
    probability = selector_probability.float().cpu().clamp(0.0, 1.0)
    correction = expert.log() - base.log()
    feature_mean = feature.mean(dim=(-2, -1))
    feature_std = feature.var(dim=(-2, -1), unbiased=False).clamp_min(1e-8).sqrt()
    vector = torch.cat(
        (
            feature_mean,
            feature_std,
            _map_statistics(base.log()),
            _map_statistics(correction),
            _map_statistics(correction.abs()),
            _map_statistics(gate),
            _map_statistics(probability),
        )
    )
    result = vector.numpy().astype(np.float32, copy=False)
    require(np.all(np.isfinite(result)), "frame router observable non-finite")
    return result


def collect_frame_observations(
    frames: list[CachedFrame],
    parent_domains: dict[str, str],
    pixel_selector: nn.Module,
    expert: nn.Module,
    pixel_threshold: float,
    device: torch.device,
) -> list[FrameAdvantageObservation]:
    pixel_selector.eval().requires_grad_(False)
    expert.eval().requires_grad_(False)
    rows: list[FrameAdvantageObservation] = []
    for frame in frames:
        feature = frame.feature[None].float().to(device)
        base = frame.base_depth_m[None].float().to(device)
        with torch.inference_mode():
            expert_outputs = expert(feature, base, frame.descriptor.output_hw)
            selector_outputs = pixel_selector(
                feature,
                base,
                expert_outputs["depth_m"],
                expert_outputs["depth_identity_gate"],
                frame.descriptor.output_hw,
            )
        truth = np.asarray(
            frame.targets["metric_depth_m"].float().cpu().numpy()
        ).squeeze()
        valid = np.asarray(
            frame.targets["metric_valid"].bool().cpu().numpy()
        ).squeeze()
        base_np = base[0, 0].float().cpu().numpy()
        expert_np = expert_outputs["depth_m"][0, 0].float().cpu().numpy()
        probability_np = (
            selector_outputs["selector_probability"][0, 0].float().cpu().numpy()
        )
        require(
            truth.shape == valid.shape == base_np.shape == expert_np.shape,
            "frame router metric shape drift",
        )
        candidate_np = np.where(probability_np >= pixel_threshold, expert_np, base_np)
        require(bool(valid.any()), "frame router truth denominator empty")
        base_error = np.abs(base_np[valid] - truth[valid])
        candidate_error = np.abs(candidate_np[valid] - truth[valid])
        rows.append(
            FrameAdvantageObservation(
                parent_id=frame.descriptor.parent_id,
                domain=parent_domains[frame.descriptor.parent_id],
                observable=observable_vector(
                    frame,
                    expert_outputs["depth_m"][0],
                    expert_outputs["depth_identity_gate"][0],
                    selector_outputs["selector_probability"][0],
                ),
                selector=SelectorObservation(
                    parent_id=frame.descriptor.parent_id,
                    domain=parent_domains[frame.descriptor.parent_id],
                    truth_depth_m=truth,
                    valid=valid,
                    base_depth_m=base_np,
                    expert_depth_m=expert_np,
                    selector_probability=probability_np,
                ),
                mae_advantage_m=float(base_error.mean() - candidate_error.mean()),
                bad_rate_advantage=float(
                    (base_error > 0.10).mean() - (candidate_error > 0.10).mean()
                ),
            )
        )
    return rows


def normalize_observables(
    fit_rows: list[FrameAdvantageObservation],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([row.observable for row in fit_rows], axis=0).astype(np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _training_arrays(
    rows: list[FrameAdvantageObservation], mean: np.ndarray, std: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.stack([(row.observable - mean) / std for row in rows], axis=0)
    targets = np.asarray(
        [
            (row.mae_advantage_m / MAE_SCALE_M, row.bad_rate_advantage / BAD_RATE_SCALE)
            for row in rows
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def train_ensemble(
    rows: list[FrameAdvantageObservation],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    ensemble_size: int,
    epochs: int,
    learning_rate: float,
    hidden: int,
    seed: int,
) -> tuple[list[FrameAdvantageQuantileRouter], list[dict[str, Any]]]:
    require(rows and ensemble_size >= 3, "frame LCB ensemble too small")
    require(epochs > 0 and learning_rate > 0.0, "frame LCB training invalid")
    inputs, targets = _training_arrays(rows, mean, std)
    inputs = inputs.to(device)
    targets = targets.to(device)
    domains = sorted({row.domain for row in rows})
    masks = {
        domain: torch.tensor(
            [row.domain == domain for row in rows], device=device, dtype=torch.bool
        )
        for domain in domains
    }
    models: list[FrameAdvantageQuantileRouter] = []
    receipts: list[dict[str, Any]] = []
    for member in range(ensemble_size):
        member_seed = seed + 1009 * member
        torch.manual_seed(member_seed)
        model = FrameAdvantageQuantileRouter(inputs.shape[1], hidden).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=learning_rate * 0.05
        )
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            prediction = model(inputs)
            domain_losses = torch.stack(
                [pinball_loss(prediction[mask], targets[mask]) for mask in masks.values()]
            )
            smooth_worst = 0.20 * (
                torch.logsumexp(domain_losses / 0.20, dim=0) - math.log(len(domains))
            )
            loss = 0.25 * domain_losses.mean() + 0.75 * smooth_worst
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            require(math.isfinite(gradient), "frame LCB gradient non-finite")
            optimizer.step()
            scheduler.step()
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 100 == 0:
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": float(loss.detach()),
                        "gradient_norm": gradient,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
        model.eval().requires_grad_(False)
        models.append(model)
        receipts.append({"member": member, "seed": member_seed, "history": history})
    return models, receipts


def ensemble_lcb_scores(
    models: list[FrameAdvantageQuantileRouter],
    rows: list[FrameAdvantageObservation],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    disagreement_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    inputs, _ = _training_arrays(rows, mean, std)
    inputs = inputs.to(device)
    with torch.inference_mode():
        predictions = torch.stack([model(inputs) for model in models], dim=0)
    average = predictions.mean(dim=0)
    disagreement = predictions.std(dim=0, unbiased=False)
    lower = average - disagreement_multiplier * disagreement
    score = lower.min(dim=1).values
    return lower.cpu().numpy(), score.cpu().numpy()


def normalized_advantage_targets(
    rows: list[FrameAdvantageObservation],
) -> np.ndarray:
    return np.asarray(
        [
            (row.mae_advantage_m / MAE_SCALE_M, row.bad_rate_advantage / BAD_RATE_SCALE)
            for row in rows
        ],
        dtype=np.float32,
    )


def knn_support_lower_bounds(
    fit_rows: list[FrameAdvantageObservation],
    query_rows: list[FrameAdvantageObservation],
    mean: np.ndarray,
    std: np.ndarray,
    *,
    neighbors: int,
    lower_quantile: float = KNN_LOWER_QUANTILE,
    exclude_same_parent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return label-aware local lower bounds and nearest-support distances."""

    require(neighbors > 0, "frame KNN neighbor count invalid")
    require(0.0 < lower_quantile < 0.5, "frame KNN quantile invalid")
    fit_matrix = np.stack(
        [(row.observable - mean) / std for row in fit_rows], axis=0
    ).astype(np.float32)
    query_matrix = np.stack(
        [(row.observable - mean) / std for row in query_rows], axis=0
    ).astype(np.float32)
    targets = normalized_advantage_targets(fit_rows)
    lower_rows: list[np.ndarray] = []
    nearest_distances: list[float] = []
    for query, query_row in zip(query_matrix, query_rows, strict=True):
        distance = np.mean((fit_matrix - query[None, :]) ** 2, axis=1)
        eligible = np.asarray(
            [
                not exclude_same_parent or row.parent_id != query_row.parent_id
                for row in fit_rows
            ],
            dtype=bool,
        )
        eligible_indices = np.flatnonzero(eligible)
        require(
            eligible_indices.size >= neighbors,
            "frame KNN support bank too small after parent exclusion",
        )
        ranked = eligible_indices[np.argsort(distance[eligible_indices], kind="stable")]
        selected = ranked[:neighbors]
        lower_rows.append(
            np.quantile(targets[selected], lower_quantile, axis=0).astype(np.float32)
        )
        nearest_distances.append(float(distance[selected[0]]))
    return np.stack(lower_rows, axis=0), np.asarray(nearest_distances, dtype=np.float32)


def calibrate_knn_support(
    fit_rows: list[FrameAdvantageObservation],
    calibration_rows: list[FrameAdvantageObservation],
    neural_lower: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pixel_threshold: float,
    *,
    candidates: Iterable[int] = DEFAULT_KNN_CANDIDATES,
) -> dict[str, Any]:
    tested: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for neighbors in sorted(set(int(value) for value in candidates)):
        knn_lower, distances = knn_support_lower_bounds(
            fit_rows,
            calibration_rows,
            mean,
            std,
            neighbors=neighbors,
        )
        combined_lower = np.minimum(neural_lower, knn_lower)
        scores = combined_lower.min(axis=1)
        calibration = calibrate_frame_gate(
            calibration_rows,
            scores,
            pixel_threshold,
            candidates=(0.0,),
        )
        row = {
            "neighbors": neighbors,
            "admissible": calibration["frame_score_threshold"] is not None,
            "calibration": calibration,
            "nearest_distance_mean": float(distances.mean()),
            "nearest_distance_max": float(distances.max()),
            "combined_lower": combined_lower,
            "scores": scores,
            "distances": distances,
        }
        tested.append(row)
        if row["admissible"]:
            eligible.append(row)
    if not eligible:
        return {
            "decision": "FRAME_ADVANTAGE_KNN_SUPPORT_NO_ADMISSIBLE_GATE",
            "neighbors": None,
            "calibration": calibrate_frame_gate(
                calibration_rows,
                np.full(len(calibration_rows), -math.inf, dtype=np.float32),
                pixel_threshold,
                candidates=(0.0,),
            ),
            "combined_lower": np.full((len(calibration_rows), 2), -math.inf),
            "scores": np.full(len(calibration_rows), -math.inf),
            "distances": np.full(len(calibration_rows), math.inf),
            "candidates": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"combined_lower", "scores", "distances"}
                }
                for row in tested
            ],
        }
    # Prefer the broadest local evidence set that remains parent-safe.  This is
    # deliberately more conservative than choosing the largest calibration gain.
    chosen = max(eligible, key=lambda row: row["neighbors"])
    return {
        "decision": "FRAME_ADVANTAGE_KNN_SUPPORT_NONTRIVIAL_GATE_FROZEN",
        "neighbors": chosen["neighbors"],
        "calibration": chosen["calibration"],
        "combined_lower": chosen["combined_lower"],
        "scores": chosen["scores"],
        "distances": chosen["distances"],
        "candidates": [
            {
                key: value
                for key, value in row.items()
                if key not in {"combined_lower", "scores", "distances"}
            }
            for row in tested
        ],
    }


def gated_selector_observations(
    rows: list[FrameAdvantageObservation],
    scores: np.ndarray,
    frame_threshold: float,
) -> list[SelectorObservation]:
    require(len(rows) == len(scores), "frame LCB score count drift")
    result: list[SelectorObservation] = []
    for row, score in zip(rows, scores, strict=True):
        source = row.selector
        probability = (
            source.selector_probability
            if float(score) >= frame_threshold
            else np.full_like(source.selector_probability, -1.0)
        )
        result.append(
            SelectorObservation(
                parent_id=source.parent_id,
                domain=source.domain,
                truth_depth_m=source.truth_depth_m,
                valid=source.valid,
                base_depth_m=source.base_depth_m,
                expert_depth_m=source.expert_depth_m,
                selector_probability=probability,
            )
        )
    return result


def calibrate_frame_gate(
    rows: list[FrameAdvantageObservation],
    scores: np.ndarray,
    pixel_threshold: float,
    *,
    candidates: Iterable[float] = DEFAULT_SCORE_THRESHOLDS,
) -> dict[str, Any]:
    tested: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in candidates)):
        observations = gated_selector_observations(rows, scores, threshold)
        summary = summarize_selector_observations(observations, pixel_threshold)
        gate = evaluate_lane_gate(summary, "selected")
        row = {
            "frame_score_threshold": threshold,
            "admissible": gate["pass"],
            "gate": gate,
            "parent_macro": summary["parent_macro"],
        }
        tested.append(row)
        if gate["pass"]:
            eligible.append(row)
    if eligible:
        chosen = min(
            eligible,
            key=lambda row: (
                row["parent_macro"]["selected_mae_delta_vs_base_m"],
                row["parent_macro"]["selected_bad_delta_vs_base"],
                -row["parent_macro"]["selected_coverage_fraction"],
                row["frame_score_threshold"],
            ),
        )
        threshold = float(chosen["frame_score_threshold"])
        decision = "FRAME_ADVANTAGE_LCB_NONTRIVIAL_GATE_FROZEN"
    else:
        threshold = float("inf")
        decision = "FRAME_ADVANTAGE_LCB_NO_ADMISSIBLE_GATE_BASE_FALLBACK"
    selected = summarize_selector_observations(
        gated_selector_observations(rows, scores, threshold), pixel_threshold
    )
    return {
        "decision": decision,
        "frame_score_threshold": threshold if math.isfinite(threshold) else None,
        "candidate_count": len(tested),
        "admissible_candidate_count": len(eligible),
        "candidates": tested,
        "selected_summary": selected,
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "frame LCB output collision")
    output_dir.mkdir(parents=True, exist_ok=False)
    require(torch.cuda.is_available(), "frame LCB training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pixel_checkpoint = args.pixel_selector_checkpoint.resolve()
    (
        pixel_payload,
        pixel_selector,
        checkpoint_pixel_threshold,
        expert_path,
        _expert_payload,
        expert,
    ) = load_selector(pixel_checkpoint, device)
    pixel_threshold = (
        float(args.pixel_threshold_candidate)
        if args.pixel_threshold_candidate is not None
        else checkpoint_pixel_threshold
    )
    require(0.0 < pixel_threshold <= 1.0, "pixel selector candidate threshold invalid")

    feature_profile = pixel_payload["expert"]["architecture"]["feature_profile"]
    training_domains = set(args.training_domain or (
        "ARKITSCENES",
        "BONN_RGBD_DYNAMIC",
        "TUM_RGBD",
    ))
    if "ARKITSCENES" in training_domains:
        descriptors, source_batches = build_frame_descriptor_batches(
            [value.resolve() for value in args.stage0a_result],
            [value.resolve() for value in args.label_dir],
        )
        arkit_frames, arkit_extraction = extract_depthart_features(
            descriptors,
            args.depthart_source.resolve(),
            args.depthart_checkpoint.resolve(),
            device,
            int(pixel_payload["seed"]),
            feature_profile=feature_profile,
        )
    else:
        source_batches = []
        arkit_frames = []
        arkit_extraction = {"skipped": True, "reason": "TRAINING_DOMAIN_FILTER"}
    if "BONN_RGBD_DYNAMIC" in training_domains:
        bonn_frames, bonn_extraction = extract_bonn_anchor_frames(
            args.cohort_manifest.resolve(),
            args.dataset_root.resolve(),
            args.archive.resolve(),
            args.catalog.resolve(),
            args.receipt.resolve(),
            args.depthart_source.resolve(),
            args.depthart_checkpoint.resolve(),
            device,
            int(pixel_payload["seed"]),
            cohort_role="fit",
        )
    else:
        bonn_frames = []
        bonn_extraction = {"skipped": True, "reason": "TRAINING_DOMAIN_FILTER"}
    tum_manifests = (
        [value.resolve() for value in args.tum_cohort_manifest]
        if args.tum_cohort_manifest
        else [value.resolve() for value in DEFAULT_TUM_MANIFESTS]
    )
    tum_frames: list[CachedFrame] = []
    tum_extractions: list[dict[str, Any]] = []
    if "TUM_RGBD" in training_domains:
        tum_roles = (
            ("fit", "evaluation")
            if args.include_consumed_tum_evaluation
            else ("fit",)
        )
        for manifest in tum_manifests:
            for role in tum_roles:
                frames, extraction = extract_tum_anchor_frames(
                    manifest,
                    args.depthart_source.resolve(),
                    args.depthart_checkpoint.resolve(),
                    device,
                    int(pixel_payload["seed"]),
                    cohort_role=role,
                )
                tum_frames.extend(frames)
                tum_extractions.append(extraction)
    require(
        not tum_frames
        or len(tum_frames)
        == 3 * len({frame.descriptor.parent_id for frame in tum_frames}),
        "frame LCB TUM duplicate parent or frame-count drift",
    )
    parent_domains = {
        **{frame.descriptor.parent_id: "ARKITSCENES" for frame in arkit_frames},
        **{
            frame.descriptor.parent_id: "BONN_RGBD_DYNAMIC" for frame in bonn_frames
        },
        **{frame.descriptor.parent_id: "TUM_RGBD" for frame in tum_frames},
    }
    frames_by_domain = {
        "ARKITSCENES": arkit_frames,
        "BONN_RGBD_DYNAMIC": bonn_frames,
        "TUM_RGBD": tum_frames,
    }
    frames_by_domain = {
        domain: frames
        for domain, frames in frames_by_domain.items()
        if domain in training_domains
    }
    require(frames_by_domain, "frame LCB training-domain set empty")
    require(all(frames_by_domain.values()), "frame LCB selected training domain empty")
    fit_parents: set[str] = set()
    calibration_parents: set[str] = set()
    split: dict[str, Any] = {
        "method": "SHA256_PARENT_DISJOINT_WITHIN_DOMAIN",
        "token": (
            FRAME_LCB_SPLIT_TOKEN
            + "|DOMAINS="
            + ",".join(sorted(training_domains))
            + f"|CONSUMED_TUM_EVALUATION={args.include_consumed_tum_evaluation}"
        ),
    }
    for domain, frames in frames_by_domain.items():
        parents = sorted({frame.descriptor.parent_id for frame in frames})
        domain_fit, domain_calibration = split_parent_roles(
            parents,
            calibration_count=calibration_parent_count(len(parents)),
            domain=domain,
            token=split["token"],
        )
        fit_parents.update(domain_fit)
        calibration_parents.update(domain_calibration)
        split[f"{domain.lower()}_fit_parents"] = domain_fit
        split[f"{domain.lower()}_calibration_parents"] = domain_calibration
    split["fit_parents"] = sorted(fit_parents)
    split["calibration_parents"] = sorted(calibration_parents)
    all_frames = [*arkit_frames, *bonn_frames, *tum_frames]
    frame_rows = collect_frame_observations(
        all_frames,
        parent_domains,
        pixel_selector,
        expert,
        pixel_threshold,
        device,
    )
    fit_rows = [row for row in frame_rows if row.parent_id in fit_parents]
    calibration_rows = [
        row for row in frame_rows if row.parent_id in calibration_parents
    ]
    require(fit_rows and calibration_rows, "frame LCB split empty")
    observable_mean, observable_std = normalize_observables(fit_rows)
    models, training_receipts = train_ensemble(
        fit_rows,
        observable_mean,
        observable_std,
        device,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden=args.hidden_channels,
        seed=args.seed,
    )
    neural_calibration_lower, _neural_calibration_scores = ensemble_lcb_scores(
        models,
        calibration_rows,
        observable_mean,
        observable_std,
        device,
        disagreement_multiplier=args.disagreement_multiplier,
    )
    knn_support = calibrate_knn_support(
        fit_rows,
        calibration_rows,
        neural_calibration_lower,
        observable_mean,
        observable_std,
        pixel_threshold,
    )
    calibration = knn_support["calibration"]
    calibration_lower = knn_support["combined_lower"]
    calibration_scores = knn_support["scores"]
    frame_threshold = calibration["frame_score_threshold"]
    neural_fit_lower, _neural_fit_scores = ensemble_lcb_scores(
        models,
        fit_rows,
        observable_mean,
        observable_std,
        device,
        disagreement_multiplier=args.disagreement_multiplier,
    )
    if knn_support["neighbors"] is not None:
        fit_knn_lower, _fit_distances = knn_support_lower_bounds(
            fit_rows,
            fit_rows,
            observable_mean,
            observable_std,
            neighbors=int(knn_support["neighbors"]),
        )
        fit_lower = np.minimum(neural_fit_lower, fit_knn_lower)
        fit_scores = fit_lower.min(axis=1)
    else:
        fit_lower = np.full_like(neural_fit_lower, -math.inf)
        fit_scores = np.full(len(fit_rows), -math.inf, dtype=np.float32)
    effective_threshold = float(frame_threshold) if frame_threshold is not None else math.inf
    fit_summary = summarize_selector_observations(
        gated_selector_observations(fit_rows, fit_scores, effective_threshold),
        pixel_threshold,
    )
    checkpoint = save_checkpoint_exclusive(
        output_dir / "frame-advantage-lcb-router.pt",
        {
            "schema": FRAME_LCB_SCHEMA,
            "architecture": {
                "input_features": int(observable_mean.size),
                "hidden_channels": args.hidden_channels,
                "ensemble_size": args.ensemble_size,
                "lower_quantile": LOWER_QUANTILE,
                "disagreement_multiplier": args.disagreement_multiplier,
            },
            "state_dicts": [
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for model in models
            ],
            "normalization": {
                "mean": observable_mean.tolist(),
                "std": observable_std.tolist(),
            },
            "knn_support": {
                "neighbors": knn_support["neighbors"],
                "lower_quantile": KNN_LOWER_QUANTILE,
                "exclude_same_parent": True,
                "fit_observables": [row.observable.tolist() for row in fit_rows],
                "fit_targets_normalized": normalized_advantage_targets(fit_rows).tolist(),
                "fit_parent_ids": [row.parent_id for row in fit_rows],
                "fit_domains": [row.domain for row in fit_rows],
            },
            "pixel_selector": {
                "path": str(pixel_checkpoint),
                "sha256": sha256_file(pixel_checkpoint),
                "checkpoint_threshold": checkpoint_pixel_threshold,
                "candidate_threshold": pixel_threshold,
                "candidate_threshold_overrides_checkpoint_fallback": (
                    abs(pixel_threshold - checkpoint_pixel_threshold) > 1e-12
                ),
            },
            "split": split,
            "frame_score_threshold": frame_threshold,
            "calibration_decision": knn_support["decision"],
            "scales": {"mae_m": MAE_SCALE_M, "bad_rate": BAD_RATE_SCALE},
            "seed": args.seed,
        },
    )
    result = {
        "schema": "blindassist_ag_frame_advantage_lcb_router_fit_result_v1",
        "status": (
            "FRAME_ADVANTAGE_KNN_LCB_ROUTER_FIT_PASS_EXTERNAL_EVALUATION_UNREAD"
            if frame_threshold is not None
            else "FRAME_ADVANTAGE_KNN_LCB_ROUTER_FIT_FALLBACK_EXTERNAL_EVALUATION_UNREAD"
        ),
        "mode": "PROJECT_CONSUMED_DEVELOPMENT",
        "question": (
            "Can a one-sided frame-level advantage LCB veto unsafe outputs from the "
            "frozen pixel selector while retaining nonzero parent-distributed coverage?"
        ),
        "frozen_candidate": {
            "pixel_selector_checkpoint_path": str(pixel_checkpoint),
            "pixel_selector_checkpoint_sha256": sha256_file(pixel_checkpoint),
            "checkpoint_pixel_threshold": checkpoint_pixel_threshold,
            "candidate_pixel_threshold": pixel_threshold,
            "candidate_threshold_overrides_checkpoint_fallback": (
                abs(pixel_threshold - checkpoint_pixel_threshold) > 1e-12
            ),
            "expert_checkpoint_path": str(expert_path),
            "expert_checkpoint_sha256": sha256_file(expert_path),
            "modified": False,
        },
        "sources": {
            "arkit_source_batches": source_batches,
            "arkit_extraction": arkit_extraction,
            "bonn_fit_extraction": bonn_extraction,
            "tum_fit_extractions": tum_extractions,
            "external_bonn_evaluation_read": False,
            "external_tum_evaluation_read": False,
            "eth3d_confirmation_read": False,
        },
        "split": split,
        "training": {
            "domains": sorted(training_domains),
            "consumed_tum_evaluation_included": args.include_consumed_tum_evaluation,
            "fit_frame_count": len(fit_rows),
            "calibration_frame_count": len(calibration_rows),
            "observable_feature_count": int(observable_mean.size),
            "ensemble_size": args.ensemble_size,
            "epochs_per_member": args.epochs,
            "learning_rate": args.learning_rate,
            "lower_quantile": LOWER_QUANTILE,
            "disagreement_multiplier": args.disagreement_multiplier,
            "receipts": training_receipts,
        },
        "calibration": {
            **calibration,
            "knn_support_decision": knn_support["decision"],
            "knn_neighbors": knn_support["neighbors"],
            "knn_lower_quantile": KNN_LOWER_QUANTILE,
            "knn_candidates": knn_support["candidates"],
            "per_frame_lower_predictions": [
                {
                    "parent_id": row.parent_id,
                    "domain": row.domain,
                    "mae_advantage_lcb_normalized": float(lower[0]),
                    "bad_rate_advantage_lcb_normalized": float(lower[1]),
                    "joint_score": float(score),
                    "truth_mae_advantage_m": row.mae_advantage_m,
                    "truth_bad_rate_advantage": row.bad_rate_advantage,
                }
                for row, lower, score in zip(
                    calibration_rows,
                    calibration_lower,
                    calibration_scores,
                    strict=True,
                )
            ],
        },
        "fit_metrics_at_frozen_gate": fit_summary,
        "checkpoint": checkpoint,
        "decision": {
            "nontrivial_gate_frozen": frame_threshold is not None,
            "fresh_parent_disjoint_evaluation_authorized": frame_threshold is not None,
            "fallback_is_success": False,
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "total_seconds": time.perf_counter() - started,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "claim_boundary": {
            "development_only": True,
            "source_depth_is_runtime_input": False,
            "boundary_support_unknown_modified": False,
            "reducer_called": False,
            "fresh_evaluation_claim_authorized": False,
            "deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(output_dir / "result.json"),
                "checkpoint": checkpoint,
                "calibration_decision": calibration["decision"],
                "frame_score_threshold": frame_threshold,
                "calibration_parent_macro": calibration["selected_summary"][
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
        "--pixel-selector-checkpoint",
        type=Path,
        default=DEFAULT_PIXEL_SELECTOR_CHECKPOINT,
    )
    parser.add_argument(
        "--pixel-threshold-candidate",
        type=float,
        default=None,
        help=(
            "Explicit Development candidate threshold for a checkpoint whose "
            "calibration terminal froze full fallback."
        ),
    )
    parser.add_argument(
        "--stage0a-result", type=Path, action="append", default=list(DEFAULT_STAGE0A_RESULTS)
    )
    parser.add_argument(
        "--label-dir", type=Path, action="append", default=list(DEFAULT_LABEL_DIRS)
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--tum-cohort-manifest", type=Path, action="append", default=None)
    parser.add_argument(
        "--training-domain",
        action="append",
        choices=("ARKITSCENES", "BONN_RGBD_DYNAMIC", "TUM_RGBD"),
        default=None,
        help="Restrict the router fit and calibration to one or more source domains.",
    )
    parser.add_argument(
        "--include-consumed-tum-evaluation",
        action="store_true",
        help=(
            "Reclassify both configured manifests' already-consumed evaluation "
            "parents as Development fit/calibration material."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--disagreement-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6101)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
