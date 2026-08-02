#!/usr/bin/env python3
"""Test early joint RGB-pair interaction from SANPO to public video."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    load_model,
    single_frame_spatial_features,
)
from run_stage_c_d6_paired_rgb_relation_backbone_canary import (
    BATCH_SIZE,
    EPOCHS,
    WEIGHT_DECAY,
    collect_sanpo_rgb_support_episodes,
    configure_deterministic_torch,
    load_frame_tensors,
)
from run_stage_c_d6_provisional_relation_transfer import (
    PUBLIC_VIDEO_STEP_MS,
    collect_public_video_actionability_episodes,
)
from run_stage_c_d6_sanpo_paired_pretraining_public_transfer_canary import (
    EXPECTED_PUBLIC_EPISODES,
    EXPECTED_PUBLIC_FRAMES,
    EXPECTED_PUBLIC_SOURCES,
    EXPECTED_SUPPORT_EPISODES,
    EXPECTED_SUPPORT_FRAMES,
    EXPECTED_SUPPORT_SOURCES,
    inventory,
    select_intervention_bearing_public_episodes,
    summarize_by_source,
)
from run_stage_c_d6_source_centered_relation_encoder_canary import (
    DEFAULT_ACTIONABILITY_MANIFEST,
    DEFAULT_FEATURE_CONTRACT,
    HIDDEN_CHANNELS,
    RelationEncoder,
    summarize_predictions,
)
from run_stage_c_d6_sanpo_spatial_relation_head import (
    SPATIAL_GRID,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    event_balanced_weights,
)
from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_joint_pair_interaction_"
    "public_transfer_canary_v1"
)
SEED = 1706
LEARNING_RATE = 3e-3
PAIR_CHANNELS = (24, 32, 64, 128)


class DepthwiseSeparableDownsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                3,
                stride=2,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.Conv2d(
                input_channels,
                output_channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.Hardswish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class JointPairInteractionModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.pair_stem = nn.Sequential(
            nn.Conv2d(
                12,
                PAIR_CHANNELS[0],
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(6, PAIR_CHANNELS[0]),
            nn.Hardswish(),
            DepthwiseSeparableDownsample(
                PAIR_CHANNELS[0],
                PAIR_CHANNELS[1],
            ),
            DepthwiseSeparableDownsample(
                PAIR_CHANNELS[1],
                PAIR_CHANNELS[2],
            ),
            DepthwiseSeparableDownsample(
                PAIR_CHANNELS[2],
                PAIR_CHANNELS[3],
            ),
        )
        self.relation = RelationEncoder()

    def train(self, mode: bool = True) -> "JointPairInteractionModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        current: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        delta = current - baseline
        pair = torch.cat(
            (current, baseline, delta, delta.abs()),
            dim=1,
        )
        pair_features = nnf.interpolate(
            self.pair_stem(pair),
            size=SPATIAL_GRID,
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            context = single_frame_spatial_features(
                self.backbone,
                current,
            )
            context = nnf.interpolate(
                context,
                size=SPATIAL_GRID,
                mode="bilinear",
                align_corners=False,
            )
        return self.relation(
            torch.cat((context, pair_features), dim=1)
        )


def build_source_baselines(
    tensors: torch.Tensor,
    reference_indices: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    return {
        source_id: tensors[indices].mean(dim=0)
        for source_id, indices in reference_indices.items()
    }


def score_batch(
    model: JointPairInteractionModel,
    tensors: torch.Tensor,
    indices: np.ndarray,
    sources: np.ndarray,
    source_baselines: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    current = tensors[indices].to(device, non_blocking=True)
    baseline = torch.stack(
        [
            source_baselines[str(source_id)]
            for source_id in sources[indices]
        ]
    ).to(device, non_blocking=True)
    return model(current, baseline)


def train_and_transfer(
    base_backbone: nn.Module,
    tensors: torch.Tensor,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
    reference_indices: dict[str, list[int]],
    train_rows: np.ndarray,
    transfer_rows: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[float], int]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = JointPairInteractionModel(
        copy.deepcopy(base_backbone)
    ).to(device)
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train = np.flatnonzero(train_rows)
    transfer = np.flatnonzero(transfer_rows)
    weights = np.zeros(len(labels), dtype=np.float32)
    weights[train] = event_balanced_weights(
        episode_ids[train].tolist(),
        labels[train],
    ).astype(np.float32)
    source_baselines = build_source_baselines(
        tensors,
        reference_indices,
    )
    generator = np.random.default_rng(SEED)
    epoch_losses = []
    for _ in range(EPOCHS):
        model.train()
        order = generator.permutation(train)
        loss_sum_epoch = 0.0
        weight_sum_epoch = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = score_batch(
                model,
                tensors,
                batch,
                sources,
                source_baselines,
                device,
            )
            targets = torch.from_numpy(
                labels[batch].astype(np.float32)
            ).to(device)
            batch_weights = torch.from_numpy(weights[batch]).to(
                device
            )
            weighted = (
                nnf.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
                * batch_weights
            ).sum()
            loss = weighted / batch_weights.sum()
            loss.backward()
            optimizer.step()
            loss_sum_epoch += float(weighted.detach().cpu())
            weight_sum_epoch += float(
                batch_weights.sum().detach().cpu()
            )
        epoch_losses.append(loss_sum_epoch / weight_sum_epoch)

    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(transfer), BATCH_SIZE):
            batch = transfer[start : start + BATCH_SIZE]
            logits = score_batch(
                model,
                tensors,
                batch,
                sources,
                source_baselines,
                device,
            )
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )
    return (
        np.concatenate(probabilities),
        epoch_losses,
        sum(
            parameter.numel()
            for parameter in trainable_parameters
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actionability-manifest",
        type=Path,
        default=DEFAULT_ACTIONABILITY_MANIFEST,
    )
    parser.add_argument(
        "--feature-contract",
        type=Path,
        default=DEFAULT_FEATURE_CONTRACT,
    )
    parser.add_argument(
        "--sanpo-support-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    args = parser.parse_args()
    configure_deterministic_torch()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError(
            "Refusing to overwrite joint-pair transfer output"
        )

    all_public_episodes = (
        collect_public_video_actionability_episodes(
            args.actionability_manifest,
            args.feature_contract,
            PUBLIC_VIDEO_STEP_MS,
        )
    )
    public_episodes, positive_public_sources = (
        select_intervention_bearing_public_episodes(
            all_public_episodes
        )
    )
    support_episodes = collect_sanpo_rgb_support_episodes(
        args.sanpo_support_manifest
    )
    support_inventory = inventory(support_episodes)
    public_inventory = inventory(public_episodes)
    if support_inventory != {
        "episode_count": EXPECTED_SUPPORT_EPISODES,
        "source_count": EXPECTED_SUPPORT_SOURCES,
        "frame_count": EXPECTED_SUPPORT_FRAMES,
    }:
        raise ValueError("Unexpected SANPO paired support inventory")
    if public_inventory != {
        "episode_count": EXPECTED_PUBLIC_EPISODES,
        "source_count": EXPECTED_PUBLIC_SOURCES,
        "frame_count": EXPECTED_PUBLIC_FRAMES,
    }:
        raise ValueError(
            "Unexpected intervention-bearing public inventory"
        )

    episodes = support_episodes + public_episodes
    (
        tensors,
        labels,
        sources,
        episode_ids,
        reference_indices,
        source_references,
    ) = load_frame_tensors(episodes, args.decode_batch_size)
    train_rows = np.char.startswith(
        episode_ids,
        "sanpo-consumed-support/",
    )
    transfer_rows = np.char.startswith(
        episode_ids,
        "public-video-actionability/",
    )
    if set(sources[train_rows]) & set(sources[transfer_rows]):
        raise ValueError("SANPO/public source overlap")

    backbone, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    probabilities, epoch_losses, trainable_parameter_count = (
        train_and_transfer(
            backbone,
            tensors,
            labels,
            sources,
            episode_ids,
            reference_indices,
            train_rows,
            transfer_rows,
            device,
        )
    )
    transfer_labels = labels[transfer_rows]
    transfer_sources = sources[transfer_rows]
    transfer_episode_ids = episode_ids[transfer_rows]
    metrics = summarize_predictions(
        probabilities,
        transfer_labels,
        transfer_sources,
        transfer_episode_ids,
    )
    metrics_by_source, source_macro_metrics = (
        summarize_by_source(
            probabilities,
            transfer_labels,
            transfer_sources,
            transfer_episode_ids,
        )
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "JOINT_PAIR_INTERACTION_PUBLIC_TRANSFER_COMPLETE"
        ),
        "inputs": {
            "actionability_manifest_sha256": sha256(
                args.actionability_manifest
            ),
            "feature_contract_sha256": sha256(
                args.feature_contract
            ),
            "sanpo_support_manifest_sha256": sha256(
                args.sanpo_support_manifest
            ),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
            "checkpoint_architecture": checkpoint.get(
                "architecture",
                "pooled",
            ),
        },
        "inventory": {
            "sanpo_train": support_inventory,
            "public_transfer": public_inventory,
            "public_transfer_sources": positive_public_sources,
        },
        "model": {
            "frozen_hftf_context": True,
            "joint_pair_input": [
                "current_rgb",
                "baseline_rgb",
                "signed_rgb_delta",
                "absolute_rgb_delta",
            ],
            "pair_channels": list(PAIR_CHANNELS),
            "pair_grid": list(SPATIAL_GRID),
            "relation_hidden_channels": list(HIDDEN_CHANNELS),
            "trainable_parameter_count": trainable_parameter_count,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "deterministic_algorithms": True,
            "tf32": False,
            "hyperparameter_search": False,
        },
        "training": {
            "source_domain": "SANPO_CONSUMED_SUPPORT_ONLY",
            "public_frames_used_for_training": 0,
            "initial_epoch_loss": epoch_losses[0],
            "final_epoch_loss": epoch_losses[-1],
            "epoch_losses": epoch_losses,
        },
        "evaluation": {
            "target_domain": (
                "PUBLIC_INTERVENTION_BEARING_SOURCES"
            ),
            "direct_transfer_without_public_finetuning": True,
            "train_transfer_source_disjoint": True,
            "threshold": 0.5,
            "uses_transfer_no_alert_truth_for_reference": True,
            "system_authority": False,
            "metrics": metrics,
            "metrics_by_source": metrics_by_source,
            "source_macro_metrics": source_macro_metrics,
        },
        "source_references": source_references,
        "evidence_limit": (
            "Consumed Development representation canary. Only "
            "SANPO support rows update the pair stem and relation "
            "head. Public no-alert labels construct target-source "
            "references, but no public row updates parameters."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(str(args.output) + ".sha256").write_text(
        sha256(args.output) + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "device": str(device),
                "trainable_parameter_count": (
                    trainable_parameter_count
                ),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "episodes"
                },
                "source_macro_metrics": source_macro_metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
