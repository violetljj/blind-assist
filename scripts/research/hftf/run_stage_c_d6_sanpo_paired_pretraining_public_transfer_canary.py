#!/usr/bin/env python3
"""Train paired RGB relations on SANPO and transfer to public sources."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    load_model,
)
from run_stage_c_d6_paired_rgb_relation_backbone_canary import (
    BATCH_SIZE,
    BACKBONE_LEARNING_RATE,
    EPOCHS,
    HEAD_LEARNING_RATE,
    PairedRgbRelationModel,
    WEIGHT_DECAY,
    collect_sanpo_rgb_support_episodes,
    configure_deterministic_torch,
    load_frame_tensors,
)
from run_stage_c_d6_provisional_relation_transfer import (
    PUBLIC_VIDEO_STEP_MS,
    collect_public_video_actionability_episodes,
)
from run_stage_c_d6_source_centered_relation_encoder_canary import (
    DEFAULT_ACTIONABILITY_MANIFEST,
    DEFAULT_FEATURE_CONTRACT,
    HIDDEN_CHANNELS,
    RelationEncoder,
    summarize_predictions,
)
from run_stage_c_d6_tartanground_paired_relation_pretraining_canary import (
    train_and_evaluate,
)
from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_sanpo_paired_"
    "pretraining_public_transfer_canary_v2"
)
PAIRED_PRETRAINED_STATE_SCHEMA = (
    "blindassist_hftf_stage_c_d6_tartanground_"
    "paired_relation_pretrained_state_v1"
)
EXPECTED_SUPPORT_EPISODES = 46
EXPECTED_SUPPORT_SOURCES = 30
EXPECTED_SUPPORT_FRAMES = 711
EXPECTED_PUBLIC_EPISODES = 18
EXPECTED_PUBLIC_SOURCES = 3
EXPECTED_PUBLIC_FRAMES = 272


def select_intervention_bearing_public_episodes(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    positive_sources = sorted(
        {
            str(episode["source_id"])
            for episode in episodes
            if int(episode["label"]) == 1
        }
    )
    selected = [
        episode
        for episode in episodes
        if str(episode["source_id"]) in positive_sources
    ]
    return selected, positive_sources


def inventory(episodes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "episode_count": len(episodes),
        "source_count": len(
            {str(episode["source_id"]) for episode in episodes}
        ),
        "frame_count": sum(
            len(episode["frames"]) for episode in episodes
        ),
    }


def summarize_by_source(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    summaries = {}
    for source_id in sorted(set(sources.tolist())):
        selected = sources == source_id
        summaries[str(source_id)] = summarize_predictions(
            probabilities[selected],
            labels[selected],
            sources[selected],
            episode_ids[selected],
        )
    metric_names = [
        "frame_alert_recall",
        "frame_no_alert_recall",
        "frame_balanced_accuracy",
        "frame_auroc",
        "episode_alert_recall",
        "episode_no_alert_recall",
        "episode_balanced_accuracy",
        "episode_auroc",
    ]
    macro = {
        name: float(
            np.mean(
                [
                    float(summary[name])
                    for summary in summaries.values()
                ]
            )
        )
        for name in metric_names
    }
    return summaries, macro


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
    parser.add_argument("--paired-pretrained-state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    args = parser.parse_args()
    configure_deterministic_torch()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError(
            "Refusing to overwrite SANPO-to-public transfer output"
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
    if sorted(set(labels[train_rows].tolist())) != [0, 1]:
        raise ValueError("SANPO training rows lack a class")
    if sorted(set(labels[transfer_rows].tolist())) != [0, 1]:
        raise ValueError("Public transfer rows lack a class")

    backbone, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    initial_model = None
    paired_pretrained_state = None
    if args.paired_pretrained_state is not None:
        paired_pretrained_state = torch.load(
            args.paired_pretrained_state,
            map_location="cpu",
            weights_only=False,
        )
        if paired_pretrained_state.get("schema") != (
            PAIRED_PRETRAINED_STATE_SCHEMA
        ):
            raise ValueError(
                "Unexpected paired pretrained state schema"
            )
        if paired_pretrained_state.get(
            "base_checkpoint_sha256"
        ) != sha256(args.checkpoint):
            raise ValueError(
                "Paired pretrained state checkpoint mismatch"
            )
        initial_model = PairedRgbRelationModel(
            copy.deepcopy(backbone)
        )
        initial_model.load_state_dict(
            paired_pretrained_state["model_state_dict"]
        )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    _, probabilities, epoch_losses = train_and_evaluate(
        backbone,
        tensors,
        labels,
        sources,
        episode_ids,
        reference_indices,
        train_rows,
        transfer_rows,
        device,
        initial_model,
    )
    metrics = summarize_predictions(
        probabilities,
        labels[transfer_rows],
        sources[transfer_rows],
        episode_ids[transfer_rows],
    )
    metrics_by_source, source_macro_metrics = (
        summarize_by_source(
            probabilities,
            labels[transfer_rows],
            sources[transfer_rows],
            episode_ids[transfer_rows],
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "SANPO_PAIRED_PRETRAINING_PUBLIC_TRANSFER_COMPLETE"
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
            "paired_pretrained_state_path": (
                str(args.paired_pretrained_state.resolve())
                if args.paired_pretrained_state is not None
                else None
            ),
            "paired_pretrained_state_sha256": (
                sha256(args.paired_pretrained_state)
                if args.paired_pretrained_state is not None
                else None
            ),
            "paired_pretraining_train_samples_sha256": (
                paired_pretrained_state.get(
                    "train_samples_sha256"
                )
                if paired_pretrained_state is not None
                else None
            ),
        },
        "inventory": {
            "sanpo_train": support_inventory,
            "public_transfer": public_inventory,
            "public_transfer_sources": positive_public_sources,
        },
        "model": {
            "pair": (
                "current_rgb_vs_episode_balanced_no_alert_"
                "reference_embeddings"
            ),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "relation_parameter_count": sum(
                parameter.numel()
                for parameter in RelationEncoder().parameters()
            ),
            "hidden_channels": list(HIDDEN_CHANNELS),
            "deterministic_algorithms": True,
            "tf32": False,
            "hyperparameter_search": False,
            "tartanground_paired_pretraining_used": (
                initial_model is not None
            ),
        },
        "training": {
            "source_domain": "SANPO_CONSUMED_SUPPORT_ONLY",
            "public_frames_used_for_training": 0,
            "initialization": (
                "TARTANGROUND_PAIRED_RELATION_PRETRAINED"
                if initial_model is not None
                else "BASE_HFTF_CHECKPOINT"
            ),
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
            "Consumed Development direct-transfer diagnosis. "
            "Only SANPO support rows train the model. Public "
            "no-alert labels construct target-source references, "
            "but no public row updates model parameters. This is "
            "not safety, App, or promotion evidence."
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
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "episodes"
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
