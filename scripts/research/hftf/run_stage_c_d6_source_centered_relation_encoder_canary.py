#!/usr/bin/env python3
"""Test a fixed nonlinear relation encoder on source-centered HFTF grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_PRETRAINED,
    ManifestFrames,
    load_model,
)
from run_stage_c_d6_provisional_relation_transfer import (
    MixedRelationFrames,
    PUBLIC_VIDEO_STEP_MS,
    collect_public_video_actionability_episodes,
)
from run_stage_c_d6_sanpo_spatial_relation_head import (
    SPATIAL_GRID,
    infer_spatial_matrices,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    event_balanced_weights,
    training_phase_labels,
)
from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_source_centered_"
    "relation_encoder_canary_v0"
)
SUPPORT_SCHEMA = (
    "blindassist_hftf_stage_c_d6_source_centered_"
    "relation_encoder_sanpo_support_canary_v1"
)
EXPECTED_EPISODES = 28
EXPECTED_SOURCES = 11
EXPECTED_FRAMES = 436
SEED = 17
EPOCHS = 200
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3
HIDDEN_CHANNELS = (32, 16)
DEFAULT_ACTIONABILITY_MANIFEST = Path(
    "artifacts.local/evidence/"
    "public-video-r789-actionability-manifest-20260719/"
    "actionability_manifest_r789.json"
)
DEFAULT_FEATURE_CONTRACT = Path(
    "configs/public_video_actionability_linear_probe_contract_r790.json"
)


class RelationEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv2d(256, HIDDEN_CHANNELS[0], 1)
        self.spatial = nn.Conv2d(
            HIDDEN_CHANNELS[0],
            HIDDEN_CHANNELS[1],
            3,
            padding=1,
        )
        self.output = nn.Linear(
            HIDDEN_CHANNELS[1]
            * SPATIAL_GRID[0]
            * SPATIAL_GRID[1],
            1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = nnf.gelu(self.project(value))
        value = nnf.gelu(self.spatial(value))
        return self.output(value.flatten(1)).squeeze(1)


def stable_fold_seed(source_id: str) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:2], "big")


def build_relation_rows(
    episodes: list[dict[str, Any]],
    matrices: list[np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    if len(episodes) != len(matrices):
        raise ValueError("Episode/feature count mismatch")
    baselines: dict[str, np.ndarray] = {}
    baseline_rows = []
    for source_id in sorted(
        {str(episode["source_id"]) for episode in episodes}
    ):
        no_alert_means = [
            np.mean(matrix, axis=0)
            for episode, matrix in zip(
                episodes,
                matrices,
                strict=True,
            )
            if episode["source_id"] == source_id
            and episode["label"] == 0
        ]
        if not no_alert_means:
            raise ValueError(
                f"Source lacks no-alert baseline: {source_id}"
            )
        baseline = np.mean(
            np.stack(no_alert_means),
            axis=0,
        )
        baselines[source_id] = baseline
        baseline_rows.append(
            {
                "source_id": source_id,
                "no_alert_episode_count": len(no_alert_means),
                "baseline_l2_norm": float(
                    np.linalg.norm(baseline)
                ),
            }
        )

    features = []
    labels = []
    sources = []
    episode_ids = []
    for episode, matrix in zip(
        episodes,
        matrices,
        strict=True,
    ):
        source_id = str(episode["source_id"])
        delta = matrix - baselines[source_id]
        grid = delta.reshape(
            len(delta),
            128,
            SPATIAL_GRID[0],
            SPATIAL_GRID[1],
        )
        relation = np.concatenate(
            (grid, np.abs(grid)),
            axis=1,
        )
        features.append(relation)
        labels.extend([int(episode["label"])] * len(matrix))
        sources.extend([source_id] * len(matrix))
        episode_ids.extend(
            [str(episode["episode_id"])] * len(matrix)
        )
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sources, dtype=str),
        np.asarray(episode_ids, dtype=str),
        baseline_rows,
    )


def collect_sanpo_support_episodes(
    manifest: dict[str, Any],
    matrices: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if len(manifest["events"]) != len(matrices):
        raise ValueError("SANPO support event/feature count mismatch")
    episodes = []
    selected_matrices = []
    for event, matrix in zip(
        manifest["events"],
        matrices,
        strict=True,
    ):
        frame_labels = training_phase_labels(event)
        for label in (0, 1):
            indices = sorted(
                index
                for index, value in frame_labels.items()
                if value == label
            )
            if not indices:
                continue
            episodes.append(
                {
                    "episode_id": (
                        "sanpo-consumed-support/"
                        f"{event['parent_event_id']}/label-{label}"
                    ),
                    "source_id": event["source_session_id"],
                    "label": label,
                }
            )
            selected_matrices.append(matrix[indices])
    return episodes, selected_matrices


def train_fold(
    features: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
    held_out_source: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.flatnonzero(sources != held_out_source)
    test = np.flatnonzero(sources == held_out_source)
    if sorted(set(labels[train].tolist())) != [0, 1]:
        raise ValueError(
            f"Training fold lacks a class: {held_out_source}"
        )
    channel_mean = features[train].mean(
        axis=(0, 2, 3),
        keepdims=True,
    )
    channel_scale = features[train].std(
        axis=(0, 2, 3),
        keepdims=True,
    )
    channel_scale[channel_scale < 1e-6] = 1.0
    train_x = torch.from_numpy(
        (features[train] - channel_mean) / channel_scale
    ).to(device)
    test_x = torch.from_numpy(
        (features[test] - channel_mean) / channel_scale
    ).to(device)
    train_y = torch.from_numpy(
        labels[train].astype(np.float32)
    ).to(device)
    sample_weights = event_balanced_weights(
        episode_ids[train].tolist(),
        labels[train],
    ).astype(np.float32)
    train_weights = torch.from_numpy(sample_weights).to(device)

    seed = stable_fold_seed(held_out_source)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = RelationEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    initial_loss = None
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = (
            nnf.binary_cross_entropy_with_logits(
                logits,
                train_y,
                reduction="none",
            )
            * train_weights
        ).sum() / train_weights.sum()
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(test_x)).cpu().numpy()
    return probabilities, {
        "held_out_source_id": held_out_source,
        "seed": seed,
        "train_frame_count": len(train),
        "test_frame_count": len(test),
        "initial_loss": initial_loss,
        "final_loss": float(loss.detach().cpu()),
    }


def summarize_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    frame_alert_recall = float(
        (predictions[labels == 1] == 1).mean()
    )
    frame_no_alert_recall = float(
        (predictions[labels == 0] == 0).mean()
    )
    episode_rows = []
    for episode_id in sorted(set(episode_ids.tolist())):
        rows = np.flatnonzero(episode_ids == episode_id)
        episode_labels = sorted(set(labels[rows].tolist()))
        if len(episode_labels) != 1:
            raise ValueError(
                f"Episode has mixed labels: {episode_id}"
            )
        score = float(probabilities[rows].mean())
        episode_rows.append(
            {
                "episode_id": episode_id,
                "source_id": str(sources[rows[0]]),
                "label": episode_labels[0],
                "score": score,
                "prediction": int(score >= 0.5),
            }
        )
    episode_labels = np.asarray(
        [row["label"] for row in episode_rows],
        dtype=np.int64,
    )
    episode_predictions = np.asarray(
        [row["prediction"] for row in episode_rows],
        dtype=np.int64,
    )
    episode_alert_recall = float(
        (
            episode_predictions[episode_labels == 1]
            == 1
        ).mean()
    )
    episode_no_alert_recall = float(
        (
            episode_predictions[episode_labels == 0]
            == 0
        ).mean()
    )
    return {
        "threshold": 0.5,
        "frame_alert_recall": frame_alert_recall,
        "frame_no_alert_recall": frame_no_alert_recall,
        "frame_balanced_accuracy": (
            frame_alert_recall + frame_no_alert_recall
        )
        / 2.0,
        "episode_alert_recall": episode_alert_recall,
        "episode_no_alert_recall": episode_no_alert_recall,
        "episode_balanced_accuracy": (
            episode_alert_recall + episode_no_alert_recall
        )
        / 2.0,
        "episodes": episode_rows,
    }


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
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sanpo-support-manifest",
        type=Path,
    )
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output_cache.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError("Refusing to overwrite relation canary outputs")

    episodes = collect_public_video_actionability_episodes(
        args.actionability_manifest,
        args.feature_contract,
        PUBLIC_VIDEO_STEP_MS,
    )
    episode_count = len(episodes)
    source_count = len(
        {episode["source_id"] for episode in episodes}
    )
    frame_count = sum(
        len(episode["frames"]) for episode in episodes
    )
    if (
        episode_count != EXPECTED_EPISODES
        or source_count != EXPECTED_SOURCES
        or frame_count != EXPECTED_FRAMES
    ):
        raise ValueError(
            "Unexpected relation encoder inventory: "
            f"episodes={episode_count}, sources={source_count}, "
            f"frames={frame_count}"
        )

    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    manifest = {
        "events": [
            {"frames": episode["frames"]}
            for episode in episodes
        ]
    }
    dataset = MixedRelationFrames(manifest)
    matrices, feature_names = infer_spatial_matrices(
        model,
        dataset,
        manifest,
        args.batch_size,
    )
    dataset.close()
    public_episode_count = len(episodes)
    support_episodes = []
    support_matrices = []
    if args.sanpo_support_manifest is not None:
        support_manifest = json.loads(
            args.sanpo_support_manifest.read_text(
                encoding="utf-8"
            )
        )
        if (
            int(support_manifest["event_count"]) != 30
            or sum(
                len(event["frames"])
                for event in support_manifest["events"]
            )
            != 1920
        ):
            raise ValueError(
                "Expected the 30-event / 1,920-frame SANPO support view"
            )
        support_dataset = ManifestFrames(
            args.sanpo_support_manifest,
            support_manifest,
        )
        support_event_matrices, support_feature_names = (
            infer_spatial_matrices(
                model,
                support_dataset,
                support_manifest,
                args.batch_size,
            )
        )
        if feature_names != support_feature_names:
            raise ValueError("Public/SANPO feature order drift")
        support_episodes, support_matrices = (
            collect_sanpo_support_episodes(
                support_manifest,
                support_event_matrices,
            )
        )
        episodes.extend(support_episodes)
        matrices.extend(support_matrices)
    features, labels, sources, episode_ids, baselines = (
        build_relation_rows(episodes, matrices)
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_cache,
        features=features,
        labels=labels,
        sources=sources,
        episode_ids=episode_ids,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    public_rows = np.char.startswith(
        episode_ids,
        "public-video-actionability/",
    )
    if (
        len(set(episode_ids[public_rows].tolist()))
        != public_episode_count
    ):
        raise ValueError("Public evaluation episode drift")
    probabilities = np.full(
        len(labels),
        np.nan,
        dtype=np.float64,
    )
    folds = []
    public_sources = sorted(set(sources[public_rows].tolist()))
    for held_out_source in public_sources:
        test = np.flatnonzero(sources == held_out_source)
        fold_probabilities, fold = train_fold(
            features,
            labels,
            sources,
            episode_ids,
            held_out_source,
            device,
        )
        probabilities[test] = fold_probabilities
        folds.append(fold)
    if not np.isfinite(probabilities[public_rows]).all():
        raise ValueError("Non-finite relation encoder predictions")
    metrics = summarize_predictions(
        probabilities[public_rows],
        labels[public_rows],
        sources[public_rows],
        episode_ids[public_rows],
    )
    report = {
        "schema": (
            SUPPORT_SCHEMA
            if args.sanpo_support_manifest is not None
            else SCHEMA
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_CENTERED_RELATION_ENCODER_CANARY_COMPLETE",
        "inputs": {
            "actionability_manifest_path": str(
                args.actionability_manifest.resolve()
            ),
            "actionability_manifest_sha256": sha256(
                args.actionability_manifest
            ),
            "feature_contract_path": str(
                args.feature_contract.resolve()
            ),
            "feature_contract_sha256": sha256(
                args.feature_contract
            ),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
            "checkpoint_architecture": checkpoint.get(
                "architecture",
                "pooled",
            ),
        },
        "inventory": {
            "public_evaluation_episode_count": episode_count,
            "public_evaluation_source_count": source_count,
            "public_evaluation_frame_count": frame_count,
            "alert_episode_count": sum(
                episode["label"] == 1
                for episode in episodes[:public_episode_count]
            ),
            "no_alert_episode_count": sum(
                episode["label"] == 0
                for episode in episodes[:public_episode_count]
            ),
            "sanpo_consumed_support_used": (
                args.sanpo_support_manifest is not None
            ),
            "sanpo_support_episode_count": len(
                support_episodes
            ),
            "sanpo_support_source_count": len(
                {
                    episode["source_id"]
                    for episode in support_episodes
                }
            ),
            "sanpo_support_frame_count": sum(
                len(matrix) for matrix in support_matrices
            ),
        },
        "feature_cache": {
            "path": str(args.output_cache.resolve()),
            "sha256": sha256(args.output_cache),
            "shape": list(features.shape),
            "fixed_hftf_feature_count": len(feature_names),
        },
        "model": {
            "input": (
                "current_minus_episode_balanced_source_no_alert_"
                "baseline_and_absolute_delta"
            ),
            "grid": list(SPATIAL_GRID),
            "input_channels": 256,
            "hidden_channels": list(HIDDEN_CHANNELS),
            "parameter_count": sum(
                parameter.numel()
                for parameter in RelationEncoder().parameters()
            ),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "threshold": 0.5,
            "hyperparameter_search": False,
        },
        "evaluation": {
            "split_unit": "public_video_source_id",
            "sanpo_support_sources_used_for_evaluation": False,
            "uses_held_out_source_no_alert_labels_for_baseline": True,
            "system_authority": False,
            "folds": folds,
            "metrics": metrics,
        },
        "source_baselines": baselines,
        "evidence_limit": (
            "Consumed Development oracle canary. The held-out source "
            "no-alert labels make this unsuitable for system, safety, "
            "calibration, App, or promotion claims."
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
