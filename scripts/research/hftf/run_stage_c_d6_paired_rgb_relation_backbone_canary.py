#!/usr/bin/env python3
"""Fine-tune the HFTF tail on paired current/no-alert RGB relations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    load_model,
    single_frame_spatial_features,
)
from run_stage_c_d6_provisional_relation_transfer import (
    MixedRelationFrames,
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
from run_stage_c_d6_sanpo_spatial_relation_head import SPATIAL_GRID
from run_stage_c_d6_sanpo_weak_relation_head import (
    event_balanced_weights,
    training_phase_labels,
)
from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_paired_rgb_"
    "relation_backbone_positive_source_canary_v3"
)
SEED = 17
EPOCHS = 10
BATCH_SIZE = 48
BACKBONE_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3
UNFROZEN_ENCODER_START = 9
EXPECTED_PUBLIC_EPISODES = 28
EXPECTED_PUBLIC_SOURCES = 11
EXPECTED_PUBLIC_FRAMES = 436
EXPECTED_SUPPORT_EPISODES = 46
EXPECTED_SUPPORT_SOURCES = 30
EXPECTED_SUPPORT_FRAMES = 711


class PairedRgbRelationModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.relation = RelationEncoder()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for module in self.backbone.encoder[
            UNFROZEN_ENCODER_START:
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.pointwise.parameters():
            parameter.requires_grad = True

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        spatial = single_frame_spatial_features(
            self.backbone,
            frames,
        )
        return nnf.interpolate(
            spatial,
            size=SPATIAL_GRID,
            mode="bilinear",
            align_corners=False,
        )

    def score(
        self,
        current: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        delta = current - baseline
        return self.relation(
            torch.cat((delta, delta.abs()), dim=1)
        )

    def train(self, mode: bool = True) -> "PairedRgbRelationModel":
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


def stable_fold_seed(source_id: str) -> int:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:2], "big")


def configure_deterministic_torch() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def collect_sanpo_rgb_support_episodes(
    manifest_path: Path,
) -> list[dict[str, Any]]:
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        int(manifest["event_count"]) != 30
        or sum(
            len(event["frames"])
            for event in manifest["events"]
        )
        != 1920
    ):
        raise ValueError(
            "Expected the 30-event / 1,920-frame SANPO support view"
        )
    episodes = []
    for event in manifest["events"]:
        labels = training_phase_labels(event)
        for label in (0, 1):
            indices = sorted(
                index
                for index, value in labels.items()
                if value == label
            )
            if not indices:
                continue
            frames = []
            for index in indices:
                frame = event["frames"][index]
                image_path = (
                    manifest_path.parent / frame["image_path"]
                ).resolve()
                if not image_path.is_file():
                    raise OSError(
                        f"Missing SANPO support frame: {image_path}"
                    )
                frames.append(
                    {
                        "image_path": str(image_path),
                        "timestamp_ms": int(
                            frame["timestamp_ms"]
                        ),
                    }
                )
            episodes.append(
                {
                    "episode_id": (
                        "sanpo-consumed-support/"
                        f"{event['parent_event_id']}/label-{label}"
                    ),
                    "source_id": event["source_session_id"],
                    "label": label,
                    "frames": frames,
                }
            )
    return episodes


def load_frame_tensors(
    episodes: list[dict[str, Any]],
    batch_size: int,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, list[int]],
    list[dict[str, Any]],
]:
    manifest = {
        "events": [
            {"frames": episode["frames"]}
            for episode in episodes
        ]
    }
    dataset = MixedRelationFrames(manifest)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    tensors = []
    for frames, _, _ in loader:
        tensors.append(frames)
    dataset.close()
    values = torch.cat(tensors, dim=0).contiguous()

    labels = []
    sources = []
    episode_ids = []
    episode_ranges = []
    offset = 0
    for episode in episodes:
        count = len(episode["frames"])
        labels.extend([int(episode["label"])] * count)
        sources.extend([str(episode["source_id"])] * count)
        episode_ids.extend(
            [str(episode["episode_id"])] * count
        )
        episode_ranges.append(
            {
                "episode_id": str(episode["episode_id"]),
                "source_id": str(episode["source_id"]),
                "label": int(episode["label"]),
                "start": offset,
                "end": offset + count,
            }
        )
        offset += count

    reference_indices: dict[str, list[int]] = {}
    baseline_rows = []
    for source_id in sorted(set(sources)):
        no_alert = [
            row
            for row in episode_ranges
            if row["source_id"] == source_id
            and row["label"] == 0
        ]
        if not no_alert:
            raise ValueError(
                f"Source lacks no-alert RGB reference: {source_id}"
            )
        indices = [
            (row["start"] + row["end"] - 1) // 2
            for row in no_alert
        ]
        reference_indices[source_id] = indices
        baseline_rows.append(
            {
                "source_id": source_id,
                "no_alert_episode_count": len(no_alert),
                "reference_frame_indices": indices,
            }
        )
    return (
        values,
        np.asarray(labels, dtype=np.int64),
        np.asarray(sources, dtype=str),
        np.asarray(episode_ids, dtype=str),
        reference_indices,
        baseline_rows,
    )


def score_batch(
    model: PairedRgbRelationModel,
    tensors: torch.Tensor,
    indices: np.ndarray,
    sources: np.ndarray,
    reference_indices: dict[str, list[int]],
    device: torch.device,
) -> torch.Tensor:
    batch_sources = sources[indices]
    unique_sources = sorted(set(batch_sources.tolist()))
    flat_references = [
        index
        for source_id in unique_sources
        for index in reference_indices[source_id]
    ]
    current_count = len(indices)
    combined_indices = np.concatenate(
        (
            indices,
            np.asarray(flat_references, dtype=np.int64),
        )
    )
    encoded = model.encode(
        tensors[combined_indices].to(
            device,
            non_blocking=True,
        )
    )
    current = encoded[:current_count]
    reference_features = encoded[current_count:]
    source_baselines: dict[str, torch.Tensor] = {}
    offset = 0
    for source_id in unique_sources:
        count = len(reference_indices[source_id])
        source_baselines[source_id] = reference_features[
            offset : offset + count
        ].mean(dim=0)
        offset += count
    baseline = torch.stack(
        [
            source_baselines[str(source_id)]
            for source_id in batch_sources
        ]
    )
    return model.score(current, baseline)


def train_positive_source_fold(
    base_backbone: nn.Module,
    tensors: torch.Tensor,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
    reference_indices: dict[str, list[int]],
    public_rows: np.ndarray,
    held_out_source: str,
    device: torch.device,
    initial_model: PairedRgbRelationModel | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.flatnonzero(sources != held_out_source)
    test = np.flatnonzero(
        public_rows & (sources == held_out_source)
    )
    if sorted(set(labels[train].tolist())) != [0, 1]:
        raise ValueError(
            f"Paired RGB fold lacks a class: {held_out_source}"
        )
    seed = stable_fold_seed(held_out_source)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = (
        copy.deepcopy(initial_model)
        if initial_model is not None
        else PairedRgbRelationModel(copy.deepcopy(base_backbone))
    ).to(device)
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("relation.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
            },
            {
                "params": model.relation.parameters(),
                "lr": HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sample_weights = event_balanced_weights(
        episode_ids[train].tolist(),
        labels[train],
    ).astype(np.float32)
    weight_by_index = np.zeros(len(labels), dtype=np.float32)
    weight_by_index[train] = sample_weights
    generator = np.random.default_rng(seed)
    epoch_losses = []
    for _ in range(EPOCHS):
        model.train()
        order = generator.permutation(train)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = score_batch(
                model,
                tensors,
                batch,
                sources,
                reference_indices,
                device,
            )
            targets = torch.from_numpy(
                labels[batch].astype(np.float32)
            ).to(device)
            weights = torch.from_numpy(
                weight_by_index[batch]
            ).to(device)
            loss_sum = (
                nnf.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
                * weights
            ).sum()
            loss = loss_sum / weights.sum()
            loss.backward()
            optimizer.step()
            total_loss += float(loss_sum.detach().cpu())
            total_weight += float(weights.sum().detach().cpu())
        epoch_losses.append(total_loss / total_weight)

    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(test), BATCH_SIZE):
            batch = test[start : start + BATCH_SIZE]
            logits = score_batch(
                model,
                tensors,
                batch,
                sources,
                reference_indices,
                device,
            )
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )
    return np.concatenate(probabilities), {
        "held_out_source_id": held_out_source,
        "seed": seed,
        "train_frame_count": len(train),
        "test_frame_count": len(test),
        "initial_epoch_loss": epoch_losses[0],
        "final_epoch_loss": epoch_losses[-1],
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
    parser.add_argument(
        "--paired-pretrained-state",
        type=Path,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    args = parser.parse_args()
    configure_deterministic_torch()
    if (
        args.output.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError("Refusing to overwrite paired RGB output")

    public_episodes = collect_public_video_actionability_episodes(
        args.actionability_manifest,
        args.feature_contract,
        PUBLIC_VIDEO_STEP_MS,
    )
    support_episodes = collect_sanpo_rgb_support_episodes(
        args.sanpo_support_manifest
    )
    public_frames = sum(
        len(episode["frames"]) for episode in public_episodes
    )
    support_frames = sum(
        len(episode["frames"]) for episode in support_episodes
    )
    if (
        len(public_episodes) != EXPECTED_PUBLIC_EPISODES
        or len(
            {
                episode["source_id"]
                for episode in public_episodes
            }
        )
        != EXPECTED_PUBLIC_SOURCES
        or public_frames != EXPECTED_PUBLIC_FRAMES
        or len(support_episodes) != EXPECTED_SUPPORT_EPISODES
        or len(
            {
                episode["source_id"]
                for episode in support_episodes
            }
        )
        != EXPECTED_SUPPORT_SOURCES
        or support_frames != EXPECTED_SUPPORT_FRAMES
    ):
        raise ValueError("Paired RGB inventory drift")

    episodes = public_episodes + support_episodes
    (
        tensors,
        labels,
        sources,
        episode_ids,
        reference_indices,
        baseline_rows,
    ) = load_frame_tensors(episodes, args.decode_batch_size)
    public_rows = np.char.startswith(
        episode_ids,
        "public-video-actionability/",
    )
    positive_public_sources = sorted(
        {
            str(episode["source_id"])
            for episode in public_episodes
            if episode["label"] == 1
        }
    )
    if len(positive_public_sources) != 3:
        raise ValueError(
            "Expected exactly three positive public sources"
        )

    base_backbone, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    initial_model = None
    pretrained_state = None
    if args.paired_pretrained_state is not None:
        pretrained_state = torch.load(
            args.paired_pretrained_state,
            map_location="cpu",
            weights_only=False,
        )
        if pretrained_state.get("schema") != (
            "blindassist_hftf_stage_c_d6_tartanground_"
            "paired_relation_pretrained_state_v1"
        ):
            raise ValueError(
                "Unexpected paired pretrained state schema"
            )
        if pretrained_state.get(
            "base_checkpoint_sha256"
        ) != sha256(args.checkpoint):
            raise ValueError(
                "Paired pretrained state checkpoint mismatch"
            )
        initial_model = PairedRgbRelationModel(
            copy.deepcopy(base_backbone)
        )
        initial_model.load_state_dict(
            pretrained_state["model_state_dict"]
        )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    probabilities = np.full(
        len(labels),
        np.nan,
        dtype=np.float64,
    )
    folds = []
    for held_out_source in positive_public_sources:
        test = np.flatnonzero(
            public_rows & (sources == held_out_source)
        )
        fold_probabilities, fold = train_positive_source_fold(
            base_backbone,
            tensors,
            labels,
            sources,
            episode_ids,
            reference_indices,
            public_rows,
            held_out_source,
            device,
            initial_model,
        )
        probabilities[test] = fold_probabilities
        folds.append(fold)
    evaluated = public_rows & np.isin(
        sources,
        positive_public_sources,
    )
    if not np.isfinite(probabilities[evaluated]).all():
        raise ValueError("Incomplete paired RGB predictions")
    metrics = summarize_predictions(
        probabilities[evaluated],
        labels[evaluated],
        sources[evaluated],
        episode_ids[evaluated],
    )
    trainable_backbone_parameters = sum(
        parameter.numel()
        for name, parameter in PairedRgbRelationModel(
            copy.deepcopy(base_backbone)
        ).named_parameters()
        if parameter.requires_grad
        and not name.startswith("relation.")
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PAIRED_RGB_RELATION_BACKBONE_CANARY_COMPLETE",
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
                pretrained_state.get("train_samples_sha256")
                if pretrained_state is not None
                else None
            ),
        },
        "inventory": {
            "public_episode_count": len(public_episodes),
            "public_source_count": EXPECTED_PUBLIC_SOURCES,
            "public_frame_count": public_frames,
            "sanpo_support_episode_count": len(
                support_episodes
            ),
            "sanpo_support_source_count": (
                EXPECTED_SUPPORT_SOURCES
            ),
            "sanpo_support_frame_count": support_frames,
            "positive_public_source_count": len(
                positive_public_sources
            ),
            "evaluated_public_sources": positive_public_sources,
        },
        "model": {
            "pair": (
                "current_rgb_vs_episode_balanced_no_alert_"
                "reference_embeddings"
            ),
            "spatial_resize": (
                "deterministic_bilinear_4x7_to_3x6"
            ),
            "unfrozen_encoder_start": UNFROZEN_ENCODER_START,
            "trainable_backbone_parameter_count": (
                trainable_backbone_parameters
            ),
            "relation_parameter_count": sum(
                parameter.numel()
                for parameter in RelationEncoder().parameters()
            ),
            "hidden_channels": list(HIDDEN_CHANNELS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hyperparameter_search": False,
            "tartanground_paired_pretraining_used": (
                initial_model is not None
            ),
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "tf32": False,
        },
        "evaluation": {
            "split_unit": "positive_public_video_source_id",
            "threshold": 0.5,
            "sanpo_support_sources_used_for_evaluation": False,
            "uses_held_out_source_no_alert_labels_for_reference": True,
            "system_authority": False,
            "folds": folds,
            "metrics": metrics,
        },
        "source_references": baseline_rows,
        "evidence_limit": (
            "Consumed Development paired-RGB oracle canary. Only the "
            "three public sources containing intervention segments are "
            "evaluated. Held-out no-alert labels are used for reference "
            "construction; no system or promotion claim is authorized."
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
