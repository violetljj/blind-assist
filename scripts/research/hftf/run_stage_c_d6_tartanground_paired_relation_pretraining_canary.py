#!/usr/bin/env python3
"""Train paired RGB relation features on TartanGround and transfer by parent."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf

from evaluate_stage_c_d5_tartanground_event_proxy import (
    decode_labels,
    lane_truth_state,
)
from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_PRETRAINED,
    load_model,
)
from run_stage_c_d6_paired_rgb_relation_backbone_canary import (
    BACKBONE_LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    HEAD_LEARNING_RATE,
    PairedRgbRelationModel,
    WEIGHT_DECAY,
    configure_deterministic_torch,
    load_frame_tensors,
    score_batch,
)
from run_stage_c_d6_source_centered_relation_encoder_canary import (
    RelationEncoder,
    summarize_predictions,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    event_balanced_weights,
)
from train_stage_c_d5_tartanground_development_student import (
    sha256,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d6_tartanground_"
    "paired_relation_pretraining_transfer_canary_v1"
)
DEFAULT_TRAIN_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-cross-environment-v1/"
    "fold-0/samples.jsonl"
)
DEFAULT_TRANSFER_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-outcome-unseen-transfer-v0/"
    "samples.jsonl"
)
EXPECTED_TRAIN_PARENTS = 6
EXPECTED_TRAIN_FRAMES = 193
EXPECTED_TRANSFER_PARENTS = 2
EXPECTED_TRANSFER_FRAMES = 66
SEED = 1706


def current_central_body_label(
    record: dict[str, Any],
) -> int | None:
    risk, known = decode_labels(record)
    risk_array = risk.numpy()
    known_array = known.numpy()
    states = [
        lane_truth_state(
            risk_array[0, 1, direction],
            known_array[0, 1, direction],
        )
        for direction in (2, 3)
    ]
    if True in states:
        return 1
    if states == [False, False]:
        return 0
    return None


def collect_paired_tartanground_episodes(
    samples_path: Path,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [
        json.loads(line)
        for line in samples_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    by_parent: dict[str, dict[int, list[dict[str, Any]]]] = {}
    ambiguous_count = 0
    for record in records:
        label = current_central_body_label(record)
        if label is None:
            ambiguous_count += 1
            continue
        by_parent.setdefault(
            str(record["parent_id"]),
            {0: [], 1: []},
        )[label].append(record)
    paired = {
        parent_id: labels
        for parent_id, labels in by_parent.items()
        if labels[0] and labels[1]
    }
    episodes = []
    for parent_id in sorted(paired):
        for label in (0, 1):
            rows = sorted(
                paired[parent_id][label],
                key=lambda row: int(row["anchor_frame_id"]),
            )
            frames = []
            for row in rows:
                current = row["history_rgb"][-1]
                if float(current["relative_time_s"]) != 0.0:
                    raise ValueError(
                        f"Current frame drift: {row['sample_id']}"
                    )
                path = Path(current["image_path"])
                if not path.is_file():
                    raise OSError(
                        f"Missing TartanGround frame: {path}"
                    )
                frames.append(
                    {
                        "image_path": str(path.resolve()),
                        "timestamp_ms": (
                            int(row["anchor_frame_id"]) * 100
                        ),
                    }
                )
            episodes.append(
                {
                    "episode_id": (
                        f"tartanground-{prefix}/"
                        f"{parent_id}/label-{label}"
                    ),
                    "source_id": parent_id,
                    "label": label,
                    "frames": frames,
                }
            )
    return episodes, {
        "record_count": len(records),
        "ambiguous_record_count": ambiguous_count,
        "labeled_parent_count": len(by_parent),
        "paired_parent_count": len(paired),
        "paired_parents": sorted(paired),
        "paired_frame_count": sum(
            len(episode["frames"]) for episode in episodes
        ),
    }


def train_and_evaluate(
    base_backbone: torch.nn.Module,
    tensors: torch.Tensor,
    labels: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
    reference_indices: dict[str, list[int]],
    train_rows: np.ndarray,
    transfer_rows: np.ndarray,
    device: torch.device,
    initial_model: PairedRgbRelationModel | None = None,
) -> tuple[
    PairedRgbRelationModel,
    np.ndarray,
    list[float],
]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
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
    train = np.flatnonzero(train_rows)
    transfer = np.flatnonzero(transfer_rows)
    sample_weights = event_balanced_weights(
        episode_ids[train].tolist(),
        labels[train],
    ).astype(np.float32)
    weight_by_index = np.zeros(len(labels), dtype=np.float32)
    weight_by_index[train] = sample_weights
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
                reference_indices,
                device,
            )
            targets = torch.from_numpy(
                labels[batch].astype(np.float32)
            ).to(device)
            weights = torch.from_numpy(
                weight_by_index[batch]
            ).to(device)
            weighted = (
                nnf.binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    reduction="none",
                )
                * weights
            ).sum()
            loss = weighted / weights.sum()
            loss.backward()
            optimizer.step()
            loss_sum_epoch += float(weighted.detach().cpu())
            weight_sum_epoch += float(weights.sum().detach().cpu())
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
                reference_indices,
                device,
            )
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy()
            )
    return model, np.concatenate(probabilities), epoch_losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-samples",
        type=Path,
        default=DEFAULT_TRAIN_SAMPLES,
    )
    parser.add_argument(
        "--transfer-samples",
        type=Path,
        default=DEFAULT_TRANSFER_SAMPLES,
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    args = parser.parse_args()
    configure_deterministic_torch()
    if (
        args.output.exists()
        or args.output_model.exists()
        or Path(str(args.output) + ".sha256").exists()
    ):
        raise ValueError(
            "Refusing to overwrite TartanGround relation output"
        )

    train_episodes, train_inventory = (
        collect_paired_tartanground_episodes(
            args.train_samples,
            "train",
        )
    )
    transfer_episodes, transfer_inventory = (
        collect_paired_tartanground_episodes(
            args.transfer_samples,
            "transfer",
        )
    )
    if (
        train_inventory["paired_parent_count"]
        != EXPECTED_TRAIN_PARENTS
        or train_inventory["paired_frame_count"]
        != EXPECTED_TRAIN_FRAMES
        or transfer_inventory["paired_parent_count"]
        != EXPECTED_TRANSFER_PARENTS
        or transfer_inventory["paired_frame_count"]
        != EXPECTED_TRANSFER_FRAMES
    ):
        raise ValueError(
            "Unexpected paired TartanGround inventory"
        )

    episodes = train_episodes + transfer_episodes
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
        "tartanground-train/",
    )
    transfer_rows = np.char.startswith(
        episode_ids,
        "tartanground-transfer/",
    )
    if set(sources[train_rows]) & set(sources[transfer_rows]):
        raise ValueError("Train/transfer parent overlap")

    backbone, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, probabilities, epoch_losses = train_and_evaluate(
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
    metrics = summarize_predictions(
        probabilities,
        labels[transfer_rows],
        sources[transfer_rows],
        episode_ids[transfer_rows],
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": (
                "blindassist_hftf_stage_c_d6_tartanground_"
                "paired_relation_pretrained_state_v1"
            ),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "train_samples_sha256": sha256(
                args.train_samples
            ),
            "base_checkpoint_sha256": sha256(args.checkpoint),
        },
        args.output_model,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TARTANGROUND_PAIRED_RELATION_TRANSFER_COMPLETE",
        "inputs": {
            "train_samples_path": str(
                args.train_samples.resolve()
            ),
            "train_samples_sha256": sha256(args.train_samples),
            "transfer_samples_path": str(
                args.transfer_samples.resolve()
            ),
            "transfer_samples_sha256": sha256(
                args.transfer_samples
            ),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
            "checkpoint_architecture": checkpoint.get(
                "architecture",
                "pooled",
            ),
        },
        "label": {
            "positive": (
                "current body height has risk truth in central "
                "direction 2 or 3"
            ),
            "negative": (
                "both current body central directions are fully "
                "known and have no risk truth"
            ),
            "ambiguous_excluded": True,
        },
        "train_inventory": train_inventory,
        "transfer_inventory": transfer_inventory,
        "model": {
            "paired_rgb_tail": True,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "relation_parameter_count": sum(
                parameter.numel()
                for parameter in RelationEncoder().parameters()
            ),
            "deterministic_algorithms": True,
            "tf32": False,
            "hyperparameter_search": False,
        },
        "training": {
            "initial_epoch_loss": epoch_losses[0],
            "final_epoch_loss": epoch_losses[-1],
            "epoch_losses": epoch_losses,
            "pretrained_state_path": str(
                args.output_model.resolve()
            ),
            "pretrained_state_sha256": sha256(
                args.output_model
            ),
        },
        "evaluation": {
            "split_unit": "parent_id",
            "train_transfer_parent_disjoint": True,
            "transfer_parent_count": (
                EXPECTED_TRANSFER_PARENTS
            ),
            "uses_transfer_no_alert_truth_for_reference": True,
            "system_authority": False,
            "metrics": metrics,
        },
        "source_references": source_references,
        "evidence_limit": (
            "Synthetic Development paired-relation pretraining "
            "diagnosis. Transfer no-alert truth is used to construct "
            "references. This is not real-event, safety, App, or "
            "promotion evidence."
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
