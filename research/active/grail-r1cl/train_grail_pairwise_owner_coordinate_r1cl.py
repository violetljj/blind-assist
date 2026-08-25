#!/usr/bin/env python3
"""Train the sole two-seed GRAIL-R1C-L pairwise owner-coordinate model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from grail_pairwise_owner_coordinate_r1cl import (
    PairCollection, PairwiseOwnerCoordinate, exchange_consistency_loss,
    slot_marginalized_loss, slot_mode_correct,
)
from grail_procthor_native_m0 import sha256_file


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.inference_mode()
def evaluate(model: PairwiseOwnerCoordinate, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = correct = 0
    by_type: dict[str, list[int]] = {}
    loss_total = 0.0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch["reference_rgb"].to(device), batch["reference_masks"].to(device),
                           batch["query_rgb"].to(device), batch["query_masks"].to(device))
        valid = batch["valid_bins"].to(device)
        outcome = slot_mode_correct(logits, valid).cpu()
        batch_loss = slot_marginalized_loss(logits, valid).item()
        loss_total += batch_loss * len(outcome)
        total += len(outcome)
        correct += int(outcome.sum())
        for object_type, passed in zip(batch["object_type"], outcome.tolist()):
            counts = by_type.setdefault(object_type, [0, 0])
            counts[0] += int(passed)
            counts[1] += 1
    return {
        "slot_correct": correct, "slot_total": total, "slot_accuracy": correct / max(total, 1),
        "loss": loss_total / max(total, 1),
        "by_type": {key: {"correct": value[0], "total": value[1], "accuracy": value[0] / max(value[1], 1)}
                    for key, value in sorted(by_type.items())},
    }


def train_seed(seed: int, train_data: PairCollection, validation_data: PairCollection,
               backbone: Path, output: Path, epochs: int, batch_size: int, workers: int) -> dict[str, Any]:
    _seed(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("R1C-L CUDA_REQUIRED_NO_CPU_FALLBACK")
    seed_root = output / f"seed-{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    type_counts: dict[str, int] = {}
    for pair in train_data.pairs:
        type_counts[pair["object_type"]] = type_counts.get(pair["object_type"], 0) + 1
    weights = [1.0 / type_counts[pair["object_type"]] for pair in train_data.pairs]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    validation_loader = DataLoader(validation_data, batch_size=batch_size * 2, shuffle=False,
                                   num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    model = PairwiseOwnerCoordinate(backbone).to(device)
    backbone_parameters = [parameter for name, parameter in model.named_parameters()
                           if name.startswith("backbone.") and parameter.requires_grad]
    head_parameters = [parameter for name, parameter in model.named_parameters()
                       if not name.startswith("backbone.") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": 1e-5},
        {"params": head_parameters, "lr": 1e-4},
    ], weight_decay=0.05)
    last_path, best_path = seed_root / "last.pt", seed_root / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_accuracy = 0, -1.0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["next_epoch"])
        best_accuracy = float(checkpoint["best_accuracy"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        generator.set_state(checkpoint["sampler_random_state"])
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = running_slot = examples = 0.0
        for batch_number, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            reference_rgb = batch["reference_rgb"].to(device, non_blocking=True)
            reference_masks = batch["reference_masks"].to(device, non_blocking=True)
            query_rgb = batch["query_rgb"].to(device, non_blocking=True)
            query_masks = batch["query_masks"].to(device, non_blocking=True)
            valid = batch["valid_bins"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward, reverse = model(reference_rgb, reference_masks, query_rgb, query_masks, return_reverse=True)
                slot_loss = slot_marginalized_loss(forward, valid)
                exchange_loss = exchange_consistency_loss(forward, reverse)
                loss = slot_loss + 0.05 * exchange_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = reference_rgb.shape[0]
            running_loss += float(loss.detach()) * count
            running_slot += float(slot_loss.detach()) * count
            examples += count
            if batch_number % 100 == 0:
                _atomic_json(seed_root / "progress.json", {
                    "phase": "train", "seed": seed, "epoch": epoch + 1,
                    "completed_units": batch_number, "total_units": len(loader),
                    "throughput": examples / max(time.monotonic() - started, 1e-6),
                    "eta_seconds": None, "last_progress_at": time.time(), "status": "running",
                })
        validation = evaluate(model, validation_loader, device)
        epoch_row = {
            "epoch": epoch + 1, "train_loss": running_loss / max(examples, 1),
            "train_slot_loss": running_slot / max(examples, 1), "validation": validation,
        }
        history.append(epoch_row)
        if validation["slot_accuracy"] > best_accuracy:
            best_accuracy = validation["slot_accuracy"]
            _atomic_torch(best_path, {
                "schema": "blindassist_grail_r1c_l_checkpoint_v1", "seed": seed,
                "epoch": epoch + 1, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "validation": validation,
            })
        _atomic_torch(last_path, {
            "schema": "blindassist_grail_r1c_l_resume_v1", "seed": seed, "next_epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            "best_accuracy": best_accuracy, "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(), "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(), "sampler_random_state": generator.get_state(),
        })
        _atomic_json(seed_root / "history.json", history)
        print(json.dumps({"seed": seed, **epoch_row}), flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    result = {
        "schema": "blindassist_grail_r1c_l_seed_result_v1", "seed": seed,
        "train_collection_sha256": sha256_file(train_data.collection_path),
        "validation_collection_sha256": sha256_file(validation_data.collection_path),
        "backbone_weights_sha256": sha256_file(backbone / "model.safetensors"),
        "best_epoch": best["epoch"], "validation": best["validation"],
        "checkpoint_sha256": sha256_file(best_path), "history": history,
    }
    _atomic_json(seed_root / "result.json", result)
    _atomic_json(seed_root / "progress.json", {
        "phase": "train", "seed": seed, "completed_units": epochs, "total_units": epochs,
        "throughput": epochs / max(time.monotonic() - started, 1e-6), "eta_seconds": 0.0,
        "last_progress_at": time.time(), "status": "complete",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-collection", type=Path, required=True)
    parser.add_argument("--validation-collection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeds = manifest["architecture"]["seeds"]
    if len(seeds) > 2:
        raise ValueError("R1C-L allows at most two seeds")
    args.output.mkdir(parents=True, exist_ok=True)
    train_data = PairCollection(args.train_collection, args.data_root / "train")
    validation_data = PairCollection(args.validation_collection, args.data_root / "validation")
    results = [train_seed(seed, train_data, validation_data, args.backbone, args.output,
                          args.epochs, args.batch_size, args.workers) for seed in seeds]
    selected = max(results, key=lambda row: (row["validation"]["slot_accuracy"], -row["seed"]))
    summary = {
        "schema": "blindassist_grail_r1c_l_training_result_v1",
        "manifest_sha256": sha256_file(args.manifest), "architecture_count": 1,
        "seed_count": len(results), "seeds": results, "selected_seed": selected["seed"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_slot_accuracy": selected["validation"]["slot_accuracy"],
        "oa_v2_validation_baseline": "PENDING_FIXED_BASELINE_MATERIALIZATION",
        "final_test_accessed": False,
    }
    _atomic_json(args.output / "result.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    raise SystemExit(main())
