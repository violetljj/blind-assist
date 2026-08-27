#!/usr/bin/env python3
"""Train matched B1-single and G1-triplet arms for R1C-G1."""

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

from grail_r1cg_g1 import (
    ActiveMultiviewPermutation,
    G1Collection,
    MODES,
    predicted_mode_indices,
    set_valued_cross_entropy,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _balanced_accuracy(rows: list[dict[str, Any]]) -> float | None:
    accuracies = []
    for mode in MODES:
        selected = [row for row in rows if row["truth"] == mode]
        if not selected:
            return None
        accuracies.append(sum(row["prediction"] == mode for row in selected) / len(selected))
    return sum(accuracies) / len(accuracies)


def metrics_from_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "discriminative_samples": len(rows),
        "correct": sum(row["prediction"] == row["truth"] for row in rows),
        "accuracy": sum(row["prediction"] == row["truth"] for row in rows) / max(len(rows), 1),
        "balanced_accuracy": _balanced_accuracy(rows),
        "by_mode": {},
        "by_type": {},
    }
    for mode in MODES:
        selected = [row for row in rows if row["truth"] == mode]
        output["by_mode"][mode] = {
            "correct": sum(row["prediction"] == mode for row in selected),
            "total": len(selected),
            "accuracy": sum(row["prediction"] == mode for row in selected) / len(selected) if selected else None,
        }
    for object_type in sorted({row["object_type"] for row in rows}):
        selected = [row for row in rows if row["object_type"] == object_type]
        output["by_type"][object_type] = {
            "total": len(selected),
            "balanced_accuracy": _balanced_accuracy(selected),
            "by_mode": {
                mode: {
                    "correct": sum(row["prediction"] == mode for row in selected if row["truth"] == mode),
                    "total": sum(row["truth"] == mode for row in selected),
                }
                for mode in MODES
            },
        }
    group_scores = []
    group_ids = sorted({row["group_id"] for row in rows})
    for group_id in group_ids:
        score = _balanced_accuracy([row for row in rows if row["group_id"] == group_id])
        if score is not None:
            group_scores.append(score)
    output["owner_group_macro"] = {
        "eligible_groups_with_both_modes": len(group_scores),
        "all_groups": len(group_ids),
        "balanced_accuracy": sum(group_scores) / len(group_scores) if group_scores else None,
    }
    return output


@torch.inference_mode()
def evaluate(model: ActiveMultiviewPermutation, loader: DataLoader,
             device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    loss_total = 0.0
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                batch["reference_rgb"].to(device, non_blocking=True),
                batch["reference_masks"].to(device, non_blocking=True),
                batch["query_rgb"].to(device, non_blocking=True),
                batch["query_masks"].to(device, non_blocking=True),
            )
        valid = batch["valid_modes"].to(device)
        loss_total += float(set_valued_cross_entropy(logits, valid)) * logits.shape[0]
        predictions = predicted_mode_indices(logits).cpu().tolist()
        truths = valid.float().argmax(dim=1).cpu().tolist()
        for index, (prediction, truth) in enumerate(zip(predictions, truths)):
            rows.append({
                "sample_id": batch["sample_id"][index],
                "house_index": int(batch["house_index"][index]),
                "group_id": batch["group_id"][index],
                "object_type": batch["object_type"][index],
                "truth": MODES[truth],
                "prediction": MODES[prediction],
            })
    metrics = metrics_from_predictions(rows)
    metrics["loss"] = loss_total / max(len(rows), 1)
    return metrics, rows


def train_arm_seed(arm: str, seed: int, train_data: G1Collection, validation_data: G1Collection,
                   backbone: Path, output: Path, epochs: int, batch_size: int,
                   workers: int) -> dict[str, Any]:
    _seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("R1C-G1 CUDA_REQUIRED_NO_CPU_FALLBACK")
    device = torch.device("cuda")
    seed_root = output / arm / f"seed-{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    strata: dict[tuple[str, str], int] = {}
    for sample in train_data.samples:
        key = (sample["object_type"], sample["valid_slot_modes"][0])
        strata[key] = strata.get(key, 0) + 1
    weights = [1.0 / strata[(sample["object_type"], sample["valid_slot_modes"][0])]
               for sample in train_data.samples]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    model = ActiveMultiviewPermutation(backbone).to(device)
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
    start_epoch, best_balanced = 0, -1.0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["arm"] != arm or checkpoint["seed"] != seed:
            raise ValueError("R1C-G1 resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["next_epoch"])
        best_balanced = float(checkpoint["best_balanced_accuracy"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        generator.set_state(checkpoint["sampler_random_state"])
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = examples = 0.0
        epoch_started = time.monotonic()
        for batch_number, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    batch["reference_rgb"].to(device, non_blocking=True),
                    batch["reference_masks"].to(device, non_blocking=True),
                    batch["query_rgb"].to(device, non_blocking=True),
                    batch["query_masks"].to(device, non_blocking=True),
                )
                loss = set_valued_cross_entropy(logits, batch["valid_modes"].to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = logits.shape[0]
            running_loss += float(loss.detach()) * count
            examples += count
            if batch_number % 100 == 0:
                elapsed = max(time.monotonic() - epoch_started, 1e-6)
                throughput = examples / elapsed
                _atomic_json(seed_root / "progress.json", {
                    "phase": "train", "arm": arm, "seed": seed, "epoch": epoch + 1,
                    "completed_units": batch_number, "total_units": len(train_loader),
                    "throughput_samples_per_second": throughput,
                    "eta_seconds": (len(train_loader) - batch_number) * batch_size / max(throughput, 1e-9),
                    "last_progress_at": time.time(), "status": "running",
                })
        validation, _ = evaluate(model, validation_loader, device)
        balanced = validation["balanced_accuracy"]
        if balanced is None:
            raise RuntimeError("R1C-G1 validation lacks one discriminative mode")
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(examples, 1),
            "validation": validation,
        }
        history.append(epoch_row)
        if balanced > best_balanced:
            best_balanced = balanced
            _atomic_torch(best_path, {
                "schema": "blindassist_grail_r1c_g1_checkpoint_v1",
                "arm": arm, "seed": seed, "epoch": epoch + 1,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "validation": validation,
            })
        _atomic_torch(last_path, {
            "schema": "blindassist_grail_r1c_g1_resume_v1",
            "arm": arm, "seed": seed, "next_epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
            "best_balanced_accuracy": best_balanced,
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(), "cuda_random_state": torch.cuda.get_rng_state_all(),
            "sampler_random_state": generator.get_state(),
        })
        _atomic_json(seed_root / "history.json", history)
        print(json.dumps({"arm": arm, "seed": seed, **epoch_row}), flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    best_metrics, predictions = evaluate(model, validation_loader, device)
    _atomic_json(seed_root / "predictions.json", predictions)
    result = {
        "schema": "blindassist_grail_r1c_g1_seed_result_v1",
        "arm": arm, "seed": seed,
        "train_collection_sha256": sha256_file(train_data.collection_path),
        "validation_collection_sha256": sha256_file(validation_data.collection_path),
        "backbone_weights_sha256": sha256_file(backbone / "model.safetensors"),
        "best_epoch": best["epoch"], "validation": best_metrics,
        "checkpoint_sha256": sha256_file(best_path),
        "predictions_sha256": sha256_file(seed_root / "predictions.json"),
        "history": history,
    }
    _atomic_json(seed_root / "result.json", result)
    _atomic_json(seed_root / "progress.json", {
        "phase": "train", "arm": arm, "seed": seed,
        "completed_units": epochs, "total_units": epochs,
        "throughput_epochs_per_second": epochs / max(time.monotonic() - started, 1e-6),
        "eta_seconds": 0.0, "last_progress_at": time.time(), "status": "complete",
    })
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-collection", type=Path, required=True)
    parser.add_argument("--validation-collection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_g1_manifest_v1":
        raise ValueError("R1C-G1 manifest schema mismatch")
    epochs = int(manifest["architecture"]["epochs"])
    seeds = manifest["architecture"]["seeds"]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in ("b1_single", "g1_triplet"):
        train_data = G1Collection(args.train_collection, args.data_root / "train", arm, True)
        validation_data = G1Collection(args.validation_collection, args.data_root / "validation", arm, True)
        for seed in seeds:
            results.append(train_arm_seed(
                arm, int(seed), train_data, validation_data, args.backbone,
                args.output, epochs, args.batch_size, args.workers,
            ))
    summary = {
        "schema": "blindassist_grail_r1c_g1_training_result_v1",
        "manifest_sha256": sha256_file(args.manifest),
        "arm_count": 2, "seed_count_per_arm": len(seeds), "runs": results,
        "final_test_accessed": False,
    }
    _atomic_json(args.output / "result.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    raise SystemExit(main())
