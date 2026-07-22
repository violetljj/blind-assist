#!/usr/bin/env python3
"""Train one frozen arm/seed of the paired R1.2d detector matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r12d_contract import CLASSES, load_json, require, resolve_bound_file, sha256_file, validate_matrix, write_json


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def layer_index(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    return None


def tensor_state_sha256(state: dict[str, torch.Tensor], layers: set[int]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if layer_index(key) not in layers:
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(bytes(str(tuple(value.shape)), "ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def architecture_path(architecture: str) -> Path:
    import ultralytics

    root = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26"
    path = root / architecture
    require(path.is_file(), f"Ultralytics architecture missing: {path}")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    matrix_path = args.matrix.resolve()
    matrix = validate_matrix(matrix_path, repo)
    arm = next((row for row in matrix["paired_arms"] if row["arm_id"] == args.arm), None)
    require(arm is not None, f"unknown arm: {args.arm}")
    require(args.seed in matrix["training"]["seeds"], f"seed not preregistered: {args.seed}")
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite training run: {output}")
    dataset = args.dataset.resolve()
    receipt_path = dataset / "dataset_receipt.json"
    receipt = load_json(receipt_path)
    require(receipt["gates"]["dataset_admission_passed"] is True, "dataset admission failed")
    require(receipt["matrix_sha256"] == sha256_file(matrix_path), "dataset/matrix hash mismatch")
    require(receipt["training_manifest_sha256"] == sha256_file(dataset / "training_manifest.jsonl"), "manifest drifted")
    pretrained = resolve_bound_file(
        repo, matrix["initialization"]["pretrained_weights_path"],
        matrix["initialization"]["pretrained_weights_sha256"], "YOLO26 pretrained weights",
    )
    arch_path = architecture_path(arm["architecture"])
    require(sha256_file(arch_path) == arm["architecture_sha256"], "architecture SHA-256 mismatch")
    set_seed(args.seed)
    from ultralytics import YOLO
    import ultralytics

    source_model = YOLO(str(pretrained)).model
    candidate_arch = YOLO(str(arch_path), task="detect")
    require([int(value) for value in candidate_arch.model.stride.tolist()] == arm["expected_strides"], "model stride drifted")
    source_state = source_model.state_dict()
    candidate_state = candidate_arch.model.state_dict()
    shared_layers = set(matrix["initialization"]["shared_backbone_layers"])
    transfer = {
        key: value for key, value in source_state.items()
        if layer_index(key) in shared_layers and key in candidate_state and candidate_state[key].shape == value.shape
    }
    expected_shared = {key for key in candidate_state if layer_index(key) in shared_layers}
    require(set(transfer) == expected_shared, f"shared backbone transfer incomplete: {len(transfer)}/{len(expected_shared)}")
    candidate_arch.model.load_state_dict(transfer, strict=False)
    backbone_sha = tensor_state_sha256(candidate_arch.model.state_dict(), shared_layers)
    source_backbone_sha = tensor_state_sha256(source_state, shared_layers)
    require(backbone_sha == source_backbone_sha, "shared backbone tensor hash mismatch")
    training = matrix["training"]
    output.parent.mkdir(parents=True, exist_ok=True)
    initialization_dir = output.parent / "_initialization"
    initialization_dir.mkdir(parents=True, exist_ok=True)
    initialization_checkpoint = initialization_dir / f"{arm['arm_id']}-seed{args.seed}.pt"
    require(not initialization_checkpoint.exists(), f"initialization checkpoint already exists: {initialization_checkpoint}")
    torch.save({
        "model": copy.deepcopy(candidate_arch.model).float(),
        "date": datetime.now(timezone.utc).isoformat(), "version": ultralytics.__version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }, initialization_checkpoint)
    initialization_checkpoint_sha = sha256_file(initialization_checkpoint)
    candidate = YOLO(str(initialization_checkpoint), task="detect")
    pretrain_observation: dict[str, Any] = {}

    def verify_trainer_initialization(trainer: Any) -> None:
        observed_sha = tensor_state_sha256(trainer.model.state_dict(), shared_layers)
        require(observed_sha == source_backbone_sha,
                f"trainer discarded shared backbone initialization: {observed_sha} != {source_backbone_sha}")
        pretrain_observation["shared_backbone_tensor_sha256"] = observed_sha
        pretrain_observation["class_count"] = int(trainer.model.nc)

    candidate.add_callback("on_pretrain_routine_end", verify_trainer_initialization)
    results = candidate.train(
        data=str((dataset / "data.yaml").resolve()), epochs=training["epochs"], imgsz=training["image_size"],
        batch=training["batch"], workers=training["workers"], device=training["device"],
        deterministic=training["deterministic"], seed=args.seed, optimizer=training["optimizer"],
        lr0=training["lr0"], lrf=training["lrf"], momentum=training["momentum"],
        weight_decay=training["weight_decay"], warmup_epochs=training["warmup_epochs"],
        cos_lr=training["cos_lr"], close_mosaic=training["close_mosaic"], amp=training["amp"],
        patience=training["patience"], project=str(output.parent), name=output.name,
        exist_ok=False, cache=False, plots=False, verbose=False, save=True, val=True,
    )
    save_dir = Path(results.save_dir).resolve()
    require(pretrain_observation.get("shared_backbone_tensor_sha256") == source_backbone_sha,
            "trainer initialization callback did not verify the shared backbone")
    require(pretrain_observation.get("class_count") == len(CLASSES), "trainer class count drifted")
    require(save_dir == output, f"unexpected training output: {save_dir}")
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    require(best.is_file() and last.is_file(), "training weights missing")
    receipt_output = {
        "schema": "blindassist_ustrf_r12d_training_run_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": sha256_file(matrix_path), "dataset_receipt_sha256": sha256_file(receipt_path),
        "training_manifest_sha256": sha256_file(dataset / "training_manifest.jsonl"),
        "arm_id": arm["arm_id"], "p2": arm["p2"], "seed": args.seed,
        "architecture_path": str(arch_path), "architecture_sha256": sha256_file(arch_path),
        "expected_strides": arm["expected_strides"], "classes": CLASSES,
        "initialization": {
            "pretrained_weights_sha256": sha256_file(pretrained), "shared_backbone_layers": sorted(shared_layers),
            "transferred_tensor_count": len(transfer), "shared_backbone_tensor_sha256": backbone_sha,
            "initialization_checkpoint_path": str(initialization_checkpoint),
            "initialization_checkpoint_sha256": initialization_checkpoint_sha,
            "trainer_pretrain_shared_backbone_tensor_sha256": pretrain_observation["shared_backbone_tensor_sha256"],
            "neck_and_head": matrix["initialization"]["neck_and_head"],
        },
        "training": training, "best_weights_path": str(best), "best_weights_sha256": sha256_file(best),
        "last_weights_sha256": sha256_file(last),
        "runtime": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(training["device"])},
        "authority": {"benchmark_only": True, "r13_inventory_read_authorized": False,
                      "production_model_replacement_authorized": False},
    }
    write_json(save_dir / "training_receipt.json", receipt_output)
    print("USTRF_R12D_TRAIN_OK", arm["arm_id"], args.seed, receipt_output["best_weights_sha256"])
    return receipt_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
