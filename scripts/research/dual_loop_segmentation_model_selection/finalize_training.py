#!/usr/bin/env python3
"""Finalize an R1 training directory from immutable per-seed receipts.

This recovery path is intentionally narrow: it never trains, selects a new
checkpoint, opens a fresh manifest, or changes a seed report.  It exists for a
post-training report-writing failure after all seed checkpoints were saved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from .train import DEFAULT_SEEDS, class_weights, load_shared, resolve, write_json
    from .models import sha256_file
except ImportError:  # pragma: no cover - direct script execution
    from train import DEFAULT_SEEDS, class_weights, load_shared, resolve, write_json
    from models import sha256_file


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = resolve(repo_root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1":
        raise ValueError("training config is not bound to R1")
    dataset_root = resolve(repo_root, args.dataset_root)
    manifest = dataset_root / str(config["training_manifest"])
    if sha256_file(manifest) != str(config["dataset_manifest_sha256"]):
        raise ValueError("training manifest SHA256 differs from the frozen config")
    output_dir = resolve(repo_root, args.output_dir)
    initialization_path = output_dir / "initialization_receipt.json"
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    seed_reports = []
    for seed in DEFAULT_SEEDS:
        path = output_dir / f"seed-{seed}" / "seed_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report.get("seed")) != seed:
            raise ValueError(f"seed receipt mismatch: {path}")
        checkpoint = Path(str(report["checkpoint"]))
        if not checkpoint.is_file() or sha256_file(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError(f"seed checkpoint missing or hash-mismatched: {checkpoint}")
        seed_reports.append(report)
    selected = max(seed_reports, key=lambda report: tuple(report["selection_key"]))
    selected_checkpoint = Path(str(selected["checkpoint"]))
    final_checkpoint = output_dir / "fp32_checkpoint.pt"
    if final_checkpoint.exists():
        existing_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        selected_payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        if existing_payload.get("seed") != selected_payload.get("seed"):
            raise ValueError("existing fp32_checkpoint.pt selected seed differs from seed receipt")
        existing_state = existing_payload.get("state_dict", {})
        selected_state = selected_payload.get("state_dict", {})
        if set(existing_state) != set(selected_state) or any(
            not torch.equal(existing_state[key], selected_state[key]) for key in selected_state
        ):
            raise ValueError("existing fp32_checkpoint.pt tensor state differs from selected seed checkpoint")
    if not final_checkpoint.exists():
        shutil.copy2(selected_checkpoint, final_checkpoint)
    shared = load_shared()
    records = shared.load_records(manifest)
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    if len(train_records) != 400 or len(dev_records) != 200:
        raise ValueError("R1 requires canonical 400 train + 200 dev records")
    weights = class_weights(shared, train_records)
    final_report = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1_training_report.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "status": "COMPLETE",
        "model_id": initialization["model_id"],
        "implementation_identity": initialization["implementation_identity"],
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(dataset_root),
        "training_manifest": str(manifest.resolve()),
        "training_manifest_sha256": sha256_file(manifest),
        "records": {"train": len(train_records), "dev": len(dev_records)},
        "sessions": {
            "train": sorted({record.session_id for record in train_records}),
            "dev": sorted({record.session_id for record in dev_records}),
        },
        "class_pixel_weights": weights.tolist(),
        "training_contract": {
            "optimizer": config["optimizer"],
            "optimizer_steps_per_seed": config["optimizer_steps"],
            "head_warmup_steps": config["head_warmup_steps"],
            "evaluation_every_steps": config["eval_every_steps"],
            "batch_size": config["batch_size"],
            "seeds": config["seeds"],
            "augmentation": config["augmentation"],
            "loss": config["loss"],
        },
        "initialization": initialization,
        "seed_reports": seed_reports,
        "selected_seed": selected["seed"],
        "selected_checkpoint": str(final_checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_file(final_checkpoint),
        "selected_dev_mask_metrics": selected["dev_mask_metrics"],
        "worst_seed": min(seed_reports, key=lambda report: tuple(report["selection_key"]))["seed"],
        "recovery": {
            "kind": "post_training_report_finalize_from_immutable_seed_receipts",
            "original_failure": "runner reached all three 1200-step seed checkpoints but referenced a deleted temporary model while writing the final report",
            "new_training_or_checkpoint_selection": False,
        },
        "fresh_holdout_consumed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "training_report.json"
    write_json(report_path, final_report)
    write_json(report_path.with_suffix(".sha256.json"), {"sha256": sha256_file(report_path)})
    write_json(output_dir / "training_progress.json", {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.training_progress.v1",
        "status": "COMPLETE",
        "model_id": initialization["model_id"],
        "completed_steps": int(config["optimizer_steps"]) * len(DEFAULT_SEEDS),
        "total_steps": int(config["optimizer_steps"]) * len(DEFAULT_SEEDS),
        "selected_seed": selected["seed"],
        "selected_checkpoint_sha256": sha256_file(final_checkpoint),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return final_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps({
        "status": report["status"],
        "model_id": report["model_id"],
        "selected_seed": report["selected_seed"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
    }, ensure_ascii=False))
