#!/usr/bin/env python3
"""Run a bounded real-TRAIN A0 optimizer/checkpoint execution smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.assistive_geometry_model import (  # noqa: E402
    DepthArtAssistiveGeometry,
    compute_b1_losses,
)
from scripts.research.assistive_geometry.assistive_geometry_training import (  # noqa: E402
    AssistiveGeometryTrainDataset,
    FrozenA0Scheduler,
    build_epoch_effective_batches,
    collate_train_samples,
    flatten_train_manifest,
)
from scripts.research.assistive_geometry.depthart_training_scan import install_depthart_training_scan  # noqa: E402
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
    write_json_exclusive,
)


A0_LOSSES = ("masked_log_depth", "valid_neighbor_log_gradient")


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"checkpoint already exists: {path}")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in targets.items()}


def nonzero_gradient_count(model: torch.nn.Module) -> int:
    return sum(
        int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item()))
        for parameter in model.metric_depthart.parameters()
    )


def choose_precision() -> tuple[str, torch.dtype, torch.amp.GradScaler]:
    if torch.cuda.is_bf16_supported():
        return "BF16", torch.bfloat16, torch.amp.GradScaler("cuda", enabled=False)
    return "FP16_GRADSCALER", torch.float16, torch.amp.GradScaler("cuda", enabled=True)


def run_effective_batch(
    model: DepthArtAssistiveGeometry,
    dataset: AssistiveGeometryTrainDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer,
    scheduler: FrozenA0Scheduler,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    require(len(indices) == 16, "A0 effective batch must contain 16 samples")
    optimizer.zero_grad(set_to_none=True)
    raw_losses: list[float] = []
    for start in range(0, 16, 4):
        batch = collate_train_samples([dataset[index] for index in indices[start : start + 4]])
        image = batch["image"].to(device)
        targets = move_targets(batch["targets"], device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(image, targets["intrinsics_tensor"])
            losses = compute_b1_losses(outputs, targets, A0_LOSSES)
            loss = losses["total"] / 4.0
        require(bool(torch.isfinite(loss).item()), "non-finite A0 microbatch loss")
        raw_losses.append(float(losses["total"].detach().float().item()))
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.metric_depthart.parameters(), 1.0).item())
    require(np.isfinite(gradient_norm), "non-finite A0 gradient norm")
    nonzero = nonzero_gradient_count(model)
    require(nonzero > 0, "A0 produced no encoder/depth gradients")
    multiplier = scheduler.prepare_next_step()
    scaler.step(optimizer)
    scaler.update()
    scheduler.mark_completed()
    return {
        "mean_unscaled_loss": float(np.mean(raw_losses)),
        "gradient_norm_before_clip": gradient_norm,
        "encoder_or_depth_parameters_with_nonzero_grad": nonzero,
        "learning_rate_multiplier": multiplier,
        "optimizer_step": scheduler.completed_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == "blindassist_assistive_geometry_b1_a0_execution_lock_protocol_v1", "A0 protocol schema drift")
    require(protocol["runner"]["sha256"] == sha256_file(Path(__file__)), "A0 runner SHA drift")
    require(args.seed in protocol["training"]["seeds"], "unfrozen A0 seed")
    require(not args.output_root.exists(), "A0 smoke output root already exists")
    manifest_path = args.manifest.resolve()
    require(sha256_file(manifest_path) == protocol["target_manifest"]["sha256"], "A0 target manifest drift")
    checkpoint = args.checkpoint.resolve()
    require(sha256_file(checkpoint) == protocol["checkpoint"]["sha256"], "A0 initialization checkpoint drift")

    manifest = load_json(manifest_path)
    frames = flatten_train_manifest(manifest)
    dataset = AssistiveGeometryTrainDataset(frames, args.seed, augment=True)
    dataset.set_epoch(0)
    batches, carry = build_epoch_effective_batches(frames, args.seed, 0)
    selected = {orientation: next(indices for family, indices in batches if family == orientation) for orientation in ("portrait", "landscape")}

    deployment = Path(__file__).resolve().parents[1] / "hftf/deployment/depthart"
    sys.path.insert(0, str(deployment))
    from export_depthart_camera_external import install_timm_compat

    install_timm_compat()
    source = args.source.resolve()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from model import load_model
    from network import tvimblock

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "A0 execution smoke requires CUDA")
    precision, amp_dtype, scaler = choose_precision()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        base = load_model(checkpoint, "S", "indoor", str(device))
        _, scan = install_depthart_training_scan(tvimblock)
        model = DepthArtAssistiveGeometry(base).to(device).train()
        model.assistive_heads.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.metric_depthart.parameters(),
            lr=2e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = FrozenA0Scheduler(optimizer, total_steps=6000)
        torch.cuda.reset_peak_memory_stats()
        rows: list[dict[str, Any]] = []
        for orientation in ("portrait", "landscape"):
            row = run_effective_batch(model, dataset, selected[orientation], optimizer, scheduler, scaler, amp_dtype, device)
            row["orientation_family"] = orientation
            row["sample_count"] = 16
            row["micro_batch_size"] = 4
            row["frame_identities"] = [
                {"video_id": frames[index]["video_id"], "frame_stem": frames[index]["frame_stem"]}
                for index in selected[orientation]
            ]
            rows.append(row)

        model_hash_before_save = state_digest(model)
        checkpoint_payload = {
            "schema": "blindassist_assistive_geometry_b1_a0_checkpoint_v1",
            "protocol_sha256": sha256_file(protocol_path),
            "initialization_checkpoint_sha256": protocol["checkpoint"]["sha256"],
            "seed": args.seed,
            "epoch": 0,
            "next_optimizer_step": scheduler.completed_steps + 1,
            "sampler": {"carry": carry, "formal_epoch_complete": False, "smoke_only": True},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": capture_rng_state(),
            "model_state_sha256": model_hash_before_save,
        }
        checkpoint_receipt = save_checkpoint_exclusive(args.output_root.resolve() / "a0-smoke-checkpoint.pt", checkpoint_payload)
        restored = torch.load(checkpoint_receipt["path"], map_location=device, weights_only=False)
        require(restored["protocol_sha256"] == sha256_file(protocol_path), "restored protocol drift")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        restore_rng_state(restored["rng"])
        require(state_digest(model) == model_hash_before_save == restored["model_state_sha256"], "checkpoint model-state roundtrip drift")
        require(scheduler.completed_steps == 2 and len(optimizer.state) > 0, "checkpoint optimizer/scheduler roundtrip drift")

    warning_messages = [str(item.message) for item in captured]
    missing_autograd = [message for message in warning_messages if "autograd kernel was not registered" in message.lower()]
    require(not missing_autograd, "A0 smoke restored missing Autograd registration warning")
    receipt = {
        "schema": "blindassist_assistive_geometry_b1_a0_train_execution_smoke_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "producer_sha256": protocol["runner"]["sha256"],
        "target_manifest_sha256": protocol["target_manifest"]["sha256"],
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "precision": precision,
        "tf32_disabled": True,
        "scan": scan,
        "missing_autograd_registration_warning_count": len(missing_autograd),
        "epoch_zero_planned_optimizer_steps": len(batches),
        "epoch_zero_carry_counts": {key: len(value) for key, value in carry.items()},
        "bounded_optimizer_steps": rows,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": checkpoint_receipt,
        "checkpoint_roundtrip_exact": True,
        "formal_training_started": False,
        "development_or_confirmation_content_opened": False,
        "teacher_import_or_execution": False,
        "terminal": "B1_A0_DEPTH_ONLY_TRAIN_EXECUTION_SMOKE_PASS",
        "authority": "Two bounded TRAIN-only optimizer steps and checkpoint smoke; no model quality, formal training, Development, Confirmation, deployment, product or safety authority.",
    }
    write_json_exclusive(args.output_root.resolve() / "receipt.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "bounded_optimizer_steps"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
