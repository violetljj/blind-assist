#!/usr/bin/env python3
"""Train one frozen Assistive Geometry additive arm from the common initialization."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
    flatten_train_manifest,
)
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
)
from scripts.research.assistive_geometry.smoke_b1_a0_train_execution import (  # noqa: E402
    choose_precision,
    state_digest,
)
from scripts.research.assistive_geometry.train_b1_a0_formal import (  # noqa: E402
    atomic_save_checkpoint,
    atomic_write_json,
    batch_digest,
    effective_batches_to_microbatches,
    load_depthart_model,
    make_loader,
    move_targets,
    utc_now,
    write_progress,
)


ARM_SPECS: dict[str, dict[str, Any]] = {
    "A1_PLUS_GROUND": {
        "losses": (
            "masked_log_depth",
            "valid_neighbor_log_gradient",
            "ground_bce",
            "ground_plane_depth",
        ),
        "head_modules": ("ground_pre", "ground_out"),
    },
    "A2_PLUS_CLEARANCE": {
        "losses": (
            "masked_log_depth",
            "valid_neighbor_log_gradient",
            "ground_bce",
            "ground_plane_depth",
            "clearance_huber",
            "occupancy_bce",
        ),
        "head_modules": (
            "ground_pre",
            "ground_out",
            "band_mlp",
            "clearance_out",
            "occupancy_out",
        ),
    },
    "A3_PLUS_FALSE_CLEAR": {
        "losses": (
            "masked_log_depth",
            "valid_neighbor_log_gradient",
            "ground_bce",
            "ground_plane_depth",
            "clearance_huber",
            "occupancy_bce",
            "false_clear_extra",
        ),
        "head_modules": (
            "ground_pre",
            "ground_out",
            "band_mlp",
            "clearance_out",
            "occupancy_out",
        ),
    },
    "A4_PLUS_CONFIDENCE": {
        "losses": (
            "masked_log_depth",
            "valid_neighbor_log_gradient",
            "ground_bce",
            "ground_plane_depth",
            "clearance_huber",
            "occupancy_bce",
            "false_clear_extra",
            "confidence_bce",
        ),
        "head_modules": (
            "ground_pre",
            "ground_out",
            "band_mlp",
            "clearance_out",
            "occupancy_out",
            "confidence_out",
        ),
    },
}


def arm_slug(arm: str) -> str:
    return arm.lower().replace("_", "-")


def configure_trainable(
    model: DepthArtAssistiveGeometry, arm: str
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    require(arm in ARM_SPECS, f"unsupported additive arm: {arm}")
    model.requires_grad_(False)
    model.metric_depthart.requires_grad_(True)
    head_parameters: list[torch.nn.Parameter] = []
    for name in ARM_SPECS[arm]["head_modules"]:
        module = getattr(model.assistive_heads, name)
        module.requires_grad_(True)
        head_parameters.extend(module.parameters())
    encoder_parameters = list(model.metric_depthart.parameters())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    require(len(trainable) == len(encoder_parameters) + len(head_parameters), "trainable parameter partition drift")
    require(len({id(parameter) for parameter in trainable}) == len(trainable), "duplicate trainable parameter")
    groups = [
        {"params": encoder_parameters, "lr": 2e-5, "group_name": "depthart_encoder_decoder"},
        {"params": head_parameters, "lr": 1e-4, "group_name": "active_assistive_heads"},
    ]
    return groups, trainable


def validate_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, Path]:
    path = args.protocol.resolve()
    protocol = load_json(path)
    require(
        protocol.get("schema") == "blindassist_assistive_geometry_b1_additive_arm_train_protocol_v1",
        "additive-arm protocol schema drift",
    )
    require(protocol.get("arm") == args.arm and args.arm in ARM_SPECS, "additive-arm identity drift")
    require(protocol["runner"]["sha256"] == sha256_file(Path(__file__)), "additive-arm runner SHA drift")
    require(tuple(protocol["active_losses"]) == ARM_SPECS[args.arm]["losses"], "active loss drift")
    require(tuple(protocol["active_head_modules"]) == ARM_SPECS[args.arm]["head_modules"], "active head drift")
    require(args.seed in protocol["execution"]["seeds"], "unfrozen additive-arm seed")
    require(args.workers in protocol["execution"]["allowed_workers"], "unfrozen worker count")
    authority_path = (REPO_ROOT / protocol["activation"]["result_path"]).resolve()
    require(sha256_file(authority_path) == protocol["activation"]["result_sha256"], "activation result drift")
    authority = load_json(authority_path)
    require(authority.get("terminal") == protocol["activation"]["required_terminal"], "activation terminal drift")
    require(authority.get("confirmation_content_opened") is False, "activation crossed Confirmation firewall")
    manifest = args.manifest.resolve()
    require(sha256_file(manifest) == protocol["inputs"]["target_manifest"]["sha256"], "TRAIN manifest drift")
    initialization = args.checkpoint.resolve()
    require(sha256_file(initialization) == protocol["inputs"]["initialization_checkpoint"]["sha256"], "common initialization drift")
    source = args.source.resolve()
    require(source == Path(protocol["inputs"]["source_root"]).resolve() and source.is_dir(), "DepthART source drift")
    expected_parent = (REPO_ROOT / protocol["execution"]["formal_output_parent"]).resolve()
    pilot_parent = (REPO_ROOT / protocol["execution"]["pilot_output_parent"]).resolve()
    expected_name = f"seed-{args.seed}" if args.mode == "formal" else f"workers-{args.workers}-seed-{args.seed}"
    require(args.output_root.resolve().parent == (pilot_parent if args.mode == "pilot" else expected_parent), "output parent drift")
    require(args.output_root.name == expected_name and not args.output_root.exists(), "output identity collision")
    return protocol, sha256_file(path), initialization, source


def checkpoint_payload(
    *,
    protocol_sha256: str,
    initialization_sha256: str,
    arm: str,
    seed: int,
    next_epoch: int,
    carry: dict[str, list[int]],
    model: DepthArtAssistiveGeometry,
    optimizer: torch.optim.Optimizer,
    scheduler: FrozenA0Scheduler,
    scaler: torch.amp.GradScaler,
    epoch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    from scripts.research.assistive_geometry.smoke_b1_a0_train_execution import capture_rng_state

    return {
        "schema": "blindassist_assistive_geometry_b1_additive_arm_checkpoint_v1",
        "protocol_sha256": protocol_sha256,
        "initialization_checkpoint_sha256": initialization_sha256,
        "arm": arm,
        "seed": seed,
        "next_epoch": next_epoch,
        "next_optimizer_step": scheduler.completed_steps + 1,
        "sampler": {"carry": carry, "formal_epoch_complete": True, "smoke_only": False},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "epoch_history": epoch_history,
        "model_state_sha256": state_digest(model),
    }


def execute(args: argparse.Namespace) -> int:
    protocol, protocol_sha, initialization, source = validate_protocol(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    progress_path = output_root / "progress.json"
    success_path = output_root / ("pilot-result.json" if args.mode == "pilot" else "result.json")
    failure_path = output_root / "failure.json"
    started = time.perf_counter()
    total_steps = args.pilot_steps if args.mode == "pilot" else 6000
    completed = 0
    phase = "initializing"
    write_progress(progress_path, phase=phase, completed=0, total=total_steps, started_at=started, status="running")
    try:
        if args.mode == "formal":
            atomic_write_json(
                output_root / "activation.json",
                {
                    "schema": "blindassist_assistive_geometry_b1_additive_arm_activation_v1",
                    "protocol_sha256": protocol_sha,
                    "arm": args.arm,
                    "seed": args.seed,
                    "role": "TRAIN_ONLY",
                    "common_initialization_not_prior_arm_checkpoint": True,
                    "development_or_confirmation_content_opened": False,
                    "activated_at": utc_now(),
                },
                exclusive=True,
            )
        frames = flatten_train_manifest(load_json(args.manifest.resolve()))
        dataset = AssistiveGeometryTrainDataset(frames, args.seed, augment=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        require(device.type == "cuda", "formal additive-arm training requires CUDA")
        precision, amp_dtype, scaler = choose_precision()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model, scan = load_depthart_model(source, initialization, args.seed, device)
            groups, trainable_parameters = configure_trainable(model, args.arm)
            optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = FrozenA0Scheduler(optimizer, total_steps=6000)
            carry: dict[str, list[int]] = {"portrait": [], "landscape": []}
            history: list[dict[str, Any]] = []
            retained_receipts: list[dict[str, Any]] = []
            input_digests: list[str] = []
            torch.cuda.reset_peak_memory_stats()
            optimization_started = time.perf_counter()
            pilot_losses: list[float] = []
            pilot_gradients: list[float] = []
            stop = False
            for epoch in range(20):
                phase = f"{arm_slug(args.arm)}-seed-{args.seed}-epoch-{epoch + 1:02d}"
                dataset.set_epoch(epoch)
                effective, next_carry = build_epoch_effective_batches(frames, args.seed, epoch, carry)
                microbatches, orientations = effective_batches_to_microbatches(effective)
                loader = make_loader(dataset, microbatches, args.workers)
                epoch_losses: list[float] = []
                epoch_gradients: list[float] = []
                optimizer.zero_grad(set_to_none=True)
                expected_orientation: str | None = None
                for micro_index, batch in enumerate(loader):
                    orientation = orientations[micro_index]
                    require({row["orientation_family"] for row in batch["metadata"]} == {orientation}, "loader orientation drift")
                    if micro_index % 4 == 0:
                        expected_orientation = orientation
                    require(orientation == expected_orientation, "effective batch orientation drift")
                    if len(input_digests) < 8:
                        input_digests.append(batch_digest(batch))
                    image = batch["image"].to(device, non_blocking=True)
                    targets = move_targets(batch["targets"], device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(image, targets["intrinsics_tensor"])
                        losses = compute_b1_losses(outputs, targets, ARM_SPECS[args.arm]["losses"])
                        scaled = losses["total"] / 4.0
                    require(bool(torch.isfinite(scaled).item()), "non-finite additive-arm loss")
                    value = float(losses["total"].detach().float().item())
                    epoch_losses.append(value)
                    pilot_losses.append(value)
                    scaler.scale(scaled).backward()
                    if (micro_index + 1) % 4:
                        continue
                    scaler.unscale_(optimizer)
                    gradient = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0).item())
                    require(np.isfinite(gradient) and gradient > 0.0, "invalid additive-arm gradient norm")
                    scheduler.prepare_next_step()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.mark_completed()
                    optimizer.zero_grad(set_to_none=True)
                    completed += 1
                    epoch_gradients.append(gradient)
                    pilot_gradients.append(gradient)
                    if completed == total_steps or completed % 10 == 0:
                        write_progress(progress_path, phase=phase, completed=completed, total=total_steps, started_at=started, status="running")
                    if args.mode == "pilot" and completed >= total_steps:
                        stop = True
                        break
                if stop:
                    break
                carry = next_carry
                history.append(
                    {
                        "epoch": epoch + 1,
                        "optimizer_steps_completed": scheduler.completed_steps,
                        "mean_train_loss": float(np.mean(epoch_losses)),
                        "mean_gradient_norm_before_clip": float(np.mean(epoch_gradients)),
                        "carry_counts": {key: len(value) for key, value in carry.items()},
                    }
                )
                payload = checkpoint_payload(
                    protocol_sha256=protocol_sha,
                    initialization_sha256=protocol["inputs"]["initialization_checkpoint"]["sha256"],
                    arm=args.arm,
                    seed=args.seed,
                    next_epoch=epoch + 1,
                    carry=carry,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch_history=history,
                )
                atomic_save_checkpoint(output_root / "latest.pt", payload)
                if epoch + 1 in (5, 10, 15, 20):
                    receipt = atomic_save_checkpoint(output_root / f"checkpoint-epoch-{epoch + 1:02d}.pt", payload, exclusive=True)
                    receipt.update(epoch=epoch + 1, optimizer_steps_completed=scheduler.completed_steps)
                    retained_receipts.append(receipt)
            missing_autograd = [str(item.message) for item in captured if "autograd kernel was not registered" in str(item.message).lower()]
            require(not missing_autograd, "missing Autograd registration warning restored")
            if args.mode == "formal":
                require(completed == scheduler.completed_steps == 6000, "formal additive-arm step count drift")
                require(carry == {"portrait": [], "landscape": []}, "formal final carry is not empty")
                require(len(history) == 20 and len(retained_receipts) == 4, "formal epoch/checkpoint count drift")
            result = {
                "schema": "blindassist_assistive_geometry_b1_additive_arm_train_result_v1",
                "protocol_sha256": protocol_sha,
                "runner_sha256": protocol["runner"]["sha256"],
                "arm": args.arm,
                "active_losses": list(ARM_SPECS[args.arm]["losses"]),
                "active_head_modules": list(ARM_SPECS[args.arm]["head_modules"]),
                "common_initialization_not_prior_arm_checkpoint": True,
                "mode": args.mode,
                "seed": args.seed,
                "workers": args.workers,
                "precision": precision,
                "completed_optimizer_steps": completed,
                "wall_seconds": time.perf_counter() - started,
                "optimization_wall_seconds": time.perf_counter() - optimization_started,
                "optimization_steps_per_second": completed / max(time.perf_counter() - optimization_started, 1e-9),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "input_batch_digests": input_digests,
                "pilot_metrics": {"mean_loss": float(np.mean(pilot_losses)), "mean_gradient_norm": float(np.mean(pilot_gradients))} if args.mode == "pilot" else None,
                "final_model_state_sha256": state_digest(model),
                "scan": scan,
                "epoch_history": history,
                "checkpoints": retained_receipts,
                "development_or_confirmation_content_opened": False,
                "teacher_import_or_execution": False,
                "terminal": "B1_ADDITIVE_ARM_TRAIN_PILOT_PASS" if args.mode == "pilot" else f"B1_{args.arm}_FORMAL_TRAIN_SEED_COMPLETE",
                "claim_ceiling": "TRAIN-only additive-arm optimization evidence; no Development, Confirmation, deployment, product or safety authority.",
            }
            atomic_write_json(success_path, result, exclusive=True)
        write_progress(progress_path, phase="complete", completed=total_steps, total=total_steps, started_at=started, status="complete")
        print(json.dumps({key: value for key, value in result.items() if key != "epoch_history"}, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "schema": "blindassist_assistive_geometry_b1_additive_arm_train_failure_v1",
            "arm": args.arm,
            "mode": args.mode,
            "seed": args.seed,
            "phase": phase,
            "completed_optimizer_steps": completed,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "development_or_confirmation_content_opened": False,
            "terminal": "B1_ADDITIVE_ARM_TRAIN_EXECUTION_FAILED_WITH_RECEIPT",
        }
        if not failure_path.exists():
            atomic_write_json(failure_path, failure, exclusive=True)
        write_progress(progress_path, phase=phase, completed=completed, total=total_steps, started_at=started, status="failed")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--pilot-steps", type=int, default=0)
    args = parser.parse_args()
    require(args.workers >= 0, "workers must be non-negative")
    if args.mode == "pilot":
        require(2 <= args.pilot_steps <= 20, "pilot steps must be in [2,20]")
    else:
        require(args.pilot_steps == 0, "formal mode forbids pilot steps")
    return args


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
