#!/usr/bin/env python3
"""Run the frozen three-seed Assistive Geometry A0 TRAIN-only execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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
)
from scripts.research.assistive_geometry.smoke_b1_a0_train_execution import (  # noqa: E402
    capture_rng_state,
    choose_precision,
    restore_rng_state,
    state_digest,
)


A0_LOSSES = ("masked_log_depth", "valid_neighbor_log_gradient")
PROGRESS_FIELDS = (
    "phase",
    "completed_units",
    "total_units",
    "throughput",
    "eta_seconds",
    "last_progress_at",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not temporary.exists(), f"partial JSON already exists: {temporary}")
    if exclusive:
        require(not path.exists(), f"JSON already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_save_checkpoint(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not temporary.exists(), f"partial checkpoint already exists: {temporary}")
    if exclusive:
        require(not path.exists(), f"checkpoint already exists: {path}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_progress(
    path: Path,
    *,
    phase: str,
    completed: int,
    total: int,
    started_at: float,
    status: str,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    throughput = completed / elapsed if completed > 0 and elapsed > 0 else 0.0
    eta = (total - completed) / throughput if throughput > 0 and completed < total else 0.0
    payload = {
        "phase": phase,
        "completed_units": completed,
        "total_units": total,
        "throughput": throughput,
        "eta_seconds": eta,
        "last_progress_at": utc_now(),
        "status": status,
    }
    atomic_write_json(path, payload)
    return payload


def effective_batches_to_microbatches(
    batches: Iterable[tuple[str, list[int]]],
) -> tuple[list[list[int]], list[str]]:
    microbatches: list[list[int]] = []
    orientations: list[str] = []
    for orientation, indices in batches:
        require(len(indices) == 16, "A0 effective batch must contain 16 samples")
        for start in range(0, 16, 4):
            microbatches.append(indices[start : start + 4])
            orientations.append(orientation)
    return microbatches, orientations


def loader_worker_init(_: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_loader(
    dataset: AssistiveGeometryTrainDataset,
    microbatches: list[list[int]],
    workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": microbatches,
        "num_workers": workers,
        "collate_fn": collate_train_samples,
        "pin_memory": True,
        "worker_init_fn": loader_worker_init,
    }
    if workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(**kwargs)


def move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


def batch_digest(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for tensor in [batch["image"], *(batch["targets"][key] for key in sorted(batch["targets"]))]:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    for metadata in batch["metadata"]:
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    return digest.hexdigest().upper()


def checkpoint_payload(
    *,
    protocol_sha256: str,
    initialization_sha256: str,
    seed: int,
    next_epoch: int,
    carry: dict[str, list[int]],
    model: DepthArtAssistiveGeometry,
    optimizer: torch.optim.Optimizer,
    scheduler: FrozenA0Scheduler,
    scaler: torch.amp.GradScaler,
    epoch_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_checkpoint_v1",
        "protocol_sha256": protocol_sha256,
        "initialization_checkpoint_sha256": initialization_sha256,
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


def validate_protocol_and_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, Path, Path]:
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    require(
        protocol.get("schema") == "blindassist_assistive_geometry_b1_a0_formal_train_execution_protocol_v1",
        "formal A0 protocol schema drift",
    )
    require(protocol["runner"]["sha256"] == sha256_file(Path(__file__)), "formal A0 runner SHA drift")
    require(args.seed in protocol["execution"]["seeds"], "unfrozen formal A0 seed")
    require(args.workers in protocol["execution"]["allowed_workers"], "unfrozen A0 worker count")

    activation = protocol["bindings"]["execution_lock_result"]
    activation_path = (REPO_ROOT / activation["path"]).resolve()
    require(sha256_file(activation_path) == activation["sha256"], "A0 execution authority drift")
    activation_payload = load_json(activation_path)
    require(
        activation_payload.get("terminal") == "B1_A0_DEPTH_ONLY_THREE_SEED_TRAIN_EXECUTION_LOCK_PASS",
        "A0 execution authority terminal drift",
    )
    require(
        activation_payload["authority"].get("formal_a0_three_seed_train_execution") is True,
        "formal A0 execution is not authorized",
    )

    manifest_path = args.manifest.resolve()
    require(sha256_file(manifest_path) == protocol["inputs"]["target_manifest"]["sha256"], "A0 manifest drift")
    checkpoint = args.checkpoint.resolve()
    require(sha256_file(checkpoint) == protocol["inputs"]["initialization_checkpoint"]["sha256"], "A0 checkpoint drift")
    source = args.source.resolve()
    require(source == Path(protocol["inputs"]["source_root"]).resolve(), "DepthART source root drift")
    require(source.is_dir(), "DepthART source root is missing")
    for binding in protocol["implementation_bindings"]:
        path = (REPO_ROOT / binding["path"]).resolve()
        require(sha256_file(path) == binding["sha256"], f"implementation drift: {binding['path']}")

    expected_parent = (REPO_ROOT / protocol["execution"]["formal_output_parent"]).resolve()
    pilot_parent = (REPO_ROOT / protocol["execution"]["pilot_output_parent"]).resolve()
    output_root = args.output_root.resolve()
    allowed_parent = pilot_parent if args.mode == "pilot" else expected_parent
    require(output_root.parent == allowed_parent, "A0 output parent drift")
    expected_name = f"seed-{args.seed}" if args.mode == "formal" else f"workers-{args.workers}-seed-{args.seed}"
    require(output_root.name == expected_name, "A0 output identity drift")
    require(not output_root.exists(), "A0 output root already exists")
    return protocol, sha256_file(protocol_path), checkpoint, source


def load_depthart_model(
    source: Path,
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> tuple[DepthArtAssistiveGeometry, dict[str, Any]]:
    deployment = Path(__file__).resolve().parents[1] / "hftf/deployment/depthart"
    sys.path.insert(0, str(deployment))
    from export_depthart_camera_external import install_timm_compat

    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from model import load_model
    from network import tvimblock

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = load_model(checkpoint, "S", "indoor", str(device))
    _, scan = install_depthart_training_scan(tvimblock)
    model = DepthArtAssistiveGeometry(base).to(device).train()
    model.assistive_heads.requires_grad_(False)
    return model, scan


def execute(args: argparse.Namespace) -> int:
    protocol, protocol_sha256, checkpoint, source = validate_protocol_and_inputs(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    progress_path = output_root / "progress.json"
    success_path = output_root / ("pilot-result.json" if args.mode == "pilot" else "result.json")
    failure_path = output_root / "failure.json"
    activation_path = output_root / "activation.json"
    started_at = time.perf_counter()
    total_steps = args.pilot_steps if args.mode == "pilot" else 6000
    completed_steps = 0
    phase = "initializing"
    write_progress(progress_path, phase=phase, completed=0, total=total_steps, started_at=started_at, status="running")

    try:
        if args.mode == "formal":
            atomic_write_json(
                activation_path,
                {
                    "schema": "blindassist_assistive_geometry_b1_a0_formal_activation_v1",
                    "protocol_sha256": protocol_sha256,
                    "seed": args.seed,
                    "created_at": utc_now(),
                    "role": "TRAIN_ONLY",
                    "development_or_confirmation_content_opened": False,
                    "terminal": "A0_FORMAL_TRAIN_EXECUTION_ACTIVATED",
                },
                exclusive=True,
            )

        manifest = load_json(args.manifest.resolve())
        frames = flatten_train_manifest(manifest)
        dataset = AssistiveGeometryTrainDataset(frames, args.seed, augment=True)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        require(device.type == "cuda", "formal A0 training requires CUDA")
        precision, amp_dtype, scaler = choose_precision()

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model, scan = load_depthart_model(source, checkpoint, args.seed, device)
            optimizer = torch.optim.AdamW(
                model.metric_depthart.parameters(),
                lr=2e-5,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            scheduler = FrozenA0Scheduler(optimizer, total_steps=6000)
            carry: dict[str, list[int]] = {"portrait": [], "landscape": []}
            epoch_history: list[dict[str, Any]] = []
            checkpoint_receipts: list[dict[str, Any]] = []
            input_batch_digests: list[str] = []
            torch.cuda.reset_peak_memory_stats()
            optimization_started_at = time.perf_counter()

            stop = False
            for epoch in range(20):
                phase = f"seed-{args.seed}-epoch-{epoch + 1:02d}"
                dataset.set_epoch(epoch)
                batches, next_carry = build_epoch_effective_batches(frames, args.seed, epoch, carry)
                microbatches, micro_orientations = effective_batches_to_microbatches(batches)
                loader = make_loader(dataset, microbatches, args.workers)
                raw_losses: list[float] = []
                step_gradients: list[float] = []
                optimizer.zero_grad(set_to_none=True)
                expected_orientation: str | None = None

                for micro_index, batch in enumerate(loader):
                    orientation = micro_orientations[micro_index]
                    observed = {row["orientation_family"] for row in batch["metadata"]}
                    require(observed == {orientation}, "A0 loader orientation drift")
                    if micro_index % 4 == 0:
                        expected_orientation = orientation
                    require(orientation == expected_orientation, "A0 effective batch orientation drift")
                    if len(input_batch_digests) < 8:
                        input_batch_digests.append(batch_digest(batch))

                    image = batch["image"].to(device, non_blocking=True)
                    targets = move_targets(batch["targets"], device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(image, targets["intrinsics_tensor"])
                        losses = compute_b1_losses(outputs, targets, A0_LOSSES)
                        scaled_loss = losses["total"] / 4.0
                    require(bool(torch.isfinite(scaled_loss).item()), "non-finite formal A0 loss")
                    raw_losses.append(float(losses["total"].detach().float().item()))
                    scaler.scale(scaled_loss).backward()

                    if (micro_index + 1) % 4 != 0:
                        continue
                    scaler.unscale_(optimizer)
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.metric_depthart.parameters(), 1.0).item())
                    require(np.isfinite(gradient_norm) and gradient_norm > 0.0, "invalid formal A0 gradient norm")
                    scheduler.prepare_next_step()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.mark_completed()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                    step_gradients.append(gradient_norm)

                    if completed_steps == total_steps or completed_steps % 10 == 0:
                        write_progress(
                            progress_path,
                            phase=phase,
                            completed=completed_steps,
                            total=total_steps,
                            started_at=started_at,
                            status="running",
                        )
                    if args.mode == "pilot" and completed_steps >= total_steps:
                        stop = True
                        break

                if stop:
                    break

                carry = next_carry
                epoch_history.append(
                    {
                        "epoch": epoch + 1,
                        "optimizer_steps_completed": scheduler.completed_steps,
                        "mean_train_loss": float(np.mean(raw_losses)),
                        "mean_gradient_norm_before_clip": float(np.mean(step_gradients)),
                        "carry_counts": {key: len(value) for key, value in carry.items()},
                    }
                )
                payload = checkpoint_payload(
                    protocol_sha256=protocol_sha256,
                    initialization_sha256=protocol["inputs"]["initialization_checkpoint"]["sha256"],
                    seed=args.seed,
                    next_epoch=epoch + 1,
                    carry=carry,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch_history=epoch_history,
                )
                atomic_save_checkpoint(output_root / "latest.pt", payload)
                if epoch + 1 in (5, 10, 15, 20):
                    retained = atomic_save_checkpoint(output_root / f"checkpoint-epoch-{epoch + 1:02d}.pt", payload, exclusive=True)
                    retained["epoch"] = epoch + 1
                    retained["optimizer_steps_completed"] = scheduler.completed_steps
                    checkpoint_receipts.append(retained)

            warning_messages = [str(item.message) for item in captured]
            missing_autograd = [message for message in warning_messages if "autograd kernel was not registered" in message.lower()]
            require(not missing_autograd, "formal A0 restored missing Autograd registration warning")

            if args.mode == "formal":
                require(completed_steps == 6000 and scheduler.completed_steps == 6000, "formal A0 step count drift")
                require(carry == {"portrait": [], "landscape": []}, "formal A0 final carry is not empty")
                require(len(epoch_history) == 20 and len(checkpoint_receipts) == 4, "formal A0 epoch/checkpoint count drift")
            else:
                require(completed_steps == args.pilot_steps, "A0 pilot step count drift")

            result = {
                "schema": (
                    "blindassist_assistive_geometry_b1_a0_pilot_result_v1"
                    if args.mode == "pilot"
                    else "blindassist_assistive_geometry_b1_a0_formal_train_result_v1"
                ),
                "protocol_sha256": protocol_sha256,
                "runner_sha256": protocol["runner"]["sha256"],
                "mode": args.mode,
                "seed": args.seed,
                "workers": args.workers,
                "precision": precision,
                "tf32_disabled": True,
                "completed_optimizer_steps": completed_steps,
                "wall_seconds": time.perf_counter() - started_at,
                "optimizer_steps_per_second": completed_steps / max(time.perf_counter() - started_at, 1e-9),
                "optimization_wall_seconds": time.perf_counter() - optimization_started_at,
                "optimization_steps_per_second": completed_steps / max(time.perf_counter() - optimization_started_at, 1e-9),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "input_batch_digests": input_batch_digests,
                "pilot_metrics": (
                    {
                        "microbatch_loss_trace": raw_losses,
                        "optimizer_step_gradient_norm_trace": step_gradients,
                        "mean_microbatch_loss": float(np.mean(raw_losses)),
                        "mean_gradient_norm_before_clip": float(np.mean(step_gradients)),
                    }
                    if args.mode == "pilot"
                    else None
                ),
                "final_model_state_sha256": state_digest(model),
                "scan": scan,
                "missing_autograd_registration_warning_count": len(missing_autograd),
                "epoch_history": epoch_history,
                "checkpoints": checkpoint_receipts,
                "development_or_confirmation_content_opened": False,
                "teacher_import_or_execution": False,
                "terminal": (
                    "B1_A0_TRAIN_PERFORMANCE_PILOT_PASS"
                    if args.mode == "pilot"
                    else "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE"
                ),
                "claim_ceiling": "TRAIN-only optimization evidence; no Development, Confirmation, deployment, product or safety authority.",
            }
            atomic_write_json(success_path, result, exclusive=True)

        write_progress(
            progress_path,
            phase="complete",
            completed=total_steps,
            total=total_steps,
            started_at=started_at,
            status="complete",
        )
        print(json.dumps({key: value for key, value in result.items() if key not in {"epoch_history"}}, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "schema": "blindassist_assistive_geometry_b1_a0_train_failure_v1",
            "mode": args.mode,
            "seed": args.seed,
            "workers": args.workers,
            "phase": phase,
            "completed_optimizer_steps": completed_steps,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": utc_now(),
            "development_or_confirmation_content_opened": False,
            "terminal": "B1_A0_TRAIN_EXECUTION_FAILED_WITH_RECEIPT",
        }
        if not failure_path.exists():
            atomic_write_json(failure_path, failure, exclusive=True)
        write_progress(
            progress_path,
            phase=phase,
            completed=completed_steps,
            total=total_steps,
            started_at=started_at,
            status="failed",
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
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
        require(2 <= args.pilot_steps <= 20, "pilot steps must be in [2, 20]")
    else:
        require(args.pilot_steps == 0, "formal mode forbids pilot steps")
    return args


def main() -> int:
    return execute(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
