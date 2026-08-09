#!/usr/bin/env python3
"""Run the isolated AG-QSF H1-only frozen-encoder TRAIN canary."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.assistive_geometry_training import (  # noqa: E402
    AssistiveGeometryTrainDataset,
    collate_train_samples,
    flatten_train_manifest,
)
from scripts.research.assistive_geometry.depthart_training_scan import (  # noqa: E402
    install_depthart_training_scan,
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
    atomic_write_json,
    write_progress,
)
from scripts.research.assistive_geometry_qsf.h1_survival import (  # noqa: E402
    DepthArtAssistiveGeometryH1,
    QsfH1TaskHeads,
    compile_h1_targets,
    compute_h1_band_losses,
    discrete_survival_nll,
)
from scripts.research.assistive_geometry_qsf.validate_h1_train_canary import (  # noqa: E402
    PROTOCOL_RELATIVE,
    runtime_preflight,
    validate_protocol,
)


def select_parent_frames(
    frames: list[dict[str, Any]],
    parent_ids: tuple[str, ...],
    frames_per_parent: int,
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_parent[str(frame["video_id"])].append(frame)
    selected: list[dict[str, Any]] = []
    for parent in parent_ids:
        values = by_parent[parent]
        require(len(values) >= frames_per_parent, f"insufficient frames for parent {parent}")
        indices = np.linspace(0, len(values) - 1, frames_per_parent, dtype=np.int64)
        require(len(set(int(value) for value in indices)) == frames_per_parent, "frame selection duplicated")
        selected.extend(values[int(index)] for index in indices)
    return selected


def load_h1_model(
    source: Path,
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> tuple[DepthArtAssistiveGeometryH1, dict[str, Any]]:
    deployment = REPO_ROOT / "scripts/research/hftf/deployment/depthart"
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
    model = DepthArtAssistiveGeometryH1(base).to(device).eval()
    model.requires_grad_(False)
    return model, scan


def extract_features(
    model: DepthArtAssistiveGeometryH1,
    frames: list[dict[str, Any]],
    *,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    dataset = AssistiveGeometryTrainDataset(frames, seed, augment=False)
    by_orientation: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        by_orientation[str(frame["orientation_family"])].append(index)
    features: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    completed = 0
    for orientation in ("portrait", "landscape"):
        indices = by_orientation[orientation]
        for start in range(0, len(indices), 4):
            batch_indices = indices[start : start + 4]
            batch = collate_train_samples([dataset[index] for index in batch_indices])
            image = batch["image"].to(device)
            intrinsics = batch["targets"]["intrinsics_tensor"].to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                pooled = model.extract_band_features(image, intrinsics)
            require(bool(torch.isfinite(pooled).all().item()), "non-finite frozen band feature")
            features.append(pooled.float().cpu())
            for key in ("clearance_m", "clearance_valid", "occupancy", "occupancy_valid"):
                targets[key].append(batch["targets"][key].cpu())
            metadata.extend(batch["metadata"])
            completed += len(batch_indices)
            if progress_callback is not None:
                progress_callback(completed, len(frames))
    result = {
        "features": torch.cat(features, dim=0),
        "targets": {key: torch.cat(values, dim=0) for key, values in targets.items()},
        "metadata": metadata,
    }
    require(result["features"].shape == (len(frames), 3, 48), "feature tensor shape drift")
    compile_h1_targets(result["targets"])
    return result


def evaluate_head(head: QsfH1TaskHeads, payload: dict[str, Any]) -> dict[str, Any]:
    head.eval()
    features = payload["features"]
    targets = payload["targets"]
    compiled = compile_h1_targets(targets)
    with torch.no_grad():
        outputs = head.forward_bands(features)
        _, per_band = discrete_survival_nll(outputs["hazard_logits"], compiled)
    valid = compiled["distribution_valid"].bool()
    occupied = compiled["occupancy_valid"].bool() & (compiled["occupancy"] >= 0.5)
    predicted_clear = outputs["occupancy_probability"] < 0.5
    event = compiled["event_observed"].bool()
    clearance_error = (outputs["clearance_m"] - compiled["clearance_m"]).abs()
    monotonic_violations = int(
        (torch.diff(outputs["occupancy_probability"], dim=-1) < -1e-7).sum().item()
    )
    return {
        "survival_nll": float(per_band[valid].mean().item()),
        "distribution_valid_count": int(valid.sum().item()),
        "total_band_count": int(valid.numel()),
        "known_coverage": float(valid.float().mean().item()),
        "false_clear_count": int((predicted_clear & occupied).sum().item()),
        "occupied_known_count": int(occupied.sum().item()),
        "false_clear_rate": float(
            (predicted_clear & occupied).sum().item() / max(int(occupied.sum().item()), 1)
        ),
        "event_count": int(event.sum().item()),
        "clearance_mae_m": float(clearance_error[event].mean().item()),
        "horizon_monotonicity_violations": monotonic_violations,
    }


def train_head(
    head: QsfH1TaskHeads,
    payload: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, float]]:
    head.train()
    parameters = [
        *head.band_mlp.parameters(),
        *head.hazard_out.parameters(),
        *head.confidence_out.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    features = payload["features"]
    targets = payload["targets"]
    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(int(training["seed"]))
    history: list[dict[str, float]] = []
    for epoch in range(int(training["epochs"])):
        order = torch.randperm(len(features), generator=generator)
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_targets = {key: value[indices] for key, value in targets.items()}
            outputs = head.forward_bands(features[indices])
            losses = compute_h1_band_losses(
                outputs,
                batch_targets,
                active_losses=training["active_losses"],
            )
            require(bool(torch.isfinite(losses["total"]).item()), "non-finite H1 head loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0).item())
            require(np.isfinite(gradient), "non-finite H1 head gradient")
            optimizer.step()
            epoch_losses.append(float(losses["total"].detach().item()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "mean_train_loss": float(np.mean(epoch_losses)),
            }
        )
    return history


def _relative_improvement(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-12)


def apply_gates(
    before_fit: dict[str, Any],
    after_fit: dict[str, Any],
    before_eval: dict[str, Any],
    after_eval: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "fit_survival_nll": _relative_improvement(
            before_fit["survival_nll"], after_fit["survival_nll"]
        )
        >= gates["fit_survival_nll_relative_improvement_min"],
        "eval_survival_nll": _relative_improvement(
            before_eval["survival_nll"], after_eval["survival_nll"]
        )
        >= gates["eval_survival_nll_relative_improvement_min"],
        "eval_false_clear": after_eval["false_clear_rate"] - before_eval["false_clear_rate"]
        <= gates["eval_false_clear_rate_increase_max"],
        "eval_clearance": after_eval["clearance_mae_m"] - before_eval["clearance_mae_m"]
        <= gates["eval_clearance_mae_increase_max_m"],
        "structural_monotonicity": after_eval["horizon_monotonicity_violations"]
        <= gates["horizon_monotonicity_violations_max"],
        "known_coverage": abs(after_eval["known_coverage"] - before_eval["known_coverage"])
        <= gates["known_coverage_delta_max"],
    }


def save_checkpoint_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"checkpoint collision: {path}")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_pilot(protocol: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
    path = REPO_ROOT / protocol["outputs"]["pilot_parent"] / "pilot-r0" / "pilot-result.json"
    require(path.is_file(), "H1 run requires completed performance pilot")
    pilot = load_json(path)
    require(pilot.get("terminal") == "H1_TRAIN_CANARY_PERFORMANCE_QUALIFIED", "pilot not qualified")
    require(pilot.get("protocol_sha256") == protocol_sha, "pilot protocol binding drift")
    require(
        pilot.get("projected_full_wall_seconds")
        <= protocol["resource_scheduling"]["maximum_projected_wall_seconds"],
        "pilot projection exceeds CANARY_LITE bound",
    )
    return pilot


def execute(args: argparse.Namespace) -> int:
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    validate_protocol(protocol)
    protocol_sha = sha256_file(protocol_path)
    preflight = runtime_preflight(protocol)
    require(preflight["status"] == "READY", preflight["terminal"])
    target_manifest = (REPO_ROOT / protocol["inputs"]["target_manifest"]["path"]).resolve()
    source = Path(protocol["inputs"]["depthart_source"]["path"]).resolve()
    checkpoint = Path(protocol["inputs"]["initialization_checkpoint"]["path"]).resolve()
    frames = flatten_train_manifest(load_json(target_manifest))
    fit_parents = tuple(protocol["roster"]["fit_parent_video_ids"])
    eval_parents = tuple(protocol["roster"]["eval_parent_video_ids"])
    pilot_mode = args.mode == "pilot"
    frames_per_parent = 1 if pilot_mode else int(protocol["roster"]["frames_per_parent"])
    selected = select_parent_frames(frames, fit_parents + eval_parents, frames_per_parent)

    expected_parent = (REPO_ROOT / protocol["outputs"]["pilot_parent"]).resolve()
    expected_name = "pilot-r0" if pilot_mode else "run-r0"
    require(args.output_root.resolve().parent == expected_parent, "H1 output parent drift")
    require(args.output_root.name == expected_name and not args.output_root.exists(), "H1 output collision")
    if not pilot_mode:
        validate_pilot(protocol, protocol_sha)
        expected_model = (REPO_ROOT / protocol["outputs"]["model_parent"] / "h1-head-r0.pt").resolve()
        require(args.model_output is not None and args.model_output.resolve() == expected_model, "model output drift")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress_path = output_root / "progress.json"
    result_path = output_root / ("pilot-result.json" if pilot_mode else "result.json")
    failure_path = output_root / "failure.json"
    total = len(selected)
    completed = 0
    write_progress(
        progress_path,
        phase="extracting_frozen_features",
        completed=0,
        total=total,
        started_at=started,
        status="running",
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        require(device.type == "cuda", "H1 feature extraction requires CUDA")
        precision, amp_dtype, _ = choose_precision()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model, scan = load_h1_model(
                source,
                checkpoint,
                int(protocol["training"]["seed"]),
                device,
            )
            torch.cuda.reset_peak_memory_stats()

            def progress(value: int, maximum: int) -> None:
                nonlocal completed
                completed = value
                write_progress(
                    progress_path,
                    phase="extracting_frozen_features",
                    completed=value,
                    total=maximum,
                    started_at=started,
                    status="running",
                )

            payload = extract_features(
                model,
                selected,
                seed=int(protocol["training"]["seed"]),
                device=device,
                amp_dtype=amp_dtype,
                progress_callback=progress,
            )
            extraction_seconds = time.perf_counter() - started
            peak_vram_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            del model
            torch.cuda.empty_cache()

        if pilot_mode:
            projected = extraction_seconds * (
                (len(fit_parents) + len(eval_parents))
                * int(protocol["roster"]["frames_per_parent"])
                / max(total, 1)
            ) + 30.0
            maximum = projected * 2.0 + 60.0
            qualified = (
                projected <= protocol["resource_scheduling"]["maximum_projected_wall_seconds"]
                and maximum <= protocol["resource_scheduling"]["maximum_projected_wall_seconds"]
            )
            result = {
                "schema": "blindassist.assistive_geometry_qsf.h1_train_canary_pilot.v1",
                "protocol_sha256": protocol_sha,
                "mode": "PERFORMANCE_PILOT",
                "scientific_outcome_access": False,
                "frame_count": total,
                "feature_shape": list(payload["features"].shape),
                "feature_finite": bool(torch.isfinite(payload["features"]).all().item()),
                "precision": precision,
                "training_scan": scan,
                "extraction_wall_seconds": extraction_seconds,
                "projected_full_wall_seconds": projected,
                "maximum_expected_wall_seconds": maximum,
                "peak_vram_mib": peak_vram_mib,
                "warnings": [str(item.message) for item in captured],
                "terminal": (
                    "H1_TRAIN_CANARY_PERFORMANCE_QUALIFIED"
                    if qualified
                    else "H1_TRAIN_CANARY_PERFORMANCE_NOT_QUALIFIED"
                ),
            }
            atomic_write_json(result_path, result, exclusive=True)
            write_progress(
                progress_path,
                phase="complete" if qualified else "not_qualified",
                completed=total,
                total=total,
                started_at=started,
                status="complete" if qualified else "failed",
            )
            return 0 if qualified else 2

        fit_set = set(fit_parents)
        fit_indices = [
            index for index, row in enumerate(payload["metadata"]) if row["video_id"] in fit_set
        ]
        eval_indices = [
            index for index, row in enumerate(payload["metadata"]) if row["video_id"] not in fit_set
        ]

        def subset(indices: list[int]) -> dict[str, Any]:
            tensor_indices = torch.as_tensor(indices)
            return {
                "features": payload["features"][tensor_indices],
                "targets": {
                    key: value[tensor_indices] for key, value in payload["targets"].items()
                },
                "metadata": [payload["metadata"][index] for index in indices],
            }

        fit_payload = subset(fit_indices)
        eval_payload = subset(eval_indices)
        torch.manual_seed(int(protocol["training"]["seed"]))
        head = QsfH1TaskHeads()
        before_fit = evaluate_head(head, fit_payload)
        before_eval = evaluate_head(head, eval_payload)
        history = train_head(head, fit_payload, protocol["training"])
        after_fit = evaluate_head(head, fit_payload)
        after_eval = evaluate_head(head, eval_payload)
        gates = apply_gates(
            before_fit,
            after_fit,
            before_eval,
            after_eval,
            protocol["gates"],
        )
        terminal = (
            "H1_TRAIN_CANARY_PASS_LEARNABILITY_SUPPORTED"
            if all(gates.values())
            else "H1_TRAIN_CANARY_FAIL_IMPLEMENTATION_VERSION_NOT_SUPPORTED"
        )
        checkpoint_receipt = save_checkpoint_exclusive(
            args.model_output.resolve(),
            {
                "schema": "blindassist.assistive_geometry_qsf.h1_head_checkpoint.v1",
                "protocol_sha256": protocol_sha,
                "seed": protocol["training"]["seed"],
                "model": head.state_dict(),
                "model_state_sha256": state_digest(head),
                "terminal": terminal,
            },
        )
        result = {
            "schema": "blindassist.assistive_geometry_qsf.h1_train_canary_result.v1",
            "protocol_sha256": protocol_sha,
            "mode": "TRAIN_ONLY_CANARY",
            "data_role": "TRAIN_CANARY_PARENT_DISJOINT",
            "fit_parent_video_ids": list(fit_parents),
            "eval_parent_video_ids": list(eval_parents),
            "parent_overlap": False,
            "frames_per_parent": frames_per_parent,
            "fit_frame_count": len(fit_indices),
            "eval_frame_count": len(eval_indices),
            "initial": {"fit": before_fit, "eval": before_eval},
            "trained": {"fit": after_fit, "eval": after_eval},
            "relative_improvement": {
                "fit_survival_nll": _relative_improvement(
                    before_fit["survival_nll"], after_fit["survival_nll"]
                ),
                "eval_survival_nll": _relative_improvement(
                    before_eval["survival_nll"], after_eval["survival_nll"]
                ),
            },
            "gates": gates,
            "epoch_history": history,
            "checkpoint": checkpoint_receipt,
            "precision": precision,
            "training_scan": scan,
            "feature_extraction_wall_seconds": extraction_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_vram_mib": peak_vram_mib,
            "warnings": [str(item.message) for item in captured],
            "development_outcome_access": False,
            "confirmation_outcome_access": False,
            "h2_executed": False,
            "terminal": terminal,
            "claim_ceiling": (
                "Parent-disjoint TRAIN-only frozen-encoder H1 learnability canary; no B1 "
                "comparison, Development, Confirmation, device, product, or safety authority."
            ),
        }
        atomic_write_json(result_path, result, exclusive=True)
        write_progress(
            progress_path,
            phase="complete",
            completed=total,
            total=total,
            started_at=started,
            status="complete",
        )
        return 0
    except Exception as error:
        atomic_write_json(
            failure_path,
            {
                "schema": "blindassist.assistive_geometry_qsf.h1_train_canary_failure.v1",
                "protocol_sha256": protocol_sha,
                "mode": args.mode,
                "completed_frames": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scientific_outcome": "NOT_EVALUABLE_DUE_TO_EXECUTION",
                "terminal": "H1_TRAIN_CANARY_EXECUTION_INVALID",
            },
            exclusive=True,
        )
        write_progress(
            progress_path,
            phase="failed",
            completed=completed,
            total=total,
            started_at=started,
            status="failed",
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument("--mode", choices=("pilot", "run"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
