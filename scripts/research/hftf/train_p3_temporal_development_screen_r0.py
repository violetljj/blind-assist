#!/usr/bin/env python3
"""Train the single, development-only P3 temporal A2-392 candidate.

This is deliberately not a P1 evaluator or a deployment trainer.  It accepts
only the hash-bound development manifests defined below and refuses any Bonn
or sealed-holdout identity before importing a ML runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_392_student_p3_r0_1 import (
    DecoupledTemporalStateHead,
    STATE_TO_INDEX,
    TRANSITIONS,
    build_temporal_evidence,
    effective_number_transition_weights,
    temporal_distillation_loss,
)
from p3_r0_1_asset_common import (
    assert_outputs_absent,
    exact_fields,
    load_json,
    require,
    resolve_inside,
    sha256_file,
    valid_sha,
    verify_bound_file,
)


PROTOCOL_SCHEMA = "blindassist_p3_temporal_development_screen_r0_protocol"
MANIFEST_SCHEMA = "blindassist_p3_temporal_development_complete_manifest_r0"
TRAINING_RESULT_SCHEMA = "blindassist_p3_temporal_development_screen_r0_training_result"
TRAINING_RESULT_FIELDS = {
    "schema", "protocol_sha256", "evidence_limit", "activation_bindings_sha256",
    "train_manifest_sha256", "validation_manifest_sha256", "a2_checkpoint_sha256",
    "teacher_depth_sha256", "seed", "epochs_completed", "best_epoch",
    "best_validation_composite_total", "history", "checkpoint", "training_duration_s",
    "sealed_holdout_opened", "terminal",
}
CLIP_LENGTH = 4
INPUT_SIZE = 392
EXPECTED_TRAINING = {
    "seed": 20260805,
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "optimizer": "AdamW",
    "learning_rate": 0.00002,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "checkpoint_selection": "lowest_validation_composite_total_ties_earliest",
}
# This is intentionally exact and shared with the development asset producer
# and evaluator.  The source root resolves RGB identities; teacher_depth_ref
# resolves into the hash-bound flat teacher-depth NPY cache.
FRAME_FIELDS = {
    "frame_id", "video_id", "parent_id", "timestamp_ns", "rgb_identity", "rgb_sha256",
    "teacher_depth_ref", "teacher_depth_sha256", "teacher_timestamp_ns", "teacher_valid",
    "tof_valid", "frozen_a2_mean_abs_log_depth_disagreement", "clearance_m",
    "geometry_state", "geometry_target_valid", "truth_depth_path", "truth_depth_sha256",
    "truth_depth_scale_m", "intrinsics_fx_fy_cx_cy",
}


def _sha(value: Any, label: str) -> str:
    value = str(value).upper()
    require(valid_sha(value), f"{label} SHA invalid")
    return value


def _binding_equal(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    exact_fields(left, {"path", "sha256"}, label)
    exact_fields(right, {"path", "sha256"}, label)
    require(left == right, f"{label} binding differs from frozen protocol")


def _teacher_index(reference: Any) -> int:
    """Parse the sole allowed reference to the flat teacher depth cache."""
    prefix = "npy-index:"
    text = str(reference)
    require(text.startswith(prefix), "teacher depth reference must be npy-index:<nonnegative integer>")
    suffix = text[len(prefix):]
    require(suffix.isdigit(), "teacher depth index is malformed")
    return int(suffix)


def _resolve_source_rgb_for_video(repo_root: Path, source_root: Path, video_id: Any, identity: Any) -> Path:
    value = str(identity).replace("\\", "/")
    require(value and not Path(value).is_absolute(), "RGB identity must be a non-empty relative path")
    video = str(video_id)
    require(video and not Path(video).is_absolute() and "/" not in video and "\\" not in video, "video identity invalid")
    path = (source_root / video / value).resolve()
    try:
        path.relative_to(source_root)
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("RGB identity leaves source root or repository") from error
    return path


def _load_protocol(path: Path, repo_root: Path) -> dict[str, Any]:
    protocol = load_json(path)
    exact_fields(protocol, {
        "schema", "status", "claim_ceiling", "training", "a2", "teacher_cache",
        "implementation", "permanent_exclusions", "inputs", "development_screen",
    }, "development protocol")
    require(protocol["schema"] == PROTOCOL_SCHEMA, "development protocol schema drift")
    require(protocol["status"] == "FROZEN_BEFORE_MODEL_LOAD_OR_TRAINING", "development protocol is not frozen")
    require(protocol["claim_ceiling"] == "DEVELOPMENT_SIGNAL_ONLY", "claim ceiling drift")
    require(protocol["training"] == EXPECTED_TRAINING, "fixed training contract drift")
    exact_fields(protocol["a2"], {"checkpoint", "training_receipt", "dav2_repo", "dav2_dpt_source"}, "A2 bindings")
    exact_fields(protocol["teacher_cache"], {"manifest", "depth"}, "teacher cache bindings")
    exact_fields(protocol["implementation"], {
        "producer_sha256", "trainer_sha256", "evaluator_sha256", "temporal_module_sha256",
    }, "implementation")
    _sha(protocol["implementation"]["trainer_sha256"], "trainer")
    _sha(protocol["implementation"]["temporal_module_sha256"], "temporal module")
    require(isinstance(protocol["inputs"], dict), "protocol inputs malformed")
    require(isinstance(protocol["development_screen"], dict), "development screen rules malformed")
    exact_fields(protocol["permanent_exclusions"], {
        "banned_parent_ids", "banned_path_fragments", "holdout_outcomes_opened",
    }, "permanent exclusions")
    require(protocol["permanent_exclusions"]["holdout_outcomes_opened"] is False, "holdout was opened")
    require(isinstance(protocol["permanent_exclusions"]["banned_parent_ids"], list), "banned parents malformed")
    require(isinstance(protocol["permanent_exclusions"]["banned_path_fragments"], list), "banned paths malformed")
    return protocol


def _load_activation_bindings(repo_root: Path, path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate materialized assets separately from the rule-only protocol."""
    activation = load_json(path)
    exact_fields(activation, {
        "schema", "protocol_sha256", "claim_ceiling", "train_manifest",
        "validation_manifest", "class_weights", "disagreement_cache", "runtime_state",
        "terminal",
    }, "activation bindings")
    require(activation["schema"] == "blindassist_p3_temporal_development_screen_r0_activation_bindings", "activation schema drift")
    require(activation["claim_ceiling"] == "DEVELOPMENT_SIGNAL_ONLY", "activation claim ceiling drift")
    require(_sha(activation["protocol_sha256"], "activation protocol") == sha256_file(protocol["_path"]), "activation protocol SHA mismatch")
    for key in ("train_manifest", "validation_manifest", "class_weights", "disagreement_cache"):
        verify_bound_file(repo_root, activation[key], key.replace("_", " "))
    require(activation["terminal"] == "P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY", "activation terminal drift")
    exact_fields(activation["runtime_state"], {
        "bonn_sealed_bundle_read", "holdout_outcomes_opened", "p3_model_constructed",
        "optimizer_constructed", "training_started", "a2_loaded_only_for_frozen_disagreement",
    }, "activation runtime state")
    state = activation["runtime_state"]
    require(state["bonn_sealed_bundle_read"] is False and state["holdout_outcomes_opened"] is False, "holdout boundary drift")
    require(state["p3_model_constructed"] is False and state["optimizer_constructed"] is False and state["training_started"] is False, "activation is not pre-training")
    require(state["a2_loaded_only_for_frozen_disagreement"] is True, "A2 asset materialization boundary drift")
    return activation


def _validate_manifest(
    repo_root: Path,
    path: Path,
    binding: dict[str, Any],
    role: str,
    teacher_records: dict[int, dict[str, Any]],
    exclusions: dict[str, Any],
    source_root: Path,
    expected_teacher_depth_sha256: str,
) -> list[dict[str, Any]]:
    verify_bound_file(repo_root, binding, f"{role} manifest")
    manifest = load_json(path)
    exact_fields(manifest, {"schema", "protocol_sha256", "evidence_limit", "role", "clips"}, f"{role} manifest")
    require(manifest["schema"] == MANIFEST_SCHEMA and manifest["role"] == role, f"{role} manifest schema/role drift")
    require(manifest["evidence_limit"] == "DEVELOPMENT_SIGNAL_ONLY", "manifest evidence ceiling drift")
    _sha(manifest["protocol_sha256"], "manifest protocol")
    clips = manifest["clips"]
    require(isinstance(clips, list) and clips, f"{role} clips missing")
    banned_parents = {str(value) for value in exclusions["banned_parent_ids"]}
    banned_paths = [str(value).replace("\\", "/").lower() for value in exclusions["banned_path_fragments"]]
    parents: set[str] = set()
    frames_seen: set[str] = set()
    for clip in clips:
        exact_fields(clip, {"clip_id", "video_id", "parent_id", "frames"}, "development clip")
        require(isinstance(clip["frames"], list) and len(clip["frames"]) == CLIP_LENGTH, "clip length must be four")
        parent = str(clip["parent_id"])
        require(parent not in banned_parents and "bonn" not in parent.lower(), "Bonn/sealed parent is forbidden")
        parents.add(parent)
        timestamps: list[int] = []
        for frame in clip["frames"]:
            exact_fields(frame, FRAME_FIELDS, "development frame")
            frame_id = str(frame["frame_id"])
            require(frame_id and frame_id not in frames_seen, "frame reuse across development clips")
            frames_seen.add(frame_id)
            require(str(frame["parent_id"]) == parent and str(frame["video_id"]) == str(clip["video_id"]), "clip identity drift")
            require(isinstance(frame["timestamp_ns"], int) and frame["timestamp_ns"] > 0, "timestamp invalid")
            timestamps.append(frame["timestamp_ns"])
            rgb_path = _resolve_source_rgb_for_video(repo_root, source_root, frame["video_id"], frame["rgb_identity"])
            normalized_path = str(rgb_path).replace("\\", "/").lower()
            require(not any(fragment in normalized_path for fragment in banned_paths), "sealed/holdout path is forbidden")
            require(rgb_path.is_file() and sha256_file(rgb_path) == _sha(frame["rgb_sha256"], "RGB"), "RGB binding mismatch")
            index = _teacher_index(frame["teacher_depth_ref"])
            require(isinstance(index, int) and index in teacher_records, "teacher index absent")
            record = teacher_records[index]
            require(str(record["frame_id"]) == frame_id and str(record["parent_id"]) == parent, "teacher roster identity mismatch")
            require(_sha(frame["teacher_depth_sha256"], "teacher depth") == expected_teacher_depth_sha256, "teacher depth SHA drift")
            require(isinstance(frame["teacher_timestamp_ns"], int) and 0 < frame["teacher_timestamp_ns"] <= frame["timestamp_ns"], "teacher timestamp invalid")
            require(isinstance(frame["teacher_valid"], bool), "teacher validity invalid")
            require(isinstance(frame["tof_valid"], bool), "sensor validity invalid")
            require(isinstance(frame["frozen_a2_mean_abs_log_depth_disagreement"], (int, float)) and math.isfinite(float(frame["frozen_a2_mean_abs_log_depth_disagreement"])) and float(frame["frozen_a2_mean_abs_log_depth_disagreement"]) >= 0.0, "frozen disagreement invalid")
            require(isinstance(frame["clearance_m"], list) and len(frame["clearance_m"]) == 3, "clearance target invalid")
            require(isinstance(frame["geometry_state"], list) and len(frame["geometry_state"]) == 3, "geometry state invalid")
            require(isinstance(frame["geometry_target_valid"], list) and len(frame["geometry_target_valid"]) == 3, "geometry validity invalid")
            require(_sha(frame["truth_depth_sha256"], "truth depth"), "truth depth SHA invalid")
            require(isinstance(frame["truth_depth_path"], str) and frame["truth_depth_path"], "truth depth path invalid")
            require(isinstance(frame["truth_depth_scale_m"], (int, float)) and math.isfinite(float(frame["truth_depth_scale_m"])) and float(frame["truth_depth_scale_m"]) > 0.0, "truth depth scale invalid")
            require(isinstance(frame["intrinsics_fx_fy_cx_cy"], list) and len(frame["intrinsics_fx_fy_cx_cy"]) == 4 and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in frame["intrinsics_fx_fy_cx_cy"]), "intrinsics invalid")
            for clearance, state, valid in zip(frame["clearance_m"], frame["geometry_state"], frame["geometry_target_valid"]):
                require(state in STATE_TO_INDEX and isinstance(valid, bool), "geometry target malformed")
                require(clearance is None or (isinstance(clearance, (int, float)) and math.isfinite(float(clearance)) and float(clearance) >= 0.0), "clearance malformed")
        require(all(0 < right - left <= 500_000_000 for left, right in zip(timestamps, timestamps[1:])), "clip timestamp gap invalid")
    require(parents, f"{role} has no parents")
    return clips


def preflight(
    repo_root: Path,
    protocol_path: Path,
    activation_path: Path,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path, Path]:
    """Perform all file/schema/identity checks before ML imports or model loading."""
    protocol = _load_protocol(protocol_path, repo_root)
    protocol["_path"] = protocol_path
    require(sha256_file(Path(__file__).resolve()) == _sha(protocol["implementation"]["trainer_sha256"], "trainer"), "trainer source hash mismatch")
    module_path = SCRIPT_DIR / "dav2_temporal_392_student_p3_r0_1.py"
    require(sha256_file(module_path) == _sha(protocol["implementation"]["temporal_module_sha256"], "temporal module"), "temporal module hash mismatch")
    require(source_root.is_dir(), "source root missing")
    try:
        source_root.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("source root leaves repository") from error
    activation = _load_activation_bindings(repo_root, activation_path, protocol)
    train_binding = activation["train_manifest"]
    validation_binding = activation["validation_manifest"]
    checkpoint_path = verify_bound_file(repo_root, protocol["a2"]["checkpoint"], "A2 checkpoint")
    receipt_path = verify_bound_file(repo_root, protocol["a2"]["training_receipt"], "A2 training receipt")
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == "blindassist_dav2_392_distillation_a2_r0_training_result", "A2 receipt schema drift")
    require(receipt.get("terminal") == "A2_DISTILLATION_TRAINING_COMPLETE_P1_UNOPENED" and receipt.get("truth_inputs_opened") is False, "A2 receipt boundary drift")
    require(receipt.get("checkpoint", {}).get("sha256") == protocol["a2"]["checkpoint"]["sha256"].upper(), "A2 selected checkpoint mismatch")
    teacher_manifest_path = verify_bound_file(repo_root, protocol["teacher_cache"]["manifest"], "teacher cache manifest")
    teacher_depth_path = verify_bound_file(repo_root, protocol["teacher_cache"]["depth"], "teacher cache depth")
    verify_bound_file(repo_root, protocol["a2"]["dav2_dpt_source"], "DA V2 DPT source")
    exact_fields(protocol["a2"]["dav2_repo"], {"path"}, "DA V2 repository")
    require(resolve_inside(repo_root, str(protocol["a2"]["dav2_repo"]["path"])).is_dir(), "DA V2 repository missing")
    teacher_manifest = load_json(teacher_manifest_path)
    require(teacher_manifest.get("truth_inputs_opened") is False and isinstance(teacher_manifest.get("records"), list), "teacher cache boundary drift")
    records = {int(row["index"]): row for row in teacher_manifest["records"]}
    require(len(records) == len(teacher_manifest["records"]), "duplicate teacher indices")
    train_path = verify_bound_file(repo_root, train_binding, "train manifest")
    validation_path = verify_bound_file(repo_root, validation_binding, "validation manifest")
    train = _validate_manifest(repo_root, train_path, train_binding, "train", records, protocol["permanent_exclusions"], source_root, _sha(protocol["teacher_cache"]["depth"]["sha256"], "teacher cache depth"))
    validation = _validate_manifest(repo_root, validation_path, validation_binding, "validation", records, protocol["permanent_exclusions"], source_root, _sha(protocol["teacher_cache"]["depth"]["sha256"], "teacher cache depth"))
    train_parents = {str(clip["parent_id"]) for clip in train}
    validation_parents = {str(clip["parent_id"]) for clip in validation}
    require(train_parents.isdisjoint(validation_parents), "train/validation parent overlap")
    return protocol, activation, train, validation, teacher_depth_path, checkpoint_path


def _class_weights(clips: list[dict[str, Any]]) -> torch.Tensor:
    counts = {name: 0 for name in TRANSITIONS}
    for clip in clips:
        frames = clip["frames"]
        for left, right in zip(frames, frames[1:]):
            for band in range(3):
                if left["geometry_target_valid"][band] and right["geometry_target_valid"][band]:
                    counts[f"{left['geometry_state'][band]}_TO_{right['geometry_state'][band]}"] += 1
    return effective_number_transition_weights(counts)


def _load_clip_batch(
    clips: list[dict[str, Any]],
    teacher_depth: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    repo_root: Path,
    source_root: Path,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    import cv2
    images: list[torch.Tensor] = []
    teacher: list[np.ndarray] = []
    timestamps: list[list[int]] = []
    teacher_timestamps: list[list[int]] = []
    teacher_valid: list[list[bool]] = []
    sensor_valid: list[list[bool]] = []
    disagreement: list[list[float]] = []
    clearance: list[list[list[float]]] = []
    states: list[list[list[int]]] = []
    geometry_valid: list[list[list[bool]]] = []
    for clip in clips:
        frames = clip["frames"]
        for frame in frames:
            rgb_path = _resolve_source_rgb_for_video(repo_root, source_root, frame["video_id"], frame["rgb_identity"])
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (192, 256):
                raise OSError(f"cannot decode development RGB: {rgb_path}")
            image, _ = model.image2tensor(bgr, INPUT_SIZE)
            images.append(image)
            teacher.append(np.asarray(teacher_depth[_teacher_index(frame["teacher_depth_ref"])], dtype=np.float32))
        timestamps.append([frame["timestamp_ns"] for frame in frames])
        teacher_timestamps.append([frame["teacher_timestamp_ns"] for frame in frames])
        teacher_valid.append([frame["teacher_valid"] for frame in frames])
        sensor_valid.append([frame["tof_valid"] for frame in frames])
        disagreement.append([frame["frozen_a2_mean_abs_log_depth_disagreement"] for frame in frames])
        clearance.append([[float(value) if value is not None else float("nan") for value in frame["clearance_m"]] for frame in frames])
        states.append([[STATE_TO_INDEX[value] for value in frame["geometry_state"]] for frame in frames])
        geometry_valid.append([frame["geometry_target_valid"] for frame in frames])
    batch = len(clips)
    return (
        torch.cat(images, dim=0).to(device),
        torch.from_numpy(np.stack(teacher).reshape(batch, CLIP_LENGTH, 192, 256)).to(device),
        torch.tensor(timestamps, device=device), torch.tensor(teacher_timestamps, device=device),
        torch.tensor(teacher_valid, device=device), torch.tensor(sensor_valid, device=device),
        torch.tensor(disagreement, dtype=torch.float32, device=device),
        torch.tensor(clearance, dtype=torch.float32, device=device),
        torch.tensor(states, device=device), torch.tensor(geometry_valid, device=device),
    )


def _composite(model: torch.nn.Module, head: DecoupledTemporalStateHead, clips: list[dict[str, Any]], teacher_depth: np.ndarray, class_weights: torch.Tensor, device: torch.device, repo_root: Path, source_root: Path) -> tuple[torch.Tensor, dict[str, float]]:
    (images, teacher, timestamps, teacher_timestamps, teacher_valid, sensor_valid, disagreement, clearance, states, geometry_valid) = _load_clip_batch(clips, teacher_depth, model, device, repo_root, source_root)
    prediction = functional.interpolate(model(images)[:, None], size=(192, 256), mode="bilinear", align_corners=True)[:, 0]
    prediction = prediction.reshape(len(clips), CLIP_LENGTH, 192, 256)
    evidence = build_temporal_evidence(timestamps, teacher_timestamps, teacher_valid, sensor_valid, disagreement)
    delta_seconds = (timestamps[:, 1:] - timestamps[:, :-1]).float() / 1_000_000_000.0
    output = head(prediction, evidence, delta_seconds)
    loss, parts = temporal_distillation_loss(prediction, teacher, torch.isfinite(teacher) & (teacher > 0.0), evidence, output, clearance, states, geometry_valid, class_weights.to(device))
    return loss, {key: float(value.detach().cpu()) for key, value in parts.items()}


def _save_checkpoint(backbone: torch.nn.Module, head: torch.nn.Module, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    state = {"backbone": {key: value.detach().cpu() for key, value in backbone.state_dict().items()}, "temporal_head": {key: value.detach().cpu() for key, value in head.state_dict().items()}}
    torch.save(state, partial)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-bindings", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    protocol_path = args.protocol.resolve()
    require(protocol_path.is_file(), "protocol missing")
    activation_path = args.activation_bindings.resolve()
    require(activation_path.is_file(), "activation bindings missing")
    try:
        activation_path.relative_to(root)
    except ValueError as error:
        raise ValueError("activation bindings leave repository") from error
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    try:
        output_root.relative_to(root)
    except ValueError as error:
        raise ValueError("output root leaves repository") from error
    assert_outputs_absent(root, [str(output_root.relative_to(root))])
    protocol, activation, train, validation, teacher_depth_path, checkpoint_path = preflight(root, protocol_path, activation_path, source_root)

    # Delayed imports: every binding, identity and holdout check above has passed.
    dav2_root = resolve_inside(root, str(protocol["a2"]["dav2_repo"]["path"]))
    require(dav2_root.is_dir(), "DA V2 repository missing")
    sys.path.insert(0, str(dav2_root / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore[import-not-found]
    config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": 20}
    device = torch.device("cuda")
    backbone = DepthAnythingV2(**config)
    backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    backbone = backbone.to(device)
    head = DecoupledTemporalStateHead().to(device)
    training = protocol["training"]
    seed = int(training["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    teacher_depth = np.load(teacher_depth_path, mmap_mode="r")
    require(teacher_depth.ndim == 3 and teacher_depth.shape[1:] == (192, 256), "teacher depth shape drift")
    weights = _class_weights(train).to(device)
    output_root.mkdir(parents=True)
    checkpoint = output_root / "p3_temporal_development_best.pth"
    best = math.inf; best_epoch: int | None = None; history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        backbone.train(); head.train()
        order = [train[index] for index in np.random.default_rng(seed + epoch).permutation(len(train))]
        totals: list[float] = []; optimizer.zero_grad(set_to_none=True)
        for number, clip in enumerate(order):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, parts = _composite(backbone, head, [clip], teacher_depth, weights, device, root, source_root)
                scaled = loss / int(training["gradient_accumulation_steps"])
            scaler.scale(scaled).backward(); totals.append(parts["total"])
            last = number + 1 == len(order)
            if (number + 1) % int(training["gradient_accumulation_steps"]) == 0 or last:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), float(training["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        backbone.eval(); head.eval(); values: list[dict[str, float]] = []
        with torch.inference_mode():
            for clip in validation:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _loss, parts = _composite(backbone, head, [clip], teacher_depth, weights, device, root, source_root)
                values.append(parts)
        validation_total = statistics.fmean(value["total"] for value in values)
        validation_summary = {key: statistics.fmean(value[key] for value in values) for key in values[0]}
        history.append({"epoch": epoch + 1, "train_composite_total_mean": statistics.fmean(totals), "validation": validation_summary})
        print(json.dumps(history[-1]), flush=True)
        if validation_total < best:
            best = validation_total; best_epoch = epoch + 1; _save_checkpoint(backbone, head, checkpoint)
    require(best_epoch is not None and checkpoint.is_file(), "no checkpoint selected")
    result = {
        "schema": TRAINING_RESULT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "evidence_limit": "DEVELOPMENT_SIGNAL_ONLY",
        "activation_bindings_sha256": sha256_file(activation_path),
        "train_manifest_sha256": activation["train_manifest"]["sha256"],
        "validation_manifest_sha256": activation["validation_manifest"]["sha256"],
        "a2_checkpoint_sha256": sha256_file(checkpoint_path),
        "teacher_depth_sha256": sha256_file(teacher_depth_path),
        "seed": seed,
        "epochs_completed": int(training["epochs"]),
        "best_epoch": best_epoch,
        "best_validation_composite_total": best,
        "history": history,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "training_duration_s": time.perf_counter() - started,
        "sealed_holdout_opened": False,
        "terminal": "P3_TEMPORAL_DEVELOPMENT_SCREEN_TRAINING_COMPLETE_EVALUATION_PENDING",
    }
    exact_fields(result, TRAINING_RESULT_FIELDS, "training result")
    (output_root / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
