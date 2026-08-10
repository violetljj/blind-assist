#!/usr/bin/env python3
"""Train a soft-distance boundary student and open a Bonn-only boundary canary."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter

from materialize_ag_st_source_native_boundary_corpus import conservative_source_boundary
from train_ag_st_source_boundary_student import (
    DEFAULT_BINDING,
    DEFAULT_MOBILENET,
    THRESHOLD_GRID,
    TarImageReader,
    _load_bound_rgb,
    _load_target,
    average_precision,
    deterministic_split,
    require,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-soft-boundary-bonn-canary-r0"
DEFAULT_BONN_ROOT = (
    REPO_ROOT
    / "artifacts.local/datasets/bonn-rgbd-dynamic-full-r0/rgbd_bonn_dataset"
)
DEFAULT_BONN_COHORT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json"
)
BONN_INTRINSICS = np.asarray(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
BONN_HEIGHT = 480
BONN_WIDTH = 640
BONN_DEPTH_SCALE = 5000.0
MAX_ASSOCIATION_DELTA_SECONDS = 0.05
SOFT_SIGMA_PX = 3.0


def soft_boundary_target(
    probability: np.ndarray,
    valid: np.ndarray,
    sigma_px: float = SOFT_SIGMA_PX,
) -> np.ndarray:
    """Turn source-derived boundary cores into a continuous distance heatmap."""

    score = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    require(score.shape == mask.shape and sigma_px > 0.0, "soft target input invalid")
    core = mask & (score >= 0.5)
    if np.any(core):
        distance = distance_transform_edt(~core).astype(np.float32)
        heat = np.exp(-0.5 * np.square(distance / sigma_px)).astype(np.float32)
        output = np.maximum(score, heat)
    else:
        output = score.copy()
    output[~mask] = 0.0
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def boundary_metrics(
    frame_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
    tolerance_px: int,
) -> dict[str, Any]:
    require(frame_values and tolerance_px >= 0, "boundary metric input invalid")
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    predicted_count = target_count = precision_hit = recall_hit = 0
    filter_size = 2 * tolerance_px + 1
    for probability, target, valid in frame_values:
        require(probability.shape == target.shape == valid.shape, "boundary metric shape drift")
        targets.append(target[valid])
        scores.append(probability[valid])
        predicted = (probability >= threshold) & valid
        truth = target & valid
        predicted_count += int(np.sum(predicted))
        target_count += int(np.sum(truth))
        truth_near = maximum_filter(truth.astype(np.uint8), size=filter_size) > 0
        predicted_near = maximum_filter(predicted.astype(np.uint8), size=filter_size) > 0
        precision_hit += int(np.sum(predicted & truth_near))
        recall_hit += int(np.sum(truth & predicted_near))
    target_flat = np.concatenate(targets)
    score_flat = np.concatenate(scores)
    require(bool(np.any(target_flat)), "boundary metric positive denominator empty")
    clipped = np.clip(score_flat, 1e-6, 1.0 - 1e-6)
    prevalence = float(np.mean(target_flat))
    precision = precision_hit / max(1, predicted_count)
    recall = recall_hit / max(1, target_count)
    return {
        "valid_pixels": int(target_flat.size),
        "positive_pixels": int(np.sum(target_flat)),
        "prevalence": prevalence,
        "constant_average_precision": prevalence,
        "student_average_precision": average_precision(target_flat, score_flat),
        "student_bce": float(
            np.mean(
                -target_flat.astype(np.float64) * np.log(clipped)
                - (~target_flat).astype(np.float64) * np.log(1.0 - clipped)
            )
        ),
        "threshold": threshold,
        "tolerance_px": tolerance_px,
        "predicted_pixels": predicted_count,
        "precision_within_tolerance": precision,
        "recall_within_tolerance": recall,
        "f1_within_tolerance": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


@dataclass(frozen=True)
class IndexRow:
    row_index: int
    timestamp_seconds: float
    absolute_path: Path


@dataclass(frozen=True)
class BonnPair:
    parent_id: str
    rgb: IndexRow
    depth: IndexRow
    delta_seconds: float


def read_index(sequence_root: Path, filename: str) -> list[IndexRow]:
    path = sequence_root / filename
    require(path.is_file(), f"Bonn index missing: {path}")
    rows: list[IndexRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 2, f"Bonn index row invalid: {path}")
        member = (sequence_root / fields[1]).resolve()
        member.relative_to(sequence_root.resolve())
        rows.append(IndexRow(len(rows), float(fields[0]), member))
    require(rows and all(a.timestamp_seconds < b.timestamp_seconds for a, b in pairwise(rows)), "Bonn index order invalid")
    return rows


def pair_selected_frames(
    bonn_root: Path,
    cohort_path: Path,
) -> list[BonnPair]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    require(cohort.get("schema") == "blindassist_ag_st_bonn_mixed_domain_cohort_v1", "Bonn cohort schema drift")
    parents = cohort.get("evaluation_parents")
    require(isinstance(parents, list) and len(parents) == 8, "Bonn evaluation parent count drift")
    output: list[BonnPair] = []
    for parent in parents:
        parent_id = str(parent["parent_id"])
        requested = {int(value) for value in parent["rgb_row_indices_zero_based"]}
        require(len(requested) == 3, f"Bonn frame selection drift: {parent_id}")
        sequence_root = (bonn_root / parent_id).resolve()
        rgb_rows = read_index(sequence_root, "rgb.txt")
        depth_rows = [row for row in read_index(sequence_root, "depth.txt") if row.absolute_path.is_file()]
        depth_times = [row.timestamp_seconds for row in depth_rows]
        used: set[int] = set()
        selected: dict[int, BonnPair] = {}
        for rgb in rgb_rows:
            insertion = bisect.bisect_left(depth_times, rgb.timestamp_seconds)
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depth_rows) and index not in used]
            if not candidates:
                continue
            best = min(candidates, key=lambda index: (abs(depth_times[index] - rgb.timestamp_seconds), index))
            delta = abs(depth_times[best] - rgb.timestamp_seconds)
            if delta > MAX_ASSOCIATION_DELTA_SECONDS:
                continue
            used.add(best)
            if rgb.row_index in requested:
                selected[rgb.row_index] = BonnPair(parent_id, rgb, depth_rows[best], delta)
        require(set(selected) == requested, f"Bonn selected pair missing: {parent_id}")
        output.extend(selected[index] for index in sorted(selected))
    require(len(output) == 24, "Bonn evaluation frame count drift")
    return output


def load_bonn_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(value.shape == (BONN_HEIGHT, BONN_WIDTH, 3), "Bonn RGB shape drift")
    return value


def load_bonn_boundary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    require(raw.shape == (BONN_HEIGHT, BONN_WIDTH) and raw.dtype == np.uint16, "Bonn depth shape/dtype drift")
    valid = raw > 0
    depth = raw.astype(np.float32) / BONN_DEPTH_SCALE
    probability, boundary_valid = conservative_source_boundary(depth, valid, BONN_INTRINSICS)
    return boundary_valid & (probability >= 0.5), boundary_valid


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import mobilenet_v3_small

    require(torch.cuda.is_available(), "soft-boundary canary requires CUDA")
    require(args.binding.is_file() and args.mobilenet_checkpoint.is_file(), "training input missing")
    require(args.bonn_root.is_dir() and args.bonn_cohort.is_file(), "Bonn canary input missing")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    require(binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS", "RGB binding invalid")
    descriptors = list(binding["frames"])
    split = deterministic_split(descriptors)
    role_by_parent = {identity: role for role, identities in split.items() for identity in identities}
    indices_by_role: dict[str, list[int]] = {role: [] for role in split}
    for index, row in enumerate(descriptors):
        indices_by_role[role_by_parent[(str(row["source"]), str(row["parent_id"]))]].append(index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    backbone_model = mobilenet_v3_small(weights=None)
    backbone_model.load_state_dict(torch.load(args.mobilenet_checkpoint, map_location="cpu", weights_only=True), strict=True)
    backbone = backbone_model.features.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def extract_features(rgb: np.ndarray) -> tuple[torch.Tensor, ...]:
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32) / 255.0
        value = ((tensor - mean) / std)[None]
        captured: list[torch.Tensor] = []
        with torch.no_grad():
            for layer_index, layer in enumerate(backbone):
                value = layer(value)
                if layer_index in (1, 3, 8, 12):
                    captured.append(value.detach().cpu().half())
        require(len(captured) == 4, "MobileNet feature capture drift")
        return tuple(captured)

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p32 = nn.Conv2d(576, 64, 1)
            self.p16 = nn.Conv2d(48, 64, 1)
            self.b16 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU())
            self.p8 = nn.Conv2d(24, 32, 1)
            self.b8 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU())
            self.p4 = nn.Conv2d(16, 24, 1)
            self.b4 = nn.Sequential(nn.Conv2d(56, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.SiLU())
            self.output = nn.Conv2d(24, 1, 1)

        def forward(self, values: tuple[torch.Tensor, ...], output_hw: tuple[int, int]) -> torch.Tensor:
            c4, c8, c16, c32 = values
            value = functional.interpolate(self.p32(c32), size=c16.shape[-2:], mode="bilinear", align_corners=False)
            value = self.b16(torch.cat((value, self.p16(c16)), dim=1))
            value = functional.interpolate(value, size=c8.shape[-2:], mode="bilinear", align_corners=False)
            value = self.b8(torch.cat((value, self.p8(c8)), dim=1))
            value = functional.interpolate(value, size=c4.shape[-2:], mode="bilinear", align_corners=False)
            value = self.b4(torch.cat((value, self.p4(c4)), dim=1))
            return functional.interpolate(self.output(value), size=output_hw, mode="bilinear", align_corners=False)[:, 0]

    started = time.monotonic()
    feature_cache: dict[int, tuple[torch.Tensor, ...]] = {}
    tar_reader = TarImageReader()
    try:
        for role in ("fit", "selection"):
            for index in indices_by_role[role]:
                feature_cache[index] = extract_features(_load_bound_rgb(descriptors[index], tar_reader))
    finally:
        tar_reader.close()
    exact_targets = {
        index: _load_target(descriptors[index])
        for role in ("fit", "selection")
        for index in indices_by_role[role]
    }
    fit_targets = {
        index: {
            **target,
            "probability": soft_boundary_target(target["probability"], target["valid"], args.soft_sigma_px),
        }
        for index, target in exact_targets.items()
        if index in indices_by_role["fit"]
    }

    decoder = Decoder().to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    trainable_parameters = sum(parameter.numel() for parameter in decoder.parameters())

    def predict(features: tuple[torch.Tensor, ...], output_hw: tuple[int, int]) -> np.ndarray:
        values = tuple(value.to(device=device, dtype=torch.float32) for value in features)
        with torch.no_grad():
            logits = decoder(values, output_hw)
        return torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)

    def selection_metrics(threshold: float) -> dict[str, Any]:
        decoder.eval()
        grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
        for index in indices_by_role["selection"]:
            target = exact_targets[index]
            grouped[str(descriptors[index]["source"])].append(
                (predict(feature_cache[index], target["valid"].shape), target["positive"], target["valid"])
            )
        by_source = {source: boundary_metrics(values, threshold, 2) for source, values in sorted(grouped.items())}
        return {
            "by_source": by_source,
            "source_macro_average_precision": float(np.mean([value["student_average_precision"] for value in by_source.values()])),
            "source_macro_f1": float(np.mean([value["f1_within_tolerance"] for value in by_source.values()])),
        }

    fit_by_source: dict[str, list[int]] = defaultdict(list)
    for index in indices_by_role["fit"]:
        fit_by_source[str(descriptors[index]["source"])].append(index)
    samples_per_source = max(len(values) for values in fit_by_source.values())
    selection_history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float] | None = None
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        rng = np.random.default_rng(args.seed + epoch)
        schedules: dict[str, list[int]] = {}
        for source, values in sorted(fit_by_source.items()):
            schedule: list[int] = []
            while len(schedule) < samples_per_source:
                schedule.extend(int(value) for value in rng.permutation(values))
            schedules[source] = schedule[:samples_per_source]
        for position in range(samples_per_source):
            for source in sorted(schedules):
                index = schedules[source][position]
                target_np = fit_targets[index]
                features = tuple(value.to(device=device, dtype=torch.float32) for value in feature_cache[index])
                target = torch.from_numpy(target_np["probability"]).to(device=device)[None]
                valid = torch.from_numpy(target_np["valid"]).to(device=device)[None]
                logits = decoder(features, target.shape[-2:])
                positive = valid & (target >= 0.5)
                positive_count = int(positive.sum().item())
                negative_count = int((valid & ~positive).sum().item())
                pos_weight = min(20.0, negative_count / max(1, positive_count))
                weights = torch.where(positive, pos_weight, 1.0)
                bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                loss_bce = (bce[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)
                probability = torch.sigmoid(logits)[valid]
                target_valid = target[valid]
                dice = 1.0 - (2.0 * torch.sum(probability * target_valid) + 1.0) / (
                    torch.sum(probability) + torch.sum(target_valid) + 1.0
                )
                loss = loss_bce + 0.5 * dice
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
        if epoch % args.selection_interval == 0 or epoch == args.epochs:
            metrics = selection_metrics(0.5)
            key = (metrics["source_macro_average_precision"], metrics["source_macro_f1"])
            selection_history.append({"epoch": epoch, "step": global_step, "metrics": metrics})
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()}
    require(best_state is not None, "selection produced no checkpoint")
    decoder.load_state_dict(best_state, strict=True)
    threshold_rows = [{"threshold": threshold, "metrics": selection_metrics(threshold)} for threshold in THRESHOLD_GRID]
    selected_threshold_row = max(
        threshold_rows,
        key=lambda row: (row["metrics"]["source_macro_f1"], row["metrics"]["source_macro_average_precision"], -abs(row["threshold"] - 0.5)),
    )
    selected_threshold = float(selected_threshold_row["threshold"])
    selected_epoch = max(selection_history, key=lambda row: (row["metrics"]["source_macro_average_precision"], row["metrics"]["source_macro_f1"]))["epoch"]

    # R7 canary files are never opened. Bonn RGB inference is complete before source depth is opened.
    bonn_pairs = pair_selected_frames(args.bonn_root, args.bonn_cohort)
    bonn_features = [extract_features(load_bonn_rgb(pair.rgb.absolute_path)) for pair in bonn_pairs]
    del backbone, backbone_model
    torch.cuda.empty_cache()
    bonn_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    bonn_by_parent: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    bonn_receipts: list[dict[str, Any]] = []
    decoder.eval()
    for pair, features in zip(bonn_pairs, bonn_features, strict=True):
        truth, valid = load_bonn_boundary(pair.depth.absolute_path)
        prediction = predict(features, truth.shape)
        item = (prediction, truth, valid)
        bonn_values.append(item)
        bonn_by_parent[pair.parent_id].append(item)
        bonn_receipts.append(
            {
                "parent_id": pair.parent_id,
                "rgb_path": str(pair.rgb.absolute_path),
                "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                "depth_path": str(pair.depth.absolute_path),
                "depth_sha256": sha256_file(pair.depth.absolute_path),
                "association_delta_seconds": pair.delta_seconds,
                "valid_pixels": int(np.sum(valid)),
                "positive_pixels": int(np.sum(truth)),
            }
        )
    bonn = boundary_metrics(bonn_values, selected_threshold, 4)
    parent_metrics = {
        parent: boundary_metrics(values, selected_threshold, 4)
        for parent, values in sorted(bonn_by_parent.items())
    }
    gates = {
        "r7_canary_labels_opened_eq_0": True,
        "bonn_parent_count_eq_8": len(parent_metrics) == 8,
        "bonn_frame_count_eq_24": len(bonn_pairs) == 24,
        "each_parent_positive_pixels_ge_1000": all(value["positive_pixels"] >= 1000 for value in parent_metrics.values()),
        "bonn_ap_gain_over_constant_ge_0p02": bonn["student_average_precision"] >= bonn["constant_average_precision"] + 0.02,
        "bonn_precision_and_recall_ge_0p10": bonn["precision_within_tolerance"] >= 0.10 and bonn["recall_within_tolerance"] >= 0.10,
        "bonn_f1_within_4px_ge_0p15": bonn["f1_within_tolerance"] >= 0.15,
    }
    passed = all(gates.values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "soft-boundary-decoder.pt"
    torch.save(
        {
            "schema": "blindassist_ag_st_soft_boundary_decoder_v1",
            "decoder_state_dict": best_state,
            "selected_threshold": selected_threshold,
            "soft_sigma_px": args.soft_sigma_px,
            "mobilenet_checkpoint_sha256": sha256_file(args.mobilenet_checkpoint),
            "r7_split": split,
        },
        checkpoint_path,
    )
    result = {
        "schema": "blindassist_ag_st_soft_boundary_bonn_canary_result_v1",
        "status": "SOFT_BOUNDARY_BONN_CANARY_PASS" if passed else "SOFT_BOUNDARY_BONN_CANARY_FAIL",
        "question": "Does continuous source-boundary distance supervision produce a transferable RGB boundary signal on a held-out sensor domain?",
        "complete_truth_required": False,
        "inputs": {
            "binding": str(args.binding.resolve()),
            "binding_sha256": sha256_file(args.binding),
            "mobilenet_checkpoint": str(args.mobilenet_checkpoint.resolve()),
            "mobilenet_checkpoint_sha256": sha256_file(args.mobilenet_checkpoint),
            "bonn_cohort": str(args.bonn_cohort.resolve()),
            "bonn_cohort_sha256": sha256_file(args.bonn_cohort),
        },
        "label_contract": {
            "training_core": "R6 source-native/source-exact boundary probability; Teacher-filled pixels UNKNOWN",
            "training_target": "max(source probability, exp(-distance_to_core^2/(2*3px^2))) on valid pixels only",
            "bonn_truth": "source-native depth 3x3 valid neighbourhood and camera-space point-to-plane boundary; no Teacher label",
        },
        "model": {
            "backbone": "frozen ImageNet MobileNetV3-Small",
            "decoder_trainable_parameters": trainable_parameters,
            "inputs": "RGB only",
            "outputs": "boundary heat probability only",
            "reducer_or_task_fields": False,
        },
        "training": {
            "roles_opened": ["fit", "selection"],
            "r7_canary_label_files_opened": 0,
            "soft_sigma_px": args.soft_sigma_px,
            "epochs": args.epochs,
            "steps": global_step,
            "selected_epoch": selected_epoch,
            "selected_threshold": selected_threshold,
            "selection_history": selection_history,
            "selected_threshold_metrics": selected_threshold_row["metrics"],
        },
        "bonn_canary": {
            "freshness_scope": "boundary-branch fresh; not project-global virgin",
            "source_depth_opened_after_checkpoint_and_threshold_selection": True,
            "overall": bonn,
            "by_parent": parent_metrics,
            "frames": bonn_receipts,
        },
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "decision": {
            "transferable_source_boundary_signal_supported": passed,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
            "if_failed": "Keep source-native labels and change representation/model evidence; never convert UNKNOWN into negatives.",
        },
        "claim_boundary": "Boundary-only WILD_LAB source-native/exact supervision and Bonn cross-sensor diagnostic; no complete truth, reducer/task utility, safety, deployment, product, or formal F1 claim.",
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--mobilenet-checkpoint", type=Path, default=DEFAULT_MOBILENET)
    parser.add_argument("--bonn-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--bonn-cohort", type=Path, default=DEFAULT_BONN_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--selection-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--soft-sigma-px", type=float, default=SOFT_SIGMA_PX)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "training": result["training"], "bonn": result["bonn_canary"]["overall"], "gates": result["gates"], "execution": result["execution"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
