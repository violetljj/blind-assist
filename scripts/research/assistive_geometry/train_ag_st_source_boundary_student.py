#!/usr/bin/env python3
"""Train a source-balanced boundary-only student on the AG-ST R6 corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter


REPO_ROOT = Path(__file__).resolve().parents[3]
MAPANYTHING_ROOT = REPO_ROOT / "artifacts.local/tools/map-anything"
sys.path.insert(0, str(MAPANYTHING_ROOT))

from download_b0_arkitscenes_assets import require, sha256_file  # noqa: E402


DEFAULT_BINDING = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/rgb_binding.json"
)
DEFAULT_MOBILENET = Path(
    "C:/Users/26442/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-source-boundary-student-r1"
SPLIT_SALT = "AG_ST_SOURCE_BOUNDARY_STUDENT_R0"
SPLIT_COUNTS = {
    "arkitscenes": {"fit": 12, "selection": 2, "canary": 2},
    "tum_rgbd": {"fit": 5, "selection": 1, "canary": 1},
    "icl_exact": {"fit": 1, "selection": 0, "canary": 0},
}
THRESHOLD_GRID = tuple(value / 10.0 for value in range(1, 10))


def deterministic_split(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    parents_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        parents_by_source[str(row["source"])].add(str(row["parent_id"]))
    output = {"fit": [], "selection": [], "canary": []}
    for source, counts in SPLIT_COUNTS.items():
        parents = sorted(
            parents_by_source[source],
            key=lambda parent: hashlib.sha256(
                f"{SPLIT_SALT}:{source}:{parent}".encode("utf-8")
            ).hexdigest(),
        )
        require(len(parents) == sum(counts.values()), f"split parent count drift: {source}")
        offset = 0
        for role in ("fit", "selection", "canary"):
            selected = parents[offset : offset + counts[role]]
            output[role].extend((source, parent) for parent in selected)
            offset += counts[role]
    require(not (set(output["fit"]) & set(output["selection"])), "fit/selection overlap")
    require(not (set(output["fit"]) & set(output["canary"])), "fit/canary overlap")
    require(not (set(output["selection"]) & set(output["canary"])), "selection/canary overlap")
    return output


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.bool_).reshape(-1)
    prediction = np.asarray(score, dtype=np.float64).reshape(-1)
    require(truth.shape == prediction.shape and truth.size > 0, "AP input invalid")
    positives = int(np.sum(truth))
    require(positives > 0, "AP positive denominator empty")
    order = np.argsort(-prediction, kind="stable")
    ranked = truth[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _binary_metrics(
    frame_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    precision_hit = recall_hit = predicted_count = target_count = 0
    for probability, target, valid in frame_values:
        targets.append(target[valid])
        scores.append(probability[valid])
        predicted = (probability >= threshold) & valid
        truth = target & valid
        predicted_count += int(np.sum(predicted))
        target_count += int(np.sum(truth))
        truth_near = maximum_filter(truth.astype(np.uint8), size=5) > 0
        predicted_near = maximum_filter(predicted.astype(np.uint8), size=5) > 0
        precision_hit += int(np.sum(predicted & truth_near))
        recall_hit += int(np.sum(truth & predicted_near))
    target_flat = np.concatenate(targets)
    score_flat = np.concatenate(scores)
    clipped = np.clip(score_flat, 1e-6, 1.0 - 1e-6)
    bce = float(
        np.mean(
            -target_flat.astype(np.float64) * np.log(clipped)
            - (~target_flat).astype(np.float64) * np.log(1.0 - clipped)
        )
    )
    precision = precision_hit / max(1, predicted_count)
    recall = recall_hit / max(1, target_count)
    return {
        "valid_pixels": int(target_flat.size),
        "positive_pixels": int(np.sum(target_flat)),
        "prevalence": float(np.mean(target_flat)),
        "average_precision": average_precision(target_flat, score_flat),
        "bce": bce,
        "threshold": threshold,
        "predicted_pixels": predicted_count,
        "precision_within_2px": precision,
        "recall_within_2px": recall,
        "f1_within_2px": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def _source_metrics(
    predictions: dict[int, np.ndarray],
    descriptors: list[dict[str, Any]],
    targets: dict[int, dict[str, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for index, probability in predictions.items():
        row = descriptors[index]
        target = targets[index]
        grouped[str(row["source"])].append(
            (probability, target["positive"], target["valid"])
        )
    by_source = {
        source: _binary_metrics(values, threshold)
        for source, values in sorted(grouped.items())
    }
    return {
        "by_source": by_source,
        "source_macro_average_precision": float(
            np.mean([value["average_precision"] for value in by_source.values()])
        ),
        "source_macro_bce": float(np.mean([value["bce"] for value in by_source.values()])),
        "source_macro_f1_within_2px": float(
            np.mean([value["f1_within_2px"] for value in by_source.values()])
        ),
    }


class TarImageReader:
    def __init__(self) -> None:
        self.handles: dict[Path, tarfile.TarFile] = {}
        self.members: dict[Path, dict[str, tarfile.TarInfo]] = {}

    def read(self, archive: Path, member: str) -> bytes:
        archive = archive.resolve()
        if archive not in self.handles:
            handle = tarfile.open(archive, mode="r:gz")
            self.handles[archive] = handle
            self.members[archive] = {item.name: item for item in handle.getmembers() if item.isfile()}
        members = self.members[archive]
        selected = members.get(member)
        if selected is None:
            matches = [item for name, item in members.items() if name.endswith("/" + member)]
            require(len(matches) == 1, f"tar RGB member ambiguous: {archive}:{member}")
            selected = matches[0]
        stream = self.handles[archive].extractfile(selected)
        require(stream is not None, f"tar RGB member unreadable: {archive}:{member}")
        return stream.read()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def _load_bound_rgb(row: dict[str, Any], tar_reader: TarImageReader) -> np.ndarray:
    if row["rgb_storage_kind"] == "file":
        source: Any = Path(row["rgb_path"])
    else:
        source = io.BytesIO(
            tar_reader.read(Path(row["rgb_source_archive"]), str(row["rgb_member"]))
        )
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    source_name = str(row["source"])
    target_h, target_w = (int(value) for value in row["label_shape_hw"])
    if source_name == "arkitscenes":
        label_to_k = {
            "IDENTITY": 0,
            "CLOCKWISE_90": -1,
            "ROTATE_180": 2,
            "COUNTERCLOCKWISE_90": 1,
        }
        rgb = np.ascontiguousarray(np.rot90(rgb, k=label_to_k[str(row["orientation"])]))
    elif source_name == "icl_exact":
        rgb = np.flipud(rgb)[2::4, 2::4]
        require(rgb.shape[:2] == (target_h, target_w), "ICL RGB sampling drift")
        return np.ascontiguousarray(rgb)
    from mapanything.utils.cropping import crop_resize_if_necessary

    processed = crop_resize_if_necessary(
        image=rgb,
        resolution=(target_w, target_h),
    )[0]
    output = np.asarray(processed.convert("RGB"), dtype=np.uint8).copy()
    require(output.shape[:2] == (target_h, target_w), "bound RGB shape drift")
    return output


def _load_target(row: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(row["label_path"]) as values:
        probability = np.asarray(values["boundary_probability_hw"], dtype=np.float32)
        valid = np.asarray(values["boundary_truth_valid_hw"], dtype=np.bool_)
    require(probability.shape == valid.shape == tuple(row["label_shape_hw"]), "target shape drift")
    return {
        "probability": probability,
        "positive": valid & (probability >= 0.5),
        "valid": valid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import mobilenet_v3_small

    require(torch.cuda.is_available(), "boundary student requires CUDA")
    require(args.binding.is_file() and args.mobilenet_checkpoint.is_file(), "student input missing")
    require(not args.output_dir.exists(), f"boundary student output exists: {args.output_dir}")
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    require(binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS", "RGB binding invalid")
    descriptors = list(binding["frames"])
    require(len(descriptors) == 81, "boundary student frame count drift")
    split = deterministic_split(descriptors)
    role_by_parent = {
        identity: role
        for role, identities in split.items()
        for identity in identities
    }
    indices_by_role: dict[str, list[int]] = {role: [] for role in split}
    for index, row in enumerate(descriptors):
        identity = (str(row["source"]), str(row["parent_id"]))
        indices_by_role[role_by_parent[identity]].append(index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    backbone_model = mobilenet_v3_small(weights=None)
    state = torch.load(args.mobilenet_checkpoint, map_location="cpu", weights_only=True)
    backbone_model.load_state_dict(state, strict=True)
    backbone = backbone_model.features.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

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
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    try:
        with torch.no_grad():
            for index, row in enumerate(descriptors):
                rgb = _load_bound_rgb(row, tar_reader)
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device=device, dtype=torch.float32) / 255.0
                value = ((tensor - mean) / std)[None]
                captured: list[torch.Tensor] = []
                for layer_index, layer in enumerate(backbone):
                    value = layer(value)
                    if layer_index in (1, 3, 8, 12):
                        captured.append(value.detach().cpu().half())
                require(len(captured) == 4, "MobileNet feature capture drift")
                feature_cache[index] = tuple(captured)
    finally:
        tar_reader.close()
    del backbone, backbone_model
    torch.cuda.empty_cache()

    fit_targets = {index: _load_target(descriptors[index]) for index in indices_by_role["fit"]}
    selection_targets = {
        index: _load_target(descriptors[index]) for index in indices_by_role["selection"]
    }
    decoder = Decoder().to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    trainable_parameters = sum(parameter.numel() for parameter in decoder.parameters())

    def forward_probability(index: int) -> np.ndarray:
        features = tuple(value.to(device=device, dtype=torch.float32) for value in feature_cache[index])
        target = fit_targets.get(index) or selection_targets.get(index)
        if target is None:
            target = _load_target(descriptors[index])
        with torch.no_grad():
            logits = decoder(features, target["probability"].shape)
        return torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)

    def evaluate(indices: list[int], targets: dict[int, dict[str, np.ndarray]], threshold: float) -> dict[str, Any]:
        decoder.eval()
        predictions = {index: forward_probability(index) for index in indices}
        return _source_metrics(predictions, descriptors, targets, threshold)

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
                target = torch.from_numpy(target_np["probability"]).to(device=device, dtype=torch.float32)[None]
                valid = torch.from_numpy(target_np["valid"]).to(device=device)[None]
                logits = decoder(features, target.shape[-2:])
                positive = valid & (target >= 0.5)
                positive_count = int(positive.sum().item())
                negative_count = int((valid & ~positive).sum().item())
                pos_weight = min(40.0, negative_count / max(1, positive_count))
                weights = torch.where(positive, pos_weight, 1.0)
                bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                loss_bce = (bce[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)
                probability = torch.sigmoid(logits)
                probability_valid = probability[valid]
                target_valid = target[valid]
                dice = 1.0 - (2.0 * torch.sum(probability_valid * target_valid) + 1.0) / (
                    torch.sum(probability_valid) + torch.sum(target_valid) + 1.0
                )
                loss = loss_bce + 0.5 * dice
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
        if epoch % args.selection_interval == 0 or epoch == args.epochs:
            metrics = evaluate(indices_by_role["selection"], selection_targets, 0.5)
            key = (
                float(metrics["source_macro_average_precision"]),
                -float(metrics["source_macro_bce"]),
            )
            selection_history.append({"epoch": epoch, "step": global_step, "metrics": metrics})
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()}
    require(best_state is not None, "boundary selection produced no checkpoint")
    decoder.load_state_dict(best_state, strict=True)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        metrics = evaluate(indices_by_role["selection"], selection_targets, threshold)
        threshold_rows.append({"threshold": threshold, "metrics": metrics})
    selected_threshold_row = max(
        threshold_rows,
        key=lambda row: (
            row["metrics"]["source_macro_f1_within_2px"],
            row["metrics"]["source_macro_average_precision"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    selected_threshold = float(selected_threshold_row["threshold"])

    # Canary labels are opened only after checkpoint and threshold selection.
    canary_targets = {index: _load_target(descriptors[index]) for index in indices_by_role["canary"]}
    canary = evaluate(indices_by_role["canary"], canary_targets, selected_threshold)
    fit_prevalence: dict[str, float] = {}
    for source, indices in fit_by_source.items():
        positive = sum(int(np.sum(fit_targets[index]["positive"])) for index in indices)
        valid = sum(int(np.sum(fit_targets[index]["valid"])) for index in indices)
        fit_prevalence[source] = positive / valid
    baseline_by_source: dict[str, dict[str, float]] = {}
    for source, metrics in canary["by_source"].items():
        prior = float(np.clip(fit_prevalence[source], 1e-6, 1 - 1e-6))
        prevalence = float(metrics["prevalence"])
        baseline_by_source[source] = {
            "fit_constant_prior": prior,
            "canary_prevalence": prevalence,
            "constant_average_precision": prevalence,
            "constant_bce": float(-prevalence * math.log(prior) - (1.0 - prevalence) * math.log(1.0 - prior)),
        }
    gates = {
        "canary_source_count_eq_2": len(canary["by_source"]) == 2,
        "canary_parent_count_eq_3": len(indices_by_role["canary"]) == 9
        and len({descriptors[index]["parent_id"] for index in indices_by_role["canary"]}) == 3,
        "each_source_positive_pixels_ge_20": all(
            int(metrics["positive_pixels"]) >= 20 for metrics in canary["by_source"].values()
        ),
        "each_source_ap_gain_ge_0p02": all(
            float(canary["by_source"][source]["average_precision"])
            >= float(baseline_by_source[source]["constant_average_precision"]) + 0.02
            for source in canary["by_source"]
        ),
        "source_macro_f1_within_2px_ge_0p10": float(canary["source_macro_f1_within_2px"]) >= 0.10,
    }
    passed = all(gates.values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "boundary-decoder.pt"
    torch.save(
        {
            "schema": "blindassist_ag_st_source_boundary_decoder_v1",
            "decoder_state_dict": best_state,
            "selected_threshold": selected_threshold,
            "split": split,
            "mobilenet_checkpoint_sha256": sha256_file(args.mobilenet_checkpoint),
        },
        checkpoint_path,
    )
    result = {
        "schema": "blindassist_ag_st_source_boundary_student_result_v1",
        "status": "SOURCE_BOUNDARY_LEARNABILITY_PASS" if passed else "SOURCE_BOUNDARY_LEARNABILITY_FAIL",
        "question": "Can a frozen pretrained encoder plus a small source-balanced decoder learn source-native/exact boundary labels across held-out ARKitScenes and TUM parents?",
        "inputs": {
            "binding": str(args.binding.resolve()),
            "binding_sha256": sha256_file(args.binding),
            "mobilenet_checkpoint": str(args.mobilenet_checkpoint.resolve()),
            "mobilenet_checkpoint_sha256": sha256_file(args.mobilenet_checkpoint),
            "frame_count": len(descriptors),
            "parent_count": 24,
        },
        "model": {
            "backbone": "torchvision MobileNetV3-Small features frozen",
            "decoder_trainable_parameters": trainable_parameters,
            "inputs": "RGB only",
            "outputs": "boundary logits only",
            "reducer_or_task_fields": False,
        },
        "split": {
            role: [{"source": source, "parent_id": parent} for source, parent in values]
            for role, values in split.items()
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "steps": global_step,
            "learning_rate": args.learning_rate,
            "source_balanced_samples_per_epoch": samples_per_source * len(fit_by_source),
            "selection_interval_epochs": args.selection_interval,
            "selection_history": selection_history,
            "selected_epoch": max(selection_history, key=lambda row: (row["metrics"]["source_macro_average_precision"], -row["metrics"]["source_macro_bce"]))["epoch"],
            "threshold_grid": list(THRESHOLD_GRID),
            "selected_threshold": selected_threshold,
            "selection_threshold_metrics": selected_threshold_row["metrics"],
            "canary_labels_opened_after_selection": True,
        },
        "canary": canary,
        "constant_prior_baseline": baseline_by_source,
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
            "source_boundary_learnability_supported": passed,
            "teacher_filled_boundary_training_authorized": False,
            "formal_f1_authority_changed": False,
            "if_failed": "Retain the corpus and diagnose source imbalance/representation; do not relabel UNKNOWN or tune on canary.",
        },
        "claim_boundary": "Boundary-only WILD_LAB learnability on held-out TRAIN-source parents; no reducer/task, formal F1, fresh real-world, safety, deployment, or product claim.",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--selection-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_epoch": result["training"]["selected_epoch"],
                "selected_threshold": result["training"]["selected_threshold"],
                "canary": result["canary"],
                "gates": result["gates"],
                "execution": result["execution"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
