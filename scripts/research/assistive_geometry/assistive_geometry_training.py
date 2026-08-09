#!/usr/bin/env python3
"""Deterministic dual-orientation TRAIN loader and A0 execution utilities."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.research.assistive_geometry.arkitscenes_truth_reader import rotate_array_upright
from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import require
from scripts.research.assistive_geometry.materialize_b1_train_targets import resize_cached_dense


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
ORIENTATIONS = ("portrait", "landscape")


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def flatten_train_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require(manifest.get("schema") == "blindassist_assistive_geometry_b1_train_target_manifest_v1", "target manifest schema drift")
    frames: list[dict[str, Any]] = []
    visits: set[str] = set()
    for video in manifest["videos"]:
        require(video["role"] == "TRAIN", f"non-TRAIN video in A0 loader: {video['video_id']}")
        visit = str(video["visit_id"])
        require(visit not in visits, f"duplicate TRAIN visit: {visit}")
        visits.add(visit)
        for frame in video["frames"]:
            frames.append({**frame, "video_id": str(video["video_id"]), "visit_id": visit})
    require(len(visits) == 16 and len(frames) == 4800, "A0 TRAIN roster/count drift")
    require(sum(frame["orientation_family"] == "portrait" for frame in frames) == 2724, "portrait count drift")
    require(sum(frame["orientation_family"] == "landscape" for frame in frames) == 2076, "landscape count drift")
    return frames


def parent_balanced_epoch_order(frames: list[dict[str, Any]], seed: int, epoch: int) -> list[int]:
    by_parent: dict[str, list[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        by_parent[str(frame["video_id"])].append(index)
    require(len(by_parent) == 16, "parent-balanced sampler requires 16 TRAIN parents")
    queues: dict[str, list[int]] = {}
    for parent, indices in by_parent.items():
        rng = np.random.default_rng(stable_seed(seed, epoch, parent, "within-parent"))
        queues[parent] = [indices[position] for position in rng.permutation(len(indices))]
    output: list[int] = []
    maximum = max(len(indices) for indices in queues.values())
    parents = sorted(queues)
    for offset in range(maximum):
        rng = np.random.default_rng(stable_seed(seed, epoch, offset, "parent-order"))
        for parent_position in rng.permutation(len(parents)):
            parent = parents[int(parent_position)]
            if offset < len(queues[parent]):
                output.append(queues[parent][offset])
    require(len(output) == len(frames) and len(set(output)) == len(frames), "epoch order coverage drift")
    return output


def build_epoch_effective_batches(
    frames: list[dict[str, Any]],
    seed: int,
    epoch: int,
    carry: dict[str, list[int]] | None = None,
    effective_batch_size: int = 16,
) -> tuple[list[tuple[str, list[int]]], dict[str, list[int]]]:
    require(effective_batch_size > 0, "effective batch size must be positive")
    incoming = {orientation: list((carry or {}).get(orientation, [])) for orientation in ORIENTATIONS}
    ordered = parent_balanced_epoch_order(frames, seed, epoch)
    streams = {orientation: incoming[orientation] for orientation in ORIENTATIONS}
    for index in ordered:
        orientation = str(frames[index]["orientation_family"])
        require(orientation in ORIENTATIONS, f"unknown orientation family: {orientation}")
        streams[orientation].append(index)
    groups: dict[str, list[list[int]]] = {}
    remainder: dict[str, list[int]] = {}
    for orientation in ORIENTATIONS:
        complete = len(streams[orientation]) // effective_batch_size
        groups[orientation] = [
            streams[orientation][start * effective_batch_size : (start + 1) * effective_batch_size]
            for start in range(complete)
        ]
        remainder[orientation] = streams[orientation][complete * effective_batch_size :]

    merged: list[tuple[str, list[int]]] = []
    consumed = {orientation: 0 for orientation in ORIENTATIONS}
    while any(consumed[orientation] < len(groups[orientation]) for orientation in ORIENTATIONS):
        available = [orientation for orientation in ORIENTATIONS if consumed[orientation] < len(groups[orientation])]
        orientation = min(
            available,
            key=lambda name: (consumed[name] / max(len(groups[name]), 1), ORIENTATIONS.index(name)),
        )
        merged.append((orientation, groups[orientation][consumed[orientation]]))
        consumed[orientation] += 1
    return merged, remainder


def color_jitter_rgb(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = image.astype(np.float32, copy=True)
    result *= float(rng.uniform(0.9, 1.1))
    channel_mean = result.mean(axis=(0, 1), keepdims=True)
    result = (result - channel_mean) * float(rng.uniform(0.9, 1.1)) + channel_mean
    gray = cv2.cvtColor(np.clip(result, 0.0, 1.0), cv2.COLOR_RGB2GRAY)[..., None]
    result = gray + (result - gray) * float(rng.uniform(0.9, 1.1))
    hsv = cv2.cvtColor(np.clip(result, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + float(rng.uniform(-0.02, 0.02)) * 360.0, 360.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0).astype(np.float32)


class AssistiveGeometryTrainDataset(Dataset[dict[str, Any]]):
    def __init__(self, frames: list[dict[str, Any]], seed: int, *, augment: bool = True) -> None:
        self.frames = frames
        self.seed = int(seed)
        self.epoch = 0
        self.augment = bool(augment)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.frames[index]
        target_path = Path(frame["target"]["path"])
        with np.load(target_path, allow_pickle=False) as payload:
            values = {key: payload[key] for key in payload.files}
        orientation = int(values["orientation_index"].item())
        target_height, target_width = (int(value) for value in values["target_hw"])
        image_bgr = cv2.imread(str(frame["rgb_source"]["path"]), cv2.IMREAD_COLOR)
        require(image_bgr is not None, f"RGB decode failed: {frame['rgb_source']['path']}")
        image_rgb = cv2.cvtColor(rotate_array_upright(image_bgr, orientation), cv2.COLOR_BGR2RGB)
        require(image_rgb.shape[:2] == values["depth_m_source"].shape, f"upright RGB/target shape drift: {frame['frame_stem']}")
        image = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0

        targets: dict[str, torch.Tensor] = {
            "dense_depth_m": torch.from_numpy(resize_cached_dense(values["depth_m_source"], (target_height, target_width), mask=False))[None],
            "depth_valid": torch.from_numpy(resize_cached_dense(values["depth_valid_source"], (target_height, target_width), mask=True))[None],
            "ground_probability": torch.from_numpy(resize_cached_dense(values["ground_probability_source"], (target_height, target_width), mask=False))[None],
            "ground_label_valid": torch.from_numpy(resize_cached_dense(values["ground_label_valid_source"], (target_height, target_width), mask=True))[None],
            "ground_plane_valid": torch.as_tensor(bool(values["ground_plane_valid"].item()), dtype=torch.bool),
            "camera_height_m": torch.as_tensor(float(values["camera_height_m"].item()), dtype=torch.float32),
            "up_camera": torch.from_numpy(values["up_camera"].astype(np.float32, copy=True)),
            "intrinsics_tensor": torch.from_numpy(values["intrinsics_tensor"].astype(np.float32, copy=True)),
            "clearance_m": torch.from_numpy(values["clearance_m"].astype(np.float32, copy=True)),
            "clearance_valid": torch.from_numpy(values["clearance_valid"].astype(np.bool_, copy=True)),
            "occupancy": torch.from_numpy(values["occupancy"].astype(np.float32, copy=True)),
            "occupancy_valid": torch.from_numpy(values["occupancy_valid"].astype(np.bool_, copy=True)),
        }
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, frame["video_id"], frame["frame_stem"], "augmentation"))
        flipped = self.augment and bool(rng.random() < 0.5)
        if flipped:
            image = np.ascontiguousarray(image[:, ::-1])
            for key in ("dense_depth_m", "depth_valid", "ground_probability", "ground_label_valid"):
                targets[key] = torch.flip(targets[key], dims=(-1,))
            targets["intrinsics_tensor"][0, 2] = target_width - 1 - targets["intrinsics_tensor"][0, 2]
            targets["up_camera"][0] = -targets["up_camera"][0]
            for key in ("clearance_m", "clearance_valid", "occupancy", "occupancy_valid"):
                targets[key] = torch.flip(targets[key], dims=(0,))
        if self.augment:
            image = color_jitter_rgb(image, rng)
        normalized = ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
        return {
            "image": torch.from_numpy(normalized),
            "targets": targets,
            "metadata": {
                "visit_id": frame["visit_id"],
                "video_id": frame["video_id"],
                "frame_stem": frame["frame_stem"],
                "orientation_family": frame["orientation_family"],
                "horizontal_flip": flipped,
            },
        }


def collate_train_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(samples), "cannot collate an empty batch")
    orientations = {sample["metadata"]["orientation_family"] for sample in samples}
    require(len(orientations) == 1, "mixed-orientation microbatch is forbidden")
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "targets": {
            key: torch.stack([sample["targets"][key] for sample in samples])
            for key in samples[0]["targets"]
        },
        "metadata": [sample["metadata"] for sample in samples],
    }


def a0_lr_multiplier(optimizer_step: int, *, warmup_steps: int = 300, total_steps: int = 6000, final_ratio: float = 0.05) -> float:
    require(1 <= optimizer_step <= total_steps, "optimizer step outside frozen A0 schedule")
    if optimizer_step <= warmup_steps:
        return optimizer_step / warmup_steps
    progress = (optimizer_step - warmup_steps) / (total_steps - warmup_steps)
    return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class FrozenA0Scheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int = 6000) -> None:
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.completed_steps = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def prepare_next_step(self) -> float:
        step = self.completed_steps + 1
        multiplier = a0_lr_multiplier(step, total_steps=self.total_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * multiplier
        return multiplier

    def mark_completed(self) -> None:
        require(self.completed_steps < self.total_steps, "A0 scheduler exceeded total steps")
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {"total_steps": self.total_steps, "completed_steps": self.completed_steps, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        require(int(state["total_steps"]) == self.total_steps, "scheduler total-step drift")
        restored_base_lrs = [float(value) for value in state["base_lrs"]]
        require(len(restored_base_lrs) == len(self.optimizer.param_groups), "scheduler parameter-group drift")
        self.base_lrs = restored_base_lrs
        self.completed_steps = int(state["completed_steps"])
