#!/usr/bin/env python3
"""Train a frozen-encoder masked factor student on AG-ST graded pseudo-labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(REPO_ROOT / "artifacts.local/tools/map-anything"))

from build_ag_st_factor_labels import (  # noqa: E402
    PROVENANCE_SOURCE_NATIVE,
    PROVENANCE_TEACHER,
    TIER_A_SOURCE,
    TIER_B_ANCHORED,
    TIER_C_TEACHER,
    TIER_UNKNOWN,
)
from download_b0_arkitscenes_assets import (  # noqa: E402
    load_json,
    require,
    sha256_file,
)
from arkitscenes_truth_reader import (  # noqa: E402
    parse_trajectory,
)
from run_ag_st_stage0a import (  # noqa: E402
    load_factor_source_frame,
    resolve_trajectory_path,
    select_source_videos,
)


DEFAULT_STAGE0A_RESULT = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-stage0a-mapanything-apache-train16-block64-r1"
    / "result.json"
)
DEFAULT_LABEL_DIR = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-superteacher-factor-labels-train16-r5"
)
DEFAULT_LABEL_RESULT = DEFAULT_LABEL_DIR / "result.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-masked-student-mobilenetv3-r0"
)
DEFAULT_MOBILENET_CHECKPOINT = Path(
    "C:/Users/26442/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
)
DEFAULT_DEPTHART_SOURCE = Path(
    "F:/ba-data/blindassist-artifacts-20260805/models/depthart/source"
)
DEFAULT_DEPTHART_CHECKPOINT = (
    DEFAULT_DEPTHART_SOURCE
    / "checkpoints"
    / "metric"
    / "depthart_metric_indoor_s_448.pth"
)
DEFAULT_SOURCE_RGB_BINDING = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/rgb_binding.json"
)

SPLIT_TOKEN = "AG_ST_MASKED_STUDENT_R0"
SOURCE_BOUNDARY_SPLIT_TOKEN = "AG_ST_SOURCE_BOUNDARY_STUDENT_R0"
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
TIER_WEIGHTS = torch.tensor((0.0, 0.25, 0.60, 1.0), dtype=torch.float32)
BOUNDARY_BAND_PX = 2.0
MAX_BOUNDARY_DISTANCE_PX = 32.0
BOUNDARY_HEAT_SIGMA_PX = 3.0
DEPTH_RESIDUAL_LOG_RANGE = 2.5
DEPTH_IDENTITY_GATE_PRIOR = 0.05
DEPTHART_SHARED_CHANNELS = 48
DEPTHART_PYRAMID_CHANNELS = 4 * DEPTHART_SHARED_CHANNELS
LOSS_WEIGHTS = {
    "depth_log_huber": 1.0,
    "support_bce": 0.50,
    "boundary_bce": 0.50,
    "boundary_distance": 0.20,
    "obstacle_bce": 0.25,
}
DEPTH_SUPPORT_LOSS_WEIGHTS = {
    "depth_log_huber": LOSS_WEIGHTS["depth_log_huber"],
    "support_bce": LOSS_WEIGHTS["support_bce"],
}
DEPTH_SUPPORT_PRECISION_LOSS_WEIGHTS = {
    "depth_log_huber": 0.50,
    "depth_abs_huber": 1.00,
    "depth_bad_0_10_soft_margin": 0.50,
    "support_bce": 0.75,
    "support_soft_dice": 0.25,
    "depth_identity_gate_bce": 0.25,
}
BOUNDARY_ONLY_LOSS_WEIGHTS = {
    "boundary_soft_heat_bce": 1.0,
    "boundary_near_distance": 0.50,
    "boundary_heat_distance_consistency": 0.10,
}
UNIFIED_CONTINUOUS_BOUNDARY_LOSS_WEIGHTS = {
    "depth_log_huber": LOSS_WEIGHTS["depth_log_huber"],
    "support_bce": LOSS_WEIGHTS["support_bce"],
    "obstacle_bce": LOSS_WEIGHTS["obstacle_bce"],
    **BOUNDARY_ONLY_LOSS_WEIGHTS,
}


@dataclass(frozen=True)
class FrameDescriptor:
    parent_id: str
    frame_index: int
    frame_stem: str
    output_hw: tuple[int, int]
    label_path: Path
    video: dict[str, Any]
    source_id: str = "arkitscenes"
    rgb_binding: dict[str, Any] | None = None


@dataclass
class CachedFrame:
    descriptor: FrameDescriptor
    feature: torch.Tensor
    base_depth_m: torch.Tensor
    targets: dict[str, torch.Tensor]


class DepthArtDenseFeatureExtractor(nn.Module):
    """Factor-only DepthART wrapper; it creates no clearance/occupancy heads."""

    def __init__(self, metric_depthart: nn.Module) -> None:
        super().__init__()
        self.metric_depthart = metric_depthart

    def decode(
        self,
        features: list[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head = self.metric_depthart.depth_head
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1 = head.scratch.layer1_rn(layer_1)
        layer_2 = head.scratch.layer2_rn(layer_2)
        layer_3 = head.scratch.layer3_rn(layer_3)
        layer_4 = head.scratch.layer4_rn(layer_4)
        path_4 = head.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        shared = head.scratch.refinenet1(path_2, layer_1)
        depth = head.scratch.output_conv1(shared)
        depth = F.interpolate(depth, output_hw, mode="bilinear", align_corners=True)
        depth = head.scratch.output_conv2(depth)
        pyramid = torch.cat(
            [
                shared,
                *[
                    F.interpolate(
                        value,
                        shared.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for value in (path_2, path_3, path_4)
                ],
            ],
            dim=1,
        )
        require(
            pyramid.shape[1] == DEPTHART_PYRAMID_CHANNELS,
            "DepthART pyramid channel drift",
        )
        return depth, shared, pyramid


class MobileNetV3DenseFeatureExtractor(nn.Module):
    """Frozen multi-scale ImageNet features aligned to the stride-4 grid."""

    TAP_INDICES = (1, 3, 8, 12)
    OUTPUT_CHANNELS = 16 + 24 + 48 + 576

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tapped: list[torch.Tensor] = []
        value = image
        for index, block in enumerate(self.features):
            value = block(value)
            if index in self.TAP_INDICES:
                tapped.append(value)
        require(len(tapped) == len(self.TAP_INDICES), "MobileNet feature tap drift")
        output_hw = tapped[0].shape[-2:]
        aligned = [
            feature
            if feature.shape[-2:] == output_hw
            else F.interpolate(feature, output_hw, mode="bilinear", align_corners=False)
            for feature in tapped
        ]
        output = torch.cat(aligned, dim=1)
        require(
            output.shape[1] == self.OUTPUT_CHANNELS,
            "MobileNet feature channel drift",
        )
        return output


def stable_hash(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def save_checkpoint_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"checkpoint collision: {path}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def shape_family(output_hw: tuple[int, int]) -> str:
    height, width = (int(value) for value in output_hw)
    if height > width:
        return "portrait"
    return f"landscape_h{height}"


def select_parent_split(
    parent_shapes: dict[str, tuple[int, int]],
    *,
    split_token: str = SPLIT_TOKEN,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Keep one selection and one canary parent per orientation stratum."""
    require(len(parent_shapes) >= 6, "masked-student split requires at least 6 parents")
    groups: dict[str, list[str]] = defaultdict(list)
    for parent, output_hw in parent_shapes.items():
        family = "portrait" if shape_family(output_hw) == "portrait" else "landscape"
        groups[family].append(str(parent))
    require(set(groups) == {"portrait", "landscape"}, "missing orientation stratum")
    require(
        all(len(values) >= 3 for values in groups.values()),
        "orientation stratum too small for train/selection/canary",
    )
    train_parents: list[str] = []
    selection_parents: list[str] = []
    canary_parents: list[str] = []
    rankings: dict[str, list[dict[str, str]]] = {}
    for family in sorted(groups):
        ranked = sorted(
            groups[family],
            key=lambda parent: stable_hash(f"{split_token}:{parent}"),
        )
        rankings[family] = [
            {
                "parent_id": parent,
                "rank_sha256": stable_hash(f"{split_token}:{parent}"),
            }
            for parent in ranked
        ]
        train_parents.extend(ranked[:-2])
        selection_parents.append(ranked[-2])
        canary_parents.append(ranked[-1])
    train_tuple = tuple(sorted(train_parents))
    selection_tuple = tuple(sorted(selection_parents))
    canary_tuple = tuple(sorted(canary_parents))
    expected = (len(parent_shapes) - 4, 2, 2)
    require(
        (len(train_tuple), len(selection_tuple), len(canary_tuple)) == expected,
        "parent split cardinality drift",
    )
    require(
        len(set(train_tuple) | set(selection_tuple) | set(canary_tuple))
        == len(parent_shapes),
        "parent split overlap or omission",
    )
    return train_tuple, selection_tuple, canary_tuple, {
        "method": (
            f"ORIENTATION_STRATIFIED_SHA256_{len(train_tuple)}_2_2_"
            "WITHOUT_LABEL_CONTENT"
        ),
        "split_token": split_token,
        "train_parents": list(train_tuple),
        "selection_parents": list(selection_tuple),
        "canary_parents": list(canary_tuple),
        "rankings": rankings,
    }


def select_multisource_parent_split(
    descriptors: list[FrameDescriptor],
    *,
    split_token: str = SPLIT_TOKEN,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Keep held parents in both ARKitScenes and TUM without reading labels."""
    by_source: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for row in descriptors:
        previous = by_source[row.source_id].setdefault(row.parent_id, row.output_hw)
        require(previous == row.output_hw, "within-source parent shape drift")
    require(
        set(by_source) == {"arkitscenes", "tum_rgbd"},
        "multisource split requires ARKitScenes and TUM RGB-D",
    )
    arkit_train, arkit_selection, arkit_canary, arkit_receipt = select_parent_split(
        by_source["arkitscenes"],
        split_token=split_token,
    )
    tum_ranked = sorted(
        by_source["tum_rgbd"],
        key=lambda parent: stable_hash(
            f"{SOURCE_BOUNDARY_SPLIT_TOKEN}:tum_rgbd:{parent}"
        ),
    )
    require(len(tum_ranked) == 7, "TUM multisource parent count drift")
    tum_train = tuple(tum_ranked[:5])
    tum_selection = (tum_ranked[5],)
    tum_canary = (tum_ranked[6],)
    train = tuple(sorted((*arkit_train, *tum_train)))
    selection = tuple(sorted((*arkit_selection, *tum_selection)))
    canary = tuple(sorted((*arkit_canary, *tum_canary)))
    require(
        (len(train), len(selection), len(canary)) == (17, 3, 3),
        "multisource split cardinality drift",
    )
    require(
        not (set(train) & set(selection))
        and not (set(train) & set(canary))
        and not (set(selection) & set(canary)),
        "multisource split overlap",
    )
    return train, selection, canary, {
        "method": "SOURCE_STRATIFIED_ARKIT_12_2_2_TUM_5_1_1_WITHOUT_LABEL_CONTENT",
        "split_token": split_token,
        "source_boundary_split_token": SOURCE_BOUNDARY_SPLIT_TOKEN,
        "train_parents": list(train),
        "selection_parents": list(selection),
        "canary_parents": list(canary),
        "by_source": {
            "arkitscenes": {
                "train_parents": list(arkit_train),
                "selection_parents": list(arkit_selection),
                "canary_parents": list(arkit_canary),
                "receipt": arkit_receipt,
            },
            "tum_rgbd": {
                "train_parents": list(tum_train),
                "selection_parents": list(tum_selection),
                "canary_parents": list(tum_canary),
                "rankings": [
                    {
                        "parent_id": parent,
                        "rank_sha256": stable_hash(
                            f"{SOURCE_BOUNDARY_SPLIT_TOKEN}:tum_rgbd:{parent}"
                        ),
                    }
                    for parent in tum_ranked
                ],
            },
        },
    }


def build_frame_descriptors(
    stage0a_result_path: Path,
    label_dir: Path,
) -> tuple[list[FrameDescriptor], dict[str, Any], dict[str, Any]]:
    stage0a = load_json(stage0a_result_path)
    require(stage0a.get("status") == "COMPLETED", "Stage 0A result is not complete")
    source = stage0a["source"]
    parent_ids = tuple(str(value) for value in source["parents"])
    require(
        len(parent_ids) >= 4 and len(parent_ids) == len(set(parent_ids)),
        "Stage 0A parent roster invalid",
    )
    manifest_path = Path(source["manifest_path"]).resolve()
    require(manifest_path.is_file(), "source manifest missing")
    require(sha256_file(manifest_path) == source["manifest_sha256"], "source manifest SHA drift")
    manifest = load_json(manifest_path)
    videos = {
        str(video["video_id"]): video
        for video in select_source_videos(
            manifest,
            parent_ids,
            role_token=str(source.get("manifest_role_token", "TRAIN")),
        )
    }
    parent_runs = {str(row["parent_id"]): row for row in stage0a["parent_runs"]}
    require(set(parent_runs) == set(parent_ids), "Stage 0A parent-run roster drift")

    descriptors: list[FrameDescriptor] = []
    for parent_id in parent_ids:
        video = videos[parent_id]
        selected = [str(value) for value in video["selected_frame_stems"]]
        summaries = parent_runs[parent_id]["frame_summaries"]
        require(len(summaries) == 3, f"expected three Stage 0A frames: {parent_id}")
        for summary in summaries:
            frame_index = int(summary["frame_index"])
            frame_stem = str(summary["frame_stem"])
            require(selected[frame_index] == frame_stem, "frame stem/index binding drift")
            output_hw = tuple(int(value) for value in summary["output_hw"])
            require(len(output_hw) == 2, "output shape invalid")
            label_path = (label_dir / f"{frame_stem}.npz").resolve()
            require(label_path.is_file(), f"factor label missing: {label_path}")
            descriptors.append(
                FrameDescriptor(
                    parent_id=parent_id,
                    frame_index=frame_index,
                    frame_stem=frame_stem,
                    output_hw=(output_hw[0], output_hw[1]),
                    label_path=label_path,
                    video=video,
                )
            )
    expected_frames = 3 * len(parent_ids)
    require(len(descriptors) == expected_frames, "masked-student frame count drift")
    require(
        len({row.frame_stem for row in descriptors}) == expected_frames,
        "duplicate frame stem",
    )
    return descriptors, stage0a, manifest


def build_frame_descriptor_batches(
    stage0a_results: list[Path],
    label_dirs: list[Path],
) -> tuple[list[FrameDescriptor], list[dict[str, Any]]]:
    """Load multiple disjoint label batches without copying their payloads."""
    require(stage0a_results, "at least one source batch is required")
    require(
        len(stage0a_results) == len(label_dirs),
        "Stage 0A result and label-dir counts differ",
    )
    descriptors: list[FrameDescriptor] = []
    receipts: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    seen_stems: set[str] = set()
    for stage0a_result, label_dir in zip(stage0a_results, label_dirs):
        stage0a_result = stage0a_result.resolve()
        label_dir = label_dir.resolve()
        label_result = (label_dir / "result.json").resolve()
        require(stage0a_result.is_file(), f"Stage 0A result missing: {stage0a_result}")
        require(label_result.is_file(), f"factor-label result missing: {label_result}")
        batch, stage0a, _ = build_frame_descriptors(stage0a_result, label_dir)
        batch_parents = {row.parent_id for row in batch}
        batch_stems = {row.frame_stem for row in batch}
        require(not (seen_parents & batch_parents), "source batch parent overlap")
        require(not (seen_stems & batch_stems), "source batch frame-stem overlap")
        seen_parents.update(batch_parents)
        seen_stems.update(batch_stems)
        descriptors.extend(batch)
        receipts.append(
            {
                "stage0a_result_path": str(stage0a_result),
                "stage0a_result_sha256": sha256_file(stage0a_result),
                "factor_label_result_path": str(label_result),
                "factor_label_result_sha256": sha256_file(label_result),
                "source_manifest_sha256": stage0a["source"]["manifest_sha256"],
                "source_role": stage0a["source"]["role"],
                "parent_count": len(batch_parents),
                "frame_count": len(batch),
                "parents": sorted(batch_parents),
            }
        )
    return descriptors, receipts


def build_tum_bound_frame_descriptors(
    label_dir: Path,
    rgb_binding_path: Path,
) -> tuple[list[FrameDescriptor], dict[str, Any]]:
    label_dir = label_dir.resolve()
    label_result_path = (label_dir / "result.json").resolve()
    rgb_binding_path = rgb_binding_path.resolve()
    require(label_result_path.is_file(), "TUM unified label result missing")
    require(rgb_binding_path.is_file(), "source RGB binding missing")
    label_result = load_json(label_result_path)
    binding = load_json(rgb_binding_path)
    require(
        label_result.get("status") == "TUM_UNIFIED_FACTOR_LABELS_PASS",
        "TUM unified labels incomplete",
    )
    require(
        binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS",
        "source RGB binding incomplete",
    )
    label_rows = {str(row["frame_id"]): row for row in label_result["frames"]}
    binding_rows = {
        str(row["frame_id"]): row
        for row in binding["frames"]
        if row["source"] == "tum_rgbd"
    }
    require(
        len(label_rows) == len(binding_rows) == 21 and set(label_rows) == set(binding_rows),
        "TUM unified label/RGB identity drift",
    )
    descriptors: list[FrameDescriptor] = []
    for frame_id in sorted(label_rows):
        label_row = label_rows[frame_id]
        rgb_row = binding_rows[frame_id]
        label_path = (label_dir / f"{frame_id}.npz").resolve()
        require(label_path.is_file(), f"TUM unified factor label missing: {label_path}")
        require(
            sha256_file(label_path) == label_row["output_sha256"],
            "TUM unified factor label SHA drift",
        )
        output_hw = tuple(int(value) for value in label_row["shape_hw"])
        require(
            output_hw == tuple(int(value) for value in rgb_row["label_shape_hw"]),
            "TUM RGB/label shape binding drift",
        )
        require(
            str(label_row["parent_id"]) == str(rgb_row["parent_id"]),
            "TUM RGB/label parent binding drift",
        )
        descriptors.append(
            FrameDescriptor(
                parent_id=str(label_row["parent_id"]),
                frame_index=len(descriptors),
                frame_stem=frame_id,
                output_hw=(output_hw[0], output_hw[1]),
                label_path=label_path,
                video={},
                source_id="tum_rgbd",
                rgb_binding=rgb_row,
            )
        )
    return descriptors, {
        "source": "tum_rgbd",
        "factor_label_result_path": str(label_result_path),
        "factor_label_result_sha256": sha256_file(label_result_path),
        "rgb_binding_path": str(rgb_binding_path),
        "rgb_binding_sha256": sha256_file(rgb_binding_path),
        "parent_count": len({row.parent_id for row in descriptors}),
        "frame_count": len(descriptors),
        "parents": sorted({row.parent_id for row in descriptors}),
    }


def aggregate_label_digest(paths: Iterable[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted((Path(value).resolve() for value in paths), key=lambda value: value.name):
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return {
        "file_count": count,
        "total_bytes": total_bytes,
        "name_size_sha256_aggregate": digest.hexdigest().upper(),
    }


def load_targets(path: Path, expected_hw: tuple[int, int]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "metric_depth_m_hw",
            "metric_depth_valid_hw",
            "quality_tier_hw",
            "provenance_code_hw",
            "support_truth_hw",
            "support_truth_valid_hw",
            "support_quality_tier_hw",
            "boundary_distance_px_hw",
            "obstacle_evidence_truth_hw",
            "evidence_truth_valid_hw",
            "evidence_quality_tier_hw",
            "intrinsics_output",
        }
        require(required <= set(payload.files), f"masked-student label schema missing fields: {path.name}")
        metric_depth = np.asarray(payload["metric_depth_m_hw"], dtype=np.float32)
        require(metric_depth.shape == expected_hw, "label/output shape drift")
        metric_valid = np.asarray(payload["metric_depth_valid_hw"], dtype=np.bool_)
        support_valid = np.asarray(payload["support_truth_valid_hw"], dtype=np.bool_)
        evidence_valid = np.asarray(payload["evidence_truth_valid_hw"], dtype=np.bool_)
        boundary_valid = np.asarray(
            payload[
                "boundary_factor_valid_hw"
                if "boundary_factor_valid_hw" in payload.files
                else "evidence_truth_valid_hw"
            ],
            dtype=np.bool_,
        )
        boundary_tier = np.asarray(
            payload[
                "boundary_quality_tier_hw"
                if "boundary_quality_tier_hw" in payload.files
                else "evidence_quality_tier_hw"
            ],
            dtype=np.uint8,
        )
        boundary_distance = np.asarray(payload["boundary_distance_px_hw"], dtype=np.float32)
        boundary_soft = (
            np.asarray(payload["boundary_soft_probability_hw"], dtype=np.float32)
            if "boundary_soft_probability_hw" in payload.files
            else np.exp(-boundary_distance / BOUNDARY_HEAT_SIGMA_PX).astype(np.float32)
        )
        require(np.isfinite(metric_depth[metric_valid]).all(), "valid metric label non-finite")
        require(np.isfinite(boundary_distance[boundary_valid]).all(), "valid boundary distance non-finite")
        require(
            np.isfinite(boundary_soft[boundary_valid]).all()
            and np.all((boundary_soft[boundary_valid] >= 0.0) & (boundary_soft[boundary_valid] <= 1.0)),
            "valid boundary soft probability invalid",
        )
        targets = {
            "metric_depth_m": torch.from_numpy(np.nan_to_num(metric_depth, nan=1.0))[None, None],
            "metric_valid": torch.from_numpy(metric_valid)[None, None],
            "metric_tier": torch.from_numpy(np.asarray(payload["quality_tier_hw"], dtype=np.uint8))[None, None],
            "metric_provenance": torch.from_numpy(np.asarray(payload["provenance_code_hw"], dtype=np.uint8))[None, None],
            "support": torch.from_numpy(np.asarray(payload["support_truth_hw"], dtype=np.float32))[None, None],
            "support_valid": torch.from_numpy(support_valid)[None, None],
            "support_tier": torch.from_numpy(np.asarray(payload["support_quality_tier_hw"], dtype=np.uint8))[None, None],
            "boundary": torch.from_numpy((boundary_distance <= BOUNDARY_BAND_PX).astype(np.float32))[None, None],
            "boundary_distance_px": torch.from_numpy(
                np.nan_to_num(boundary_distance, nan=MAX_BOUNDARY_DISTANCE_PX)
            )[None, None],
            "boundary_soft": torch.from_numpy(
                np.nan_to_num(boundary_soft, nan=0.0)
            )[None, None],
            "boundary_valid": torch.from_numpy(boundary_valid)[None, None],
            "boundary_tier": torch.from_numpy(boundary_tier)[None, None],
            "obstacle": torch.from_numpy(
                np.asarray(payload["obstacle_evidence_truth_hw"], dtype=np.float32)
            )[None, None],
            "evidence_valid": torch.from_numpy(evidence_valid)[None, None],
            "evidence_tier": torch.from_numpy(
                np.asarray(payload["evidence_quality_tier_hw"], dtype=np.uint8)
            )[None, None],
        }
        intrinsics = np.asarray(payload["intrinsics_output"], dtype=np.float64)
    return targets, intrinsics


def load_bound_rgb(row: dict[str, Any]) -> np.ndarray:
    storage_kind = str(row["rgb_storage_kind"])
    if storage_kind == "file":
        source_path = Path(row["rgb_path"])
        require(source_path.is_file(), f"bound RGB file missing: {source_path}")
        payload = source_path.read_bytes()
    else:
        require(storage_kind == "tar_member", "unsupported bound RGB storage kind")
        archive_path = Path(row["rgb_source_archive"])
        require(archive_path.is_file(), f"bound RGB archive missing: {archive_path}")
        member_name = str(row["rgb_member"])
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = {item.name: item for item in archive.getmembers() if item.isfile()}
            selected = members.get(member_name)
            if selected is None:
                matches = [
                    item for name, item in members.items() if name.endswith("/" + member_name)
                ]
                require(len(matches) == 1, "bound RGB tar member ambiguous")
                selected = matches[0]
            stream = archive.extractfile(selected)
            require(stream is not None, "bound RGB tar member unreadable")
            payload = stream.read()
    require(
        hashlib.sha256(payload).hexdigest().upper() == str(row["rgb_sha256"]),
        "bound RGB payload SHA drift",
    )
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def preprocess_rgb(
    descriptor: FrameDescriptor,
    trajectory: np.ndarray | None,
    expected_intrinsics: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mapanything.utils.cropping import crop_resize_if_necessary

    height, width = descriptor.output_hw
    if descriptor.source_id == "tum_rgbd":
        require(descriptor.rgb_binding is not None, "TUM RGB binding missing")
        processed_image = crop_resize_if_necessary(
            image=load_bound_rgb(descriptor.rgb_binding),
            resolution=(width, height),
        )[0]
        processed_intrinsics = expected_intrinsics
    else:
        require(trajectory is not None, "ARKit trajectory missing")
        frame = load_factor_source_frame(descriptor.video, descriptor.frame_index, trajectory)
        processed_image, processed_intrinsics = crop_resize_if_necessary(
            image=np.asarray(frame["rgb_upright"], dtype=np.uint8),
            resolution=(width, height),
            intrinsics=np.asarray(frame["intrinsics_upright"], dtype=np.float64),
        )
    image = np.asarray(processed_image, dtype=np.uint8)
    intrinsics = np.asarray(processed_intrinsics, dtype=np.float64)
    require(image.shape == (height, width, 3), "student RGB transform shape drift")
    require(
        np.allclose(intrinsics, expected_intrinsics, atol=1e-5, rtol=0.0),
        "student RGB/label intrinsics drift",
    )
    image_float = image.astype(np.float32) / 255.0
    normalized = ((image_float - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
    return torch.from_numpy(normalized), torch.from_numpy(intrinsics.astype(np.float32))


def load_mobilenet_backbone(
    checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[MobileNetV3DenseFeatureExtractor, dict[str, Any]]:
    from torchvision.models import mobilenet_v3_small

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = mobilenet_v3_small(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = base.load_state_dict(state, strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "MobileNet checkpoint load drift",
    )
    model = MobileNetV3DenseFeatureExtractor(base.features).to(device).eval()
    model.requires_grad_(False)
    return model, {
        "backend": "torchvision.models.mobilenet_v3_small.features",
        "tap_indices": list(MobileNetV3DenseFeatureExtractor.TAP_INDICES),
        "output_channels": MobileNetV3DenseFeatureExtractor.OUTPUT_CHANNELS,
    }


def load_depthart_backbone(
    source: Path,
    checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[DepthArtDenseFeatureExtractor, dict[str, Any]]:
    deployment = REPO_ROOT / "scripts/research/hftf/deployment/depthart"
    sys.path.insert(0, str(deployment))
    from export_depthart_camera_external import install_timm_compat

    install_timm_compat()
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from model import load_model
    from network import tvimblock
    from depthart_selective_scan import install_depthart

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base = load_model(checkpoint, "S", "indoor", str(device))
    previous_scan = install_depthart(tvimblock)
    scan = {
        "backend": "depthart_selective_scan.cross_selective_scan",
        "replaced": f"{previous_scan.__module__}.{previous_scan.__name__}",
        "autograd_required": False,
    }
    model = DepthArtDenseFeatureExtractor(base).to(device).eval()
    model.requires_grad_(False)
    return model, scan


def extract_depthart_features(
    descriptors: list[FrameDescriptor],
    source: Path,
    checkpoint: Path,
    device: torch.device,
    seed: int,
    feature_profile: str = "shared",
) -> tuple[list[CachedFrame], dict[str, Any]]:
    require(device.type == "cuda", "DepthART feature extraction requires CUDA")
    require(
        feature_profile in {"shared", "decoder_pyramid"},
        "DepthART feature profile invalid",
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, scan = load_depthart_backbone(source, checkpoint, device, seed)
    trajectories: dict[str, np.ndarray] = {}
    cached: list[CachedFrame] = []
    padded_shapes: set[tuple[int, int]] = set()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for descriptor in descriptors:
        targets, expected_intrinsics = load_targets(descriptor.label_path, descriptor.output_hw)
        if descriptor.source_id == "arkitscenes" and descriptor.parent_id not in trajectories:
            trajectories[descriptor.parent_id] = parse_trajectory(
                resolve_trajectory_path(descriptor.video)
            )
        image, intrinsics = preprocess_rgb(
            descriptor,
            trajectories.get(descriptor.parent_id),
            expected_intrinsics,
        )
        image_batch = image[None].to(device)
        intrinsics_batch = intrinsics[None].to(device)
        height, width = descriptor.output_hw
        padded_height = int(math.ceil(height / 32.0) * 32)
        padded_width = int(math.ceil(width / 32.0) * 32)
        padded_shapes.add((padded_height, padded_width))
        image_batch = F.pad(
            image_batch,
            (0, padded_width - width, 0, padded_height - height),
            mode="constant",
            value=0.0,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            cameras = model.metric_depthart.cam_embedder(
                intrinsics_batch,
                padded_height,
                padded_width,
                device,
            )
            features = model.metric_depthart.pretrained.forward_with_adapters(
                image_batch,
                adapters=[
                    model.metric_depthart.daa1,
                    model.metric_depthart.daa2,
                    model.metric_depthart.daa3,
                    model.metric_depthart.daa4,
                ],
                cams=list(cameras),
            )
            relative_depth, shared, pyramid = model.decode(
                list(features),
                (padded_height, padded_width),
            )
            scale = model.metric_depthart.sfh(features[3], cameras[3])
            base_depth = (
                relative_depth
                * scale.view(-1, 1, 1, 1)
                * model.metric_depthart.max_depth
            )
        selected_feature = shared if feature_profile == "shared" else pyramid
        content_feature_height = int(
            round(selected_feature.shape[-2] * height / padded_height)
        )
        content_feature_width = int(
            round(selected_feature.shape[-1] * width / padded_width)
        )
        require(
            content_feature_height * padded_height
            == selected_feature.shape[-2] * height
            and content_feature_width * padded_width
            == selected_feature.shape[-1] * width,
            "DepthART padded feature/content ratio is not integral",
        )
        selected_feature = selected_feature[
            ...,
            :content_feature_height,
            :content_feature_width,
        ]
        base_depth = base_depth[..., :height, :width]
        require(
            bool(torch.isfinite(selected_feature).all().item()),
            "non-finite frozen dense feature",
        )
        require(bool(torch.isfinite(base_depth).all().item()), "non-finite DepthART base depth")
        cached.append(
            CachedFrame(
                descriptor=descriptor,
                feature=selected_feature[0].to(dtype=torch.float16, device="cpu"),
                base_depth_m=base_depth[0].float().clamp(0.05, 20.0).cpu(),
                targets=targets,
            )
        )
    peak = int(torch.cuda.max_memory_allocated())
    feature_shapes = sorted({tuple(frame.feature.shape) for frame in cached})
    del model
    torch.cuda.empty_cache()
    return cached, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(cached),
        "feature_profile": feature_profile,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "peak_cuda_allocated_bytes": peak,
        "feature_shapes_chw": [list(values) for values in feature_shapes],
        "bottom_right_padded_input_shapes_hw": [
            list(values) for values in sorted(padded_shapes)
        ],
        "scan_backend": scan,
    }


def extract_mobilenet_features(
    descriptors: list[FrameDescriptor],
    checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[list[CachedFrame], dict[str, Any]]:
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, backend = load_mobilenet_backbone(checkpoint, device, seed)
    trajectories: dict[str, np.ndarray] = {}
    cached: list[CachedFrame] = []
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    for descriptor in descriptors:
        targets, expected_intrinsics = load_targets(
            descriptor.label_path,
            descriptor.output_hw,
        )
        if descriptor.source_id == "arkitscenes" and descriptor.parent_id not in trajectories:
            trajectories[descriptor.parent_id] = parse_trajectory(
                resolve_trajectory_path(descriptor.video)
            )
        image, _ = preprocess_rgb(
            descriptor,
            trajectories.get(descriptor.parent_id),
            expected_intrinsics,
        )
        image_batch = image[None].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            feature = model(image_batch)
        require(bool(torch.isfinite(feature).all().item()), "non-finite MobileNet feature")
        height, width = descriptor.output_hw
        cached.append(
            CachedFrame(
                descriptor=descriptor,
                feature=feature[0].to(dtype=torch.float16, device="cpu"),
                base_depth_m=torch.ones((1, height, width), dtype=torch.float32),
                targets=targets,
            )
        )
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    feature_shapes = sorted({tuple(frame.feature.shape) for frame in cached})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached, {
        "elapsed_seconds": time.perf_counter() - started,
        "frame_count": len(cached),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_enabled else None,
        "peak_cuda_allocated_bytes": peak,
        "feature_shapes_chw": [list(values) for values in feature_shapes],
        "backend": backend,
    }


class DilatedPyramidTrunk(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden,
                        hidden,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(value)
        fused = self.fuse(torch.cat([branch(projected) for branch in self.branches], dim=1))
        return projected + fused


class MaskedFactorStudent(nn.Module):
    def __init__(
        self,
        channels: int = 48,
        hidden: int = 32,
        *,
        depth_mode: str = "residual",
        head_profile: str = "basic",
        use_base_depth_feature: bool = False,
        depth_gate_profile: str = "none",
    ) -> None:
        super().__init__()
        require(depth_mode in {"residual", "absolute_log"}, "invalid depth mode")
        require(
            head_profile in {"basic", "dilated_pyramid", "factor_split_dilated"},
            "invalid head profile",
        )
        require(hidden % 8 == 0, "head hidden channels must be divisible by 8")
        require(
            depth_gate_profile in {"none", "identity_sigmoid"},
            "invalid depth gate profile",
        )
        require(
            depth_gate_profile == "none" or depth_mode == "residual",
            "identity depth gate requires residual depth mode",
        )
        self.depth_mode = depth_mode
        self.head_profile = head_profile
        self.use_base_depth_feature = bool(use_base_depth_feature)
        self.depth_gate_profile = depth_gate_profile
        trunk_channels = channels + int(self.use_base_depth_feature)
        self.trunk = (
            DilatedPyramidTrunk(trunk_channels, hidden)
            if head_profile in {"dilated_pyramid", "factor_split_dilated"}
            else nn.Sequential(
                nn.Conv2d(trunk_channels, hidden, 1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
        )
        self.boundary_trunk = (
            DilatedPyramidTrunk(trunk_channels, hidden)
            if head_profile == "factor_split_dilated"
            else None
        )
        self.depth_residual = nn.Conv2d(hidden, 1, 1)
        self.depth_identity_gate_logits = (
            nn.Conv2d(hidden, 1, 1)
            if depth_gate_profile == "identity_sigmoid"
            else None
        )
        self.support_logits = nn.Conv2d(hidden, 1, 1)
        self.boundary_logits = nn.Conv2d(hidden, 1, 1)
        self.boundary_distance_logits = nn.Conv2d(hidden, 1, 1)
        self.obstacle_logits = nn.Conv2d(hidden, 1, 1)
        for layer in (
            self.depth_residual,
            self.support_logits,
            self.boundary_logits,
            self.boundary_distance_logits,
            self.obstacle_logits,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.depth_identity_gate_logits is not None:
            nn.init.zeros_(self.depth_identity_gate_logits.weight)
            nn.init.constant_(
                self.depth_identity_gate_logits.bias,
                math.log(
                    DEPTH_IDENTITY_GATE_PRIOR
                    / (1.0 - DEPTH_IDENTITY_GATE_PRIOR)
                ),
            )

    @staticmethod
    def _upsample(value: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, output_hw, mode="bilinear", align_corners=False)

    def initialize_priors(self, priors: dict[str, float]) -> None:
        def logit(probability: float) -> float:
            bounded = min(max(float(probability), 1e-4), 1.0 - 1e-4)
            return math.log(bounded / (1.0 - bounded))

        with torch.no_grad():
            if self.depth_mode == "absolute_log":
                self.depth_residual.bias.fill_(
                    math.log(max(float(priors["metric_depth_m"]), 0.05))
                )
            self.support_logits.bias.fill_(logit(priors["support_probability"]))
            self.boundary_logits.bias.fill_(logit(priors["boundary_probability"]))
            self.obstacle_logits.bias.fill_(logit(priors["obstacle_probability"]))
            self.boundary_distance_logits.bias.fill_(
                logit(priors["boundary_distance_px"] / MAX_BOUNDARY_DISTANCE_PX)
            )

    def forward(
        self,
        feature: torch.Tensor,
        base_depth_m: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        trunk_input = feature
        if self.use_base_depth_feature:
            base_guidance = F.interpolate(
                base_depth_m.clamp(0.05, 20.0).log(),
                feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            trunk_input = torch.cat([feature, base_guidance], dim=1)
        latent = self.trunk(trunk_input)
        boundary_latent = (
            self.boundary_trunk(trunk_input)
            if self.boundary_trunk is not None
            else latent
        )
        depth_value = self._upsample(self.depth_residual(latent), output_hw)
        depth_gate = None
        if self.depth_identity_gate_logits is not None:
            depth_gate = torch.sigmoid(
                self._upsample(self.depth_identity_gate_logits(latent), output_hw)
            )
        if self.depth_mode == "residual":
            log_correction = DEPTH_RESIDUAL_LOG_RANGE * torch.tanh(depth_value)
            if depth_gate is not None:
                log_correction = depth_gate * log_correction
            predicted_depth = base_depth_m * torch.exp(log_correction)
        else:
            predicted_depth = torch.exp(
                depth_value.clamp(math.log(0.05), math.log(20.0))
            )
        outputs = {
            "depth_m": predicted_depth,
            "base_depth_m": base_depth_m,
            "support_logits": self._upsample(self.support_logits(latent), output_hw),
            "boundary_logits": self._upsample(
                self.boundary_logits(boundary_latent), output_hw
            ),
            "boundary_distance_px": MAX_BOUNDARY_DISTANCE_PX
            * torch.sigmoid(
                self._upsample(
                    self.boundary_distance_logits(boundary_latent), output_hw
                )
            ),
            "obstacle_logits": self._upsample(self.obstacle_logits(latent), output_hw),
        }
        if depth_gate is not None:
            outputs["depth_identity_gate"] = depth_gate
        return outputs


def tier_weights(tiers: torch.Tensor) -> torch.Tensor:
    values = tiers.to(dtype=torch.long)
    require(bool(((values >= TIER_UNKNOWN) & (values <= TIER_A_SOURCE)).all()), "tier outside range")
    return TIER_WEIGHTS.to(values.device)[values]


def masked_weighted_mean(
    value: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    selected = valid.bool() & torch.isfinite(value) & (weight > 0)
    if not bool(selected.any()):
        return value.sum() * 0.0
    return (value[selected] * weight[selected]).sum() / weight[selected].sum().clamp_min(1e-12)


def compute_student_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    predicted_depth = outputs["depth_m"].clamp(0.05, 20.0)
    target_depth = targets["metric_depth_m"].clamp(0.05, 20.0)
    depth_error = F.smooth_l1_loss(
        predicted_depth.log(),
        target_depth.log(),
        reduction="none",
        beta=0.05,
    )
    depth_loss = masked_weighted_mean(
        depth_error,
        targets["metric_valid"],
        tier_weights(targets["metric_tier"]),
    )

    support_raw = F.binary_cross_entropy_with_logits(
        outputs["support_logits"],
        targets["support"],
        reduction="none",
        pos_weight=torch.as_tensor(
            class_weights["support_pos_weight"],
            device=outputs["support_logits"].device,
        ),
    )
    support_loss = masked_weighted_mean(
        support_raw,
        targets["support_valid"],
        tier_weights(targets["support_tier"]),
    )

    boundary_raw = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"],
        targets.get("boundary_soft", targets["boundary"]),
        reduction="none",
        pos_weight=torch.as_tensor(
            class_weights["boundary_pos_weight"],
            device=outputs["boundary_logits"].device,
        ),
    )
    boundary_weights = tier_weights(
        targets.get("boundary_tier", targets["evidence_tier"])
    )
    boundary_valid = targets.get("boundary_valid", targets["evidence_valid"])
    boundary_loss = masked_weighted_mean(
        boundary_raw,
        boundary_valid,
        boundary_weights,
    )
    distance_raw = F.smooth_l1_loss(
        torch.log1p(outputs["boundary_distance_px"]),
        torch.log1p(targets["boundary_distance_px"]),
        reduction="none",
        beta=0.20,
    )
    distance_loss = masked_weighted_mean(
        distance_raw,
        boundary_valid,
        boundary_weights,
    )
    obstacle_raw = F.binary_cross_entropy_with_logits(
        outputs["obstacle_logits"],
        targets["obstacle"],
        reduction="none",
    )
    obstacle_loss = masked_weighted_mean(
        obstacle_raw,
        targets["evidence_valid"],
        tier_weights(targets["evidence_tier"]),
    )
    raw = {
        "depth_log_huber": depth_loss,
        "support_bce": support_loss,
        "boundary_bce": boundary_loss,
        "boundary_distance": distance_loss,
        "obstacle_bce": obstacle_loss,
    }
    weighted = {name: raw[name] * LOSS_WEIGHTS[name] for name in raw}
    total = sum(weighted.values(), predicted_depth.sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": value for name, value in raw.items()},
        **{f"weighted/{name}": value for name, value in weighted.items()},
    }


def compute_depth_support_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Optimize only the two factors that replicated on fresh parents."""
    predicted_depth = outputs["depth_m"].clamp(0.05, 20.0)
    target_depth = targets["metric_depth_m"].clamp(0.05, 20.0)
    depth_raw = F.smooth_l1_loss(
        predicted_depth.log(),
        target_depth.log(),
        reduction="none",
        beta=0.05,
    )
    depth_loss = masked_weighted_mean(
        depth_raw,
        targets["metric_valid"],
        tier_weights(targets["metric_tier"]),
    )
    support_raw = F.binary_cross_entropy_with_logits(
        outputs["support_logits"],
        targets["support"],
        reduction="none",
        pos_weight=torch.as_tensor(
            class_weights["support_pos_weight"],
            device=outputs["support_logits"].device,
        ),
    )
    support_loss = masked_weighted_mean(
        support_raw,
        targets["support_valid"],
        tier_weights(targets["support_tier"]),
    )
    raw = {
        "depth_log_huber": depth_loss,
        "support_bce": support_loss,
    }
    weighted = {
        name: raw[name] * DEPTH_SUPPORT_LOSS_WEIGHTS[name]
        for name in raw
    }
    total = sum(weighted.values(), predicted_depth.sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": value for name, value in raw.items()},
        **{f"weighted/{name}": value for name, value in weighted.items()},
    }


def compute_depth_support_precision_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Favor meter precision and calibrated support without non-core gradients."""
    predicted_depth = outputs["depth_m"].clamp(0.05, 20.0)
    depth_weights = tier_weights(targets["metric_tier"])
    depth_valid = targets["metric_valid"] & torch.isfinite(
        targets["metric_depth_m"]
    )
    target_depth = torch.where(
        depth_valid,
        targets["metric_depth_m"].clamp(0.05, 20.0),
        predicted_depth.detach(),
    )
    log_raw = F.smooth_l1_loss(
        predicted_depth.log(),
        target_depth.log(),
        reduction="none",
        beta=0.05,
    )
    abs_raw = F.smooth_l1_loss(
        predicted_depth,
        target_depth,
        reduction="none",
        beta=0.05,
    )
    absolute_error = (predicted_depth - target_depth).abs()
    margin_raw = 0.02 * F.softplus((absolute_error - 0.10) / 0.02)

    support_probability = torch.sigmoid(outputs["support_logits"])
    support_weights = tier_weights(targets["support_tier"])
    support_valid = targets["support_valid"]
    support_target = torch.where(
        support_valid,
        targets["support"],
        torch.zeros_like(targets["support"]),
    )
    support_bce_raw = F.binary_cross_entropy_with_logits(
        outputs["support_logits"],
        support_target,
        reduction="none",
    )
    selected_weights = support_weights * support_valid.to(support_weights.dtype)
    dice_denominator = (
        selected_weights * (support_probability + support_target)
    ).sum()
    if bool((selected_weights > 0).any()):
        support_dice = 1.0 - (
            2.0
            * (selected_weights * support_probability * support_target).sum()
            + 1e-6
        ) / (dice_denominator + 1e-6)
    else:
        support_dice = outputs["support_logits"].sum() * 0.0

    raw = {
        "depth_log_huber": masked_weighted_mean(
            log_raw,
            depth_valid,
            depth_weights,
        ),
        "depth_abs_huber": masked_weighted_mean(
            abs_raw,
            depth_valid,
            depth_weights,
        ),
        "depth_bad_0_10_soft_margin": masked_weighted_mean(
            margin_raw,
            depth_valid,
            depth_weights,
        ),
        "support_bce": masked_weighted_mean(
            support_bce_raw,
            support_valid,
            support_weights,
        ),
        "support_soft_dice": support_dice,
    }
    if "depth_identity_gate" in outputs:
        base_depth = outputs["base_depth_m"].clamp(0.05, 20.0)
        required_log_correction = (
            target_depth.log() - base_depth.log()
        ).abs()
        gate_target = (required_log_correction / 0.25).clamp(0.0, 1.0)
        gate_raw = F.binary_cross_entropy(
            outputs["depth_identity_gate"].clamp(1e-5, 1.0 - 1e-5),
            gate_target,
            reduction="none",
        )
        raw["depth_identity_gate_bce"] = masked_weighted_mean(
            gate_raw,
            depth_valid,
            depth_weights,
        )
    weighted = {
        name: raw[name] * DEPTH_SUPPORT_PRECISION_LOSS_WEIGHTS[name]
        for name in raw
    }
    total = sum(weighted.values(), predicted_depth.sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": value for name, value in raw.items()},
        **{f"weighted/{name}": value for name, value in weighted.items()},
    }


def compute_boundary_only_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Train sparse physical boundaries as a continuous distance-derived heat field."""
    target_distance = targets["boundary_distance_px"].clamp(
        0.0,
        MAX_BOUNDARY_DISTANCE_PX,
    )
    predicted_distance = outputs["boundary_distance_px"].clamp(
        0.0,
        MAX_BOUNDARY_DISTANCE_PX,
    )
    target_heat = targets.get(
        "boundary_soft",
        torch.exp(-target_distance / BOUNDARY_HEAT_SIGMA_PX),
    )
    predicted_heat = torch.sigmoid(outputs["boundary_logits"])
    evidence_weights = tier_weights(
        targets.get("boundary_tier", targets["evidence_tier"])
    )
    valid = targets.get("boundary_valid", targets["evidence_valid"])

    # A small far-field floor still penalizes hallucinated boundaries, while
    # the continuous near-field term prevents the 0.48% two-pixel band from
    # disappearing inside millions of easy negatives.
    heat_weights = evidence_weights * (0.05 + 20.0 * target_heat)
    heat_raw = F.binary_cross_entropy_with_logits(
        outputs["boundary_logits"],
        target_heat,
        reduction="none",
    )
    heat_loss = masked_weighted_mean(heat_raw, valid, heat_weights)

    distance_weights = evidence_weights * (
        0.05 + 10.0 * torch.exp(-target_distance / 6.0)
    )
    distance_raw = F.smooth_l1_loss(
        torch.log1p(predicted_distance),
        torch.log1p(target_distance),
        reduction="none",
        beta=0.15,
    )
    distance_loss = masked_weighted_mean(distance_raw, valid, distance_weights)
    consistency_raw = F.smooth_l1_loss(
        predicted_heat,
        torch.exp(-predicted_distance / BOUNDARY_HEAT_SIGMA_PX),
        reduction="none",
        beta=0.05,
    )
    consistency_loss = masked_weighted_mean(
        consistency_raw,
        valid,
        evidence_weights,
    )
    raw = {
        "boundary_soft_heat_bce": heat_loss,
        "boundary_near_distance": distance_loss,
        "boundary_heat_distance_consistency": consistency_loss,
    }
    weighted = {
        name: raw[name] * BOUNDARY_ONLY_LOSS_WEIGHTS[name]
        for name in raw
    }
    total = sum(weighted.values(), predicted_distance.sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": value for name, value in raw.items()},
        **{f"weighted/{name}": value for name, value in weighted.items()},
    }


def compute_unified_continuous_boundary_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Keep dense-factor losses while giving boundary its validated continuous objective."""

    multifactor = compute_student_losses(outputs, targets, class_weights)
    boundary = compute_boundary_only_losses(outputs, targets)
    selected = {
        name: multifactor[f"raw/{name}"]
        for name in ("depth_log_huber", "support_bce", "obstacle_bce")
    }
    selected.update(
        {
            name: boundary[f"raw/{name}"]
            for name in BOUNDARY_ONLY_LOSS_WEIGHTS
        }
    )
    weighted = {
        name: selected[name] * UNIFIED_CONTINUOUS_BOUNDARY_LOSS_WEIGHTS[name]
        for name in selected
    }
    total = sum(weighted.values(), outputs["depth_m"].sum() * 0.0)
    return {
        "total": total,
        **{f"raw/{name}": value for name, value in selected.items()},
        **{f"weighted/{name}": value for name, value in weighted.items()},
    }


def compute_training_priors(frames: list[CachedFrame]) -> tuple[dict[str, float], dict[str, float]]:
    support_positive = support_count = 0.0
    boundary_positive = boundary_count = 0.0
    obstacle_sum = obstacle_count = 0.0
    distance_sum = distance_count = 0.0
    depth_samples: list[np.ndarray] = []
    for frame in frames:
        targets = frame.targets
        metric_mask = targets["metric_valid"].bool()
        support_mask = targets["support_valid"].bool()
        boundary_mask = targets.get("boundary_valid", targets["evidence_valid"]).bool()
        evidence_mask = targets["evidence_valid"].bool()
        depth_samples.append(
            targets["metric_depth_m"][metric_mask].numpy().astype(np.float32, copy=False)
        )
        support_positive += float(targets["support"][support_mask].sum())
        support_count += int(support_mask.sum())
        boundary_positive += float(targets["boundary"][boundary_mask].sum())
        boundary_count += int(boundary_mask.sum())
        obstacle_sum += float(targets["obstacle"][evidence_mask].sum())
        obstacle_count += int(evidence_mask.sum())
        distance_sum += float(targets["boundary_distance_px"][boundary_mask].sum())
        distance_count += int(boundary_mask.sum())
    require(support_count > 0 and boundary_count > 0, "training factor denominator empty")
    require(depth_samples and sum(value.size for value in depth_samples) > 0, "training depth denominator empty")
    metric_depth_m = float(np.median(np.concatenate(depth_samples)))
    support_probability = support_positive / support_count
    boundary_probability = boundary_positive / boundary_count
    obstacle_probability = obstacle_sum / obstacle_count
    boundary_distance = distance_sum / distance_count

    def positive_weight(probability: float, maximum: float) -> float:
        return min(max(math.sqrt((1.0 - probability) / max(probability, 1e-6)), 0.25), maximum)

    priors = {
        "metric_depth_m": metric_depth_m,
        "support_probability": support_probability,
        "boundary_probability": boundary_probability,
        "obstacle_probability": obstacle_probability,
        "boundary_distance_px": min(max(boundary_distance, 0.01), MAX_BOUNDARY_DISTANCE_PX - 0.01),
    }
    class_weights = {
        "support_pos_weight": positive_weight(support_probability, 10.0),
        "boundary_pos_weight": positive_weight(boundary_probability, 20.0),
    }
    return priors, class_weights


def move_targets(
    targets: dict[str, torch.Tensor],
    device: torch.device,
    *,
    flip: bool = False,
) -> dict[str, torch.Tensor]:
    output = {key: value.to(device) for key, value in targets.items()}
    if flip:
        output = {key: torch.flip(value, dims=(-1,)) for key, value in output.items()}
    return output


def build_epoch_order(
    source_ids: list[str],
    rng: np.random.Generator,
    *,
    source_balanced: bool,
) -> list[int]:
    if not source_balanced:
        return [int(value) for value in rng.permutation(len(source_ids))]
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, source_id in enumerate(source_ids):
        by_source[source_id].append(index)
    require(len(by_source) >= 2, "source-balanced order requires at least two sources")
    samples_per_source = max(len(values) for values in by_source.values())
    schedules: dict[str, list[int]] = {}
    for source_id, values in sorted(by_source.items()):
        schedule: list[int] = []
        while len(schedule) < samples_per_source:
            schedule.extend(int(value) for value in rng.permutation(values))
        schedules[source_id] = schedule[:samples_per_source]
    return [
        schedules[source_id][position]
        for position in range(samples_per_source)
        for source_id in sorted(schedules)
    ]


def train_student(
    model: MaskedFactorStudent,
    frames: list[CachedFrame],
    class_weights: dict[str, float],
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    objective_profile: str = "multifactor",
    source_balanced: bool = False,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    require(epochs > 0 and learning_rate > 0, "training configuration invalid")
    model.train()
    require(
        objective_profile
        in {
            "multifactor",
            "depth_support",
            "depth_support_precision",
            "boundary_only",
            "unified_continuous_boundary",
        },
        "invalid objective profile",
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = []
    source_visits: dict[str, int] = defaultdict(int)
    optimizer_steps = 0
    for epoch in range(epochs):
        order = build_epoch_order(
            [row.descriptor.source_id for row in frames],
            rng,
            source_balanced=source_balanced,
        )
        loss_values: dict[str, list[float]] = defaultdict(list)
        gradient_values: list[float] = []
        for index in order:
            frame = frames[int(index)]
            source_visits[frame.descriptor.source_id] += 1
            flip = bool(rng.random() < 0.5)
            feature = frame.feature[None].float().to(device)
            base_depth = frame.base_depth_m[None].to(device)
            targets = move_targets(frame.targets, device, flip=flip)
            if flip:
                feature = torch.flip(feature, dims=(-1,))
                base_depth = torch.flip(base_depth, dims=(-1,))
            outputs = model(feature, base_depth, frame.descriptor.output_hw)
            if objective_profile == "boundary_only":
                losses = compute_boundary_only_losses(outputs, targets)
            elif objective_profile == "unified_continuous_boundary":
                losses = compute_unified_continuous_boundary_losses(
                    outputs,
                    targets,
                    class_weights,
                )
            elif objective_profile == "depth_support_precision":
                losses = compute_depth_support_precision_losses(outputs, targets)
            elif objective_profile == "depth_support":
                losses = compute_depth_support_losses(
                    outputs,
                    targets,
                    class_weights,
                )
            else:
                losses = compute_student_losses(outputs, targets, class_weights)
            require(bool(torch.isfinite(losses["total"]).item()), "non-finite student loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
            require(math.isfinite(gradient), "non-finite student gradient")
            optimizer.step()
            optimizer_steps += 1
            gradient_values.append(gradient)
            for name, value in losses.items():
                loss_values[name].append(float(value.detach().item()))
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "mean_total_loss": float(np.mean(loss_values["total"])),
                    "mean_gradient_norm": float(np.mean(gradient_values)),
                    **{
                        f"mean_{name.replace('/', '_')}": float(np.mean(values))
                        for name, values in loss_values.items()
                        if name.startswith("raw/")
                    },
                }
            )
    return history, {
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_steps": optimizer_steps,
        "source_balanced_sampling": source_balanced,
        "source_optimizer_visits": dict(sorted(source_visits.items())),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def fit_scalar_support_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    steps: int,
    learning_rate: float = 0.05,
) -> dict[str, float]:
    """Fit temperature and bias on frozen train-only support logits."""
    require(steps > 0 and learning_rate > 0.0, "calibration configuration invalid")
    selected = (
        torch.isfinite(logits)
        & torch.isfinite(targets)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    require(bool(selected.any()), "support calibration denominator empty")
    logits = logits[selected].detach()
    targets = targets[selected].detach()
    weights = weights[selected].detach()

    def loss_for(temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raw = F.binary_cross_entropy_with_logits(
            logits / temperature + bias,
            targets,
            reduction="none",
        )
        return (raw * weights).sum() / weights.sum().clamp_min(1e-12)

    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    bias = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.Adam((log_temperature, bias), lr=learning_rate)
    unit_temperature = torch.ones((), device=logits.device)
    zero_bias = torch.zeros((), device=logits.device)
    before = float(loss_for(unit_temperature, zero_bias).detach())
    best_loss = before
    best_temperature = 1.0
    best_bias = 0.0
    for _ in range(steps):
        temperature = log_temperature.clamp(
            math.log(0.25),
            math.log(4.0),
        ).exp()
        bounded_bias = bias.clamp(-4.0, 4.0)
        loss = loss_for(temperature, bounded_bias)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_temperature = float(temperature.detach())
            best_bias = float(bounded_bias.detach())
    require(
        math.isfinite(best_loss)
        and math.isfinite(best_temperature)
        and math.isfinite(best_bias),
        "non-finite support calibration",
    )
    return {
        "temperature": best_temperature,
        "bias_delta": best_bias,
        "train_weighted_bce_before": before,
        "train_weighted_bce_after": best_loss,
        "train_weighted_bce_relative_reduction": relative_improvement(
            before,
            best_loss,
        )
        or 0.0,
    }


def calibrate_support_head(
    model: MaskedFactorStudent,
    frames: list[CachedFrame],
    device: torch.device,
    *,
    steps: int,
    maximum_pixels: int = 2_000_000,
) -> dict[str, Any]:
    require(frames and maximum_pixels > 0, "support calibration input invalid")
    model.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for frame in frames:
            targets = move_targets(frame.targets, device)
            output = model(
                frame.feature[None].float().to(device),
                frame.base_depth_m[None].to(device),
                frame.descriptor.output_hw,
            )
            valid = targets["support_valid"] & torch.isfinite(targets["support"])
            weights = tier_weights(targets["support_tier"])
            selected = valid & (weights > 0)
            logits_parts.append(output["support_logits"][selected].detach())
            target_parts.append(targets["support"][selected].detach())
            weight_parts.append(weights[selected].detach())
    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    weights = torch.cat(weight_parts)
    candidate_pixels = int(logits.numel())
    require(candidate_pixels > 0, "support calibration pixels empty")
    stride = max(1, math.ceil(candidate_pixels / maximum_pixels))
    logits = logits[::stride]
    targets = targets[::stride]
    weights = weights[::stride]
    fitted = fit_scalar_support_calibration(
        logits,
        targets,
        weights,
        steps=steps,
    )
    temperature = float(fitted["temperature"])
    bias_delta = float(fitted["bias_delta"])
    with torch.no_grad():
        model.support_logits.weight.div_(temperature)
        model.support_logits.bias.div_(temperature).add_(bias_delta)
    return {
        "method": "TRAIN_ONLY_SCALAR_TEMPERATURE_AND_BIAS_FOLDED_INTO_SUPPORT_HEAD",
        "steps": steps,
        "candidate_pixels": candidate_pixels,
        "used_pixels": int(logits.numel()),
        "sampling_stride": stride,
        **fitted,
    }


def _binary_counts(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[int, int, int, int]:
    mask = valid.bool()
    prediction = predicted.bool()
    truth = target >= 0.5
    return (
        int((prediction & truth & mask).sum()),
        int((prediction & ~truth & mask).sum()),
        int((~prediction & truth & mask).sum()),
        int(mask.sum()),
    )


def evaluate_frames(
    model: MaskedFactorStudent,
    frames: list[CachedFrame],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accumulators: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for frame in frames:
            parent = frame.descriptor.parent_id
            feature = frame.feature[None].float().to(device)
            base_depth = frame.base_depth_m[None].to(device)
            targets = move_targets(frame.targets, device)
            outputs = model(feature, base_depth, frame.descriptor.output_hw)
            for scope in ("__all__", parent):
                values = accumulators[scope]
                depth_mask = targets["metric_valid"].bool()
                depth_error = (outputs["depth_m"] - targets["metric_depth_m"]).abs()
                values["depth_abs_sum"] += float(depth_error[depth_mask].sum())
                values["depth_bad_sum"] += int((depth_error[depth_mask] > 0.10).sum())
                values["depth_count"] += int(depth_mask.sum())
                teacher_mask = depth_mask & (
                    targets["metric_provenance"] == PROVENANCE_TEACHER
                )
                values["teacher_depth_abs_sum"] += float(depth_error[teacher_mask].sum())
                values["teacher_depth_count"] += int(teacher_mask.sum())

                support_mask = targets["support_valid"].bool()
                support_bce = F.binary_cross_entropy_with_logits(
                    outputs["support_logits"], targets["support"], reduction="none"
                )
                values["support_bce_sum"] += float(support_bce[support_mask].sum())
                support_counts = _binary_counts(
                    torch.sigmoid(outputs["support_logits"]) >= 0.5,
                    targets["support"],
                    support_mask,
                )
                values["support_tp"] += support_counts[0]
                values["support_fp"] += support_counts[1]
                values["support_fn"] += support_counts[2]
                values["support_count"] += support_counts[3]

                boundary_mask = targets.get(
                    "boundary_valid", targets["evidence_valid"]
                ).bool()
                evidence_mask = targets["evidence_valid"].bool()
                boundary_bce = F.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"], targets["boundary"], reduction="none"
                )
                values["boundary_bce_sum"] += float(boundary_bce[boundary_mask].sum())
                boundary_soft_target = targets.get(
                    "boundary_soft",
                    torch.exp(
                        -targets["boundary_distance_px"] / BOUNDARY_HEAT_SIGMA_PX
                    ),
                )
                boundary_soft_bce = F.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"],
                    boundary_soft_target,
                    reduction="none",
                )
                values["boundary_soft_bce_sum"] += float(
                    boundary_soft_bce[boundary_mask].sum()
                )
                boundary_counts = _binary_counts(
                    torch.sigmoid(outputs["boundary_logits"]) >= 0.5,
                    targets["boundary"],
                    boundary_mask,
                )
                values["boundary_tp"] += boundary_counts[0]
                values["boundary_fp"] += boundary_counts[1]
                values["boundary_fn"] += boundary_counts[2]
                values["boundary_count"] += boundary_counts[3]
                distance_boundary_counts = _binary_counts(
                    outputs["boundary_distance_px"] <= BOUNDARY_BAND_PX,
                    targets["boundary"],
                    boundary_mask,
                )
                values["distance_boundary_tp"] += distance_boundary_counts[0]
                values["distance_boundary_fp"] += distance_boundary_counts[1]
                values["distance_boundary_fn"] += distance_boundary_counts[2]
                values["distance_boundary_count"] += distance_boundary_counts[3]
                distance_error = (
                    outputs["boundary_distance_px"]
                    - targets["boundary_distance_px"]
                ).abs()
                values["distance_abs_sum"] += float(distance_error[boundary_mask].sum())
                values["distance_count"] += int(boundary_mask.sum())
                obstacle_bce = F.binary_cross_entropy_with_logits(
                    outputs["obstacle_logits"], targets["obstacle"], reduction="none"
                )
                values["obstacle_bce_sum"] += float(obstacle_bce[evidence_mask].sum())
                values["obstacle_count"] += int(evidence_mask.sum())

    def finalize(values: dict[str, float]) -> dict[str, Any]:
        def ratio(numerator: str, denominator: str) -> float | None:
            count = values[denominator]
            return float(values[numerator] / count) if count > 0 else None

        def f1(prefix: str) -> float | None:
            denominator = 2 * values[f"{prefix}_tp"] + values[f"{prefix}_fp"] + values[f"{prefix}_fn"]
            return float(2 * values[f"{prefix}_tp"] / denominator) if denominator > 0 else None

        return {
            "depth_mae_m": ratio("depth_abs_sum", "depth_count"),
            "depth_bad_0_10m_rate": ratio("depth_bad_sum", "depth_count"),
            "depth_valid_pixels": int(values["depth_count"]),
            "teacher_only_depth_mae_m": ratio(
                "teacher_depth_abs_sum", "teacher_depth_count"
            ),
            "teacher_only_depth_pixels": int(values["teacher_depth_count"]),
            "support_bce": ratio("support_bce_sum", "support_count"),
            "support_f1": f1("support"),
            "support_valid_pixels": int(values["support_count"]),
            "boundary_bce": ratio("boundary_bce_sum", "boundary_count"),
            "boundary_soft_bce": ratio(
                "boundary_soft_bce_sum", "boundary_count"
            ),
            "boundary_f1": f1("boundary"),
            "boundary_distance_f1": f1("distance_boundary"),
            "boundary_valid_pixels": int(values["boundary_count"]),
            "boundary_distance_mae_px": ratio(
                "distance_abs_sum", "distance_count"
            ),
            "obstacle_bce": ratio("obstacle_bce_sum", "obstacle_count"),
        }

    overall = finalize(accumulators["__all__"])
    parent_rows = [
        {"parent_id": parent, **finalize(accumulators[parent])}
        for parent in sorted({frame.descriptor.parent_id for frame in frames})
    ]
    macro: dict[str, Any] = {}
    metric_names = (
        "depth_mae_m",
        "depth_bad_0_10m_rate",
        "teacher_only_depth_mae_m",
        "support_bce",
        "support_f1",
        "boundary_bce",
        "boundary_soft_bce",
        "boundary_f1",
        "boundary_distance_f1",
        "boundary_distance_mae_px",
        "obstacle_bce",
    )
    for name in metric_names:
        finite = [float(row[name]) for row in parent_rows if row[name] is not None]
        macro[name] = float(np.mean(finite)) if finite else None
        macro[f"{name}_evaluable_parents"] = len(finite)
    return {"overall": overall, "parent_macro": macro, "parents": parent_rows}


def relative_improvement(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float((before - after) / max(abs(before), 1e-12))


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    objective_profile = str(args.objective_profile)
    head_profile = str(args.head_profile)
    head_hidden_channels = int(args.head_hidden_channels)
    use_base_depth_feature = bool(args.use_base_depth_feature)
    depth_gate_profile = str(args.depth_gate_profile)
    stage0a_results = [args.stage0a_result.resolve()] + [
        value.resolve()
        for value in getattr(args, "additional_stage0a_result", [])
    ]
    label_dirs = [args.label_dir.resolve()] + [
        value.resolve()
        for value in getattr(args, "additional_label_dir", [])
    ]
    require(
        len(stage0a_results) == len(label_dirs),
        "additional Stage 0A result and label-dir counts differ",
    )
    encoder_name = str(args.encoder)
    depthart_source = args.depthart_source.resolve()
    if encoder_name == "mobilenet_v3_small":
        checkpoint = args.mobilenet_checkpoint.resolve()
        feature_channels = MobileNetV3DenseFeatureExtractor.OUTPUT_CHANNELS
        depth_mode = "absolute_log"
        encoder_label = "FROZEN_TORCHVISION_MOBILENET_V3_SMALL_IMAGENET1K"
        feature_profile = "mobilenet_pyramid"
    else:
        checkpoint = args.depthart_checkpoint.resolve()
        feature_profile = str(args.depthart_feature_profile)
        feature_channels = (
            DEPTHART_PYRAMID_CHANNELS
            if feature_profile == "decoder_pyramid"
            else DEPTHART_SHARED_CHANNELS
        )
        depth_mode = "residual"
        encoder_label = "FROZEN_DEPTHART_S_METRIC_INDOOR"
    output_dir = args.output_dir.resolve()
    require(checkpoint.is_file(), f"{encoder_name} checkpoint missing")
    if encoder_name == "depthart_s":
        require(depthart_source.is_dir(), "DepthART source missing")
    require(not output_dir.exists(), "masked-student output collision")
    output_dir.mkdir(parents=True, exist_ok=False)

    descriptors, source_batches = build_frame_descriptor_batches(
        stage0a_results,
        label_dirs,
    )
    tum_label_dir = getattr(args, "tum_label_dir", None)
    if tum_label_dir is not None:
        require(
            bool(getattr(args, "source_stratified_split", False)),
            "TUM unified factors require source-stratified split",
        )
        tum_descriptors, tum_receipt = build_tum_bound_frame_descriptors(
            Path(tum_label_dir),
            Path(args.source_rgb_binding),
        )
        require(
            not ({row.parent_id for row in descriptors} & {row.parent_id for row in tum_descriptors}),
            "ARKit/TUM parent overlap",
        )
        require(
            not ({row.frame_stem for row in descriptors} & {row.frame_stem for row in tum_descriptors}),
            "ARKit/TUM frame identity overlap",
        )
        descriptors.extend(tum_descriptors)
        source_batches.append(tum_receipt)
    parent_shapes: dict[str, tuple[int, int]] = {}
    for descriptor in descriptors:
        previous = parent_shapes.setdefault(descriptor.parent_id, descriptor.output_hw)
        require(previous == descriptor.output_hw, "within-parent output shape drift")
    if bool(getattr(args, "source_stratified_split", False)):
        train_parents, selection_parents, canary_parents, split_receipt = (
            select_multisource_parent_split(
                descriptors,
                split_token=str(args.split_token),
            )
        )
    else:
        train_parents, selection_parents, canary_parents, split_receipt = select_parent_split(
            parent_shapes,
            split_token=str(args.split_token),
        )
    fit_all_consumed_parents = bool(args.fit_all_consumed_parents)
    fit_parents = (
        tuple(sorted(parent_shapes))
        if fit_all_consumed_parents
        else train_parents
    )
    split_receipt["fit_parents"] = list(fit_parents)
    split_receipt["internal_selection_and_canary_included_in_fit"] = (
        fit_all_consumed_parents
    )
    train_descriptors = [row for row in descriptors if row.parent_id in fit_parents]
    selection_descriptors = [
        row for row in descriptors if row.parent_id in selection_parents
    ]
    canary_descriptors = [row for row in descriptors if row.parent_id in canary_parents]
    expected_frame_split = (
        len(descriptors)
        if fit_all_consumed_parents
        else 3 * len(train_parents),
        3 * len(selection_parents),
        3 * len(canary_parents),
    )
    require(
        (len(train_descriptors), len(selection_descriptors), len(canary_descriptors))
        == expected_frame_split,
        "frame split drift",
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "masked-student canary requires CUDA")
    if encoder_name == "mobilenet_v3_small":
        cached, extraction = extract_mobilenet_features(
            descriptors,
            checkpoint,
            device,
            args.seed,
        )
    else:
        cached, extraction = extract_depthart_features(
            descriptors,
            depthart_source,
            checkpoint,
            device,
            args.seed,
            feature_profile=feature_profile,
        )
    train_frames = [row for row in cached if row.descriptor.parent_id in fit_parents]
    selection_frames = [
        row for row in cached if row.descriptor.parent_id in selection_parents
    ]
    canary_frames = [row for row in cached if row.descriptor.parent_id in canary_parents]
    priors, class_weights = compute_training_priors(train_frames)

    torch.manual_seed(args.seed)
    model = MaskedFactorStudent(
        channels=feature_channels,
        hidden=head_hidden_channels,
        depth_mode=depth_mode,
        head_profile=head_profile,
        use_base_depth_feature=use_base_depth_feature,
        depth_gate_profile=depth_gate_profile,
    ).to(device)
    model.initialize_priors(priors)
    if objective_profile == "boundary_only":
        for unused_head in (
            model.depth_residual,
            model.support_logits,
            model.obstacle_logits,
        ):
            unused_head.requires_grad_(False)
    elif objective_profile in {"depth_support", "depth_support_precision"}:
        for unused_head in (
            model.boundary_logits,
            model.boundary_distance_logits,
            model.obstacle_logits,
        ):
            unused_head.requires_grad_(False)
    before_train = evaluate_frames(model, train_frames, device)
    before_selection = evaluate_frames(model, selection_frames, device)
    before_canary = evaluate_frames(model, canary_frames, device)
    source_ids = sorted({row.descriptor.source_id for row in cached})

    def evaluate_by_source(frames: list[CachedFrame]) -> dict[str, Any]:
        return {
            source_id: evaluate_frames(
                model,
                [row for row in frames if row.descriptor.source_id == source_id],
                device,
            )
            for source_id in source_ids
            if any(row.descriptor.source_id == source_id for row in frames)
        }

    before_train_by_source = evaluate_by_source(train_frames)
    before_selection_by_source = evaluate_by_source(selection_frames)
    before_canary_by_source = evaluate_by_source(canary_frames)
    history, training = train_student(
        model,
        train_frames,
        class_weights,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        objective_profile=objective_profile,
        source_balanced=bool(getattr(args, "source_balanced_sampling", False)),
    )
    support_calibration: dict[str, Any] | None = None
    if int(args.support_calibration_steps) > 0:
        require(
            objective_profile == "depth_support_precision",
            "support calibration is only enabled for depth_support_precision",
        )
        support_calibration = calibrate_support_head(
            model,
            train_frames,
            device,
            steps=int(args.support_calibration_steps),
        )
    after_train = evaluate_frames(model, train_frames, device)
    after_selection = evaluate_frames(model, selection_frames, device)
    after_canary = evaluate_frames(model, canary_frames, device)
    after_train_by_source = evaluate_by_source(train_frames)
    after_selection_by_source = evaluate_by_source(selection_frames)
    after_canary_by_source = evaluate_by_source(canary_frames)

    checkpoint_receipt = save_checkpoint_exclusive(
        output_dir / "masked-factor-head.pt",
        {
            "schema": "blindassist_ag_st_masked_factor_student_checkpoint_v1",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": {
                "frozen_encoder": encoder_label,
                "input_feature_channels": feature_channels,
                "feature_profile": feature_profile,
                "head_hidden_channels": head_hidden_channels,
                "head_profile": head_profile,
                "use_base_depth_feature": use_base_depth_feature,
                "depth_gate_profile": depth_gate_profile,
                "depth_mode": depth_mode,
                "objective_profile": objective_profile,
                "outputs": [
                    "metric_depth_residual",
                    *(
                        ["depth_identity_gate"]
                        if depth_gate_profile == "identity_sigmoid"
                        else []
                    ),
                    "support_logit",
                    "boundary_logit",
                    "boundary_distance_px",
                    "obstacle_logit",
                ],
            },
            "split": split_receipt,
            "priors": priors,
            "class_weights": class_weights,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "objective_profile": objective_profile,
            "support_calibration": support_calibration,
            "source_balanced_sampling": bool(
                getattr(args, "source_balanced_sampling", False)
            ),
        },
    )

    def metric_improvements(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
        before_macro = before["parent_macro"]
        after_macro = after["parent_macro"]
        return {
            "depth_mae": relative_improvement(
                before_macro["depth_mae_m"], after_macro["depth_mae_m"]
            ),
            "support_bce": relative_improvement(
                before_macro["support_bce"], after_macro["support_bce"]
            ),
            "boundary_bce": relative_improvement(
                before_macro["boundary_bce"], after_macro["boundary_bce"]
            ),
            "boundary_soft_bce": relative_improvement(
                before_macro["boundary_soft_bce"],
                after_macro["boundary_soft_bce"],
            ),
            "boundary_distance_mae": relative_improvement(
                before_macro["boundary_distance_mae_px"],
                after_macro["boundary_distance_mae_px"],
            ),
            "obstacle_bce": relative_improvement(
                before_macro["obstacle_bce"], after_macro["obstacle_bce"]
            ),
        }

    selection_improvements = metric_improvements(before_selection, after_selection)
    canary_improvements = metric_improvements(before_canary, after_canary)
    if objective_profile == "boundary_only":
        boundary_signals = {
            "selection_soft_bce_improved": (
                selection_improvements["boundary_soft_bce"] is not None
                and selection_improvements["boundary_soft_bce"] > 0.0
            ),
            "canary_soft_bce_improved": (
                canary_improvements["boundary_soft_bce"] is not None
                and canary_improvements["boundary_soft_bce"] > 0.0
            ),
            "selection_distance_mae_improved": (
                selection_improvements["boundary_distance_mae"] is not None
                and selection_improvements["boundary_distance_mae"] > 0.0
            ),
            "canary_distance_mae_improved": (
                canary_improvements["boundary_distance_mae"] is not None
                and canary_improvements["boundary_distance_mae"] > 0.0
            ),
            "selection_fixed_f1_nonzero": (
                after_selection["parent_macro"]["boundary_f1"] or 0.0
            ) > 0.0,
            "canary_fixed_f1_nonzero": (
                after_canary["parent_macro"]["boundary_f1"] or 0.0
            ) > 0.0,
        }
        required_boundary_signals = tuple(boundary_signals.values())[:4]
        supported = sum(required_boundary_signals)
        status = (
            "PARENT_DISJOINT_BOUNDARY_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported == 4
            else "PARENT_DISJOINT_PARTIAL_BOUNDARY_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported >= 2
            else "PARENT_DISJOINT_BOUNDARY_LEARNABILITY_SIGNAL_NOT_SUPPORTED"
        )
        core_signals = boundary_signals
        total_core_signals = 4
    elif objective_profile in {"depth_support", "depth_support_precision"}:
        depth_support_signals = {
            "selection_depth_mae_improved": (
                selection_improvements["depth_mae"] is not None
                and selection_improvements["depth_mae"] > 0.0
            ),
            "selection_support_bce_improved": (
                selection_improvements["support_bce"] is not None
                and selection_improvements["support_bce"] > 0.0
            ),
            "canary_depth_mae_improved": (
                canary_improvements["depth_mae"] is not None
                and canary_improvements["depth_mae"] > 0.0
            ),
            "canary_support_bce_improved": (
                canary_improvements["support_bce"] is not None
                and canary_improvements["support_bce"] > 0.0
            ),
        }
        supported = sum(depth_support_signals.values())
        total_core_signals = 4
        status = (
            "PARENT_DISJOINT_DEPTH_SUPPORT_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported == total_core_signals
            else "PARENT_DISJOINT_PARTIAL_DEPTH_SUPPORT_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported > 0
            else "PARENT_DISJOINT_DEPTH_SUPPORT_LEARNABILITY_SIGNAL_NOT_SUPPORTED"
        )
        core_signals = depth_support_signals
    else:
        core_signals = {
            name: value is not None and value > 0.0
            for name, value in canary_improvements.items()
            if name in {"depth_mae", "support_bce", "boundary_bce"}
        }
        supported = sum(core_signals.values())
        total_core_signals = 3
        status = (
            "PARENT_DISJOINT_MULTI_FACTOR_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported == 3
            else "PARENT_DISJOINT_PARTIAL_FACTOR_LEARNABILITY_SIGNAL_SUPPORTED"
            if supported > 0
            else "PARENT_DISJOINT_FACTOR_LEARNABILITY_SIGNAL_NOT_SUPPORTED"
        )
    if fit_all_consumed_parents:
        status = (
            "ALL_CONSUMED_PARENT_DEPTH_SUPPORT_PRECISION_FIT_COMPLETED_"
            "AWAITING_FRESH_EVALUATION"
        )
    label_digest = aggregate_label_digest(row.label_path for row in descriptors)
    encoder_inputs: dict[str, Any] = {
        "encoder": encoder_label,
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": sha256_file(checkpoint),
    }
    if encoder_name == "depthart_s":
        encoder_inputs["depthart_source"] = str(depthart_source)
    source_inputs: dict[str, Any] = {"source_batches": source_batches}
    if len(source_batches) == 1:
        source_inputs.update(
            {
                "stage0a_result_path": source_batches[0]["stage0a_result_path"],
                "stage0a_result_sha256": source_batches[0]["stage0a_result_sha256"],
                "factor_label_result_path": source_batches[0]["factor_label_result_path"],
                "factor_label_result_sha256": source_batches[0]["factor_label_result_sha256"],
                "source_manifest_sha256": source_batches[0]["source_manifest_sha256"],
            }
        )
    result = {
        "schema": "blindassist_ag_st_masked_student_wild_lab_result_v1",
        "status": status,
        "mode": "WILD_LAB_REVERSIBLE_EXPLORATION",
        "question": (
            f"Can {encoder_label} RGB features learn sparse AG-ST physical-boundary distance/heat supervision across internal parent-disjoint roles?"
            if objective_profile == "boundary_only"
            else f"Can {encoder_label} RGB features reduce metric depth error and calibrate support across {len(parent_shapes)} consumed parents using absolute-meter, 0.10 m soft-margin, BCE, and Dice supervision?"
            if objective_profile == "depth_support_precision"
            else f"Can {encoder_label} RGB features strengthen replicated AG-ST depth/support factors across {len(parent_shapes)} consumed parents without obstacle or boundary gradients?"
            if objective_profile == "depth_support"
            else f"Can {encoder_label} RGB features learn graded AG-ST depth/support/boundary pseudo-labels across {len(source_ids)} sources and internal parent-disjoint selection/canary roles?"
        ),
        "inputs": {
            "factor_label_payloads": label_digest,
            **source_inputs,
            **encoder_inputs,
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "split": split_receipt,
        "architecture": {
            "encoder": encoder_label,
            "head": (
                "FACTOR_SPLIT_DILATED_1_2_4_BOUNDARY_AND_MULTI_FACTOR_HEAD"
                if head_profile == "factor_split_dilated"
                else "DILATED_1_2_4_PYRAMID_MULTI_FACTOR_DENSE_HEAD"
                if head_profile == "dilated_pyramid"
                else "ONE_BY_ONE_THEN_THREE_BY_THREE_MULTI_FACTOR_DENSE_HEAD"
            ),
            "feature_profile": feature_profile,
            "head_profile": head_profile,
            "head_hidden_channels": head_hidden_channels,
            "use_base_depth_feature": use_base_depth_feature,
            "depth_gate_profile": depth_gate_profile,
            "objective_profile": objective_profile,
            "encoder_trainable_parameters": 0,
            "head_total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "head_trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "depth_output": (
                (
                    "BASE_DEPTH_TIMES_IDENTITY_GATE_TIMES_TANH_LOG_RESIDUAL_"
                    f"RANGE_{DEPTH_RESIDUAL_LOG_RANGE}"
                    if depth_gate_profile == "identity_sigmoid"
                    else f"BASE_DEPTH_TIMES_TANH_LOG_RESIDUAL_RANGE_{DEPTH_RESIDUAL_LOG_RANGE}"
                )
                if depth_mode == "residual"
                else "ABSOLUTE_LOG_DEPTH"
            ),
            "boundary_target": f"BOUNDARY_DISTANCE_LE_{BOUNDARY_BAND_PX:.1f}_PX",
        },
        "supervision": {
            "tier_weights": {
                "A_SOURCE": float(TIER_WEIGHTS[TIER_A_SOURCE]),
                "B_ANCHORED": float(TIER_WEIGHTS[TIER_B_ANCHORED]),
                "C_TEACHER": float(TIER_WEIGHTS[TIER_C_TEACHER]),
                "UNKNOWN": float(TIER_WEIGHTS[TIER_UNKNOWN]),
            },
            "priors_from_train_only": priors,
            "class_weights_from_train_only": class_weights,
            "loss_weights": (
                BOUNDARY_ONLY_LOSS_WEIGHTS
                if objective_profile == "boundary_only"
                else UNIFIED_CONTINUOUS_BOUNDARY_LOSS_WEIGHTS
                if objective_profile == "unified_continuous_boundary"
                else DEPTH_SUPPORT_PRECISION_LOSS_WEIGHTS
                if objective_profile == "depth_support_precision"
                else DEPTH_SUPPORT_LOSS_WEIGHTS
                if objective_profile == "depth_support"
                else LOSS_WEIGHTS
            ),
            "boundary_heat_sigma_px": BOUNDARY_HEAT_SIGMA_PX,
            "support_calibration": support_calibration,
            "source_balanced_sampling": bool(
                getattr(args, "source_balanced_sampling", False)
            ),
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "support_calibration_steps": int(args.support_calibration_steps),
            "feature_extraction": extraction,
            "training": training,
            "total_seconds": time.perf_counter() - started,
        },
        "metrics": {
            "before_train": before_train,
            "after_train": after_train,
            "before_selection": before_selection,
            "after_selection": after_selection,
            "before_canary": before_canary,
            "after_canary": after_canary,
            "before_train_by_source": before_train_by_source,
            "after_train_by_source": after_train_by_source,
            "before_selection_by_source": before_selection_by_source,
            "after_selection_by_source": after_selection_by_source,
            "before_canary_by_source": before_canary_by_source,
            "after_canary_by_source": after_canary_by_source,
            "selection_parent_macro_relative_improvement": selection_improvements,
            "canary_parent_macro_relative_improvement": canary_improvements,
            "canary_core_factor_improvement_signals": core_signals,
        },
        "history": history,
        "checkpoint": checkpoint_receipt,
        "decision": {
            "complete_truth_required_for_this_canary": False,
            "second_teacher_required_for_this_canary": False,
            "selection_used_for_model_or_threshold_choice": False,
            "internal_selection_and_canary_included_in_fit": (
                fit_all_consumed_parents
            ),
            "student_signal_supported_factor_count": supported,
            "student_signal_total_core_factor_count": total_core_signals,
            "next_step": (
                "Evaluate once on the pre-reserved fresh confirmation cohort without fitting or threshold selection."
                if fit_all_consumed_parents
                else "Scale physical-boundary supervision to fresh parent/source sequences."
                if objective_profile == "boundary_only" and supported == 4
                else "Keep boundary isolated and revise its representation before scaling."
                if objective_profile == "boundary_only"
                else "Evaluate once on the pre-reserved fresh parent cohort without fitting or threshold selection."
                if objective_profile
                in {"depth_support", "depth_support_precision"}
                and supported == total_core_signals
                else "Do not scale the combined depth/support student until the failed internal signal is localized."
                if objective_profile
                in {"depth_support", "depth_support_precision"}
                else "Scale pseudo-label materialization to more parent-disjoint RGB sequences and repeat the frozen-encoder student test."
                if supported == total_core_signals
                else "Inspect unsupported factor residuals before scaling or adding another Teacher."
            ),
        },
        "claim_boundary": (
            f"All-consumed-parent pseudo-label fit over {len(parent_shapes)} ARKitScenes parents awaiting an independently reserved fresh evaluation."
            if fit_all_consumed_parents
            else f"Internal parent-disjoint pseudo-label learnability diagnostic over {len(parent_shapes)} previously consumed parents from {len(source_ids)} source(s)."
        )
        + " No objective truth, cross-dataset generalization, formal F1 authorization, deployment, product, or safety claim.",
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": status,
                "result": str((output_dir / "result.json").resolve()),
                "selection_parent_macro_relative_improvement": selection_improvements,
                "canary_parent_macro_relative_improvement": canary_improvements,
                "before_canary": before_canary["parent_macro"],
                "after_canary": after_canary["parent_macro"],
                "total_seconds": result["execution"]["total_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a-result", type=Path, default=DEFAULT_STAGE0A_RESULT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--additional-stage0a-result",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--additional-label-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--tum-label-dir",
        type=Path,
        default=None,
        help="Optional R13 TUM unified factor-label directory.",
    )
    parser.add_argument(
        "--source-rgb-binding",
        type=Path,
        default=DEFAULT_SOURCE_RGB_BINDING,
    )
    parser.add_argument("--source-stratified-split", action="store_true")
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument(
        "--encoder",
        choices=("mobilenet_v3_small", "depthart_s"),
        default="mobilenet_v3_small",
    )
    parser.add_argument(
        "--objective-profile",
        choices=(
            "multifactor",
            "depth_support",
            "depth_support_precision",
            "boundary_only",
            "unified_continuous_boundary",
        ),
        default="multifactor",
    )
    parser.add_argument("--split-token", default=SPLIT_TOKEN)
    parser.add_argument(
        "--depthart-feature-profile",
        choices=("shared", "decoder_pyramid"),
        default="shared",
    )
    parser.add_argument(
        "--head-profile",
        choices=("basic", "dilated_pyramid", "factor_split_dilated"),
        default="basic",
    )
    parser.add_argument("--head-hidden-channels", type=int, default=32)
    parser.add_argument("--use-base-depth-feature", action="store_true")
    parser.add_argument(
        "--depth-gate-profile",
        choices=("none", "identity_sigmoid"),
        default="none",
    )
    parser.add_argument(
        "--mobilenet-checkpoint",
        type=Path,
        default=DEFAULT_MOBILENET_CHECKPOINT,
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--support-calibration-steps", type=int, default=0)
    parser.add_argument("--fit-all-consumed-parents", action="store_true")
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
