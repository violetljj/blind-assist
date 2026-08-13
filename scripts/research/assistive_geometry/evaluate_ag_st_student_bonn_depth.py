#!/usr/bin/env python3
"""Evaluate a frozen AG-ST depth student on source-native Bonn RGB-D depth.

The source depth is opened only after RGB/K inference and is never passed to
DepthART or the masked student.  This evaluator is a DEVELOPMENT-only,
CROSS_DATASET depth diagnostic; it does not evaluate support or make a dataset
license conclusion.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    DEPTHART_PYRAMID_CHANNELS,
    DEPTHART_SHARED_CHANNELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DepthArtDenseFeatureExtractor,
    MaskedFactorStudent,
    load_depthart_backbone,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BONN_ROOT = (
    REPO_ROOT
    / "artifacts.local"
    / "datasets"
    / "bonn-rgbd-dynamic-full-r0"
    / "rgbd_bonn_dataset"
)
DEFAULT_BONN_ARCHIVE = (
    REPO_ROOT
    / "artifacts.local"
    / "downloads"
    / "bonn-rgbd-dynamic"
    / "rgbd_bonn_dataset.zip"
)
DEFAULT_BONN_CATALOG = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "hftf"
    / "p3-public-rgbd-source-admission-r0"
    / "bonn-valid-pairs-identity-catalog-attempt-03.json"
)
DEFAULT_BONN_RECEIPT = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "hftf"
    / "p3-public-rgbd-source-admission-r0"
    / "bonn-valid-pairs-identity-receipt-attempt-03.json"
)

EXPECTED_ARCHIVE_SHA256 = (
    "D2AFDC286ECDB02A3E57CB98E3908185B1133A3F79F181970064EB177221A84B"
)
EXPECTED_CATALOG_SHA256 = (
    "0F6307FBDB26295B7EB2CE407F3D586A43D2709653443F34CCFBB97DDB14E20E"
)
EXPECTED_RECEIPT_SHA256 = (
    "6E4089B3C2E5535D51104914EDE1DD783475EFB237FEBD517F80F1A7A20F6D56"
)
CATALOG_SCHEMA = "blindassist_p3_public_rgbd_source_admission_r0_catalog"
RECEIPT_SCHEMA = "blindassist_p3_public_bonn_identity_inventory_r0_receipt"
CHECKPOINT_SCHEMA = "blindassist_ag_st_masked_factor_student_checkpoint_v1"
RESULT_SCHEMA = "blindassist_ag_st_student_bonn_source_native_depth_development_v1"
COHORT_SCHEMA = "blindassist_ag_st_bonn_mixed_domain_cohort_v1"

BONN_WIDTH = 640
BONN_HEIGHT = 480
BONN_DEPTH_SCALE = 5000.0
BONN_INTRINSICS = np.asarray(
    [
        [542.822841, 0.0, 315.593520],
        [0.0, 542.576870, 237.756098],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
MAX_RGB_DEPTH_DELTA_SECONDS = 0.05
BAD_DEPTH_THRESHOLD_M = 0.10

# Zero-based row indices in each sequence's source-native rgb.txt.
FIXED_FRAME_INDICES_BY_SEQUENCE: dict[str, tuple[int, int, int]] = {
    "rgbd_bonn_kidnapping_box": (272, 545, 818),
    "rgbd_bonn_kidnapping_box2": (323, 647, 970),
    "rgbd_bonn_placing_nonobstructing_box": (180, 360, 540),
    "rgbd_bonn_placing_nonobstructing_box2": (169, 338, 507),
    "rgbd_bonn_placing_obstructing_box": (249, 499, 750),
    "rgbd_bonn_removing_nonobstructing_box": (123, 247, 370),
    "rgbd_bonn_removing_obstructing_box": (240, 481, 722),
    "rgbd_bonn_synchronous2": (89, 179, 268),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


@dataclass(frozen=True)
class TumIndexRow:
    row_index: int
    timestamp_seconds: float
    relative_path: Path
    absolute_path: Path


@dataclass(frozen=True)
class BonnFramePair:
    sequence_id: str
    rgb: TumIndexRow
    depth: TumIndexRow
    association_delta_seconds: float


def read_tum_index(sequence_root: Path, filename: str) -> list[TumIndexRow]:
    sequence_root = sequence_root.resolve()
    path = sequence_root / filename
    require(path.is_file(), f"missing TUM index: {path}")
    rows: list[TumIndexRow] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        require(
            len(fields) == 2,
            f"{path}:{line_number}: expected timestamp and relative path",
        )
        timestamp = float(fields[0])
        require(math.isfinite(timestamp), f"{path}:{line_number}: non-finite timestamp")
        relative = Path(fields[1])
        require(not relative.is_absolute(), f"{path}:{line_number}: absolute member path")
        absolute = (sequence_root / relative).resolve()
        try:
            absolute.relative_to(sequence_root)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: member escapes sequence root") from error
        rows.append(
            TumIndexRow(
                row_index=len(rows),
                timestamp_seconds=timestamp,
                relative_path=relative,
                absolute_path=absolute,
            )
        )
    require(rows, f"index contains no rows: {path}")
    require(
        all(
            left.timestamp_seconds < right.timestamp_seconds
            for left, right in pairwise(rows)
        ),
        f"timestamps are not strictly increasing: {path}",
    )
    return rows


def pair_rgb_depth_unique(
    sequence_id: str,
    rgb_rows: list[TumIndexRow],
    depth_rows: list[TumIndexRow],
    *,
    maximum_delta_seconds: float = MAX_RGB_DEPTH_DELTA_SECONDS,
) -> dict[int, BonnFramePair]:
    """Greedily pair nearest source rows without reusing a depth member.

    Candidate and used-set semantics intentionally match the frozen Bonn
    identity materializer: only the insertion-neighbor rows are considered and
    a paired depth row is never admitted again.
    """

    require(maximum_delta_seconds >= 0.0, "negative association threshold")
    usable_depth = [row for row in depth_rows if row.absolute_path.is_file()]
    require(usable_depth, f"no materialized depth rows: {sequence_id}")
    depth_times = [row.timestamp_seconds for row in usable_depth]
    used: set[int] = set()
    pairs: dict[int, BonnFramePair] = {}
    for rgb in rgb_rows:
        insertion = bisect.bisect_left(depth_times, rgb.timestamp_seconds)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(usable_depth) and index not in used
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda index: (
                abs(depth_times[index] - rgb.timestamp_seconds),
                index,
            ),
        )
        delta = abs(depth_times[selected] - rgb.timestamp_seconds)
        if delta > maximum_delta_seconds:
            continue
        used.add(selected)
        pairs[rgb.row_index] = BonnFramePair(
            sequence_id=sequence_id,
            rgb=rgb,
            depth=usable_depth[selected],
            association_delta_seconds=delta,
        )
    require(
        len({pair.depth.absolute_path for pair in pairs.values()}) == len(pairs),
        f"depth member reused: {sequence_id}",
    )
    return pairs


def load_cohort_indices(
    path: Path,
    role: str,
) -> dict[str, tuple[int, int, int]]:
    payload = load_json(path)
    require(payload.get("schema") == COHORT_SCHEMA, "Bonn cohort schema drift")
    require(role in {"fit", "evaluation"}, "unsupported Bonn cohort role")
    rows = payload.get(f"{role}_parents")
    require(isinstance(rows, list) and len(rows) >= 4, "Bonn cohort role empty")
    output: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        require(isinstance(row, dict), "Bonn cohort row must be an object")
        parent_id = str(row["parent_id"])
        indices = tuple(int(value) for value in row["rgb_row_indices_zero_based"])
        require(len(indices) == 3 and len(set(indices)) == 3, "Bonn cohort frame drift")
        require(all(value >= 0 for value in indices), "negative Bonn cohort index")
        require(parent_id not in output, "duplicate Bonn cohort parent")
        output[parent_id] = indices
    return output


def fixed_frame_pairs(
    dataset_root: Path,
    frame_indices_by_sequence: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, list[BonnFramePair]]:
    dataset_root = dataset_root.resolve()
    frame_indices = frame_indices_by_sequence or FIXED_FRAME_INDICES_BY_SEQUENCE
    output: dict[str, list[BonnFramePair]] = {}
    for sequence_id, indices in frame_indices.items():
        sequence_root = dataset_root / sequence_id
        require(sequence_root.is_dir(), f"missing Bonn sequence: {sequence_root}")
        rgb_rows = read_tum_index(sequence_root, "rgb.txt")
        depth_rows = read_tum_index(sequence_root, "depth.txt")
        require(
            all(row.absolute_path.is_file() for row in rgb_rows),
            f"missing RGB index member: {sequence_id}",
        )
        pairing = pair_rgb_depth_unique(sequence_id, rgb_rows, depth_rows)
        missing = [index for index in indices if index not in pairing]
        require(not missing, f"fixed RGB rows lack unique depth pairs: {sequence_id}:{missing}")
        output[sequence_id] = [pairing[index] for index in indices]
    return output


def load_rgb_native(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(
        rgb.shape == (BONN_HEIGHT, BONN_WIDTH, 3),
        f"Bonn RGB shape drift: {path}:{rgb.shape}",
    )
    return rgb


def load_depth_native(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    require(
        raw.shape == (BONN_HEIGHT, BONN_WIDTH),
        f"Bonn depth shape drift: {path}:{raw.shape}",
    )
    require(raw.dtype == np.uint16, f"Bonn depth dtype drift: {path}:{raw.dtype}")
    return depth_uint16_to_metres(raw)


def depth_uint16_to_metres(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(raw)
    require(value.dtype == np.uint16, f"depth must be uint16, got {value.dtype}")
    require(value.ndim == 2, f"depth must be rank two, got {value.shape}")
    valid = value > 0
    depth_m = value.astype(np.float32) / BONN_DEPTH_SCALE
    return depth_m, valid


def normalize_rgb_native(rgb: np.ndarray) -> torch.Tensor:
    value = np.asarray(rgb, dtype=np.uint8)
    require(
        value.shape == (BONN_HEIGHT, BONN_WIDTH, 3),
        f"Bonn RGB shape drift: {value.shape}",
    )
    image_float = value.astype(np.float32) / 255.0
    normalized = (
        (image_float - IMAGENET_MEAN) / IMAGENET_STD
    ).transpose(2, 0, 1).copy()
    return torch.from_numpy(normalized)


def depth_error_sums(
    truth_m: np.ndarray,
    prediction_m: np.ndarray,
    source_valid: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(truth_m, dtype=np.float32)
    prediction = np.asarray(prediction_m, dtype=np.float32)
    valid = np.asarray(source_valid, dtype=np.bool_)
    require(
        truth.shape == prediction.shape == valid.shape,
        "depth metric shape mismatch",
    )
    require(bool(valid.any()), "source-native depth denominator empty")
    require(bool(np.isfinite(truth[valid]).all()), "non-finite valid source depth")
    require(bool(np.isfinite(prediction[valid]).all()), "non-finite student prediction")
    absolute = np.abs(prediction[valid] - truth[valid])
    return {
        "absolute_error_sum_m": float(absolute.astype(np.float64).sum()),
        "bad_gt_0_10_m_count": int((absolute > BAD_DEPTH_THRESHOLD_M).sum()),
        "valid_pixel_count": int(absolute.size),
    }


def merge_depth_error_sums(
    rows: Iterable[dict[str, float | int]],
) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        "absolute_error_sum_m": 0.0,
        "bad_gt_0_10_m_count": 0,
        "valid_pixel_count": 0,
    }
    for row in rows:
        output["absolute_error_sum_m"] = float(output["absolute_error_sum_m"]) + float(
            row["absolute_error_sum_m"]
        )
        output["bad_gt_0_10_m_count"] = int(output["bad_gt_0_10_m_count"]) + int(
            row["bad_gt_0_10_m_count"]
        )
        output["valid_pixel_count"] = int(output["valid_pixel_count"]) + int(
            row["valid_pixel_count"]
        )
    require(int(output["valid_pixel_count"]) > 0, "depth metric denominator empty")
    return output


def finalize_depth_metrics(sums: dict[str, float | int]) -> dict[str, float | int]:
    count = int(sums["valid_pixel_count"])
    require(count > 0, "depth metric denominator empty")
    return {
        "mae_m": float(sums["absolute_error_sum_m"]) / count,
        "bad_gt_0_10_m_fraction": int(sums["bad_gt_0_10_m_count"]) / count,
        "bad_gt_0_10_m_count": int(sums["bad_gt_0_10_m_count"]),
        "valid_pixel_count": count,
    }


def parent_macro_metrics(parent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(parent_rows, "parent metric set empty")
    baseline_mae = [float(row["baseline"]["mae_m"]) for row in parent_rows]
    student_mae = [float(row["student"]["mae_m"]) for row in parent_rows]
    baseline_bad = [
        float(row["baseline"]["bad_gt_0_10_m_fraction"]) for row in parent_rows
    ]
    student_bad = [
        float(row["student"]["bad_gt_0_10_m_fraction"]) for row in parent_rows
    ]

    def comparison(before: list[float], after: list[float]) -> dict[str, float | None]:
        before_mean = float(np.mean(before))
        after_mean = float(np.mean(after))
        return {
            "initialized_baseline": before_mean,
            "student": after_mean,
            "absolute_reduction": before_mean - after_mean,
            "relative_reduction": (
                (before_mean - after_mean) / before_mean if before_mean > 0.0 else None
            ),
        }

    return {
        "parent_count": len(parent_rows),
        "aggregation": (
            "pool source-valid pixels within each parent, then unweighted mean "
            "across parents"
        ),
        "mae_m": comparison(baseline_mae, student_mae),
        "bad_gt_0_10_m_fraction": comparison(baseline_bad, student_bad),
    }


def checkpoint_parent_ids(payload: dict[str, Any]) -> set[str]:
    split = payload.get("split", {})
    require(isinstance(split, dict), "checkpoint split missing")
    return {
        str(parent)
        for role in (
            "train_parents",
            "selection_parents",
            "canary_parents",
            "fit_parents",
        )
        for parent in split.get(role, [])
    }


def checkpoint_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == CHECKPOINT_SCHEMA, "student checkpoint schema drift")
    architecture = payload.get("architecture")
    require(isinstance(architecture, dict), "student checkpoint architecture missing")
    require(
        architecture.get("frozen_encoder") == "FROZEN_DEPTHART_S_METRIC_INDOOR",
        "student checkpoint encoder is not metric DepthART-S indoor",
    )
    feature_profile = str(architecture.get("feature_profile", "shared"))
    require(
        feature_profile in {"shared", "decoder_pyramid"},
        "student feature profile drift",
    )
    channels = int(architecture["input_feature_channels"])
    expected_channels = (
        DEPTHART_PYRAMID_CHANNELS
        if feature_profile == "decoder_pyramid"
        else DEPTHART_SHARED_CHANNELS
    )
    require(channels == expected_channels, "student feature channel/profile drift")
    objective = str(
        architecture.get("objective_profile", payload.get("objective_profile", "multifactor"))
    )
    require(objective != "boundary_only", "boundary-only checkpoint has no depth claim")
    return {
        "feature_profile": feature_profile,
        "channels": channels,
        "hidden": int(architecture.get("head_hidden_channels", 32)),
        "head_profile": str(architecture.get("head_profile", "basic")),
        "use_base_depth_feature": bool(
            architecture.get("use_base_depth_feature", False)
        ),
        "depth_gate_profile": str(
            architecture.get("depth_gate_profile", "none")
        ),
        "depth_mode": str(architecture["depth_mode"]),
        "objective_profile": objective,
    }


def build_students(
    checkpoint: dict[str, Any],
    architecture: dict[str, Any],
    device: torch.device,
) -> tuple[MaskedFactorStudent, MaskedFactorStudent]:
    seed = int(checkpoint["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {
        "channels": architecture["channels"],
        "hidden": architecture["hidden"],
        "depth_mode": architecture["depth_mode"],
        "head_profile": architecture["head_profile"],
        "use_base_depth_feature": architecture["use_base_depth_feature"],
        "depth_gate_profile": architecture["depth_gate_profile"],
    }
    baseline = MaskedFactorStudent(**kwargs).to(device).eval()
    baseline.initialize_priors(checkpoint["priors"])
    student = MaskedFactorStudent(**kwargs).to(device).eval()
    incompatible = student.load_state_dict(checkpoint["state_dict"], strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "student state-dict drift",
    )
    return baseline, student


def extract_rgb_only_feature(
    extractor: DepthArtDenseFeatureExtractor,
    rgb: np.ndarray,
    feature_profile: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a frozen feature/base-depth pair from RGB plus fixed K."""

    return extract_rgb_only_feature_with_intrinsics(
        extractor,
        rgb,
        BONN_INTRINSICS,
        feature_profile,
        device,
        amp_dtype,
    )


def extract_rgb_only_feature_with_intrinsics(
    extractor: DepthArtDenseFeatureExtractor,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    feature_profile: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen DepthART features from native 640x480 RGB and explicit K."""

    image = normalize_rgb_native(rgb)[None].to(device)
    intrinsics_value = np.asarray(intrinsics, dtype=np.float32)
    require(intrinsics_value.shape == (3, 3), "camera intrinsics shape drift")
    intrinsics_tensor = torch.from_numpy(intrinsics_value.copy())[None].to(device)
    output_hw = (BONN_HEIGHT, BONN_WIDTH)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        cameras = extractor.metric_depthart.cam_embedder(
            intrinsics_tensor,
            BONN_HEIGHT,
            BONN_WIDTH,
            device,
        )
        features = extractor.metric_depthart.pretrained.forward_with_adapters(
            image,
            adapters=[
                extractor.metric_depthart.daa1,
                extractor.metric_depthart.daa2,
                extractor.metric_depthart.daa3,
                extractor.metric_depthart.daa4,
            ],
            cams=list(cameras),
        )
        relative_depth, shared, pyramid = extractor.decode(list(features), output_hw)
        scale = extractor.metric_depthart.sfh(features[3], cameras[3])
        base_depth = (
            relative_depth
            * scale.view(-1, 1, 1, 1)
            * extractor.metric_depthart.max_depth
        ).float().clamp(0.05, 20.0)
        selected_feature = shared if feature_profile == "shared" else pyramid
    require(bool(torch.isfinite(selected_feature).all()), "non-finite Bonn feature")
    require(bool(torch.isfinite(base_depth).all()), "non-finite Bonn base depth")
    return selected_feature, base_depth


def infer_rgb_only_depths(
    extractor: DepthArtDenseFeatureExtractor,
    baseline: MaskedFactorStudent,
    student: MaskedFactorStudent,
    rgb: np.ndarray,
    feature_profile: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer both heads from RGB plus fixed K; source depth is not an argument."""

    selected_feature, base_depth = extract_rgb_only_feature(
        extractor,
        rgb,
        feature_profile,
        device,
        amp_dtype,
    )
    output_hw = (BONN_HEIGHT, BONN_WIDTH)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        initialized_depth = baseline(
            selected_feature,
            base_depth,
            output_hw,
        )["depth_m"]
        student_depth = student(
            selected_feature,
            base_depth,
            output_hw,
        )["depth_m"]
    require(bool(torch.isfinite(initialized_depth).all()), "non-finite baseline depth")
    require(bool(torch.isfinite(student_depth).all()), "non-finite student depth")
    return (
        initialized_depth[0, 0].float().cpu().numpy(),
        student_depth[0, 0].float().cpu().numpy(),
    )


def validate_source_receipts(
    dataset_root: Path,
    archive_path: Path,
    catalog_path: Path,
    receipt_path: Path,
    cohort_parents: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, description in (
        (dataset_root, "Bonn dataset root"),
        (archive_path, "Bonn archive"),
        (catalog_path, "Bonn identity catalog"),
        (receipt_path, "Bonn integrity receipt"),
    ):
        require(path.exists(), f"{description} missing: {path}")
    catalog_sha256 = sha256_file(catalog_path)
    receipt_sha256 = sha256_file(receipt_path)
    require(catalog_sha256 == EXPECTED_CATALOG_SHA256, "Bonn catalog SHA drift")
    require(receipt_sha256 == EXPECTED_RECEIPT_SHA256, "Bonn receipt SHA drift")
    catalog = load_json(catalog_path)
    receipt = load_json(receipt_path)
    require(catalog.get("schema") == CATALOG_SCHEMA, "Bonn catalog schema drift")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "Bonn receipt schema drift")
    archive = receipt.get("archive", {})
    require(
        str(archive.get("sha256", "")).upper() == EXPECTED_ARCHIVE_SHA256,
        "Bonn archive receipt SHA drift",
    )
    require(
        int(archive.get("bytes", -1)) == archive_path.stat().st_size,
        "Bonn archive byte receipt drift",
    )
    catalog_bonn = [
        source
        for source in catalog.get("sources", [])
        if source.get("dataset_id") == "bonn_rgbd_dynamic"
    ]
    require(len(catalog_bonn) == 1, "Bonn catalog source missing or ambiguous")
    inventory_ids = {
        str(row["parent_id"])
        for row in catalog_bonn[0].get("identity_inventory", [])
    }
    required_parents = cohort_parents or set(FIXED_FRAME_INDICES_BY_SEQUENCE)
    require(
        required_parents <= inventory_ids,
        "fixed Bonn cohort absent from identity catalog",
    )
    parent_receipts = {
        str(row["parent_id"]): row for row in receipt.get("parents", [])
    }
    require(
        required_parents <= set(parent_receipts),
        "fixed Bonn cohort absent from integrity receipt",
    )
    index_receipts: dict[str, Any] = {}
    for sequence_id in sorted(required_parents):
        sequence_root = dataset_root / sequence_id
        source = parent_receipts[sequence_id]
        rgb_sha256 = sha256_file(sequence_root / "rgb.txt")
        depth_sha256 = sha256_file(sequence_root / "depth.txt")
        require(
            rgb_sha256 == str(source["rgb_index_sha256"]).upper(),
            f"RGB index SHA drift: {sequence_id}",
        )
        require(
            depth_sha256 == str(source["depth_index_sha256"]).upper(),
            f"depth index SHA drift: {sequence_id}",
        )
        index_receipts[sequence_id] = {
            "rgb_index_sha256": rgb_sha256,
            "depth_index_sha256": depth_sha256,
            "paired_rgb_depth_identity_count": int(
                source["paired_rgb_depth_identity_count"]
            ),
            "missing_depth_reference_count": int(
                source["missing_depth_reference_count"]
            ),
        }
    provenance = {
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": EXPECTED_ARCHIVE_SHA256,
            "sha256_authority": "BOUND_FROZEN_INTEGRITY_RECEIPT",
            "live_archive_rehash_performed": False,
        },
        "catalog": {
            "path": str(catalog_path),
            "schema": CATALOG_SCHEMA,
            "sha256": catalog_sha256,
        },
        "integrity_receipt": {
            "path": str(receipt_path),
            "schema": RECEIPT_SCHEMA,
            "sha256": receipt_sha256,
        },
        "sequence_index_receipts": index_receipts,
    }
    return catalog, receipt, provenance


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset_root = args.dataset_root.resolve()
    archive_path = args.archive.resolve()
    catalog_path = args.catalog.resolve()
    receipt_path = args.receipt.resolve()
    student_checkpoint_path = args.student_checkpoint.resolve()
    depthart_source = args.depthart_source.resolve()
    depthart_checkpoint = args.depthart_checkpoint.resolve()
    output_path = args.output.resolve()
    cohort_manifest = args.cohort_manifest.resolve() if args.cohort_manifest else None
    frame_indices = (
        load_cohort_indices(cohort_manifest, str(args.cohort_role))
        if cohort_manifest is not None
        else FIXED_FRAME_INDICES_BY_SEQUENCE
    )
    require(student_checkpoint_path.is_file(), "student checkpoint missing")
    require(depthart_source.is_dir(), "DepthART source missing")
    require(depthart_checkpoint.is_file(), "DepthART checkpoint missing")
    require(not output_path.exists(), f"output collision: {output_path}")
    require(torch.cuda.is_available(), "Bonn DepthART evaluation requires CUDA")

    _, receipt, source_provenance = validate_source_receipts(
        dataset_root,
        archive_path,
        catalog_path,
        receipt_path,
        set(frame_indices),
    )
    pairs_by_parent = fixed_frame_pairs(dataset_root, frame_indices)
    receipt_by_parent = {
        str(row["parent_id"]): row for row in receipt.get("parents", [])
    }
    for parent_id, pairs in pairs_by_parent.items():
        all_pairs = pair_rgb_depth_unique(
            parent_id,
            read_tum_index(dataset_root / parent_id, "rgb.txt"),
            read_tum_index(dataset_root / parent_id, "depth.txt"),
        )
        require(
            len(all_pairs)
            == int(receipt_by_parent[parent_id]["paired_rgb_depth_identity_count"]),
            f"paired identity count drift: {parent_id}",
        )
        require(len(pairs) == 3, f"fixed frame count drift: {parent_id}")

    checkpoint = torch.load(
        student_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    require(isinstance(checkpoint, dict), "student checkpoint root must be a mapping")
    architecture = checkpoint_architecture(checkpoint)
    checkpoint_parents = checkpoint_parent_ids(checkpoint)
    evaluation_parents = set(frame_indices)
    overlap = sorted(checkpoint_parents & evaluation_parents)
    require(not overlap, f"checkpoint/Bonn parent overlap: {overlap}")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    extractor, scan = load_depthart_backbone(
        depthart_source,
        depthart_checkpoint,
        device,
        int(checkpoint["seed"]),
    )
    baseline, student = build_students(checkpoint, architecture, device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    parent_rows: list[dict[str, Any]] = []
    frame_receipts: list[dict[str, Any]] = []
    for parent_id, pairs in pairs_by_parent.items():
        baseline_sums: list[dict[str, float | int]] = []
        student_sums: list[dict[str, float | int]] = []
        for pair in pairs:
            rgb = load_rgb_native(pair.rgb.absolute_path)
            # The source-native depth is intentionally not opened until both RGB/K
            # predictions exist.
            initialized_depth, student_depth = infer_rgb_only_depths(
                extractor,
                baseline,
                student,
                rgb,
                architecture["feature_profile"],
                device,
                amp_dtype,
            )
            truth_depth_m, source_valid = load_depth_native(pair.depth.absolute_path)
            initialized_metrics = depth_error_sums(
                truth_depth_m,
                initialized_depth,
                source_valid,
            )
            student_metrics = depth_error_sums(
                truth_depth_m,
                student_depth,
                source_valid,
            )
            baseline_sums.append(initialized_metrics)
            student_sums.append(student_metrics)
            frame_receipts.append(
                {
                    "parent_id": parent_id,
                    "rgb_row_index_zero_based": pair.rgb.row_index,
                    "rgb_timestamp_seconds": pair.rgb.timestamp_seconds,
                    "rgb_relative_path": pair.rgb.relative_path.as_posix(),
                    "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                    "depth_row_index_zero_based": pair.depth.row_index,
                    "depth_timestamp_seconds": pair.depth.timestamp_seconds,
                    "depth_relative_path": pair.depth.relative_path.as_posix(),
                    "depth_sha256": sha256_file(pair.depth.absolute_path),
                    "rgb_depth_delta_seconds": pair.association_delta_seconds,
                    "source_valid_pixel_count": int(source_valid.sum()),
                    "source_valid_fraction": float(source_valid.mean()),
                    "initialized_baseline": finalize_depth_metrics(
                        initialized_metrics
                    ),
                    "student": finalize_depth_metrics(student_metrics),
                }
            )
        parent_rows.append(
            {
                "parent_id": parent_id,
                "frame_count": len(pairs),
                "baseline": finalize_depth_metrics(
                    merge_depth_error_sums(baseline_sums)
                ),
                "student": finalize_depth_metrics(
                    merge_depth_error_sums(student_sums)
                ),
            }
        )

    metrics = parent_macro_metrics(parent_rows)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "DEVELOPMENT_CROSS_DATASET_DEPTH_DIAGNOSTIC_COMPLETE",
        "mode": "DEVELOPMENT",
        "dataset_relation": "CROSS_DATASET",
        "question": (
            "Does the frozen AG-ST masked depth student improve over its checkpoint-"
            "initialized baseline on eight fixed Bonn RGB-D parents using only "
            "source-native valid depth for evaluation?"
        ),
        "provenance": {
            **source_provenance,
            "dataset_root": str(dataset_root),
            "student_checkpoint": {
                "path": str(student_checkpoint_path),
                "sha256": sha256_file(student_checkpoint_path),
                "schema": CHECKPOINT_SCHEMA,
            },
            "depthart": {
                "source_path": str(depthart_source),
                "checkpoint_path": str(depthart_checkpoint),
                "checkpoint_sha256": sha256_file(depthart_checkpoint),
                "extractor": "DepthArtDenseFeatureExtractor",
                "scan_backend": scan,
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "cohort_manifest": (
                {
                    "path": str(cohort_manifest),
                    "sha256": sha256_file(cohort_manifest),
                    "role": str(args.cohort_role),
                }
                if cohort_manifest is not None
                else None
            ),
        },
        "cohort": {
            "parent_ids": list(frame_indices),
            "parent_count": len(frame_indices),
            "frames_per_parent": 3,
            "frame_indices_zero_based": {
                parent: list(indices)
                for parent, indices in frame_indices.items()
            },
            "frame_receipts": frame_receipts,
        },
        "parent_firewall": {
            "checkpoint_consumed_parents": sorted(checkpoint_parents),
            "evaluation_parents": sorted(evaluation_parents),
            "overlap": overlap,
            "evaluation_data_used_for_checkpoint_fitting_or_selection": False,
            "freshness_scope": "FRESH_RELATIVE_TO_AG_ST_CHECKPOINT_ONLY",
        },
        "inference_contract": {
            "model_inputs": ["RGB", "FIXED_BONN_INTRINSICS"],
            "source_depth_is_model_input": False,
            "source_depth_opened_after_predictions": True,
            "pose_is_model_input": False,
            "intrinsics_fx_fy_cx_cy": [
                float(BONN_INTRINSICS[0, 0]),
                float(BONN_INTRINSICS[1, 1]),
                float(BONN_INTRINSICS[0, 2]),
                float(BONN_INTRINSICS[1, 2]),
            ],
            "native_resolution_wh": [BONN_WIDTH, BONN_HEIGHT],
            "depth_scale_divisor": BONN_DEPTH_SCALE,
            "source_validity": "uint16_depth_greater_than_zero",
            "rgb_depth_pairing": (
                "timestamp-nearest greedy one-to-one used-set; maximum 0.05 seconds"
            ),
        },
        "architecture": architecture,
        "metrics": {
            "per_parent": parent_rows,
            "parent_macro": metrics,
            "bad_depth_definition": "absolute_depth_error_m > 0.10",
            "denominator": "source-native uint16 depth > 0 only",
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "total_seconds": time.perf_counter() - started,
            "training_performed": False,
            "optimizer_constructed": False,
            "threshold_selected": False,
        },
        "claim_boundary": {
            "authority": "DEVELOPMENT_CROSS_DATASET_DEPTH_DIAGNOSTIC_ONLY",
            "depth_comparison_only": True,
            "support_evaluated": False,
            "support_claim_authorized": False,
            "boundary_or_obstacle_evaluated": False,
            "task_outcome_read": False,
            "task_utility_claim_authorized": False,
            "deployment_product_or_safety_claim_authorized": False,
            "license_review_performed_by_evaluator": False,
            "license_conclusion": "NOT_MADE_BY_THIS_EVALUATOR",
            "foundation_model_training_ancestry_conclusion": "NOT_MADE",
        },
    }
    write_json_exclusive(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "parent_macro": metrics,
                "claim_boundary": result["claim_boundary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument(
        "--cohort-role",
        choices=("fit", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint",
        type=Path,
        default=DEFAULT_DEPTHART_CHECKPOINT,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
