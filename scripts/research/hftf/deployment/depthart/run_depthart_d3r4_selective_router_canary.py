#!/usr/bin/env python3
"""Train/evaluate the D3R4 selective-horizon release/veto router canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d3r3_phase_b_source_truth import (
    load_json,
    member_map,
    parse_pincam_payload,
    timestamp_from_stem,
)
from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import (
    install_timm_compat,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
STATE_UNKNOWN = -1
STATE_CLEAR = 0
STATE_OCCUPIED = 1
FEATURE_ORDER = (
    "baseline_clear",
    "baseline_occupied",
    "baseline_unknown",
    "baseline_clearance_present",
    "baseline_clearance_m_or_zero",
    "valid_depth_fraction",
    "ground_support_fraction",
    "ground_residual_m_or_zero",
    "log1p_band_support_points",
    "log1p_band_intrusion_points",
    "intrusion_proximity_within_horizon",
    "observed_forward_fraction_through_horizon",
    "observed_forward_normalized_6m",
    "horizon_normalized_2m",
    "band_left",
    "band_center",
)
THRESHOLD_CANDIDATES = (0.5, 0.6, 0.7, 0.8, 0.9)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def truth_state(band: dict[str, Any] | None, horizon: float) -> int:
    value = (band or {}).get("occupied_by_horizon", {}).get(str(horizon))
    if value is True:
        return STATE_OCCUPIED
    if value is False:
        return STATE_CLEAR
    return STATE_UNKNOWN


def baseline_state(band: dict[str, Any] | None, horizon: float) -> int:
    return truth_state(band, horizon)


def candidate_features(
    geometry: dict[str, Any], band_name: str, horizon: float
) -> tuple[list[float], int, bool]:
    band = geometry.get("bands", {}).get(band_name) or {}
    state = baseline_state(band, horizon)
    clearance = band.get("clearance_m")
    clearance_present = clearance is not None and math.isfinite(float(clearance))
    clearance_value = float(clearance) if clearance_present else 0.0
    plane = geometry.get("ground_plane") or {}
    valid_fraction = float(geometry.get("valid_depth_fraction", 0.0))
    ground_support = float(plane.get("support_fraction", 0.0))
    ground_residual = float(plane.get("median_residual_m", 0.0))
    support_points = int(band.get("support_points", 0))
    intrusion_points = int(band.get("intrusion_points", 0))
    observed_forward = float(band.get("observed_forward_m", 0.0))
    proximity = max(0.0, min(1.0, 1.0 - clearance_value / horizon)) if clearance_present else 0.0
    features = [
        float(state == STATE_CLEAR),
        float(state == STATE_OCCUPIED),
        float(state == STATE_UNKNOWN),
        float(clearance_present),
        clearance_value,
        valid_fraction,
        ground_support,
        ground_residual,
        math.log1p(support_points),
        math.log1p(intrusion_points),
        proximity,
        min(max(observed_forward / horizon, 0.0), 1.0),
        min(max(observed_forward / 6.0, 0.0), 1.0),
        horizon / 2.0,
        float(band_name == "left"),
        float(band_name == "center"),
    ]
    require(len(features) == 16 and all(math.isfinite(value) for value in features), "invalid candidate features")
    hard_evidence = bool(
        geometry.get("ground_plane") is not None
        and valid_fraction >= 0.05
        and ground_support >= 0.02
        and support_points >= 20
    )
    return features, state, hard_evidence


def source_truth_rows(
    base: dict[str, Any], extension: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    rows = list(base["processed"]) + list(extension["processed_extension"])
    result = {str(row["video_id"]): row for row in rows}
    require(len(result) == 53, "source-truth lookup drift")
    return result


def role_plan(
    phase_a_manifest: dict[str, Any], rgb_result: dict[str, Any],
    selective: dict[str, Any], base_truth: dict[str, Any], extension_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    phase_a = {
        str(row["video_id"]): row
        for row in phase_a_manifest["processed"] if row["eligible"] is True
    }
    rgb = {str(row["video_id"]): row for row in rgb_result["processed"]}
    truth = source_truth_rows(base_truth, extension_truth)
    result: list[dict[str, Any]] = []
    for role in ("TRAIN", "DEVELOPMENT"):
        for identity in selective["candidate_role_split"][role]["identities"]:
            video_id = str(identity["video_id"])
            checkpoint = phase_a[video_id]
            rgb_row = rgb[video_id]
            truth_row = truth[video_id]
            stems = list(checkpoint["selected_frame_stems"])
            require(len(stems) == 300, "selected stem count drift")
            require(rgb_row["role"] == role, "RGB role drift")
            result.append({
                "role": role,
                "role_order": int(identity["role_order"]),
                "pool_order": int(identity["pool_order"]),
                "visit_id": str(identity["visit_id"]),
                "video_id": video_id,
                "selected_frame_stems": stems,
                "unavailable_stems": list(rgb_row["effective_multimodal_unavailable_stems"]),
                "truth_row": truth_row,
            })
    require(len(result) == 16 and len({row["video_id"] for row in result}) == 16, "role roster drift")
    return result


def _source_roots(video_id: str, base_root: Path, extension_root: Path) -> tuple[Path, Path]:
    base = base_root / "source" / "Training" / video_id
    extension = extension_root / "source" / "Training" / video_id
    depth_root = base if base.is_dir() else extension
    require(depth_root.is_dir(), f"depth/conf source missing: {video_id}")
    return depth_root, depth_root


def _load_depthart(source_root: Path, checkpoint: Path) -> tuple[nn.Module, Any]:
    install_timm_compat()
    sys.path.insert(0, str(source_root / "metric"))
    sys.path.insert(0, str(source_root / "deploy" / "shared"))
    sys.path.insert(0, str(source_root / "deploy" / "shared" / "selective_scan"))
    from common import preprocess  # type: ignore
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    model = load_model(checkpoint, "S", "indoor", "cuda").eval()
    install_depthart(tvimblock)
    return model, preprocess


def _append_unknown_frame(
    arrays: dict[str, list[Any]], parent_index: int, frame_index: int
) -> None:
    for band_index in range(3):
        for horizon_index in range(3):
            arrays["features"].append([0.0] * 16)
            arrays["truth_state"].append(STATE_UNKNOWN)
            arrays["baseline_state"].append(STATE_UNKNOWN)
            arrays["hard_evidence"].append(False)
            arrays["source_available"].append(False)
            arrays["parent_index"].append(parent_index)
            arrays["frame_index"].append(frame_index)
            arrays["band_index"].append(band_index)
            arrays["horizon_index"].append(horizon_index)


def materialize_role(
    role: str,
    identities: list[dict[str, Any]],
    *,
    phase_a_root: Path,
    base_depth_root: Path,
    extension_depth_root: Path,
    rgb_root: Path,
    model: nn.Module,
    preprocess: Any,
    batch_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, list[Any]] = {
        key: [] for key in (
            "features", "truth_state", "baseline_state", "hard_evidence",
            "source_available", "parent_index", "frame_index", "band_index", "horizon_index",
        )
    }
    policy = TruthReaderPolicy()
    policy.validate()
    identity_summaries: list[dict[str, Any]] = []
    for parent_index, identity in enumerate(identities):
        video_id = identity["video_id"]
        stems = identity["selected_frame_stems"]
        unavailable = set(identity["unavailable_stems"])
        phase_a = phase_a_root / "source" / "Training" / video_id
        depth_root, _ = _source_roots(video_id, base_depth_root, extension_depth_root)
        rgb_path = rgb_root / "source" / "Training" / video_id / "lowres_wide.zip"
        trajectory = parse_trajectory(phase_a / "lowres_wide.traj")
        available_frame_count = 0
        with (
            zipfile.ZipFile(rgb_path) as rgb_zip,
            zipfile.ZipFile(phase_a / "lowres_wide_intrinsics.zip") as intrinsics_zip,
            zipfile.ZipFile(depth_root / "lowres_depth.zip") as depth_zip,
            zipfile.ZipFile(depth_root / "confidence.zip") as confidence_zip,
        ):
            rgb_map = member_map(rgb_zip, ".png")
            intrinsics_map = member_map(intrinsics_zip, ".pincam")
            depth_map = member_map(depth_zip, ".png")
            confidence_map = member_map(confidence_zip, ".png")
            for start in range(0, 300, batch_size):
                pending: list[dict[str, Any]] = []
                for frame_index in range(start, min(start + batch_size, 300)):
                    stem = stems[frame_index]
                    if stem in unavailable:
                        _append_unknown_frame(arrays, parent_index, frame_index)
                        continue
                    require(
                        stem in rgb_map and stem in intrinsics_map
                        and stem in depth_map and stem in confidence_map,
                        f"effective source coverage drift: {video_id}/{stem}",
                    )
                    rgb = np.asarray(Image.open(rgb_zip.open(rgb_map[stem])).convert("RGB"))
                    depth_raw = np.asarray(Image.open(depth_zip.open(depth_map[stem]))).copy()
                    confidence = np.asarray(Image.open(confidence_zip.open(confidence_map[stem]))).copy()
                    intrinsics, _ = parse_pincam_payload(
                        intrinsics_zip.read(intrinsics_map[stem]), intrinsics_map[stem]
                    )
                    pose, _ = interpolate_camera_to_world(
                        trajectory, timestamp_from_stem(stem),
                        policy.maximum_pose_bracketing_gap_seconds,
                    )
                    canonical = canonicalize_frame(rgb, depth_raw, confidence, intrinsics, pose)
                    require(canonical["rotation_index"] in (1, 3), "portrait orientation drift")
                    up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
                    sensor_truth = derive_assistive_truth(
                        depth_mm_to_metres(canonical["depth_raw_mm"]),
                        canonical["confidence"],
                        canonical["intrinsics"],
                        up_camera,
                        policy,
                    )
                    bgr = cv2.cvtColor(canonical["rgb"], cv2.COLOR_RGB2BGR)
                    image, intrinsics_tensor = preprocess(
                        bgr,
                        np.asarray(canonical["intrinsics"], dtype=np.float32),
                        448,
                        448,
                    )
                    pending.append({
                        "frame_index": frame_index,
                        "image": image,
                        "intrinsics": intrinsics_tensor,
                        "up_camera": up_camera,
                        "truth": sensor_truth,
                    })
                if not pending:
                    continue
                images = torch.cat([row["image"] for row in pending], dim=0).cuda()
                intrinsics_tensors = torch.cat([row["intrinsics"] for row in pending], dim=0).cuda()
                with torch.inference_mode():
                    predicted = model(images, intrinsics_tensors).detach().float().cpu().numpy()
                require(predicted.shape == (len(pending), 608, 448), "DepthART output shape drift")
                require(np.all(np.isfinite(predicted)), "DepthART non-finite output")
                for item, depth_m, intrinsics_tensor in zip(
                    pending, predicted, intrinsics_tensors.detach().cpu().numpy(), strict=True
                ):
                    geometry = derive_assistive_truth(
                        depth_m,
                        np.full(depth_m.shape, 2, dtype=np.uint8),
                        intrinsics_tensor,
                        np.asarray(item["up_camera"], dtype=np.float64),
                        policy,
                    )
                    for band_index, band_name in enumerate(BANDS):
                        truth_band = item["truth"].get("bands", {}).get(band_name)
                        for horizon_index, horizon in enumerate(HORIZONS):
                            features, base_state, hard = candidate_features(
                                geometry, band_name, horizon
                            )
                            arrays["features"].append(features)
                            arrays["truth_state"].append(truth_state(truth_band, horizon))
                            arrays["baseline_state"].append(base_state)
                            arrays["hard_evidence"].append(hard)
                            arrays["source_available"].append(True)
                            arrays["parent_index"].append(parent_index)
                            arrays["frame_index"].append(item["frame_index"])
                            arrays["band_index"].append(band_index)
                            arrays["horizon_index"].append(horizon_index)
                    available_frame_count += 1
        require(available_frame_count == 300 - len(unavailable), "available frame count drift")
        identity_summaries.append({
            "parent_index": parent_index,
            "visit_id": identity["visit_id"],
            "video_id": video_id,
            "available_frame_count": available_frame_count,
            "source_unavailable_frame_count": len(unavailable),
        })
        print(json.dumps({
            "stage": "MATERIALIZE_DEPTHART_FEATURES",
            "role": role,
            "completed": parent_index + 1,
            "total": len(identities),
            "video_id": video_id,
            "available": available_frame_count,
        }, sort_keys=True), flush=True)
    result = {
        "features": np.asarray(arrays["features"], dtype=np.float64),
        "truth_state": np.asarray(arrays["truth_state"], dtype=np.int8),
        "baseline_state": np.asarray(arrays["baseline_state"], dtype=np.int8),
        "hard_evidence": np.asarray(arrays["hard_evidence"], dtype=bool),
        "source_available": np.asarray(arrays["source_available"], dtype=bool),
        "parent_index": np.asarray(arrays["parent_index"], dtype=np.int16),
        "frame_index": np.asarray(arrays["frame_index"], dtype=np.int16),
        "band_index": np.asarray(arrays["band_index"], dtype=np.int8),
        "horizon_index": np.asarray(arrays["horizon_index"], dtype=np.int8),
    }
    require(result["features"].shape == (len(identities) * 300 * 9, 16), "dataset shape drift")
    return result, {
        "role": role,
        "identity_count": len(identities),
        "cell_count": len(result["truth_state"]),
        "truth_known_cell_count": int(np.sum(result["truth_state"] >= 0)),
        "source_unavailable_cell_count": int(np.sum(~result["source_available"])),
        "identities": identity_summaries,
    }


class CertificateHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)[:, 0]


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum()
    negative = labels.numel() - positive
    require(float(positive) > 0 and float(negative) > 0, "balanced BCE requires both classes")
    weights = torch.where(
        labels > 0.5,
        labels.numel() / (2.0 * positive),
        labels.numel() / (2.0 * negative),
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=weights
    )


def train_heads(
    train: dict[str, np.ndarray], steps: int = 1000, seed: int = 41
) -> tuple[CertificateHead, CertificateHead, dict[str, Any], np.ndarray, np.ndarray]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    features = torch.as_tensor(train["features"], dtype=torch.float64)
    valid_features = torch.as_tensor(train["source_available"], dtype=torch.bool)
    mean = features[valid_features].mean(dim=0)
    std = features[valid_features].std(dim=0, unbiased=False)
    normalized = (features - mean) / (std + 1e-6)
    truth = torch.as_tensor(train["truth_state"], dtype=torch.int64)
    baseline = torch.as_tensor(train["baseline_state"], dtype=torch.int64)
    hard = torch.as_tensor(train["hard_evidence"], dtype=torch.bool)
    horizon = torch.as_tensor(train["horizon_index"], dtype=torch.int64)
    known = truth >= 0
    release_mask = known & hard & (horizon == 0) & (baseline != STATE_CLEAR)
    veto_mask = known & hard & (baseline != STATE_OCCUPIED)
    release_labels = (truth[release_mask] == STATE_CLEAR).to(torch.float64)
    veto_labels = (truth[veto_mask] == STATE_OCCUPIED).to(torch.float64)
    require(int(release_mask.sum()) > 0 and int(veto_mask.sum()) > 0, "empty training mask")
    release = CertificateHead().to(dtype=torch.float64)
    veto = CertificateHead().to(dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        list(release.parameters()) + list(veto.parameters()),
        lr=0.005,
        weight_decay=0.0001,
    )
    initial: dict[str, float] | None = None
    final: dict[str, float] = {}
    for step in range(steps + 1):
        release_logits = release(normalized[release_mask])
        veto_logits = veto(normalized[veto_mask])
        release_loss = balanced_bce(release_logits, release_labels)
        veto_loss = balanced_bce(veto_logits, veto_labels)
        contradiction_mask = known & hard & (horizon == 0)
        contradiction = (
            torch.sigmoid(release(normalized[contradiction_mask]))
            * torch.sigmoid(veto(normalized[contradiction_mask]))
        ).mean()
        loss = release_loss + veto_loss + 0.1 * contradiction
        metrics = {
            "total": float(loss.detach()),
            "release": float(release_loss.detach()),
            "veto": float(veto_loss.detach()),
            "contradiction": float(contradiction.detach()),
        }
        if step == 0:
            initial = metrics
        if step == steps:
            final = metrics
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    require(initial is not None, "initial training metrics missing")
    training = {
        "seed": seed,
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "release_training_rows": int(release_mask.sum()),
        "release_positive_rows": int(release_labels.sum()),
        "veto_training_rows": int(veto_mask.sum()),
        "veto_positive_rows": int(veto_labels.sum()),
        "initial_loss": initial,
        "final_loss": final,
    }
    return release, veto, training, mean.numpy(), std.numpy()


def predict_probabilities(
    dataset: dict[str, np.ndarray], release: CertificateHead, veto: CertificateHead,
    mean: np.ndarray, std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.as_tensor(
        (dataset["features"] - mean) / (std + 1e-6), dtype=torch.float64
    )
    with torch.inference_mode():
        release_prob = torch.sigmoid(release(features)).numpy()
        veto_prob = torch.sigmoid(veto(features)).numpy()
    return release_prob, veto_prob


def route_states(
    dataset: dict[str, np.ndarray], release_prob: np.ndarray, veto_prob: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    opposite = 1.0 - threshold
    baseline = dataset["baseline_state"].copy()
    hard = dataset["hard_evidence"]
    horizon = dataset["horizon_index"]
    parent = dataset["parent_index"]
    frame = dataset["frame_index"]
    band = dataset["band_index"]
    result = baseline.copy()
    release_enabled = horizon == 0
    contradiction = hard & release_enabled & (release_prob >= threshold) & (veto_prob >= threshold)
    release_action = hard & release_enabled & (release_prob >= threshold) & (veto_prob <= opposite)
    veto_action = hard & (veto_prob >= threshold) & (
        (~release_enabled) | (release_prob <= opposite)
    )
    result[contradiction] = STATE_UNKNOWN
    result[release_action] = STATE_CLEAR
    result[veto_action] = STATE_OCCUPIED
    projected = 0
    # Enforce horizon safety independently for every parent/frame/band triplet.
    for parent_index in np.unique(parent):
        for frame_index in np.unique(frame[parent == parent_index]):
            base_mask = (parent == parent_index) & (frame == frame_index)
            for band_index in range(3):
                indices = np.flatnonzero(base_mask & (band == band_index))
                if len(indices) != 3:
                    continue
                indices = indices[np.argsort(horizon[indices])]
                blocked = False
                for index in indices:
                    if result[index] in (STATE_OCCUPIED, STATE_UNKNOWN):
                        blocked = True
                    elif blocked and result[index] == STATE_CLEAR:
                        result[index] = STATE_UNKNOWN
                        projected += 1
    return result, {
        "release_actions": int(np.sum(release_action)),
        "veto_actions": int(np.sum(veto_action)),
        "contradictions": int(np.sum(contradiction)),
        "projection_to_unknown": projected,
    }


def metrics(dataset: dict[str, np.ndarray], states: np.ndarray) -> dict[str, Any]:
    truth = dataset["truth_state"]
    known_truth = truth >= 0
    clear_truth = truth == STATE_CLEAR
    occupied_truth = truth == STATE_OCCUPIED
    known_pred = states >= 0
    require(int(np.sum(known_truth)) > 0 and int(np.sum(clear_truth)) > 0, "metric denominator empty")

    def metric_block(mask: np.ndarray) -> dict[str, Any]:
        local_known = known_truth & mask
        local_clear = clear_truth & mask
        local_occupied = occupied_truth & mask
        return {
            "cell_count": int(np.sum(mask)),
            "truth_known_count": int(np.sum(local_known)),
            "truth_clear_count": int(np.sum(local_clear)),
            "truth_occupied_count": int(np.sum(local_occupied)),
            "known_coverage_all_cells": float(np.sum(known_pred & mask) / max(int(np.sum(mask)), 1)),
            "false_clear_all_known": float(
                np.sum((states == STATE_CLEAR) & local_occupied) / max(int(np.sum(local_known)), 1)
            ),
            "false_clear_given_occupied": float(
                np.sum((states == STATE_CLEAR) & local_occupied) / max(int(np.sum(local_occupied)), 1)
            ),
            "false_block_given_clear": float(
                np.sum((states == STATE_OCCUPIED) & local_clear) / max(int(np.sum(local_clear)), 1)
            ),
            "unknown_given_known": float(
                np.sum((states == STATE_UNKNOWN) & local_known) / max(int(np.sum(local_known)), 1)
            ),
        }

    result = {"pooled": metric_block(np.ones(len(truth), dtype=bool))}
    result["by_horizon"] = {
        f"{horizon:.1f}m": metric_block(dataset["horizon_index"] == index)
        for index, horizon in enumerate(HORIZONS)
    }
    parent_rows = []
    for parent_index in sorted(np.unique(dataset["parent_index"])):
        block = metric_block(dataset["parent_index"] == parent_index)
        block["parent_index"] = int(parent_index)
        parent_rows.append(block)
    result["by_parent"] = parent_rows
    return result


def threshold_score(baseline: dict[str, Any], candidate: dict[str, Any]) -> float:
    base = baseline["pooled"]
    cand = candidate["pooled"]
    coverage_drop = max(0.0, base["known_coverage_all_cells"] - cand["known_coverage_all_cells"])
    return (
        cand["false_clear_all_known"]
        + cand["false_block_given_clear"]
        + 2.0 * max(0.0, coverage_drop - 0.02)
        + 0.25 * cand["unknown_given_known"]
    )


def serialize_head(head: CertificateHead) -> dict[str, Any]:
    return {
        name: value.detach().cpu().numpy().tolist()
        for name, value in head.state_dict().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--base-source-truth-result", type=Path, required=True)
    parser.add_argument("--extension-result", type=Path, required=True)
    parser.add_argument("--selective-canary", type=Path, required=True)
    parser.add_argument("--rgb-result", type=Path, required=True)
    parser.add_argument("--depthart-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-depth-root", type=Path, required=True)
    parser.add_argument("--extension-depth-root", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"fresh output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    phase_a = load_json(args.phase_a_manifest)
    base_truth = load_json(args.base_source_truth_result)
    extension_truth = load_json(args.extension_result)
    selective = load_json(args.selective_canary)
    rgb = load_json(args.rgb_result)
    require(selective.get("status") == "D3R4_SELECTIVE_HORIZON_SOURCE_SUPPORT_PASS", "selective canary drift")
    require(rgb.get("status") == "D3R4_RGB_SOURCE_PASS", "RGB source drift")
    plan = role_plan(phase_a, rgb, selective, base_truth, extension_truth)
    require(args.checkpoint.is_file(), "DepthART checkpoint missing")
    require(
        sha256_file(args.checkpoint)
        == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65",
        "DepthART checkpoint drift",
    )
    model, preprocess = _load_depthart(args.depthart_source, args.checkpoint)
    datasets: dict[str, dict[str, np.ndarray]] = {}
    dataset_meta: dict[str, Any] = {}
    started = time.time()
    for role in ("TRAIN", "DEVELOPMENT"):
        role_identities = [row for row in plan if row["role"] == role]
        dataset, metadata = materialize_role(
            role,
            role_identities,
            phase_a_root=args.phase_a_manifest.parent,
            base_depth_root=args.base_depth_root,
            extension_depth_root=args.extension_depth_root,
            rgb_root=args.rgb_root,
            model=model,
            preprocess=preprocess,
            batch_size=args.batch_size,
        )
        datasets[role] = dataset
        dataset_meta[role] = metadata
        np.savez_compressed(args.output_root / f"{role.lower()}-dataset.npz", **dataset)
    del model
    torch.cuda.empty_cache()

    # Development arrays are not passed into training or threshold selection.
    release, veto, training, mean, std = train_heads(datasets["TRAIN"])
    train_release, train_veto = predict_probabilities(
        datasets["TRAIN"], release, veto, mean, std
    )
    train_baseline = metrics(datasets["TRAIN"], datasets["TRAIN"]["baseline_state"])
    threshold_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_CANDIDATES:
        states, actions = route_states(
            datasets["TRAIN"], train_release, train_veto, threshold
        )
        candidate = metrics(datasets["TRAIN"], states)
        threshold_rows.append({
            "threshold": threshold,
            "opposite_max": 1.0 - threshold,
            "score": threshold_score(train_baseline, candidate),
            "metrics": candidate,
            "actions": actions,
        })
    selected_threshold_row = min(
        threshold_rows, key=lambda row: (row["score"], -row["threshold"])
    )
    selected_threshold = float(selected_threshold_row["threshold"])
    checkpoint = {
        "schema": "blindassist_depthart_d3r4_selective_router_checkpoint_v1",
        "feature_order": list(FEATURE_ORDER),
        "architecture": "two independent Linear(16,16)-SiLU-Linear(16,1) heads",
        "release_enabled_horizons_m": [1.0],
        "veto_enabled_horizons_m": [1.0, 1.5, 2.0],
        "far_clear_action": "KEEP_BASELINE_OR_UNKNOWN_NEVER_RELEASE",
        "threshold": selected_threshold,
        "opposite_certificate_max": 1.0 - selected_threshold,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "release_head": serialize_head(release),
        "veto_head": serialize_head(veto),
        "training": training,
        "threshold_selection_role": "TRAIN_ONLY",
    }
    checkpoint_path = args.output_root / "router-checkpoint.json"
    atomic_json(checkpoint_path, checkpoint)

    # Only after the complete checkpoint/threshold is frozen do we open DEVELOPMENT arrays.
    dev_release, dev_veto = predict_probabilities(
        datasets["DEVELOPMENT"], release, veto, mean, std
    )
    dev_states, dev_actions = route_states(
        datasets["DEVELOPMENT"], dev_release, dev_veto, selected_threshold
    )
    dev_baseline = metrics(
        datasets["DEVELOPMENT"], datasets["DEVELOPMENT"]["baseline_state"]
    )
    dev_candidate = metrics(datasets["DEVELOPMENT"], dev_states)
    base = dev_baseline["pooled"]
    candidate = dev_candidate["pooled"]
    false_clear_improvement = base["false_clear_all_known"] - candidate["false_clear_all_known"]
    false_block_improvement = base["false_block_given_clear"] - candidate["false_block_given_clear"]
    coverage_decrease = base["known_coverage_all_cells"] - candidate["known_coverage_all_cells"]
    mechanism_supported = bool(
        dev_actions["release_actions"] + dev_actions["veto_actions"] > 0
        and false_clear_improvement >= 0.01
        and false_block_improvement >= -0.01
        and coverage_decrease <= 0.02
    )
    result = {
        "schema": "blindassist_depthart_d3r4_selective_router_canary_result_v1",
        "status": (
            "D3R4_SELECTIVE_ROUTER_MECHANISM_SUPPORTED"
            if mechanism_supported
            else "D3R4_SELECTIVE_ROUTER_MECHANISM_NOT_SUPPORTED"
        ),
        "problem": "D3R3 all-horizon bidirectional learning was not observable because far CLEAR was parent-concentrated.",
        "hypothesis": "Near-only CLEAR release plus all-horizon OCCUPIED veto reduces false-clear without materially increasing false-block or UNKNOWN.",
        "dataset": dataset_meta,
        "feature_order": list(FEATURE_ORDER),
        "truth_reader_policy": asdict(TruthReaderPolicy()),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "parameter_count": sum(parameter.numel() for parameter in release.parameters())
            + sum(parameter.numel() for parameter in veto.parameters()),
        },
        "training": training,
        "train_threshold_search": threshold_rows,
        "selected_threshold": selected_threshold,
        "development": {
            "baseline": dev_baseline,
            "candidate": dev_candidate,
            "actions": dev_actions,
            "false_clear_all_known_improvement": false_clear_improvement,
            "false_block_given_clear_improvement": false_block_improvement,
            "known_coverage_decrease": coverage_decrease,
        },
        "decision_rule": {
            "false_clear_improvement_min": 0.01,
            "false_block_improvement_min": -0.01,
            "known_coverage_decrease_max": 0.02,
        },
        "mechanism_supported": mechanism_supported,
        "source_unavailable_as_negative": False,
        "far_clear_as_negative": False,
        "development_used_for_training_or_threshold": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "elapsed_seconds_diagnostic_only": time.time() - started,
        "next_action": (
            "REFINE_AND_CONFIRM_D3R4_SELECTIVE_ROUTER_ON_NEW_IDENTITY_DISJOINT_DATA"
            if mechanism_supported
            else "ANALYZE_D3R4_FAILURE_AND_REDEFINE_FEATURE_OR_ACTION_MECHANISM"
        ),
    }
    atomic_json(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "threshold": selected_threshold,
        "baseline_false_clear": base["false_clear_all_known"],
        "candidate_false_clear": candidate["false_clear_all_known"],
        "baseline_false_block": base["false_block_given_clear"],
        "candidate_false_block": candidate["false_block_given_clear"],
        "known_coverage_decrease": coverage_decrease,
        "actions": dev_actions,
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
