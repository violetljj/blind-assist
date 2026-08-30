#!/usr/bin/env python3
"""Reference-conditioned RGB endpoint-coordinate field on registered 3RScan doors.

Geometry and depth select a fresh, fully visible reference/rescan pair before RGB
is decoded.  During replay, registered geometry supervises only the reference
image.  The query prediction reads RGB plus frozen DINOv2 features; query pose,
depth, instance geometry, and the scan transform remain evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from plyfile import PlyData
from torch import nn
from transformers import AutoModel


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-reference-pixel-field-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-reference-pixel-field-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-reference-pixel-field-result-v1"
PREFLIGHT_SCHEMA = "blindassist-l10-3rscan-reference-pixel-field-preflight-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def protocol_paths(protocol: dict[str, Any], artifact_root: Path) -> tuple[Path, Path]:
    expected_drive = str(protocol["storage"]["artifact_root_argument_must_resolve_to_drive"]).upper()
    resolved = artifact_root.resolve()
    require(resolved.drive.upper() == expected_drive, f"ARTIFACT_ROOT_DRIVE:{resolved}")
    data_root = resolved / protocol["storage"]["dataset_relative_path"]
    model_root = resolved / protocol["storage"]["model_relative_path"]
    require(data_root.is_dir(), f"DATA_ROOT_MISSING:{data_root}")
    require(model_root.is_dir(), f"MODEL_ROOT_MISSING:{model_root}")
    return data_root, model_root


def verify_predecessor(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor = protocol["predecessor"]
    entrypoint = HERE / predecessor["entrypoint"]
    cohort_path = HERE / predecessor["cohort"]
    result_path = HERE / predecessor["result"]
    for path, expected in (
        (entrypoint, predecessor["entrypoint_sha256"]),
        (cohort_path, predecessor["cohort_sha256"]),
        (result_path, predecessor["result_sha256"]),
    ):
        require(path.is_file(), f"PREDECESSOR_MISSING:{path}")
        require(sha256(path) == expected, f"PREDECESSOR_HASH:{path.name}")
    result = load_json(result_path)
    require(result.get("status") == predecessor["required_status"], "PREDECESSOR_STATUS")
    return load_json(cohort_path), result


def parse_info(text: str) -> dict[str, Any]:
    def value(key: str) -> str:
        match = re.search(rf"^{re.escape(key)} = (.+)$", text, flags=re.MULTILINE)
        require(match is not None, f"SEQUENCE_INFO_MISSING:{key}")
        return str(match.group(1)).strip()

    def intrinsic(key: str) -> np.ndarray:
        values = [float(item) for item in value(key).split()]
        require(len(values) == 16, f"INTRINSIC_LENGTH:{key}:{len(values)}")
        return np.asarray(values, dtype=np.float64).reshape(4, 4)[:3, :3]

    return {
        "color_width": int(value("m_colorWidth")),
        "color_height": int(value("m_colorHeight")),
        "depth_width": int(value("m_depthWidth")),
        "depth_height": int(value("m_depthHeight")),
        "color_intrinsic": intrinsic("m_calibrationColorIntrinsic"),
        "depth_intrinsic": intrinsic("m_calibrationDepthIntrinsic"),
        "frames": int(value("m_frames.size")),
    }


def read_pose(archive: zipfile.ZipFile, frame: int) -> np.ndarray:
    name = f"frame-{frame:06d}.pose.txt"
    require(name in archive.namelist(), f"POSE_MISSING:{name}")
    pose = np.loadtxt(io.BytesIO(archive.read(name)), dtype=np.float64)
    require(pose.shape == (4, 4) and np.isfinite(pose).all(), f"POSE_INVALID:{name}")
    require(np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5), f"POSE_AFFINE:{name}")
    return pose


def pose_frames(archive: zipfile.ZipFile) -> list[int]:
    frames = []
    for name in archive.namelist():
        match = re.fullmatch(r"frame-(\d{6})\.pose\.txt", name)
        if match is not None:
            frames.append(int(match.group(1)))
    return sorted(frames)


def decode_depth(archive: zipfile.ZipFile, frame: int) -> np.ndarray:
    name = f"frame-{frame:06d}.depth.pgm"
    require(name in archive.namelist(), f"DEPTH_MISSING:{name}")
    depth = cv2.imdecode(np.frombuffer(archive.read(name), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    require(depth is not None and depth.dtype == np.uint16, f"DEPTH_INVALID:{name}")
    return depth


def decode_rgb(archive: zipfile.ZipFile, frame: int) -> tuple[np.ndarray, str]:
    name = f"frame-{frame:06d}.color.jpg"
    require(name in archive.namelist(), f"RGB_MISSING:{name}")
    payload = archive.read(name)
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(bgr is not None and bgr.ndim == 3, f"RGB_INVALID:{name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hashlib.sha256(payload).hexdigest()


def project_points(
    points: np.ndarray,
    camera_to_scan: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(camera_to_scan)
    camera = np.column_stack((points, np.ones(len(points), dtype=np.float64))) @ inverse.T
    xyz = camera[:, :3]
    positive = xyz[:, 2] > 0.05
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected = xyz[positive] @ intrinsic.T
    pixels[positive] = projected[:, :2] / projected[:, 2:3]
    inside = (
        positive
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return xyz, pixels, inside


def projected_hull(
    points: np.ndarray,
    camera_to_scan: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _, pixels, inside = project_points(points, camera_to_scan, intrinsic, width, height)
    require(np.count_nonzero(inside) >= 4, "PROJECTED_HULL_TOO_FEW")
    hull = cv2.convexHull(pixels[inside].astype(np.float32)).reshape(-1, 2)
    x0, y0 = np.min(hull, axis=0)
    x1, y1 = np.max(hull, axis=0)
    return hull, {
        "inside_vertices": int(np.count_nonzero(inside)),
        "inside_vertex_fraction": float(np.count_nonzero(inside) / len(points)),
        "projected_area_pixels": float(cv2.contourArea(hull)),
        "image_margin_pixels": float(min(x0, y0, width - 1 - x1, height - 1 - y1)),
        "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
    }


def frame_visibility(
    points: np.ndarray,
    pose: np.ndarray,
    info: dict[str, Any],
    depth: np.ndarray,
    tolerance_metres: float,
) -> dict[str, Any]:
    require(
        depth.shape == (info["depth_height"], info["depth_width"]),
        f"DEPTH_SHAPE:{depth.shape}",
    )
    _, color_stats = projected_hull(
        points,
        pose,
        info["color_intrinsic"],
        info["color_width"],
        info["color_height"],
    )
    camera, pixels, inside = project_points(
        points,
        pose,
        info["depth_intrinsic"],
        info["depth_width"],
        info["depth_height"],
    )
    indices = np.flatnonzero(inside)
    compared = visible = 0
    if len(indices):
        xs = np.rint(pixels[indices, 0]).astype(np.int32).clip(0, info["depth_width"] - 1)
        ys = np.rint(pixels[indices, 1]).astype(np.int32).clip(0, info["depth_height"] - 1)
        observed = depth[ys, xs].astype(np.float64) / 1000.0
        valid = observed > 0.0
        compared = int(np.count_nonzero(valid))
        visible = int(
            np.count_nonzero(
                np.abs(observed[valid] - camera[indices[valid], 2]) <= tolerance_metres
            )
        )
    return {
        **color_stats,
        "depth_compared_vertices": compared,
        "depth_visible_vertices": visible,
        "depth_visible_ratio": float(visible / compared) if compared else 0.0,
    }


def eligible(stats: dict[str, Any], rules: dict[str, Any]) -> bool:
    return bool(
        stats["inside_vertex_fraction"] >= float(rules["minimum_inside_vertex_fraction"])
        and stats["projected_area_pixels"] >= float(rules["minimum_projected_area_pixels"])
        and stats["image_margin_pixels"] >= float(rules["minimum_image_margin_pixels"])
        and stats["depth_compared_vertices"] >= int(rules["minimum_depth_compared_vertices"])
        and stats["depth_visible_vertices"] >= int(rules["minimum_depth_visible_vertices"])
        and stats["depth_visible_ratio"] >= float(rules["minimum_depth_visible_ratio"])
    )


def select_frame(
    data_root: Path,
    scan_id: str,
    target_id: int,
    rules: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    scan_root = data_root / scan_id
    points = extent.ply_instance_points(
        scan_root / "labels.instances.annotated.v2.ply", {target_id}
    )[target_id]
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    with zipfile.ZipFile(scan_root / "sequence.zip") as archive:
        info = parse_info(archive.read("_info.txt").decode("utf-8"))
        for frame in pose_frames(archive):
            try:
                pose = read_pose(archive, frame)
            except ValueError:
                continue
            opened["pose_members"] += 1
            if not np.isfinite(pose).all():
                continue
            try:
                depth = decode_depth(archive, frame)
                opened["depth_members"] += 1
                stats = frame_visibility(
                    points,
                    pose,
                    info,
                    depth,
                    float(rules["depth_consistency_metres"]),
                )
            except ValueError:
                continue
            if not eligible(stats, rules):
                continue
            key = (
                float(stats["depth_visible_vertices"]),
                float(stats["depth_visible_ratio"]),
                float(stats["projected_area_pixels"]),
                float(stats["image_margin_pixels"]),
                -float(frame),
            )
            candidates.append(
                (
                    key,
                    {
                        "frame": frame,
                        "color_size": [info["color_width"], info["color_height"]],
                        **stats,
                    },
                )
            )
    return (max(candidates, key=lambda row: row[0])[1] if candidates else None), opened


def source_record(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(artifact_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def freeze(
    protocol_path: Path,
    artifact_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    data_root, model_root = protocol_paths(protocol, artifact_root)
    predecessor_cohort, _ = verify_predecessor(protocol)
    predecessor_protocol = load_json(HERE / "l10_3rscan_registered_extent_protocol_v1.json")
    excluded = {
        (
            str(row["reference_scan_id"]),
            str(row["rescan_id"]),
            int(row["target_instance_id"]),
        )
        for row in predecessor_cohort["episodes"]
    }
    candidates = extent.candidate_rows(predecessor_protocol, data_root, require_geometry=True)
    rules = protocol["source_selector"]["frame_rules"]
    cache: dict[tuple[str, int], tuple[dict[str, Any] | None, dict[str, int]]] = {}
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    considered = 0
    selected: list[dict[str, Any]] = []
    used_references: set[str] = set()

    def cached(scan_id: str, target_id: int) -> dict[str, Any] | None:
        key = (scan_id, target_id)
        if key not in cache:
            cache[key] = select_frame(data_root, scan_id, target_id, rules)
            for name, count in cache[key][1].items():
                opened[name] += count
        return cache[key][0]

    for candidate in candidates:
        triple = (
            str(candidate["reference_scan_id"]),
            str(candidate["rescan_id"]),
            int(candidate["target_instance_id"]),
        )
        if triple in excluded or triple[0] in used_references:
            continue
        reference_zip = data_root / triple[0] / "sequence.zip"
        query_zip = data_root / triple[1] / "sequence.zip"
        if not reference_zip.is_file() or not query_zip.is_file():
            continue
        considered += 1
        reference_frame = cached(triple[0], triple[2])
        query_frame = cached(triple[1], triple[2])
        if reference_frame is None or query_frame is None:
            continue
        episode_id = f"PF{len(selected) + 1:02d}"
        selected.append(
            {
                "episode_id": episode_id,
                **candidate,
                "reference": reference_frame,
                "query": query_frame,
            }
        )
        used_references.add(triple[0])
        if len(selected) == int(protocol["source_selector"]["cohort_size"]):
            break

    require(
        len(selected) == int(protocol["source_selector"]["cohort_size"]),
        f"PIXEL_COHORT_NOT_EVALUABLE:{len(selected)}",
    )
    require(opened["rgb_members"] == 0, "RGB_OPENED_BEFORE_FREEZE")
    manifest: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for scan_id in (episode["reference_scan_id"], episode["rescan_id"]):
            for name in (
                "semseg.v2.json",
                "labels.instances.annotated.v2.ply",
                "sequence.zip",
            ):
                path = data_root / scan_id / name
                manifest[f"{scan_id}/{name}"] = source_record(path, artifact_root)
    for name in ("3RScan.json", "objects.json"):
        manifest[name] = source_record(data_root / name, artifact_root)
    model_weights = model_root / "model.safetensors"
    require(
        sha256(model_weights) == protocol["backbone"]["weights_sha256"],
        "DINO_WEIGHTS_HASH",
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_REFERENCE_CONDITIONED_ENDPOINT_FIELD_CANARY",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
        "storage": {
            "artifact_drive": artifact_root.resolve().drive,
            "dataset_relative_path": protocol["storage"]["dataset_relative_path"],
            "repository_drive_payload_bytes": 0,
        },
        "predecessor_cohort_sha256": protocol["predecessor"]["cohort_sha256"],
        "selection": {
            "candidate_rows_with_downloaded_geometry": len(candidates),
            "candidate_rows_considered": considered,
            "excluded_predecessor_target_triples": len(excluded),
            "opened_members": opened,
            "rules": rules,
        },
        "source_manifest": dict(sorted(manifest.items())),
        "model": {
            "path": model_weights.resolve().relative_to(artifact_root.resolve()).as_posix(),
            "bytes": model_weights.stat().st_size,
            "sha256": sha256(model_weights),
        },
        "episodes": selected,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, cohort)
    return cohort


def load_model(model_root: Path, device: torch.device) -> nn.Module:
    model = AutoModel.from_pretrained(model_root, local_files_only=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def preprocess(images: Iterable[np.ndarray], backbone: dict[str, Any]) -> torch.Tensor:
    tensors = []
    for image in images:
        require(image.ndim == 3 and image.shape[2] == 3, "RGB_SHAPE")
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        tensors.append(tensor)
    batch = torch.stack(tensors)
    batch = torch_functional.interpolate(
        batch,
        size=(int(backbone["input_height"]), int(backbone["input_width"])),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(backbone["normalization_mean"], dtype=batch.dtype)[None, :, None, None]
    std = torch.tensor(backbone["normalization_std"], dtype=batch.dtype)[None, :, None, None]
    return (batch - mean) / std


def encode_features(
    model: nn.Module,
    images: list[np.ndarray],
    backbone: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    pixels = preprocess(images, backbone).to(device)
    with torch.inference_mode():
        hidden = model(pixel_values=pixels).last_hidden_state[:, 1:]
    grid_height = int(backbone["input_height"]) // int(backbone["patch_size"])
    grid_width = int(backbone["input_width"]) // int(backbone["patch_size"])
    require(hidden.shape[1] == grid_height * grid_width, f"DINO_TOKEN_COUNT:{hidden.shape}")
    return torch_functional.normalize(hidden.float(), dim=-1), (grid_height, grid_width)


def grid_centres(width: int, height: int, grid: tuple[int, int]) -> np.ndarray:
    grid_height, grid_width = grid
    xs = (np.arange(grid_width, dtype=np.float32) + 0.5) * width / grid_width
    ys = (np.arange(grid_height, dtype=np.float32) + 0.5) * height / grid_height
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.reshape(-1), yy.reshape(-1)))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    return np.asarray(
        [cv2.pointPolygonTest(polygon, tuple(float(v) for v in point), False) >= 0 for point in points],
        dtype=bool,
    )


def reference_coordinate_system(
    points: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = extent.portal_frame(points)
    portal = extent.project_uv(points, *frame).astype(np.float64)
    minimum = np.min(portal, axis=0)
    span = np.maximum(np.max(portal, axis=0) - minimum, 1e-6)
    normalized = (portal - minimum) / span
    _, pixels, inside = project_points(points, pose, intrinsic, width, height)
    require(np.count_nonzero(inside) >= 16, "REFERENCE_COORDINATE_SUPPORT")
    image_to_portal, _ = cv2.findHomography(
        pixels[inside].astype(np.float32), normalized[inside].astype(np.float32), method=0
    )
    require(image_to_portal is not None, "REFERENCE_COORDINATE_HOMOGRAPHY")
    canonical_hull = cv2.convexHull(normalized.astype(np.float32)).reshape(-1, 2)
    return image_to_portal, canonical_hull


class EndpointHead(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_size, 3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.linear(values)
        return output[:, 0], output[:, 1:]


def train_endpoint_head(
    reference_features: torch.Tensor,
    labels: np.ndarray,
    coordinates: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[EndpointHead, dict[str, Any]]:
    device = reference_features.device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    head = EndpointHead(reference_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    target = torch.from_numpy(labels.astype(np.float32)).to(device)
    coordinate_target = torch.from_numpy(coordinates.astype(np.float32)).to(device)
    positive = target > 0.5
    positive_count = int(torch.count_nonzero(positive).item())
    negative_count = int(len(target) - positive_count)
    require(positive_count >= 4 and negative_count >= 4, "REFERENCE_PATCH_LABEL_SUPPORT")
    positive_weight = torch.tensor(negative_count / positive_count, device=device)
    initial_loss = final_loss = 0.0
    epochs = int(config["epochs"])
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, predicted_coordinates = head(reference_features)
        mask_loss = torch_functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=positive_weight
        )
        coordinate_loss = torch_functional.smooth_l1_loss(
            predicted_coordinates[positive], coordinate_target[positive]
        )
        loss = mask_loss + float(config["coordinate_loss_weight"]) * coordinate_loss
        loss.backward()
        optimizer.step()
        if epoch == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
    head.eval()
    return head, {
        "positive_reference_patches": positive_count,
        "negative_reference_patches": negative_count,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "epochs": epochs,
        "device": str(device),
    }


def decode_field(
    head: EndpointHead,
    query_features: torch.Tensor,
    query_size: tuple[int, int],
    grid: tuple[int, int],
    canonical_hull: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    with torch.inference_mode():
        logits, coordinate_tensor = head(query_features)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        coordinates = coordinate_tensor.detach().cpu().numpy()
    width, height = query_size
    centres = grid_centres(width, height, grid)
    coordinate_valid = np.all(
        (coordinates >= float(config["valid_coordinate_minimum"]))
        & (coordinates <= float(config["valid_coordinate_maximum"])),
        axis=1,
    )
    active = (probabilities >= float(config["mask_probability_threshold"])) & coordinate_valid
    active_grid = active.reshape(grid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active_grid, connectivity=8)
    components = []
    for label in range(1, count):
        indices = np.flatnonzero(labels.reshape(-1) == label)
        components.append(
            (
                float(np.sum(probabilities[indices])),
                int(stats[label, cv2.CC_STAT_AREA]),
                -label,
                indices,
            )
        )
    if not components:
        return None, {
            "active_patches": int(np.count_nonzero(active)),
            "selected_component_patches": 0,
            "failure": "NO_ACTIVE_COMPONENT",
            "maximum_probability": float(np.max(probabilities)),
        }
    _, component_size, _, indices = max(components, key=lambda row: row[:3])
    if component_size < int(config["minimum_component_patches"]):
        return None, {
            "active_patches": int(np.count_nonzero(active)),
            "selected_component_patches": component_size,
            "failure": "COMPONENT_TOO_SMALL",
            "maximum_probability": float(np.max(probabilities)),
        }
    portal_to_query, inliers = cv2.findHomography(
        coordinates[indices].astype(np.float32),
        centres[indices].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config["homography_ransac_threshold_pixels"]),
        maxIters=2000,
        confidence=0.99,
    )
    if portal_to_query is None or inliers is None or int(np.count_nonzero(inliers)) < 4:
        return None, {
            "active_patches": int(np.count_nonzero(active)),
            "selected_component_patches": component_size,
            "failure": "FIELD_HOMOGRAPHY_NOT_EVALUABLE",
            "maximum_probability": float(np.max(probabilities)),
        }
    prediction = cv2.perspectiveTransform(
        canonical_hull.reshape(1, -1, 2).astype(np.float32), portal_to_query
    ).reshape(-1, 2)
    finite = np.isfinite(prediction).all()
    if not finite:
        return None, {
            "active_patches": int(np.count_nonzero(active)),
            "selected_component_patches": component_size,
            "failure": "FIELD_POLYGON_NONFINITE",
            "maximum_probability": float(np.max(probabilities)),
        }
    endpoint_coordinates = np.asarray([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    endpoints = cv2.perspectiveTransform(
        endpoint_coordinates.reshape(1, -1, 2), portal_to_query
    ).reshape(-1, 2)
    return prediction, {
        "active_patches": int(np.count_nonzero(active)),
        "selected_component_patches": component_size,
        "homography_inliers": int(np.count_nonzero(inliers)),
        "maximum_probability": float(np.max(probabilities)),
        "mean_selected_probability": float(np.mean(probabilities[indices])),
        "predicted_canonical_coordinate_min": np.min(coordinates[indices], axis=0).tolist(),
        "predicted_canonical_coordinate_max": np.max(coordinates[indices], axis=0).tolist(),
        "endpoint_midline_pixels": endpoints.tolist(),
        "failure": None,
    }


def polygon_mask(polygon: np.ndarray | None, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon is None or len(polygon) < 3:
        return mask.astype(bool)
    clipped = np.column_stack(
        (
            np.clip(polygon[:, 0], 0, width - 1),
            np.clip(polygon[:, 1], 0, height - 1),
        )
    )
    cv2.fillConvexPoly(mask, cv2.convexHull(clipped.astype(np.float32)).astype(np.int32), 1)
    return mask.astype(bool)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    moments = cv2.moments(polygon.astype(np.float32))
    require(abs(moments["m00"]) > 1e-9, "POLYGON_CENTROID_EMPTY")
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def planar_truth_centroid(points: np.ndarray) -> np.ndarray:
    frame = extent.portal_frame(points)
    hull = extent.convex_hull(extent.project_uv(points, *frame))
    centroid = extent.polygon_centroid(hull)
    return frame[0] + frame[1] * centroid[0] + frame[2] * centroid[1]


def lift_pixel_to_plane(
    pixel: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
    plane_points: np.ndarray,
) -> np.ndarray | None:
    centre = np.mean(plane_points, axis=0)
    _, _, vectors = np.linalg.svd(plane_points - centre, full_matrices=False)
    normal = vectors[-1]
    camera_ray = np.asarray(
        [
            (pixel[0] - intrinsic[0, 2]) / intrinsic[0, 0],
            (pixel[1] - intrinsic[1, 2]) / intrinsic[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    origin = pose[:3, 3]
    direction = pose[:3, :3] @ camera_ray
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-9:
        return None
    scale = float(np.dot(normal, centre - origin) / denominator)
    if scale <= 0.0:
        return None
    return origin + direction * scale


def evaluate_polygon(
    prediction: np.ndarray | None,
    truth: np.ndarray,
    query_door_polygons: dict[int, np.ndarray],
    target_id: int,
    width: int,
    height: int,
    query_pose: np.ndarray,
    query_intrinsic: np.ndarray,
    query_target_points: np.ndarray,
) -> dict[str, Any]:
    truth_mask = polygon_mask(truth, width, height)
    prediction_mask = polygon_mask(prediction, width, height)
    overlaps = {
        instance_id: mask_iou(prediction_mask, polygon_mask(polygon, width, height))
        for instance_id, polygon in query_door_polygons.items()
    }
    selected_id = max(overlaps, key=lambda value: (overlaps[value], -value)) if overlaps else None
    if prediction is None or not np.any(prediction_mask):
        return {
            "pixel_iou": 0.0,
            "centroid_inside_target": False,
            "pixel_centroid_error": None,
            "world_centroid_error_metres": None,
            "selected_instance_id": selected_id,
            "target_top1": False,
            "instance_pixel_ious": {str(k): v for k, v in sorted(overlaps.items())},
            "prediction_polygon_pixels": None,
            "truth_polygon_pixels": truth.tolist(),
        }
    prediction_centroid = polygon_centroid(cv2.convexHull(prediction.astype(np.float32)).reshape(-1, 2))
    truth_centroid = polygon_centroid(truth)
    lifted = lift_pixel_to_plane(
        prediction_centroid, query_pose, query_intrinsic, query_target_points
    )
    truth_world = planar_truth_centroid(query_target_points)
    return {
        "pixel_iou": mask_iou(prediction_mask, truth_mask),
        "centroid_inside_target": bool(
            cv2.pointPolygonTest(truth.astype(np.float32), tuple(prediction_centroid), False) >= 0
        ),
        "pixel_centroid_error": float(np.linalg.norm(prediction_centroid - truth_centroid)),
        "world_centroid_error_metres": (
            float(np.linalg.norm(lifted - truth_world)) if lifted is not None else None
        ),
        "selected_instance_id": selected_id,
        "target_top1": bool(selected_id == target_id),
        "instance_pixel_ious": {str(k): v for k, v in sorted(overlaps.items())},
        "prediction_polygon_pixels": prediction.tolist(),
        "truth_polygon_pixels": truth.tolist(),
    }


def verify_manifest(cohort: dict[str, Any], artifact_root: Path) -> None:
    for key, record in cohort["source_manifest"].items():
        path = artifact_root / record["path"]
        require(path.is_file(), f"SOURCE_MISSING:{key}")
        require(path.stat().st_size == int(record["bytes"]), f"SOURCE_SIZE:{key}")
        require(sha256(path) == record["sha256"], f"SOURCE_HASH:{key}")


def arm_prediction(
    arm_name: str,
    fraction: float,
    reference_points: np.ndarray,
    reference_pose: np.ndarray,
    reference_info: dict[str, Any],
    reference_features: torch.Tensor,
    query_features: torch.Tensor,
    query_size: tuple[int, int],
    grid: tuple[int, int],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    full_polygon, _ = projected_hull(
        reference_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    portal_frame = extent.portal_frame(reference_points)
    portal_uv = extent.project_uv(reference_points, *portal_frame)
    supervised_points = (
        reference_points
        if fraction >= 1.0
        else extent.partial_fragment(reference_points, portal_uv, fraction)
    )
    supervision_polygon, _ = projected_hull(
        supervised_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    image_to_portal, canonical_hull = reference_coordinate_system(
        reference_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    centres = grid_centres(
        reference_info["color_width"], reference_info["color_height"], grid
    )
    labels = points_in_polygon(centres, supervision_polygon)
    coordinates = cv2.perspectiveTransform(
        centres.reshape(1, -1, 2).astype(np.float32), image_to_portal
    ).reshape(-1, 2)
    head, training = train_endpoint_head(
        reference_features,
        labels,
        coordinates,
        config,
        seed,
    )
    prediction, field = decode_field(
        head,
        query_features,
        query_size,
        grid,
        canonical_hull,
        config,
    )
    return prediction, {
        "arm": arm_name,
        "retained_width_fraction": fraction,
        "training": training,
        "field": field,
        "reference_full_polygon_pixels": full_polygon.tolist(),
        "reference_supervision_polygon_pixels": supervision_polygon.tolist(),
    }


def preview(
    output_path: Path,
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_full: np.ndarray,
    reference_partial: np.ndarray,
    query_truth: np.ndarray,
    oracle_complete: np.ndarray,
    partial_prediction: np.ndarray | None,
    complete_prediction: np.ndarray | None,
) -> None:
    reference = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
    query = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)

    def draw(image: np.ndarray, polygon: np.ndarray | None, color: tuple[int, int, int], width: int) -> None:
        if polygon is not None and len(polygon) >= 3:
            cv2.polylines(
                image,
                [np.rint(polygon).astype(np.int32)],
                True,
                color,
                width,
                cv2.LINE_AA,
            )

    draw(reference, reference_full, (0, 255, 0), 3)
    draw(reference, reference_partial, (0, 180, 255), 3)
    draw(query, query_truth, (0, 255, 0), 3)
    draw(query, oracle_complete, (255, 100, 0), 3)
    draw(query, partial_prediction, (0, 180, 255), 3)
    draw(query, complete_prediction, (255, 0, 255), 3)
    canvas = np.concatenate((reference, query), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require(cv2.imwrite(str(output_path), canvas), f"PREVIEW_WRITE:{output_path}")


def replay(
    protocol_path: Path,
    cohort_path: Path,
    artifact_root: Path,
    output_path: Path,
    preview_dir: Path | None,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    cohort = load_json(cohort_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "PROTOCOL_HASH")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__).resolve()), "ENTRYPOINT_HASH")
    data_root, model_root = protocol_paths(protocol, artifact_root)
    verify_predecessor(protocol)
    verify_manifest(cohort, artifact_root)
    require(
        sha256(model_root / "model.safetensors") == protocol["backbone"]["weights_sha256"],
        "DINO_WEIGHTS_HASH",
    )
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    device = torch.device("cuda:0")
    model = load_model(model_root, device)
    rows: list[dict[str, Any]] = []

    for episode_index, episode in enumerate(cohort["episodes"]):
        target_id = int(episode["target_instance_id"])
        reference_scan = str(episode["reference_scan_id"])
        query_scan = str(episode["rescan_id"])
        reference_zip = data_root / reference_scan / "sequence.zip"
        query_zip = data_root / query_scan / "sequence.zip"
        reference_points = extent.ply_instance_points(
            data_root / reference_scan / "labels.instances.annotated.v2.ply", {target_id}
        )[target_id]

        # Reference geometry is allowed supervision. Query archives expose RGB only
        # until both arm predictions are sealed below.
        with zipfile.ZipFile(reference_zip) as archive:
            reference_info = parse_info(archive.read("_info.txt").decode("utf-8"))
            reference_pose = read_pose(archive, int(episode["reference"]["frame"]))
            reference_rgb, reference_rgb_hash = decode_rgb(
                archive, int(episode["reference"]["frame"])
            )
        with zipfile.ZipFile(query_zip) as archive:
            query_rgb, query_rgb_hash = decode_rgb(archive, int(episode["query"]["frame"]))

        features, grid = encode_features(
            model, [reference_rgb, query_rgb], protocol["backbone"], device
        )
        reference_features = features[0]
        query_features = features[1]
        predictions: dict[str, np.ndarray | None] = {}
        arm_debug: dict[str, dict[str, Any]] = {}
        for arm_offset, (arm_name, arm) in enumerate(protocol["arms"].items()):
            prediction, diagnostics = arm_prediction(
                arm_name,
                float(arm["retained_width_fraction"]),
                reference_points,
                reference_pose,
                reference_info,
                reference_features,
                query_features,
                (query_rgb.shape[1], query_rgb.shape[0]),
                grid,
                protocol["learned_field"],
                int(protocol["learned_field"]["seed"]) + episode_index * 10 + arm_offset,
            )
            predictions[arm_name] = prediction
            arm_debug[arm_name] = diagnostics

        # Evaluator-only query pose, geometry, transform, and instance labels start here.
        with zipfile.ZipFile(query_zip) as archive:
            query_info = parse_info(archive.read("_info.txt").decode("utf-8"))
            query_pose = read_pose(archive, int(episode["query"]["frame"]))
        query_instance_ids = set(int(value) for value in episode["rescan_door_instance_ids"])
        query_instances = extent.ply_instance_points(
            data_root / query_scan / "labels.instances.annotated.v2.ply", query_instance_ids
        )
        query_target = query_instances[target_id]
        query_width, query_height = query_rgb.shape[1], query_rgb.shape[0]
        query_truth, _ = projected_hull(
            query_target,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        query_door_polygons: dict[int, np.ndarray] = {}
        for instance_id, points in query_instances.items():
            try:
                polygon, _ = projected_hull(
                    points,
                    query_pose,
                    query_info["color_intrinsic"],
                    query_width,
                    query_height,
                )
                query_door_polygons[instance_id] = polygon
            except ValueError:
                continue

        rescan_to_reference = extent.provider_matrix(episode["transform"])
        reference_to_rescan = np.linalg.inv(rescan_to_reference)
        reference_in_query = extent.transform_points(reference_points, reference_to_rescan)
        reference_frame = extent.portal_frame(reference_points)
        reference_uv = extent.project_uv(reference_points, *reference_frame)
        partial_reference = extent.partial_fragment(
            reference_points,
            reference_uv,
            float(protocol["arms"]["partial_reference"]["retained_width_fraction"]),
        )
        partial_in_query = extent.transform_points(partial_reference, reference_to_rescan)
        oracle_complete, _ = projected_hull(
            reference_in_query,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        oracle_partial, _ = projected_hull(
            partial_in_query,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        metrics = {
            arm_name: evaluate_polygon(
                prediction,
                query_truth,
                query_door_polygons,
                target_id,
                query_width,
                query_height,
                query_pose,
                query_info["color_intrinsic"],
                query_target,
            )
            for arm_name, prediction in predictions.items()
        }
        oracle_metrics = {
            "partial_reference": evaluate_polygon(
                oracle_partial,
                query_truth,
                query_door_polygons,
                target_id,
                query_width,
                query_height,
                query_pose,
                query_info["color_intrinsic"],
                query_target,
            ),
            "complete_reference": evaluate_polygon(
                oracle_complete,
                query_truth,
                query_door_polygons,
                target_id,
                query_width,
                query_height,
                query_pose,
                query_info["color_intrinsic"],
                query_target,
            ),
        }
        complete_iou = float(metrics["complete_reference"]["pixel_iou"])
        oracle_complete_iou = float(oracle_metrics["complete_reference"]["pixel_iou"])
        row = {
            "episode_id": episode["episode_id"],
            "reference_scan_id": reference_scan,
            "rescan_id": query_scan,
            "target_instance_id": target_id,
            "reference_frame": int(episode["reference"]["frame"]),
            "query_frame": int(episode["query"]["frame"]),
            "reference_rgb_sha256": reference_rgb_hash,
            "query_rgb_sha256": query_rgb_hash,
            "feature_grid": [grid[1], grid[0]],
            "model_inputs_sealed_before_evaluator_open": True,
            "arms": {
                name: {**arm_debug[name], "metrics": metrics[name]}
                for name in protocol["arms"]
            },
            "registered_geometry_ceiling": oracle_metrics,
            "complete_to_registered_ceiling_iou_ratio": (
                complete_iou / oracle_complete_iou if oracle_complete_iou > 0.0 else 0.0
            ),
        }
        rows.append(row)
        if preview_dir is not None:
            reference_full, _ = projected_hull(
                reference_points,
                reference_pose,
                reference_info["color_intrinsic"],
                reference_info["color_width"],
                reference_info["color_height"],
            )
            reference_partial_polygon, _ = projected_hull(
                partial_reference,
                reference_pose,
                reference_info["color_intrinsic"],
                reference_info["color_width"],
                reference_info["color_height"],
            )
            preview(
                preview_dir / f"{episode['episode_id']}.png",
                reference_rgb,
                query_rgb,
                reference_full,
                reference_partial_polygon,
                query_truth,
                oracle_complete,
                predictions["partial_reference"],
                predictions["complete_reference"],
            )

    require(len(rows) == 1, "CANARY_EPISODE_COUNT")
    row = rows[0]
    partial = row["arms"]["partial_reference"]["metrics"]
    complete = row["arms"]["complete_reference"]["metrics"]
    gate = protocol["decision_gate"]
    world_error = complete["world_centroid_error_metres"]
    conditions = {
        "complete_target_top1": bool(complete["target_top1"])
        if bool(gate["complete_target_top1_required"])
        else True,
        "complete_centroid_inside": bool(complete["centroid_inside_target"])
        if bool(gate["complete_centroid_inside_required"])
        else True,
        "complete_pixel_iou": float(complete["pixel_iou"])
        >= float(gate["complete_pixel_iou_minimum"]),
        "complete_world_centroid_error": world_error is not None
        and float(world_error) <= float(gate["complete_world_centroid_error_metres_maximum"]),
        "complete_to_registered_ceiling": float(
            row["complete_to_registered_ceiling_iou_ratio"]
        )
        >= float(gate["complete_to_registered_ceiling_iou_ratio_minimum"]),
        "complete_iou_gain_over_partial": float(complete["pixel_iou"])
        - float(partial["pixel_iou"])
        >= float(gate["minimum_complete_iou_gain_over_partial"]),
    }
    passed = all(conditions.values())
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "L10_3RSCAN_REFERENCE_PIXEL_ENDPOINT_FIELD_DEVELOPMENT_CANARY_MET"
            if passed
            else "L10_3RSCAN_REFERENCE_PIXEL_ENDPOINT_FIELD_DEVELOPMENT_CANARY_NOT_MET"
        ),
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
        "storage": {
            "artifact_drive": artifact_root.resolve().drive,
            "dataset_payload_on_repository_drive": False,
        },
        "backend": {
            "framework": "torch",
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(),
            "peak_allocated_mebibytes": float(torch.cuda.max_memory_allocated() / 1024**2),
            "selection_reason": "GPU_FIRST_DINO_DENSE_FEATURE_INFERENCE_AND_LINEAR_HEAD_TRAINING",
        },
        "aggregate": {
            "episodes": 1,
            "partial_pixel_iou": float(partial["pixel_iou"]),
            "complete_pixel_iou": float(complete["pixel_iou"]),
            "complete_iou_gain_over_partial": float(complete["pixel_iou"])
            - float(partial["pixel_iou"]),
            "registered_complete_pixel_iou_ceiling": float(
                row["registered_geometry_ceiling"]["complete_reference"]["pixel_iou"]
            ),
            "complete_to_registered_ceiling_iou_ratio": float(
                row["complete_to_registered_ceiling_iou_ratio"]
            ),
            "complete_world_centroid_error_metres": world_error,
            "complete_target_top1": bool(complete["target_top1"]),
            "complete_centroid_inside": bool(complete["centroid_inside_target"]),
        },
        "decision_conditions": conditions,
        "episodes": rows,
        "stop_rule_observed": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, result)
    return result


def preflight(protocol_path: Path, artifact_root: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    data_root, model_root = protocol_paths(protocol, artifact_root)
    verify_predecessor(protocol)
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    device = torch.device("cuda:0")
    model = load_model(model_root, device)
    backbone = protocol["backbone"]
    dummy = np.zeros((540, 960, 3), dtype=np.uint8)
    features, grid = encode_features(model, [dummy], backbone, device)
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "L10_3RSCAN_REFERENCE_PIXEL_FIELD_PREFLIGHT_PASS",
        "storage": {
            "artifact_root": str(artifact_root.resolve()),
            "artifact_drive": artifact_root.resolve().drive,
            "data_root": str(data_root.resolve()),
            "model_root": str(model_root.resolve()),
        },
        "backend": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(),
            "feature_shape": list(features.shape),
            "feature_grid": [grid[1], grid[0]],
        },
    }
    atomic_write_json(output_path, result)
    return result


def self_test() -> dict[str, Any]:
    square = np.asarray([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    shifted = square + np.asarray([10, 0], dtype=np.float32)
    first = polygon_mask(square, 64, 64)
    second = polygon_mask(shifted, 64, 64)
    require(abs(mask_iou(first, first) - 1.0) < 1e-9, "SELF_IOU_IDENTITY")
    require(0.0 < mask_iou(first, second) < 1.0, "SELF_IOU_OVERLAP")
    centres = grid_centres(64, 32, (2, 4))
    require(centres.shape == (8, 2), "SELF_GRID_SHAPE")
    return {
        "status": "SELF_TEST_PASS",
        "overlap_iou": mask_iou(first, second),
        "grid_centres": centres.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_protocol_v1.json",
    )
    preflight_parser.add_argument("--artifact-root", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_protocol_v1.json",
    )
    freeze_parser.add_argument("--artifact-root", type=Path, required=True)
    freeze_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_cohort_v1.json",
    )

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_protocol_v1.json",
    )
    replay_parser.add_argument(
        "--cohort",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_cohort_v1.json",
    )
    replay_parser.add_argument("--artifact-root", type=Path, required=True)
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_reference_pixel_field_result_v1.json",
    )
    replay_parser.add_argument("--preview-dir", type=Path)

    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.protocol.resolve(), args.artifact_root.resolve(), args.output.resolve())
    elif args.command == "freeze":
        result = freeze(args.protocol.resolve(), args.artifact_root.resolve(), args.output.resolve())
    elif args.command == "replay":
        result = replay(
            args.protocol.resolve(),
            args.cohort.resolve(),
            args.artifact_root.resolve(),
            args.output.resolve(),
            args.preview_dir.resolve() if args.preview_dir else None,
        )
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
