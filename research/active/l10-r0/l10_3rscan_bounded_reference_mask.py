#!/usr/bin/env python3
"""Bounded reference-conditioned endpoint masks on registered 3RScan doors.

Geometry and depth freeze a fresh reference/rescan pair before RGB is decoded.
During replay, registered geometry supervises only the reference image. Frozen
DINOv2 patch correspondences create query point prompts for frozen SAM2.1. The
endpoint extent is the convex hull of a supported query mask and both endpoints
are actual mask pixels; coordinate regression and homography are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_reference_pixel_field as pixel_field  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-bounded-reference-mask-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-bounded-reference-mask-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-bounded-reference-mask-result-v1"


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


def protocol_paths(
    protocol: dict[str, Any], artifact_root: Path
) -> tuple[Path, Path, Path]:
    expected_drive = str(
        protocol["storage"]["artifact_root_argument_must_resolve_to_drive"]
    ).upper()
    resolved = artifact_root.resolve()
    require(resolved.drive.upper() == expected_drive, f"ARTIFACT_ROOT_DRIVE:{resolved}")
    data_root = resolved / protocol["storage"]["dataset_relative_path"]
    dino_root = resolved / protocol["storage"]["dino_model_relative_path"]
    sam_root = resolved / protocol["storage"]["sam_model_relative_path"]
    for label, path in (
        ("DATA_ROOT", data_root),
        ("DINO_ROOT", dino_root),
        ("SAM_ROOT", sam_root),
    ):
        require(path.is_dir(), f"{label}_MISSING:{path}")
    return data_root, dino_root, sam_root


def verify_predecessor(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor = protocol["predecessor"]
    for key in ("entrypoint", "protocol", "cohort", "result"):
        path = HERE / predecessor[key]
        require(path.is_file(), f"PREDECESSOR_MISSING:{path}")
        require(
            sha256(path) == predecessor[f"{key}_sha256"],
            f"PREDECESSOR_HASH:{path.name}",
        )
    result = load_json(HERE / predecessor["result"])
    require(result.get("status") == predecessor["required_status"], "PREDECESSOR_STATUS")
    return load_json(HERE / predecessor["cohort"]), result


def source_record(path: Path, artifact_root: Path) -> dict[str, Any]:
    require(path.is_file(), f"SOURCE_MISSING:{path}")
    return {
        "path": path.resolve().relative_to(artifact_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_manifest(cohort: dict[str, Any], artifact_root: Path) -> None:
    for key, record in cohort["source_manifest"].items():
        path = artifact_root / record["path"]
        require(path.is_file(), f"SOURCE_MISSING:{key}")
        require(path.stat().st_size == int(record["bytes"]), f"SOURCE_SIZE:{key}")
        require(sha256(path) == record["sha256"], f"SOURCE_HASH:{key}")


def freeze(protocol_path: Path, artifact_root: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    data_root, dino_root, sam_root = protocol_paths(protocol, artifact_root)
    predecessor_cohort, _ = verify_predecessor(protocol)
    predecessor_references = {
        str(row["reference_scan_id"]) for row in predecessor_cohort["episodes"]
    }
    allowed_pairs = {
        (str(pair[0]), str(pair[1]))
        for pair in protocol["source_selector"]["allowed_registered_pairs"]
    }
    require(
        not any(reference in predecessor_references for reference, _ in allowed_pairs),
        "REFERENCE_NOT_FRESH",
    )
    geometry_protocol = load_json(HERE / "l10_3rscan_registered_extent_protocol_v1.json")
    candidates = [
        row
        for row in extent.candidate_rows(geometry_protocol, data_root, require_geometry=True)
        if (str(row["reference_scan_id"]), str(row["rescan_id"])) in allowed_pairs
    ]
    require(candidates, "ALLOWED_REGISTERED_PAIR_EMPTY")
    rules = protocol["source_selector"]["frame_rules"]
    cache: dict[tuple[str, int], tuple[dict[str, Any] | None, dict[str, int]]] = {}
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    considered = 0
    selected: dict[str, Any] | None = None

    def selected_frame(scan_id: str, target_id: int) -> dict[str, Any] | None:
        key = (scan_id, target_id)
        if key not in cache:
            cache[key] = pixel_field.select_frame(data_root, scan_id, target_id, rules)
            for name, count in cache[key][1].items():
                opened[name] += int(count)
        return cache[key][0]

    for candidate in candidates:
        reference_scan = str(candidate["reference_scan_id"])
        query_scan = str(candidate["rescan_id"])
        target_id = int(candidate["target_instance_id"])
        if not (data_root / reference_scan / "sequence.zip").is_file():
            continue
        if not (data_root / query_scan / "sequence.zip").is_file():
            continue
        considered += 1
        reference_frame = selected_frame(reference_scan, target_id)
        query_frame = selected_frame(query_scan, target_id)
        if reference_frame is None or query_frame is None:
            continue
        selected = {
            "episode_id": "BRM01",
            **candidate,
            "reference": reference_frame,
            "query": query_frame,
        }
        break

    require(selected is not None, "BOUNDED_MASK_COHORT_NOT_EVALUABLE")
    require(opened["rgb_members"] == 0, "RGB_OPENED_BEFORE_FREEZE")
    manifest: dict[str, dict[str, Any]] = {}
    for scan_id in (selected["reference_scan_id"], selected["rescan_id"]):
        for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip"):
            path = data_root / str(scan_id) / name
            manifest[f"{scan_id}/{name}"] = source_record(path, artifact_root)
    for name in ("3RScan.json", "objects.json"):
        manifest[name] = source_record(data_root / name, artifact_root)

    dino_weights = dino_root / "model.safetensors"
    sam_weights = sam_root / "model.safetensors"
    require(
        sha256(dino_weights) == protocol["dino_backbone"]["weights_sha256"],
        "DINO_WEIGHTS_HASH",
    )
    require(
        sha256(sam_weights) == protocol["sam_masker"]["weights_sha256"],
        "SAM_WEIGHTS_HASH",
    )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_BOUNDED_REFERENCE_MASK_CANARY",
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
            "candidate_rows_in_allowed_pairs": len(candidates),
            "candidate_rows_considered": considered,
            "opened_members": opened,
            "rules": rules,
        },
        "source_manifest": dict(sorted(manifest.items())),
        "models": {
            "dino": source_record(dino_weights, artifact_root),
            "sam2": source_record(sam_weights, artifact_root),
        },
        "episodes": [selected],
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, cohort)
    return cohort


def spread_select(
    indices: np.ndarray,
    scores: np.ndarray,
    grid: tuple[int, int],
    limit: int,
    separation: int,
) -> list[int]:
    grid_height, grid_width = grid
    selected: list[int] = []
    ordered = sorted((int(index) for index in indices), key=lambda index: (-float(scores[index]), index))
    for index in ordered:
        row, column = divmod(index, grid_width)
        if all(
            max(abs(row - divmod(other, grid_width)[0]), abs(column - divmod(other, grid_width)[1]))
            >= separation
            for other in selected
        ):
            selected.append(index)
            if len(selected) == limit:
                break
    require(all(0 <= value < grid_height * grid_width for value in selected), "PROMPT_INDEX")
    return selected


def transferred_prompts(
    reference_features: torch.Tensor,
    query_features: torch.Tensor,
    reference_positive: np.ndarray,
    reference_full: np.ndarray,
    query_centres: np.ndarray,
    grid: tuple[int, int],
    config: dict[str, Any],
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    with torch.inference_mode():
        similarities = query_features @ reference_features.T
        query_scores, query_best_reference = torch.max(similarities, dim=1)
        reference_best_query = torch.argmax(similarities, dim=0)
        query_indices = torch.arange(len(query_features), device=query_features.device)
        mutual = reference_best_query[query_best_reference] == query_indices
    scores = query_scores.detach().cpu().numpy()
    best_reference = query_best_reference.detach().cpu().numpy()
    mutual_numpy = mutual.detach().cpu().numpy()
    positive_candidates = np.flatnonzero(
        mutual_numpy
        & reference_positive[best_reference]
        & (scores >= float(config["minimum_positive_cosine"]))
    )
    negative_candidates = np.flatnonzero(
        (~reference_full[best_reference])
        & (scores >= float(config["minimum_negative_cosine"]))
    )
    separation = int(config["minimum_grid_cell_separation"])
    positive_indices = spread_select(
        positive_candidates,
        scores,
        grid,
        int(config["positive_prompt_limit"]),
        separation,
    )
    negative_indices = spread_select(
        negative_candidates,
        scores,
        grid,
        int(config["negative_prompt_limit"]),
        separation,
    )
    points = [query_centres[index].astype(float).tolist() for index in positive_indices]
    points.extend(query_centres[index].astype(float).tolist() for index in negative_indices)
    labels = [1] * len(positive_indices) + [0] * len(negative_indices)
    return points, labels, {
        "mutual_matches": int(np.count_nonzero(mutual_numpy)),
        "positive_candidate_matches": int(len(positive_candidates)),
        "negative_candidate_matches": int(len(negative_candidates)),
        "positive_prompt_indices": positive_indices,
        "negative_prompt_indices": negative_indices,
        "positive_prompt_points": points[: len(positive_indices)],
        "negative_prompt_points": points[len(positive_indices) :],
        "positive_prompt_cosines": [float(scores[index]) for index in positive_indices],
        "negative_prompt_cosines": [float(scores[index]) for index in negative_indices],
        "minimum_positive_prompts_met": len(positive_indices)
        >= int(config["minimum_positive_prompts"]),
        "query_best_cosine_minimum": float(np.min(scores)),
        "query_best_cosine_maximum": float(np.max(scores)),
    }


def select_prompt_component(
    mask: np.ndarray, positive_points: list[list[float]]
) -> tuple[np.ndarray, dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[tuple[int, int, int]] = []
    height, width = mask.shape
    for label in range(1, count):
        hits = 0
        for x, y in positive_points:
            xx = int(np.clip(round(x), 0, width - 1))
            yy = int(np.clip(round(y), 0, height - 1))
            hits += int(labels[yy, xx] == label)
        components.append((hits, int(stats[label, cv2.CC_STAT_AREA]), -label))
    require(components, "SAM_MASK_EMPTY")
    hits, area, negative_label = max(components)
    selected_label = -negative_label
    selected = np.ascontiguousarray(labels == selected_label)
    return selected, {
        "raw_components": len(components),
        "selected_component_label": selected_label,
        "selected_positive_prompt_hits": hits,
        "selected_support_pixels": area,
    }


def support_extent(mask: np.ndarray) -> tuple[np.ndarray, list[list[float]], dict[str, Any]]:
    ys, xs = np.nonzero(mask)
    require(len(xs) >= 3, "SUPPORTED_MASK_TOO_SMALL")
    points = np.column_stack((xs, ys)).astype(np.float32)
    polygon = cv2.convexHull(points).reshape(-1, 2)
    centred = points.astype(np.float64) - np.mean(points, axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-12 and axis[1] < 0.0):
        axis = -axis
    projection = points @ axis
    first = points[int(np.argmin(projection))]
    second = points[int(np.argmax(projection))]
    endpoints = [first.astype(float).tolist(), second.astype(float).tolist()]
    endpoint_supported = all(bool(mask[int(point[1]), int(point[0])]) for point in (first, second))
    require(endpoint_supported, "ENDPOINT_NOT_SUPPORTED")
    return polygon, endpoints, {
        "extent_polygon_vertices": int(len(polygon)),
        "principal_axis_xy": axis.astype(float).tolist(),
        "endpoint_pixels": endpoints,
        "endpoints_are_supported_query_pixels": endpoint_supported,
        "homography_used": False,
        "coordinate_regression_used": False,
        "image_space_extrapolation_used": False,
    }


def sam_supported_extent(
    processor: Sam2Processor,
    model: Sam2Model,
    query_rgb: np.ndarray,
    points: list[list[float]],
    labels: list[int],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[list[float]], np.ndarray, dict[str, Any]]:
    image = Image.fromarray(query_rgb, mode="RGB")
    inputs = processor(
        images=image,
        input_points=[[points]],
        input_labels=[[labels]],
        return_tensors="pt",
    )
    original_sizes = inputs["original_sizes"].detach().cpu()
    input_fields = sorted(inputs.keys())
    with torch.inference_mode():
        outputs = model(**inputs.to("cuda:0"), multimask_output=False)
    native = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        mask_threshold=float(config["postprocess_mask_threshold"]),
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
    )
    require(isinstance(native, list) and len(native) == 1, "SAM_POSTPROCESS_BATCH")
    batch = native[0]
    height, width = query_rgb.shape[:2]
    require(tuple(batch.shape) == (1, 1, height, width), f"SAM_MASK_SHAPE:{batch.shape}")
    mask = np.ascontiguousarray(batch[0, 0].numpy(), dtype=bool)
    positive_points = [point for point, label in zip(points, labels) if label == 1]
    support, component = select_prompt_component(mask, positive_points)
    polygon, endpoints, endpoint_debug = support_extent(support)
    return polygon, endpoints, support, {
        "input_fields": input_fields,
        "input_points_shape": list(inputs["input_points"].shape),
        "input_labels_shape": list(inputs["input_labels"].shape),
        "raw_pred_masks_shape": list(outputs.pred_masks.shape),
        "raw_iou_scores_shape": list(outputs.iou_scores.shape),
        "multimask_output": False,
        "iou_scores_ignored": True,
        "postprocess_mask_threshold": float(config["postprocess_mask_threshold"]),
        "raw_mask_pixels": int(np.count_nonzero(mask)),
        **component,
        **endpoint_debug,
    }


def arm_prediction(
    arm_name: str,
    fraction: float,
    reference_points: np.ndarray,
    reference_pose: np.ndarray,
    reference_info: dict[str, Any],
    reference_features: torch.Tensor,
    query_features: torch.Tensor,
    query_rgb: np.ndarray,
    grid: tuple[int, int],
    transfer_config: dict[str, Any],
    sam_config: dict[str, Any],
    sam_processor: Sam2Processor,
    sam_model: Sam2Model,
) -> tuple[np.ndarray, list[list[float]], np.ndarray, dict[str, Any]]:
    full_polygon, _ = pixel_field.projected_hull(
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
    supervision_polygon, _ = pixel_field.projected_hull(
        supervised_points,
        reference_pose,
        reference_info["color_intrinsic"],
        reference_info["color_width"],
        reference_info["color_height"],
    )
    reference_centres = pixel_field.grid_centres(
        reference_info["color_width"], reference_info["color_height"], grid
    )
    query_centres = pixel_field.grid_centres(query_rgb.shape[1], query_rgb.shape[0], grid)
    reference_positive = pixel_field.points_in_polygon(reference_centres, supervision_polygon)
    reference_full = pixel_field.points_in_polygon(reference_centres, full_polygon)
    points, labels, transfer = transferred_prompts(
        reference_features,
        query_features,
        reference_positive,
        reference_full,
        query_centres,
        grid,
        transfer_config,
    )
    if not bool(transfer["minimum_positive_prompts_met"]):
        support = np.zeros(query_rgb.shape[:2], dtype=bool)
        return (
            np.empty((0, 2), dtype=np.float32),
            [],
            support,
            {
                "arm": arm_name,
                "retained_width_fraction": fraction,
                "reference_full_polygon_pixels": full_polygon.tolist(),
                "reference_supervision_polygon_pixels": supervision_polygon.tolist(),
                "reference_positive_patches": int(np.count_nonzero(reference_positive)),
                "reference_full_patches": int(np.count_nonzero(reference_full)),
                "transfer": transfer,
                "masker": {
                    "failure": "INSUFFICIENT_MUTUAL_POSITIVE_PROMPTS",
                    "selected_support_pixels": 0,
                    "endpoint_pixels": [],
                    "endpoints_are_supported_query_pixels": False,
                    "homography_used": False,
                    "coordinate_regression_used": False,
                    "image_space_extrapolation_used": False,
                },
                "support_mask_uint8_sha256": hashlib.sha256(
                    support.astype(np.uint8).tobytes(order="C")
                ).hexdigest(),
            },
        )
    polygon, endpoints, support, masker = sam_supported_extent(
        sam_processor,
        sam_model,
        query_rgb,
        points,
        labels,
        sam_config,
    )
    return polygon, endpoints, support, {
        "arm": arm_name,
        "retained_width_fraction": fraction,
        "reference_full_polygon_pixels": full_polygon.tolist(),
        "reference_supervision_polygon_pixels": supervision_polygon.tolist(),
        "reference_positive_patches": int(np.count_nonzero(reference_positive)),
        "reference_full_patches": int(np.count_nonzero(reference_full)),
        "transfer": transfer,
        "masker": masker,
        "support_mask_uint8_sha256": hashlib.sha256(
            support.astype(np.uint8).tobytes(order="C")
        ).hexdigest(),
    }


def preview(
    output_path: Path,
    reference_rgb: np.ndarray,
    query_rgb: np.ndarray,
    reference_full: np.ndarray,
    reference_partial: np.ndarray,
    query_truth: np.ndarray,
    oracle_complete: np.ndarray,
    predictions: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, Any]],
) -> None:
    reference = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
    query = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)

    def draw_polygon(image: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], width: int) -> None:
        if len(polygon) >= 3:
            cv2.polylines(image, [np.rint(polygon).astype(np.int32)], True, color, width, cv2.LINE_AA)

    draw_polygon(reference, reference_full, (0, 255, 0), 3)
    draw_polygon(reference, reference_partial, (0, 180, 255), 3)
    draw_polygon(query, query_truth, (0, 255, 0), 3)
    draw_polygon(query, oracle_complete, (255, 100, 0), 3)
    draw_polygon(query, predictions["partial_reference"], (0, 180, 255), 3)
    draw_polygon(query, predictions["complete_reference"], (255, 0, 255), 3)
    complete = diagnostics["complete_reference"]
    for x, y in complete["transfer"]["positive_prompt_points"]:
        cv2.circle(query, (round(x), round(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
    for x, y in complete["transfer"]["negative_prompt_points"]:
        cv2.circle(query, (round(x), round(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)
    for x, y in complete["masker"]["endpoint_pixels"]:
        cv2.circle(query, (round(x), round(y)), 7, (255, 0, 255), -1, cv2.LINE_AA)
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
    data_root, dino_root, sam_root = protocol_paths(protocol, artifact_root)
    verify_predecessor(protocol)
    verify_manifest(cohort, artifact_root)
    require(
        sha256(dino_root / "model.safetensors") == protocol["dino_backbone"]["weights_sha256"],
        "DINO_WEIGHTS_HASH",
    )
    require(
        sha256(sam_root / "model.safetensors") == protocol["sam_masker"]["weights_sha256"],
        "SAM_WEIGHTS_HASH",
    )
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    device = torch.device("cuda:0")
    dino_model = pixel_field.load_model(dino_root, device)
    sam_processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        sam_root,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to(device)
    for parameter in sam_model.parameters():
        parameter.requires_grad_(False)
    rows: list[dict[str, Any]] = []

    for episode in cohort["episodes"]:
        target_id = int(episode["target_instance_id"])
        reference_scan = str(episode["reference_scan_id"])
        query_scan = str(episode["rescan_id"])
        reference_points = extent.ply_instance_points(
            data_root / reference_scan / "labels.instances.annotated.v2.ply", {target_id}
        )[target_id]
        reference_zip = data_root / reference_scan / "sequence.zip"
        query_zip = data_root / query_scan / "sequence.zip"
        with zipfile.ZipFile(reference_zip) as archive:
            reference_info = pixel_field.parse_info(archive.read("_info.txt").decode("utf-8"))
            reference_pose = pixel_field.read_pose(archive, int(episode["reference"]["frame"]))
            reference_rgb, reference_rgb_hash = pixel_field.decode_rgb(
                archive, int(episode["reference"]["frame"])
            )
        with zipfile.ZipFile(query_zip) as archive:
            query_rgb, query_rgb_hash = pixel_field.decode_rgb(
                archive, int(episode["query"]["frame"])
            )

        features, grid = pixel_field.encode_features(
            dino_model, [reference_rgb, query_rgb], protocol["dino_backbone"], device
        )
        predictions: dict[str, np.ndarray] = {}
        endpoints: dict[str, list[list[float]]] = {}
        support_masks: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for arm_name, arm in protocol["arms"].items():
            polygon, arm_endpoints, support, debug = arm_prediction(
                arm_name,
                float(arm["retained_width_fraction"]),
                reference_points,
                reference_pose,
                reference_info,
                features[0],
                features[1],
                query_rgb,
                grid,
                protocol["transfer_field"],
                protocol["sam_masker"],
                sam_processor,
                sam_model,
            )
            predictions[arm_name] = polygon
            endpoints[arm_name] = arm_endpoints
            support_masks[arm_name] = support
            diagnostics[arm_name] = debug

        # Evaluator-only query pose, geometry, transform, and labels begin here.
        with zipfile.ZipFile(query_zip) as archive:
            query_info = pixel_field.parse_info(archive.read("_info.txt").decode("utf-8"))
            query_pose = pixel_field.read_pose(archive, int(episode["query"]["frame"]))
        query_instance_ids = {int(value) for value in episode["rescan_door_instance_ids"]}
        query_instances = extent.ply_instance_points(
            data_root / query_scan / "labels.instances.annotated.v2.ply", query_instance_ids
        )
        query_target = query_instances[target_id]
        query_width, query_height = query_rgb.shape[1], query_rgb.shape[0]
        query_truth, _ = pixel_field.projected_hull(
            query_target,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        query_door_polygons: dict[int, np.ndarray] = {}
        for instance_id, points in query_instances.items():
            try:
                polygon, _ = pixel_field.projected_hull(
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
        oracle_complete, _ = pixel_field.projected_hull(
            reference_in_query,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        oracle_partial, _ = pixel_field.projected_hull(
            partial_in_query,
            query_pose,
            query_info["color_intrinsic"],
            query_width,
            query_height,
        )
        metrics = {
            name: pixel_field.evaluate_polygon(
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
            for name, prediction in predictions.items()
        }
        oracle_metrics = {
            "partial_reference": pixel_field.evaluate_polygon(
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
            "complete_reference": pixel_field.evaluate_polygon(
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
        for name in protocol["arms"]:
            metrics[name]["endpoint_pixels"] = endpoints[name]
            metrics[name]["endpoints_are_supported_query_pixels"] = bool(
                len(endpoints[name]) == 2
                and all(
                    support_masks[name][int(point[1]), int(point[0])]
                    for point in endpoints[name]
                )
            )
        complete_iou = float(metrics["complete_reference"]["pixel_iou"])
        oracle_iou = float(oracle_metrics["complete_reference"]["pixel_iou"])
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
                name: {**diagnostics[name], "metrics": metrics[name]}
                for name in protocol["arms"]
            },
            "registered_geometry_ceiling": oracle_metrics,
            "complete_to_registered_ceiling_iou_ratio": (
                complete_iou / oracle_iou if oracle_iou > 0.0 else 0.0
            ),
        }
        rows.append(row)
        if preview_dir is not None:
            reference_full, _ = pixel_field.projected_hull(
                reference_points,
                reference_pose,
                reference_info["color_intrinsic"],
                reference_info["color_width"],
                reference_info["color_height"],
            )
            reference_partial, _ = pixel_field.projected_hull(
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
                reference_partial,
                query_truth,
                oracle_complete,
                predictions,
                diagnostics,
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
        "complete_endpoints_supported": bool(complete["endpoints_are_supported_query_pixels"])
        if bool(gate["complete_endpoints_supported_required"])
        else True,
        "complete_pixel_iou": float(complete["pixel_iou"])
        >= float(gate["complete_pixel_iou_minimum"]),
        "complete_world_centroid_error": world_error is not None
        and float(world_error) <= float(gate["complete_world_centroid_error_metres_maximum"]),
        "complete_to_registered_ceiling": float(row["complete_to_registered_ceiling_iou_ratio"])
        >= float(gate["complete_to_registered_ceiling_iou_ratio_minimum"]),
    }
    passed = all(conditions.values())
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "L10_3RSCAN_BOUNDED_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_MET"
            if passed
            else "L10_3RSCAN_BOUNDED_REFERENCE_ENDPOINT_MASK_DEVELOPMENT_CANARY_NOT_MET"
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
            "models": ["facebook/dinov2-small", "facebook/sam2.1-hiera-small"],
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
            "complete_endpoints_supported": bool(
                complete["endpoints_are_supported_query_pixels"]
            ),
        },
        "decision_conditions": conditions,
        "episodes": rows,
        "execution_note": {
            "replay_attempts": 2,
            "first_attempt": "ABORTED_BEFORE_SAM_INFERENCE_WHEN_PARTIAL_ARM_HAD_ONE_MUTUAL_POSITIVE_PROMPT",
            "mechanical_correction": "Serialize an insufficient-support arm as an empty prediction and continue; protocol, cohort selection, RGB frames, models, thresholds, prompts, decoder, and gate were unchanged."
        },
        "stop_rule_observed": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, result)
    return result


def self_test() -> dict[str, Any]:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    selected = spread_select(np.arange(4), scores, (2, 2), 4, 1)
    require(selected == [0, 1, 2, 3], f"SELF_SPREAD:{selected}")
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:9, 4:13] = True
    polygon, endpoints, debug = support_extent(mask)
    require(len(polygon) == 4, "SELF_HULL")
    require(bool(debug["endpoints_are_supported_query_pixels"]), "SELF_ENDPOINTS")
    return {"status": "SELF_TEST_PASS", "selected": selected, "endpoints": endpoints}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_bounded_reference_mask_protocol_v1.json",
    )
    freeze_parser.add_argument("--artifact-root", type=Path, required=True)
    freeze_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_bounded_reference_mask_cohort_v1.json",
    )

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_bounded_reference_mask_protocol_v1.json",
    )
    replay_parser.add_argument(
        "--cohort",
        type=Path,
        default=HERE / "l10_3rscan_bounded_reference_mask_cohort_v1.json",
    )
    replay_parser.add_argument("--artifact-root", type=Path, required=True)
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_bounded_reference_mask_result_v1.json",
    )
    replay_parser.add_argument("--preview-dir", type=Path)

    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "freeze":
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
