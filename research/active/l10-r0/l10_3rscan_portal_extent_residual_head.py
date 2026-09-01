#!/usr/bin/env python3
"""Learn target-disjoint completion of a partial portal plane extent."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
from typing import Any
import zipfile

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402
import l10_3rscan_query_mask_3d_track as track  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402
import l10_scenenn_real_posed_portal_transfer as plane  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-portal-extent-residual-head-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-portal-extent-residual-head-result-v1"


def _load_image(archive: zipfile.ZipFile, member: str) -> tuple[Image.Image, str]:
    payload = archive.read(member)
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.convert("RGB")
    return image, hashlib.sha256(payload).hexdigest()


def _frame_and_bounds(points: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    pixel.require(len(points) >= 64, f"PARTIAL_POINTS_TOO_FEW:{len(points)}")
    normal, offset, fit = plane.fit_plane(points)
    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    horizontal = np.cross(vertical, normal)
    pixel.require(np.linalg.norm(horizontal) >= 0.35, "PARTIAL_PLANE_NOT_VERTICAL")
    horizontal /= np.linalg.norm(horizontal)
    if horizontal[0] < 0.0 or (abs(horizontal[0]) < 1e-12 and horizontal[1] < 0.0):
        horizontal = -horizontal
    residual = np.abs(points @ normal + offset)
    cutoff = float(np.quantile(residual, 0.70))
    kept = points[residual <= cutoff + 1e-12]
    origin = np.mean(kept, axis=0)
    uv = np.column_stack(((kept - origin) @ horizontal, (kept - origin) @ vertical))
    q = np.quantile(uv, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99], axis=0)
    bounds = {
        "u0": float(q[0, 0]),
        "u1": float(q[-1, 0]),
        "v0": float(q[0, 1]),
        "v1": float(q[-1, 1]),
    }
    pixel.require(bounds["u1"] - bounds["u0"] > 0.05, "PARTIAL_WIDTH_TOO_SMALL")
    pixel.require(bounds["v1"] - bounds["v0"] > 0.05, "PARTIAL_HEIGHT_TOO_SMALL")
    frame = {
        "origin": origin,
        "horizontal": horizontal,
        "vertical": vertical,
        "normal": normal,
        "uv_quantiles": q,
        "retained_points": np.asarray([len(kept)], dtype=np.float64),
        "plane_median": np.asarray([fit["median_residual_m"]], dtype=np.float64),
        "plane_p90": np.asarray([fit["p90_residual_m"]], dtype=np.float64),
    }
    return frame, bounds


def _truth_bounds(points: np.ndarray, frame: dict[str, np.ndarray]) -> dict[str, float]:
    uv = np.column_stack(
        (
            (points - frame["origin"]) @ frame["horizontal"],
            (points - frame["origin"]) @ frame["vertical"],
        )
    )
    q = np.quantile(uv, [0.01, 0.99], axis=0)
    return {"u0": float(q[0, 0]), "u1": float(q[1, 0]), "v0": float(q[0, 1]), "v1": float(q[1, 1])}


def _features(
    frame: dict[str, np.ndarray],
    bounds: dict[str, float],
    view_rows: list[dict[str, Any]],
) -> np.ndarray:
    width = bounds["u1"] - bounds["u0"]
    height = bounds["v1"] - bounds["v0"]
    q = frame["uv_quantiles"]
    boxes = np.asarray(
        [
            [
                float(row["bbox_xyxy"][0]) / float(row["color_size"][0]),
                float(row["bbox_xyxy"][1]) / float(row["color_size"][1]),
                float(row["bbox_xyxy"][2]) / float(row["color_size"][0]),
                float(row["bbox_xyxy"][3]) / float(row["color_size"][1]),
            ]
            for row in view_rows
        ],
        dtype=np.float64,
    )
    normalized_u = (q[:, 0] - bounds["u0"]) / width
    normalized_v = (q[:, 1] - bounds["v0"]) / height
    return np.asarray(
        [
            math.log(width),
            math.log(height),
            math.log(float(frame["retained_points"][0])),
            float(frame["plane_median"][0]),
            float(frame["plane_p90"][0]),
            normalized_u[1],
            normalized_u[2],
            normalized_u[3],
            normalized_u[4],
            normalized_u[5],
            normalized_v[1],
            normalized_v[2],
            normalized_v[3],
            normalized_v[4],
            normalized_v[5],
            float(len(view_rows)),
            float(np.mean(boxes[:, 0])),
            float(np.mean(boxes[:, 1])),
            float(np.mean(boxes[:, 2])),
            float(np.mean(boxes[:, 3])),
            float(np.min(boxes)),
            float(np.min(1.0 - boxes)),
            float(np.std(boxes[:, 0])),
            float(np.std(boxes[:, 2])),
        ],
        dtype=np.float32,
    )


def _target(partial: dict[str, float], truth: dict[str, float]) -> np.ndarray:
    pw = partial["u1"] - partial["u0"]
    ph = partial["v1"] - partial["v0"]
    pc_u = 0.5 * (partial["u0"] + partial["u1"])
    pc_v = 0.5 * (partial["v0"] + partial["v1"])
    tw = truth["u1"] - truth["u0"]
    th = truth["v1"] - truth["v0"]
    tc_u = 0.5 * (truth["u0"] + truth["u1"])
    tc_v = 0.5 * (truth["v0"] + truth["v1"])
    return np.asarray(
        [(tc_u - pc_u) / pw, (tc_v - pc_v) / ph, math.log(tw / pw), math.log(th / ph)],
        dtype=np.float32,
    )


def _decode(partial: dict[str, float], residual: np.ndarray) -> dict[str, float]:
    pw = partial["u1"] - partial["u0"]
    ph = partial["v1"] - partial["v0"]
    centre_u = 0.5 * (partial["u0"] + partial["u1"]) + float(residual[0]) * pw
    centre_v = 0.5 * (partial["v0"] + partial["v1"]) + float(residual[1]) * ph
    width = pw * math.exp(float(np.clip(residual[2], -1.5, 1.5)))
    height = ph * math.exp(float(np.clip(residual[3], -1.5, 1.5)))
    return {"u0": centre_u - width / 2, "u1": centre_u + width / 2, "v0": centre_v - height / 2, "v1": centre_v + height / 2}


def _rect_iou(a: dict[str, float], b: dict[str, float]) -> float:
    iw = max(0.0, min(a["u1"], b["u1"]) - max(a["u0"], b["u0"]))
    ih = max(0.0, min(a["v1"], b["v1"]) - max(a["v0"], b["v0"]))
    intersection = iw * ih
    union = (a["u1"] - a["u0"]) * (a["v1"] - a["v0"]) + (b["u1"] - b["u0"]) * (b["v1"] - b["v0"]) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _corners(bounds: dict[str, float], frame: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [frame["origin"] + frame["horizontal"] * u + frame["vertical"] * v for u, v in ((bounds["u0"], bounds["v0"]), (bounds["u0"], bounds["v1"]), (bounds["u1"], bounds["v0"]), (bounds["u1"], bounds["v1"]))],
        dtype=np.float64,
    )


def _bbox(points: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> list[float]:
    _, pixels, inside = pixel.project_points(points, pose, intrinsic, width, height)
    selected = pixels[inside]
    pixel.require(len(selected) >= 2, "PROJECTED_EXTENT_TOO_FEW")
    return [float(np.min(selected[:, 0])), float(np.min(selected[:, 1])), float(np.max(selected[:, 0])), float(np.max(selected[:, 1]))]


def _bbox_iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _serialize_frame(frame: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: value.tolist() for key, value in frame.items() if key not in {"uv_quantiles"}}


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for name in ("manifest", "row3_cohort"):
        row = protocol[name]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{name.upper()}_HASH")
    masker = protocol["masker"]
    pixel.require(pixel.sha256(Path(masker["model_path"])) == masker["model_sha256"], "MASKER_HASH")
    manifest = pixel.load_json(HERE / protocol["manifest"]["path"])
    cohort = pixel.load_json(HERE / protocol["row3_cohort"]["path"])
    data_root = Path(protocol["source"]["data_root"])

    seed = int(protocol["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    import torch
    from transformers import Sam2Model, Sam2Processor

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processor = Sam2Processor.from_pretrained(Path(masker["model_root"]), local_files_only=True)
    sam = Sam2Model.from_pretrained(Path(masker["model_root"]), local_files_only=True, use_safetensors=True, dtype=torch.float32).eval().to("cuda:0")

    positives = [row for row in manifest["samples"] if row["role"] == "positive"]
    observed: dict[str, dict[str, Any]] = {}
    scan_geometry: dict[tuple[str, int], np.ndarray] = {}
    opened_rgb: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for sample in positives:
        scan_id = str(sample["scan_id"])
        instance_id = int(sample["instance_id"])
        key = (scan_id, instance_id)
        if key not in scan_geometry:
            geometry = extent.ply_instance_points(data_root / scan_id / "labels.instances.annotated.v2.ply", {instance_id})
            pixel.require(instance_id in geometry, f"INSTANCE_GEOMETRY_MISSING:{scan_id}:{instance_id}")
            scan_geometry[key] = geometry[instance_id]
        frame_number = int(str(sample["zip_member"]).split(".")[0].split("-")[1])
        with zipfile.ZipFile(data_root / scan_id / "sequence.zip") as archive:
            info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
            image, image_hash = _load_image(archive, sample["zip_member"])
            depth = pixel.decode_depth(archive, frame_number)
            pose = pixel.read_pose(archive, frame_number)
        masks, receipt = nids.sam_base._sam_masks(processor, sam, image, [sample["bbox_xyxy"]], image.size, torch, np)
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        points = track._lift(mask, depth, pose, info)
        if len(points) < int(protocol["observation"]["minimum_lifted_points"]):
            dropped.append({"sample_id": sample["sample_id"], "reason": "TOO_FEW_LIFTED_POINTS"})
            continue
        observed[sample["sample_id"]] = {"points": points, "row": sample, "sam": receipt}
        opened_rgb.append({"sample_id": sample["sample_id"], "sha256": image_hash})

    del sam, processor
    gc.collect()
    torch.cuda.empty_cache()

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in observed.values():
        groups[(str(item["row"]["identity_key"]), str(item["row"]["scan_id"]))].append(item)
    examples: list[dict[str, Any]] = []
    maximum_views = int(protocol["observation"]["maximum_union_views"])
    for (identity_key, scan_id), items in sorted(groups.items()):
        split = str(manifest["split_by_identity"][identity_key])
        instance_id = int(items[0]["row"]["instance_id"])
        truth_points = scan_geometry[(scan_id, instance_id)]
        for count in range(1, min(maximum_views, len(items)) + 1):
            for selected in combinations(items, count):
                try:
                    points = np.concatenate([item["points"] for item in selected], axis=0)
                    local_frame, partial = _frame_and_bounds(points)
                    truth = _truth_bounds(truth_points, local_frame)
                except (AssertionError, np.linalg.LinAlgError) as exc:
                    dropped.append({"sample_id": "+".join(item["row"]["sample_id"] for item in selected), "reason": str(exc)})
                    continue
                rows = [item["row"] for item in selected]
                examples.append(
                    {
                        "identity_key": identity_key,
                        "scan_id": scan_id,
                        "split": split,
                        "sample_ids": [row["sample_id"] for row in rows],
                        "x": _features(local_frame, partial, rows),
                        "y": _target(partial, truth),
                        "partial": partial,
                        "truth": truth,
                    }
                )
    train_rows = [row for row in examples if row["split"] == "train"]
    validation_rows = [row for row in examples if row["split"] == "validation"]
    pixel.require(train_rows and validation_rows, "TRAIN_VALIDATION_EMPTY")
    x_train = np.stack([row["x"] for row in train_rows])
    y_train = np.stack([row["y"] for row in train_rows])
    x_validation = np.stack([row["x"] for row in validation_rows])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale
    x_validation = (x_validation - mean) / scale

    hidden = int(protocol["training"]["hidden_width"])
    model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 4))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["training"]["learning_rate"]), weight_decay=float(protocol["training"]["weight_decay"]))
    tx = torch.from_numpy(x_train.astype(np.float32))
    ty = torch.from_numpy(y_train.astype(np.float32))
    vx = torch.from_numpy(x_validation.astype(np.float32))
    best_state = None
    best_epoch = 0
    best_score = -1.0
    patience = int(protocol["training"]["patience"])
    maximum_epochs = int(protocol["training"]["maximum_epochs"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        prediction = model(tx)
        loss = torch.nn.functional.smooth_l1_loss(prediction, ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5:
            continue
        model.eval()
        with torch.no_grad():
            residuals = model(vx).numpy()
        score = float(np.mean([_rect_iou(_decode(row["partial"], residual), row["truth"]) for row, residual in zip(validation_rows, residuals, strict=True)]))
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    pixel.require(best_state is not None, "MODEL_SELECTION_EMPTY")
    model.load_state_dict(best_state)
    model.eval()

    def evaluate(rows: list[dict[str, Any]], values: np.ndarray) -> dict[str, Any]:
        with torch.no_grad():
            residuals = model(torch.from_numpy(values.astype(np.float32))).numpy()
        baseline = np.asarray([_rect_iou(row["partial"], row["truth"]) for row in rows])
        learned = np.asarray([_rect_iou(_decode(row["partial"], residual), row["truth"]) for row, residual in zip(rows, residuals, strict=True)])
        return {
            "example_count": len(rows),
            "identity_count": len({row["identity_key"] for row in rows}),
            "baseline_mean_iou": float(np.mean(baseline)),
            "learned_mean_iou": float(np.mean(learned)),
            "mean_iou_gain": float(np.mean(learned - baseline)),
            "baseline_success_at_0_8": int(np.count_nonzero(baseline >= 0.8)),
            "learned_success_at_0_8": int(np.count_nonzero(learned >= 0.8)),
        }

    train_metrics = evaluate(train_rows, x_train)
    validation_metrics = evaluate(validation_rows, x_validation)

    reference_keys = list(cohort["panel"]["reference_keys"])
    reference_rows = [cohort["images"][key] for key in reference_keys]
    reference_scan = str(cohort["candidate"]["reference_scan_id"])
    row3_points = []
    row3_rgb = []
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(Path(masker["model_root"]), local_files_only=True)
    sam = Sam2Model.from_pretrained(Path(masker["model_root"]), local_files_only=True, use_safetensors=True, dtype=torch.float32).eval().to("cuda:0")
    with zipfile.ZipFile(data_root / reference_scan / "sequence.zip") as archive:
        reference_info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        for row in reference_rows:
            image, image_hash = _load_image(archive, row["zip_member"])
            depth = pixel.decode_depth(archive, int(row["frame"]))
            pose = pixel.read_pose(archive, int(row["frame"]))
            masks, _ = nids.sam_base._sam_masks(processor, sam, image, [row["bbox_xyxy"]], image.size, torch, np)
            row3_points.append(track._lift(np.ascontiguousarray(masks[0], dtype=np.bool_), depth, pose, reference_info))
            row3_rgb.append({"key": row["episode_id"], "sha256": image_hash})
    del sam, processor
    gc.collect()
    torch.cuda.empty_cache()
    row3_union = np.concatenate(row3_points, axis=0)
    row3_frame, row3_partial = _frame_and_bounds(row3_union)
    row3_x = (_features(row3_frame, row3_partial, reference_rows) - mean) / scale
    with torch.no_grad():
        row3_residual = model(torch.from_numpy(row3_x[None].astype(np.float32))).numpy()[0]
    row3_prediction = _decode(row3_partial, row3_residual)

    action_row = cohort["images"][cohort["panel"]["action_query_key"]]
    rescan_id = str(cohort["candidate"]["rescan_id"])
    with zipfile.ZipFile(data_root / rescan_id / "sequence.zip") as archive:
        action_info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
        action_pose = pixel.read_pose(archive, int(action_row["frame"]))
    rescan_to_reference = extent.provider_matrix(cohort["candidate"]["transform"])
    reference_to_rescan = np.linalg.inv(rescan_to_reference)
    partial_rescan = extent.transform_points(_corners(row3_partial, row3_frame), reference_to_rescan)
    predicted_rescan = extent.transform_points(_corners(row3_prediction, row3_frame), reference_to_rescan)
    reference_truth = extent.ply_instance_points(data_root / reference_scan / "labels.instances.annotated.v2.ply", {int(cohort["candidate"]["target_instance_id"])})[int(cohort["candidate"]["target_instance_id"])]
    oracle_rescan = extent.transform_points(reference_truth, reference_to_rescan)
    baseline_bbox = _bbox(partial_rescan, action_pose, action_info["color_intrinsic"], int(action_info["color_width"]), int(action_info["color_height"]))
    predicted_bbox = _bbox(predicted_rescan, action_pose, action_info["color_intrinsic"], int(action_info["color_width"]), int(action_info["color_height"]))
    oracle_bbox = _bbox(oracle_rescan, action_pose, action_info["color_intrinsic"], int(action_info["color_width"]), int(action_info["color_height"]))
    truth_bbox = [float(value) for value in action_row["bbox_xyxy"]]
    baseline_iou = _bbox_iou(baseline_bbox, truth_bbox)
    learned_iou = _bbox_iou(predicted_bbox, truth_bbox)
    oracle_iou = _bbox_iou(oracle_bbox, truth_bbox)
    gate = protocol["gate"]
    gate_met = validation_metrics["mean_iou_gain"] > float(gate["minimum_validation_mean_iou_gain"]) and learned_iou >= float(gate["minimum_row3_action_iou"]) and learned_iou > baseline_iou

    state = {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "TARGET_DISJOINT_TRAIN_VALIDATION_AND_CONSUMED_ROW3_DEVELOPMENT",
        "status": "L10_3RSCAN_PORTAL_EXTENT_RESIDUAL_HEAD_DEVELOPMENT_GATE_MET" if gate_met else "L10_3RSCAN_PORTAL_EXTENT_RESIDUAL_HEAD_DEVELOPMENT_GATE_NOT_MET",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "dataset": {
            "positive_images": len(positives),
            "opened_rgb_members": len(opened_rgb),
            "sam_mask_calls": len(opened_rgb) + len(reference_rows),
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "dropped": dropped,
        },
        "model": {
            "feature_count": int(x_train.shape[1]),
            "hidden_width": hidden,
            "selected_epoch": best_epoch,
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
            "state_dict": state,
        },
        "metrics": {"train": train_metrics, "validation": validation_metrics},
        "row3_consumed_development": {
            "reference_scan_id": reference_scan,
            "action_scan_id": rescan_id,
            "reference_view_count": len(reference_rows),
            "partial_lifted_points": int(len(row3_union)),
            "frame": _serialize_frame(row3_frame),
            "partial_bounds_uv": row3_partial,
            "predicted_bounds_uv": row3_prediction,
            "residual": row3_residual.tolist(),
            "baseline_action_bbox_xyxy": baseline_bbox,
            "learned_action_bbox_xyxy": predicted_bbox,
            "oracle_action_bbox_xyxy": oracle_bbox,
            "truth_action_bbox_xyxy": truth_bbox,
            "baseline_action_iou": baseline_iou,
            "learned_action_iou": learned_iou,
            "oracle_action_iou": oracle_iou,
            "learned_gain": learned_iou - baseline_iou,
            "opened_rgb": row3_rgb,
        },
        "gate": {**gate, "met": gate_met},
        "conclusion": "L10_3RSCAN_PORTAL_EXTENT_RESIDUAL_HEAD_DEVELOPMENT_GATE_MET" if gate_met else "L10_3RSCAN_PORTAL_EXTENT_RESIDUAL_HEAD_DEVELOPMENT_GATE_NOT_MET",
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
