#!/usr/bin/env python3
"""Evaluate three paired FP-aware DDRNet seeds on consumed Development only."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for connected-component evaluation") from exc

from . import CANDIDATE_ID, PROTOCOL_ID
from .contract import (
    PENDING_VALIDATION_TERMINAL,
    validate_config_contract,
    validate_config_sha256,
)
from .models import DDRNet23SlimSegmenter, load_exact_checkpoint, sha256_file


HAZARD_IDS = (1, 2)
EVALUATION_ROLES = ("consumed_old_blind", "r1_consumed_fresh")


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"zero-row input: {path}")
    return rows


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    text = json.dumps(value, ensure_ascii=False, indent=2 if pretty else None, separators=separators)
    return (text + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(json_bytes(value, pretty=True))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def pack_ids(ids: np.ndarray) -> str:
    value = np.asarray(ids, dtype=np.uint8)
    if value.shape != (256, 256):
        raise ValueError(f"prediction shape must be 256x256, got {value.shape}")
    return base64.b64encode(zlib.compress(value.tobytes(), level=9)).decode("ascii")


def unpack_ids(value: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(value))
    result = np.frombuffer(raw, dtype=np.uint8)
    if result.size != 256 * 256:
        raise ValueError("packed prediction has an invalid element count")
    return result.reshape(256, 256)


def pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError("pixel metric masks must be equal two-dimensional arrays")
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(pred.size - tp - fp - fn)
    empty = tp + fp + fn == 0
    precision = float(tp / (tp + fp)) if tp + fp else (1.0 if empty else None)
    recall = float(tp / (tp + fn)) if tp + fn else (1.0 if empty else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": int(pred.sum()),
        "truth_pixels": int(target.sum()),
        "precision": precision,
        "recall": recall,
        "false_positive_area_fraction": float(fp / pred.size),
    }


def aggregate_confusion(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate empty confusion rows")
    tp = sum(int(row["tp"]) for row in values)
    fp = sum(int(row["fp"]) for row in values)
    fn = sum(int(row["fn"]) for row in values)
    tn = sum(int(row["tn"]) for row in values)
    total = tp + fp + fn + tn
    empty = tp + fp + fn == 0
    precision = float(tp / (tp + fp)) if tp + fp else (1.0 if empty else None)
    recall = float(tp / (tp + fn)) if tp + fn else (1.0 if empty else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": sum(int(row["predicted_pixels"]) for row in values),
        "truth_pixels": sum(int(row["truth_pixels"]) for row in values),
        "pixel_count": total,
        "precision": precision,
        "recall": recall,
        "false_positive_area_fraction": float(fp / total),
    }


def component_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, pred_count = ndimage.label(np.asarray(predicted, dtype=bool), structure=structure)
    truth_labels, truth_count = ndimage.label(np.asarray(truth, dtype=bool), structure=structure)
    pred_hits = 0
    for label_id in range(1, int(pred_count) + 1):
        if np.any(truth_labels[pred_labels == label_id] > 0):
            pred_hits += 1
    truth_hits = 0
    for label_id in range(1, int(truth_count) + 1):
        if np.any(pred_labels[truth_labels == label_id] > 0):
            truth_hits += 1
    return {
        "predicted_component_count": int(pred_count),
        "truth_component_count": int(truth_count),
        "hit_predicted_component_count": pred_hits,
        "hit_truth_component_count": truth_hits,
        "false_activation_component_count": int(pred_count) - pred_hits,
    }


def aggregate_components(rows: Iterable[dict[str, Any]], frame_count: int) -> dict[str, Any]:
    values = list(rows)
    predicted = sum(int(row["predicted_component_count"]) for row in values)
    truth = sum(int(row["truth_component_count"]) for row in values)
    hit_predicted = sum(int(row["hit_predicted_component_count"]) for row in values)
    hit_truth = sum(int(row["hit_truth_component_count"]) for row in values)
    false_count = sum(int(row["false_activation_component_count"]) for row in values)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": float(hit_predicted / predicted) if predicted else (1.0 if truth == 0 else None),
        "component_recall": float(hit_truth / truth) if truth else (1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / frame_count),
    }


def load_trace(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]))
        if key in result:
            raise ValueError(f"duplicate trace identity: {key}")
        detections = row.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"trace detections must be a list: {key}")
        result[key] = row
    return result


def box_union_mask(
    detections: Iterable[dict[str, Any]],
    *,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    mask = np.zeros((256, 256), dtype=bool)
    for detection in detections:
        left = max(0.0, min(float(source_width), float(detection["left"])))
        right = max(0.0, min(float(source_width), float(detection["right"])))
        top = max(0.0, min(float(source_height), float(detection["top"])))
        bottom = max(0.0, min(float(source_height), float(detection["bottom"])))
        if right <= left or bottom <= top:
            continue
        x0 = max(0, min(256, math.floor(left * 256 / source_width)))
        x1 = max(0, min(256, math.ceil(right * 256 / source_width)))
        y0 = max(0, min(256, math.floor(top * 256 / source_height)))
        y1 = max(0, min(256, math.ceil(bottom * 256 / source_height)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def load_evaluation_inputs(
    repo_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
    view_manifest = resolve(repo_root, config["evaluation"]["canonical_view_manifest"]["path"])
    if sha256_file(view_manifest) != config["evaluation"]["canonical_view_manifest"]["sha256"]:
        raise ValueError("canonical view manifest SHA256 mismatch")
    selected = [row for row in read_jsonl(view_manifest) if row.get("role") in EVALUATION_ROLES]
    expected_counts = config["evaluation"]["role_counts"]
    actual_counts = {
        role: sum(row.get("role") == role for row in selected) for role in EVALUATION_ROLES
    }
    if actual_counts != expected_counts or len(selected) != 320:
        raise ValueError(f"consumed evaluation membership mismatch: {actual_counts}")
    traces: dict[tuple[str, int, str], dict[str, Any]] = {}
    for role in EVALUATION_ROLES:
        binding = config["evaluation"]["yolo_traces"][role]
        path = resolve(repo_root, binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"{role} YOLO trace SHA256 mismatch")
        local = load_trace(path)
        overlap = set(traces) & set(local)
        if overlap:
            raise ValueError(f"YOLO trace identities overlap across roles: {list(overlap)[:3]}")
        traces.update(local)
    selected_keys = {
        (str(row["source_id"]), int(row["frame_id"]), str(row["image_sha256"]))
        for row in selected
    }
    if selected_keys != set(traces):
        raise ValueError(
            f"evaluation/trace membership mismatch: "
            f"missing={len(selected_keys - set(traces))} extra={len(set(traces) - selected_keys)}"
        )
    return sorted(selected, key=lambda row: (str(row["role"]), str(row["session_id"]), int(row["frame_id"]))), traces


def load_images_and_truth(
    repo_root: Path,
    view_root: Path,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    images: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    source_sizes: list[tuple[int, int]] = []
    for row in rows:
        image_path = resolve(repo_root, row["image_repo_relative_path"])
        truth_path = view_root / row["canonical_mask_path"]
        if sha256_file(image_path) != row["image_sha256"]:
            raise ValueError(f"image SHA256 mismatch: {row['id']}")
        if sha256_file(truth_path) != row["canonical_mask_sha256"]:
            raise ValueError(f"truth SHA256 mismatch: {row['id']}")
        with Image.open(image_path) as image:
            source_sizes.append(image.size)
            rgb = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
            images.append(np.asarray(rgb, dtype=np.float32))
        with Image.open(truth_path) as image:
            if image.mode != "L" or image.size != (256, 256):
                raise ValueError(f"canonical truth contract mismatch: {row['id']}")
            truth = np.asarray(image, dtype=np.uint8)
        if np.any(truth > 3):
            raise ValueError(f"truth ID outside 0..3: {row['id']}")
        truths.append(truth)
    return np.stack(images), truths, source_sizes


def predict_checkpoint(
    *,
    architecture: Path,
    source_checkpoint: Path,
    checkpoint: Path,
    images: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = DDRNet23SlimSegmenter(architecture, source_checkpoint).to(device)
    load_exact_checkpoint(model, checkpoint)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(images[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            ).permute(0, 3, 1, 2)
            predictions.append(model(tensor).argmax(dim=1).cpu().numpy().astype(np.uint8))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(predictions, axis=0)


def validate_checkpoint_payload(
    checkpoint: Path,
    *,
    expected_seed: int,
    candidate: bool,
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a mapping: {checkpoint}")
    if payload.get("model_id") != "DDRNet-23-Slim":
        raise ValueError(f"checkpoint model identity mismatch: {checkpoint}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(f"checkpoint seed identity mismatch: {checkpoint}")
    if candidate:
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"candidate checkpoint protocol mismatch: {checkpoint}")
        if payload.get("candidate_id") != CANDIDATE_ID:
            raise ValueError(f"candidate checkpoint identity mismatch: {checkpoint}")


def build_frame_row(
    *,
    manifest: dict[str, Any],
    trace: dict[str, Any],
    source_size: tuple[int, int],
    truth_ids: np.ndarray,
    predicted_ids: np.ndarray,
    seed: int,
    arm: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    detector = box_union_mask(
        trace["detections"],
        source_width=source_size[0],
        source_height=source_size[1],
    )
    raw_hazard = np.isin(predicted_ids, HAZARD_IDS)
    candidate = raw_hazard & ~detector
    truth_hazard = np.isin(truth_ids, HAZARD_IDS)
    residual_truth = truth_hazard & ~detector
    arm_a = detector
    arm_c = detector | candidate
    return {
        "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.frame.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "formal_authority": False,
        "seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
        "view_row_id": manifest["id"],
        "role": manifest["role"],
        "source_id": manifest["source_id"],
        "session_id": manifest["session_id"],
        "frame_id": int(manifest["frame_id"]),
        "image_sha256": manifest["image_sha256"],
        "canonical_mask_sha256": manifest["canonical_mask_sha256"],
        "predicted_ids_zlib_base64": pack_ids(predicted_ids),
        "metrics": {
            "A": pixel_metrics(arm_a, truth_hazard),
            "C": pixel_metrics(arm_c, truth_hazard),
            "candidate": pixel_metrics(candidate, residual_truth),
            "boundary_candidate": pixel_metrics(
                (predicted_ids == 1) & ~detector,
                (truth_ids == 1) & ~detector,
            ),
            "obstacle_candidate": pixel_metrics(
                (predicted_ids == 2) & ~detector,
                (truth_ids == 2) & ~detector,
            ),
            "components": component_metrics(candidate, residual_truth),
        },
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("zero frame rows")
    metrics = {
        name: aggregate_confusion(row["metrics"][name] for row in rows)
        for name in ("A", "C", "candidate", "boundary_candidate", "obstacle_candidate")
    }
    components = aggregate_components(
        (row["metrics"]["components"] for row in rows),
        frame_count=len(rows),
    )
    metrics["components"] = components
    metrics["delta_recall_C_minus_A"] = float(metrics["C"]["recall"] - metrics["A"]["recall"])
    metrics["delta_false_positive_area_fraction_C_minus_A"] = float(
        metrics["C"]["false_positive_area_fraction"]
        - metrics["A"]["false_positive_area_fraction"]
    )
    return metrics


def aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session_id"])].append(row)
        by_role[str(row["role"])].append(row)
    return {
        "frame_count": len(rows),
        "overall": aggregate_rows(rows),
        "sessions": {
            key: aggregate_rows(value) for key, value in sorted(by_session.items())
        },
        "roles": {key: aggregate_rows(value) for key, value in sorted(by_role.items())},
    }


def safe_retention(candidate: float | int, baseline: float | int, label: str) -> float:
    if float(baseline) <= 0:
        raise ValueError(f"baseline denominator is non-positive for {label}")
    return float(float(candidate) / float(baseline))


def compare_seed(
    *,
    seed: int,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, Any]:
    baseline_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    fp_reduction = 1.0 - safe_retention(
        candidate_overall["candidate"]["fp"],
        baseline_overall["candidate"]["fp"],
        "candidate FP",
    )
    overall_recall_retention = safe_retention(
        candidate_overall["candidate"]["tp"],
        baseline_overall["candidate"]["tp"],
        "overall candidate TP",
    )
    session_retentions = {
        session: safe_retention(
            candidate["sessions"][session]["candidate"]["tp"],
            baseline["sessions"][session]["candidate"]["tp"],
            f"{session} candidate TP",
        )
        for session in sorted(baseline["sessions"])
    }
    boundary_retention = safe_retention(
        candidate_overall["boundary_candidate"]["tp"],
        baseline_overall["boundary_candidate"]["tp"],
        "boundary candidate TP",
    )
    obstacle_retention = safe_retention(
        candidate_overall["obstacle_candidate"]["tp"],
        baseline_overall["obstacle_candidate"]["tp"],
        "obstacle candidate TP",
    )
    false_component_reduction = 1.0 - safe_retention(
        candidate_overall["components"]["false_activation_component_count"],
        baseline_overall["components"]["false_activation_component_count"],
        "false activation components",
    )
    relative_values = {
        "fp_pixel_reduction": fp_reduction,
        "overall_recall_retention": overall_recall_retention,
        "minimum_session_recall_retention": min(session_retentions.values()),
        "boundary_recall_retention": boundary_retention,
        "obstacle_recall_retention": obstacle_retention,
    }
    relative_pass = {
        "fp_pixel_reduction": fp_reduction >= gates["min_fp_pixel_reduction"],
        "overall_recall_retention": overall_recall_retention
        >= gates["min_overall_recall_retention"],
        "minimum_session_recall_retention": min(session_retentions.values())
        >= gates["min_session_recall_retention"],
        "boundary_recall_retention": boundary_retention
        >= gates["min_boundary_recall_retention"],
        "obstacle_recall_retention": obstacle_retention
        >= gates["min_obstacle_recall_retention"],
    }
    absolute_values = {
        "delta_recall_C_minus_A": candidate_overall["delta_recall_C_minus_A"],
        "delta_false_positive_area_fraction_C_minus_A": candidate_overall[
            "delta_false_positive_area_fraction_C_minus_A"
        ],
        "candidate_component_recall": candidate_overall["components"]["component_recall"],
        "false_activation_components_per_frame": candidate_overall["components"][
            "false_activation_components_per_frame"
        ],
    }
    absolute_pass = {
        "delta_recall_C_minus_A": absolute_values["delta_recall_C_minus_A"]
        >= gates["min_delta_recall_C_minus_A"],
        "delta_false_positive_area_fraction_C_minus_A": absolute_values[
            "delta_false_positive_area_fraction_C_minus_A"
        ]
        <= gates["max_delta_false_positive_area_fraction_C_minus_A"],
        "candidate_component_recall": absolute_values["candidate_component_recall"]
        >= gates["min_candidate_component_recall"],
        "false_activation_components_per_frame": absolute_values[
            "false_activation_components_per_frame"
        ]
        <= gates["max_false_activation_components_per_frame"],
    }
    margins = {
        "relative_fp": fp_reduction - gates["min_fp_pixel_reduction"],
        "relative_overall": overall_recall_retention - gates["min_overall_recall_retention"],
        "relative_session": min(session_retentions.values())
        - gates["min_session_recall_retention"],
        "relative_boundary": boundary_retention - gates["min_boundary_recall_retention"],
        "relative_obstacle": obstacle_retention - gates["min_obstacle_recall_retention"],
        "absolute_recall": absolute_values["delta_recall_C_minus_A"]
        - gates["min_delta_recall_C_minus_A"],
        "absolute_fp": gates["max_delta_false_positive_area_fraction_C_minus_A"]
        - absolute_values["delta_false_positive_area_fraction_C_minus_A"],
        "absolute_component_recall": absolute_values["candidate_component_recall"]
        - gates["min_candidate_component_recall"],
        "absolute_false_components": gates["max_false_activation_components_per_frame"]
        - absolute_values["false_activation_components_per_frame"],
    }
    return {
        "seed": seed,
        "relative": relative_values,
        "relative_gate_pass": relative_pass,
        "session_recall_retentions": session_retentions,
        "absolute": absolute_values,
        "absolute_gate_pass": absolute_pass,
        "false_activation_component_reduction": false_component_reduction,
        "all_nine_gates_passed": all(relative_pass.values()) and all(absolute_pass.values()),
        "minimum_gate_margin": min(margins.values()),
        "gate_margins": margins,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    config_path = resolve(repo_root, args.config)
    config = read_json(config_path)
    validate_config_sha256(sha256_file(config_path))
    validate_config_contract(config)
    if args.preflight_only:
        manifest_rows, traces = load_evaluation_inputs(repo_root, config)
        return {
            "status": "EVALUATION_PREFLIGHT_VALID",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "evaluation_frame_count": len(manifest_rows),
            "evaluation_session_count": len({str(row["session_id"]) for row in manifest_rows}),
            "trace_row_count": len(traces),
            "roles": list(EVALUATION_ROLES),
            "truth_pixels_or_candidate_outcomes_accessed": False,
        }
    output_root = resolve(repo_root, args.output_root or config["output"]["evaluation_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_root}")
    training_report_path = resolve(
        repo_root,
        args.training_report or config["output"]["training_report"],
    )
    training_report = read_json(training_report_path)
    if (
        training_report.get("protocol_id") != PROTOCOL_ID
        or training_report.get("candidate_id") != CANDIDATE_ID
        or training_report.get("status") != "TRAINING_COMPLETE"
    ):
        raise ValueError("candidate training report identity/status mismatch")
    if training_report.get("config_sha256") != sha256_file(config_path):
        raise ValueError("candidate training report config binding mismatch")
    if training_report.get("cross_seed_selection") != "FORBIDDEN_NOT_PERFORMED":
        raise ValueError("candidate training report performed or omitted cross-seed selection")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    architecture_binding = config["inputs"]["ddrnet_architecture"]
    source_binding = config["inputs"]["ddrnet_source_checkpoint"]
    architecture = resolve(repo_root, architecture_binding["path"])
    source_checkpoint = resolve(repo_root, source_binding["path"])
    if sha256_file(architecture) != architecture_binding["sha256"]:
        raise ValueError("DDRNet architecture SHA256 mismatch")
    if sha256_file(source_checkpoint) != source_binding["sha256"]:
        raise ValueError("DDRNet source checkpoint SHA256 mismatch")
    manifest_rows, traces = load_evaluation_inputs(repo_root, config)
    view_manifest = resolve(repo_root, config["evaluation"]["canonical_view_manifest"]["path"])
    images, truths, source_sizes = load_images_and_truth(
        repo_root,
        view_manifest.parent,
        manifest_rows,
    )
    all_frame_rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, dict[str, Any]] = {}
    candidate_reports_by_seed = {
        int(report["seed"]): report for report in training_report["seed_reports"]
    }
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline_binding = config["inputs"][f"r1_baseline_seed_{seed}"]
        baseline_checkpoint = resolve(repo_root, baseline_binding["path"])
        if sha256_file(baseline_checkpoint) != baseline_binding["sha256"]:
            raise ValueError(f"baseline seed {seed} checkpoint SHA256 mismatch")
        validate_checkpoint_payload(
            baseline_checkpoint,
            expected_seed=seed,
            candidate=False,
        )
        candidate_report = candidate_reports_by_seed.get(seed)
        if candidate_report is None:
            raise ValueError(f"missing candidate seed report: {seed}")
        candidate_checkpoint = Path(candidate_report["checkpoint"]).resolve()
        if sha256_file(candidate_checkpoint) != candidate_report["checkpoint_sha256"]:
            raise ValueError(f"candidate seed {seed} checkpoint SHA256 mismatch")
        validate_checkpoint_payload(
            candidate_checkpoint,
            expected_seed=seed,
            candidate=True,
        )
        arm_rows: dict[str, list[dict[str, Any]]] = {}
        for arm, checkpoint, checkpoint_sha in (
            ("R1_BASELINE", baseline_checkpoint, baseline_binding["sha256"]),
            ("FP_AWARE_CANDIDATE", candidate_checkpoint, candidate_report["checkpoint_sha256"]),
        ):
            predictions = predict_checkpoint(
                architecture=architecture,
                source_checkpoint=source_checkpoint,
                checkpoint=checkpoint,
                images=images,
                device=device,
                batch_size=int(config["evaluation"]["batch_size"]),
            )
            rows: list[dict[str, Any]] = []
            for manifest, truth, size, prediction in zip(
                manifest_rows,
                truths,
                source_sizes,
                predictions,
                strict=True,
            ):
                key = (
                    str(manifest["source_id"]),
                    int(manifest["frame_id"]),
                    str(manifest["image_sha256"]),
                )
                rows.append(
                    build_frame_row(
                        manifest=manifest,
                        trace=traces[key],
                        source_size=size,
                        truth_ids=truth,
                        predicted_ids=prediction,
                        seed=seed,
                        arm=arm,
                        checkpoint_sha256=checkpoint_sha,
                    )
                )
            arm_rows[arm] = rows
            all_frame_rows.extend(rows)
        baseline_summary = aggregate_model(arm_rows["R1_BASELINE"])
        candidate_summary = aggregate_model(arm_rows["FP_AWARE_CANDIDATE"])
        comparison = compare_seed(
            seed=seed,
            baseline=baseline_summary,
            candidate=candidate_summary,
            gates=config["evaluation"]["gates"],
        )
        seed_summaries[str(seed)] = {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "comparison": comparison,
        }
    all_frame_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["arm"]),
            str(row["role"]),
            str(row["session_id"]),
            int(row["frame_id"]),
        )
    )
    output_root.mkdir(parents=True)
    frames_path = output_root / "frame_predictions.jsonl"
    write_jsonl(frames_path, all_frame_rows)
    comparisons = [seed_summaries[str(int(seed))]["comparison"] for seed in config["training"]["seeds"]]
    all_pass = all(item["all_nine_gates_passed"] for item in comparisons)
    worst = min(comparisons, key=lambda item: (item["minimum_gate_margin"], item["seed"]))
    metric_terminal = (
        "FP_WEIGHTED_SAMPLING_SUPPORTED_DEVELOPMENT_ONLY"
        if all_pass
        else "FP_WEIGHTED_SAMPLING_NOT_SUPPORTED"
    )
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.result.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "status": "EVALUATION_COMPLETE_UNVALIDATED",
        "terminal": PENDING_VALIDATION_TERMINAL,
        "provisional_metric_terminal": metric_terminal,
        "all_three_seeds_passed": all_pass,
        "worst_seed": worst["seed"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_report": str(training_report_path),
        "training_report_sha256": sha256_file(training_report_path),
        "frame_predictions": str(frames_path),
        "frame_predictions_sha256": sha256_file(frames_path),
        "frame_row_count": len(all_frame_rows),
        "evaluation_frame_count_per_arm_seed": len(manifest_rows),
        "evaluation_roles": list(EVALUATION_ROLES),
        "seed_summaries": seed_summaries,
        "decision_rule": {
            "required": "all three same-seed pairs pass every relative and absolute gate",
            "gates": config["evaluation"]["gates"],
            "no_best_seed_selection": True,
        },
        "authority": {
            "stage": "DEVELOPMENT",
            "fresh_or_confirmation_outcome_accessed": False,
            "int8_or_runtime_authority": False,
            "android_or_alert_authority": False,
            "default_app_unchanged": True,
        },
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-report")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal": result.get("terminal"),
                "all_three_seeds_passed": result.get("all_three_seeds_passed"),
                "worst_seed": result.get("worst_seed"),
                "frame_row_count": result.get("frame_row_count"),
                "evaluation_frame_count": result.get("evaluation_frame_count"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
