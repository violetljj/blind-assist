#!/usr/bin/env python3
"""Audit REveL Dynamic RGB frames and YOLO labels with CUDA vector checks.

This is a source-data audit, not a detector evaluation.  The archive supplies
annotated 2D helmet-colour classes, so it can establish external per-frame
2D-object truth and temporal continuity.  It deliberately cannot establish
metric range, physical TTC, body-local clearance, or device safety behaviour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path.name}:{line_number}: expected five YOLO fields")
        try:
            class_id = int(fields[0]); values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"{path.name}:{line_number}: non-numeric label") from error
        if class_id < 0 or not all(np.isfinite(value) for value in values):
            raise ValueError(f"{path.name}:{line_number}: invalid class or non-finite coordinate")
        rows.append((class_id, *values))
    return rows


def _iou_xywh(first: Any, second: Any) -> Any:
    """Vectorised IoU for [cx, cy, width, height] normalised boxes."""
    import torch

    first_min = first[:, :2] - first[:, 2:] / 2; first_max = first[:, :2] + first[:, 2:] / 2
    second_min = second[:, :2] - second[:, 2:] / 2; second_max = second[:, :2] + second[:, 2:] / 2
    intersection = (torch.minimum(first_max, second_max) - torch.maximum(first_min, second_min)).clamp_min(0)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    first_area = first[:, 2] * first[:, 3]; second_area = second[:, 2] * second[:, 3]
    return intersection_area / (first_area + second_area - intersection_area).clamp_min(1e-12)


def _layout(dataset_root: Path) -> tuple[Path, Path]:
    images = dataset_root / "extracted" / "images" / "images"
    labels = dataset_root / "extracted" / "labels" / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("expected extracted/images/images and extracted/labels/labels")
    return images, labels


def audit(dataset_root: Path, image_samples: int = 256) -> dict[str, Any]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the REveL label geometry and motion audit")
    if image_samples < 1:
        raise ValueError("image_samples must be positive")
    images_root, labels_root = _layout(dataset_root)
    image_by_stem = {path.stem: path for path in images_root.glob("*.jpg")}
    label_by_stem = {path.stem: path for path in labels_root.glob("*.txt")}
    common_stems = sorted(image_by_stem.keys() & label_by_stem.keys(), key=int)
    if not common_stems:
        raise ValueError("no image/label pairs")
    rows: list[tuple[str, list[tuple[int, float, float, float, float]]]] = []
    class_counts: Counter[int] = Counter(); boxes: list[tuple[float, float, float, float]] = []
    per_frame_class: dict[tuple[str, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    for stem in common_stems:
        entries = _parse_label(label_by_stem[stem]); rows.append((stem, entries))
        for class_id, cx, cy, width, height in entries:
            class_counts[class_id] += 1; boxes.append((cx, cy, width, height))
            per_frame_class[(stem, class_id)].append((cx, cy, width, height))
    device = torch.device("cuda")
    box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
    valid_geometry = (
        (box_tensor[:, :2] >= 0).all(dim=1) & (box_tensor[:, :2] <= 1).all(dim=1)
        & (box_tensor[:, 2:] > 0).all(dim=1) & (box_tensor[:, 2:] <= 1).all(dim=1)
        & ((box_tensor[:, :2] - box_tensor[:, 2:] / 2) >= 0).all(dim=1)
        & ((box_tensor[:, :2] + box_tensor[:, 2:] / 2) <= 1).all(dim=1)
    )
    timestamps = np.asarray([int(stem) for stem in common_stems], dtype=np.int64)
    deltas_s = np.diff(timestamps).astype(np.float64) / 1_000_000_000.0
    transitions: dict[int, list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]]] = defaultdict(list)
    for previous, current in zip(common_stems, common_stems[1:]):
        for class_id in class_counts:
            before = per_frame_class.get((previous, class_id), []); after = per_frame_class.get((current, class_id), [])
            if len(before) == 1 and len(after) == 1:
                transitions[class_id].append((before[0], after[0]))
    temporal: dict[str, Any] = {}
    for class_id, pairs in sorted(transitions.items()):
        if not pairs:
            continue
        previous = torch.tensor([pair[0] for pair in pairs], dtype=torch.float32, device=device)
        current = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32, device=device)
        centre_delta = torch.linalg.vector_norm(current[:, :2] - previous[:, :2], dim=1)
        iou = _iou_xywh(previous, current)
        temporal[str(class_id)] = {
            "unique_single_box_consecutive_pairs": len(pairs),
            "median_normalized_centre_displacement": float(centre_delta.median().item()),
            "p95_normalized_centre_displacement": float(torch.quantile(centre_delta, .95).item()),
            "median_consecutive_iou": float(iou.median().item()),
        }
    sample_indexes = np.unique(np.linspace(0, len(common_stems) - 1, num=min(image_samples, len(common_stems)), dtype=int))
    image_sizes = Counter()
    for index in sample_indexes:
        with Image.open(image_by_stem[common_stems[int(index)]]) as image:
            image_sizes[f"{image.width}x{image.height}"] += 1
    report = {
        "format": "blindassist_revel_dynamic_rgb_labels_audit_v1",
        "dataset": "REveL Dynamic images.zip + labels.zip",
        "pairing": {
            "image_files": len(image_by_stem), "label_files": len(label_by_stem), "paired_frames": len(common_stems),
            "image_without_label": len(image_by_stem) - len(common_stems), "label_without_image": len(label_by_stem) - len(common_stems),
        },
        "temporal": {
            "first_timestamp_ns": int(timestamps[0]), "last_timestamp_ns": int(timestamps[-1]),
            "median_frame_interval_s": float(np.median(deltas_s)), "p95_frame_interval_s": float(np.quantile(deltas_s, .95)),
            "median_frame_rate_hz": float(1 / np.median(deltas_s)), "class_conditional_single_box_motion": temporal,
        },
        "labels": {
            "nonempty_label_frames": sum(bool(entries) for _, entries in rows), "annotated_boxes": len(boxes),
            "boxes_by_helmet_colour_class": {str(key): value for key, value in sorted(class_counts.items())},
            "valid_normalized_box_fraction": float(valid_geometry.float().mean().item()),
        },
        "sampled_image_dimensions": {"sample_count": len(sample_indexes), "sizes": dict(image_sizes)},
        "archives": {name: {"bytes": (dataset_root / name).stat().st_size, "sha256": _sha256(dataset_root / name)} for name in ("images.zip", "labels.zip")},
        "admission": {
            "external_2d_dynamic_object_truth_admitted": True,
            "admitted_for": ["offline 2D target-label integrity", "class-conditional 2D temporal continuity metrics"],
            "not_admitted_for": ["metric depth", "physical TTC", "body-local safe corridor", "assistive event truth", "on-device safety"],
            "reason": "downloaded archives contain RGB images and 2D YOLO labels, but not the Dynamic bag's Vicon/LiDAR/IMU timing and calibration data",
        },
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)},
        "production_authority": False,
    }
    qa = dataset_root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "revel_dynamic_rgb_labels_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-samples", type=int, default=256)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.image_samples)
    print(json.dumps({"paired_frames": report["pairing"]["paired_frames"], "boxes": report["labels"]["annotated_boxes"], "device": report["compute_backend"]["device"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
