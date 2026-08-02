#!/usr/bin/env python3
"""Run a real semantic-segmentation model and emit a frame-keyed sidecar.

The sidecar contains only normalized discovery signals. It deliberately keeps
the semantic model, label map, input trace hash, and device in a separate
manifest so a segmentation response cannot be confused with the old image
space risk proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from scripts.research.candidate_event_mining.pipeline import ContractError, read_jsonl, sha256_file, write_json, write_jsonl


SIDECAR_SCHEMA = "blindassist_candidate_event_mining_segmentation_sidecar_manifest_v1"
ALLOWED_WALKABLE = ("floor", "road", "sidewalk", "path")
OBSTACLE_LABELS = (
    "person",
    "car",
    "truck",
    "bus",
    "motorbike",
    "bicycle",
    "wall",
    "building",
    "fence",
    "pole",
    "streetlight",
    "door",
    "table",
    "chair",
    "tree",
    "plant",
    "signboard",
)


def _score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"non-finite segmentation score: {value}")
    return max(0.0, min(1.0, value))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ContractError("segmentation sidecar input trace is empty")
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("session_id", "")), int(row.get("frame_index", -1)))
        if not all(key[:2]) or key[2] < 0:
            raise ContractError(f"invalid segmentation frame key: {key}")
        if key in keys:
            raise ContractError(f"duplicate segmentation frame key: {key}")
        keys.add(key)
        frame_ref = Path(str(row.get("frame_ref", ""))).resolve()
        if not frame_ref.is_file():
            raise ContractError(f"segmentation frame is missing: {frame_ref}")
    return rows


def _region_mean(values: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    height, width = values.shape
    iy0 = max(0, min(height - 1, int(round(y0 * height))))
    iy1 = max(iy0 + 1, min(height, int(round(y1 * height))))
    ix0 = max(0, min(width - 1, int(round(x0 * width))))
    ix1 = max(ix0 + 1, min(width, int(round(x1 * width))))
    return float(values[iy0:iy1, ix0:ix1].mean())


def _semantic_features(
    probabilities: np.ndarray,
    label_ids: dict[str, int],
) -> dict[str, float]:
    if probabilities.ndim != 3 or not np.isfinite(probabilities).all():
        raise ContractError("SegFormer probabilities must be finite CxHxW")
    walkable_ids = [label_ids[name] for name in ALLOWED_WALKABLE if name in label_ids]
    if not walkable_ids:
        raise ContractError("semantic model has no walkable ADE20K labels")
    obstacle_ids = [label_ids[name] for name in OBSTACLE_LABELS if name in label_ids]
    walkable = np.clip(probabilities[walkable_ids].sum(axis=0), 0.0, 1.0)
    nonwalkable = np.clip(1.0 - walkable, 0.0, 1.0)
    obstacle = (
        np.clip(probabilities[obstacle_ids].sum(axis=0), 0.0, 1.0)
        if obstacle_ids
        else np.zeros_like(walkable)
    )
    lower_center = _region_mean(nonwalkable, 0.58, 1.0, 0.22, 0.78)
    far_center = _region_mean(nonwalkable, 0.42, 0.62, 0.35, 0.65)
    center_obstacle = _region_mean(obstacle, 0.42, 1.0, 0.22, 0.78)
    lower_obstacle = _region_mean(obstacle, 0.62, 1.0, 0.15, 0.85)
    left = _region_mean(nonwalkable, 0.58, 0.98, 0.05, 0.22)
    right = _region_mean(nonwalkable, 0.58, 0.98, 0.78, 0.95)
    vertical_boundary = float(np.abs(np.diff(walkable, axis=0)).mean())
    horizontal_boundary = float(np.abs(np.diff(walkable, axis=1)).mean())
    boundary = _score(6.0 * (vertical_boundary + horizontal_boundary))
    return {
        "segmentation.alert": _score(0.45 * lower_center + 0.35 * center_obstacle + 0.20 * boundary),
        "segmentation.risk": _score(0.55 * lower_center + 0.45 * lower_obstacle),
        "segmentation.front_risk": _score(0.60 * lower_center + 0.40 * center_obstacle),
        "segmentation.boundary_level_change": _score(0.55 * boundary + 0.45 * max(0.0, lower_center - far_center)),
        "segmentation.parallel_curb": _score(0.5 * (left + right) + 0.5 * abs(left - right)),
        "segmentation.walkable_support": _score(1.0 - lower_center),
        "segmentation.obstacle_area": _score(lower_obstacle),
        "segmentation.semantic_boundary": boundary,
    }


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, torch.device, dict[str, Any]]:
    source = str(args.model_source)
    local_source = Path(source).resolve() if Path(source).is_dir() else None
    load_kwargs: dict[str, Any] = {"use_fast": False}
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir.resolve())
    processor = AutoImageProcessor.from_pretrained(source, **load_kwargs)
    model = SegformerForSemanticSegmentation.from_pretrained(
        source,
        cache_dir=(str(args.cache_dir.resolve()) if args.cache_dir is not None else None),
    )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ContractError("--device cuda requested but CUDA is unavailable")
    device = torch.device(device_name)
    model = model.to(device).eval()
    materialized = None
    if args.materialize_model_dir is not None:
        materialized = args.materialize_model_dir.resolve()
        materialized.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(materialized)
        model.save_pretrained(materialized)
    labels = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    label_ids = {name: index for index, name in labels.items()}
    required = [name for name in ALLOWED_WALKABLE if name not in label_ids]
    if len(required) == len(ALLOWED_WALKABLE):
        raise ContractError("semantic model label map has no known walkable labels")
    return processor, model, device, {
        "source": source,
        "local_source": str(local_source) if local_source else None,
        "materialized_model_dir": str(materialized) if materialized else None,
        "runtime": "transformers.SegformerForSemanticSegmentation",
        "model_type": str(model.config.model_type),
        "revision": getattr(model.config, "_commit_hash", None),
        "label_map_sha256": _sha256_json(labels),
        "label_ids": labels,
        "walkable_labels_used": [name for name in ALLOWED_WALKABLE if name in label_ids],
        "obstacle_labels_used": [name for name in OBSTACLE_LABELS if name in label_ids],
        "device": str(device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.frame_trace.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest_output.resolve()
    for path in (output, manifest_path):
        if path.exists():
            raise ContractError(f"refusing to overwrite output: {path}")
    rows = _read_trace(input_path)
    processor, model, device, model_meta = _load_model(args)
    sidecar_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = []
        for row in batch_rows:
            image = cv2.imread(str(Path(str(row["frame_ref"])).resolve()), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ContractError(f"cannot decode segmentation frame: {row['frame_ref']}")
            images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1).cpu()
            # Upsampling is not needed for an absolute semantic-region score;
            # keep the model output grid and use normalized region coordinates.
            probabilities_np = probabilities.numpy()
        for row, probability in zip(batch_rows, probabilities_np):
            sidecar_rows.append(
                {
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "frame_index": int(row["frame_index"]),
                    "signals": _semantic_features(
                        probability,
                        {label: int(index) for index, label in model_meta["label_ids"].items()},
                    ),
                }
            )
    write_jsonl(output, sidecar_rows)
    manifest = {
        "schema": SIDECAR_SCHEMA,
        "sidecar_id": "cem-real-segformer-ade20k-r0",
        "input_trace": {"path": str(input_path), "sha256": sha256_file(input_path), "frame_count": len(rows)},
        "sidecar": {"path": str(output), "sha256": sha256_file(output), "frame_count": len(sidecar_rows)},
        "model": model_meta,
        "real_semantic_segmentation": True,
        "image_space_risk_proxy": False,
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "authorization": {"event_truth": False, "training": False, "production": False, "safety": False},
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-trace", type=Path, required=True)
    parser.add_argument("--model-source", required=True, help="Local model directory or Hugging Face model id")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"F:\ba-data\blindassist-candidate-event-mining\models\hf-cache"))
    parser.add_argument("--materialize-model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (ContractError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "frame_count": manifest["sidecar"]["frame_count"], "output": manifest["sidecar"]["path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
