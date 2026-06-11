from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SRC = ".downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_CHECKPOINT = ".downloads/depth-lab/checkpoints/depth_anything_v2_vits.pth"
DEFAULT_DATASET = "test-artifacts.local/datasets/blindassist-evalset-20260527-impl"
DEFAULT_OUTPUT = "test-artifacts.local/depth-fusion/smoke-depth-anything-v2-pytorch.json"


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def model_config(encoder: str) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


def manifest_rows(dataset_root: Path, limit: int) -> list[dict[str, Any]]:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"BlindAssist manifest not found: {manifest}")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_value = row.get("image_path") or row.get("image") or row.get("file_name") or row.get("relative_path")
        if not image_value:
            continue
        image_path = dataset_root / image_value
        if image_path.is_file():
            row["_image_path"] = image_path
            rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        raise AssertionError(f"No evalset images found under {dataset_root}")
    return rows


def summarize(values: np.ndarray) -> dict[str, Any]:
    flat = values.astype(np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"finite": 0, "min": None, "max": None, "mean": None, "all_zero": True}
    return {
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "all_zero": bool(np.allclose(finite, 0.0)),
    }


def bbox_summary(depth: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
    h, w = depth.shape[:2]
    objects = row.get("objects") or []
    sampled = 0
    invalid = 0
    primary_object_id = row.get("primary_object_id")
    primary_mean = None
    object_means: list[float] = []
    for obj in objects:
        bbox = obj.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            invalid += 1
            continue
        x1, y1, x2, y2 = bbox
        left = max(0, min(w - 1, int(np.floor(x1))))
        top = max(0, min(h - 1, int(np.floor(y1))))
        right = max(left + 1, min(w, int(np.ceil(x2))))
        bottom = max(top + 1, min(h, int(np.ceil(y2))))
        patch = depth[top:bottom, left:right]
        stats = summarize(patch)
        if stats["finite"] <= 0 or stats["all_zero"]:
            invalid += 1
            continue
        sampled += 1
        mean = float(stats["mean"])
        object_means.append(mean)
        if obj.get("id") == primary_object_id:
            primary_mean = mean
    return {
        "objects": len(objects),
        "bbox_regions_sampled": sampled,
        "bbox_regions_invalid": invalid,
        "bbox_mean_min": float(np.min(object_means)) if object_means else None,
        "bbox_mean_max": float(np.max(object_means)) if object_means else None,
        "primary_object_mean": primary_mean,
    }


def load_model(src_root: Path, checkpoint: Path, encoder: str):
    import torch

    sys.path.insert(0, str(src_root))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    model = DepthAnythingV2(**model_config(encoder))
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test official Depth Anything V2 PyTorch weights on BlindAssist EvalSet.")
    parser.add_argument("--src-root", default=DEFAULT_SRC)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=252)
    parser.add_argument("--image-limit", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    src_root = resolve(project_root, args.src_root)
    checkpoint = resolve(project_root, args.checkpoint)
    dataset_root = resolve(project_root, args.dataset_root)
    output = resolve(project_root, args.output)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Depth Anything V2 source root not found: {src_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {checkpoint}")

    import cv2
    import torch

    model = load_model(src_root, checkpoint, args.encoder)
    rows = []
    timings_ms = []
    with torch.no_grad():
        for row in manifest_rows(dataset_root, args.image_limit):
            image_path = row["_image_path"]
            raw = cv2.imread(str(image_path))
            if raw is None:
                raise AssertionError(f"Failed to read image: {image_path}")
            start = time.perf_counter()
            depth = model.infer_image(raw, input_size=args.input_size)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
            stats = summarize(depth)
            if stats["finite"] <= 0 or stats["all_zero"]:
                raise AssertionError(f"Invalid depth output for {image_path.name}: {stats}")
            bbox_stats = bbox_summary(depth, row)
            if bbox_stats["objects"] > 0 and bbox_stats["bbox_regions_sampled"] <= 0:
                raise AssertionError(f"No valid bbox depth samples for {image_path.name}: {bbox_stats}")
            rows.append(
                {
                    "image": image_path.name,
                    "elapsed_ms": elapsed_ms,
                    "depth_shape": list(depth.shape),
                    **stats,
                    **bbox_stats,
                }
            )

    payload = {
        "source": "Depth Anything V2 Small PyTorch checkpoint",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "src_root": str(src_root.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "encoder": args.encoder,
        "input_size": args.input_size,
        "image_count": len(rows),
        "elapsed_ms_mean": float(np.mean(timings_ms)) if timings_ms else None,
        "elapsed_ms_p95": float(np.percentile(timings_ms, 95)) if timings_ms else None,
        "assertions": "passed",
        "images": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
