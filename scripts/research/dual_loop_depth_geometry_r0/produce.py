"""Produce truth-blind target-ROI relative-depth summaries on fixed REveL RGB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


SCHEMA = "blindassist.dual_loop_target_depth_features.v1"
EXPECTED_FRAME_ROWS = 512
EXPECTED_TARGET_ROWS = 770
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def quantile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction))


def target_summary(
    depth: np.ndarray,
    box: list[float],
    frame_quantiles: tuple[float, float, float],
) -> dict[str, float]:
    height, width = depth.shape
    left = max(0, min(width - 1, int(math.floor(box[0] * width))))
    top = max(0, min(height - 1, int(math.floor(box[1] * height))))
    right = max(left + 1, min(width, int(math.ceil(box[2] * width))))
    bottom = max(top + 1, min(height, int(math.ceil(box[3] * height))))
    roi = depth[top:bottom, left:right]
    finite = roi[np.isfinite(roi)]
    if finite.size == 0:
        raise ValueError("target ROI has no finite relative-depth value")
    q25 = quantile(finite, 0.25)
    median = quantile(finite, 0.50)
    q75 = quantile(finite, 0.75)
    frame_q10, frame_q50, frame_q90 = frame_quantiles
    span = frame_q90 - frame_q10
    normalized = (median - frame_q10) / span if span > 1e-12 else 0.5
    return {
        "roi_depth_q25": q25,
        "roi_depth_median": median,
        "roi_depth_q75": q75,
        "roi_depth_iqr": q75 - q25,
        "roi_depth_frame_normalized": float(min(1.0, max(0.0, normalized))),
        "roi_pixel_count": int(finite.size),
        "frame_depth_q10": frame_q10,
        "frame_depth_q50": frame_q50,
        "frame_depth_q90": frame_q90,
    }


def produce(
    details_path: Path,
    image_root: Path,
    output_path: Path,
    receipt_path: Path,
    model_id: str,
    batch_size: int,
) -> dict[str, Any]:
    rows = read_jsonl(details_path)
    if len(rows) != EXPECTED_FRAME_ROWS:
        raise ValueError(f"expected {EXPECTED_FRAME_ROWS} frame rows, found {len(rows)}")
    target_count = sum(len(row.get("ground_truth", [])) for row in rows)
    if target_count != EXPECTED_TARGET_ROWS:
        raise ValueError(f"expected {EXPECTED_TARGET_ROWS} target rows, found {target_count}")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("discovery output namespace already exists")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
    model_revision = getattr(model.config, "_commit_hash", None)
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    processed_frames = 0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images: list[Image.Image] = []
        for row in batch_rows:
            image_path = image_root / str(row["image_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            images.append(Image.open(image_path).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            predictions = model(**inputs).predicted_depth.detach().float().cpu().numpy()
        for row, depth in zip(batch_rows, predictions, strict=True):
            finite_frame = depth[np.isfinite(depth)]
            if finite_frame.size == 0:
                raise ValueError("frame has no finite relative-depth value")
            frame_quantiles = (
                quantile(finite_frame, 0.10),
                quantile(finite_frame, 0.50),
                quantile(finite_frame, 0.90),
            )
            for target in row.get("ground_truth", []):
                summary = target_summary(depth, target["xyxy_normalized"], frame_quantiles)
                output_rows.append(
                    {
                        "schema": SCHEMA,
                        "selected_index": int(row["selected_index"]),
                        "image_name": str(row["image_name"]),
                        "source_timestamp_ns": int(row["source_timestamp_ns"]),
                        "normalized_area": float(target["normalized_area"]),
                        "target_box_xyxy_normalized": [
                            float(value) for value in target["xyxy_normalized"]
                        ],
                        "model_id": model_id,
                        "model_revision": model_revision,
                        "larger_output_means_closer": True,
                        **summary,
                    }
                )
        processed_frames += len(batch_rows)

    atomic_write_jsonl(output_path, output_rows)
    receipt = {
        "schema": "blindassist.dual_loop_target_depth_producer_receipt.v1",
        "status": "COMPLETE",
        "vicon_truth_opened": False,
        "oracle_roi_opened": True,
        "details_path": details_path.as_posix(),
        "details_sha256": sha256_file(details_path),
        "image_root": image_root.as_posix(),
        "model_id": model_id,
        "model_revision": model_revision,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "batch_size": batch_size,
        "frame_rows": processed_frames,
        "target_rows": len(output_rows),
        "wall_seconds": time.perf_counter() - started,
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    # REveL contains heterogeneous aspect ratios; a batch larger than one would
    # require padding that changes the relative-depth geometry.
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            produce(
                args.details,
                args.image_root,
                args.output,
                args.receipt,
                args.model_id,
                args.batch_size,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
