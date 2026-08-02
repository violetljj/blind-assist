#!/usr/bin/env python3
"""Extract front-crop MobileNet spatial maps for JRDB replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_PRETRAINED,
    MEAN,
    STD,
    MobileNetFeatures,
    load_jsonl,
    sha256,
)


def front_crop_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    if image_bgr.shape[:2] != (480, 3760):
        raise ValueError(
            f"Expected JRDB stitched image 3760x480, got {image_bgr.shape}"
        )
    crop_width = 1254
    start = (image_bgr.shape[1] - crop_width) // 2
    crop = image_bgr[:, start : start + crop_width]
    resized = cv2.resize(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        (224, 128),
        interpolation=cv2.INTER_LINEAR,
    )
    value = resized.astype(np.float32) / 255.0
    value = (value - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".json").exists():
        raise ValueError("Refusing to overwrite JRDB feature cache")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 104:
        raise ValueError("Expected 104 JRDB replication samples")
    expected_hashes = {}
    for record in records:
        for path, digest in zip(
            record["history_image_paths"],
            record["history_image_sha256"],
        ):
            expected_hashes[str(Path(path).resolve())] = str(digest)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = MobileNetFeatures(args.pretrained).to(device).eval()
    features_by_path: dict[str, np.ndarray] = {}
    paths = sorted(expected_hashes)
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        frames = []
        for path_text in batch_paths:
            path = Path(path_text)
            if sha256(path) != expected_hashes[path_text]:
                raise ValueError(f"JRDB image hash mismatch: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Unable to decode JRDB image: {path}")
            frames.append(front_crop_tensor(image))
        with torch.inference_mode():
            values = model.features(
                torch.stack(frames).to(device, non_blocking=True)
            )
        if values.shape[1:] != (576, 4, 7):
            raise ValueError(
                f"Unexpected spatial feature shape: {values.shape}"
            )
        for path_text, feature in zip(
            batch_paths,
            values.cpu().numpy(),
        ):
            features_by_path[path_text] = feature.astype(np.float16)

    matrix = np.stack(
        [
            np.stack(
                [
                    features_by_path[str(Path(path).resolve())]
                    for path in record["history_image_paths"]
                ]
            )
            for record in records
        ]
    )
    if matrix.shape != (104, 5, 576, 4, 7):
        raise ValueError(f"Unexpected JRDB feature matrix: {matrix.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(
            [record["sample_id"] for record in records]
        ),
        features=matrix,
    )
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d9_jrdb_"
            "front_crop_spatial_feature_cache_v0"
        ),
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "design": {
            "source_image": "JRDB 3760x480 image_stitched RGB360",
            "front_crop": (
                "centered 1254-pixel horizontal crop, approximately 120 deg"
            ),
            "resize": [128, 224],
            "feature_shape": list(matrix.shape),
            "storage_dtype": str(matrix.dtype),
        },
        "device": str(device),
        "unique_image_count": len(paths),
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
        },
    }
    report_path = Path(str(args.output) + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "device": str(device),
                "unique_images": len(paths),
                "feature_shape": list(matrix.shape),
                "feature_sha256": report["output"]["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
