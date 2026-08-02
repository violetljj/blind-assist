#!/usr/bin/env python3
"""Extract frozen MobileNet spatial maps for TartanGround onset samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_PRETRAINED,
    MobileNetFeatures,
    frame_tensor,
    load_jsonl,
    sha256,
)
from materialize_stage_c_d16_tartanground_future_onset import (
    DEFAULT_FOLD_MANIFEST,
)


DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d16-tartanground-future-onset-v0/samples.jsonl"
)
SCHEMA = (
    "blindassist_hftf_stage_c_d16_tartanground_"
    "mobilenet_spatial_feature_cache_v0"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    report_path = Path(str(args.output) + ".json")
    if args.output.exists() or report_path.exists():
        raise ValueError("Refusing to overwrite D16 feature cache")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 495:
        raise ValueError("Expected 495 D16 onset samples")
    expected_hashes: dict[str, str] = {}
    for record in records:
        if len(record["history_rgb"]) != 5:
            raise ValueError("Expected five history RGB frames")
        for item in record["history_rgb"]:
            path_text = str(Path(item["image_path"]).resolve())
            digest = str(item["image_sha256"])
            previous = expected_hashes.get(path_text)
            if previous is not None and previous != digest:
                raise ValueError(f"Conflicting image hash: {path_text}")
            expected_hashes[path_text] = digest

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
                raise ValueError(f"TartanGround image hash mismatch: {path}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Unable to decode TartanGround image: {path}")
            frames.append(frame_tensor(image))
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
                    features_by_path[
                        str(Path(item["image_path"]).resolve())
                    ]
                    for item in record["history_rgb"]
                ]
            )
            for record in records
        ]
    )
    if matrix.shape != (495, 5, 576, 4, 7):
        raise ValueError(
            f"Unexpected TartanGround feature matrix: {matrix.shape}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(
            [record["sample_id"] for record in records]
        ),
        features=matrix,
    )
    report = {
        "schema": SCHEMA,
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "fold_manifest_path": str(DEFAULT_FOLD_MANIFEST.resolve()),
            "fold_manifest_sha256": sha256(DEFAULT_FOLD_MANIFEST),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "design": {
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
        "authority": {
            "role": "Development synthetic onset feature cache",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
    }
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
