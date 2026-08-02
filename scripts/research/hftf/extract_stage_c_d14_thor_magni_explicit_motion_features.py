#!/usr/bin/env python3
"""Extract direction-preserving RAFT motion features for THOR onset."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow import raft_small

from evaluate_stage_c_d6_sanpo_raft_motion_representation import (
    DEFAULT_RAFT_WEIGHTS,
    EXPECTED_RAFT_SHA256,
    remove_global_affine,
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    DEFAULT_SAMPLES,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d14_thor_magni_"
    "direction_preserving_raft_motion_features_v0"
)
DEFAULT_RGB_CACHE = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d10-thor-magni-trainable-rgb-cache-v0/"
    "history_rgb_uint8.npy"
)
GRID_ROWS = 3
GRID_COLUMNS = 6


class ConsecutivePairDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]
):
    def __init__(self, cache_path: Path) -> None:
        self.cache = np.load(cache_path, mmap_mode="r")
        if self.cache.shape != (1078, 5, 128, 224, 3):
            raise ValueError(f"Unexpected RGB cache: {self.cache.shape}")

    def __len__(self) -> int:
        return self.cache.shape[0] * 4

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        sample_index = index // 4
        pair_index = index % 4

        def image(frame_index: int) -> torch.Tensor:
            value = np.array(
                self.cache[sample_index, frame_index],
                copy=True,
            )
            return (
                torch.from_numpy(value)
                .permute(2, 0, 1)
                .float()
                .div(127.5)
                .sub(1.0)
            )

        return (
            image(pair_index),
            image(pair_index + 1),
            sample_index,
            pair_index,
        )


def direction_preserving_grid(
    flow: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    _, height, width = flow.shape
    raw_x = flow[0] / width
    raw_y = flow[1] / height
    residual_x = residual[0] / width
    residual_y = residual[1] / height
    raw_magnitude = np.sqrt(raw_x**2 + raw_y**2)
    residual_magnitude = np.sqrt(
        residual_x**2 + residual_y**2
    )
    output = np.empty(
        (8, GRID_ROWS, GRID_COLUMNS),
        dtype=np.float32,
    )
    for row in range(GRID_ROWS):
        y0 = row * height // GRID_ROWS
        y1 = (row + 1) * height // GRID_ROWS
        for column in range(GRID_COLUMNS):
            x0 = column * width // GRID_COLUMNS
            x1 = (column + 1) * width // GRID_COLUMNS
            cells = [
                values[y0:y1, x0:x1]
                for values in (
                    raw_x,
                    raw_y,
                    raw_magnitude,
                    residual_x,
                    residual_y,
                    residual_magnitude,
                )
            ]
            output[:6, row, column] = [
                float(np.mean(cell)) for cell in cells
            ]
            output[6, row, column] = float(
                np.quantile(cells[2], 0.90)
            )
            output[7, row, column] = float(
                np.quantile(cells[5], 0.90)
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB_CACHE)
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=DEFAULT_RAFT_WEIGHTS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = Path(str(args.output) + ".json")
    partial_path = args.output.with_name(
        args.output.stem + ".partial" + args.output.suffix
    )
    if args.output.exists() or report_path.exists():
        raise ValueError("Refusing to overwrite D14 motion features")
    if partial_path.exists():
        partial_path.unlink()
    if sha256(args.raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("Unexpected RAFT-small weights SHA-256")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for D14 RAFT extraction")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    sample_ids = np.asarray(
        [record["sample_id"] for record in records],
        dtype=str,
    )
    if len(sample_ids) != 1078:
        raise ValueError("Expected 1,078 D12 samples")
    cache_report = json.loads(
        Path(str(args.rgb_cache) + ".json").read_text(encoding="utf-8")
    )
    if (
        cache_report["design"]["sample_ids"] != sample_ids.tolist()
        or cache_report["output"]["sha256"] != sha256(args.rgb_cache)
    ):
        raise ValueError("D10 RGB cache binding mismatch")

    model = raft_small(weights=None, progress=False)
    model.load_state_dict(
        torch.load(
            args.raft_weights,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = ConsecutivePairDataset(args.rgb_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    features = np.empty(
        (1078, 4, 8, GRID_ROWS, GRID_COLUMNS),
        dtype=np.float16,
    )
    diagnostic_counts: dict[str, int] = {}
    with torch.inference_mode():
        for previous, current, sample_index, pair_index in loader:
            flows = model(
                previous.to(device, non_blocking=True),
                current.to(device, non_blocking=True),
            )[-1].cpu().numpy()
            for local_index, flow in enumerate(flows):
                residual, diagnostic = remove_global_affine(flow)
                reason = str(diagnostic["reason"])
                diagnostic_counts[reason] = (
                    diagnostic_counts.get(reason, 0) + 1
                )
                if residual is None:
                    raise ValueError(
                        "RAFT global-motion removal was not evaluable"
                    )
                features[
                    int(sample_index[local_index]),
                    int(pair_index[local_index]),
                ] = direction_preserving_grid(
                    flow,
                    residual,
                ).astype(np.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("xb") as output:
        np.savez_compressed(
            output,
            sample_ids=sample_ids,
            features=features,
        )
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial_path, args.output)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "D14_DIRECTION_PRESERVING_RAFT_FEATURES_MATERIALIZED",
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": cache_report["output"]["sha256"],
            "raft_weights_path": str(args.raft_weights.resolve()),
            "raft_weights_sha256": EXPECTED_RAFT_SHA256,
        },
        "design": {
            "pairs": "four consecutive pairs from five history frames",
            "flow": "pretrained torchvision RAFT-small",
            "global_motion": (
                "RANSAC partial-affine with fixed median-translation fallback"
            ),
            "grid": [GRID_ROWS, GRID_COLUMNS],
            "channels": [
                "raw_mean_x",
                "raw_mean_y",
                "raw_mean_magnitude",
                "residual_mean_x",
                "residual_mean_y",
                "residual_mean_magnitude",
                "raw_p90_magnitude",
                "residual_p90_magnitude",
            ],
            "selection": "none; all raw and residual channels retained",
        },
        "counts": {
            "samples": len(sample_ids),
            "pairs": len(dataset),
            "diagnostics": dict(sorted(diagnostic_counts.items())),
        },
        "output": {
            "path": str(args.output.resolve()),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
        "authority": {
            "role": "Development explicit-motion feature cache",
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
                "status": report["status"],
                "counts": report["counts"],
                "output": report["output"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
