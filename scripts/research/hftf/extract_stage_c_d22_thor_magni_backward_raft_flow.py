#!/usr/bin/env python3
"""Extract current-to-history dense RAFT flow for THOR D22 transfer."""

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
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from extract_stage_c_d18_tartanground_backward_raft_flow import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
    resize_flow,
    sample_id_digest,
)
from extract_stage_c_d14_thor_magni_explicit_motion_features import (
    DEFAULT_RGB_CACHE,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    DEFAULT_SAMPLES,
)


SCHEMA = "blindassist_hftf_stage_c_d22_thor_magni_backward_raft_flow_v0"
EXPECTED_SAMPLES = 1078
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d22-thor-magni-backward-raft-flow-v0/"
    "current_to_history_flow_f16.npy"
)


class ThorCurrentToHistoryPairDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]
):
    def __init__(
        self,
        cache_path: Path,
        expected_samples: int = EXPECTED_SAMPLES,
    ) -> None:
        self.cache = np.load(cache_path, mmap_mode="r")
        expected = (expected_samples, 5, 128, 224, 3)
        if self.cache.shape != expected or self.cache.dtype != np.uint8:
            raise ValueError(
                f"Unexpected D22 RGB cache: {self.cache.shape}/{self.cache.dtype}"
            )

    def __len__(self) -> int:
        return self.cache.shape[0] * PAIRS_PER_SAMPLE

    def image(self, sample_index: int, frame_index: int) -> torch.Tensor:
        value = np.array(
            self.cache[sample_index, frame_index],
            copy=True,
        )
        return (
            torch.from_numpy(value)
            .permute(2, 0, 1)
            .float()
            .div_(127.5)
            .sub_(1.0)
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        sample_index = index // PAIRS_PER_SAMPLE
        history_index = index % PAIRS_PER_SAMPLE
        return (
            self.image(sample_index, 4),
            self.image(sample_index, history_index),
            sample_index,
            history_index,
        )


def validate_rgb_binding(
    records: list[dict[str, Any]],
    rgb_cache_path: Path,
) -> dict[str, Any]:
    report_path = Path(str(rgb_cache_path) + ".json")
    if not rgb_cache_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("D22 RGB cache or report missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_ids = [str(record["sample_id"]) for record in records]
    if report["design"]["sample_ids"] != expected_ids:
        raise ValueError("D22 RGB cache sample ordering mismatch")
    if report["output"]["sha256"] != sha256(rgb_cache_path):
        raise ValueError("D22 RGB cache SHA-256 mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB_CACHE)
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=DEFAULT_RAFT_WEIGHTS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    partial_path = args.output.with_name(
        args.output.stem + ".partial" + args.output.suffix
    )
    if args.output.exists() or report_path.exists():
        raise FileExistsError("D22 flow extraction is non-overwriting")
    if partial_path.exists():
        partial_path.unlink()
    if not args.samples.is_file() or not args.raft_weights.is_file():
        raise FileNotFoundError("D22 samples or RAFT weights missing")
    if sha256(args.raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("D22 RAFT weight SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("D22 flow extraction requires CUDA")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: str(row["sample_id"]))
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError("D22 requires the exact D12 sample count")
    rgb_report = validate_rgb_binding(records, args.rgb_cache)

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
    dataset = ThorCurrentToHistoryPairDataset(args.rgb_cache)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flows = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float16,
        shape=(
            EXPECTED_SAMPLES,
            PAIRS_PER_SAMPLE,
            2,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
        ),
    )
    maximum_absolute_flow = 0.0
    with torch.inference_mode():
        for batch_index, (
            current,
            earlier,
            sample_index,
            history_index,
        ) in enumerate(loader):
            full_flow = model(
                current.to(device, non_blocking=True),
                earlier.to(device, non_blocking=True),
            )[-1]
            low_flow = resize_flow(full_flow).cpu().numpy()
            if not np.all(np.isfinite(low_flow)):
                raise ValueError("D22 RAFT produced non-finite flow")
            maximum_absolute_flow = max(
                maximum_absolute_flow,
                float(np.max(np.abs(low_flow))),
            )
            for local_index in range(len(low_flow)):
                flows[
                    int(sample_index[local_index]),
                    int(history_index[local_index]),
                ] = low_flow[local_index].astype(np.float16)
            if batch_index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "batch": batch_index,
                            "batches": len(loader),
                            "pairs_complete": min(
                                (batch_index + 1) * args.batch_size,
                                len(dataset),
                            ),
                        }
                    ),
                    flush=True,
                )
    flows.flush()
    del flows
    with partial_path.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(partial_path, args.output)

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "D22_THOR_MAGNI_BACKWARD_RAFT_FLOW_MATERIALIZED",
        "authority": {
            "role": "Development independent-source dense-flow cache",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": rgb_report["output"]["sha256"],
            "rgb_cache_source_samples_sha256": rgb_report["inputs"][
                "samples_sha256"
            ],
            "raft_weights_path": str(args.raft_weights.resolve()),
            "raft_weights_sha256": EXPECTED_RAFT_SHA256,
        },
        "design": {
            "direction": (
                "current frame to each of four earlier history frames; "
                "usable as backward sampling coordinates"
            ),
            "source_resolution": [128, 224],
            "output_resolution": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "component_scaling": (
                "x and y displacement scaled with output width and height"
            ),
            "model": "torchvision pretrained RAFT-small C_T_V2",
            "selection": "none; every D12 sample and history pair retained",
        },
        "counts": {
            "samples": len(records),
            "pairs": len(dataset),
            "sample_ids_sha256": sample_id_digest(records),
        },
        "output": {
            "path": str(args.output.resolve()),
            "shape": [
                EXPECTED_SAMPLES,
                PAIRS_PER_SAMPLE,
                2,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
            ],
            "dtype": "float16",
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
            "maximum_absolute_flow_low_resolution": maximum_absolute_flow,
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
