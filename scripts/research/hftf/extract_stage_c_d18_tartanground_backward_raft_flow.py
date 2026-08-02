#!/usr/bin/env python3
"""Extract current-to-history RAFT flow for D18 feature alignment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.optical_flow import raft_small
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from evaluate_stage_c_d6_sanpo_raft_motion_representation import (
    DEFAULT_RAFT_WEIGHTS,
    EXPECTED_RAFT_SHA256,
)
from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    DEFAULT_SAMPLES,
    validate_records,
)


SCHEMA = "blindassist_hftf_stage_c_d18_tartanground_backward_raft_flow_v0"
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 112
EXPECTED_SAMPLES = 495
PAIRS_PER_SAMPLE = 4
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d18-tartanground-backward-raft-flow-v0/"
    "current_to_history_flow_f16.npy"
)


class CurrentToHistoryPairDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]
):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records) * PAIRS_PER_SAMPLE

    def image(self, path: str) -> torch.Tensor:
        value = self.cache.get(path)
        if value is None:
            with Image.open(path) as source:
                resized = tvf.resize(
                    source.convert("RGB"),
                    [128, 224],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            value = tvf.pil_to_tensor(resized).float().div_(127.5).sub_(1.0)
            self.cache[path] = value
        return value

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        sample_index = index // PAIRS_PER_SAMPLE
        history_index = index % PAIRS_PER_SAMPLE
        history = self.records[sample_index]["history_rgb"]
        current = self.image(str(history[-1]["image_path"]))
        earlier = self.image(str(history[history_index]["image_path"]))
        return current, earlier, sample_index, history_index


def resize_flow(
    flow: torch.Tensor,
    height: int = OUTPUT_HEIGHT,
    width: int = OUTPUT_WIDTH,
) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("RAFT flow must have shape Bx2xHxW")
    source_height, source_width = flow.shape[-2:]
    resized = nnf.interpolate(
        flow,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    resized[:, 0].mul_(width / source_width)
    resized[:, 1].mul_(height / source_height)
    return resized


def sample_id_digest(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(record["sample_id"]) for record in records) + "\n"
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
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
        raise FileExistsError("D18 flow extraction is non-overwriting")
    if partial_path.exists():
        partial_path.unlink()
    if not args.samples.is_file() or not args.raft_weights.is_file():
        raise FileNotFoundError("D18 samples or RAFT weights missing")
    if sha256(args.raft_weights) != EXPECTED_RAFT_SHA256:
        raise ValueError("D18 RAFT weight SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("D18 flow extraction requires CUDA")
    records = load_jsonl(args.samples)
    validate_records(records)
    if len(records) != EXPECTED_SAMPLES:
        raise ValueError("D18 requires the exact D16 sample count")

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
    dataset = CurrentToHistoryPairDataset(records)
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
                raise ValueError("D18 RAFT produced non-finite flow")
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
    with partial_path.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial_path, args.output)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "D18_TARTANGROUND_BACKWARD_RAFT_FLOW_MATERIALIZED",
        "authority": {
            "role": "Development frozen dense-correspondence cache",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
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
            "selection": "none; every D16 sample and history pair retained",
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
            "maximum_absolute_flow_low_resolution": (
                maximum_absolute_flow
            ),
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
