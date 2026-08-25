#!/usr/bin/env python3
"""Two-sample real-backbone forward/loss smoke for the active R1C-L route."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch

from grail_pairwise_owner_coordinate_r1cl import (
    PairwiseOwnerCoordinate,
    YAW_BINS,
    exchange_consistency_loss,
    slot_marginalized_loss,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BA_ENV_CUDA_UNAVAILABLE")

    started = time.monotonic()
    device = torch.device("cuda")
    model = PairwiseOwnerCoordinate(args.backbone).to(device).train()
    reference = torch.randn(2, 3, 224, 224, device=device)
    query = torch.randn(2, 3, 224, 224, device=device)
    reference_masks = torch.rand(2, 2, 224, 224, device=device)
    query_masks = torch.rand(2, 2, 224, 224, device=device)
    valid = torch.zeros(2, YAW_BINS, dtype=torch.bool, device=device)
    valid[0, [0, 1, 35]] = True
    valid[1, [17, 18, 19]] = True

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        forward, reverse = model(
            reference, reference_masks, query, query_masks, return_reverse=True
        )
        slot = slot_marginalized_loss(forward, valid)
        exchange = exchange_consistency_loss(forward, reverse)
        loss = slot + 0.05 * exchange
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError("BA_SMOKE_NONFINITE_LOSS")
    loss.backward()
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "PASS",
        "samples": 2,
        "shape": list(forward.shape),
        "loss": float(loss.detach()),
        "slot_loss": float(slot.detach()),
        "exchange_loss": float(exchange.detach()),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_device": torch.cuda.get_device_name(0),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
