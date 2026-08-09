#!/usr/bin/env python3
"""Attempt 02 wrapper: keep checkpoint RNG tensors on CPU during restore."""

from __future__ import annotations

import torch

from scripts.research.assistive_geometry import smoke_b1_a0_train_execution as attempt_01


def main() -> int:
    original_load = torch.load

    def load_checkpoint_on_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    torch.load = load_checkpoint_on_cpu
    try:
        return attempt_01.main()
    finally:
        torch.load = original_load


if __name__ == "__main__":
    raise SystemExit(main())
