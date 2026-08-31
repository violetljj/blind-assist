#!/usr/bin/env python3
"""Replay the carrier with a deterministic reference-support-only fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

import l10_3rscan_cycle_component_open_set_posthoc as carrier


def replay(protocol_path: Path, output_path: Path) -> None:
    original_file = carrier.__file__
    original_prompt = carrier.dual.dual_surface_cycle_affine_prompt

    def support_cascade(
        extent_masks: dict[str, np.ndarray],
        warp: torch.Tensor,
        certainty: torch.Tensor,
        reference_mask: np.ndarray,
        query_size: tuple[int, int],
        matcher: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        try:
            prompt, receipt = original_prompt(
                extent_masks, warp, certainty, reference_mask, query_size, matcher
            )
            receipt["reference_support_cascade"] = {
                "path": "NATIVE_SAM_SUPPORT",
                "specialist_requested": False,
                "primary_support_pixels": int(np.count_nonzero(reference_mask)),
            }
            return prompt, receipt
        except ValueError as exc:
            if str(exc) != "NO_REFERENCE_CYCLES":
                raise
        extent_mask = extent_masks[carrier.base.mask_sha256(reference_mask)]
        specialist_extents = {carrier.base.mask_sha256(extent_mask): extent_mask}
        prompt, receipt = original_prompt(
            specialist_extents, warp, certainty, extent_mask, query_size, matcher
        )
        receipt["reference_support_cascade"] = {
            "path": "FROZEN_REFERENCE_BOUND_EXTENT_SUPPORT",
            "specialist_requested": True,
            "request_reason": "NATIVE_SAM_ZERO_REFERENCE_CYCLES",
            "primary_support_pixels": int(np.count_nonzero(reference_mask)),
            "specialist_support_pixels": int(np.count_nonzero(extent_mask)),
        }
        return prompt, receipt

    carrier.__file__ = str(Path(__file__).resolve())
    carrier.dual.dual_surface_cycle_affine_prompt = support_cascade
    try:
        carrier.replay(protocol_path, output_path)
    finally:
        carrier.dual.dual_surface_cycle_affine_prompt = original_prompt
        carrier.__file__ = original_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
