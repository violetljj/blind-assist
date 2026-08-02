#!/usr/bin/env python3
"""Tests for D18 dense backward-flow extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_stage_c_d18_tartanground_backward_raft_flow import resize_flow


class D18BackwardFlowExtractionTests(unittest.TestCase):
    def test_resize_flow_scales_displacement_components(self) -> None:
        flow = torch.zeros(1, 2, 128, 224)
        flow[:, 0] = 8.0
        flow[:, 1] = -6.0
        resized = resize_flow(flow, 64, 112)
        torch.testing.assert_close(
            resized[:, 0],
            torch.full((1, 64, 112), 4.0),
        )
        torch.testing.assert_close(
            resized[:, 1],
            torch.full((1, 64, 112), -3.0),
        )


if __name__ == "__main__":
    unittest.main()
