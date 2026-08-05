#!/usr/bin/env python3

import unittest

import torch

from scripts.research.hftf.materialize_dav2_block_skip_a4_bs25_p1_cache import (
    FROZEN_SKIP_INDICES,
    apply_frozen_block_skip,
)


class FakePretrained:
    def __init__(self) -> None:
        self.chunked_blocks = False
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(12)])


class FakeModel:
    def __init__(self) -> None:
        self.pretrained = FakePretrained()
        self.intermediate_layer_idx = {"vits": [2, 5, 8, 11]}


class BlockSkipTest(unittest.TestCase):
    def test_replaces_only_frozen_blocks(self) -> None:
        model = FakeModel()
        apply_frozen_block_skip(model)
        identities = tuple(
            index
            for index, block in enumerate(model.pretrained.blocks)
            if isinstance(block, torch.nn.Identity)
        )
        self.assertEqual(identities, FROZEN_SKIP_INDICES)


if __name__ == "__main__":
    unittest.main()
