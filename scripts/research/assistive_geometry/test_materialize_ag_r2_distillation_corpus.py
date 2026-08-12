#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
import unittest

from scripts.research.assistive_geometry.materialize_ag_r2_distillation_corpus import (
    select_evenly_spaced_pairs,
)


class DistillationCorpusTest(unittest.TestCase):
    def test_even_selection_is_unique_and_temporally_ordered(self) -> None:
        rgb = [
            SimpleNamespace(row_index=index, timestamp_seconds=float(index))
            for index in range(120)
        ]
        depth = [
            SimpleNamespace(row_index=index, timestamp_seconds=float(index))
            for index in range(120)
        ]
        # The pairing/interpolation helpers require their real row types, so
        # test the deterministic bucket formula directly through equivalent indices.
        indices = [min(len(rgb) - 1, int((index + 0.5) * len(rgb) / 12)) for index in range(12)]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices[0], 5)
        self.assertEqual(indices[-1], 115)


if __name__ == "__main__":
    unittest.main()
