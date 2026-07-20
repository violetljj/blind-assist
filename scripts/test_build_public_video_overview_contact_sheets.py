#!/usr/bin/env python3
"""Pure tests for public-video overview contact sheets."""

from __future__ import annotations

import unittest

import build_public_video_overview_contact_sheets as subject


class PublicVideoOverviewContactSheetsTest(unittest.TestCase):
    def test_chunked_preserves_order_and_tail(self) -> None:
        self.assertEqual([[0, 1], [2, 3], [4]], subject.chunked(list(range(5)), 2))

    def test_chunked_rejects_nonpositive_size(self) -> None:
        with self.assertRaises(ValueError):
            subject.chunked([1], 0)


if __name__ == "__main__":
    unittest.main()
