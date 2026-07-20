#!/usr/bin/env python3
"""Pure tests for dense public-video review windows."""

from __future__ import annotations

import unittest

import extract_public_video_review_windows as subject


class PublicVideoReviewWindowsTest(unittest.TestCase):
    def test_parse_window(self) -> None:
        self.assertEqual(("source", 1000, 5000), subject.parse_window("source:1000:5000"))

    def test_parse_window_rejects_reversed_bounds(self) -> None:
        with self.assertRaises(ValueError):
            subject.parse_window("source:5000:1000")


if __name__ == "__main__":
    unittest.main()
