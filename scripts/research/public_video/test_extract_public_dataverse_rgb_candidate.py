#!/usr/bin/env python3
from __future__ import annotations

import unittest

from extract_public_dataverse_rgb_candidate import ExtractionError, select_rgb_members


class ExtractPublicDataverseRgbCandidateTests(unittest.TestCase):
    def test_selects_only_contract_rgb_members(self) -> None:
        names = [
            "10/Color/Color_15_34_16_830.png",
            "10/Color/Color_15_34_16_867.png",
            "10/Depth/Depth_15_34_16_830.png",
            "10/Labels/Label_15_34_16_830.xml",
        ]
        self.assertEqual(
            select_rgb_members(names, prefix="10/Color/", expected_count=2),
            ["10/Color/Color_15_34_16_830.png", "10/Color/Color_15_34_16_867.png"],
        )

    def test_rejects_incomplete_or_non_rgb_contract(self) -> None:
        with self.assertRaises(ExtractionError):
            select_rgb_members(["10/Color/a.png"], prefix="10/Color/", expected_count=2)
        with self.assertRaises(ExtractionError):
            select_rgb_members(["10/Labels/a.png"], prefix="10/Labels/", expected_count=1)


if __name__ == "__main__":
    unittest.main()
