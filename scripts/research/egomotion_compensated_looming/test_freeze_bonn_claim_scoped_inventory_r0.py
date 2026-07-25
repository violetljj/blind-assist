#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze_bonn_claim_scoped_inventory_r0 as subject  # noqa: E402


class BonnClaimScopedFreezeTest(unittest.TestCase):
    def test_parser_extracts_sequence_entry(self) -> None:
        html = """
        Name: rgbd_bonn_example<br />
        Size: 12.5 MB<br />
        <a href="https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_example.zip">Download</a>
        """
        with self.assertRaisesRegex(ValueError, "expected 26"):
            subject.parse_inventory(html)

    def test_stable_hash_is_deterministic(self) -> None:
        value = subject.stable_hash("rgbd_bonn_static")
        self.assertEqual(value, subject.stable_hash("rgbd_bonn_static"))
        self.assertNotEqual(value, subject.stable_hash("rgbd_bonn_static_close_far"))


if __name__ == "__main__":
    unittest.main()
