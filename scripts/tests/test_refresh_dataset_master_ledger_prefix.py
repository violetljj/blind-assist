from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import refresh_dataset_master_ledger_prefix as refresh


class PrefixDiscoveryTest(unittest.TestCase):
    def test_prefix_discovery_keeps_canonical_relative_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts.local"
            prefix = root / "datasets" / "arkit-scenes" / "raw" / "Validation" / "42445021"
            frame = prefix / "lowres_wide" / "42445021_1.000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"fixture")

            candidates = refresh.discover_prefix("fixture", root, root / "datasets" / "arkit-scenes")

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].group_rel, "datasets/arkit-scenes/raw/Validation/42445021")
            self.assertEqual(candidates[0].rel_path, "datasets/arkit-scenes/raw/Validation/42445021/lowres_wide/42445021_1.000.png")


if __name__ == "__main__":
    unittest.main()
