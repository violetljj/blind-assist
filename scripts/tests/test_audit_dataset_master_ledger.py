from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import audit_dataset_master_ledger as ledger


class ArkitScenesDiscoveryTest(unittest.TestCase):
    def test_validation_media_is_grouped_by_video_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifacts.local"
            parent = (
                root
                / "datasets"
                / "p3-r0-2-1-arkit-validation-extension-20260805"
                / "raw"
                / "Validation"
                / "42445021"
            )
            files = {
                "lowres_wide/42445021_48879.676.png": "rgb",
                "lowres_depth/42445021_48879.676.png": "depth",
                "confidence/42445021_48879.676.png": "confidence",
                "lowres_wide_intrinsics/42445021_48879.676.pincam": "metadata",
                "lowres_wide.traj": "pose",
            }
            for relative_path in files:
                path = parent / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            candidates, _ = ledger.discover_candidates([("fixture", root)])

            self.assertEqual(len(candidates), len(files))
            self.assertEqual(
                {candidate.group_rel for candidate in candidates},
                {
                    "datasets/p3-r0-2-1-arkit-validation-extension-20260805/"
                    "raw/Validation/42445021"
                },
            )
            self.assertEqual(
                {candidate.classification["modality"] for candidate in candidates},
                set(files.values()),
            )
            session = ledger.build_sessions(candidates)[0]
            path_text = f"{session.root_id} {session.group_rel}"
            self.assertEqual(ledger.infer_dataset(path_text), "ARKitScenes")


if __name__ == "__main__":
    unittest.main()
