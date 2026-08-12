from __future__ import annotations

import unittest
from unittest import mock

from scripts.research.taro_o1r_r8_clear_runtime import run_ray_space_canary as runner


class RaySpaceCanaryRunnerTests(unittest.TestCase):
    def test_manifest_rejects_missing_file(self) -> None:
        manifest = {
            "schema": "example.schema",
            "terminal": "DONE",
            "file_count_before_manifest": 1,
            "files": {"missing.json": {"path": "missing.json", "bytes": 1, "sha256": "0" * 64}},
        }
        with mock.patch.object(runner, "_repo_path", return_value=runner.REPO_ROOT / "does-not-exist"):
            with self.assertRaisesRegex(runner.RaySpaceCanaryError, "manifest file drift"):
                runner._verify_manifest("root", manifest, "example.schema", "DONE")

    def test_manifest_rejects_self_inclusion(self) -> None:
        manifest = {
            "schema": "example.schema",
            "terminal": "DONE",
            "file_count_before_manifest": 1,
            "files": {"manifest.json": {"path": "manifest.json", "bytes": 1, "sha256": "0" * 64}},
        }
        with self.assertRaisesRegex(runner.RaySpaceCanaryError, "manifest count drift"):
            runner._verify_manifest("root", manifest, "example.schema", "DONE")


if __name__ == "__main__":
    unittest.main()
