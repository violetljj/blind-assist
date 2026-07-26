from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r3 import (
    authority,
)


class BonnMetadataAuthorityR3Test(unittest.TestCase):
    @staticmethod
    def runner_source() -> str:
        return (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_metadata_authority_r3.py"
        ).read_text(encoding="utf-8")

    def test_top_level_imports_are_stdlib_only(self) -> None:
        tree = ast.parse(self.runner_source())
        top_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported = {
            (
                node.module.split(".")[0]
                if isinstance(node, ast.ImportFrom) and node.module
                else alias.name.split(".")[0]
            )
            for node in top_imports
            for alias in node.names
        }
        self.assertEqual(
            imported,
            {"__future__", "datetime", "json", "os", "sys"},
        )

    def test_claim_precedes_project_import(self) -> None:
        source = self.runner_source()
        claim_call = source.index("_claim_first(repo_root_text, started_at)")
        project_import = source.index(
            "from scripts.research.egomotion_compensated_looming"
        )
        self.assertLess(claim_call, project_import)

    def test_forbidden_preclaim_operations_absent(self) -> None:
        source_before_project_import = self.runner_source().split(
            "from scripts.research.egomotion_compensated_looming", 1
        )[0]
        for forbidden in [
            "argparse",
            ".resolve(",
            ".exists(",
            ".stat(",
            ".glob(",
            ".listdir(",
        ]:
            self.assertNotIn(forbidden, source_before_project_import)

    def test_claim_is_exclusive_and_persistent(self) -> None:
        runner = self.runner_source()
        self.assertIn("os.O_CREAT | os.O_EXCL", runner)
        self.assertIn("os.fsync(descriptor)", runner)
        self.assertEqual(runner.count("os.open("), 1)

    def test_canonical_r3_output_is_separate(self) -> None:
        paths = authority.canonical_paths(Path("E:/repo"))
        self.assertIn("rcle_phase_b_bonn_entry_r3", str(paths["output"]))
        self.assertEqual(paths["run_claim"].parent, paths["output"])

    def test_no_cli_path_override(self) -> None:
        runner = self.runner_source()
        for forbidden in [
            "--output-root",
            "--lock",
            "--official-page",
            "--receipt",
        ]:
            self.assertNotIn(forbidden, runner)


if __name__ == "__main__":
    unittest.main()
