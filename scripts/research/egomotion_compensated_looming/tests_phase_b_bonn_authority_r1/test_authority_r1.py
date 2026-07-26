from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r1 import (
    authority,
)


class BonnMetadataAuthorityR1Test(unittest.TestCase):
    def test_exclusive_claim_rejects_second_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_claim.json"
            authority.create_exclusive_claim(path, {"claim": 1})
            with self.assertRaises(FileExistsError):
                authority.create_exclusive_claim(path, {"claim": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"claim": 1},
            )

    def test_runner_cli_has_no_path_override(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_metadata_authority_r1.py"
        ).read_text(encoding="utf-8")
        for forbidden in [
            "--output-root",
            "--lock",
            "--official-page",
            "--receipt",
        ]:
            self.assertNotIn(forbidden, runner)
        self.assertIn("--validate-existing", runner)

    def test_canonical_paths_are_repo_bound(self) -> None:
        repo = Path("E:/example/repo")
        paths = authority.canonical_paths(repo)
        self.assertEqual(
            paths["output"],
            repo
            / "artifacts.local"
            / "evidence"
            / "rcle_phase_b_bonn_entry_r1"
            / "authority_gate_r1",
        )
        self.assertEqual(paths["run_claim"].parent, paths["output"])

    def test_fixed_identity_hashes_are_complete(self) -> None:
        for value in [
            authority.DESIGN_LOCK_SHA256,
            authority.R0_RECEIPT_SHA256,
            authority.R0_IMPLEMENTATION_LOCK_SHA256,
            authority.OFFICIAL_PAGE_SHA256,
            authority.HISTORICAL_MANIFEST_SHA256,
            authority.COHORT_IDENTITY_SHA256,
        ]:
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_validate_only_parser_default_is_false(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--validate-existing", action="store_true")
        self.assertFalse(parser.parse_args([]).validate_existing)
        self.assertTrue(
            parser.parse_args(["--validate-existing"]).validate_existing
        )


if __name__ == "__main__":
    unittest.main()
