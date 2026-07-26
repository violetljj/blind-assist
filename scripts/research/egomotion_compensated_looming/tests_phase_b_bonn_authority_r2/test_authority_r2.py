from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_authority_r2 import (
    authority,
)


class BonnMetadataAuthorityR2Test(unittest.TestCase):
    def test_claim_is_exclusive_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "run_claim.json"
            authority.create_preclaim_first(claim, {"first": True})
            with self.assertRaises(FileExistsError):
                authority.create_preclaim_first(claim, {"second": True})
            self.assertTrue(claim.exists())

    def test_runner_orders_claim_before_receipt_build(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_metadata_authority_r2.py"
        ).read_text(encoding="utf-8")
        claim_index = runner.index("create_preclaim_first(")
        build_index = runner.index("build_receipt(")
        self.assertLess(claim_index, build_index)
        self.assertNotIn("validate_implementation_lock", runner)

    def test_runner_has_no_path_override(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_metadata_authority_r2.py"
        ).read_text(encoding="utf-8")
        for forbidden in [
            "--output-root",
            "--lock",
            "--official-page",
            "--receipt",
        ]:
            self.assertNotIn(forbidden, runner)

    def test_canonical_r2_output_is_separate(self) -> None:
        paths = authority.canonical_paths(Path("E:/repo"))
        self.assertIn("rcle_phase_b_bonn_entry_r2", str(paths["output"]))
        self.assertEqual(paths["run_claim"].parent, paths["output"])


if __name__ == "__main__":
    unittest.main()
