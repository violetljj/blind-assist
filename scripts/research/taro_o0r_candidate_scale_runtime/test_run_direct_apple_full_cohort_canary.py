#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_full_cohort_canary as runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter


class DirectAppleFullCohortRunnerTest(unittest.TestCase):
    def test_activation_write_failure_terminalizes_consumed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "one-shot"
            writer = FactorEvidenceWriter(root, 1024 * 1024)
            original = writer.write_json

            def fail_execution_receipt_once(relative: str, value: object) -> dict[str, object]:
                if relative == "execution-receipt.json":
                    raise runner.FullCohortRunError("SYNTHETIC_ACTIVATION_WRITE_FAILURE", "synthetic activation write failure")
                return original(relative, value)

            with mock.patch.object(writer, "write_json", side_effect=fail_execution_receipt_once):
                with self.assertRaises(runner.FullCohortRunError):
                    runner._activate_execution_writer(writer, {"schema": "synthetic.execution.v1"})

            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["error_code"], "SYNTHETIC_ACTIVATION_WRITE_FAILURE")
            self.assertIn("failure.json", manifest["files"])
            self.assertTrue(manifest["one_shot_root_consumed"])

    def test_terminal_print_failure_does_not_escape(self) -> None:
        with mock.patch("builtins.print", side_effect=OSError("synthetic console failure")):
            runner._best_effort_terminal_print({"terminal": "SYNTHETIC_COMPLETE"})

    def test_committed_manifest_blocks_late_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "one-shot"
            writer = FactorEvidenceWriter(root, 1024 * 1024)
            writer.activate({"schema": "synthetic.execution.v1"})
            files = dict(sorted(writer.file_receipts.items()))
            writer.write_json(
                "manifest.json",
                {
                    "schema": "synthetic.manifest.v1",
                    "files": files,
                    "file_count_before_manifest": len(files),
                },
            )

            receipts_before = dict(writer.file_receipts)
            runner._write_consumed_failure(writer, OSError("synthetic late failure"))

            self.assertEqual(writer.file_receipts, receipts_before)
            self.assertFalse((root / "failure.json").exists())


if __name__ == "__main__":
    unittest.main()
