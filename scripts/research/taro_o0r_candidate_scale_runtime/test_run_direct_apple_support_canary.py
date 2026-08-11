#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_support_canary as runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter


class DirectAppleSupportRunnerTest(unittest.TestCase):
    def test_activation_write_failure_terminalizes_consumed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "one-shot"
            writer = FactorEvidenceWriter(root, 1024 * 1024)
            original = writer.write_json

            def fail_execution_receipt_once(relative: str, value: object) -> dict[str, object]:
                if relative == "execution-receipt.json":
                    raise runner.DirectAppleSupportRunError("SYNTHETIC_ACTIVATION_WRITE_FAILURE", "synthetic activation write failure")
                return original(relative, value)

            with mock.patch.object(writer, "write_json", side_effect=fail_execution_receipt_once):
                with self.assertRaises(runner.DirectAppleSupportRunError):
                    runner._activate_execution_writer(writer, {"schema": "synthetic.execution.v1"})

            self.assertTrue(root.is_dir())
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["error_code"], "SYNTHETIC_ACTIVATION_WRITE_FAILURE")
            self.assertTrue(failure["one_shot_consumed"])
            self.assertIn("failure.json", manifest["files"])
            self.assertTrue(manifest["one_shot_root_consumed"])


if __name__ == "__main__":
    unittest.main()
