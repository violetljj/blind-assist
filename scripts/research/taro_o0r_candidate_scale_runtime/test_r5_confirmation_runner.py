#!/usr/bin/env python3
"""Focused failure-boundary tests for the TARO R5 one-shot runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation_io as r5io
from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_hybrid_adapter_fit_confirmation as runner


class R5ConfirmationRunnerTests(unittest.TestCase):
    def test_missing_plan_fails_before_model_load_or_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            r3 = base / "r3"
            evidence = base / "r5"
            source.mkdir()
            r3.mkdir()
            called = False

            def validate(_):
                return {
                    "roots": {
                        "source_root": str(source),
                        "r3_evidence_root": str(r3),
                        "r5_evidence_root": str(evidence),
                    },
                    "frame_plan_path": str(base / "missing-plan.json.gz"),
                }

            def model_loader(*args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("model loader must not run")

            with self.assertRaises(r5io.R5ConfirmationIOError) as caught:
                runner.execute_confirmation(base / "lock.json", lock_validator=validate, model_loader=model_loader)
            self.assertEqual("R5_GZIP_JSON_READ_FAILED", caught.exception.code)
            self.assertFalse(called)
            self.assertFalse(evidence.exists())

    def test_io_rejects_payload_roles_outside_phase_allowlist(self) -> None:
        with self.assertRaises(r5io.R5ConfirmationIOError) as caught:
            r5io.read_bound_payload(None, None, "trajectory")  # type: ignore[arg-type]
        self.assertEqual("R5_PAYLOAD_ROLE_INVALID", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
