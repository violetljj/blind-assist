"""Tests for the independent stress-package validator."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.research.synthetic_stress import validate_stress_r0 as validator


class SyntheticStressValidationTests(unittest.TestCase):
    def test_missing_package_is_reported_as_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = validator.validate(Path(temporary))
            self.assertEqual(result["status"], "INVALID")
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
