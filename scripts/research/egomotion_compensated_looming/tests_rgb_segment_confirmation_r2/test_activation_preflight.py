from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    REPO
    / "scripts"
    / "research"
    / "egomotion_compensated_looming"
    / "rgb_segment_confirmation_r2"
    / "activation_preflight.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("activation_preflight", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ActivationPreflightTests(unittest.TestCase):
    def test_claim_is_atomic_and_second_writer_fails(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            claim = Path(temporary) / "claim.json"
            module.exclusive_claim(claim, {"claim": "first"})
            with self.assertRaises(module.ActivationPreflightFailure):
                module.exclusive_claim(claim, {"claim": "second"})
            self.assertEqual(
                json.loads(claim.read_text(encoding="utf-8")),
                {"claim": "first"},
            )

    def test_resource_probe_is_real_and_positive(self) -> None:
        module = load_module()
        result = module.validate_resource_probe(module.process_peak_memory())
        self.assertGreater(result["peak_rss_bytes"], 0)
        self.assertGreater(result["peak_commit_bytes"], 0)

    def test_resource_probe_rejects_throwaway_or_null_values(self) -> None:
        module = load_module()
        invalid = [
            None,
            {},
            {"peak_rss_bytes": None, "peak_commit_bytes": None},
            {"peak_rss_bytes": 1, "peak_commit_bytes": 0},
            {
                "peak_rss_bytes": 1,
                "peak_commit_bytes": 1,
                "extra": 1,
            },
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(module.ActivationPreflightFailure):
                    module.validate_resource_probe(value)

    def test_secret_scan_records_hash_not_secret_text(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe.json"
            unsafe = root / "unsafe.txt"
            safe.write_text('{"status":"PASS"}\n', encoding="utf-8")
            unsafe.write_text("Authorization: private-value\n", encoding="utf-8")
            result = module.secret_scan([safe, unsafe])
            self.assertEqual(result["decision"], "FAIL")
            rendered = json.dumps(result)
            self.assertNotIn("private-value", rendered)
            self.assertEqual(result["findings"][0]["line"], 1)


if __name__ == "__main__":
    unittest.main()
