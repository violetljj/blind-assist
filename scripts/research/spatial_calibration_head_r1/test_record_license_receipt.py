#!/usr/bin/env python3

import json
import unittest

from record_license_receipt import CONFIRMATION, build_receipt
from validate_protocol import DEFAULT_PROTOCOL


class LicenseReceiptTest(unittest.TestCase):
    def test_exact_user_confirmation_is_required(self) -> None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            build_receipt(protocol, "p", "r", "a", "yes", "user")
        receipt = build_receipt(protocol, "p", "r", "a", CONFIRMATION, "user")
        self.assertTrue(receipt["media_download_authorized"])
        self.assertFalse(receipt["authorized_scope"]["sealed_metric_assets"])


if __name__ == "__main__":
    unittest.main()
