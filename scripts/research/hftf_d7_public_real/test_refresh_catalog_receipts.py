from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from refresh_catalog_receipts import _latest_materialized_receipts


class RefreshCatalogReceiptsTest(unittest.TestCase):
    def test_downstream_review_receipt_cannot_replace_source_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            receipts = root / "receipts"
            receipts.mkdir()
            source_path = receipts / "sanpo_candidate_receipt_d7-r1.json"
            source_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "SANPO-Real",
                        "schema": "hftf_d7_public_real_sanpo_candidate_receipt_v1",
                        "generated_at_utc": "2026-08-02T10:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            review_path = receipts / "review_bundle_receipt_d7-r1.json"
            review_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "SANPO-Real",
                        "schema": "hftf_d7_public_real_review_bundle_receipt_v1",
                        "generated_at_utc": "2026-08-02T11:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            selected = _latest_materialized_receipts(root)

            self.assertEqual(selected["SANPO-Real"][0], source_path)


if __name__ == "__main__":
    unittest.main()
