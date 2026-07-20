import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_silver_inverse_counterfactuals as audit


def pair_rows():
    source = {
        "parent_source_id": "source-a",
        "source_page_url": "https://commons.wikimedia.org/wiki/File:Example.jpg",
        "original_file_path": "source/example.jpg",
        "original_file_sha256": "a" * 64,
        "license": "CC0",
        "artist": "Example",
        "lineage_rule": "hold_out_source_and_all_descendants",
    }
    return [
        {"label": 0, "attributes": {"counterfactual_pair_id": "p1", "risk_state": "clear", "synthetic": True, "endpoint_role": "gpt_removed_obstacle"}, "source": source},
        {"label": 1, "attributes": {"counterfactual_pair_id": "p1", "risk_state": "risk", "synthetic": False, "endpoint_role": "real_licensed_risk"}, "source": source},
    ]


class InverseCounterfactualAuditTest(unittest.TestCase):
    def test_accepts_exact_inverse_pair_contract(self):
        result = audit.validate_pair_rows(pair_rows())
        self.assertEqual(1, result["pair_count"])
        self.assertEqual(1, result["parent_source_count"])

    def test_rejects_synthetic_risk_endpoint(self):
        rows = copy.deepcopy(pair_rows())
        rows[1]["attributes"]["synthetic"] = True
        with self.assertRaisesRegex(ValueError, "invalid real risk"):
            audit.validate_pair_rows(rows)

    def test_rejects_reused_parent_source_across_pairs(self):
        rows = pair_rows()
        second = copy.deepcopy(rows)
        for row in second:
            row["attributes"]["counterfactual_pair_id"] = "p2"
        with self.assertRaisesRegex(ValueError, "unique per pair"):
            audit.validate_pair_rows(rows + second)


if __name__ == "__main__":
    unittest.main()
