from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0 import (
    DEFAULT_WORKERS,
    decimal_median,
    numbers_equivalent,
)
from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    validate_burned_fixture,
    validate_execution_contract,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "burned_floor3_2_w3_even_median.json"
)


class NumericTemplateTest(unittest.TestCase):
    def test_even_sample_decimal_median(self) -> None:
        self.assertEqual(
            decimal_median(
                [
                    Decimal("0.2532449718901444"),
                    Decimal("0.25457027587838793"),
                ]
            ),
            Decimal("0.253907623884266165"),
        )

    def test_decimal_float_representation_uses_prefrozen_tolerance(self) -> None:
        self.assertTrue(
            numbers_equivalent(
                Decimal("0.2539076238842660"),
                0.25390762388426613,
            )
        )
        self.assertFalse(numbers_equivalent(Decimal("0.25"), 0.250001))

    def test_burned_fixture_smoke(self) -> None:
        rows = [
            {
                "window_index": 3,
                "geometry_evaluable": True,
                "geometry_signed_radial_expansion_per_s": "0.1",
            },
            {
                "window_index": 3,
                "geometry_evaluable": True,
                "geometry_signed_radial_expansion_per_s": "0.3",
            },
        ]
        payload = "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            fixture = Path(directory) / "fixture.json"
            ledger.write_bytes(payload)
            fixture.write_text(
                json.dumps(
                    {
                        "schema": "rcle.motion_diverse.burned_median_fixture.v1",
                        "source_access": "BURNED_REGRESSION_ONLY",
                        "source_protocol": (
                            "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_"
                            "CROSS_SEQUENCE_HOLDOUT_R0"
                        ),
                        "source_window": 3,
                        "source_geometry_pair_ledger_sha256": hashlib.sha256(
                            payload
                        ).hexdigest(),
                        "evaluable_value_count": 2,
                        "center_values": ["0.1", "0.3"],
                        "producer_serialized_median": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            receipt = validate_burned_fixture(fixture, ledger)
            self.assertEqual(receipt["status"], "BURNED_FIXTURE_SMOKE_PASS")
            self.assertEqual(receipt["evaluable_value_count"], 2)
            self.assertEqual(receipt["default_workers"], 8)

    def test_future_contract_is_fail_closed(self) -> None:
        contract = {
            "execution": {"default_workers": DEFAULT_WORKERS},
            "geometry_only_selection": {
                "window_duration_s": "10",
                "required_positive_windows": 2,
                "required_below_reference_windows": 2,
            },
            "candidate": {"metadata_rank": 1, "candidate_id": "candidate_a"},
            "numeric_equivalence": {
                "relative_tolerance": "1e-12",
                "absolute_tolerance": "1e-15",
                "finite_only": True,
            },
            "candidate_replacement_allowed": False,
            "post_outcome_window_addition_allowed": False,
            "rgb_download_before_geometry_admission": False,
        }
        validate_execution_contract(contract)
        contract["candidate"]["candidate_id"] = "floor3_3"
        with self.assertRaisesRegex(ValueError, "FLOOR3_3_FORBIDDEN"):
            validate_execution_contract(contract)


if __name__ == "__main__":
    unittest.main()
