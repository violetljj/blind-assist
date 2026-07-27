from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.rgb_algorithm_cid_sims_floor3_2_cross_sequence_holdout_r0 import (
    formal_runner as runner,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_"
    "CROSS_SEQUENCE_HOLDOUT_R0_CONTRACT_2026-07-27.json"
)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def summary(index: int, role: str) -> dict:
    anchor = Decimal("1000")
    start = anchor + index * Decimal("10")
    return {
        "window_index": index,
        "start_timestamp_s": str(start),
        "end_timestamp_s": str(start + Decimal("10")),
        "candidate_pair_count": 299,
        "role": role,
        "median_signed_radial_expansion_per_s": (
            0.2 if role == "POSITIVE_APPROACH_WINDOW" else 0.0
        ),
        "first_positive_geometry_timestamp_s": (
            float(start) + 0.1
            if role == "POSITIVE_APPROACH_WINDOW"
            else None
        ),
    }


class HoldoutRulesTest(unittest.TestCase):
    def test_candidate_grid_uses_all_complete_windows(self) -> None:
        contract = load_contract()
        timestamps = [
            Decimal("1000") + Decimal(index) / Decimal("30")
            for index in range(3601)
        ]
        windows = runner.candidate_windows(timestamps, contract)
        self.assertEqual(
            [item["window_index"] for item in windows],
            list(range(12)),
        )
        self.assertTrue(all(item["identity_eligible"] for item in windows))

    def test_candidate_grid_rejects_empty_depth_identity(self) -> None:
        contract = load_contract()
        with self.assertRaisesRegex(ValueError, "NO_DEPTH_TIMESTAMPS"):
            runner.candidate_windows([], contract)

    def test_selection_requires_exact_two_plus_two(self) -> None:
        contract = load_contract()
        rows = [
            summary(3, "POSITIVE_APPROACH_WINDOW"),
            summary(5, "POSITIVE_APPROACH_WINDOW"),
            summary(7, "BELOW_TRIGGER_REFERENCE_WINDOW"),
        ]
        self.assertEqual(runner.select_windows(rows, contract), [])

    def test_selection_uses_joint_spacing_and_earliest_tuple(self) -> None:
        contract = load_contract()
        rows = [
            summary(3, "POSITIVE_APPROACH_WINDOW"),
            summary(4, "BELOW_TRIGGER_REFERENCE_WINDOW"),
            summary(5, "BELOW_TRIGGER_REFERENCE_WINDOW"),
            summary(6, "POSITIVE_APPROACH_WINDOW"),
            summary(7, "BELOW_TRIGGER_REFERENCE_WINDOW"),
            summary(8, "POSITIVE_APPROACH_WINDOW"),
            summary(9, "BELOW_TRIGGER_REFERENCE_WINDOW"),
            summary(10, "POSITIVE_APPROACH_WINDOW"),
        ]
        selected = runner.select_windows(rows, contract)
        self.assertEqual(
            [item["window_index"] for item in selected],
            [3, 5, 7, 10],
        )

    def test_role_requires_fixed_denominator_fraction_and_run(self) -> None:
        contract = load_contract()
        window = {
            "window_index": 3,
            "start_timestamp_s": "0",
            "end_timestamp_s": "10",
            "frame_count": 300,
            "pair_count": 299,
            "identity_eligible": True,
            "identity_reason": None,
        }
        rows = []
        for index in range(299):
            rows.append(
                {
                    "window_index": 3,
                    "pair_index": index,
                    "previous_timestamp_s": index / 30,
                    "current_timestamp_s": (index + 1) / 30,
                    "geometry_evaluable": True,
                    "geometry_abstention_reason": None,
                    "geometry_signed_radial_expansion_per_s": (
                        0.1 if index < 240 else 0.02
                    ),
                    "geometry_band": (
                        "POSITIVE_APPROACH_GEOMETRY"
                        if index < 240
                        else "WEAK_POSITIVE_RADIAL"
                    ),
                }
            )
        result = runner.summarize_geometry_window(
            window, rows, contract
        )
        self.assertEqual(result["role"], "POSITIVE_APPROACH_WINDOW")

    def test_abstention_cannot_be_imputed_into_role(self) -> None:
        contract = load_contract()
        window = {
            "window_index": 3,
            "start_timestamp_s": "0",
            "end_timestamp_s": "10",
            "frame_count": 300,
            "pair_count": 299,
            "identity_eligible": True,
            "identity_reason": None,
        }
        rows = [
            {
                "window_index": 3,
                "pair_index": index,
                "previous_timestamp_s": index / 30,
                "current_timestamp_s": (index + 1) / 30,
                "geometry_evaluable": index < 200,
                "geometry_abstention_reason": (
                    None if index < 200 else "NO_VALID_GEOMETRY_SAMPLES"
                ),
                "geometry_signed_radial_expansion_per_s": 0.2,
                "geometry_band": (
                    "POSITIVE_APPROACH_GEOMETRY" if index < 200 else None
                ),
            }
            for index in range(299)
        ]
        result = runner.summarize_geometry_window(
            window, rows, contract
        )
        self.assertEqual(result["role"], "AMBIGUOUS_OR_INELIGIBLE")

    def test_direction_requires_every_positive_to_exceed_every_low(self) -> None:
        contract = load_contract()
        selected = [
            summary(3, "POSITIVE_APPROACH_WINDOW"),
            summary(5, "BELOW_TRIGGER_REFERENCE_WINDOW"),
            summary(7, "POSITIVE_APPROACH_WINDOW"),
            summary(9, "BELOW_TRIGGER_REFERENCE_WINDOW"),
        ]
        rgb = []
        for item in selected:
            positive = item["role"] == "POSITIVE_APPROACH_WINDOW"
            rgb.append(
                {
                    "window_index": item["window_index"],
                    "role": item["role"],
                    "window_start_s": float(item["start_timestamp_s"]),
                    "window_end_s": float(item["end_timestamp_s"]),
                    "candidate_pair_count": 299,
                    "evaluable_pair_count": 299,
                    "pair_coverage": 1.0,
                    "abstention_count": 0,
                    "abstention_reasons": {},
                    "median_compensated_expansion_per_s": (
                        0.2 if positive else 0.0
                    ),
                    "trigger_count": 299 if positive else 0,
                    "trigger_coverage_fixed_denominator": (
                        1.0 if positive else 0.0
                    ),
                    "trigger_coverage_evaluable": (
                        1.0 if positive else 0.0
                    ),
                    "first_trigger_delay_s": 0.1 if positive else None,
                    "longest_consecutive_trigger_pair_count": (
                        299 if positive else 0
                    ),
                    "longest_consecutive_trigger_duration_s": (
                        9.9 if positive else 0.0
                    ),
                }
            )
        terminal, aggregate = runner.aggregate_rgb(
            selected, rgb, [], contract
        )
        self.assertEqual(
            terminal, "CROSS_SEQUENCE_DIRECTION_REPLICATED / VALID"
        )
        self.assertTrue(aggregate["direction_replicated"])

    def test_runtime_identity_is_frozen(self) -> None:
        self.assertEqual(
            runner.runtime_identity(),
            load_contract()["runtime_identity"],
        )

    def test_identity_manifest_precedes_rgb_member_read_in_source(self) -> None:
        source = (
            Path(runner.__file__).read_text(encoding="utf-8")
        )
        write_position = source.index(
            'run_dir / "selected_rgb_identity.json", identity'
        )
        read_position = source.index(
            "manifest = materialize_selected_rgb_cache("
        )
        self.assertLess(write_position, read_position)

    def test_validator_does_not_import_formal_runner_or_rgb_producer(self) -> None:
        validator_source = (
            Path(runner.__file__).with_name("validator.py").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("import formal_runner", validator_source)
        self.assertNotIn(
            "rgb_algorithm_development_canary_cid_sims_r0",
            validator_source,
        )

    def test_contract_names_cross_sequence_not_cross_source(self) -> None:
        contract = load_contract()
        self.assertEqual(
            contract["authority"]["maximum_claim"],
            "CROSS_SEQUENCE_SAME_SOURCE_DEVELOPMENT_HOLDOUT_ONLY",
        )
        self.assertFalse(
            contract["authority"]["cross_source_confirmation_authorized"]
        )


if __name__ == "__main__":
    unittest.main()
