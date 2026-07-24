"""Mutation and integration tests for ordered isolated one-shot opening R1."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eligible_target_attribution_ordered_isolated_opening import (  # noqa: E402
    _load_verified_token_ledger,
    apply_ordered_isolated_opener,
    assert_opener_input,
    atomic_write_json,
    build_terminal_from_frozen_tokens,
    load_and_verify_config,
    load_json,
    mutate_background_namespace,
    sha256_file,
    validate_outputs,
)


class EligibleAttributionOrderedIsolatedOpeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = HERE.parents[2]
        cls.config_path = (
            cls.repo
            / "configs"
            / "ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json"
        )

    def _ledger(
        self,
        candidate_id: str = "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
    ) -> dict:
        return {
            "schema": (
                "blindassist_ustrf_eligible_target_attribution_token_ledger_r1"
            ),
            "stage": (
                "ELIGIBLE-TARGET-ATTRIBUTION-ORDERED-ISOLATED-OPENING-R1"
            ),
            "authority": (
                "truth_assisted_oracle_attribution_token_not_runtime_authority"
            ),
            "candidate_id": candidate_id,
            "event_id": "event-x",
            "source_id": "source-x",
            "sequence_id": "sequence-x",
            "frame_count": 3,
            "frames": [
                {
                    "frame_id": 10,
                    "source_capture_timestamp_ns": 10,
                    "reset_segment": 0,
                    "state_reset_before_frame": True,
                    "route_known": True,
                    "eligible_attributed_track_ids": [],
                    "background_namespace_active": True,
                    "background_namespace_opening_count": 1,
                },
                {
                    "frame_id": 11,
                    "source_capture_timestamp_ns": 11,
                    "reset_segment": 0,
                    "state_reset_before_frame": False,
                    "route_known": True,
                    "eligible_attributed_track_ids": [7],
                    "background_namespace_active": True,
                    "background_namespace_opening_count": 0,
                },
                {
                    "frame_id": 12,
                    "source_capture_timestamp_ns": 12,
                    "reset_segment": 0,
                    "state_reset_before_frame": False,
                    "route_known": True,
                    "eligible_attributed_track_ids": [7],
                    "background_namespace_active": True,
                    "background_namespace_opening_count": 0,
                },
            ],
        }

    def test_pre_token_state_cannot_open_or_consume_delivery(self) -> None:
        result = apply_ordered_isolated_opener(self._ledger(), 2)
        self.assertEqual(result["support_start_frame"], 11)
        self.assertEqual(result["qualification_frame"], 12)
        self.assertEqual(result["deliveries"][0]["frame_id"], 12)
        mutated = apply_ordered_isolated_opener(
            mutate_background_namespace(self._ledger()), 2
        )
        self.assertEqual(result["deliveries"], mutated["deliveries"])

    def test_c1_requires_same_attributed_track_continuity(self) -> None:
        ledger = self._ledger("C1_CAUSAL_ROUTE_RELATION_FSM")
        ledger["frames"][1]["eligible_attributed_track_ids"] = [7]
        ledger["frames"][2]["eligible_attributed_track_ids"] = [8]
        result = apply_ordered_isolated_opener(ledger, 2)
        self.assertEqual(result["deliveries"], [])

    def test_c2_event_identity_allows_track_handoff(self) -> None:
        ledger = self._ledger()
        ledger["frames"][1]["eligible_attributed_track_ids"] = [7]
        ledger["frames"][2]["eligible_attributed_track_ids"] = [8]
        result = apply_ordered_isolated_opener(ledger, 2)
        self.assertEqual(len(result["deliveries"]), 1)
        self.assertEqual(result["qualification_frame"], 12)

    def test_reset_breaks_qualification_run(self) -> None:
        ledger = self._ledger()
        ledger["frames"][2]["state_reset_before_frame"] = True
        ledger["frames"][2]["reset_segment"] = 1
        result = apply_ordered_isolated_opener(ledger, 2)
        self.assertEqual(result["deliveries"], [])

    def test_unknown_route_cannot_carry_or_create_token(self) -> None:
        ledger = self._ledger()
        ledger["frames"][1]["route_known"] = False
        ledger["frames"][1]["eligible_attributed_track_ids"] = []
        result = apply_ordered_isolated_opener(ledger, 2)
        self.assertEqual(result["deliveries"], [])
        contaminated = self._ledger()
        contaminated["frames"][1]["route_known"] = False
        with self.assertRaisesRegex(
            RuntimeError, "eligible_token_on_unknown_route"
        ):
            apply_ordered_isolated_opener(contaminated, 2)

    def test_sustained_token_delivers_once(self) -> None:
        ledger = self._ledger()
        ledger["frames"].append(
            {
                **ledger["frames"][-1],
                "frame_id": 13,
                "source_capture_timestamp_ns": 13,
            }
        )
        ledger["frame_count"] = 4
        result = apply_ordered_isolated_opener(ledger, 2)
        self.assertEqual(len(result["deliveries"]), 1)

    def test_opener_rejects_truth_or_baseline_state(self) -> None:
        for key in (
            "truth",
            "truth_box",
            "observed_tracks",
            "baseline_active_keys_after",
            "baseline_deliveries",
            "guard_events",
        ):
            ledger = self._ledger()
            ledger[key] = []
            with self.assertRaisesRegex(
                RuntimeError, "opener_ledger_keys_drift"
            ):
                assert_opener_input(ledger)

    def test_frame_gap_timestamp_and_segment_drift_fail_closed(self) -> None:
        mutations = []
        frame_gap = self._ledger()
        frame_gap["frames"][1]["frame_id"] = 12
        mutations.append((frame_gap, "opener_frame_id_gap"))
        timestamp = self._ledger()
        timestamp["frames"][1]["source_capture_timestamp_ns"] = 10
        mutations.append((timestamp, "opener_timestamp_not_monotonic"))
        segment = self._ledger()
        segment["frames"][1]["reset_segment"] = 1
        mutations.append((segment, "opener_segment_changed_without_reset"))
        for ledger, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    apply_ordered_isolated_opener(ledger, 2)

    def test_threshold_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "opener_min_alert_drift"):
            apply_ordered_isolated_opener(self._ledger(), 1)

    def test_config_authority_mutation_fails_closed(self) -> None:
        config = load_json(self.config_path)
        config["authority"]["selection"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                __import__("json").dumps(config), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "authority_drift"):
                load_and_verify_config(self.repo, path)

    def test_inventory_token_mutation_is_detected(self) -> None:
        config, _ = load_and_verify_config(self.repo, self.config_path)
        inventory_path = (
            self.repo
            / config["outputs"]["root"]
            / config["outputs"]["token_inventory"]
        )
        inventory = load_json(inventory_path)
        source_row = inventory["inventory"][0]
        source_ledger = load_json(self.repo / source_row["path"])
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.json"
            atomic_write_json(ledger_path, source_ledger)
            row = {
                **source_row,
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
            }
            mutated = {
                **source_ledger,
                "event_id": f"{source_ledger['event_id']}-mutated",
            }
            atomic_write_json(ledger_path, mutated)
            with self.assertRaisesRegex(RuntimeError, "token_ledger_sha_drift"):
                _load_verified_token_ledger(self.repo, row)

    def test_terminal_exact_recomputation_and_validation(self) -> None:
        terminal = build_terminal_from_frozen_tokens(
            self.repo, self.config_path
        )
        self.assertTrue(terminal["mechanism_gate"]["passed"])
        self.assertEqual(
            terminal["mechanism_gate"]["token_qualified_cells"], 33
        )
        validation = validate_outputs(self.repo, self.config_path)
        self.assertEqual(validation["status"], "VALID")

    def test_terminal_opener_phase_does_not_decode_raw_truth(self) -> None:
        with patch(
            "eligible_target_attribution_ordered_isolated_opening."
            "_load_truth_index",
            side_effect=AssertionError("raw truth decoded in opener phase"),
        ):
            terminal = build_terminal_from_frozen_tokens(
                self.repo, self.config_path
            )
        self.assertTrue(terminal["mechanism_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
