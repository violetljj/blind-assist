from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate_stage_c_d3_q0_screening as aggregator
import stage_c_d3_q0_common as common


CONTRACT_SHA = "b" * 64
ROSTER_SHA = "c" * 64
CARRY_AUTHORITY = {
    "q0_protocol_sha256": "1" * 64,
    "metadata_roster_sha256": ROSTER_SHA,
    "q0_execution_contract_sha256": "2" * 64,
    "q0_invalid_result_sha256": "3" * 64,
    "q0_screening_invalid_sha256": "4" * 64,
}


def _slot(index: int) -> dict[str, object]:
    return {
        "session_id": f"{index:064x}",
        "d3_roster_slot_index": index,
        "metadata_eligible_rank": index,
        "role": "d3_q0_locked_truth_screening_slot",
        "media_pose_content_opened": False,
        "support_or_truth_computed": False,
        "effect_computed": False,
    }


def _slots() -> list[dict[str, object]]:
    return [_slot(index) for index in range(1, 41)]


def _hashes(slot_attempt_sha256: str = "a" * 64) -> dict[str, str]:
    return {
        key: (
            slot_attempt_sha256
            if key == "slot_attempt_sha256"
            else "a" * 64
        )
        for key in common.AUTHORITY_HASH_KEYS
    }


def _screening_attempt() -> dict[str, object]:
    return {
        "schema": common.SCREENING_ATTEMPT_SCHEMA,
        "status": common.SCREENING_ATTEMPT_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "contract_sha256": CONTRACT_SHA,
        "roster_sha256": ROSTER_SHA,
        "first_slot_index": 2,
        "first_network_request_started": False,
        "slot_replacement_authorized": False,
        "budget_expansion_authorized": False,
    }


def _carry(slot: dict[str, object]) -> dict[str, object]:
    return {
        "schema": common.CARRY_FORWARD_SCHEMA,
        "terminal": common.CARRY_FORWARD_TERMINAL,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "q0_1_execution_contract_sha256": CONTRACT_SHA,
        **CARRY_AUTHORITY,
        "original_slot_index": 1,
        "session_id": slot["session_id"],
        "burn_reason": "SCHEMA_INVALID_AFTER_MEDIA_SUPPORT_TRUTH_OPEN",
        "original_attempt_durable": True,
        "media_support_truth_opened": True,
        "selector_schema_valid": False,
        "selector_admitted": False,
        "permanently_burned": True,
        "counts_toward_original_40_budget": True,
        "counts_as_qualified": False,
        "counts_as_not_qualified": False,
        "counts_as_slot_failure": False,
        "reopen_authorized": False,
        "recompute_authorized": False,
        "rerun_authorized": False,
        "replacement_authorized": False,
        "first_remaining_original_slot": 2,
        "maximum_remaining_original_slots": 39,
        "preserve_original_indices": True,
        "preserve_original_order": True,
        "sealed_payload_read": False,
        "invalid_selector_read": False,
        "outcome_fields_imported": False,
    }


def _slot_attempt(slot: dict[str, object]) -> dict[str, object]:
    return {
        "schema": common.SLOT_ATTEMPT_SCHEMA,
        "status": common.SLOT_ATTEMPT_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "contract_sha256": CONTRACT_SHA,
        "roster_sha256": ROSTER_SHA,
        "slot_index": slot["d3_roster_slot_index"],
        "session_id": slot["session_id"],
        "internal_retries_per_request": 3,
        "content_request_started": False,
        "slot_retry_authorized": False,
        "source_replacement_authorized": False,
        "candidate_arm_clearance_authorized": False,
        "effect_metric_authorized": False,
    }


def _selector(
    slot: dict[str, object],
    *,
    qualified: bool,
    slot_attempt_sha256: str = "a" * 64,
) -> dict[str, object]:
    strata = []
    for height, horizon in common.STRATA:
        risk = 5 if qualified else 4
        safe = 21 if qualified else 22
        known = risk + safe
        gates = {
            "coverage": known / 252.0 >= 0.1,
            "risk": risk >= 5,
            "safe": safe >= 20,
            "unknown_to_safe": True,
        }
        strata.append(
            {
                "height": height,
                "horizon_s": horizon,
                "denominator": 252,
                "common_known_count": known,
                "common_known_coverage": known / 252.0,
                "truth_risk_count": risk,
                "truth_safe_count": safe,
                "unknown_to_safe_violation_count": 0,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    return {
        "schema": common.SELECTOR_SCHEMA,
        "terminal": (
            common.SELECTOR_QUALIFIED
            if qualified
            else common.SELECTOR_NOT_QUALIFIED
        ),
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": CONTRACT_SHA,
        "metadata_roster_sha256": ROSTER_SHA,
        "slot_index": slot["d3_roster_slot_index"],
        "session_id": slot["session_id"],
        "source_authority_and_content_hashes": _hashes(
            slot_attempt_sha256
        ),
        "strata": strata,
        "qualified": qualified,
    }


def _failure(
    slot: dict[str, object],
    slot_attempt_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema": common.FAILURE_SCHEMA,
        "terminal": common.SLOT_FAILURE_TERMINAL,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": CONTRACT_SHA,
        "metadata_roster_sha256": ROSTER_SHA,
        "slot_attempt_sha256": slot_attempt_sha256,
        "slot_index": slot["d3_roster_slot_index"],
        "session_id": slot["session_id"],
        "error": {"type": "FixtureError", "message": "fixture"},
        "slot_consumed": True,
        "rerun_authorized": False,
        "source_replacement_authorized": False,
    }


def _write(path: Path, value: dict[str, object]) -> None:
    common.write_json_exclusive_fsync(path, value)


def _ensure_control(root: Path, slots: list[dict[str, object]]) -> None:
    attempt = common.aggregate_paths(root)["screening_attempt"]
    if not attempt.exists():
        _write(attempt, _screening_attempt())
    carry = common.slot_layout(root, slots[0])["carry_forward"]
    if not carry.exists():
        _write(carry, _carry(slots[0]))


def _write_slot_attempt(
    root: Path,
    slots: list[dict[str, object]],
    slot: dict[str, object],
) -> str:
    if int(slot["d3_roster_slot_index"]) < 2:
        raise AssertionError("slot 1 is permanently burned")
    _ensure_control(root, slots)
    path = common.slot_layout(root, slot)["attempt"]
    if not path.exists():
        _write(path, _slot_attempt(slot))
    return common.sha256(path)


def _write_selector(
    root: Path,
    slots: list[dict[str, object]],
    slot: dict[str, object],
    qualified: bool,
) -> None:
    layout = common.slot_layout(root, slot)
    attempt_sha = _write_slot_attempt(root, slots, slot)
    _write(
        layout["selector"],
        _selector(
            slot,
            qualified=qualified,
            slot_attempt_sha256=attempt_sha,
        ),
    )


def _state(root: Path, slots: list[dict[str, object]]) -> dict[str, object]:
    return common.scan_screening_state(
        root,
        slots,
        CONTRACT_SHA,
        ROSTER_SHA,
        CARRY_AUTHORITY,
    )


def _context(
    root: Path,
    slots: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "root": root,
        "slots": slots,
        "contract_sha256": CONTRACT_SHA,
        "roster_sha256": ROSTER_SHA,
        "carry_forward_authority": CARRY_AUTHORITY,
    }


class D3Q01StateTest(unittest.TestCase):
    def test_slot_layout_uses_short_token_and_carry_path(self) -> None:
        layout = common.slot_layout(Path("screening"), _slot(7))
        self.assertEqual(
            {
                "slot_root",
                "attempt",
                "content_index",
                "sealed_payload",
                "selector",
                "failure",
                "carry_forward",
            },
            set(layout),
        )
        self.assertTrue(layout["slot_root"].name.startswith("slot-07-"))

    def test_selector_closed_schema_accepts_only_nested_attempt_hash(self) -> None:
        slot = _slot(2)
        value = _selector(slot, qualified=True)
        self.assertNotIn("slot_attempt_sha256", value)
        self.assertIn(
            "slot_attempt_sha256",
            value["source_authority_and_content_hashes"],
        )
        self.assertTrue(
            common.validate_selector(
                value, slot, CONTRACT_SHA, ROSTER_SHA
            )["qualified"]
        )

    def test_selector_rejects_synthetic_duplicate_top_level_field(self) -> None:
        slot = _slot(2)
        value = _selector(slot, qualified=False)
        value["slot_attempt_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "extra"):
            common.validate_selector(
                value, slot, CONTRACT_SHA, ROSTER_SHA
            )

    def test_selector_rejects_gate_or_count_inconsistency(self) -> None:
        slot = _slot(2)
        value = _selector(slot, qualified=True)
        value["strata"][0]["truth_safe_count"] = 20
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            common.validate_selector(
                value, slot, CONTRACT_SHA, ROSTER_SHA
            )

    def test_failure_closed_schema(self) -> None:
        slot = _slot(2)
        value = _failure(slot)
        self.assertTrue(
            common.validate_failure(
                value, slot, CONTRACT_SHA, ROSTER_SHA
            )["slot_consumed"]
        )
        value["error"]["secret"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "extra"):
            common.validate_failure(
                value, slot, CONTRACT_SHA, ROSTER_SHA
            )

    def test_carry_forward_is_closed_and_outcome_free(self) -> None:
        slot = _slot(1)
        value = _carry(slot)
        self.assertFalse(value["sealed_payload_read"])
        self.assertFalse(value["invalid_selector_read"])
        self.assertFalse(value["outcome_fields_imported"])
        common.validate_carry_forward(
            value, slot, CONTRACT_SHA, CARRY_AUTHORITY
        )
        value["qualified"] = False
        with self.assertRaisesRegex(ValueError, "extra"):
            common.validate_carry_forward(
                value, slot, CONTRACT_SHA, CARRY_AUTHORITY
            )

    def test_carry_forward_authority_drift_is_rejected(self) -> None:
        slot = _slot(1)
        value = _carry(slot)
        value["q0_invalid_result_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "authority mismatch"):
            common.validate_carry_forward(
                value, slot, CONTRACT_SHA, CARRY_AUTHORITY
            )

    def test_exclusive_writer_is_durable_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            final = Path(temp) / "receipt.json"
            common.write_json_exclusive_fsync(final, {"ok": True})
            self.assertEqual(
                common.durable_json_sha256({"ok": True}),
                common.sha256(final),
            )
            with self.assertRaises(FileExistsError):
                common.write_json_exclusive_fsync(final, {"ok": False})

    def test_empty_root_requires_control_plane_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = _state(Path(temp), _slots())
        self.assertTrue(state["control_plane_uninitialized"])
        self.assertEqual(0, state["consumed_count"])
        self.assertIsNone(state["next_slot"])

    def test_control_plane_points_to_original_slot_2(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_control(root, slots)
            state = _state(root, slots)
        self.assertEqual(1, state["consumed_count"])
        self.assertEqual(0, state["newly_opened_count"])
        self.assertEqual(2, state["next_slot"]["d3_roster_slot_index"])
        self.assertEqual([], state["qualified_rows"])
        self.assertEqual([], state["failure_rows"])

    def test_slot_1_any_noncarry_artifact_is_rejected(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_control(root, slots)
            _write(
                common.slot_layout(root, slots[0])["attempt"],
                _slot_attempt(slots[0]),
            )
            with self.assertRaisesRegex(ValueError, "beyond"):
                _state(root, slots)

    def test_interrupted_slot_2_is_not_reopened_as_next(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_slot_attempt(root, slots, slots[1])
            state = _state(root, slots)
        self.assertIsNone(state["next_slot"])
        self.assertEqual(
            2, state["interrupted_slot"]["d3_roster_slot_index"]
        )

    def test_slot_2_gap_before_slot_3_is_rejected(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_control(root, slots)
            _write_selector(root, slots, slots[2], False)
            with self.assertRaisesRegex(ValueError, "prefix gap"):
                _state(root, slots)

    def test_selector_and_failure_are_ambiguous(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slot = slots[1]
            layout = common.slot_layout(root, slot)
            attempt_sha = _write_slot_attempt(root, slots, slot)
            _write(
                layout["selector"],
                _selector(
                    slot,
                    qualified=False,
                    slot_attempt_sha256=attempt_sha,
                ),
            )
            _write(layout["failure"], _failure(slot, attempt_sha))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                _state(root, slots)

    def test_sixth_qualified_source_stops_at_original_slot_7(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[1:7]:
                _write_selector(root, slots, slot, True)
            state = _state(root, slots)
        self.assertEqual(common.QUALIFICATION_TERMINAL, state["terminal"])
        self.assertEqual(7, state["consumed_count"])
        self.assertEqual(6, state["newly_opened_count"])
        self.assertEqual(6, len(state["qualified_rows"]))

    def test_any_artifact_after_sixth_qualified_is_rejected(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[1:7]:
                _write_selector(root, slots, slot, True)
            _write_selector(root, slots, slots[7], False)
            with self.assertRaisesRegex(ValueError, "after sixth"):
                _state(root, slots)

    def test_burn_plus_39_new_slots_is_budget_terminal(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, slot in enumerate(slots[1:]):
                _write_selector(root, slots, slot, index < 5)
            state = _state(root, slots)
        self.assertEqual(common.BUDGET_TERMINAL, state["terminal"])
        self.assertEqual(40, state["consumed_count"])
        self.assertEqual(39, state["newly_opened_count"])
        self.assertEqual(5, len(state["qualified_rows"]))

    def test_aggregate_reads_no_sealed_payload_and_validates_selection(
        self,
    ) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[1:7]:
                layout = common.slot_layout(root, slot)
                _write_selector(root, slots, slot, True)
                layout["sealed_payload"].write_text(
                    "not-json and forbidden to read",
                    encoding="utf-8",
                )
            paths = common.aggregate_paths(root)
            context = _context(root, slots)
            original_scan = aggregator.scan_screening_state

            def scan_after_attempt(*args: object, **kwargs: object) -> object:
                self.assertTrue(paths["aggregate_attempt"].is_file())
                return original_scan(*args, **kwargs)

            with mock.patch.object(
                aggregator,
                "scan_screening_state",
                side_effect=scan_after_attempt,
            ):
                result = aggregator.aggregate_screening(context)
            self.assertFalse(result["sealed_payload_read"])
            self.assertEqual(1, result["carry_forward_burned_slot_count"])
            self.assertEqual(0, result["failure_receipt_count"])
            self.assertEqual(
                result,
                common.validate_selection(
                    paths["selection"],
                    slots,
                    CONTRACT_SHA,
                    ROSTER_SHA,
                    CARRY_AUTHORITY,
                ),
            )

    def test_orphan_aggregate_attempt_freezes_without_rescan(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = common.aggregate_paths(root)
            _write(
                paths["aggregate_attempt"],
                {
                    "schema": common.AGGREGATE_ATTEMPT_SCHEMA,
                    "status": common.AGGREGATE_ATTEMPT_STATUS,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "execution_contract_sha256": CONTRACT_SHA,
                    "metadata_roster_sha256": ROSTER_SHA,
                    "selector_or_failure_receipts_read_before_attempt": False,
                    "sealed_payload_read": False,
                    "rerun_authorized": False,
                },
            )
            with (
                mock.patch.object(
                    aggregator,
                    "scan_screening_state",
                    side_effect=AssertionError("must not rescan"),
                ),
                self.assertRaisesRegex(ValueError, "prior durable"),
            ):
                aggregator.aggregate_screening(_context(root, slots))
            invalid = common.load_json(paths["invalid"])
            self.assertEqual(
                "D3_QUALIFICATION_INVALID_STOP", invalid["terminal"]
            )

    def test_validate_selection_rejects_nonexact_receipts(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[1:7]:
                _write_selector(root, slots, slot, True)
            selection = aggregator.aggregate_screening(
                _context(root, slots)
            )
            cases = []
            reordered = json.loads(json.dumps(selection))
            reordered["selected_sources"][0], reordered[
                "selected_sources"
            ][1] = (
                reordered["selected_sources"][1],
                reordered["selected_sources"][0],
            )
            cases.append((reordered, "frozen slot order"))
            wrong_hash = json.loads(json.dumps(selection))
            wrong_hash["selected_sources"][0][
                "selector_sha256"
            ] = "d" * 64
            cases.append((wrong_hash, "hash mismatch"))
            extra = json.loads(json.dumps(selection))
            extra["unexpected"] = True
            cases.append((extra, "extra"))
            for value, pattern in cases:
                with (
                    self.subTest(pattern=pattern),
                    self.assertRaisesRegex(ValueError, pattern),
                ):
                    common.validate_selection(
                        value,
                        slots,
                        CONTRACT_SHA,
                        ROSTER_SHA,
                        CARRY_AUTHORITY,
                    )

    def test_aggregate_budget_terminal_preserves_burn_accounting(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, slot in enumerate(slots[1:]):
                _write_selector(root, slots, slot, index < 5)
            result = aggregator.aggregate_screening(
                _context(root, slots)
            )
        self.assertEqual(common.BUDGET_TERMINAL, result["terminal"])
        self.assertEqual(40, result["consumed_slot_count"])
        self.assertEqual(39, result["newly_opened_slot_count"])
        self.assertEqual(1, result["carry_forward_burned_slot_count"])
        self.assertEqual(5, result["qualified_source_count"])

    def test_aggregate_invalid_prefix_writes_only_invalid_terminal(
        self,
    ) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_control(root, slots)
            _write_selector(root, slots, slots[2], False)
            with self.assertRaisesRegex(ValueError, "prefix gap"):
                aggregator.aggregate_screening(_context(root, slots))
            paths = common.aggregate_paths(root)
            self.assertTrue(paths["invalid"].exists())
            self.assertFalse(paths["selection"].exists())
            self.assertFalse(paths["exhausted"].exists())

    def test_validate_contract_rejects_incomplete_parent_chain(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "protocol.json"
            roster_path = root / "roster.json"
            contract_path = root / "contract.json"
            _write(
                protocol_path,
                {
                    "schema": common.PROTOCOL_SCHEMA,
                    "status": common.PROTOCOL_STATUS,
                    "source_pool": {
                        "maximum_truth_screened_slots": 40,
                        "required_qualified_sources": 6,
                    },
                },
            )
            _write(
                roster_path,
                {
                    "schema": common.ROSTER_SCHEMA,
                    "terminal": common.ROSTER_TERMINAL,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "requested_roster_slot_count": 40,
                    "locked_roster_slot_count": 40,
                    "exclusions": {"excluded_parent_count": 84},
                    "locked_slots": slots,
                    "firewall": {"future_truth_opened": False},
                    "authorization": {
                        "freeze_qualifier_effect_execution_contract": True,
                        "d3_effect_authorized": False,
                    },
                },
            )
            implementation = Path(aggregator.__file__).resolve()
            _write(
                contract_path,
                {
                    "schema": common.CONTRACT_SCHEMA,
                    "status": common.CONTRACT_STATUS,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "parents": {
                        "d3_q0_protocol": {
                            "path": str(protocol_path),
                            "sha256": common.sha256(protocol_path),
                        },
                        "metadata_roster": {
                            "path": str(roster_path),
                            "sha256": common.sha256(roster_path),
                        },
                    },
                    "implementations": {
                        aggregator.IMPLEMENTATION_KEY: {
                            "path": str(implementation),
                            "sha256": common.sha256(implementation),
                        }
                    },
                },
            )
            with self.assertRaisesRegex(
                ValueError, "metadata_roster_result"
            ):
                common.validate_execution_contract(
                    contract_path,
                    aggregator.IMPLEMENTATION_KEY,
                    implementation,
                    verify_git=False,
                )


if __name__ == "__main__":
    unittest.main()
