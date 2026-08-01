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


def _screening_attempt(
    *,
    contract_sha: str = "b" * 64,
    roster_sha: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema": common.SCREENING_ATTEMPT_SCHEMA,
        "status": common.SCREENING_ATTEMPT_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "contract_sha256": contract_sha,
        "roster_sha256": roster_sha,
        "first_slot_index": 1,
        "first_network_request_started": False,
        "slot_replacement_authorized": False,
        "budget_expansion_authorized": False,
    }


def _slot_attempt(
    slot: dict[str, object],
    *,
    contract_sha: str = "b" * 64,
    roster_sha: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema": common.SLOT_ATTEMPT_SCHEMA,
        "status": common.SLOT_ATTEMPT_STATUS,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "contract_sha256": contract_sha,
        "roster_sha256": roster_sha,
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
    contract_sha: str = "b" * 64,
    roster_sha: str = "c" * 64,
    slot_attempt_sha256: str = "a" * 64,
) -> dict[str, object]:
    strata = []
    for height, horizon in common.STRATA:
        risk = 5 if qualified else 4
        safe = 21 if qualified else 22
        common_count = risk + safe
        gates = {
            "coverage": common_count / 252.0 >= 0.1,
            "risk": risk >= 5,
            "safe": safe >= 20,
            "unknown_to_safe": True,
        }
        strata.append(
            {
                "height": height,
                "horizon_s": horizon,
                "denominator": 252,
                "common_known_count": common_count,
                "common_known_coverage": common_count / 252.0,
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
        "execution_contract_sha256": contract_sha,
        "metadata_roster_sha256": roster_sha,
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
    *,
    contract_sha: str = "b" * 64,
    roster_sha: str = "c" * 64,
    slot_attempt_sha256: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema": common.FAILURE_SCHEMA,
        "terminal": common.SLOT_FAILURE_TERMINAL,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": contract_sha,
        "metadata_roster_sha256": roster_sha,
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


def _ensure_screening_attempt(root: Path) -> None:
    path = common.aggregate_paths(root)["screening_attempt"]
    if not path.exists():
        _write(path, _screening_attempt())


def _write_slot_attempt(
    root: Path,
    slot: dict[str, object],
) -> str:
    _ensure_screening_attempt(root)
    path = common.slot_layout(root, slot)["attempt"]
    if not path.exists():
        _write(path, _slot_attempt(slot))
    return common.sha256(path)


def _write_selector(
    root: Path,
    slot: dict[str, object],
    qualified: bool,
) -> None:
    layout = common.slot_layout(root, slot)
    attempt_sha256 = _write_slot_attempt(root, slot)
    _write(
        layout["selector"],
        _selector(
            slot,
            qualified=qualified,
            slot_attempt_sha256=attempt_sha256,
        ),
    )


class D3Q0StateTest(unittest.TestCase):
    def test_slot_layout_uses_short_canonical_token(self) -> None:
        root = Path("screening")
        slot = _slot(7)
        layout = common.slot_layout(root, slot)
        token = common.hashlib.sha256(
            str(slot["session_id"]).encode("ascii")
        ).hexdigest()[:12]
        self.assertEqual(
            f"slot-07-{token}",
            layout["slot_root"].name,
        )
        self.assertEqual(
            {
                "slot_root",
                "attempt",
                "content_index",
                "sealed_payload",
                "selector",
                "failure",
            },
            set(layout),
        )

    def test_selector_closed_schema_and_arithmetic(self) -> None:
        slot = _slot(1)
        value = _selector(slot, qualified=True)
        self.assertTrue(
            common.validate_selector(
                value,
                slot,
                "b" * 64,
                "c" * 64,
            )["qualified"]
        )
        value["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "extra"):
            common.validate_selector(
                value,
                slot,
                "b" * 64,
                "c" * 64,
            )

    def test_selector_rejects_gate_or_count_inconsistency(self) -> None:
        slot = _slot(1)
        value = _selector(slot, qualified=True)
        value["strata"][0]["truth_safe_count"] = 20
        with self.assertRaisesRegex(ValueError, "arithmetic"):
            common.validate_selector(
                value,
                slot,
                "b" * 64,
                "c" * 64,
            )

    def test_failure_closed_schema(self) -> None:
        slot = _slot(1)
        value = _failure(slot)
        self.assertTrue(
            common.validate_failure(
                value,
                slot,
                "b" * 64,
                "c" * 64,
            )["slot_consumed"]
        )
        value["error"]["secret"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "extra"):
            common.validate_failure(
                value,
                slot,
                "b" * 64,
                "c" * 64,
            )

    def test_exclusive_writer_uses_tmp_and_rejects_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            final = Path(temp) / "receipt.json"
            temporary = final.with_name(final.name + ".tmp")
            common.write_json_exclusive_fsync(final, {"ok": True})
            self.assertEqual({"ok": True}, common.load_json(final))
            self.assertEqual(
                common.durable_json_sha256({"ok": True}),
                common.sha256(final),
            )
            self.assertFalse(temporary.exists())
            with self.assertRaises(FileExistsError):
                common.write_json_exclusive_fsync(final, {"ok": False})
            blocked = Path(temp) / "blocked.json"
            blocked_temporary = blocked.with_name(blocked.name + ".tmp")
            blocked_temporary.write_text("owned-by-prior-attempt", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                common.write_json_exclusive_fsync(blocked, {"ok": False})
            self.assertFalse(blocked.exists())
            self.assertEqual(
                "owned-by-prior-attempt",
                blocked_temporary.read_text(encoding="utf-8"),
            )

    def test_empty_state_points_to_first_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = common.scan_screening_state(
                Path(temp),
                _slots(),
                "b" * 64,
                "c" * 64,
            )
        self.assertEqual(0, state["consumed_count"])
        self.assertEqual(1, state["next_slot"]["d3_roster_slot_index"])
        self.assertIsNone(state["terminal"])

    def test_empty_slot_directory_is_safe_before_durable_attempt(
        self,
    ) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_screening_attempt(root)
            common.slot_layout(root, slots[0])["slot_root"].mkdir()
            state = common.scan_screening_state(
                root,
                slots,
                "b" * 64,
                "c" * 64,
            )
        self.assertEqual(0, state["consumed_count"])
        self.assertEqual(1, state["next_slot"]["d3_roster_slot_index"])

    def test_consumed_selector_requires_bound_durable_attempt(
        self,
    ) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _ensure_screening_attempt(root)
            layout = common.slot_layout(root, slots[0])
            _write(
                layout["selector"],
                _selector(slots[0], qualified=False),
            )
            with self.assertRaisesRegex(
                ValueError,
                "lacks a durable attempt",
            ):
                common.scan_screening_state(
                    root,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )

    def test_interrupted_slot_is_not_next_slot(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = common.slot_layout(root, slots[0])
            _write_slot_attempt(root, slots[0])
            state = common.scan_screening_state(
                root,
                slots,
                "b" * 64,
                "c" * 64,
            )
        self.assertIsNone(state["next_slot"])
        self.assertEqual(
            1,
            state["interrupted_slot"]["d3_roster_slot_index"],
        )

    def test_prefix_gap_is_rejected(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_selector(root, slots[0], False)
            _write_selector(root, slots[2], False)
            with self.assertRaisesRegex(ValueError, "prefix gap"):
                common.scan_screening_state(
                    root,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )

    def test_selector_and_failure_is_ambiguous(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = common.slot_layout(root, slots[0])
            attempt_sha256 = _write_slot_attempt(root, slots[0])
            _write(
                layout["selector"],
                _selector(
                    slots[0],
                    qualified=False,
                    slot_attempt_sha256=attempt_sha256,
                ),
            )
            _write(
                layout["failure"],
                _failure(
                    slots[0],
                    slot_attempt_sha256=attempt_sha256,
                ),
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                common.scan_screening_state(
                    root,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )

    def test_sixth_qualified_source_is_success_terminal(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[:6]:
                _write_selector(root, slot, True)
            state = common.scan_screening_state(
                root,
                slots,
                "b" * 64,
                "c" * 64,
            )
        self.assertEqual(common.QUALIFICATION_TERMINAL, state["terminal"])
        self.assertEqual(6, state["consumed_count"])
        self.assertEqual(6, len(state["qualified_rows"]))

    def test_any_artifact_after_sixth_qualified_is_rejected(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[:6]:
                _write_selector(root, slot, True)
            _write_selector(root, slots[6], False)
            with self.assertRaisesRegex(ValueError, "after sixth"):
                common.scan_screening_state(
                    root,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )

    def test_all_40_consumed_without_six_is_budget_terminal(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, slot in enumerate(slots):
                _write_selector(root, slot, index < 5)
            state = common.scan_screening_state(
                root,
                slots,
                "b" * 64,
                "c" * 64,
            )
        self.assertEqual(common.BUDGET_TERMINAL, state["terminal"])
        self.assertEqual(40, state["consumed_count"])
        self.assertEqual(5, len(state["qualified_rows"]))

    def test_aggregate_reads_no_sealed_payload_and_is_exclusive(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[:6]:
                layout = common.slot_layout(root, slot)
                _write_selector(root, slot, True)
                layout["sealed_payload"].write_text(
                    "not-json and forbidden to read",
                    encoding="utf-8",
                )
            paths = common.aggregate_paths(root)
            context = {
                "root": root,
                "slots": slots,
                "contract_sha256": "b" * 64,
                "roster_sha256": "c" * 64,
            }
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
            self.assertEqual(common.QUALIFICATION_TERMINAL, result["terminal"])
            self.assertFalse(result["sealed_payload_read"])
            self.assertTrue(paths["aggregate_attempt"].exists())
            self.assertEqual(
                result["aggregate_attempt_sha256"],
                common.sha256(paths["aggregate_attempt"]),
            )
            self.assertEqual(
                result,
                common.validate_selection(
                    paths["selection"],
                    slots,
                    "b" * 64,
                    "c" * 64,
                ),
            )
            with self.assertRaises(FileExistsError):
                aggregator.aggregate_screening(context)

    def test_orphan_aggregate_attempt_freezes_invalid_without_rescan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = common.aggregate_paths(root)
            context = {
                "root": root,
                "slots": _slots(),
                "contract_sha256": "b" * 64,
                "roster_sha256": "c" * 64,
            }
            _write(
                paths["aggregate_attempt"],
                {
                    "schema": common.AGGREGATE_ATTEMPT_SCHEMA,
                    "status": common.AGGREGATE_ATTEMPT_STATUS,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "execution_contract_sha256": "b" * 64,
                    "metadata_roster_sha256": "c" * 64,
                    "selector_or_failure_receipts_read_before_attempt": False,
                    "sealed_payload_read": False,
                    "rerun_authorized": False,
                },
            )
            with (
                mock.patch.object(
                    aggregator,
                    "scan_screening_state",
                    side_effect=AssertionError(
                        "orphan recovery must not reread receipts"
                    ),
                ),
                self.assertRaisesRegex(ValueError, "prior durable"),
            ):
                aggregator.aggregate_screening(context)
            invalid = common.load_json(paths["invalid"])
            self.assertEqual(
                invalid["terminal"],
                "D3_QUALIFICATION_INVALID_STOP",
            )
            self.assertFalse(invalid["rerun_authorized"])

    def test_validate_selection_rejects_nonexact_receipts(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for slot in slots[:6]:
                _write_selector(root, slot, True)
            context = {
                "root": root,
                "slots": slots,
                "contract_sha256": "b" * 64,
                "roster_sha256": "c" * 64,
            }
            selection = aggregator.aggregate_screening(context)
            reordered = json.loads(json.dumps(selection))
            reordered["selected_sources"][0], reordered["selected_sources"][1] = (
                reordered["selected_sources"][1],
                reordered["selected_sources"][0],
            )
            with self.assertRaisesRegex(ValueError, "frozen slot order"):
                common.validate_selection(
                    reordered,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )
            wrong_hash = json.loads(json.dumps(selection))
            wrong_hash["selected_sources"][0]["selector_sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                common.validate_selection(
                    wrong_hash,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )
            extra = json.loads(json.dumps(selection))
            extra["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "extra"):
                common.validate_selection(
                    extra,
                    slots,
                    "b" * 64,
                    "c" * 64,
                )

    def test_aggregate_budget_terminal(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, slot in enumerate(slots):
                _write_selector(root, slot, index < 5)
            context = {
                "root": root,
                "slots": slots,
                "contract_sha256": "b" * 64,
                "roster_sha256": "c" * 64,
            }
            result = aggregator.aggregate_screening(context)
        self.assertEqual(common.BUDGET_TERMINAL, result["terminal"])
        self.assertEqual(5, result["qualified_source_count"])

    def test_aggregate_invalid_prefix_writes_only_invalid_terminal(self) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_selector(root, slots[1], False)
            context = {
                "root": root,
                "slots": slots,
                "contract_sha256": "b" * 64,
                "roster_sha256": "c" * 64,
            }
            with self.assertRaisesRegex(ValueError, "prefix gap"):
                aggregator.aggregate_screening(context)
            paths = common.aggregate_paths(root)
            self.assertTrue(paths["invalid"].exists())
            self.assertFalse(paths["selection"].exists())
            self.assertFalse(paths["exhausted"].exists())

    def test_validate_execution_contract_rejects_incomplete_parent_chain(
        self,
    ) -> None:
        slots = _slots()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "protocol.json"
            roster_path = root / "roster.json"
            contract_path = root / "contract.json"
            protocol = {
                "schema": common.PROTOCOL_SCHEMA,
                "status": common.PROTOCOL_STATUS,
                "source_pool": {
                    "maximum_truth_screened_slots": 40,
                    "required_qualified_sources": 6,
                },
            }
            roster = {
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
            }
            _write(protocol_path, protocol)
            _write(roster_path, roster)
            implementation = Path(aggregator.__file__).resolve()
            contract = {
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
                "screening": {
                    "slot_count": 40,
                    "required_qualified_sources": 6,
                    "maximum_path_chars_exclusive": 240,
                },
                "canonical_artifacts": {
                    "screening_root": str(
                        root / common.CANONICAL_RELATIVE_ROOT
                    )
                },
                "authorization": {
                    "reference_support_screening_authorized": True,
                    "effect_before_six_source_selection_authorized": False,
                    "rgb_student_training_authorized": False,
                    "rgb_student_execution_authorized": False,
                    "reserved_official_test_open_authorized": False,
                    "research_mainline_changed": False,
                    "default_app_changed": False,
                    "android_changed": False,
                    "production_authorized": False,
                    "safety_claim_authorized": False,
                },
                "network": {"retries": 3},
            }
            _write(contract_path, contract)
            with (
                mock.patch.object(common, "repo_root", return_value=root),
                self.assertRaisesRegex(
                    ValueError,
                    "metadata_roster_result",
                ),
            ):
                common.validate_execution_contract(
                    contract_path,
                    aggregator.IMPLEMENTATION_KEY,
                    implementation,
                    verify_git=False,
                )


if __name__ == "__main__":
    unittest.main()
