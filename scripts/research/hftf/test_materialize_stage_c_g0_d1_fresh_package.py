#!/usr/bin/env python3
"""Tests for the G0-D1 one-shot fresh package materializer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import materialize_stage_c_g0_d1_fresh_package as target


def _label(
    *,
    known_cells: int = 36,
    positive_cells: int = 5,
) -> dict[str, object]:
    known = np.zeros((2, 6, 6), dtype=np.uint8)
    risk = np.full((2, 6, 6), None, dtype=object)
    clearance = np.full((2, 6, 6), None, dtype=object)
    for height in range(2):
        for flat in range(known_cells):
            index = (height, flat // 6, flat % 6)
            known[index] = 1
            is_positive = flat < positive_cells
            risk[index] = int(is_positive)
            clearance[index] = -0.1 if is_positive else 0.2
    return {
        "known_target": known.tolist(),
        "risk_target_nullable": risk.tolist(),
        "clearance_target_m_nullable": clearance.tolist(),
    }


def _truth_rows(
    *,
    known_cells: int = 36,
    positive_cells: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session_id in target.EXPECTED_SOURCE_IDS:
        for frame in range(25):
            rows.append(
                {
                    "session_id": session_id,
                    "labels": _label(
                        known_cells=known_cells,
                        positive_cells=positive_cells,
                    ),
                    "frame": frame,
                }
            )
    return rows


class FreshMaterializerTests(unittest.TestCase):
    def test_execution_receipt_consumes_materialization_before_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "contract.json"
            contract_path.write_text("{}\n", encoding="utf-8")
            contract = {
                "canonical_artifacts": {
                    "fresh_package_execution_root": (
                        "artifacts.local/evidence/hftf/"
                        "stage-c-g0-d1-fresh-package-execution-20260801"
                    )
                }
            }
            with mock.patch.object(
                target, "_repository_root", return_value=root
            ):
                execution_root = target._start_execution(
                    contract_path, contract
                )
                receipt = target._load_json(
                    execution_root / "execution_receipt.json"
                )
                self.assertEqual(
                    receipt["status"],
                    (
                        "STARTED_BEFORE_FIRST_FRESH_PACKAGE_"
                        "SOURCE_OR_MEDIA_READ"
                    ),
                )
                with self.assertRaisesRegex(
                    FileExistsError, "already consumed"
                ):
                    target._start_execution(contract_path, contract)

    def test_nullable_labels_enforces_unknown_null_and_semantic_equivalence(
        self,
    ) -> None:
        known = np.zeros((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        known[0, 0, 0] = True
        support[0, 0, 0] = 2
        clearance[0, 0, 0] = -0.125
        labels = target._nullable_labels(known, support, clearance)
        self.assertEqual(labels["known_target"][0][0][0], 1)
        self.assertEqual(labels["risk_target_nullable"][0][0][0], 1)
        self.assertEqual(
            labels["clearance_target_m_nullable"][0][0][0], -0.125
        )
        self.assertIsNone(labels["risk_target_nullable"][1][5][5])
        self.assertIsNone(
            labels["clearance_target_m_nullable"][1][5][5]
        )

    def test_nullable_labels_rejects_risk_clearance_disagreement(self) -> None:
        known = np.ones((6, 6, 2), dtype=np.bool_)
        support = np.full((6, 6, 2), 2, dtype=np.int64)
        clearance = np.full((6, 6, 2), 0.1, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "semantics disagree"):
            target._nullable_labels(known, support, clearance)

    def test_opportunity_gate_is_per_source_height(self) -> None:
        report, adequate = target._opportunity(
            _truth_rows(), target.EXPECTED_SOURCE_IDS
        )
        self.assertTrue(adequate)
        self.assertEqual(list(report), list(target.EXPECTED_SOURCE_IDS))
        for source in report.values():
            for height in target.HEIGHTS:
                self.assertEqual(source[height]["frame_count"], 25)
                self.assertEqual(source[height]["positive_known"], 125)
                self.assertEqual(source[height]["negative_known"], 775)
                self.assertTrue(source[height]["gate_pass"])

    def test_opportunity_failure_is_not_rescued_or_replaced(self) -> None:
        report, adequate = target._opportunity(
            _truth_rows(known_cells=3, positive_cells=1),
            target.EXPECTED_SOURCE_IDS,
        )
        self.assertFalse(adequate)
        self.assertEqual(set(report), set(target.EXPECTED_SOURCE_IDS))
        self.assertTrue(
            all(
                not metrics[height]["gate_pass"]
                for metrics in report.values()
                for height in target.HEIGHTS
            )
        )

    def test_contract_status_fails_before_parent_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "contract.json"
            with mock.patch.object(
                target,
                "_load_json",
                return_value={
                    "schema": target.CONTRACT_SCHEMA,
                    "status": "NOT_FROZEN",
                },
            ), mock.patch.object(
                target, "_bound_parent"
            ) as bound:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    target._load_context(contract_path)
                bound.assert_not_called()

    def test_select_sources_requires_exact_fixed_order_and_bindings(
        self,
    ) -> None:
        planned = []
        bindings = []
        for session_id in target.EXPECTED_SOURCE_IDS:
            planned.append(
                {
                    "session_id": session_id,
                    "role": "fresh_evaluation",
                    "official_split": "train",
                    "g0_source_role": (
                        "one_shot_fresh_evaluation_metadata_planned_only"
                    ),
                    "media_geometry_teacher_or_student_outcome_open": False,
                    "fresh_evidence_obtained": False,
                    "selected_source_frames": list(range(0, 50, 2)),
                    "source_fps": 20.0,
                    "target_fps": 10.0,
                }
            )
            bindings.append(
                {"session_id": session_id}
            )
        source_plan = {
            "schema": target.SOURCE_PLAN_SCHEMA,
            "terminal": target.SOURCE_PLAN_TERMINAL,
            "roles": {"one_shot_fresh_evaluation": planned},
        }
        contract = {
            "implementations": {
                "fresh_source_authority_verifier": {
                    "sha256": "f" * 64
                }
            },
            "fresh_source_contract": {
                "source_order": list(target.EXPECTED_SOURCE_IDS),
                "sources": bindings,
            }
        }
        selected = target._select_sources(source_plan, contract)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            selected[0]["source_binding"],
            {"session_id": target.EXPECTED_SOURCE_IDS[0]},
        )
        contract["fresh_source_contract"]["source_order"].reverse()
        with self.assertRaisesRegex(ValueError, "Exact frozen"):
            target._select_sources(source_plan, contract)


if __name__ == "__main__":
    unittest.main()
