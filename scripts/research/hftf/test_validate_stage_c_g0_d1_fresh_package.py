#!/usr/bin/env python3
"""Tests for independent G0-D1 fresh package validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_stage_c_g0_d1_fresh_package as target


def _valid_label(
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
            positive = flat < positive_cells
            risk[index] = int(positive)
            clearance[index] = -0.2 if positive else 0.3
    return {
        "known_target": known.tolist(),
        "risk_target_nullable": risk.tolist(),
        "clearance_target_m_nullable": clearance.tolist(),
    }


class FreshValidatorTests(unittest.TestCase):
    def test_strict_labels_accepts_exact_nullable_contract(self) -> None:
        known, risk, clearance = target._strict_labels(_valid_label())
        self.assertEqual(known.shape, (2, 6, 6))
        self.assertEqual(int(known.sum()), 72)
        self.assertEqual(int(risk.sum()), 10)
        self.assertTrue(np.isfinite(clearance[known]).all())

    def test_strict_labels_rejects_unknown_to_safe_coercion(self) -> None:
        labels = _valid_label(known_cells=1, positive_cells=1)
        labels["risk_target_nullable"][0][0][1] = 0
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            target._strict_labels(labels)

    def test_strict_labels_rejects_clearance_range_or_sign_drift(self) -> None:
        labels = _valid_label()
        labels["clearance_target_m_nullable"][0][0][0] = 0.1
        with self.assertRaisesRegex(ValueError, "risk/clearance"):
            target._strict_labels(labels)
        labels = _valid_label()
        labels["clearance_target_m_nullable"][0][0][0] = -0.6
        with self.assertRaisesRegex(ValueError, "risk/clearance"):
            target._strict_labels(labels)

    def test_independent_label_json_matches_strict_schema(self) -> None:
        known = np.zeros((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        known[0, 0, 0] = True
        support[0, 0, 0] = 2
        clearance[0, 0, 0] = -0.2
        labels = target._label_json(known, support, clearance)
        parsed_known, parsed_risk, parsed_clearance = (
            target._strict_labels(labels)
        )
        self.assertTrue(parsed_known[0, 0, 0])
        self.assertEqual(parsed_risk[0, 0, 0], 1)
        self.assertEqual(parsed_clearance[0, 0, 0], -0.2)
        self.assertIsNone(labels["risk_target_nullable"][1][5][5])

    def test_validate_rows_enforces_input_firewall_and_exact_rederivation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            image_path.write_bytes(b"current-rgb")
            image_digest = target._sha256(image_path)
            label = _valid_label()
            frozen = {
                "sample_id": "sample-1",
                "session_id": target.EXPECTED_SOURCE_IDS[0],
                "source_frame_index": 0,
                "manifest_id": "manifest-1",
                "image_path": str(image_path.resolve()),
                "image_sha256": image_digest,
                "source_depth_sha256": "a" * 64,
                "source_mask_sha256": "b" * 64,
                "camera_poses_sha256": "c" * 64,
                "authority_report_sha256": "d" * 64,
                "labels": label,
            }
            input_row = {
                "schema": target.INPUT_SCHEMA,
                "sample_id": "sample-1",
                "session_id": target.EXPECTED_SOURCE_IDS[0],
                "source_frame_index": 0,
                "manifest_id": "manifest-1",
                "current_rgb": {
                    "path": str(image_path.resolve()),
                    "sha256": image_digest,
                },
            }
            truth = {
                "schema": target.TRUTH_SCHEMA,
                "sample_id": "sample-1",
                "session_id": target.EXPECTED_SOURCE_IDS[0],
                "source_frame_index": 0,
                "manifest_id": "manifest-1",
                "labels": label,
            }
            receipt = {
                "schema": target.RECEIPT_SCHEMA,
                "sample_id": "sample-1",
                "session_id": target.EXPECTED_SOURCE_IDS[0],
                "source_frame_index": 0,
                "manifest_id": "manifest-1",
                "teacher_view": "REFERENCE_STRIDE4_OFFSET2_CURRENT",
                "source_depth_sha256": "a" * 64,
                "source_mask_sha256": "b" * 64,
                "camera_poses_sha256": "c" * 64,
                "authority_report_sha256": "d" * 64,
                "labels_sha256": target._sha256_bytes(
                    target._canonical_bytes(label)
                ),
                "student_loader_authorized": False,
            }
            with mock.patch.object(
                target, "EXPECTED_SOURCE_IDS", (target.EXPECTED_SOURCE_IDS[0],)
            ):
                with self.assertRaisesRegex(ValueError, "record count"):
                    target._validate_rows(
                        [input_row], [truth], [receipt], [frozen]
                    )
            rows = [dict(input_row) for _ in range(75)]
            truths = [dict(truth) for _ in range(75)]
            receipts = [dict(receipt) for _ in range(75)]
            expected = [dict(frozen) for _ in range(75)]
            target._validate_rows(rows, truths, receipts, expected)
            rows[0] = {**rows[0], "labels": label}
            with self.assertRaisesRegex(ValueError, "firewall"):
                target._validate_rows(rows, truths, receipts, expected)

    def test_opportunity_not_evaluable_has_no_source_replacement(self) -> None:
        truths: list[dict[str, object]] = []
        for session_id in target.EXPECTED_SOURCE_IDS:
            for frame in range(25):
                truths.append(
                    {
                        "session_id": session_id,
                        "frame": frame,
                        "labels": _valid_label(
                            known_cells=2, positive_cells=1
                        ),
                    }
                )
        report, adequate = target._opportunity(truths)
        self.assertFalse(adequate)
        self.assertEqual(list(report), list(target.EXPECTED_SOURCE_IDS))
        self.assertEqual(
            target.NOT_EVALUABLE,
            "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT",
        )

    def test_flat_bindings_match_predictor_and_evaluator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prediction_inputs.jsonl").write_bytes(b"inputs\n")
            (root / "truth_labels.jsonl").write_bytes(b"truth\n")
            opportunity = {
                session_id: {
                    height: {"unknown_to_safe_violations": 0}
                    for height in ("body", "head")
                }
                for session_id in target.EXPECTED_SOURCE_IDS
            }
            fields = target._flat_validation_bindings(root, opportunity)
            self.assertEqual(fields["prediction_input_count"], 75)
            self.assertEqual(fields["truth_label_count"], 75)
            self.assertEqual(
                fields["unknown_to_safe_violation_count"], 0
            )
            self.assertEqual(
                set(fields["source_frame_indices"]),
                set(target.EXPECTED_SOURCE_IDS),
            )
            self.assertTrue(
                all(
                    frames == list(range(0, 50, 2))
                    for frames in fields["source_frame_indices"].values()
                )
            )
            self.assertEqual(
                fields["prediction_inputs_sha256"],
                target._sha256(root / "prediction_inputs.jsonl"),
            )
            self.assertEqual(
                fields["truth_labels_sha256"],
                target._sha256(root / "truth_labels.jsonl"),
            )

    def test_prediction_authorization_is_truth_payload_free(self) -> None:
        report = {
            "terminal": target.READY,
            "contract_sha256": "a" * 64,
            "package_validator_sha256": "b" * 64,
            "prediction_inputs_path": "C:/frozen/prediction_inputs.jsonl",
            "prediction_inputs_sha256": "c" * 64,
            "prediction_input_count": 75,
            "source_order": list(target.EXPECTED_SOURCE_IDS),
            "source_frame_indices": {
                session_id: list(range(0, 50, 2))
                for session_id in target.EXPECTED_SOURCE_IDS
            },
            "truth_labels_path": "C:/quarantined/truth_labels.jsonl",
            "truth_labels_sha256": "d" * 64,
            "teacher_receipts_sha256": "e" * 64,
            "opportunity_gate": {"secret": 1},
        }
        authorization = target._prediction_authorization(report)
        self.assertEqual(
            authorization["schema"],
            target.PREDICTION_AUTHORIZATION_SCHEMA,
        )
        self.assertEqual(
            authorization["terminal"],
            target.PREDICTION_AUTHORIZATION_READY,
        )
        self.assertTrue(
            authorization["authorization"][
                "fresh_prediction_authorized"
            ]
        )
        self.assertFalse(
            authorization["authorization"][
                "truth_join_authorized_before_predictions_frozen"
            ]
        )
        serialized = target._canonical_bytes(authorization).decode()
        for forbidden in (
            "truth_labels",
            "teacher_receipts",
            "opportunity",
            "unknown_to_safe",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_inadequate_package_never_authorizes_prediction(self) -> None:
        report = {
            "terminal": target.NOT_EVALUABLE,
            "contract_sha256": "a" * 64,
            "package_validator_sha256": "b" * 64,
            "prediction_inputs_path": "C:/frozen/prediction_inputs.jsonl",
            "prediction_inputs_sha256": "c" * 64,
            "prediction_input_count": 75,
            "source_order": list(target.EXPECTED_SOURCE_IDS),
            "source_frame_indices": {
                session_id: list(range(0, 50, 2))
                for session_id in target.EXPECTED_SOURCE_IDS
            },
        }
        authorization = target._prediction_authorization(report)
        self.assertEqual(
            authorization["terminal"],
            target.PREDICTION_AUTHORIZATION_NOT_EVALUABLE,
        )
        self.assertFalse(
            authorization["authorization"][
                "fresh_prediction_authorized"
            ]
        )

    def test_wrong_contract_status_fails_before_any_parent_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract_path = Path(directory) / "contract.json"
            with mock.patch.object(
                target,
                "_load_json",
                return_value={
                    "schema": target.CONTRACT_SCHEMA,
                    "status": "DRAFT",
                },
            ), mock.patch.object(
                target, "_bound_parent"
            ) as bound:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    target._validate_contract(contract_path)
                bound.assert_not_called()


if __name__ == "__main__":
    unittest.main()
