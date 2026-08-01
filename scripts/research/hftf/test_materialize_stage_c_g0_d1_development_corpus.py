#!/usr/bin/env python3
"""Targeted tests for the G0-D1 development corpus materializer."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import materialize_stage_c_g0_d1_development_corpus as target


class NullableLabelsTest(unittest.TestCase):
    def test_transposes_to_height_theta_distance_and_nulls_unknown(self) -> None:
        known = np.zeros((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        known[1, 2, 0] = True
        known[4, 5, 1] = True
        support[4, 5, 1] = 2
        clearance[4, 5, 1] = -0.25

        labels = target._nullable_labels(known, support, clearance)

        self.assertEqual(np.asarray(labels["known_target"]).shape, (2, 6, 6))
        self.assertEqual(labels["known_target"][0][1][2], 1)
        self.assertEqual(labels["risk_target_nullable"][0][1][2], 0)
        self.assertEqual(
            labels["clearance_target_m_nullable"][0][1][2], 1.0
        )
        self.assertEqual(labels["known_target"][1][4][5], 1)
        self.assertEqual(labels["risk_target_nullable"][1][4][5], 1)
        self.assertEqual(
            labels["clearance_target_m_nullable"][1][4][5], -0.25
        )
        self.assertIsNone(labels["risk_target_nullable"][0][0][0])
        self.assertIsNone(
            labels["clearance_target_m_nullable"][0][0][0]
        )

    def test_rejects_nonfinite_known_clearance(self) -> None:
        known = np.ones((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        clearance[0, 0, 0] = np.inf

        with self.assertRaisesRegex(ValueError, "finite"):
            target._nullable_labels(known, support, clearance)

    def test_rejects_risk_clearance_semantic_mismatch(self) -> None:
        known = np.ones((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        support[0, 0, 0] = 2

        with self.assertRaisesRegex(ValueError, "semantics disagree"):
            target._nullable_labels(known, support, clearance)


class StudentFirewallTest(unittest.TestCase):
    @staticmethod
    def valid_record() -> dict[str, object]:
        known = np.zeros((6, 6, 2), dtype=np.bool_)
        support = np.zeros((6, 6, 2), dtype=np.int64)
        clearance = np.ones((6, 6, 2), dtype=np.float64)
        known[0, 0, 0] = True
        return {
            "sample_id": "sample-1",
            "session_id": "session-1",
            "role": "train",
            "source_frame_index": 0,
            "manifest_id": "manifest-1",
            "current_rgb": {
                "path": "C:/data/images/frame.png",
                "sha256": "a" * 64,
            },
            "labels": target._nullable_labels(
                known, support, clearance
            ),
        }

    def test_accepts_exact_requested_student_schema(self) -> None:
        self.assertTrue(
            target._student_record_firewall(self.valid_record())
        )

    def test_rejects_teacher_or_future_field(self) -> None:
        for forbidden_key in (
            "source_depth",
            "teacher_receipt",
            "future_rgb",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                record = self.valid_record()
                record[forbidden_key] = "forbidden"
                self.assertFalse(target._student_record_firewall(record))

    def test_rejects_nonnull_unknown_or_bad_known_equivalence(self) -> None:
        unknown = self.valid_record()
        unknown["labels"]["risk_target_nullable"][0][0][1] = 0
        self.assertFalse(target._student_record_firewall(unknown))

        mismatch = self.valid_record()
        mismatch["labels"]["risk_target_nullable"][0][0][0] = 1
        self.assertFalse(target._student_record_firewall(mismatch))


class SourceSelectionTest(unittest.TestCase):
    @staticmethod
    def plan() -> dict[str, object]:
        frames = list(range(0, 50, 2))
        development = []
        for index in range(9):
            prior_role = "train" if index < 6 else "dev"
            development.append(
                {
                    "session_id": f"development-{index}",
                    "role": prior_role,
                    "official_split": "train",
                    "source_plan_origin": "f0_metadata_plan_first_nine",
                    "g0_source_role": (
                        "development_reuse_outcome_open_train"
                        if index < 6
                        else
                        "development_reuse_outcome_open_model_selection"
                    ),
                    "fresh_evidence_credit": False,
                    "source_fps": 20.0,
                    "target_fps": 10.0,
                    "selected_source_frames": frames,
                }
            )
        fresh = [
            {
                "session_id": f"fresh-{index}",
                "g0_source_role": (
                    "one_shot_fresh_evaluation_metadata_planned_only"
                ),
                "media_geometry_teacher_or_student_outcome_open": False,
                "fresh_evidence_obtained": False,
            }
            for index in range(3)
        ]
        heldout = [
            {
                "session_id": f"heldout-{index}",
                "g0_source_role": (
                    "metadata_only_future_heldout_reservation"
                ),
                "media_geometry_teacher_or_student_outcome_open": False,
                "fresh_evidence_obtained": False,
            }
            for index in range(3)
        ]
        return {
            "schema": target.SOURCE_PLAN_SCHEMA,
            "terminal": target.SOURCE_PLAN_READY,
            "role_counts": copy.deepcopy(target.EXPECTED_ROLE_COUNTS),
            "roles": {
                "development_reuse": development,
                "one_shot_fresh_evaluation": fresh,
                "reserved_fresh_heldout": heldout,
            },
        }

    def test_maps_exact_six_train_three_model_selection(self) -> None:
        selected = target._select_development_sources(self.plan())

        self.assertEqual(len(selected), 9)
        self.assertEqual(
            [item["materialized_role"] for item in selected],
            ["train"] * 6 + ["model_selection"] * 3,
        )
        self.assertTrue(
            all(
                not str(item["session_id"]).startswith(
                    ("fresh-", "heldout-")
                )
                for item in selected
            )
        )

    def test_rejects_role_drift_overlap_and_open_fresh_media(self) -> None:
        drift = self.plan()
        drift["roles"]["development_reuse"][6]["role"] = "train"
        with self.assertRaises(ValueError):
            target._select_development_sources(drift)

        overlap = self.plan()
        overlap["roles"]["one_shot_fresh_evaluation"][0][
            "session_id"
        ] = "development-0"
        with self.assertRaises(ValueError):
            target._select_development_sources(overlap)

        opened = self.plan()
        opened["roles"]["one_shot_fresh_evaluation"][0][
            "media_geometry_teacher_or_student_outcome_open"
        ] = True
        with self.assertRaises(ValueError):
            target._select_development_sources(opened)

    def test_accepts_frozen_five_fps_source_and_rejects_frame_step_drift(
        self,
    ) -> None:
        mixed = self.plan()
        mixed["roles"]["development_reuse"][2].update(
            {
                "source_fps": 5.0,
                "target_fps": 5.0,
                "selected_source_frames": list(range(25)),
            }
        )
        self.assertEqual(
            len(target._select_development_sources(mixed)),
            9,
        )
        mixed["roles"]["development_reuse"][2][
            "selected_source_frames"
        ] = list(range(0, 50, 2))
        with self.assertRaisesRegex(ValueError, "six-train"):
            target._select_development_sources(mixed)


class SerializationAndAuthorizationTest(unittest.TestCase):
    @staticmethod
    def frozen_contract(
        design_path: Path,
        *,
        include_implementation: bool = True,
    ) -> dict[str, object]:
        amendment_path = design_path.with_name(
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
            "D1_TIMELINE_AMENDMENT_2026-08-01.json"
        )
        contract: dict[str, object] = {
            "schema": target.EXECUTION_CONTRACT_SCHEMA,
            "status": target.EXECUTION_CONTRACT_STATUS,
            "parents": {
                "d1_scientific_design": {
                    "path": design_path.name,
                    "sha256": target._sha256(design_path),
                },
                "d1_timeline_amendment": {
                    "path": amendment_path.name,
                    "sha256": target._sha256(amendment_path),
                }
            },
            "implementations": {},
        }
        if include_implementation:
            contract["implementations"] = {
                "development_corpus_materializer": {
                    "path": target.IMPLEMENTATION_PATH,
                    "sha256": target._sha256(
                        Path(target.__file__).resolve()
                    ),
                    "execution_authorized": True,
                }
            }
        return contract

    def test_canonical_jsonl_is_order_stable_and_strict(self) -> None:
        left = target._canonical_jsonl([{"b": 2, "a": 1}])
        right = target._canonical_jsonl([{"a": 1, "b": 2}])
        self.assertEqual(left, right)
        self.assertEqual(left, b'{"a":1,"b":2}\n')
        with self.assertRaises(ValueError):
            target._canonical_jsonl([{"bad": float("nan")}])

    def test_atomic_publish_writes_exact_files_and_refuses_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            target._atomic_publish(
                output,
                b'{"sample":1}\n',
                b'{"receipt":1}\n',
                {"schema": "test", "terminal": target.READY},
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "student_samples.jsonl",
                    "teacher_receipts.jsonl",
                    "dataset_spec.json",
                },
            )
            self.assertEqual(
                (output / "student_samples.jsonl").read_bytes(),
                b'{"sample":1}\n',
            )
            with self.assertRaises(FileExistsError):
                target._atomic_publish(output, b"", b"", {})

    def test_unfrozen_execution_contract_fails_before_source_open(
        self,
    ) -> None:
        design_path = (
            target._repository_root()
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_D1_"
            "2026-08-01.json"
        )
        contract = self.frozen_contract(
            design_path, include_implementation=False
        )
        with self.assertRaisesRegex(
            ValueError, "implementation receipt is not frozen"
        ):
            target._implementation_receipt(contract)

    def test_frozen_parent_graph_and_actual_source_plan_validate_in_memory(
        self,
    ) -> None:
        design_path = (
            target._repository_root()
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_D1_"
            "2026-08-01.json"
        )
        contract_path = design_path.with_name(
            "HFTF_STAGE_C_CURRENT_CLEARANCE_LEARNABILITY_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        )
        contract = self.frozen_contract(design_path)
        real_load_json = target._load_json

        def load_with_in_memory_receipt(path: Path) -> dict[str, object]:
            if path.resolve() == contract_path.resolve():
                return copy.deepcopy(contract)
            return real_load_json(path)

        with mock.patch.object(
            target, "_load_json", side_effect=load_with_in_memory_receipt
        ):
            context = target._load_context(contract_path)

        selected = context[-1]
        self.assertEqual(len(selected), 9)
        self.assertEqual(
            [item["materialized_role"] for item in selected],
            ["train"] * 6 + ["model_selection"] * 3,
        )


if __name__ == "__main__":
    unittest.main()
