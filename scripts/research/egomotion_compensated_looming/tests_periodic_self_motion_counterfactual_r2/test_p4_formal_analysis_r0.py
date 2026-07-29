from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p4_formal_analysis_r0 as subject,
)


class FormalFixture:
    def __init__(self) -> None:
        self.identity_sha = "1" * 64
        self.activation_sha = "2" * 64
        self.authority = {
            "manipulation_receipt_sha256": "7" * 64,
            "manipulation_producer_receipt_sha256": "a" * 64,
            "trajectory_manifest_sha256": "8" * 64,
            "runner_sha256": "9" * 64,
        }
        identities = []
        for block in subject.BLOCKS:
            for ordinal in range(20):
                cluster = f"{block}__MAIN_{ordinal:02d}"
                for arm_ordinal, arm in enumerate(subject.MAIN_ARMS):
                    identities.append(
                        self._identity(
                            block,
                            ordinal,
                            "MAIN_FACTORIAL",
                            arm,
                            arm_ordinal,
                            cluster,
                            f"FORMAL_{block}_MAIN_{ordinal:02d}__{arm}",
                        )
                    )
            for ordinal in range(2):
                cluster = f"{block}__GUARD_{ordinal:02d}"
                for arm_ordinal, arm in enumerate(subject.GUARD_ARMS):
                    identities.append(
                        self._identity(
                            block,
                            ordinal,
                            "POSITIVE_GUARDRAIL",
                            arm,
                            arm_ordinal,
                            cluster,
                            f"FORMAL_{block}_GUARD_{ordinal:02d}__{arm}",
                        )
                    )
        self.identity_lock = {
            "schema": subject.IDENTITY_SCHEMA,
            "protocol_id": subject.PROTOCOL_ID,
            "formal_execution_authorized": False,
            "counts": {
                "main_clusters": 80,
                "guardrail_clusters": 8,
                "main_sequences": 480,
                "guardrail_sequences": 16,
                "total_sequences": 496,
                "frames": 298592,
                "pairs": 298096,
            },
            "identities": identities,
            "identity_set_sha256": hashlib.sha256(
                subject.canonical_bytes(identities)
            ).hexdigest(),
        }
        self.identities = {
            item["sequence_id"]: item for item in identities
        }
        self.receipts = {}
        self.receipt_hashes = {}
        self.ledger_hashes = {}
        entries = []
        for sequence, identity in self.identities.items():
            receipt_path = f"artifacts.local/formal/arms/{sequence}/receipt.json"
            ledger_path = (
                f"artifacts.local/formal/arms/{sequence}/pair_ledger.jsonl"
            )
            frame_path = (
                f"artifacts.local/formal/arms/{sequence}/frame_manifest.jsonl"
            )
            reduced_path = (
                f"artifacts.local/formal/arms/{sequence}/reduced_metrics.json"
            )
            ledger_sha = hashlib.sha256(
                f"ledger:{sequence}".encode()
            ).hexdigest()
            frame_sha = hashlib.sha256(
                f"frames:{sequence}".encode()
            ).hexdigest()
            reduced_sha = hashlib.sha256(
                f"reduced:{sequence}".encode()
            ).hexdigest()
            receipt = {
                "schema": subject.ARM_SCHEMA,
                "protocol_id": subject.PROTOCOL_ID,
                "run_id": "P4_FIXTURE",
                "activation_sha256": self.activation_sha,
                "identity_manifest_sha256": self.identity_sha,
                **self.authority,
                **{
                    key: identity[key]
                    for key in (
                        "sequence_id",
                        "cluster_id",
                        "block",
                        "ordinal",
                        "role",
                        "arm",
                        "numeric_seed_uint64",
                        "frame_count",
                        "pair_count",
                    )
                },
                "pair_ledger_path": ledger_path,
                "pair_ledger_sha256": ledger_sha,
                "frame_manifest_path": frame_path,
                "frame_manifest_sha256": frame_sha,
                "reduced_metrics_path": reduced_path,
                "reduced_metrics_sha256": reduced_sha,
                "terminal": "ARM_COMPLETE",
            }
            receipt_sha = hashlib.sha256(
                subject.canonical_bytes(receipt)
            ).hexdigest()
            self.receipts[receipt_path] = receipt
            self.receipt_hashes[receipt_path] = receipt_sha
            trigger_count = (
                599 if identity["role"] == "POSITIVE_GUARDRAIL" else 0
            )
            self.receipts[reduced_path] = {
                "scheduled_pair_count": 601,
                "evaluable_pair_count": 601,
                "trigger_count": trigger_count,
                "trigger_density": trigger_count / 601,
                "quality_failure_union_count": 0,
                "quality_failure_union_density": 0.0,
                "clean_trackable_pair_count": 601,
                "clean_sequence_trackable": True,
            }
            self.receipt_hashes[reduced_path] = reduced_sha
            self.ledger_hashes[ledger_path] = ledger_sha
            self.ledger_hashes[frame_path] = frame_sha
            entries.append(
                {
                    "sequence_id": sequence,
                    "receipt_path": receipt_path,
                    "receipt_sha256": receipt_sha,
                }
            )
        self.bundle = {
            "schema": subject.BUNDLE_SCHEMA,
            "protocol_id": subject.PROTOCOL_ID,
            "run_id": "P4_FIXTURE",
            "activation_sha256": self.activation_sha,
            "identity_manifest_sha256": self.identity_sha,
            **self.authority,
            "prerequisite_gates": {
                "geometry_validation": "PASS",
                "quality_strength_lock": "PASS",
                "formal_main_manipulation": "PASS",
            },
            "arms": entries,
            "arm_count": 496,
            "frame_count": 298592,
            "pair_count": 298096,
            "residual_worker_pids": [],
            "rgb_frames_retained": False,
            "sequence16_android_realtime": False,
            "scientific_outcome_interpreted": False,
            "terminal": "BUNDLE_COMPLETE",
        }
        self.ledger_overrides = {}

    @staticmethod
    def _identity(
        block: str,
        ordinal: int,
        role: str,
        arm: str,
        arm_ordinal: int,
        cluster_id: str,
        sequence_id: str,
    ) -> dict:
        source_arm = {
            "MONOTONIC_APPROACH_ONLY__CLEAN": "MONOTONIC_APPROACH",
            "MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF__CLEAN": (
                "MONOTONIC_APPROACH_PLUS_PERIODIC"
            ),
        }.get(arm, arm)
        return {
            "sequence_id": sequence_id,
            "cluster_id": cluster_id,
            "block": block,
            "ordinal": ordinal,
            "role": role,
            "source_arm_id": source_arm,
            "arm": arm,
            "arm_ordinal": arm_ordinal,
            "numeric_seed_uint64": subject._exact_seed(
                role, block, ordinal
            ),
            "scene_geometry_sha256": "3" * 64,
            "trajectory_sha256": "4" * 64,
            "frame_count": 602,
            "pair_count": 601,
        }

    def ledger(self, path: str):
        if path in self.ledger_overrides:
            return self.ledger_overrides[path]
        sequence = path.split("/")[-2]
        if path.endswith("frame_manifest.jsonl"):
            return [
                {
                    "frame_index": index,
                    "rgb_sha256": "b" * 64,
                    "valid_mask_sha256": "c" * 64,
                }
                for index in range(602)
            ]
        identity = self.identities[sequence]
        response = (
            0.02
            if identity["role"] == "POSITIVE_GUARDRAIL"
            else 0.0
        )
        return [
            {
                **{
                    key: identity[key]
                    for key in (
                        "sequence_id",
                        "cluster_id",
                        "block",
                        "ordinal",
                        "role",
                        "arm",
                    )
                },
                "pair_index": index,
                "evaluable": True,
                "compensated_expansion_median_per_s": response,
                "compensated_three_pair_trigger": (
                    response > 0.01 and index >= 2
                ),
                "detected_feature_count": 100,
                "forward_backward_consistent_count": 90,
                "forward_backward_consistent_fraction": 0.9,
                "median_forward_backward_error_px": 0.2,
                "occupied_3x3_cells": 8,
            }
            for index in range(601)
        ]

    def assemble(self, bundle=None):
        return subject.assemble_loaded_bundle(
            self.bundle if bundle is None else bundle,
            self.identity_lock,
            lambda path: self.receipts[path],
            lambda path: self.receipt_hashes[path],
            self.ledger,
            lambda path: self.ledger_hashes[path],
            bundle_sha256="5" * 64,
            identity_lock_sha256=self.identity_sha,
        )


class FormalAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FormalFixture()
        cls.result = cls.fixture.assemble()

    def test_complete_496_bundle_closes_and_analyzes(self) -> None:
        self.assertEqual("VALID_COMPLETE", self.result["execution_state"])
        self.assertEqual(496, self.result["closure"]["atomic_receipt_count"])
        self.assertEqual(
            "NO_SEPARATION_HOLD", self.result["scientific_terminal"]
        )
        self.assertTrue(self.result["clean_tracking_gate"]["passed"])
        self.assertTrue(self.result["positive_guardrail_gate"]["passed"])
        self.assertEqual(
            20_000, self.result["analysis"]["bootstrap"]["replicates"]
        )

    def test_missing_arm_is_execution_incomplete(self) -> None:
        bundle = copy.deepcopy(self.fixture.bundle)
        bundle["arms"].pop()
        with self.assertRaisesRegex(
            subject.ClosureError, "EXECUTION_INCOMPLETE:BUNDLE_ARM_COUNT"
        ):
            self.fixture.assemble(bundle)

    def test_duplicate_arm_is_invalid(self) -> None:
        bundle = copy.deepcopy(self.fixture.bundle)
        bundle["arms"][-1] = copy.deepcopy(bundle["arms"][0])
        with self.assertRaisesRegex(
            subject.ClosureError, "INVALID:BUNDLE_SEQUENCE_KEYSET"
        ):
            self.fixture.assemble(bundle)

    def test_identity_seed_mutation_is_invalid(self) -> None:
        identity_lock = copy.deepcopy(self.fixture.identity_lock)
        identity_lock["identities"][0]["numeric_seed_uint64"] ^= 1
        identity_lock["identity_set_sha256"] = hashlib.sha256(
            subject.canonical_bytes(identity_lock["identities"])
        ).hexdigest()
        with self.assertRaisesRegex(
            subject.ClosureError, "INVALID:IDENTITY_NUMERIC_BINDING"
        ):
            subject.validate_identity_lock(identity_lock)

    def test_receipt_hash_mutation_is_invalid(self) -> None:
        with self.assertRaisesRegex(
            subject.ClosureError, "INVALID:ATOMIC_RECEIPT_HASH"
        ):
            subject.assemble_loaded_bundle(
                self.fixture.bundle,
                self.fixture.identity_lock,
                lambda path: self.fixture.receipts[path],
                lambda path: "0" * 64,
                self.fixture.ledger,
                lambda path: self.fixture.ledger_hashes[path],
                bundle_sha256="5" * 64,
                identity_lock_sha256=self.fixture.identity_sha,
            )

    def test_missing_receipt_is_execution_incomplete(self) -> None:
        with self.assertRaisesRegex(
            subject.ClosureError,
            "EXECUTION_INCOMPLETE:ATOMIC_RECEIPT_MISSING",
        ):
            subject.assemble_loaded_bundle(
                self.fixture.bundle,
                self.fixture.identity_lock,
                lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
                lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
                self.fixture.ledger,
                lambda path: self.fixture.ledger_hashes[path],
                bundle_sha256="5" * 64,
                identity_lock_sha256=self.fixture.identity_sha,
            )

    def test_pair_trigger_abstention_and_order_mutations_fail(self) -> None:
        identity = next(iter(self.fixture.identities.values()))
        rows = self.fixture.ledger(
            f"artifacts.local/formal/arms/{identity['sequence_id']}/pair_ledger.jsonl"
        )
        forged = copy.deepcopy(rows)
        forged[2]["compensated_three_pair_trigger"] = True
        with self.assertRaisesRegex(
            subject.ClosureError, "FORGED_THREE_PAIR_TRIGGER"
        ):
            subject.reduce_pair_ledger(forged, identity)
        abstention = copy.deepcopy(rows)
        abstention[0]["evaluable"] = False
        with self.assertRaisesRegex(
            subject.ClosureError, "ABSTENTION_RESPONSE_PRESENT"
        ):
            subject.reduce_pair_ledger(abstention, identity)
        reordered = copy.deepcopy(rows)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(
            subject.ClosureError, "PAIR_ORDER_OR_IDENTITY"
        ):
            subject.reduce_pair_ledger(reordered, identity)

    def test_strict_threshold_resets_streak(self) -> None:
        identity = next(iter(self.fixture.identities.values()))
        rows = self.fixture.ledger(
            f"artifacts.local/formal/arms/{identity['sequence_id']}/pair_ledger.jsonl"
        )
        for index, value in enumerate((0.02, 0.02, 0.01, 0.02, 0.02)):
            rows[index]["compensated_expansion_median_per_s"] = value
            rows[index].pop("compensated_three_pair_trigger")
        reduced = subject.reduce_pair_ledger(rows, identity)
        self.assertEqual(0, reduced["trigger_count"])

    def test_prerequisite_gate_cannot_be_bypassed(self) -> None:
        bundle = copy.deepcopy(self.fixture.bundle)
        bundle["prerequisite_gates"]["formal_main_manipulation"] = "FAIL"
        with self.assertRaisesRegex(
            subject.ClosureError, "INVALID:PREREQUISITE_GATES"
        ):
            self.fixture.assemble(bundle)

    def test_gate_failure_has_terminal_precedence(self) -> None:
        terminal, subtype = subject._classify_terminal(
            self.result["analysis"], clean_gate_pass=True, guard_gate_pass=False
        )
        self.assertEqual("INTERVENTION_NOT_EVALUABLE", terminal)
        self.assertIsNone(subtype)

    def test_result_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            subject._write_exclusive(path, {"first": True})
            with self.assertRaises(FileExistsError):
                subject._write_exclusive(path, {"second": True})


if __name__ == "__main__":
    unittest.main()
