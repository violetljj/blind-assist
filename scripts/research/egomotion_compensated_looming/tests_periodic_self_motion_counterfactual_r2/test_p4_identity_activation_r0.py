from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ..periodic_self_motion_counterfactual_r2 import (
    p4_identity_activation_r0 as p4,
)


class P4IdentityActivationR0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[4]
        cls.temporary = tempfile.TemporaryDirectory(dir=cls.root / "tmp")
        cls.work = Path(cls.temporary.name)
        cls.scheduler_path = cls.work / "scheduler_amendment.json"
        cls.identity_path = cls.work / "formal_identity_lock.json"
        scheduler = p4.build_scheduler_amendment(cls.root)
        p4.write_exclusive(cls.scheduler_path, scheduler)
        identity = p4.build_formal_identity_lock(
            cls.root,
            cls.scheduler_path,
        )
        p4.write_exclusive(cls.identity_path, identity)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_scheduler_amendment_is_user_scoped_and_science_invariant(self) -> None:
        value = p4.load_json(self.scheduler_path)
        p4.validate_scheduler_amendment(self.root, value)
        self.assertEqual(value["scheduler"]["workers"], 8)
        self.assertEqual(value["scheduler"]["openblas_threads_per_worker"], 18)
        self.assertEqual(value["scheduler"]["opencv_threads_per_worker"], 1)
        self.assertEqual(
            value["user_authorization"]["authorization_id"],
            p4.DEFAULT_USER_AUTHORIZATION_ID,
        )
        self.assertFalse(
            value["scientific_lock"]["algorithm_or_threshold_changed"]
        )
        self.assertFalse(value["formal_execution_authorized"])
        mutated = copy.deepcopy(value)
        mutated["scientific_lock"]["threshold_per_s"] = 0.02
        with self.assertRaisesRegex(
            p4.InvalidP4IdentityActivation,
            "SCHEDULER_AMENDMENT_DRIFT",
        ):
            p4.validate_scheduler_amendment(self.root, mutated)

    def test_formal_identity_lock_is_exact_480_plus_16(self) -> None:
        value = p4.load_json(self.identity_path)
        p4.validate_formal_identity_lock(
            self.root,
            self.scheduler_path,
            value,
        )
        counts = value["counts"]
        self.assertEqual(counts["main_sequences"], 480)
        self.assertEqual(counts["guardrail_sequences"], 16)
        self.assertEqual(counts["total_sequences"], 496)
        self.assertEqual(counts["total_frames"], 496 * 602)
        self.assertEqual(counts["total_pairs"], 496 * 601)
        identities = value["identities"]
        self.assertEqual(len(identities), 496)
        self.assertEqual(len({item["sequence_id"] for item in identities}), 496)
        self.assertEqual(
            value["identity_set_sha256"],
            hashlib.sha256(p4.canonical_bytes(identities)).hexdigest(),
        )
        self.assertEqual(
            value["canonicalization"],
            "UTF8_JSON_SORT_KEYS_COMPACT_ENSURE_ASCII_FALSE_ALLOW_NAN_FALSE",
        )

    def test_guard_short_names_are_explicitly_mapped_to_canonical(self) -> None:
        identities = p4.load_json(self.identity_path)["identities"]
        guards = [
            item for item in identities if item["role"] == "POSITIVE_GUARDRAIL"
        ]
        self.assertEqual(len(guards), 16)
        observed = {
            item["source_arm_id"]: item["arm"]
            for item in guards
        }
        self.assertEqual(observed, p4.P1_GUARD_TO_CANONICAL)
        self.assertTrue(
            all(item["arm"].endswith("__CLEAN") for item in guards)
        )
        self.assertEqual(
            guards[0]["sequence_id"],
            (
                "FORMAL_ADVIO_13_GUARD_00__"
                "MONOTONIC_APPROACH_ONLY__CLEAN"
            ),
        )

    def test_seed_identity_mutations_fail_closed(self) -> None:
        value = p4.load_json(self.identity_path)
        mutated = copy.deepcopy(value)
        mutated["identities"][0]["numeric_seed_uint64"] += 1
        mutated["identity_set_sha256"] = hashlib.sha256(
            p4.canonical_bytes(mutated["identities"])
        ).hexdigest()
        with self.assertRaisesRegex(
            p4.InvalidP4IdentityActivation,
            "FORMAL_IDENTITY_LOCK_DRIFT",
        ):
            p4.validate_formal_identity_lock(
                self.root,
                self.scheduler_path,
                mutated,
            )
        self.assertNotEqual(
            p4.derive_seed("MAIN", "ADVIO_13", 0),
            p4.derive_seed("GUARD", "ADVIO_13", 0),
        )

    def test_activation_builder_binds_phases_implementation_and_tests(self) -> None:
        with mock.patch.object(p4, "live_research_worker_pids", return_value=[]):
            value = p4.build_activation_lock(
                self.root,
                self.scheduler_path,
                self.identity_path,
                issued_at_utc="2026-07-29T00:00:00+00:00",
            )
            p4.validate_activation_lock(self.root, value)
        self.assertTrue(value["formal_execution_authorized"])
        self.assertTrue(value["p4_activated"])
        self.assertTrue(value["execution"]["one_shot"])
        self.assertEqual(value["execution"]["formal_sequence_count"], 496)
        by_phase = {item["phase"]: item for item in value["bindings"]}
        self.assertIn("P0_CONTRACT", by_phase)
        self.assertIn("P1_ALL_SEED_MANIFEST", by_phase)
        self.assertIn("P2_INDEPENDENT_RECEIPT", by_phase)
        self.assertIn("P3_SCHEDULER_SUCCESSOR_RECEIPT", by_phase)
        self.assertIn("P4_IDENTITY_ACTIVATION_IMPLEMENTATION", by_phase)
        self.assertIn("P4_IDENTITY_ACTIVATION_TESTS", by_phase)
        self.assertIn("P4_MANIPULATION_IMPLEMENTATION", by_phase)
        self.assertIn("P4_MANIPULATION_INDEPENDENT_VALIDATOR", by_phase)
        self.assertIn("P4_FORMAL_RUNNER", by_phase)
        self.assertIn("P4_FORMAL_ANALYSIS", by_phase)
        self.assertIn("P4_FORMAL_INDEPENDENT_VALIDATOR", by_phase)
        self.assertFalse(
            value["authority_ceiling"]["scientific_lock_change_authorized"]
        )

    def test_activation_binding_and_precondition_mutations_fail(self) -> None:
        with mock.patch.object(p4, "live_research_worker_pids", return_value=[]):
            value = p4.build_activation_lock(
                self.root,
                self.scheduler_path,
                self.identity_path,
                issued_at_utc="2026-07-29T00:00:00+00:00",
            )
        mutated = copy.deepcopy(value)
        mutated["bindings"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            p4.InvalidP4IdentityActivation,
            "ACTIVATION_BINDING_DRIFT",
        ):
            p4.validate_activation_lock(self.root, mutated)
        with mock.patch.object(
            p4,
            "live_research_worker_pids",
            return_value=[12345],
        ):
            with self.assertRaisesRegex(
                p4.InvalidP4IdentityActivation,
                "RESIDUAL_WORKERS",
            ):
                p4.build_activation_lock(
                    self.root,
                    self.scheduler_path,
                    self.identity_path,
                )

    def test_activation_is_not_issued_and_exclusive_write_is_fail_closed(self) -> None:
        activation_path = self.work / "activation.json"
        self.assertFalse(activation_path.exists())
        fixture = {
            "schema": "test",
            "terminal": "NOT_AN_ACTIVATION",
            "formal_execution_authorized": False,
        }
        p4.write_exclusive(activation_path, fixture)
        with self.assertRaises(FileExistsError):
            p4.write_exclusive(activation_path, fixture)
        self.assertEqual(
            json.loads(activation_path.read_text(encoding="utf-8")),
            fixture,
        )


if __name__ == "__main__":
    unittest.main()
