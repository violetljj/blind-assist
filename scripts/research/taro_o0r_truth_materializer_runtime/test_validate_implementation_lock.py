#!/usr/bin/env python3
"""Mutation tests for the TARO O0R truth-materializer implementation lock."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o0r_truth_materializer_runtime.materializer import load_json
from scripts.research.taro_o0r_truth_materializer_runtime.validate_implementation_lock import (
    REPO_ROOT,
    validate_lock,
)


LOCK_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_2026-08-10.json"


class ImplementationLockValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = load_json(LOCK_PATH)

    def errors_for(self, value: dict[str, object]) -> list[str]:
        return validate_lock(value, REPO_ROOT)

    def test_canonical_lock_is_valid(self) -> None:
        self.assertEqual(self.errors_for(copy.deepcopy(self.lock)), [])

    def test_schema_status_and_authority_mutations_fail(self) -> None:
        for path, value in (
            (("schema",), "wrong"),
            (("status",), "HEAD_RUN"),
            (("scientific_status",), "PASS"),
            (("execution_authority", "head_or_network"), True),
        ):
            with self.subTest(path=path):
                mutated = copy.deepcopy(self.lock)
                target = mutated
                for key in path[:-1]:
                    target = target[key]  # type: ignore[index]
                target[path[-1]] = value  # type: ignore[index]
                self.assertTrue(self.errors_for(mutated))

    def test_binding_hash_and_path_mutations_fail(self) -> None:
        mutated = copy.deepcopy(self.lock)
        mutated["bindings"][0]["sha256"] = "0" * 64
        self.assertTrue(any("hash/bytes" in error for error in self.errors_for(mutated)))
        mutated = copy.deepcopy(self.lock)
        mutated["bindings"][0]["path"] = "docs/research/taro/not-the-contract.json"
        self.assertTrue(any("role/path" in error for error in self.errors_for(mutated)))

    def test_test_count_and_source_access_mutations_fail(self) -> None:
        mutated = copy.deepcopy(self.lock)
        mutated["synthetic_validation"]["tests_run"] = 20
        self.assertTrue(self.errors_for(mutated))
        mutated = copy.deepcopy(self.lock)
        mutated["synthetic_validation"]["network_requests"] = 1
        self.assertTrue(self.errors_for(mutated))

    def test_root_and_successor_mutations_fail(self) -> None:
        mutated = copy.deepcopy(self.lock)
        mutated["exclusive_roots"][0]["exists"] = True
        self.assertTrue(self.errors_for(mutated))
        mutated = copy.deepcopy(self.lock)
        mutated["unique_successor"]["execution_authority"] = True
        self.assertTrue(self.errors_for(mutated))

    def test_shared_frame_uncertainty_interface_is_absent(self) -> None:
        source = (REPO_ROOT / "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py").read_text(encoding="utf-8")
        self.assertNotIn("def representative_uncertainty_observation(", source)
        self.assertIn("for query, lookup in zip(query_receipts, uncertainty_lookups, strict=True)", source)


if __name__ == "__main__":
    unittest.main()
