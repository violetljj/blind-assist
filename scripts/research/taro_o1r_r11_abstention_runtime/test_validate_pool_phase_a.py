from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_phase_a as runner
from scripts.research.taro_o1r_r11_abstention_runtime import validate_pool_phase_a as validator


class PhaseAValidatorTests(unittest.TestCase):
    def test_validator_is_independent_and_freezes_exact_counts(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_phase_a", source)
        self.assertEqual((validator.PARENT_COUNT, validator.FRAME_COUNT, validator.QUERY_COUNT), (48, 1043, 9387))
        self.assertEqual(validator.PRE_TERMINAL_FILE_COUNT, 5218)
        self.assertEqual(validator.FINAL_FILE_COUNT, 5219)
        self.assertEqual(sum(validator.FROZEN_FRAME_COUNTS), 1043)
        self.assertEqual(validator.EXPECTED_BINDING_PATHS, runner.EXPECTED_BINDINGS)
        self.assertEqual(validator.EXPECTED_RESOURCE_BUDGET, runner.EXPECTED_RESOURCE_BUDGET)
        self.assertEqual(validator.EXPECTED_RUNTIME_ENVIRONMENT, runner.EXPECTED_RUNTIME_ENVIRONMENT)
        self.assertEqual(validator.EXPECTED_CANDIDATE_IDENTITY, runner.EXPECTED_CANDIDATE_IDENTITY)
        self.assertEqual(
            validator.EXPECTED_DEPTHART_RUNTIME_IDENTITY,
            runner._expected_runtime_identity(
                runner.EXPECTED_CANDIDATE_IDENTITY,
                runner.EXPECTED_RUNTIME_ENVIRONMENT,
            ),
        )

    def test_seal_and_schema_mutation_fail_closed(self) -> None:
        schema = "blindassist.taro.o1r.r11_validator_fixture.v1"
        record = {"schema": schema, "value": 1}
        record["content_sha256"] = validator.adapter.canonical_sha256(record)
        self.assertEqual(validator._validate_seal(record, schema), record)
        mutated = copy.deepcopy(record)
        mutated["value"] = 2
        with self.assertRaises(validator.PhaseAValidationError):
            validator._validate_seal(mutated, schema)
        with self.assertRaises(validator.PhaseAValidationError):
            validator._validate_seal(record, schema + ".drift")

    def test_missing_or_empty_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(validator.PhaseAValidationError, "terminal.json"):
                validator.validate_evidence(Path(directory))

    def test_expected_layout_is_five_files_per_frame_plus_controls(self) -> None:
        rows = validator._frame_rows(
            validator.run_pool_inventory.validate_inventory(
                validator._load_json(validator._repo_path(validator.INVENTORY_PATH))
            )
        )
        files = {
            path
            for parent, video, token in rows
            for path in validator._relative_paths(parent, video, token).values()
        }
        files.update({"execution-receipt.json", "candidate-completion.json", "phase-a-completion.json"})
        self.assertEqual(len(files), validator.PRE_TERMINAL_FILE_COUNT)


if __name__ == "__main__":
    unittest.main()
