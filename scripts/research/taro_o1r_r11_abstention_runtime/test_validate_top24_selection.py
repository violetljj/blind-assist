from __future__ import annotations

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r11_abstention_runtime import (
    validate_top24_selection as validator,
)


def seal(value: dict) -> dict:
    record = copy.deepcopy(value)
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(adapter.canonical_json_bytes(value) + b"\n")


def feature(eligible: bool) -> dict:
    return {
        "query_receipt": {} if eligible else None,
        "r6_state": "UNKNOWN",
        "positive_obstacle_veto": False,
        "occupied_hits": [[[False]]],
        "far_valid_anchor_count": 6,
        "far_fractions": [0.0, 0.0, 0.0],
        "observed_support_points": 0,
    }


class Top24IndependentValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "selection"
        self.root.mkdir()
        self.lock_path = Path(self.temporary.name) / "execution-lock.json"
        self.lock_path.write_text("synthetic lock\n", encoding="utf-8")
        self.completion = {
            "content_sha256": validator.PHASE_A_COMPLETION_CONTENT_SHA256,
            "source_frame_hash_sequence_sha256": "A" * 64,
        }
        self.sources = self._sources()
        self._write_valid_root()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _sources(self) -> list[dict]:
        sources: list[dict] = []
        for parent_index, ((parent_id, video_id), count) in enumerate(
            zip(validator.EXPECTED_IDENTITIES, validator.FROZEN_FRAME_COUNTS, strict=True)
        ):
            for frame_index in range(count):
                sources.append(
                    {
                        "parent_id": parent_id,
                        "video_id": video_id,
                        "source_phase_has_label_input": False,
                        "training_steps": 0,
                        "network_requests": 0,
                        "query_features": [feature(parent_index < 24) for _ in range(9)],
                        "content_sha256": adapter.canonical_sha256(
                            [parent_id, video_id, frame_index]
                        ),
                    }
                )
        self.assertEqual(len(sources), validator.FRAME_COUNT)
        return sources

    @staticmethod
    def _zeros() -> dict[str, int]:
        return {field: 0 for field in validator.ZERO_FIELDS}

    def _write_valid_root(self) -> None:
        scores = validator._score_sources(self.sources)
        ranked = validator._rank(scores)
        selected = ranked[: validator.SELECTED_PARENT_COUNT]
        scores_record = seal(
            {
                "schema": validator.SCORES_SCHEMA,
                "selector": validator.FROZEN_SELECTOR,
                "phase_a_completion_sha256": self.completion["content_sha256"],
                "source_frame_hash_sequence_sha256": self.completion[
                    "source_frame_hash_sequence_sha256"
                ],
                "parent_count": validator.PARENT_COUNT,
                "frame_count": validator.FRAME_COUNT,
                "query_count": validator.QUERY_COUNT,
                "parent_scores": scores,
                "ranked_parent_identities": [
                    [row["parent_id"], row["video_id"]] for row in ranked
                ],
                "all_48_source_records_sealed_before_scoring": True,
                "all_48_parent_scores_sealed_before_faro": True,
                **self._zeros(),
            }
        )
        selection = seal(
            {
                "schema": validator.SELECTION_SCHEMA,
                "selector": validator.FROZEN_SELECTOR,
                "parent_scores_sha256": scores_record["content_sha256"],
                "selected_parent_count": validator.SELECTED_PARENT_COUNT,
                "selected_parent_identities": [
                    [row["parent_id"], row["video_id"]] for row in selected
                ],
                "selected_parent_scores": selected,
                "selection_sealed_before_faro": True,
                "read_unselected_faro": False,
                "source_reselection_after_faro": False,
                "parent_reselection_after_faro": False,
                "candidate_or_threshold_reselection_after_faro": False,
                "unknown_is_negative": False,
                **self._zeros(),
            }
        )
        execution = seal(
            {
                "schema": validator.EXECUTION_SCHEMA,
                "execution_lock_sha256": materializer.sha256_file(self.lock_path),
                "execution_lock_content_sha256": "L" * 64,
                "started_at_utc": "2026-08-13T00:00:00+00:00",
                "phase_a_root": validator.PHASE_A_ROOT,
                "phase_a_audit_content_sha256": validator.PHASE_A_AUDIT_CONTENT_SHA256,
                "frozen_selector": validator.FROZEN_SELECTOR,
                "one_shot_consumed_on_root_creation": True,
                **self._zeros(),
            }
        )
        result = seal(
            {
                "schema": validator.RESULT_SCHEMA,
                "terminal": validator.PASS_TERMINAL,
                "passed": True,
                "execution_valid": True,
                "parent_count": validator.PARENT_COUNT,
                "frame_count": validator.FRAME_COUNT,
                "query_count": validator.QUERY_COUNT,
                "selected_parent_count": validator.SELECTED_PARENT_COUNT,
                "selected_parent_identities": selection["selected_parent_identities"],
                "phase_a_terminal_content_sha256": validator.PHASE_A_TERMINAL_CONTENT_SHA256,
                "phase_a_completion_sha256": self.completion["content_sha256"],
                "phase_a_audit_content_sha256": validator.PHASE_A_AUDIT_CONTENT_SHA256,
                "parent_scores_sha256": scores_record["content_sha256"],
                "selection_sha256": selection["content_sha256"],
                "all_48_source_records_sealed_before_scoring": True,
                "all_48_parent_scores_sealed_before_faro": True,
                "selection_sealed_before_faro": True,
                "unknown_is_negative": False,
                "phase_a_prior_file_validations": validator.PHASE_A_PRE_TERMINAL_FILE_COUNT,
                "phase_a_prior_bytes_validated": 123456,
                "phase_a_lineage_decodes": validator.FRAME_COUNT,
                "phase_a_source_receipt_decodes": validator.FRAME_COUNT,
                "source_frame_records_validated": validator.FRAME_COUNT,
                "query_features_scored": validator.QUERY_COUNT,
                **self._zeros(),
                "elapsed_seconds": 1.25,
                "peak_rss_bytes": 1_000_000,
                "resource_budget": validator.EXPECTED_RESOURCE_BUDGET,
                "one_shot_consumed": True,
                "unique_successor": "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK",
                "claim_ceiling": validator.EXPECTED_CLAIM_CEILING,
            }
        )
        write_json(self.root / "execution-receipt.json", execution)
        write_json(self.root / "parent-scores.json", scores_record)
        write_json(self.root / "selection.json", selection)
        files = {
            name: {
                "path": name,
                "bytes": (self.root / name).stat().st_size,
                "sha256": materializer.sha256_file(self.root / name),
            }
            for name in (
                "execution-receipt.json",
                "parent-scores.json",
                "selection.json",
            )
        }
        terminal = seal(
            {
                "schema": validator.TERMINAL_SCHEMA,
                "terminal": validator.PASS_TERMINAL,
                "passed": True,
                "execution_valid": True,
                "result": result,
                "files": files,
                "file_count_before_terminal": 3,
                "bytes_before_terminal": sum(row["bytes"] for row in files.values()),
                "one_shot_consumed": True,
            }
        )
        write_json(self.root / "terminal.json", terminal)

    def _validate(self) -> dict:
        lock = {"_path": self.lock_path, "content_sha256": "L" * 64}
        with (
            mock.patch.object(validator, "validate_execution_lock", return_value=lock),
            mock.patch.object(
                validator,
                "_validate_phase_a_predecessor",
                return_value=({}, self.completion),
            ),
            mock.patch.object(
                validator,
                "_load_phase_a_sources",
                return_value=self.sources,
            ),
        ):
            return validator.validate_evidence(self.root, self.lock_path)

    def _replace_and_reterminal(self, name: str, value: dict) -> None:
        write_json(self.root / name, value)
        terminal = json.loads((self.root / "terminal.json").read_text(encoding="utf-8"))
        terminal.pop("content_sha256")
        for relative in terminal["files"]:
            path = self.root / relative
            terminal["files"][relative] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": materializer.sha256_file(path),
            }
        terminal["bytes_before_terminal"] = sum(
            row["bytes"] for row in terminal["files"].values()
        )
        write_json(self.root / "terminal.json", seal(terminal))

    def test_valid_synthetic_root_recomputes_all_48_and_top24(self) -> None:
        result = self._validate()
        self.assertTrue(result["passed"])
        self.assertTrue(result["independently_recomputed_eligible_counts"])
        self.assertEqual(result["root_file_count"], 4)
        self.assertEqual(result["selected_parent_count"], 24)
        self.assertEqual(result["faro_reads"], 0)
        self.assertFalse(result["unknown_is_negative"])

    def test_resealed_parent_score_mutation_is_rejected(self) -> None:
        record = json.loads((self.root / "parent-scores.json").read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["parent_scores"][0]["eligible_query_count"] -= 1
        self._replace_and_reterminal("parent-scores.json", seal(record))
        with self.assertRaisesRegex(validator.Top24ValidationError, "recomputed"):
            self._validate()

    def test_resealed_top24_tie_break_mutation_is_rejected(self) -> None:
        record = json.loads((self.root / "selection.json").read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["selected_parent_identities"][0], record["selected_parent_identities"][-1] = (
            record["selected_parent_identities"][-1],
            record["selected_parent_identities"][0],
        )
        self._replace_and_reterminal("selection.json", seal(record))
        with self.assertRaisesRegex(validator.Top24ValidationError, "top24"):
            self._validate()

    def test_firewall_ledger_mutation_is_rejected(self) -> None:
        record = json.loads((self.root / "execution-receipt.json").read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["faro_reads"] = 1
        self._replace_and_reterminal("execution-receipt.json", seal(record))
        with self.assertRaisesRegex(validator.Top24ValidationError, "firewall"):
            self._validate()

    def test_resealed_extra_result_side_field_is_rejected(self) -> None:
        record = json.loads((self.root / "selection.json").read_text(encoding="utf-8"))
        record.pop("content_sha256")
        record["faro_payload_sha256"] = "F" * 64
        self._replace_and_reterminal("selection.json", seal(record))
        with self.assertRaisesRegex(validator.Top24ValidationError, "top24"):
            self._validate()

    def test_exact_four_file_root_rejects_extra_file(self) -> None:
        (self.root / "extra.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(validator.Top24ValidationError, "four-file"):
            self._validate()

    def test_execution_lock_binding_and_policy_are_independently_checked(self) -> None:
        repo = Path(self.temporary.name) / "repo"
        lock_path = repo / "lock.json"
        bound = repo / "bound.txt"
        bound.parent.mkdir(parents=True)
        bound.write_text("bound\n", encoding="utf-8")
        binding = {
            "role": "ONLY",
            "path": "bound.txt",
            "bytes": bound.stat().st_size,
            "sha256": materializer.sha256_file(bound),
        }
        lock = seal(
            {
                "schema": validator.LOCK_SCHEMA,
                "lock_id": validator.LOCK_ID,
                "status": "AUTHORIZED_UNCONSUMED",
                "consumed": False,
                "argv": validator.EXPECTED_ARGV,
                "phase_a_root": validator.PHASE_A_ROOT,
                "phase_a_repaired_audit": validator.PHASE_A_AUDIT,
                "output_root": validator.EVIDENCE_ROOT,
                "overwrite": False,
                "rerun": False,
                "frozen_selector": validator.FROZEN_SELECTOR,
                "user_authority": validator.EXPECTED_USER_AUTHORITY,
                "execution_authority": validator.EXPECTED_AUTHORITY,
                "resource_budget": validator.EXPECTED_RESOURCE_BUDGET,
                "one_shot_policy": validator.EXPECTED_ONE_SHOT_POLICY,
                "phase_a_terminal_content_sha256": validator.PHASE_A_TERMINAL_CONTENT_SHA256,
                "phase_a_audit_content_sha256": validator.PHASE_A_AUDIT_CONTENT_SHA256,
                "implementation_on_origin_master": True,
                "implementation_commit": "A" * 40,
                "bindings": [binding],
            }
        )
        write_json(lock_path, lock)
        with (
            mock.patch.object(validator, "REPO_ROOT", repo),
            mock.patch.object(validator, "LOCK_RELATIVE", "lock.json"),
            mock.patch.object(validator, "EXPECTED_BINDINGS", {"ONLY": "bound.txt"}),
            mock.patch.object(validator, "ARTIFACT_BINDING_ROLES", set()),
            mock.patch.object(validator, "_commit_is_on_master", return_value=True),
            mock.patch.object(validator, "_git_bytes", return_value=bound.read_bytes()) as git_bytes,
            mock.patch.object(validator, "_validate_frozen_selector_artifact"),
        ):
            validated = validator.validate_execution_lock(lock_path)
            self.assertEqual(validated["content_sha256"], lock["content_sha256"])
            git_bytes.assert_called_once_with("A" * 40, "bound.txt")
            bound.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(validator.Top24ValidationError, "binding drift"):
                validator.validate_execution_lock(lock_path)

        bound.write_text("bound\n", encoding="utf-8")
        with (
            mock.patch.object(validator, "REPO_ROOT", repo),
            mock.patch.object(validator, "LOCK_RELATIVE", "lock.json"),
            mock.patch.object(validator, "EXPECTED_BINDINGS", {"ONLY": "bound.txt"}),
            mock.patch.object(validator, "ARTIFACT_BINDING_ROLES", set()),
            mock.patch.object(validator, "_commit_is_on_master", return_value=True),
            mock.patch.object(validator, "_git_bytes", return_value=b"implementation drift\n"),
            mock.patch.object(validator, "_validate_frozen_selector_artifact"),
            self.assertRaisesRegex(validator.Top24ValidationError, "implementation-commit binding drift"),
        ):
            validator.validate_execution_lock(lock_path)

    def test_validator_does_not_import_producer_or_result_side_runtime(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        joined = "\n".join(imports).lower()
        self.assertNotIn("run_top24_selection", joined)
        for forbidden in ("faro", "depthart", "torch", "requests", "urllib"):
            self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
