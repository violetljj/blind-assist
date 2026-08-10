#!/usr/bin/env python3
"""Focused fail-closed tests for factor-headroom lock preparation."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts.research.taro_o0r_factor_headroom_runtime import (
    prepare_factor_headroom_execution as prepare,
)


_RUNNER_CONSTANTS = {
    "EXECUTION_LOCK_SCHEMA",
    "EXECUTION_LOCK_ID",
    "EXPECTED_ROOTS",
    "EXPECTED_AUTHORITY",
    "EXPECTED_BUDGET",
    "EXPECTED_RUNTIME",
    "EXPECTED_BINDING_PATHS",
    "STRUCTURALLY_NOT_APPLICABLE_STRATA",
}


def _literal_runner_constants() -> dict[str, object]:
    """Read dependency-free literal constants without importing NumPy/Torch."""

    tree = ast.parse(Path(__file__).with_name("run_factor_headroom.py").read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in _RUNNER_CONSTANTS:
            values[target.id] = ast.literal_eval(node.value)
    if set(values) != _RUNNER_CONSTANTS:
        raise AssertionError(f"runner literal contract is incomplete: {sorted(_RUNNER_CONSTANTS - set(values))}")
    return values


class PrepareFactorHeadroomExecutionTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name).resolve()
        constants = _literal_runner_constants()
        self.contract = SimpleNamespace(
            **constants,
            adapter=SimpleNamespace(
                BASELINE_MODEL_ID="depthart-test-model",
                BASELINE_CHECKPOINT_SHA256="",
            ),
        )
        self.base_commit = "1" * 40
        self.execution_commit = "2" * 40

        self.truth_root = self.repo / self.contract.EXPECTED_ROOTS["TRUTH_EVIDENCE"]
        self.factor_root = self.repo / self.contract.EXPECTED_ROOTS["FACTOR_EVIDENCE"]
        (self.repo / self.contract.EXPECTED_ROOTS["SOURCE"]).mkdir(parents=True)
        self.depthart_source = self.repo / "vendor" / "DepthART"
        self.depthart_source.mkdir(parents=True)
        self.checkpoint = self.repo / "artifacts.local" / "models" / "depthart-test.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"synthetic checkpoint identity")
        self.checkpoint_sha256 = prepare.sha256_file(self.checkpoint)
        self.contract.adapter.BASELINE_CHECKPOINT_SHA256 = self.checkpoint_sha256

        self._write_r3_evidence()
        self._write_runner_bindings()
        self.upstream_bindings = prepare.DEFAULT_UPSTREAM_BINDINGS
        for relative in self.upstream_bindings.values():
            self._write(relative, b"{}\n")

    def _write(self, relative: str, payload: bytes) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _write_json(self, relative: str, payload: object) -> Path:
        return self._write(
            relative,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n",
        )

    def _write_r3_evidence(self, *, passed: bool = True) -> None:
        self.truth_root.mkdir(parents=True, exist_ok=True)
        result = {
            "schema": prepare.R3_RESULT_SCHEMA,
            "passed": passed,
            "scientific_status": "TRUTH_ONLY_ADMISSION_PASS" if passed else "NOT_EVALUABLE",
            "terminal": prepare.R3_PASS_TERMINAL if passed else "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE",
            "gates": {"passed": passed, "failure_codes": [] if passed else ["MINIMUM_EVALUABLE_O0R_PARENTS"]},
            "model_outputs_absent": True,
            "depthart_inference_count": 0,
            "factorial_execution_count": 0,
        }
        completion = {
            "schema": prepare.R3_COMPLETION_SCHEMA,
            "passed": passed,
            "terminal": result["terminal"],
            "one_shot_consumed": True,
            "elapsed_seconds": 7.5,
        }
        self._write_json(f"{self.contract.EXPECTED_ROOTS['TRUTH_EVIDENCE']}/result.json", result)
        self._write_json(
            f"{self.contract.EXPECTED_ROOTS['TRUTH_EVIDENCE']}/completion-receipt.json",
            completion,
        )
        for relative, payload in {
            "execution-receipt.json": b"{}\n",
            "download-receipts.json.gz": b"download-receipts",
            "exact-frame-plan.json.gz": b"exact-frame-plan",
            "uncertainty-model-receipt.json": b"{}\n",
            "uncertainty-model-artifact.json.gz": b"uncertainty-model",
            "frame-failures.json.gz": b"frame-failures",
            "truth-frames/41069021/000001.json.gz": b"compact-truth-frame",
        }.items():
            self._write(f"{self.contract.EXPECTED_ROOTS['TRUTH_EVIDENCE']}/{relative}", payload)
        self._rewrite_manifest()

    def _rewrite_manifest(self) -> None:
        files: dict[str, dict[str, object]] = {}
        for path in sorted(self.truth_root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            relative = path.relative_to(self.truth_root).as_posix()
            files[relative] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": prepare.sha256_file(path),
            }
        manifest = {
            "schema": prepare.R3_MANIFEST_SCHEMA,
            "truth_root_consumed": True,
            "file_count_before_manifest": len(files),
            "bytes_before_manifest": sum(int(row["bytes"]) for row in files.values()),
            "files": files,
        }
        self._write_json(f"{self.contract.EXPECTED_ROOTS['TRUTH_EVIDENCE']}/manifest.json", manifest)

    def _write_runner_bindings(self) -> None:
        implementation_path = self.contract.EXPECTED_BINDING_PATHS["FACTOR_IMPLEMENTATION_LOCK"]
        for role, relative in self.contract.EXPECTED_BINDING_PATHS.items():
            if role == "FACTOR_IMPLEMENTATION_LOCK" or relative == implementation_path:
                continue
            path = self.repo / relative
            if path.exists():
                continue
            payload = b"# synthetic bound runtime\n" if path.suffix == ".py" else b"{}\n"
            self._write(relative, payload)
        runtime_root = self.repo / Path(self.contract.EXPECTED_BINDING_PATHS["FACTOR_RUNNER"]).parent
        self._write(
            (runtime_root.relative_to(self.repo) / "test_synthetic_runtime.py").as_posix(),
            b"# discovered focused test module\n",
        )
        self._write((runtime_root.relative_to(self.repo) / "README.md").as_posix(), b"synthetic runtime\n")

    def _git_probe(self, root: Path) -> dict[str, object]:
        commit = prepare.EXPECTED_DEPTHART_SOURCE_COMMIT if root == self.depthart_source else self.base_commit
        return {"commit": commit, "clean": True, "status": []}

    def _passing_test_runner(self, command, cwd, environment):
        self.assertEqual(cwd, self.repo)
        self.assertEqual(dict(environment), prepare.REQUIRED_ENVIRONMENT)
        return {
            "command": list(command),
            "returncode": 0,
            "tests_run": 9,
            "tests_passed": 9,
            "failures": 0,
            "errors": 0,
            "network_requests": 0,
            "runtime": copy.deepcopy(self.contract.EXPECTED_RUNTIME),
            "stdout_sha256": "A" * 64,
            "stderr_sha256": "B" * 64,
        }

    def _prepare(self, *, contract=None, git_probe=None, test_runner=None):
        active_contract = contract or self.contract
        with (
            mock.patch.object(prepare, "_load_runner_contract", return_value=active_contract),
            mock.patch.object(prepare, "DEFAULT_CHECKPOINT_SHA256", self.checkpoint_sha256),
            mock.patch.object(prepare, "_default_git_probe", side_effect=git_probe or self._git_probe),
            mock.patch.object(
                prepare,
                "_default_test_runner",
                side_effect=test_runner or self._passing_test_runner,
            ),
        ):
            return prepare.prepare_factor_headroom_execution(
                repo_root=self.repo,
                r3_truth_root=self.truth_root,
                factor_root=self.factor_root,
                depthart_source_root=self.depthart_source,
                depthart_source_commit=prepare.EXPECTED_DEPTHART_SOURCE_COMMIT,
                checkpoint_path=self.checkpoint,
                checkpoint_sha256=self.checkpoint_sha256,
                implementation_base_commit=self.base_commit,
                execution_commit=self.execution_commit,
            )

    def assert_preparation_error(self, code: str, callback) -> None:
        with self.assertRaises(prepare.PreparationError) as raised:
            callback()
        self.assertEqual(raised.exception.code, code)

    def test_prepares_exact_runner_bound_locks_without_writes(self) -> None:
        files_before = sorted(path.relative_to(self.repo).as_posix() for path in self.repo.rglob("*") if path.is_file())
        prepared = self._prepare()
        files_after = sorted(path.relative_to(self.repo).as_posix() for path in self.repo.rglob("*") if path.is_file())

        self.assertEqual(files_after, files_before)
        self.assertEqual(prepared["status"], "READY_FOR_SEPARATE_LOCK_COMMIT_NO_EXECUTION")
        self.assertEqual(prepared["writes_performed"], 0)
        self.assertFalse(prepared["factor_root_created"])
        self.assertFalse(self.factor_root.exists())
        implementation_receipt = prepared["implementation_lock_receipt"]
        self.assertFalse((self.repo / implementation_receipt["path"]).exists())
        self.assertFalse((self.repo / prepared["execution_lock_receipt"]["path"]).exists())
        self.assertEqual(
            implementation_receipt["sha256"],
            prepare.sha256_bytes(prepare.canonical_lock_bytes(prepared["implementation_lock"])),
        )

        lock = prepared["execution_lock"]
        self.assertEqual(lock["schema"], self.contract.EXECUTION_LOCK_SCHEMA)
        self.assertEqual(lock["lock_id"], self.contract.EXECUTION_LOCK_ID)
        self.assertEqual(lock["roots"], self.contract.EXPECTED_ROOTS)
        self.assertEqual(lock["execution_authority"], self.contract.EXPECTED_AUTHORITY)
        self.assertEqual(lock["resource_budget"], self.contract.EXPECTED_BUDGET)
        self.assertEqual(lock["runtime"], self.contract.EXPECTED_RUNTIME)
        self.assertEqual(lock["resource_budget"]["maximum_cuda_allocated_bytes"], 8_500_000_000)
        self.assertEqual(
            lock["structurally_not_applicable_strata"],
            self.contract.STRUCTURALLY_NOT_APPLICABLE_STRATA,
        )
        self.assertEqual(lock["overwrite"], False)
        self.assertEqual(lock["rerun"], False)
        self.assertEqual(lock["argv_alternatives"], [])
        self.assertEqual(
            lock["argv"],
            [
                self.contract.EXPECTED_BINDING_PATHS["FACTOR_RUNNER"],
                "--execution-lock",
                prepare.DEFAULT_EXECUTION_LOCK_PATH,
            ],
        )
        self.assertEqual(
            {row["role"]: row["path"] for row in lock["bindings"]},
            self.contract.EXPECTED_BINDING_PATHS,
        )
        self.assertEqual(set(lock["depthart_assets"]), {
            "source_root",
            "source_git_commit",
            "checkpoint_path",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "model_id",
        })

    def test_rejects_tampered_or_unlisted_r3_files(self) -> None:
        (self.truth_root / "exact-frame-plan.json.gz").write_bytes(b"tampered")
        self.assert_preparation_error("R3_LEDGER_FILE_MISMATCH", self._prepare)

        self._write_r3_evidence()
        (self.truth_root / "unlisted.bin").write_bytes(b"unlisted")
        self.assert_preparation_error("R3_LEDGER_FILE_SET_MISMATCH", self._prepare)

    def test_accepts_exact_not_evaluable_for_descriptive_canary_only(self) -> None:
        self._write_r3_evidence(passed=False)
        prepared = self._prepare()
        admission = prepared["implementation_lock"]["r3_truth_admission"]
        self.assertFalse(admission["formal_headroom_authorized"])
        self.assertTrue(admission["descriptive_partial_factor_canary_authorized"])
        self.assertEqual(["MINIMUM_EVALUABLE_O0R_PARENTS"], admission["r3_failure_codes"])

    def test_rejects_internally_inconsistent_r3_terminal(self) -> None:
        result_path = self.truth_root / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["passed"] = False
        self._write_json(result_path.relative_to(self.repo).as_posix(), result)
        self.assert_preparation_error("R3_RESULT_IDENTITY_INVALID", self._prepare)

    def test_rejects_existing_factor_root(self) -> None:
        self.factor_root.mkdir(parents=True)
        self.assert_preparation_error("FACTOR_ROOT_ALREADY_EXISTS", self._prepare)

    def test_rejects_depthart_commit_and_checkpoint_drift(self) -> None:
        def bad_probe(root: Path) -> dict[str, object]:
            value = self._git_probe(root)
            if root == self.depthart_source:
                value["commit"] = "3" * 40
            return value

        self.assert_preparation_error("DEPTHART_SOURCE_COMMIT_MISMATCH", lambda: self._prepare(git_probe=bad_probe))

    def test_preserves_unrelated_dirty_work_but_rejects_bound_taro_drift(self) -> None:
        def unrelated_probe(root: Path) -> dict[str, object]:
            value = self._git_probe(root)
            if root.resolve() == self.repo.resolve():
                value = dict(value)
                value["clean"] = False
                value["status"] = [" M scripts/research/assistive_geometry/unrelated_user_work.py"]
            return value

        prepared = self._prepare(git_probe=unrelated_probe)
        identity = prepared["implementation_lock"]["repository_identity"]
        self.assertFalse(identity["clean"])
        self.assertTrue(identity["protected_paths_clean"])
        self.assertEqual(1, len(identity["unrelated_dirty_status"]))

        def protected_probe(root: Path) -> dict[str, object]:
            value = self._git_probe(root)
            if root.resolve() == self.repo.resolve():
                value = dict(value)
                value["clean"] = False
                value["status"] = [" M scripts/research/taro_o0r_factor_headroom_runtime/run_factor_headroom.py"]
            return value

        self.assert_preparation_error("REPOSITORY_PROTECTED_PATH_DIRTY", lambda: self._prepare(git_probe=protected_probe))
        self.checkpoint.write_bytes(b"tampered checkpoint")
        self.assert_preparation_error("CHECKPOINT_SHA256_MISMATCH", self._prepare)

    def test_rejects_failed_focused_tests_and_runtime_drift(self) -> None:
        def failed(command, cwd, environment):
            receipt = self._passing_test_runner(command, cwd, environment)
            receipt.update(returncode=1, tests_passed=8, failures=1)
            return receipt

        self.assert_preparation_error("FOCUSED_TESTS_FAILED", lambda: self._prepare(test_runner=failed))

        def drifted(command, cwd, environment):
            receipt = self._passing_test_runner(command, cwd, environment)
            receipt["runtime"] = {**self.contract.EXPECTED_RUNTIME, "torch": "drift"}
            return receipt

        self.assert_preparation_error("FOCUSED_TEST_RUNTIME_DRIFT", lambda: self._prepare(test_runner=drifted))

    def test_rejects_runtime_mutation_during_focused_tests(self) -> None:
        def mutating(command, cwd, environment):
            receipt = self._passing_test_runner(command, cwd, environment)
            runner = self.repo / self.contract.EXPECTED_BINDING_PATHS["FACTOR_RUNNER"]
            runner.write_bytes(b"# mutated while tests ran\n")
            return receipt

        self.assert_preparation_error(
            "FACTOR_RUNTIME_CHANGED_DURING_TESTS",
            lambda: self._prepare(test_runner=mutating),
        )

    def test_rejects_factor_root_created_during_focused_tests(self) -> None:
        def consuming(command, cwd, environment):
            receipt = self._passing_test_runner(command, cwd, environment)
            self.factor_root.mkdir(parents=True)
            return receipt

        self.assert_preparation_error(
            "FACTOR_ROOT_CREATED_DURING_TESTS",
            lambda: self._prepare(test_runner=consuming),
        )

    def test_rejects_runner_budget_or_stratum_policy_drift(self) -> None:
        bad_budget = SimpleNamespace(**vars(self.contract))
        bad_budget.EXPECTED_BUDGET = {**self.contract.EXPECTED_BUDGET, "wall_seconds": 28_801}
        self.assert_preparation_error("RUNNER_BUDGET_EXCEEDS_AUTHORITY", lambda: self._prepare(contract=bad_budget))

        bad_policy = SimpleNamespace(**vars(self.contract))
        bad_policy.STRUCTURALLY_NOT_APPLICABLE_STRATA = {}
        self.assert_preparation_error("RUNNER_STRATUM_POLICY_DRIFT", lambda: self._prepare(contract=bad_policy))

        bad_authority = SimpleNamespace(**vars(self.contract))
        bad_authority.EXPECTED_AUTHORITY = {**self.contract.EXPECTED_AUTHORITY, "device": True}
        self.assert_preparation_error("RUNNER_AUTHORITY_DRIFT", lambda: self._prepare(contract=bad_authority))


if __name__ == "__main__":
    unittest.main()
