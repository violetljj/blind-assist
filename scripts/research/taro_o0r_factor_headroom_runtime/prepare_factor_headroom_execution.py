#!/usr/bin/env python3
"""Prepare, but never write, TARO O0R factor-headroom execution locks.

This module is a fail-closed bridge from a completed R3 truth-only terminal to
the future DepthART one-shot.  R3 PASS authorizes the formal factorial analysis;
the exact retained NOT_EVALUABLE terminal authorizes only the threshold-free
descriptive partial-factor canary.  It validates the complete R3 evidence
ledger, factor-root absence, repository and DepthART source identities, the
frozen checkpoint, and every focused test in the factor runtime.  On success it
returns canonical implementation-lock and execution-lock payloads.

No function in this file creates the factor root or writes a docs lock.  The
CLI prints the prepared payload to stdout only.  A later governance step must
commit the exact canonical payload before any scientific execution.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


R3_RESULT_SCHEMA = "blindassist.taro.o0r.truth_only_result.v1"
R3_COMPLETION_SCHEMA = "blindassist.taro.o0r.truth_only_execution_completion_receipt.v1"
R3_MANIFEST_SCHEMA = "blindassist.taro.o0r.truth_materializer_manifest.v1"
R3_PASS_TERMINAL = "TARO_O0R_TRUTH_ONLY_ADMISSION_PASS"

IMPLEMENTATION_LOCK_SCHEMA = "blindassist.taro.o0r.factor_headroom_r3_implementation_lock.v1"
IMPLEMENTATION_LOCK_ID = "TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_IMPLEMENTATION_LOCK"

DEFAULT_EXECUTION_LOCK_PATH = (
    "docs/research/taro/"
    "TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_ONE_SHOT_EXECUTION_LOCK_2026-08-10.json"
)
DEFAULT_CHECKPOINT_SHA256 = "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65"
EXPECTED_DEPTHART_SOURCE_COMMIT = "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c"

REQUIRED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
}

DEFAULT_UPSTREAM_BINDINGS = {
    "R3_RECOVERY_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R3_LOCK_2026-08-10.json",
    "R3_MATERIALIZER_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R3_IMPLEMENTATION_LOCK_2026-08-10.json",
    "R3_EXECUTION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_R3_EXECUTION_LOCK_2026-08-10.json",
    "TRUTH_PREFLIGHT": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER_CONTRACT": "docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "MATERIALIZER": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "TRUTH_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
}

_RUNNER_ROOT_ROLES = {"SOURCE", "TRUTH_EVIDENCE", "FACTOR_EVIDENCE"}
_RUNNER_AUTHORITY_TRUE = {
    "depthart_inference",
    "factorial_execution",
    "descriptive_partial_factor_canary",
    "source_cache_reuse",
    "truth_recomputation_after_candidate_seal",
}
_RUNNER_AUTHORITY_FALSE = {"training", "network", "device", "product", "safety"}
_RUNNER_BUDGET_FIELDS = {
    "wall_seconds",
    "peak_rss_bytes",
    "maximum_evidence_bytes",
    "maximum_cuda_allocated_bytes",
    "training_steps",
    "network_requests",
}
_RUNNER_RUNTIME_FIELDS = {
    "python",
    "numpy",
    "opencv",
    "torch",
    "torch_cuda",
    "cuda_device",
    "psutil",
    "pillow",
}
_REQUIRED_RUNNER_BINDING_ROLES = {
    "FACTOR_IMPLEMENTATION_LOCK",
    "R3_RESULT",
    "R3_COMPLETION",
    "R3_MANIFEST",
    "R3_EXACT_FRAME_PLAN",
    "R3_DOWNLOAD_RECEIPTS",
    "R3_UNCERTAINTY_RECEIPT",
    "R3_UNCERTAINTY_ARTIFACT",
    "TRUTH_PREFLIGHT",
    "R3_IMPLEMENTATION_LOCK",
    "R3_EXECUTION_LOCK",
    "SOURCE_ADAPTER",
    "MATERIALIZER",
    "TRUTH_RUNNER",
    "DEPTHART_RUNNER",
    "CANDIDATE_INPUTS",
    "CANDIDATE_PHASE",
    "EVIDENCE_WRITER",
    "UNCERTAINTY_LOADER",
    "UNCERTAINTY_REFIT",
    "TRUTH_RECOMPUTE",
    "FACTOR_HEADROOM",
    "FACTOR_CANARY",
    "FACTOR_EVALUATOR",
    "STATISTICS",
    "FACTOR_RUNNER",
}

_SHA256 = re.compile(r"[0-9A-Fa-f]{64}")
_GIT_SHA = re.compile(r"[0-9A-Fa-f]{40}")
_TEST_COUNT = re.compile(r"Ran\s+(\d+)\s+tests?", re.IGNORECASE)

GitProbe = Callable[[Path], Mapping[str, Any]]
FocusedTestRunner = Callable[[tuple[str, ...], Path, Mapping[str, str]], Mapping[str, Any]]


class PreparationError(ValueError):
    """The proposed lock cannot safely authorize execution."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.context = context


def _load_runner_contract() -> Any:
    """Import the runner as the sole execution-lock contract authority."""

    from scripts.research.taro_o0r_factor_headroom_runtime import run_factor_headroom

    return run_factor_headroom


def _runner_mapping(runner_contract: Any, field: str) -> dict[str, Any]:
    value = getattr(runner_contract, field, None)
    _require(
        isinstance(value, Mapping),
        "RUNNER_CONTRACT_INVALID",
        "runner contract field must be a mapping",
        field=field,
    )
    return dict(value)


def _require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise PreparationError(code, message, **context)


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used to hash both future lock payloads."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreparationError("CANONICAL_JSON_INVALID", "lock payload is not canonical JSON") from error


def canonical_lock_bytes(value: Mapping[str, Any]) -> bytes:
    """Exact bytes a later writer must use for a prepared lock file."""

    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(code, "JSON artifact cannot be read", path=str(path)) from error
    _require(isinstance(value, dict), code, "JSON artifact must be an object", path=str(path))
    return value


def _absolute_lexical(path: Path, *, base: Path | None = None) -> Path:
    candidate = path if path.is_absolute() else (base / path if base is not None else path)
    return Path(os.path.abspath(os.fspath(candidate)))


def _relative_path(value: str, code: str) -> str:
    _require(isinstance(value, str) and bool(value), code, "path must be a non-empty string")
    _require("\\" not in value, code, "bound paths must use forward slashes", path=value)
    path = PurePosixPath(value)
    _require(not path.is_absolute(), code, "bound path must be repository-relative", path=value)
    _require(all(part not in ("", ".", "..") for part in path.parts), code, "bound path contains unsafe components", path=value)
    normalized = path.as_posix()
    _require(normalized == value, code, "bound path is not normalized", path=value)
    return normalized


def _display_path(path: Path, repo_root: Path) -> str:
    lexical_path = _absolute_lexical(path)
    lexical_root = _absolute_lexical(repo_root)
    try:
        return lexical_path.relative_to(lexical_root).as_posix()
    except ValueError:
        return lexical_path.as_posix()


def _safe_evidence_file(root: Path, relative: str) -> Path:
    normalized = _relative_path(relative, "R3_LEDGER_PATH_INVALID")
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreparationError("R3_LEDGER_PATH_ESCAPE", "ledger file escapes the R3 truth root", path=relative) from error
    _require(candidate.is_file() and not candidate.is_symlink(), "R3_LEDGER_FILE_INVALID", "ledger entry is not a regular file", path=relative)
    return candidate


def _file_receipt(path: Path, display_path: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "BOUND_FILE_INVALID", "bound path is not a regular file", path=str(path))
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_r3_truth_evidence(r3_truth_root: Path, repo_root: Path) -> dict[str, Any]:
    """Validate a consumed R3 terminal plus its complete on-disk file ledger."""

    root = _absolute_lexical(r3_truth_root, base=repo_root)
    _require(root.is_dir(), "R3_TRUTH_ROOT_MISSING", "R3 truth evidence root is absent", root=str(root))
    _require(not root.is_symlink(), "R3_TRUTH_ROOT_SYMLINK", "R3 truth root cannot be a symlink", root=str(root))
    result_path = root / "result.json"
    completion_path = root / "completion-receipt.json"
    manifest_path = root / "manifest.json"
    for required in (result_path, completion_path, manifest_path):
        _require(required.is_file() and not required.is_symlink(), "R3_REQUIRED_ARTIFACT_MISSING", "required R3 artifact is absent", path=str(required))

    result = _load_json(result_path, "R3_RESULT_INVALID")
    completion = _load_json(completion_path, "R3_COMPLETION_INVALID")
    manifest = _load_json(manifest_path, "R3_MANIFEST_INVALID")

    _require(result.get("schema") == R3_RESULT_SCHEMA, "R3_RESULT_SCHEMA_INVALID", "R3 result schema drift")
    gates = result.get("gates")
    _require(isinstance(gates, dict) and isinstance(result.get("passed"), bool), "R3_GATES_INVALID", "R3 truth admission gates are malformed")
    if result["passed"]:
        _require(
            result.get("scientific_status") == "TRUTH_ONLY_ADMISSION_PASS"
            and result.get("terminal") == R3_PASS_TERMINAL
            and gates.get("passed") is True
            and gates.get("failure_codes") == [],
            "R3_RESULT_IDENTITY_INVALID",
            "R3 PASS identity/gates drift",
        )
    else:
        _require(
            result.get("scientific_status") == "NOT_EVALUABLE"
            and result.get("terminal") == "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE"
            and gates.get("passed") is False
            and isinstance(gates.get("failure_codes"), list)
            and bool(gates["failure_codes"]),
            "R3_RESULT_IDENTITY_INVALID",
            "R3 retained NOT_EVALUABLE identity/gates drift",
        )
    _require(
        result.get("model_outputs_absent") is True
        and result.get("depthart_inference_count") == 0
        and result.get("factorial_execution_count") == 0,
        "R3_OUTCOME_LEAKAGE",
        "R3 truth result contains model or factorial execution",
    )

    _require(completion.get("schema") == R3_COMPLETION_SCHEMA, "R3_COMPLETION_SCHEMA_INVALID", "R3 completion schema drift")
    _require(
        completion.get("passed") is result["passed"]
        and completion.get("terminal") == result["terminal"]
        and completion.get("one_shot_consumed") is True,
        "R3_COMPLETION_INVALID",
        "R3 completion receipt does not match its consumed terminal",
    )
    elapsed = completion.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) >= 0.0,
        "R3_COMPLETION_RESOURCE_INVALID",
        "R3 completion elapsed time is invalid",
    )

    _require(manifest.get("schema") == R3_MANIFEST_SCHEMA, "R3_MANIFEST_SCHEMA_INVALID", "R3 manifest schema drift")
    _require(manifest.get("truth_root_consumed") is True, "R3_MANIFEST_UNCONSUMED", "R3 manifest does not mark the truth root consumed")
    files = manifest.get("files")
    _require(isinstance(files, dict) and bool(files), "R3_LEDGER_INVALID", "R3 manifest file ledger is empty or malformed")
    _require(
        manifest.get("file_count_before_manifest") == len(files),
        "R3_LEDGER_COUNT_MISMATCH",
        "R3 manifest file count differs from its ledger",
    )

    ledger_bytes = 0
    normalized_files: dict[str, dict[str, Any]] = {}
    for relative, raw_receipt in sorted(files.items()):
        path = _safe_evidence_file(root, relative)
        _require(isinstance(raw_receipt, dict), "R3_LEDGER_RECEIPT_INVALID", "ledger receipt must be an object", path=relative)
        _require(
            set(raw_receipt) == {"path", "bytes", "sha256"}
            and raw_receipt.get("path") == relative
            and isinstance(raw_receipt.get("bytes"), int)
            and not isinstance(raw_receipt.get("bytes"), bool)
            and raw_receipt["bytes"] >= 0
            and isinstance(raw_receipt.get("sha256"), str)
            and bool(_SHA256.fullmatch(raw_receipt["sha256"])),
            "R3_LEDGER_RECEIPT_INVALID",
            "ledger receipt fields are malformed",
            path=relative,
        )
        observed = _file_receipt(path, relative)
        _require(observed == raw_receipt, "R3_LEDGER_FILE_MISMATCH", "ledger file bytes or hash drift", path=relative)
        normalized_files[relative] = observed
        ledger_bytes += observed["bytes"]

    _require(
        manifest.get("bytes_before_manifest") == ledger_bytes,
        "R3_LEDGER_BYTES_MISMATCH",
        "R3 manifest byte count differs from its ledger",
    )
    actual_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            child = directory_path / name
            _require(not child.is_symlink(), "R3_LEDGER_DIRECTORY_SYMLINK", "R3 evidence contains a symlink directory", path=str(child))
        for name in file_names:
            child = directory_path / name
            _require(not child.is_symlink(), "R3_LEDGER_FILE_SYMLINK", "R3 evidence contains a symlink file", path=str(child))
            actual_files.add(child.relative_to(root).as_posix())
    _require(
        actual_files == set(normalized_files) | {"manifest.json"},
        "R3_LEDGER_FILE_SET_MISMATCH",
        "on-disk R3 files differ from the complete manifest ledger",
        unlisted=sorted(actual_files - set(normalized_files) - {"manifest.json"}),
        missing=sorted(set(normalized_files) - actual_files),
    )

    required_ledger_files = {
        "execution-receipt.json",
        "download-receipts.json.gz",
        "exact-frame-plan.json.gz",
        "uncertainty-model-receipt.json",
        "uncertainty-model-artifact.json.gz",
        "frame-failures.json.gz",
        "result.json",
        "completion-receipt.json",
    }
    _require(required_ledger_files <= set(normalized_files), "R3_LEDGER_REQUIRED_FILE_MISSING", "R3 ledger lacks required truth evidence files")
    _require(
        any(path.startswith("truth-frames/") and path.endswith(".json.gz") for path in normalized_files),
        "R3_TRUTH_FRAME_LEDGER_EMPTY",
        "R3 ledger contains no compact truth-frame commitment",
    )

    result_receipt = _file_receipt(result_path, _display_path(result_path, repo_root))
    completion_receipt = _file_receipt(completion_path, _display_path(completion_path, repo_root))
    manifest_receipt = _file_receipt(manifest_path, _display_path(manifest_path, repo_root))
    return {
        "terminal": result["terminal"],
        "scientific_status": result["scientific_status"],
        "r3_passed": result["passed"],
        "formal_headroom_authorized": result["passed"],
        "descriptive_partial_factor_canary_authorized": True,
        "r3_failure_codes": list(gates.get("failure_codes", [])),
        "result": result_receipt,
        "completion": completion_receipt,
        "manifest": manifest_receipt,
        "ledger_file_count": len(normalized_files),
        "ledger_bytes": ledger_bytes,
        "ledger_canonical_sha256": sha256_bytes(canonical_json_bytes(normalized_files)),
        "manifest_excluded_from_ledger": True,
        "full_file_set_verified": True,
    }


def _default_git_probe(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    _require(head.returncode == 0 and status.returncode == 0, "GIT_PROBE_FAILED", "git identity probe failed", root=str(root))
    return {
        "commit": head.stdout.strip(),
        "clean": not bool(status.stdout.strip()),
        "status": status.stdout.splitlines(),
    }


def _validate_git_identity(
    root: Path,
    expected_commit: str,
    *,
    probe: GitProbe,
    prefix: str,
) -> dict[str, Any]:
    _require(root.is_dir(), f"{prefix}_ROOT_MISSING", "git source root is absent", root=str(root))
    _require(isinstance(expected_commit, str) and bool(_GIT_SHA.fullmatch(expected_commit)), f"{prefix}_COMMIT_INVALID", "expected git commit is malformed")
    observed = probe(root)
    _require(isinstance(observed, Mapping), f"{prefix}_GIT_PROBE_INVALID", "git probe did not return a mapping")
    commit = observed.get("commit")
    clean = observed.get("clean")
    _require(isinstance(commit, str) and bool(_GIT_SHA.fullmatch(commit)), f"{prefix}_COMMIT_INVALID", "observed git commit is malformed")
    _require(commit.lower() == expected_commit.lower(), f"{prefix}_COMMIT_MISMATCH", "git commit differs from its frozen identity", expected=expected_commit, actual=commit)
    _require(clean is True and observed.get("status", []) == [], f"{prefix}_WORKTREE_DIRTY", "git source tree is dirty", status=observed.get("status"))
    return {"root": str(root), "commit": commit.lower(), "clean": True}


def _porcelain_paths(line: str) -> tuple[str, ...]:
    """Extract lexical paths from one porcelain-v1 status row."""

    _require(isinstance(line, str) and len(line) >= 4, "REPOSITORY_GIT_STATUS_INVALID", "git status row is malformed")
    payload = line[3:].strip()
    _require(bool(payload), "REPOSITORY_GIT_STATUS_INVALID", "git status row has no path")
    paths = tuple(part.strip().strip('"').replace("\\", "/") for part in payload.split(" -> "))
    _require(all(path and not path.startswith("/") for path in paths), "REPOSITORY_GIT_STATUS_INVALID", "git status path is malformed")
    return paths


def _path_intersects_protected(path: str, protected_paths: Sequence[str]) -> bool:
    normalized = path.rstrip("/")
    return any(
        normalized == protected.rstrip("/")
        or normalized.startswith(protected.rstrip("/") + "/")
        or protected.rstrip("/").startswith(normalized + "/")
        for protected in protected_paths
    )


def _validate_repository_identity(
    root: Path,
    expected_commit: str,
    *,
    protected_paths: Sequence[str],
    probe: GitProbe,
) -> dict[str, Any]:
    """Require the bound TARO scope clean while preserving unrelated user work."""

    _require(root.is_dir(), "REPOSITORY_ROOT_MISSING", "git source root is absent", root=str(root))
    _require(isinstance(expected_commit, str) and bool(_GIT_SHA.fullmatch(expected_commit)), "REPOSITORY_COMMIT_INVALID", "expected git commit is malformed")
    protected = tuple(sorted({_relative_path(path, "REPOSITORY_PROTECTED_PATH_INVALID") for path in protected_paths}))
    _require(bool(protected), "REPOSITORY_PROTECTED_PATHS_EMPTY", "repository protected path set is empty")
    observed = probe(root)
    _require(isinstance(observed, Mapping), "REPOSITORY_GIT_PROBE_INVALID", "git probe did not return a mapping")
    commit = observed.get("commit")
    status = observed.get("status")
    clean = observed.get("clean")
    _require(isinstance(commit, str) and bool(_GIT_SHA.fullmatch(commit)), "REPOSITORY_COMMIT_INVALID", "observed git commit is malformed")
    _require(commit.lower() == expected_commit.lower(), "REPOSITORY_COMMIT_MISMATCH", "git commit differs from its frozen identity", expected=expected_commit, actual=commit)
    _require(isinstance(status, list) and all(isinstance(row, str) for row in status), "REPOSITORY_GIT_STATUS_INVALID", "git status snapshot is malformed")
    _require(clean is (not bool(status)), "REPOSITORY_GIT_STATUS_INVALID", "git clean flag and status snapshot disagree")
    protected_dirty = [
        row
        for row in status
        if any(_path_intersects_protected(path, protected) for path in _porcelain_paths(row))
    ]
    _require(not protected_dirty, "REPOSITORY_PROTECTED_PATH_DIRTY", "hash-bound TARO paths are dirty", status=protected_dirty)
    return {
        "root": str(root),
        "commit": commit.lower(),
        "clean": clean,
        "protected_paths_clean": True,
        "unrelated_dirty_status": list(status),
    }


def _discover_runtime(repo_root: Path, runtime_root_relative: str, runner_relative: str) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    runtime_relative = _relative_path(runtime_root_relative, "RUNTIME_ROOT_INVALID")
    runner_path_relative = _relative_path(runner_relative, "RUNNER_PATH_INVALID")
    runtime_root = repo_root.joinpath(*PurePosixPath(runtime_relative).parts)
    _require(runtime_root.is_dir(), "RUNTIME_ROOT_MISSING", "factor runtime root is absent", root=str(runtime_root))
    _require(repo_root.joinpath(*PurePosixPath(runner_path_relative).parts).is_file(), "FACTOR_RUNNER_MISSING", "factor one-shot runner is absent", path=runner_path_relative)

    files = sorted(
        path
        for path in runtime_root.iterdir()
        if path.is_file() and (path.suffix == ".py" or path.name == "README.md")
    )
    _require(bool(files), "RUNTIME_BINDINGS_EMPTY", "factor runtime contains no bindable files")
    receipts: list[dict[str, Any]] = []
    roles: set[str] = set()
    test_modules: list[str] = []
    for path in files:
        relative = path.relative_to(repo_root).as_posix()
        role_token = re.sub(r"[^A-Za-z0-9]+", "_", path.name).strip("_").upper()
        role = f"FACTOR_RUNTIME_{role_token}"
        _require(role not in roles, "RUNTIME_ROLE_COLLISION", "runtime filenames collide after role normalization", path=relative)
        roles.add(role)
        receipts.append({"role": role, **_file_receipt(path, relative)})
        if path.name.startswith("test_") and path.suffix == ".py":
            test_modules.append(".".join(PurePosixPath(relative).with_suffix("").parts))
    _require(bool(test_modules), "FOCUSED_TESTS_MISSING", "factor runtime contains no focused tests")
    return receipts, tuple(sorted(test_modules))


def _binding_receipts(repo_root: Path, bindings: Mapping[str, str]) -> list[dict[str, Any]]:
    _require(isinstance(bindings, Mapping) and bool(bindings), "UPSTREAM_BINDINGS_INVALID", "upstream bindings must be a non-empty mapping")
    receipts: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for role, raw_relative in sorted(bindings.items()):
        _require(isinstance(role, str) and bool(role), "UPSTREAM_BINDING_ROLE_INVALID", "upstream binding role is invalid")
        relative = _relative_path(raw_relative, "UPSTREAM_BINDING_PATH_INVALID")
        _require(relative not in observed_paths, "UPSTREAM_BINDING_DUPLICATE", "two upstream roles bind the same path", path=relative)
        observed_paths.add(relative)
        path = repo_root.joinpath(*PurePosixPath(relative).parts)
        receipts.append({"role": role, **_file_receipt(path, relative)})
    return receipts


def _execution_binding_receipts(
    repo_root: Path,
    expected_paths: Mapping[str, str],
    implementation_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the runner's exact role/path binding set, with no alternatives."""

    _require(
        isinstance(expected_paths, Mapping)
        and expected_paths.get("FACTOR_IMPLEMENTATION_LOCK") == implementation_receipt.get("path"),
        "RUNNER_BINDING_CONTRACT_INVALID",
        "runner implementation-lock binding differs from the prepared lock path",
    )
    receipts: list[dict[str, Any]] = []
    for role, raw_relative in sorted(expected_paths.items()):
        relative = _relative_path(raw_relative, "RUNNER_BINDING_PATH_INVALID")
        if role == "FACTOR_IMPLEMENTATION_LOCK":
            receipt = dict(implementation_receipt)
        else:
            path = repo_root.joinpath(*PurePosixPath(relative).parts)
            receipt = _file_receipt(path, relative)
        receipts.append({"role": role, **receipt})
    _require(
        {row["role"]: row["path"] for row in receipts} == dict(expected_paths),
        "RUNNER_BINDING_CONTRACT_INVALID",
        "prepared execution bindings differ from runner EXPECTED_BINDING_PATHS",
    )
    return receipts


def _default_test_runner(
    command: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    process_environment = dict(os.environ)
    process_environment.update(environment)
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = _TEST_COUNT.search(combined)
    tests_run = int(match.group(1)) if match else 0
    runtime_probe = subprocess.run(
        [
            command[0],
            "-c",
            (
                "import json,platform,cv2,numpy as np,torch,psutil,PIL;"
                "print(json.dumps({'python':platform.python_version(),'numpy':np.__version__,"
                "'opencv':cv2.__version__,'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
                "'cuda_device':str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CUDA_UNAVAILABLE',"
                "'psutil':psutil.__version__,'pillow':PIL.__version__},sort_keys=True))"
            ),
        ],
        cwd=cwd,
        env=process_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        observed_runtime = json.loads(runtime_probe.stdout) if runtime_probe.returncode == 0 else None
    except json.JSONDecodeError:
        observed_runtime = None
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "tests_run": tests_run,
        "tests_passed": tests_run if completed.returncode == 0 else 0,
        "failures": 0 if completed.returncode == 0 else 1,
        "errors": 0,
        "network_requests": 0,
        "runtime": observed_runtime,
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }


def _run_and_validate_tests(
    command: tuple[str, ...],
    *,
    repo_root: Path,
    runner: FocusedTestRunner,
    expected_runtime: Mapping[str, str],
) -> dict[str, Any]:
    raw = runner(command, repo_root, REQUIRED_ENVIRONMENT)
    _require(isinstance(raw, Mapping), "FOCUSED_TEST_RECEIPT_INVALID", "focused test runner returned no receipt")
    receipt = dict(raw)
    _require(receipt.get("command") == list(command), "FOCUSED_TEST_COMMAND_DRIFT", "focused test command differs from the complete discovered suite")
    tests_run = receipt.get("tests_run")
    _require(isinstance(tests_run, int) and not isinstance(tests_run, bool) and tests_run > 0, "FOCUSED_TEST_COUNT_INVALID", "focused test count is invalid")
    _require(
        receipt.get("returncode") == 0
        and receipt.get("tests_passed") == tests_run
        and receipt.get("failures") == 0
        and receipt.get("errors") == 0
        and receipt.get("network_requests") == 0,
        "FOCUSED_TESTS_FAILED",
        "factor runtime focused tests did not pass exactly",
        receipt=receipt,
    )
    _require(
        receipt.get("runtime") == dict(expected_runtime),
        "FOCUSED_TEST_RUNTIME_DRIFT",
        "focused tests did not run in the runner's EXPECTED_RUNTIME",
        observed=receipt.get("runtime"),
        expected=dict(expected_runtime),
    )
    for field in ("stdout_sha256", "stderr_sha256"):
        _require(isinstance(receipt.get(field), str) and bool(_SHA256.fullmatch(receipt[field])), "FOCUSED_TEST_RECEIPT_INVALID", "focused test output hash is malformed", field=field)
    return receipt


def _future_lock_receipt(payload: Mapping[str, Any], relative_path: str) -> dict[str, Any]:
    encoded = canonical_lock_bytes(payload)
    return {"path": relative_path, "bytes": len(encoded), "sha256": sha256_bytes(encoded)}


def prepare_factor_headroom_execution(
    *,
    repo_root: Path,
    r3_truth_root: Path,
    factor_root: Path,
    depthart_source_root: Path,
    depthart_source_commit: str,
    checkpoint_path: Path,
    checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    implementation_base_commit: str,
    execution_commit: str,
    python_executable: str = sys.executable,
    date: str = "2026-08-10",
) -> dict[str, Any]:
    """Return hash-bound lock templates after every pre-execution gate passes.

    The returned mapping is in-memory only.  This function performs read-only
    hashing plus the focused test subprocess; it never creates or writes the
    implementation lock, execution lock, or factor evidence root.
    """

    repository = _absolute_lexical(repo_root)
    _require(repository.is_dir(), "REPOSITORY_ROOT_MISSING", "repository root is absent", root=str(repository))
    runner_contract = _load_runner_contract()
    runner_expected_paths = _runner_mapping(runner_contract, "EXPECTED_BINDING_PATHS")
    runner_roots = _runner_mapping(runner_contract, "EXPECTED_ROOTS")
    runner_authority = _runner_mapping(runner_contract, "EXPECTED_AUTHORITY")
    runner_budget = _runner_mapping(runner_contract, "EXPECTED_BUDGET")
    runner_runtime = _runner_mapping(runner_contract, "EXPECTED_RUNTIME")
    structurally_not_applicable_strata = _runner_mapping(
        runner_contract,
        "STRUCTURALLY_NOT_APPLICABLE_STRATA",
    )
    _require(
        isinstance(getattr(runner_contract, "EXECUTION_LOCK_SCHEMA", None), str)
        and bool(runner_contract.EXECUTION_LOCK_SCHEMA)
        and isinstance(getattr(runner_contract, "EXECUTION_LOCK_ID", None), str)
        and bool(runner_contract.EXECUTION_LOCK_ID),
        "RUNNER_CONTRACT_INVALID",
        "runner execution-lock schema/id are missing",
    )
    _require(
        set(runner_roots) == _RUNNER_ROOT_ROLES
        and all(_relative_path(value, "RUNNER_ROOT_CONTRACT_DRIFT") == value for value in runner_roots.values()),
        "RUNNER_ROOT_CONTRACT_DRIFT",
        "runner EXPECTED_ROOTS has missing, extra, or unsafe roles",
        roots=runner_roots,
    )
    _require(
        set(runner_budget) == _RUNNER_BUDGET_FIELDS,
        "RUNNER_BUDGET_INVALID",
        "runner EXPECTED_BUDGET has missing or extra fields",
        budget=runner_budget,
    )
    for field in (
        "wall_seconds",
        "peak_rss_bytes",
        "maximum_evidence_bytes",
        "maximum_cuda_allocated_bytes",
    ):
        _require(
            isinstance(runner_budget.get(field), int)
            and not isinstance(runner_budget.get(field), bool)
            and runner_budget[field] > 0,
            "RUNNER_BUDGET_INVALID",
            "runner EXPECTED_BUDGET has a malformed positive integer",
            field=field,
            value=runner_budget.get(field),
        )
    for field in ("network_requests", "training_steps"):
        _require(
            isinstance(runner_budget.get(field), int)
            and not isinstance(runner_budget.get(field), bool)
            and runner_budget[field] == 0,
            "RUNNER_BUDGET_INVALID",
            "runner EXPECTED_BUDGET must freeze network and training to zero",
            field=field,
            value=runner_budget.get(field),
        )
    _require(
        runner_budget.get("network_requests") == 0
        and runner_budget.get("training_steps") == 0
        and 0 < runner_budget.get("maximum_evidence_bytes", 0) <= 2_147_483_648
        and 0 < runner_budget.get("wall_seconds", 0) <= 28_800
        and 0 < runner_budget.get("peak_rss_bytes", 0) <= 17_179_869_184,
        "RUNNER_BUDGET_EXCEEDS_AUTHORITY",
        "runner EXPECTED_BUDGET exceeds the frozen preparation ceiling",
        budget=runner_budget,
    )
    _require(
        set(runner_authority) == _RUNNER_AUTHORITY_TRUE | _RUNNER_AUTHORITY_FALSE
        and all(runner_authority.get(field) is True for field in _RUNNER_AUTHORITY_TRUE)
        and all(runner_authority.get(field) is False for field in _RUNNER_AUTHORITY_FALSE),
        "RUNNER_AUTHORITY_DRIFT",
        "runner EXPECTED_AUTHORITY differs from the bounded one-shot authority",
    )
    _require(
        set(runner_runtime) == _RUNNER_RUNTIME_FIELDS
        and all(isinstance(value, str) and bool(value) for value in runner_runtime.values()),
        "RUNNER_RUNTIME_INVALID",
        "runner EXPECTED_RUNTIME has missing, extra, or malformed fields",
        runtime=runner_runtime,
    )
    _require(
        _REQUIRED_RUNNER_BINDING_ROLES <= set(runner_expected_paths)
        and all(
            isinstance(role, str)
            and bool(role)
            and _relative_path(path, "RUNNER_BINDING_PATH_INVALID") == path
            for role, path in runner_expected_paths.items()
        ),
        "RUNNER_BINDING_CONTRACT_INVALID",
        "runner EXPECTED_BINDING_PATHS omits a mandatory role or contains an unsafe path",
    )
    _require(
        structurally_not_applicable_strata
        == {
            "orientation": "SOURCE_CONTRACT_FIXES_ALL_REGISTERED_RASTERS_TO_LANDSCAPE_1440X1920_NO_PORTRAIT_LEVEL_EXISTS"
        },
        "RUNNER_STRATUM_POLICY_DRIFT",
        "runner structurally-not-applicable stratum policy drifted",
    )
    runner_path = _relative_path(runner_expected_paths["FACTOR_RUNNER"], "RUNNER_PATH_INVALID")
    runtime_relative = PurePosixPath(runner_path).parent.as_posix()
    implementation_path = _relative_path(
        runner_expected_paths["FACTOR_IMPLEMENTATION_LOCK"],
        "IMPLEMENTATION_LOCK_PATH_INVALID",
    )
    execution_path = _relative_path(DEFAULT_EXECUTION_LOCK_PATH, "EXECUTION_LOCK_PATH_INVALID")
    _require(
        not os.path.lexists(repository.joinpath(*PurePosixPath(implementation_path).parts)),
        "IMPLEMENTATION_LOCK_ALREADY_EXISTS",
        "future implementation lock path already exists",
    )
    _require(
        not os.path.lexists(repository.joinpath(*PurePosixPath(execution_path).parts)),
        "EXECUTION_LOCK_ALREADY_EXISTS",
        "future execution lock path already exists",
    )
    _require(isinstance(date, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)), "LOCK_DATE_INVALID", "lock date must be YYYY-MM-DD")
    _require(isinstance(implementation_base_commit, str) and bool(_GIT_SHA.fullmatch(implementation_base_commit)), "IMPLEMENTATION_COMMIT_INVALID", "implementation base commit is malformed")
    _require(isinstance(execution_commit, str) and bool(_GIT_SHA.fullmatch(execution_commit)), "EXECUTION_COMMIT_INVALID", "execution commit is malformed")
    _require(isinstance(python_executable, str) and bool(python_executable), "PYTHON_EXECUTABLE_INVALID", "focused-test Python executable is missing")

    factor_path = _absolute_lexical(factor_root, base=repository)
    truth_path = _absolute_lexical(r3_truth_root, base=repository)
    _require(
        _display_path(truth_path, repository) == runner_roots.get("TRUTH_EVIDENCE")
        and _display_path(factor_path, repository) == runner_roots.get("FACTOR_EVIDENCE"),
        "RUNNER_ROOT_CONTRACT_DRIFT",
        "requested truth/factor roots differ from runner EXPECTED_ROOTS",
        expected=runner_roots,
    )
    truth_relative = runner_roots["TRUTH_EVIDENCE"]
    expected_r3_paths = {
        "R3_RESULT": f"{truth_relative}/result.json",
        "R3_COMPLETION": f"{truth_relative}/completion-receipt.json",
        "R3_MANIFEST": f"{truth_relative}/manifest.json",
        "R3_EXACT_FRAME_PLAN": f"{truth_relative}/exact-frame-plan.json.gz",
        "R3_DOWNLOAD_RECEIPTS": f"{truth_relative}/download-receipts.json.gz",
        "R3_UNCERTAINTY_RECEIPT": f"{truth_relative}/uncertainty-model-receipt.json",
        "R3_UNCERTAINTY_ARTIFACT": f"{truth_relative}/uncertainty-model-artifact.json.gz",
    }
    _require(
        all(runner_expected_paths.get(role) == path for role, path in expected_r3_paths.items()),
        "RUNNER_R3_BINDING_DRIFT",
        "runner R3 artifact bindings do not resolve inside EXPECTED_ROOTS truth evidence",
        expected=expected_r3_paths,
    )
    source_path = repository.joinpath(*PurePosixPath(runner_roots["SOURCE"]).parts)
    _require(source_path.is_dir(), "R3_SOURCE_ROOT_MISSING", "runner EXPECTED_ROOTS source cache is absent", root=str(source_path))
    _require(factor_path != truth_path, "FACTOR_ROOT_COLLIDES_WITH_TRUTH", "factor root equals the R3 truth root")
    _require(not os.path.lexists(factor_path), "FACTOR_ROOT_ALREADY_EXISTS", "factor-headroom one-shot root already exists", root=str(factor_path))

    repository_protected_paths = tuple(runner_expected_paths.values()) + tuple(DEFAULT_UPSTREAM_BINDINGS.values()) + (
        runtime_relative,
        implementation_path,
        execution_path,
    )
    repository_identity_before_tests = _validate_repository_identity(
        repository,
        implementation_base_commit,
        probe=_default_git_probe,
        protected_paths=repository_protected_paths,
    )

    depthart_source = _absolute_lexical(depthart_source_root, base=repository)
    _require(
        isinstance(depthart_source_commit, str)
        and bool(_GIT_SHA.fullmatch(depthart_source_commit))
        and depthart_source_commit.lower() == EXPECTED_DEPTHART_SOURCE_COMMIT,
        "DEPTHART_SOURCE_COMMIT_CONTRACT_DRIFT",
        "DepthART source commit differs from the factor runner's frozen source identity",
    )
    depthart_identity_before_tests = _validate_git_identity(
        depthart_source,
        depthart_source_commit,
        probe=_default_git_probe,
        prefix="DEPTHART_SOURCE",
    )
    checkpoint = _absolute_lexical(checkpoint_path, base=repository)
    _require(isinstance(checkpoint_sha256, str) and bool(_SHA256.fullmatch(checkpoint_sha256)), "CHECKPOINT_SHA256_INVALID", "expected checkpoint SHA-256 is malformed")
    _require(
        checkpoint_sha256.upper() == runner_contract.adapter.BASELINE_CHECKPOINT_SHA256
        and checkpoint_sha256.upper() == DEFAULT_CHECKPOINT_SHA256,
        "CHECKPOINT_CONTRACT_DRIFT",
        "checkpoint identity differs from the factor runner/source-adapter contract",
    )
    checkpoint_receipt_before_tests = _file_receipt(checkpoint, _display_path(checkpoint, repository))
    _require(checkpoint_receipt_before_tests["sha256"] == checkpoint_sha256.upper(), "CHECKPOINT_SHA256_MISMATCH", "DepthART checkpoint hash differs from the frozen identity")

    runtime_receipts_before_tests, test_modules = _discover_runtime(repository, runtime_relative, runner_path)
    test_command = (python_executable, "-m", "unittest", *test_modules, "-q")
    test_receipt = _run_and_validate_tests(
        test_command,
        repo_root=repository,
        runner=_default_test_runner,
        expected_runtime=runner_runtime,
    )
    _require(
        not os.path.lexists(factor_path),
        "FACTOR_ROOT_CREATED_DURING_TESTS",
        "focused tests created the reserved one-shot factor root",
        root=str(factor_path),
    )
    _require(
        not os.path.lexists(repository.joinpath(*PurePosixPath(implementation_path).parts))
        and not os.path.lexists(repository.joinpath(*PurePosixPath(execution_path).parts)),
        "LOCK_PATH_CREATED_DURING_TESTS",
        "focused tests created a reserved docs lock path",
    )
    repository_identity = _validate_repository_identity(
        repository,
        implementation_base_commit,
        probe=_default_git_probe,
        protected_paths=repository_protected_paths,
    )
    _require(
        repository_identity == repository_identity_before_tests,
        "REPOSITORY_IDENTITY_CHANGED_DURING_TESTS",
        "repository identity changed while focused tests ran",
    )
    depthart_identity = _validate_git_identity(
        depthart_source,
        depthart_source_commit,
        probe=_default_git_probe,
        prefix="DEPTHART_SOURCE",
    )
    _require(
        depthart_identity == depthart_identity_before_tests,
        "DEPTHART_IDENTITY_CHANGED_DURING_TESTS",
        "DepthART source identity changed while focused tests ran",
    )
    checkpoint_receipt = _file_receipt(checkpoint, _display_path(checkpoint, repository))
    _require(
        checkpoint_receipt == checkpoint_receipt_before_tests,
        "CHECKPOINT_CHANGED_DURING_TESTS",
        "DepthART checkpoint changed while focused tests ran",
    )
    runtime_receipts, test_modules_after = _discover_runtime(repository, runtime_relative, runner_path)
    _require(
        runtime_receipts == runtime_receipts_before_tests and test_modules_after == test_modules,
        "FACTOR_RUNTIME_CHANGED_DURING_TESTS",
        "factor runtime file set or hashes changed while focused tests ran",
    )

    r3 = validate_r3_truth_evidence(truth_path, repository)
    upstream_receipts = _binding_receipts(repository, DEFAULT_UPSTREAM_BINDINGS)

    r3_bindings = [
        {"role": "R3_RESULT", **r3["result"]},
        {"role": "R3_COMPLETION", **r3["completion"]},
        {"role": "R3_MANIFEST", **r3["manifest"]},
    ]
    bindings = sorted(
        [*upstream_receipts, *runtime_receipts, *r3_bindings, {"role": "DEPTHART_CHECKPOINT", **checkpoint_receipt}],
        key=lambda row: row["role"],
    )
    binding_roles = [row["role"] for row in bindings]
    binding_paths = [row["path"] for row in bindings]
    _require(len(set(binding_roles)) == len(binding_roles), "BINDING_ROLE_DUPLICATE", "implementation binding roles are not unique")
    _require(len(set(binding_paths)) == len(binding_paths), "BINDING_PATH_DUPLICATE", "implementation binding paths are not unique")

    implementation_lock = {
        "schema": IMPLEMENTATION_LOCK_SCHEMA,
        "lock_id": IMPLEMENTATION_LOCK_ID,
        "date": date,
        "research_mode": "WILD_LAB",
        "status": "IMPLEMENTATION_LOCK_PASS",
        "implementation_base_commit": implementation_base_commit.lower(),
        "repository_identity": repository_identity,
        "bindings": bindings,
        "r3_truth_admission": r3,
        "depthart_identity": {
            "model_id": runner_contract.adapter.BASELINE_MODEL_ID,
            "source": depthart_identity,
            "checkpoint": checkpoint_receipt,
        },
        "focused_validation": test_receipt,
        "runtime_contract": runner_runtime,
        "resource_contract": runner_budget,
        "structurally_not_applicable_strata": structurally_not_applicable_strata,
        "execution_authority": {
            "implementation_lock": True,
            "depthart_inference": False,
            "factorial_execution": False,
            "descriptive_partial_factor_canary": False,
            "training": False,
            "network": False,
            "device": False,
            "product": False,
            "safety": False,
        },
        "unique_successor": runner_contract.EXECUTION_LOCK_ID,
        "claim_ceiling": "Hash-bound landscape-only consumed R3 terminal, DepthART identity and factor-runtime implementation only; NOT_EVALUABLE authorizes descriptive partial-factor canary but never formal headroom, device, product or safety inference.",
    }
    implementation_receipt = _future_lock_receipt(implementation_lock, implementation_path)

    argv = [runner_path, "--execution-lock", execution_path]
    execution_bindings = _execution_binding_receipts(
        repository,
        runner_expected_paths,
        implementation_receipt,
    )
    execution_lock = {
        "schema": runner_contract.EXECUTION_LOCK_SCHEMA,
        "lock_id": runner_contract.EXECUTION_LOCK_ID,
        "date": date,
        "status": "AUTHORIZED_UNCONSUMED",
        "consumed": False,
        "overwrite": False,
        "rerun": False,
        "execution_commit": execution_commit.lower(),
        "implementation_lock": implementation_receipt,
        "bindings": execution_bindings,
        "r3_truth_admission": {
            "terminal": r3["terminal"],
            "formal_headroom_authorized": r3["formal_headroom_authorized"],
            "descriptive_partial_factor_canary_authorized": r3["descriptive_partial_factor_canary_authorized"],
            "r3_failure_codes": r3["r3_failure_codes"],
            "result_sha256": r3["result"]["sha256"],
            "completion_sha256": r3["completion"]["sha256"],
            "manifest_sha256": r3["manifest"]["sha256"],
            "ledger_canonical_sha256": r3["ledger_canonical_sha256"],
            "full_file_set_verified": True,
        },
        "depthart_assets": {
            "source_root": _display_path(depthart_source, repository),
            "source_git_commit": depthart_identity["commit"],
            "checkpoint_path": checkpoint_receipt["path"],
            "checkpoint_bytes": checkpoint_receipt["bytes"],
            "checkpoint_sha256": checkpoint_receipt["sha256"],
            "model_id": runner_contract.adapter.BASELINE_MODEL_ID,
        },
        "roots": runner_roots,
        "factor_root_must_be_absent_at_start": True,
        "one_shot_consumption_event": "FACTOR_EVIDENCE_ROOT_CREATION",
        "required_environment": dict(REQUIRED_ENVIRONMENT),
        "runtime": runner_runtime,
        "resource_budget": runner_budget,
        "argv": argv,
        "argv_alternatives": [],
        "execution_authority": runner_authority,
        "structurally_not_applicable_strata": structurally_not_applicable_strata,
        "claim_ceiling": "One exact landscape-only ARKitScenes R3 WILD_LAB DepthART execution; formal headroom only after R3 PASS, otherwise post-hoc descriptive partial-factor canary only; no portrait, training, wearable, active-observation, device, product or safety authority.",
    }
    execution_receipt = _future_lock_receipt(execution_lock, execution_path)
    return {
        "schema": "blindassist.taro.o0r.factor_headroom_lock_preparation.v1",
        "status": "READY_FOR_SEPARATE_LOCK_COMMIT_NO_EXECUTION",
        "writes_performed": 0,
        "factor_root_created": False,
        "implementation_lock": implementation_lock,
        "implementation_lock_receipt": implementation_receipt,
        "execution_lock": execution_lock,
        "execution_lock_receipt": execution_receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--r3-truth-root", type=Path, required=True)
    parser.add_argument("--factor-root", type=Path, required=True)
    parser.add_argument("--depthart-source-root", type=Path, required=True)
    parser.add_argument("--depthart-source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--implementation-base-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()
    prepared = prepare_factor_headroom_execution(
        repo_root=args.repo_root,
        r3_truth_root=args.r3_truth_root,
        factor_root=args.factor_root,
        depthart_source_root=args.depthart_source_root,
        depthart_source_commit=args.depthart_source_commit,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        implementation_base_commit=args.implementation_base_commit,
        execution_commit=args.execution_commit,
        python_executable=args.python_executable,
    )
    sys.stdout.buffer.write(canonical_json_bytes(prepared) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHECKPOINT_SHA256",
    "DEFAULT_EXECUTION_LOCK_PATH",
    "DEFAULT_UPSTREAM_BINDINGS",
    "EXPECTED_DEPTHART_SOURCE_COMMIT",
    "IMPLEMENTATION_LOCK_ID",
    "IMPLEMENTATION_LOCK_SCHEMA",
    "PreparationError",
    "REQUIRED_ENVIRONMENT",
    "canonical_json_bytes",
    "canonical_lock_bytes",
    "prepare_factor_headroom_execution",
    "sha256_bytes",
    "sha256_file",
    "validate_r3_truth_evidence",
]
