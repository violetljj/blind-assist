#!/usr/bin/env python3
"""Lock the exact-commit TartanGround differential-drive catalog.

This stage is intentionally catalog-only.  It fetches one frozen toolkit
commit, reads two Git objects from that commit, and never contacts the dataset
host or opens a dataset ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_"
    "execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_D5_S0_DESIGN_BEFORE_EXACT_COMMIT_FETCH_OR_CATALOG_READ"
)
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_attempt"
)
PREFLIGHT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_preflight"
)
CATALOG_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_diff_catalog"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_result"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a_tartanground_catalog_failure"
)

CATALOG_LOCKED = (
    "D5_S0A_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_"
    "S0B_STRUCTURAL_AUTHORITY"
)
CATALOG_INSUFFICIENT = (
    "D5_S0A_TARTANGROUND_DIFF_CATALOG_CAPACITY_INSUFFICIENT_STOP"
)
CATALOG_INVALID = "D5_S0A_TARTANGROUND_DIFF_CATALOG_INVALID_STOP"

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0a-tartanground-catalog-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0A_TARTANGROUND_CATALOG_EXECUTION_CONTRACT_"
    "2026-08-02.json"
)
PLANNER_RELATIVE_PATH = Path(
    "scripts/research/hftf/"
    "plan_stage_c_d5_s0a_tartanground_catalog.py"
)
TEST_RELATIVE_PATH = Path(
    "scripts/research/hftf/"
    "test_plan_stage_c_d5_s0a_tartanground_catalog.py"
)

FILENAMES = {
    "attempt": "attempt.json",
    "preflight": "preflight.json",
    "catalog": "catalog.json",
    "result": "result.json",
    "failure": "failure.json",
}
TOOLKIT_DIRNAME = "toolkit"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
TARGET_PATH_RE = re.compile(
    r"^(?P<environment>[^/]+)/Data_diff/"
    r"(?P<trajectory>P1[0-9]{3})/(?P<archive>[^/]+\.zip)$"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_bound(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return candidate.resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(resolved)


def test_definition_count(path: Path) -> int:
    return len(re.findall(r"^\s+def test_", path.read_text(), re.MULTILINE))


def git_local(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd or repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_tracked_clean(path: Path, label: str) -> None:
    relative = path.resolve().relative_to(repo_root()).as_posix()
    if not git_local("ls-files", "--error-unmatch", "--", relative):
        raise ValueError(f"{label} is not tracked")
    if git_local("status", "--porcelain", "--", relative):
        raise ValueError(f"{label} is not clean")


def write_bytes_exclusive_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        raise OSError(f"Durable exact-byte reopen verification failed: {path}")


def write_json_exclusive_fsync(path: Path, value: Any) -> None:
    write_bytes_exclusive_fsync(path, canonical_json_bytes(value))


def require_canonical_root(path: Path) -> Path:
    expected = (repo_root() / CANONICAL_ROOT).resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"Noncanonical output root: {actual}")
    return actual


def artifact_state(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir()}


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected D5-S0A contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Unexpected D5-S0A contract status")

    design_binding = contract["parents"]["d5_s0_design"]
    design_path = resolve_bound(str(design_binding["path"]))
    if sha256(design_path) != str(design_binding["sha256"]):
        raise ValueError("D5-S0 design hash drift")
    design = load_json(design_path)
    if design.get("status") != design_binding["required_status"]:
        raise ValueError("D5-S0 design status drift")

    implementation_paths: list[tuple[Path, str]] = []
    for label, binding in contract["implementations"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"Implementation hash drift: {label}")
        implementation_paths.append((path, label))

    test_binding = contract["implementation_tests"]["planner_test"]
    test_path = resolve_bound(str(test_binding["path"]))
    if sha256(test_path) != str(test_binding["sha256"]):
        raise ValueError("Planner test hash drift")
    expected_tests = int(contract["implementation_tests"]["test_count"])
    if test_definition_count(test_path) != expected_tests:
        raise ValueError("Planner test count drift")

    source = contract["source_lock"]
    commit = str(source["toolkit_commit"])
    if not HEX40_RE.fullmatch(commit):
        raise ValueError("Toolkit commit is not exact lowercase SHA-1")
    if source["manifest_path"] != "tartanair/download_ground_files.txt":
        raise ValueError("Unexpected manifest path")
    if source["gitmodules_path"] != ".gitmodules":
        raise ValueError("Unexpected .gitmodules path")
    if contract["network"]["git_fetch_attempts"] != 1:
        raise ValueError("S0A must use exactly one Git fetch attempt")
    if not contract["authorization"]["exact_commit_fetch_authorized"]:
        raise ValueError("Exact-commit fetch is not authorized")
    if contract["authorization"]["dataset_host_request_authorized"]:
        raise ValueError("Dataset host request must remain forbidden")

    if verify_git:
        if git_local("rev-parse", "HEAD") != git_local(
            "rev-parse", "origin/master"
        ):
            raise ValueError("HEAD differs from origin/master")
        for path, label in [
            (contract_path, "execution contract"),
            (design_path, "D5-S0 design"),
            *implementation_paths,
            (test_path, "planner test"),
        ]:
            require_tracked_clean(path, label)

    return {
        "contract": contract,
        "contract_path": contract_path,
        "design_path": design_path,
        "implementation_paths": implementation_paths,
        "test_path": test_path,
    }


def parse_gitmodules(value: bytes) -> dict[str, str]:
    text = value.decode("utf-8")
    section_re = re.compile(r'^\[submodule "([^"]+)"\]\s*$', re.MULTILINE)
    matches = list(section_re.finditer(text))
    if not matches:
        raise ValueError(".gitmodules contains no submodule sections")
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        path_match = re.search(r"^\s*path\s*=\s*(\S+)\s*$", body, re.MULTILINE)
        url_match = re.search(r"^\s*url\s*=\s*(\S+)\s*$", body, re.MULTILINE)
        if not path_match or not url_match:
            raise ValueError(f"Incomplete submodule section: {match.group(1)}")
        path = path_match.group(1)
        if path in result:
            raise ValueError(f"Duplicate submodule path: {path}")
        result[path] = url_match.group(1)
    return result


def parse_ls_tree_gitlink(text: str, expected_path: str) -> str:
    match = re.fullmatch(
        r"160000 commit ([0-9a-f]{40})\t" + re.escape(expected_path),
        text.strip(),
    )
    if not match:
        raise ValueError(f"Unexpected gitlink tree entry: {expected_path}")
    return match.group(1)


def parse_manifest(
    value: bytes,
    *,
    required_archives: Sequence[str],
) -> dict[str, Any]:
    text = value.decode("utf-8")
    lines = text.splitlines()
    nonempty_line_count = sum(bool(line.strip()) for line in lines)
    if not nonempty_line_count:
        raise ValueError("Empty download_ground_files manifest")

    seen_paths: set[str] = set()
    parents: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    target_line_count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed manifest row {line_number}")
        path_text = fields[0]
        declared_size = " ".join(fields[1:])
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)? G", declared_size):
            raise ValueError(
                f"Unexpected declared size format at row {line_number}"
            )
        try:
            if Decimal(declared_size.removesuffix(" G")) <= 0:
                raise ValueError(
                    f"Nonpositive declared size at row {line_number}"
                )
        except InvalidOperation as error:
            raise ValueError(
                f"Invalid declared size at row {line_number}"
            ) from error
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe manifest path at row {line_number}")
        if path_text in seen_paths:
            raise ValueError(f"Duplicate manifest path: {path_text}")
        seen_paths.add(path_text)
        match = TARGET_PATH_RE.fullmatch(path_text)
        if not match:
            continue
        target_line_count += 1
        key = (match.group("environment"), match.group("trajectory"))
        archives = parents.setdefault(key, {})
        archive = match.group("archive")
        if archive in archives:
            raise ValueError(f"Duplicate archive for parent: {path_text}")
        archives[archive] = {
            "path": path_text,
            "declared_size": declared_size,
        }

    required = tuple(required_archives)
    rows: list[dict[str, Any]] = []
    for (environment, trajectory), archives in sorted(parents.items()):
        missing = sorted(set(required) - set(archives))
        rows.append(
            {
                "parent_id": f"{environment}/Data_diff/{trajectory}",
                "environment": environment,
                "robot_version": "diff",
                "trajectory_id": trajectory,
                "archives": {
                    name: archives[name] for name in sorted(archives)
                },
                "required_catalog_archives_present": not missing,
                "missing_required_catalog_archives": missing,
            }
        )
    eligible = [
        row for row in rows if row["required_catalog_archives_present"]
    ]
    environments = sorted({str(row["environment"]) for row in eligible})
    return {
        "manifest_nonempty_line_count": nonempty_line_count,
        "manifest_unique_path_count": len(seen_paths),
        "target_diff_archive_line_count": target_line_count,
        "target_diff_parent_count": len(rows),
        "required_catalog_complete_parent_count": len(eligible),
        "required_catalog_complete_environment_count": len(environments),
        "required_catalog_complete_environments": environments,
        "parents": rows,
    }


GitRunner = Callable[[Sequence[str], Path], bytes]


def subprocess_git_runner(args: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def git_read_object(
    runner: GitRunner,
    toolkit: Path,
    commit: str,
    path: str,
) -> bytes:
    return runner(["cat-file", "blob", f"{commit}:{path}"], toolkit)


def fetch_and_read_catalog(
    contract: dict[str, Any],
    root: Path,
    *,
    git_runner: GitRunner,
) -> dict[str, Any]:
    source = contract["source_lock"]
    toolkit = root / TOOLKIT_DIRNAME
    toolkit.mkdir()
    git_runner(["init", "."], toolkit)
    git_runner(
        ["remote", "add", "origin", str(source["toolkit_repository"])],
        toolkit,
    )
    commit = str(source["toolkit_commit"])
    git_runner(
        [
            "-c",
            "protocol.version=2",
            "fetch",
            "--no-tags",
            "--depth=1",
            "--recurse-submodules=no",
            "origin",
            commit,
        ],
        toolkit,
    )
    fetched = git_runner(["rev-parse", "FETCH_HEAD"], toolkit).decode().strip()
    if fetched != commit:
        raise ValueError("Fetched commit differs from frozen toolkit commit")

    gitmodules_bytes = git_read_object(
        git_runner, toolkit, commit, str(source["gitmodules_path"])
    )
    observed_modules = parse_gitmodules(gitmodules_bytes)
    expected_modules = {
        str(row["path"]): str(row["url"])
        for row in source["submodule_gitlinks"]
    }
    if observed_modules != expected_modules:
        raise ValueError(".gitmodules path or URL drift")
    observed_gitlinks: dict[str, str] = {}
    for row in source["submodule_gitlinks"]:
        path = str(row["path"])
        tree = git_runner(["ls-tree", commit, "--", path], toolkit)
        observed_gitlinks[path] = parse_ls_tree_gitlink(
            tree.decode("utf-8"), path
        )
        if observed_gitlinks[path] != str(row["commit"]):
            raise ValueError(f"Submodule gitlink commit drift: {path}")

    if (toolkit / ".git" / "modules").exists():
        raise ValueError("Submodule checkout or initialization observed")
    manifest_bytes = git_read_object(
        git_runner, toolkit, commit, str(source["manifest_path"])
    )
    parsed = parse_manifest(
        manifest_bytes,
        required_archives=contract["catalog_gate"]["required_archives"],
    )
    return {
        "toolkit_commit": commit,
        "manifest_bytes": len(manifest_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "gitmodules_bytes": len(gitmodules_bytes),
        "gitmodules_sha256": sha256_bytes(gitmodules_bytes),
        "verified_gitlinks": observed_gitlinks,
        **parsed,
    }


def frozen_source_identity(contract: dict[str, Any]) -> dict[str, str]:
    source = contract["source_lock"]
    return {
        "toolkit_repository": str(source["toolkit_repository"]),
        "toolkit_commit": str(source["toolkit_commit"]),
        "manifest_path": str(source["manifest_path"]),
        "gitmodules_path": str(source["gitmodules_path"]),
    }


def validate_existing_terminal(
    root: Path,
    names: set[str],
    *,
    contract_path: Path | None = None,
) -> bool:
    contract_path = (
        contract_path.resolve()
        if contract_path is not None
        else (repo_root() / CONTRACT_RELATIVE_PATH).resolve()
    )
    try:
        contract = load_json(contract_path)
        if (
            contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("status") != CONTRACT_STATUS
        ):
            return False
        expected_identity = frozen_source_identity(contract)
        contract_sha256 = sha256(contract_path)
    except (KeyError, OSError, ValueError):
        return False

    locked_names = {
        TOOLKIT_DIRNAME,
        FILENAMES["attempt"],
        FILENAMES["preflight"],
        FILENAMES["catalog"],
        FILENAMES["result"],
    }
    if names == locked_names:
        try:
            attempt = load_json(root / FILENAMES["attempt"])
            if (
                attempt.get("schema") != ATTEMPT_SCHEMA
                or attempt.get("status")
                != "ATTEMPT_FSYNCED_BEFORE_FIRST_GIT_NETWORK_REQUEST"
                or attempt.get("execution_contract_sha256")
                != contract_sha256
                or attempt.get("toolkit_repository")
                != expected_identity["toolkit_repository"]
                or attempt.get("toolkit_commit")
                != expected_identity["toolkit_commit"]
            ):
                return False
            preflight = load_json(root / FILENAMES["preflight"])
            if (
                preflight.get("schema") != PREFLIGHT_SCHEMA
                or preflight.get("status")
                != (
                    "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_GIT_"
                    "NETWORK_REQUEST"
                )
                or preflight.get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
            ):
                return False
            catalog = load_json(root / FILENAMES["catalog"])
            if (
                catalog.get("schema") != CATALOG_SCHEMA
                or catalog.get("status")
                != "EXACT_COMMIT_CATALOG_INVENTORY_COMPLETE"
                or catalog.get("source_identity") != expected_identity
                or catalog.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or catalog.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
            ):
                return False
            result = load_json(root / FILENAMES["result"])
            if result.get("schema") != RESULT_SCHEMA:
                return False
            if result.get("terminal") not in {
                CATALOG_LOCKED,
                CATALOG_INSUFFICIENT,
            }:
                return False
            if result.get("source_identity") != expected_identity:
                return False
            bindings = result["bindings"]
            for key in ("attempt", "preflight", "catalog"):
                if sha256(root / FILENAMES[key]) != bindings[f"{key}_sha256"]:
                    return False
            fetched = git_local(
                "rev-parse", "FETCH_HEAD", cwd=root / TOOLKIT_DIRNAME
            )
            return fetched == expected_identity["toolkit_commit"]
        except (KeyError, OSError, ValueError, subprocess.SubprocessError):
            return False
    if FILENAMES["failure"] in names and FILENAMES["result"] not in names:
        try:
            failure = load_json(root / FILENAMES["failure"])
            if (
                failure.get("schema") != FAILURE_SCHEMA
                or failure.get("terminal") != CATALOG_INVALID
                or failure.get("resume_or_rerun_authorized") is not False
                or set(failure.get("observed_top_level_names", []))
                != names - {FILENAMES["failure"]}
            ):
                return False
            for key in ("attempt", "preflight"):
                path = root / FILENAMES[key]
                recorded = failure.get(f"{key}_sha256")
                if path.is_file():
                    if recorded != sha256(path):
                        return False
                elif recorded is not None:
                    return False
            attempt_path = root / FILENAMES["attempt"]
            if attempt_path.is_file():
                attempt = load_json(attempt_path)
                if (
                    attempt.get("schema") != ATTEMPT_SCHEMA
                    or attempt.get("execution_contract_sha256")
                    != contract_sha256
                    or attempt.get("toolkit_repository")
                    != expected_identity["toolkit_repository"]
                    or attempt.get("toolkit_commit")
                    != expected_identity["toolkit_commit"]
                ):
                    return False
            return True
        except (OSError, ValueError):
            return False
    return False


def freeze_existing_partial(root: Path, names: set[str]) -> int:
    if FILENAMES["failure"] in names:
        raise ValueError("Existing failure is corrupt or ambiguous")
    write_json_exclusive_fsync(
        root / FILENAMES["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": CATALOG_INVALID,
            "reason": "existing_partial_or_unknown_canonical_root",
            "observed_top_level_names": sorted(names),
            "attempt_sha256": (
                sha256(root / FILENAMES["attempt"])
                if (root / FILENAMES["attempt"]).is_file()
                else None
            ),
            "preflight_sha256": (
                sha256(root / FILENAMES["preflight"])
                if (root / FILENAMES["preflight"]).is_file()
                else None
            ),
            "resume_or_rerun_authorized": False,
            "dataset_host_request_made": False,
            "dataset_zip_opened": False,
        },
    )
    return 2


def execute(
    contract_path: Path,
    root: Path,
    *,
    git_runner: GitRunner = subprocess_git_runner,
    verify_git: bool = True,
) -> dict[str, Any]:
    context = validate_contract(contract_path, verify_git=verify_git)
    contract = context["contract"]
    root.mkdir(parents=True, exist_ok=False)

    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_GIT_NETWORK_REQUEST",
        "execution_contract_path": display_path(context["contract_path"]),
        "execution_contract_sha256": sha256(context["contract_path"]),
        "toolkit_repository": contract["source_lock"]["toolkit_repository"],
        "toolkit_commit": contract["source_lock"]["toolkit_commit"],
        "git_fetch_attempts_authorized": 1,
        "dataset_host_request_authorized": False,
        "dataset_zip_open_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["attempt"], attempt)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_GIT_NETWORK_REQUEST",
        "attempt_sha256": sha256(root / FILENAMES["attempt"]),
        "head": git_local("rev-parse", "HEAD") if verify_git else None,
        "origin_master": (
            git_local("rev-parse", "origin/master") if verify_git else None
        ),
        "design_sha256": sha256(context["design_path"]),
        "implementation_sha256": {
            label: sha256(path)
            for path, label in context["implementation_paths"]
        },
        "test_sha256": sha256(context["test_path"]),
        "canonical_output_root": display_path(root),
        "dataset_host_request_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["preflight"], preflight)

    observed = fetch_and_read_catalog(
        contract, root, git_runner=git_runner
    )
    catalog = {
        "schema": CATALOG_SCHEMA,
        "status": "EXACT_COMMIT_CATALOG_INVENTORY_COMPLETE",
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
        },
        "source_identity": {
            "toolkit_repository": contract["source_lock"][
                "toolkit_repository"
            ],
            "toolkit_commit": observed.pop("toolkit_commit"),
            "manifest_path": contract["source_lock"]["manifest_path"],
            "gitmodules_path": contract["source_lock"]["gitmodules_path"],
        },
        "catalog_observation": observed,
        "dataset_host_request_made": False,
        "dataset_zip_opened": False,
        "member_payload_read": False,
        "pose_value_read_or_retained": False,
        "structural_authority_evaluated": False,
        "source_feasibility_decided": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["catalog"], catalog)

    gate = contract["catalog_gate"]
    count = int(
        catalog["catalog_observation"][
            "required_catalog_complete_parent_count"
        ]
    )
    environments = int(
        catalog["catalog_observation"][
            "required_catalog_complete_environment_count"
        ]
    )
    passes = (
        count >= int(gate["minimum_distinct_diff_trajectory_parents"])
        and environments >= int(gate["minimum_distinct_environments"])
    )
    terminal = CATALOG_LOCKED if passes else CATALOG_INSUFFICIENT
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "catalog_sha256": sha256(root / FILENAMES["catalog"]),
        },
        "source_identity": catalog["source_identity"],
        "catalog_gate": {
            "minimum_distinct_diff_trajectory_parents": gate[
                "minimum_distinct_diff_trajectory_parents"
            ],
            "observed_required_catalog_complete_parent_count": count,
            "minimum_distinct_environments": gate[
                "minimum_distinct_environments"
            ],
            "observed_required_catalog_complete_environment_count": environments,
            "passes_catalog_capacity_only": passes,
        },
        "structural_authority_evaluated": False,
        "d5_s0_source_feasibility_terminal_reached": False,
        "s0b_execution_authorized_automatically": False,
        "dataset_payload_authorized": False,
        "ecology_or_effect_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "production_or_safety_claim_authorized": False,
        "next_authority": (
            "freeze_separate_hash_bound_d5_s0b_structural_authority_"
            "execution_contract"
            if passes
            else "stop_d5_tartanground_diff_route_for_catalog_capacity"
        ),
    }
    write_json_exclusive_fsync(root / FILENAMES["result"], result)
    return result


def execute_with_failure_closure(
    contract_path: Path,
    root: Path,
    *,
    git_runner: GitRunner = subprocess_git_runner,
    verify_git: bool = True,
) -> dict[str, Any]:
    try:
        return execute(
            contract_path,
            root,
            git_runner=git_runner,
            verify_git=verify_git,
        )
    except BaseException as error:
        if root.exists() and FILENAMES["failure"] not in artifact_state(root):
            try:
                write_json_exclusive_fsync(
                    root / FILENAMES["failure"],
                    {
                        "schema": FAILURE_SCHEMA,
                        "terminal": CATALOG_INVALID,
                        "reason": f"{type(error).__name__}: {error}",
                        "observed_top_level_names": sorted(
                            artifact_state(root)
                        ),
                        "attempt_sha256": (
                            sha256(root / FILENAMES["attempt"])
                            if (root / FILENAMES["attempt"]).is_file()
                            else None
                        ),
                        "preflight_sha256": (
                            sha256(root / FILENAMES["preflight"])
                            if (root / FILENAMES["preflight"]).is_file()
                            else None
                        ),
                        "resume_or_rerun_authorized": False,
                        "dataset_host_request_made": False,
                        "dataset_zip_opened": False,
                    },
                )
            except BaseException:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-contract",
        type=Path,
        default=repo_root() / CONTRACT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / CANONICAL_ROOT,
    )
    args = parser.parse_args()
    root = require_canonical_root(args.output_root)
    names = artifact_state(root)
    if names:
        if validate_existing_terminal(root, names):
            raise ValueError("D5-S0A validated terminal already exists")
        return freeze_existing_partial(root, names)
    result = execute_with_failure_closure(
        args.execution_contract.resolve(), root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
