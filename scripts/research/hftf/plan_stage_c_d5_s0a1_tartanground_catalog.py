#!/usr/bin/env python3
"""Lock the TartanGround catalog using path-token-only parsing.

S0A.1 is a new control-plane protocol.  It does not read or reuse the failed
S0A root.  Manifest suffix tokens are discarded and can never affect a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0a_tartanground_catalog import (  # noqa: E402
    git_local,
    load_json,
    parse_gitmodules,
    parse_ls_tree_gitlink,
    require_tracked_clean,
    resolve_bound,
    sha256,
    test_definition_count,
    write_json_exclusive_fsync,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_"
    "execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_S0A1_DESIGN_BEFORE_NEW_EXACT_COMMIT_FETCH_OR_"
    "MANIFEST_READ"
)
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_attempt"
)
PREFLIGHT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_preflight"
)
CATALOG_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_diff_catalog"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_result"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0a1_tartanground_catalog_failure"
)

CATALOG_LOCKED = (
    "D5_S0A1_TARTANGROUND_DIFF_CATALOG_LOCKED_REQUIRES_"
    "S0B_STRUCTURAL_AUTHORITY"
)
CATALOG_INSUFFICIENT = (
    "D5_S0A1_TARTANGROUND_DIFF_CATALOG_CAPACITY_INSUFFICIENT_STOP"
)
CATALOG_INVALID = "D5_S0A1_TARTANGROUND_DIFF_CATALOG_INVALID_STOP"

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0a1-tartanground-catalog-20260802"
)
FAILED_S0A_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0a-tartanground-catalog-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0A1_TARTANGROUND_CATALOG_EXECUTION_CONTRACT_"
    "2026-08-02.json"
)
PLANNER_RELATIVE_PATH = Path(
    "scripts/research/hftf/"
    "plan_stage_c_d5_s0a1_tartanground_catalog.py"
)
TEST_RELATIVE_PATH = Path(
    "scripts/research/hftf/"
    "test_plan_stage_c_d5_s0a1_tartanground_catalog.py"
)
TOOLKIT_DIRNAME = "toolkit"
FILENAMES = {
    "attempt": "attempt.json",
    "preflight": "preflight.json",
    "catalog": "catalog.json",
    "result": "result.json",
    "failure": "failure.json",
}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
TARGET_PATH_RE = re.compile(
    r"^(?P<environment>[^/]+)/Data_diff/"
    r"(?P<trajectory>P1[0-9]{3})/(?P<archive>[^/]+\.zip)$"
)
ASCII_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(resolved)


def require_canonical_root(path: Path) -> Path:
    expected = (repo_root() / CANONICAL_ROOT).resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(f"Noncanonical S0A.1 output root: {actual}")
    if actual == (repo_root() / FAILED_S0A_ROOT).resolve():
        raise ValueError("S0A.1 may not use the failed S0A root")
    return actual


def artifact_state(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir()}


def local_fetch_head_or_none(root: Path) -> str | None:
    toolkit = root / TOOLKIT_DIRNAME
    if not toolkit.is_dir():
        return None
    try:
        value = git_local("rev-parse", "FETCH_HEAD", cwd=toolkit)
    except (OSError, subprocess.SubprocessError):
        return None
    return value if HEX40_RE.fullmatch(value) else None


def frozen_source_identity(contract: dict[str, Any]) -> dict[str, str]:
    source = contract["source_lock"]
    return {
        "toolkit_repository": str(source["toolkit_repository"]),
        "toolkit_commit": str(source["toolkit_commit"]),
        "manifest_path": str(source["manifest_path"]),
        "gitmodules_path": str(source["gitmodules_path"]),
    }


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected D5-S0A.1 contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Unexpected D5-S0A.1 contract status")

    parent_paths: list[tuple[Path, str]] = []
    for label, binding in contract["parents"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"Parent hash drift: {label}")
        value = load_json(path)
        required_terminal = binding.get("required_terminal")
        required_status = binding.get("required_status")
        if required_terminal and value.get("terminal") != required_terminal:
            raise ValueError(f"Parent terminal drift: {label}")
        if required_status and value.get("status") != required_status:
            raise ValueError(f"Parent status drift: {label}")
        parent_paths.append((path, label))

    implementation_paths: list[tuple[Path, str]] = []
    for label, binding in contract["implementations"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"Implementation hash drift: {label}")
        implementation_paths.append((path, label))

    test_binding = contract["implementation_tests"]["planner_test"]
    test_path = resolve_bound(str(test_binding["path"]))
    if sha256(test_path) != str(test_binding["sha256"]):
        raise ValueError("S0A.1 planner test hash drift")
    if test_definition_count(test_path) != int(
        contract["implementation_tests"]["test_count"]
    ):
        raise ValueError("S0A.1 planner test count drift")

    source = contract["source_lock"]
    if not HEX40_RE.fullmatch(str(source["toolkit_commit"])):
        raise ValueError("Toolkit commit is not exact lowercase SHA-1")
    if source["manifest_path"] != "tartanair/download_ground_files.txt":
        raise ValueError("Unexpected manifest path")
    if source["gitmodules_path"] != ".gitmodules":
        raise ValueError("Unexpected .gitmodules path")
    if contract["network"]["git_fetch_attempts"] != 1:
        raise ValueError("S0A.1 must use one new Git fetch")
    parser = contract["manifest_parser"]
    if (
        not parser["first_token_only_is_path_identity"]
        or not parser["all_suffix_tokens_discarded"]
        or parser["suffix_retained_or_used"]
    ):
        raise ValueError("Opaque suffix parser boundary drift")
    if contract["authorization"]["read_failed_s0a_root_authorized"]:
        raise ValueError("Failed S0A root read must remain forbidden")
    if contract["authorization"]["dataset_host_request_authorized"]:
        raise ValueError("Dataset host request must remain forbidden")

    if verify_git:
        if git_local("rev-parse", "HEAD") != git_local(
            "rev-parse", "origin/master"
        ):
            raise ValueError("HEAD differs from origin/master")
        for path, label in [
            (contract_path, "S0A.1 execution contract"),
            *parent_paths,
            *implementation_paths,
            (test_path, "S0A.1 planner test"),
        ]:
            require_tracked_clean(path, label)

    return {
        "contract": contract,
        "contract_path": contract_path,
        "parent_paths": parent_paths,
        "implementation_paths": implementation_paths,
        "test_path": test_path,
    }


def parse_manifest_path_tokens(
    value: bytes,
    *,
    required_archives: Sequence[str],
) -> dict[str, Any]:
    text = value.decode("utf-8")
    raw_lines = text.split("\n")
    lines = [
        line[:-1] if line.endswith("\r") else line for line in raw_lines
    ]
    nonempty = [
        (index, line)
        for index, line in enumerate(lines, 1)
        if line.strip(" \t\f\v")
    ]
    if not nonempty:
        raise ValueError("Blank-only download_ground_files manifest")

    seen_paths: set[str] = set()
    parents: dict[tuple[str, str], dict[str, str]] = {}
    target_line_count = 0
    for line_number, line in nonempty:
        stripped = line.strip(" \t\f\v")
        fields = ASCII_HORIZONTAL_WHITESPACE_RE.split(stripped, maxsplit=1)
        path_text = fields[0]
        if not path_text:
            raise ValueError(f"Missing path token at row {line_number}")
        if "\\" in path_text or "\x00" in path_text:
            raise ValueError(f"Unsafe path token at row {line_number}")
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path token at row {line_number}")
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
        archives[archive] = path_text

    required = set(required_archives)
    rows: list[dict[str, Any]] = []
    for (environment, trajectory), archives in sorted(parents.items()):
        missing = sorted(required - set(archives))
        rows.append(
            {
                "parent_id": f"{environment}/Data_diff/{trajectory}",
                "environment": environment,
                "robot_version": "diff",
                "trajectory_id": trajectory,
                "archive_paths": {
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
        "manifest_nonempty_line_count": len(nonempty),
        "manifest_unique_path_token_count": len(seen_paths),
        "target_diff_archive_path_count": target_line_count,
        "target_diff_parent_count": len(rows),
        "required_catalog_complete_parent_count": len(eligible),
        "required_catalog_complete_environment_count": len(environments),
        "required_catalog_complete_environments": environments,
        "suffix_tokens_read_validated_retained_or_used": False,
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

    gitmodules_bytes = git_runner(
        ["cat-file", "blob", f"{commit}:{source['gitmodules_path']}"],
        toolkit,
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

    manifest_bytes = git_runner(
        ["cat-file", "blob", f"{commit}:{source['manifest_path']}"],
        toolkit,
    )
    parsed = parse_manifest_path_tokens(
        manifest_bytes,
        required_archives=contract["catalog_gate"]["required_archives"],
    )
    return {
        "toolkit_commit": commit,
        "gitmodules_bytes": len(gitmodules_bytes),
        "gitmodules_sha256": hashlib.sha256(gitmodules_bytes).hexdigest(),
        "verified_gitlinks": observed_gitlinks,
        **parsed,
    }


def validate_catalog_result_semantics(
    contract: dict[str, Any],
    catalog: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    observation = catalog.get("catalog_observation")
    if not isinstance(observation, dict):
        return False
    rows = observation.get("parents")
    if not isinstance(rows, list):
        return False
    eligible = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("required_catalog_archives_present") is True
    ]
    if len({str(row.get("parent_id")) for row in rows}) != len(rows):
        return False
    parent_count = len(eligible)
    environment_count = len(
        {str(row.get("environment")) for row in eligible}
    )
    if (
        observation.get("target_diff_parent_count") != len(rows)
        or observation.get("required_catalog_complete_parent_count")
        != parent_count
        or observation.get("required_catalog_complete_environment_count")
        != environment_count
        or observation.get(
            "suffix_tokens_read_validated_retained_or_used"
        )
        is not False
    ):
        return False

    gate = contract["catalog_gate"]
    minimum_parents = int(gate["minimum_distinct_diff_trajectory_parents"])
    minimum_environments = int(gate["minimum_distinct_environments"])
    passes = (
        parent_count >= minimum_parents
        and environment_count >= minimum_environments
    )
    expected_terminal = CATALOG_LOCKED if passes else CATALOG_INSUFFICIENT
    expected_gate = {
        "minimum_distinct_diff_trajectory_parents": minimum_parents,
        "observed_required_catalog_complete_parent_count": parent_count,
        "minimum_distinct_environments": minimum_environments,
        "observed_required_catalog_complete_environment_count": (
            environment_count
        ),
        "passes_catalog_capacity_only": passes,
    }
    expected_next = (
        "freeze_separate_hash_bound_d5_s0b_structural_authority_"
        "execution_contract"
        if passes
        else "stop_d5_tartanground_diff_route_for_catalog_capacity"
    )
    if (
        result.get("terminal") != expected_terminal
        or result.get("catalog_gate") != expected_gate
        or result.get("next_authority") != expected_next
    ):
        return False
    required_false = {
        "suffix_tokens_read_validated_retained_or_used",
        "structural_authority_evaluated",
        "d5_s0_source_feasibility_terminal_reached",
        "s0b_execution_authorized_automatically",
        "dataset_payload_ecology_or_effect_authorized",
        "research_mainline_changed",
        "default_app_changed",
        "production_or_safety_claim_authorized",
    }
    if any(result.get(key) is not False for key in required_false):
        return False
    catalog_required_false = {
        "failed_s0a_root_read",
        "dataset_host_request_made",
        "dataset_zip_opened",
        "structural_authority_evaluated",
        "source_feasibility_decided",
    }
    return not any(
        catalog.get(key) is not False for key in catalog_required_false
    )


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
        expected = frozen_source_identity(contract)
        contract_hash = sha256(contract_path)
    except (KeyError, OSError, ValueError):
        return False

    locked = {
        TOOLKIT_DIRNAME,
        FILENAMES["attempt"],
        FILENAMES["preflight"],
        FILENAMES["catalog"],
        FILENAMES["result"],
    }
    if names == locked:
        try:
            attempt = load_json(root / FILENAMES["attempt"])
            preflight = load_json(root / FILENAMES["preflight"])
            catalog = load_json(root / FILENAMES["catalog"])
            result = load_json(root / FILENAMES["result"])
            if (
                attempt.get("schema") != ATTEMPT_SCHEMA
                or attempt.get("status")
                != "ATTEMPT_FSYNCED_BEFORE_FIRST_NEW_GIT_NETWORK_REQUEST"
                or attempt.get("execution_contract_sha256") != contract_hash
                or attempt.get("toolkit_repository")
                != expected["toolkit_repository"]
                or attempt.get("toolkit_commit") != expected["toolkit_commit"]
            ):
                return False
            if (
                preflight.get("schema") != PREFLIGHT_SCHEMA
                or preflight.get("status")
                != (
                    "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_NEW_GIT_"
                    "NETWORK_REQUEST"
                )
                or preflight.get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
            ):
                return False
            if (
                catalog.get("schema") != CATALOG_SCHEMA
                or catalog.get("status")
                != "EXACT_COMMIT_PATH_TOKEN_CATALOG_INVENTORY_COMPLETE"
                or catalog.get("source_identity") != expected
                or catalog.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or catalog.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
                or catalog.get("catalog_observation", {}).get(
                    "suffix_tokens_read_validated_retained_or_used"
                )
                is not False
            ):
                return False
            if (
                result.get("schema") != RESULT_SCHEMA
                or result.get("source_identity") != expected
                or not validate_catalog_result_semantics(
                    contract, catalog, result
                )
            ):
                return False
            for key in ("attempt", "preflight", "catalog"):
                if result["bindings"][f"{key}_sha256"] != sha256(
                    root / FILENAMES[key]
                ):
                    return False
            return (
                git_local(
                    "rev-parse",
                    "FETCH_HEAD",
                    cwd=root / TOOLKIT_DIRNAME,
                )
                == expected["toolkit_commit"]
            )
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
                if path.is_file() and recorded != sha256(path):
                    return False
                if not path.exists() and recorded is not None:
                    return False
            if (root / FILENAMES["attempt"]).is_file():
                attempt = load_json(root / FILENAMES["attempt"])
                if (
                    attempt.get("schema") != ATTEMPT_SCHEMA
                    or attempt.get("status")
                    != (
                        "ATTEMPT_FSYNCED_BEFORE_FIRST_NEW_GIT_"
                        "NETWORK_REQUEST"
                    )
                    or attempt.get("execution_contract_sha256")
                    != contract_hash
                    or attempt.get("toolkit_commit")
                    != expected["toolkit_commit"]
                    or attempt.get("toolkit_repository")
                    != expected["toolkit_repository"]
                ):
                    return False
            preflight_path = root / FILENAMES["preflight"]
            if preflight_path.is_file():
                preflight = load_json(preflight_path)
                if (
                    preflight.get("schema") != PREFLIGHT_SCHEMA
                    or preflight.get("status")
                    != (
                        "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_NEW_GIT_"
                        "NETWORK_REQUEST"
                    )
                    or preflight.get("attempt_sha256")
                    != sha256(root / FILENAMES["attempt"])
                ):
                    return False
            observed_fetch_head = local_fetch_head_or_none(root)
            if failure.get("fetched_commit") != observed_fetch_head:
                return False
            if (
                observed_fetch_head is not None
                and observed_fetch_head != expected["toolkit_commit"]
            ):
                return False
            return True
        except (KeyError, OSError, ValueError):
            return False
    return False


def freeze_existing_partial(root: Path, names: set[str]) -> int:
    if FILENAMES["failure"] in names:
        raise ValueError("Existing S0A.1 failure is corrupt or ambiguous")
    write_json_exclusive_fsync(
        root / FILENAMES["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": CATALOG_INVALID,
            "reason": "existing_partial_or_unknown_s0a1_root",
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
            "fetched_commit": local_fetch_head_or_none(root),
            "resume_or_rerun_authorized": False,
            "failed_s0a_root_read": False,
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
        "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_NEW_GIT_NETWORK_REQUEST",
        "execution_contract_path": display_path(context["contract_path"]),
        "execution_contract_sha256": sha256(context["contract_path"]),
        "toolkit_repository": contract["source_lock"]["toolkit_repository"],
        "toolkit_commit": contract["source_lock"]["toolkit_commit"],
        "git_fetch_attempts_authorized": 1,
        "failed_s0a_root_read_authorized": False,
        "dataset_host_request_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["attempt"], attempt)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": (
            "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_NEW_GIT_NETWORK_REQUEST"
        ),
        "attempt_sha256": sha256(root / FILENAMES["attempt"]),
        "head": git_local("rev-parse", "HEAD") if verify_git else None,
        "origin_master": (
            git_local("rev-parse", "origin/master") if verify_git else None
        ),
        "parent_sha256": {
            label: sha256(path) for path, label in context["parent_paths"]
        },
        "implementation_sha256": {
            label: sha256(path)
            for path, label in context["implementation_paths"]
        },
        "test_sha256": sha256(context["test_path"]),
        "canonical_output_root": display_path(root),
        "failed_s0a_root_read": False,
        "dataset_host_request_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["preflight"], preflight)

    observed = fetch_and_read_catalog(
        contract, root, git_runner=git_runner
    )
    source_identity = frozen_source_identity(contract)
    observed.pop("toolkit_commit")
    catalog = {
        "schema": CATALOG_SCHEMA,
        "status": "EXACT_COMMIT_PATH_TOKEN_CATALOG_INVENTORY_COMPLETE",
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
        },
        "source_identity": source_identity,
        "catalog_observation": observed,
        "failed_s0a_root_read": False,
        "dataset_host_request_made": False,
        "dataset_zip_opened": False,
        "structural_authority_evaluated": False,
        "source_feasibility_decided": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["catalog"], catalog)

    gate = contract["catalog_gate"]
    parent_count = int(
        observed["required_catalog_complete_parent_count"]
    )
    environment_count = int(
        observed["required_catalog_complete_environment_count"]
    )
    passes = (
        parent_count
        >= int(gate["minimum_distinct_diff_trajectory_parents"])
        and environment_count >= int(gate["minimum_distinct_environments"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": CATALOG_LOCKED if passes else CATALOG_INSUFFICIENT,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "catalog_sha256": sha256(root / FILENAMES["catalog"]),
        },
        "source_identity": source_identity,
        "catalog_gate": {
            "minimum_distinct_diff_trajectory_parents": gate[
                "minimum_distinct_diff_trajectory_parents"
            ],
            "observed_required_catalog_complete_parent_count": parent_count,
            "minimum_distinct_environments": gate[
                "minimum_distinct_environments"
            ],
            "observed_required_catalog_complete_environment_count": (
                environment_count
            ),
            "passes_catalog_capacity_only": passes,
        },
        "suffix_tokens_read_validated_retained_or_used": False,
        "structural_authority_evaluated": False,
        "d5_s0_source_feasibility_terminal_reached": False,
        "s0b_execution_authorized_automatically": False,
        "dataset_payload_ecology_or_effect_authorized": False,
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
                        "fetched_commit": local_fetch_head_or_none(root),
                        "resume_or_rerun_authorized": False,
                        "failed_s0a_root_read": False,
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
            raise ValueError("D5-S0A.1 validated terminal already exists")
        return freeze_existing_partial(root, names)
    result = execute_with_failure_closure(
        args.execution_contract.resolve(), root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
