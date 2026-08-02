#!/usr/bin/env python3
"""Lock the exact toolkit Python import closure for S0B provider resolution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_d5_s0a_tartanground_catalog import (  # noqa: E402
    git_local,
    load_json,
    require_tracked_clean,
    resolve_bound,
    sha256,
    test_definition_count,
    write_json_exclusive_fsync,
)


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure_"
    "execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_S0B_DESIGN_BEFORE_P0A_EXACT_COMMIT_FETCH_OR_SOURCE_READ"
)
ATTEMPT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure_attempt"
)
PREFLIGHT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure_preflight"
)
TREE_SCHEMA = "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_tree"
CLOSURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure_result"
)
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d5_s0b_p0a_toolkit_source_closure_failure"
)

CLOSURE_LOCKED = (
    "D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_LOCKED_REQUIRES_"
    "P0B_PROVIDER_RESOLUTION"
)
CLOSURE_NOT_EVALUABLE = (
    "D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_NOT_EVALUABLE"
)
CLOSURE_INVALID = (
    "D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_INVALID_STOP"
)

CANONICAL_ROOT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-s0b-p0a-toolkit-source-closure-20260802"
)
CONTRACT_RELATIVE_PATH = Path(
    "docs/research/hftf/"
    "HFTF_STAGE_C_D5_S0B_P0A_TOOLKIT_SOURCE_CLOSURE_"
    "EXECUTION_CONTRACT_2026-08-02.json"
)
FILENAMES = {
    "attempt": "attempt.json",
    "preflight": "preflight.json",
    "tree": "tree.json",
    "closure": "closure.json",
    "result": "result.json",
    "failure": "failure.json",
}
TOOLKIT_DIRNAME = "toolkit"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


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
        raise ValueError(f"Noncanonical P0A output root: {actual}")
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


def validate_contract(
    contract_path: Path,
    *,
    verify_git: bool,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unexpected P0A contract schema")
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("Unexpected P0A contract status")

    parent_paths: list[tuple[Path, str]] = []
    for label, binding in contract["parents"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"P0A parent hash drift: {label}")
        value = load_json(path)
        if binding.get("required_status") and value.get("status") != binding[
            "required_status"
        ]:
            raise ValueError(f"P0A parent status drift: {label}")
        if binding.get("required_terminal") and value.get(
            "terminal"
        ) != binding["required_terminal"]:
            raise ValueError(f"P0A parent terminal drift: {label}")
        parent_paths.append((path, label))

    implementation_paths: list[tuple[Path, str]] = []
    for label, binding in contract["implementations"].items():
        path = resolve_bound(str(binding["path"]))
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"P0A implementation hash drift: {label}")
        implementation_paths.append((path, label))
    test_binding = contract["implementation_tests"]["planner_test"]
    test_path = resolve_bound(str(test_binding["path"]))
    if sha256(test_path) != str(test_binding["sha256"]):
        raise ValueError("P0A test hash drift")
    if test_definition_count(test_path) != int(
        contract["implementation_tests"]["test_count"]
    ):
        raise ValueError("P0A test count drift")

    source = contract["source_lock"]
    if not HEX40_RE.fullmatch(str(source["toolkit_commit"])):
        raise ValueError("P0A toolkit commit is not exact SHA-1")
    closure = contract["source_closure"]
    if closure["seed_path"] != "tartanair/__init__.py":
        raise ValueError("P0A seed path drift")
    if int(closure["maximum_python_blobs"]) <= 0:
        raise ValueError("P0A blob cap invalid")
    if int(closure["maximum_total_source_bytes"]) <= 0:
        raise ValueError("P0A byte cap invalid")
    if (
        closure[
            "source_byte_budget_enforced_before_content_read_using_git_object_size"
        ]
        is not True
    ):
        raise ValueError("P0A pre-read source budget enforcement drift")
    for field in (
        "direct_and_simple_alias_dynamic_import_calls_recorded_not_executed",
        "indirect_dynamic_import_or_exec_signals_recorded_not_executed",
        "p0b_must_not_evaluable_if_dynamic_evidence_nonzero",
        "zero_dynamic_evidence_is_not_runtime_completeness_proof",
    ):
        if closure[field] is not True:
            raise ValueError(f"P0A dynamic-import policy drift: {field}")
    if contract["authorization"]["dataset_host_request_authorized"]:
        raise ValueError("P0A dataset host request must remain forbidden")
    if contract["authorization"]["provider_semantics_or_url_mapping_authorized"]:
        raise ValueError("P0A provider interpretation must remain forbidden")

    if verify_git:
        if git_local("rev-parse", "HEAD") != git_local(
            "rev-parse", "origin/master"
        ):
            raise ValueError("HEAD differs from origin/master")
        for path, label in [
            (contract_path, "P0A execution contract"),
            *parent_paths,
            *implementation_paths,
            (test_path, "P0A planner test"),
        ]:
            require_tracked_clean(path, label)
    return {
        "contract": contract,
        "contract_path": contract_path,
        "parent_paths": parent_paths,
        "implementation_paths": implementation_paths,
        "test_path": test_path,
    }


def parse_tree_paths(value: bytes) -> list[str]:
    if not value.endswith(b"\x00"):
        raise ValueError("P0A tree listing is not NUL terminated")
    paths = [item.decode("utf-8") for item in value[:-1].split(b"\x00")]
    if not paths or any(not item for item in paths):
        raise ValueError("P0A tree listing is empty or malformed")
    if paths != sorted(paths):
        raise ValueError("P0A tree listing is not sorted")
    if len(set(paths)) != len(paths):
        raise ValueError("P0A tree listing contains duplicates")
    for item in paths:
        path = PurePosixPath(item)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item
            or not item.startswith("tartanair/")
        ):
            raise ValueError(f"Unsafe P0A tree path: {item}")
    return paths


def python_module_candidates(module: str) -> tuple[str, str]:
    base = module.replace(".", "/")
    return f"{base}.py", f"{base}/__init__.py"


def current_package(path: str) -> list[str]:
    parts = list(PurePosixPath(path).parts)
    if parts[-1] == "__init__.py":
        return parts[:-1]
    return parts[:-1]


def resolve_module(
    module: str,
    python_paths: set[str],
) -> str | None:
    candidates = [
        path for path in python_module_candidates(module) if path in python_paths
    ]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous local module: {module}")
    return candidates[0] if candidates else None


def resolve_module_chain(
    module: str,
    python_paths: set[str],
) -> list[str]:
    parts = module.split(".")
    targets: list[str] = []
    for index in range(1, len(parts)):
        package_init = "/".join(parts[:index]) + "/__init__.py"
        if package_init in python_paths:
            targets.append(package_init)
    exact = resolve_module(module, python_paths)
    if exact:
        targets.append(exact)
    return list(dict.fromkeys(targets))


def dynamic_import_evidence(tree: ast.AST) -> tuple[int, int]:
    importlib_names: set[str] = set()
    builtins_names: set[str] = set()
    dynamic_callable_names = {"__import__"}
    indirect_callable_names = {"exec", "eval"}
    getattr_names = {"getattr"}
    assignments: list[tuple[ast.AST, ast.AST]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "importlib" or (
                    alias.name.startswith("importlib.")
                    and alias.asname is None
                ):
                    importlib_names.add(local_name)
                elif alias.name == "builtins":
                    builtins_names.add(local_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        dynamic_callable_names.add(alias.asname or alias.name)
            elif node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        dynamic_callable_names.add(alias.asname or alias.name)
                    elif alias.name in {"exec", "eval"}:
                        indirect_callable_names.add(
                            alias.asname or alias.name
                        )
                    elif alias.name == "getattr":
                        getattr_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assignments.append((target, node.value))
            else:
                assignments.append((node.target, node.value))

    def is_dynamic_callable_reference(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in dynamic_callable_names
        if not isinstance(value, ast.Attribute):
            return False
        return (
            value.attr == "import_module"
            and isinstance(value.value, ast.Name)
            and value.value.id in importlib_names
        ) or (
            value.attr == "__import__"
            and isinstance(value.value, ast.Name)
            and value.value.id in builtins_names
        )

    def is_indirect_callable_reference(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in indirect_callable_names
        return (
            isinstance(value, ast.Attribute)
            and value.attr in {"exec", "eval"}
            and isinstance(value.value, ast.Name)
            and value.value.id in builtins_names
        )

    def contains_import_runtime_reference(value: ast.AST) -> bool:
        for child in ast.walk(value):
            if isinstance(child, ast.Name) and child.id in (
                importlib_names
                | builtins_names
                | dynamic_callable_names
                | indirect_callable_names
            ):
                return True
            if is_dynamic_callable_reference(child):
                return True
            if is_indirect_callable_reference(child):
                return True
        return False

    changed = True
    while changed:
        changed = False
        for target, value in assignments:
            if isinstance(target, ast.Name) and isinstance(value, ast.Name):
                if (
                    value.id in importlib_names
                    and target.id not in importlib_names
                ):
                    importlib_names.add(target.id)
                    changed = True
                if (
                    value.id in builtins_names
                    and target.id not in builtins_names
                ):
                    builtins_names.add(target.id)
                    changed = True
            if (
                isinstance(target, ast.Name)
                and target.id not in dynamic_callable_names
                and is_dynamic_callable_reference(value)
            ):
                dynamic_callable_names.add(target.id)
                changed = True
            if (
                isinstance(target, ast.Name)
                and target.id not in indirect_callable_names
                and is_indirect_callable_reference(value)
            ):
                indirect_callable_names.add(target.id)
                changed = True
            if (
                isinstance(target, ast.Name)
                and target.id not in getattr_names
                and isinstance(value, ast.Name)
                and value.id in getattr_names
            ):
                getattr_names.add(target.id)
                changed = True

    direct_calls = 0
    indirect_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if is_dynamic_callable_reference(node.func):
            direct_calls += 1
            continue
        if is_indirect_callable_reference(node.func):
            indirect_calls += 1
            continue
        if isinstance(node.func, ast.Subscript) and (
            contains_import_runtime_reference(node.func)
        ):
            indirect_calls += 1
            continue
        if isinstance(node.func, ast.Name) and node.func.id in getattr_names:
            if node.args:
                owner = node.args[0]
                attribute = node.args[1] if len(node.args) > 1 else None
                owner_is_import_runtime = (
                    isinstance(owner, ast.Name)
                    and owner.id in importlib_names | builtins_names
                )
                attribute_is_import_runtime = (
                    isinstance(attribute, ast.Constant)
                    and attribute.value in {"import_module", "__import__"}
                )
                if owner_is_import_runtime or attribute_is_import_runtime:
                    indirect_calls += 1
    for target, value in assignments:
        if not isinstance(target, ast.Name) and (
            is_dynamic_callable_reference(value)
            or is_indirect_callable_reference(value)
        ):
            indirect_calls += 1
        elif isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)) and (
            contains_import_runtime_reference(value)
        ):
            indirect_calls += 1
    return direct_calls, indirect_calls


def local_import_targets(
    path: str,
    source: str,
    python_paths: set[str],
) -> tuple[list[str], list[str], int, int]:
    tree = ast.parse(source, filename=path)
    targets: set[str] = set()
    unresolved: set[str] = set()
    dynamic_calls, indirect_dynamic_calls = dynamic_import_evidence(tree)
    package = current_package(path)

    def add_module(module: str) -> bool:
        chain = resolve_module_chain(module, python_paths)
        targets.update(chain)
        return resolve_module(module, python_paths) is not None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tartanair" or alias.name.startswith(
                    "tartanair."
                ):
                    if not add_module(alias.name):
                        unresolved.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            bases: list[str] = []
            if node.level:
                ascend = node.level - 1
                if ascend >= len(package):
                    unresolved.add(
                        f"relative-level-{node.level}:{node.module or ''}"
                    )
                    continue
                base_parts = package[: len(package) - ascend]
                if node.module:
                    base_parts += node.module.split(".")
                bases.append(".".join(base_parts))
            elif node.module and (
                node.module == "tartanair"
                or node.module.startswith("tartanair.")
            ):
                bases.append(node.module)
            for base in bases:
                resolved_base = add_module(base)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    child = f"{base}.{alias.name}" if base else alias.name
                    add_module(child)
                if not resolved_base and not any(
                    resolve_module(
                        f"{base}.{alias.name}" if base else alias.name,
                        python_paths,
                    )
                    for alias in node.names
                    if alias.name != "*"
                ):
                    unresolved.add(base)
    return (
        sorted(targets),
        sorted(unresolved),
        dynamic_calls,
        indirect_dynamic_calls,
    )


GitRunner = Callable[[Sequence[str], Path], bytes]


def subprocess_git_runner(args: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def build_source_closure(
    contract: dict[str, Any],
    toolkit: Path,
    tree_paths: list[str],
    *,
    git_runner: GitRunner,
) -> dict[str, Any]:
    rules = contract["source_closure"]
    seed = str(rules["seed_path"])
    python_paths = {path for path in tree_paths if path.endswith(".py")}
    if seed not in python_paths:
        return {
            "terminal": CLOSURE_NOT_EVALUABLE,
            "reason": "exact_seed_path_missing",
        }
    maximum_blobs = int(rules["maximum_python_blobs"])
    maximum_bytes = int(rules["maximum_total_source_bytes"])
    commit = str(contract["source_lock"]["toolkit_commit"])
    queue: deque[str] = deque([seed])
    queued = {seed}
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    external_or_unresolved: set[str] = set()
    dynamic_calls = 0
    indirect_dynamic_calls = 0

    while queue:
        path = queue.popleft()
        if len(rows) >= maximum_blobs:
            raise ValueError(
                "P0A source closure blob budget would be exceeded"
            )
        size_text = git_runner(
            ["cat-file", "-s", f"{commit}:{path}"], toolkit
        ).decode("ascii").strip()
        if not re.fullmatch(r"0|[1-9][0-9]*", size_text):
            raise ValueError(f"Invalid Git blob size metadata: {path}")
        object_size = int(size_text)
        if object_size > maximum_bytes - total_bytes:
            raise ValueError(
                "P0A source closure byte budget would be exceeded"
            )
        blob = git_runner(
            ["cat-file", "blob", f"{commit}:{path}"], toolkit
        )
        if len(blob) != object_size:
            raise ValueError(f"Git blob size receipt mismatch: {path}")
        total_bytes += object_size
        source = blob.decode("utf-8")
        imports, unresolved, calls, indirect_calls = local_import_targets(
            path, source, python_paths
        )
        dynamic_calls += calls
        indirect_dynamic_calls += indirect_calls
        external_or_unresolved.update(unresolved)
        for target in imports:
            if target not in queued:
                queued.add(target)
                queue.append(target)
        oid = git_runner(
            ["rev-parse", f"{commit}:{path}"], toolkit
        ).decode().strip()
        if not HEX40_RE.fullmatch(oid):
            raise ValueError(f"Invalid Git blob OID: {path}")
        rows.append(
            {
                "path": path,
                "git_blob_oid": oid,
                "bytes": object_size,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "local_import_targets": imports,
                "unresolved_local_imports": unresolved,
                "dynamic_import_call_count": calls,
                "indirect_dynamic_import_or_exec_count": indirect_calls,
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "terminal": CLOSURE_LOCKED,
        "seed_path": seed,
        "python_tree_path_count": len(python_paths),
        "closure_blob_count": len(rows),
        "closure_total_source_bytes": total_bytes,
        "closure_rows": rows,
        "unresolved_local_imports": sorted(external_or_unresolved),
        "dynamic_import_call_count": dynamic_calls,
        "indirect_dynamic_import_or_exec_count": indirect_dynamic_calls,
        "no_dynamic_import_evidence_under_frozen_detector": (
            dynamic_calls == 0 and indirect_dynamic_calls == 0
        ),
        "provider_semantics_interpreted": False,
        "url_literal_or_template_extracted": False,
        "dataset_host_request_made": False,
    }


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
        "status": "ATTEMPT_FSYNCED_BEFORE_FIRST_P0A_GIT_NETWORK_REQUEST",
        "execution_contract_path": display_path(context["contract_path"]),
        "execution_contract_sha256": sha256(context["contract_path"]),
        "toolkit_repository": contract["source_lock"]["toolkit_repository"],
        "toolkit_commit": contract["source_lock"]["toolkit_commit"],
        "dataset_host_request_authorized": False,
        "provider_semantics_interpretation_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["attempt"], attempt)
    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0A_GIT_NETWORK_REQUEST",
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
        "dataset_host_request_authorized": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["preflight"], preflight)

    toolkit = root / TOOLKIT_DIRNAME
    toolkit.mkdir()
    source = contract["source_lock"]
    commit = str(source["toolkit_commit"])
    git_runner(["init", "."], toolkit)
    git_runner(
        ["remote", "add", "origin", str(source["toolkit_repository"])],
        toolkit,
    )
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
        raise ValueError("P0A fetched commit drift")
    tree_bytes = git_runner(
        ["ls-tree", "-r", "-z", "--name-only", commit, "--", "tartanair"],
        toolkit,
    )
    tree_paths = parse_tree_paths(tree_bytes)
    tree = {
        "schema": TREE_SCHEMA,
        "status": "EXACT_COMMIT_TARTANAIR_TREE_LOCKED",
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
        },
        "toolkit_commit": commit,
        "path_count": len(tree_paths),
        "python_path_count": sum(path.endswith(".py") for path in tree_paths),
        "paths": tree_paths,
        "blob_content_read": False,
        "dataset_host_request_made": False,
    }
    write_json_exclusive_fsync(root / FILENAMES["tree"], tree)
    closure_observation = build_source_closure(
        contract, toolkit, tree_paths, git_runner=git_runner
    )
    closure_terminal = closure_observation.pop("terminal")
    closure = {
        "schema": CLOSURE_SCHEMA,
        "status": (
            "EXACT_COMMIT_PYTHON_IMPORT_CLOSURE_LOCKED"
            if closure_terminal == CLOSURE_LOCKED
            else "EXACT_SEED_PATH_MISSING_SOURCE_CLOSURE_NOT_EVALUABLE"
        ),
        "terminal": closure_terminal,
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "tree_sha256": sha256(root / FILENAMES["tree"]),
        },
        "toolkit_repository": source["toolkit_repository"],
        "toolkit_commit": commit,
        "observation": closure_observation,
    }
    write_json_exclusive_fsync(root / FILENAMES["closure"], closure)
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": closure["terminal"],
        "bindings": {
            "attempt_sha256": sha256(root / FILENAMES["attempt"]),
            "preflight_sha256": sha256(root / FILENAMES["preflight"]),
            "tree_sha256": sha256(root / FILENAMES["tree"]),
            "closure_sha256": sha256(root / FILENAMES["closure"]),
        },
        "toolkit_repository": source["toolkit_repository"],
        "toolkit_commit": commit,
        "provider_semantics_interpreted": False,
        "provider_url_template_or_mapping_established": False,
        "dataset_host_request_made": False,
        "p0b_execution_authorized_automatically": False,
        "next_authority": (
            "freeze_hash_bound_p0b_provider_resolution_contract"
            if closure["terminal"] == CLOSURE_LOCKED
            else "stop_provider_resolution_as_source_authority_not_evaluable"
        ),
    }
    write_json_exclusive_fsync(root / FILENAMES["result"], result)
    return result


def validate_locked_semantics(
    contract: dict[str, Any],
    tree: dict[str, Any],
    closure: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or tree.get("status") != "EXACT_COMMIT_TARTANAIR_TREE_LOCKED"
    ):
        return False
    paths = tree.get("paths")
    if not isinstance(paths, list) or any(
        not isinstance(path, str) for path in paths
    ):
        return False
    try:
        encoded = b"\x00".join(path.encode("utf-8") for path in paths) + b"\x00"
        if parse_tree_paths(encoded) != paths:
            return False
    except (UnicodeError, ValueError):
        return False
    python_paths = {path for path in paths if path.endswith(".py")}
    if (
        tree.get("path_count") != len(paths)
        or tree.get("python_path_count") != len(python_paths)
        or tree.get("blob_content_read") is not False
        or tree.get("dataset_host_request_made") is not False
    ):
        return False
    attempt_sha = tree.get("bindings", {}).get("attempt_sha256")
    preflight_sha = tree.get("bindings", {}).get("preflight_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(attempt_sha or ""))
        or not re.fullmatch(r"[0-9a-f]{64}", str(preflight_sha or ""))
    ):
        return False

    terminal = closure.get("terminal")
    if terminal not in {CLOSURE_LOCKED, CLOSURE_NOT_EVALUABLE}:
        return False
    if result.get("terminal") != terminal:
        return False
    expected_status = (
        "EXACT_COMMIT_PYTHON_IMPORT_CLOSURE_LOCKED"
        if terminal == CLOSURE_LOCKED
        else "EXACT_SEED_PATH_MISSING_SOURCE_CLOSURE_NOT_EVALUABLE"
    )
    if closure.get("status") != expected_status:
        return False
    expected_next = (
        "freeze_hash_bound_p0b_provider_resolution_contract"
        if terminal == CLOSURE_LOCKED
        else "stop_provider_resolution_as_source_authority_not_evaluable"
    )
    if result.get("next_authority") != expected_next:
        return False
    if any(
        result.get(key) is not False
        for key in (
            "provider_semantics_interpreted",
            "provider_url_template_or_mapping_established",
            "dataset_host_request_made",
            "p0b_execution_authorized_automatically",
        )
    ):
        return False

    observation = closure.get("observation")
    if not isinstance(observation, dict):
        return False
    if terminal == CLOSURE_NOT_EVALUABLE:
        return (
            contract["source_closure"]["seed_path"] not in python_paths
            and observation == {"reason": "exact_seed_path_missing"}
        )
    rows = observation.get("closure_rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        return False
    row_paths = [str(row.get("path")) for row in rows]
    if (
        row_paths != sorted(row_paths)
        or len(set(row_paths)) != len(row_paths)
        or not set(row_paths) <= python_paths
        or contract["source_closure"]["seed_path"] not in row_paths
        or observation.get("closure_blob_count") != len(rows)
        or len(rows)
        > int(contract["source_closure"]["maximum_python_blobs"])
        or observation.get("closure_total_source_bytes")
        != sum(int(row.get("bytes", -1)) for row in rows)
        or observation.get("closure_total_source_bytes", -1)
        > int(contract["source_closure"]["maximum_total_source_bytes"])
        or observation.get("python_tree_path_count") != len(python_paths)
    ):
        return False
    row_path_set = set(row_paths)
    dynamic_total = 0
    indirect_dynamic_total = 0
    unresolved: set[str] = set()
    adjacency: dict[str, set[str]] = {}
    for row in rows:
        if (
            not HEX40_RE.fullmatch(str(row.get("git_blob_oid", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
            or int(row.get("bytes", -1)) < 0
        ):
            return False
        targets = row.get("local_import_targets")
        missing = row.get("unresolved_local_imports")
        if (
            not isinstance(targets, list)
            or targets != sorted(targets)
            or not set(str(item) for item in targets) <= row_path_set
            or not isinstance(missing, list)
            or missing != sorted(missing)
        ):
            return False
        row_dynamic = int(row.get("dynamic_import_call_count", -1))
        row_indirect_dynamic = int(
            row.get("indirect_dynamic_import_or_exec_count", -1)
        )
        if row_dynamic < 0 or row_indirect_dynamic < 0:
            return False
        adjacency[str(row["path"])] = {str(item) for item in targets}
        unresolved.update(str(item) for item in missing)
        dynamic_total += row_dynamic
        indirect_dynamic_total += row_indirect_dynamic
    reachable: set[str] = set()
    queue = deque([str(contract["source_closure"]["seed_path"])])
    while queue:
        path = queue.popleft()
        if path in reachable:
            continue
        reachable.add(path)
        queue.extend(sorted(adjacency.get(path, set()) - reachable))
    return (
        reachable == row_path_set
        and observation.get("unresolved_local_imports") == sorted(unresolved)
        and observation.get("dynamic_import_call_count") == dynamic_total
        and observation.get("indirect_dynamic_import_or_exec_count")
        == indirect_dynamic_total
        and observation.get("no_dynamic_import_evidence_under_frozen_detector")
        is (dynamic_total == 0 and indirect_dynamic_total == 0)
        and observation.get("provider_semantics_interpreted") is False
        and observation.get("url_literal_or_template_extracted") is False
        and observation.get("dataset_host_request_made") is False
    )


def validate_existing_terminal(root: Path, names: set[str]) -> bool:
    locked = {
        TOOLKIT_DIRNAME,
        FILENAMES["attempt"],
        FILENAMES["preflight"],
        FILENAMES["tree"],
        FILENAMES["closure"],
        FILENAMES["result"],
    }
    contract_path = (repo_root() / CONTRACT_RELATIVE_PATH).resolve()
    if names == locked:
        try:
            contract = load_json(contract_path)
            result = load_json(root / FILENAMES["result"])
            closure = load_json(root / FILENAMES["closure"])
            tree = load_json(root / FILENAMES["tree"])
            attempt = load_json(root / FILENAMES["attempt"])
            preflight = load_json(root / FILENAMES["preflight"])
            if (
                contract.get("schema") != CONTRACT_SCHEMA
                or result.get("schema") != RESULT_SCHEMA
                or closure.get("schema") != CLOSURE_SCHEMA
                or tree.get("schema") != TREE_SCHEMA
                or attempt.get("schema") != ATTEMPT_SCHEMA
                or attempt.get("status")
                != "ATTEMPT_FSYNCED_BEFORE_FIRST_P0A_GIT_NETWORK_REQUEST"
                or preflight.get("schema") != PREFLIGHT_SCHEMA
                or preflight.get("status")
                != (
                    "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0A_GIT_"
                    "NETWORK_REQUEST"
                )
            ):
                return False
            commit = str(contract["source_lock"]["toolkit_commit"])
            repository = str(contract["source_lock"]["toolkit_repository"])
            if (
                result.get("toolkit_repository") != repository
                or result.get("toolkit_commit") != commit
                or closure.get("toolkit_repository") != repository
                or closure.get("toolkit_commit") != commit
                or tree.get("toolkit_commit") != commit
                or attempt.get("toolkit_repository") != repository
                or attempt.get("toolkit_commit") != commit
                or attempt.get("execution_contract_sha256")
                != sha256(contract_path)
                or preflight.get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
            ):
                return False
            for key in ("attempt", "preflight", "tree", "closure"):
                if result["bindings"][f"{key}_sha256"] != sha256(
                    root / FILENAMES[key]
                ):
                    return False
            if (
                tree.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or tree.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
                or closure.get("bindings", {}).get("attempt_sha256")
                != sha256(root / FILENAMES["attempt"])
                or closure.get("bindings", {}).get("preflight_sha256")
                != sha256(root / FILENAMES["preflight"])
                or closure.get("bindings", {}).get("tree_sha256")
                != sha256(root / FILENAMES["tree"])
                or not validate_locked_semantics(
                    contract, tree, closure, result
                )
            ):
                return False
            return local_fetch_head_or_none(root) == commit
        except (KeyError, OSError, ValueError, subprocess.SubprocessError):
            return False
    if FILENAMES["failure"] in names and FILENAMES["result"] not in names:
        try:
            contract = load_json(contract_path)
            failure = load_json(root / FILENAMES["failure"])
            if (
                contract.get("schema") != CONTRACT_SCHEMA
                or failure.get("schema") != FAILURE_SCHEMA
                or failure.get("terminal") != CLOSURE_INVALID
                or failure.get("resume_or_rerun_authorized") is not False
                or failure.get("dataset_host_request_made") is not False
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
            attempt_path = root / FILENAMES["attempt"]
            if attempt_path.is_file():
                attempt = load_json(attempt_path)
                if (
                    attempt.get("schema") != ATTEMPT_SCHEMA
                    or attempt.get("status")
                    != (
                        "ATTEMPT_FSYNCED_BEFORE_FIRST_P0A_GIT_"
                        "NETWORK_REQUEST"
                    )
                    or attempt.get("execution_contract_sha256")
                    != sha256(contract_path)
                    or attempt.get("toolkit_repository")
                    != contract["source_lock"]["toolkit_repository"]
                    or attempt.get("toolkit_commit")
                    != contract["source_lock"]["toolkit_commit"]
                ):
                    return False
            preflight_path = root / FILENAMES["preflight"]
            if preflight_path.is_file():
                preflight = load_json(preflight_path)
                if (
                    preflight.get("schema") != PREFLIGHT_SCHEMA
                    or preflight.get("status")
                    != (
                        "LOCAL_BINDINGS_VALIDATED_BEFORE_FIRST_P0A_GIT_"
                        "NETWORK_REQUEST"
                    )
                    or preflight.get("attempt_sha256")
                    != sha256(attempt_path)
                ):
                    return False
            observed = local_fetch_head_or_none(root)
            if failure.get("fetched_commit") != observed:
                return False
            return observed is None or observed == contract["source_lock"][
                "toolkit_commit"
            ]
        except (KeyError, OSError, ValueError, subprocess.SubprocessError):
            return False
    return False


def freeze_existing_partial(root: Path, names: set[str]) -> int:
    if FILENAMES["failure"] in names:
        raise ValueError("Existing P0A failure is corrupt or ambiguous")
    write_json_exclusive_fsync(
        root / FILENAMES["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": CLOSURE_INVALID,
            "reason": "existing_partial_or_unknown_p0a_root",
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
            "dataset_host_request_made": False,
        },
    )
    return 2


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
                        "terminal": CLOSURE_INVALID,
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
                        "dataset_host_request_made": False,
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
            raise ValueError("P0A validated terminal already exists")
        return freeze_existing_partial(root, names)
    result = execute_with_failure_closure(
        args.execution_contract.resolve(), root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
