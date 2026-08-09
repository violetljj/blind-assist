#!/usr/bin/env python3
"""Fail-closed preparation and shared-resource checks for the AG-QSF route."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROUTE_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0"
PROTOCOL_RELATIVE = Path(
    "docs/research/assistive-geometry-qsf/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09.json"
)
EXPECTED_SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_ONLY_IMPLEMENTATION_AND_TRAIN_CANARY_LOCK"
IDENTITY_BASES = {"SHA256", "GIT_COMMIT", "MANIFEST_SHA256"}
OWNED_OUTPUT_ROOTS = {
    "artifacts.local/evidence/assistive-geometry-qsf",
    "artifacts.local/models/assistive-geometry-qsf",
    "artifacts.local/work/assistive-geometry-qsf",
}
ALLOWED_KINDS = {
    "SOURCE_DATA",
    "DERIVED_TRAIN_CACHE",
    "INITIALIZATION",
    "CODE_CONTRACT",
    "PROTOCOL",
    "SYNTHETIC_FIXTURE",
    "TEST_TOOL",
    "LITERATURE",
    "OPERATIONAL_LESSON",
    "TRAIN_DIAGNOSTIC",
}
FORBIDDEN_KINDS = {
    "ACTIVE_CHECKPOINT",
    "ACTIVE_RUN_STATE",
    "PROTECTED_OUTCOME",
    "SELECTION_RESULT",
    "CONFIRMATION_IDENTITY",
}
ALLOWED_DATA_ROLES = {
    "TRAIN",
    "CANARY",
    "PROJECT_CONSUMED_DEVELOPMENT",
    "DISCLOSED_CROSS_PROGRAM_CANARY",
    "REGRESSION_ONLY",
    "NOT_APPLICABLE",
}
FORBIDDEN_DATA_ROLES = {
    "DEVELOPMENT_SELECTION",
    "DEVELOPMENT_CALIBRATION",
    "CONFIRMATION",
    "SEALED_UNSEEN",
    "DEPLOYMENT",
}
FORBIDDEN_PATH_PREFIXES = {
    "artifacts.local/evidence/hftf/assistive-geometry-b1-a0-formal-train-"
}
OUTCOME_ACCESS_LEVELS = {
    "NONE",
    "METADATA_ONLY",
    "CONTENT_INSPECTED",
    "OUTPUT_INSPECTED_DIAGNOSTIC_ONLY",
}
SELECTION_INFLUENCES = {"NONE", "DISCLOSED_DEVELOPMENT_ONLY"}
CLAIM_USES = {
    "SCHEMA_ONLY",
    "TRAIN_INPUT_ONLY",
    "TRAIN_TARGET_INPUT_ONLY",
    "INITIALIZATION_ONLY",
    "SYNTHETIC_MECHANICS_ONLY",
    "TEST_ONLY",
    "LITERATURE_CONTEXT_ONLY",
    "DIAGNOSTIC_ONLY",
    "CANARY_INPUT_ONLY",
    "REGRESSION_ONLY",
}
EXPECTED_CANDIDATE_ORDER = [
    "H1_ONLY",
    "H2_ONLY",
    "H1_PLUS_H2_CONDITIONAL_NOT_TRAINABLE",
]
EXPECTED_INFORMATION_ALLOWED = {
    "TRACKED_CONTRACTS",
    "NON_PROTECTED_GEOMETRY_CONVENTIONS",
    "TRAIN_ONLY_OPERATIONAL_LESSONS",
    "FAILURE_MODES_WITHOUT_PROTECTED_OUTCOME",
    "LITERATURE_AND_PUBLIC_METHODS",
}
EXPECTED_INFORMATION_FORBIDDEN = {
    "B1_DEVELOPMENT_OR_CONFIRMATION_OUTCOME",
    "FOREIGN_ACTIVE_CHECKPOINT_OR_PROGRESS",
    "FOREIGN_SELECTION_OR_THRESHOLD_DECISION",
    "UNDISCLOSED_CLAIM_RELEVANT_OUTCOME",
}
HARD_ISOLATION_KEYS = {
    "run_identity_separate",
    "checkpoint_separate",
    "optimizer_scheduler_rng_separate",
    "target_cache_separate",
    "progress_receipt_separate",
    "artifact_root_separate",
    "foreign_process_control_forbidden",
}
STARTER_INTERFACES = {
    "ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT": (
        "PROTOCOL",
        "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json",
        "GEOMETRY_SCHEMA_ONLY",
    ),
    "ASSISTIVE_GEOMETRY_TRUTH_READER_CONVENTIONS": (
        "CODE_CONTRACT",
        "scripts/research/assistive_geometry/arkitscenes_truth_reader.py",
        "SOURCE_UPRIGHT_GEOMETRY_CONVENTIONS_ONLY",
    ),
    "ASSISTIVE_GEOMETRY_HYPOTHESIS_CANARY_LITE_R0": (
        "SYNTHETIC_FIXTURE",
        "scripts/research/assistive_geometry/run_hypothesis_canary_lite.py",
        "MATH_MECHANICS_AND_COUNTEREXAMPLES_ONLY",
    ),
}


class ValidationError(ValueError):
    """A QSF preparation or isolation contract was violated."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(repo_root: Path, logical: PurePosixPath) -> Path:
    return repo_root.joinpath(*logical.parts)


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _git_blob(repo_root: Path, commit: str, logical: PurePosixPath) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{logical}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValidationError(f"git identity does not contain {logical}: {detail}")
    return result.stdout


def _validate_directory_manifest(
    resource_id: str,
    logical: PurePosixPath,
    resource_path: Path,
    manifest_path: Path,
) -> None:
    document = _load_json(manifest_path)
    if document.get("schema_version") != 1 or document.get("complete") is not True:
        raise ValidationError(f"{resource_id}: directory manifest must be complete schema v1")
    root_raw = document.get("root")
    if not isinstance(root_raw, str):
        raise ValidationError(f"{resource_id}: directory manifest root is missing")
    manifest_root = _clean_relative(root_raw, f"{resource_id}.directory_manifest.root")
    if tuple(part.casefold() for part in manifest_root.parts) != tuple(
        part.casefold() for part in logical.parts
    ):
        raise ValidationError(f"{resource_id}: directory manifest root mismatch")

    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError(f"{resource_id}: directory manifest entries must be non-empty")
    declared: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValidationError(f"{resource_id}: directory manifest entry must be an object")
        relative_raw = entry.get("path")
        if not isinstance(relative_raw, str):
            raise ValidationError(f"{resource_id}: directory manifest entry path is missing")
        relative = _clean_relative(relative_raw, f"{resource_id}.directory_manifest.entry.path")
        normalized = str(relative).casefold()
        if normalized in declared:
            raise ValidationError(f"{resource_id}: duplicate directory manifest entry: {relative}")
        declared.add(normalized)
        member = resource_path.joinpath(*relative.parts)
        if not member.is_file() or _is_linklike(member):
            raise ValidationError(f"{resource_id}: directory member is missing or not a regular file: {relative}")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0 or member.stat().st_size != size:
            raise ValidationError(f"{resource_id}: directory member size mismatch: {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError(f"{resource_id}: directory member SHA256 is invalid: {relative}")
        if _sha256(member) != digest:
            raise ValidationError(f"{resource_id}: directory member SHA256 mismatch: {relative}")

    actual: set[str] = set()
    for member in resource_path.rglob("*"):
        if _is_linklike(member):
            raise ValidationError(f"{resource_id}: linked directory members are not shareable")
        if not member.is_file() or member == manifest_path:
            continue
        actual.add(member.relative_to(resource_path).as_posix().casefold())
    if actual != declared:
        missing = sorted(actual - declared)
        extra = sorted(declared - actual)
        raise ValidationError(
            f"{resource_id}: directory manifest is incomplete; unlisted={missing}, absent={extra}"
        )


def _required(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in mapping:
        raise ValidationError(f"missing required field: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise ValidationError(f"{key} must be {expected_type.__name__}")
    return value


def _clean_relative(value: str, field: str) -> PurePosixPath:
    if not value or "\\" in value or ":" in value:
        raise ValidationError(f"{field} must be a non-empty POSIX-style relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValidationError(f"{field} must remain inside the repository logical namespace")
    return path


def _is_same_or_child(path: PurePosixPath, root: PurePosixPath) -> bool:
    path_parts = tuple(part.casefold() for part in path.parts)
    root_parts = tuple(part.casefold() for part in root.parts)
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _validate_owned_roots(protocol: dict[str, Any]) -> tuple[PurePosixPath, ...]:
    raw_roots = _required(protocol, "owned_output_roots", list)
    roots = tuple(_clean_relative(str(value), "owned_output_roots") for value in raw_roots)
    if len(roots) != 3 or len(set(roots)) != len(roots):
        raise ValidationError("owned_output_roots must contain three distinct roots")
    if {str(root).casefold() for root in roots} != {value.casefold() for value in OWNED_OUTPUT_ROOTS}:
        raise ValidationError("owned_output_roots drifted from the fixed QSF namespaces")
    for root in roots:
        if not _is_same_or_child(root, PurePosixPath("artifacts.local")):
            raise ValidationError(f"owned output root is outside artifacts.local: {root}")
        if "assistive-geometry-qsf" not in root.parts:
            raise ValidationError(f"owned output root lacks QSF namespace: {root}")
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if _is_same_or_child(left, right) or _is_same_or_child(right, left):
                raise ValidationError(f"owned output roots overlap: {left} and {right}")
    return roots


def validate_protocol(protocol: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if protocol.get("schema_version") != 1:
        raise ValidationError("protocol schema_version must be 1")
    if protocol.get("route_id") != ROUTE_ID:
        raise ValidationError("unexpected route_id")
    if protocol.get("research_style") != "WILD_LAB":
        raise ValidationError("QSF preparation must remain WILD_LAB")
    if protocol.get("execution_profile") != "CANARY_LITE":
        raise ValidationError("QSF preparation must remain CANARY_LITE")
    if protocol.get("local_successor") != EXPECTED_SUCCESSOR:
        raise ValidationError("QSF must have exactly one expected local successor")
    if protocol.get("candidate_order") != EXPECTED_CANDIDATE_ORDER:
        raise ValidationError("candidate order drifted")

    parallel = _required(protocol, "parallel_policy", dict)
    for key in (
        "parallel_execution_authorized",
        "independent_scheduler_required",
        "no_wait_on_foreign_route",
        "no_mutation_of_foreign_route_state",
        "foreign_route_status_or_successor_unchanged",
    ):
        if parallel.get(key) is not True:
            raise ValidationError(f"parallel_policy.{key} must be true")

    authority = _required(protocol, "execution_authority", dict)
    if authority.get("protocol_and_implementation") is not True:
        raise ValidationError("protocol and implementation authority must be enabled")
    for key in (
        "real_train_canary",
        "h2_implementation_or_materialization",
        "development_outcome_access",
        "confirmation_outcome_access",
        "h1_plus_h2_training",
        "android_or_device",
        "default_app_change",
    ):
        if authority.get(key) is not False:
            raise ValidationError(f"execution_authority.{key} must remain false")
    for key in ("h1_only_implementation", "h2_schema_placeholder_only"):
        if authority.get(key) is not True:
            raise ValidationError(f"execution_authority.{key} must be true")

    roots = _validate_owned_roots(protocol)
    scheduling = _required(protocol, "resource_scheduling", dict)
    if scheduling.get("foreign_formal_train_priority") is not True:
        raise ValidationError("foreign formal training must retain resource priority")
    if scheduling.get("future_gpu_canary_requires_separate_preflight") is not True:
        raise ValidationError("future GPU canary must require a separate preflight")
    active_limits = set(_required(scheduling, "while_foreign_formal_train_running", list))
    required_limits = {
        "CPU_ONLY",
        "SYNTHETIC_ONLY",
        "LIGHT_IO_ONLY",
        "NO_GPU_OR_VRAM_COMPETITION",
        "NO_LONG_MATERIALIZATION",
    }
    if active_limits != required_limits:
        raise ValidationError("foreign formal-train resource limits drifted")
    policy = _required(protocol, "shared_resource_policy", dict)
    if policy.get("manifest_required") is not True or policy.get("access_mode") != "READ_ONLY":
        raise ValidationError("shared resources require a read-only manifest")
    for key in (
        "immutable_identity_required",
        "provenance_required",
        "license_scope_required",
        "outcome_access_disclosure_required",
        "selection_influence_disclosure_required",
    ):
        if policy.get(key) is not True:
            raise ValidationError(f"shared_resource_policy.{key} must be true")
    allowed_kinds = set(_required(policy, "allowed_kinds", list))
    forbidden_kinds = set(_required(policy, "forbidden_kinds", list))
    if allowed_kinds != ALLOWED_KINDS or forbidden_kinds != FORBIDDEN_KINDS:
        raise ValidationError("shared resource kind policy drifted")
    if allowed_kinds & forbidden_kinds:
        raise ValidationError("allowed and forbidden resource kinds overlap")
    allowed_roles = set(_required(policy, "allowed_data_roles", list))
    forbidden_roles = set(_required(policy, "forbidden_data_roles", list))
    if allowed_roles != ALLOWED_DATA_ROLES or forbidden_roles != FORBIDDEN_DATA_ROLES:
        raise ValidationError("shared data-role policy drifted")
    if allowed_roles & forbidden_roles:
        raise ValidationError("allowed and forbidden data roles overlap")
    forbidden_prefixes = {
        str(_clean_relative(str(value), "forbidden_logical_path_prefixes")).casefold()
        for value in _required(policy, "forbidden_logical_path_prefixes", list)
    }
    if forbidden_prefixes != FORBIDDEN_PATH_PREFIXES:
        raise ValidationError("foreign active-run path isolation drifted")
    if set(_required(policy, "allowed_outcome_access", list)) != OUTCOME_ACCESS_LEVELS:
        raise ValidationError("allowed_outcome_access drifted")
    if set(_required(policy, "allowed_selection_influence", list)) != SELECTION_INFLUENCES:
        raise ValidationError("allowed_selection_influence drifted")
    if set(_required(policy, "allowed_claim_uses", list)) != CLAIM_USES:
        raise ValidationError("allowed_claim_uses drifted")

    starter = _required(protocol, "starter_shared_interfaces", list)
    if len(starter) != len(STARTER_INTERFACES) or {
        item.get("resource_id") for item in starter if isinstance(item, dict)
    } != set(STARTER_INTERFACES):
        raise ValidationError("starter_shared_interfaces drifted")
    for item in starter:
        if not isinstance(item, dict):
            raise ValidationError("starter shared interface must be an object")
        logical = _clean_relative(str(item.get("logical_path", "")), "starter.logical_path")
        expected_kind, expected_path, expected_authority = STARTER_INTERFACES[str(item.get("resource_id"))]
        if (
            item.get("kind") != expected_kind
            or str(logical) != expected_path
            or item.get("producer_route") != "BLINDASSIST_ASSISTIVE_GEOMETRY"
            or item.get("authority") != expected_authority
        ):
            raise ValidationError(f"starter shared interface contract drifted: {item.get('resource_id')}")
        if not (repo_root / Path(*logical.parts)).is_file():
            raise ValidationError(f"starter shared interface does not exist: {logical}")
        if item.get("access") != "READ_ONLY":
            raise ValidationError(f"starter shared interface is not read-only: {logical}")
        value = str(item.get("identity_value", ""))
        if item.get("identity_basis") != "GIT_COMMIT" or not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValidationError(f"starter shared interface lacks a full git identity: {logical}")
        current = _logical_path(repo_root, logical).read_bytes()
        if current != _git_blob(repo_root, value, logical):
            raise ValidationError(f"starter shared interface drifted from pinned git blob: {logical}")

    information = _required(protocol, "information_sharing", dict)
    if set(_required(information, "allowed", list)) != EXPECTED_INFORMATION_ALLOWED:
        raise ValidationError("allowed information-sharing policy drifted")
    if set(_required(information, "forbidden", list)) != EXPECTED_INFORMATION_FORBIDDEN:
        raise ValidationError("forbidden information-sharing policy drifted")
    isolation = _required(protocol, "hard_isolation", dict)
    if set(isolation) != HARD_ISOLATION_KEYS or any(isolation.get(key) is not True for key in HARD_ISOLATION_KEYS):
        raise ValidationError("hard isolation policy drifted")

    return {
        "route_id": ROUTE_ID,
        "status": protocol.get("status"),
        "local_successor": EXPECTED_SUCCESSOR,
        "owned_output_roots": [str(root) for root in roots],
        "starter_shared_interface_count": len(starter),
    }


def _validate_identity(
    identity: Any,
    resource_id: str,
    logical: PurePosixPath,
    repo_root: Path,
) -> None:
    if not isinstance(identity, dict):
        raise ValidationError(f"{resource_id}: identity must be an object")
    basis = identity.get("basis")
    value = identity.get("value")
    if basis not in IDENTITY_BASES or not isinstance(value, str) or not value:
        raise ValidationError(f"{resource_id}: invalid immutable identity")
    resource_path = _logical_path(repo_root, logical)
    if not resource_path.exists():
        raise ValidationError(f"{resource_id}: shared resource does not exist: {logical}")
    if _is_linklike(resource_path):
        raise ValidationError(f"{resource_id}: shared resource root cannot be a symlink or junction")
    if basis in {"SHA256", "MANIFEST_SHA256"} and not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError(f"{resource_id}: {basis} must be 64 lowercase hex characters")
    if basis == "GIT_COMMIT" and not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValidationError(f"{resource_id}: GIT_COMMIT must be 40 lowercase hex characters")
    if basis == "SHA256":
        if not resource_path.is_file() or _sha256(resource_path) != value:
            raise ValidationError(f"{resource_id}: shared file SHA256 mismatch")
    elif basis == "GIT_COMMIT":
        if not resource_path.is_file() or resource_path.read_bytes() != _git_blob(repo_root, value, logical):
            raise ValidationError(f"{resource_id}: shared file does not match pinned git blob")
    elif basis == "MANIFEST_SHA256":
        manifest_raw = identity.get("manifest_path")
        if not isinstance(manifest_raw, str):
            raise ValidationError(f"{resource_id}: MANIFEST_SHA256 requires manifest_path")
        manifest_logical = _clean_relative(manifest_raw, f"{resource_id}.identity.manifest_path")
        manifest_path = _logical_path(repo_root, manifest_logical)
        if not manifest_path.is_file() or _is_linklike(manifest_path) or _sha256(manifest_path) != value:
            raise ValidationError(f"{resource_id}: manifest SHA256 mismatch")
        if resource_path.is_dir() and not _is_same_or_child(manifest_logical, logical):
            raise ValidationError(f"{resource_id}: manifest must be inside the shared directory")
        if resource_path.is_file() and manifest_logical != logical:
            raise ValidationError(f"{resource_id}: file resource must identify itself")
        if resource_path.is_dir():
            _validate_directory_manifest(
                resource_id,
                logical,
                resource_path,
                manifest_path,
            )


def validate_resource_manifest(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1 or manifest.get("route_id") != ROUTE_ID:
        raise ValidationError("resource manifest identity does not match QSF R0")
    manifest_id = manifest.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id or manifest_id.startswith("REPLACE_"):
        raise ValidationError("resource manifest requires a run-scoped manifest_id")

    policy = _required(protocol, "shared_resource_policy", dict)
    allowed_kinds = set(policy["allowed_kinds"])
    forbidden_kinds = set(policy["forbidden_kinds"])
    allowed_roles = set(policy["allowed_data_roles"])
    forbidden_roles = set(policy["forbidden_data_roles"])
    forbidden_path_prefixes = tuple(
        str(_clean_relative(str(value), "forbidden_logical_path_prefixes")).casefold()
        for value in _required(policy, "forbidden_logical_path_prefixes", list)
    )
    allowed_outcomes = set(policy["allowed_outcome_access"])
    allowed_influences = set(policy["allowed_selection_influence"])
    allowed_claim_uses = set(policy["allowed_claim_uses"])
    owned_roots = _validate_owned_roots(protocol)

    resources = _required(manifest, "resources", list)
    if not resources:
        raise ValidationError("resource manifest must contain at least one resource")
    seen: set[str] = set()
    diagnostic_only = 0
    for resource in resources:
        if not isinstance(resource, dict):
            raise ValidationError("resource entry must be an object")
        resource_id = resource.get("resource_id")
        if not isinstance(resource_id, str) or not resource_id or resource_id in seen:
            raise ValidationError("resource_id must be non-empty and unique")
        seen.add(resource_id)

        kind = resource.get("kind")
        if kind in forbidden_kinds or kind not in allowed_kinds:
            raise ValidationError(f"{resource_id}: resource kind is not shareable: {kind}")
        if resource.get("access") != "READ_ONLY" or resource.get("immutable") is not True:
            raise ValidationError(f"{resource_id}: shared resource must be immutable and READ_ONLY")
        logical = _clean_relative(str(resource.get("logical_path", "")), f"{resource_id}.logical_path")
        if str(logical).casefold().startswith(forbidden_path_prefixes):
            raise ValidationError(f"{resource_id}: foreign active-run path is not shareable")
        if any(_is_same_or_child(logical, root) for root in owned_roots):
            raise ValidationError(f"{resource_id}: owned output cannot be declared as a shared input")
        _validate_identity(resource.get("identity"), resource_id, logical, repo_root)

        role = resource.get("data_role")
        if role in forbidden_roles or role not in allowed_roles:
            raise ValidationError(f"{resource_id}: data role is not shareable: {role}")
        for field in ("producer_route", "provenance", "license_scope", "selection_influence", "claim_use"):
            value = resource.get(field)
            if not isinstance(value, str) or not value or value.startswith("REPLACE_"):
                raise ValidationError(f"{resource_id}: {field} must be disclosed")
        if resource.get("selection_influence") not in allowed_influences:
            raise ValidationError(f"{resource_id}: selection influence is not allowed")
        if resource.get("claim_use") not in allowed_claim_uses:
            raise ValidationError(f"{resource_id}: claim use is not allowed")

        outcome = resource.get("outcome_access")
        if outcome not in allowed_outcomes:
            raise ValidationError(f"{resource_id}: outcome access is not allowed: {outcome}")
        if outcome == "OUTPUT_INSPECTED_DIAGNOSTIC_ONLY":
            if kind not in {"OPERATIONAL_LESSON", "TRAIN_DIAGNOSTIC"}:
                raise ValidationError(f"{resource_id}: inspected output is limited to diagnostics")
            if resource.get("claim_use") != "DIAGNOSTIC_ONLY":
                raise ValidationError(f"{resource_id}: inspected output must remain DIAGNOSTIC_ONLY")
            diagnostic_only += 1

    return {
        "manifest_id": manifest_id,
        "resource_count": len(resources),
        "diagnostic_only_count": diagnostic_only,
        "access": "READ_ONLY",
    }


def validate_planned_outputs(paths: Iterable[str], protocol: dict[str, Any]) -> list[str]:
    owned_roots = _validate_owned_roots(protocol)
    accepted: list[str] = []
    for value in paths:
        path = _clean_relative(value, "planned_output")
        if not any(_is_same_or_child(path, root) for root in owned_roots):
            raise ValidationError(f"planned output is outside QSF-owned roots: {path}")
        accepted.append(str(path))
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-manifest", type=Path)
    parser.add_argument("--planned-output", action="append", default=[])
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    protocol_path = repo_root / PROTOCOL_RELATIVE
    protocol = _load_json(protocol_path)
    report: dict[str, Any] = {"protocol": validate_protocol(protocol, repo_root)}
    if args.resource_manifest:
        report["resource_manifest"] = validate_resource_manifest(
            _load_json(args.resource_manifest), protocol, repo_root
        )
    if args.planned_output:
        report["planned_outputs"] = validate_planned_outputs(args.planned_output, protocol)
    report["terminal"] = "QSF_PREPARATION_VALID"
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
