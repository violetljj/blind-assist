#!/usr/bin/env python3
"""Validate a host-research performance preflight receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "blindassist.host_research_preflight.v1"
TASK_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{2,79}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WORKLOAD_CLASSES = {
    "python_serial",
    "cpu_data_parallel",
    "native_numeric",
    "cuda_batch",
    "decode_io",
    "memory_bound",
    "mixed_pipeline",
}
BACKENDS = {
    "serial",
    "cpu_process_pool",
    "cpu_threads",
    "cuda",
    "bounded_io",
    "mixed",
}
REQUIRED_PROGRESS_FIELDS = {
    "phase",
    "completed_units",
    "total_units",
    "throughput",
    "eta_seconds",
    "last_progress_at",
    "status",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _positive_number(value: Any, label: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be a positive number")
        return 0.0
    number = float(value)
    if number <= 0:
        errors.append(f"{label} must be greater than zero")
    return number


def _positive_integer(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{label} must be a positive integer")
        return 0
    return value


def _required_true(value: Any, label: str, errors: list[str]) -> None:
    if value is not True:
        errors.append(f"{label} must be true")


def _required_text(value: Any, label: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string")
        return ""
    return value.strip()


def _repo_path(
    repo_root: Path,
    value: Any,
    label: str,
    errors: list[str],
    *,
    must_exist: bool,
) -> Path | None:
    text = _required_text(value, label, errors)
    if not text:
        return None
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label} must be a repository-relative path")
        return None
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        errors.append(f"{label} escapes the repository root")
        return None
    if must_exist and not resolved.is_file():
        errors.append(f"{label} does not exist as a file: {text}")
    return resolved


def _artifact_path(
    repo_root: Path,
    value: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    resolved = _repo_path(
        repo_root,
        value,
        label,
        errors,
        must_exist=False,
    )
    if resolved is None:
        return None
    artifacts_root = (repo_root / "artifacts.local").resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError:
        errors.append(f"{label} must be under artifacts.local/")
    return resolved


def validate_receipt(
    payload: Any,
    repo_root: Path,
    expected_script: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    root = _object(payload, "receipt", errors)
    if root.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    task_id = root.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        errors.append("task_id must be a stable uppercase identifier")

    execution_class = root.get("execution_class")
    if execution_class not in {"long", "formal"}:
        errors.append("execution_class must be long or formal")

    implementation = _object(
        root.get("implementation"),
        "implementation",
        errors,
    )
    script_path = _repo_path(
        repo_root,
        implementation.get("script"),
        "implementation.script",
        errors,
        must_exist=True,
    )
    script_sha256 = implementation.get("sha256")
    if (
        not isinstance(script_sha256, str)
        or not SHA256_PATTERN.fullmatch(script_sha256)
    ):
        errors.append("implementation.sha256 must be lowercase SHA-256")
    elif script_path is not None and script_path.is_file():
        if _sha256(script_path) != script_sha256:
            errors.append("implementation.sha256 does not match the script")
    if expected_script is not None and script_path is not None:
        if script_path != expected_script.resolve():
            errors.append("implementation.script does not match --expected-script")

    workload = _object(root.get("workload"), "workload", errors)
    if workload.get("class") not in WORKLOAD_CLASSES:
        errors.append("workload.class is unsupported")
    _required_true(
        workload.get("real_data_mechanics_match"),
        "workload.real_data_mechanics_match",
        errors,
    )
    input_identity = _required_text(
        workload.get("input_identity"),
        "workload.input_identity",
        errors,
    )
    if input_identity and not re.search(
        r"(?:^|:)[0-9a-f]{64}$",
        input_identity,
    ):
        errors.append(
            "workload.input_identity must end with a lowercase SHA-256"
        )

    pilot = _object(root.get("pilot"), "pilot", errors)
    pilot_units = _positive_integer(
        pilot.get("representative_units"),
        "pilot.representative_units",
        errors,
    )
    _positive_number(
        pilot.get("wall_seconds"),
        "pilot.wall_seconds",
        errors,
    )
    full_units = _positive_integer(
        pilot.get("projected_full_units"),
        "pilot.projected_full_units",
        errors,
    )
    projected_seconds = _positive_number(
        pilot.get("projected_full_wall_seconds"),
        "pilot.projected_full_wall_seconds",
        errors,
    )
    maximum_seconds = _positive_number(
        pilot.get("maximum_expected_wall_seconds"),
        "pilot.maximum_expected_wall_seconds",
        errors,
    )
    if pilot_units and full_units and full_units < pilot_units:
        errors.append(
            "pilot.projected_full_units must not be smaller than "
            "representative_units"
        )
    if projected_seconds and maximum_seconds:
        if maximum_seconds < projected_seconds:
            errors.append(
                "pilot.maximum_expected_wall_seconds must cover the projection"
            )
        if maximum_seconds > projected_seconds * 4:
            errors.append(
                "pilot.maximum_expected_wall_seconds exceeds 4x projection"
            )
    _required_true(
        pilot.get("same_access_mechanics"),
        "pilot.same_access_mechanics",
        errors,
    )
    if pilot.get("output_equivalence") != "PASS":
        errors.append("pilot.output_equivalence must be PASS")
    progress_samples = _positive_integer(
        pilot.get("progress_samples"),
        "pilot.progress_samples",
        errors,
    )
    if progress_samples and progress_samples < 2:
        errors.append("pilot.progress_samples must be at least 2")

    scheduler = _object(root.get("scheduler"), "scheduler", errors)
    backend = scheduler.get("backend")
    if backend not in BACKENDS:
        errors.append("scheduler.backend is unsupported")
    workers = _positive_integer(
        scheduler.get("workers"),
        "scheduler.workers",
        errors,
    )
    _required_text(scheduler.get("reason"), "scheduler.reason", errors)
    _required_true(
        scheduler.get("comparison_performed"),
        "scheduler.comparison_performed",
        errors,
    )
    _required_true(
        scheduler.get("scientific_parameters_unchanged"),
        "scheduler.scientific_parameters_unchanged",
        errors,
    )
    _positive_number(
        scheduler.get("estimated_gib_per_worker"),
        "scheduler.estimated_gib_per_worker",
        errors,
    )
    _positive_number(
        scheduler.get("reserve_memory_gib"),
        "scheduler.reserve_memory_gib",
        errors,
    )
    if not isinstance(scheduler.get("requires_ac_power"), bool):
        errors.append("scheduler.requires_ac_power must be boolean")
    inject_workers = scheduler.get("inject_workers")
    if not isinstance(inject_workers, bool):
        errors.append("scheduler.inject_workers must be boolean")
    if backend == "serial" and workers not in {0, 1}:
        errors.append("serial scheduler must use exactly one worker")
    if backend in {"cuda", "mixed"}:
        _positive_number(
            scheduler.get("minimum_free_vram_gib"),
            "scheduler.minimum_free_vram_gib",
            errors,
        )

    progress = _object(root.get("progress"), "progress", errors)
    progress_path = _artifact_path(
        repo_root,
        progress.get("path"),
        "progress.path",
        errors,
    )
    fields = progress.get("fields")
    if not isinstance(fields, list) or not all(
        isinstance(item, str) for item in fields
    ):
        errors.append("progress.fields must be a list of strings")
    else:
        missing_fields = sorted(REQUIRED_PROGRESS_FIELDS - set(fields))
        if missing_fields:
            errors.append(
                "progress.fields missing: " + ", ".join(missing_fields)
            )
    interval = _positive_number(
        progress.get("update_interval_seconds"),
        "progress.update_interval_seconds",
        errors,
    )
    if interval > 60:
        errors.append("progress.update_interval_seconds must be at most 60")
    _required_true(
        progress.get("verified_in_pilot"),
        "progress.verified_in_pilot",
        errors,
    )

    terminal = _object(root.get("terminal"), "terminal", errors)
    success_path = _artifact_path(
        repo_root,
        terminal.get("success_path"),
        "terminal.success_path",
        errors,
    )
    failure_path = _artifact_path(
        repo_root,
        terminal.get("failure_path"),
        "terminal.failure_path",
        errors,
    )
    if progress_path is not None and progress_path in {
        success_path,
        failure_path,
    }:
        errors.append("progress.path must differ from terminal paths")
    if success_path is not None and success_path == failure_path:
        errors.append("terminal success and failure paths must differ")

    if execution_class == "formal":
        formal = _object(root.get("formal"), "formal", errors)
        _required_true(formal.get("one_shot"), "formal.one_shot", errors)
        _required_true(
            formal.get("claim_created_by_runner_only"),
            "formal.claim_created_by_runner_only",
            errors,
        )
        claim = _artifact_path(
            repo_root,
            formal.get("claim_path"),
            "formal.claim_path",
            errors,
        )
        output = _artifact_path(
            repo_root,
            formal.get("output_path"),
            "formal.output_path",
            errors,
        )
        failure = _artifact_path(
            repo_root,
            formal.get("failure_receipt_path"),
            "formal.failure_receipt_path",
            errors,
        )
        if claim is not None and output is not None and claim == output:
            errors.append("formal claim and output paths must differ")
        if failure is not None and output is not None and failure == output:
            errors.append("formal failure and output paths must differ")
        if output is not None and output != success_path:
            errors.append(
                "formal.output_path must equal terminal.success_path"
            )
        if failure is not None and failure != failure_path:
            errors.append(
                "formal.failure_receipt_path must equal terminal.failure_path"
            )
        _required_text(
            formal.get("activation_authority"),
            "formal.activation_authority",
            errors,
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-script", type=Path)
    arguments = parser.parse_args()

    repo_root = arguments.repo_root.resolve()
    try:
        payload = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "status": "PERFORMANCE_NOT_QUALIFIED",
                    "errors": [f"cannot read receipt: {error}"],
                },
                sort_keys=True,
            )
        )
        return 2

    errors = validate_receipt(
        payload,
        repo_root,
        arguments.expected_script,
    )
    result = {
        "status": (
            "QUALIFIED" if not errors else "PERFORMANCE_NOT_QUALIFIED"
        ),
        "task_id": payload.get("task_id"),
        "execution_class": payload.get("execution_class"),
        "errors": errors,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
