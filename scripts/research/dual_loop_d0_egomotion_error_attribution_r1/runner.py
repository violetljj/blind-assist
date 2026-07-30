from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from analysis import analyze_event_table
from bindings import load_prestart_bundle
from contract import (
    EVENT_COUNT,
    PROTOCOL_ID,
    PROTOCOL_RELATIVE_PATH,
    PROTOCOL_SHA256,
    canonical_json_bytes,
    canonical_json_line,
    find_repo_root,
    load_protocol,
    sha256_file,
)
from producer import build_event_table, read_camera_from_marker, read_vicon_tracks
from validate_implementation_lock import (
    create_formal_start_marker,
    validate as validate_implementation_lock,
    validate_activation_identity,
)


FORMAL_OUTPUT_RELATIVE_PATH = Path(
    "artifacts.local/evidence/dual-loop/"
    "d0-egomotion-error-attribution-r1/run-r1"
)


class RunnerError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RunnerError(f"{path} must contain a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("atomic write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_json_bytes(value) + b"\n")


def _git_clean_and_at_origin(repo_root: Path) -> dict[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if status:
        raise RunnerError("formal activation requires a clean worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/master"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if head != origin:
        raise RunnerError("HEAD does not equal origin/master")
    return {"head": head, "origin_master": origin}


def _activation_authorizes(activation: dict[str, Any]) -> bool:
    return (
        activation.get("formal_execution_authorized") is True
        and activation.get("authority")
        == {
            "formal_execution_authorized": True,
            "successor_execution_authorized": False,
            "confirmation_authorized": False,
            "product_or_safety_authorized": False,
        }
    )


def run_producer(
    *,
    repo_root: Path,
    activation_path: Path,
    implementation_lock_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    activation_path = activation_path.resolve()
    implementation_lock_path = implementation_lock_path.resolve()
    output_root = output_root.resolve()
    expected_output = (root / FORMAL_OUTPUT_RELATIVE_PATH).resolve()
    if output_root != expected_output:
        raise RunnerError("formal output root differs from frozen protocol")
    if output_root.exists():
        raise RunnerError("formal output namespace already exists")

    protocol = load_protocol(root)
    implementation_validation = validate_implementation_lock(
        implementation_lock_path,
        root,
    )
    if implementation_validation.get("status") != "VALID":
        raise RunnerError(
            "implementation lock is invalid: "
            f"{implementation_validation.get('failures', [])}"
        )
    activation = _load_json(activation_path)
    if activation_path.read_bytes() != canonical_json_bytes(activation) + b"\n":
        raise RunnerError("activation is not canonical JSON")
    activation_identity = validate_activation_identity(
        activation,
        implementation_lock_path,
        root,
        expected_formal_output_root=FORMAL_OUTPUT_RELATIVE_PATH.as_posix(),
    )
    if activation_identity["status"] != "VALID_IDENTITY":
        raise RunnerError(
            f"activation identity invalid: {activation_identity['failures']}"
        )
    if not _activation_authorizes(activation):
        raise RunnerError("activation does not authorize formal execution")
    git_identity = _git_clean_and_at_origin(root)
    if activation.get("repository") != git_identity:
        raise RunnerError(
            "live repository identity differs from activation and implementation lock"
        )

    # Prestart may parse burned JSON and hash the bag, but does not open messages
    # or derive any D0 metric.
    bundle = load_prestart_bundle(root, protocol)
    activation_sha = sha256_file(activation_path)
    implementation_lock_sha = sha256_file(implementation_lock_path)
    output_root.mkdir(parents=False, exist_ok=False)
    formal_start = {
        "schema_version": "blindassist.d0_formal_start.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "activation": {
            "path": str(activation_path),
            "sha256": activation_sha,
        },
        "implementation_lock": {
            "path": str(implementation_lock_path),
            "sha256": implementation_lock_sha,
        },
        "repository": git_identity,
        "state": "FORMAL_STARTED",
        "d0_metric_computation_pending": True,
        "vicon_bag_messages_opened": False,
    }
    formal_start_path = output_root / "formal_start.json"
    try:
        create_formal_start_marker(formal_start_path, formal_start)
        _atomic_json(
            output_root / "progress.json",
            {
                "protocol_id": PROTOCOL_ID,
                "state": "FORMAL_STARTED",
                "completed_event_count": 0,
                "expected_event_count": EVENT_COUNT,
            },
        )
        frozen = protocol["frozen_inputs"]
        tracks = read_vicon_tracks(root / frozen["revel_dynamic_bag"]["path"])
        calibration = read_camera_from_marker(
            root / frozen["revel_calibration"]["path"]
        )
        event_rows = build_event_table(bundle, tracks, calibration)
        if len(event_rows) != EVENT_COUNT:
            raise RunnerError("event-table row-count drift")
        event_path = output_root / "event_table.jsonl"
        _atomic_write(
            event_path,
            b"".join(canonical_json_line(row) for row in event_rows),
        )
        event_hash = sha256_file(event_path)
        analysis = analyze_event_table(event_rows, event_hash)
        analysis_path = output_root / "analysis.json"
        _atomic_json(analysis_path, analysis)
        receipt = {
            "schema_version": "blindassist.d0_producer_receipt.v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "status": "PRODUCER_COMPLETE_NOT_YET_VALID",
            "formal_start_sha256": sha256_file(formal_start_path),
            "event_table": {
                "path": str(event_path),
                "sha256": event_hash,
                "rows": len(event_rows),
            },
            "analysis": {
                "path": str(analysis_path),
                "sha256": sha256_file(analysis_path),
            },
            "frozen_inputs": bundle.binding_summary,
            "binding_summary": {
                "activation_sha256": activation_sha,
                "implementation_lock_sha256": implementation_lock_sha,
                "head": git_identity["head"],
            },
            "forbidden_access": {
                "old_f1b_decision_opened": False,
                "production_ab_trace_opened": False,
                "confirmation_opened": False,
            },
            "errors": [],
        }
        receipt_path = output_root / "producer_receipt.json"
        _atomic_json(receipt_path, receipt)
        os.replace(
            output_root / "progress.json",
            output_root / ".progress.previous.json",
        )
        _atomic_json(
            output_root / "progress.json",
            {
                "protocol_id": PROTOCOL_ID,
                "state": "PRODUCER_COMPLETE_NOT_YET_VALID",
                "completed_event_count": len(event_rows),
                "expected_event_count": EVENT_COUNT,
            },
        )
        (output_root / ".progress.previous.json").unlink()
        return receipt
    except BaseException as error:
        if formal_start_path.exists():
            failure_path = output_root / "failure_receipt.json"
            if not failure_path.exists():
                _atomic_json(
                    failure_path,
                    {
                        "schema_version": "blindassist.d0_failure_receipt.v1",
                        "protocol_id": PROTOCOL_ID,
                        "protocol_sha256": PROTOCOL_SHA256,
                        "status": "EXECUTION_INVALID",
                        "consumed": True,
                        "rerun_authorized": False,
                        "scientific_exit": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "formal_start_sha256": sha256_file(formal_start_path),
                    },
                )
        else:
            try:
                output_root.rmdir()
            except OSError:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the activated one-shot D0-R1 producer and analysis."
    )
    parser.add_argument("--repo-root", type=Path, default=find_repo_root())
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Must equal the frozen run-r1 output root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (repo_root / FORMAL_OUTPUT_RELATIVE_PATH).resolve()
    )
    receipt = run_producer(
        repo_root=repo_root,
        activation_path=args.activation,
        implementation_lock_path=args.implementation_lock,
        output_root=output_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
