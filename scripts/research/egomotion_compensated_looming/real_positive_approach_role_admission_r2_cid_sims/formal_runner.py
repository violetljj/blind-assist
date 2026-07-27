"""Stable formal one-shot orchestration for role-admission R2.

The runner is the sole claim creator.  It never retries a failed phase and
leaves all partial evidence in place.  SUCCESS.json and FAILURE.json are
mutually exclusive terminal paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
    )

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    acquire,
    bootstrap_claim,
    producer,
    validator,
)


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS"
TOTAL_UNITS = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_progress(
    path: Path,
    *,
    phase: str,
    completed_units: int,
    started: float,
    status: str,
    error: str | None = None,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    throughput = completed_units / elapsed
    remaining = max(TOTAL_UNITS - completed_units, 0)
    value: dict[str, Any] = {
        "schema_version": "rcle.formal_progress.v1",
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "completed_units": completed_units,
        "total_units": TOTAL_UNITS,
        "throughput": throughput,
        "eta_seconds": remaining / throughput if throughput > 0 else None,
        "last_progress_at": _utc_now(),
        "status": status,
    }
    if error is not None:
        value["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _within(path: Path, root: Path) -> bool:
    try:
        path_text = os.path.abspath(os.fspath(path))
        root_text = os.path.abspath(os.fspath(root))
        return os.path.commonpath((path_text, root_text)) == root_text
    except (OSError, ValueError):
        return False


def run_formal(
    *,
    contract: Path,
    source_authority: Path,
    burned_manifest: Path,
    implementation_lock: Path,
    run_dir: Path,
    workers: int = 1,
    expected_hashes: Mapping[str, str] | None = None,
    verify_implementation_files: bool = True,
    opener: Callable[..., Any] = acquire._official_urlopen,
    claim_path: Path | None = None,
    archive_path: Path | None = None,
    source_receipt_path: Path | None = None,
    formal_dir: Path | None = None,
    progress_path: Path | None = None,
    success_path: Path | None = None,
    failure_path: Path | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("R2_WORKERS_MUST_BE_POSITIVE")
    claim_path = claim_path or run_dir / "claim.json"
    archive_path = archive_path or run_dir / "source" / acquire.ARCHIVE_NAME
    source_receipt_path = source_receipt_path or run_dir / "source_receipt.json"
    formal_dir = formal_dir or run_dir / "formal"
    progress_path = progress_path or run_dir / "progress.json"
    success_path = success_path or run_dir / "SUCCESS.json"
    failure_path = failure_path or run_dir / "FAILURE.json"
    outputs = (
        claim_path,
        archive_path,
        source_receipt_path,
        formal_dir,
        progress_path,
        success_path,
        failure_path,
    )
    if not all(_within(path, run_dir) for path in outputs):
        raise ValueError("R2_RUN_OUTPUT_OUTSIDE_RUN_DIR")
    if success_path.exists() or failure_path.exists():
        raise FileExistsError("R2_TERMINAL_PATH_ALREADY_EXISTS")
    if claim_path.exists():
        raise FileExistsError("R2_ONE_SHOT_OUTPUT_ALREADY_EXISTS")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    _atomic_progress(
        progress_path,
        phase="PREACCESS_READY",
        completed_units=completed,
        started=started,
        status="RUNNING",
    )
    try:
        claim = bootstrap_claim.create_claim(
            contract,
            source_authority,
            burned_manifest,
            implementation_lock,
            claim_path,
            expected_hashes=expected_hashes,
            claim_created_by_runner_only=True,
            verify_implementation_files=verify_implementation_files,
        )
        if claim.get("claim_created_by_runner_only") is not True:
            raise ValueError("R2_RUNNER_CLAIM_FLAG")
        if archive_path.exists() or source_receipt_path.exists():
            raise FileExistsError("R2_ONE_SHOT_OUTPUT_ALREADY_EXISTS")
        completed = 1
        _atomic_progress(
            progress_path,
            phase="CLAIM_EXCLUSIVELY_CREATED",
            completed_units=completed,
            started=started,
            status="RUNNING",
        )
        acquisition = acquire.acquire_once(
            claim_path,
            contract,
            source_authority,
            burned_manifest,
            implementation_lock,
            archive_path,
            source_receipt_path,
            opener=opener,
            progress_callback=lambda byte_count: _atomic_progress(
                progress_path,
                phase=f"OFFICIAL_SOURCE_GET_BYTES_{byte_count}",
                completed_units=completed,
                started=started,
                status="RUNNING",
            ),
        )
        completed = 2
        _atomic_progress(
            progress_path,
            phase="SOURCE_RECEIPT_AND_MEMBER_INVENTORY_FROZEN",
            completed_units=completed,
            started=started,
            status="RUNNING",
        )
        bindings = {
            "archive": archive_path,
            "acquisition_receipt": source_receipt_path,
            "contract": contract,
            "claim": claim_path,
            "source_authority": source_authority,
            "burned_manifest": burned_manifest,
            "implementation_lock": implementation_lock,
        }
        result = producer.run(formal_dir, bindings, workers=workers)
        completed = 3
        _atomic_progress(
            progress_path,
            phase="PRODUCER_COMPLETE",
            completed_units=completed,
            started=started,
            status="RUNNING",
        )
        validation = validator.validate(formal_dir, bindings)
        if validation["validation_terminal"] != "VALID":
            raise ValueError("R2_VALIDATOR_INVALID")
        completed = 4
        terminal = {
            "schema_version": "rcle.formal_success.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "SUCCESS",
            "completed_at_utc": _utc_now(),
            "workers": workers,
            "claim_created_by_runner_only": True,
            "archive_sha256": acquisition["archive_sha256"],
            "archive_member_inventory_sha256": acquisition[
                "archive_member_inventory_sha256"
            ],
            "scientific_terminal": result["terminal"],
            "validation_terminal": validation["validation_terminal"],
            "performance_qualification_may_be_created": validation[
                "performance_qualification_may_be_created"
            ],
        }
        _write_exclusive(success_path, terminal)
        _atomic_progress(
            progress_path,
            phase="VALIDATED_TERMINAL",
            completed_units=completed,
            started=started,
            status="SUCCEEDED",
        )
        return terminal
    except acquire.SourceAccessHold as error:
        try:
            validation = validator.validate_source_hold(
                formal_dir,
                source_receipt_path,
                {
                    "archive": archive_path,
                    "contract": contract,
                    "claim": claim_path,
                    "source_authority": source_authority,
                    "burned_manifest": burned_manifest,
                    "implementation_lock": implementation_lock,
                },
            )
        except BaseException as validation_error:
            failure = {
                "schema_version": "rcle.formal_failure.v1",
                "protocol_id": PROTOCOL_ID,
                "status": "FAILURE",
                "failed_at_utc": _utc_now(),
                "completed_units": completed,
                "claim_created_by_runner_only": True,
                "claim_consumed": claim_path.exists(),
                "scientific_terminal": "INVALID_R2_EVIDENCE / INVALID",
                "validation_terminal": "INVALID",
                "error_type": type(validation_error).__name__,
                "error": str(validation_error),
                "retry_authorized": False,
                "replacement_source_authorized": False,
            }
            if not success_path.exists() and not failure_path.exists():
                _write_exclusive(failure_path, failure)
            _atomic_progress(
                progress_path,
                phase="INVALID_SOURCE_HOLD_VALIDATION_NO_RETRY",
                completed_units=completed,
                started=started,
                status="FAILED",
                error=f"{type(validation_error).__name__}:{validation_error}",
            )
            raise
        if validation["validation_terminal"] != "VALID":
            invalid_error = ValueError("R2_SOURCE_HOLD_VALIDATOR_INVALID")
            failure = {
                "schema_version": "rcle.formal_failure.v1",
                "protocol_id": PROTOCOL_ID,
                "status": "FAILURE",
                "failed_at_utc": _utc_now(),
                "completed_units": completed,
                "claim_created_by_runner_only": True,
                "claim_consumed": claim_path.exists(),
                "scientific_terminal": "INVALID_R2_EVIDENCE / INVALID",
                "validation_terminal": "INVALID",
                "error_type": type(invalid_error).__name__,
                "error": str(invalid_error),
                "retry_authorized": False,
                "replacement_source_authorized": False,
            }
            if not success_path.exists() and not failure_path.exists():
                _write_exclusive(failure_path, failure)
            _atomic_progress(
                progress_path,
                phase="INVALID_SOURCE_HOLD_VALIDATION_NO_RETRY",
                completed_units=completed,
                started=started,
                status="FAILED",
                error=str(invalid_error),
            )
            raise invalid_error from error
        terminal = {
            "schema_version": "rcle.formal_success.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "SUCCESS",
            "completed_at_utc": _utc_now(),
            "workers": workers,
            "claim_created_by_runner_only": True,
            "source_access_complete": False,
            "scientific_terminal": acquire.HOLD,
            "validation_terminal": validation["validation_terminal"],
            "validation_scope": validation["validation_scope"],
            "performance_qualification_may_be_created": False,
            "source_error_code": error.code,
        }
        _write_exclusive(success_path, terminal)
        _atomic_progress(
            progress_path,
            phase="VALIDATED_SOURCE_HOLD_NO_RETRY",
            completed_units=TOTAL_UNITS,
            started=started,
            status="SUCCEEDED",
        )
        return terminal
    except BaseException as error:
        failure = {
            "schema_version": "rcle.formal_failure.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "FAILURE",
            "failed_at_utc": _utc_now(),
            "completed_units": completed,
            "claim_created_by_runner_only": True,
            "claim_consumed": claim_path.exists(),
            "scientific_terminal": (
                "INVALID_R2_EVIDENCE / INVALID"
                if claim_path.exists()
                else None
            ),
            "validation_terminal": (
                "INVALID" if claim_path.exists() else "NOT_RUN"
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_authorized": False,
            "replacement_source_authorized": False,
        }
        if not success_path.exists() and not failure_path.exists():
            _write_exclusive(failure_path, failure)
        _atomic_progress(
            progress_path,
            phase="FAILED_NO_RETRY",
            completed_units=completed,
            started=started,
            status="FAILED",
            error=f"{type(error).__name__}:{error}",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--burned-manifest", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument(
        "--expected-implementation-lock-sha256",
        required=True,
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--claim-path", type=Path)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--source-receipt-path", type=Path)
    parser.add_argument("--formal-dir", type=Path)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--success-path", type=Path)
    parser.add_argument("--failure-path", type=Path)
    args = parser.parse_args()
    result = run_formal(
        contract=args.contract,
        source_authority=args.source_authority,
        burned_manifest=args.burned_manifest,
        implementation_lock=args.implementation_lock,
        run_dir=args.run_dir,
        expected_hashes={
            **bootstrap_claim._expected_defaults(),
            "implementation_lock": args.expected_implementation_lock_sha256,
        },
        workers=args.workers,
        claim_path=args.claim_path,
        archive_path=args.archive_path,
        source_receipt_path=args.source_receipt_path,
        formal_dir=args.formal_dir,
        progress_path=args.progress_path,
        success_path=args.success_path,
        failure_path=args.failure_path,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
