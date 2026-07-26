#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import sys
from typing import Any


REPO_ROOT = r"E:\linnan\linnan"
RUNNER_PATH = (
    r"E:\linnan\linnan\scripts\research\egomotion_compensated_looming"
    r"\run_phase_b_bonn_b1a.py"
)
CANONICAL_OUTPUT = (
    r"E:\linnan\linnan\artifacts.local\evidence"
    r"\rcle_phase_b_bonn_b1\b1a_geometry_admission"
)
RUN_CLAIM_PATH = CANONICAL_OUTPUT + r"\run_claim.json"
FAILURE_RECEIPT_PATH = CANONICAL_OUTPUT + r"\failure_receipt.json"

PREREGISTRATION_SHA256 = (
    "f3974b2c0096dae2334b1d6c8cd563d892b09288df4f2085604b8fee88d4cfd0"
)
DESIGN_LOCK_SHA256 = (
    "c53c9edaf7012df481b2ba286902af87f1716e3a5d4f57f27398303c4f74420e"
)
# This value is deliberately resolved by the implementation-review lock flow.
# The runner hashes itself before the claim, avoiding a runner/lock hash cycle.
EXPECTED_IMPLEMENTATION_LOCK_SHA256 = (
    "84bb2c71064e539267602fc8ad51517c15e02b46366fa006f954e55b66b261f4"
)

B0_RECEIPT_SHA256 = (
    "dc0ffe9a890b539478ff4c035b4dfadea6c21347a11b36f164810a18eb811f86"
)
WINDOW_DENOMINATOR_SHA256 = (
    "f1e6f7f2e54da349d004af744573884e6273089f67bda86d5f0eb812234aa05b"
)
SOURCE_AUTHORITY_SHA256 = {
    "bonn_official_page": (
        "2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186"
    ),
    "tum_file_formats": (
        "721c8df093ade2b0078215c3154f6f1a3641a0c691b5123cd037e87b61b30107"
    ),
}
ARCHIVE_SHA256_BY_SEQUENCE = {
    "rgbd_bonn_crowd2": (
        "e751ca1b64165f3789d1c396d5c5c3d25e7ceb49cab01fad9ec87993f6244840"
    ),
    "rgbd_bonn_balloon_tracking": (
        "3c63ec5d06ffc7b97f2f3f965f4bdf7e52b72f38cd98e0b532456e0ef7e3c421"
    ),
    "rgbd_bonn_balloon_tracking2": (
        "3ebbfd803be0c7a992f9f6c222a7170231879b479ec2b70350c777ecd5add789"
    ),
    "rgbd_bonn_moving_obstructing_box2": (
        "cc5d3ec67cb9dadd9905de3ea9b72120f228e0bd4da865858fe44cc2a1ccd643"
    ),
    "rgbd_bonn_balloon2": (
        "9e84087740fbe845f6ed3a6b652656068d960c36c28148321f2cb108dd257aba"
    ),
    "rgbd_bonn_moving_nonobstructing_box2": (
        "b7d41c21d31103e12d538631da427e0464ffeee6165a1837fa0b460e4fa7a9d6"
    ),
}

_PRODUCER_MODULES = (
    "scripts.research.egomotion_compensated_looming."
    "rcle_phase_b_bonn_b1a.producer",
    "scripts.research.egomotion_compensated_looming."
    "rcle_phase_b_bonn_b1a.protocol",
    "scripts.research.egomotion_compensated_looming."
    "rcle_phase_b_bonn_b1a",
)
_VALIDATOR_MODULES = (
    "scripts.research.egomotion_compensated_looming."
    "rcle_phase_b_bonn_b1a.validator",
    "scripts.research.egomotion_compensated_looming."
    "rcle_phase_b_bonn_b1a",
)


def _sha256_self() -> str:
    digest = hashlib.sha256()
    with open(RUNNER_PATH, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _claim_document(runner_sha256: str) -> dict[str, Any]:
    return {
        "archive_sha256_by_sequence": ARCHIVE_SHA256_BY_SEQUENCE,
        "application_data_operations_before_claim": 0,
        "argv": [],
        "b0_receipt_sha256": B0_RECEIPT_SHA256,
        "bootstrap_runner_sha256": runner_sha256,
        "canonical_output": CANONICAL_OUTPUT,
        "canonical_run_claim": RUN_CLAIM_PATH,
        "claim_permanently_retained": True,
        "claimed_at": datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(),
        "delete_replace_or_rewrite_claim": "FORBIDDEN",
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "exclusive_create": True,
        "implementation_lock_sha256": EXPECTED_IMPLEMENTATION_LOCK_SHA256,
        "maximum_claims": 1,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "protocol_id": "RCLE_PHASE_B_BONN_B1A",
        "schema_version": "rcle.phase_b.bonn_b1a.run_claim.v1",
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
        "success_failure_exception_or_interrupt_consumes_claim": True,
        "window_denominator_sha256": WINDOW_DENOMINATOR_SHA256,
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while materializing permanent run claim")
        offset += written


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _create_permanent_claim(runner_sha256: str) -> dict[str, Any]:
    claim = _claim_document(runner_sha256)
    payload = (
        json.dumps(
            claim,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    # This is the first application-data operation on the formal path.
    descriptor = os.open(
        RUN_CLAIM_PATH,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o644,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim


def _write_failure_receipt(
    claim: dict[str, Any],
    runner_sha256: str,
    error: BaseException,
) -> None:
    failure = {
        "bootstrap_runner_sha256": runner_sha256,
        "claim": claim,
        "error_message": str(error),
        "error_type": type(error).__name__,
        "protocol_id": "RCLE_PHASE_B_BONN_B1A",
        "run_claim_sha256": _sha256_path(RUN_CLAIM_PATH),
        "schema_version": "rcle.phase_b.bonn_b1a.failure_receipt.v1",
        "terminal_state": "INVALID_EXECUTION_CLOSE_B1",
    }
    payload = (
        json.dumps(
            failure,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path = (
        CANONICAL_OUTPUT
        + rf"\.failure_receipt.{os.getpid()}.tmp"
    )
    descriptor = os.open(
        temporary_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o644,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name == "nt":
        move_file_ex = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).MoveFileExW
        move_file_ex.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(
            temporary_path,
            FAILURE_RECEIPT_PATH,
            0x00000001 | 0x00000008,
        ):
            error_code = ctypes.get_last_error()
            raise OSError(
                error_code,
                "MoveFileExW failure receipt publish failed",
                FAILURE_RECEIPT_PATH,
            )
    else:
        os.replace(temporary_path, FAILURE_RECEIPT_PATH)
        directory = os.open(CANONICAL_OUTPUT, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _import_first(module_names: tuple[str, ...]) -> Any:
    import importlib

    missing: list[str] = []
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            missing.append(module_name)
    raise ModuleNotFoundError(
        "no compatible B1A module found: " + ", ".join(missing)
    )


def _validate_existing() -> Any:
    from pathlib import Path

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    module = _import_first(_VALIDATOR_MODULES)
    validator = getattr(module, "validate_existing", None)
    if validator is None:
        validator = getattr(module, "validate", None)
    if not callable(validator):
        raise RuntimeError(
            "B1A validator must expose validate_existing(repo_root)"
        )
    return validator(Path(REPO_ROOT))


def _run_formal(claim: dict[str, Any]) -> Any:
    from pathlib import Path

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    module = _import_first(_PRODUCER_MODULES)
    validate_lock = getattr(module, "validate_implementation_lock", None)
    if not callable(validate_lock):
        raise RuntimeError(
            "B1A producer must expose validate_implementation_lock(repo_root)"
        )
    repo_root = Path(REPO_ROOT)
    implementation_lock = validate_lock(repo_root)
    if not isinstance(implementation_lock, dict):
        raise RuntimeError("B1A implementation lock validator returned non-object")
    if implementation_lock.get("canonical_execution_authorized") is not True:
        raise RuntimeError(
            "canonical B1A execution is not authorized by implementation lock"
        )

    producer = getattr(module, "run_b1a", None)
    if producer is None:
        producer = getattr(module, "run", None)
    if not callable(producer):
        raise RuntimeError("B1A producer must expose run_b1a(repo_root, claim)")
    return producer(repo_root, claim)


def _print_result(result: Any) -> None:
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv == ["--validate-existing"]:
        _print_result(_validate_existing())
        return 0
    if argv:
        raise SystemExit(
            "usage: run_phase_b_bonn_b1a.py [--validate-existing]"
        )

    runner_sha256 = _sha256_self()
    claim = _create_permanent_claim(runner_sha256)
    try:
        result = _run_formal(claim)
    except BaseException as error:
        _write_failure_receipt(claim, runner_sha256, error)
        raise
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
