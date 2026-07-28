from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization|cookie|token|password|secret)\s*[:=]"),
    re.compile(r"(?i)[?&](token|signature|sig|key|credential)="),
)


class ActivationPreflightFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_claim(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ActivationPreflightFailure("CLAIM_NAMESPACE_ALREADY_CONSUMED") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial claim remains consumed by design. Never delete or reuse it.
        raise


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_peak_memory() -> dict[str, int]:
    if os.name != "nt":
        raise ActivationPreflightFailure("WINDOWS_RESOURCE_PROBE_REQUIRED")
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ActivationPreflightFailure("GET_PROCESS_MEMORY_INFO_FAILED")
    result = {
        "peak_rss_bytes": int(counters.PeakWorkingSetSize),
        "peak_commit_bytes": int(counters.PeakPagefileUsage),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in result.values()
    ):
        raise ActivationPreflightFailure("INVALID_RESOURCE_PROBE_RESULT")
    return result


def validate_resource_probe(probe_result: Any) -> dict[str, int]:
    if not isinstance(probe_result, dict):
        raise ActivationPreflightFailure("RESOURCE_PROBE_NOT_OBJECT")
    if set(probe_result) != {"peak_rss_bytes", "peak_commit_bytes"}:
        raise ActivationPreflightFailure("RESOURCE_PROBE_KEYS")
    for value in probe_result.values():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ActivationPreflightFailure("RESOURCE_PROBE_VALUE")
    return probe_result


def secret_scan(paths: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append(
                    {
                        "path": str(path),
                        "line": line_number,
                        "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    }
                )
    return {
        "decision": "PASS" if not findings else "FAIL",
        "scanned_file_count": len(paths),
        "findings": findings,
    }
