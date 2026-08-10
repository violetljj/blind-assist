#!/usr/bin/env python3
"""Exclusive, budgeted evidence writing for TARO O0R factor headroom."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


class FactorEvidenceError(RuntimeError):
    """Stable fail-closed error raised by the factor evidence writer."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FactorEvidenceError(code, message, **context)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def deterministic_gzip(payload: bytes, *, compresslevel: int = 9) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=compresslevel, mtime=0) as stream:
        stream.write(payload)
    return output.getvalue()


def _safe_relative(relative: str) -> str:
    require(isinstance(relative, str) and bool(relative) and "\\" not in relative, "EVIDENCE_PATH_INVALID", "evidence path must be a non-empty POSIX path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts), "EVIDENCE_PATH_INVALID", "evidence path escapes its root", path=relative)
    return path.as_posix()


class FactorEvidenceWriter:
    """One-shot writer whose root creation irreversibly consumes execution."""

    def __init__(self, root: Path, maximum_bytes: int) -> None:
        self.root = root.resolve()
        self.maximum_bytes = int(maximum_bytes)
        require(self.maximum_bytes > 0, "EVIDENCE_BUDGET_INVALID", "evidence byte budget must be positive")
        self.bytes_written = 0
        self.activated = False
        self.file_receipts: dict[str, dict[str, Any]] = {}

    def activate(self, execution_receipt: Mapping[str, Any]) -> dict[str, Any]:
        require(not self.activated and not self.root.exists(), "FACTOR_ROOT_COLLISION", "factor evidence root already exists", root=str(self.root))
        self.root.mkdir(parents=True, exist_ok=False)
        self.activated = True
        return self.write_json("execution-receipt.json", dict(execution_receipt))

    def destination(self, relative: str) -> Path:
        require(self.activated, "WRITER_NOT_ACTIVATED", "factor evidence writer is not activated")
        normalized = _safe_relative(relative)
        destination = (self.root / Path(*PurePosixPath(normalized).parts)).resolve()
        require(destination != self.root and self.root in destination.parents, "EVIDENCE_PATH_INVALID", "evidence path escapes its root", path=relative)
        return destination

    def write_bytes(self, relative: str, payload: bytes) -> dict[str, Any]:
        normalized = _safe_relative(relative)
        require(isinstance(payload, bytes), "EVIDENCE_PAYLOAD_INVALID", "evidence payload must be bytes")
        destination = self.destination(normalized)
        require(normalized not in self.file_receipts and not destination.exists(), "EVIDENCE_OVERWRITE_FORBIDDEN", "factor evidence path already exists", path=normalized)
        require(self.bytes_written + len(payload) <= self.maximum_bytes, "EVIDENCE_BUDGET_EXCEEDED", "factor evidence byte budget exceeded", requested_bytes=len(payload), remaining_bytes=self.maximum_bytes - self.bytes_written)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        require(not temporary.exists(), "EVIDENCE_PARTIAL_COLLISION", "factor evidence partial path already exists", path=str(temporary))
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        receipt = {"path": normalized, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        self.bytes_written += len(payload)
        self.file_receipts[normalized] = dict(receipt)
        return dict(receipt)

    def write_json(self, relative: str, value: Any) -> dict[str, Any]:
        return self.write_bytes(relative, adapter.canonical_json_bytes(value) + b"\n")

    def write_json_gzip(self, relative: str, value: Any) -> dict[str, Any]:
        return self.write_bytes(relative, deterministic_gzip(adapter.canonical_json_bytes(value) + b"\n"))


__all__ = [
    "FactorEvidenceError",
    "FactorEvidenceWriter",
    "deterministic_gzip",
    "sha256_bytes",
]
