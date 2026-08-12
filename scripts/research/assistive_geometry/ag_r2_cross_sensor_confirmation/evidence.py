"""Exclusive, atomic, hash-manifested evidence writing."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .contract import canonical_bytes, require


def _safe_relative(value: str) -> str:
    require("\\" not in value, "F2_EVIDENCE_BACKSLASH_PATH")
    path = PurePosixPath(value)
    require(value != "" and not path.is_absolute() and ".." not in path.parts, "F2_EVIDENCE_UNSAFE_PATH")
    normalized = path.as_posix()
    require(normalized == value and normalized not in (".", ""), "F2_EVIDENCE_NONCANONICAL_PATH")
    return normalized


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


class EvidenceWriter:
    """A root is consumed at construction and no file can ever be overwritten."""

    def __init__(self, root: Path, start_receipt: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.files: dict[str, dict[str, Any]] = {}
        self.write_json("start-receipt.json", dict(start_receipt))

    def _destination(self, relative: str) -> Path:
        canonical = _safe_relative(relative)
        destination = (self.root / Path(*PurePosixPath(canonical).parts)).resolve()
        require(destination.parent == self.root or self.root in destination.parents, "F2_EVIDENCE_PATH_ESCAPE")
        require(canonical not in self.files and not destination.exists(), "F2_EVIDENCE_OVERWRITE")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def write_bytes(self, relative: str, payload: bytes) -> dict[str, Any]:
        destination = self._destination(relative)
        partial = destination.with_name(destination.name + ".partial")
        require(not partial.exists(), "F2_EVIDENCE_PARTIAL_COLLISION")
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
        receipt = {"path": relative, "bytes": len(payload), "sha256": _sha_bytes(payload)}
        self.files[relative] = receipt
        return dict(receipt)

    def write_json(self, relative: str, value: Any) -> dict[str, Any]:
        return self.write_bytes(relative, canonical_bytes(value) + b"\n")

    def write_npz(self, relative: str, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        destination = self._destination(relative)
        partial = destination.with_name(destination.name + ".partial")
        require(not partial.exists(), "F2_EVIDENCE_PARTIAL_COLLISION")
        with partial.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
        payload = destination.read_bytes()
        receipt = {"path": relative, "bytes": len(payload), "sha256": _sha_bytes(payload)}
        self.files[relative] = receipt
        return dict(receipt)

    def finalize(self, terminal: str) -> dict[str, Any]:
        require("manifest.json" not in self.files, "F2_EVIDENCE_ALREADY_FINALIZED")
        manifest = {
            "schema": "blindassist.ag.r2.cross_sensor_factor_confirmation_manifest.v1",
            "evidence_root_consumed": True,
            "terminal": terminal,
            "file_count_before_manifest": len(self.files),
            "bytes_before_manifest": sum(int(row["bytes"]) for row in self.files.values()),
            "files": {key: self.files[key] for key in sorted(self.files)},
        }
        self.write_json("manifest.json", manifest)
        return manifest

