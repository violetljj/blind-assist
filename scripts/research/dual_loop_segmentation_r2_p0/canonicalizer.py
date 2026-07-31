"""Frozen SANPO/native and canonical-mask decoder for R2-P0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class CanonicalizationError(ValueError):
    """Raised when a source mask cannot be converted without ambiguity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanonicalizationError(f"expected JSON object: {path}")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    mapping = contract.get("source_native_to_canonical")
    if not isinstance(mapping, dict):
        raise CanonicalizationError("source-native mapping is missing")
    parsed = {int(key): int(value) for key, value in mapping.items()}
    if set(parsed) != set(range(31)):
        raise CanonicalizationError("source-native mapping must cover exactly IDs 0..30")
    if not set(parsed.values()).issubset({0, 1, 2, 3}):
        raise CanonicalizationError("source-native mapping emits IDs outside canonical 0..3")
    output = contract.get("output", {})
    if (
        output.get("width") != 256
        or output.get("height") != 256
        or output.get("resize") != "PIL_NEAREST"
        or output.get("png_mode") != "L"
        or output.get("allowed_ids") != [0, 1, 2, 3]
    ):
        raise CanonicalizationError("canonical output contract mismatch")
    normalized = dict(contract)
    normalized["_parsed_mapping"] = parsed
    return normalized


def decode_source_native(path: Path, contract: dict[str, Any]) -> np.ndarray:
    accepted = set(contract["source_decoder"]["accepted_png_modes"])
    with Image.open(path) as image:
        mode = image.mode
        if mode not in accepted:
            raise CanonicalizationError(f"source-native PNG mode {mode!r} is not allowed")
        raw = np.asarray(image)
    if mode in {"L", "P"}:
        native = raw
    elif mode in {"RGB", "RGBA"}:
        if raw.ndim != 3 or raw.shape[2] < 3:
            raise CanonicalizationError(f"source-native {mode} tensor shape is invalid")
        native = raw[..., 0]
    else:  # pragma: no cover - guarded by the frozen accepted set
        raise CanonicalizationError(f"unsupported source-native mode: {mode}")
    native = np.asarray(native)
    if native.ndim != 2 or not np.issubdtype(native.dtype, np.integer):
        raise CanonicalizationError("source-native decoder did not produce a 2D integer mask")
    unique = {int(value) for value in np.unique(native)}
    unknown = sorted(unique - set(contract["_parsed_mapping"]))
    if unknown:
        raise CanonicalizationError(f"source-native mask contains unmapped IDs: {unknown}")
    return native.astype(np.uint8, copy=False)


def decode_canonical(path: Path, contract: dict[str, Any]) -> np.ndarray:
    accepted = set(contract["canonical_passthrough_decoder"]["accepted_png_modes"])
    with Image.open(path) as image:
        mode = image.mode
        if mode not in accepted:
            raise CanonicalizationError(f"canonical passthrough PNG mode {mode!r} is not allowed")
        value = np.asarray(image)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise CanonicalizationError("canonical passthrough decoder requires a 2D integer mask")
    unique = {int(item) for item in np.unique(value)}
    unknown = sorted(unique - {0, 1, 2, 3})
    if unknown:
        raise CanonicalizationError(f"canonical passthrough mask contains invalid IDs: {unknown}")
    return value.astype(np.uint8, copy=False)


def canonicalize_array(path: Path, decoder: str, contract: dict[str, Any]) -> np.ndarray:
    if decoder == "source_native":
        native = decode_source_native(path, contract)
        canonical = np.full(native.shape, 255, dtype=np.uint8)
        for source_id, canonical_id in contract["_parsed_mapping"].items():
            canonical[native == source_id] = canonical_id
        if np.any(canonical == 255):
            raise CanonicalizationError("source-native mapping left pixels unmapped")
    elif decoder == "canonical_passthrough":
        canonical = decode_canonical(path, contract)
    else:
        raise CanonicalizationError(f"unknown decoder identity: {decoder!r}")
    image = Image.fromarray(canonical, mode="L")
    resized = image.resize((256, 256), resample=Image.Resampling.NEAREST)
    result = np.asarray(resized, dtype=np.uint8)
    unique = set(int(value) for value in np.unique(result))
    if not unique.issubset({0, 1, 2, 3}):
        raise CanonicalizationError(f"canonical output contains invalid IDs: {sorted(unique)}")
    return result


def write_canonical_png(path: Path, value: np.ndarray, contract: dict[str, Any]) -> None:
    array = np.asarray(value, dtype=np.uint8)
    if array.shape != (256, 256):
        raise CanonicalizationError(f"canonical output shape must be 256x256, got {array.shape}")
    if not set(int(item) for item in np.unique(array)).issubset({0, 1, 2, 3}):
        raise CanonicalizationError("refusing to write non-canonical IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array, mode="L").save(
        temporary,
        format="PNG",
        optimize=bool(contract["output"]["png_optimize"]),
        compress_level=int(contract["output"]["png_compress_level"]),
    )
    temporary.replace(path)
