"""Independently recompute and validate an R2-P0 materialized canonical view."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from . import PROTOCOL_ID


INDEPENDENT_MAP = {
    0: 3, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 0, 7: 3,
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1,
    16: 2, 17: 0, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2,
    24: 2, 25: 2, 26: 2, 27: 3, 28: 2, 29: 3, 30: 3,
}


class CanonicalViewValidationError(ValueError):
    """Raised when a canonical view is incomplete or not independently reproducible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanonicalViewValidationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CanonicalViewValidationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CanonicalViewValidationError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise CanonicalViewValidationError(f"zero-row manifest is invalid: {path}")
    return rows


def _safe_repo_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CanonicalViewValidationError(f"path escapes repository: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _decode_independent(path: Path, decoder: str) -> np.ndarray:
    with Image.open(path) as image:
        mode = image.mode
        raw = np.asarray(image)
    if decoder == "source_native":
        if mode in {"L", "P"}:
            native = raw
        elif mode in {"RGB", "RGBA"} and raw.ndim == 3 and raw.shape[2] >= 3:
            native = raw[..., 0]
        else:
            raise CanonicalViewValidationError(f"independent decoder rejects source mode {mode!r}")
        unknown = sorted(set(int(value) for value in np.unique(native)) - set(INDEPENDENT_MAP))
        if unknown:
            raise CanonicalViewValidationError(f"independent decoder found unknown native IDs: {unknown}")
        canonical = np.full(native.shape, 255, dtype=np.uint8)
        for source_id, canonical_id in INDEPENDENT_MAP.items():
            canonical[native == source_id] = canonical_id
    elif decoder == "canonical_passthrough":
        if mode not in {"L", "P"} or raw.ndim != 2:
            raise CanonicalViewValidationError(
                f"independent canonical decoder rejects mode {mode!r}"
            )
        canonical = raw.astype(np.uint8, copy=False)
        unknown = sorted(set(int(value) for value in np.unique(canonical)) - {0, 1, 2, 3})
        if unknown:
            raise CanonicalViewValidationError(f"independent canonical IDs invalid: {unknown}")
    else:
        raise CanonicalViewValidationError(f"unknown decoder: {decoder!r}")
    resized = Image.fromarray(canonical, mode="L").resize(
        (256, 256),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.uint8)


def validate(
    *,
    repo_root: Path,
    view_root: Path,
    canonicalizer_path: Path,
    contract_path: Path,
    schema_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    view_root = view_root.resolve()
    receipt = _read_json(view_root / "receipt.json")
    manifest_path = view_root / str(receipt.get("manifest"))
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "CANONICAL_VIEW_MATERIALIZED"
        or receipt.get("formal_authority") is not False
    ):
        raise CanonicalViewValidationError("canonical view receipt identity/status mismatch")
    if sha256_file(manifest_path) != receipt.get("manifest_sha256"):
        raise CanonicalViewValidationError("canonical view manifest SHA256 mismatch")
    identity = receipt.get("identity", {})
    expected_identity = {
        "canonicalizer_code_sha256": sha256_file(canonicalizer_path.resolve()),
        "canonicalizer_config_sha256": sha256_file(contract_path.resolve()),
        "canonical_view_schema_sha256": sha256_file(schema_path.resolve()),
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise CanonicalViewValidationError(f"{key} mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(receipt.get("row_count", -1)):
        raise CanonicalViewValidationError("canonical view row count mismatch")
    ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    source_native_ids: set[int] = set()
    canonical_ids: set[int] = set()
    for index, row in enumerate(rows, start=1):
        if (
            row.get("schema_version")
            != "blindassist.dual_loop_segmentation_r2_p0.materialized_canonical_view_row.v1"
            or row.get("protocol_id") != PROTOCOL_ID
            or row.get("formal_authority") is not False
        ):
            raise CanonicalViewValidationError(f"row {index}: identity/authority mismatch")
        row_id = str(row.get("id"))
        if row_id in ids:
            raise CanonicalViewValidationError(f"row {index}: duplicate identity")
        ids.add(row_id)
        source_path = _safe_repo_path(repo_root, str(row["source_mask_repo_relative_path"]))
        if sha256_file(source_path) != row.get("source_mask_sha256"):
            raise CanonicalViewValidationError(f"row {index}: source mask SHA256 mismatch")
        canonical_path = (view_root / str(row["canonical_mask_path"])).resolve()
        try:
            canonical_path.relative_to(view_root)
        except ValueError as exc:
            raise CanonicalViewValidationError(f"row {index}: canonical path escapes view") from exc
        if sha256_file(canonical_path) != row.get("canonical_mask_sha256"):
            raise CanonicalViewValidationError(f"row {index}: canonical mask SHA256 mismatch")
        independent = _decode_independent(source_path, str(row["source_mask_decoder"]))
        with Image.open(canonical_path) as image:
            if image.mode != "L" or image.size != (256, 256):
                raise CanonicalViewValidationError(f"row {index}: canonical encoding mismatch")
            materialized = np.asarray(image, dtype=np.uint8)
        if not np.array_equal(independent, materialized):
            raise CanonicalViewValidationError(f"row {index}: independent recompute mismatch")
        row_ids_value = sorted(int(value) for value in np.unique(materialized))
        if row_ids_value != row.get("canonical_ids"):
            raise CanonicalViewValidationError(f"row {index}: canonical ID ledger mismatch")
        if not set(row_ids_value).issubset({0, 1, 2, 3}):
            raise CanonicalViewValidationError(f"row {index}: canonical ID outside 0..3")
        if str(row["source_mask_decoder"]) == "source_native":
            with Image.open(source_path) as source_image:
                raw = np.asarray(source_image)
                native = raw if source_image.mode in {"L", "P"} else raw[..., 0]
            source_native_ids.update(int(value) for value in np.unique(native))
        canonical_ids.update(row_ids_value)
        role_counts[str(row["role"])] += 1
        source_counts[str(row["source_id"])] += 1
        session_counts[str(row["session_id"])] += 1
    if dict(sorted(role_counts.items())) != receipt.get("role_counts"):
        raise CanonicalViewValidationError("role aggregate differs from receipt")
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_r2_p0.canonical_view_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "formal_authority": False,
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(rows),
        "identity_count": len(ids),
        "role_counts": dict(sorted(role_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "session_counts": dict(sorted(session_counts.items())),
        "source_native_ids_observed": sorted(source_native_ids),
        "canonical_ids_observed": sorted(canonical_ids),
        "all_native_ids_0_30_covered": source_native_ids.issuperset(set(range(31))),
        "all_canonical_ids_within_0_3": canonical_ids.issubset({0, 1, 2, 3}),
        "independent_full_recompute": True,
        "zero_row_policy": "FAIL_CLOSED",
        "atomic_publish": receipt.get("atomic_publish") is True,
        "interruption_resume_supported": receipt.get("interruption_resume_supported") is True,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_path.with_suffix(".sha256.json").write_text(
        json.dumps({"sha256": sha256_file(output_path)}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--canonicalizer", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    value = validate(
        repo_root=args.repo_root,
        view_root=args.view_root,
        canonicalizer_path=args.canonicalizer,
        contract_path=args.contract,
        schema_path=args.schema,
        output_path=args.output,
    )
    print(json.dumps({"status": value["status"], "rows": value["row_count"]}))
