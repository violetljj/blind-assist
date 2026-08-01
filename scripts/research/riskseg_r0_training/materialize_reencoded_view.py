"""Materialize the frozen 520-frame RISKSEG-R0 four-class mask view."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

try:
    from . import PROTOCOL_ID
except ImportError:  # pragma: no cover - direct-script entry point
    PROTOCOL_ID = "RISKSEG_R0"

from scripts.research.dual_loop_segmentation_r2_p0.canonicalizer import (
    CanonicalizationError,
    canonicalize_array,
    load_contract,
    write_canonical_png,
)


LEDGER_SCHEMA = "blindassist.riskseg_r0.data_role_ledger.v1"
ROW_SCHEMA = "blindassist.riskseg_r0.reencoded_view_row.v1"
RECEIPT_SCHEMA = "blindassist.riskseg_r0.reencoded_view_receipt.v1"
EXPECTED_TOTAL_ROWS = 520
CANONICALIZATION_CONTRACT_REPO_PATH = Path(
    "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json"
)
CANONICALIZER_REPO_PATH = Path(
    "scripts/research/dual_loop_segmentation_r2_p0/canonicalizer.py"
)
EXPECTED_CANONICALIZATION_CONTRACT_SHA256 = (
    "a1818b8b52ece55cc046c424defc4451a800790e15905c9c8f4b18a630fe67e1"
)
EXPECTED_CANONICALIZER_SHA256 = (
    "142ba3683270a85bad33d5f5e3773cd263c28fa9801823a6a8785655d309d1aa"
)
SOURCE_SLICE_RULES = {
    "dev_200": {
        "role": "dev",
        "decoder": "canonical_passthrough",
        "dataset_root_from_manifest": "parent",
    },
    "train_consumed_old_blind_120": {
        "role": "train",
        "decoder": "canonical_passthrough",
        "dataset_root_from_manifest": "parent.parent",
    },
    "train_r1_consumed_fresh_200": {
        "role": "train",
        "decoder": "source_native",
        "dataset_root_from_manifest": "parent",
    },
}
EXPECTED_CLASS_MAPPING = {0: 0, 1: 2, 2: 1, 3: 3}
CLASS_NAMES = {
    0: "walkable",
    1: "blocking_obstacle",
    2: "boundary_level_change",
    3: "unknown_nonwalkable",
}


class MaterializationError(RuntimeError):
    """Raised when a frozen-data invariant fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MaterializationError(f"{context}: invalid SHA-256")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MaterializationError(f"blank JSONL row: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MaterializationError(
                        f"expected JSON object: {path}:{line_number}"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"invalid JSONL: {path}:{exc.lineno}") from exc
    except OSError as exc:
        raise MaterializationError(f"cannot read JSONL: {path}") from exc
    return rows


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=separators,
            sort_keys=pretty,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_json_bytes(value, pretty=pretty))
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_repo_path(repo_root: Path, raw_path: Any, *, context: str) -> Path:
    raw = str(raw_path)
    pure = PurePath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        raise MaterializationError(f"{context}: path escape rejected: {raw!r}")
    resolved = (repo_root / Path(raw)).resolve()
    if not _is_relative_to(resolved, repo_root):
        raise MaterializationError(f"{context}: path escapes repository: {raw!r}")
    if not resolved.is_file():
        raise MaterializationError(f"{context}: missing file: {raw!r}")
    return resolved


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise MaterializationError(f"path is outside repository: {path}") from exc


def _resolve_source_asset(
    *,
    dataset_root: Path,
    raw_path: Any,
    context: str,
) -> Path:
    raw = str(raw_path)
    pure = PurePath(raw)
    if not raw or pure.is_absolute() or ".." in pure.parts:
        raise MaterializationError(f"{context}: path escape rejected: {raw!r}")

    candidate = (dataset_root / Path(raw)).resolve()
    if not _is_relative_to(candidate, dataset_root):
        raise MaterializationError(f"{context}: path escapes declared dataset root")
    if not candidate.is_file():
        raise MaterializationError(f"{context}: source asset not found: {raw!r}")
    return candidate


def _source_row_frame_id(row: dict[str, Any], *, row_id: str) -> int:
    for key in ("frame_id", "frame_index", "sequence_frame_index"):
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError) as exc:
                raise MaterializationError(f"{row_id}: invalid {key}") from exc
    raise MaterializationError(f"{row_id}: missing frame identity")


def _normalize_mapping(raw_mapping: Any) -> dict[int, int]:
    if not isinstance(raw_mapping, dict):
        raise MaterializationError("ledger class mapping is missing")
    try:
        mapping = {int(key): int(value) for key, value in raw_mapping.items()}
    except (TypeError, ValueError) as exc:
        raise MaterializationError("ledger class mapping is not integer-valued") from exc
    if mapping != EXPECTED_CLASS_MAPPING:
        raise MaterializationError(
            f"ledger class mapping drift: expected {EXPECTED_CLASS_MAPPING}, got {mapping}"
        )
    return mapping


def _role_contracts(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_roles = ledger.get("roles")
    if not isinstance(raw_roles, dict):
        raise MaterializationError("ledger roles are missing")
    result: dict[str, dict[str, Any]] = {}
    all_sessions: set[str] = set()
    total_rows = 0
    for role_name in ("train", "dev"):
        role = raw_roles.get(role_name)
        if not isinstance(role, dict) or not isinstance(role.get("sessions"), dict):
            raise MaterializationError(f"ledger role {role_name!r} is malformed")
        sessions: dict[str, int] = {}
        for session_id, raw_count in role["sessions"].items():
            count = int(raw_count)
            if not session_id or count <= 0:
                raise MaterializationError(f"ledger role {role_name!r} has invalid session")
            if session_id in all_sessions:
                raise MaterializationError(f"train/dev session overlap: {session_id}")
            all_sessions.add(str(session_id))
            sessions[str(session_id)] = count
        expected_rows = int(role.get("frame_count", -1))
        expected_sessions = int(role.get("session_count", -1))
        if sum(sessions.values()) != expected_rows:
            raise MaterializationError(f"ledger {role_name} session/frame count drift")
        if len(sessions) != expected_sessions:
            raise MaterializationError(f"ledger {role_name} session count drift")
        total_rows += expected_rows
        result[role_name] = {
            "role_contract": str(role.get("role", "")),
            "frame_count": expected_rows,
            "session_count": expected_sessions,
            "sessions": sessions,
        }
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise MaterializationError(
            f"frozen train/dev total must be {EXPECTED_TOTAL_ROWS}, got {total_rows}"
        )
    return result


def _load_source_specs(
    *,
    repo_root: Path,
    artifacts_root: Path,
    ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_specs = ledger.get("source_manifests")
    if not isinstance(raw_specs, list) or len(raw_specs) != len(SOURCE_SLICE_RULES):
        raise MaterializationError("ledger must declare exactly three source manifests")
    seen_slices: set[str] = set()
    seen_paths: set[Path] = set()
    specs: list[dict[str, Any]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise MaterializationError("malformed source manifest declaration")
        role_slice = str(raw_spec.get("role_slice", ""))
        if role_slice not in SOURCE_SLICE_RULES or role_slice in seen_slices:
            raise MaterializationError(f"unknown or duplicate role_slice: {role_slice!r}")
        seen_slices.add(role_slice)
        manifest_path = _safe_repo_path(
            repo_root,
            raw_spec.get("path"),
            context=f"{role_slice} manifest",
        )
        if not _is_relative_to(manifest_path, artifacts_root):
            raise MaterializationError(f"{role_slice}: source manifest outside artifacts.local")
        if manifest_path in seen_paths:
            raise MaterializationError(f"duplicate source manifest path: {manifest_path}")
        seen_paths.add(manifest_path)
        expected_sha = _require_sha256(
            raw_spec.get("sha256"), context=f"{role_slice} manifest"
        )
        actual_sha = sha256_file(manifest_path)
        if actual_sha != expected_sha:
            raise MaterializationError(
                f"{role_slice}: source manifest SHA-256 mismatch"
            )
        rule = SOURCE_SLICE_RULES[role_slice]
        if rule["dataset_root_from_manifest"] == "parent":
            dataset_root = manifest_path.parent.resolve()
        elif rule["dataset_root_from_manifest"] == "parent.parent":
            dataset_root = manifest_path.parent.parent.resolve()
        else:  # pragma: no cover - fixed table
            raise MaterializationError(f"{role_slice}: invalid dataset-root rule")
        if not _is_relative_to(dataset_root, artifacts_root):
            raise MaterializationError(f"{role_slice}: dataset root outside artifacts.local")
        specs.append(
            {
                "role_slice": role_slice,
                "role": rule["role"],
                "decoder": rule["decoder"],
                "manifest_path": manifest_path,
                "manifest_repo_path": _repo_relative(repo_root, manifest_path),
                "dataset_root": dataset_root,
                "dataset_root_repo_path": _repo_relative(repo_root, dataset_root),
                "manifest_sha256": actual_sha,
                "selected_row_count": int(raw_spec.get("selected_row_count", -1)),
                "whole_manifest_row_count": (
                    int(raw_spec["whole_manifest_row_count"])
                    if "whole_manifest_row_count" in raw_spec
                    else None
                ),
            }
        )
    if seen_slices != set(SOURCE_SLICE_RULES):
        raise MaterializationError("source manifest role slices drifted")
    if sum(spec["selected_row_count"] for spec in specs) != EXPECTED_TOTAL_ROWS:
        raise MaterializationError("source manifest selected-row total drifted")
    return specs


def _mapping_sha256(mapping: dict[int, int]) -> str:
    payload = json.dumps(
        {str(key): value for key, value in sorted(mapping.items())},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _png_bytes(value: np.ndarray, contract: dict[str, Any]) -> bytes:
    output = contract["output"]
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="L").save(
        buffer,
        format="PNG",
        optimize=bool(output["png_optimize"]),
        compress_level=int(output["png_compress_level"]),
    )
    return buffer.getvalue()


def _canonicalize_source(
    *,
    source_path: Path,
    decoder: str,
    contract: dict[str, Any],
) -> np.ndarray:
    try:
        return canonicalize_array(source_path, decoder, contract)
    except CanonicalizationError as exc:
        raise MaterializationError(
            f"{decoder} canonicalization failed for {source_path}: {exc}"
        ) from exc


def _reencode_canonical(
    canonical: np.ndarray,
    target_path: Path,
    contract: dict[str, Any],
) -> tuple[dict[int, int], tuple[int, int]]:
    if canonical.shape != (256, 256):
        raise MaterializationError(
            f"old-canonical mask must be 256x256, got {canonical.shape}"
        )
    observed = {int(value) for value in np.unique(canonical)}
    unknown = sorted(observed - set(EXPECTED_CLASS_MAPPING))
    if unknown:
        raise MaterializationError(f"old-canonical mask contains unknown IDs {unknown}")
    lookup = np.asarray(
        [EXPECTED_CLASS_MAPPING[index] for index in range(4)], dtype=np.uint8
    )
    mapped = lookup[canonical.astype(np.int64, copy=False)]
    try:
        write_canonical_png(target_path, mapped, contract)
    except CanonicalizationError as exc:
        raise MaterializationError(f"cannot write RISKSEG-R0 mask: {exc}") from exc
    counts = np.bincount(mapped.reshape(-1), minlength=4)
    return {index: int(counts[index]) for index in range(4)}, (
        int(mapped.shape[1]),
        int(mapped.shape[0]),
    )


def _validate_coverage(
    *,
    ledger: dict[str, Any],
    session_pixels: dict[str, dict[str, Counter[int]]],
) -> dict[str, Any]:
    overlay = ledger.get("riskseg_r0_forward_only_overlay")
    raw_minima = overlay.get("minimum_nonzero_class_coverage_sessions") if isinstance(overlay, dict) else None
    if not isinstance(raw_minima, dict):
        raise MaterializationError("ledger class-coverage minima are missing")
    expected_minima = {
        "train_blocking_obstacle": ("train", 1),
        "train_boundary_level_change": ("train", 2),
        "dev_blocking_obstacle": ("dev", 1),
        "dev_boundary_level_change": ("dev", 2),
    }
    result: dict[str, Any] = {}
    failures: list[str] = []
    for key, (role, class_id) in expected_minima.items():
        minimum = int(raw_minima.get(key, -1))
        if minimum < 0:
            raise MaterializationError(f"ledger coverage minimum missing: {key}")
        sessions = sorted(
            session_id
            for session_id, counts in session_pixels[role].items()
            if counts[class_id] > 0
        )
        result[key] = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "minimum_sessions": minimum,
            "nonzero_session_count": len(sessions),
            "nonzero_sessions": sessions,
            "passed": len(sessions) >= minimum,
        }
        if len(sessions) < minimum:
            failures.append(f"{key}: {len(sessions)} < {minimum}")
    if failures:
        raise MaterializationError(
            "DATA_SPLIT_NOT_READY: insufficient nonzero class/session coverage: "
            + "; ".join(failures)
        )
    return result


def materialize(
    *,
    repo_root: Path,
    ledger_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    artifacts_root = (repo_root / "artifacts.local").resolve()
    ledger_path = ledger_path.resolve()
    output_root = output_root.resolve()
    if not _is_relative_to(ledger_path, repo_root) or not ledger_path.is_file():
        raise MaterializationError("ledger must be an existing repository file")
    if not _is_relative_to(output_root, artifacts_root) or output_root == artifacts_root:
        raise MaterializationError("output must be a child directory of artifacts.local")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    ledger = _read_json(ledger_path)
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise MaterializationError("ledger schema mismatch")
    if ledger.get("protocol_id") != PROTOCOL_ID:
        raise MaterializationError("ledger protocol mismatch")
    mapping = _normalize_mapping(
        ledger.get("class_contract", {}).get("legacy_to_riskseg_r0_id")
        if isinstance(ledger.get("class_contract"), dict)
        else None
    )
    canonicalization_contract_path = (
        repo_root / CANONICALIZATION_CONTRACT_REPO_PATH
    ).resolve()
    canonicalizer_path = (repo_root / CANONICALIZER_REPO_PATH).resolve()
    if sha256_file(canonicalization_contract_path) != EXPECTED_CANONICALIZATION_CONTRACT_SHA256:
        raise MaterializationError("frozen canonicalization contract SHA-256 mismatch")
    if sha256_file(canonicalizer_path) != EXPECTED_CANONICALIZER_SHA256:
        raise MaterializationError("frozen canonicalizer SHA-256 mismatch")
    try:
        canonicalization_contract = load_contract(canonicalization_contract_path)
    except CanonicalizationError as exc:
        raise MaterializationError(f"invalid frozen canonicalization contract: {exc}") from exc
    native_to_old_mapping = dict(canonicalization_contract["_parsed_mapping"])
    canonical_identity_mapping = {index: index for index in range(4)}
    native_to_old_mapping_sha = _mapping_sha256(native_to_old_mapping)
    canonical_identity_mapping_sha = _mapping_sha256(canonical_identity_mapping)
    old_to_new_mapping_sha = _mapping_sha256(mapping)
    role_contracts = _role_contracts(ledger)
    source_specs = _load_source_specs(
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        ledger=ledger,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    ).resolve()
    published = False
    try:
        result_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        role_counts: Counter[str] = Counter()
        session_counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "dev": Counter(),
        }
        role_pixels: dict[str, Counter[int]] = {
            "train": Counter(),
            "dev": Counter(),
        }
        total_pixels: Counter[int] = Counter()
        session_pixels: dict[str, dict[str, Counter[int]]] = {
            "train": defaultdict(Counter),
            "dev": defaultdict(Counter),
        }

        for spec in source_specs:
            source_rows = _read_jsonl(spec["manifest_path"])
            if (
                spec["whole_manifest_row_count"] is not None
                and len(source_rows) != spec["whole_manifest_row_count"]
            ):
                raise MaterializationError(
                    f"{spec['role_slice']}: whole-manifest row count drift"
                )
            role = spec["role"]
            permitted_sessions = role_contracts[role]["sessions"]
            selected = [
                row for row in source_rows if str(row.get("session_id", "")) in permitted_sessions
            ]
            if len(selected) != spec["selected_row_count"]:
                raise MaterializationError(
                    f"{spec['role_slice']}: selected row count drift "
                    f"({len(selected)} != {spec['selected_row_count']})"
                )

            for source_row in selected:
                row_id = str(source_row.get("id", ""))
                if not row_id:
                    raise MaterializationError(f"{spec['role_slice']}: row has no ID")
                if row_id in seen_ids:
                    raise MaterializationError(f"duplicate selected row ID: {row_id}")
                seen_ids.add(row_id)
                session_id = str(source_row.get("session_id", ""))
                expected_session_count = permitted_sessions.get(session_id)
                if expected_session_count is None:
                    raise MaterializationError(f"{row_id}: session drift: {session_id!r}")

                image_path = _resolve_source_asset(
                    dataset_root=spec["dataset_root"],
                    raw_path=source_row.get("image_path"),
                    context=f"{row_id} image",
                )
                mask_path = _resolve_source_asset(
                    dataset_root=spec["dataset_root"],
                    raw_path=source_row.get("semantic_mask_path"),
                    context=f"{row_id} mask",
                )
                expected_image_sha = _require_sha256(
                    source_row.get("image_sha256"), context=f"{row_id} image"
                )
                expected_mask_sha = _require_sha256(
                    source_row.get("semantic_mask_sha256"), context=f"{row_id} mask"
                )
                if sha256_file(image_path) != expected_image_sha:
                    raise MaterializationError(f"{row_id}: image SHA-256 mismatch")
                if sha256_file(mask_path) != expected_mask_sha:
                    raise MaterializationError(f"{row_id}: source mask SHA-256 mismatch")

                digest = hashlib.sha256(
                    f"{role}:{session_id}:{row_id}".encode("utf-8")
                ).hexdigest()[:24]
                relative_mask = Path("masks") / role / f"{digest}.png"
                target_mask = staging / relative_mask
                try:
                    with Image.open(image_path) as source_image:
                        image_size = source_image.size
                    with Image.open(mask_path) as source_mask:
                        source_mask_size = source_mask.size
                except OSError as exc:
                    raise MaterializationError(
                        f"{row_id}: cannot decode source image or mask"
                    ) from exc
                if tuple(image_size) != tuple(source_mask_size):
                    raise MaterializationError(
                        f"{row_id}: source image/mask size mismatch "
                        f"({tuple(image_size)} != {tuple(source_mask_size)})"
                    )
                old_canonical = _canonicalize_source(
                    source_path=mask_path,
                    decoder=spec["decoder"],
                    contract=canonicalization_contract,
                )
                old_canonical_png_sha = hashlib.sha256(
                    _png_bytes(old_canonical, canonicalization_contract)
                ).hexdigest()
                class_counts, mask_size = _reencode_canonical(
                    old_canonical,
                    target_mask,
                    canonicalization_contract,
                )
                output_mask_sha = sha256_file(target_mask)
                first_stage_mapping_sha = (
                    native_to_old_mapping_sha
                    if spec["decoder"] == "source_native"
                    else canonical_identity_mapping_sha
                )

                role_counts[role] += 1
                session_counts[role][session_id] += 1
                for class_id, count in class_counts.items():
                    role_pixels[role][class_id] += count
                    total_pixels[class_id] += count
                    session_pixels[role][session_id][class_id] += count
                result_rows.append(
                    {
                        "schema_version": ROW_SCHEMA,
                        "protocol_id": PROTOCOL_ID,
                        "id": row_id,
                        "role": role,
                        "role_contract": role_contracts[role]["role_contract"],
                        "source_session_id": session_id,
                        "source_sequence_id": source_row.get("sequence_id"),
                        "source_frame_id": _source_row_frame_id(
                            source_row, row_id=row_id
                        ),
                        "source_manifest_role_slice": spec["role_slice"],
                        "source_manifest_path": spec["manifest_repo_path"],
                        "source_manifest_sha256": spec["manifest_sha256"],
                        "source_dataset_root": spec["dataset_root_repo_path"],
                        "source_mask_decoder": spec["decoder"],
                        "source_image_path": _repo_relative(repo_root, image_path),
                        "source_image_sha256": expected_image_sha,
                        "source_mask_path": _repo_relative(repo_root, mask_path),
                        "source_mask_sha256": expected_mask_sha,
                        "native_source_mask_sha256": expected_mask_sha,
                        "old_canonical_png_sha256": old_canonical_png_sha,
                        "output_mask_path": relative_mask.as_posix(),
                        "output_mask_sha256": output_mask_sha,
                        "source_to_old_canonical_mapping_sha256": first_stage_mapping_sha,
                        "old_canonical_to_riskseg_r0_mapping_sha256": old_to_new_mapping_sha,
                        "canonicalization_contract_sha256": EXPECTED_CANONICALIZATION_CONTRACT_SHA256,
                        "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
                        "legacy_to_riskseg_r0_id": {
                            str(key): value for key, value in mapping.items()
                        },
                        "width": mask_size[0],
                        "height": mask_size[1],
                        "class_pixel_counts": {
                            str(class_id): class_counts[class_id]
                            for class_id in range(4)
                        },
                    }
                )

        if len(result_rows) != EXPECTED_TOTAL_ROWS:
            raise MaterializationError(
                f"selected row total drift ({len(result_rows)} != {EXPECTED_TOTAL_ROWS})"
            )
        for role, contract in role_contracts.items():
            if role_counts[role] != contract["frame_count"]:
                raise MaterializationError(f"{role}: role frame count drift")
            actual_sessions = dict(session_counts[role])
            if actual_sessions != contract["sessions"]:
                missing = {
                    key: value
                    for key, value in contract["sessions"].items()
                    if actual_sessions.get(key) != value
                }
                raise MaterializationError(f"{role}: exact session/frame count drift: {missing}")

        coverage = _validate_coverage(ledger=ledger, session_pixels=session_pixels)
        manifest_path = staging / "manifest.jsonl"
        _write_jsonl(manifest_path, result_rows)
        ledger_sha = sha256_file(ledger_path)
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "RISKSEG_R0_REENCODED_VIEW_MATERIALIZED",
            "row_count": len(result_rows),
            "role_counts": dict(sorted(role_counts.items())),
            "session_counts": {
                role: dict(sorted(counts.items()))
                for role, counts in session_counts.items()
            },
            "class_order": {
                str(class_id): CLASS_NAMES[class_id] for class_id in range(4)
            },
            "legacy_to_riskseg_r0_id": {
                str(key): value for key, value in mapping.items()
            },
            "class_pixel_counts": {
                "total": {
                    str(class_id): total_pixels[class_id] for class_id in range(4)
                },
                **{
                    role: {
                        str(class_id): role_pixels[role][class_id]
                        for class_id in range(4)
                    }
                    for role in ("train", "dev")
                },
            },
            "session_class_pixel_counts": {
                role: {
                    session_id: {
                        str(class_id): counts[class_id] for class_id in range(4)
                    }
                    for session_id, counts in sorted(session_pixels[role].items())
                }
                for role in ("train", "dev")
            },
            "coverage_gates": coverage,
            "ledger": {
                "path": _repo_relative(repo_root, ledger_path),
                "sha256": ledger_sha,
            },
            "source_manifests": [
                {
                    "role_slice": spec["role_slice"],
                    "decoder": spec["decoder"],
                    "dataset_root": spec["dataset_root_repo_path"],
                    "path": spec["manifest_repo_path"],
                    "sha256": spec["manifest_sha256"],
                    "selected_row_count": spec["selected_row_count"],
                }
                for spec in source_specs
            ],
            "canonicalization": {
                "contract_path": CANONICALIZATION_CONTRACT_REPO_PATH.as_posix(),
                "contract_sha256": EXPECTED_CANONICALIZATION_CONTRACT_SHA256,
                "canonicalizer_path": CANONICALIZER_REPO_PATH.as_posix(),
                "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
                "source_native_to_old_canonical_mapping_sha256": native_to_old_mapping_sha,
                "canonical_passthrough_identity_mapping_sha256": canonical_identity_mapping_sha,
                "old_canonical_to_riskseg_r0_mapping_sha256": old_to_new_mapping_sha,
                "old_canonical_png_encoding": {
                    "width": 256,
                    "height": 256,
                    "mode": "L",
                    "optimize": bool(canonicalization_contract["output"]["png_optimize"]),
                    "compress_level": int(
                        canonicalization_contract["output"]["png_compress_level"]
                    ),
                },
            },
            "manifest": "manifest.jsonl",
            "manifest_sha256": sha256_file(manifest_path),
            "atomic_publish": True,
            "source_images_referenced_not_copied": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(staging / "receipt.json", receipt)
        staging.replace(output_root)
        published = True
        return receipt
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "docs/research/dual-loop/"
            "RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    ledger_path = args.ledger
    if not ledger_path.is_absolute():
        ledger_path = repo_root / ledger_path
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    receipt = materialize(
        repo_root=repo_root,
        ledger_path=ledger_path,
        output_root=output_root,
    )
    print(
        json.dumps(
            {"status": receipt["status"], "row_count": receipt["row_count"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
