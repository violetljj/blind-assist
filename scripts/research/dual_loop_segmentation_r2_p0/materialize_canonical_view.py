"""Materialize a resumable, SHA-closed canonical mask view for R2-P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, __version__ as pillow_version

from . import PROTOCOL_ID
from .canonicalizer import (
    CanonicalizationError,
    canonicalize_array,
    load_contract,
    read_json,
    sha256_file,
    write_canonical_png,
)


ROW_SCHEMA = "blindassist.dual_loop_segmentation_r2_p0.materialized_canonical_view_row.v1"
RECEIPT_SCHEMA = "blindassist.dual_loop_segmentation_r2_p0.materialized_canonical_view_receipt.v1"


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value, pretty=pretty))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CanonicalizationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CanonicalizationError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def _safe_existing(root: Path, value: str) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CanonicalizationError(f"path escapes declared dataset root: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CanonicalizationError(f"source path is outside repository: {path}") from exc


def _row_frame_id(row: dict[str, Any]) -> int:
    for key in ("frame_id", "frame_index", "sequence_frame_index"):
        if key in row:
            return int(row[key])
    raise CanonicalizationError(f"manifest row {row.get('id')!r} has no frame identity")


def _identity_sha(inputs: list[dict[str, str]]) -> str:
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_source_specs(repo_root: Path, config_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs: list[dict[str, Any]] = []
    identities: list[dict[str, str]] = []
    for config_path in config_paths:
        config_path = config_path.resolve()
        config = read_json(config_path)
        if config.get("protocol_id") != PROTOCOL_ID:
            raise CanonicalizationError(f"source config protocol mismatch: {config_path}")
        identities.append(
            {
                "path": _repo_relative(repo_root, config_path),
                "sha256": sha256_file(config_path),
            }
        )
        raw_specs = config.get("sources")
        if not isinstance(raw_specs, list) or not raw_specs:
            raise CanonicalizationError(f"source config has no sources: {config_path}")
        specs.extend(raw_specs)
    return specs, identities


def materialize(
    *,
    repo_root: Path,
    contract_path: Path,
    schema_path: Path,
    source_config_paths: Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    artifacts_root = (repo_root / "artifacts.local").resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(artifacts_root)
    except ValueError as exc:
        raise CanonicalizationError("canonical view output must stay under artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite canonical view: {output_root}")
    staging = output_root.with_name(output_root.name + ".staging")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    contract = load_contract(contract_path.resolve())
    specs, source_config_identities = _load_source_specs(repo_root, source_config_paths)
    code_path = Path(__file__).with_name("canonicalizer.py").resolve()
    identities = {
        "canonicalizer_code_sha256": sha256_file(code_path),
        "canonicalizer_config_sha256": sha256_file(contract_path.resolve()),
        "canonical_view_schema_sha256": sha256_file(schema_path.resolve()),
        "source_configs": source_config_identities,
    }
    state_identity = _identity_sha(
        [
            {"kind": "canonicalizer_code", "sha256": identities["canonicalizer_code_sha256"]},
            {"kind": "canonicalizer_config", "sha256": identities["canonicalizer_config_sha256"]},
            {"kind": "view_schema", "sha256": identities["canonical_view_schema_sha256"]},
            *[
                {"kind": f"source_config:{item['path']}", "sha256": item["sha256"]}
                for item in source_config_identities
            ],
        ]
    )
    state_path = staging / "state.json"
    if staging.exists():
        state = read_json(state_path)
        if state.get("state_identity_sha256") != state_identity:
            raise CanonicalizationError("staging identity differs; refusing unsafe resume")
    else:
        staging.mkdir(parents=True)
        _write_atomic(
            state_path,
            {
                "schema_version": "blindassist.dual_loop_segmentation_r2_p0.materialization_state.v1",
                "protocol_id": PROTOCOL_ID,
                "state_identity_sha256": state_identity,
                "status": "RUNNING",
                "completed_rows": 0,
            },
        )
    result_rows: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    for spec in specs:
        role = str(spec["role"])
        manifest_path = (repo_root / str(spec["manifest"])).resolve()
        if sha256_file(manifest_path) != str(spec["manifest_sha256"]):
            raise CanonicalizationError(f"{role}: source manifest SHA256 mismatch")
        dataset_root = (repo_root / str(spec["dataset_root"])).resolve()
        rows = _read_jsonl(manifest_path)
        split = spec.get("split")
        if split is not None:
            rows = [row for row in rows if row.get("split") == split]
        if len(rows) != int(spec["expected_rows"]):
            raise CanonicalizationError(
                f"{role}: expected {spec['expected_rows']} rows, got {len(rows)}"
            )
        for source_row in rows:
            row_id = str(source_row["id"])
            global_id = f"{role}:{row_id}"
            if global_id in row_ids:
                raise CanonicalizationError(f"duplicate materialized identity: {global_id}")
            row_ids.add(global_id)
            mask_path = _safe_existing(dataset_root, str(source_row["semantic_mask_path"]))
            image_path: Path | None = None
            image_sha: str | None = None
            if source_row.get("image_path"):
                image_path = _safe_existing(dataset_root, str(source_row["image_path"]))
                image_sha = str(source_row.get("image_sha256") or sha256_file(image_path))
                if sha256_file(image_path) != image_sha:
                    raise CanonicalizationError(f"{global_id}: image SHA256 mismatch")
            source_sha = str(source_row.get("semantic_mask_sha256") or sha256_file(mask_path))
            if sha256_file(mask_path) != source_sha:
                raise CanonicalizationError(f"{global_id}: source mask SHA256 mismatch")
            canonical = canonicalize_array(mask_path, str(spec["decoder"]), contract)
            name_digest = hashlib.sha256(global_id.encode("utf-8")).hexdigest()[:20]
            relative_mask = Path("masks") / role / f"{name_digest}.png"
            target_mask = staging / relative_mask
            write_canonical_png(target_mask, canonical, contract)
            with Image.open(target_mask) as written:
                if written.mode != "L" or written.size != (256, 256):
                    raise CanonicalizationError(f"{global_id}: canonical PNG encoding mismatch")
                written_ids = sorted(int(value) for value in np.unique(np.asarray(written)))
            if not set(written_ids).issubset({0, 1, 2, 3}):
                raise CanonicalizationError(f"{global_id}: canonical PNG contains invalid IDs")
            source_id = str(source_row.get("source_id") or source_row.get("session_id") or role)
            session_id = str(source_row.get("session_id") or source_id)
            result_rows.append(
                {
                    "schema_version": ROW_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "id": global_id,
                    "source_row_id": row_id,
                    "role": role,
                    "source_id": source_id,
                    "session_id": session_id,
                    "sequence_id": source_row.get("sequence_id"),
                    "frame_id": _row_frame_id(source_row),
                    "source_capture_timestamp_ns": int(
                        source_row.get("source_capture_timestamp_ns", _row_frame_id(source_row))
                    ),
                    "scene_bucket": str(source_row.get("scene_bucket", role)),
                    "image_repo_relative_path": _repo_relative(repo_root, image_path) if image_path else None,
                    "image_sha256": image_sha,
                    "source_mask_repo_relative_path": _repo_relative(repo_root, mask_path),
                    "source_mask_sha256": source_sha,
                    "source_mask_decoder": str(spec["decoder"]),
                    "canonical_mask_path": relative_mask.as_posix(),
                    "canonical_mask_sha256": sha256_file(target_mask),
                    "canonical_width": 256,
                    "canonical_height": 256,
                    "canonical_ids": written_ids,
                    "canonicalizer_code_sha256": identities["canonicalizer_code_sha256"],
                    "canonicalizer_config_sha256": identities["canonicalizer_config_sha256"],
                    "canonical_view_schema_sha256": identities["canonical_view_schema_sha256"],
                    "label_authority": str(source_row.get("label_authority", "synthetic")),
                    "formal_authority": False,
                }
            )
            role_counts[role] += 1
            _write_atomic(
                state_path,
                {
                    "schema_version": "blindassist.dual_loop_segmentation_r2_p0.materialization_state.v1",
                    "protocol_id": PROTOCOL_ID,
                    "state_identity_sha256": state_identity,
                    "status": "RUNNING",
                    "completed_rows": len(result_rows),
                    "last_id": global_id,
                },
            )
    if not result_rows:
        raise CanonicalizationError("materialized canonical view would contain zero rows")
    manifest_path = staging / "manifest.jsonl"
    _write_jsonl_atomic(manifest_path, result_rows)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "CANONICAL_VIEW_MATERIALIZED",
        "formal_authority": False,
        "output_root": str(output_root),
        "manifest": "manifest.jsonl",
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(result_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "identity": identities,
        "state_identity_sha256": state_identity,
        "atomic_publish": True,
        "interruption_resume_supported": True,
        "pillow_version": pillow_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(staging / "receipt.json", receipt)
    _write_atomic(
        state_path,
        {
            "schema_version": "blindassist.dual_loop_segmentation_r2_p0.materialization_state.v1",
            "protocol_id": PROTOCOL_ID,
            "state_identity_sha256": state_identity,
            "status": "COMPLETE",
            "completed_rows": len(result_rows),
            "manifest_sha256": receipt["manifest_sha256"],
        },
    )
    staging.replace(output_root)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--sources-config", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = materialize(
        repo_root=args.repo_root,
        contract_path=args.contract,
        schema_path=args.schema,
        source_config_paths=args.sources_config,
        output_root=args.output_root,
    )
    print(json.dumps({"status": result["status"], "rows": result["row_count"]}, ensure_ascii=False))
