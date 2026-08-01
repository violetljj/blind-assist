#!/usr/bin/env python3
"""Acquire only the frozen HFTF Stage C E0 EgoWalk files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


SCHEMA = "blindassist_hftf_stage_c_fresh_foot_ground_student_canary_e0"
STATUS = "FROZEN_BEFORE_FRESH_RGB_DEPTH_OR_GEOMETRY_LABEL_OUTCOME"
LOCK_SCHEMA = "blindassist_hftf_stage_c_e0_fresh_source_lock_result"
RESULT_SCHEMA = "blindassist_hftf_stage_c_e0_fresh_media_acquisition"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _allow_patterns(protocol: dict[str, Any]) -> list[str]:
    patterns = set(protocol["dataset_binding"]["metadata_files"])
    for source in protocol["frozen_sources"]:
        for kind in ("pose", "rgb", "depth"):
            patterns.add(str(source["files"][kind]["path"]))
    return sorted(patterns)


def _validate_lock(
    protocol: dict[str, Any],
    protocol_path: Path,
    lock: dict[str, Any],
) -> None:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unexpected E0 source-lock schema")
    if lock.get("terminal") != "E0_FRESH_SOURCE_LOCK_VALIDATED":
        raise ValueError("E0 source lock is not validated")
    if lock.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0 source-lock protocol binding mismatch")
    locked = [
        (item["role"], item["trajectory"])
        for item in lock["selected_sources"]
    ]
    frozen = [
        (item["role"], item["trajectory"])
        for item in protocol["frozen_sources"]
    ]
    if locked != frozen:
        raise ValueError("E0 source-lock cohort mismatch")
    if lock.get("rgb_or_depth_media_content_read"):
        raise ValueError("Source lock unexpectedly read fresh media")
    if not lock.get("exact_selected_media_acquisition_authorized"):
        raise ValueError("Exact E0 media acquisition is not authorized")


def _validate_download(
    protocol: dict[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    metadata = protocol["dataset_binding"]["metadata_files"]
    for relative, expected_hash in sorted(metadata.items()):
        path = output_root / relative
        files.append(
            {
                "kind": "metadata",
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        if files[-1]["sha256"] != expected_hash:
            raise ValueError(f"Metadata hash mismatch: {relative}")
    for source in protocol["frozen_sources"]:
        for kind in ("pose", "rgb", "depth"):
            expected = source["files"][kind]
            relative = str(expected["path"])
            path = output_root / relative
            actual_size = path.stat().st_size
            actual_hash = _sha256(path)
            if actual_size != int(expected["size_bytes"]):
                raise ValueError(f"Frozen size mismatch: {relative}")
            if actual_hash != expected["sha256"]:
                raise ValueError(f"Frozen hash mismatch: {relative}")
            files.append(
                {
                    "kind": kind,
                    "role": source["role"],
                    "trajectory": source["trajectory"],
                    "path": relative,
                    "size_bytes": actual_size,
                    "sha256": actual_hash,
                }
            )
    return files


def run(
    protocol_path: Path,
    source_lock_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("Stage C E0 protocol is not frozen")
    source_lock = _load_json(source_lock_path)
    _validate_lock(protocol, protocol_path, source_lock)
    patterns = _allow_patterns(protocol)
    snapshot_download(
        repo_id=protocol["dataset_binding"]["dataset_repo"],
        repo_type="dataset",
        revision=protocol["dataset_binding"]["dataset_revision"],
        allow_patterns=patterns,
        local_dir=output_root,
    )
    files = _validate_download(protocol, output_root)
    return {
        "schema": RESULT_SCHEMA,
        "terminal": "E0_FRESH_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": _sha256(source_lock_path),
        "dataset_repo": protocol["dataset_binding"]["dataset_repo"],
        "dataset_revision": protocol["dataset_binding"]["dataset_revision"],
        "output_root": str(output_root.resolve()),
        "allow_patterns": patterns,
        "downloaded_file_count": len(files),
        "downloaded_files": files,
        "frozen_media_total_bytes": sum(
            item["size_bytes"]
            for item in files
            if item["kind"] in {"pose", "rgb", "depth"}
        ),
        "fresh_rgb_or_depth_media_content_read": True,
        "selected_sources_burned": True,
        "fresh_geometry_label_outcome_read": False,
        "student_output_read": False,
        "transport_decode_audit_authorized": True,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_path(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    artifacts_root = (repo_root / "artifacts.local").resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("E0 media/output must stay under artifacts.local") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    source_lock = (repo_root / args.source_lock).resolve()
    output_root = _require_artifacts_path(
        repo_root / args.output_root, repo_root
    )
    manifest = _require_artifacts_path(repo_root / args.manifest, repo_root)
    result = run(protocol, source_lock, output_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "downloaded_file_count": result["downloaded_file_count"],
                "frozen_media_total_bytes": result[
                    "frozen_media_total_bytes"
                ],
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
