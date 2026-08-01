#!/usr/bin/env python3
"""Acquire only frozen HFTF Stage C E0.1 dev/heldout media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_stage_c_e0_fresh_media as e0_acquire  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_foot_ground_student_canary_e0_1"
STATUS = "FROZEN_BEFORE_FRESH_EVALUATION_RGB_DEPTH_OR_LABEL_OUTCOME"
LOCK_SCHEMA = (
    "blindassist_hftf_stage_c_e0_1_fresh_evaluation_source_lock"
)
RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_e0_1_fresh_evaluation_media_acquisition"
)


def _load_json(path: Path) -> dict[str, Any]:
    return e0_acquire._load_json(path)


def _sha256(path: Path) -> str:
    return e0_acquire._sha256(path)


def _validate_lock(
    protocol: dict[str, Any],
    protocol_path: Path,
    lock: dict[str, Any],
) -> None:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unexpected E0.1 source-lock schema")
    if (
        lock.get("terminal")
        != "E0_1_FRESH_EVALUATION_SOURCE_LOCK_VALIDATED"
    ):
        raise ValueError("E0.1 evaluation source lock is not validated")
    if lock.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0.1 source-lock protocol binding mismatch")
    if (
        lock.get("fresh_evaluation_sources")
        != protocol["fresh_evaluation_sources"]
    ):
        raise ValueError("E0.1 source-lock cohort mismatch")
    if lock.get("fresh_evaluation_rgb_or_depth_read"):
        raise ValueError("E0.1 source lock unexpectedly read media")
    if not lock.get(
        "exact_fresh_evaluation_media_acquisition_authorized"
    ):
        raise ValueError("E0.1 exact acquisition is not authorized")


def _e0_compatible_protocol(
    protocol: dict[str, Any],
    e0_protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_binding": {
            "dataset_repo": protocol["fresh_evaluation_selection"][
                "dataset_repo"
            ],
            "dataset_revision": protocol["fresh_evaluation_selection"][
                "dataset_revision"
            ],
            "metadata_files": e0_protocol["dataset_binding"][
                "metadata_files"
            ],
        },
        "frozen_sources": protocol["fresh_evaluation_sources"],
    }


def run(
    protocol_path: Path,
    source_lock_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if protocol.get("schema") != SCHEMA or protocol.get("status") != STATUS:
        raise ValueError("Stage C E0.1 protocol is not frozen")
    lock = _load_json(source_lock_path)
    _validate_lock(protocol, protocol_path, lock)
    e0_protocol_path = (
        protocol_path.parent
        / protocol["parent_bindings"]["e0_protocol_path"]
    ).resolve()
    e0_protocol = _load_json(e0_protocol_path)
    compatible = _e0_compatible_protocol(protocol, e0_protocol)
    patterns = e0_acquire._allow_patterns(compatible)
    snapshot_download(
        repo_id=compatible["dataset_binding"]["dataset_repo"],
        repo_type="dataset",
        revision=compatible["dataset_binding"]["dataset_revision"],
        allow_patterns=patterns,
        local_dir=output_root,
    )
    files = e0_acquire._validate_download(compatible, output_root)
    return {
        "schema": RESULT_SCHEMA,
        "terminal": (
            "E0_1_FRESH_EVALUATION_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND"
        ),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": _sha256(source_lock_path),
        "dataset_repo": compatible["dataset_binding"]["dataset_repo"],
        "dataset_revision": compatible["dataset_binding"][
            "dataset_revision"
        ],
        "output_root": str(output_root),
        "allow_patterns": patterns,
        "downloaded_file_count": len(files),
        "downloaded_files": files,
        "fresh_evaluation_media_total_bytes": sum(
            item["size_bytes"]
            for item in files
            if item["kind"] in {"pose", "rgb", "depth"}
        ),
        "fresh_evaluation_rgb_or_depth_read": True,
        "new_dev_and_heldout_burned": True,
        "fresh_evaluation_geometry_label_outcome_read": False,
        "student_output_read": False,
        "fresh_transport_audit_authorized": True,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _artifacts_path(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as error:
        raise ValueError("E0.1 acquisition path must stay under artifacts.local") from error
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
    lock = (repo_root / args.source_lock).resolve()
    output_root = _artifacts_path(repo_root / args.output_root, repo_root)
    manifest = _artifacts_path(repo_root / args.manifest, repo_root)
    result = run(protocol, lock, output_root, repo_root)
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
                "fresh_evaluation_media_total_bytes": result[
                    "fresh_evaluation_media_total_bytes"
                ],
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
