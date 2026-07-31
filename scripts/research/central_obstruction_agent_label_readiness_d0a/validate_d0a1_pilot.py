#!/usr/bin/env python3
"""Independently validate D0-A1 pilot inputs and their candidate-output firewall."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .freeze_d0a1_pilot import (
    DEFAULT_LOCK,
    MANIFEST_NAME,
    RECEIPT_NAME,
    PilotFreezeError,
    build_bundle,
    load_json,
    load_lock,
    make_contact_sheet,
    repo_path,
)
from .freeze_input_universe import canonical_bytes, sha256_file


VALIDATION_NAME = "pilot-input-validation.json"


class PilotValidationError(ValueError):
    """Fail-closed D0-A1 independent-validation error."""


def without_review_paths(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"review_image_path", "review_image_sha256"}
    }


def validate(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest_path = output_root / MANIFEST_NAME
    receipt_path = output_root / RECEIPT_NAME
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise PilotValidationError("pilot manifest or receipt is missing")
    manifest = load_json(manifest_path, where="pilot manifest")
    receipt = load_json(receipt_path, where="pilot receipt")
    if receipt.get("output_manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise PilotValidationError("manifest receipt hash mismatch")
    if manifest.get("lock", {}).get("sha256") != sha256_file(lock_path):
        raise PilotValidationError("lock hash mismatch")
    try:
        expected, _, clip_frames = build_bundle(
            repo_root=repo_root,
            lock_path=lock_path,
            frozen_at_utc=manifest.get("frozen_at_utc"),
        )
    except PilotFreezeError as error:
        raise PilotValidationError(str(error)) from error
    for key in (
        "schema_version",
        "protocol_id",
        "phase",
        "evidence_instance",
        "status",
        "frozen_at_utc",
        "lock",
        "prompt",
        "d0a0_manifest_sha256",
        "d0a0_reuse_role_ledger_sha256",
        "candidate_output_access",
        "labels_generated",
        "production_source_overlap_count",
        "calibration_source_count",
        "pilot_clip_count",
        "pilot_observation_count",
        "calibration_sources",
        "source_metadata",
        "next_permitted_action",
        "d0a2_production_labeling_authorized",
        "d0b_authorized",
    ):
        if manifest.get(key) != expected.get(key):
            raise PilotValidationError(f"manifest field mismatch: {key}")
    actual_observations = manifest.get("observations")
    if not isinstance(actual_observations, list) or len(actual_observations) != len(expected["observations"]):
        raise PilotValidationError("observation count mismatch")
    actual_by_key = {
        (row.get("clip_id"), row.get("source_frame_index")): row
        for row in actual_observations
        if isinstance(row, dict)
    }
    if len(actual_by_key) != len(actual_observations):
        raise PilotValidationError("observation identity is duplicated")
    expected_by_key = {
        (row["clip_id"], row["source_frame_index"]): row
        for row in expected["observations"]
    }
    if set(actual_by_key) != set(expected_by_key):
        raise PilotValidationError("observation identity set mismatch")
    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        if without_review_paths(actual_row) != expected_row:
            raise PilotValidationError(f"observation reconstruction mismatch: {key}")
        review_path = repo_path(repo_root, actual_row.get("review_image_path"), where="review image")
        if not review_path.is_file() or sha256_file(review_path) != actual_row.get("review_image_sha256"):
            raise PilotValidationError(f"review image hash mismatch: {key}")
        rendered = cv2.imread(str(review_path), cv2.IMREAD_COLOR)
        expected_rendered = dict(clip_frames[actual_row["clip_id"]])[actual_row["source_frame_index"]]
        if rendered is None or rendered.shape != expected_rendered.shape or not np.array_equal(rendered, expected_rendered):
            raise PilotValidationError(f"review image pixels mismatch: {key}")
    contact_rows = manifest.get("contact_sheets")
    if not isinstance(contact_rows, list) or len(contact_rows) != len(clip_frames):
        raise PilotValidationError("contact-sheet inventory mismatch")
    contacts = {row.get("clip_id"): row for row in contact_rows if isinstance(row, dict)}
    if set(contacts) != set(clip_frames):
        raise PilotValidationError("contact-sheet clip set mismatch")
    for clip_id, frames in clip_frames.items():
        row = contacts[clip_id]
        path = repo_path(repo_root, row.get("path"), where="contact sheet")
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise PilotValidationError(f"contact-sheet hash mismatch: {clip_id}")
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        expected_sheet = make_contact_sheet([frame for _, frame in frames])
        if decoded is None or decoded.shape != expected_sheet.shape:
            raise PilotValidationError(f"contact-sheet geometry mismatch: {clip_id}")
    if (
        manifest.get("candidate_output_access") is not False
        or manifest.get("labels_generated") is not False
        or manifest.get("production_source_overlap_count") != 0
        or manifest.get("d0a2_production_labeling_authorized") is not False
        or manifest.get("d0b_authorized") is not False
    ):
        raise PilotValidationError("D0-A1 entry safety boundary mismatch")
    return {
        "schema_version": "blindassist.central_obstruction_d0a1_pilot_input_validation.v1",
        "protocol_id": manifest["protocol_id"],
        "phase": "D0-A1",
        "evidence_instance": manifest["evidence_instance"],
        "status": "VALID",
        "decision": "D0_A1_PILOT_INPUTS_VALID_PRIMARY_REVIEW_NOT_RUN",
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
        "lock_sha256": sha256_file(lock_path),
        "calibration_source_count": manifest["calibration_source_count"],
        "pilot_clip_count": manifest["pilot_clip_count"],
        "pilot_observation_count": manifest["pilot_observation_count"],
        "production_source_overlap_count": 0,
        "candidate_output_access": False,
        "labels_generated": False,
        "isolated_review_pass_count": 0,
        "readiness_evaluated": False,
        "next_permitted_action": "D0-A1_PRIMARY_SOURCE_ONLY_LABEL_PASS",
        "d0a2_production_labeling_authorized": False,
        "d0b_authorized": False,
        "errors": [],
    }


def write_validation(
    *,
    repo_root: Path,
    lock_path: Path,
    output_root: Path,
) -> Path:
    destination = output_root / VALIDATION_NAME
    if destination.exists():
        raise PilotValidationError("pilot validation already exists; refusing overwrite")
    result = validate(repo_root=repo_root, lock_path=lock_path, output_root=output_root)
    temporary = output_root / f".{VALIDATION_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(canonical_bytes(result))
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.lock.resolve()
    lock = load_lock(repo_root, lock_path)
    output_root = args.output_root or repo_path(repo_root, lock["output_root"], where="output_root")
    path = write_validation(
        repo_root=repo_root,
        lock_path=lock_path,
        output_root=output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "VALID",
                "validation": str(path),
                "sha256": sha256_file(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
